#!/usr/bin/env python3
import argparse
import os
import yaml
import logging
import torch
import numpy as np
import h5py
import torch.nn as nn
import torch.nn.functional as F
import pathlib
import mdtraj as md
from torch_cluster import knn_graph
from torch_geometric.nn import ChebConv
from typing import Dict, List, Tuple
from sklearn.neighbors import NearestNeighbors

# ===================================================================
# (A) Argument Parsing & Setup
# ===================================================================

parser = argparse.ArgumentParser(description="Generate Pooled Embeddings from a Novel PDB")
parser.add_argument('--config', type=str, required=True, help='Path to the generation YAML configuration file.')
parser.add_argument('--pdb', type=str, required=True, help='Path to the novel input PDB file.')
parser.add_argument('--output', type=str, required=True, help='Path for the output HDF5 file.')
parser.add_argument('--debug', action='store_true', help='Enable debug level logging.')
args = parser.parse_args()

# Setup logging
log_level = logging.DEBUG if args.debug else logging.INFO
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler()])
logger = logging.getLogger()

logger.info("Novel PDB Inference Script Started")

# ===================================================================
# (B) Configuration Loading
# ===================================================================

config_path = pathlib.Path(args.config)
if not config_path.is_file():
    logger.error(f"Configuration file not found: {args.config}"); exit(1)

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    logger.info(f"Loaded configuration from {config_path}")

# Extract parameters
paths = config.get('paths', {})
hno_cfg = config.get('hno_encoder', {})
diff_cfg = config.get('diffusion_model', {})
gen_cfg = config.get('generation_settings', {})

HNO_CKPT_PATH = pathlib.Path(paths['hno_checkpoint_path'])
DIFF_CKPT_PATH = pathlib.Path(paths['diffusion_checkpoint_path'])
CANONICAL_REF_PDB = pathlib.Path(paths['canonical_reference_pdb'])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ===================================================================
# (C) Model Definitions (Must match trained models)
# ===================================================================

# --- REAL HNO Encoder ---
class HNO(nn.Module):
    def __init__(self, hidden_dim, K):
        super().__init__()
        self.conv1 = ChebConv(3, hidden_dim, K=K)
        self.bano1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.bano2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.bano3 = nn.BatchNorm1d(hidden_dim)
        self.conv4 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.mlpRep = nn.Linear(hidden_dim, 3)

    def forward_representation(self, x, edge_index):
        x = x.float()
        x = self.bano1(F.leaky_relu(self.conv1(x, edge_index)))
        x = self.bano2(F.leaky_relu(self.conv2(x, edge_index)))
        x = self.bano3(F.relu(self.conv3(x, edge_index)))
        x = self.conv4(x, edge_index)
        return F.normalize(x, p=2.0, dim=1)

# --- REAL Diffusion Model ---
class ConditionerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(128, output_dim)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.net(x)

class DiffusionModel(nn.Module):
    def __init__(self, data_dim, cond_input_dim, cond_encoded_dim, hidden_dim):
        super().__init__()
        self.encoder = ConditionerEncoder(cond_input_dim, cond_encoded_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1 + cond_encoded_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, data_dim)
        )
    def forward(self, x, t, cond_2d):
        t_norm = t.float().unsqueeze(1) / diff_cfg['diffusion_steps']
        encoded_cond = self.encoder(cond_2d)
        net_input = torch.cat([x, t_norm, encoded_cond], dim=1)
        return self.net(net_input)

# ===================================================================
# (D) PDB Processing & Alignment Utilities
# ===================================================================

def get_heavy_atom_coords(pdb_path: pathlib.Path) -> np.ndarray:
    traj = md.load_pdb(str(pdb_path))
    heavy_indices = traj.topology.select('not element H')
    traj.atom_slice(heavy_indices, inplace=True)
    return traj.xyz[0] * 10  # Convert nm to Angstroms

def get_ca_indices(pdb_path: pathlib.Path) -> np.ndarray:
    traj = md.load_pdb(str(pdb_path))
    heavy_indices = traj.topology.select('not element H')
    heavy_traj = traj.atom_slice(heavy_indices, inplace=False)
    return heavy_traj.topology.select('name CA')

def find_mutual_nn_pairs(ref_coords_ca: np.ndarray, target_coords_ca: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    nn_ref_to_target = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_coords_ca)
    _, indices1 = nn_ref_to_target.kneighbors(ref_coords_ca)
    nn_target_to_ref = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(ref_coords_ca)
    _, indices2 = nn_target_to_ref.kneighbors(target_coords_ca)
    
    ref_indices, target_indices = [], []
    for i, target_idx in enumerate(indices1.flatten()):
        if indices2[target_idx, 0] == i:
            ref_indices.append(i)
            target_indices.append(target_idx)
    return np.array(ref_indices), np.array(target_indices)

def align_by_core(structure_to_align: np.ndarray, core_indices_to_align: np.ndarray,
                  reference_structure: np.ndarray, core_indices_reference: np.ndarray) -> np.ndarray:
    P_core = reference_structure[core_indices_reference]
    Q_core = structure_to_align[core_indices_to_align]
    
    centroid_P_core = P_core.mean(axis=0)
    centroid_Q_core = Q_core.mean(axis=0)
    
    P_c = P_core - centroid_P_core
    Q_c = Q_core - centroid_Q_core
    
    H = Q_c.T @ P_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    
    return (structure_to_align - centroid_Q_core) @ R + centroid_P_core

# ===================================================================
# (E) Main Inference Logic
# ===================================================================

@torch.no_grad()
def main():
    # 1. Load Models
    logger.info("Loading pre-trained models...")
    hno_model = HNO(hno_cfg['hidden_dim'], hno_cfg['cheb_order']).to(DEVICE)
    hno_checkpoint = torch.load(HNO_CKPT_PATH, map_location=DEVICE)
    hno_model.load_state_dict(hno_checkpoint['model_state_dict'])
    hno_model.eval()

    diffusion_model = DiffusionModel(
        diff_cfg['data_dim'], diff_cfg['cond_input_dim'],
        diff_cfg['cond_encoded_dim'], diff_cfg['hidden_dim']
    ).to(DEVICE)
    diff_checkpoint = torch.load(DIFF_CKPT_PATH, map_location=DEVICE)
    diffusion_model.load_state_dict(diff_checkpoint['model_state_dict'])
    diffusion_model.eval()
    logger.info("Models loaded successfully.")

    # 2. Process PDBs and Align
    logger.info("Processing and aligning PDB files...")
    ref_coords = get_heavy_atom_coords(CANONICAL_REF_PDB)
    new_coords = get_heavy_atom_coords(pathlib.Path(args.pdb))
    
    ref_ca_indices = get_ca_indices(CANONICAL_REF_PDB)
    new_ca_indices = get_ca_indices(pathlib.Path(args.pdb))

    core_ref_idx, core_new_idx = find_mutual_nn_pairs(ref_coords[ref_ca_indices], new_coords[new_ca_indices])
    
    aligned_new_coords = align_by_core(new_coords, new_ca_indices[core_new_idx], ref_coords, ref_ca_indices[core_ref_idx])
    logger.info(f"Aligned new PDB to canonical reference using {len(core_ref_idx)} core atom pairs.")
    
    # 3. Generate Novel z_ref
    logger.info("Generating novel z_ref from new PDB...")
    coords_tensor = torch.from_numpy(aligned_new_coords).float().to(DEVICE)
    edge_index = knn_graph(coords_tensor, k=hno_cfg['knn_value'])
    novel_z_ref = hno_model.forward_representation(coords_tensor, edge_index)
    logger.info(f"Generated novel z_ref with shape: {novel_z_ref.shape}")

    # 4. Generate Pooled Embeddings
    logger.info(f"Generating {gen_cfg['num_gen']} pooled embeddings...")
    T = diff_cfg['diffusion_steps']
    betas = torch.linspace(diff_cfg['beta_start'], diff_cfg['beta_end'], T, device=DEVICE)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    shape = (gen_cfg['num_gen'], diff_cfg['data_dim'])
    conditioner_batch = novel_z_ref.unsqueeze(0).repeat(shape[0], 1, 1)
    x_t = torch.randn(shape, device=DEVICE)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=DEVICE, dtype=torch.long)
        predicted_noise = diffusion_model(x_t, t_batch, conditioner_batch)
        alpha_t = alphas[t]
        beta_t = betas[t]
        sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)
        model_mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alphas_cumprod[t]) * predicted_noise)
        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = model_mean + torch.sqrt(beta_t) * noise
        else:
            x_t = model_mean
    
    generated_embeddings = x_t.cpu().numpy()
    logger.info(f"Generated embeddings with shape: {generated_embeddings.shape}")

    # 5. Save Outputs
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_path}")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset("generated_pooled_embeddings", data=generated_embeddings)
        f.create_dataset("novel_z_ref_conditioner", data=novel_z_ref.cpu().numpy())
    
    logger.info("Inference complete.")

if __name__ == "__main__":
    main()
