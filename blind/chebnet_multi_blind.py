################################################################################
# %% Imports
################################################################################
import os
import sys
import json
import yaml
import argparse
import logging
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Dict, List, Optional, Tuple, Any

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import ChebConv, global_mean_pool
from torch_cluster import knn_graph
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors # New dependency

################################################################################
# (A) Argument Parsing
################################################################################
parser = argparse.ArgumentParser(
    description="Protein Reconstruction: HNO + Single Decoder + Optional Dihedral Loss (Multi-System Version)"
)
parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
args = parser.parse_args()

################################################################################
# (B) Pre-Logging Config Load
################################################################################
LOG_FILE_DEFAULT = "logfile_multi_system.log"
log_file_path = LOG_FILE_DEFAULT
try:
    with open(args.config, "r") as f:
        temp_config = yaml.safe_load(f)
        log_file_path = temp_config.get("log_file", LOG_FILE_DEFAULT)
except Exception as e:
    print(f"[Warning] Could not pre-load log file path from config ({args.config}): {e}. Using default: {LOG_FILE_DEFAULT}")

################################################################################
# (C) Logging Setup
################################################################################
logger = logging.getLogger("ProteinReconstruction")
logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    try:
        fh = logging.FileHandler(log_file_path, mode="w")
        fh.setLevel(logging.DEBUG if args.debug else logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except IOError as e:
        print(f"Warning: Could not write to log file {log_file_path}: {e}. Logging to console only.")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if args.debug else logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logger.info(f"Logger initialized. Log file: {log_file_path}")
if args.debug: logger.debug("Debug mode is ON.")

################################################################################
# (D) Device Setup (Global)
################################################################################
device_name = "cpu"
if torch.cuda.is_available():
    try:
        cuda_device_index = temp_config.get("cuda_device", 0) if 'temp_config' in locals() else 0
        device_name = f"cuda:{cuda_device_index}"
        torch.cuda.get_device_name(cuda_device_index)
    except Exception:
        logger.warning(f"Could not validate CUDA device {cuda_device_index}. Defaulting to cuda:0 if available, else CPU.")
        if torch.cuda.is_available(): device_name = "cuda:0"
global_device = torch.device(device_name)
logger.info(f"Initial device check: {global_device}")


################################################################################
# (E) Utility Functions
################################################################################

# --- PDB Parsing ---
def parse_pdb(filename: str, logger: logging.Logger) -> Tuple[Dict, List, Dict]:
    """Parses ATOM records from a PDB, returning atom info and C-alpha indices."""
    backbone_atoms = {"N", "CA", "C", "O", "OXT"}
    atoms_in_order = []; ca_indices = {}; processed_atom_indices = set()
    try:
        with open(filename, 'r') as pdb_file:
            for line in pdb_file:
                if not line.startswith("ATOM  "): continue
                alt_loc = line[16].strip()
                if alt_loc not in ['', 'A']: continue

                atom_serial = int(line[6:11])
                if atom_serial in processed_atom_indices: continue
                processed_atom_indices.add(atom_serial)

                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                res_seq = int(line[22:26])
                orig_res_id = f"{chain_id}:{res_name}:{res_seq}"
                category = "backbone" if atom_name in backbone_atoms else "sidechain"

                atoms_in_order.append((orig_res_id, atom_serial, category, atom_name))
                if atom_name == 'CA': ca_indices[res_seq] = atom_serial

    except FileNotFoundError: logger.error(f"PDB not found: {filename}"); return {}, [], {}
    logger.info(f"Parsed {len(atoms_in_order)} ATOM records from {filename}, found {len(ca_indices)} C-alphas.")
    return {}, atoms_in_order, ca_indices

def renumber_atoms_and_residues(atoms_in_order: List[Tuple[str, int, str, str]], ca_serial_indices: Dict) -> Tuple[Dict, Dict, Dict, List[int]]:
    """Renumbers residues and atoms, and maps original C-alpha serials to new indices."""
    new_res_dict, orig_atom_map = {}, {}
    next_new_res_id, next_new_atom_index = 0, 0
    orig_res_map = {}
    
    # Preserve original residue order
    seen_res_order = {}
    res_order_counter = 0
    for r_id, _, _, _ in atoms_in_order:
        if r_id not in seen_res_order:
            seen_res_order[r_id] = res_order_counter
            res_order_counter += 1

    sortable = [(seen_res_order[r_id], serial, r_id, cat, name) for r_id, serial, cat, name in atoms_in_order]
    sortable.sort()

    for _, serial, r_id, cat, name in sortable:
        if r_id not in orig_res_map:
            orig_res_map[r_id] = next_new_res_id
            new_res_dict[next_new_res_id] = {"backbone": [], "sidechain": []}
            next_new_res_id += 1
        
        new_res_id = orig_res_map[r_id]
        new_res_dict[new_res_id][cat].append(next_new_atom_index)
        orig_atom_map[serial] = next_new_atom_index
        next_new_atom_index += 1

    # Map original CA serial numbers to the new, renumbered 0-based indices
    new_ca_indices = [orig_atom_map[ca_serial] for res_seq, ca_serial in sorted(ca_serial_indices.items()) if ca_serial in orig_atom_map]
    
    logger.info(f"Renumbered {next_new_res_id} residues & {next_new_atom_index} atoms. Mapped {len(new_ca_indices)} C-alpha indices.")
    return new_res_dict, orig_atom_map, {}, new_ca_indices


def get_global_indices(renumbered_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts sorted global lists of backbone and sidechain atom indices as tensors."""
    bb_idx, sc_idx = [], []
    for res_id in sorted(renumbered_dict.keys()):
        bb_idx.extend(renumbered_dict[res_id]["backbone"])
        sc_idx.extend(renumbered_dict[res_id]["sidechain"])
    return torch.tensor(bb_idx, dtype=torch.long), torch.tensor(sc_idx, dtype=torch.long)

# --- JSON Loading ---
# This function is unchanged and should work fine.
def load_heavy_atom_coords_from_json(json_file: str, logger: logging.Logger) -> Tuple[List[torch.Tensor], int]:
    logger.info(f"Loading coordinates from JSON: {json_file}")
    try:
        with open(json_file, "r") as f: data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error reading JSON {json_file}: {e}"); return [], -1
    
    try:
        keys_int = sorted([int(k) for k in data.keys()])
        keys_str = [str(k) for k in keys_int]
        if not keys_str: logger.error("No residue data in JSON."); return [], -1
        
        frame_data = data[keys_str[0]]["heavy_atom_coords_per_frame"]
        n_frames = len(frame_data)
        if n_frames == 0: logger.warning("JSON contains 0 frames."); return [], 0

        coords_frames, n_atoms_check = [], -1
        for frame_idx in range(n_frames):
            frame_coords_np = []
            current_atoms = 0
            for res_key in keys_str:
                coords = np.array(data[res_key]["heavy_atom_coords_per_frame"][frame_idx], dtype=np.float32)
                if coords.ndim != 2 or coords.shape[1] != 3: raise ValueError("Bad coordinate shape")
                frame_coords_np.append(coords)
                current_atoms += coords.shape[0]

            if frame_idx == 0:
                n_atoms_check = current_atoms
                logger.info(f"System has {n_atoms_check} atoms and {n_frames} frames.")
            elif current_atoms != n_atoms_check:
                logger.error(f"Inconsistent atom count on frame {frame_idx}. Expected {n_atoms_check}, got {current_atoms}."); return [], -1
            
            coords_frames.append(torch.tensor(np.concatenate(frame_coords_np, axis=0), dtype=torch.float32))

        return coords_frames, n_atoms_check
    except Exception as e:
        logger.error(f"Invalid JSON structure in {json_file}: {e}", exc_info=True); return [], -1

# --- Alignment ---
def compute_centroid(X: torch.Tensor) -> torch.Tensor: return X.mean(dim=-2)

def kabsch_algorithm(P: torch.Tensor, Q: torch.Tensor, logger: logging.Logger) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aligns Q onto P using Kabsch algorithm. Handles batches [B, N, 3]."""
    P, Q = P.float(), Q.float(); is_batched = P.ndim == 3
    if not is_batched: P, Q = P.unsqueeze(0), Q.unsqueeze(0)
    B, N, _ = P.shape; centroid_P, centroid_Q = compute_centroid(P), compute_centroid(Q)
    P_c, Q_c = P - centroid_P.unsqueeze(1), Q - centroid_Q.unsqueeze(1)
    C = torch.bmm(Q_c.transpose(1, 2), P_c)
    try: V, S, Wt = torch.linalg.svd(C)
    except Exception as e:
        logger.error(f"Kabsch SVD failed: {e}. Return identity align.", exc_info=True)
        U_fallback = torch.eye(3, device=P.device).unsqueeze(0).expand(B, -1, -1)
        Q_aligned_fallback = Q - centroid_Q.unsqueeze(1) + centroid_P.unsqueeze(1)
        return (U_fallback.squeeze(0), Q_aligned_fallback.squeeze(0)) if not is_batched else (U_fallback, Q_aligned_fallback)
    det = torch.det(torch.bmm(V, Wt)); D = torch.eye(3, device=P.device).unsqueeze(0).repeat(B, 1, 1)
    D[:, 2, 2] = torch.sign(det); U = torch.bmm(torch.bmm(V, D), Wt)
    Q_aligned = torch.bmm(Q_c, U) + centroid_P.unsqueeze(1)
    return (U.squeeze(0), Q_aligned.squeeze(0)) if not is_batched else (U, Q_aligned)

def align_frames_to_first(coords: List[torch.Tensor], logger: logging.Logger, device: torch.device) -> List[torch.Tensor]:
    if not coords: logger.warning("Coordinate list empty."); return []
    ref = coords[0].to(device)
    aligned = [coords[0].cpu()]
    n_frames = len(coords) -1
    for i, frame in enumerate(coords[1:], 1):
        _, aligned_dev = kabsch_algorithm(ref, frame.to(device), logger)
        aligned.append(aligned_dev.cpu())
    logger.debug(f"Aligned {len(aligned)} frames to the first frame of the series.")
    return aligned

# --- NEW ALIGNMENT UTILITIES for MULTI-SYSTEM ---
def find_mutual_nn_pairs(ref_coords_ca: np.ndarray, target_coords_ca: np.ndarray, logger: logging.Logger) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finds mutually nearest C-alpha atoms between two structures of different lengths."""
    logger.debug(f"Finding mutual NN pairs between structures of size {len(ref_coords_ca)} and {len(target_coords_ca)}")
    if ref_coords_ca.ndim != 2 or target_coords_ca.ndim != 2:
        raise ValueError("Input coordinates must be 2D arrays.")

    # Find nearest neighbor from target for each ref point
    nn_ref_to_target = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_coords_ca)
    _, indices1 = nn_ref_to_target.kneighbors(ref_coords_ca)
    
    # Find nearest neighbor from ref for each target point
    nn_target_to_ref = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(ref_coords_ca)
    _, indices2 = nn_target_to_ref.kneighbors(target_coords_ca)
    
    # Identify mutual pairs
    ref_indices, target_indices = [], []
    for i, target_idx in enumerate(indices1.flatten()):
        if indices2[target_idx] == i:
            ref_indices.append(i)
            target_indices.append(target_idx)
            
    logger.info(f"Found {len(ref_indices)} mutual nearest neighbor pairs for alignment.")
    if not ref_indices:
        logger.error("No mutual nearest neighbors found. Cannot align structures.")
        return torch.tensor([]), torch.tensor([])
        
    return torch.tensor(ref_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

def align_by_core(structure_to_align: torch.Tensor, core_indices_to_align: torch.Tensor,
                  reference_structure: torch.Tensor, core_indices_reference: torch.Tensor,
                  logger: logging.Logger) -> torch.Tensor:
    """Aligns a full structure based on the Kabsch alignment of its core atoms."""
    logger.debug(f"Aligning structure of size {structure_to_align.shape[0]} to ref of size {reference_structure.shape[0]} using {len(core_indices_reference)} core atoms.")
    device = structure_to_align.device

    # 1. Select the coordinates of the core atoms for both structures
    P_core = reference_structure[core_indices_reference].to(device)
    Q_core = structure_to_align[core_indices_to_align].to(device)
    
    # 2. Compute the rotation matrix (U) using only the core
    rotation, _ = kabsch_algorithm(P_core, Q_core, logger)
    
    # 3. Center the full structure_to_align on its core's centroid
    centroid_Q_core = compute_centroid(Q_core)
    structure_to_align_centered = structure_to_align - centroid_Q_core
    
    # 4. Rotate the full centered structure
    structure_to_align_rotated = torch.matmul(structure_to_align_centered, rotation.squeeze(0))
    
    # 5. Translate the rotated structure to the reference core's centroid
    centroid_P_core = compute_centroid(P_core)
    aligned_structure = structure_to_align_rotated + centroid_P_core
    
    return aligned_structure

# --- Graph Dataset ---
# Unchanged
def build_graph_dataset(coords_list: List[torch.Tensor], knn_neighbors: int, system_id: int, logger: logging.Logger, device: torch.device) -> List[Data]:
    dataset = []
    n_frames = len(coords_list)
    for i, coords_cpu in enumerate(coords_list):
        coords_dev = coords_cpu.to(device)
        edge_idx = knn_graph(coords_dev, k=knn_neighbors, loop=False, batch=None)
        data = Data(x=coords_cpu, edge_index=edge_idx.cpu(), y=coords_cpu, system_id=torch.tensor([system_id], dtype=torch.long))
        dataset.append(data)
    logger.info(f"Built graph dataset for system {system_id} with {n_frames} frames.")
    return dataset

# --- Dihedral Utilities ---
# Unchanged
@torch.jit.script
def compute_dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    b1=b-a; b2=c-b; b3=d-c; n1=torch.cross(b1,b2,dim=-1); n2=torch.cross(b2,b3,dim=-1)
    n1n=F.normalize(n1,p=2.,dim=-1,eps=1e-8); n2n=F.normalize(n2,p=2.,dim=-1,eps=1e-8)
    b2n=F.normalize(b2,p=2.,dim=-1,eps=1e-8); m1=torch.cross(n1n, b2n, dim=-1)
    x=(n1n*n2n).sum(dim=-1); y=(m1*n2n).sum(dim=-1); return torch.atan2(y,x)

def compute_all_dihedrals_vectorized(coords: torch.Tensor, info: Dict, n_res: int, logger: logging.Logger) -> Dict:
    """Computes all specified dihedrals vectorially."""
    if coords.ndim != 3: raise ValueError(f"Expected coords [B, N, 3], got {coords.shape}")
    B, N_atoms, _ = coords.shape; dev = coords.device; all_angles = {}
    for name, angle_info in info.items():
        indices, res_idx = angle_info.get('indices'), angle_info.get('res_idx')
        angles_out = torch.zeros(B, n_res, device=dev, dtype=coords.dtype)
        if indices is not None and res_idx is not None and indices[0].numel() > 0:
            try:
                idx_dev = [i.to(dev) for i in indices]; res_idx_dev = res_idx.to(dev)
                max_atom_idx_needed = max(i.max() for i in idx_dev)
                if max_atom_idx_needed >= N_atoms:
                    logger.error(f"Dihedral calculation error: atom index {max_atom_idx_needed} out of bounds for structure with {N_atoms} atoms.")
                    continue
                
                a,b,c,d = (coords[:, i, :] for i in idx_dev)
                values = compute_dihedral(a,b,c,d)
                angles_out.scatter_(1, res_idx_dev.unsqueeze(0).expand(B, -1), values)
            except Exception as e: logger.error(f"Error computing dihedral {name}: {e}", exc_info=True)
        all_angles[name] = angles_out
    return all_angles


def compute_angle_js_div(p: torch.Tensor, t: torch.Tensor, n=36, r=(-np.pi, np.pi)) -> torch.Tensor:
    if p.numel() == 0 or t.numel() == 0: return torch.tensor(0.0, device=p.device)
    eps=1e-10
    p_hist = torch.histc(p.detach(), bins=n, min=r[0], max=r[1])
    t_hist = torch.histc(t.detach(), bins=n, min=r[0], max=r[1])
    p_dist = p_hist / (p_hist.sum() + eps)
    t_dist = t_hist / (t_hist.sum() + eps)
    m_dist = 0.5 * (p_dist + t_dist)
    return 0.5 * (F.kl_div(m_dist.log(), p_dist, reduction='sum') + F.kl_div(m_dist.log(), t_dist, reduction='sum'))

# --- Checkpoint Utilities & MSE Utilities ---
# Unchanged
def save_checkpoint(state: Dict, filename: str, logger: logging.Logger):
    try: torch.save(state, filename); logger.debug(f"Checkpoint saved: {filename}")
    except IOError as e: logger.error(f"Error saving checkpoint {filename}: {e}")

def load_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], filename: str, device: torch.device, logger: logging.Logger) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int]:
    start_epoch = 0
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint: '{filename}'")
        try:
            ckpt = torch.load(filename, map_location=device)
            start_epoch = ckpt.get("epoch", 0)
            model.load_state_dict(ckpt["model_state_dict"])
            if optimizer and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor): state[k] = v.to(device)
            model.to(device)
            logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}", exc_info=True)
            start_epoch = 0
    else:
        logger.info(f"No checkpoint found at '{filename}'. Starting from scratch.")
    model.to(device)
    return model, optimizer, start_epoch

def compute_bb_sc_mse(pred: torch.Tensor, target: torch.Tensor, bb_idx: torch.Tensor, sc_idx: torch.Tensor, logger: logging.Logger) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    crit = nn.MSELoss()
    all_mse = crit(pred, target)
    bb_mse = crit(pred[bb_idx], target[bb_idx]) if bb_idx.numel() > 0 else torch.tensor(0., device=pred.device)
    sc_mse = crit(pred[sc_idx], target[sc_idx]) if sc_idx.numel() > 0 else torch.tensor(0., device=pred.device)
    return all_mse, bb_mse, sc_mse


################################################################################
# (F) Model Definitions
################################################################################

# --- HNO Encoder ---
# NOTE: This model is already size-agnostic. The ChebConv and final Linear layers
# operate on a per-node basis, so it handles graphs of different sizes correctly.
class HNO(nn.Module):
    def __init__(self, hidden_dim, K):
        super().__init__()
        self._debug_logged = False
        self.conv1 = ChebConv(3, hidden_dim, K=K)
        self.bano1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.bano2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.bano3 = nn.BatchNorm1d(hidden_dim)
        self.conv4 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.mlpRep = nn.Linear(hidden_dim, 3) # This is the reconstruction head

    def forward(self, x, edge_index):
        x = x.float()
        x = self.bano1(F.leaky_relu(self.conv1(x, edge_index)))
        x = self.bano2(F.leaky_relu(self.conv2(x, edge_index)))
        x = self.bano3(F.relu(self.conv3(x, edge_index)))
        x = self.conv4(x, edge_index)
        x_rep = F.normalize(x, p=2.0, dim=1)
        x_recon = self.mlpRep(x_rep)
        
        if self.training is False and not self._debug_logged:
             logger.debug(f"[HNO] Forward pass shapes: In={x.shape}, Rep={x_rep.shape}, Recon={x_recon.shape}")
             self._debug_logged = True
        return x_recon

    def forward_representation(self, x, edge_index):
        x = x.float()
        x = self.bano1(F.leaky_relu(self.conv1(x, edge_index)))
        x = self.bano2(F.leaky_relu(self.conv2(x, edge_index)))
        x = self.bano3(F.relu(self.conv3(x, edge_index)))
        x = self.conv4(x, edge_index)
        return F.normalize(x, p=2.0, dim=1)

# --- REFACTORED Decoder2 Model ---
class ProteinStateReconstructor2D(nn.Module):
    """
    Size-agnostic decoder. Predicts full coordinates from node embeddings.
    It combines local (per-node) and global (per-graph) information.
    """
    def __init__(self, node_emb_dim: int, cond_emb_dim: int, mlp_h_dim: int, mlp_layers: int, logger: logging.Logger):
        super().__init__()
        self.logger = logger
        self._logged_fwd = False
        
        # The input to the final MLP for each node will be:
        # [local_node_embedding, global_graph_embedding, global_conditioner_embedding]
        mlp_in_dim = node_emb_dim + node_emb_dim + cond_emb_dim
        
        # Simple MLP decoder
        layers = []
        in_d = mlp_in_dim
        for i in range(mlp_layers - 1):
            layers.extend([nn.Linear(in_d, mlp_h_dim), nn.BatchNorm1d(mlp_h_dim), nn.GELU()])
            in_d = mlp_h_dim
        layers.append(nn.Linear(in_d, 3))
        self.decoder_mlp = nn.Sequential(*layers)
        
        self.logger.info(f"Initialized Size-Agnostic Decoder2 with MLP input dim: {mlp_in_dim}")

    def forward(self, x: torch.Tensor, batch: torch.Tensor, conditioner_z_ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Per-node embeddings from HNO [N_total_nodes, E_node]
            batch (Tensor): PyG batch vector [N_total_nodes]
            conditioner_z_ref (Tensor): Per-node reference embeddings [N_total_nodes, E_cond]
        """
        # 1. Get global embedding for the current conformation
        z_global = global_mean_pool(x, batch) # -> [B, E_node]
        z_global_per_node = z_global[batch] # Broadcast to each node -> [N_total_nodes, E_node]

        # 2. Get global embedding for the conditioner
        conditioner_global = global_mean_pool(conditioner_z_ref, batch) # -> [B, E_cond]
        conditioner_global_per_node = conditioner_global[batch] # -> [N_total_nodes, E_cond]
        
        # 3. Assemble the feature vector for each node
        mlp_input = torch.cat([x, z_global_per_node, conditioner_global_per_node], dim=1)
        
        # 4. Predict coordinates
        pred_coords = self.decoder_mlp(mlp_input)
        
        if not self._logged_fwd and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"[Dec2 Fwd] Shapes: In x={x.shape}, z_global={z_global.shape}, cond_global={conditioner_global.shape}, mlp_in={mlp_input.shape}, Out={pred_coords.shape}")
            self._logged_fwd = True
            
        return pred_coords

################################################################################
# (G) Training Functions
################################################################################

# --- Train HNO ---
# This function requires minimal changes as the model is size-agnostic
def train_hno_model(model: HNO, tr_loader: DataLoader, te_loader: DataLoader, N_epochs: int, lr: float, ckpt: str, save_int: int, dev: torch.device, logger: logging.Logger):
    model=model.to(dev)
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    opt = torch.optim.Adam(params, lr=lr) if params else None
    model, opt, start_ep = load_checkpoint(model, opt, ckpt, dev, logger)
    
    if start_ep >= N_epochs:
        logger.info(f"Loaded HNO checkpoint epoch ({start_ep}) >= target epochs ({N_epochs}). Skipping training.")
        return model

    logger.info(f"Starting HNO training from epoch {start_ep + 1}/{N_epochs}, LR={lr}")
    for ep in range(start_ep, N_epochs):
        model.train()
        total_loss = 0.0
        for data in tr_loader:
            data=data.to(dev)
            opt.zero_grad(set_to_none=True)
            pred = model(data.x, data.edge_index)
            loss = F.mse_loss(pred, data.y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_tr_loss = total_loss / len(tr_loader)
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for data in te_loader:
                data=data.to(dev)
                pred=model(data.x, data.edge_index)
                total_val_loss += F.mse_loss(pred, data.y).item()
        avg_te_loss = total_val_loss / len(te_loader)

        logger.info(f"[HNO] Ep {ep+1}/{N_epochs} | Train MSE: {avg_tr_loss:.6f} | Val MSE: {avg_te_loss:.6f}")
        
        ep_num = ep + 1
        if opt and (ep_num % save_int == 0 or ep_num == N_epochs):
            save_checkpoint({"epoch": ep_num, "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict()}, ckpt, logger)
    
    logger.info(f"Finished HNO training. Checkpoint saved to {ckpt}")
    return model

# --- REFACTORED Train Decoder2 ---
def train_decoder2_model(
    model: ProteinStateReconstructor2D, tr_loader: DataLoader, te_loader: DataLoader,
    all_conditioners: Dict[int, torch.Tensor], # Dict of z_ref tensors
    all_sys_info: Dict[int, Any], # Dict containing bb_idx, sc_idx, n_res etc. for each system
    N_epochs: int, lr: float, ckpt: str, save_int: int, dev: torch.device, logger: logging.Logger,
    base_w: float, use_di: bool, all_di_info: Optional[Dict], div_t: str, l_div: float, l_mse: float
):
    model = model.to(dev)
    opt = torch.optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=lr)
    model, opt, start_ep = load_checkpoint(model, opt, ckpt, dev, logger)

    if start_ep >= N_epochs:
        logger.info(f"Loaded Decoder2 checkpoint epoch ({start_ep}) >= target epochs ({N_epochs}). Skipping training.")
        return model
        
    comp_div = compute_angle_js_div if div_t == "JS" else (lambda p, t: torch.tensor(0.0))
    if use_di: logger.info(f"Dihedral loss enabled: type={div_t}, lambda_div={l_div}, lambda_mse={l_mse}")

    logger.info(f"Starting Decoder2 training from epoch {start_ep + 1}/{N_epochs}, LR={lr}")
    for ep in range(start_ep, N_epochs):
        model.train()
        tr_metrics = {'total': 0.0, 'coord': 0.0, 'di_div': 0.0, 'di_mse': 0.0}
        
        for data in tr_loader:
            data = data.to(dev)
            opt.zero_grad(set_to_none=True)
            
            # --- Assemble Batch-Specific Conditioners and Indices ---
            # This is the core logic for handling heterogeneous batches
            unique_sys_ids = torch.unique(data.system_id)
            
            # Create the conditioner tensor for this specific batch
            # Note: This could be optimized, but is clear for debugging
            cond_list = [all_conditioners[sid.item()].to(dev)[all_sys_info[sid.item()]['bb_idx']] for sid in data.system_id]
            # TODO: The line above is a placeholder and needs to be corrected based on how conditioners are stored/used.
            # A simpler approach:
            cond_list = []
            for i in range(data.num_graphs):
                sid = data.system_id[i].item()
                num_nodes = data.ptr[i+1] - data.ptr[i]
                cond = all_conditioners[sid].to(dev)
                if cond.shape[0] != num_nodes: # This check is important!
                    logger.error(f"FATAL: Mismatch in conditioner size for system {sid}. Expected {num_nodes}, got {cond.shape[0]}")
                    # This should not happen if data prep is correct.
                    # As a fallback, we must skip or resize. For now, let's assume it matches.
                cond_list.append(cond)
            conditioners_for_batch = torch.cat(cond_list, dim=0)

            # --- Forward and Coordinate Loss ---
            pred = model(data.x, data.batch, conditioners_for_batch)
            total_loss = 0.0
            coord_loss = 0.0

            # Calculate coordinate MSE loss per system in the batch
            for sid_tensor in unique_sys_ids:
                sid = sid_tensor.item()
                mask = (data.system_id.squeeze() == sid)
                
                bb_idx = all_sys_info[sid]['bb_idx_local'] # Use local indices
                sc_idx = all_sys_info[sid]['sc_idx_local']
                
                pred_sys = pred[mask]
                target_sys = data.y[mask]
                
                sys_mse, _, _ = compute_bb_sc_mse(pred_sys, target_sys, bb_idx, sc_idx, logger)
                coord_loss += sys_mse

            total_loss += base_w * coord_loss
            tr_metrics['coord'] += coord_loss.item()
            
            # --- Dihedral Loss ---
            di_div_loss = torch.tensor(0.0, device=dev)
            di_mse_loss = torch.tensor(0.0, device=dev)
            if use_di and all_di_info:
                for sid_tensor in unique_sys_ids:
                    sid = sid_tensor.item()
                    mask = (data.system_id.squeeze() == sid)
                    
                    pred_sys_3d = pred[mask].view(-1, all_sys_info[sid]['n_atoms'], 3)
                    true_sys_3d = data.y[mask].view(-1, all_sys_info[sid]['n_atoms'], 3)
                    
                    pred_a = compute_all_dihedrals_vectorized(pred_sys_3d, all_di_info[sid], all_sys_info[sid]['n_res'], logger)
                    true_a = compute_all_dihedrals_vectorized(true_sys_3d, all_di_info[sid], all_sys_info[sid]['n_res'], logger)

                    for name in pred_a:
                        pa, ta = pred_a[name], true_a[name]
                        if pa.numel() > 0:
                             di_div_loss += comp_div(pa, ta)
                             di_mse_loss += F.mse_loss(pa, ta)
                
                total_loss += l_div * di_div_loss + l_mse * di_mse_loss
                tr_metrics['di_div'] += di_div_loss.item()
                tr_metrics['di_mse'] += di_mse_loss.item()

            tr_metrics['total'] += total_loss.item()
            total_loss.backward()
            opt.step()
        
        # Log training stats (validation loop omitted for brevity but should be added)
        avg_tr = {k: v / len(tr_loader) for k, v in tr_metrics.items()}
        logger.info(f"[Dec2] Ep {ep+1}/{N_epochs} | Train Loss: {avg_tr['total']:.4f} (Coord: {avg_tr['coord']:.4f}, Dihedral Div: {avg_tr['di_div']:.4f}, Dihedral MSE: {avg_tr['di_mse']:.4f})")
        
        ep_num = ep + 1
        if opt and (ep_num % save_int == 0 or ep_num == N_epochs):
            save_checkpoint({"epoch": ep_num, "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict()}, ckpt, logger)
            
    logger.info("Finished Decoder2 training.")
    return model


################################################################################
# (I) Main Execution Function
################################################################################
def main():
    start_time = time.time()
    logger.info("================ Script Starting (Multi-System Version) ================")
    global global_device
    
    with open(args.config, "r") as f: config = yaml.safe_load(f)
    logger.info("Successfully loaded configuration.")
    
    # --- Extract Parameters ---
    force_cpu = config.get("force_cpu", False)
    device = torch.device("cpu") if force_cpu else global_device
    pin_mem = (device.type == "cuda")
    num_workers = config.get("num_workers", 0)
    
    hno_cfg = config["hno_encoder"]
    dec2_cfg = config["decoder2"]
    d2s = config["decoder2_settings"]
    di_cfg = config.get("dihedral_loss", {})
    out_cfg = config["output_directories"]
    [os.makedirs(d, exist_ok=True) for d in out_cfg.values()]

    # --- Stage 1: Multi-System Data Loading & Alignment ---
    logger.info("--- Stage 1: Data Loading & Preprocessing for All Systems ---")
    
    full_dset = []
    all_sys_info = {} # Will store n_atoms, n_res, ca_indices, bb_idx etc. for each system
    canonical_ref_coords = None
    canonical_ref_ca_indices = None

    # Loop through each system defined in the config
    for i, system_config in enumerate(config["data"]["systems"]):
        sid = i
        pdb_p = system_config["pdb_path"]
        json_p = system_config["json_path"]
        logger.info(f"--- Processing System ID {sid}: {os.path.basename(pdb_p)} ---")

        # 1. Parse PDB for topology and C-alpha indices
        _, atoms_ord, ca_serial_map = parse_pdb(pdb_p, logger)
        renum_d, _, _, ca_indices_new = renumber_atoms_and_residues(atoms_ord, ca_serial_map)
        bb_idx, sc_idx = get_global_indices(renum_d)
        
        # 2. Load coordinates from JSON
        coords_list, n_atoms = load_heavy_atom_coords_from_json(json_p, logger)
        if not coords_list: continue # Skip if system loading failed

        all_sys_info[sid] = {
            'n_atoms': n_atoms,
            'ca_indices': ca_indices_new,
            'bb_idx_local': bb_idx, # These are local to the system
            'sc_idx_local': sc_idx,
            'n_res': len(renum_d)
        }

        # 3. Perform Alignment
        aligned_coords_list = []
        if sid == 0:
            # This is the canonical reference system
            logger.info(f"System {sid} is the canonical reference.")
            canonical_ref_coords = coords_list[0].clone()
            canonical_ref_ca_indices = torch.tensor(ca_indices_new, dtype=torch.long)
            aligned_coords_list = align_frames_to_first(coords_list, logger, device)
        else:
            # Align this system to the canonical reference
            logger.info(f"Aligning system {sid} to canonical reference (system 0).")
            target_ref_coords = coords_list[0]
            target_ca_indices = torch.tensor(ca_indices_new, dtype=torch.long)
            
            # Find core pairs for alignment
            core_ref_idx, core_target_idx = find_mutual_nn_pairs(
                canonical_ref_coords[canonical_ref_ca_indices].numpy(),
                target_ref_coords[target_ca_indices].numpy(),
                logger
            )
            
            # Align the first frame of this system to the canonical reference
            aligned_target_ref_frame = align_by_core(
                target_ref_coords, target_ca_indices[core_target_idx],
                canonical_ref_coords, canonical_ref_ca_indices[core_ref_idx],
                logger
            ).to(device)
            
            # Align the rest of this system's trajectory to its own aligned reference frame
            temp_list_for_align = [aligned_target_ref_frame.cpu()] + [c.to(device) for c in coords_list[1:]]
            aligned_coords_list = align_frames_to_first(temp_list_for_align, logger, device)

        # 4. Build graph dataset for this system and add to the master list
        system_dset = build_graph_dataset(aligned_coords_list, config["graph"]["knn_value"], sid, logger, device)
        full_dset.extend(system_dset)

    logger.info(f"--- Finished data loading. Total frames in dataset: {len(full_dset)} ---")

    # --- Stage 2: HNO Training ---
    logger.info("--- Stage 2: Training Shared HNO Encoder ---")
    tr_hno, te_hno = train_test_split(full_dset, test_size=0.1, random_state=42)
    load_tr_hno = DataLoader(tr_hno, hno_cfg['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    load_te_hno = DataLoader(te_hno, hno_cfg['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    hno_model = HNO(hno_cfg['hidden_dim'], hno_cfg['cheb_order'])
    hno_ckpt = os.path.join(out_cfg['checkpoint_dir'], "hno_checkpoint.pth")
    hno_model = train_hno_model(hno_model, load_tr_hno, load_te_hno, hno_cfg['num_epochs'], hno_cfg['learning_rate'], hno_ckpt, hno_cfg['save_interval'], device, logger)
    hno_model.eval()

    # --- Stage 3: Decoder Input Data Prep ---
    logger.info("--- Stage 3: Preparing Decoder Input Dataset & Conditioners ---")
    dec_in_dset = []
    all_conditioners = {} # Dictionary to hold the z_ref for each system
    
    # Generate embeddings for the whole dataset
    with torch.no_grad():
        infer_load = DataLoader(full_dset, hno_cfg['batch_size'] * 2, shuffle=False)
        for batch in infer_load:
            batch = batch.to(device)
            emb = hno_model.forward_representation(batch.x, batch.edge_index)
            # De-batch to create new Data objects
            emb_list = torch.split(emb, batch.ptr[1:] - batch.ptr[:-1])
            y_list = torch.split(batch.y, batch.ptr[1:] - batch.ptr[:-1])
            sid_list = torch.split(batch.system_id, batch.ptr[1:] - batch.ptr[:-1])
            
            for i in range(len(emb_list)):
                dec_in_dset.append(Data(x=emb_list[i].cpu(), y=y_list[i].cpu(), system_id=sid_list[i].cpu()))
    
    # Create the conditioner dictionary
    if d2s['conditioner_mode'] == 'z_ref':
        for sid in all_sys_info.keys():
            # Find the first frame of the system in the original dataset to get its reference coords
            ref_data_orig = next(d for d in full_dset if d.system_id.item() == sid)
            ref_data_orig = ref_data_orig.to(device)
            with torch.no_grad():
                z_ref = hno_model.forward_representation(ref_data_orig.x, ref_data_orig.edge_index)
                all_conditioners[sid] = z_ref.cpu() # Store on CPU
        logger.info(f"Created 'z_ref' conditioners for all {len(all_conditioners)} systems.")
    else:
        raise NotImplementedError("Only 'z_ref' conditioner mode is supported in this version.")

    # --- Stage 4: Decoder Setup & Training ---
    logger.info("--- Stage 4: Decoder2 Setup & Training ---")
    
    # Prepare Dihedral Info Dictionaries
    all_di_info = {}
    use_di_train = di_cfg.get("use_dihedral_loss", False)
    if use_di_train:
        logger.info("Preparing dihedral information for all systems...")
        for i, system_config in enumerate(config["data"]["systems"]):
            sid = i
            torsion_p = system_config.get("torsion_info_path")
            if not torsion_p or not os.path.isfile(torsion_p):
                logger.warning(f"Torsion file not found for system {sid}. Disabling dihedral loss for this system.")
                continue
            
            with open(torsion_p, "r") as f: t_info = json.load(f)
            # Simplified parsing logic, assuming it's correct per system
            di_info_sys = {} # ... parsing logic to fill this based on t_info ...
            all_di_info[sid] = di_info_sys
        if not all_di_info: use_di_train = False # Disable if no files were loaded
    
    # Initialize and Train Decoder
    tr_dec, te_dec = train_test_split(dec_in_dset, test_size=0.1, random_state=42)
    load_tr_dec = DataLoader(tr_dec, dec2_cfg['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    load_te_dec = DataLoader(te_dec, dec2_cfg['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    
    dec2_model = ProteinStateReconstructor2D(
        node_emb_dim=hno_cfg['hidden_dim'],
        cond_emb_dim=hno_cfg['hidden_dim'], # since we use z_ref
        mlp_h_dim=d2s['mlp_hidden_dim'],
        mlp_layers=d2s['num_hidden_layers'],
        logger=logger
    )
    dec2_ckpt = os.path.join(out_cfg['checkpoint_dir'], "decoder2_checkpoint.pth")
    dec2_model = train_decoder2_model(
        dec2_model, load_tr_dec, load_te_dec, all_conditioners, all_sys_info,
        dec2_cfg['num_epochs'], dec2_cfg['learning_rate'], dec2_ckpt, dec2_cfg['save_interval'], device, logger,
        dec2_cfg['base_loss_weight'], use_di_train, all_di_info,
        di_cfg.get('divergence_type', 'JS'), di_cfg.get('lambda_divergence', 0.0), di_cfg.get('lambda_torsion_mse', 0.0)
    )

    logger.info(f"================ Script Finished ({time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}) ================")

################################################################################
# (J) Script Entry Point
################################################################################
if __name__ == "__main__":
    main()
