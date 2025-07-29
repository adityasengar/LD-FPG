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

# ===================================================================
# (A) Argument Parsing & Setup
# ===================================================================

parser = argparse.ArgumentParser(description="Decode Novel Latent Embeddings to 3D Structures")
parser.add_argument('--config', type=str, required=True, help='Path to the decoding YAML configuration file.')
parser.add_argument('--input', type=str, required=True, help='Path to the HDF5 file from generate_from_new_pdb.py.')
parser.add_argument('--output', type=str, required=True, help='Path for the final output HDF5 file containing coordinates.')
parser.add_argument('--debug', action='store_true', help='Enable debug level logging.')
args = parser.parse_args()

# Setup logging
log_level = logging.DEBUG if args.debug else logging.INFO
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler()])
logger = logging.getLogger()

logger.info("Novel Latent Decoder Script Started")

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
d2s = config.get('decoder2_settings', {})

DECODER_CKPT_PATH = pathlib.Path(paths['decoder_checkpoint_path'])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ===================================================================
# (C) Model Definition (Must match trained Decoder)
# ===================================================================

class ProteinStateReconstructor2D(nn.Module):
    def __init__(self, node_emb_dim: int, cond_emb_dim: int,
                 output_height: int, output_width: int,
                 mlp_h_dim: int, mlp_layers: int, logger: logging.Logger):
        super().__init__()
        self.logger = logger
        # The pooling layer is not used in the forward pass here, but its parameters
        # are kept to correctly calculate the mlp_in_dim, ensuring that the
        # architecture matches the saved checkpoint weights.
        self.pool_layer = nn.AdaptiveAvgPool2d((output_height, output_width))
        pooled_dim = output_height * output_width
        mlp_in_dim = cond_emb_dim + pooled_dim

        layers = []
        in_d = mlp_in_dim
        for i in range(mlp_layers - 1):
            layers.extend([nn.Linear(in_d, mlp_h_dim), nn.BatchNorm1d(mlp_h_dim), nn.GELU()])
            in_d = mlp_h_dim
        layers.append(nn.Linear(in_d, 3))
        self.decoder_mlp = nn.Sequential(*layers)
        self.logger.info(f"Instantiated Decoder. MLP Input Dim: {mlp_in_dim}")

    def forward(self, pooled_embedding: torch.Tensor, conditioner_z_ref: torch.Tensor) -> torch.Tensor:
        """
        A corrected forward pass for generation from a pre-pooled embedding.
        'pooled_embedding' is the direct output from the diffusion model.
        'conditioner_z_ref' is the static per-atom conditioner.
        """
        num_atoms = conditioner_z_ref.shape[0]
        
        # The input 'pooled_embedding' is already the correct, flattened, pooled vector.
        # We just need to broadcast it to each atom.
        # It should have shape [pooled_dim], we unsqueeze to [1, pooled_dim] for expand.
        pooled_per_node = pooled_embedding.unsqueeze(0).expand(num_atoms, -1)
        
        # Concatenate the static conditioner with the broadcasted pooled vector
        mlp_input = torch.cat([conditioner_z_ref, pooled_per_node], dim=1)
        
        # Predict coordinates
        pred_coords = self.decoder_mlp(mlp_input)
        return pred_coords

# ===================================================================
# (D) Main Decoding Logic
# ===================================================================

@torch.no_grad()
def main():
    # 1. Load Decoder Model
    logger.info("Loading pre-trained decoder model...")
    if not DECODER_CKPT_PATH.is_file():
        logger.error(f"Decoder checkpoint not found at {DECODER_CKPT_PATH}"); exit(1)

    model = ProteinStateReconstructor2D(
        node_emb_dim=d2s['node_emb_dim'],
        cond_emb_dim=d2s['cond_emb_dim'],
        output_height=d2s['output_height'],
        output_width=d2s['output_width'],
        mlp_h_dim=d2s['mlp_hidden_dim'],
        mlp_layers=d2s['num_hidden_layers'],
        logger=logger
    ).to(DEVICE)

    checkpoint = torch.load(DECODER_CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Decoder model loaded successfully.")

    # 2. Load Input Data
    input_h5_path = pathlib.Path(args.input)
    if not input_h5_path.is_file():
        logger.error(f"Input HDF5 file not found: {input_h5_path}"); exit(1)
        
    logger.info(f"Loading data from {input_h5_path}...")
    with h5py.File(input_h5_path, 'r') as f:
        pooled_embeddings = torch.from_numpy(f['generated_pooled_embeddings'][:]).float().to(DEVICE)
        novel_z_ref = torch.from_numpy(f['novel_z_ref_conditioner'][:]).float().to(DEVICE)
    
    logger.info(f"Loaded {pooled_embeddings.shape[0]} embeddings and a conditioner of shape {novel_z_ref.shape}")

    # 3. Decode each embedding to a structure
    all_generated_coords = []
    for i in range(pooled_embeddings.shape[0]):
        pooled_emb = pooled_embeddings[i]
        coords = model(pooled_emb, novel_z_ref)
        all_generated_coords.append(coords.cpu().numpy())
    
    # 4. Save final coordinates
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stacked_coords = np.array(all_generated_coords)
    
    with h5py.File(output_path, 'w') as f_out:
        f_out.create_dataset("generated_coords", data=stacked_coords, compression="gzip")
    
    logger.info(f"Decoding complete. Saved {stacked_coords.shape[0]} structures to {output_path}")

if __name__ == "__main__":
    main()
