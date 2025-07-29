#!/usr/bin/env python3
import argparse
import os
import yaml
import logging
import torch
import numpy as np
import h5py
import torch.nn as nn
import pathlib

# ===================================================================
# (A) Argument Parsing & Setup
# ===================================================================

parser = argparse.ArgumentParser(description="Multi-System Structure Generator from Diffusion Embeddings")
parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file.')
parser.add_argument('--debug', action='store_true', help='Enable debug level logging.')
args = parser.parse_args()

# Setup logging
log_level = logging.DEBUG if args.debug else logging.INFO
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler()])
logger = logging.getLogger()

logger.info("Multi-System Structure Generator Script Started")

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
CONDITIONERS_H5_PATH = pathlib.Path(paths['conditioners_h5_path'])
GEN_EMBEDDINGS_DIR = pathlib.Path(paths['generated_embeddings_dir'])
OUTPUT_H5_PATH = pathlib.Path(paths['output_structures_h5_path'])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Ensure output directory exists
OUTPUT_H5_PATH.parent.mkdir(parents=True, exist_ok=True)

# ===================================================================
# (C) Model Definition (Must match the trained model)
# ===================================================================

# This class is a simplified version of the one in chebnet_multi_final.py
# It is redefined here to make this script standalone for generation.
class ProteinStateReconstructor2D(nn.Module):
    def __init__(self, node_emb_dim: int, cond_emb_dim: int,
                 output_height: int, output_width: int,
                 mlp_h_dim: int, mlp_layers: int, logger: logging.Logger):
        super().__init__()
        self.logger = logger
        # The pooling layer is not used in this simplified forward pass, but its
        # parameters are kept to correctly calculate the mlp_in_dim, ensuring
        # that the architecture matches the saved checkpoint weights.
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
# (D) Main Generation Logic
# ===================================================================

def main():
    # 1. Instantiate and load the trained decoder model
    logger.info("Loading trained decoder model...")
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

    # 2. Find all generated embedding files to process
    if not GEN_EMBEDDINGS_DIR.is_dir():
        logger.error(f"Generated embeddings directory not found: {GEN_EMBEDDINGS_DIR}"); exit(1)
    
    embedding_files = sorted(list(GEN_EMBEDDINGS_DIR.glob("generated_embeddings_sys*.h5")))
    if not embedding_files:
        logger.error(f"No generated embedding files found in {GEN_EMBEDDINGS_DIR}"); exit(1)
    
    logger.info(f"Found {len(embedding_files)} generated embedding files to process.")

    # 3. NEW: Use a nested dictionary for aggregation: {sid: {exp_id: [coords]}}
    aggregated_coords = {}

    # 4. Loop through each file, generate structures, and store in the dictionary
    for emb_file in embedding_files:
        try:
            # --- Robustly parse SID and EXP_ID from filename ---
            filename_parts = emb_file.stem.split('_')
            sid_part = next((part for part in filename_parts if part.startswith('sys')), None)
            exp_part = next((part for part in filename_parts if part.startswith('exp')), None)

            if not sid_part or not exp_part:
                logger.warning(f"Could not parse SID and EXP from filename {emb_file.name}. Skipping.")
                continue
            
            sid = int(sid_part.replace('sys', ''))
            exp_id = int(exp_part.replace('exp', ''))
            logger.info(f"--- Processing SID: {sid}, EXP: {exp_id} from file {emb_file.name} ---")

            # a. Load the corresponding z_ref conditioner
            with h5py.File(CONDITIONERS_H5_PATH, 'r') as f_cond:
                group_name = f"system_{sid}"
                if group_name not in f_cond:
                    logger.warning(f"Group '{group_name}' not in conditioner file. Skipping.")
                    continue
                z_ref = torch.from_numpy(f_cond[group_name]['z_ref'][:]).float().to(DEVICE)

            # b. Load the generated pooled embeddings
            with h5py.File(emb_file, 'r') as f_emb:
                group_key = next((k for k in f_emb.keys() if k.startswith('system_')), None)
                if not group_key: continue
                dset_name = "embeddings" if "embeddings" in f_emb[group_key] else "generated_embeddings"
                if dset_name not in f_emb[group_key]: continue
                pooled_embeddings = torch.from_numpy(f_emb[group_key][dset_name][:]).float().to(DEVICE)
            
            logger.info(f"Loaded {pooled_embeddings.shape[0]} embeddings and conditioner of shape {z_ref.shape}")

            # c. Decode each embedding and store in nested dictionary
            if sid not in aggregated_coords:
                aggregated_coords[sid] = {}
            if exp_id not in aggregated_coords[sid]:
                aggregated_coords[sid][exp_id] = []

            with torch.no_grad():
                for i in range(pooled_embeddings.shape[0]):
                    pooled_emb = pooled_embeddings[i]
                    coords = model(pooled_emb, z_ref).cpu().numpy()
                    aggregated_coords[sid][exp_id].append(coords)

        except Exception as e:
            logger.error(f"Failed to process file {emb_file.name}: {e}", exc_info=True)

    # 5. NEW: Write the aggregated results using the nested structure
    with h5py.File(OUTPUT_H5_PATH, 'w') as f_out:
        logger.info(f"Writing aggregated structures to {OUTPUT_H5_PATH}")
        for sid, exp_data in sorted(aggregated_coords.items()):
            sys_group = f_out.create_group(f"system_{sid}")
            logger.info(f"Created group for system {sid}")
            for exp_id, coords_list in sorted(exp_data.items()):
                if not coords_list: continue
                
                stacked_coords = np.array(coords_list)
                exp_group = sys_group.create_group(f"exp_{exp_id}")
                exp_group.create_dataset("coords", data=stacked_coords, compression="gzip")
                logger.info(f"  - Saved {stacked_coords.shape[0]} structures for exp_{exp_id}")

    logger.info("--- All systems processed. Final structures saved. ---")


if __name__ == "__main__":
    main()
