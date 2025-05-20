#!/usr/bin/env python3
import os
import sys
import yaml
import h5py
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any

# Need imports for classes/functions used
from torch_geometric.nn import ChebConv # For HNO definition
from torch_cluster import knn_graph # Needed if calculating z_ref

##############################################################
# Copied Class Definitions from Script M (Response #16)
##############################################################


class HNO(nn.Module):
    def __init__(self, hidden_dim, K):
        super().__init__()
        self._debug_logged = False  # For one-time debug logging
        logger.debug(f"Initializing HNO with hidden_dim={hidden_dim}, K={K}")
        sys.stdout.flush()
        # Input dimension is 3 (x, y, z coordinates)
        self.conv1 = ChebConv(3, hidden_dim, K=K)
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.conv3 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.conv4 = ChebConv(hidden_dim, hidden_dim, K=K)
        # BatchNorm applied on the feature dimension (hidden_dim)
        self.bano1 = nn.BatchNorm1d(hidden_dim)
        self.bano2 = nn.BatchNorm1d(hidden_dim)
        self.bano3 = nn.BatchNorm1d(hidden_dim)
        # Final MLP maps back to 3D coordinates
        self.mlpRep = nn.Linear(hidden_dim, 3)

    def forward(self, x, edge_index, log_debug=False):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = self.bano1(F.leaky_relu(x))
        x = self.conv2(x, edge_index)
        x = self.bano2(F.leaky_relu(x))
        x = self.conv3(x, edge_index)
        x = self.bano3(F.relu(x)) # Note: ReLU here, LeakyReLU before
        x = self.conv4(x, edge_index)
        x = F.normalize(x, p=2.0, dim=1) # dim=1 is the feature dimension
        x = self.mlpRep(x)
        return x

    def forward_representation(self, x, edge_index, log_debug=False):
        x = x.float() # Ensure input is float
        x = self.conv1(x, edge_index)
        x = self.bano1(F.leaky_relu(x))
        x = self.conv2(x, edge_index)
        x = self.bano2(F.leaky_relu(x))
        x = self.conv3(x, edge_index)
        x = self.bano3(F.relu(x))
        x = self.conv4(x, edge_index)
        x = F.normalize(x, p=2.0, dim=1)
        return x

# --- MLP Builder ---
def build_decoder_mlp(in_dim: int, out_dim: int, N_layers: int, h_dim: int = 128) -> nn.Sequential:
    """Builds MLP with BatchNorm."""
    layers: List[nn.Module] = []; curr = in_dim
    if N_layers<=0: raise ValueError("MLP layers must be >= 1.")
    elif N_layers==1: layers.append(nn.Linear(curr, out_dim))
    else:
        layers.extend([nn.Linear(curr, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU()]); curr = h_dim
        for _ in range(N_layers - 2): layers.extend([nn.Linear(curr, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU()])
        layers.append(nn.Linear(curr, out_dim))
    return nn.Sequential(*layers)

# --- Decoder2 Model Definition (Needed for loading state_dict) ---
class ProteinStateReconstructor2D(nn.Module):
    _logged_fwd = False
    def __init__(self, in_dim: int, N_nodes: int, cond_dim: int, pool_type: str = "blind", res_indices: Optional[List[List[int]]] = None, pool_size: Tuple[int, int] = (20, 4), mlp_h_dim: int = 128, mlp_layers: int = 2, pool2_size: Optional[Tuple[int, int]] = None, use_pool2: bool = False, use_attn: bool = False, attn_type: str = "global", logger: logging.Logger = logging.getLogger()): # Added logger default
        super().__init__(); self.N_nodes=N_nodes; self.in_dim=in_dim; self.cond_dim=cond_dim; self.logger=logger; self.pool_type=pool_type
        self.seg_indices: List[torch.LongTensor] = []
        if pool_type=="blind": self.seg_indices.append(torch.arange(N_nodes,dtype=torch.long))
        elif pool_type=="residue":
            if not res_indices: self.logger.warning("Residue pooling selected but no indices provided during init."); self.seg_indices.append(torch.arange(N_nodes,dtype=torch.long)) # Fallback?
            else: self.seg_indices = [torch.tensor(idx, dtype=torch.long) for idx in res_indices if idx];
            if not self.seg_indices: raise ValueError("Empty residue segments.")
        else: raise ValueError(f"Unknown pool_type={pool_type}")
        self.N_seg = len(self.seg_indices); #self.logger.info(f"Dec2 Init: Pool='{pool_type}', Segs={self.N_seg}")
        self.seg_pools = nn.ModuleList([nn.AdaptiveAvgPool2d(pool_size) for _ in self.seg_indices])
        self.prim_pool_dim = pool_size[0]*pool_size[1]; self.final_pool_dim = self.prim_pool_dim*self.N_seg # Calculate initial pooled dim based on primary pool concat
        self.glob_pool2 = None
        if use_pool2 and pool2_size and self.N_seg>0:
            self.glob_pool2=nn.AdaptiveAvgPool2d(pool2_size);
            self.final_pool_dim=pool2_size[0]*pool2_size[1]; # Recalculate if secondary pooling is used
            #self.logger.info(f"Dec2 Init: Use Pool2. FinalPoolDim={self.final_pool_dim}")
        #else: self.logger.info(f"Dec2 Init: Primary Pool only. ConcatPoolDim={self.final_pool_dim}")
        self.attn = None;
        if use_attn: self.logger.warning("Attn impl omitted.")
        mlp_in_dim=self.cond_dim+self.final_pool_dim; # Input to MLP is conditioner + final_pooled_dim
        self.decoder=build_decoder_mlp(mlp_in_dim, 3, mlp_layers, mlp_h_dim) # This is the part we need
        self.logger.info(f"Dec2 Init MLP: In={mlp_in_dim}, Out=3, Layers={mlp_layers}, Hidden={mlp_h_dim}")
    # get_pooled_latent and forward are NOT used directly in this script.
# --- End of Copied Class Definitions ---


##############################################################
# (D) Main Decoding Function
##############################################################
def main():
    parser = argparse.ArgumentParser(description="Decode diffused GLOBAL pooled embeddings using trained Decoder2 MLP.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config used for TRAINING the main script (Script M).")
    # Checkpoints from Script M training
    parser.add_argument("--hno_ckpt", type=str, default="checkpoints/hno_checkpoint.pth", help="Path to trained HNO ckpt (needed only if conditioner_mode='z_ref' AND z_ref file is missing).")
    parser.add_argument("--decoder2_ckpt", type=str, default="checkpoints/decoder2_checkpoint.pth", help="Path to trained Decoder2 ckpt (from Script M).")
    # Inputs: Diffused embeddings and Conditioner references from Script M
    parser.add_argument("--diff_emb_file", type=str, required=True, help="HDF5 file with diffused embeddings (output from diffusion script).")
    parser.add_argument("--diff_emb_key", type=str, default="generated_embeddings", help="Dataset key for diffused embeddings in HDF5 file.")
    parser.add_argument("--conditioner_x_ref_pt", type=str, default="structures/X_ref_coords.pt", help="Path to saved X_ref_coords.pt (generated by Script M).")
    parser.add_argument("--conditioner_z_ref_pt", type=str, default="latent_reps/z_ref_embedding.pt", help="Path to saved z_ref_embedding.pt (generated by Script M, required if conditioner_mode='z_ref').")
    # Output
    parser.add_argument("--output_file", type=str, default="structures/full_coords_diff.h5", help="Output HDF5 file for decoded diffusion coordinates.")
    parser.add_argument("--output_key", type=str, default="full_coords_diff", help="Dataset key for output coords in HDF5 file.")
    # Optional
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for decoding.")
    parser.add_argument("--cuda_device", type=int, default=0, help="GPU device index if CUDA available.")

    args = parser.parse_args()

    # Basic Logging Setup for this script
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s')

    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
        try:
            torch.cuda.set_device(device) # Set default device
        except Exception as e:
            logging.warning(f"Could not set CUDA device {args.cuda_device}: {e}. Defaulting to cpu.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    logging.info(f"Using Device: {device}")

    # --- Load Training Config ---
    try:
        with open(args.config, "r") as f: config = yaml.safe_load(f)
        logging.info(f"Loaded training config from: {args.config}")
    except Exception as e: logging.error(f"Failed to load config: {e}"); sys.exit(1)

    # --- Extract Necessary Parameters from Config ---
    try:
        hno_hdim = config["hno_encoder"]["hidden_dim"]
        hno_k = config["hno_encoder"]["cheb_order"]
        graph_cfg = config["graph"]; knn = graph_cfg["knn_value"] # Needed if calculating z_ref
        d2s = config["decoder2_settings"]
        d2_c_mode = d2s.get("conditioner_mode","z_ref"); d2_p_type = d2s.get("pooling_type","blind")
        d2_ph, d2_pw = d2s.get("output_height",20), d2s.get("output_width",4); d2_mlp_h = d2s.get("mlp_hidden_dim",128)
        d2_mlp_l = d2s.get("num_hidden_layers",2); d2_use_p2 = d2s.get("use_second_level_pooling",False);
        d2_ph2, d2_pw2 = d2s.get("output_height2"), d2s.get("output_width2")
        # --- FIX: Added missing lines ---
        d2_use_a = d2s.get("use_cross_attention", False)
        d2_a_type = d2s.get("cross_attention_type", "global")
        # --- End Fix ---
    except KeyError as e: logging.error(f"Missing key in config: {e}"); sys.exit(1)

    # --- Load Conditioner Data ---
    try:
        X_ref_t = torch.load(args.conditioner_x_ref_pt, map_location='cpu').float() # Load X_ref [N, 3] on CPU
        N_atoms = X_ref_t.shape[0]
        logging.info(f"Loaded X_ref, N_atoms = {N_atoms}")
    except Exception as e: logging.error(f"Failed to load X_ref from {args.conditioner_x_ref_pt}: {e}"); sys.exit(1)

    conditioner_cpu = None; cond_dim = -1
    if d2_c_mode == "X_ref":
        conditioner_cpu = X_ref_t; cond_dim = 3; logging.info(f"Using X_ref conditioner (Dim={cond_dim})")
    elif d2_c_mode == "z_ref":
        if os.path.isfile(args.conditioner_z_ref_pt):
             try:
                  conditioner_cpu = torch.load(args.conditioner_z_ref_pt, map_location='cpu').float()
                  cond_dim = conditioner_cpu.shape[1]
                  logging.info(f"Loaded pre-computed z_ref conditioner (Dim={cond_dim})")
             except Exception as e: logging.error(f"Failed to load z_ref file {args.conditioner_z_ref_pt}: {e}"); sys.exit(1)
        else: # Need to compute z_ref
             logging.info(f"z_ref file not found ({args.conditioner_z_ref_pt}). Computing z_ref using HNO...")
             try:
                 hno_model = HNO(hidden_dim=hno_hdim, K=hno_k).to(device)
                 if not os.path.isfile(args.hno_ckpt): raise FileNotFoundError(f"HNO checkpoint not found at {args.hno_ckpt}")
                 hno_ckpt_data = torch.load(args.hno_ckpt, map_location=device)
                 hno_model.load_state_dict(hno_ckpt_data["model_state_dict"]); hno_model.eval()
                 logging.info(f"Loaded HNO model from: {args.hno_ckpt}")
                 logging.info("Calculating edge index for X_ref...")
                 edge_index = knn_graph(X_ref_t.to(device), k=knn, loop=False)
                 logging.info("Calculating z_ref via HNO forward_representation...")
                 with torch.no_grad(): z_ref_dev = hno_model.forward_representation(X_ref_t.to(device), edge_index)
                 conditioner_cpu = z_ref_dev.cpu(); cond_dim = conditioner_cpu.shape[1]
                 logging.info(f"Computed z_ref as conditioner (Dim={cond_dim})")
                 try: torch.save(conditioner_cpu, args.conditioner_z_ref_pt); logging.info(f"Saved computed z_ref to {args.conditioner_z_ref_pt}")
                 except Exception as save_e: logging.warning(f"Could not save computed z_ref: {save_e}")
                 del hno_model, edge_index # Clean up
                 if torch.cuda.is_available(): torch.cuda.empty_cache()
             except Exception as e: logging.error(f"Failed to compute z_ref: {e}"); sys.exit(1)
    else: logging.error(f"Invalid conditioner_mode: {d2_c_mode}"); sys.exit(1)
    if conditioner_cpu is None or conditioner_cpu.shape[0] != N_atoms: logging.error(f"Conditioner failed validation (None or wrong N_atoms {conditioner_cpu.shape[0]} vs {N_atoms})."); sys.exit(1)

    # --- Load Diffused Embeddings ---
    try:
        logging.info(f"Loading diffused embeddings: {args.diff_emb_file} [Key: {args.diff_emb_key}]")
        with h5py.File(args.diff_emb_file, "r") as f: diff_emb_np = f[args.diff_emb_key][:]
        diff_emb_t = torch.tensor(diff_emb_np, dtype=torch.float) # Keep on CPU initially for shape checks
        N_gen = diff_emb_t.shape[0]
        logging.info(f"Loaded diffusion embeddings shape: {diff_emb_t.shape}")
    except Exception as e: logging.error(f"Failed to load diffused embeddings: {e}"); sys.exit(1)

    # --- Instantiate Decoder2 to get parameters and MLP ---
    res_indices_dummy = [list(range(N_atoms))] if d2_p_type == "blind" else None
    if d2_p_type == "residue": logging.warning("Residue pooling selected, ensure diffusion embs match expected structure.")
    temp_logger = logging.getLogger("dummy"); temp_logger.setLevel(logging.CRITICAL)
    try:
        decoder2_instance_for_state = ProteinStateReconstructor2D(
            in_dim=hno_hdim, N_nodes=N_atoms, cond_dim=cond_dim, pool_type=d2_p_type,
            res_indices=res_indices_dummy, pool_size=(d2_ph, d2_pw), mlp_h_dim=d2_mlp_h,
            mlp_layers=d2_mlp_l, pool2_size=((d2_ph2, d2_pw2) if d2_use_p2 and d2_ph2 and d2_pw2 else None),
            use_pool2=d2_use_p2, use_attn=d2_use_a, attn_type=d2_a_type, logger=temp_logger
        )
        final_pool_dim_expected = decoder2_instance_for_state.final_pool_dim
        logging.info(f"Decoder expects final pooled dimension: {final_pool_dim_expected}")
    except Exception as e: logging.error(f"Failed to instantiate dummy Decoder2 model structure: {e}"); sys.exit(1)

    # --- Validate and Reshape loaded diffusion embeddings ---
    diff_emb_global_t = None # Will hold final [N_gen, final_pool_dim_expected] tensor
    if diff_emb_t.ndim == 3: # Shape [N_gen, R, PD_seg] - Assume from Script D
        logging.info("Loaded embeddings seem per-segment/residue. Flattening to global representation.")
        loaded_R, loaded_PD_seg = diff_emb_t.shape[1], diff_emb_t.shape[2]
        # Flatten to [N_gen, R * PD_seg]
        diff_emb_global_t = diff_emb_t.reshape(N_gen, -1)
        loaded_pooled_dim = diff_emb_global_t.shape[1]
        logging.info(f"Flattened per-segment embeddings to global shape: {diff_emb_global_t.shape}")
    elif diff_emb_t.ndim == 2: # Shape [N_gen, SomeDim] - Assume this is already global
        logging.info("Loaded embeddings seem global.")
        diff_emb_global_t = diff_emb_t
        loaded_pooled_dim = diff_emb_global_t.shape[1]
    else: logging.error(f"Loaded diffusion embeddings unexpected ndim={diff_emb_t.ndim}."); sys.exit(1)

    # Final dimension check
    if loaded_pooled_dim != final_pool_dim_expected:
        logging.error(f"Dimension mismatch! Loaded/Flattened diffusion emb dim ({loaded_pooled_dim}) != Expected decoder pooled dim ({final_pool_dim_expected}).")
        sys.exit(1)
    diff_emb_global_t = diff_emb_global_t.to(device) # Move final embeddings to device

    # --- Load Decoder2 Checkpoint and Extract MLP ---
    try:
        logging.info(f"Loading Decoder2 checkpoint from: {args.decoder2_ckpt}")
        decoder2_model = ProteinStateReconstructor2D( # Instantiate with correct parameters
            in_dim=hno_hdim, N_nodes=N_atoms, cond_dim=cond_dim, pool_type=d2_p_type,
            res_indices=res_indices_dummy, pool_size=(d2_ph, d2_pw), mlp_h_dim=d2_mlp_h,
            mlp_layers=d2_mlp_l, pool2_size=((d2_ph2, d2_pw2) if d2_use_p2 and d2_ph2 and d2_pw2 else None),
            use_pool2=d2_use_p2, use_attn=d2_use_a, attn_type=d2_a_type, logger=temp_logger
        ).to(device)
        dec2_ckpt_data = torch.load(args.decoder2_ckpt, map_location=device)
        decoder2_model.load_state_dict(dec2_ckpt_data["model_state_dict"])
        decoder_mlp = decoder2_model.decoder # Extract the MLP part
        decoder_mlp.eval()
        logging.info("Successfully loaded Decoder2 MLP weights.")
    except Exception as e: logging.error(f"Failed to load Decoder2 checkpoint/MLP: {e}"); sys.exit(1)

    # --- Prepare Conditioner for Batches ---
    conditioner_dev = conditioner_cpu.to(device) # Move final conditioner to device

    # --- Decoding Loop ---
    logging.info(f"Starting decoding for {N_gen} samples (Batch Size: {args.batch_size})...")
    all_coords_list = []
    with torch.no_grad():
        for i in range(0, N_gen, args.batch_size):
            b_start, b_end = i, min(i + args.batch_size, N_gen)
            bs = b_end - b_start

            pooled_batch = diff_emb_global_t[b_start:b_end] # [bs, final_pooled_dim]
            cond_expanded = conditioner_dev.unsqueeze(0).expand(bs, -1, -1) # [bs, N, Cdim]

            # Replicate the logic: Expand global pooled, concat with expanded conditioner
            pooled_expanded = pooled_batch.unsqueeze(1).expand(-1, N_atoms, -1) # [bs, N, final_pooled_dim]
            mlp_input_batch = torch.cat([cond_expanded, pooled_expanded], dim=-1) # [bs, N, Cdim + final_pool_dim]
            mlp_input_flat = mlp_input_batch.view(bs * N_atoms, -1)

            # Use the extracted MLP
            coords_flat = decoder_mlp(mlp_input_flat) # [bs*N, 3]
            coords_batch = coords_flat.view(bs, N_atoms, 3) # [bs, N, 3]

            all_coords_list.append(coords_batch.cpu().numpy())
            if (i // args.batch_size + 1) % 20 == 0: # Log progress less frequently
                 logging.info(f"  Decoded up to sample {b_end}/{N_gen}")

    # --- Concatenate and Save Results ---
    try:
        final_coords_np = np.concatenate(all_coords_list, axis=0)
        logging.info(f"Concatenated decoded coords shape: {final_coords_np.shape}")
        if final_coords_np.shape != (N_gen, N_atoms, 3):
             logging.warning("Final coords shape mismatch!") # Sanity check

        logging.info(f"Saving decoded structures to: {args.output_file} (Key: {args.output_key})")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with h5py.File(args.output_file, "w") as f:
            f.create_dataset(args.output_key, data=final_coords_np, chunks=(1, N_atoms, 3), compression="gzip") # Add chunking/compression
        logging.info(f"[SUCCESS] Wrote decoded coordinates.")

    except Exception as e: logging.error(f"Failed to concatenate or save results: {e}"); sys.exit(1)


if __name__ == "__main__":
    main()
