import os
import sys
import json
import yaml
import argparse
import logging
import math # For JS Div log (if enabled later)
import numpy as np
from typing import List, Dict, Optional, Tuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py  # for final outputs
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import ChebConv
from torch_cluster import knn_graph
from sklearn.model_selection import train_test_split

#################################################################
# Argument Parsing & Config Loading
#################################################################
parser = argparse.ArgumentParser(description="Protein Reconstruction with Pretrained HNO & Two-Step Decoders + Optional Dihedral Losses + Diffusion Override")
parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

# --- ADDED: Diffusion Override Arguments ---
parser.add_argument("--use_diffusion", action="store_true",
                    help="If set, attempt to override the internal pooling with diffused embeddings at final export.")
parser.add_argument("--diffused_backbone_h5", type=str, default=None,
                    help="Path to HDF5 with diffused backbone embeddings (e.g., dataset='generated_diffusion').")
parser.add_argument("--diffused_sidechain_h5", type=str, default=None,
                    help="Path to HDF5 with diffused sidechain embeddings (e.g., dataset='generated_diffusion').")
# --- END ADDED ---

args = parser.parse_args()

try:
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ERROR: Configuration file not found at {args.config}")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"ERROR: Could not parse configuration file {args.config}: {e}")
    sys.exit(1)

#################################################################
# Logging Setup
#################################################################
use_debug = config.get("use_debug_logs", False) or args.debug
log_file = config.get("log_file", "logfile.log")

# Setup logger instance
logger = logging.getLogger("ProteinReconstruction")
logger.setLevel(logging.DEBUG if use_debug else logging.INFO)

# Prevent duplicate handlers if logger is accessed multiple times
if not logger.handlers:
    # File Handler
    try:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG if use_debug else logging.INFO)
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except IOError as e:
        print(f"Warning: Could not write to log file {log_file}: {e}. Logging to console only.")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if use_debug else logging.INFO)
    # Use a simpler format for console if desired
    # console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    # ch.setFormatter(console_formatter)
    if 'formatter' not in locals(): # Define formatter if file handler failed
         formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logger.info("Logger initialized.")
if use_debug:
    logger.debug("Debug mode is ON.")
else:
    logger.info("Debug mode is OFF.")

#################################################################
# Device Setup
#################################################################
force_cpu = config.get("force_cpu", False)
if force_cpu:
     device_name = "cpu"
elif torch.cuda.is_available():
     # Allow specifying CUDA device index in config, e.g., cuda_device: 0
     cuda_device_index = config.get("cuda_device", 0)
     device_name = f"cuda:{cuda_device_index}"
     try:
         # Test if the specified device is valid
         torch.cuda.get_device_name(cuda_device_index)
     except (AssertionError, RuntimeError) as e:
         logger.warning(f"Specified CUDA device {cuda_device_index} not available or invalid: {e}. Falling back to CPU.")
         device_name = "cpu"
else:
     device_name = "cpu"
device = torch.device(device_name)
logger.info(f"Using device: {device}")

#################################################################
# Utility: Checkpoint Save/Load
#################################################################
def save_checkpoint(state: Dict, filename: str, logger: logging.Logger):
    """Saves model and optimizer state dict."""
    try:
        torch.save(state, filename)
        logger.debug(f"Checkpoint saved to {filename}")
    except IOError as e:
        logger.error(f"Error saving checkpoint to {filename}: {e}")
    sys.stdout.flush()

def load_checkpoint(model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    filename: str,
                    device: torch.device) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int]:
    """Loads model and optimizer state dict. Returns model, optimizer, start_epoch."""
    start_epoch = 0
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint from '{filename}'")
        try:
            # Load checkpoint onto the specified device directly
            checkpoint = torch.load(filename, map_location=device)
            start_epoch = checkpoint.get("epoch", 0) # Use .get for safety

            # Load model state
            try:
                 model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError as e:
                 logger.warning(f"Could not load model state dict strictly: {e}. Trying non-strict loading.")
                 # Try loading with strict=False if keys don't match exactly
                 model.load_state_dict(checkpoint["model_state_dict"], strict=False)

            # Load optimizer state if optimizer is provided and state exists
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                try:
                     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                     logger.info("Optimizer state loaded successfully.")
                     # Move optimizer state to the correct device (important if device changed)
                     for state in optimizer.state.values():
                         for k, v in state.items():
                             if isinstance(v, torch.Tensor):
                                 state[k] = v.to(device)
                except Exception as e:
                     logger.warning(f"Could not load optimizer state: {e}. Optimizer state ignored.")
            elif optimizer is not None:
                 logger.warning("Optimizer state dict not found in checkpoint. Optimizer state ignored.")

            # Ensure model is on the correct device *after* loading state dict
            model.to(device)
            logger.info(f"Loaded checkpoint. Resuming from epoch {start_epoch + 1}")

        except Exception as e:
            logger.error(f"Error loading checkpoint from '{filename}': {e}", exc_info=True)
            # Reset start_epoch if loading failed
            start_epoch = 0
            logger.warning("Training from scratch due to checkpoint loading error.")
        sys.stdout.flush()
    else:
        logger.info(f"No checkpoint found at '{filename}'. Training from scratch.")
        # Ensure model is on the correct device even when training from scratch
        model.to(device)
        sys.stdout.flush()
    return model, optimizer, start_epoch


#################################################################
# (A) PDB Parsing & Backbone/Sidechain Extraction
#################################################################
def parse_pdb(filename: str, logger: logging.Logger) -> Tuple[Dict, List]:
    """Parses ATOM records from a PDB file, handling alternate locations."""
    backbone_atoms = {"N", "CA", "C", "O", "OXT"}
    atoms_in_order = [] # List of (orig_res_id, orig_atom_index, category)
    atom_counter = 0
    processed_atom_indices = set() # To track unique atom serial numbers processed

    try:
        with open(filename, 'r') as pdb_file:
            for line in pdb_file:
                if not line.startswith("ATOM  ") and not line.startswith("HETATM"): # Process both for robustness? Assume ATOM only for now
                    continue

                atom_counter += 1
                record_type = line[0:6].strip()
                try:
                    # PDB format indices (1-based)
                    atom_serial = int(line[6:11])
                    atom_name = line[12:16].strip()
                    alt_loc = line[16].strip() # Alternate location indicator
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    res_seq = int(line[22:26])
                    #icode = line[26].strip() # Insertion code
                except ValueError as e:
                    logger.warning(f"Skipping malformed {record_type} line {atom_counter} (parsing error: {e}): {line.strip()}")
                    continue

                # Handle alternate locations: Keep only blank or 'A'
                if alt_loc != '' and alt_loc != 'A':
                    continue

                # Handle duplicate atom serial numbers (often from alt locs missed by above filter)
                # Keep only the first occurrence encountered
                if atom_serial in processed_atom_indices:
                    continue
                processed_atom_indices.add(atom_serial)

                # Create a unique original residue identifier
                orig_res_id = f"{chain_id}:{res_name}:{res_seq}"

                # Classify atom
                category = "backbone" if atom_name in backbone_atoms else "sidechain"

                # Store info needed for renumbering
                atoms_in_order.append((orig_res_id, atom_serial, category))

    except FileNotFoundError:
        logger.error(f"PDB file not found: {filename}")
        return {}, []
    except Exception as e:
        logger.error(f"Error reading PDB file {filename}: {e}", exc_info=True)
        return {}, []

    if not atoms_in_order:
        logger.error(f"No valid ATOM records (with altLoc='' or 'A') found in PDB file: {filename}")
    else:
        logger.info(f"Parsed {len(atoms_in_order)} unique ATOM records from {filename}.")

    return {}, atoms_in_order # We don't need original_dict, just the ordered list

def renumber_atoms_and_residues(atoms_in_order: List[Tuple[str, int, str]], logger: logging.Logger) -> Tuple[Dict, Dict]:
    """Renumbers residues and atoms consecutively starting from 0."""
    new_res_dict = {} # Maps new_res_id -> {"backbone": [new_atom_indices], "sidechain": [new_atom_indices]}
    orig_atom_to_new_atom_map = {} # Maps orig_atom_serial -> new_atom_index
    orig_res_to_new_res_map = {} # Maps orig_res_id -> new_res_id
    next_new_res_id = 0
    next_new_atom_index = 0

    # Determine residue appearance order based on the input list
    seen_res_ids_order = {}
    res_order_counter = 0
    for orig_res_id, _, _ in atoms_in_order:
        if orig_res_id not in seen_res_ids_order:
            seen_res_ids_order[orig_res_id] = res_order_counter
            res_order_counter += 1

    # Create tuples for stable sorting: (residue_appearance_order, original_atom_serial, orig_res_id, category)
    sortable_atoms = [
        (seen_res_ids_order[orig_res_id], orig_atom_serial, orig_res_id, category)
        for orig_res_id, orig_atom_serial, category in atoms_in_order
    ]
    # Sort primarily by residue appearance, secondarily by original atom serial number
    sortable_atoms.sort()

    # Perform renumbering based on sorted order
    for _, orig_atom_serial, orig_res_id, category in sortable_atoms:
        # Assign new residue ID if first time seeing this original residue ID
        if orig_res_id not in orig_res_to_new_res_map:
            orig_res_to_new_res_map[orig_res_id] = next_new_res_id
            new_res_dict[next_new_res_id] = {"backbone": [], "sidechain": []}
            next_new_res_id += 1

        # Get new residue ID and add new atom index
        new_res_id = orig_res_to_new_res_map[orig_res_id]
        new_res_dict[new_res_id][category].append(next_new_atom_index)
        orig_atom_to_new_atom_map[orig_atom_serial] = next_new_atom_index
        next_new_atom_index += 1

    logger.info(f"Renumbered {next_new_res_id} residues and {next_new_atom_index} atoms consecutively.")
    return new_res_dict, orig_atom_to_new_atom_map # Return mapping if needed later

def get_global_indices(renumbered_dict: Dict) -> Tuple[List[int], List[int]]:
    """Extracts sorted global lists of backbone and sidechain atom indices."""
    backbone_indices, sidechain_indices = [], []
    # Sort by the new residue ID to ensure consistent global ordering
    for res_id in sorted(renumbered_dict.keys()):
        # Indices within each residue should already be sorted by atom appearance
        backbone_indices.extend(renumbered_dict[res_id]["backbone"])
        sidechain_indices.extend(renumbered_dict[res_id]["sidechain"])
    # The combined lists should be globally sorted because of the renumbering process
    return backbone_indices, sidechain_indices


#################################################################
# (B) Load JSON heavy-atom coordinates
#################################################################
def load_heavy_atom_coords_from_json(json_file: str, logger: logging.Logger) -> Tuple[List[torch.Tensor], int]:
    """Loads coordinates from JSON, assuming keys are '0', '1', ... (new residue IDs)."""
    logger.info(f"Loading heavy atom coordinates from JSON: {json_file}")
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_file}")
        return [], -1
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file {json_file}: {e}")
        return [], -1

    # Expect keys to be NEW residue IDs (0-based integers as strings)
    try:
        residue_keys_sorted_int = sorted([int(k) for k in data.keys()])
        residue_keys_sorted_str = [str(k) for k in residue_keys_sorted_int] # Keep string keys for dict access
        logger.info(f"Found data for {len(residue_keys_sorted_str)} residues in JSON.")
    except ValueError:
        logger.error(f"Residue keys in {json_file} must be sortable integers ('0', '1', ...). Check JSON format.")
        return [], -1

    if not residue_keys_sorted_str:
        logger.error(f"No residue data found in {json_file}.")
        return [], -1

    # Determine number of frames and structure from the first residue
    first_res_key = residue_keys_sorted_str[0]
    try:
        frame_data = data[first_res_key].get("heavy_atom_coords_per_frame")
        if not isinstance(frame_data, list):
             raise TypeError("'heavy_atom_coords_per_frame' is not a list.")
        num_frames = len(frame_data)
        if num_frames == 0:
             raise ValueError("First residue has 0 frames.")
        # Check coordinate structure of the first atom in the first frame
        first_coords = np.array(frame_data[0][0])
        if first_coords.shape != (3,):
             raise ValueError(f"Expected coordinate shape (3,), but got {first_coords.shape}")
    except (KeyError, IndexError, TypeError, ValueError) as e:
        logger.error(f"Invalid structure or data for first residue ('{first_res_key}') in {json_file}. Cannot determine frames/coords. Error: {e}")
        return [], -1

    logger.info(f"Number of frames found in JSON: {num_frames}")
    coords_per_frame_list = [] # List to store tensors for each frame
    total_atoms_check = -1 # For consistency check

    # Iterate through frames
    for frame_idx in range(num_frames):
        frame_coords_list_np = [] # List of numpy arrays for this frame
        current_frame_atoms = 0
        # Iterate through residues in sorted order
        for res_key in residue_keys_sorted_str:
             try:
                 coords_this_res_raw = data[res_key]["heavy_atom_coords_per_frame"][frame_idx]
                 coords_this_res = np.array(coords_this_res_raw, dtype=np.float32)
                 # Validate shape: [num_atoms_in_res, 3]
                 if coords_this_res.ndim != 2 or coords_this_res.shape[1] != 3:
                      raise ValueError(f"Invalid coordinate shape {coords_this_res.shape}, expected [N, 3].")
                 frame_coords_list_np.append(coords_this_res)
                 current_frame_atoms += coords_this_res.shape[0]
             except (KeyError, IndexError, ValueError, TypeError) as e:
                 logger.error(f"Error processing residue {res_key} frame {frame_idx} in {json_file}: {e}")
                 return [], -1 # Abort on error

        # Check atom count consistency after processing all residues for the frame
        if frame_idx == 0:
            total_atoms_check = current_frame_atoms
            logger.info(f"Total atoms found in first frame from JSON: {total_atoms_check}")
        elif current_frame_atoms != total_atoms_check:
            logger.error(f"Inconsistent atom count in frame {frame_idx} ({current_frame_atoms}) vs first frame ({total_atoms_check}). Cannot proceed.")
            return [], -1

        # Concatenate numpy arrays for the frame -> [total_atoms, 3]
        try:
            frame_coords_np = np.concatenate(frame_coords_list_np, axis=0)
            # Convert to PyTorch tensor and add to list
            coords_per_frame_list.append(torch.tensor(frame_coords_np, dtype=torch.float32))
        except ValueError as e:
             logger.error(f"Error concatenating coordinates for frame {frame_idx}: {e}. Check atom counts within residues.")
             return [], -1

    if not coords_per_frame_list:
         logger.error("Failed to load any coordinate frames from JSON.")
         return [], -1

    return coords_per_frame_list, total_atoms_check

#################################################################
# (C) Kabsch Alignment
#################################################################
def compute_centroid(X: torch.Tensor) -> torch.Tensor:
    """Computes centroid by averaging over the atom dimension (assumed to be -2)."""
    return X.mean(dim=-2)

def kabsch_algorithm(P: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aligns Q onto P using Kabsch algorithm. Handles batches."""
    P = P.float()
    Q = Q.float()
    is_batched = P.ndim == 3
    if not is_batched:
        P = P.unsqueeze(0)
        Q = Q.unsqueeze(0)
    B, N, _3 = P.shape # Batch size, Num atoms, Coords

    # Center coordinates
    centroid_P = compute_centroid(P) # [B, 3]
    centroid_Q = compute_centroid(Q) # [B, 3]
    P_centered = P - centroid_P.unsqueeze(1) # [B, N, 3]
    Q_centered = Q - centroid_Q.unsqueeze(1) # [B, N, 3]

    # Covariance matrix H = Q_centered^T * P_centered
    C = torch.bmm(Q_centered.transpose(1, 2), P_centered) # [B, 3, 3]

    # SVD
    try:
        # Use torch.linalg.svd for robustness
        V, S, Wt = torch.linalg.svd(C) # V=[B,3,3], S=[B,3], Wt=[B,3,3]
    except torch._C._LinAlgError as e:
        logger.error(f"SVD failed during Kabsch: {e}. Returning identity alignment.")
        identity_U = torch.eye(3, device=P.device).unsqueeze(0).expand(B, -1, -1)
        Q_aligned_fallback = Q - centroid_Q.unsqueeze(1) + centroid_P.unsqueeze(1)
        if not is_batched: return identity_U.squeeze(0), Q_aligned_fallback.squeeze(0)
        return identity_U, Q_aligned_fallback

    # Ensure proper rotation (determinant +1)
    det_VWt = torch.det(torch.bmm(V, Wt)) # [B]
    D = torch.eye(3, device=P.device).unsqueeze(0).repeat(B, 1, 1) # [B, 3, 3]
    D[:, 2, 2] = torch.sign(det_VWt)
    U = torch.bmm(torch.bmm(V, D), Wt) # Rotation matrix [B, 3, 3]

    # Apply rotation and translate back
    Q_aligned_centered = torch.bmm(Q_centered, U)
    Q_aligned = Q_aligned_centered + centroid_P.unsqueeze(1)

    if not is_batched:
        U = U.squeeze(0)
        Q_aligned = Q_aligned.squeeze(0)

    return U, Q_aligned

def align_frames_to_first(coords_list: List[torch.Tensor], logger: logging.Logger, device: torch.device) -> List[torch.Tensor]:
    """Aligns all coordinate frames to the first frame using Kabsch."""
    logger.info("Aligning coordinate frames to the first frame...")
    if not coords_list:
        logger.warning("Coordinate list is empty, cannot align.")
        return []
    # Reference frame on target device
    reference = coords_list[0].float().to(device)
    # Store aligned frames on CPU
    aligned_coords_list = [coords_list[0].cpu()] # Keep first frame as is on CPU
    logger.debug(f"Reference frame shape: {reference.shape} on {reference.device}")

    num_frames_to_align = len(coords_list) - 1
    for i, coords in enumerate(coords_list[1:], start=1):
        coords_on_device = coords.float().to(device)
        _, coords_aligned_device = kabsch_algorithm(reference, coords_on_device)
        # Store aligned result on CPU
        aligned_coords_list.append(coords_aligned_device.cpu())

        if i % 500 == 0 or i == num_frames_to_align:
             logger.info(f"Aligned {i}/{num_frames_to_align} frames...")

    logger.info("Finished aligning frames.")
    return aligned_coords_list

#################################################################
# (D) Build PyG Graph Dataset
#################################################################
def build_graph_dataset(coords_list: List[torch.Tensor],
                        knn_neighbors: int = 4,
                        logger: Optional[logging.Logger] = None,
                        device: torch.device = torch.device('cpu')) -> List[Data]:
    """Builds PyTorch Geometric dataset with k-NN graphs."""
    if logger:
        logger.info(f"Building PyG dataset using k-NN graph (k={knn_neighbors}) on device '{device}'...")
    dataset = []
    num_frames = len(coords_list)
    for i, coords in enumerate(coords_list):
        # Calculate k-NN graph on the specified device
        coords_device = coords.to(device)
        # batch=None explicitly for single graph processing
        edge_index = knn_graph(coords_device, k=knn_neighbors, loop=False, batch=None)

        # Create Data object on CPU
        # Store original aligned coordinates as both input features 'x' and target 'y'
        data = Data(x=coords.cpu(), edge_index=edge_index.cpu(), y=coords.cpu())
        dataset.append(data)

        if logger and ((i + 1) % 500 == 0 or (i + 1) == num_frames):
            logger.info(f"Built graph for {i+1}/{num_frames} frames...")
    if logger:
        logger.info("Finished building PyG dataset.")
    return dataset

#################################################################
# (E) Blind Pooling Module (2D)
#################################################################
class BlindPooling2D(nn.Module):
    """Pools [B, N, E] input to [B, H*W] using AdaptiveAvgPool2d over N*E grid."""
    def __init__(self, H: int, W: int):
        super().__init__()
        self.pool2d = nn.AdaptiveAvgPool2d((H, W))
        self.output_dim = H * W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, N, E]
        if x.shape[1] == 0: # Handle empty sequence dimension N
             return torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype)
        B, N, E = x.shape
        # Add channel dim: [B, 1, N, E] (Treat N as Height, E as Width)
        x_4d = x.unsqueeze(1)
        pooled = self.pool2d(x_4d)  # -> [B, 1, H, W]
        pooled_flat = pooled.view(B, self.output_dim) # -> [B, H*W]
        return pooled_flat

#################################################################
# (F) HNO Model (ChebConv based)
#################################################################
class HNO(nn.Module):
    """Graph Neural Network Encoder using ChebConv layers."""
    def __init__(self, hidden_dim: int, K: int):
        super().__init__()
        self._debug_logged_fwd = False
        self._debug_logged_rep = False
        logger.debug(f"Initializing HNO with hidden_dim={hidden_dim}, K={K}")

        self.conv1 = ChebConv(3, hidden_dim, K=K) # Input features = 3 (coords)
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.conv3 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.conv4 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.bano1 = nn.BatchNorm1d(hidden_dim)
        self.bano2 = nn.BatchNorm1d(hidden_dim)
        self.bano3 = nn.BatchNorm1d(hidden_dim)
        # Final layer predicts coordinates (reconstruction) or representation
        self.mlpRep = nn.Linear(hidden_dim, 3) # Predicts 3D coordinates

    def _log_shape(self, name: str, tensor: torch.Tensor, log_debug: bool, flag_attr: str):
         """Helper for conditional debug logging of tensor shapes."""
         if log_debug and not getattr(self, flag_attr, False):
              logger.debug(f"[HNO {name}] Shape: {tensor.shape}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, log_debug: bool = False) -> torch.Tensor:
        """Forward pass for coordinate reconstruction."""
        log_now = log_debug and not self._debug_logged_fwd
        self._log_shape("Input", x, log_now, "_debug_logged_fwd")
        x = x.float()

        x = self.conv1(x, edge_index)
        x = self.bano1(F.leaky_relu(x))
        self._log_shape("After conv1+bano1", x, log_now, "_debug_logged_fwd")

        x = self.conv2(x, edge_index)
        x = self.bano2(F.leaky_relu(x))
        self._log_shape("After conv2+bano2", x, log_now, "_debug_logged_fwd")

        x = self.conv3(x, edge_index)
        x = self.bano3(F.relu(x))
        self._log_shape("After conv3+bano3", x, log_now, "_debug_logged_fwd")

        x = self.conv4(x, edge_index)
        self._log_shape("After conv4", x, log_now, "_debug_logged_fwd")

        x = F.normalize(x, p=2.0, dim=1) # Use p=2.0 (float)
        self._log_shape("After normalize", x, log_now, "_debug_logged_fwd")

        x = self.mlpRep(x)
        self._log_shape("Output (mlpRep)", x, log_now, "_debug_logged_fwd")

        if log_now: self._debug_logged_fwd = True
        return x

    def forward_representation(self, x: torch.Tensor, edge_index: torch.Tensor, log_debug: bool = False) -> torch.Tensor:
        """Forward pass to get latent representation [N, hidden_dim]."""
        log_now = log_debug and not self._debug_logged_rep
        self._log_shape("Rep Input", x, log_now, "_debug_logged_rep")
        x = x.float()

        x = self.conv1(x, edge_index)
        x = self.bano1(F.leaky_relu(x))
        self._log_shape("Rep After conv1+bano1", x, log_now, "_debug_logged_rep")

        x = self.conv2(x, edge_index)
        x = self.bano2(F.leaky_relu(x))
        self._log_shape("Rep After conv2+bano2", x, log_now, "_debug_logged_rep")

        x = self.conv3(x, edge_index)
        x = self.bano3(F.relu(x))
        self._log_shape("Rep After conv3+bano3", x, log_now, "_debug_logged_rep")

        x = self.conv4(x, edge_index)
        self._log_shape("Rep After conv4", x, log_now, "_debug_logged_rep")

        x = F.normalize(x, p=2.0, dim=1) # Use p=2.0 (float)
        self._log_shape("Rep Output", x, log_now, "_debug_logged_rep")

        if log_now: self._debug_logged_rep = True
        return x

#################################################################
# (G) Helper: Build simple MLP
#################################################################
def build_mlp(input_dim: int, output_dim: int, hidden_dim: int = 128, num_layers: int = 2,
              use_layernorm: bool = True, final_activation: Optional[nn.Module] = None) -> nn.Sequential:
    """Builds a multi-layer perceptron."""
    layers = []
    in_dim = input_dim
    if num_layers <= 0:
         raise ValueError("Number of MLP layers must be at least 1.")
    elif num_layers == 1:
         layers.append(nn.Linear(in_dim, output_dim))
    else:
         # First hidden layer
         layers.append(nn.Linear(in_dim, hidden_dim))
         layers.append(nn.ReLU())
         if use_layernorm: layers.append(nn.LayerNorm(hidden_dim))
         in_dim = hidden_dim
         # Intermediate hidden layers
         for _ in range(num_layers - 2):
              layers.append(nn.Linear(in_dim, hidden_dim))
              layers.append(nn.ReLU())
              if use_layernorm: layers.append(nn.LayerNorm(hidden_dim))
         # Final output layer
         layers.append(nn.Linear(in_dim, output_dim))

    if final_activation is not None:
         layers.append(final_activation)

    return nn.Sequential(*layers)

#################################################################
# (H) Backbone Decoder
#################################################################
class BackboneDecoder(nn.Module):
    """Decodes backbone coordinates from HNO latent space and reference."""
    def __init__(self, num_total_atoms: int, backbone_indices: torch.Tensor, emb_dim: int,
                 pooling_dim: Tuple[int, int] = (20, 4), mlp_depth: int = 2, mlp_hidden_dim: int = 128):
        super().__init__()
        self._debug_logged = False
        self.num_total_atoms = num_total_atoms
        self.register_buffer("backbone_indices", backbone_indices.cpu(), persistent=False)
        self.backbone_count = len(self.backbone_indices)
        if self.backbone_count == 0: logger.warning("BackboneDecoder initialized with 0 backbone atoms.")

        self.pool_backbone = BlindPooling2D(*pooling_dim)
        self.pool_output_dim = self.pool_backbone.output_dim # Use Script #1's naming

        self.mlp_input_dim = self.backbone_count * (emb_dim + self.pool_output_dim) if self.backbone_count > 0 else 0
        self.mlp_output_dim = self.backbone_count * 3

        self.mlp_flat = nn.Identity()
        if self.backbone_count > 0:
            self.mlp_flat = build_mlp(
                input_dim=self.mlp_input_dim, output_dim=self.mlp_output_dim,
                hidden_dim=mlp_hidden_dim, num_layers=mlp_depth, use_layernorm=True)

        self.last_pooled_backbone = None

    # --- MODIFIED: Added override_pooled_backbone ---
    def forward(self, hno_latent: torch.Tensor, z_ref: torch.Tensor, log_debug: bool = False,
                override_pooled_backbone: Optional[torch.Tensor] = None) -> torch.Tensor:

        if hno_latent.ndim == 3:
            B, N, E = hno_latent.shape
            x = hno_latent
            if N != self.num_total_atoms:
                logger.warning(f"BackboneDecoder Warning: Input N ({N}) != expected num_total_atoms ({self.num_total_atoms}).")
        elif hno_latent.ndim == 2:
            B_times_N, E = hno_latent.shape
            if B_times_N == 0:
                 return torch.empty(0, self.backbone_count, 3, device=hno_latent.device, dtype=hno_latent.dtype)
            if self.num_total_atoms == 0: raise ValueError("BackboneDecoder: num_total_atoms is 0.")
            if B_times_N % self.num_total_atoms != 0:
                 raise ValueError(f"BackboneDecoder: Input B*N ({B_times_N}) not divisible by num_total_atoms ({self.num_total_atoms}).")
            B = B_times_N // self.num_total_atoms
            N = self.num_total_atoms
            x = hno_latent.view(B, N, E)
        else:
            raise ValueError(f"BackboneDecoder: Unsupported hno_latent input ndim: {hno_latent.ndim}. Expected 2 or 3.")

        if self.backbone_count == 0:
             return torch.empty(B, 0, 3, device=hno_latent.device, dtype=hno_latent.dtype)

        z_ref = z_ref.to(hno_latent.device)
        should_log = (log_debug and not self._debug_logged)
        # x is now guaranteed to be [B, N, E]

        # --- Prepare MLP inputs ---
        backbone_emb = x[:, self.backbone_indices, :] # [B, bb_count, E]

        # --- Conditional Pooling ---
        if override_pooled_backbone is None:
            pooled_backbone = self.pool_backbone(backbone_emb) # [B, pool_out_dim]
            # Store internally generated pooled embedding only when not overriding
            self.last_pooled_backbone = pooled_backbone.detach().cpu()
        else:
            # Use the provided override
            if override_pooled_backbone.shape != (B, self.pool_output_dim):
                 raise ValueError(f"BackboneDecoder override_pooled_backbone has shape {override_pooled_backbone.shape}, expected ({B}, {self.pool_output_dim})")
            pooled_backbone = override_pooled_backbone.to(hno_latent.device) # Ensure device
            # Do not update self.last_pooled_backbone when overriding
        # --- End Conditional Pooling ---

        if z_ref is None: raise ValueError("[BackboneDecoder] z_ref is None.")
        z_ref_backbone_single = z_ref[self.backbone_indices, :] # [bb_count, E]
        z_ref_backbone = z_ref_backbone_single.unsqueeze(0).expand(B, -1, -1) # [B, bb_count, E]

        pooled_backbone_expanded = pooled_backbone.unsqueeze(1).expand(-1, self.backbone_count, -1) # [B, bb_count, pool_out_dim]
        combined = torch.cat([z_ref_backbone, pooled_backbone_expanded], dim=-1) # [B, bb_count, E + pool_out_dim]
        combined_flat = combined.view(B, -1) # [B, bb_count * (E + pool_out_dim)]

        # --- Predict ---
        pred_bb_flat = self.mlp_flat(combined_flat) # [B, bb_count * 3]
        pred_bb = pred_bb_flat.view(B, self.backbone_count, 3) # [B, bb_count, 3]

        if should_log:
            logger.debug(f"[BackboneDecoder] Input: hno_latent(viewed) {x.shape}, z_ref {z_ref.shape}")
            logger.debug(f"[BackboneDecoder] backbone_emb {backbone_emb.shape}, pooled_backbone {pooled_backbone.shape}")
            if override_pooled_backbone is not None:
                 logger.debug(f"[BackboneDecoder] Used override_pooled_backbone: {override_pooled_backbone.shape}")
            logger.debug(f"[BackboneDecoder] combined_flat {combined_flat.shape} -> mlp_flat -> pred_bb_flat {pred_bb_flat.shape}")
            logger.debug(f"[BackboneDecoder] Output: pred_bb {pred_bb.shape}")
            self._debug_logged = True

        return pred_bb
    # --- END MODIFIED ---

#################################################################
# (I) Sidechain Decoder
#################################################################
class SidechainDecoder(nn.Module):
    """Decodes full coordinates using backbone prediction and latent info."""
    def __init__(self, num_total_atoms: int, sidechain_indices: torch.Tensor, backbone_indices: torch.Tensor,
                 emb_dim: int, pooling_dim: Tuple[int, int] = (20, 4), mlp_depth: int = 2,
                 mlp_hidden_dim: int = 128, arch_type: int = 0):
        super().__init__()
        self._debug_logged = False
        self.num_total_atoms = num_total_atoms
        self.emb_dim = emb_dim
        self.arch_type = arch_type
        self.register_buffer("sidechain_indices", sidechain_indices.cpu(), persistent=False)
        self.register_buffer("backbone_indices", backbone_indices.cpu(), persistent=False)
        self.sidechain_count = len(self.sidechain_indices)
        self.backbone_count = len(self.backbone_indices)
        if self.sidechain_count == 0: logger.warning("SidechainDecoder initialized with 0 sidechain atoms.")

        self.pool_sidechain = BlindPooling2D(*pooling_dim)
        self.pool_output_dim = self.pool_sidechain.output_dim if self.sidechain_count > 0 else 0 # Use Script #1's naming

        # Optional reduction layers
        sc_zref_reduced_dim = 0
        self.sc_zref_reduce = None
        if arch_type >= 1 and self.sidechain_count > 0:
             sc_zref_input_dim = self.sidechain_count * emb_dim
             if sc_zref_input_dim > 0:
                  self.sc_zref_reduce = nn.Linear(sc_zref_input_dim, 128)
                  sc_zref_reduced_dim = 128

        bb_reduced_dim = 0
        self.bb_reduce = None
        if arch_type == 2 and self.backbone_count > 0:
             bb_input_dim = self.backbone_count * 3
             if bb_input_dim > 0:
                  self.bb_reduce = nn.Linear(bb_input_dim, 128)
                  bb_reduced_dim = 128

        # Determine MLP input dimension based on architecture
        final_in_dim = 0
        bb_term_dim = self.backbone_count * 3 if self.backbone_count > 0 else 0
        if arch_type == 0:   final_in_dim = bb_term_dim + self.pool_output_dim
        elif arch_type == 1: final_in_dim = bb_term_dim + self.pool_output_dim + sc_zref_reduced_dim
        elif arch_type == 2: final_in_dim = bb_reduced_dim + self.pool_output_dim + sc_zref_reduced_dim
        else: raise ValueError(f"Unsupported SidechainDecoder arch_type: {arch_type}")

        # Main MLP predicts sidechain coords
        self.mlp_sidechain = nn.Identity() # Default if no sidechains
        if self.sidechain_count > 0 and final_in_dim > 0 :
            self.mlp_sidechain = build_mlp(
                input_dim=final_in_dim, output_dim=self.sidechain_count * 3,
                hidden_dim=mlp_hidden_dim, num_layers=mlp_depth, use_layernorm=True)
        elif self.sidechain_count > 0 and final_in_dim == 0:
             logger.warning("Sidechain MLP input dimension is 0, check architecture/counts.")

        self.last_pooled_sidechain = None

    # --- MODIFIED: Added override_pooled_sidechain ---
    def forward(self, hno_latent: torch.Tensor, predicted_backbone: torch.Tensor,
                z_ref: torch.Tensor, log_debug: bool = False,
                override_pooled_sidechain: Optional[torch.Tensor] = None) -> torch.Tensor:

        if hno_latent.ndim == 3:
            B, N, E = hno_latent.shape
            x = hno_latent
            if N != self.num_total_atoms:
                 logger.warning(f"SidechainDecoder Warning: Input N ({N}) != expected num_total_atoms ({self.num_total_atoms}).")
        elif hno_latent.ndim == 2:
            B_times_N, E = hno_latent.shape
            if B_times_N == 0:
                 return torch.empty(0, self.num_total_atoms, 3, device=hno_latent.device, dtype=predicted_backbone.dtype)
            if self.num_total_atoms == 0: raise ValueError("SidechainDecoder: num_total_atoms is 0.")
            if B_times_N % self.num_total_atoms != 0:
                 raise ValueError(f"SidechainDecoder: Input B*N ({B_times_N}) not divisible by num_total_atoms ({self.num_total_atoms}).")
            B = B_times_N // self.num_total_atoms
            N = self.num_total_atoms
            x = hno_latent.view(B, N, E)
        else:
            raise ValueError(f"SidechainDecoder: Unsupported hno_latent input ndim: {hno_latent.ndim}. Expected 2 or 3.")

        should_log = (log_debug and not self._debug_logged)
        # x is now guaranteed to be [B, N, E]

        # --- Prepare MLP Inputs ---
        # 1. Pooled sidechain context (current frame or override)
        pooled_sidechain = torch.empty(B, 0, device=hno_latent.device) # Default if no sidechains
        if self.sidechain_count > 0:
            sidechain_emb = x[:, self.sidechain_indices, :]
            # --- Conditional Pooling ---
            if override_pooled_sidechain is None:
                pooled_sidechain = self.pool_sidechain(sidechain_emb) # [B, pool_out_dim]
                # Store internally generated pooled embedding only when not overriding
                self.last_pooled_sidechain = pooled_sidechain.detach().cpu()
            else:
                # Use the provided override
                if override_pooled_sidechain.shape != (B, self.pool_output_dim):
                     raise ValueError(f"SidechainDecoder override_pooled_sidechain has shape {override_pooled_sidechain.shape}, expected ({B}, {self.pool_output_dim})")
                pooled_sidechain = override_pooled_sidechain.to(hno_latent.device) # Ensure device
                # Do not update self.last_pooled_sidechain when overriding
            # --- End Conditional Pooling ---
        # If sidechain_count is 0, pooled_sidechain remains shape [B, 0]

        # 2. Backbone information (predicted coords)
        bb_flat = predicted_backbone.view(B, self.backbone_count * 3) # [B, bb_count * 3]
        bb_reduced = None
        if self.arch_type == 2 and self.bb_reduce:
             bb_reduced = self.bb_reduce(bb_flat) # [B, 128]

        # 3. Sidechain reference information (from z_ref)
        sc_zref_reduced = None
        if self.arch_type >= 1 and self.sc_zref_reduce:
            if z_ref is None: raise ValueError(f"Arch type {self.arch_type} requires z_ref.")
            if self.sidechain_count > 0:
                 sc_zref_single = z_ref[self.sidechain_indices, :]
                 sc_zref = sc_zref_single.unsqueeze(0).expand(B, -1, -1)
                 sc_zref_flat = sc_zref.view(B, self.sidechain_count * E)
                 sc_zref_reduced = self.sc_zref_reduce(sc_zref_flat) # [B, 128]
            else: # If arch >= 1 but no SC atoms, need zero tensor of expected dim
                 sc_zref_reduced = torch.zeros(B, self.sc_zref_reduce.out_features, device=hno_latent.device)

        # --- Concatenate inputs based on arch_type ---
        final_input_list = []
        if self.arch_type == 0:
             if self.backbone_count > 0: final_input_list.append(bb_flat)
             if self.pool_output_dim > 0: final_input_list.append(pooled_sidechain)
        elif self.arch_type == 1:
             if self.backbone_count > 0: final_input_list.append(bb_flat)
             if self.pool_output_dim > 0: final_input_list.append(pooled_sidechain)
             if sc_zref_reduced is not None: final_input_list.append(sc_zref_reduced)
        elif self.arch_type == 2:
             if bb_reduced is not None: final_input_list.append(bb_reduced)
             if self.pool_output_dim > 0: final_input_list.append(pooled_sidechain)
             if sc_zref_reduced is not None: final_input_list.append(sc_zref_reduced)

        if final_input_list:
             final_input = torch.cat(final_input_list, dim=-1)
        else: # Should only happen if backbone_count=0 and pool_output_dim=0
             final_input = torch.empty(B, 0, device=hno_latent.device)

        # --- Predict Sidechain Coordinates ---
        pred_sidechain_coords = torch.empty(B, 0, 3, device=hno_latent.device, dtype=predicted_backbone.dtype)
        if self.sidechain_count > 0 and isinstance(self.mlp_sidechain, nn.Sequential): # Check if MLP exists
            sidechain_coords_flat = self.mlp_sidechain(final_input) # [B, sc_count * 3]
            pred_sidechain_coords = sidechain_coords_flat.view(B, self.sidechain_count, 3)

        # --- Combine Backbone and Sidechain ---
        full_coords = torch.zeros(B, self.num_total_atoms, 3, device=hno_latent.device, dtype=predicted_backbone.dtype)
        if self.backbone_count > 0:
            full_coords[:, self.backbone_indices, :] = predicted_backbone
        if self.sidechain_count > 0:
            full_coords[:, self.sidechain_indices, :] = pred_sidechain_coords

        if should_log:
             logger.debug(f"[SidechainDecoder arch={self.arch_type}] Input: hno_latent(viewed) {x.shape}, pred_bb {predicted_backbone.shape}, z_ref {z_ref.shape}")
             if override_pooled_sidechain is not None:
                  logger.debug(f"[SidechainDecoder] Used override_pooled_sidechain: {override_pooled_sidechain.shape}")
             logger.debug(f"[SidechainDecoder] final_input {final_input.shape}")
             if self.sidechain_count > 0: logger.debug(f"[SidechainDecoder] pred_sc {pred_sidechain_coords.shape}")
             logger.debug(f"[SidechainDecoder] Output: full_coords {full_coords.shape}")
             self._debug_logged = True

        return full_coords
    # --- END MODIFIED ---

#################################################################
# (X) Dihedral Angle Utilities
#################################################################
@torch.jit.script
def compute_dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Computes dihedral angle(s). Input shapes [B, 3] or [B, N_angles, 3]."""
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    # Use p=2.0 (float) and add eps for stability
    n1_norm = F.normalize(n1, p=2.0, dim=-1, eps=1e-8)
    n2_norm = F.normalize(n2, p=2.0, dim=-1, eps=1e-8)
    b2_norm = F.normalize(b2, p=2.0, dim=-1, eps=1e-8)
    m1 = torch.cross(n1_norm, b2_norm, dim=-1)
    x = (n1_norm * n2_norm).sum(dim=-1)
    y = (m1 * n2_norm).sum(dim=-1)
    angle = torch.atan2(y, x)
    return angle

def compute_all_dihedrals_vectorized(coords: torch.Tensor,
                                     dihedral_info_precomputed: Dict[str, Dict],
                                     num_res: int) -> Dict[str, torch.Tensor]:
    """Computes all specified dihedrals (phi, psi, chi1-5) vectorially."""
    B = coords.shape[0]
    device = coords.device
    all_angles_out = {} # Dictionary to store angle tensors: name -> [B, num_res]

    for angle_name, info in dihedral_info_precomputed.items():
        indices = info.get('indices') # List of 4 index tensors
        res_idx_tensor = info.get('res_idx') # Tensor mapping calculation index to residue index

        # Initialize output tensor for this angle type with zeros
        angles_out_tensor = torch.zeros(B, num_res, device=device, dtype=coords.dtype)

        # Check if valid indices exist for this angle type
        if indices is not None and res_idx_tensor is not None and indices[0].numel() > 0:
            num_angles_of_this_type = indices[0].numel()
            # Gather coordinates using the specific indices
            try:
                a = coords[:, indices[0], :] # Shape [B, num_angles, 3]
                b = coords[:, indices[1], :]
                c = coords[:, indices[2], :]
                d = coords[:, indices[3], :]
            except IndexError as e:
                 logger.error(f"IndexError gathering coords for {angle_name}: {e}. Max index needed might exceed N={coords.shape[1]}.")
                 all_angles_out[angle_name] = angles_out_tensor # Store zeros and continue
                 continue

            # Compute all angles of this type simultaneously
            angle_values = compute_dihedral(a, b, c, d) # Shape [B, num_angles]

            # Place calculated values into the correct residue slots
            batch_indices = torch.arange(B, device=device).unsqueeze(1) # [B, 1]
            try:
                 # res_idx_tensor is [num_angles], need [1, num_angles] for broadcasting with batch_indices
                 angles_out_tensor[batch_indices, res_idx_tensor.unsqueeze(0)] = angle_values
            except IndexError as e:
                 logger.error(f"IndexError scattering angles for {angle_name}: {e}. Max res index needed={res_idx_tensor.max().item()}, num_res={num_res}")
                 # Output remains zeros

        all_angles_out[angle_name] = angles_out_tensor

    return all_angles_out

def compute_angle_kl_div(pred_flat_valid: torch.Tensor, true_flat_valid: torch.Tensor, nbins=36, angle_range=(-np.pi, np.pi)):
    min_angle, max_angle = angle_range
    pred_flat_valid_detached = pred_flat_valid.detach()
    true_flat_valid_detached = true_flat_valid.detach()
    if pred_flat_valid_detached.numel() == 0 or true_flat_valid_detached.numel() == 0: return torch.tensor(0.0, device=pred_flat_valid.device)
    edges = torch.linspace(min_angle, max_angle, nbins + 1, device=pred_flat_valid_detached.device)
    pred_hist = torch.histc(pred_flat_valid_detached, bins=nbins, min=min_angle, max=max_angle)
    true_hist = torch.histc(true_flat_valid_detached, bins=nbins, min=min_angle, max=max_angle)
    epsilon = 1e-10
    pred_dist = pred_hist / (pred_hist.sum() + epsilon)
    true_dist = true_hist / (true_hist.sum() + epsilon)
    pred_log_dist = torch.log(pred_dist + epsilon)
    kl_val = F.kl_div(pred_log_dist, true_dist, reduction='sum', log_target=False) # KL(True || Pred)
    return kl_val

def compute_angle_js_div(pred_flat_valid: torch.Tensor, true_flat_valid: torch.Tensor, nbins=36, angle_range=(-np.pi, np.pi)):
    min_angle, max_angle = angle_range
    pred_flat_valid_detached = pred_flat_valid.detach()
    true_flat_valid_detached = true_flat_valid.detach()
    if pred_flat_valid_detached.numel() == 0 or true_flat_valid_detached.numel() == 0: return torch.tensor(0.0, device=pred_flat_valid.device)
    edges = torch.linspace(min_angle, max_angle, nbins + 1, device=pred_flat_valid_detached.device)
    pred_hist = torch.histc(pred_flat_valid_detached, bins=nbins, min=min_angle, max=max_angle)
    true_hist = torch.histc(true_flat_valid_detached, bins=nbins, min=min_angle, max=max_angle)
    epsilon = 1e-10
    Q = pred_hist / (pred_hist.sum() + epsilon) # Predicted
    P = true_hist / (true_hist.sum() + epsilon) # True
    M = 0.5 * (P + Q)
    log_M = torch.log(M + epsilon)
    # KL(P || M) = sum(P * log(P/M)) => F.kl_div(input=logM, target=P)
    kl_pm = F.kl_div(log_M, P, reduction='sum', log_target=False)
    # KL(Q || M) = sum(Q * log(Q/M)) => F.kl_div(input=logM, target=Q)
    kl_qm = F.kl_div(log_M, Q, reduction='sum', log_target=False)
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd

def compute_angle_wasserstein(pred_flat_valid: torch.Tensor, true_flat_valid: torch.Tensor, nbins=36, angle_range=(-np.pi, np.pi)):
    min_angle, max_angle = angle_range
    pred_flat_valid_detached = pred_flat_valid.detach()
    true_flat_valid_detached = true_flat_valid.detach()
    if pred_flat_valid_detached.numel() == 0 or true_flat_valid_detached.numel() == 0: return torch.tensor(0.0, device=pred_flat_valid.device)
    edges = torch.linspace(min_angle, max_angle, nbins + 1, device=pred_flat_valid_detached.device)
    pred_hist = torch.histc(pred_flat_valid_detached, bins=nbins, min=min_angle, max=max_angle)
    true_hist = torch.histc(true_flat_valid_detached, bins=nbins, min=min_angle, max=max_angle)
    epsilon = 1e-10
    pred_dist = pred_hist / (pred_hist.sum() + epsilon)
    true_dist = true_hist / (true_hist.sum() + epsilon)
    pred_cdf = torch.cumsum(pred_dist, dim=0)
    true_cdf = torch.cumsum(true_dist, dim=0)
    # L1 distance between CDFs (approximation of Wasserstein-1)
    wasserstein_l1 = torch.sum(torch.abs(pred_cdf - true_cdf))
    return wasserstein_l1


#################################################################
# (J) Training Routines (Modified Backbone, Modified Sidechain)
#################################################################
def train_hno_model(model: nn.Module, # Type hint for HNO model
                    train_loader: DataLoader,
                    test_loader: DataLoader,
                    num_epochs: int,
                    learning_rate: float,
                    checkpoint_path: str,
                    save_interval: int = 10,
                    device: torch.device = torch.device('cpu') # Pass device
                   ) -> nn.Module:
    """
    Trains the HNO encoder model using coordinate reconstruction loss (MSE).
    """
    model = model.to(device) # Ensure model starts on the right device

    # Filter out parameters that don't require gradients
    try:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    except ValueError: # Happens if model has no trainable parameters
         logger.warning("HNO model has no trainable parameters. Skipping optimizer creation/training.")
         optimizer = None

    criterion = nn.MSELoss()

    # Load checkpoint if exists (moves model to device again, loads optimizer state)
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)

    logger.info(f"Starting HNO training from epoch {start_epoch+1}, total epochs={num_epochs}, LR={learning_rate}")
    sys.stdout.flush()

    for epoch in range(start_epoch, num_epochs):
        model.train() # Set model to training mode
        train_loss_val = 0.0
        num_batches = len(train_loader)

        # Check if optimizer exists before training loop
        if optimizer is None:
             logger.warning(f"No optimizer found for HNO model. Cannot train epoch {epoch+1}. Skipping...")
             break # Exit training loop if no optimizer

        for batch_idx, data in enumerate(train_loader):
            # Data object contains: x (input coords), edge_index, y (target coords), batch
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Forward pass: Predict coordinates from input coordinates
            # Log debug info only for the very first batch of the first epoch if enabled
            log_flag = (epoch == start_epoch and batch_idx == 0 and use_debug)
            pred = model(data.x, data.edge_index, log_debug=log_flag)

            # Calculate loss against ground truth coordinates (data.y)
            loss = criterion(pred, data.y)

            # Backward pass and optimization
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_val += loss.item()
            # End Training Batch Loop

        # Calculate average training loss for the epoch
        avg_train_loss = train_loss_val / num_batches if num_batches > 0 else 0.0

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        test_loss_val = 0.0
        num_val_batches = len(test_loader)
        with torch.no_grad(): # Disable gradient calculations for validation
            for data in test_loader:
                data = data.to(device)
                pred = model(data.x, data.edge_index)
                loss = criterion(pred, data.y)
                test_loss_val += loss.item()
        # Calculate average validation loss
        avg_test_loss = test_loss_val / num_val_batches if num_val_batches > 0 else 0.0

        # Log epoch results
        logger.info(f"[HNO] Epoch {epoch+1}/{num_epochs} => TRAIN MSE={avg_train_loss:.6f}, TEST MSE={avg_test_loss:.6f}")
        sys.stdout.flush()

        # --- Save Checkpoint ---
        current_epoch_num = epoch + 1
        # Save checkpoint if interval is reached or it's the last epoch
        if optimizer is not None and (current_epoch_num % save_interval == 0 or current_epoch_num == num_epochs):
            checkpoint_state = {
                "epoch": current_epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            # Overwrite the main checkpoint file (or save epoch-specific ones)
            save_checkpoint(checkpoint_state, checkpoint_path, logger)
            logger.info(f"HNO checkpoint saved at epoch {current_epoch_num} -> {checkpoint_path}")
            sys.stdout.flush()
        # End Epoch Loop

    logger.info(f"Finished training HNO model. Final checkpoint at {checkpoint_path}")
    return model


def train_backbone_decoder(
    model: BackboneDecoder, train_loader: DataLoader, test_loader: DataLoader,
    device: torch.device, logger: logging.Logger, config: dict, checkpoint_path: str,
    z_ref: torch.Tensor,
    dihedral_info_precomputed: Optional[Dict[str, Dict]] = None,
    dihedral_mask_all: Optional[torch.Tensor] = None, # Shape [num_res, num_angle_types]
    num_res: Optional[int] = None
):
    """ Train Backbone Decoder. Uses Coord MSE + Optional Dihedral Loss (Div + MSE) for Phi/Psi."""
    fraction_dihedral = 0.1

    lr = config.get("learning_rate", 0.001)
    epochs = config.get("num_epochs", 50)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr) if list(model.parameters()) else None
    coord_criterion = nn.MSELoss()

    num_total_atoms = model.num_total_atoms
    start_epoch = 0
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
    model.to(device) # Ensure model is on device after loading
    z_ref = z_ref.to(device)

    # --- Dihedral Loss Setup ---
    use_dihedral_loss = config.get("use_dihedral", False)
    lambda_1 = config.get("lambda_1", 0.0) # Weight for TOTAL BB DIVERGENCE loss (Phi + Psi)
    lambda_2 = config.get("lambda_2", 0.0) # Weight for TOTAL BB Torsion MSE loss (Phi + Psi)
    divergence_type = config.get("divergence_type", "KL").upper()
    backbone_angle_types = ['phi', 'psi'] # Angles relevant for backbone loss
    phi_mask, psi_mask = None, None

    if use_dihedral_loss:
        if dihedral_info_precomputed is None or dihedral_mask_all is None or num_res is None:
            logger.warning("Backbone dihedral loss requested but precomputed info missing. Disabling.")
            use_dihedral_loss = False
        else:
            # Extract masks for phi and psi
            try:
                 angle_types_all = ['phi', 'psi', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5']
                 phi_mask_idx = angle_types_all.index('phi')
                 psi_mask_idx = angle_types_all.index('psi')
                 phi_mask = dihedral_mask_all[:, phi_mask_idx].to(device) # [num_res]
                 psi_mask = dihedral_mask_all[:, psi_mask_idx].to(device) # [num_res]
                 logger.info(f"Backbone Decoder => use_dihedral_loss=True (Phi/Psi), Type={divergence_type}, lambda_Div={lambda_1:.4f}, lambda_MSE={lambda_2:.4f}")
            except (ValueError, IndexError) as e:
                 logger.error(f"Error extracting phi/psi masks from dihedral_mask_all: {e}. Disabling dihedral loss.")
                 use_dihedral_loss = False

    # Select divergence function
    compute_divergence = None
    if use_dihedral_loss:
        if divergence_type == "JS": compute_divergence = compute_angle_js_div
        elif divergence_type == "WASSERSTEIN": compute_divergence = compute_angle_wasserstein
        elif divergence_type == "KL": compute_divergence = compute_angle_kl_div
        else:
            logger.warning(f"Unknown backbone divergence_type '{divergence_type}'. Defaulting to KL.")
            divergence_type = "KL"
            compute_divergence = compute_angle_kl_div
    # --- End Dihedral Loss Setup ---

    logger.info(f"Starting Backbone Decoder training from epoch {start_epoch+1}, total epochs={epochs}, LR={lr}")
    best_train_loss = float("inf")
  
    for epoch in range(start_epoch, epochs):
        model.train()
        # Accumulators
        total_loss_bb_mse, total_loss_torsion_mse_phi, total_loss_torsion_mse_psi = 0.0, 0.0, 0.0
        total_loss_div_phi, total_loss_div_psi, total_loss_combined = 0.0, 0.0, 0.0

        num_batches = len(train_loader)
        for i, data in enumerate(train_loader):
            data = data.to(device)
            if optimizer: optimizer.zero_grad(set_to_none=True)
            log_flag = (i == 0 and epoch == start_epoch and use_debug) # Log first batch of first training epoch

            # Standard forward pass for training (no override)
            pred_bb = model(data.x, z_ref=z_ref, log_debug=log_flag)

            B_times_N, _ = data.y.shape
            B = B_times_N // num_total_atoms if num_total_atoms > 0 and B_times_N % num_total_atoms == 0 else 0
            if B == 0: continue
            coords_3d_gt = data.y.view(B, num_total_atoms, 3)
            gt_backbone = coords_3d_gt[:, model.backbone_indices, :]

            loss_bb_mse = coord_criterion(pred_bb, gt_backbone)
            current_loss = loss_bb_mse

            loss_torsion_mse_phi_batch, loss_torsion_mse_psi_batch = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            loss_div_phi_batch, loss_div_psi_batch = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

            if use_dihedral_loss and compute_divergence is not None and random.random()<fraction_dihedral:
                # Reconstruct full structure approximately for angle calculation
                full_pred = torch.zeros_like(coords_3d_gt)
                full_pred[:, model.backbone_indices, :] = pred_bb

                predicted_angles_dict = compute_all_dihedrals_vectorized(full_pred, dihedral_info_precomputed, num_res)
                true_angles_dict = compute_all_dihedrals_vectorized(coords_3d_gt, dihedral_info_precomputed, num_res)

                # --- Phi Loss ---
                phi_pred = predicted_angles_dict.get('phi')
                phi_true = true_angles_dict.get('phi')
                if phi_pred is not None and phi_true is not None and phi_mask is not None and phi_mask.any():
                    phi_mask_expanded = phi_mask.view(1, -1).expand(B, -1)
                    phi_pred_valid_flat = phi_pred[phi_mask_expanded]
                    phi_true_valid_flat = phi_true[phi_mask_expanded]
                    if phi_pred_valid_flat.numel() > 0:
                         loss_torsion_mse_phi_batch = F.mse_loss(phi_pred_valid_flat, phi_true_valid_flat)
                         loss_div_phi_batch = compute_divergence(phi_pred_valid_flat, phi_true_valid_flat)

                # --- Psi Loss ---
                psi_pred = predicted_angles_dict.get('psi')
                psi_true = true_angles_dict.get('psi')
                if psi_pred is not None and psi_true is not None and psi_mask is not None and psi_mask.any():
                    psi_mask_expanded = psi_mask.view(1, -1).expand(B, -1)
                    psi_pred_valid_flat = psi_pred[psi_mask_expanded]
                    psi_true_valid_flat = psi_true[psi_mask_expanded]
                    if psi_pred_valid_flat.numel() > 0:
                         loss_torsion_mse_psi_batch = F.mse_loss(psi_pred_valid_flat, psi_true_valid_flat)
                         loss_div_psi_batch = compute_divergence(psi_pred_valid_flat, psi_true_valid_flat)

                # --- Combine and Add to Loss ---
                loss_div_total_batch = loss_div_phi_batch + loss_div_psi_batch
                loss_torsion_mse_total_batch = loss_torsion_mse_phi_batch + loss_torsion_mse_psi_batch
                current_loss = current_loss + lambda_1 * loss_div_total_batch + lambda_2 * loss_torsion_mse_total_batch

            # Backpropagation
            if optimizer and current_loss.requires_grad:
                current_loss.backward()
                optimizer.step()

            # Accumulate
            total_loss_bb_mse += loss_bb_mse.item()
            total_loss_torsion_mse_phi += loss_torsion_mse_phi_batch.item()
            total_loss_torsion_mse_psi += loss_torsion_mse_psi_batch.item()
            total_loss_div_phi += loss_div_phi_batch.item()
            total_loss_div_psi += loss_div_psi_batch.item()
            total_loss_combined += current_loss.item()
            # End Train Batch Loop

        # --- Averages and Logging ---
        if num_batches == 0: continue
        avg_loss_bb_mse = total_loss_bb_mse / num_batches
        avg_loss_torsion_mse_phi = total_loss_torsion_mse_phi / num_batches
        avg_loss_torsion_mse_psi = total_loss_torsion_mse_psi / num_batches
        avg_loss_div_phi = total_loss_div_phi / num_batches
        avg_loss_div_psi = total_loss_div_psi / num_batches
        avg_loss_combined = total_loss_combined / num_batches

        # --- Validation ---
        model.eval()
        val_total_loss_bb_mse, val_total_loss_torsion_mse_phi, val_total_loss_torsion_mse_psi = 0.0, 0.0, 0.0
        val_total_loss_div_phi, val_total_loss_div_psi, val_total_loss_combined = 0.0, 0.0, 0.0
        num_val_batches = len(test_loader)

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                # Standard forward pass for validation
                pred_bb = model(data.x, z_ref=z_ref, log_debug=False)
                B_times_N, _ = data.y.shape
                B = B_times_N // num_total_atoms if num_total_atoms > 0 and B_times_N % num_total_atoms == 0 else 0
                if B == 0: continue
                coords_3d_gt = data.y.view(B, num_total_atoms, 3)
                gt_backbone = coords_3d_gt[:, model.backbone_indices, :]
                loss_bb_mse = coord_criterion(pred_bb, gt_backbone)
                val_loss = loss_bb_mse
                loss_torsion_mse_phi_batch, loss_torsion_mse_psi_batch = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                loss_div_phi_batch, loss_div_psi_batch = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

                if use_dihedral_loss and compute_divergence is not None:
                    full_pred = torch.zeros_like(coords_3d_gt)
                    full_pred[:, model.backbone_indices, :] = pred_bb
                    predicted_angles_dict = compute_all_dihedrals_vectorized(full_pred, dihedral_info_precomputed, num_res)
                    true_angles_dict = compute_all_dihedrals_vectorized(coords_3d_gt, dihedral_info_precomputed, num_res)

                    phi_pred = predicted_angles_dict.get('phi')
                    phi_true = true_angles_dict.get('phi')
                    if phi_pred is not None and phi_true is not None and phi_mask is not None and phi_mask.any():
                        phi_mask_expanded = phi_mask.view(1, -1).expand(B, -1)
                        phi_pred_valid_flat = phi_pred[phi_mask_expanded]
                        phi_true_valid_flat = phi_true[phi_mask_expanded]
                        if phi_pred_valid_flat.numel() > 0:
                             loss_torsion_mse_phi_batch = F.mse_loss(phi_pred_valid_flat, phi_true_valid_flat)
                             loss_div_phi_batch = compute_divergence(phi_pred_valid_flat, phi_true_valid_flat)

                    psi_pred = predicted_angles_dict.get('psi')
                    psi_true = true_angles_dict.get('psi')
                    if psi_pred is not None and psi_true is not None and psi_mask is not None and psi_mask.any():
                        psi_mask_expanded = psi_mask.view(1, -1).expand(B, -1)
                        psi_pred_valid_flat = psi_pred[psi_mask_expanded]
                        psi_true_valid_flat = psi_true[psi_mask_expanded]
                        if psi_pred_valid_flat.numel() > 0:
                             loss_torsion_mse_psi_batch = F.mse_loss(psi_pred_valid_flat, psi_true_valid_flat)
                             loss_div_psi_batch = compute_divergence(psi_pred_valid_flat, psi_true_valid_flat)

                    loss_div_total_batch = loss_div_phi_batch + loss_div_psi_batch
                    loss_torsion_mse_total_batch = loss_torsion_mse_phi_batch + loss_torsion_mse_psi_batch
                    val_loss = val_loss + lambda_1 * loss_div_total_batch + lambda_2 * loss_torsion_mse_total_batch

                # Accumulate validation losses
                val_total_loss_bb_mse += loss_bb_mse.item()
                val_total_loss_torsion_mse_phi += loss_torsion_mse_phi_batch.item()
                val_total_loss_torsion_mse_psi += loss_torsion_mse_psi_batch.item()
                val_total_loss_div_phi += loss_div_phi_batch.item()
                val_total_loss_div_psi += loss_div_psi_batch.item()
                val_total_loss_combined += val_loss.item()
                # End validation batch loop

        # Calculate average validation losses
        if num_val_batches == 0: continue
        avg_val_bb_mse = val_total_loss_bb_mse / num_val_batches
        avg_val_torsion_mse_phi = val_total_loss_torsion_mse_phi / num_val_batches
        avg_val_torsion_mse_psi = val_total_loss_torsion_mse_psi / num_val_batches
        avg_val_div_phi = val_total_loss_div_phi / num_val_batches
        avg_val_div_psi = val_total_loss_div_psi / num_val_batches
        avg_val_combined = val_total_loss_combined / num_val_batches

        # --- Log Epoch Results ---
        div_label = divergence_type.upper()
        log_msg = (
            f"[BackboneDecoder] Epoch {epoch+1}/{epochs} => \n"
            f"  TRAIN: BB_MSE={avg_loss_bb_mse:.4f} | "
            f"Phi(MSE={avg_loss_torsion_mse_phi:.4f}, {div_label}={avg_loss_div_phi:.4f}) | "
            f"Psi(MSE={avg_loss_torsion_mse_psi:.4f}, {div_label}={avg_loss_div_psi:.4f}) | "
            f"TOTAL_Loss={avg_loss_combined:.4f}\n"
            f"  TEST : BB_MSE={avg_val_bb_mse:.4f} | "
            f"Phi(MSE={avg_val_torsion_mse_phi:.4f}, {div_label}={avg_val_div_phi:.4f}) | "
            f"Psi(MSE={avg_val_torsion_mse_psi:.4f}, {div_label}={avg_val_div_psi:.4f}) | "
            f"TOTAL_Loss={avg_val_combined:.4f}"
        )
        logger.info(log_msg)
        sys.stdout.flush()

        # --- Save Checkpoint ---
        current_epoch_num = epoch + 1
        if avg_loss_combined < best_train_loss and optimizer:
             checkpoint_state = {
                 "epoch": current_epoch_num,
                 "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict()}
             save_checkpoint(checkpoint_state, checkpoint_path, logger)
             best_train_loss = avg_loss_combined
             logger.info(f"[Backbone] ↓ new best TRAIN loss {best_train_loss:.4f} – checkpoint saved to {checkpoint_path}")
             sys.stdout.flush()

        # End epoch loop

    logger.info(f"Finished training Backbone decoder. Final checkpoint at {checkpoint_path}")
    # ---- reload the best weights we just saved ----
    model, _, _ = load_checkpoint(model, None, checkpoint_path, device)
    model.eval()                          # switch to inference mode
    logger.info(f"Best backbone weights re‑loaded from {checkpoint_path}")
    return model



def train_sidechain_decoder(
    model: SidechainDecoder, train_loader: DataLoader, test_loader: DataLoader,
    backbone_decoder: BackboneDecoder, device: torch.device, logger: logging.Logger,
    config: dict, checkpoint_path: str, z_ref: torch.Tensor,
    dihedral_info_precomputed: Optional[Dict[str, Dict]] = None,
    dihedral_mask_all: Optional[torch.Tensor] = None, # Shape [num_res, num_angle_types]
    num_res: Optional[int] = None
):
    """ Train Sidechain Decoder. Uses Coord MSE + Optional Dihedral Loss (Div + MSE) for Chi1-5."""
    fraction_dihedral = 0.1
  
    lr = config.get("learning_rate", 0.001)
    epochs = config.get("num_epochs", 50)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    has_params = any(True for _ in model.parameters()) # Check if model actually has parameters
    optimizer = torch.optim.Adam(trainable_params, lr=lr) if has_params else None
    coord_criterion = nn.MSELoss()

    num_atoms = model.num_total_atoms
    start_epoch = 0
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
    model.to(device) # Ensure model on device
    backbone_decoder = backbone_decoder.to(device).eval() # Ensure BB decoder on device and eval
    z_ref = z_ref.to(device)

    # --- Sidechain Dihedral Loss Setup ---
    use_dihedral_sc = config.get("use_dihedral_sc", False)
    lambda_1_sc = config.get("lambda_1_sc", 0.0) # Weight for TOTAL SC DIVERGENCE loss
    lambda_2_sc = config.get("lambda_2_sc", 0.0) # Weight for TOTAL SC Torsion MSE loss
    divergence_type_sc = config.get("divergence_type_sc", "KL").upper()
    sidechain_angle_types = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']
    sc_angle_mask_indices = [] # Indices into columns of dihedral_mask_all

    if use_dihedral_sc:
        if dihedral_info_precomputed is None or dihedral_mask_all is None or num_res is None:
            logger.warning("Sidechain dihedral loss requested but precomputed info missing. Disabling.")
            use_dihedral_sc = False
        else:
            try:
                 angle_types_all = ['phi', 'psi', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5']
                 # Get column indices for chi1-5 in the mask
                 sc_angle_mask_indices = [angle_types_all.index(name) for name in sidechain_angle_types]
                 # Ensure mask is on device
                 dihedral_mask_all = dihedral_mask_all.to(device)
                 logger.info(f"Sidechain Decoder => use_dihedral_sc=True (Chi1-5), Type={divergence_type_sc}, lambda_Div={lambda_1_sc:.4f}, lambda_MSE={lambda_2_sc:.4f}")
            except (ValueError, IndexError) as e:
                 logger.error(f"Error setting up sidechain masks from dihedral_mask_all: {e}. Disabling SC dihedral loss.")
                 use_dihedral_sc = False

    # Select divergence function
    compute_divergence_sc = None
    if use_dihedral_sc:
        if divergence_type_sc == "JS": compute_divergence_sc = compute_angle_js_div
        elif divergence_type_sc == "WASSERSTEIN": compute_divergence_sc = compute_angle_wasserstein
        elif divergence_type_sc == "KL": compute_divergence_sc = compute_angle_kl_div
        else:
            logger.warning(f"Unknown sidechain divergence_type '{divergence_type_sc}'. Defaulting to KL.")
            divergence_type_sc = "KL"
            compute_divergence_sc = compute_angle_kl_div
    # --- End Sidechain Dihedral Loss Setup ---

    logger.info(f"Starting Sidechain Decoder training from epoch {start_epoch+1}, total epochs={epochs}, LR={lr}")
    best_train_loss = float("inf")
    for epoch in range(start_epoch, epochs):
        model.train()
        backbone_decoder.eval()

        # Accumulators
        total_train_loss, total_train_bb_mse, total_train_sc_mse = 0.0, 0.0, 0.0
        total_train_torsion_mse_sc, total_train_div_sc = 0.0, 0.0

        num_batches = len(train_loader)
        for i, data in enumerate(train_loader):
            data = data.to(device)
            if optimizer: optimizer.zero_grad(set_to_none=True)
            log_flag = (i == 0 and epoch == start_epoch and use_debug)

            # Use standard forward pass for training
            with torch.no_grad():
                pred_bb = backbone_decoder(data.x, z_ref=z_ref, log_debug=False) # No override here
            full_pred = model(data.x, pred_bb, z_ref=z_ref, log_debug=log_flag) # No override here

            B_times_N, _ = data.y.shape
            B = B_times_N // num_atoms if num_atoms > 0 and B_times_N % num_atoms == 0 else 0
            if B == 0: continue
            coords_3d_gt = data.y.view(B, num_atoms, 3)

            # --- Base Coordinate Loss ---
            bb_mse = torch.tensor(0.0, device=device)
            if model.backbone_count > 0:
                 bb_pred_from_full = full_pred[:, model.backbone_indices, :]
                 bb_gt = coords_3d_gt[:, model.backbone_indices, :]
                 bb_mse = coord_criterion(bb_pred_from_full, bb_gt)
            sc_mse = torch.tensor(0.0, device=device)
            if model.sidechain_count > 0:
                sc_pred = full_pred[:, model.sidechain_indices, :]
                sc_gt = coords_3d_gt[:, model.sidechain_indices, :]
                sc_mse = coord_criterion(sc_pred, sc_gt)
            current_loss = bb_mse + sc_mse

            # --- Sidechain Dihedral Losses ---
            loss_torsion_mse_sc_batch = torch.tensor(0.0, device=device)
            loss_div_sc_batch = torch.tensor(0.0, device=device)
            if use_dihedral_sc and compute_divergence_sc is not None and random.random() < fraction_dihedral:
                predicted_angles_dict = compute_all_dihedrals_vectorized(full_pred, dihedral_info_precomputed, num_res)
                true_angles_dict = compute_all_dihedrals_vectorized(coords_3d_gt, dihedral_info_precomputed, num_res)

                # Iterate through chi angle types ['chi1', ..., 'chi5']
                for angle_idx, angle_name in enumerate(sidechain_angle_types):
                    mask_col_idx = sc_angle_mask_indices[angle_idx]
                    pred_angles = predicted_angles_dict.get(angle_name)
                    true_angles = true_angles_dict.get(angle_name)
                    # Extract the mask column for this chi angle type
                    mask = dihedral_mask_all[:, mask_col_idx] # [num_res] bool

                    if pred_angles is not None and true_angles is not None and mask.any():
                        mask_expanded = mask.view(1, -1).expand(B, -1)
                        pred_valid_flat = pred_angles[mask_expanded]
                        true_valid_flat = true_angles[mask_expanded]

                        if pred_valid_flat.numel() > 0:
                            # Add MSE for this chi type to the batch total
                            loss_torsion_mse_sc_batch += F.mse_loss(pred_valid_flat, true_valid_flat)
                            # Add Divergence for this chi type to the batch total
                            loss_div_sc_batch += compute_divergence_sc(pred_valid_flat, true_valid_flat)

                # Add weighted total SC dihedral losses to current loss
                current_loss = current_loss + lambda_1_sc * loss_div_sc_batch + lambda_2_sc * loss_torsion_mse_sc_batch
            # --- End Dihedral Loss ---

            # Backpropagation
            if optimizer and current_loss.requires_grad:
                current_loss.backward()
                optimizer.step()

            # Accumulate losses
            total_train_loss += current_loss.item()
            total_train_bb_mse += bb_mse.item()
            total_train_sc_mse += sc_mse.item()
            total_train_torsion_mse_sc += loss_torsion_mse_sc_batch.item()
            total_train_div_sc += loss_div_sc_batch.item()
            # End train batch loop

        # --- Averages and Logging ---
        if num_batches == 0: continue
        avg_train_loss = total_train_loss / num_batches
        avg_train_bb = total_train_bb_mse / num_batches
        avg_train_sc = total_train_sc_mse / num_batches
        avg_train_torsion_mse_sc = total_train_torsion_mse_sc / num_batches
        avg_train_div_sc = total_train_div_sc / num_batches

        # --- Validation ---
        model.eval()
        val_total_loss, val_total_bb_mse, val_total_sc_mse = 0.0, 0.0, 0.0
        val_total_torsion_mse_sc, val_total_div_sc = 0.0, 0.0
        num_val_batches = len(test_loader)

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                # Standard forward pass for validation
                pred_bb = backbone_decoder(data.x, z_ref=z_ref, log_debug=False)
                full_pred = model(data.x, pred_bb, z_ref=z_ref, log_debug=False)
                B_times_N, _ = data.y.shape
                B = B_times_N // num_atoms if num_atoms > 0 and B_times_N % num_atoms == 0 else 0
                if B == 0: continue
                coords_3d_gt = data.y.view(B, num_atoms, 3)

                # Base loss
                bb_mse = torch.tensor(0.0, device=device)
                if model.backbone_count > 0:
                     bb_pred_from_full = full_pred[:, model.backbone_indices, :]
                     bb_gt = coords_3d_gt[:, model.backbone_indices, :]
                     bb_mse = coord_criterion(bb_pred_from_full, bb_gt)
                sc_mse = torch.tensor(0.0, device=device)
                if model.sidechain_count > 0:
                     sc_pred = full_pred[:, model.sidechain_indices, :]
                     sc_gt = coords_3d_gt[:, model.sidechain_indices, :]
                     sc_mse = coord_criterion(sc_pred, sc_gt)
                val_loss = bb_mse + sc_mse

                # SC Dihedral loss
                loss_torsion_mse_sc_batch = torch.tensor(0.0, device=device)
                loss_div_sc_batch = torch.tensor(0.0, device=device)
                if use_dihedral_sc and compute_divergence_sc is not None:
                    predicted_angles_dict = compute_all_dihedrals_vectorized(full_pred, dihedral_info_precomputed, num_res)
                    true_angles_dict = compute_all_dihedrals_vectorized(coords_3d_gt, dihedral_info_precomputed, num_res)
                    for angle_idx, angle_name in enumerate(sidechain_angle_types):
                        mask_col_idx = sc_angle_mask_indices[angle_idx]
                        pred_angles = predicted_angles_dict.get(angle_name)
                        true_angles = true_angles_dict.get(angle_name)
                        mask = dihedral_mask_all[:, mask_col_idx]
                        if pred_angles is not None and true_angles is not None and mask.any():
                            mask_expanded = mask.view(1, -1).expand(B, -1)
                            pred_valid_flat = pred_angles[mask_expanded]
                            true_valid_flat = true_angles[mask_expanded]
                            if pred_valid_flat.numel() > 0:
                                 loss_torsion_mse_sc_batch += F.mse_loss(pred_valid_flat, true_valid_flat)
                                 loss_div_sc_batch += compute_divergence_sc(pred_valid_flat, true_valid_flat)
                    val_loss = val_loss + lambda_1_sc * loss_div_sc_batch + lambda_2_sc * loss_torsion_mse_sc_batch

                # Accumulate validation losses
                val_total_loss += val_loss.item()
                val_total_bb_mse += bb_mse.item()
                val_total_sc_mse += sc_mse.item()
                val_total_torsion_mse_sc += loss_torsion_mse_sc_batch.item()
                val_total_div_sc += loss_div_sc_batch.item()
                # End validation batch loop

        # --- Calculate Averages and Log ---
        if num_val_batches == 0: continue
        avg_test_loss = val_total_loss / num_val_batches
        avg_test_bb = val_total_bb_mse / num_val_batches
        avg_test_sc = val_total_sc_mse / num_val_batches
        avg_test_torsion_mse_sc = val_total_torsion_mse_sc / num_val_batches
        avg_test_div_sc = val_total_div_sc / num_val_batches

        div_label_sc = divergence_type_sc.upper()
        log_msg = (
            f"[SidechainDecoder] Epoch {epoch+1}/{epochs} => \n"
            f"  TRAIN: TotalLoss={avg_train_loss:.4f} | Coord(BB={avg_train_bb:.4f}, SC={avg_train_sc:.4f}) | "
            f"SC_Dihedral(MSE={avg_train_torsion_mse_sc:.4f}, {div_label_sc}={avg_train_div_sc:.4f})\n"
            f"  TEST : TotalLoss={avg_test_loss:.4f} | Coord(BB={avg_test_bb:.4f}, SC={avg_test_sc:.4f}) | "
            f"SC_Dihedral(MSE={avg_test_torsion_mse_sc:.4f}, {div_label_sc}={avg_test_div_sc:.4f})"
        )
        logger.info(log_msg)
        sys.stdout.flush()

        # --- Save Checkpoint ---
        current_epoch_num = epoch + 1
        if avg_train_loss < best_train_loss and optimizer:
            checkpoint_state = {
                 "epoch": current_epoch_num,
                 "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict()}
            save_checkpoint(checkpoint_state, checkpoint_path, logger)
            logger.info(f"Sidechain decoder checkpoint saved at epoch {current_epoch_num} -> {checkpoint_path}")
            best_train_loss = avg_train_loss
            logger.info(f"[Side‑chain] ↓ new best TRAIN loss {best_train_loss:.4f} – checkpoint saved to {checkpoint_path}")
            sys.stdout.flush()
        # End Epoch Loop

    logger.info(f"Finished training Sidechain decoder. Final checkpoint at {checkpoint_path}")
    model, _, _ = load_checkpoint(model, None, checkpoint_path, device)
    model.eval()
    logger.info(f"Best side‑chain weights re‑loaded from {checkpoint_path}")
    return model



#################################################################
# (K) Output Generation (Modified for Diffusion Override)
#################################################################


#################################################################
# (K) Output Generation (Modified for Diffusion Override & Debugging)
#################################################################
@torch.no_grad()
def export_final_outputs(
    raw_dataset: List[Data], dec_dataset: List[Data], hno_model: HNO,
    backbone_decoder: BackboneDecoder, sidechain_decoder: SidechainDecoder,
    z_ref: torch.Tensor, num_atoms: int, struct_dir: str, latent_dir: str,
    device: torch.device,
    use_diff: bool = False, diff_bb: Optional[torch.Tensor] = None, diff_sc: Optional[torch.Tensor] = None
):
    """
    Exports final predictions and intermediate results to HDF5 files.
    Includes standard export and optional diffusion override export with added debugging.
    """
    global logger # Assuming logger is defined globally

    logger.info("Exporting final outputs: Ground Truth, HNO, Backbone, Full Coords, Pooled Embeddings.")

    hno_model.eval().to(device)
    backbone_decoder.eval().to(device)
    sidechain_decoder.eval().to(device)
    z_ref = z_ref.to(device)

    # Get indices and counts from models
    # Ensure these are tensors on CPU for indexing numpy arrays later if needed, although models keep buffers on CPU anyway.
    backbone_indices = backbone_decoder.backbone_indices.cpu() # Get from buffer
    sidechain_indices = sidechain_decoder.sidechain_indices.cpu() # Get from buffer
    backbone_count = len(backbone_indices)
    sidechain_count = len(sidechain_indices)

    # Define STANDARD output file paths
    gt_path = os.path.join(struct_dir, "ground_truth_aligned.h5")
    hno_recon_path = os.path.join(struct_dir, "hno_reconstructions.h5")
    backbone_path = os.path.join(struct_dir, "backbone_coords.h5")
    full_path = os.path.join(struct_dir, "full_coords.h5")
    backbone_pooled_path = os.path.join(latent_dir, "backbone_pooled.h5")
    sidechain_pooled_path = os.path.join(latent_dir, "sidechain_pooled.h5")

    # Define potential diffusion output paths
    backbone_path_diff = os.path.join(struct_dir, "backbone_coords_diff.h5")
    full_path_diff = os.path.join(struct_dir, "full_coords_diff.h5")

    total_samples = len(raw_dataset)
    if len(dec_dataset) != total_samples:
        logger.warning(f"Dataset length mismatch: Raw ({total_samples}) vs Dec ({len(dec_dataset)}). Using minimum.")
        total_samples = min(total_samples, len(dec_dataset))
    logger.info(f"Exporting {total_samples} standard samples.")

    bb_pool_dim = backbone_decoder.pool_output_dim
    sc_pool_dim = sidechain_decoder.pool_output_dim if sidechain_count > 0 else 0

    # --- Standard Export Loop ---
    dset_gt, dset_hno, dset_bb, dset_full, dset_bbpool, dset_scpool = None, None, None, None, None, None
    try:
        logger.debug("DEBUG: Entering standard export 'try' block.")
        with h5py.File(gt_path, "w") as gt_h5, \
             h5py.File(hno_recon_path, "w") as hno_h5, \
             h5py.File(backbone_path, "w") as bb_h5, \
             h5py.File(full_path, "w") as full_h5, \
             h5py.File(backbone_pooled_path, "w") as bbp_h5, \
             h5py.File(sidechain_pooled_path, "w") as scp_h5:

            logger.debug("DEBUG: Opened standard HDF5 files.")
            # Create datasets (standard)
            dset_gt = gt_h5.create_dataset("ground_truth_coords", (total_samples, num_atoms, 3), dtype='float32') if num_atoms > 0 else None
            dset_hno = hno_h5.create_dataset("hno_coords", (total_samples, num_atoms, 3), dtype='float32') if num_atoms > 0 else None
            dset_bb = bb_h5.create_dataset("backbone_coords", (total_samples, backbone_count, 3), dtype='float32') if backbone_count > 0 else None
            dset_full = full_h5.create_dataset("full_coords", (total_samples, num_atoms, 3), dtype='float32') if num_atoms > 0 else None
            dset_bbpool = bbp_h5.create_dataset("backbone_pooled", (total_samples, bb_pool_dim), dtype='float32') if bb_pool_dim > 0 else None
            dset_scpool = scp_h5.create_dataset("sidechain_pooled", (total_samples, sc_pool_dim), dtype='float32') if sc_pool_dim > 0 else None
            logger.debug(f"DEBUG: Created standard datasets (GT: {dset_gt is not None}, HNO: {dset_hno is not None}, BB: {dset_bb is not None}, Full: {dset_full is not None}, BBPool: {dset_bbpool is not None}, SCPool: {dset_scpool is not None})")

            logger.debug("DEBUG: Starting standard export loop.")
            for idx in range(total_samples):
                raw_data = raw_dataset[idx].to(device)
                dec_data = dec_dataset[idx].to(device)
                hno_latent = dec_data.x # Shape [N, E]
                hno_latent_batch = hno_latent.unsqueeze(0) # Shape [1, N, E]

                # 1. Ground Truth
                if dset_gt is not None:
                    dset_gt[idx] = dec_data.y.cpu().numpy()

                # 2. HNO Reconstruction
                if dset_hno is not None:
                    hno_recon = hno_model(raw_data.x, raw_data.edge_index)
                    dset_hno[idx] = hno_recon.cpu().numpy()

                # --- Process using INTERNAL pooling ---
                pred_bb = torch.empty(1, 0, 3, device=device) # Default

                # 3. Backbone Prediction (Standard)
                if dset_bb is not None:
                    pred_bb = backbone_decoder(hno_latent_batch, z_ref=z_ref) # NO override
                    bb_np_array = pred_bb.squeeze(0).cpu().numpy()
                    dset_bb[idx] = bb_np_array
                    if dset_bbpool is not None and hasattr(backbone_decoder, "last_pooled_backbone") and backbone_decoder.last_pooled_backbone is not None:
                         dset_bbpool[idx] = backbone_decoder.last_pooled_backbone.numpy()

                # 4. Full Structure Prediction (Standard)
                if dset_full is not None:
                    full_pred = sidechain_decoder(hno_latent_batch, pred_bb, z_ref=z_ref) # NO override
                    full_np_array = full_pred.squeeze(0).cpu().numpy()
                    dset_full[idx] = full_np_array
                    if dset_scpool is not None and hasattr(sidechain_decoder, "last_pooled_sidechain") and sidechain_decoder.last_pooled_sidechain is not None:
                        dset_scpool[idx] = sidechain_decoder.last_pooled_sidechain.numpy()

                if (idx + 1) % 500 == 0 or (idx + 1) == total_samples:
                     logger.info(f"Exported standard results for {idx+1}/{total_samples} samples...")
            logger.debug("DEBUG: Finished standard export loop.")
        logger.debug("DEBUG: Exited standard export 'with h5py.File...' block.")

    except Exception as e:
        logger.error(f"Error during standard HDF5 export: {e}", exc_info=True)
        logger.debug(f"DEBUG standard export exception details:", exc_info=True)
        # Depending on severity, you might want to return here or allow diffusion export to proceed

    # --- Diffusion Override Export ---
    dset_bb_diff, dset_full_diff = None, None # For final logging status check
    logger.debug(f"DEBUG export_final_outputs: Checking diffusion export condition. Received use_diff = {use_diff}")
    if use_diff:
        logger.debug("DEBUG: use_diff is True, proceeding with diffusion checks.")
        if diff_bb is None or diff_sc is None:
             logger.error("use_diff=True but diff_bb or diff_sc is None in export_final_outputs. Cannot perform diffusion export.")
             logger.debug(f"DEBUG export_final_outputs: Skipping diffusion export because diff_bb is None: {diff_bb is None}, or diff_sc is None: {diff_sc is None}")
        else:
            logger.debug(f"DEBUG: diff_bb shape: {diff_bb.shape}, diff_sc shape: {diff_sc.shape}")
            N_data = len(dec_dataset)
            N_diff_bb = diff_bb.shape[0]
            N_diff_sc = diff_sc.shape[0]
            N_diff = min(N_data, N_diff_bb, N_diff_sc)
            logger.debug(f"DEBUG: Calculated N_data={N_data}, N_diff_bb={N_diff_bb}, N_diff_sc={N_diff_sc}, N_diff={N_diff}")

            # Corrected Logic: Export if N_diff > 0
            if N_diff == 0:
                 logger.warning("No diffused embeddings available or dataset empty (N_diff=0). Skipping diffusion export.")
                 logger.debug("DEBUG: Skipping diffusion export because N_diff is 0.")
            else: # N_diff > 0
                if N_diff < N_data:
                    logger.warning(f"Number of diffused embeddings ({N_diff_bb} BB, {N_diff_sc} SC) is less than dataset size ({N_data}). Exporting only {N_diff} diffused samples.")
                # This log should always appear if N_diff > 0
                logger.info(f"Exporting {N_diff} samples using diffused embeddings.")
                try:
                    logger.debug("DEBUG: Entered diffusion export 'try' block.")
                    with h5py.File(backbone_path_diff, "w") as bb_diff_h5, \
                         h5py.File(full_path_diff, "w") as full_diff_h5:

                        logger.debug(f"DEBUG: Attempting to open diffusion HDF5 files: {backbone_path_diff}, {full_path_diff}")
                        logger.debug("DEBUG: Successfully opened diffusion HDF5 files for writing.")

                        # Create diffusion datasets
                        dset_bb_diff = bb_diff_h5.create_dataset("backbone_coords_diff", (N_diff, backbone_count, 3), dtype='float32') if backbone_count > 0 else None
                        logger.debug(f"DEBUG: Created dset_bb_diff: {dset_bb_diff} (Type: {type(dset_bb_diff)})")

                        dset_full_diff = full_diff_h5.create_dataset("full_coords_diff", (N_diff, num_atoms, 3), dtype='float32') if num_atoms > 0 else None
                        logger.debug(f"DEBUG: Created dset_full_diff: {dset_full_diff} (Type: {type(dset_full_diff)})")

                        # Check if datasets were actually created before looping
                        skip_loop = dset_bb_diff is None and dset_full_diff is None
                        logger.debug(f"DEBUG: Checking loop skip condition: dset_bb_diff is None ({dset_bb_diff is None}) AND dset_full_diff is None ({dset_full_diff is None}) -> Skip = {skip_loop}")
                        if skip_loop:
                             logger.warning("Neither backbone nor full diff datasets seem to have been created (check counts or HDF5 permissions?). Skipping diff export loop.")
                        else:
                             logger.debug(f"DEBUG export_final_outputs: Proceeding with diffusion export loop for N_diff = {N_diff} samples.")
                             for idx in range(N_diff):
                                # --- Limit verbose logs to first few iterations ---
                                log_this_iter = idx < 3 # Log verbosely for idx 0, 1, 2

                                if log_this_iter:
                                    logger.debug(f"DEBUG Loop idx={idx}: Starting processing.")
                                try:
                                    dec_data = dec_dataset[idx].to(device)
                                    hno_latent = dec_data.x
                                    hno_latent_batch = hno_latent.unsqueeze(0)

                                    bb_over = diff_bb[idx:idx+1, :].to(device)
                                    sc_over = diff_sc[idx:idx+1, :].to(device)
                                    if log_this_iter:
                                        logger.debug(f"DEBUG Loop idx={idx}: bb_over shape {bb_over.shape}, sc_over shape {sc_over.shape}")

                                    pred_bb_diff = torch.empty(1, 0, 3, device=device)
                                    if dset_bb_diff is not None:
                                        if log_this_iter:
                                            logger.debug(f"DEBUG Loop idx={idx}: Calling backbone_decoder with override.")
                                        pred_bb_diff = backbone_decoder(hno_latent_batch, z_ref=z_ref, override_pooled_backbone=bb_over)

                                        if log_this_iter: # Log stats/shape only for first few
                                            logger.debug(f"DEBUG Loop idx={idx}: Shape of pred_bb_diff: {pred_bb_diff.shape}")
                                            if pred_bb_diff.numel() > 0: # Avoid errors on empty tensors
                                                logger.debug(f"DEBUG Loop idx={idx}: Stats pred_bb_diff: min={pred_bb_diff.min().item():.3f}, max={pred_bb_diff.max().item():.3f}, mean={pred_bb_diff.mean().item():.3f}, has_nan={torch.isnan(pred_bb_diff).any().item()}")
                                            else:
                                                 logger.debug(f"DEBUG Loop idx={idx}: pred_bb_diff is empty.")


                                        bb_np_array = pred_bb_diff.squeeze(0).cpu().numpy()
                                        if log_this_iter: # Log shape only for first few
                                            logger.debug(f"DEBUG Loop idx={idx}: Shape of bb_np_array for HDF5: {bb_np_array.shape}")
                                        dset_bb_diff[idx] = bb_np_array
                                        # if log_this_iter: # Log assignment only for first few if needed
                                        #     logger.debug(f"DEBUG Loop idx={idx}: Assigned to dset_bb_diff.")


                                    if dset_full_diff is not None:
                                        if log_this_iter:
                                            logger.debug(f"DEBUG Loop idx={idx}: Calling sidechain_decoder with override.")
                                        full_pred_diff = sidechain_decoder(hno_latent_batch, pred_bb_diff, z_ref=z_ref, override_pooled_sidechain=sc_over)

                                        if log_this_iter: # Log stats/shape only for first few
                                            logger.debug(f"DEBUG Loop idx={idx}: Shape of full_pred_diff: {full_pred_diff.shape}")
                                            if full_pred_diff.numel() > 0:
                                                logger.debug(f"DEBUG Loop idx={idx}: Stats full_pred_diff: min={full_pred_diff.min().item():.3f}, max={full_pred_diff.max().item():.3f}, mean={full_pred_diff.mean().item():.3f}, has_nan={torch.isnan(full_pred_diff).any().item()}")
                                            else:
                                                 logger.debug(f"DEBUG Loop idx={idx}: full_pred_diff is empty.")


                                        full_np_array = full_pred_diff.squeeze(0).cpu().numpy()
                                        if log_this_iter: # Log shape only for first few
                                            logger.debug(f"DEBUG Loop idx={idx}: Shape of full_np_array for HDF5: {full_np_array.shape}")
                                        dset_full_diff[idx] = full_np_array
                                        # if log_this_iter: # Log assignment only for first few if needed
                                        #     logger.debug(f"DEBUG Loop idx={idx}: Assigned to dset_full_diff.")

                                    # Keep the INFO progress log unconditional
                                    if (idx + 1) % 500 == 0 or (idx + 1) == N_diff:
                                        logger.info(f"Exported diffusion override results for {idx+1}/{N_diff} samples...")

                                except Exception as loop_e:
                                     logger.error(f"Error during diffusion export loop at index {idx}: {loop_e}", exc_info=True)
                                     logger.debug(f"DEBUG Loop idx={idx}: Exception details:", exc_info=True)

                             logger.debug("DEBUG: Finished diffusion export loop.") # Keep this unconditional


                    logger.debug("DEBUG: Exited diffusion 'with h5py.File...' block.")
                    logger.info(f"Saved diffusion-based coords to:\n  {backbone_path_diff}\n  {full_path_diff}")

                except Exception as e:
                     logger.error(f"Error during diffusion HDF5 export setup or file handling: {e}", exc_info=True)
                     logger.debug(f"DEBUG: Diffusion export setup exception details:", exc_info=True)
    else:
        # Log reason for skipping if use_diff was False initially or became False
        logger.debug(f"DEBUG export_final_outputs: Skipping diffusion export block because use_diff flag is {use_diff}.")
        # Assuming 'args' is accessible or passed if needed for this specific debug line
        # if 'args' in globals() and not args.use_diffusion:
        #     logger.debug("DEBUG export_final_outputs: --use_diffusion flag was likely not provided.")
        # else:
        #      logger.debug("DEBUG export_final_outputs: --use_diffusion flag WAS provided, but override was likely disabled due to loading/shape errors in main.")
        logger.info("Diffusion override not requested or embeddings not validated; skipping diff output.")




#################################################################
# (L) Main Orchestration Function
#################################################################
def main():
    logger.info(f"Script started. Using device: {device}")

    # --- Setup Directories ---
    out_dirs = config.get("output_directories", {})
    ckpt_dir = out_dirs.get("checkpoint_dir", "checkpoints")
    struct_dir = out_dirs.get("structure_dir", "structures")
    latent_dir = out_dirs.get("latent_dir", "latent_reps")
    try: os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(struct_dir, exist_ok=True); os.makedirs(latent_dir, exist_ok=True)
    except OSError as e: logger.error(f"Error creating output directories: {e}"); sys.exit(1)

    # --- Load Config Parameters ---
    json_path = config.get("json_path")
    pdb_filename = config.get("pdb_filename")
    if not json_path or not pdb_filename: logger.error("Missing 'json_path' or 'pdb_filename' in config."); sys.exit(1)

    num_workers = config.get("num_workers", 0)
    pin_memory = (device.type == "cuda")

    # --- Data Loading and Preprocessing ---
    coords_per_frame, json_num_atoms = load_heavy_atom_coords_from_json(json_path, logger)
    if not coords_per_frame: logger.error("Failed to load coordinates from JSON."); sys.exit(1)

    logger.info(f"Parsing PDB: {pdb_filename}")
    _, atoms_in_order = parse_pdb(pdb_filename, logger)
    if not atoms_in_order: logger.error("Failed to parse PDB."); sys.exit(1)
    renumbered_dict, _ = renumber_atoms_and_residues(atoms_in_order, logger)
    bb_indices_list, sc_indices_list = get_global_indices(renumbered_dict)
    backbone_indices = torch.tensor(bb_indices_list, dtype=torch.long) # CPU tensor
    sidechain_indices = torch.tensor(sc_indices_list, dtype=torch.long) # CPU tensor
    num_atoms_pdb = len(backbone_indices) + len(sidechain_indices)

    if json_num_atoms != num_atoms_pdb: logger.error(f"Atom count mismatch! JSON ({json_num_atoms}) != PDB ({num_atoms_pdb})."); sys.exit(1)
    num_atoms = num_atoms_pdb
    logger.info(f"Found {len(backbone_indices)} backbone and {len(sidechain_indices)} sidechain atoms (Total: {num_atoms}).")

    coords_aligned = align_frames_to_first(coords_per_frame, logger, device)
    if not coords_aligned: logger.error("Failed to align coordinates."); sys.exit(1)

    knn_value = config.get("knn_value", 4)
    dataset = build_graph_dataset(coords_aligned, knn_neighbors=knn_value, logger=logger, device=device)
    if not dataset: logger.error("Failed to build graph dataset."); sys.exit(1)

    # --- 1) HNO Training/Loading ---
    hno_conf = config.get("hno_training", {})
    hno_batch_size = hno_conf.get("batch_size", 32)
    train_data_hno, test_data_hno = train_test_split(dataset, test_size=0.1, random_state=42)
    train_loader_hno = DataLoader(train_data_hno, batch_size=hno_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    test_loader_hno = DataLoader(test_data_hno, batch_size=hno_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    cheb_order = config.get("cheb_order", 3)
    hidden_dim = config.get("hidden_dim", 128)
    hno_model = HNO(hidden_dim, K=cheb_order) # Initialized on CPU
    hno_ckpt = os.path.join(ckpt_dir, config.get("hno_ckpt", "hno_model.pth"))
    logger.info(f"Training/loading HNO => {hno_conf.get('num_epochs', 0)} epochs, LR={hno_conf.get('learning_rate', 0.001)}")
    hno_model = train_hno_model(
        model=hno_model, train_loader=train_loader_hno, test_loader=test_loader_hno,
        num_epochs=hno_conf.get("num_epochs", 0), learning_rate=hno_conf.get("learning_rate", 0.001),
        checkpoint_path=hno_ckpt, save_interval=hno_conf.get("save_interval", 10), device=device)
    hno_model.eval().to(device) # Ensure on correct device and eval mode

    # --- 2) Build Decoder Dataset ---
    logger.info("Building decoder dataset (latent embeddings)...")
    dec_dataset = []
    inference_batch_size = config.get("inference_batch_size", hno_batch_size * 2)
    inference_loader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    with torch.no_grad():
        for data_batch in inference_loader:
            data_batch = data_batch.to(device)
            # Use forward_representation to get embeddings
            x_emb_batch = hno_model.forward_representation(data_batch.x, data_batch.edge_index)
            y_batch = data_batch.y # Ground truth coordinates
            num_graphs = data_batch.num_graphs
            node_slices = torch.cumsum(torch.bincount(data_batch.batch), 0)
            node_slices = torch.cat([torch.tensor([0], device=device), node_slices])
            # Split batch back into individual graphs
            for i in range(num_graphs):
                 start, end = node_slices[i], node_slices[i+1]
                 # Store embedding as 'x' and true coords as 'y' on CPU
                 dec_dataset.append(Data(x=x_emb_batch[start:end].cpu(), y=y_batch[start:end].cpu()))
    logger.info(f"Built decoder dataset with {len(dec_dataset)} samples.")

    # --- 3) Split Decoder Dataset ---
    dec_batch_size = config.get("decoder_batch_size", 16)
    train_data_dec, test_data_dec = train_test_split(dec_dataset, test_size=0.1, random_state=42)
    train_loader_dec = DataLoader(train_data_dec, batch_size=dec_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    test_loader_dec = DataLoader(test_data_dec, batch_size=dec_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # --- 4) Calculate z_ref ---
    logger.info("Computing z_ref...")
    with torch.no_grad():
        # Use the first sample from the original dataset (which has coords in 'x')
        first_frame_data = dataset[0].to(device)
        z_ref = hno_model.forward_representation(first_frame_data.x, first_frame_data.edge_index, log_debug=use_debug)
    logger.debug(f"z_ref calculated => shape {z_ref.shape}, device {z_ref.device}")

    # --- ADDED: Save X_ref and z_ref ---
    try:
        x_ref_path = os.path.join(struct_dir, "X_ref_coords.pt")
        z_ref_path = os.path.join(struct_dir, "z_ref_embedding.pt")
        # Ensure X_ref (first aligned coords) is on CPU before saving
        X_ref_cpu = coords_aligned[0].cpu() # coords_aligned is already on CPU
        torch.save(X_ref_cpu, x_ref_path)
        # Ensure z_ref is on CPU before saving
        z_ref_cpu = z_ref.cpu()
        torch.save(z_ref_cpu, z_ref_path)
        logger.info(f"Saved reference coordinates to: {x_ref_path}")
        logger.info(f"Saved reference embedding to: {z_ref_path}")
    except Exception as e:
        logger.error(f"Error saving X_ref or z_ref: {e}", exc_info=True)
    # --- END ADDED ---


    # --- 5) Precompute Dihedral Info ---
    torsion_json_path = config.get("torsion_info_path", "condensed_residues.json")
    dihedral_info_precomputed = {} # Dict: name -> {'indices': [...], 'res_idx': tensor}
    dihedral_mask_all = None     # Tensor: [num_res, 7] bool mask
    num_res = None
    angle_types_all = ['phi', 'psi', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5']
    num_angle_types = len(angle_types_all)

    if os.path.isfile(torsion_json_path):
        try:
            with open(torsion_json_path, "r") as f: torsion_info = json.load(f)
            logger.info(f"Torsion info loaded from {torsion_json_path}")
            logger.info("Precomputing ALL dihedral angle indices and masks...")

            indices_lists = {name: [[], [], [], []] for name in angle_types_all}
            residue_indices_lists = {name: [] for name in angle_types_all}

            try:
                 torsion_keys_sorted = sorted([int(k) for k in torsion_info.keys()])
                 num_res = len(torsion_keys_sorted)
                 logger.info(f"Found torsion info for {num_res} residues.")
            except ValueError: logger.error("Invalid torsion JSON keys."); torsion_info = None

            if torsion_info and num_res is not None and num_res > 0:
                 valid_angle_mask_list = [[False] * num_angle_types for _ in range(num_res)]
                 # Loop through residues and angle types
                 for r_idx, res_id_int in enumerate(torsion_keys_sorted):
                      res_str = str(res_id_int)
                      res_data = torsion_info.get(res_str, {})
                      torsion_atoms = res_data.get("torsion_atoms", {})
                      chi_atoms = torsion_atoms.get("chi", {})
                      for type_idx, angle_name in enumerate(angle_types_all):
                           indices = None
                           if angle_name in ['phi', 'psi']: indices = torsion_atoms.get(angle_name, None)
                           elif angle_name.startswith('chi') and angle_name in chi_atoms: indices = chi_atoms.get(angle_name, None)

                           if isinstance(indices, list) and len(indices) == 4 and None not in indices:
                                if all(0 <= idx < num_atoms for idx in indices):
                                     for list_idx, atom_idx in enumerate(indices):
                                          indices_lists[angle_name][list_idx].append(atom_idx)
                                     residue_indices_lists[angle_name].append(r_idx)
                                     valid_angle_mask_list[r_idx][type_idx] = True

                 # Convert lists to tensors
                 try:
                      for angle_name in angle_types_all:
                           if residue_indices_lists[angle_name]:
                                dihedral_info_precomputed[angle_name] = {
                                     'indices': [torch.tensor(lst, dtype=torch.long, device=device) for lst in indices_lists[angle_name]],
                                     'res_idx': torch.tensor(residue_indices_lists[angle_name], dtype=torch.long, device=device) }
                           else: dihedral_info_precomputed[angle_name] = {'indices': None, 'res_idx': None}
                      dihedral_mask_all = torch.tensor(valid_angle_mask_list, dtype=torch.bool, device=device)
                      logger.info("Finished precomputing dihedral info.")
                      if use_debug and dihedral_mask_all is not None:
                          logger.debug(f"DEBUG [main]: dihedral_mask_all shape: {dihedral_mask_all.shape}")
                          # Add more debug prints for mask sums if needed

                 except Exception as e: logger.error(f"Error converting dihedral lists to tensors: {e}", exc_info=True); dihedral_info_precomputed={}; dihedral_mask_all=None; num_res=None
            else: logger.warning("Invalid or empty torsion info, dihedral loss disabled."); dihedral_info_precomputed={}; dihedral_mask_all=None; num_res=None
        except Exception as e: logger.error(f"Error loading/processing torsion file {torsion_json_path}: {e}", exc_info=True); dihedral_info_precomputed={}; dihedral_mask_all=None; num_res=None
    else: logger.warning(f"Torsion file not found: {torsion_json_path}. Dihedral loss disabled.")

    # --- 6) Backbone Decoder Training/Loading ---
    stepA_conf = config.get("decoderB_training", {})
    pooling_dim_backbone = tuple(config.get("pooling_dim_backbone", [20, 4]))
    backbone_decoder_ckpt = os.path.join(ckpt_dir, config.get("bb_decoder_ckpt", "decoder_backbone.pth"))
    backbone_decoder_model = BackboneDecoder(
        num_total_atoms=num_atoms, backbone_indices=backbone_indices, emb_dim=hidden_dim,
        pooling_dim=pooling_dim_backbone, mlp_depth=stepA_conf.get("decoder_depth", 2),
        mlp_hidden_dim=stepA_conf.get("mlp_hidden_dim", 128))
    logger.info("--- Training/loading Backbone Decoder ---")
    backbone_decoder_model = train_backbone_decoder(
        model=backbone_decoder_model, train_loader=train_loader_dec, test_loader=test_loader_dec,
        device=device, logger=logger, config=stepA_conf, checkpoint_path=backbone_decoder_ckpt,
        z_ref=z_ref, dihedral_info_precomputed=dihedral_info_precomputed,
        dihedral_mask_all=dihedral_mask_all, num_res=num_res)
    backbone_decoder_model.eval().to(device) # Ensure on device and eval mode

    # --- 7) Sidechain Decoder Training/Loading ---
    stepB_conf = config.get("decoderSC_training", {})
    pooling_dim_sidechain = tuple(config.get("pooling_dim_sidechain", [20, 4]))
    sidechain_decoder_ckpt = os.path.join(ckpt_dir, config.get("sc_decoder_ckpt","decoder_sidechain.pth"))
    sidechain_decoder_model = SidechainDecoder(
        num_total_atoms=num_atoms, sidechain_indices=sidechain_indices, backbone_indices=backbone_indices,
        emb_dim=hidden_dim, pooling_dim=pooling_dim_sidechain, mlp_depth=stepB_conf.get("decoder_depth", 2),
        mlp_hidden_dim=stepB_conf.get("mlp_hidden_dim", 128), arch_type=stepB_conf.get("arch_type", 0))
    logger.info("--- Training/loading Sidechain Decoder ---")
    sidechain_decoder_model = train_sidechain_decoder(
        model=sidechain_decoder_model, train_loader=train_loader_dec, test_loader=test_loader_dec,
        backbone_decoder=backbone_decoder_model, device=device, logger=logger,
        config=stepB_conf, checkpoint_path=sidechain_decoder_ckpt, z_ref=z_ref,
        dihedral_info_precomputed=dihedral_info_precomputed, dihedral_mask_all=dihedral_mask_all, num_res=num_res)
    sidechain_decoder_model.eval().to(device) # Ensure on device and eval mode

    logger.info("All training tasks completed successfully!")

    # --- ADDED: Step 7.5 - Load Optional Diffused Embeddings ---
    # --- MODIFIED: Step 7.5 - Load Optional Diffused Embeddings with Debugging ---
    diff_bb_torch = None
    diff_sc_torch = None
    use_diffusion_override = False # Flag to pass to export function
    bb_pool_dim_expected = backbone_decoder_model.pool_output_dim
    sc_pool_dim_expected = sidechain_decoder_model.pool_output_dim

    # Log expected dimensions regardless of whether override is used
    logger.debug(f"DEBUG: Expected BB pool dim: {bb_pool_dim_expected}")
    logger.debug(f"DEBUG: Expected SC pool dim: {sc_pool_dim_expected}")

    if args.use_diffusion:
        if not args.diffused_backbone_h5 or not os.path.isfile(args.diffused_backbone_h5):
            logger.warning(f"use_diffusion=True but diffused_backbone_h5 path invalid or not found: {args.diffused_backbone_h5}. Skipping diffusion override.")
        elif not args.diffused_sidechain_h5 or not os.path.isfile(args.diffused_sidechain_h5):
            logger.warning(f"use_diffusion=True but diffused_sidechain_h5 path invalid or not found: {args.diffused_sidechain_h5}. Skipping diffusion override.")
        else:
            logger.info("Loading diffused pooled embeddings to potentially override final structure generation.")
            try:
                # Load data
                with h5py.File(args.diffused_backbone_h5, "r") as f:
                    if "generated_diffusion" in f: diff_bb_np = f["generated_diffusion"][:]
                    elif "backbone_pooled" in f: diff_bb_np = f["backbone_pooled"][:]
                    else: raise KeyError(f"Cannot find expected dataset ('generated_diffusion' or 'backbone_pooled') in {args.diffused_backbone_h5}")

                with h5py.File(args.diffused_sidechain_h5, "r") as f:
                     if "generated_diffusion" in f: diff_sc_np = f["generated_diffusion"][:]
                     elif "sidechain_pooled" in f: diff_sc_np = f["sidechain_pooled"][:]
                     else: raise KeyError(f"Cannot find expected dataset ('generated_diffusion' or 'sidechain_pooled') in {args.diffused_sidechain_h5}")

                logger.debug(f"DEBUG: Loaded raw diffused shapes: BB={diff_bb_np.shape if diff_bb_np is not None else 'None'}, SC={diff_sc_np.shape if diff_sc_np is not None else 'None'}")

                # Reshape if needed
                if diff_bb_np is not None and diff_bb_np.ndim > 2:
                    N = diff_bb_np.shape[0]
                    expected_elements = N * bb_pool_dim_expected
                    if diff_bb_np.size == expected_elements:
                        diff_bb_np = diff_bb_np.reshape(N, bb_pool_dim_expected)
                        logger.debug(f"DEBUG: Reshaped BB diffused to {diff_bb_np.shape}")
                    else:
                        logger.warning(f"Cannot reshape BB diffused array {diff_bb_np.shape} to expected elements {expected_elements}. Shape mismatch.")

                if diff_sc_np is not None and diff_sc_np.ndim > 2:
                    N = diff_sc_np.shape[0]
                    expected_elements = N * sc_pool_dim_expected
                    if diff_sc_np.size == expected_elements:
                         diff_sc_np = diff_sc_np.reshape(N, sc_pool_dim_expected)
                         logger.debug(f"DEBUG: Reshaped SC diffused to {diff_sc_np.shape}")
                    else:
                         logger.warning(f"Cannot reshape SC diffused array {diff_sc_np.shape} to expected elements {expected_elements}. Shape mismatch.")

                # Log shapes *after* potential reshaping, before the check
                logger.debug(f"DEBUG: Shape of diff_bb_np before final check: {diff_bb_np.shape if diff_bb_np is not None else 'None'}")
                logger.debug(f"DEBUG: Shape of diff_sc_np before final check: {diff_sc_np.shape if diff_sc_np is not None else 'None'}")

                # Perform the final dimension check
                bb_check_ok = diff_bb_np is not None and diff_bb_np.ndim == 2 and diff_bb_np.shape[1] == bb_pool_dim_expected
                sc_check_ok = diff_sc_np is not None and diff_sc_np.ndim == 2 and diff_sc_np.shape[1] == sc_pool_dim_expected
                logger.debug(f"DEBUG: BB final shape check result: {bb_check_ok} (Actual Dim: {diff_bb_np.shape[1] if diff_bb_np is not None and diff_bb_np.ndim == 2 else 'N/A'}, Expected Dim: {bb_pool_dim_expected})")
                logger.debug(f"DEBUG: SC final shape check result: {sc_check_ok} (Actual Dim: {diff_sc_np.shape[1] if diff_sc_np is not None and diff_sc_np.ndim == 2 else 'N/A'}, Expected Dim: {sc_pool_dim_expected})")

                if bb_check_ok and sc_check_ok:
                    diff_bb_torch = torch.from_numpy(diff_bb_np).float()
                    diff_sc_torch = torch.from_numpy(diff_sc_np).float()
                    logger.info(f"Loaded and processed diffused embeddings: BB={diff_bb_torch.shape}, SC={diff_sc_torch.shape}")
                    use_diffusion_override = True
                    logger.debug(f"DEBUG: Setting use_diffusion_override = {use_diffusion_override} (Checks Passed)")
                else:
                     logger.error("Final shape check failed for diffused embeddings. Disabling override.")
                     use_diffusion_override = False
                     logger.debug(f"DEBUG: Setting use_diffusion_override = {use_diffusion_override} (Checks Failed)")

            except Exception as e:
                logger.error(f"Error loading or processing diffused embeddings: {e}", exc_info=True)
                logger.warning("Disabling diffusion override due to loading error.")
                use_diffusion_override = False
                diff_bb_torch = None
                diff_sc_torch = None
                logger.debug(f"DEBUG: Setting use_diffusion_override = {use_diffusion_override} (Exception during loading)")
    # --- END MODIFIED ---
    '''
    # --- 8) Export Final Outputs ---
    logger.info("--- Exporting Final Outputs ---")
    # Add debug log before calling export
    logger.debug(f"DEBUG: Calling export_final_outputs with use_diff = {use_diffusion_override}")
    export_final_outputs(
        raw_dataset=dataset, dec_dataset=dec_dataset, hno_model=hno_model,
        backbone_decoder=backbone_decoder_model, sidechain_decoder=sidechain_decoder_model,
        z_ref=z_ref, num_atoms=num_atoms, struct_dir=struct_dir, latent_dir=latent_dir, device=device,
        use_diff=use_diffusion_override, # Pass the determined flag
        diff_bb=diff_bb_torch,
        diff_sc=diff_sc_torch
        )
    '''
    sys.stdout.flush()
    logger.info("Script finished successfully.")

if __name__ == "__main__":
    main()
