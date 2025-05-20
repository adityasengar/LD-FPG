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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import ChebConv
from torch_cluster import knn_graph
from sklearn.model_selection import train_test_split

################################################################################
# Argument Parsing
################################################################################
parser = argparse.ArgumentParser(
    description="All-Atom Protein Reconstruction with Contiguous Indexing & Optional Debug (Optimized Decoder)"
)
parser.add_argument(
    "--config", type=str, required=True, help="Path to the YAML configuration file."
)
parser.add_argument("--debug", action="store_true", help="Enable debug-level logging.")
parser.add_argument(
    "--log_file", type=str, default="logfile_optimized.log", help="Path to the log file."
)
args = parser.parse_args()

################################################################################
# Logging Setup
################################################################################
logger = logging.getLogger("ProteinReconstructionOptimized") # Changed logger name
logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

fh = logging.FileHandler(args.log_file, mode="w")
fh.setLevel(logging.DEBUG if args.debug else logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG if args.debug else logging.INFO)

formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

logger.info("Logger initialized.")
if args.debug:
    logger.debug("Debug mode is ON.")
else:
    logger.info("Debug mode is OFF; minimal logs will be shown.")

################################################################################
# Load YAML Configuration
################################################################################
def load_config(config_path):
    with open(config_path, "r") as f:
        conf = yaml.safe_load(f)
    return conf

try:
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
except Exception as e:
    logger.error(f"Error loading config: {e}")
    sys.exit(1)

# Extract main parameters
json_path = config["json_path"]
knn_value = config["knn_value"]
cheb_order = config["cheb_order"]
hidden_dim = config["hidden_dim"]
recon_output = config["recon_output"]

# Decoder 1 (HNO) parameters
decoder1_lr = config["decoder1"]["learning_rate"]
decoder1_epochs = config["decoder1"]["num_epochs"]
decoder1_bsize = config["decoder1"]["batch_size"]

# Decoder 2 parameters
decoder2_lr = config["decoder2"]["learning_rate"]
decoder2_epochs = config["decoder2"]["num_epochs"]
decoder2_bsize = config["decoder2"]["batch_size"]

# Decoder 2 settings
d2_output_height = config["decoder2_settings"]["output_height"]
d2_output_width = config["decoder2_settings"]["output_width"]
d2_pooling_type = config["decoder2_settings"]["pooling_type"]  # 'blind' or 'residue'
d2_weight_value = config["decoder2_settings"]["weight_value"]
d2_output_height2 = config["decoder2_settings"].get("output_height2", None)
d2_output_width2 = config["decoder2_settings"].get("output_width2", None)

use_second_level_pooling = config["decoder2_settings"].get(
    "use_second_level_pooling", False
)
use_cross_attention = config["decoder2_settings"].get("use_cross_attention", False)
cross_attention_type = config["decoder2_settings"].get(
    "cross_attention_type", "global"
)
conditioner_mode = config["decoder2_settings"].get(
    "conditioner_mode", "X_ref"
)
d2_num_hidden_layers = config["decoder2_settings"].get(
    "num_hidden_layers", 2
)

# Override settings for ProteinStateReconstructor2D
override_residues_config = config["decoder2_settings"].get("override_residues", None) # Expect list or null
special_res_file_config = config["decoder2_settings"].get("special_res_file", None) # Expect path string or null


device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

################################################################################
# Directory Setup
################################################################################
current_dir = os.getcwd()
checkpoint_dir = os.path.join(current_dir, "checkpoints") # Changed dir name
latent_reps_dir = os.path.join(current_dir, "latent_reps") # Changed dir name
structures_dir = os.path.join(current_dir, "structures")   # Changed dir name

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(latent_reps_dir, exist_ok=True)
os.makedirs(structures_dir, exist_ok=True)

logger.debug(
    f"Directories:\n  checkpoints: {checkpoint_dir}\n  latent_reps: {latent_reps_dir}\n  structures: {structures_dir}"
)

################################################################################
# Utility Functions
################################################################################
def save_checkpoint(state, filename):
    torch.save(state, filename)
    logger.debug(f"Checkpoint saved to {filename}")
    sys.stdout.flush()

def load_checkpoint(model, optimizer, filename):
    start_epoch = 0
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint from '{filename}'")
        sys.stdout.flush()
        checkpoint = torch.load(filename, map_location=device)
        start_epoch = checkpoint["epoch"]
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            logger.warning(f"Could not load model state_dict strictly: {e}. Trying non-strict loading.")
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Move optimizer state to device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        logger.info(f"Loaded checkpoint at epoch {start_epoch}")
        sys.stdout.flush()
    else:
        logger.info(f"No checkpoint found at '{filename}'. Training from scratch.")
        sys.stdout.flush()
    return model, optimizer, start_epoch

def compute_centroid(X):
    return X.mean(dim=0)

def kabsch_algorithm(P, Q):
    centroid_P = compute_centroid(P)
    centroid_Q = compute_centroid(Q)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    C = torch.mm(Q_centered.T, P_centered)
    try:
        V, S, W = torch.linalg.svd(C) # More modern SVD
    except AttributeError: # Fallback for older PyTorch
        V, S, W = torch.svd(C)


    d = torch.det(torch.mm(V, W.T))
    if d < 0:
        V[:, -1] = -V[:, -1]

    U = torch.mm(V, W.T)
    Q_aligned = torch.mm(Q_centered, U) + centroid_P
    return U, Q_aligned

################################################################################
# Reindexing Residue Indices
################################################################################
def remap_residue_indices(residue_atom_indices):
    flattened = [atom for residue in residue_atom_indices for atom in residue]
    if not flattened: # Handle empty case
        logger.warning("Residue_atom_indices is empty or contains only empty residues.")
        return [], {}
    unique_atoms = sorted(set(flattened))
    logger.debug(
        f"Unique atoms found: {len(unique_atoms)}. "
        f"Max original ID = {max(unique_atoms)}, Min = {min(unique_atoms)}"
    )
    sys.stdout.flush()

    old2new_map = {old_id: idx for idx, old_id in enumerate(unique_atoms)}
    if args.debug:
        first_few_items = list(old2new_map.items())[:5]
        logger.debug(f"[old2new_map sample] -> {first_few_items}")
        sys.stdout.flush()

    new_residue_indices = []
    for residue in residue_atom_indices:
        new_list = [old2new_map[a] for a in residue if a in old2new_map] # Ensure atom is in map
        new_residue_indices.append(new_list)
    
    # Filter out empty lists that might result if a residue had no mappable atoms
    new_residue_indices = [res for res in new_residue_indices if res]

    return new_residue_indices, old2new_map

################################################################################
# Data Loading & Alignment
################################################################################
def load_heavy_atom_coords_from_json(json_file):
    logger.info(f"Loading JSON from {json_file}")
    sys.stdout.flush()
    with open(json_file, "r") as f:
        data = json.load(f)

    if not data:
        logger.error(f"JSON file {json_file} is empty or invalid.")
        return [], []

    residue_keys_sorted = sorted(data.keys(), key=lambda x: int(x)) # Assuming keys are numeric strings
    if not residue_keys_sorted:
        logger.error("No residue keys found in JSON data.")
        return [], []
    logger.debug(f"Number of residues in JSON: {len(residue_keys_sorted)}")
    sys.stdout.flush()

    num_frames = len(data[residue_keys_sorted[0]]["heavy_atom_coords_per_frame"])
    logger.info(f"Number of frames = {num_frames}")
    sys.stdout.flush()

    original_residue_atom_indices = []
    for res_key in residue_keys_sorted:
        idx_list = data[res_key]["heavy_atom_indices"]
        original_residue_atom_indices.append(idx_list)

    coords_per_frame = []
    for frame_idx in range(num_frames):
        frame_coords_list = []
        for res_key in residue_keys_sorted:
            coords_this_res = data[res_key]["heavy_atom_coords_per_frame"][frame_idx]
            frame_coords_list.append(np.array(coords_this_res, dtype=np.float32))
        if not frame_coords_list: # If a frame has no coordinates
            logger.warning(f"Frame {frame_idx} resulted in no coordinates. Skipping.")
            continue
        try:
            frame_coords = np.concatenate(frame_coords_list, axis=0)
            coords_per_frame.append(torch.tensor(frame_coords, dtype=torch.float32))
        except ValueError as e:
             logger.error(f"Error concatenating coordinates for frame {frame_idx}: {e}. Check for empty coord lists.")
             # Potentially skip this frame or handle error
             continue


    return coords_per_frame, original_residue_atom_indices

def align_frames_to_first(coords_list):
    if not coords_list:
        logger.warning("Coordinate list is empty. Cannot align.")
        return []
    logger.info("Aligning all frames to the first frame via Kabsch...")
    sys.stdout.flush()
    reference = coords_list[0].to(device) # Move reference to device for Kabsch
    aligned = [coords_list[0]] # First frame is the reference, keep on CPU

    for i, coords in enumerate(coords_list[1:], 1): # Start from the second frame
        coords_dev = coords.to(device)
        try:
            _, coords_aligned_dev = kabsch_algorithm(reference, coords_dev)
            aligned.append(coords_aligned_dev.cpu()) # Move aligned back to CPU
        except Exception as e:
            logger.error(f"Kabsch alignment failed for frame {i}: {e}. Appending original.")
            aligned.append(coords) # Append original if alignment fails

        if args.debug and i < 5:
            logger.debug(f"Frame {i}: aligned shape={aligned[-1].shape}")
            sys.stdout.flush()
    return aligned

################################################################################
# Build Graph Dataset
################################################################################
def build_graph_dataset(coords_list, knn_neighbors=4):
    logger.info("Building PyG dataset with knn_graph.")
    sys.stdout.flush()
    dataset = []
    for idx, coords_cpu in enumerate(coords_list):
        coords_dev = coords_cpu.to(device) # Move to device for knn_graph
        try:
            edge_index = knn_graph(coords_dev, k=knn_neighbors, batch=None, loop=False)
            # Create Data object with tensors on CPU
            data = Data(x=coords_cpu, edge_index=edge_index.cpu(), y=coords_cpu)
            dataset.append(data)
        except Exception as e:
            logger.error(f"Failed to build graph for frame {idx}: {e}. Skipping this frame.")
            continue

        if args.debug and idx < 5:
            logger.debug(
                f"[Dataset] Frame {idx}: x.shape={coords_cpu.shape}, edge_index.shape={edge_index.cpu().shape}"
            )
            sys.stdout.flush()
    return dataset

################################################################################
# HNO Model
################################################################################
class HNO(nn.Module):
    _logged_forward_once = False

    def __init__(self, hidden_dim, K):
        super().__init__()
        logger.debug(f"Initializing HNO with hidden_dim={hidden_dim}, K={K}")
        sys.stdout.flush()
        self.conv1 = ChebConv(3, hidden_dim, K=K)
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.conv3 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.conv4 = ChebConv(hidden_dim, hidden_dim, K=K)

        self.bano1 = nn.BatchNorm1d(hidden_dim)
        self.bano2 = nn.BatchNorm1d(hidden_dim)
        self.bano3 = nn.BatchNorm1d(hidden_dim)
        self.mlpRep = nn.Linear(hidden_dim, 3)

    def forward(self, x, edge_index):
        if args.debug and (not HNO._logged_forward_once) and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[HNO:forward] Input x.shape={x.shape}")
            sys.stdout.flush()

        x = self.conv1(x, edge_index)
        if args.debug and (not HNO._logged_forward_once) and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[HNO:forward] After conv1 => {x.shape}")
            sys.stdout.flush()
        x = F.leaky_relu(x)
        x = self.bano1(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.bano2(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x) # Note: Original uses F.relu here
        x = self.bano3(x)

        x_repr = self.conv4(x, edge_index) # Representation before normalization for mlpRep
        x_norm_repr = F.normalize(x_repr, p=2, dim=1) # Normalized for potential use as z_ref
        
        if args.debug and (not HNO._logged_forward_once) and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[HNO:forward] After conv4 (before mlpRep) => {x_repr.shape}")
            sys.stdout.flush()
            
        # In original Code 1, mlpRep was applied to the output of conv4 *before* normalization
        # if it was for the main forward pass, but to the *normalized* output in forward_representation.
        # Let's use the un-normalized for direct reconstruction, and forward_representation will give normalized.
        x_out = self.mlpRep(x_repr) # Use un-normalized for reconstruction
        
        if args.debug and (not HNO._logged_forward_once) and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[HNO:forward] Output => {x_out.shape}")
            HNO._logged_forward_once = True # Log only once
            sys.stdout.flush()
        return x_out


    def forward_representation(self, x, edge_index):
        # This explicitly returns the L2 normalized representation after conv4
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.bano1(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.bano2(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.bano3(x)

        x = self.conv4(x, edge_index)
        x = F.normalize(x, p=2, dim=1) # Key: normalize for z_ref
        return x

################################################################################
# Train HNO
################################################################################
def train_hno_model(
    model,
    train_loader,
    test_loader,
    num_epochs,
    learning_rate,
    checkpoint_path,
    save_interval=10,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    if start_epoch >= num_epochs:
        logger.info(f"HNO training already completed up to epoch {start_epoch}/{num_epochs}. Skipping training.")
        model.eval()
        return model

    logger.info(
        f"Starting HNO from epoch {start_epoch + 1}, total epochs={num_epochs}, LR={learning_rate}"
    )
    sys.stdout.flush()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)
            loss = criterion(pred, data.y) # Target is data.y (ground truth coordinates)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                pred = model(data.x, data.edge_index)
                val_loss = criterion(pred, data.y) # Target is data.y
                test_loss += val_loss.item()
        avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0

        logger.info(
            f"[HNO] Epoch {epoch+1}/{num_epochs} => Train Loss={avg_loss:.6f}, Test Loss={avg_test_loss:.6f}"
        )
        sys.stdout.flush()

        if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
            checkpoint_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint_state, checkpoint_path)
            logger.info(f"HNO checkpoint saved at epoch {epoch+1} -> {checkpoint_path}")
            sys.stdout.flush()
    model.eval()
    return model

################################################################################
# Simple Cross-Attention Module (Kept for structural consistency with original Code 1)
################################################################################
class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, kv_dim, attention_dim=64):
        super().__init__()
        self.query_layer = nn.Linear(query_dim, attention_dim, bias=False)
        self.key_layer = nn.Linear(kv_dim, attention_dim, bias=False)
        self.value_layer = nn.Linear(kv_dim, attention_dim, bias=False)
        self.scale = attention_dim**-0.5

    def forward(self, query, key, value):
        original_shape = query.shape
        if query.dim() == 3:
            B, N_seg, q_dim = query.shape
            query = query.view(B * N_seg, q_dim)
        else:
            B, q_dim = query.shape
            N_seg = 1

        B_kv, num_kv, kv_d = key.shape
        Q = self.query_layer(query).unsqueeze(1)
        K = self.key_layer(key)
        V = self.value_layer(value)
        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.squeeze(1)
        if N_seg > 1:
            out = out.view(B, N_seg, -1)
        return out

################################################################################
# MLP Builder for Decoder2 (Kept as is)
################################################################################
def build_decoder_mlp(input_dim, output_dim, num_hidden_layers, hidden_dim_mlp=128): # Renamed hidden_dim for clarity
    layers = []
    current_dim = input_dim
    if num_hidden_layers < 1: # Direct connection
        layers.append(nn.Linear(current_dim, output_dim))
    else:
        for i in range(num_hidden_layers -1): # Corrected loop for num_hidden_layers
            layers.append(nn.Linear(current_dim, hidden_dim_mlp))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim_mlp))
            current_dim = hidden_dim_mlp
        layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)

################################################################################
# ProteinStateReconstructor2D (MODIFIED with atom_to_residue_map)
################################################################################
class ProteinStateReconstructor2D(nn.Module):
    _logged_forward_once = False

    def __init__(
        self,
        input_dim,        # HNO's hidden_dim
        num_nodes,        # Total number of unique atoms after remapping
        pooling_type="blind",
        residue_atom_indices=None, # List of lists (remapped atom indices per residue)
        output_size_per_segment=(1, 10),
        second_level_output_size=None,
        use_second_level_pooling=False,
        use_cross_attention=False, # Not actively used in this optimized version's core logic
        cross_attention_type=None, # Not actively used
        conditioner_mode="z_ref",
        num_hidden_layers=2,
        mlp_hidden_dim=128, # Added for clarity
        override_residues=None,
        special_res_file=None,
    ):
        super().__init__()
        self.pooling_type = pooling_type
        self.num_nodes = num_nodes
        self.input_dim = input_dim # This is the dimension of HNO embeddings
        self.conditioner_mode = conditioner_mode # 'X_ref' or 'z_ref'
        self.use_second_level_pooling = use_second_level_pooling
        self.output_size_per_segment = output_size_per_segment
        self.second_level_output_size = second_level_output_size

        self.override_residues = override_residues if override_residues is not None else []
        self.generated_arr = None

        if special_res_file is not None and len(self.override_residues) > 0:
            logger.info(f"Attempting to load override embeddings from: {special_res_file}")
            try:
                with h5py.File(special_res_file, "r") as f:
                    # Assuming dataset name, adjust if different
                    arr = f["generated_embeddings"][:] # Or "pooled_latent" etc.
                if np.isnan(arr).any():
                    logger.error("Generated override embeddings contain NaN values!")
                else:
                    logger.info(f"Generated embeddings loaded with shape {arr.shape} and appear valid.")
                    self.generated_arr = arr
                    if self.generated_arr.shape[1] != len(self.override_residues):
                        logger.error(
                            f"Mismatch: special_res_file has {self.generated_arr.shape[1]} residues for override, "
                            f"but override_residues has length {len(self.override_residues)}. Disabling override."
                        )
                        self.generated_arr = None # Disable if mismatch
            except Exception as e:
                logger.error(f"Could not load or process special_res_file {special_res_file}: {e}. Override disabled.")
                self.generated_arr = None
        elif special_res_file is not None and not self.override_residues:
            logger.info(f"special_res_file '{special_res_file}' provided, but override_residues list is empty. No override will occur.")


        # Determine conditioner dimension based on mode and input_dim (HNO's output)
        # This is used for the final MLP's input size calculation.
        # The actual conditioner tensor is passed during forward.
        self.conditioner_dim_expected = 3 if conditioner_mode == "X_ref" else self.input_dim

        if pooling_type == "blind":
            # For blind pooling, all atoms effectively belong to one segment
            self.segments_indices_map = [list(range(num_nodes))] # A single segment containing all atoms
            self.segment_pools = nn.ModuleList(
                [nn.AdaptiveAvgPool2d(self.output_size_per_segment)]
            )
            self.num_effective_segments = 1
            seg_pooled_size = self.output_size_per_segment[0] * self.output_size_per_segment[1]
            self.pooled_dim_per_segment = seg_pooled_size # This is D_pool_level1
            self.final_effective_pooled_dim = self.pooled_dim_per_segment # No second level for blind
            self.global_pool2 = None
            # For blind pooling, atom_to_residue_map maps all atoms to segment 0
            self.register_buffer('atom_to_segment_map', torch.zeros(num_nodes, dtype=torch.long), persistent=False)

        elif pooling_type == "residue":
            if residue_atom_indices is None or not any(residue_atom_indices): # Check if empty or list of empty lists
                raise ValueError("residue_atom_indices is required and must not be empty for residue-level pooling.")
            
            self.segments_indices_map = residue_atom_indices # Store the list of lists of atom indices
            self.num_effective_segments = len(self.segments_indices_map)
            self.segment_pools = nn.ModuleList()
            for _ in self.segments_indices_map:
                pool = nn.AdaptiveAvgPool2d(self.output_size_per_segment)
                self.segment_pools.append(pool)
            
            self.pooled_dim_per_segment = self.output_size_per_segment[0] * self.output_size_per_segment[1] # D_pool_level1

            if use_second_level_pooling and second_level_output_size is not None and self.num_effective_segments > 0 :
                self.global_pool2 = nn.AdaptiveAvgPool2d(second_level_output_size)
                self.final_effective_pooled_dim = second_level_output_size[0] * second_level_output_size[1]
            else:
                self.global_pool2 = None
                self.final_effective_pooled_dim = self.pooled_dim_per_segment

            # --- Create atom_to_residue_map for 'residue' pooling ---
            atom_to_res_map_list = [-1] * num_nodes
            valid_map = True
            for res_idx, atom_indices_in_res in enumerate(self.segments_indices_map):
                for atom_idx in atom_indices_in_res:
                    if 0 <= atom_idx < num_nodes:
                        if atom_to_res_map_list[atom_idx] != -1:
                            logger.warning(f"Atom {atom_idx} assigned to multiple residues. Using first assignment (res {atom_to_res_map_list[atom_idx]}). Check residue_atom_indices for overlaps.")
                        else:
                            atom_to_res_map_list[atom_idx] = res_idx
                    else:
                        logger.error(f"Atom index {atom_idx} in residue {res_idx} is out of bounds (0-{num_nodes-1}).")
                        valid_map = False
            
            if not valid_map:
                raise ValueError("Invalid atom indices found in residue_atom_indices. See logs.")

            if -1 in atom_to_res_map_list:
                unmapped_indices = [i for i, x in enumerate(atom_to_res_map_list) if x == -1]
                logger.warning(f"{len(unmapped_indices)} atoms not mapped to any residue (e.g., {unmapped_indices[:5]}). "
                               f"These atoms will effectively not receive specific residue-pooled features if not handled.")
                # Decide on a strategy: error, or assign to a dummy residue index, or ensure all atoms are covered.
                # For now, we proceed, but if an atom with map value -1 is accessed, it will error.
                # It's better to ensure residue_atom_indices covers all 0..num_nodes-1 atoms.

            self.register_buffer('atom_to_segment_map', torch.tensor(atom_to_res_map_list, dtype=torch.long), persistent=False)
        else:
            raise ValueError(f"Unknown pooling_type={pooling_type}")

        # Final MLP input dimension: effective pooled dimension + conditioner dimension
        self.final_mlp_in_dim = self.final_effective_pooled_dim + self.conditioner_dim_expected
        
        self.decoder = build_decoder_mlp(
            input_dim=self.final_mlp_in_dim,
            output_dim=3, # Predicts 3D coordinates
            num_hidden_layers=num_hidden_layers,
            hidden_dim_mlp=mlp_hidden_dim, # Pass the specific hidden_dim for MLP
        )
        logger.info(f"ProteinStateReconstructor2D initialized. Pooling: {self.pooling_type}, "
                    f"Num Segments: {self.num_effective_segments}, "
                    f"Pooled dim per seg: {self.pooled_dim_per_segment}, "
                    f"Final effective pooled dim: {self.final_effective_pooled_dim}, "
                    f"Conditioner dim expected: {self.conditioner_dim_expected}, "
                    f"MLP input dim: {self.final_mlp_in_dim}")


    def get_pooled_latent(self, x, conditioner=None): # x is [B*N_nodes, input_dim]
        """
        Standard pooling step without overrides. conditioner is not used here.
        Returns pooled_latent of shape [B, NumEffectiveSegments, FinalEffectivePooledDim]
        """
        B_times_N, _ = x.shape
        batch_size = B_times_N // self.num_nodes
        x_reshaped = x.view(batch_size, self.num_nodes, -1) # [B, N, D_in]

        if self.pooling_type == "blind":
            # x_reshaped[:, self.segments_indices_map[0], :] is equivalent to x_reshaped
            seg_x = x_reshaped.unsqueeze(1)  # [B, 1, N_nodes, D_in]
            p_2d = self.segment_pools[0](seg_x)  # [B, 1, H, W] (here D_in is pooled as W)
            pooled_flat = p_2d.squeeze(1).view(batch_size, -1) # [B, H*W]
            # Result shape [B, 1 segment, pooled_dim_per_segment]
            pooled_latent_to_return = pooled_flat.unsqueeze(1)
        
        elif self.pooling_type == "residue":
            pooled_segments_level1_list = []
            for i, atom_indices_in_segment in enumerate(self.segments_indices_map):
                if not atom_indices_in_segment: # Skip empty segments
                     # Append zeros or handle as error. For shape consistency, append zeros.
                     pooled_segments_level1_list.append(torch.zeros(batch_size, self.pooled_dim_per_segment, device=x.device, dtype=x.dtype))
                     continue
                # Select atoms for this segment: [B, N_atoms_in_seg, D_in]
                seg_x_atoms = x_reshaped[:, atom_indices_in_segment, :]
                seg_x_atoms_unsqueezed = seg_x_atoms.unsqueeze(1) # [B, 1, N_atoms_in_seg, D_in]
                
                p_2d = self.segment_pools[i](seg_x_atoms_unsqueezed) # [B, 1, H, W] (D_in pooled as W)
                p_2d_flat = p_2d.squeeze(1).view(batch_size, -1) # [B, pooled_dim_per_segment]
                pooled_segments_level1_list.append(p_2d_flat)
            
            if not pooled_segments_level1_list: # Should not happen if init checks pass
                 return torch.empty(batch_size, 0, self.final_effective_pooled_dim, device=x.device, dtype=x.dtype)

            level1_concat = torch.stack(pooled_segments_level1_list, dim=1)  # [B, NumResidues, pooled_dim_per_segment]

            if self.global_pool2 is not None:
                level1_concat_4d = level1_concat.unsqueeze(1) # [B, 1, NumResidues, pooled_dim_per_segment]
                # global_pool2 pools over NumResidues (dim 2) and pooled_dim_per_segment (dim 3)
                pooled_2d_level2 = self.global_pool2(level1_concat_4d) # [B, 1, H2, W2]
                pooled_2d_level2_flat = pooled_2d_level2.squeeze(1).view(batch_size, -1) # [B, final_effective_pooled_dim]
                # Repeat this global vector for each segment "slot"
                pooled_latent_to_return = pooled_2d_level2_flat.unsqueeze(1).repeat(1, self.num_effective_segments, 1)
            else:
                pooled_latent_to_return = level1_concat
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")
            
        return pooled_latent_to_return


    def forward(self, x, batch, conditioner_ref_frame, use_override=True):
        """
        x: Input atom embeddings [B*N_nodes, input_dim] from HNO.
        batch: PyG batch object (not directly used if B*N_nodes is given).
        conditioner_ref_frame: Conditioner tensor [num_nodes, conditioner_dim_expected] (e.g., X_ref or z_ref).
        """
        current_dev = x.device
        B_times_N, _ = x.shape # x is [B*N, D_in]
        
        if B_times_N == 0: # Handle empty batch
            logger.warning("Decoder2 forward received an empty input tensor x.")
            return torch.empty(0, 3, device=current_dev, dtype=x.dtype)
            
        batch_size = B_times_N // self.num_nodes
        if batch_size * self.num_nodes != B_times_N:
            raise ValueError(f"Mismatch in x.shape ({x.shape}) vs. num_nodes ({self.num_nodes}) & inferred batch_size ({batch_size})")

        # x_reshaped: [B, num_nodes, input_dim]
        x_reshaped = x.view(batch_size, self.num_nodes, self.input_dim)

        # Expand conditioner_ref_frame: [num_nodes, D_cond] -> [B, num_nodes, D_cond]
        conditioner_expanded = conditioner_ref_frame.to(current_dev).unsqueeze(0).expand(batch_size, -1, -1)
        if conditioner_expanded.shape[2] != self.conditioner_dim_expected:
            raise ValueError(f"Conditioner dimension mismatch. Expected {self.conditioner_dim_expected}, "
                             f"got {conditioner_expanded.shape[2]} from conditioner_ref_frame.")

        # 1. Get pooled latent features (before override)
        # This call uses x, which is the full [B*N, D_in] tensor
        pooled_latent_for_map = self.get_pooled_latent(x)
        # pooled_latent_for_map shape: [B, NumEffectiveSegments, FinalEffectivePooledDim]

        # 2. Apply override if specified
        if use_override and self.generated_arr is not None and self.override_residues:
            num_avail_override_samples = self.generated_arr.shape[0]
            if batch_size > num_avail_override_samples:
                logger.warning(
                    f"Batch size ({batch_size}) > available override samples ({num_avail_override_samples}). "
                    f"Using modulo for override indices or disabling. For safety, disabling for this batch."
                ) # Or implement a strategy like modulo
            else:
                arr_slice = self.generated_arr[:batch_size] # Take first B samples
                arr_tensor = torch.from_numpy(arr_slice).to(current_dev).float() # [B, M, embed_dim_override]
                
                # Ensure override embedding dim matches pooled_latent_for_map's feature dim
                if arr_tensor.shape[2] != pooled_latent_for_map.shape[2]:
                    logger.error(f"Override embedding dimension ({arr_tensor.shape[2]}) mismatch with "
                                 f"pooled_latent_for_map dimension ({pooled_latent_for_map.shape[2]}). Skipping override.")
                else:
                    for i, r_idx in enumerate(self.override_residues):
                        if 0 <= r_idx < pooled_latent_for_map.shape[1]: # Check if residue index is valid
                            pooled_latent_for_map[:, r_idx, :] = arr_tensor[:, i, :]
                        else:
                            logger.warning(f"Override residue index {r_idx} is out of bounds for "
                                           f"pooled_latent_for_map with {pooled_latent_for_map.shape[1]} segments.")
        
        # 3. Assemble MLP Input using atom_to_segment_map
        # atom_to_segment_map has shape [num_nodes]
        # Its values range from 0 to NumEffectiveSegments-1
        current_atom_map = self.atom_to_segment_map.to(current_dev)
        
        # Filter map if it contains -1 (unmapped atoms) before gathering
        # This is a safety check; ideally, the map is dense and correct.
        if (current_atom_map < 0).any():
            logger.error("atom_to_segment_map contains invalid (negative) indices. This will cause an error.")
            # Raise error or handle: e.g., create a default feature for these atoms
            raise IndexError("Invalid indices in atom_to_segment_map")

        # Gather:
        # pooled_latent_for_map shape: [B, NumEffectiveSegments, FinalEffectivePooledDim]
        # current_atom_map shape:    [num_nodes]
        # Result per_atom_pooled_rep: [B, num_nodes, FinalEffectivePooledDim]
        try:
            per_atom_pooled_rep = pooled_latent_for_map[:, current_atom_map, :]
        except IndexError as e:
            logger.error(f"IndexError during gather operation. pooled_latent_for_map.shape={pooled_latent_for_map.shape}, "
                         f"max(current_atom_map)={current_atom_map.max() if current_atom_map.numel() > 0 else 'N/A'}. Error: {e}")
            raise

        # Concatenate with conditioner
        # conditioner_expanded shape: [B, num_nodes, D_cond]
        combined_features = torch.cat([per_atom_pooled_rep, conditioner_expanded], dim=-1)
        # combined_features shape: [B, num_nodes, FinalEffectivePooledDim + D_cond]
        
        if combined_features.shape[2] != self.final_mlp_in_dim:
             raise ValueError(f"MLP input dimension mismatch. Expected {self.final_mlp_in_dim}, "
                              f"got {combined_features.shape[2]}. Check pooled and conditioner dims.")

        final_mlp_input_flat = combined_features.view(B_times_N, self.final_mlp_in_dim)
        
        # 4. Decode
        X_pred = self.decoder(final_mlp_input_flat)  # [B*N_nodes, 3]

        if args.debug and (not ProteinStateReconstructor2D._logged_forward_once) and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Decoder2:forward] x_in.shape={x.shape}, conditioner_ref_frame.shape={conditioner_ref_frame.shape}")
            logger.debug(f"  x_reshaped.shape={x_reshaped.shape}, conditioner_expanded.shape={conditioner_expanded.shape}")
            logger.debug(f"  pooled_latent_for_map.shape={pooled_latent_for_map.shape}")
            logger.debug(f"  per_atom_pooled_rep.shape={per_atom_pooled_rep.shape}")
            logger.debug(f"  combined_features.shape={combined_features.shape}")
            logger.debug(f"  final_mlp_input_flat.shape={final_mlp_input_flat.shape}")
            logger.debug(f"  X_pred.shape={X_pred.shape}")
            ProteinStateReconstructor2D._logged_forward_once = True
            sys.stdout.flush()

        return X_pred


################################################################################
# Train Decoder2
################################################################################
def train_protein_state_reconstructor(
    model,
    train_loader,
    test_loader,
    num_epochs,
    learning_rate,
    checkpoint_path,
    conditioner_data, # This is the reference frame data (X_ref or z_ref) [N_nodes, D_cond]
    save_interval=10,
    weight_value=1.0, # Not explicitly used in this version's loss, but kept for signature
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    if start_epoch >= num_epochs:
        logger.info(f"Decoder2 training already completed up to epoch {start_epoch}/{num_epochs}. Skipping training.")
        model.eval()
        return model

    logger.info(
        f"Starting Decoder2 from epoch {start_epoch + 1}, total epochs={num_epochs}, LR={learning_rate}"
    )
    sys.stdout.flush()

    # conditioner_data is already on CPU, move to device once if static, or pass as is
    # The model's forward pass will move it to x.device

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        num_batches = len(train_loader)
        if num_batches == 0:
            logger.warning("Train loader is empty for Decoder2. Skipping epoch.")
            continue

        for data in train_loader: # data.x is HNO embeddings, data.coords is GT
            data = data.to(device)
            optimizer.zero_grad()
            # Pass the single reference frame conditioner_data
            X_pred = model(data.x, data.batch, conditioner_data)
            coords_gt = data.coords
            loss = criterion(X_pred, coords_gt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / num_batches

        model.eval()
        test_loss = 0
        num_test_batches = len(test_loader)
        if num_test_batches == 0:
            logger.warning("Test loader is empty for Decoder2.")
            avg_test_loss = 0.0
        else:
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    X_pred = model(data.x, data.batch, conditioner_data)
                    coords_gt = data.coords
                    val_loss = criterion(X_pred, coords_gt)
                    test_loss += val_loss.item()
            avg_test_loss = test_loss / num_test_batches

        logger.info(
            f"[Decoder2] Epoch {epoch+1}/{num_epochs} => Train Loss={avg_loss:.6f}, Test Loss={avg_test_loss:.6f}"
        )
        sys.stdout.flush()

        if (epoch + 1) % save_interval == 0 or (epoch+1) == num_epochs:
            checkpoint_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint_state, checkpoint_path)
            logger.info(
                f"Decoder2 checkpoint saved at epoch {epoch+1} -> {checkpoint_path}"
            )
            sys.stdout.flush()
    model.eval()
    return model

################################################################################
# Main Execution
################################################################################
if __name__ == "__main__":
    logger.info("Starting main script execution (Optimized Decoder)...")
    sys.stdout.flush()

    coords_per_frame, original_residue_atom_indices = load_heavy_atom_coords_from_json(json_path)
    if not coords_per_frame:
        logger.error("Failed to load coordinates. Exiting.")
        sys.exit(1)

    logger.info("Remapping residue_atom_indices to contiguous IDs...")
    sys.stdout.flush()
    # new_residue_indices contains atom indices remapped from 0 to N-1
    # old2new_map maps original atom IDs to these new 0..N-1 indices
    new_residue_indices, old2new_map = remap_residue_indices(original_residue_atom_indices)
    num_unique_atoms = len(old2new_map)
    if num_unique_atoms == 0:
        logger.error("No unique atoms found after remapping. Exiting.")
        sys.exit(1)
    logger.info(f"Number of unique atoms after reindexing: {num_unique_atoms}")
    sys.stdout.flush()
    
    if not new_residue_indices and d2_pooling_type == "residue":
        logger.error("Residue pooling selected, but no residue definitions found after remapping. Exiting.")
        sys.exit(1)


    aligned_frames = align_frames_to_first(coords_per_frame)
    if not aligned_frames:
        logger.error("Frame alignment failed. Exiting.")
        sys.exit(1)

    dataset = build_graph_dataset(aligned_frames, knn_neighbors=knn_value)
    if not dataset:
        logger.error("Dataset construction failed. Exiting.")
        sys.exit(1)
        
    first_data = dataset[0]
    if args.debug:
        logger.debug(f"Atom Indices Mapping (old -> new, sample): {list(old2new_map.items())[:5]}")
        logger.debug(f"First 5 atoms in data.x (remapped coords): {first_data.x[:5]}")
        sys.stdout.flush()

    logger.info(f"Dataset size: {len(dataset)} frames.")
    sys.stdout.flush()

    train_dataset_hno, test_dataset_hno = train_test_split(dataset, test_size=0.1, random_state=42)
    # Ensure num_workers and pin_memory are used if configured globally
    num_workers_config = config.get("num_workers", 0)
    pin_memory_config = True if device == "cuda" else False

    train_loader_hno = DataLoader(
        train_dataset_hno, batch_size=decoder1_bsize, shuffle=True, # Shuffle train
        num_workers=num_workers_config, pin_memory=pin_memory_config, drop_last=True
    )
    test_loader_hno = DataLoader(
        test_dataset_hno, batch_size=decoder1_bsize, shuffle=False,
        num_workers=num_workers_config, pin_memory=pin_memory_config
    )

    hno_model = HNO(hidden_dim, cheb_order)
    hno_ckpt_path = os.path.join(checkpoint_dir, "hno_checkpoint.pth")
    logger.info(
        f"Training HNO => epochs={decoder1_epochs}, LR={decoder1_lr}, bsize={decoder1_bsize}"
    )
    sys.stdout.flush()
    hno_model = train_hno_model( # Reassign to get the potentially loaded/trained model
        hno_model,
        train_loader_hno,
        test_loader_hno,
        num_epochs=decoder1_epochs,
        learning_rate=decoder1_lr,
        checkpoint_path=hno_ckpt_path,
        save_interval=config.get("hno_save_interval", 10),
    )

    if recon_output == 1: # Save HNO outputs
        logger.info("Generating HNO reconstructions & latent embeddings for entire dataset...")
        sys.stdout.flush()
        hno_model.eval()
        hno_recon_path = os.path.join(structures_dir, "hno_reconstructed_coords.h5")
        hno_latent_path = os.path.join(latent_reps_dir, "hno_latent_embeddings.h5")
        
        # Use the full dataset for HNO output generation for consistency, not just train split
        num_samples_hno_output = len(dataset) 
        hno_output_loader = DataLoader(dataset, batch_size=decoder1_bsize, shuffle=False, 
                                       num_workers=num_workers_config, pin_memory=pin_memory_config)

        pooled_dim_hno = hidden_dim

        with h5py.File(hno_recon_path, "w") as recon_h5, h5py.File(hno_latent_path, "w") as latent_h5:
            recon_dset_hno = recon_h5.create_dataset(
                "reconstructions", shape=(num_samples_hno_output, num_unique_atoms, 3), dtype="float32"
            )
            latent_dset_hno = latent_h5.create_dataset(
                "latent_embeddings", shape=(num_samples_hno_output, num_unique_atoms, pooled_dim_hno), dtype="float32"
            )
            
            current_idx_hno = 0
            with torch.no_grad():
                for data_batch in hno_output_loader:
                    data_batch = data_batch.to(device)
                    # HNO's forward_representation gives normalized embeddings
                    latent_batch = hno_model.forward_representation(data_batch.x, data_batch.edge_index)
                    # HNO's forward gives reconstructed coordinates
                    reconstruction_batch = hno_model(data_batch.x, data_batch.edge_index)

                    batch_actual_size = data_batch.num_graphs # Number of graphs in this batch

                    # Correctly slice and store batched outputs
                    start = current_idx_hno
                    end = current_idx_hno + batch_actual_size
                    
                    # Assuming data_batch.x are ground truth coordinates for these reconstructions
                    # This might be slightly off if recon_output==2's intent was to save GT as well.
                    # The original code saved data.x, which could be ground truth or a previous reconstruction.
                    # Let's save the actual reconstructions.
                    recon_dset_hno[start:end] = reconstruction_batch.view(batch_actual_size, num_unique_atoms, 3).cpu().numpy()
                    latent_dset_hno[start:end] = latent_batch.view(batch_actual_size, num_unique_atoms, pooled_dim_hno).cpu().numpy()
                    current_idx_hno = end
        logger.info(f"HNO outputs saved. Reconstructions: {hno_recon_path}, Latents: {hno_latent_path}")


    logger.info("Preparing dataset for Decoder2 using HNO latent embeddings...")
    sys.stdout.flush()
    decoder2_input_dataset = [] # data.x will be HNO embeddings, data.coords GT coords
    hno_model.eval() # Ensure HNO is in eval mode
    
    # Use a DataLoader for efficient batch processing for generating decoder inputs
    decoder_input_gen_loader = DataLoader(dataset, batch_size=decoder1_bsize, shuffle=False,
                                          num_workers=num_workers_config, pin_memory=pin_memory_config)
    with torch.no_grad():
        for data_batch in decoder_input_gen_loader:
            # data_batch contains original x (coords), edge_index, y (coords)
            data_batch_dev = data_batch.to(device)
            # Generate latent embeddings (these will be input 'x' to the decoder)
            latent_embeddings_batch = hno_model.forward_representation(data_batch_dev.x, data_batch_dev.edge_index)
            
            # Split batched embeddings and original coordinates (data.y) back into individual Data objects
            # batch.ptr is [0, N1, N1+N2, ...], counts are N1, N2, ...
            counts = data_batch.ptr[1:] - data_batch.ptr[:-1]
            num_graphs_in_batch = data_batch.num_graphs

            current_pos_emb = 0
            current_pos_y = 0
            for i in range(num_graphs_in_batch):
                num_nodes_this_graph = counts[i].item()
                
                emb_this_graph = latent_embeddings_batch[current_pos_emb : current_pos_emb + num_nodes_this_graph]
                coords_this_graph = data_batch.y[current_pos_y : current_pos_y + num_nodes_this_graph] # Original y from dataset
                edge_idx_this_graph = data_batch.edge_index # This needs careful handling if edge_indices are different per graph
                                                          # For kNN on full graphs, they are different.
                                                          # PyG DataLoader handles batching of edge_index.
                                                          # When splitting, we need to get the original edge_index for this graph.
                                                          # This is complex if not careful.
                                                          # Simpler: iterate original dataset if edge_index is critical for decoder directly.
                                                          # However, decoder2 here doesn't use edge_index.

                # Create new Data object for decoder: x=embedding, coords=original_coords_GT
                # We don't need edge_index for ProteinStateReconstructor2D's forward pass
                d = Data(x=emb_this_graph.cpu(), coords=coords_this_graph.cpu())
                decoder2_input_dataset.append(d)
                
                current_pos_emb += num_nodes_this_graph
                current_pos_y += num_nodes_this_graph


    if not decoder2_input_dataset:
        logger.error("Decoder2 input dataset is empty. Exiting.")
        sys.exit(1)

    train_data_d2, test_data_d2 = train_test_split(decoder2_input_dataset, test_size=0.1, random_state=42)
    train_loader_d2 = DataLoader(
        train_data_d2, batch_size=decoder2_bsize, shuffle=True, # Shuffle train
        num_workers=num_workers_config, pin_memory=pin_memory_config, drop_last=True
    )
    test_loader_d2 = DataLoader(
        test_data_d2, batch_size=decoder2_bsize, shuffle=False,
        num_workers=num_workers_config, pin_memory=pin_memory_config
    )

    # Prepare the "conditioner" data (single reference frame)
    X_ref_raw_coords = dataset[0].x.cpu() # Shape [num_unique_atoms, 3], on CPU
    conditioner_data_for_training = None # This will be [N_nodes, D_cond] on CPU

    if conditioner_mode == "X_ref":
        conditioner_data_for_training = X_ref_raw_coords
        logger.info(f"Using raw X_ref (shape {X_ref_raw_coords.shape}) as conditioner.")
        sys.stdout.flush()
    elif conditioner_mode == "z_ref":
        hno_model.eval() # Ensure HNO is in eval mode
        with torch.no_grad():
            # Compute z_ref using the first graph from the original dataset
            ref_graph_dev = dataset[0].to(device)
            X_ref_emb = hno_model.forward_representation(ref_graph_dev.x, ref_graph_dev.edge_index)
        conditioner_data_for_training = X_ref_emb.cpu() # Keep on CPU
        logger.info(f"Using z_ref (shape {conditioner_data_for_training.shape}) as conditioner.")
        sys.stdout.flush()
    else:
        logger.error(f"Invalid conditioner_mode: {conditioner_mode}. Choose 'X_ref' or 'z_ref'.")
        sys.exit(1)

    # num_nodes for the decoder is num_unique_atoms
    np.save(os.path.join(structures_dir, "X_ref_coords.npy"), X_ref_raw_coords.numpy())
    if conditioner_mode == "z_ref":
        np.save(os.path.join(latent_reps_dir, "z_ref_embedding.npy"), conditioner_data_for_training.numpy())


    d2_ckpt_path = os.path.join(checkpoint_dir, f"decoder2_{d2_pooling_type}.pth")
    if d2_pooling_type == "residue" and use_second_level_pooling:
         d2_ckpt_path = os.path.join(checkpoint_dir, f"decoder2_{d2_pooling_type}_2level.pth")


    logger.info(
        f"Creating ProteinStateReconstructor2D with pooling_type={d2_pooling_type}, "
        f"num_nodes={num_unique_atoms}, "
        f"conditioner_mode={conditioner_mode}, second_level_pooling={use_second_level_pooling}."
    )
    sys.stdout.flush()

    protein_state_model_2d = ProteinStateReconstructor2D(
        input_dim=hidden_dim, # HNO's output embedding dim
        num_nodes=num_unique_atoms,
        pooling_type=d2_pooling_type,
        residue_atom_indices=new_residue_indices if d2_pooling_type == "residue" else None,
        output_size_per_segment=(d2_output_height, d2_output_width),
        second_level_output_size=(d2_output_height2, d2_output_width2) if d2_output_height2 and d2_output_width2 else None,
        use_second_level_pooling=use_second_level_pooling,
        use_cross_attention=use_cross_attention, # Retained for signature
        cross_attention_type=cross_attention_type, # Retained for signature
        conditioner_mode=conditioner_mode,
        num_hidden_layers=d2_num_hidden_layers,
        mlp_hidden_dim=config["decoder2_settings"].get("mlp_hidden_dim", 128), # Get mlp hidden dim from config
        override_residues=override_residues_config,
        special_res_file=special_res_file_config
    )

    logger.info(
        f"Training Decoder2 => epochs={decoder2_epochs}, LR={decoder2_lr}, bsize={decoder2_bsize}"
    )
    sys.stdout.flush()
    protein_state_model_2d = train_protein_state_reconstructor( # Reassign
        protein_state_model_2d,
        train_loader_d2,
        test_loader_d2,
        num_epochs=decoder2_epochs,
        learning_rate=decoder2_lr,
        checkpoint_path=d2_ckpt_path,
        conditioner_data=conditioner_data_for_training, # Pass the CPU tensor
        save_interval=config.get("decoder2_save_interval", 10),
        weight_value=d2_weight_value,
    )

    if recon_output == 1: # Save Decoder2 outputs
        logger.info("Generating Decoder2 reconstructions on the full dataset (using optimized model)...")
        sys.stdout.flush()

        dec2_recon_path = os.path.join(
            structures_dir, f"decoder2_output_{d2_pooling_type}_{conditioner_mode}.h5"
        )
        protein_state_model_2d.eval() # Ensure model is in eval mode

        # Use the full decoder2_input_dataset for generating outputs
        num_samples_dec2_output = len(decoder2_input_dataset)
        dec2_output_loader = DataLoader(decoder2_input_dataset, batch_size=decoder2_bsize, shuffle=False,
                                        num_workers=num_workers_config, pin_memory=pin_memory_config)


        with h5py.File(dec2_recon_path, "w") as recon_h5:
            dset_with_override = recon_h5.create_dataset(
                "reconstructions_with_override", shape=(num_samples_dec2_output, num_unique_atoms, 3), dtype="float32"
            )
            dset_no_override = recon_h5.create_dataset(
                "reconstructions_no_override", shape=(num_samples_dec2_output, num_unique_atoms, 3), dtype="float32"
            )
            dset_ground_truth = recon_h5.create_dataset(
                "ground_truth_coords", shape=(num_samples_dec2_output, num_unique_atoms, 3), dtype="float32"
            )


            logger.debug(f"HDF5 dataset for Decoder2 outputs created.")
            sys.stdout.flush()

            current_idx_d2 = 0
            with torch.no_grad():
                for data_batch in dec2_output_loader: # data_batch.x are embeddings, data_batch.coords are GT
                    data_batch_dev = data_batch.to(device)
                    
                    # Call forward with override
                    X_pred_override = protein_state_model_2d(
                        data_batch_dev.x, data_batch_dev.batch, conditioner_data_for_training, use_override=True
                    )
                    # Call forward without override
                    X_pred_no_override = protein_state_model_2d(
                        data_batch_dev.x, data_batch_dev.batch, conditioner_data_for_training, use_override=False
                    )

                    batch_actual_size = X_pred_override.shape[0] // num_unique_atoms # Infer from output

                    start = current_idx_d2
                    end = current_idx_d2 + batch_actual_size
                    
                    dset_with_override[start:end] = X_pred_override.view(batch_actual_size, num_unique_atoms, 3).cpu().numpy()
                    dset_no_override[start:end] = X_pred_no_override.view(batch_actual_size, num_unique_atoms, 3).cpu().numpy()
                    dset_ground_truth[start:end] = data_batch.coords.view(batch_actual_size, num_unique_atoms, 3).cpu().numpy() # Save GT
                    
                    current_idx_d2 = end

                    if args.debug and current_idx_d2 % (10 * decoder2_bsize) < decoder2_bsize : # Log roughly every 10 batches
                        logger.debug(f"[Decoder2:Generation] Processed {current_idx_d2}/{num_samples_dec2_output} samples.")
                        sys.stdout.flush()
            logger.info(f"Decoder2 reconstructions saved to {dec2_recon_path}.")
            sys.stdout.flush()

    # Generate and Save Pooled Latent Embeddings from Decoder2 (if needed)
    # This uses the get_pooled_latent method which might show different performance characteristics
    # if it's not as optimized as the main forward pass, but it's for post-hoc analysis.
    if config.get("save_decoder2_pooled_latent", True): # Add a config toggle for this
        logger.info("Generating and saving Decoder2 pooled latent embeddings (full dataset)...")
        pooled_latent_path_d2 = os.path.join(latent_reps_dir, "decoder2_pooled_latent_embeddings.h5")
        all_pooled_latents_list = []
        protein_state_model_2d.eval()

        dec2_pooled_loader = DataLoader(decoder2_input_dataset, batch_size=decoder2_bsize, shuffle=False,
                                        num_workers=num_workers_config, pin_memory=pin_memory_config)
        with torch.no_grad():
            for data_batch in dec2_pooled_loader: # data_batch.x are HNO embeddings
                data_batch_dev_x = data_batch.x.to(device) # Only need x for get_pooled_latent
                
                # get_pooled_latent expects [B*N, D_in] and handles reshape internally
                # Its output is [B, NumEffectiveSegs, FinalEffectivePooledDim]
                pooled_lat_batch = protein_state_model_2d.get_pooled_latent(data_batch_dev_x, None)
                all_pooled_latents_list.append(pooled_lat_batch.cpu().numpy())
        
        if all_pooled_latents_list:
            all_pooled_cat = np.concatenate(all_pooled_latents_list, axis=0)
            with h5py.File(pooled_latent_path_d2, "w") as latent_h5_d2:
                latent_h5_d2.create_dataset("pooled_latent", data=all_pooled_cat)
            logger.info(f"Decoder2 pooled latent embeddings saved to {pooled_latent_path_d2} with shape {all_pooled_cat.shape}")
        else:
            logger.warning("No pooled latents generated for Decoder2.")
        sys.stdout.flush()


    logger.info("All tasks completed successfully!")
    sys.stdout.flush()
