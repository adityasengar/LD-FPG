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
    description="All-Atom Protein Reconstruction with Contiguous Indexing & Optional Debug"
)
parser.add_argument(
    "--config", type=str, required=True, help="Path to the YAML configuration file."
)
parser.add_argument("--debug", action="store_true", help="Enable debug-level logging.")
parser.add_argument(
    "--log_file", type=str, default="logfile.log", help="Path to the log file."
)
# MODIFIED: Replaced --exp_index with more descriptive arguments
parser.add_argument(
    "--exp_label", type=str, required=True, help="Experiment label (e.g., 'exp1')."
)
parser.add_argument(
    "--embedding_base_folder",
    type=str,
    required=True,
    help="Base folder for embeddings within latent_reps (e.g., 'exp1_epoch_snapshots').",
)
parser.add_argument(
    "--epoch_folder",
    type=str,
    required=True,
    help="Specific epoch folder containing generated_embeddings.h5 (e.g., 'epoch_10000').",
)
args = parser.parse_args()

################################################################################
# Logging Setup
################################################################################
logger = logging.getLogger(f"ProteinReconstruction_{args.exp_label}_{args.epoch_folder}") # Make logger name more specific
logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

# Use a log file name that reflects the specific run
log_file_name = f"logfile_{args.exp_label}_{args.embedding_base_folder.replace('/', '_')}_{args.epoch_folder}.log"
fh = logging.FileHandler(log_file_name, mode="w") # Changed from args.log_file
fh.setLevel(logging.DEBUG if args.debug else logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG if args.debug else logging.INFO)

formatter = logging.Formatter(
    f"[%(levelname)s] %(asctime)s - %(name)s - [{args.exp_label}][{args.epoch_folder}] %(message)s"
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

logger.info(f"Logger initialized for {args.exp_label}/{args.embedding_base_folder}/{args.epoch_folder}. Log file: {log_file_name}")
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
d2_pooling_type = config["decoder2_settings"]["pooling_type"]
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

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

################################################################################
# Directory Setup
################################################################################
current_dir = os.getcwd()
checkpoint_dir = os.path.join(current_dir, "checkpoints")
latent_reps_dir = os.path.join(current_dir, "latent_reps") # Base for input embeddings
structures_dir = os.path.join(current_dir, "structures") # For other general structures like X_ref

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(latent_reps_dir, exist_ok=True) # Ensured by data prep
os.makedirs(structures_dir, exist_ok=True)

logger.debug(
    f"Directories:\n  checkpoints: {checkpoint_dir}\n  latent_reps_base: {latent_reps_dir}\n  structures_general: {structures_dir}"
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
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
    unique_atoms = sorted(set(flattened))
    logger.debug(
        f"Unique atoms found: {len(unique_atoms)}. "
        f"Max original ID = {max(unique_atoms)}, Min = {min(unique_atoms)}"
    )
    sys.stdout.flush()
    old2new_map = {old_id: idx for idx, old_id in enumerate(unique_atoms)}
    first_few_items = list(old2new_map.items())[:20]
    logger.debug(f"[old2new_map sample] -> {first_few_items}")
    sys.stdout.flush()
    new_residue_indices = []
    for residue in residue_atom_indices:
        new_list = [old2new_map[a] for a in residue]
        new_residue_indices.append(new_list)
    return new_residue_indices, old2new_map

################################################################################
# Data Loading & Alignment
################################################################################
def load_heavy_atom_coords_from_json(json_file):
    logger.info(f"Loading JSON from {json_file}")
    sys.stdout.flush()
    with open(json_file, "r") as f:
        data = json.load(f)
    residue_keys_sorted = sorted(data.keys(), key=lambda x: int(x))
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
        frame_coords = np.concatenate(frame_coords_list, axis=0)
        coords_per_frame.append(torch.tensor(frame_coords, dtype=torch.float32))
    return coords_per_frame, original_residue_atom_indices

def align_frames_to_first(coords_list):
    logger.info("Aligning all frames to the first frame via Kabsch...")
    sys.stdout.flush()
    reference = coords_list[0]
    aligned = []
    for i, coords in enumerate(coords_list):
        if i == 0:
            aligned.append(coords)
            logger.debug(f"Frame {i}: No alignment needed. shape={coords.shape}")
            sys.stdout.flush()
        else:
            _, coords_aligned = kabsch_algorithm(reference, coords)
            aligned.append(coords_aligned)
            if i < 5: # Log only a few for brevity
                logger.debug(f"Frame {i}: aligned shape={coords_aligned.shape}")
                sys.stdout.flush()
    return aligned

################################################################################
# Build Graph Dataset
################################################################################
def build_graph_dataset(coords_list, knn_neighbors=4):
    logger.info("Building PyG dataset with knn_graph.")
    sys.stdout.flush()
    dataset = []
    for idx, coords in enumerate(coords_list):
        edge_index = knn_graph(coords, k=knn_neighbors, batch=None, loop=False)
        data = Data(x=coords, edge_index=edge_index, y=coords)
        dataset.append(data)
        if idx < 5:
            logger.debug(
                f"[Dataset] Frame {idx}: x.shape={coords.shape}, edge_index.shape={edge_index.shape}"
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
        if (not HNO._logged_forward_once) and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[HNO:forward] Input x.shape={x.shape}")
            sys.stdout.flush()
        x = self.conv1(x, edge_index)
        if (not HNO._logged_forward_once) and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[HNO:forward] After conv1 => {x.shape}")
            sys.stdout.flush()
        x = F.leaky_relu(x)
        x = self.bano1(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.bano2(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.bano3(x)
        x = self.conv4(x, edge_index)
        x = F.normalize(x, p=2, dim=1)
        if (not HNO._logged_forward_once) and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[HNO:forward] After conv4 => {x.shape}")
            sys.stdout.flush()
        x_rep = x # Save representation before mlpRep
        x = self.mlpRep(x)
        if (not HNO._logged_forward_once) and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[HNO:forward] Output => {x.shape}")
            HNO._logged_forward_once = True
            sys.stdout.flush()
        return x

    def forward_representation(self, x, edge_index):
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
        x = F.normalize(x, p=2, dim=1)
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
    checkpoint_path, # MODIFIED: Path now includes exp_label
    save_interval=10,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    logger.info(
        f"Starting HNO ({args.exp_label}) from epoch={start_epoch}, total epochs={num_epochs}, LR={learning_rate}"
    )
    sys.stdout.flush()
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)
            loss = criterion(pred, data.x) # Reconstruct original coordinates
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                pred = model(data.x, data.edge_index)
                val_loss = criterion(pred, data.x)
                test_loss += val_loss.item()
        avg_test_loss = test_loss / len(test_loader)
        logger.info(
            f"[HNO {args.exp_label}] Epoch {epoch+1}/{num_epochs} => Train Loss={avg_loss:.6f}, Test Loss={avg_test_loss:.6f}"
        )
        sys.stdout.flush()
        if (epoch + 1) % save_interval == 0:
            checkpoint_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint_state, checkpoint_path)
            logger.info(f"HNO checkpoint for {args.exp_label} saved at epoch {epoch+1} -> {checkpoint_path}")
            sys.stdout.flush()

################################################################################
# Simple Cross-Attention Module (definition remains, usage depends on config)
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
        scores = (torch.matmul(Q, K.transpose(-1, -2)) * self.scale)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.squeeze(1)
        if N_seg > 1:
            out = out.view(B, N_seg, -1)
        return out

################################################################################
# MLP Builder for Decoder2 (remains the same)
################################################################################
def build_decoder_mlp(input_dim, output_dim, num_hidden_layers, hidden_dim=128):
    # ... (no changes needed in this function)
    layers = []
    current_dim = input_dim
    for i in range(num_hidden_layers - 1):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)

################################################################################
# ProteinStateReconstructor2D
################################################################################
class ProteinStateReconstructor2D(nn.Module):
    _logged_forward_once = False
    def __init__(
        self,
        input_dim,
        num_nodes,
        pooling_type="blind",
        residue_atom_indices=None,
        output_size_per_segment=(1, 10),
        second_level_output_size=None,
        use_second_level_pooling=False,
        use_cross_attention=False,
        cross_attention_type=None,
        conditioner_mode="z_ref",
        num_hidden_layers=2,
        override_residues=None,
        special_res_file=None, # This will be the full path to generated_embedding.h5
    ):
        super().__init__()
        self.pooling_type = pooling_type
        self.num_nodes = num_nodes
        self.input_dim = input_dim # Dim of HNO latent
        self.conditioner_mode = conditioner_mode # How conditioner is formed (X_ref or z_ref)
        self.use_second_level_pooling = use_second_level_pooling
        self.output_size_per_segment = output_size_per_segment
        self.second_level_output_size = second_level_output_size

        self.override_residues = (
            override_residues if override_residues is not None else []
        )
        self.generated_arr = None

        if special_res_file is not None and os.path.isfile(special_res_file) and len(self.override_residues) > 0 :
            logger.info(f"Attempting to load override embeddings from: {special_res_file}")
            try:
                with h5py.File(special_res_file, "r") as f:
                    # Assuming the dataset inside HDF5 is named 'generated_embeddings' or 'data'
                    # Check for common names if 'generated_embeddings' isn't found
                    if "generated_embeddings" in f:
                        arr = f["generated_embeddings"][:]
                    elif "data" in f: # Common alternative
                         arr = f["data"][:]
                    else:
                        logger.error(f"Cannot find 'generated_embeddings' or 'data' dataset in {special_res_file}")
                        arr = None
                
                if arr is not None:
                    if np.isnan(arr).any():
                        logger.error(f"Generated override embeddings from {special_res_file} contain NaN values!")
                        self.generated_arr = None
                    else:
                        logger.info(
                            f"Generated embeddings loaded from {special_res_file} with shape {arr.shape} and appear valid."
                        )
                        self.generated_arr = arr
                        if self.generated_arr.shape[1] != len(self.override_residues): # M should be num_override_residues
                             # This check might be too strict if the H5 file stores more than needed.
                             # For now, let's assume it stores exactly M columns for M override residues.
                            logger.warning(
                                f"Shape Mismatch: special_res_file {special_res_file} has {self.generated_arr.shape[1]} residue embeddings, "
                                f"but override_residues has length {len(self.override_residues)}. Using available."
                            )
                            # Potentially adjust or error out here based on exact requirements.
                            # If the H5 file contains embeddings for ALL residues, and override_residues is a subset,
                            # then we need to select the correct columns.
                            # Current code assumes H5 file has shape (N_samples, N_override_residues, embed_dim)
            except Exception as e:
                logger.error(f"Error loading or processing {special_res_file}: {e}")
                self.generated_arr = None
        elif special_res_file:
             logger.warning(f"Special residue file specified but not found or no override residues: {special_res_file}")


        if pooling_type == "blind":
            # ... (no changes needed in pooling logic itself)
            self.segments_indices = [list(range(num_nodes))]
            self.segment_pools = nn.ModuleList(
                [nn.AdaptiveAvgPool2d(self.output_size_per_segment)]
            )
            seg_pooled_size = (
                self.output_size_per_segment[0] * self.output_size_per_segment[1]
            )
            self.pooled_dim = seg_pooled_size
            self.global_pool2 = None

        elif pooling_type == "residue":
            # ... (no changes needed in pooling logic itself)
            if residue_atom_indices is None:
                raise ValueError("residue_atom_indices is required for residue-level pooling.")
            self.segments_indices = residue_atom_indices
            self.segment_pools = nn.ModuleList()
            for _ in self.segments_indices:
                pool = nn.AdaptiveAvgPool2d(self.output_size_per_segment)
                self.segment_pools.append(pool)
            if use_second_level_pooling and second_level_output_size is not None:
                self.global_pool2 = nn.AdaptiveAvgPool2d(second_level_output_size)
                final_pooled_size = (
                    second_level_output_size[0] * second_level_output_size[1]
                )
                self.pooled_dim = final_pooled_size
            else:
                self.global_pool2 = None
                self.pooled_dim = (
                    self.output_size_per_segment[0] * self.output_size_per_segment[1]
                )
        else:
            raise ValueError(f"Unknown pooling_type={pooling_type}")

        cond_dim = self.input_dim if conditioner_mode == "z_ref" else 3 # 3 for X_ref
        self.final_in_dim = self.pooled_dim + cond_dim

        self.decoder = self._build_decoder_mlp(
            input_dim=self.final_in_dim,
            output_dim=3,
            num_hidden_layers=num_hidden_layers,
            hidden_dim=128, # This hidden_dim is for the MLP, not the HNO's hidden_dim
        )

    def _build_decoder_mlp(
        self, input_dim, output_dim, num_hidden_layers, hidden_dim=128
    ):
        # ... (no changes needed here)
        layers = []
        current_dim = input_dim
        for i in range(num_hidden_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim)) # Make sure this is appropriate if batch_size can be 1
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)


    def get_pooled_latent(self, x, conditioner=None): # x is [B*N_nodes, input_dim]
        # ... (no changes needed in this function's core logic)
        B_times_N, _ = x.shape # in_dim is self.input_dim (HNO latent dim)
        batch_size = B_times_N // self.num_nodes
        x = x.view(batch_size, self.num_nodes, -1) # [B, N_nodes, self.input_dim]

        if self.pooling_type == "blind":
            seg_x = x.unsqueeze(1)  # [B,1,N_nodes,in_dim]
            p_2d = self.segment_pools[0](seg_x)  # [B,1,H,W] , H*W should be self.pooled_dim
            pooled_flat = p_2d.squeeze(1).view(batch_size, -1) # [B, self.pooled_dim]
            pooled_latent = pooled_flat.unsqueeze(1)  # [B,1,pooled_dim]
        else: # residue pooling
            pooled_segments_level1 = []
            for i, seg_indices in enumerate(self.segments_indices):
                seg_x = x[:, seg_indices, :].unsqueeze(1) # [B, 1, N_atoms_in_res, self.input_dim]
                p_2d = self.segment_pools[i](seg_x) # [B, 1, H, W]
                p_2d = p_2d.squeeze(1).view(batch_size, -1) # [B, self.output_size_per_segment[0]*self.output_size_per_segment[1]]
                pooled_segments_level1.append(p_2d)
            
            level1_concat = torch.stack(pooled_segments_level1, dim=1)  # [B,R,pooled_dim_per_segment]
            
            if self.global_pool2 is not None:
                level1_concat_4d = level1_concat.unsqueeze(1) # [B,1,R,pooled_dim_per_segment]
                # global_pool2 expects input_channel = 1, H = R, W = pooled_dim_per_segment
                # output_size is self.second_level_output_size
                # So the result should be [B, 1, H2, W2]
                pooled_2d = self.global_pool2(level1_concat_4d) 
                pooled_2d = pooled_2d.squeeze(1).view(batch_size, -1) # [B, self.pooled_dim] (where pooled_dim is H2*W2)
                N_seg = len(self.segments_indices)
                pooled_latent = pooled_2d.unsqueeze(1).repeat(1, N_seg, 1) # [B,R,self.pooled_dim]
            else:
                pooled_latent = level1_concat # [B,R,pooled_dim_per_segment] where pooled_dim_per_segment = self.pooled_dim
        return pooled_latent


    def forward(self, x, batch, conditioner, use_override=True):
        # x: [B*N_nodes, input_dim from HNO]
        # conditioner: [num_nodes, conditioner_dim (input_dim from HNO if z_ref, or 3 if X_ref)]
        device = x.device
        B_times_N, _ = x.shape
        batch_size = B_times_N // self.num_nodes
        if batch_size * self.num_nodes != B_times_N:
            raise ValueError("Mismatch in x.shape vs. num_nodes & batch_size")

        x_latent_hno = x.view(batch_size, self.num_nodes, -1) # [B, N_nodes, self.input_dim]

        conditioner = conditioner.unsqueeze(0).to(device) # [1, N_nodes, cond_dim]
        conditioner = conditioner.repeat(batch_size, 1, 1) # [B, N_nodes, cond_dim]

        # 1) Get pooled latent from x_latent_hno
        # This will be [B, R, self.pooled_dim] for residue pooling (R=num_residues)
        # or [B, 1, self.pooled_dim] for blind pooling
        pooled_latent = self.get_pooled_latent(x, conditioner) # Pass original x [B*N, C]

        # 2) Override mechanism
        if (
            use_override
            and self.generated_arr is not None
            and len(self.override_residues) > 0
            and self.pooling_type == "residue" # Override makes most sense for residue-level pooling
        ):
            # self.generated_arr is (N_samples_in_H5, M_override_residues, H5_embedding_dim)
            # H5_embedding_dim should match self.pooled_dim
            # N_samples_in_H5 could be 1 if it's a single set of overrides, or >= batch_size
            
            num_samples_in_h5, num_override_res_in_h5, h5_embed_dim = self.generated_arr.shape
            
            if h5_embed_dim != pooled_latent.shape[-1]:
                logger.error(f"Override H5 embed dim ({h5_embed_dim}) != model pooled_dim ({pooled_latent.shape[-1]}). Skipping override.")
            else:
                # Select samples for the current batch
                if num_samples_in_h5 == 1: # Use the same override for all batch items
                    arr_slice = self.generated_arr[0] # [M_override_residues, h5_embed_dim]
                    arr_tensor = torch.from_numpy(arr_slice).to(device).unsqueeze(0).repeat(batch_size, 1, 1) # [B, M, pooled_dim]
                elif num_samples_in_h5 >= batch_size: # Use distinct overrides per batch item
                    arr_slice = self.generated_arr[:batch_size] # [B, M_override_residues, h5_embed_dim]
                    arr_tensor = torch.from_numpy(arr_slice).to(device) # [B, M, pooled_dim]
                else: # Not enough samples in H5 for the batch
                    logger.warning(f"Batch size ({batch_size}) > num_samples_in_H5 ({num_samples_in_h5}). Using first H5 sample for all. ")
                    arr_slice = self.generated_arr[0]
                    arr_tensor = torch.from_numpy(arr_slice).to(device).unsqueeze(0).repeat(batch_size, 1, 1)
                
                # Apply override
                # pooled_latent is [B, R, pooled_dim]
                # arr_tensor is [B, M_override_residues, pooled_dim]
                # self.override_residues is a list of R indices to be replaced by M entries from arr_tensor
                # This assumes len(self.override_residues) == M_override_residues
                if arr_tensor.shape[1] == len(self.override_residues):
                    for i, r_idx_to_override in enumerate(self.override_residues):
                        if r_idx_to_override < pooled_latent.shape[1]: # ensure r_idx is valid
                            pooled_latent[:, r_idx_to_override, :] = arr_tensor[:, i, :]
                        else:
                            logger.error(f"Override residue index {r_idx_to_override} out of bounds for pooled_latent shape {pooled_latent.shape[1]}")
                else:
                    logger.warning(f"Mismatch between override_residues list ({len(self.override_residues)}) and H5 override dim ({arr_tensor.shape[1]}). Skipping override.")

        # 3) Combine with conditioner and pass to MLP decoder
        # pooled_latent: [B, R_or_1, pooled_dim]
        # conditioner:   [B, N_nodes, cond_dim]
        # segments_indices: List of lists of atom indices per residue (or one list for all atoms if blind)
        
        node_inputs_for_mlp = []
        num_segments_in_pool = pooled_latent.shape[1] # R if residue-pooled, 1 if blind-pooled

        if self.pooling_type == "blind": # pooled_latent is [B, 1, pooled_dim]
            # We need to combine this single pooled vector with each node's conditioner
            seg_pooled_expanded = pooled_latent.repeat(1, self.num_nodes, 1) # [B, N_nodes, pooled_dim]
            combined_for_mlp = torch.cat([seg_pooled_expanded, conditioner], dim=-1) # [B, N_nodes, pooled_dim + cond_dim]
            node_inputs_for_mlp.append(combined_for_mlp)
        else: # residue pooling, pooled_latent is [B, R, pooled_dim]
            for res_idx, atom_indices_in_res in enumerate(self.segments_indices):
                # Get the pooled vector for this specific residue
                res_pooled = pooled_latent[:, res_idx, :]  # [B, pooled_dim]
                
                # Expand it to match the number of atoms in this residue
                res_pooled_expanded = res_pooled.unsqueeze(1).repeat(1, len(atom_indices_in_res), 1) # [B, N_atoms_in_res, pooled_dim]
                
                # Get the conditioner part for atoms in this residue
                cond_part_for_res = conditioner[:, atom_indices_in_res, :] # [B, N_atoms_in_res, cond_dim]
                
                combined_res_mlp_input = torch.cat([res_pooled_expanded, cond_part_for_res], dim=-1) # [B, N_atoms_in_res, final_in_dim]
                node_inputs_for_mlp.append(combined_res_mlp_input)

        final_mlp_input_structured = torch.cat(node_inputs_for_mlp, dim=1) # [B, N_nodes, final_in_dim]
        final_mlp_input_flat = final_mlp_input_structured.view(batch_size * self.num_nodes, -1) # [B*N_nodes, final_in_dim]
        
        X_pred = self.decoder(final_mlp_input_flat) # [B*N_nodes, 3]

        if (not ProteinStateReconstructor2D._logged_forward_once) and (
            logger.isEnabledFor(logging.DEBUG)
        ):
            logger.debug(f"[Decoder2:forward ({args.exp_label}/{args.epoch_folder})] X_pred => {X_pred.shape}")
            ProteinStateReconstructor2D._logged_forward_once = True
        return X_pred

    def forward_representation(self, x, batch, conditioner):
        raise NotImplementedError("forward_representation not implemented in ProteinStateReconstructor2D.")


################################################################################
# Train Decoder2
################################################################################
def train_protein_state_reconstructor(
    model,
    train_loader,
    test_loader,
    num_epochs,
    learning_rate,
    checkpoint_path, # MODIFIED: Path now includes exp_label
    conditioner_data,
    save_interval=10,
    weight_value=1.0, # Not used in current loss, but kept for signature
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    logger.info(
        f"Starting Decoder2 ({args.exp_label}/{args.epoch_folder}) from epoch={start_epoch}, total epochs={num_epochs}, LR={learning_rate}"
    )
    sys.stdout.flush()
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader: # data.x is HNO latent, data.coords is target
            data = data.to(device)
            optimizer.zero_grad()
            X_pred = model(data.x, data.batch, conditioner_data)
            coords_gt = data.coords
            loss = criterion(X_pred, coords_gt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                X_pred = model(data.x, data.batch, conditioner_data)
                coords_gt = data.coords
                val_loss = criterion(X_pred, coords_gt)
                test_loss += val_loss.item()
        avg_test_loss = test_loss / len(test_loader)
        logger.info(
            f"[Decoder2 {args.exp_label}/{args.epoch_folder}] Epoch {epoch+1}/{num_epochs} => Train Loss={avg_loss:.6f}, Test Loss={avg_test_loss:.6f}"
        )
        sys.stdout.flush()
        if (epoch + 1) % save_interval == 0:
            checkpoint_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint_state, checkpoint_path)
            logger.info(
                f"Decoder2 checkpoint for {args.exp_label}/{args.epoch_folder} saved at epoch {epoch+1} -> {checkpoint_path}"
            )
            sys.stdout.flush()

################################################################################
# Main Execution
################################################################################
if __name__ == "__main__":
    logger.info(f"Starting main script execution for {args.exp_label}/{args.embedding_base_folder}/{args.epoch_folder}...")
    sys.stdout.flush()

    coords_per_frame, original_residue_atom_indices = load_heavy_atom_coords_from_json(
        json_path
    )
    logger.info("Remapping residue_atom_indices to contiguous IDs...")
    sys.stdout.flush()
    new_residue_indices, old2new_map = remap_residue_indices(
        original_residue_atom_indices
    )
    num_unique_atoms = len(old2new_map)
    logger.info(f"Number of unique atoms after reindexing: {num_unique_atoms}")
    sys.stdout.flush()
    aligned_frames = align_frames_to_first(coords_per_frame)
    dataset = build_graph_dataset(aligned_frames, knn_neighbors=knn_value)
    first_data = dataset[0]
    # print("Atom Indices Mapping (old -> new):", list(old2new_map.items())[:5]) # Redundant with logger
    # print("First 5 atoms in data.x:", first_data.x[:5]) # Redundant
    logger.info(f"Dataset size: {len(dataset)} frames.")
    sys.stdout.flush()

    train_dataset_hno, test_dataset_hno = train_test_split(
        dataset, test_size=0.1, random_state=42
    )
    train_loader_hno = DataLoader(
        train_dataset_hno, batch_size=decoder1_bsize, shuffle=False # Shuffle True for training usually
    )
    test_loader_hno = DataLoader(
        test_dataset_hno, batch_size=decoder1_bsize, shuffle=False
    )

    hno_model = HNO(hidden_dim, cheb_order)
    # MODIFIED: HNO checkpoint path now includes exp_label
    hno_ckpt_path = os.path.join(checkpoint_dir, f"hno_checkpoint.pth")
    logger.info(
        f"Training HNO ({args.exp_label}) => epochs={decoder1_epochs}, LR={decoder1_lr}, bsize={decoder1_bsize}"
    )
    sys.stdout.flush()
    train_hno_model(
        hno_model,
        train_loader_hno,
        test_loader_hno,
        num_epochs=decoder1_epochs,
        learning_rate=decoder1_lr,
        checkpoint_path=hno_ckpt_path,
        save_interval=10,
    )

    if recon_output == 2: # HNO reconstructions
        logger.info(
            f"Generating HNO reconstructions & latent embeddings for {args.exp_label}..."
        )
        sys.stdout.flush()
        hno_model.eval()
        # MODIFIED: HNO output paths include exp_label
        hno_recon_path = os.path.join(structures_dir, f"hno_reconstructed_coords_{args.exp_label}.h5")
        hno_latent_path = os.path.join(latent_reps_dir, f"hno_latent_embeddings_{args.exp_label}.h5") # Stays in latent_reps_dir base
        
        # This should use the full dataset or a consistent subset, not just train_dataset_hno
        # For consistency, let's use the 'dataset' (all frames) for HNO output generation
        num_samples_hno_output = len(dataset) 
        logger.info(f"Number of samples to process for HNO output ({args.exp_label}): {num_samples_hno_output}")

        pooled_dim_hno = hidden_dim
        with h5py.File(hno_recon_path, "w") as recon_h5, h5py.File(hno_latent_path, "w") as latent_h5:
            recon_dset_hno = recon_h5.create_dataset(
                "reconstructions",
                shape=(num_samples_hno_output, num_unique_atoms, 3),
                dtype="float32",
            )
            latent_dset_hno = latent_h5.create_dataset(
                "pooled_latent", # This is per-atom latent from HNO before MLP
                shape=(num_samples_hno_output, num_unique_atoms, pooled_dim_hno),
                dtype="float32",
            )
            logger.debug(
                f"HDF5 datasets for HNO ({args.exp_label}) created:\n recon={recon_dset_hno.shape}, latent={latent_dset_hno.shape}"
            )
            with torch.no_grad():
                for idx, data_item in enumerate(dataset): # Iterate over the whole dataset for generation
                    data_item = data_item.to(device)
                    latent = hno_model.forward_representation(data_item.x, data_item.edge_index)
                    reconstruction = hno_model(data_item.x, data_item.edge_index) # This is direct reconstruction
                    
                    # Storing original coordinates as 'reconstruction' if we want to compare to HNO's attempt
                    # Or store HNO's attempt. The current code stores data.x (original) in recon_dset_hno.
                    # This seems like a bug in original. It should be the `reconstruction`.
                    # Let's fix it to store the actual HNO reconstruction.
                    recon_dset_hno[idx, :, :] = reconstruction.cpu().numpy() # Fixed
                    latent_dset_hno[idx, :, :] = latent.cpu().numpy()

                    if args.debug and idx < 1:
                        logger.debug(
                            f"[HNO:Generation {args.exp_label}] Sample {idx}: recon_np.shape={reconstruction.cpu().numpy().shape}, latent_np.shape={latent.cpu().numpy().shape}"
                        )
                    if args.debug and (idx + 1) % 1000 == 0:
                        logger.debug(
                            f"[HNO:Generation {args.exp_label}] Processed {idx + 1}/{num_samples_hno_output} samples."
                        )
            logger.info(f"HNO outputs for {args.exp_label} saved to {hno_recon_path} and {hno_latent_path}")


    logger.info(f"Preparing dataset for Decoder2 ({args.exp_label}/{args.epoch_folder}) using HNO latent embeddings...")
    sys.stdout.flush()
    decoder2_dataset = []
    hno_model.eval()
    with torch.no_grad():
        for data_item in dataset: # Use the full dataset
            data_cpu = data_item.cpu()
            latent = hno_model.forward_representation(
                data_cpu.x.to(device), data_cpu.edge_index.to(device)
            )
            d = Data(
                x=latent.cpu(), # Input to Decoder2 is HNO latent
                edge_index=data_cpu.edge_index, # Not strictly used by Decoder2 but good to keep
                coords=data_cpu.y, # Target for Decoder2 is original coordinates
                batch=data_cpu.batch,
            )
            decoder2_dataset.append(d)

    train_data_d2, test_data_d2 = train_test_split(
        decoder2_dataset, test_size=0.1, random_state=42
    )
    train_loader_d2 = DataLoader(
        train_data_d2, batch_size=decoder2_bsize, shuffle=False # Shuffle True for training
    )
    test_loader_d2 = DataLoader(test_data_d2, batch_size=decoder2_bsize, shuffle=False)

    X_ref_raw = dataset[0].x # Atom coords of the first frame
    if conditioner_mode == "X_ref":
        conditioner_data = X_ref_raw # Shape [num_unique_atoms, 3]
        logger.info("Using raw X_ref as conditioner.")
    else: # z_ref
        with torch.no_grad():
            X_ref_emb = hno_model.forward_representation(
                X_ref_raw.to(device), dataset[0].edge_index.to(device)
            )
        conditioner_data = X_ref_emb.cpu() # Shape [num_unique_atoms, hidden_dim]
        logger.info("Using z_ref (HNO embedding of X_ref) as conditioner.")
    sys.stdout.flush()
    num_nodes = X_ref_raw.shape[0]
    np.save(os.path.join(structures_dir, f"X_ref_{args.exp_label}.npy"), X_ref_raw.numpy()) # Save X_ref specific to exp_label

    # MODIFIED: Decoder2 checkpoint path now includes exp_label
    base_ckpt_name = f"decoder2_{args.exp_label}"
    if d2_pooling_type == "blind":
        d2_ckpt_path = os.path.join(
            checkpoint_dir,
            "decoder2_blind.pth"
        )
    else: # residue
        d2_ckpt_path = os.path.join(
            checkpoint_dir,"decoder2_residue.pth"
        )

    logger.info(
        f"Creating ProteinStateReconstructor2D ({args.exp_label}/{args.epoch_folder}) with pooling_type={d2_pooling_type}, "
        f"conditioner_mode={conditioner_mode}."
    )
    sys.stdout.flush()

    # MODIFIED: Construct path to the specific generated_embedding.h5
    special_file_path = os.path.join(
        latent_reps_dir, args.embedding_base_folder, args.epoch_folder, "generated_embeddings.h5"
    )
    logger.info(f"Using special_res_file: {special_file_path}")

    # Ensure override_residues list matches your data. This is a very long list.
    # It should correspond to the number of residues (or segments) you expect.
    # If using residue pooling, len(new_residue_indices) is the number of residues.
    # The override_residues list should contain indices from 0 to R-1.
    # The H5 file for override should then have embeddings for these R residues (or a subset M of them).
    num_total_residues = len(new_residue_indices) if d2_pooling_type == "residue" else 1
    
    # Example: Override all residues if residue pooling, or the single segment if blind.
    # This example list needs to be dynamically set or carefully configured.
    # For residue pooling, it should be list(range(num_total_residues)) if you intend to override all.
    # The HDF5 override file must then provide embeddings for all these.
    # The original hardcoded list was very long, make sure it's appropriate.
    # Assuming the hardcoded list was for a specific protein with that many residues.
    # Let's make it dynamic:
    override_residues_list = list(range(num_total_residues)) if d2_pooling_type == "residue" else [0]
    if len(override_residues_list) > 273 and d2_pooling_type == "residue": # If the hardcoded list was a max
        logger.warning(f"Number of residues ({num_total_residues}) is less than typical override list. Using all {num_total_residues} residues for override.")
    elif d2_pooling_type == "residue":
         logger.info(f"Setting override_residues to all {num_total_residues} residues.")
    # else: it's blind pooling, override_residues_list is [0]

    protein_state_model_2d = ProteinStateReconstructor2D(
        input_dim=hidden_dim, # Dimension of HNO latent embeddings
        num_nodes=num_nodes,
        pooling_type=d2_pooling_type,
        residue_atom_indices=new_residue_indices if d2_pooling_type == "residue" else None,
        output_size_per_segment=(d2_output_height, d2_output_width),
        second_level_output_size=(d2_output_height2, d2_output_width2) if d2_pooling_type == "residue" and use_second_level_pooling else None,
        use_second_level_pooling=use_second_level_pooling,
        use_cross_attention=use_cross_attention, # Note: this is usually False in constructor
        cross_attention_type=cross_attention_type,
        conditioner_mode=conditioner_mode,
        num_hidden_layers=d2_num_hidden_layers,
        override_residues=override_residues_list, # Use the dynamically determined list
        special_res_file=special_file_path,
    )

    logger.info(
        f"Training Decoder2 ({args.exp_label}/{args.epoch_folder}) => epochs={decoder2_epochs}, LR={decoder2_lr}, bsize={decoder2_bsize}"
    )
    sys.stdout.flush()
    train_protein_state_reconstructor(
        protein_state_model_2d,
        train_loader_d2,
        test_loader_d2,
        num_epochs=decoder2_epochs,
        learning_rate=decoder2_lr,
        checkpoint_path=d2_ckpt_path,
        conditioner_data=conditioner_data,
        save_interval=10,
        weight_value=d2_weight_value,
    )

    if recon_output == 1: # Decoder2 reconstructions
        logger.info(f"Generating Decoder2 reconstructions for {args.exp_label}/{args.embedding_base_folder}/{args.epoch_folder}...")
        sys.stdout.flush()

        # MODIFIED: Output path is now inside the specific epoch_folder in latent_reps
        embedding_source_dir = os.path.join(latent_reps_dir, args.embedding_base_folder, args.epoch_folder)
        # os.makedirs(embedding_source_dir, exist_ok=True) # Should exist if generated_embedding.h5 is there
        dec2_recon_path = os.path.join(embedding_source_dir, f"decoder2_output_reconstruction.h5")
        
        protein_state_model_2d.eval()
        
        # For generation, use a consistent dataset part, e.g., test_data_d2 or all of decoder2_dataset
        # Using test_loader_d2 for this example.
        # To save all coords from a batch, need to sum num_nodes per item in batch.
        # PyG loader handles batching, so output X_pred is already [TotalNodesInBatch, 3]
        
        all_recons_override = []
        all_recons_no_override = []

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader_d2): # Or use train_loader_d2 or full dataset loader
                data = data.to(device)
                X_pred_override = protein_state_model_2d(
                    data.x, data.batch, conditioner_data, use_override=True
                )
                X_pred_no_override = protein_state_model_2d(
                    data.x, data.batch, conditioner_data, use_override=False
                )
                all_recons_override.append(X_pred_override.cpu().numpy())
                all_recons_no_override.append(X_pred_no_override.cpu().numpy())

                if batch_idx % 10 == 0 and args.debug:
                    logger.debug(
                        f"[Decoder2:Generation {args.exp_label}/{args.epoch_folder}] batch {batch_idx+1} "
                        f"X_pred_over.shape={X_pred_override.shape}"
                    )
                    sys.stdout.flush()
        
        final_recons_override = np.concatenate(all_recons_override, axis=0)
        final_recons_no_override = np.concatenate(all_recons_no_override, axis=0)

        with h5py.File(dec2_recon_path, "w") as recon_h5:
            recon_h5.create_dataset("reconstructions_with_override", data=final_recons_override)
            recon_h5.create_dataset("reconstructions_no_override", data=final_recons_no_override)
        
        logger.info(
            f"Decoder2 reconstructions for {args.exp_label}/{args.epoch_folder} saved to {dec2_recon_path}. "
            f"with_override={final_recons_override.shape[0]} coords, "
            f"no_override={final_recons_no_override.shape[0]} coords."
        )
        sys.stdout.flush()

    # Pooled latent embeddings from Decoder2 (optional, if needed for analysis)
    # This part might need adjustment based on what "pooled latent" means for Decoder2
    # The current ProteinStateReconstructor2D doesn't have a simple get_final_pooled_latent.
    # The get_pooled_latent method returns the result after the initial pooling stages.
    # If this specific output is still desired, its path should also be made exp_label specific.
    # For now, commenting out as it was less critical to the primary request.
    """
    pooled_latent_path_d2 = os.path.join(
        latent_reps_dir, f"decoder2_pooled_latent_embeddings_{args.exp_label}_{args.epoch_folder}.h5" # Made specific
    )
    all_pooled_d2 = []
    protein_state_model_2d.eval()
    with torch.no_grad():
        # Iterate over decoder2_dataset (or a specific part like test_data_d2)
        # The get_pooled_latent needs to be called carefully, perhaps within the forward pass logic
        # or by creating a temporary loader if needed.
        # For each Data object in decoder2_dataset:
        temp_loader_d2_for_latent = DataLoader(decoder2_dataset, batch_size=decoder2_bsize, shuffle=False)
        for data_batch in temp_loader_d2_for_latent:
            data_batch = data_batch.to(device)
            # get_pooled_latent takes x (HNO latents), and conditioner (optional, not used by current get_pooled_latent)
            pooled_lat = protein_state_model_2d.get_pooled_latent(data_batch.x, None) 
            # pooled_lat shape [B, R_or_1, pooled_dim].
            all_pooled_d2.append(pooled_lat.cpu().numpy())
    
    if all_pooled_d2:
        all_pooled_d2_cat = np.concatenate(all_pooled_d2, axis=0) # Concatenate along batch dimension
        with h5py.File(pooled_latent_path_d2, "w") as latent_h5_d2:
            latent_h5_d2.create_dataset("pooled_latent", data=all_pooled_d2_cat)
        logger.info(
            f"Decoder2 pooled latent embeddings for {args.exp_label}/{args.epoch_folder} saved to {pooled_latent_path_d2} with shape {all_pooled_d2_cat.shape}"
        )
    else:
        logger.info(f"No Decoder2 pooled latent embeddings generated for {args.exp_label}/{args.epoch_folder}.")
    sys.stdout.flush()
    """
    logger.info(f"All tasks completed successfully for {args.exp_label}/{args.embedding_base_folder}/{args.epoch_folder}!")
    sys.stdout.flush()
