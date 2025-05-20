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
from torch_geometric.nn import ChebConv
from torch_cluster import knn_graph
from sklearn.model_selection import train_test_split

################################################################################
# (A) Argument Parsing
################################################################################
parser = argparse.ArgumentParser(
    description="Protein Reconstruction: HNO + Single Decoder + Optional Dihedral Loss (Mod Script 2)"
)
parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
args = parser.parse_args()

################################################################################
# (B) Pre-Logging Config Load (Minimal, just for log file path)
################################################################################
LOG_FILE_DEFAULT = "logfile_script2_mod.log"
log_file_path = LOG_FILE_DEFAULT
try:
    # Temporarily load config just for the log file path
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
    # File Handler
    try:
        fh = logging.FileHandler(log_file_path, mode="w")
        fh.setLevel(logging.DEBUG if args.debug else logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except IOError as e:
        print(f"Warning: Could not write to log file {log_file_path}: {e}. Logging to console only.")
        # Define formatter here if file handler failed
        if 'formatter' not in locals():
            formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if args.debug else logging.INFO)
    if 'formatter' not in locals(): # Ensure formatter exists
         formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logger.info(f"Logger initialized. Log file: {log_file_path}")
if args.debug: logger.debug("Debug mode is ON.")
else: logger.info("Debug mode is OFF.")

################################################################################
# (D) Device Setup (Global)
################################################################################
device_name = "cpu"
if torch.cuda.is_available():
    try:
        cuda_device_index = temp_config.get("cuda_device", 0) if 'temp_config' in locals() else 0
        device_name = f"cuda:{cuda_device_index}"
        torch.cuda.get_device_name(cuda_device_index) # Test validity
    except Exception:
        logger.warning(f"Could not validate CUDA device {cuda_device_index}. Defaulting to cuda:0 if available, else CPU.")
        if torch.cuda.is_available(): device_name = "cuda:0"
global_device = torch.device(device_name)
logger.info(f"Initial device check: {global_device}")


################################################################################
# (E) Utility Functions
################################################################################

# --- PDB Parsing ---
def parse_pdb(filename: str, logger: logging.Logger) -> Tuple[Dict, List]:
    """Parses ATOM records from a PDB file, handling alternate locations."""
    backbone_atoms = {"N", "CA", "C", "O", "OXT"}
    atoms_in_order = []; processed_atom_indices = set()
    try:
        with open(filename, 'r') as pdb_file:
            for line_num, line in enumerate(pdb_file, 1):
                if not line.startswith("ATOM  "): continue
                try:
                    atom_serial = int(line[6:11]); atom_name = line[12:16].strip()
                    alt_loc = line[16].strip(); res_name = line[17:20].strip()
                    chain_id = line[21].strip(); res_seq = int(line[22:26])
                except ValueError as e: logger.warning(f"Skipping PDB line {line_num}: {e}"); continue
                if alt_loc != '' and alt_loc != 'A': continue
                if atom_serial in processed_atom_indices: continue
                processed_atom_indices.add(atom_serial)
                orig_res_id = f"{chain_id}:{res_name}:{res_seq}"
                category = "backbone" if atom_name in backbone_atoms else "sidechain"
                atoms_in_order.append((orig_res_id, atom_serial, category))
    except FileNotFoundError: logger.error(f"PDB not found: {filename}"); return {}, []
    except Exception as e: logger.error(f"Error reading PDB {filename}: {e}", exc_info=True); return {}, []
    if not atoms_in_order: logger.error(f"No valid ATOM records found: {filename}")
    else: logger.info(f"Parsed {len(atoms_in_order)} ATOM records from {filename}.")
    return {}, atoms_in_order

def renumber_atoms_and_residues(atoms_in_order: List[Tuple[str, int, str]], logger: logging.Logger) -> Tuple[Dict, Dict]:
    """Renumbers residues and atoms consecutively starting from 0."""
    new_res_dict, orig_atom_map, orig_res_map = {}, {}, {}
    next_new_res_id, next_new_atom_index = 0, 0; seen_res_order, res_order_counter = {}, 0
    for r_id, _, _ in atoms_in_order:
        if r_id not in seen_res_order: seen_res_order[r_id] = res_order_counter; res_order_counter += 1
    sortable = [(seen_res_order[r_id], serial, r_id, cat) for r_id, serial, cat in atoms_in_order]; sortable.sort()
    for _, serial, r_id, cat in sortable:
        if r_id not in orig_res_map: orig_res_map[r_id] = next_new_res_id; new_res_dict[next_new_res_id] = {"backbone": [], "sidechain": []}; next_new_res_id += 1
        new_res_id = orig_res_map[r_id]; new_res_dict[new_res_id][cat].append(next_new_atom_index)
        orig_atom_map[serial] = next_new_atom_index; next_new_atom_index += 1
    logger.info(f"Renumbered {next_new_res_id} residues & {next_new_atom_index} atoms.")
    return new_res_dict, orig_atom_map

def get_global_indices(renumbered_dict: Dict) -> Tuple[List[int], List[int]]:
    """Extracts sorted global lists of backbone and sidechain atom indices."""
    bb_idx, sc_idx = [], []
    for res_id in sorted(renumbered_dict.keys()): bb_idx.extend(renumbered_dict[res_id]["backbone"]); sc_idx.extend(renumbered_dict[res_id]["sidechain"])
    return bb_idx, sc_idx

# --- JSON Loading ---
def load_heavy_atom_coords_from_json(json_file: str, logger: logging.Logger) -> Tuple[List[torch.Tensor], int]:
    """Loads coordinates from JSON assuming 0-based integer keys."""
    logger.info(f"Loading coordinates from JSON: {json_file}")
    try:
        with open(json_file, "r") as f: data = json.load(f)
    except FileNotFoundError: logger.error(f"JSON not found: {json_file}"); return [], -1
    except json.JSONDecodeError as e: logger.error(f"JSON decoding error: {e}"); return [], -1
    try:
        keys_int = sorted([int(k) for k in data.keys()]); keys_str = [str(k) for k in keys_int]
        if not keys_str: logger.error("No residue data in JSON."); return [], -1
        logger.info(f"Found {len(keys_str)} residues in JSON.")
        frame_data = data[keys_str[0]]["heavy_atom_coords_per_frame"]
        N_frames = len(frame_data)
        if N_frames == 0 or np.array(frame_data[0][0]).shape != (3,): raise ValueError("Invalid frame data")
        logger.info(f"Found {N_frames} frames in JSON.")
    except Exception as e: logger.error(f"Invalid JSON structure or keys: {e}"); return [], -1
    coords_frames, N_atoms_check = [], -1
    for frame_idx in range(N_frames):
        frame_coords_np, current_atoms = [], 0
        for res_key in keys_str:
            try:
                coords = np.array(data[res_key]["heavy_atom_coords_per_frame"][frame_idx], dtype=np.float32)
                if coords.ndim != 2 or coords.shape[1] != 3: raise ValueError("Bad Shape")
                frame_coords_np.append(coords); current_atoms += coords.shape[0]
            except Exception as e: logger.error(f"Error processing res {res_key} frame {frame_idx}: {e}"); return [], -1
        if frame_idx == 0: N_atoms_check = current_atoms; logger.info(f"Atoms/frame from JSON: {N_atoms_check}")
        elif current_atoms != N_atoms_check: logger.error("Inconsistent atom count"); return [], -1
        try: coords_frames.append(torch.tensor(np.concatenate(frame_coords_np, axis=0), dtype=torch.float32))
        except ValueError as e: logger.error(f"Concat error frame {frame_idx}: {e}"); return [], -1
    if not coords_frames: logger.error("Failed to load frames."); return [], -1
    return coords_frames, N_atoms_check

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
    """Aligns all coordinate frames to the first frame using Kabsch. Returns list on CPU."""
    logger.info("Aligning coordinate frames...")
    if not coords: logger.warning("Coordinate list empty."); return []
    ref = coords[0].float().to(device); aligned = [coords[0].cpu()]
    N_frames = len(coords) - 1
    for i, frame in enumerate(coords[1:], 1):
        _, aligned_dev = kabsch_algorithm(ref, frame.float().to(device), logger)
        aligned.append(aligned_dev.cpu())
        if (i % 500 == 0 or i == N_frames) and N_frames > 0: logger.info(f"Aligned {i}/{N_frames} frames...")
    logger.info("Finished aligning frames.")
    return aligned

# --- Graph Dataset ---
def build_graph_dataset(coords_list: List[torch.Tensor], knn_neighbors: int, logger: logging.Logger, device: torch.device) -> List[Data]: # Fixed param name
    """Builds PyG dataset with k-NN graphs. Returns Data objects on CPU."""
    logger.info(f"Building PyG dataset (k={knn_neighbors}) using device '{device}'...") # Fixed param name
    dataset = []; N_frames = len(coords_list)
    for i, coords_cpu in enumerate(coords_list):
        coords_dev = coords_cpu.to(device)
        edge_idx = knn_graph(coords_dev, k=knn_neighbors, loop=False, batch=None) # Fixed param name
        data = Data(x=coords_cpu, edge_index=edge_idx.cpu(), y=coords_cpu) # Ensure y exists
        dataset.append(data)
        if ((i + 1) % 500 == 0 or (i + 1) == N_frames) and N_frames > 0: logger.info(f"Built graph {i+1}/{N_frames}...")
    logger.info("Finished building PyG dataset.")
    return dataset

# --- Dihedral Utilities ---
@torch.jit.script
def compute_dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    b1=b-a; b2=c-b; b3=d-c; n1=torch.cross(b1,b2,dim=-1); n2=torch.cross(b2,b3,dim=-1)
    n1n=F.normalize(n1,p=2.,dim=-1,eps=1e-8); n2n=F.normalize(n2,p=2.,dim=-1,eps=1e-8)
    b2n=F.normalize(b2,p=2.,dim=-1,eps=1e-8); m1=torch.cross(n1n, b2n, dim=-1)
    x=(n1n*n2n).sum(dim=-1); y=(m1*n2n).sum(dim=-1); return torch.atan2(y,x)

def compute_all_dihedrals_vectorized(coords: torch.Tensor, info: Dict, N_res: int, logger: logging.Logger) -> Dict:
    """Computes all specified dihedrals vectorially."""
    if coords.ndim != 3: raise ValueError(f"Expected coords [B, N, 3], got {coords.shape}")
    B, N_atoms, _ = coords.shape; dev = coords.device; all_angles = {}
    for name, angle_info in info.items():
        indices, res_idx = angle_info.get('indices'), angle_info.get('res_idx')
        angles_out = torch.zeros(B, N_res, device=dev, dtype=coords.dtype)
        if indices is not None and res_idx is not None and indices[0].numel() > 0:
            try:
                idx_dev = [i.to(dev) for i in indices]; res_idx_dev = res_idx.to(dev)
                # Check bounds BEFORE indexing
                max_atom_idx_needed = max(i.max() for i in idx_dev)
                max_res_idx_needed = res_idx_dev.max()
                if max_atom_idx_needed >= N_atoms: raise IndexError(f"Atom index {max_atom_idx_needed} >= {N_atoms}")
                if max_res_idx_needed >= N_res: raise IndexError(f"Residue index {max_res_idx_needed} >= {N_res}")

                a,b,c,d = (coords[:, i, :] for i in idx_dev)
                values = compute_dihedral(a,b,c,d)
                angles_out[torch.arange(B,device=dev).unsqueeze(1), res_idx_dev.unsqueeze(0)] = values
            except IndexError as e: logger.error(f"Idx error computing {name}: {e}", exc_info=False)
            except Exception as e: logger.error(f"Error computing {name}: {e}", exc_info=True)
        all_angles[name] = angles_out
    return all_angles

def compute_angle_kl_div(p: torch.Tensor, t: torch.Tensor, n=36, r=(-np.pi, np.pi)) -> torch.Tensor:
    pd, td = p.detach(), t.detach(); dev = p.device
    if pd.numel()==0 or td.numel()==0: return torch.tensor(0.0, device=dev)
    e=torch.linspace(r[0],r[1],n+1,device=pd.device); ph=torch.histc(pd,n,r[0],r[1]); th=torch.histc(td,n,r[0],r[1])
    eps=1e-10; p_dist=ph/(ph.sum()+eps); t_dist=th/(th.sum()+eps); pld=torch.log(p_dist+eps)
    return F.kl_div(pld, t_dist, reduction='sum', log_target=False)

def compute_angle_js_div(p: torch.Tensor, t: torch.Tensor, n=36, r=(-np.pi, np.pi)) -> torch.Tensor:
    pd, td = p.detach(), t.detach(); dev = p.device
    if pd.numel()==0 or td.numel()==0: return torch.tensor(0.0, device=dev)
    e=torch.linspace(r[0],r[1],n+1,device=pd.device); ph=torch.histc(pd,n,r[0],r[1]); th=torch.histc(td,n,r[0],r[1])
    eps=1e-10; Q=ph/(ph.sum()+eps); P=th/(th.sum()+eps); M=0.5*(P+Q); lm=torch.log(M+eps)
    kl_pm=F.kl_div(lm, P, reduction='sum', log_target=False); kl_qm=F.kl_div(lm, Q, reduction='sum', log_target=False)
    return F.relu(0.5*(kl_pm+kl_qm))

def compute_angle_wasserstein(p: torch.Tensor, t: torch.Tensor, n=36, r=(-np.pi, np.pi)) -> torch.Tensor:
    pd, td = p.detach(), t.detach(); dev = p.device
    if pd.numel()==0 or td.numel()==0: return torch.tensor(0.0, device=dev)
    e=torch.linspace(r[0],r[1],n+1,device=pd.device); ph=torch.histc(pd,n,r[0],r[1]); th=torch.histc(td,n,r[0],r[1])
    eps=1e-10; p_dist=ph/(ph.sum()+eps); t_dist=th/(th.sum()+eps); pcdf=torch.cumsum(p_dist,0); tcdf=torch.cumsum(t_dist,0)
    return torch.sum(torch.abs(pcdf - tcdf))

# --- Checkpoint Utilities ---
def save_checkpoint(state: Dict, filename: str, logger: logging.Logger):
    try: torch.save(state, filename); logger.debug(f"Checkpoint saved: {filename}")
    except IOError as e: logger.error(f"Error saving checkpoint {filename}: {e}")
    sys.stdout.flush()

def load_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], filename: str, device: torch.device, logger: logging.Logger) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int]:
    start_epoch = 0
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint: '{filename}'")
        try:
            ckpt = torch.load(filename, map_location=device); start_epoch = ckpt.get("epoch", 0)
            try: model.load_state_dict(ckpt["model_state_dict"])
            except RuntimeError: model.load_state_dict(ckpt["model_state_dict"], strict=False); logger.warning("Loaded model non-strictly.")
            if optimizer and "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"]); logger.info("Optimizer state loaded.")
                    for state in optimizer.state.values(): # Move optimizer state to device
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor): state[k] = v.to(device)
                except Exception as e: logger.warning(f"Could not load optimizer state: {e}")
            elif optimizer: logger.warning("Optimizer state not in checkpoint.")
            model.to(device); logger.info(f"Checkpoint loaded. Resuming after epoch {start_epoch}") # Epoch COMPLETED
        except Exception as e: logger.error(f"Err loading ckpt: {e}", exc_info=True); start_epoch = 0; logger.warning("Training from scratch.")
    else: logger.info(f"No ckpt at '{filename}'. Training from scratch."); model.to(device)
    return model, optimizer, start_epoch # Return completed epoch

# --- MSE Utilities ---
def compute_bb_sc_mse(pred: torch.Tensor, target: torch.Tensor, bb_idx: torch.Tensor, sc_idx: torch.Tensor, logger: logging.Logger) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes Overall, Backbone, Sidechain MSE. Assumes flat inputs, indices on device."""
    crit = nn.MSELoss(); target = target.to(pred.dtype); all_mse = crit(pred, target)
    bb_mse, sc_mse = torch.tensor(0.,device=pred.device), torch.tensor(0.,device=pred.device)
    try:
        if bb_idx.numel() > 0: bb_mse = crit(pred[bb_idx], target[bb_idx])
        if sc_idx.numel() > 0: sc_mse = crit(pred[sc_idx], target[sc_idx])
    except IndexError: logger.error("MSE Indexing Error", exc_info=False)
    return all_mse, bb_mse, sc_mse

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


################################################################################
# (F) Model Definitions
################################################################################

# --- HNO Encoder ---
# --- HNO Encoder (Reverted to Simpler Version + Renamed Layers for Checkpoint Compatibility) ---

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
        # x: [N, 3] node features (coordinates)
        # edge_index: [2, E] graph connectivity
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO] Input x shape: {x.shape}, edge_index shape: {edge_index.shape}")

        # Ensure input is float
        x = x.float()

        x = self.conv1(x, edge_index)
        # BatchNorm expects [N, C] or [B, C, L], here it's [N, hidden_dim]
        x = self.bano1(F.leaky_relu(x))
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO] After conv1+bano1: {x.shape}")

        x = self.conv2(x, edge_index)
        x = self.bano2(F.leaky_relu(x))
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO] After conv2+bano2: {x.shape}")

        x = self.conv3(x, edge_index)
        x = self.bano3(F.relu(x)) # Note: ReLU here, LeakyReLU before
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO] After conv3+bano3: {x.shape}")

        # Last graph conv layer
        x = self.conv4(x, edge_index)
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO] After conv4: {x.shape}")

        # Apply L2 normalization on the node embeddings
        x = F.normalize(x, p=2.0, dim=1) # dim=1 is the feature dimension

        # Final linear layer to predict coordinates
        x = self.mlpRep(x)

        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO] Output (after mlpRep): {x.shape}")
            self._debug_logged = True # Only log shapes once per instance

        return x

    def forward_representation(self, x, edge_index, log_debug=False):
        """
        Returns the latent representation after the final conv and normalization
        (before final MLP). Output shape: [N, hidden_dim]
        """
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO.rep] Input x shape: {x.shape}")

        x = x.float() # Ensure input is float

        x = self.conv1(x, edge_index)
        x = self.bano1(F.leaky_relu(x))
        if log_debug and not self._debug_logged:
             logger.debug(f"[HNO.rep] After conv1+bano1: {x.shape}")

        x = self.conv2(x, edge_index)
        x = self.bano2(F.leaky_relu(x))
        if log_debug and not self._debug_logged:
             logger.debug(f"[HNO.rep] After conv2+bano2: {x.shape}")

        x = self.conv3(x, edge_index)
        x = self.bano3(F.relu(x))
        if log_debug and not self._debug_logged:
             logger.debug(f"[HNO.rep] After conv3+bano3: {x.shape}")

        x = self.conv4(x, edge_index)
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO.rep] After conv4: {x.shape}")
            # Don't set self._debug_logged = True here, allow forward() to log too

        # Normalize features before returning
        x = F.normalize(x, p=2.0, dim=1)
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO.rep] Output representation shape: {x.shape}")
            # Mark as logged only if this specific method was asked to log
            # self._debug_logged = True # Let forward() control the flag primarily

        return x


# --- Optional Cross Attention ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, att_dim=64):
        super().__init__(); self.s = att_dim**-0.5; self.q=nn.Linear(q_dim,att_dim,bias=False)
        self.k=nn.Linear(kv_dim,att_dim,bias=False); self.v=nn.Linear(kv_dim,att_dim,bias=False)
    def forward(self, q, k, v):
        Q=self.q(q); K=self.k(k); V=self.v(v)
        sc = torch.matmul(Q, K.transpose(-1, -2)) * self.s
        return torch.matmul(F.softmax(sc, dim=-1), V)

# --- Decoder2 Model ---
class ProteinStateReconstructor2D(nn.Module):
    """Single-step decoder. Predicts full coordinates from HNO embeddings + conditioner."""
    _logged_fwd = False
    def __init__(self, in_dim: int, N_nodes: int, cond_dim: int, pool_type: str = "blind", res_indices: Optional[List[List[int]]] = None, pool_size: Tuple[int, int] = (20, 4), mlp_h_dim: int = 128, mlp_layers: int = 2, pool2_size: Optional[Tuple[int, int]] = None, use_pool2: bool = False, use_attn: bool = False, attn_type: str = "global", logger: logging.Logger = logging.getLogger()):
        super().__init__(); self.N_nodes=N_nodes; self.in_dim=in_dim; self.cond_dim=cond_dim; self.logger=logger; self.pool_type=pool_type
        self.seg_indices: List[torch.LongTensor] = []
        if pool_type=="blind": self.seg_indices.append(torch.arange(N_nodes,dtype=torch.long))
        elif pool_type=="residue":
            if not res_indices: raise ValueError("Residue indices needed."); self.seg_indices = [torch.tensor(idx, dtype=torch.long) for idx in res_indices if idx];
            if not self.seg_indices: raise ValueError("Empty residue segments.")
        else: raise ValueError(f"Unknown pool_type={pool_type}")
        self.N_seg = len(self.seg_indices); self.logger.info(f"Dec2: Pool='{pool_type}', Segs={self.N_seg}")
        self.seg_pools = nn.ModuleList([nn.AdaptiveAvgPool2d(pool_size) for _ in self.seg_indices])
        self.prim_pool_dim = pool_size[0]*pool_size[1]; self.final_pool_dim = self.prim_pool_dim*self.N_seg
        self.glob_pool2 = None
        if use_pool2 and pool2_size and self.N_seg>0: self.glob_pool2=nn.AdaptiveAvgPool2d(pool2_size); self.final_pool_dim=pool2_size[0]*pool2_size[1]; self.logger.info(f"Dec2: Use Pool2. FinalPoolDim={self.final_pool_dim}")
        else: self.logger.info(f"Dec2: Primary Pool only. ConcatPoolDim={self.final_pool_dim}")
        self.attn = None
        if use_attn: self.logger.warning("Attn impl omitted.")
        mlp_in_dim=self.cond_dim+self.final_pool_dim; self.decoder=build_decoder_mlp(mlp_in_dim, 3, mlp_layers, mlp_h_dim)
        self.logger.info(f"Dec2 MLP: In={mlp_in_dim}, Out=3, Layers={mlp_layers}, Hidden={mlp_h_dim}")

    def get_pooled_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Pools input [B*N, E] to get global latent [B, final_pooled_dim]."""
        if x.ndim!=2 or x.shape[0]%self.N_nodes!=0: raise ValueError(f"Bad x shape {x.shape}")
        B=x.shape[0]//self.N_nodes; x_r=x.view(B,self.N_nodes,self.in_dim); seg_pooled=[]
        for i, seg_idx in enumerate(self.seg_indices):
            idx_dev = seg_idx.to(x.device)
            if len(idx_dev)==0: seg_pooled.append(torch.zeros(B,self.prim_pool_dim,device=x.device,dtype=x.dtype)); continue
            seg_x = x_r[:,idx_dev,:].unsqueeze(1); pool_f=self.seg_pools[i](seg_x).view(B,-1); seg_pooled.append(pool_f)
        if not seg_pooled: return torch.zeros(B, self.final_pool_dim, device=x.device, dtype=x.dtype)
        l1_stack = torch.stack(seg_pooled, dim=1) # [B, num_segments, primary_dim]
        return self.glob_pool2(l1_stack.unsqueeze(1)).view(B,-1) if self.glob_pool2 else l1_stack.view(B,-1) # Return [B, final_dim]

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor], conditioner: torch.Tensor) -> torch.Tensor:
        """Predicts coords [B*N, 3] from embeddings [B*N, E] and conditioner [N, Cdim]."""
        dev=x.device; cond=conditioner.to(dev)
        if x.shape[1]!=self.in_dim or cond.shape!=(self.N_nodes,self.cond_dim): raise ValueError("Shape mismatch")
        B_N=x.shape[0]
        if B_N%self.N_nodes!=0: B=batch.max().item()+1 if batch is not None else -1; assert B>0 and B*self.N_nodes==B_N,"Batch size err"
        else: B=B_N//self.N_nodes
        pool_g=self.get_pooled_latent(x); pool_g_ex=pool_g.unsqueeze(1).expand(-1,self.N_nodes,-1) # [B, N, final_pool_dim]
        cond_ex=cond.unsqueeze(0).expand(B,-1,-1) # [B, N, cond_dim]
        mlp_in=torch.cat([cond_ex, pool_g_ex],dim=-1); pred=self.decoder(mlp_in.view(B_N,-1))
        if not ProteinStateReconstructor2D._logged_fwd and self.logger.isEnabledFor(logging.DEBUG): self.logger.debug(f"[Dec2 Fwd] Shapes: In={x.shape} Cond={cond.shape} Pool={pool_g.shape} MLPIn={mlp_in.shape[-1]} Out={pred.shape}"); ProteinStateReconstructor2D._logged_fwd=True
        return pred


################################################################################
# (G) Training Functions
################################################################################

# --- Train HNO ---

def train_hno_model(model: HNO, tr_loader: DataLoader, te_loader: DataLoader, bb_idx: torch.Tensor, sc_idx: torch.Tensor, N_epochs: int, lr: float, ckpt: str, save_int: int, dev: torch.device, logger: logging.Logger):
    model=model.to(dev); bb_idx, sc_idx=bb_idx.to(dev), sc_idx.to(dev)
    params=list(filter(lambda p: p.requires_grad, model.parameters())); opt=torch.optim.Adam(params, lr=lr) if params else None
    model, opt, start_ep = load_checkpoint(model, opt, ckpt, dev, logger) # start_ep is COMPLETED epoch
    train_start_epoch = start_ep # Start training epoch AFTER the loaded one
    logger.info(f"Start HNO train from epoch {train_start_epoch + 1}/{N_epochs}, LR={lr}")

    if train_start_epoch >= N_epochs:
        logger.info(f"Loaded checkpoint epoch ({start_ep}) >= target epochs ({N_epochs}). Skipping HNO training.")
        return model

    for ep in range(train_start_epoch, N_epochs):
        model.train(); tr_all, tr_bb, tr_sc = 0.0, 0.0, 0.0; n_tr = len(tr_loader)
        if not opt: logger.warning("No optimizer for HNO"); break
        for i, data in enumerate(tr_loader):
            data=data.to(dev); opt.zero_grad(set_to_none=True)
            # log=(ep==train_start_epoch and i==0 and logger.isEnabledFor(logging.DEBUG)) # Original log flag calculation (not used in call now)

            # --- MODIFIED CALL: Removed 'log_debug' argument ---
            pred = model(data.x, data.edge_index)
            # ----------------------------------------------------

            all_mse, bb_mse, sc_mse = compute_bb_sc_mse(pred, data.y, bb_idx, sc_idx, logger)
            loss=all_mse; loss.backward(); opt.step()
            tr_all+=all_mse.item(); tr_bb+=bb_mse.item(); tr_sc+=sc_mse.item()
        avg_tr = [x/n_tr if n_tr else 0 for x in [tr_all, tr_bb, tr_sc]]
        model.eval(); te_all, te_bb, te_sc = 0.0, 0.0, 0.0; n_te = len(te_loader)
        with torch.no_grad():
            for data in te_loader:
                data=data.to(dev)
                 # --- MODIFIED CALL: Removed 'log_debug' argument ---
                pred=model(data.x, data.edge_index)
                 # ----------------------------------------------------
                a,b,s = compute_bb_sc_mse(pred, data.y, bb_idx, sc_idx, logger); te_all+=a.item(); te_bb+=b.item(); te_sc+=s.item()
        avg_te = [x/n_te if n_te else 0 for x in [te_all, te_bb, te_sc]]
        logger.info(f"[HNO] Ep {ep+1} TR MSE={avg_tr[0]:.5f}(BB={avg_tr[1]:.5f},SC={avg_tr[2]:.5f}) | TE MSE={avg_te[0]:.5f}(BB={avg_te[1]:.5f},SC={avg_te[2]:.5f})")
        ep_num = ep + 1 # Current epoch number (1-based)
        if opt and (ep_num % save_int == 0 or ep_num == N_epochs): save_checkpoint({"epoch":ep_num, "model_state_dict":model.state_dict(), "optimizer_state_dict":opt.state_dict()}, ckpt, logger)
    logger.info(f"Finished HNO training. Ckpt: {ckpt}")
    return model


'''def train_hno_model(model: HNO, tr_loader: DataLoader, te_loader: DataLoader, bb_idx: torch.Tensor, sc_idx: torch.Tensor, N_epochs: int, lr: float, ckpt: str, save_int: int, dev: torch.device, logger: logging.Logger):
    model=model.to(dev); bb_idx, sc_idx=bb_idx.to(dev), sc_idx.to(dev)
    params=list(filter(lambda p: p.requires_grad, model.parameters())); opt=torch.optim.Adam(params, lr=lr) if params else None
    model, opt, start_ep = load_checkpoint(model, opt, ckpt, dev, logger) # start_ep is COMPLETED epoch
    train_start_epoch = start_ep # Start training epoch AFTER the loaded one
    logger.info(f"Start HNO train from epoch {train_start_epoch + 1}/{N_epochs}, LR={lr}")

    if train_start_epoch >= N_epochs:
        logger.info(f"Loaded checkpoint epoch ({start_ep}) >= target epochs ({N_epochs}). Skipping HNO training.")
        return model

    for ep in range(train_start_epoch, N_epochs):
        model.train(); tr_all, tr_bb, tr_sc = 0.0, 0.0, 0.0; n_tr = len(tr_loader)
        if not opt: logger.warning("No optimizer for HNO"); break
        for i, data in enumerate(tr_loader):
            data=data.to(dev); opt.zero_grad(set_to_none=True); log=(ep==train_start_epoch and i==0 and logger.isEnabledFor(logging.DEBUG))
            pred = model(data.x, data.edge_index, log_debug=log)
            all_mse, bb_mse, sc_mse = compute_bb_sc_mse(pred, data.y, bb_idx, sc_idx, logger)
            loss=all_mse; loss.backward(); opt.step()
            tr_all+=all_mse.item(); tr_bb+=bb_mse.item(); tr_sc+=sc_mse.item()
        avg_tr = [x/n_tr if n_tr else 0 for x in [tr_all, tr_bb, tr_sc]]
        model.eval(); te_all, te_bb, te_sc = 0.0, 0.0, 0.0; n_te = len(te_loader)
        with torch.no_grad():
            for data in te_loader: data=data.to(dev); pred=model(data.x, data.edge_index); a,b,s = compute_bb_sc_mse(pred, data.y, bb_idx, sc_idx, logger); te_all+=a.item(); te_bb+=b.item(); te_sc+=s.item()
        avg_te = [x/n_te if n_te else 0 for x in [te_all, te_bb, te_sc]]
        logger.info(f"[HNO] Ep {ep+1} TR MSE={avg_tr[0]:.5f}(BB={avg_tr[1]:.5f},SC={avg_tr[2]:.5f}) | TE MSE={avg_te[0]:.5f}(BB={avg_te[1]:.5f},SC={avg_te[2]:.5f})")
        ep_num = ep + 1 # Current epoch number (1-based)
        if opt and (ep_num % save_int == 0 or ep_num == N_epochs): save_checkpoint({"epoch":ep_num, "model_state_dict":model.state_dict(), "optimizer_state_dict":opt.state_dict()}, ckpt, logger)
    logger.info(f"Finished HNO training. Ckpt: {ckpt}")
    return model
'''


import random  # Make sure this is somewhere in your imports.

def train_decoder2_model(
    model: ProteinStateReconstructor2D,
    tr_loader: DataLoader,
    te_loader: DataLoader,
    bb_idx: torch.Tensor,
    sc_idx: torch.Tensor,
    conditioner: torch.Tensor,
    N_atoms: int,
    N_epochs: int,
    lr: float,
    ckpt: str,
    save_int: int,
    dev: torch.device,
    logger: logging.Logger,
    base_w: float = 1.0,  # Weight for coordinate-based MSE
    use_di: bool = False, # Whether dihedral-based loss is allowed at all
    di_info: Optional[Dict] = None,
    di_mask: Optional[torch.Tensor] = None,
    N_res: Optional[int] = None,
    l_div: float = 0.0,  # Weight for the dihedral distribution-divergence loss
    l_mse: float = 0.0,  # Weight for the dihedral MSE term
    div_t: str = "KL",   # Type of divergence (KL, JS, or WASSERSTEIN)
    fraction_dihedral: float = 0.1  # New: fraction of batches to apply dihedral
):
    """
    Trains the decoder model for a given number of epochs, optionally computing
    dihedral-based loss for a random fraction of batches. The fraction is controlled
    by 'fraction_dihedral' (default=1.0 means always apply dihedral).

    Args:
        model: Decoder model instance (ProteinStateReconstructor2D).
        tr_loader: Training DataLoader with Data objects.
        te_loader: Validation DataLoader.
        bb_idx: 1D Tensor of backbone-atom indices (on CPU or device).
        sc_idx: 1D Tensor of sidechain-atom indices.
        conditioner: Tensor used as the conditional input (e.g., X_ref or z_ref).
        N_atoms: Number of atoms per frame.
        N_epochs: Number of epochs to train.
        lr: Learning rate for optimizer.
        ckpt: Path to checkpoint file for loading/saving.
        save_int: Save checkpoint every 'save_int' epochs.
        dev: Torch device (cpu or cuda).
        logger: Logging instance.
        base_w: Weight for the coordinate-based MSE. (Default=1.0)
        use_di: If True, dihedral-based losses can be computed.
        di_info: Precomputed dihedral index info (dict).
        di_mask: Boolean mask of shape [N_res, num_angle_types], indicates valid angles.
        N_res: Number of residues.
        l_div: Loss coefficient for distribution-based dihedral difference (KL, JS, or Wass).
        l_mse: Loss coefficient for direct dihedral MSE.
        div_t: Divergence type: 'KL', 'JS', or 'WASSERSTEIN'.
        fraction_dihedral: Fraction of training batches where dihedral-based loss is applied.

    Returns:
        model: The trained decoder model (with final parameters).
    """
    # Make sure the relevant dihedral utilities are in scope (from Code 2).
    # e.g. compute_all_dihedrals_vectorized, compute_angle_kl_div, etc.

    model = model.to(dev)
    bb_idx, sc_idx = bb_idx.to(dev), sc_idx.to(dev)
    mask_dev = di_mask.to(dev) if di_mask is not None else None

    # Build an optimizer
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    opt = torch.optim.Adam(params, lr=lr) if params else None

    # Load checkpoint if it exists
    model, opt, start_ep = load_checkpoint(model, opt, ckpt, dev, logger)
    train_start_epoch = start_ep
    best_train_loss = float("inf")
    logger.info(f"Start Dec2 train from epoch {train_start_epoch + 1}/{N_epochs}, LR={lr}, BaseW={base_w}")

    # Pick the correct divergence function
    comp_div = None
    valid_di = False
    if use_di:
        if div_t == "JS":
            comp_div = compute_angle_js_div
        elif div_t == "WASSERSTEIN":
            comp_div = compute_angle_wasserstein
        elif div_t == "KL":
            comp_div = compute_angle_kl_div
        else:
            logger.warning(f"Unknown div_type '{div_t}', defaulting to KL.")
            div_t = "KL"
            comp_div = compute_angle_kl_div

        # Check if we have everything needed (di_info, mask, N_res)
        if di_info and mask_dev is not None and N_res and comp_div:
            valid_di = True
            logger.info(f"  Dihedral Loss ENABLED: Type={div_t}, "
                        f"lambda_div={l_div}, lambda_mse={l_mse}, fraction={fraction_dihedral}")
        else:
            logger.warning("Dihedral components missing or invalid. Disabling dihedral loss.")
    else:
        logger.info("  Dihedral Loss DISABLED.")

    # Define the angle types we compute
    angles = ['phi', 'psi', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5']

    if train_start_epoch >= N_epochs:
        logger.info(f"Loaded checkpoint epoch ({start_ep}) >= target epochs ({N_epochs}). "
                    f"Skipping Decoder2 training.")
        return model

    # ------------------------ Training Loop ------------------------
    for ep in range(train_start_epoch, N_epochs):
        model.train()
        tr_metrics = [0.0]*6  # [tot, coord, bb, sc, div, d_mse]
        n_tr = len(tr_loader)

        if not opt:
            logger.warning("No optimizer for Decoder2 (empty param list?)")
            break

        for i, data in enumerate(tr_loader):
            data = data.to(dev)
            opt.zero_grad(set_to_none=True)

            # Forward pass: coordinate predictions
            pred = model(data.x, data.batch, conditioner)
            c_mse, b_mse, s_mse = compute_bb_sc_mse(pred, data.y, bb_idx, sc_idx, logger)
            loss = c_mse * base_w

            # By default, the dihedral terms are zero
            div_l = torch.tensor(0., device=dev)
            d_mse_l = torch.tensor(0., device=dev)

            # Decide if we incorporate the dihedral portion for this batch
            if use_di and valid_di:
                # For example, use random fraction
                if random.random() < fraction_dihedral:
                    B_N = pred.shape[0]
                    if B_N > 0 and B_N % N_atoms == 0:
                        B = B_N // N_atoms
                        pred3d = pred.view(B, N_atoms, 3)
                        true3d = data.y.view(B, N_atoms, 3)

                        try:
                            # Compute dihedrals for predicted & target
                            pred_a = compute_all_dihedrals_vectorized(pred3d, di_info, N_res, logger)
                            true_a = compute_all_dihedrals_vectorized(true3d, di_info, N_res, logger)

                            for name in angles:
                                pa = pred_a.get(name)
                                ta = true_a.get(name)
                                if pa is not None and ta is not None:
                                    angle_idx = angles.index(name)
                                    if (mask_dev.ndim == 2 and
                                        mask_dev.shape[0] == N_res and
                                        mask_dev.shape[1] == len(angles)):

                                        mask = mask_dev[:, angle_idx]
                                        if mask.any():
                                            mask_ex = mask.view(1, -1).expand(B, -1)
                                            pv = pa[mask_ex]
                                            tv = ta[mask_ex]
                                            if pv.numel() > 0:
                                                d_mse_l += F.mse_loss(pv, tv)
                                                div_l += comp_div(pv, tv)
                                    else:
                                        logger.error(f"Mask shape mismatch: {mask_dev.shape}, "
                                                     f"expected ({N_res}, {len(angles)})")
                                        valid_di = False
                                        break
                            if valid_di:
                                loss += l_div * div_l + l_mse * d_mse_l

                        except Exception as e:
                            logger.error(f"Dih. loss error in batch {i}: {e}", exc_info=False)

                    elif B_N > 0:
                        logger.warning(f"Batch size {B_N} not a multiple of N_atoms={N_atoms}. "
                                       f"Skipping dihedral for batch {i}.")

            # Backprop and update
            if loss.requires_grad:
                loss.backward()
                opt.step()

            # Collect training stats
            batch_metrics = [
                loss.item(),
                c_mse.item(),
                b_mse.item(),
                s_mse.item(),
                div_l.item(),
                d_mse_l.item()
            ]
            for j in range(6):
                tr_metrics[j] += batch_metrics[j]
        # end for (training data)

        # Compute average training stats
        avg_tr = [x / n_tr if n_tr else 0 for x in tr_metrics]

        # ------------------ Validation Loop ------------------
        model.eval()
        te_metrics = [0.0]*6  # [tot, coord, bb, sc, div, d_mse]
        n_te = len(te_loader)

        with torch.no_grad():
            for data in te_loader:
                data = data.to(dev)
                pred = model(data.x, data.batch, conditioner)

                c_mse, b_mse, s_mse = compute_bb_sc_mse(pred, data.y, bb_idx, sc_idx, logger)
                loss = c_mse * base_w

                div_l = torch.tensor(0., device=dev)
                d_mse_l = torch.tensor(0., device=dev)

                # Typically we either do full dihedral or none in validation
                if use_di and valid_di:
                    B_N = pred.shape[0]
                    if B_N > 0 and B_N % N_atoms == 0:
                        B = B_N // N_atoms
                        pred3d = pred.view(B, N_atoms, 3)
                        true3d = data.y.view(B, N_atoms, 3)

                        try:
                            pred_a = compute_all_dihedrals_vectorized(pred3d, di_info, N_res, logger)
                            true_a = compute_all_dihedrals_vectorized(true3d, di_info, N_res, logger)

                            for name in angles:
                                pa = pred_a.get(name)
                                ta = true_a.get(name)
                                if pa is not None and ta is not None:
                                    angle_idx = angles.index(name)
                                    if (mask_dev.ndim == 2 and
                                        mask_dev.shape[0] == N_res and
                                        mask_dev.shape[1] == len(angles)):

                                        mask = mask_dev[:, angle_idx]
                                        if mask.any():
                                            mask_ex = mask.view(1, -1).expand(B, -1)
                                            pv = pa[mask_ex]
                                            tv = ta[mask_ex]
                                            if pv.numel() > 0:
                                                d_mse_l += F.mse_loss(pv, tv)
                                                div_l += comp_div(pv, tv)
                                    else:
                                        logger.error("Val Mask shape mismatch: "
                                                     f"{mask_dev.shape}")
                                        valid_di = False
                                        break
                            if valid_di:
                                loss += l_div * div_l + l_mse * d_mse_l

                        except Exception:
                            pass  # ignore dihedral errors in validation

                # Accumulate validation stats
                batch_metrics = [
                    loss.item(),
                    c_mse.item(),
                    b_mse.item(),
                    s_mse.item(),
                    div_l.item(),
                    d_mse_l.item()
                ]
                for j in range(6):
                    te_metrics[j] += batch_metrics[j]

        # end val loop
        avg_te = [x / n_te if n_te else 0 for x in te_metrics]

        # Print epoch summary
        log_str = (f"[Dec2] Ep {ep+1} "
                   f"TR: Tot={avg_tr[0]:.4f} Coord={avg_tr[1]:.4f} "
                   f"(BB={avg_tr[2]:.4f}, SC={avg_tr[3]:.4f}) "
                   f"Dih(MSE={avg_tr[5]:.4f}, {div_t}={avg_tr[4]:.4f}) | "
                   f"TE: Tot={avg_te[0]:.4f} Coord={avg_te[1]:.4f} "
                   f"(BB={avg_te[2]:.4f}, SC={avg_te[3]:.4f}) "
                   f"Dih(MSE={avg_te[5]:.4f}, {div_t}={avg_te[4]:.4f})")
        logger.info(log_str)

        ep_num = ep + 1
        # Periodically save the model
        if opt:
            current_train_loss = avg_tr[0]  # TOT is stored in index 0
            if current_train_loss < best_train_loss:
                best_train_loss = current_train_loss  # update
                save_checkpoint({
                    "epoch": ep_num,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict()
                }, ckpt, logger)
                logger.info(
                    f"[Dec2] Saved checkpoint at epoch {ep_num}, improved train loss={current_train_loss:.5f}"
                )
            else:
                logger.info(
                    f"[Dec2] Epoch {ep_num} not saved (train loss={current_train_loss:.5f} >= best={best_train_loss:.5f})"
                )
    logger.info(f"Finished Dec2 training. Ckpt: {ckpt}")
    # Reload the newly saved best checkpoint so 'model' in memory matches it
    best_ckpt_data = torch.load(ckpt, map_location=dev)
    model.load_state_dict(best_ckpt_data["model_state_dict"])
    model.eval()
    return model



################################################################################
# (H) Export Final Outputs Function
################################################################################
@torch.no_grad()
def export_final_outputs(
    hno: 'HNO', # Use forward declaration if classes defined later
    dec2: 'ProteinStateReconstructor2D',
    full_dset: List['Data'], # Dataset with original coords for GT and HNO input
    dec_in_dset: List['Data'], # Dataset with embeddings for Decoder2 input/pooled
    conditioner: torch.Tensor, # X_ref or z_ref on CPU
    N_atoms: int,
    struct_dir: str,
    latent_dir: str,
    dev: torch.device,
    logger: logging.Logger
):
    """
    Exports final predictions and intermediate results to HDF5 files.
    Logs file existence at the end based on os.path.isfile.

    Writes:
     1) structures/ground_truth_aligned.h5 (dataset key: 'ground_truth_coords')
     2) structures/hno_reconstructions.h5  (dataset key: 'hno_coords')
     3) structures/full_coords.h5          (dataset key: 'full_coords', from Decoder2)
     4) latent_reps/hno_embeddings.h5      (dataset key: 'hno_embeddings')
     5) latent_reps/pooled_embedding.h5    (dataset key: 'pooled_embedding', from Decoder2)
    """
    logger.info("--- Exporting Final Outputs ---")
    hno.eval().to(dev); dec2.eval().to(dev); cond_dev = conditioner.to(dev)

    # Define output file paths
    paths = {
        'gt': os.path.join(struct_dir, "ground_truth_aligned.h5"),
        'hno_rec': os.path.join(struct_dir, "hno_reconstructions.h5"),
        'full': os.path.join(struct_dir, "full_coords.h5"),
        'hno_emb': os.path.join(latent_dir, "hno_embeddings.h5"),
        'pool': os.path.join(latent_dir, "pooled_embedding.h5")
    }

    N_frames = len(full_dset)
    if N_frames == 0 or len(dec_in_dset) != N_frames:
        logger.warning("Dataset empty or mismatched. Skipping export.")
        return

    # Determine dimensions
    emb_dim = hno.conv4.out_channels # Assuming c4 is the last conv layer in HNO
    pool_dim = 0
    try:
        # Ensure the sample data exists and get pooled dim
        if dec_in_dset:
            pool_dim = dec2.get_pooled_latent(dec_in_dset[0].x.to(dev)).shape[1]
        else:
            logger.warning("Decoder input dataset empty, cannot determine pool dim.")
    except Exception as e:
        logger.error(f"Could not get pooled dim: {e}.")

    logger.info(f"Export Details: Frames={N_frames}, Atoms={N_atoms}, EmbDim={emb_dim}, PooledDim={pool_dim}")

    # Using separate file handles and explicit closing in finally for robustness
    files_opened: List[h5py.File] = []
    dsets_created_flags: Dict[str, bool] = {k: False for k in paths} # Track if create_dataset was called

    try:
        # Open files
        f_gt = h5py.File(paths['gt'],"w"); files_opened.append(f_gt)
        f_hno = h5py.File(paths['hno_rec'],"w"); files_opened.append(f_hno)
        f_full = h5py.File(paths['full'],"w"); files_opened.append(f_full)
        f_emb = h5py.File(paths['hno_emb'],"w"); files_opened.append(f_emb)
        f_pool = h5py.File(paths['pool'],"w"); files_opened.append(f_pool)

        # Create datasets
        dset_gt = dset_hno_rec = dset_full = dset_hno_emb = dset_pool = None # Initialize
        if N_atoms > 0:
            dset_gt = f_gt.create_dataset("ground_truth_coords",(N_frames,N_atoms,3),dtype='f4'); dsets_created_flags['gt'] = True
            dset_hno_rec = f_hno.create_dataset("hno_coords",(N_frames,N_atoms,3),dtype='f4'); dsets_created_flags['hno_rec'] = True
            dset_full = f_full.create_dataset("full_coords",(N_frames,N_atoms,3),dtype='f4'); dsets_created_flags['full'] = True
        if N_atoms > 0 and emb_dim > 0:
            dset_hno_emb = f_emb.create_dataset("hno_embeddings",(N_frames,N_atoms,emb_dim),dtype='f4'); dsets_created_flags['hno_emb'] = True
        if pool_dim > 0:
            dset_pool = f_pool.create_dataset("pooled_embedding",(N_frames,pool_dim),dtype='f4'); dsets_created_flags['pool'] = True

        logger.debug("HDF5 datasets potentially created (check flags).")

        # Data Writing Loop (Use DataLoader for efficiency if needed, otherwise iterate)
        for i in range(N_frames):
            d_orig, d_dec = full_dset[i].to(dev), dec_in_dset[i].to(dev)

            if dset_gt: dset_gt[i] = d_orig.y.cpu().numpy()
            if dset_hno_rec: dset_hno_rec[i] = hno(d_orig.x, d_orig.edge_index).cpu().numpy()
            if dset_hno_emb: dset_hno_emb[i] = d_dec.x.cpu().numpy() # Embedding already computed
            if dset_full: dset_full[i] = dec2(d_dec.x, None, cond_dev).cpu().numpy() # No batch index needed for B=1
            if dset_pool: dset_pool[i] = dec2.get_pooled_latent(d_dec.x).squeeze(0).cpu().numpy()

            if (i + 1) % 500 == 0 or (i + 1) == N_frames:
                logger.info(f"Exported frame {i + 1}/{N_frames}")

    except Exception as e:
        logger.error(f"HDF5 export failed during processing: {e}", exc_info=True)
    finally:
        # Ensure all opened files are closed
        for f in files_opened:
            try:
                f.close()
            except Exception as close_e:
                logger.error(f"Error closing HDF5 file handle: {close_e}", exc_info=False)

    # --- Corrected Final Logging Logic ---
    logger.info("Export process finished. Verifying file existence:")
    all_paths_map = { # Map internal key to name and path
        "gt": ("Ground Truth", paths['gt']),
        "hno_rec": ("HNO Recon", paths['hno_rec']),
        "full": ("Full Coords", paths['full']),
        "hno_emb": ("HNO Embeddings", paths['hno_emb']),
        "pool": ("Pooled Embedding", paths['pool'])
    }

    for key, (name, file_path) in all_paths_map.items():
        try:
            if os.path.isfile(file_path):
                # Check if the dataset was intended to be created based on dims
                should_exist = False
                if key in ['gt', 'hno_rec', 'full'] and N_atoms > 0: should_exist = True
                elif key == 'hno_emb' and N_atoms > 0 and emb_dim > 0: should_exist = True
                elif key == 'pool' and pool_dim > 0: should_exist = True

                if not should_exist: # File exists but wasn't expected based on dims
                     logger.warning(f"  {name} => {file_path} (Exists but N/A based on dimensions)")
                elif os.path.getsize(file_path) > 50: # Check size > 50 bytes (HDF5 header is small)
                    logger.info(f"  {name} => {file_path} (Exists)")
                else:
                    logger.warning(f"  {name} => {file_path} (Exists but is suspiciously small/empty!)")
            else:
                logger.warning(f"  {name} => {file_path} (File NOT found!)")
        except OSError as e:
             logger.error(f"  Error checking file {name} at {file_path}: {e}")
        except Exception as e:
            logger.error(f"  Unexpected error checking file {name} at {file_path}: {e}")

################################################################################
# (I) Main Execution Function
################################################################################
def main():
    """Main orchestration function."""
    start_time = time.time()
    logger.info("================ Script Starting ================")
    global global_device

    # --- Load Full Configuration ---
    try:
        with open(args.config, "r") as f: config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {args.config}")
    except Exception as e: logger.error(f"Failed to load config: {e}", exc_info=True); sys.exit(1)

    # --- Extract Parameters & Setup ---
    try: # Use .get for safety with defaults where possible
        force_cpu = config.get("force_cpu", False); cuda_idx = config.get("cuda_device", 0)
        num_workers = config.get("num_workers", 0); data_cfg = config["data"]; json_p = data_cfg["json_path"]; pdb_p = data_cfg["pdb_path"]
        graph_cfg = config["graph"]; knn = graph_cfg["knn_value"]; hno_cfg = config["hno_encoder"]; hno_hdim = hno_cfg["hidden_dim"]
        hno_k = hno_cfg["cheb_order"]; hno_ep = hno_cfg["num_epochs"]; hno_lr = hno_cfg["learning_rate"]; hno_bs = hno_cfg["batch_size"]
        hno_si = hno_cfg.get("save_interval", 500); dec2_cfg = config["decoder2"]; dec2_ep = dec2_cfg["num_epochs"]; dec2_lr = dec2_cfg["learning_rate"]
        dec2_bs = dec2_cfg["batch_size"]; dec2_base_w = dec2_cfg.get("base_loss_weight", 1.0); dec2_si = dec2_cfg.get("save_interval", 500)
        d2s = config["decoder2_settings"]; d2_c_mode = d2s.get("conditioner_mode","z_ref"); d2_p_type = d2s.get("pooling_type","blind")
        d2_ph, d2_pw = d2s.get("output_height",20), d2s.get("output_width",4); d2_mlp_h = d2s.get("mlp_hidden_dim",128)
        d2_mlp_l = d2s.get("num_hidden_layers",2); d2_use_p2 = d2s.get("use_second_level_pooling",False); d2_ph2, d2_pw2 = d2s.get("output_height2"), d2s.get("output_width2")
        d2_use_a = d2s.get("use_cross_attention",False); d2_a_type = d2s.get("cross_attention_type","global"); di_cfg = config.get("dihedral_loss",{})
        use_di_cfg = di_cfg.get("use_dihedral_loss",False); torsion_p = di_cfg.get("torsion_info_path","torsion.json"); l_div = di_cfg.get("lambda_divergence",0.0)
        l_mse = di_cfg.get("lambda_torsion_mse",0.0); div_t = di_cfg.get("divergence_type","KL").upper(); out_cfg = config.get("output_directories",{})
        ckpt_dir = out_cfg.get("checkpoint_dir","checkpoints"); struct_dir = out_cfg.get("structure_dir","structures"); latent_dir = out_cfg.get("latent_dir","latent_reps")
    except KeyError as e: logger.error(f"Missing config key: {e}"); sys.exit(1)

    # --- Finalize Device ---
    device = torch.device("cpu") if force_cpu else global_device
    pin_mem = (device.type == "cuda")
    logger.info(f"Final device: {device}")
    try: [os.makedirs(d, exist_ok=True) for d in [ckpt_dir, struct_dir, latent_dir]]
    except OSError as e: logger.error(f"Dir creation failed: {e}"); sys.exit(1)

    # --- Stage 1: Data Loading & Prep ---
    logger.info("--- Stage 1: Data Loading & Preprocessing ---")
    _, atoms_ord = parse_pdb(pdb_p, logger);
    if not atoms_ord: sys.exit(1)
    renum_d, _ = renumber_atoms_and_residues(atoms_ord, logger)
    bb_list, sc_list = get_global_indices(renum_d); N_atoms_pdb = len(bb_list)+len(sc_list)
    bb_t, sc_t = torch.tensor(bb_list, dtype=torch.long), torch.tensor(sc_list, dtype=torch.long) # CPU tensors
    logger.info(f"PDB: Atoms={N_atoms_pdb} (BB={len(bb_list)}, SC={len(sc_list)})")
    coords_f, N_atoms_json = load_heavy_atom_coords_from_json(json_p, logger)
    if not coords_f: sys.exit(1)
    if N_atoms_json!=N_atoms_pdb: logger.error(f"Atom count mismatch JSON({N_atoms_json}) != PDB({N_atoms_pdb})"); sys.exit(1)
    N_atoms = N_atoms_pdb
    aligned_c = align_frames_to_first(coords_f, logger, device);
    if not aligned_c: sys.exit(1)
    full_dset = build_graph_dataset(aligned_c, knn_neighbors=knn, logger=logger, device=device); # Use corrected knn_neighbors
    if not full_dset: sys.exit(1)

    # --- Stage 2: HNO Training/Loading ---
    logger.info("--- Stage 2: HNO Encoder ---")
    tr_hno, te_hno = train_test_split(full_dset, test_size=0.1, random_state=42)
    load_tr_hno = DataLoader(tr_hno,hno_bs,shuffle=True,num_workers=num_workers,pin_memory=pin_mem,drop_last=True)
    load_te_hno = DataLoader(te_hno,hno_bs,shuffle=False,num_workers=num_workers,pin_memory=pin_mem)
    hno_model = HNO(hno_hdim, hno_k); hno_ckpt = os.path.join(ckpt_dir, "hno_checkpoint.pth")
    hno_model = train_hno_model(hno_model,load_tr_hno,load_te_hno,bb_t,sc_t,hno_ep,hno_lr,hno_ckpt,hno_si,device,logger)
    hno_model.eval()

    # --- Stage 3: Decoder Input Data Prep ---
    logger.info("--- Stage 3: Decoder2 Input Dataset ---")
    dec_in_dset = []; infer_load = DataLoader(full_dset,hno_bs*2,shuffle=False,num_workers=num_workers,pin_memory=pin_mem)
    with torch.no_grad():
        for batch_idx, batch in enumerate(infer_load): # Use enumerate
            batch=batch.to(device); emb=hno_model.forward_representation(batch.x,batch.edge_index)
            y_batch = batch.y
            # Check for None y_batch - robust check
            if y_batch is None:
                logger.error(f"batch.y is None at batch index {batch_idx}! Skipping batch.")
                continue
            counts = batch.ptr[1:]-batch.ptr[:-1]
            # Validate counts before splitting
            if counts.numel() > 0 and counts.sum() == emb.shape[0] and counts.sum() == y_batch.shape[0]:
                emb_l=torch.split(emb,counts.tolist()); y_l=torch.split(y_batch,counts.tolist())
                if len(emb_l) == len(y_l): # Ensure lists match length
                   for i in range(len(emb_l)): dec_in_dset.append(Data(x=emb_l[i].cpu(), y=y_l[i].cpu()))
                else: logger.error(f"Split list length mismatch batch {batch_idx}")
            elif counts.sum() != emb.shape[0] or counts.sum() != y_batch.shape[0]:
                 logger.error(f"Node count mismatch batch {batch_idx}: CountsSum={counts.sum()}, Emb={emb.shape[0]}, Y={y_batch.shape[0]}")
            # Handle case where counts might be empty if batch is empty (less likely with drop_last=True)
            elif counts.numel() == 0 and emb.shape[0] == 0 and y_batch.shape[0] == 0:
                 pass # Empty batch, do nothing
            else: logger.error(f"Unhandled count/shape mismatch batch {batch_idx}")

    logger.info(f"Decoder2 input dataset size: {len(dec_in_dset)}")
    if not dec_in_dset: logger.error("Decoder input dataset is empty! Cannot proceed."); sys.exit(1)


    # --- Stage 4: Decoder2 Setup & Training ---
    logger.info("--- Stage 4: Decoder2 Setup & Training ---")
    tr_dec, te_dec = train_test_split(dec_in_dset, test_size=0.1, random_state=42)
    load_tr_dec = DataLoader(tr_dec,dec2_bs,shuffle=True,num_workers=num_workers,pin_memory=pin_mem,drop_last=True)
    load_te_dec = DataLoader(te_dec,dec2_bs,shuffle=False,num_workers=num_workers,pin_memory=pin_mem)

    # Determine Conditioner
    ff_data=full_dset[0].to(device); Xref_dev=ff_data.x; cond_dim=-1; cond_cpu=None; zref_cpu=None
    if d2_c_mode=="X_ref": cond_cpu=Xref_dev.cpu(); cond_dim=3; logger.info("Using X_ref conditioner")
    elif d2_c_mode=="z_ref":
        with torch.no_grad(): zref_dev = hno_model.forward_representation(Xref_dev, ff_data.edge_index.to(device))
        cond_cpu=zref_dev.cpu(); zref_cpu=cond_cpu; cond_dim=zref_dev.shape[1]; logger.info("Using z_ref conditioner")
    else: logger.error(f"Invalid conditioner: {d2_c_mode}"); sys.exit(1)
    try: torch.save(Xref_dev.cpu(), os.path.join(struct_dir,"X_ref_coords.pt")); logger.info("Saved X_ref.")
    except Exception as e: logger.error(f"Save X_ref failed: {e}")
    if zref_cpu is not None:
        try: torch.save(zref_cpu, os.path.join(latent_dir,"z_ref_embedding.pt")); logger.info("Saved z_ref.")
        except Exception as e: logger.error(f"Save z_ref failed: {e}")

    # Dihedral Precomputation (Concise Logging)
    logger.info("--- Attempting Dihedral Precomputation ---")
    di_info={}; mask_all=None; N_res=None; precomp_ok=False; angle_types_all=['phi','psi','chi1','chi2','chi3','chi4','chi5']
    if use_di_cfg:
        torsion_path_abs = os.path.abspath(torsion_p) # Use absolute path for check
        if os.path.isfile(torsion_path_abs):
            logger.info(f"Found torsion file: {torsion_path_abs}")
            try:
                with open(torsion_path_abs,"r") as f: t_info=json.load(f)
                idx_l,res_l = {n:[[] for _ in range(4)] for n in angle_types_all},{n:[] for n in angle_types_all}
                keys_s = sorted([int(k) for k in t_info.keys()]); N_res = len(keys_s)
                mask_l = [[False]*len(angle_types_all) for _ in range(N_res)]; skip_c=0
                for r_idx, r_id in enumerate(keys_s):
                    r_d=t_info.get(str(r_id),{}); t_a=r_d.get("torsion_atoms",{}); c_a=t_a.get("chi",{})
                    for type_idx, name in enumerate(angle_types_all):
                        indices = t_a.get(name) if name in ['phi','psi'] else c_a.get(name)
                        if isinstance(indices, list) and len(indices)==4 and all(isinstance(i,int) for i in indices):
                             if all(0<=i<N_atoms for i in indices): # Check bounds
                                 for k, atom_idx in enumerate(indices): idx_l[name][k].append(atom_idx)
                                 res_l[name].append(r_idx); mask_l[r_idx][type_idx] = True
                             else: skip_c+=1
                if skip_c > 0: logger.warning(f"Skipped {skip_c} angles (out-of-bounds atom indices).")
                try: # Convert lists
                    for name in angle_types_all:
                        if res_l[name]: di_info[name] = {'indices': [torch.tensor(l,dtype=torch.long) for l in idx_l[name]], 'res_idx': torch.tensor(res_l[name],dtype=torch.long)}
                        else: di_info[name] = {'indices': None, 'res_idx': None}
                    mask_all=torch.tensor(mask_l, dtype=torch.bool); precomp_ok=True
                    logger.info(f"Successfully precomputed dihedral info for {N_res} residues.")
                except Exception as e: logger.error(f"Dihedral tensor conversion failed: {e}", exc_info=True)
            except Exception as e: logger.error(f"Dihedral JSON loading/processing failed: {e}", exc_info=True)
        else: logger.warning(f"Torsion file not found at resolved path: {torsion_path_abs}")
    use_di_train = use_di_cfg and precomp_ok # Final decision flag
    if use_di_cfg and not precomp_ok: logger.warning("Dihedral requested but precomp failed. Disabling loss.")
    logger.info(f"Final dihedral loss status for training: {'ENABLED' if use_di_train else 'DISABLED'}")

    # Initialize Decoder2
    res_idx_pool=None # Placeholder
    if d2_p_type == "residue": logger.warning("Residue pooling needs index generation from renum_d.")
    sec_pool = (d2_ph2, d2_pw2) if d2_use_p2 and d2_ph2 and d2_pw2 else None
    dec2_model = ProteinStateReconstructor2D(hno_hdim,N_atoms,cond_dim,d2_p_type,res_idx_pool,(d2_ph,d2_pw),d2_mlp_h,d2_mlp_l,sec_pool,d2_use_p2,d2_use_a,d2_a_type,logger)
    dec2_ckpt = os.path.join(ckpt_dir, "decoder2_checkpoint.pth")
    dec2_model = train_decoder2_model(dec2_model,load_tr_dec,load_te_dec,bb_t,sc_t,cond_cpu,N_atoms,dec2_ep,dec2_lr,dec2_ckpt,dec2_si,device,logger,dec2_base_w,use_di_train,di_info,mask_all,N_res,l_div,l_mse,div_t)
    dec2_model.eval()

    # --- Export Results ---
    export_final_outputs(hno_model,dec2_model,full_dset,dec_in_dset,cond_cpu,N_atoms,struct_dir,latent_dir,device,logger)

    elapsed = time.time() - start_time
    logger.info(f"================ Script Finished ({time.strftime('%H:%M:%S', time.gmtime(elapsed))}) ================")
    sys.stdout.flush()


################################################################################
# (J) Script Entry Point
################################################################################
if __name__ == "__main__":
    main()
