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
import random
import mdtraj as md
from typing import Dict, List, Optional, Tuple, Any

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import ChebConv, global_mean_pool
from torch_cluster import knn_graph
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors # New dependency

# (NEW) Local flag to control output verbosity
DISABLE_DETAILED_OUTPUTS = True # Set to True to skip non-essential file exports


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
    chain_map = {}; chain_idx_counter = 0
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
                chain_id_char = line[21].strip()
                if chain_id_char not in chain_map:
                    chain_map[chain_id_char] = chain_idx_counter
                    chain_idx_counter += 1
                chain_id_int = chain_map[chain_id_char]

                res_seq = int(line[22:26])
                orig_res_id = f"{chain_id_int}:{res_name}:{res_seq}"
                category = "backbone" if atom_name in backbone_atoms else "sidechain"

                atoms_in_order.append((orig_res_id, atom_serial, category, atom_name))
                if atom_name == 'CA': ca_indices[res_seq] = atom_serial

    except FileNotFoundError: logger.error(f"PDB not found: {filename}"); return {}, [], {}
    logger.info(f"Parsed {len(atoms_in_order)} ATOM records from {filename}, found {len(ca_indices)} C-alphas.")
    return {}, atoms_in_order, ca_indices

def renumber_atoms_and_residues(atoms_in_order: List[Tuple[str, int, str, str]], ca_serial_indices: Dict) -> Tuple[Dict, Dict, Dict, List[int], Dict]:
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
    return new_res_dict, orig_atom_map, {}, new_ca_indices, orig_res_map


def get_global_indices(renumbered_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts sorted global lists of backbone and sidechain atom indices as tensors."""
    bb_idx, sc_idx = [], []
    for res_id in sorted(renumbered_dict.keys()):
        bb_idx.extend(renumbered_dict[res_id]["backbone"])
        sc_idx.extend(renumbered_dict[res_id]["sidechain"])
    return torch.tensor(bb_idx, dtype=torch.long), torch.tensor(sc_idx, dtype=torch.long)

# --- JSON Loading ---
def load_heavy_atom_coords_from_json(json_file: str, logger: logging.Logger, max_frames: Optional[int] = None) -> Tuple[List[torch.Tensor], int]:
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
        n_frames_total = len(frame_data)
        if n_frames_total == 0: logger.warning("JSON contains 0 frames."); return [], 0

        frame_indices_to_load = range(n_frames_total)
        if max_frames is not None and max_frames > 0 and max_frames < n_frames_total:
            logger.info(f"Randomly sampling {max_frames} frames out of {n_frames_total} available.")
            frame_indices_to_load = sorted(random.sample(range(n_frames_total), max_frames))
        else:
            logger.info(f"Loading all {n_frames_total} available frames.")

        coords_frames, n_atoms_check = [], -1
        for frame_idx in frame_indices_to_load:
            frame_coords_np = []
            current_atoms = 0
            for res_key in keys_str:
                coords = np.array(data[res_key]["heavy_atom_coords_per_frame"][frame_idx], dtype=np.float32)
                if coords.ndim != 2 or coords.shape[1] != 3: raise ValueError("Bad coordinate shape")
                frame_coords_np.append(coords)
                current_atoms += coords.shape[0]

            if n_atoms_check == -1:
                n_atoms_check = current_atoms
                logger.info(f"System has {n_atoms_check} atoms and {len(frame_indices_to_load)} frames will be loaded.")
            elif current_atoms != n_atoms_check:
                logger.error(f"Inconsistent atom count on frame {frame_idx}. Expected {n_atoms_check}, got {current_atoms}."); return [], -1
            
            coords_frames.append(torch.tensor(np.concatenate(frame_coords_np, axis=0), dtype=torch.float32))

        return coords_frames, n_atoms_check
    except Exception as e:
        logger.error(f"Invalid JSON structure in {json_file}: {e}", exc_info=True); return [], -1

# --- Alignment ---
def compute_centroid(X: torch.Tensor) -> torch.Tensor: return X.mean(dim=-2)

def kabsch_algorithm(P: torch.Tensor, Q: torch.Tensor, logger: logging.Logger) -> Tuple[torch.Tensor, torch.Tensor]:
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

# --- ROBUST ALIGNMENT UTILITIES for MULTI-SYSTEM ---
def find_mutual_nn_pairs(ref_coords_ca: np.ndarray, target_coords_ca: np.ndarray, logger: logging.Logger) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finds mutually nearest C-alpha atoms between two structures of different lengths."""
    logger.debug(f"Finding mutual NN pairs between structures of size {len(ref_coords_ca)} and {len(target_coords_ca)}")
    if ref_coords_ca.ndim != 2 or target_coords_ca.ndim != 2:
        raise ValueError("Input coordinates must be 2D arrays.")

    nn_ref_to_target = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_coords_ca)
    _, indices1 = nn_ref_to_target.kneighbors(ref_coords_ca)
    
    nn_target_to_ref = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(ref_coords_ca)
    _, indices2 = nn_target_to_ref.kneighbors(target_coords_ca)
    
    ref_indices, target_indices = [], []
    for i, target_idx in enumerate(indices1.flatten()):
        # *** PATCH 1: Correctly index the result from kneighbors ***
        if indices2[target_idx, 0] == i:
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

    P_core = reference_structure[core_indices_reference].to(device)
    Q_core = structure_to_align[core_indices_to_align].to(device)
    
    rotation, _ = kabsch_algorithm(P_core, Q_core, logger)
    
    centroid_Q_core = compute_centroid(Q_core)
    structure_to_align_centered = structure_to_align - centroid_Q_core
    
    structure_to_align_rotated = torch.matmul(structure_to_align_centered, rotation.squeeze(0))
    
    centroid_P_core = compute_centroid(P_core)
    aligned_structure = structure_to_align_rotated + centroid_P_core
    
    return aligned_structure

# --- Graph Dataset ---
def build_graph_dataset(aligned_coords_list: List[torch.Tensor], unaligned_coords_list: List[torch.Tensor], knn_neighbors: int, system_id: int, logger: logging.Logger, device: torch.device) -> List[Data]:
    dataset = []
    n_frames = len(aligned_coords_list)
    logger.debug(f"[Graph Build] Building dataset for system {system_id} with {n_frames} frames. First frame coord shape: {aligned_coords_list[0].shape}")
    for i, (aligned_coords_cpu, unaligned_coords_cpu) in enumerate(zip(aligned_coords_list, unaligned_coords_list)):
        coords_dev = aligned_coords_cpu.to(device)
        edge_idx = knn_graph(coords_dev, k=knn_neighbors, loop=False, batch=None)
        if logger.isEnabledFor(logging.DEBUG) and i == 0:
            logger.debug(f"[Graph Build] Frame 0: coords_cpu shape: {aligned_coords_cpu.shape}, edge_index shape: {edge_idx.shape}")
        data = Data(x=aligned_coords_cpu, edge_index=edge_idx.cpu(), y=aligned_coords_cpu, y_unaligned=unaligned_coords_cpu, system_id=torch.tensor([system_id], dtype=torch.long))
        dataset.append(data)
    logger.info(f"Built graph dataset for system {system_id} with {n_frames} frames.")
    return dataset

# --- Dihedral Utilities ---
@torch.jit.script
def compute_dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    b1=b-a; b2=c-b; b3=d-c; n1=torch.cross(b1,b2,dim=-1); n2=torch.cross(b2,b3,dim=-1)
    n1n=F.normalize(n1,p=2.,dim=-1,eps=1e-8); n2n=F.normalize(n2,p=2.,dim=-1,eps=1e-8)
    b2n=F.normalize(b2,p=2.,dim=-1,eps=1e-8); m1=torch.cross(n1n, b2n, dim=-1)
    x=(n1n*n2n).sum(dim=-1); y=(m1*n2n).sum(dim=-1); return torch.atan2(y,x)

def compute_all_dihedrals_vectorized(coords: torch.Tensor, info: Dict, n_res: int, logger: logging.Logger) -> Dict:
    if coords.ndim != 3: raise ValueError(f"Expected coords [B, N, 3], got {coords.shape}")
    B, N_atoms, _ = coords.shape; dev = coords.device; all_angles = {}
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[Dihedral Calc] Input coords shape: {coords.shape}")
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
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[Dihedral Calc] Angle '{name}': computed values shape: {values.shape}, scatter indices shape: {res_idx_dev.shape}")
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

# --- Data Augmentation Utilities ---
def calculate_ca_rmsd(coords1: torch.Tensor, coords2: torch.Tensor, ca_indices: torch.Tensor) -> torch.Tensor:
    """Calculates C-alpha RMSD between two coordinate tensors."""
    ca_coords1 = coords1[ca_indices]
    ca_coords2 = coords2[ca_indices]
    return torch.sqrt(torch.mean((ca_coords1 - ca_coords2) ** 2))

def find_optimal_sigma(X_ref: torch.Tensor, ca_indices: torch.Tensor, target_rmsd: float, logger: logging.Logger, 
                       initial_sigma: float = 0.5, tolerance: float = 0.1, max_iter: int = 10) -> float:
    """
    Finds the optimal sigma for noise generation to meet a target C-alpha RMSD
    by directly comparing the noisy structure to the reference, without model inference.
    """
    sigma = initial_sigma
    X_ref_ca = X_ref[ca_indices].cpu() # Work with C-alpha atoms on CPU for simplicity

    logger.info(f"[Sigma Calibration] Starting calibration for target RMSD {target_rmsd:.3f} Å")
    for i in range(max_iter):
        # Generate one noisy sample by adding noise to a CLONE of the reference
        noise = torch.randn_like(X_ref) * sigma
        X_noisy = X_ref.clone() + noise
        X_noisy_ca = X_noisy[ca_indices].cpu()
        
        # Calculate C-alpha RMSD against the original, unmodified reference
        current_rmsd = torch.sqrt(torch.mean((X_ref_ca - X_noisy_ca) ** 2)).item()
        
        logger.debug(f"[Sigma Calibration] Iter {i+1}/{max_iter}: sigma={sigma:.4f}, current_rmsd={current_rmsd:.4f} Å")

        if abs(current_rmsd - target_rmsd) < tolerance:
            logger.info(f"[Sigma Calibration] Success! Final sigma={sigma:.4f} yields RMSD={current_rmsd:.4f} Å (within tolerance {tolerance:.3f} Å)")
            return sigma
        
        # Adjust sigma proportionally
        # Add a small epsilon to avoid division by zero if current_rmsd is 0
        sigma = sigma * (target_rmsd / (current_rmsd + 1e-6))

    logger.warning(f"[Sigma Calibration] Failed to converge within {max_iter} iterations. Using final sigma={sigma:.4f} (yielded RMSD={current_rmsd:.4f} Å)")
    return sigma

def run_dssp_analysis(pdb_path: str, orig_res_map: Dict[str, int], logger: logging.Logger) -> Dict[int, str]:
    """
    Runs DSSP analysis on a PDB file and maps the results to 0-based residue indices.
    """
    try:
        traj = md.load(pdb_path)
        ss_raw = md.compute_dssp(traj, simplified=True)
        ss_first_frame = ss_raw[0]
        
        ss_map = {}
        # MDTraj provides 1-based residue indices in its topology
        for res_md in traj.topology.residues:
            # res_md.resSeq is the original 1-based residue number from the PDB
            # res_md.index is the 0-based index within mdtraj
            
            # Construct the original residue ID string using the integer chain index to match parse_pdb
            orig_res_id = f"{res_md.chain.index}:{res_md.name}:{res_md.resSeq}"
            
            if orig_res_id in orig_res_map:
                internal_res_idx = orig_res_map[orig_res_id]
                ss_code = ss_first_frame[res_md.index]
                simplified_code = 'H' if ss_code == 'H' else 'L'
                ss_map[internal_res_idx] = simplified_code
                
        logger.info(f"Successfully ran DSSP and mapped {len(ss_map)} residues.")
        return ss_map
    except Exception as e:
        logger.error(f"Failed to run DSSP analysis on {pdb_path}: {e}", exc_info=True)
        return {}

def get_deletable_segments(ss_map: Dict[int, str], total_residues: int, logger: logging.Logger) -> List[List[int]]:
    """
    Identifies contiguous segments of residues that are candidates for deletion.
    A segment is a candidate if it is a loop ('L') or a helix ('H') adjacent to a loop.
    Excludes segments containing the first two or last two residues.
    """
    if not ss_map:
        logger.warning("Secondary structure map is empty. No deletable segments can be identified.")
        return []

    # Identify all loop residues
    loop_residues = {res_idx for res_idx, code in ss_map.items() if code == 'L'}
    
    deletable_residues = set()
    for res_idx, code in ss_map.items():
        # Rule: A residue is deletable if it's a loop
        if code == 'L':
            deletable_residues.add(res_idx)
        # Rule: A residue is deletable if it's a helix adjacent to a loop
        elif code == 'H':
            is_adjacent_to_loop = (res_idx - 1 in loop_residues) or (res_idx + 1 in loop_residues)
            if is_adjacent_to_loop:
                deletable_residues.add(res_idx)

    # Safety Rule: Exclude the first two and last two residues of the entire chain
    protected_residues = {0, 1, total_residues - 2, total_residues - 1}
    eligible_residues = sorted(list(deletable_residues - protected_residues))
    
    if not eligible_residues:
        logger.warning("No eligible residues for deletion after applying safety rules.")
        return []

    # Group eligible residues into contiguous segments
    segments = []
    if eligible_residues:
        current_segment = [eligible_residues[0]]
        for i in range(1, len(eligible_residues)):
            if eligible_residues[i] == eligible_residues[i-1] + 1:
                current_segment.append(eligible_residues[i])
            else:
                segments.append(current_segment)
                current_segment = [eligible_residues[i]]
        segments.append(current_segment)
        
    logger.info(f"Identified {len(segments)} deletable segments from {len(eligible_residues)} eligible residues.")
    return segments

# --- Checkpoint Utilities & MSE Utilities ---
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
class HNO(nn.Module):
    def __init__(self, hidden_dim, K):
        super().__init__()
        self._debug_logged_train = False
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
        x_in = x
        x = self.bano1(F.leaky_relu(self.conv1(x, edge_index)))
        x = self.bano2(F.leaky_relu(self.conv2(x, edge_index)))
        x = self.bano3(F.relu(self.conv3(x, edge_index)))
        x = self.conv4(x, edge_index)
        x_rep = F.normalize(x, p=2.0, dim=1)
        x_recon = self.mlpRep(x_rep)
        
        if self.training and not self._debug_logged_train and logger.isEnabledFor(logging.DEBUG):
             logger.debug(f"[HNO Train Fwd] Input shape: {x_in.shape}, Edge index shape: {edge_index.shape}")
             logger.debug(f"[HNO Train Fwd] Output shapes: Rep={x_rep.shape}, Recon={x_recon.shape}")
             self._debug_logged_train = True
        return x_recon

    def forward_representation(self, x, edge_index):
        x = x.float()
        x = self.bano1(F.leaky_relu(self.conv1(x, edge_index)))
        x = self.bano2(F.leaky_relu(self.conv2(x, edge_index)))
        x = self.bano3(F.relu(self.conv3(x, edge_index)))
        x = self.conv4(x, edge_index)
        return F.normalize(x, p=2.0, dim=1)

# --- ROBUST Decoder2 Model (CORRECTED LOGIC) ---
class ProteinStateReconstructor2D(nn.Module):
    """
    Size-agnostic decoder. For each graph, it pools the dynamic embedding to a
    fixed size using AdaptiveAvgPool2d, flattens it, and concatenates this
    global dynamic vector to the static (z_ref) conditioner for each node.
    """
    def __init__(self, node_emb_dim: int, cond_emb_dim: int,
                 output_height: int, output_width: int,
                 mlp_h_dim: int, mlp_layers: int, logger: logging.Logger):
        super().__init__()
        self.logger = logger
        self._logged_fwd = False
        self.node_emb_dim = node_emb_dim

        # 1. Define the 2D pooling layer
        self.pool_layer = nn.AdaptiveAvgPool2d((output_height, output_width))
        pooled_dim = output_height * output_width

        # 2. Define the final MLP input dimension
        # It's the static conditioner + the flattened, pooled dynamic embedding
        mlp_in_dim = cond_emb_dim + pooled_dim

        # 3. Build the MLP
        layers = []
        in_d = mlp_in_dim
        for i in range(mlp_layers - 1):
            layers.extend([nn.Linear(in_d, mlp_h_dim), nn.BatchNorm1d(mlp_h_dim), nn.GELU()])
            in_d = mlp_h_dim
        layers.append(nn.Linear(in_d, 3))
        self.decoder_mlp = nn.Sequential(*layers)

        self.logger.info(f"Initialized Decoder2 (Corrected). Pool size: ({output_height}, {output_width}). MLP Input Dim: {mlp_in_dim}")

    def forward(self, x: torch.Tensor, batch: torch.Tensor, conditioner_z_ref: torch.Tensor) -> torch.Tensor:
        if not self._logged_fwd and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"[Decoder Fwd Start] Input x shape: {x.shape}, conditioner_z_ref shape: {conditioner_z_ref.shape}")

        # Since graphs in the batch have different sizes, we must loop.
        pooled_vectors = []
        for i in range(batch.max().item() + 1):
            # a. Get the dynamic embedding for the current graph
            graph_mask = (batch == i)
            x_graph = x[graph_mask] # Shape: [n_nodes_in_graph, node_emb_dim]
            n_nodes, n_emb = x_graph.shape

            # b. Reshape for 2D pooling: [B, C, H, W] -> [1, 1, n_nodes, n_emb]
            x_graph_4d = x_graph.unsqueeze(0).unsqueeze(0)

            # c. Apply pooling and flatten
            pooled_graph = self.pool_layer(x_graph_4d)
            flattened_pooled = pooled_graph.view(1, -1) # Shape: [1, pooled_dim]
            pooled_vectors.append(flattened_pooled)

        # d. Combine pooled vectors for the whole batch
        batch_pooled = torch.cat(pooled_vectors, dim=0) # Shape: [num_graphs, pooled_dim]

        # e. Broadcast the correct pooled vector to each node in its respective graph
        pooled_per_node = batch_pooled[batch] # Shape: [total_nodes, pooled_dim]

        # f. Concatenate the static conditioner with the broadcasted pooled vector
        mlp_input = torch.cat([conditioner_z_ref, pooled_per_node], dim=1)

        # g. Predict coordinates
        pred_coords = self.decoder_mlp(mlp_input)

        if not self._logged_fwd and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"[Decoder Fwd Pools] Pooled vector shape (per graph): {pooled_vectors[0].shape}")
            self.logger.debug(f"[Decoder Fwd Pools] Broadcasted pooled shape: {pooled_per_node.shape}")
            self.logger.debug(f"[Decoder Fwd End] mlp_input shape: {mlp_input.shape}, Final output pred_coords shape: {pred_coords.shape}")
            self._logged_fwd = True

        return pred_coords

################################################################################
# (G) Training Functions
################################################################################

# --- Train HNO ---
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
        for i, data in enumerate(tr_loader):
            data=data.to(dev)
            if ep == start_ep and i == 0 and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[HNO Batching] Batch contains {data.num_graphs} graphs.")
                logger.debug(f"[HNO Batching] data.x shape: {data.x.shape}")
                logger.debug(f"[HNO Batching] data.ptr: {data.ptr}")
                logger.debug(f"[HNO Batching] Total nodes in batch: {data.num_nodes}")
                logger.debug(f"[HNO Batching] System IDs in batch: {data.system_id.squeeze().tolist()}")

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

# --- ROBUST Train Decoder2 ---
def train_decoder2_model(
    model: ProteinStateReconstructor2D, tr_loader: DataLoader, te_loader: DataLoader,
    all_conditioners: Dict[int, torch.Tensor],
    all_z_ref_ensembles: Dict[int, torch.Tensor],
    use_z_ref_ensemble: bool,
    all_sys_info: Dict[int, Any],
    N_epochs: int, lr: float, ckpt: str, save_int: int, dev: torch.device, logger: logging.Logger,
    base_w: float, use_di: bool, all_di_info: Optional[Dict], div_t: str, l_div: float, l_mse: float
):
    model = model.to(dev)
    opt = torch.optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=lr)
    model, opt, start_ep = load_checkpoint(model, opt, ckpt, dev, logger)

    if use_z_ref_ensemble:
        logger.info("Decoder training with augmented 'z_ref_ensemble' as conditioner.")
        if not all_z_ref_ensembles:
            logger.error("`use_z_ref_ensemble` is true, but the ensemble dictionary is empty. Aborting.")
            return model
    else:
        logger.info("Decoder training with clean 'z_ref' as conditioner.")

    if start_ep >= N_epochs:
        logger.info(f"Loaded Decoder2 checkpoint epoch ({start_ep}) >= target epochs ({N_epochs}). Skipping training.")
        return model
        
    comp_div = compute_angle_js_div if div_t == "JS" else (lambda p, t: torch.tensor(0.0))
    if use_di: logger.info(f"Dihedral loss enabled: type={div_t}, lambda_div={l_div}, lambda_mse={l_mse}")

    logger.info(f"Starting Decoder2 training from epoch {start_ep + 1}/{N_epochs}, LR={lr}")
    for ep in range(start_ep, N_epochs):
        model.train()
        tr_metrics = {'total': 0.0, 'coord': 0.0, 'di_div': 0.0, 'di_mse': 0.0}
        
        for i, data in enumerate(tr_loader):
            data = data.to(dev)
            opt.zero_grad(set_to_none=True)
            
            # --- Assemble Batch-Specific Conditioners ---
            cond_list = []
            for j in range(data.num_graphs):
                sid = data.system_id[j].item()
                num_nodes = data.ptr[j+1] - data.ptr[j]

                if use_z_ref_ensemble:
                    # Check if an ensemble exists for this specific system
                    if sid in all_z_ref_ensembles:
                        # If yes, sample from it (this is a base system with noise aug)
                        ensemble = all_z_ref_ensembles[sid]
                        rand_idx = torch.randint(0, ensemble.shape[0], (1,)).item()
                        cond = ensemble[rand_idx].to(dev)
                        if i == 0 and j < 4 and logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"[Decoder Verify] Graph {j} (SID: {sid}) is using RANDOM conditioner from its ensemble.")
                    else:
                        # If no, fall back to the single z_ref (this is a variant)
                        cond = all_conditioners[sid].to(dev)
                        if i == 0 and j < 4 and logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"[Decoder Verify] Graph {j} (SID: {sid}) is a variant, using FIXED conditioner.")
                else:
                    # Use the clean, single z_ref if ensemble mode is off entirely
                    cond = all_conditioners[sid].to(dev)
                    if i == 0 and j < 4 and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[Decoder Verify] Graph {j} (SID: {sid}) is using the FIXED conditioner (ensemble mode off).")

                if cond.shape[0] != num_nodes:
                    raise ValueError(f"FATAL: Mismatch in conditioner size for system {sid}. Expected {num_nodes}, got {cond.shape[0]}")
                cond_list.append(cond)
            conditioners_for_batch = torch.cat(cond_list, dim=0)

            # --- Forward Pass ---
            pred = model(data.x, data.batch, conditioners_for_batch)
            
            # --- ROBUST Loss Calculation (Graph by Graph) ---
            total_loss = torch.tensor(0.0, device=dev)
            for j in range(data.num_graphs):
                graph_start, graph_end = data.ptr[j], data.ptr[j+1]
                sid = data.system_id[j].item()

                pred_graph = pred[graph_start:graph_end]
                target_graph = data.y[graph_start:graph_end]
                
                if i == 0 and j < 2 and logger.isEnabledFor(logging.DEBUG): # Log first 2 graphs of first batch
                    logger.debug(f"[Decoder Loss Loop] Graph {j}, SID {sid}: pred_graph shape: {pred_graph.shape}, target_graph shape: {target_graph.shape}")

                # Coordinate Loss for this graph
                bb_idx = all_sys_info[sid]['bb_idx_local']
                sc_idx = all_sys_info[sid]['sc_idx_local']
                
                coord_mse_graph, _, _ = compute_bb_sc_mse(pred_graph, target_graph, bb_idx, sc_idx, logger)
                total_loss += base_w * coord_mse_graph
                tr_metrics['coord'] += coord_mse_graph.item()

                # Dihedral Loss for this graph
                if use_di and sid in all_di_info:
                    pred_3d = pred_graph.unsqueeze(0) # Add batch dim
                    target_3d = target_graph.unsqueeze(0)
                    
                    pred_a = compute_all_dihedrals_vectorized(pred_3d, all_di_info[sid], all_sys_info[sid]['n_res'], logger)
                    true_a = compute_all_dihedrals_vectorized(target_3d, all_di_info[sid], all_sys_info[sid]['n_res'], logger)
                    
                    di_div_loss_graph = torch.tensor(0.0, device=dev)
                    di_mse_loss_graph = torch.tensor(0.0, device=dev)
                    for name in pred_a:
                        pa, ta = pred_a[name], true_a[name]
                        if pa.numel() > 0:
                             di_div_loss_graph += comp_div(pa, ta)
                             di_mse_loss_graph += F.mse_loss(pa, ta)
                    
                    total_loss += l_div * di_div_loss_graph + l_mse * di_mse_loss_graph
                    tr_metrics['di_div'] += di_div_loss_graph.item()
                    tr_metrics['di_mse'] += di_mse_loss_graph.item()

            # Average loss over the number of graphs in the batch
            avg_batch_loss = total_loss / data.num_graphs
            tr_metrics['total'] += avg_batch_loss.item()
            
            avg_batch_loss.backward()
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
# (H) Final Data Export Function
################################################################################
@torch.no_grad()
def export_final_outputs_multi(
    hno: 'HNO',
    dec2: 'ProteinStateReconstructor2D',
    full_dset: List['Data'],
    dec_in_dset: List['Data'],
    all_conditioners: Dict[int, torch.Tensor],
    all_sys_info: Dict[int, Any],
    all_X_refs: Dict[int, torch.Tensor],
    all_X_ref_ensembles: Dict[int, torch.Tensor],
    all_z_ref_ensembles: Dict[int, torch.Tensor],
    struct_dir: str,
    latent_dir: str,
    dev: torch.device,
    logger: logging.Logger
):
    """
    Exports final ground truth, reconstructions, and embeddings for all systems
    to HDF5 files, with data for each system stored in a separate group.
    Also saves the reference conditioners (z_ref and X_ref) and their ensembles.
    """
    logger.info("--- Stage 5: Starting Final Data Export ---")
    hno.eval()
    dec2.eval()

    # 1. Initialize data storage
    outputs_by_sid = {
        sid: {'gt_aligned': [], 'gt_unaligned': [], 'hno_rec': [], 'full_rec': [], 'hno_emb': [], 'pooled_emb': []}
        for sid in all_sys_info.keys()
    }

    # 2. Loop through datasets and perform inference
    logger.info("Running inference on all frames...")
    full_loader = DataLoader(full_dset, batch_size=32, shuffle=False)
    dec_in_loader = DataLoader(dec_in_dset, batch_size=32, shuffle=False)

    # Store intermediate results to avoid recomputing embeddings
    temp_embs = {}

    with torch.no_grad():
        # First, get all HNO reconstructions and embeddings
        for i, data in enumerate(full_loader):
            data = data.to(dev)
            hno_recon = hno(data.x, data.edge_index)
            hno_embedding = hno.forward_representation(data.x, data.edge_index)
            
            for j in range(data.num_graphs):
                sid = data.system_id[j].item()
                start, end = data.ptr[j], data.ptr[j+1]
                
                outputs_by_sid[sid]['gt_aligned'].append(data.y[start:end].cpu())
                outputs_by_sid[sid]['gt_unaligned'].append(data.y_unaligned[start:end].cpu())
                outputs_by_sid[sid]['hno_rec'].append(hno_recon[start:end].cpu())
                outputs_by_sid[sid]['hno_emb'].append(hno_embedding[start:end].cpu())
                
                # Store for decoder pass
                if sid not in temp_embs: temp_embs[sid] = []
                temp_embs[sid].append(hno_embedding[start:end])


        # Second, get all Decoder reconstructions and pooled embeddings
        for i, emb_data in enumerate(dec_in_loader):
            emb_data = emb_data.to(dev)
            
            # Assemble conditioners for the batch
            cond_list = [all_conditioners[sid.item()].to(dev) for sid in emb_data.system_id]
            conditioner = torch.cat(cond_list, dim=0)

            # Decoder forward pass
            full_recon = dec2(emb_data.x, emb_data.batch, conditioner)

            # Pooled embedding logic (replicated from decoder's forward)
            pooled_vectors = []
            for j in range(emb_data.num_graphs):
                graph_mask = (emb_data.batch == j)
                x_graph = emb_data.x[graph_mask]
                
                x_graph_4d = x_graph.unsqueeze(0).unsqueeze(0)
                pooled_graph = dec2.pool_layer(x_graph_4d)
                flattened_pooled = pooled_graph.view(1, -1)
                pooled_vectors.append(flattened_pooled)
            
            batch_pooled = torch.cat(pooled_vectors, dim=0)

            # Split results and store
            for j in range(emb_data.num_graphs):
                sid = emb_data.system_id[j].item()
                start, end = emb_data.ptr[j], emb_data.ptr[j+1]
                
                outputs_by_sid[sid]['full_rec'].append(full_recon[start:end].cpu())
                outputs_by_sid[sid]['pooled_emb'].append(batch_pooled[j].cpu())


    # 3. Write stacked data to HDF5 files
    logger.info("Writing exported data to HDF5 files...")
    os.makedirs(struct_dir, exist_ok=True)
    os.makedirs(latent_dir, exist_ok=True)

    # Define output files
    coord_files = {
        'gt_aligned': os.path.join(struct_dir, 'gt_coords_aligned.h5'),
        'gt_unaligned': os.path.join(struct_dir, 'gt_coords_unaligned.h5'),
        'hno_rec': os.path.join(struct_dir, 'hno_reconstructed_coords.h5'),
        'full_rec': os.path.join(struct_dir, 'full_reconstructed_coords.h5')
    }
    emb_files = {
        'hno_emb': os.path.join(latent_dir, 'hno_embeddings.h5'),
        'pooled_emb': os.path.join(latent_dir, 'pooled_embeddings.h5')
    }

    # Write coordinate data
    if not DISABLE_DETAILED_OUTPUTS:
        for key, path in coord_files.items():
            with h5py.File(path, 'w') as f:
                logger.debug(f"Writing to {path}")
                for sid, data in outputs_by_sid.items():
                    if data[key]:
                        stacked_data = torch.stack(data[key]).numpy()
                        grp = f.create_group(f"system_{sid}")
                        grp.create_dataset("coords", data=stacked_data, compression="gzip")
                        logger.debug(f"  - Wrote system_{sid}/coords with shape {stacked_data.shape}")
    else:
        logger.info("Skipping export of all coordinate files in `structure_dir` due to local setting.")

    # Write embedding data
    for key, path in emb_files.items():
        if DISABLE_DETAILED_OUTPUTS and key == 'hno_emb':
            logger.info(f"Skipping export of {path} due to local setting.")
            continue
        with h5py.File(path, 'w') as f:
            logger.debug(f"Writing to {path}")
            for sid, data in outputs_by_sid.items():
                if data[key]:
                    stacked_data = torch.stack(data[key]).numpy()
                    grp = f.create_group(f"system_{sid}")
                    grp.create_dataset("embeddings", data=stacked_data, compression="gzip")
                    logger.debug(f"  - Wrote system_{sid}/embeddings with shape {stacked_data.shape}")

    # --- Modified logic to save reference conditioners and ensembles ---
    ref_cond_path = os.path.join(latent_dir, 'reference_conditioners.h5')
    logger.info(f"Writing reference conditioners and ensembles to {ref_cond_path}")
    with h5py.File(ref_cond_path, 'w') as f:
        for sid, z_ref in all_conditioners.items():
            if sid in all_X_refs:
                grp = f.create_group(f"system_{sid}")
                grp.create_dataset("z_ref", data=z_ref.numpy(), compression="gzip")
                grp.create_dataset("X_ref", data=all_X_refs[sid].numpy(), compression="gzip")
                logger.debug(f"  - Wrote system_{sid}/z_ref with shape {z_ref.shape}")
                logger.debug(f"  - Wrote system_{sid}/X_ref with shape {all_X_refs[sid].shape}")

                # Save ensembles if they exist for this system
                if sid in all_X_ref_ensembles and sid in all_z_ref_ensembles:
                    x_ensemble = all_X_ref_ensembles[sid].numpy()
                    z_ensemble = all_z_ref_ensembles[sid].numpy()
                    grp.create_dataset("X_ref_ensemble", data=x_ensemble, compression="gzip")
                    grp.create_dataset("z_ref_ensemble", data=z_ensemble, compression="gzip")
                    logger.debug(f"  - Wrote system_{sid}/X_ref_ensemble with shape {x_ensemble.shape}")
                    logger.debug(f"  - Wrote system_{sid}/z_ref_ensemble with shape {z_ensemble.shape}")
            else:
                logger.warning(f"Could not find X_ref for system {sid}. Skipping save for this system in reference_conditioners.h5")

    logger.info("--- Finished Final Data Export ---")


################################################################################
# (I) Main Execution Function
################################################################################
def main():
    start_time = time.time()
    logger.info("================ Script Starting (Multi-System Version) ================")
    global global_device
    
    with open(args.config, "r") as f: config = yaml.safe_load(f)
    logger.info("Successfully loaded configuration.")
    
    force_cpu = config.get("force_cpu", False)
    device = torch.device("cpu") if force_cpu else global_device
    pin_mem = (device.type == "cuda")
    num_workers = config.get("num_workers", 0)
    
    hno_cfg = config["hno_encoder"]
    dec2_cfg = config["decoder2"]
    d2s = config["decoder2_settings"]
    di_cfg = config.get("dihedral_loss", {})
    
    # Get structural augmentation config
    struct_aug_cfg = config.get("structural_augmentation", {"enabled": False})
    aug_cfg = config.get("data_augmentation", {}) # For noise-based augmentation

    out_cfg = config["output_directories"]
    [os.makedirs(d, exist_ok=True) for d in out_cfg.values()]

    logger.info("--- Stage 1: Data Loading & Preprocessing for All Systems ---")
    
    base_systems_data = []
    systems_to_process_configs = config["data"]["systems"]
    
    exec_settings = config.get("execution_settings", {})
    limit_systems = exec_settings.get("limit_systems")
    max_frames = exec_settings.get("max_frames_per_system")

    if limit_systems is not None and isinstance(limit_systems, int) and limit_systems > 0:
        systems_to_process_configs = systems_to_process_configs[:limit_systems]
        logger.info(f"[INFO] Limiting run to the first {limit_systems} systems as specified in the config.")

    for i, system_config in enumerate(systems_to_process_configs):
        sid = i
        pdb_p = system_config["pdb_path"]
        json_p = system_config["json_path"]
        logger.info(f"--- Loading Base System ID {sid}: {os.path.basename(pdb_p)} ---")

        _, atoms_ord, ca_serial_map = parse_pdb(pdb_p, logger)
        coords_list, n_atoms = load_heavy_atom_coords_from_json(json_p, logger, max_frames=max_frames)
        if not coords_list:
            logger.warning(f"Skipping system {sid} due to no coordinates.")
            continue
        
        base_systems_data.append({
            "pdb_path": pdb_p,
            "atoms_ord": atoms_ord,
            "ca_serial_map": ca_serial_map,
            "coords_list": coords_list,
            "is_variant": False,
            "base_sid": sid
        })

    # --- NEW Stage 1.5: Structural Augmentation ---
    final_systems_to_process = list(base_systems_data) # Start with original systems

    if struct_aug_cfg.get("enabled", False):
        logger.info("--- Stage 1.5: Starting Structural Augmentation ---")
        variants_per_system = struct_aug_cfg.get("variants_per_system", 5)
        
        for base_system_data in base_systems_data:
            base_sid = base_system_data["base_sid"]
            pdb_path = base_system_data["pdb_path"]
            logger.info(f"--- Generating variants for Base System ID {base_sid} ---")

            # 1. Get mappings for the original, undamaged system
            renum_d_orig, _, _, _, orig_res_map = renumber_atoms_and_residues(
                base_system_data["atoms_ord"], base_system_data["ca_serial_map"]
            )
            total_residues = len(renum_d_orig)

            # 2. Perform SS analysis
            ss_map = run_dssp_analysis(pdb_path, orig_res_map, logger)
            if not ss_map:
                logger.warning(f"Cannot generate variants for SID {base_sid} due to DSSP failure.")
                continue

            # 3. Identify deletable segments
            deletable_segments = get_deletable_segments(ss_map, total_residues, logger)
            if not deletable_segments:
                logger.warning(f"No deletable segments found for SID {base_sid}. Skipping variant generation.")
                continue

            # 4. Generate damaged variants
            for i in range(variants_per_system):
                logger.debug(f"Creating variant {i+1}/{variants_per_system} for SID {base_sid}")
                
                # a. Randomly select a segment and deletion length
                segment_to_delete_from = random.choice(deletable_segments)
                max_len = min(len(segment_to_delete_from), 6)
                if max_len < 2: continue # Skip if the segment is too short
                
                deletion_len = random.randint(2, max_len)
                start_idx_in_segment = random.randint(0, len(segment_to_delete_from) - deletion_len)
                
                residues_to_delete = set(segment_to_delete_from[start_idx_in_segment : start_idx_in_segment + deletion_len])
                
                # b. Identify all atoms belonging to the residues to be deleted
                atom_indices_to_delete = set()
                for res_idx in residues_to_delete:
                    atom_indices_to_delete.update(renum_d_orig[res_idx]["backbone"])
                    atom_indices_to_delete.update(renum_d_orig[res_idx]["sidechain"])

                # c. Create new damaged data
                original_atoms_ord = base_system_data["atoms_ord"]
                
                # We need a map from the original 0-based index to the atom tuple
                idx_to_atom_tuple = {idx: atom_tuple for idx, atom_tuple in enumerate(original_atoms_ord)}

                atoms_ord_damaged = [
                    atom_tuple for idx, atom_tuple in idx_to_atom_tuple.items() 
                    if idx not in atom_indices_to_delete
                ]

                coords_list_damaged = []
                original_coords_list = base_system_data["coords_list"]
                
                # Create a keep_mask for atom indices
                keep_indices = sorted(list(set(range(len(original_atoms_ord))) - atom_indices_to_delete))
                keep_mask = torch.tensor(keep_indices, dtype=torch.long)

                for frame_coords in original_coords_list:
                    coords_list_damaged.append(frame_coords[keep_mask])

                final_systems_to_process.append({
                    "pdb_path": pdb_path, # For reference, not reparsing
                    "atoms_ord": atoms_ord_damaged,
                    "ca_serial_map": base_system_data["ca_serial_map"], # Will be filtered by renumbering
                    "coords_list": coords_list_damaged,
                    "is_variant": True,
                    "base_sid": base_sid
                })
        logger.info(f"Finished augmentation. Total systems to process: {len(final_systems_to_process)}")


    # --- Stage 2: Final Data Processing and Graph Building ---
    full_dset = []
    all_sys_info = {}
    all_X_refs = {}
    canonical_ref_coords = None
    canonical_ref_ca_indices = None
    
    # This loop now processes both original and variant systems
    for i, system_data in enumerate(final_systems_to_process):
        sid = i # Each system, original or variant, gets a new unique ID
        
        log_prefix = f"Variant of {system_data['base_sid']}" if system_data['is_variant'] else f"Base System {system_data['base_sid']}"
        logger.info(f"--- Processing Final System ID {sid} ({log_prefix}) ---")

        # CRUCIAL: Re-run renumbering and indexing for every system
        renum_d, _, _, ca_indices_new, _ = renumber_atoms_and_residues(system_data["atoms_ord"], system_data["ca_serial_map"])
        bb_idx, sc_idx = get_global_indices(renum_d)
        
        coords_list = system_data["coords_list"]
        n_atoms = coords_list[0].shape[0]
        unaligned_coords_list = [c.clone() for c in coords_list]
        
        all_sys_info[sid] = {
            'n_atoms': n_atoms,
            'ca_indices': torch.tensor(ca_indices_new, dtype=torch.long),
            'bb_idx_local': bb_idx,
            'sc_idx_local': sc_idx,
            'n_res': len(renum_d),
            'is_variant': system_data['is_variant'],
            'base_sid_ref': system_data['base_sid']
        }

        aligned_coords_list = []
        if sid == 0: # The first system is always the canonical reference
            logger.info(f"System {sid} is the canonical reference.")
            canonical_ref_coords = coords_list[0].clone()
            all_X_refs[sid] = canonical_ref_coords.clone()
            canonical_ref_ca_indices = torch.tensor(ca_indices_new, dtype=torch.long)
            aligned_coords_list = align_frames_to_first(coords_list, logger, device)
        else:
            logger.info(f"Aligning system {sid} to canonical reference (system 0).")
            target_ref_coords = coords_list[0]
            target_ca_indices = torch.tensor(ca_indices_new, dtype=torch.long)
            
            core_ref_idx, core_target_idx = find_mutual_nn_pairs(
                canonical_ref_coords[canonical_ref_ca_indices].numpy(),
                target_ref_coords[target_ca_indices].numpy(),
                logger
            )
            
            aligned_target_ref_frame = align_by_core(
                target_ref_coords, target_ca_indices[core_target_idx],
                canonical_ref_coords, canonical_ref_ca_indices[core_ref_idx],
                logger
            )
            
            all_X_refs[sid] = aligned_target_ref_frame.cpu().clone()
            temp_list_for_align = [aligned_target_ref_frame.cpu()] + [c.to(device) for c in coords_list[1:]]
            aligned_coords_list = align_frames_to_first(temp_list_for_align, logger, device)

        system_dset = build_graph_dataset(aligned_coords_list, unaligned_coords_list, config["graph"]["knn_value"], sid, logger, device)
        full_dset.extend(system_dset)

    logger.info(f"--- Finished data processing. Total frames in dataset: {len(full_dset)} ---")

    # --- Stage 3: Training Shared HNO Encoder ---
    tr_hno, te_hno = train_test_split(full_dset, test_size=0.1, random_state=42)
    load_tr_hno = DataLoader(tr_hno, hno_cfg['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    load_te_hno = DataLoader(te_hno, hno_cfg['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    hno_model = HNO(hno_cfg['hidden_dim'], hno_cfg['cheb_order'])
    hno_ckpt = os.path.join(out_cfg['checkpoint_dir'], "hno_checkpoint.pth")
    hno_model = train_hno_model(hno_model, load_tr_hno, load_te_hno, hno_cfg['num_epochs'], hno_cfg['learning_rate'], hno_ckpt, hno_cfg['save_interval'], device, logger)
    hno_model.eval()

    # --- Stage 4: Preparing Decoder Input Dataset & Conditioners ---
    dec_in_dset = []
    all_conditioners = {}
    all_X_ref_ensembles = {} # Legacy from noise augmentation, can be removed if not used
    all_z_ref_ensembles = {} # Legacy from noise augmentation, can be removed if not used
    
    with torch.no_grad():
        infer_load = DataLoader(full_dset, hno_cfg['batch_size'] * 2, shuffle=False)
        for batch in infer_load:
            batch = batch.to(device)
            emb = hno_model.forward_representation(batch.x, batch.edge_index)
            split_sizes = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
            emb_list = torch.split(emb, split_sizes)
            y_list = torch.split(batch.y, split_sizes)
            
            for j in range(len(emb_list)):
                dec_in_dset.append(Data(x=emb_list[j].cpu(), y=y_list[j].cpu(), system_id=batch.system_id[j].cpu().reshape(1)))
    
    if d2s['conditioner_mode'] == 'z_ref':
        for sid in all_sys_info.keys():
            ref_coords = all_X_refs[sid].to(device)
            edge_index_ref = knn_graph(ref_coords, k=config["graph"]["knn_value"], loop=False, batch=None)
            with torch.no_grad():
                z_ref = hno_model.forward_representation(ref_coords, edge_index_ref)
                all_conditioners[sid] = z_ref.cpu()
        logger.info(f"Created 'z_ref' conditioners for all {len(all_conditioners)} systems.")
    else:
        raise NotImplementedError("Only 'z_ref' conditioner mode is supported in this version.")

    # --- Stage 4.5: Noise-based Data Augmentation (applied to all systems) ---
    if aug_cfg.get("enabled", False):
        logger.info("--- Stage 4.5: Starting Noise-Based Data Augmentation for ALL systems (base and variants) ---")
        ensemble_size = aug_cfg.get("ensemble_size", 100)
        target_rmsd = aug_cfg.get("target_ca_rmsd", 1.0)
        
        for sid, X_ref in all_X_refs.items():
            logger.info(f"--- Augmenting data with noise for System ID {sid} ---")
            ca_indices = all_sys_info[sid]['ca_indices']
            
            optimal_sigma = find_optimal_sigma(
                X_ref, ca_indices, target_rmsd, logger
            )

            X_ref_ensemble_list = []
            for _ in range(ensemble_size):
                noise = torch.randn_like(X_ref) * optimal_sigma
                X_noisy = X_ref.clone() + noise
                X_ref_ensemble_list.append(X_noisy)
            
            X_ref_ensemble_tensor = torch.stack(X_ref_ensemble_list)
            all_X_ref_ensembles[sid] = X_ref_ensemble_tensor.cpu()
            logger.info(f"Generated X_ref_ensemble for SID {sid} with shape {X_ref_ensemble_tensor.shape}")

            z_ref_ensemble_list = []
            with torch.no_grad():
                for i in range(ensemble_size):
                    noisy_struct = X_ref_ensemble_tensor[i].to(device)
                    edge_index_noisy = knn_graph(noisy_struct, k=config["graph"]["knn_value"], loop=False, batch=None)
                    z_ref_single = hno_model.forward_representation(noisy_struct, edge_index_noisy)
                    z_ref_ensemble_list.append(z_ref_single.cpu())

            z_ref_ensemble_tensor = torch.stack(z_ref_ensemble_list)
            all_z_ref_ensembles[sid] = z_ref_ensemble_tensor
            logger.info(f"Generated z_ref_ensemble for SID {sid} with shape {z_ref_ensemble_tensor.shape}")

    # --- Stage 5: Decoder2 Setup & Training ---
    all_di_info = {}
    use_di_train = di_cfg.get("use_dihedral_loss", False)
    if use_di_train:
        logger.info("Dihedral loss is configured but parsing logic is not implemented in this version.")

    tr_dec, te_dec = train_test_split(dec_in_dset, test_size=0.1, random_state=42)
    load_tr_dec = DataLoader(tr_dec, dec2_cfg['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    load_te_dec = DataLoader(te_dec, dec2_cfg['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    
    dec2_model = ProteinStateReconstructor2D(
        node_emb_dim=hno_cfg['hidden_dim'],
        cond_emb_dim=hno_cfg['hidden_dim'],
        output_height=d2s['output_height'],
        output_width=d2s['output_width'],
        mlp_h_dim=d2s['mlp_hidden_dim'],
        mlp_layers=d2s['num_hidden_layers'],
        logger=logger
    )
    dec2_ckpt = os.path.join(out_cfg['checkpoint_dir'], "decoder2_checkpoint.pth")
    dec2_model = train_decoder2_model(
        model=dec2_model, tr_loader=load_tr_dec, te_loader=load_te_dec,
        all_conditioners=all_conditioners,
        all_z_ref_ensembles=all_z_ref_ensembles, # Pass empty dict, not used by this aug
        use_z_ref_ensemble=dec2_cfg.get('use_z_ref_ensemble', False),
        all_sys_info=all_sys_info,
        N_epochs=dec2_cfg['num_epochs'], lr=dec2_cfg['learning_rate'],
        ckpt=dec2_ckpt, save_int=dec2_cfg['save_interval'],
        dev=device, logger=logger,
        base_w=dec2_cfg['base_loss_weight'],
        use_di=use_di_train, all_di_info=all_di_info,
        div_t=di_cfg.get('divergence_type', 'JS'),
        l_div=di_cfg.get('lambda_divergence', 0.0),
        l_mse=di_cfg.get('lambda_torsion_mse', 0.0)
    )

    # --- Stage 6: Final Export ---
    export_final_outputs_multi(
        hno=hno_model,
        dec2=dec2_model,
        full_dset=full_dset,
        dec_in_dset=dec_in_dset,
        all_conditioners=all_conditioners,
        all_sys_info=all_sys_info,
        all_X_refs=all_X_refs,
        all_X_ref_ensembles=all_X_ref_ensembles, # Pass empty dict
        all_z_ref_ensembles=all_z_ref_ensembles, # Pass empty dict
        struct_dir=out_cfg['structure_dir'],
        latent_dir=out_cfg['latent_dir'],
        dev=device,
        logger=logger
    )

    logger.info(f"================ Script Finished ({time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}) ================")

################################################################################
# (J) Script Entry Point
################################################################################
if __name__ == "__main__":
    main()