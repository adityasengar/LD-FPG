import json
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Any

from torch_geometric.data import Data
from torch_cluster import knn_graph

# --- PDB Parsing ---
def parse_pdb(filename: str, logger: logging.Logger) -> Tuple[Dict, List]:
    """Parses ATOM records from a PDB file, handling alternate locations."""
    backbone_atoms = {"N", "CA", "C", "O", "OXT"} # Define backbone atom names
    atoms_in_order: List[Tuple[str, int, str]] = []
    processed_atom_indices = set() # To handle unique atom serial numbers

    try:
        with open(filename, 'r') as pdb_file:
            for line_num, line in enumerate(pdb_file, 1):
                if not line.startswith("ATOM  "): # Standard ATOM record start
                    continue
                try:
                    atom_serial = int(line[6:11])
                    atom_name = line[12:16].strip()
                    alt_loc = line[16].strip()
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    res_seq = int(line[22:26])
                except ValueError as e:
                    logger.warning(f"Skipping PDB line {line_num} due to parsing error: {e}")
                    continue

                # Handle alternate locations - take the first one (usually 'A' or empty)
                if alt_loc != '' and alt_loc != 'A':
                    continue
                # Ensure unique atom processing if serial numbers are reused for alt locs
                if atom_serial in processed_atom_indices and alt_loc == '': # Common for non-altloc
                    logger.warning(f"Duplicate atom serial {atom_serial} at line {line_num} without alt_loc. Possible PDB format issue. Taking first encountered.")
                    continue
                elif alt_loc != '' and f"{atom_serial}_{alt_loc}" in processed_atom_indices: # If tracking alt locs specifically
                     continue

                # Mark atom as processed
                processed_atom_indices.add(atom_serial if alt_loc == '' else f"{atom_serial}_{alt_loc}")


                orig_res_id = f"{chain_id}:{res_name}:{res_seq}" # Unique original residue ID
                category = "backbone" if atom_name in backbone_atoms else "sidechain"
                atoms_in_order.append((orig_res_id, atom_serial, category))
    except FileNotFoundError:
        logger.error(f"PDB file not found: {filename}")
        return {}, []
    except Exception as e:
        logger.error(f"Error reading PDB file {filename}: {e}", exc_info=True)
        return {}, []

    if not atoms_in_order:
        logger.error(f"No valid ATOM records found or parsed from PDB file: {filename}")
    else:
        logger.info(f"Successfully parsed {len(atoms_in_order)} ATOM records from {filename}.")
    return {}, atoms_in_order # First dict is unused in original, kept for signature

def renumber_atoms_and_residues(atoms_in_order: List[Tuple[str, int, str]], logger: logging.Logger) -> Tuple[Dict, Dict]:
    """Renumbers residues and atoms consecutively starting from 0, maintaining order."""
    new_res_dict: Dict[int, Dict[str, List[int]]] = {}
    orig_atom_map: Dict[int, int] = {} # Maps original atom serial to new 0-based index
    orig_res_map: Dict[str, int] = {}  # Maps original residue ID string to new 0-based residue index

    next_new_res_id = 0
    next_new_atom_index = 0

    # Preserve original residue order by assigning an order index
    seen_res_order: Dict[str, int] = {}
    res_order_counter = 0
    for orig_res_id, _, _ in atoms_in_order:
        if orig_res_id not in seen_res_order:
            seen_res_order[orig_res_id] = res_order_counter
            res_order_counter += 1

    # Create a sortable list that respects original residue and atom order
    sortable_atoms = []
    for orig_res_id, atom_serial, category in atoms_in_order:
        sortable_atoms.append((seen_res_order[orig_res_id], atom_serial, orig_res_id, category))
    sortable_atoms.sort() # Sort by residue order, then by atom serial

    for _, atom_serial, orig_res_id, category in sortable_atoms:
        if orig_res_id not in orig_res_map:
            orig_res_map[orig_res_id] = next_new_res_id
            new_res_dict[next_new_res_id] = {"backbone": [], "sidechain": []}
            next_new_res_id += 1

        current_new_res_id = orig_res_map[orig_res_id]
        new_res_dict[current_new_res_id][category].append(next_new_atom_index)
        orig_atom_map[atom_serial] = next_new_atom_index
        next_new_atom_index += 1

    logger.info(f"Renumbered to {next_new_res_id} residues and {next_new_atom_index} atoms.")
    return new_res_dict, orig_atom_map

def get_global_indices(renumbered_residue_dict: Dict) -> Tuple[List[int], List[int]]:
    """Extracts sorted global lists of backbone and sidechain atom indices from renumbered dict."""
    backbone_indices: List[int] = []
    sidechain_indices: List[int] = []
    for res_id in sorted(renumbered_residue_dict.keys()): # Ensure residue order
        backbone_indices.extend(sorted(renumbered_residue_dict[res_id]["backbone"]))
        sidechain_indices.extend(sorted(renumbered_residue_dict[res_id]["sidechain"]))
    return backbone_indices, sidechain_indices

# --- JSON Loading ---
def load_heavy_atom_coords_from_json(json_file: str, logger: logging.Logger) -> Tuple[List[torch.Tensor], int]:
    """Loads heavy atom coordinates from the specified JSON file.
    Assumes JSON structure: { "res_id_str": {"heavy_atom_coords_per_frame": [ [[x,y,z], ...], ... ]}, ... }
    Residue IDs are stringified integers, sorted numerically. Atoms are concatenated in this residue order.
    """
    logger.info(f"Loading heavy atom coordinates from JSON file: {json_file}")
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_file}")
        return [], -1
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file {json_file}: {e}")
        return [], -1

    coords_per_frame_list: List[torch.Tensor] = []
    num_atoms_per_frame = -1

    try:
        # Extract and sort residue keys numerically
        residue_keys_int = sorted([int(k) for k in data.keys()])
        residue_keys_str = [str(k) for k in residue_keys_int]

        if not residue_keys_str:
            logger.error("No residue data found in the JSON file.")
            return [], -1
        logger.info(f"Found {len(residue_keys_str)} residues in JSON data.")

        # Determine number of frames from the first residue's data
        first_res_data = data[residue_keys_str[0]].get("heavy_atom_coords_per_frame")
        if first_res_data is None or not isinstance(first_res_data, list):
            logger.error("Missing or invalid 'heavy_atom_coords_per_frame' in first residue.")
            return [], -1
        num_frames = len(first_res_data)
        if num_frames == 0:
            logger.warning("No frames found in the JSON data.")
            return [], -1
        logger.info(f"Found {num_frames} frames in JSON data.")

        for frame_idx in range(num_frames):
            current_frame_coords_np_list = []
            current_num_atoms = 0
            for res_key in residue_keys_str:
                res_data = data[res_key]
                try:
                    atom_coords_for_res_frame = np.array(res_data["heavy_atom_coords_per_frame"][frame_idx], dtype=np.float32)
                    if atom_coords_for_res_frame.ndim != 2 or atom_coords_for_res_frame.shape[1] != 3:
                        raise ValueError(f"Coordinate array for residue {res_key}, frame {frame_idx} has incorrect shape: {atom_coords_for_res_frame.shape}")
                    current_frame_coords_np_list.append(atom_coords_for_res_frame)
                    current_num_atoms += atom_coords_for_res_frame.shape[0]
                except Exception as e:
                    logger.error(f"Error processing residue {res_key}, frame {frame_idx}: {e}")
                    return [], -1

            if frame_idx == 0:
                num_atoms_per_frame = current_num_atoms
                logger.info(f"Determined {num_atoms_per_frame} atoms per frame from JSON data.")
            elif current_num_atoms != num_atoms_per_frame:
                logger.error(f"Inconsistent number of atoms in frame {frame_idx} ({current_num_atoms}) compared to first frame ({num_atoms_per_frame}).")
                return [], -1

            try:
                # Concatenate coordinates for all residues in the current frame
                full_frame_coords_np = np.concatenate(current_frame_coords_np_list, axis=0)
                coords_per_frame_list.append(torch.tensor(full_frame_coords_np, dtype=torch.float32))
            except ValueError as e: # Handle empty list concatenation
                logger.error(f"Error concatenating coordinates for frame {frame_idx}: {e}")
                return [], -1


    except Exception as e:
        logger.error(f"Failed to process JSON data structure: {e}", exc_info=True)
        return [], -1

    if not coords_per_frame_list:
        logger.error("No coordinate frames were successfully loaded from JSON.")
        return [], -1

    return coords_per_frame_list, num_atoms_per_frame

# --- Alignment ---
def compute_centroid(X: torch.Tensor) -> torch.Tensor:
    """Computes the centroid of a point cloud X [..., N, 3]."""
    return X.mean(dim=-2) # Mean over the N atoms

def kabsch_algorithm(P: torch.Tensor, Q: torch.Tensor, logger: logging.Logger) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aligns point cloud Q onto point cloud P using the Kabsch algorithm.
    Assumes P and Q are of shape [N, 3] or [B, N, 3].
    Returns rotation matrix U and aligned Q.
    """
    P_float, Q_float = P.float(), Q.float() # Ensure float for SVD
    is_batched = P_float.ndim == 3

    if not is_batched:
        P_batch = P_float.unsqueeze(0)
        Q_batch = Q_float.unsqueeze(0)
    else:
        P_batch = P_float
        Q_batch = Q_float

    B, N, _ = P_batch.shape
    centroid_P = compute_centroid(P_batch) # [B, 3]
    centroid_Q = compute_centroid(Q_batch) # [B, 3]

    P_centered = P_batch - centroid_P.unsqueeze(1) # [B, N, 3]
    Q_centered = Q_batch - centroid_Q.unsqueeze(1) # [B, N, 3]

    # Covariance matrix C = Q_centered^T * P_centered
    C = torch.bmm(Q_centered.transpose(1, 2), P_centered) # [B, 3, 3]

    try:
        V, S, Wt = torch.linalg.svd(C) # V [B,3,3], S [B,3], Wt [B,3,3] ( Wt is W transpose)
    except Exception as e: # Catch LinAlgError for singular matrices
        logger.error(f"Kabsch SVD failed: {e}. Returning identity alignment.", exc_info=True)
        # Fallback: return Q centered at P's centroid without rotation
        U_identity = torch.eye(3, device=P.device).unsqueeze(0).expand(B, -1, -1)
        Q_aligned_fallback = Q_batch - centroid_Q.unsqueeze(1) + centroid_P.unsqueeze(1) # Translate Q to P's centroid
        return (U_identity.squeeze(0), Q_aligned_fallback.squeeze(0)) if not is_batched else (U_identity, Q_aligned_fallback)


    # Ensure a right-handed coordinate system (handle reflections)
    determinant = torch.det(torch.bmm(V, Wt)) # [B]
    D_diag = torch.eye(3, device=P.device).unsqueeze(0).repeat(B, 1, 1) # [B, 3, 3]
    D_diag[:, 2, 2] = torch.sign(determinant) # [B]

    # Optimal rotation matrix U = V * D * Wt
    U = torch.bmm(torch.bmm(V, D_diag), Wt) # [B, 3, 3]

    # Align Q: Q_aligned = Q_centered * U + centroid_P
    Q_aligned_batch = torch.bmm(Q_centered, U) + centroid_P.unsqueeze(1)

    if not is_batched:
        return U.squeeze(0), Q_aligned_batch.squeeze(0)
    else:
        return U, Q_aligned_batch

def align_frames_to_first(coords_list: List[torch.Tensor], logger: logging.Logger, device: torch.device) -> List[torch.Tensor]:
    """Aligns all coordinate frames in a list to the first frame using Kabsch.
    Returns the list of aligned coordinate Tensors on CPU.
    """
    logger.info("Starting alignment of coordinate frames to the first frame...")
    if not coords_list:
        logger.warning("Coordinate list is empty. No frames to align.")
        return []

    reference_frame = coords_list[0].float().to(device)
    aligned_coords_list = [coords_list[0].cpu()] # First frame is the reference, already "aligned"

    num_total_frames_to_align = len(coords_list) - 1
    for i, frame_coords in enumerate(coords_list[1:], 1):
        _, aligned_frame_device = kabsch_algorithm(reference_frame, frame_coords.float().to(device), logger)
        aligned_coords_list.append(aligned_frame_device.cpu())
        if (i % 500 == 0 or i == num_total_frames_to_align) and num_total_frames_to_align > 0:
            logger.info(f"Aligned {i}/{num_total_frames_to_align} frames...")

    logger.info("Finished aligning all coordinate frames.")
    return aligned_coords_list

# --- Graph Dataset ---
def build_graph_dataset(coords_list: List[torch.Tensor], knn_k_value: int, logger: logging.Logger, device: torch.device) -> List[Data]:
    """Builds a PyTorch Geometric dataset with k-NN graphs for each frame.
    Returns a list of Data objects, with features and edges on CPU.
    """
    logger.info(f"Building PyTorch Geometric dataset with k-NN graphs (k={knn_k_value}). Processing on device '{device}'...")
    pyg_dataset: List[Data] = []
    num_frames = len(coords_list)

    for i, frame_coords_cpu in enumerate(coords_list):
        frame_coords_device = frame_coords_cpu.to(device) # Move to device for knn_graph
        # Construct k-NN graph. loop=False to avoid self-loops unless intended.
        # batch=None as we process one graph at a time.
        edge_index_device = knn_graph(frame_coords_device, k=knn_k_value, loop=False, batch=None)

        # Create Data object, ensuring all components are on CPU for dataset storage
        data_object = Data(x=frame_coords_cpu, edge_index=edge_index_device.cpu(), y=frame_coords_cpu)
        pyg_dataset.append(data_object)

        if ((i + 1) % 500 == 0 or (i + 1) == num_frames) and num_frames > 0:
            logger.info(f"Built graph for frame {i+1}/{num_frames}...")

    logger.info("Finished building PyTorch Geometric dataset.")
    return pyg_dataset
