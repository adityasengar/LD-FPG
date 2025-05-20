# data_utils.py
# Description: Utilities for PDB parsing, JSON coordinate loading,
# Kabsch alignment, and PyTorch Geometric dataset construction.

from common_imports import (
    json, logging, torch, np,
    Dict, List, Tuple, Any, # from typing
    PyGData, knn_graph # from torch_geometric
)

# --- PDB Parsing ---
def parse_pdb(filename: str, logger: logging.Logger) -> Tuple[Dict, List[Tuple[str, int, str]]]:
    """Parses ATOM records from a PDB file, handling alternate locations."""
    backbone_atoms = {"N", "CA", "C", "O", "OXT"}
    atoms_in_order: List[Tuple[str, int, str]] = []
    processed_atom_keys = set() # To handle unique atom identifiers (serial_altloc)

    try:
        with open(filename, 'r') as pdb_file:
            for line_num, line in enumerate(pdb_file, 1):
                if not line.startswith("ATOM  "):
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

                # Handle alternate locations - take the first one encountered (usually 'A' or empty)
                # Create a unique key for atom including alt_loc if present
                atom_key = f"{atom_serial}_{alt_loc}" if alt_loc else str(atom_serial)

                if alt_loc != '' and alt_loc != 'A': # As per original logic, only take 'A' or no alt loc
                    if str(atom_serial) in processed_atom_keys and alt_loc != 'A': # If base atom already processed
                        continue
                    elif alt_loc != 'A': # If it's an alt loc other than A, and base not processed, skip
                        continue

                if atom_key in processed_atom_keys: # If this specific atom_serial(_altloc) already processed
                    continue
                processed_atom_keys.add(atom_key)
                if alt_loc == 'A' and str(atom_serial) not in processed_atom_keys: # also add base if 'A' is first
                    processed_atom_keys.add(str(atom_serial))


                orig_res_id = f"{chain_id}:{res_name}:{res_seq}"
                category = "backbone" if atom_name in backbone_atoms else "sidechain"
                atoms_in_order.append((orig_res_id, atom_serial, category)) # Store original serial for mapping
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
    return {}, atoms_in_order

def renumber_atoms_and_residues(atoms_in_order: List[Tuple[str, int, str]], logger: logging.Logger) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[int, int]]:
    """Renumbers residues and atoms consecutively starting from 0, maintaining order."""
    new_res_dict: Dict[int, Dict[str, List[int]]] = {}
    orig_atom_serial_to_new_index_map: Dict[int, int] = {}
    orig_res_id_str_to_new_index_map: Dict[str, int] = {}

    next_new_res_id = 0
    next_new_atom_index = 0

    # Preserve original residue order by assigning an order index
    # This ensures that residues are renumbered based on their first appearance in the PDB
    residue_appearance_order: Dict[str, int] = {}
    res_order_counter = 0
    for orig_res_id, _, _ in atoms_in_order:
        if orig_res_id not in residue_appearance_order:
            residue_appearance_order[orig_res_id] = res_order_counter
            res_order_counter += 1

    # Create a list of tuples that can be sorted to maintain original PDB order
    # (residue_appearance_order, original_atom_serial, original_residue_id_str, category)
    sortable_atom_entries = []
    for orig_res_id, atom_serial, category in atoms_in_order:
        sortable_atom_entries.append(
            (residue_appearance_order[orig_res_id], atom_serial, orig_res_id, category)
        )
    sortable_atom_entries.sort() # Sort by residue order, then by original atom serial

    for _, atom_serial, orig_res_id, category in sortable_atom_entries:
        if orig_res_id not in orig_res_id_str_to_new_index_map:
            orig_res_id_str_to_new_index_map[orig_res_id] = next_new_res_id
            new_res_dict[next_new_res_id] = {"backbone": [], "sidechain": []}
            next_new_res_id += 1

        current_new_res_id = orig_res_id_str_to_new_index_map[orig_res_id]
        new_res_dict[current_new_res_id][category].append(next_new_atom_index)
        orig_atom_serial_to_new_index_map[atom_serial] = next_new_atom_index
        next_new_atom_index += 1

    logger.info(f"Renumbered to {next_new_res_id} residues and {next_new_atom_index} atoms.")
    return new_res_dict, orig_atom_serial_to_new_index_map

def get_global_indices(renumbered_residue_dict: Dict[int, Dict[str, List[int]]]) -> Tuple[List[int], List[int]]:
    """Extracts sorted global lists of backbone and sidechain atom indices from renumbered dict."""
    backbone_indices: List[int] = []
    sidechain_indices: List[int] = []
    # Iterate through new residue IDs in their sorted (0, 1, 2...) order
    for res_id_new in sorted(renumbered_residue_dict.keys()):
        # Atoms within each category (backbone, sidechain) should also be sorted if not already
        # The renumber_atoms_and_residues function should already add them in order.
        backbone_indices.extend(renumbered_residue_dict[res_id_new]["backbone"])
        sidechain_indices.extend(renumbered_residue_dict[res_id_new]["sidechain"])
    return backbone_indices, sidechain_indices

# --- JSON Loading ---
def load_heavy_atom_coords_from_json(json_file: str, logger: logging.Logger) -> Tuple[List[torch.Tensor], int]:
    """Loads heavy atom coordinates from the specified JSON file."""
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
        residue_keys_int = sorted([int(k) for k in data.keys()])
        residue_keys_str = [str(k) for k in residue_keys_int]

        if not residue_keys_str:
            logger.error("No residue data found in the JSON file.")
            return [], -1
        logger.info(f"Found {len(residue_keys_str)} residues in JSON data.")

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
                        raise ValueError(f"Coord array for res {res_key}, frame {frame_idx} has shape {atom_coords_for_res_frame.shape}")
                    current_frame_coords_np_list.append(atom_coords_for_res_frame)
                    current_num_atoms += atom_coords_for_res_frame.shape[0]
                except Exception as e:
                    logger.error(f"Error processing residue {res_key}, frame {frame_idx}: {e}")
                    return [], -1

            if frame_idx == 0:
                num_atoms_per_frame = current_num_atoms
                logger.info(f"Determined {num_atoms_per_frame} atoms per frame from JSON data.")
            elif current_num_atoms != num_atoms_per_frame:
                logger.error(f"Inconsistent atom count in frame {frame_idx} ({current_num_atoms}) vs first frame ({num_atoms_per_frame}).")
                return [], -1
            try:
                full_frame_coords_np = np.concatenate(current_frame_coords_np_list, axis=0)
                coords_per_frame_list.append(torch.tensor(full_frame_coords_np, dtype=torch.float32))
            except ValueError as e:
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
    return X.mean(dim=-2)

def kabsch_algorithm(P: torch.Tensor, Q: torch.Tensor, logger: logging.Logger) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aligns point cloud Q onto point cloud P using Kabsch. Handles batches [B, N, 3] or single [N, 3]."""
    P_float, Q_float = P.float(), Q.float()
    is_batched = P_float.ndim == 3
    P_batch = P_float.unsqueeze(0) if not is_batched else P_float
    Q_batch = Q_float.unsqueeze(0) if not is_batched else Q_float

    B, N, _ = P_batch.shape
    centroid_P = compute_centroid(P_batch)
    centroid_Q = compute_centroid(Q_batch)
    P_centered = P_batch - centroid_P.unsqueeze(1)
    Q_centered = Q_batch - centroid_Q.unsqueeze(1)
    C = torch.bmm(Q_centered.transpose(1, 2), P_centered)

    try:
        V, S, Wt = torch.linalg.svd(C)
    except Exception as e:
        logger.error(f"Kabsch SVD failed: {e}. Returning identity alignment.", exc_info=True)
        U_identity = torch.eye(3, device=P.device).unsqueeze(0).expand(B, -1, -1)
        Q_aligned_fallback = Q_batch - centroid_Q.unsqueeze(1) + centroid_P.unsqueeze(1)
        return (U_identity.squeeze(0), Q_aligned_fallback.squeeze(0)) if not is_batched else (U_identity, Q_aligned_fallback)

    determinant = torch.det(torch.bmm(V, Wt))
    D_diag = torch.eye(3, device=P.device).unsqueeze(0).repeat(B, 1, 1)
    D_diag[:, 2, 2] = torch.sign(determinant)
    U = torch.bmm(torch.bmm(V, D_diag), Wt)
    Q_aligned_batch = torch.bmm(Q_centered, U) + centroid_P.unsqueeze(1)

    return (U.squeeze(0), Q_aligned_batch.squeeze(0)) if not is_batched else (U, Q_aligned_batch)

def align_frames_to_first(coords_list: List[torch.Tensor], logger: logging.Logger, device: torch.device) -> List[torch.Tensor]:
    """Aligns all coordinate frames to the first frame using Kabsch. Returns list on CPU."""
    logger.info("Starting alignment of coordinate frames to the first frame...")
    if not coords_list:
        logger.warning("Coordinate list is empty. No frames to align.")
        return []
    reference_frame = coords_list[0].float().to(device)
    aligned_coords_list = [coords_list[0].cpu()]
    num_total_frames_to_align = len(coords_list) - 1

    for i, frame_coords in enumerate(coords_list[1:], 1):
        _, aligned_frame_device = kabsch_algorithm(reference_frame, frame_coords.float().to(device), logger)
        aligned_coords_list.append(aligned_frame_device.cpu())
        if (i % 500 == 0 or i == num_total_frames_to_align) and num_total_frames_to_align > 0:
            logger.info(f"Aligned {i}/{num_total_frames_to_align} frames...")
    logger.info("Finished aligning all coordinate frames.")
    return aligned_coords_list

# --- Graph Dataset ---
def build_graph_dataset(coords_list: List[torch.Tensor], knn_k_value: int, logger: logging.Logger, device: torch.device) -> List[PyGData]:
    """Builds PyG dataset with k-NN graphs. Returns Data objects on CPU."""
    logger.info(f"Building PyTorch Geometric dataset with k-NN graphs (k={knn_k_value}). Processing on device '{device}'...")
    pyg_dataset: List[PyGData] = []
    num_frames = len(coords_list)

    for i, frame_coords_cpu in enumerate(coords_list):
        frame_coords_device = frame_coords_cpu.to(device)
        edge_index_device = knn_graph(frame_coords_device, k=knn_k_value, loop=False, batch=None)
        data_object = PyGData(x=frame_coords_cpu, edge_index=edge_index_device.cpu(), y=frame_coords_cpu)
        pyg_dataset.append(data_object)
        if ((i + 1) % 500 == 0 or (i + 1) == num_frames) and num_frames > 0:
            logger.info(f"Built graph for frame {i+1}/{num_frames}...")
    logger.info("Finished building PyTorch Geometric dataset.")
    return pyg_dataset
