import os
import sys
import json
import yaml
import argparse
import logging
import h5py # Though direct use is now in export_utils
import torch
import numpy as np # Though direct use might be minimal here now
import time
from typing import Dict, List, Optional # Keep for type hints from config

# PyG imports
from torch_geometric.loader import DataLoader as PyGDataLoader # Explicitly PyG DataLoader
from sklearn.model_selection import train_test_split

# Custom module imports
from data_utils import (parse_pdb, renumber_atoms_and_residues, get_global_indices,
                        load_heavy_atom_coords_from_json, align_frames_to_first, build_graph_dataset)
from models import HNO, ProteinStateReconstructor2D
from trainers import train_hno_model, train_decoder2_model
from checkpoint_utils import load_checkpoint # Though specific use is within trainers
from export_utils import export_final_outputs

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Protein Autoencoder Training (HNO Encoder + Decoder2) for LD-FPG Step 1"
)
parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
parser.add_argument("--debug", action="store_true", help="Enable debug level logging.")
args = parser.parse_args()

# --- Pre-Logging Config Load (Minimal, for log file path) ---
LOG_FILE_DEFAULT_PATH = "autoencoder_training_run.log"
log_file_path_from_config = LOG_FILE_DEFAULT_PATH
temp_config_for_log = None
try:
    with open(args.config, "r") as f_temp_config:
        temp_config_for_log = yaml.safe_load(f_temp_config)
    log_file_path_from_config = temp_config_for_log.get("log_file", LOG_FILE_DEFAULT_PATH)
except Exception as e_log_config:
    print(f"[Warning] Could not pre-load log file path from config ({args.config}): {e_log_config}. "
          f"Using default: {LOG_FILE_DEFAULT_PATH}")

# --- Logging Setup ---
# Using a more specific logger name for the main script
logger = logging.getLogger("AutoencoderTrainingScript") # Changed from ProteinReconstruction
logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

if not logger.handlers: # Avoid adding handlers multiple times if script/module is reloaded
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    # File Handler
    try:
        file_handler = logging.FileHandler(log_file_path_from_config, mode="w") # Overwrite log each run
        file_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except IOError as e_file_log:
        print(f"Warning: Could not write to log file {log_file_path_from_config}: {e_file_log}. Logging to console only.")
        if 'formatter' not in locals(): # Ensure formatter exists if file handler failed early
            formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
    if 'formatter' not in locals(): # Ensure formatter exists
            formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info(f"Logger initialized. Logging to: {log_file_path_from_config}")
logger.debug("Debug mode is ON.") if args.debug else logger.info("Debug mode is OFF.")

# --- Device Setup ---
# Device setup now uses the fully loaded config later in main()
# global_device variable is set inside main() for clarity.

# --- Main Execution Function ---
def main():
    """Main orchestration function for autoencoder training."""
    script_start_time = time.time()
    logger.info("================ Autoencoder Training Script Starting ================")

    # --- Load Full Configuration ---
    try:
        with open(args.config, "r") as f_config:
            config = yaml.safe_load(f_config)
        logger.info(f"Successfully loaded full configuration from {args.config}")
    except Exception as e_config:
        logger.error(f"Failed to load full configuration file: {e_config}", exc_info=True)
        sys.exit(1)

    # --- Extract Parameters & Setup ---
    try:
        # General settings
        force_cpu_flag = config.get("force_cpu", False)
        cuda_device_idx = config.get("cuda_device", 0)
        num_dataloader_workers = config.get("num_workers", 0)

        # Data paths
        data_paths_cfg = config["data"]
        json_coord_path = data_paths_cfg["json_path"]
        pdb_ref_path = data_paths_cfg["pdb_path"]

        # Graph settings
        graph_settings_cfg = config["graph"]
        knn_k_val = graph_settings_cfg["knn_value"]

        # HNO Encoder settings
        hno_encoder_cfg = config["hno_encoder"]
        hno_hidden_dim = hno_encoder_cfg["hidden_dim"]
        hno_cheb_order = hno_encoder_cfg["cheb_order"]
        hno_num_epochs = hno_encoder_cfg["num_epochs"]
        hno_learning_rate = hno_encoder_cfg["learning_rate"]
        hno_batch_size = hno_encoder_cfg["batch_size"]
        hno_save_interval = hno_encoder_cfg.get("save_interval", 500)

        # Decoder2 settings
        decoder2_train_cfg = config["decoder2"]
        decoder2_num_epochs = decoder2_train_cfg["num_epochs"]
        decoder2_learning_rate = decoder2_train_cfg["learning_rate"]
        decoder2_batch_size = decoder2_train_cfg["batch_size"]
        decoder2_base_loss_w = decoder2_train_cfg.get("base_loss_weight", 1.0)
        decoder2_save_interval = decoder2_train_cfg.get("save_interval", 500)

        decoder2_arch_cfg = config["decoder2_settings"]
        d2_conditioner_mode = decoder2_arch_cfg.get("conditioner_mode", "z_ref")
        d2_pooling_type = decoder2_arch_cfg.get("pooling_type", "blind")
        d2_pool_height_primary = decoder2_arch_cfg.get("output_height", 20)
        d2_pool_width_primary = decoder2_arch_cfg.get("output_width", 4)
        d2_mlp_hidden_dim = decoder2_arch_cfg.get("mlp_hidden_dim", 128)
        d2_num_mlp_layers = decoder2_arch_cfg.get("num_hidden_layers", 2) # Corrected key name from "num_hidden_layers"
        d2_use_secondary_pool = decoder2_arch_cfg.get("use_second_level_pooling", False)
        d2_pool_height_secondary = decoder2_arch_cfg.get("output_height2")
        d2_pool_width_secondary = decoder2_arch_cfg.get("output_width2")
        # d2_use_cross_attention = decoder2_arch_cfg.get("use_cross_attention", False) # Not used in model for now
        # d2_cross_attention_type = decoder2_arch_cfg.get("cross_attention_type", "global")

        # Dihedral loss settings
        dihedral_loss_cfg = config.get("dihedral_loss", {})
        use_dihedral_loss = dihedral_loss_cfg.get("use_dihedral_loss", False)
        torsion_info_json_path = dihedral_loss_cfg.get("torsion_info_path", "torsion.json")
        lambda_div_loss = dihedral_loss_cfg.get("lambda_divergence", 0.0)
        lambda_mse_loss = dihedral_loss_cfg.get("lambda_torsion_mse", 0.0)
        dihedral_div_type = dihedral_loss_cfg.get("divergence_type", "KL").upper()
        fraction_dihedral_batches = dihedral_loss_cfg.get("fraction_dihedral", 0.1)


        # Output directories
        output_dirs_cfg = config.get("output_directories", {})
        checkpoint_output_dir = output_dirs_cfg.get("checkpoint_dir", "checkpoints")
        structure_output_dir = output_dirs_cfg.get("structure_dir", "structures")
        latent_output_dir = output_dirs_cfg.get("latent_dir", "latent_reps")

    except KeyError as e_key:
        logger.error(f"Missing critical configuration key: {e_key}. Please check your YAML file.")
        sys.exit(1)

    # --- Finalize Device Selection ---
    selected_device = torch.device("cpu")
    if not force_cpu_flag and torch.cuda.is_available():
        try:
            torch.cuda.get_device_name(cuda_device_idx) # Validate device index
            selected_device = torch.device(f"cuda:{cuda_device_idx}")
        except Exception as e_cuda:
            logger.warning(f"Could not validate CUDA device index {cuda_device_idx} ({e_cuda}). "
                           f"Defaulting to cuda:0 if available, otherwise CPU.")
            if torch.cuda.is_available(): selected_device = torch.device("cuda:0")
    logger.info(f"Using device: {selected_device}")
    pin_memory_flag = (selected_device.type == "cuda")

    # --- Create Output Directories ---
    try:
        for dir_path in [checkpoint_output_dir, structure_output_dir, latent_output_dir]:
            os.makedirs(dir_path, exist_ok=True)
        logger.info("Output directories ensured.")
    except OSError as e_dir:
        logger.error(f"Failed to create output directories: {e_dir}", exc_info=True)
        sys.exit(1)

    # --- Stage 1: Data Loading & Preprocessing ---
    logger.info("--- Stage 1: Data Loading & Preprocessing ---")
    _, atoms_ordered_list = parse_pdb(pdb_ref_path, logger)
    if not atoms_ordered_list:
        logger.error("Failed to parse PDB file. Exiting.")
        sys.exit(1)

    renumbered_res_dict, _ = renumber_atoms_and_residues(atoms_ordered_list, logger)
    backbone_idx_list, sidechain_idx_list = get_global_indices(renumbered_res_dict)
    num_atoms_from_pdb = len(backbone_idx_list) + len(sidechain_idx_list)

    # Convert to tensors (will be moved to device in trainer functions)
    backbone_indices_tensor = torch.tensor(backbone_idx_list, dtype=torch.long)
    sidechain_indices_tensor = torch.tensor(sidechain_idx_list, dtype=torch.long)
    logger.info(f"PDB Parsed: TotalAtoms={num_atoms_from_pdb} (Backbone={len(backbone_idx_list)}, Sidechain={len(sidechain_idx_list)})")

    coords_frames_list, num_atoms_from_json = load_heavy_atom_coords_from_json(json_coord_path, logger)
    if not coords_frames_list:
        logger.error("Failed to load coordinates from JSON. Exiting.")
        sys.exit(1)
    if num_atoms_from_json != num_atoms_from_pdb:
        logger.error(f"Atom count mismatch: JSON ({num_atoms_from_json}) vs PDB ({num_atoms_from_pdb}). Exiting.")
        sys.exit(1)
    num_total_atoms = num_atoms_from_pdb # Consistent atom count

    aligned_coords_cpu_list = align_frames_to_first(coords_frames_list, logger, selected_device)
    if not aligned_coords_cpu_list:
        logger.error("Frame alignment failed. Exiting.")
        sys.exit(1)

    # This dataset contains original coordinates (x and y) and edge_index
    full_pyg_dataset_original_coords = build_graph_dataset(aligned_coords_cpu_list, knn_k_val, logger, selected_device)
    if not full_pyg_dataset_original_coords:
        logger.error("Failed to build PyG dataset from original coordinates. Exiting.")
        sys.exit(1)

    # --- Stage 2: HNO Encoder Training/Loading ---
    logger.info("--- Stage 2: HNO Encoder Training/Loading ---")
    train_hno_data, val_hno_data = train_test_split(full_pyg_dataset_original_coords, test_size=0.1, random_state=42)
    train_hno_loader = PyGDataLoader(train_hno_data, batch_size=hno_batch_size, shuffle=True,
                                  num_workers=num_dataloader_workers, pin_memory=pin_memory_flag, drop_last=True)
    val_hno_loader = PyGDataLoader(val_hno_data, batch_size=hno_batch_size, shuffle=False,
                                num_workers=num_dataloader_workers, pin_memory=pin_memory_flag)

    hno_model_instance = HNO(hidden_dim=hno_hidden_dim, K_cheb_order=hno_cheb_order)
    hno_checkpoint_file = os.path.join(checkpoint_output_dir, "hno_encoder_checkpoint.pth")
    hno_model_instance = train_hno_model(
        hno_model_instance, train_hno_loader, val_hno_loader,
        backbone_indices_tensor, sidechain_indices_tensor, # Pass global indices
        hno_num_epochs, hno_learning_rate, hno_checkpoint_file,
        hno_save_interval, selected_device, logger
    )
    hno_model_instance.eval() # Ensure model is in eval mode after training

    # --- Stage 3: Prepare Decoder Input Dataset (using trained HNO to get embeddings) ---
    logger.info("--- Stage 3: Preparing Decoder2 Input Dataset (Atom Embeddings) ---")
    decoder_input_pyg_dataset: List[torch_geometric.data.Data] = [] # Corrected type hint
    # Use a DataLoader for potentially large datasets to process in batches
    inference_original_loader = PyGDataLoader(full_pyg_dataset_original_coords, batch_size=hno_batch_size * 2, # Can use larger BS for inference
                                           shuffle=False, num_workers=num_dataloader_workers, pin_memory=pin_memory_flag)
    with torch.no_grad():
        for batch_idx, original_data_batch in enumerate(inference_original_loader):
            original_data_batch = original_data_batch.to(selected_device)
            # Get atom-wise embeddings Z
            atom_embeddings_batch = hno_model_instance.forward_representation(original_data_batch.x, original_data_batch.edge_index)
            target_coords_batch = original_data_batch.y # Ground truth coordinates

            # Handle PyG batching: split batch back into individual graph Data objects
            # counts will tell how many nodes are in each graph in the batch
            node_counts_per_graph = original_data_batch.ptr[1:] - original_data_batch.ptr[:-1]

            if node_counts_per_graph.numel() > 0 and \
               node_counts_per_graph.sum() == atom_embeddings_batch.shape[0] and \
               node_counts_per_graph.sum() == target_coords_batch.shape[0]:

                list_of_atom_embeddings = torch.split(atom_embeddings_batch, node_counts_per_graph.tolist())
                list_of_target_coords = torch.split(target_coords_batch, node_counts_per_graph.tolist())

                if len(list_of_atom_embeddings) == len(list_of_target_coords):
                    for i in range(len(list_of_atom_embeddings)):
                        decoder_input_pyg_dataset.append(
                            Data(x=list_of_atom_embeddings[i].cpu(), y=list_of_target_coords[i].cpu())
                        )
                else:
                    logger.error(f"Mismatch in lengths of split embedding/target lists for batch {batch_idx}. Skipping batch.")
            elif node_counts_per_graph.numel() == 0 and atom_embeddings_batch.shape[0] == 0 and target_coords_batch.shape[0] == 0:
                pass # Empty batch processed, do nothing
            else: # Mismatch in total nodes
                logger.error(f"Node count sum mismatch when creating decoder input for batch {batch_idx}. "
                             f"CountsSum={node_counts_per_graph.sum() if node_counts_per_graph.numel() > 0 else 'N/A'}, "
                             f"EmbShape0={atom_embeddings_batch.shape[0]}, TargetShape0={target_coords_batch.shape[0]}. Skipping batch.")


    logger.info(f"Decoder2 input dataset created with {len(decoder_input_pyg_dataset)} samples.")
    if not decoder_input_pyg_dataset:
        logger.error("Decoder input dataset is empty! This usually means atom embedding generation failed. Cannot proceed.")
        sys.exit(1)

    # --- Stage 4: Decoder2 Setup & Training ---
    logger.info("--- Stage 4: Decoder2 Setup & Training ---")
    train_decoder_data, val_decoder_data = train_test_split(decoder_input_pyg_dataset, test_size=0.1, random_state=42)
    train_decoder_loader = PyGDataLoader(train_decoder_data, batch_size=decoder2_batch_size, shuffle=True,
                                      num_workers=num_dataloader_workers, pin_memory=pin_memory_flag, drop_last=True)
    val_decoder_loader = PyGDataLoader(val_decoder_data, batch_size=decoder2_batch_size, shuffle=False,
                                    num_workers=num_dataloader_workers, pin_memory=pin_memory_flag)

    # Determine Conditioner (X_ref or z_ref)
    # Use the first frame from the *original coordinates* dataset for reference
    first_frame_original_data = full_pyg_dataset_original_coords[0].to(selected_device)
    x_ref_coords_device = first_frame_original_data.x # These are the coordinates of the first frame
    conditioner_dim_val = -1
    conditioner_tensor_cpu = None # Store the chosen conditioner on CPU for export
    z_ref_embedding_cpu = None    # Specifically store z_ref if used

    if d2_conditioner_mode.lower() == "x_ref":
        conditioner_tensor_cpu = x_ref_coords_device.cpu()
        conditioner_dim_val = 3 # Coordinates are 3D
        logger.info("Using X_ref (reference coordinates) as conditioner for Decoder2.")
    elif d2_conditioner_mode.lower() == "z_ref":
        with torch.no_grad():
            z_ref_embedding_device = hno_model_instance.forward_representation(
                x_ref_coords_device, first_frame_original_data.edge_index.to(selected_device) # Edge index also needed
            )
        conditioner_tensor_cpu = z_ref_embedding_device.cpu()
        z_ref_embedding_cpu = conditioner_tensor_cpu # Keep a specific copy
        conditioner_dim_val = z_ref_embedding_device.shape[1]
        logger.info(f"Using z_ref (reference atom embeddings) as conditioner for Decoder2. Dim: {conditioner_dim_val}")
    else:
        logger.error(f"Invalid conditioner_mode: '{d2_conditioner_mode}'. Choose 'X_ref' or 'z_ref'. Exiting.")
        sys.exit(1)

    # Save reference structures/embeddings for later use (e.g., by diffusion sampling script)
    try:
        torch.save(x_ref_coords_device.cpu(), os.path.join(structure_output_dir, "X_ref_coords.pt"))
        logger.info(f"Saved X_ref_coords.pt to {structure_output_dir}")
    except Exception as e_save_xref: logger.error(f"Failed to save X_ref_coords.pt: {e_save_xref}")
    if z_ref_embedding_cpu is not None:
        try:
            torch.save(z_ref_embedding_cpu, os.path.join(latent_output_dir, "z_ref_embedding.pt"))
            logger.info(f"Saved z_ref_embedding.pt to {latent_output_dir}")
        except Exception as e_save_zref: logger.error(f"Failed to save z_ref_embedding.pt: {e_save_zref}")


    # Dihedral Precomputation
    logger.info("--- Dihedral Information Precomputation (if enabled) ---")
    dihedral_info_for_loss: Dict = {}
    dihedral_mask_for_loss: Optional[torch.Tensor] = None
    num_total_residues_for_loss: Optional[int] = None
    can_use_dihedral_loss = False

    if use_dihedral_loss:
        torsion_info_abs_path = os.path.abspath(torsion_info_json_path)
        if os.path.isfile(torsion_info_abs_path):
            logger.info(f"Found torsion information file: {torsion_info_abs_path}")
            try:
                with open(torsion_info_abs_path, "r") as f_torsion:
                    torsion_data_json = json.load(f_torsion)

                # Process torsion_data_json to populate dihedral_info_for_loss, dihedral_mask_for_loss, num_total_residues_for_loss
                # This part is complex and depends heavily on the exact format of condensed_residues.json
                # Assuming format: { "res_id_int_str": {"torsion_atoms": {"phi": [i,j,k,l], "psi": ..., "chi": {"chi1": ...}}}, ...}
                # And that atom indices are 0-based global indices matching num_total_atoms.

                angle_types_defined = ['phi', 'psi', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5']
                # Initialize structures to hold indices
                indices_lists_per_angle: Dict[str, List[List[int]]] = {name: [[] for _ in range(4)] for name in angle_types_defined}
                residue_indices_per_angle: Dict[str, List[int]] = {name: [] for name in angle_types_defined}

                # Sort residue keys numerically to ensure consistent processing order
                sorted_residue_ids_str = sorted(torsion_data_json.keys(), key=int)
                num_total_residues_for_loss = len(sorted_residue_ids_str)
                mask_list_of_lists = [[False] * len(angle_types_defined) for _ in range(num_total_residues_for_loss)]
                skipped_angles_count = 0

                for res_idx_numeric_order, res_id_str in enumerate(sorted_residue_ids_str):
                    residue_entry = torsion_data_json.get(res_id_str, {})
                    torsion_atoms_entry = residue_entry.get("torsion_atoms", {})
                    chi_atoms_entry = torsion_atoms_entry.get("chi", {}) # chi angles are often nested

                    for angle_type_idx, angle_name in enumerate(angle_types_defined):
                        atom_quadruplet_indices = None
                        if angle_name in ['phi', 'psi']:
                            atom_quadruplet_indices = torsion_atoms_entry.get(angle_name)
                        else: # chi angles
                            atom_quadruplet_indices = chi_atoms_entry.get(angle_name)

                        if isinstance(atom_quadruplet_indices, list) and len(atom_quadruplet_indices) == 4 and \
                           all(isinstance(idx, int) for idx in atom_quadruplet_indices):
                            # Validate atom indices are within bounds
                            if all(0 <= atom_idx < num_total_atoms for atom_idx in atom_quadruplet_indices):
                                for k_idx_in_quad, atom_idx_val in enumerate(atom_quadruplet_indices):
                                    indices_lists_per_angle[angle_name][k_idx_in_quad].append(atom_idx_val)
                                residue_indices_per_angle[angle_name].append(res_idx_numeric_order)
                                mask_list_of_lists[res_idx_numeric_order][angle_type_idx] = True
                            else:
                                skipped_angles_count +=1
                                # logger.debug(f"Skipped angle {angle_name} for res {res_id_str}: indices {atom_quadruplet_indices} out of bounds for N_atoms={num_total_atoms}")

                if skipped_angles_count > 0:
                    logger.warning(f"Skipped {skipped_angles_count} dihedral angles due to out-of-bounds atom indices during precomputation.")

                # Convert lists to tensors for dihedral_info_for_loss
                for angle_name_str_iter in angle_types_defined:
                    if residue_indices_per_angle[angle_name_str_iter]: # If any angles of this type were found
                        dihedral_info_for_loss[angle_name_str_iter] = {
                            'indices': [torch.tensor(idx_list_for_pos, dtype=torch.long) for idx_list_for_pos in indices_lists_per_angle[angle_name_str_iter]],
                            'res_idx': torch.tensor(residue_indices_per_angle[angle_name_str_iter], dtype=torch.long)
                        }
                    else: # No valid angles of this type found
                         dihedral_info_for_loss[angle_name_str_iter] = {'indices': None, 'res_idx': None}


                dihedral_mask_for_loss = torch.tensor(mask_list_of_lists, dtype=torch.bool)
                can_use_dihedral_loss = True
                logger.info(f"Successfully precomputed dihedral information for {num_total_residues_for_loss} residues.")

            except json.JSONDecodeError as e_json:
                logger.error(f"Error decoding torsion info JSON file '{torsion_info_abs_path}': {e_json}")
            except Exception as e_torsion_proc:
                logger.error(f"Error processing torsion information from '{torsion_info_abs_path}': {e_torsion_proc}", exc_info=True)
        else:
            logger.warning(f"Torsion information file not found at resolved path: {torsion_info_abs_path}. Dihedral loss cannot be enabled.")

    final_use_dihedral_loss_flag = use_dihedral_loss and can_use_dihedral_loss
    if use_dihedral_loss and not can_use_dihedral_loss:
        logger.warning("Dihedral loss was requested in config, but precomputation failed. Disabling for training.")
    logger.info(f"Final dihedral loss status for Decoder2 training: {'ENABLED' if final_use_dihedral_loss_flag else 'DISABLED'}")


    # Initialize Decoder2 Model
    residue_indices_for_pooling = None # For 'residue' pooling type, needs proper setup
    if d2_pooling_type.lower() == "residue":
        # This would require renumbered_res_dict to be processed into a list of lists of global atom indices per residue
        logger.warning("Residue pooling type selected for Decoder2, but `residue_segment_indices` "
                       "derivation from `renumbered_res_dict` is not fully implemented here. "
                       "Blind pooling will effectively occur if not correctly set up.")
        # Example placeholder: if renumbered_res_dict maps res_idx_new -> {'backbone': [...], 'sidechain': [...]},
        # residue_indices_for_pooling = [renumbered_res_dict[k]['backbone'] + renumbered_res_dict[k]['sidechain']
        #                               for k in sorted(renumbered_res_dict.keys())]


    secondary_pool_dims = (d2_pool_height_secondary, d2_pool_width_secondary) \
        if d2_use_secondary_pool and d2_pool_height_secondary is not None and d2_pool_width_secondary is not None else None

    decoder2_model_instance = ProteinStateReconstructor2D(
        atom_embedding_dim=hno_hidden_dim, # d_z from HNO
        num_atoms_total=num_total_atoms,
        conditioner_dim=conditioner_dim_val,
        pooling_type=d2_pooling_type,
        residue_segment_indices=residue_indices_for_pooling, # Pass the prepared list for residue pooling
        primary_pool_output_size=(d2_pool_height_primary, d2_pool_width_primary),
        mlp_hidden_dim=d2_mlp_hidden_dim,
        num_mlp_layers=d2_num_mlp_layers,
        use_secondary_pool=d2_use_secondary_pool,
        secondary_pool_output_size=secondary_pool_dims,
        # use_cross_attention=d2_use_cross_attention, # Not used
        # cross_attention_type=d2_cross_attention_type,
        logger_instance=logger # Pass the main script's logger
    )
    decoder2_checkpoint_file = os.path.join(checkpoint_output_dir, "decoder2_checkpoint.pth")

    decoder2_model_instance = train_decoder2_model(
        decoder2_model_instance, train_decoder_loader, val_decoder_loader,
        backbone_indices_tensor, sidechain_indices_tensor, # Global indices
        conditioner_tensor_cpu, # Pass the chosen conditioner (on CPU, trainer moves to device)
        num_total_atoms, decoder2_num_epochs, decoder2_learning_rate,
        decoder2_checkpoint_file, decoder2_save_interval, selected_device, logger,
        base_coord_loss_weight=decoder2_base_loss_w,
        use_dihedral_loss_terms=final_use_dihedral_loss_flag,
        dihedral_angle_info=dihedral_info_for_loss,
        dihedral_valid_mask=dihedral_mask_for_loss,
        num_total_residues=num_total_residues_for_loss,
        lambda_dihedral_divergence=lambda_div_loss,
        lambda_dihedral_mse=lambda_mse_loss,
        dihedral_divergence_type=dihedral_div_type,
        fraction_batches_for_dihedral_loss=fraction_dihedral_batches
    )
    decoder2_model_instance.eval() # Ensure model is in eval mode

    # --- Stage 5: Export Final Outputs from this Autoencoder stage ---
    logger.info("--- Stage 5: Exporting Autoencoder Outputs ---")
    export_final_outputs(
        hno_encoder_model=hno_model_instance,
        decoder2_model=decoder2_model_instance,
        full_original_dataset=full_pyg_dataset_original_coords, # For GT and HNO input
        decoder_input_dataset=decoder_input_pyg_dataset,       # For Decoder2 input (embeddings)
        conditioner_tensor_cpu=conditioner_tensor_cpu,         # Conditioner used by Decoder2
        num_total_atoms=num_total_atoms,
        output_structure_dir=structure_output_dir,
        output_latent_dir=latent_output_dir,
        device=selected_device,
        logger=logger
    )

    elapsed_time_seconds = time.time() - script_start_time
    logger.info(f"================ Autoencoder Training Script Finished ({time.strftime('%H:%M:%S', time.gmtime(elapsed_time_seconds))}) ================")
    sys.stdout.flush()


# --- Script Entry Point ---
if __name__ == "__main__":
    main()
