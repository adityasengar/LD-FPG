# main.py
# Description: Main executable script for training the autoencoder (HNO Encoder + Decoder2).
# This corresponds to Step 1 of the LD-FPG pipeline (Blind Pooling variant).

from common_imports import (
    os, sys, json, yaml, argparse, logging, torch, time,
    Dict, List, Optional, # from typing
    PyGDataLoader, PyGData, train_test_split
)

# Custom project module imports
from data_utils import (parse_pdb, renumber_atoms_and_residues, get_global_indices,
                        load_heavy_atom_coords_from_json, align_frames_to_first, build_graph_dataset)
from models import HNO, ProteinStateReconstructor2D
from trainers import train_hno_model, train_decoder2_model
from export_utils import export_final_outputs
# checkpoint_utils is used internally by trainers

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Protein Autoencoder Training (HNO Encoder + Decoder2) - LD-FPG Step 1"
)
parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
parser.add_argument("--debug", action="store_true", help="Enable debug level logging.")
args = parser.parse_args()

# --- Pre-Logging Config Load (Minimal, for log file path) ---
LOG_FILE_DEFAULT_PATH = "autoencoder_run.log" # Default if not in config
log_file_path_from_config = LOG_FILE_DEFAULT_PATH
try:
    with open(args.config, "r") as f_temp_config:
        temp_config_for_log = yaml.safe_load(f_temp_config)
    log_file_path_from_config = temp_config_for_log.get("log_file", LOG_FILE_DEFAULT_PATH)
except Exception as e_log_config:
    print(f"[Warning] Could not pre-load log file path from config ({args.config}): {e_log_config}. "
          f"Using default: {LOG_FILE_DEFAULT_PATH}")

# --- Logging Setup ---
logger = logging.getLogger("AutoencoderMainScript")
logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(module)s - %(message)s")
    try: # File Handler
        file_handler = logging.FileHandler(log_file_path_from_config, mode="w")
        file_handler.setLevel(logging.DEBUG if args.debug else logging.INFO); file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except IOError as e_file_log: print(f"Warning: Log file error {log_file_path_from_config}: {e_file_log}.")
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if args.debug else logging.INFO); console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
logger.info(f"Logging to: {log_file_path_from_config}. Debug mode: {'ON' if args.debug else 'OFF'}.")

# --- Main Execution Function ---
def main_workflow():
    script_start_time = time.time()
    logger.info("================ Autoencoder Training Workflow Starting ================")
    try:
        with open(args.config, "r") as f_config: config = yaml.safe_load(f_config)
        logger.info(f"Configuration loaded from {args.config}")
    except Exception as e_config: logger.error(f"Config loading failed: {e_config}", exc_info=True); sys.exit(1)

    # Parameter Extraction (simplified for brevity, ensure all keys from your YAML are covered)
    try:
        force_cpu = config.get("force_cpu", False); cuda_idx = config.get("cuda_device", 0)
        num_workers = config.get("num_workers", 0)
        data_cfg = config["data"]; json_p = data_cfg["json_path"]; pdb_p = data_cfg["pdb_path"]
        graph_cfg = config["graph"]; knn = graph_cfg["knn_value"]
        hno_cfg = config["hno_encoder"]; hno_hdim = hno_cfg["hidden_dim"]; hno_k_order = hno_cfg["cheb_order"]
        hno_ep = hno_cfg["num_epochs"]; hno_lr = hno_cfg["learning_rate"]; hno_bs = hno_cfg["batch_size"]
        hno_si = hno_cfg.get("save_interval", 500)
        dec2_cfg = config["decoder2"]; dec2_ep = dec2_cfg["num_epochs"]; dec2_lr = dec2_cfg["learning_rate"]
        dec2_bs = dec2_cfg["batch_size"]; dec2_base_w = dec2_cfg.get("base_loss_weight", 1.0)
        dec2_si = dec2_cfg.get("save_interval", 500)
        d2s = config["decoder2_settings"]; d2_c_mode = d2s.get("conditioner_mode","z_ref")
        d2_p_type = d2s.get("pooling_type","blind"); d2_ph, d2_pw = d2s.get("output_height",20), d2s.get("output_width",4)
        d2_mlp_h = d2s.get("mlp_hidden_dim",128); d2_mlp_l = d2s.get("num_hidden_layers",2)
        d2_use_p2 = d2s.get("use_second_level_pooling",False); d2_ph2, d2_pw2 = d2s.get("output_height2"), d2s.get("output_width2")
        di_cfg = config.get("dihedral_loss",{}); use_di_loss = di_cfg.get("use_dihedral_loss",False)
        torsion_p = di_cfg.get("torsion_info_path","torsion.json"); l_div = di_cfg.get("lambda_divergence",0.0)
        l_mse = di_cfg.get("lambda_torsion_mse",0.0); div_t = di_cfg.get("divergence_type","KL").upper()
        frac_di = di_cfg.get("fraction_dihedral", 0.1)
        out_cfg = config.get("output_directories",{}); ckpt_dir = out_cfg.get("checkpoint_dir","checkpoints")
        struct_dir = out_cfg.get("structure_dir","structures"); latent_dir = out_cfg.get("latent_dir","latent_reps")
    except KeyError as e_key: logger.error(f"Missing config key: {e_key}"); sys.exit(1)

    # Device Setup
    device = torch.device("cpu")
    if not force_cpu and torch.cuda.is_available():
        try: device = torch.device(f"cuda:{cuda_idx}"); torch.cuda.get_device_name(cuda_idx)
        except Exception: logger.warning(f"CUDA device {cuda_idx} invalid. Using cuda:0 or CPU."); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    pin_mem = (device.type == "cuda")
    try: [os.makedirs(d, exist_ok=True) for d in [ckpt_dir, struct_dir, latent_dir]]
    except OSError as e_dir: logger.error(f"Dir creation error: {e_dir}"); sys.exit(1)

    # Stage 1: Data Loading & Prep
    logger.info("--- Stage 1: Data Loading & Preprocessing ---")
    _, atoms_ord_list = parse_pdb(pdb_p, logger);
    if not atoms_ord_list: logger.error("PDB parsing failed."); sys.exit(1)
    renum_res_dict, _ = renumber_atoms_and_residues(atoms_ord_list, logger)
    bb_idx_list, sc_idx_list = get_global_indices(renum_res_dict); N_atoms_pdb = len(bb_idx_list)+len(sc_idx_list)
    bb_t, sc_t = torch.tensor(bb_idx_list, dtype=torch.long), torch.tensor(sc_idx_list, dtype=torch.long)
    coords_frames, N_atoms_json = load_heavy_atom_coords_from_json(json_p, logger)
    if not coords_frames or N_atoms_json!=N_atoms_pdb: logger.error("JSON coord loading or atom count mismatch."); sys.exit(1)
    N_total_atoms = N_atoms_pdb
    aligned_coords = align_frames_to_first(coords_frames, logger, device)
    if not aligned_coords: logger.error("Frame alignment failed."); sys.exit(1)
    full_orig_pyg_dset = build_graph_dataset(aligned_coords, knn, logger, device) # knn was knn_k_val
    if not full_orig_pyg_dset: logger.error("PyG dataset build failed."); sys.exit(1)

    # Stage 2: HNO Encoder
    logger.info("--- Stage 2: HNO Encoder ---")
    tr_hno_dset, val_hno_dset = train_test_split(full_orig_pyg_dset, test_size=0.1, random_state=42)
    tr_hno_loader = PyGDataLoader(tr_hno_dset, hno_bs, shuffle=True, num_workers=num_workers, pin_memory=pin_mem, drop_last=True)
    val_hno_loader = PyGDataLoader(val_hno_dset, hno_bs, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    hno = HNO(hno_hdim, hno_k_order); hno_ckpt_path = os.path.join(ckpt_dir, "hno_encoder_checkpoint.pth")
    hno = train_hno_model(hno, tr_hno_loader, val_hno_loader, bb_t, sc_t, hno_ep, hno_lr, hno_ckpt_path, hno_si, device, logger)
    hno.eval()

    # Stage 3: Decoder Input Data (Embeddings)
    logger.info("--- Stage 3: Preparing Decoder2 Input Dataset (Atom Embeddings) ---")
    dec_in_pyg_dset: List[PyGData] = []
    # Using PyGDataLoader for batching during embedding generation
    all_frames_loader_for_embed = PyGDataLoader(full_orig_pyg_dset, batch_size=hno_bs*2, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    with torch.no_grad():
        for orig_batch_data in all_frames_loader_for_embed:
            orig_batch_data = orig_batch_data.to(device)
            atom_embeds_batch = hno.forward_representation(orig_batch_data.x, orig_batch_data.edge_index)
            target_coords_batch = orig_batch_data.y # Ground truth coordinates
            node_counts = orig_batch_data.ptr[1:] - orig_batch_data.ptr[:-1]
            if node_counts.numel() > 0 and node_counts.sum() == atom_embeds_batch.shape[0]:
                list_embeds = torch.split(atom_embeds_batch, node_counts.tolist())
                list_coords = torch.split(target_coords_batch, node_counts.tolist())
                for emb, coords_y in zip(list_embeds, list_coords):
                    dec_in_pyg_dset.append(PyGData(x=emb.cpu(), y=coords_y.cpu())) # Store on CPU
            elif node_counts.numel() == 0 and atom_embeds_batch.shape[0] == 0: pass # Empty batch
            else: logger.warning("Node count mismatch in embedding generation batch.")
    if not dec_in_pyg_dset: logger.error("Decoder input dataset (embeddings) is empty!"); sys.exit(1)
    logger.info(f"Decoder2 input dataset (embeddings) created: {len(dec_in_pyg_dset)} samples.")

    # Stage 4: Decoder2
    logger.info("--- Stage 4: Decoder2 ---")
    tr_dec_dset, val_dec_dset = train_test_split(dec_in_pyg_dset, test_size=0.1, random_state=42)
    tr_dec_loader = PyGDataLoader(tr_dec_dset, dec2_bs, shuffle=True, num_workers=num_workers, pin_memory=pin_mem, drop_last=True)
    val_dec_loader = PyGDataLoader(val_dec_dset, dec2_bs, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    first_frame_data_dev = full_orig_pyg_dset[0].to(device) # For conditioner source
    Xref_coords_dev = first_frame_data_dev.x
    cond_tensor_cpu: Optional[torch.Tensor] = None; cond_dim_val = -1; zref_embed_cpu: Optional[torch.Tensor] = None
    if d2_c_mode.lower() == "x_ref":
        cond_tensor_cpu = Xref_coords_dev.cpu(); cond_dim_val = 3
    elif d2_c_mode.lower() == "z_ref":
        with torch.no_grad(): zref_dev = hno.forward_representation(Xref_coords_dev, first_frame_data_dev.edge_index)
        cond_tensor_cpu = zref_dev.cpu(); zref_embed_cpu = cond_tensor_cpu; cond_dim_val = zref_dev.shape[1]
    else: logger.error(f"Invalid conditioner_mode: {d2_c_mode}"); sys.exit(1)
    if cond_tensor_cpu is None : logger.error(f"Conditioner tensor is None!"); sys.exit(1)
    try: torch.save(Xref_coords_dev.cpu(), os.path.join(struct_dir,"X_ref_coords.pt")); logger.info("Saved X_ref.")
    except Exception as e: logger.error(f"Save X_ref failed: {e}")
    if zref_embed_cpu is not None:
      try: torch.save(zref_embed_cpu, os.path.join(latent_dir,"z_ref_embedding.pt")); logger.info("Saved z_ref.")
      except Exception as e: logger.error(f"Save z_ref failed: {e}")


    di_info_loss: Dict = {}; di_mask_loss: Optional[torch.Tensor] = None; N_res_loss: Optional[int] = None; use_di_final = False
    if use_di_loss:
        torsion_abs_p = os.path.abspath(torsion_p)
        if os.path.isfile(torsion_abs_p):
            try:
                with open(torsion_abs_p,"r") as f_t: t_json_data=json.load(f_t)
                angle_list=['phi','psi','chi1','chi2','chi3','chi4','chi5']
                idx_ll,res_ll = {n:[[] for _ in range(4)] for n in angle_list},{n:[] for n in angle_list}
                sorted_res_keys = sorted(t_json_data.keys(), key=int); N_res_loss=len(sorted_res_keys)
                mask_ll = [[False]*len(angle_list) for _ in range(N_res_loss)]; skip_c=0
                for r_i_order, r_id_s in enumerate(sorted_res_keys):
                    r_entry=t_json_data.get(r_id_s,{}); t_atoms=r_entry.get("torsion_atoms",{}); c_atoms=t_atoms.get("chi",{})
                    for type_i, name_a in enumerate(angle_list):
                        indices_q = t_atoms.get(name_a) if name_a in ['phi','psi'] else c_atoms.get(name_a)
                        if isinstance(indices_q,list) and len(indices_q)==4 and all(isinstance(i_val,int) for i_val in indices_q):
                            if all(0<=i_val<N_total_atoms for i_val in indices_q):
                                for k_q,atom_i_val in enumerate(indices_q): idx_ll[name_a][k_q].append(atom_i_val)
                                res_ll[name_a].append(r_i_order); mask_ll[r_i_order][type_i]=True
                            else: skip_c+=1
                if skip_c > 0: logger.warning(f"Skipped {skip_c} angles (out-of-bounds atom indices in torsion JSON).")
                for name_a_iter in angle_list:
                    if res_ll[name_a_iter]: di_info_loss[name_a_iter] = {'indices': [torch.tensor(l_idx,dtype=torch.long) for l_idx in idx_ll[name_a_iter]], 'res_idx': torch.tensor(res_ll[name_a_iter],dtype=torch.long)}
                    else: di_info_loss[name_a_iter] = {'indices': None, 'res_idx': None}
                di_mask_loss=torch.tensor(mask_ll,dtype=torch.bool); use_di_final=True
                logger.info(f"Dihedral info precomputed for {N_res_loss} residues.")
            except Exception as e_di: logger.error(f"Dihedral JSON processing error: {e_di}", exc_info=True)
        else: logger.warning(f"Torsion file not found: {torsion_abs_p}. Dihedral loss disabled.")
    if use_di_loss and not use_di_final: logger.warning("Dihedral loss configured but precomp failed. Disabled.")

    res_indices_for_pool: Optional[List[List[int]]] = None
    if d2_p_type.lower() == "residue":
        logger.info("Residue pooling selected. Preparing residue segment indices.")
        # Assuming renum_res_dict maps: new_res_id (int) -> {"backbone": [global_atom_indices], "sidechain": [...]}
        if renum_res_dict:
             res_indices_for_pool = [
                 sorted(renum_res_dict[k]['backbone'] + renum_res_dict[k]['sidechain'])
                 for k in sorted(renum_res_dict.keys()) # Iterate in order of new residue IDs
             ]
             if not all(res_indices_for_pool): # Check for empty residue segments
                 logger.warning("Some residue segments are empty after processing for pooling.")
        else: logger.error("Cannot prepare residue indices for pooling: renumbered_res_dict is empty.")


    sec_pool_dims = (d2_ph2, d2_pw2) if d2_use_p2 and d2_ph2 is not None and d2_pw2 is not None else None
    dec2 = ProteinStateReconstructor2D(hno_hdim, N_total_atoms, cond_dim_val, d2_p_type, res_indices_for_pool,
                                     (d2_ph,d2_pw), d2_mlp_h, d2_mlp_l,
                                     use_secondary_pool=d2_use_p2, secondary_pool_output_size=sec_pool_dims,
                                     logger_instance=logger)
    dec2_ckpt_path = os.path.join(ckpt_dir, "decoder2_checkpoint.pth")
    dec2 = train_decoder2_model(dec2, tr_dec_loader, val_dec_loader, bb_t, sc_t, cond_tensor_cpu, N_total_atoms,
                                dec2_ep, dec2_lr, dec2_ckpt_path, dec2_si, device, logger, dec2_base_w,
                                use_di_final, di_info_loss, di_mask_loss, N_res_loss, l_div, l_mse, div_t, frac_di)
    dec2.eval()

    # Stage 5: Export Results
    logger.info("--- Stage 5: Exporting Autoencoder Outputs ---")
    export_final_outputs(hno, dec2, full_orig_pyg_dset, dec_in_pyg_dset, cond_tensor_cpu, N_total_atoms,
                         struct_dir, latent_dir, device, logger)

    elapsed = time.time() - script_start_time
    logger.info(f"================ Autoencoder Workflow Finished ({time.strftime('%H:%M:%S', time.gmtime(elapsed))}) ================")
    sys.stdout.flush()

# --- Script Entry Point ---
if __name__ == "__main__":
    main_workflow()
