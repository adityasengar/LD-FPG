# export_utils.py
# Description: Functionality to export final ground truth data, model reconstructions,
# and learned latent representations to HDF5 files.

from common_imports import (
    os, h5py, torch, logging,
    List, Dict, Any, # from typing
    PyGData # from torch_geometric
)

# Custom project module imports
from models import HNO, ProteinStateReconstructor2D # For type hinting

logger_export = logging.getLogger(__name__)

@torch.no_grad()
def export_final_outputs(
    hno_encoder_model: HNO, decoder2_model: ProteinStateReconstructor2D,
    full_original_dataset: List[PyGData], decoder_input_dataset: List[PyGData],
    conditioner_tensor_cpu: torch.Tensor, num_total_atoms: int,
    output_structure_dir: str, output_latent_dir: str,
    device: torch.device, logger: logging.Logger
):
    logger.info("--- Starting Export of Final Outputs ---")
    hno_encoder_model.eval().to(device)
    decoder2_model.eval().to(device)
    conditioner_on_device = conditioner_tensor_cpu.to(device)

    output_paths = {
        'gt_coords': os.path.join(output_structure_dir, "ground_truth_aligned.h5"),
        'hno_reconstructed_coords': os.path.join(output_structure_dir, "hno_reconstructions.h5"),
        'decoder_decoded_coords': os.path.join(output_structure_dir, "full_decoded_coords.h5"),
        'hno_atom_embeddings': os.path.join(output_latent_dir, "hno_atom_embeddings.h5"),
        'decoder_pooled_latents': os.path.join(output_latent_dir, "decoder_pooled_latents.h5")
    }
    num_frames = len(full_original_dataset)
    if num_frames == 0 or len(decoder_input_dataset) != num_frames:
        logger.warning(f"Dataset empty or mismatched. Skipping export.")
        return

    atom_embedding_dim = hno_encoder_model.conv4.out_channels
    pooled_latent_dim = 0
    if decoder_input_dataset and decoder_input_dataset[0].x is not None:
        try:
            sample_emb = decoder_input_dataset[0].x.to(device)
            if sample_emb.shape[0] == num_total_atoms: # Ensure it's for a single structure
                 pooled_latent_dim = decoder2_model.get_pooled_latent(sample_emb).shape[1] # Input is [N,E] for B=1
            else:
                 logger.warning(f"Sample embedding shape {sample_emb.shape} mismatch with N_atoms {num_total_atoms} for pooled_dim check.")
        except Exception as e: logger.error(f"Could not get pooled_dim: {e}.", exc_info=True)
    logger.info(f"Export Params: Frames={num_frames}, Atoms={num_total_atoms}, EmbDim={atom_embedding_dim}, PoolDim={pooled_latent_dim}")

    hdf5_files: Dict[str, Any] = {}; dset_flags: Dict[str, bool] = {k: False for k in output_paths}
    try:
        for key, path_str in output_paths.items(): hdf5_files[key] = h5py.File(path_str, "w")
        dset_refs: Dict[str, Any] = {}
        if num_total_atoms > 0:
            dset_refs['gt'] = hdf5_files['gt_coords'].create_dataset("ground_truth_coords",(num_frames,num_total_atoms,3),dtype='f4'); dset_flags['gt_coords']=True
            dset_refs['hno_rec'] = hdf5_files['hno_reconstructed_coords'].create_dataset("hno_coords",(num_frames,num_total_atoms,3),dtype='f4'); dset_flags['hno_reconstructed_coords']=True
            dset_refs['dec2_rec'] = hdf5_files['decoder_decoded_coords'].create_dataset("full_coords",(num_frames,num_total_atoms,3),dtype='f4'); dset_flags['decoder_decoded_coords']=True
        if num_total_atoms > 0 and atom_embedding_dim > 0:
            dset_refs['hno_emb'] = hdf5_files['hno_atom_embeddings'].create_dataset("hno_embeddings",(num_frames,num_total_atoms,atom_embedding_dim),dtype='f4'); dset_flags['hno_atom_embeddings']=True
        if pooled_latent_dim > 0:
            dset_refs['pool_emb'] = hdf5_files['decoder_pooled_latents'].create_dataset("pooled_embedding",(num_frames,pooled_latent_dim),dtype='f4'); dset_flags['decoder_pooled_latents']=True

        for i in range(num_frames):
            orig_data, dec_in_data = full_original_dataset[i].to(device), decoder_input_dataset[i].to(device)
            if dset_refs.get('gt'): dset_refs['gt'][i] = orig_data.y.cpu().numpy()
            if dset_refs.get('hno_rec'): dset_refs['hno_rec'][i] = hno_encoder_model(orig_data.x, orig_data.edge_index).cpu().numpy()
            if dset_refs.get('hno_emb'): dset_refs['hno_emb'][i] = dec_in_data.x.cpu().numpy() # Embeddings are dec_in_data.x
            if dset_refs.get('dec2_rec'): dset_refs['dec2_rec'][i] = decoder2_model(dec_in_data.x, None, conditioner_on_device).cpu().numpy()
            if dset_refs.get('pool_emb'): dset_refs['pool_emb'][i] = decoder2_model.get_pooled_latent(dec_in_data.x).squeeze(0).cpu().numpy()
            if (i + 1) % 500 == 0 or (i + 1) == num_frames: logger.info(f"Exported frame {i + 1}/{num_frames}")
    except Exception as e: logger.error(f"HDF5 export failed: {e}", exc_info=True)
    finally:
        for f_handle in hdf5_files.values():
            if f_handle: try: f_handle.close()
            except Exception as ce: logger.error(f"Error closing HDF5: {ce}")

    logger.info("Export finished. Verifying files:")
    # (File verification logic from your original script can be adapted here, using dset_flags)
    for key_path, (name_desc, file_path_val) in {
        'gt_coords': ("GT Coords", output_paths['gt_coords']),
        'hno_rec': ("HNO Recon", output_paths['hno_reconstructed_coords']),
        'dec2_rec': ("Dec2 Recon", output_paths['decoder_decoded_coords']),
        'hno_emb': ("HNO Embeds", output_paths['hno_atom_embeddings']),
        'pool_emb': ("Pooled Embeds", output_paths['decoder_pooled_latents'])
    }.items(): # Use the internal keys for dset_refs if those differ from output_paths keys
        exists = os.path.isfile(file_path_val)
        size_ok = os.path.getsize(file_path_val) > 50 if exists else False
        created_flag = dset_flags.get(key_path, False) # Check if creation was attempted based on valid dims

        if exists and size_ok: logger.info(f"  {name_desc} -> {file_path_val} (Exists, OK)")
        elif exists and not size_ok: logger.warning(f"  {name_desc} -> {file_path_val} (Exists but small/empty!)")
        elif not exists and created_flag: logger.warning(f"  {name_desc} -> {file_path_val} (NOT found, but creation was expected!)")
        elif not exists and not created_flag: logger.info(f"  {name_desc} -> {file_path_val} (Not found, not created due to dims)")
        else: logger.warning(f"  {name_desc} -> {file_path_val} (Status unclear)")
