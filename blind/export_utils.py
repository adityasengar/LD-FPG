import os
import h5py
import torch
import logging
from typing import List, Dict, Any

# Assuming models.py contains HNO and ProteinStateReconstructor2D definitions
# and torch_geometric.data.Data is used.
from torch_geometric.data import Data # Or from wherever Data is imported
from models import HNO, ProteinStateReconstructor2D # Make sure these are accessible

logger_export = logging.getLogger(__name__) # Module-specific logger

@torch.no_grad()
def export_final_outputs(
    hno_encoder_model: HNO,
    decoder2_model: ProteinStateReconstructor2D,
    full_original_dataset: List[Data], # Dataset with original coords for GT and HNO input
    decoder_input_dataset: List[Data], # Dataset with embeddings for Decoder2 input
    conditioner_tensor_cpu: torch.Tensor, # X_ref or z_ref, expected on CPU
    num_total_atoms: int,
    output_structure_dir: str,
    output_latent_dir: str,
    device: torch.device,
    logger: logging.Logger # Pass logger instance
):
    """
    Exports final predictions and intermediate results to HDF5 files.
    Writes:
     1) ground_truth_aligned.h5 (original aligned coordinates)
     2) hno_reconstructions.h5  (coordinates reconstructed by HNO encoder's MLP head)
     3) full_decoded_coords.h5 (coordinates from Decoder2 using HNO embeddings)
     4) hno_atom_embeddings.h5  (atom-wise embeddings from HNO encoder)
     5) decoder_pooled_latents.h5 (pooled latent embeddings from Decoder2's pooling stage)
    """
    logger.info("--- Starting Export of Final Outputs ---")
    hno_encoder_model.eval().to(device)
    decoder2_model.eval().to(device)
    conditioner_on_device = conditioner_tensor_cpu.to(device)

    # Define output file paths
    output_paths = {
        'gt_coords': os.path.join(output_structure_dir, "ground_truth_aligned.h5"),
        'hno_reconstructed_coords': os.path.join(output_structure_dir, "hno_reconstructions.h5"),
        'decoder_decoded_coords': os.path.join(output_structure_dir, "full_decoded_coords.h5"),
        'hno_atom_embeddings': os.path.join(output_latent_dir, "hno_atom_embeddings.h5"),
        'decoder_pooled_latents': os.path.join(output_latent_dir, "decoder_pooled_latents.h5")
    }

    num_frames = len(full_original_dataset)
    if num_frames == 0 or len(decoder_input_dataset) != num_frames:
        logger.warning(f"Dataset empty (Full: {num_frames}, DecoderInput: {len(decoder_input_dataset)}) or mismatched. Skipping export.")
        return

    # Determine dimensions for HDF5 datasets
    # Atom embedding dimension from HNO (e.g., output of conv4)
    atom_embedding_dim = hno_encoder_model.conv4.out_channels # Accessing directly, ensure this is robust
    
    # Pooled latent dimension from Decoder2
    pooled_latent_dim = 0
    if decoder_input_dataset:
        try:
            # Get a sample embedding (decoder_input_dataset[0].x is [N_atoms, atom_embedding_dim])
            sample_atom_embeddings_for_pooling = decoder_input_dataset[0].x.to(device)
            # The get_pooled_latent method expects [B*N, E], so simulate B=1
            pooled_latent_dim = decoder2_model.get_pooled_latent(sample_atom_embeddings_for_pooling).shape[1]
        except Exception as e:
            logger.error(f"Could not determine pooled_latent_dim from decoder2_model.get_pooled_latent(): {e}", exc_info=True)
            logger.warning("Pooled latent dimension could not be determined. Skipping pooled latent export if dim is 0.")

    logger.info(f"Export Parameters: NumFrames={num_frames}, NumAtoms={num_total_atoms}, "
                f"AtomEmbeddingDim={atom_embedding_dim}, PooledLatentDim={pooled_latent_dim}")

    # HDF5 file handling
    hdf5_files_opened: Dict[str, Any] = {} # Using Any for h5py.File type hint flexibility
    hdf5_datasets_created_flags: Dict[str, bool] = {key: False for key in output_paths}

    try:
        # Open all HDF5 files
        for key, path_str in output_paths.items():
            hdf5_files_opened[key] = h5py.File(path_str, "w")
        logger.debug("HDF5 files opened for writing.")

        # Create datasets within HDF5 files
        dset_refs: Dict[str, Any] = {}
        if num_total_atoms > 0:
            dset_refs['gt_coords'] = hdf5_files_opened['gt_coords'].create_dataset(
                "ground_truth_coords", (num_frames, num_total_atoms, 3), dtype='f4')
            hdf5_datasets_created_flags['gt_coords'] = True

            dset_refs['hno_reconstructed_coords'] = hdf5_files_opened['hno_reconstructed_coords'].create_dataset(
                "hno_coords", (num_frames, num_total_atoms, 3), dtype='f4')
            hdf5_datasets_created_flags['hno_reconstructed_coords'] = True

            dset_refs['decoder_decoded_coords'] = hdf5_files_opened['decoder_decoded_coords'].create_dataset(
                "full_coords", (num_frames, num_total_atoms, 3), dtype='f4')
            hdf5_datasets_created_flags['decoder_decoded_coords'] = True

        if num_total_atoms > 0 and atom_embedding_dim > 0:
            dset_refs['hno_atom_embeddings'] = hdf5_files_opened['hno_atom_embeddings'].create_dataset(
                "hno_embeddings", (num_frames, num_total_atoms, atom_embedding_dim), dtype='f4')
            hdf5_datasets_created_flags['hno_atom_embeddings'] = True

        if pooled_latent_dim > 0:
            dset_refs['decoder_pooled_latents'] = hdf5_files_opened['decoder_pooled_latents'].create_dataset(
                "pooled_embedding", (num_frames, pooled_latent_dim), dtype='f4') # Note the key from original
            hdf5_datasets_created_flags['decoder_pooled_latents'] = True
        logger.debug("HDF5 datasets created (or skipped if dims were zero).")

        # Iterate through data and write to HDF5
        for i in range(num_frames):
            original_data_item = full_original_dataset[i].to(device) # Contains original .x and .edge_index
            decoder_input_item = decoder_input_dataset[i].to(device) # Contains .x as embeddings, .y as original coords

            # 1. Ground Truth Coordinates
            if dset_refs.get('gt_coords'):
                dset_refs['gt_coords'][i] = original_data_item.y.cpu().numpy() # .y should be coords

            # 2. HNO Reconstructed Coordinates (from original coords input to HNO)
            if dset_refs.get('hno_reconstructed_coords'):
                hno_rec_coords = hno_encoder_model(original_data_item.x, original_data_item.edge_index)
                dset_refs['hno_reconstructed_coords'][i] = hno_rec_coords.cpu().numpy()

            # 3. Decoder Decoded Coordinates (from HNO embeddings input to Decoder2)
            if dset_refs.get('decoder_decoded_coords'):
                # Decoder2 takes flat embeddings [N,E], conditioner [N,E_cond]
                # decoder_input_item.x is [N, atom_embedding_dim]
                full_decoded_coords = decoder2_model(decoder_input_item.x, None, conditioner_on_device) # batch_indices=None for single graph
                dset_refs['decoder_decoded_coords'][i] = full_decoded_coords.cpu().numpy()

            # 4. HNO Atom-wise Embeddings (these are the .x in decoder_input_dataset)
            if dset_refs.get('hno_atom_embeddings'):
                dset_refs['hno_atom_embeddings'][i] = decoder_input_item.x.cpu().numpy()

            # 5. Decoder Pooled Latents
            if dset_refs.get('decoder_pooled_latents'):
                # get_pooled_latent expects [B*N, E], so for B=1, it's [N, E]
                pooled_latent = decoder2_model.get_pooled_latent(decoder_input_item.x) # Output [1, pooled_dim]
                dset_refs['decoder_pooled_latents'][i] = pooled_latent.squeeze(0).cpu().numpy()


            if (i + 1) % 500 == 0 or (i + 1) == num_frames:
                logger.info(f"Exported data for frame {i + 1}/{num_frames}")

    except Exception as e:
        logger.error(f"Error during HDF5 export processing: {e}", exc_info=True)
    finally:
        for key, f_handle in hdf5_files_opened.items():
            try:
                if f_handle: # Check if it's a valid file object
                    f_handle.close()
            except Exception as close_e:
                logger.error(f"Error closing HDF5 file '{output_paths[key]}': {close_e}", exc_info=False)

    # Verify file existence and basic integrity
    logger.info("Export process finished. Verifying file existence and basic integrity:")
    for key, (name_desc, file_path) in {
        'gt_coords': ("Ground Truth Coords", output_paths['gt_coords']),
        'hno_reconstructed_coords': ("HNO Recon Coords", output_paths['hno_reconstructed_coords']),
        'decoder_decoded_coords': ("Decoder Decoded Coords", output_paths['decoder_decoded_coords']),
        'hno_atom_embeddings': ("HNO Atom Embeddings", output_paths['hno_atom_embeddings']),
        'decoder_pooled_latents': ("Decoder Pooled Latents", output_paths['decoder_pooled_latents'])
    }.items():
        try:
            if os.path.isfile(file_path):
                # Check if dataset was supposed to be created based on dimensions
                should_exist_flag = False
                if key in ['gt_coords', 'hno_reconstructed_coords', 'decoder_decoded_coords'] and num_total_atoms > 0:
                    should_exist_flag = True
                elif key == 'hno_atom_embeddings' and num_total_atoms > 0 and atom_embedding_dim > 0:
                    should_exist_flag = True
                elif key == 'decoder_pooled_latents' and pooled_latent_dim > 0:
                    should_exist_flag = True
                
                if not should_exist_flag and hdf5_datasets_created_flags.get(key, False): # Exists but wasn't expected from flags
                     logger.warning(f"  {name_desc} -> {file_path} (Exists but possibly based on incorrect zero dim assumption during creation check)")
                elif not should_exist_flag and not hdf5_datasets_created_flags.get(key, False): # Does not exist and wasn't expected
                     logger.info(f"  {name_desc} -> {file_path} (Not created, as expected due to zero dimensions)")
                elif os.path.getsize(file_path) > 50: # Basic check for non-empty file
                    logger.info(f"  {name_desc} -> {file_path} (Exists and seems non-empty)")
                else: # Exists but is very small
                    logger.warning(f"  {name_desc} -> {file_path} (Exists but is suspiciously small or empty!)")
            else: # File does not exist
                if hdf5_datasets_created_flags.get(key, False): # Was expected to be created
                    logger.warning(f"  {name_desc} -> {file_path} (File NOT found, but dataset creation was attempted!)")
                else: # Not found and not expected (e.g. due to zero dim)
                    logger.info(f"  {name_desc} -> {file_path} (File NOT found, likely not created due to zero dimensions)")
        except OSError as e:
            logger.error(f"  Error checking file status for {name_desc} at {file_path}: {e}")
        except Exception as e:
            logger.error(f"  Unexpected error checking file {name_desc} at {file_path}: {e}", exc_info=True)
