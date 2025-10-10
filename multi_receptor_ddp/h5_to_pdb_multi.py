#!/usr/bin/env python3
import argparse
import os
import yaml
import logging
import h5py
import numpy as np
import pathlib
import mdtraj as md

# ===================================================================
# (A) Argument Parsing & Setup
# ===================================================================

parser = argparse.ArgumentParser(description="Multi-System HDF5 to PDB Converter")
parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file.')
parser.add_argument('--debug', action='store_true', help='Enable debug level logging.')
args = parser.parse_args()

# Setup logging
log_level = logging.DEBUG if args.debug else logging.INFO
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler()])
logger = logging.getLogger()

logger.info("HDF5 to PDB Converter Script Started")

# ===================================================================
# (B) Configuration Loading
# ===================================================================

config_path = pathlib.Path(args.config)
if not config_path.is_file():
    logger.error(f"Configuration file not found: {args.config}"); exit(1)

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    logger.info(f"Loaded configuration from {config_path}")

# Extract parameters
paths = config.get('paths', {})
settings = config.get('settings', {})

H5_INPUT_PATH = pathlib.Path(paths['generated_h5_path'])
PDB_OUTPUT_DIR = pathlib.Path(paths['pdb_output_dir'])
TEMPLATE_PDB_PATHS = paths.get('template_pdb_paths', [])

FRAMES_PER_SYSTEM = settings.get('max_frames_to_save', 100)

# ===================================================================
# (C) Main Conversion Logic
# ===================================================================

def main():
    if not H5_INPUT_PATH.is_file():
        logger.error(f"Input HDF5 file not found: {H5_INPUT_PATH}"); exit(1)
    
    if not TEMPLATE_PDB_PATHS:
        logger.error("No 'template_pdb_paths' provided in the configuration. Cannot create PDBs."); exit(1)

    PDB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"PDB files will be saved to: {PDB_OUTPUT_DIR.resolve()}")

    # Load all template topologies into memory
    templates = {}
    for i, pdb_path in enumerate(TEMPLATE_PDB_PATHS):
        try:
            templates[i] = md.load_pdb(pdb_path)
            logger.info(f"Loaded template for System ID {i} from {pdb_path}")
        except Exception as e:
            logger.error(f"Could not load template PDB for System ID {i} from {pdb_path}: {e}")
    
    if not templates:
        logger.error("Failed to load any valid PDB templates. Aborting."); exit(1)

    # Open the HDF5 file and process each system and experiment
    with h5py.File(H5_INPUT_PATH, 'r') as f:
        logger.info(f"Scanning HDF5 file: {H5_INPUT_PATH}")
        
        # Use visititems to traverse the nested structure
        f.visititems(lambda name, obj: convert_dataset(name, obj, templates))

    logger.info("--- PDB file generation complete. ---")

def convert_dataset(name, obj, templates):
    """A callback function for h5py's visititems to process each dataset."""
    # We only care about datasets named 'coords'
    if not name.endswith('/coords') or not isinstance(obj, h5py.Dataset):
        return

    try:
        path_parts = name.split('/')
        # Expected path: 'system_0/exp_1/coords'
        if len(path_parts) != 3:
            logger.warning(f"Skipping dataset with unexpected path structure: {name}")
            return

        sys_group_name, exp_group_name, _ = path_parts
        sid = int(sys_group_name.replace('system_', ''))
        
        if sid not in templates:
            logger.warning(f"No PDB template found for System ID {sid} (from path {name}). Skipping.")
            return

        logger.info(f"--- Processing {name} ---")
        
        coords_data = obj[:]
        num_frames, num_atoms, _ = coords_data.shape
        
        template_traj = templates[sid]
        if template_traj.n_atoms != num_atoms:
            logger.error(f"Atom count mismatch for {name}! Template has {template_traj.n_atoms}, data has {num_atoms}. Skipping.")
            return

        # Create a nested directory for this system's PDBs
        output_dir = PDB_OUTPUT_DIR / sys_group_name / exp_group_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine how many frames to save
        frames_to_save = min(num_frames, FRAMES_PER_SYSTEM)
        logger.info(f"Saving {frames_to_save} frames out of {num_frames} available.")

        # Create and save PDB files
        for i in range(frames_to_save):
            frame_coords_nm = coords_data[i] * 0.1
            new_traj = md.Trajectory([frame_coords_nm], topology=template_traj.topology)
            output_pdb_path = output_dir / f"generated_frame_{i+1}.pdb"
            new_traj.save_pdb(str(output_pdb_path))

        logger.info(f"Successfully saved {frames_to_save} PDB files to {output_dir}")

    except Exception as e:
        logger.error(f"Failed to process dataset at path {name}: {e}", exc_info=True)



if __name__ == "__main__":
    main()
