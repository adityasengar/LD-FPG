#!/usr/bin/env python3
import argparse
import os
import logging
import h5py
import numpy as np
import pathlib
import mdtraj as md

# ===================================================================
# (A) Argument Parsing & Setup
# ===================================================================

parser = argparse.ArgumentParser(description="Convert Novel Generated HDF5 Coordinates to PDB Files")
parser.add_argument('--input', type=str, required=True, help='Path to the HDF5 file containing the generated coordinates.')
parser.add_argument('--template_pdb', type=str, required=True, help='Path to the original PDB file that was used for generation, to be used as a template.')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output PDB files.')
parser.add_argument('--max_frames', type=int, default=100, help='The maximum number of PDB files to save from the HDF5 file.')
parser.add_argument('--debug', action='store_true', help='Enable debug level logging.')
args = parser.parse_args()

# Setup logging
log_level = logging.DEBUG if args.debug else logging.INFO
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler()])
logger = logging.getLogger()

logger.info("HDF5 to PDB Converter for Novel Structures Started")

# ===================================================================
# (B) Main Conversion Logic
# ===================================================================

def main():
    h5_input_path = pathlib.Path(args.input)
    template_pdb_path = pathlib.Path(args.template_pdb)
    pdb_output_dir = pathlib.Path(args.output_dir)

    if not h5_input_path.is_file():
        logger.error(f"Input HDF5 file not found: {h5_input_path}"); exit(1)
    
    if not template_pdb_path.is_file():
        logger.error(f"Template PDB file not found: {template_pdb_path}"); exit(1)

    pdb_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"PDB files will be saved to: {pdb_output_dir.resolve()}")

    # Load the template topology
    try:
        template_traj = md.load_pdb(str(template_pdb_path))
        # We need the heavy-atom topology to match the generated coordinates
        heavy_indices = template_traj.topology.select('not element H')
        template_topology = template_traj.atom_slice(heavy_indices, inplace=False).topology
        logger.info(f"Loaded template topology from {template_pdb_path} ({template_topology.n_atoms} heavy atoms)")
    except Exception as e:
        logger.error(f"Could not load template PDB: {e}"); exit(1)

    # Open the HDF5 file and process the coordinates
    with h5py.File(h5_input_path, 'r') as f:
        if 'generated_coords' not in f:
            logger.error(f"Dataset 'generated_coords' not found in {h5_input_path}. Aborting."); exit(1)
            
        coords_data = f['generated_coords'][:]
        num_frames, num_atoms, _ = coords_data.shape
        
        if template_topology.n_atoms != num_atoms:
            logger.error(f"Atom count mismatch! Template has {template_topology.n_atoms}, generated data has {num_atoms}. Cannot proceed.")
            exit(1)

        frames_to_save = min(num_frames, args.max_frames)
        logger.info(f"Saving {frames_to_save} frames out of {num_frames} available.")

        for i in range(frames_to_save):
            try:
                # Coordinates are in Angstroms, convert to nm for mdtraj
                frame_coords_nm = coords_data[i] * 0.1
                
                new_traj = md.Trajectory([frame_coords_nm], topology=template_topology)
                
                output_pdb_path = pdb_output_dir / f"generated_novel_frame_{i+1}.pdb"
                new_traj.save_pdb(str(output_pdb_path))
            except Exception as e:
                logger.error(f"Failed to save frame {i+1}: {e}")

    logger.info(f"--- PDB file generation complete. Saved {frames_to_save} files. ---")

if __name__ == "__main__":
    main()
