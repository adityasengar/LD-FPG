#!/usr/bin/env python3

import numpy as np
import h5py
import argparse
import os

def read_and_reshape_h5(file_path, key='reconstructions', reshape_dim=(2191, 3)):
    """
    Reads the dataset with the given key from the HDF5 file.
    If the loaded data is 2D, reshapes it to (N, *reshape_dim).
    Otherwise, returns the data as is.
    """
    with h5py.File(file_path, 'r') as hf:
        data = np.array(hf[key])
        print(f"Data shape from {file_path} (key: {key}): {data.shape}")
    if len(data.shape) == 2:
        # Assume data is concatenated coordinates for multiple structures.
        N = data.shape[0] // reshape_dim[0]
        reshaped_data = data.reshape(N, *reshape_dim)
        return reshaped_data
    else:
        return data

def load_pdb(pdb_path):
    """
    Reads a PDB file and returns its full lines and the atom lines (those starting with ATOM or HETATM).
    """
    with open(pdb_path, 'r') as file:
        lines = file.readlines()
    atom_lines = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
    return lines, atom_lines

def generate_pdb_files(atom_lines, reshaped_data, output_dir, num_files=10, prefix="pdbO_"):
    """
    For each slice in reshaped_data (assumed shape: [N, num_atoms, 3]), update the atom coordinates
    (using the same number of atoms as in atom_lines) and write out a new PDB file.
    """
    num_files = min(num_files, reshaped_data.shape[0])
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        coordinates = reshaped_data[i]
        new_pdb_lines = []
        for j, line in enumerate(atom_lines[:coordinates.shape[0]]):
            x, y, z = coordinates[j]
            # Update columns 31-54 with new coordinates
            new_line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            new_pdb_lines.append(new_line)
        output_path = os.path.join(output_dir, f"{prefix}{i+1}.pdb")
        with open(output_path, 'w') as file:
            file.writelines(new_pdb_lines)
        print(f"PDB file saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate PDB files from HDF5 reconstructions (HNO and Decoder2)."
    )
    parser.add_argument("--hno_file", type=str, default="",
                        help="Path to the HNO reconstruction HDF5 file (optional)")
    parser.add_argument("--decoder2_file", type=str, default="",
                        help="Path to the Decoder2 reconstruction HDF5 file (optional; expects keys 'reconstructions_with_override' and 'reconstructions_no_override')")
    parser.add_argument("--pdb_file", type=str, required=True,
                        help="Path to the base PDB file (e.g., base.pdb)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory where subfolders will be created")
    parser.add_argument("--num_files", type=int, default=10,
                        help="Number of PDB files to generate from each reconstruction (default: 10)")
    args = parser.parse_args()

    # Check that the pdb_file exists (this one is required)
    if not os.path.exists(args.pdb_file):
        print(f"PDB file not found: {args.pdb_file}")
        return

    # Load the base PDB file
    all_lines, atom_lines = load_pdb(args.pdb_file)

    # Create output subfolders for each type (even if one branch is skipped, the others will be made)
    hno_out = os.path.join(args.output_dir, "HNO_reconstructions")
    dec2_with_out = os.path.join(args.output_dir, "Decoder2_with_override")
    dec2_without_out = os.path.join(args.output_dir, "Decoder2_without_override")
    
    # Process HNO file if present
    if args.hno_file and os.path.exists(args.hno_file):
        hno_data = read_and_reshape_h5(args.hno_file, key="reconstructions", reshape_dim=(2191, 3))
        os.makedirs(hno_out, exist_ok=True)
        print("Generating PDB files for HNO reconstructions")
        generate_pdb_files(atom_lines, hno_data, hno_out, num_files=args.num_files, prefix="hno_")
    else:
        print("HNO file not found or not provided; skipping HNO reconstructions.")

    # Process Decoder2 file if present
    if args.decoder2_file and os.path.exists(args.decoder2_file):
        dec2_with = read_and_reshape_h5(args.decoder2_file, key="reconstructions_with_override")
        dec2_without = read_and_reshape_h5(args.decoder2_file, key="reconstructions_no_override")
        os.makedirs(dec2_with_out, exist_ok=True)
        os.makedirs(dec2_without_out, exist_ok=True)
        print("Generating PDB files for Decoder2 reconstructions (with override)")
        generate_pdb_files(atom_lines, dec2_with, dec2_with_out, num_files=args.num_files, prefix="dec2_with_")
        print("Generating PDB files for Decoder2 reconstructions (without override)")
        generate_pdb_files(atom_lines, dec2_without, dec2_without_out, num_files=args.num_files, prefix="dec2_without_")
    else:
        print("Decoder2 file not found or not provided; skipping Decoder2 reconstructions.")

if __name__ == "__main__":
    main()
