import MDAnalysis as mda
import glob
import re
import argparse
import os

def pdbs_to_xtc(pdb_pattern, output_xtc):
    """
    Converts a numerically sorted series of PDB files into a single XTC trajectory.

    Parameters:
    - pdb_pattern (str): A glob pattern to find the PDB files (e.g., "generated_*.pdb").
    - output_xtc (str): The path for the output XTC file.
    """
    # 1. Find all files matching the pattern
    pdb_files = glob.glob(pdb_pattern)

    if not pdb_files:
        print(f"Error: No files found matching the pattern '{pdb_pattern}'.")
        return

    # 2. Sort the files numerically to ensure correct frame order
    # This extracts the number from filenames like 'gen_1.pdb', 'gen_10.pdb'
    pdb_files.sort(key=lambda f: int(re.search(r'(\d+)', f).group(1)))
    
    print(f"Found {len(pdb_files)} PDB files to process.")
    
    # 3. Use the first PDB file to define the topology (atom count, etc.)
    first_pdb = pdb_files[0]
    try:
        u = mda.Universe(first_pdb)
    except Exception as e:
        print(f"Error loading the first PDB file '{first_pdb}': {e}")
        return
        
    num_atoms = len(u.atoms)
    print(f"Using '{os.path.basename(first_pdb)}' as the topology reference ({num_atoms} atoms).")

    # 4. Initialize the writer for the XTC file
    with mda.Writer(output_xtc, n_atoms=num_atoms) as writer:
        # 5. Loop through each PDB file and write it as a frame
        for i, pdb_file in enumerate(pdb_files):
            try:
                # Load the current PDB to get its coordinates
                frame_universe = mda.Universe(pdb_file)
                
                # Sanity check: ensure atom count is consistent
                if len(frame_universe.atoms) != num_atoms:
                    print(f"Warning: Skipping {os.path.basename(pdb_file)} because its atom count "
                          f"({len(frame_universe.atoms)}) differs from the reference ({num_atoms}).")
                    continue
                
                # Write the atoms from the current PDB as a new frame
                writer.write(frame_universe.atoms)

                # Optional: Print progress
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(pdb_files)} frames...")

            except Exception as e:
                print(f"Warning: Could not process file {os.path.basename(pdb_file)}. Error: {e}")

    print(f"\n✅ Successfully created trajectory file: {output_xtc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine multiple PDB files into a single XTC trajectory file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-p", "--pattern", 
        type=str, 
        required=True,
        help="Input PDB file pattern, e.g., 'path/to/generated_*.pdb'.\n"
             "Use quotes around the pattern to prevent shell expansion."
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        required=True, 
        help="Output XTC file name, e.g., 'trajectory.xtc'."
    )
    
    args = parser.parse_args()
    pdbs_to_xtc(args.pattern, args.output)
