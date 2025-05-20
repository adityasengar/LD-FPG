import json
import argparse

def infer_element(atom_name):
    """
    A simple heuristic to guess the element from the atom name.
    For example, 'CA' -> 'C', 'N' -> 'N', '1H' -> 'H'.
    """
    if not atom_name:
        return 'X'
    # If atom name starts with a digit, e.g. '1H', take the second character:
    if atom_name[0].isdigit():
        return atom_name[1].upper() if len(atom_name) > 1 else 'X'
    # Common two-letter elements for quick detection:
    two_letter_elements = {'FE','MG','ZN','CU','MN','NI','CA','NA','K','CL','BR'}
    # Try two-letter match first:
    upper_name = atom_name.upper()
    if len(atom_name) >= 2 and upper_name[:2] in two_letter_elements:
        return upper_name[:2]
    # Otherwise, fall back to first character:
    return atom_name[0].upper()

def format_pdb_atom_line(atom_serial, atom_name, res_name, chain_id, res_seq, x, y, z, element):
    """
    Returns a single PDB ATOM line for one atom.
    Columns used (approx. PDB v3.3):
      1-6   "ATOM  "
      7-11  Atom serial number (right-aligned)
      13-16 Atom name (right-aligned)
      18-20 Residue name
      22    Chain ID
      23-26 Residue sequence number
      31-38 X coord (8.3f)
      39-46 Y coord (8.3f)
      47-54 Z coord (8.3f)
      55-60 Occupancy (6.2f) => 1.00
      61-66 Temp factor (6.2f) => 0.00
      77-78 Element symbol (right-aligned)
    """
    atom_name_field = f"{atom_name:>4}"  # Right-align the atom name in a 4-char field
    return (
        f"ATOM  "
        f"{atom_serial:5d} "
        f"{atom_name_field}"
        f" {res_name:>3} "
        f"{chain_id}"
        f"{res_seq:4d}"
        f"    "
        f"{x:8.3f}"
        f"{y:8.3f}"
        f"{z:8.3f}"
        f"  1.00  0.00           "
        f"{element:>2}"
        f"\n"
    )

def export_atom_indices_names_coords_first_frame_to_dat_and_pdb(json_path, dat_path, pdb_path):
    """
    1) Reads the JSON file containing residue and atom information,
       then exports a .dat file with:
         Residue_Number Residue_Name Atom_Index Atom_Name X Y Z
       (first-frame coordinates).
    
    2) Simultaneously creates a .pdb file for the same first-frame heavy atoms.
    """
    # -- (A) Load JSON data --
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Sort residue keys numerically
    residue_keys_sorted = sorted(data.keys(), key=lambda x: int(x))
    
    # -- (B) Open files for writing --
    with open(dat_path, 'w') as dat_file, open(pdb_path, 'w') as pdb_file:
        # Write header line to .dat
        dat_file.write("Residue_Number Residue_Name Atom_Index Atom_Name X Y Z\n")
        # Write a simple header to .pdb (optional)
        pdb_file.write("HEADER    HEAVY ATOMS - FIRST FRAME\n")
        
        # We'll keep everything on chain 'A' (you can adjust if needed)
        chain_id = 'A'
        
        # -- (C) Iterate over each residue in sorted order --
        for res_key in residue_keys_sorted:
            residue_data = data[res_key]
    
            # Extract the residue number and residue name
            res_num = residue_data.get("res_num", "N/A")
            resname = residue_data.get("resname", "N/A")
    
            # Each 'heavy_atoms' entry aligns with the same index in coords
            heavy_atoms_list = residue_data.get("heavy_atoms", [])
            coords_per_frame = residue_data.get("heavy_atom_coords_per_frame", [])
    
            # For safety, check that there's at least one frame
            if not coords_per_frame:
                print(f"Warning: Residue {res_num} ({resname}) has no frames. Skipping.")
                continue  # No frames to process
    
            # We only need the coordinates of the first frame
            first_frame_coords = coords_per_frame[0]  # shape: (num_heavy_atoms, 3)
    
            # -- (D) Iterate through each heavy atom in this residue --
            for atom_idx, atom_info in enumerate(heavy_atoms_list):
                atom_index = atom_info.get("index", "N/A")  # e.g. 123
                atom_name = atom_info.get("name", "N/A")    # e.g. CA, C, N, etc.
                
                # Safely extract coordinates
                try:
                    x, y, z = first_frame_coords[atom_idx]
                except IndexError:
                    print(f"Error: Missing coords for atom index {atom_index} in residue {res_num}.")
                    x, y, z = ("nan", "nan", "nan")
                except TypeError:
                    print(f"Error: Invalid coords for atom index {atom_index} in residue {res_num}.")
                    x, y, z = ("nan", "nan", "nan")
    
                # -- (1) Write line to .dat --
                dat_line = f"{res_num} {resname} {atom_index} {atom_name} {x} {y} {z}\n"
                dat_file.write(dat_line)
                
                # -- (2) Write ATOM line to .pdb --
                # Use the same atom_index as the "serial number" in the PDB.
                # If you prefer a strict 1-based count, you can adapt it here.
                try:
                    atom_serial = int(atom_index)
                except ValueError:
                    # If the index is not numeric, fall back to a local counter or skip
                    atom_serial = atom_idx + 1
                
                element_symbol = infer_element(atom_name)
                
                # Convert res_num to int if possible
                try:
                    res_seq = int(res_num)
                except ValueError:
                    # If res_num isn't numeric, fallback to a dummy
                    res_seq = 1
                
                pdb_line = format_pdb_atom_line(
                    atom_serial=atom_serial,
                    atom_name=atom_name,
                    res_name=resname,
                    chain_id=chain_id,
                    res_seq=res_seq,
                    x=float(x), y=float(y), z=float(z),
                    element=element_symbol
                )
                pdb_file.write(pdb_line)
        
        # Write an END record in the PDB
        pdb_file.write("END\n")
    
    print(f"Data successfully exported to:\n  {dat_path}\n  {pdb_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Export atom indices, names, and coordinates (first frame) to both .dat and .pdb files."
    )
    parser.add_argument("json_path",  type=str, help="Path to the input JSON file (e.g., 'residues_data_active_full.json').")
    parser.add_argument("dat_path",   type=str, help="Path to the output .dat file (e.g., 'atom_indices_names_coords_first_frame.dat').")
    parser.add_argument("pdb_path",   type=str, help="Path to the output .pdb file (e.g., 'atom_indices_names_coords_first_frame.pdb').")
    
    args = parser.parse_args()
    export_atom_indices_names_coords_first_frame_to_dat_and_pdb(args.json_path, args.dat_path, args.pdb_path)

if __name__ == "__main__":
    main()
