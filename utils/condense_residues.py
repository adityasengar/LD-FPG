'''
Load the original time-series JSON (the one that looks like your serialized residues data, containing "heavy_atoms", "torsion_atoms", etc.).

Create a contiguous remapping of old atom indices to new indices. Separate backbone vs. side-chain atoms (using atom_dict["is_backbone"]). Remap torsion quadruplets (𝜙,𝜓,𝜒𝑛)
Write a smaller final JSON under keys "0", "1", etc. Each entry has the form:

{
  "new_res_id": 0,
  "old_res_id": 1,
  "resname": "ARG",
  "backbone": [0, 1, 2, 3, 4],        // new index numbering
  "sidechain": [5, 6, 7, 8, 9],
  "torsion_atoms": {
    "phi": [0, 1, 2, 3],            // new numbering for the quadruplet
    "psi": [1, 2, 3, 10],
    "chi": {
      "chi1": [1, 2, 5, 20],
      ...
    }
  }
}

'''


#!/usr/bin/env python3
import json
import numpy as np
import sys
import os

def build_old2new_map(original_data):
    """
    Collect all old atom indices from each residue's 'heavy_atom_indices'
    and build a contiguous mapping old_index -> new_index.
    """
    # Flatten all heavy_atom_indices
    all_old_indices = []
    for res_id_str, res_info in original_data.items():
        # 'heavy_atom_indices' is presumably a list of old indices
        all_old_indices.extend(res_info["heavy_atom_indices"])

    unique_old_indices = sorted(set(all_old_indices))
    old2new = {}
    for new_idx, old_idx in enumerate(unique_old_indices):
        old2new[old_idx] = new_idx
    return old2new

def condense_json(original_data, old2new_map):
    """
    Build a condensed dictionary with:
      - new_res_id
      - resname
      - backbone (list of new indices)
      - sidechain (list of new indices)
      - torsion_atoms: {phi, psi, chi} in new indexing
    """
    condensed_data = {}

    # Sort residue IDs by integer value
    sorted_res_ids = sorted(original_data.keys(), key=lambda x: int(x))
    new_res_id_counter = 0

    for old_res_str in sorted_res_ids:
        old_res_id = int(old_res_str)
        res_info = original_data[old_res_str]

        # 1) Identify backbone vs. side-chain in the old indexing
        backbone_old = []
        sidechain_old = []
        # 'heavy_atoms' is a list of dicts => each has {'index', 'is_backbone', ...}
        for atom_dict in res_info["heavy_atoms"]:
            old_idx = atom_dict["index"]
            if atom_dict["is_backbone"]:
                backbone_old.append(old_idx)
            else:
                sidechain_old.append(old_idx)

        # 2) Remap those to new indexing
        backbone_new = [old2new_map[i] for i in backbone_old]
        sidechain_new = [old2new_map[i] for i in sidechain_old]

        # 3) Remap torsion atoms
        torsion_remapped = {"phi": None, "psi": None, "chi": {}}
        old_torsions = res_info.get("torsion_atoms", {})
        
        # (a) phi
        phi_old = old_torsions.get("phi", None)
        if phi_old and None not in phi_old:
            phi_new = [old2new_map[a] for a in phi_old]
        else:
            phi_new = None
        torsion_remapped["phi"] = phi_new

        # (b) psi
        psi_old = old_torsions.get("psi", None)
        if psi_old and None not in psi_old:
            psi_new = [old2new_map[a] for a in psi_old]
        else:
            psi_new = None
        torsion_remapped["psi"] = psi_new

        # (c) chi dict
        chi_dict_old = old_torsions.get("chi", {})
        chi_dict_new = {}
        for chi_name, chi_atoms_old in chi_dict_old.items():
            if chi_atoms_old and None not in chi_atoms_old:
                chi_atoms_new = [old2new_map[a] for a in chi_atoms_old]
            else:
                chi_atoms_new = None
            chi_dict_new[chi_name] = chi_atoms_new
        torsion_remapped["chi"] = chi_dict_new

        # 4) Build the final condensed entry
        condensed_data[str(new_res_id_counter)] = {
            "new_res_id": new_res_id_counter,
            "old_res_id": old_res_id,
            "resname": res_info.get("resname", "UNK"),
            "backbone": backbone_new,
            "sidechain": sidechain_new,
            "torsion_atoms": torsion_remapped
        }

        new_res_id_counter += 1

    return condensed_data

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {os.path.basename(__file__)} <input_json> <output_json>")
        sys.exit(1)

    input_json = sys.argv[1]
    output_json = sys.argv[2]

    if not os.path.isfile(input_json):
        print(f"ERROR: Could not find input JSON: {input_json}")
        sys.exit(1)

    print(f"Loading original data from {input_json}...")
    with open(input_json, "r") as f:
        original_data = json.load(f)

    # Build old->new mapping
    print("Building old->new index mapping...")
    old2new = build_old2new_map(original_data)
    print(f"Mapping covers {len(old2new)} unique atom indices.")

    # Build condensed JSON
    print("Condensing data...")
    condensed_data = condense_json(original_data, old2new)

    # Save
    with open(output_json, "w") as f:
        json.dump(condensed_data, f, indent=2)

    print(f"Condensed JSON saved to: {output_json}")
    print("Done.")

if __name__ == "__main__":
    main()
