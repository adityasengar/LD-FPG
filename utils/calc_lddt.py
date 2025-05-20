#!/usr/bin/env python3
"""
calc_lddt_sample.py
====================

This script calculates lDDT scores between predicted structures (from an HDF5 file)
and a reference structure (X_ref.npy or .pt), possibly restricting to the backbone only.
It also supports random subsampling of the models to compute mean/stdev of lDDT.

Key features:
    --max_samples (default=0) => if >0, randomly pick that many models from the HDF5 dataset
    --seed => set random seed for reproducibility
    --key => Optional HDF5 key. If not provided, uses the first key found in the file.

Usage Examples:

1) All-atom, using entire file (explicit key):
    python calc_lddt.py --h5file full_coords.h5 --xref X_ref.npy

2) All-atom, using entire file (auto-detect key):
    python calc_lddt.py --h5file full_coords.h5 --xref X_ref.npy

3) Backbone-only, using entire file (auto-detect key):
    python calc_lddt.py --h5file full_coords.h5 --xref X_ref.npy \
      --backbone_only --pdb heavy_chain.pdb

4) Diffusion all-atom, sampling 1000 models (auto-detect key):
    python calc_lddt.py --h5file full_coords_diff.h5 --xref X_ref.npy \
      --max_samples 1000 --seed 42

5) Diffusion backbone-only, sampling 1000 models (explicit key):
    python calc_lddt.py --h5file full_coords_diff.h5 --xref X_ref.npy \
      --backbone_only --pdb heavy_chain.pdb --max_samples 1000 --seed 42

"""

import argparse
import h5py
import torch
import numpy as np
import random

def load_h5_structure(h5file, key):
    with h5py.File(h5file, "r") as f:
        if key not in f:
            avail_keys = list(f.keys())
            # This error should ideally not be reached if key auto-detection works
            raise KeyError(f"Key '{key}' not found in '{h5file}'. Available keys: {avail_keys}")
        data = f[key][:]
    return torch.from_numpy(data).float()

def load_xref(xref_file):
    if xref_file.endswith('.pt'):
        return torch.load(xref_file)
    else:
        arr = np.load(xref_file)
        return torch.from_numpy(arr).float()

def parse_backbone_indices_from_pdb(pdb_file):
    allowed = {"N", "CA", "C", "O", "OXT"}
    backbone_indices = []
    current_idx = 0
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                if atom_name in allowed:
                    backbone_indices.append(current_idx)
                current_idx += 1
            else:
                pass
    return sorted(backbone_indices)

def kabsch_alignment_torch(P, Q):
    centroid_P = P.mean(dim=0)
    centroid_Q = Q.mean(dim=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    C = torch.mm(Q_centered.t(), P_centered)
    U, S, Vt = torch.linalg.svd(C) # Use torch.linalg.svd for compatibility
    # Check for reflection
    d = torch.det(torch.mm(Vt.t(), U.t()))
    if d < 0:
        Vt[-1, :] = -Vt[-1, :] # Adjust the last row of Vt
    R = torch.mm(Vt.t(), U.t())
    Q_aligned = torch.mm(Q_centered, R) + centroid_P
    return Q_aligned

def calculate_lddt_global_torch(ref_coords, model_coords, cutoff=15.0, seq_sep=2, thresholds=[0.5, 1.0, 2.0, 4.0]):
    device = ref_coords.device
    L = ref_coords.shape[0]
    diff_ref = ref_coords.unsqueeze(0) - ref_coords.unsqueeze(1)
    ref_dists = torch.norm(diff_ref, dim=2)
    diff_model = model_coords.unsqueeze(0) - model_coords.unsqueeze(1)
    model_dists = torch.norm(diff_model, dim=2)

    idx = torch.arange(L, device=device)
    seq_sep_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= seq_sep
    cutoff_mask = ref_dists <= cutoff
    valid_mask = seq_sep_mask & cutoff_mask

    diff_matrix = torch.abs(model_dists - ref_dists).unsqueeze(-1)  # (L,L,1)
    thr_tensor = torch.tensor(thresholds, device=device).view(1,1,-1)
    hits = (diff_matrix < thr_tensor).float()  # (L,L,T)
    hit_scores = hits.sum(dim=-1) / float(len(thresholds))

    per_atom_lddt = torch.zeros(L, device=device)
    num_valid_comparisons = valid_mask.sum(dim=1).float() # Precompute number of valid comparisons per atom

    # Vectorized calculation for per_atom_lddt
    masked_scores = hit_scores * valid_mask.float() # Zero out scores where mask is False
    summed_scores = masked_scores.sum(dim=1)

    # Avoid division by zero for atoms with no valid comparisons
    per_atom_lddt = torch.where(num_valid_comparisons > 0, summed_scores / num_valid_comparisons, torch.zeros_like(summed_scores))

    overall_lddt = per_atom_lddt.mean().item()
    return overall_lddt, per_atom_lddt


def main():
    parser = argparse.ArgumentParser(
        description="Compute lDDT (all-atom or backbone-only) with optional sampling of models."
    )
    parser.add_argument("--h5file", required=True)
    # Make --key optional
    parser.add_argument("--key", required=False, default=None,
                        help="Key for the dataset in HDF5 file. If not provided, uses the first key found.")
    parser.add_argument("--xref", required=True, help="Path to reference structure (X_ref.npy or .pt).")
    parser.add_argument("--cutoff", type=float, default=15.0)
    parser.add_argument("--seq_sep", type=int, default=2)

    parser.add_argument("--backbone_only", action="store_true",
                        help="If set, do backbone-only lDDT. Must pass --pdb to get backbone indices.")
    parser.add_argument("--pdb", type=str, default=None,
                        help="PDB file used to extract backbone indices if --backbone_only is set.")
    # Sampling:
    parser.add_argument("--max_samples", type=int, default=0,
                        help="If >0, randomly sample that many models from the dataset.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Set random seed for reproducibility if sampling.")

    args = parser.parse_args()

    # Determine the HDF5 key to use
    h5_key_to_use = args.key
    if h5_key_to_use is None:
        try:
            with h5py.File(args.h5file, "r") as f:
                keys = list(f.keys())
                if not keys:
                    raise ValueError(f"HDF5 file '{args.h5file}' contains no keys.")
                h5_key_to_use = keys[0]
                print(f"No --key provided, automatically using first key found: '{h5_key_to_use}'")
        except Exception as e:
             raise ValueError(f"Error opening or reading keys from HDF5 file '{args.h5file}': {e}")

    # Load reference
    ref_all = load_xref(args.xref)  # shape (L_all,3)
    print(f"Loaded reference from {args.xref}, shape={tuple(ref_all.shape)}")

    # If backbone-only, parse indices from PDB => slice ref
    if args.backbone_only:
        if not args.pdb:
            raise ValueError("Must provide --pdb when using --backbone_only.")
        backbone_indices = parse_backbone_indices_from_pdb(args.pdb)
        if not backbone_indices:
             raise ValueError(f"No backbone atoms found in PDB file '{args.pdb}'. Check PDB format and atom names.")
        # Ensure indices are within bounds of the reference structure
        max_ref_idx = ref_all.shape[0] - 1
        if max(backbone_indices) > max_ref_idx:
             raise ValueError(f"Backbone index {max(backbone_indices)} from PDB is out of bounds for reference structure with {ref_all.shape[0]} atoms.")
        ref_coords = ref_all[backbone_indices, :]
        print(f"Backbone-only mode: using {len(backbone_indices)} backbone atoms from PDB={args.pdb}")
    else:
        backbone_indices = None
        ref_coords = ref_all

    # Load predicted coords using the determined key
    pred_coords = load_h5_structure(args.h5file, h5_key_to_use)  # shape (N, L, 3)
    N_total = pred_coords.shape[0]
    print(f"Loaded predicted coords from {args.h5file}, key='{h5_key_to_use}', shape={pred_coords.shape}")

    # Dimension check
    pred_len = pred_coords.shape[1] # Length dimension from predicted coords
    ref_len_effective = ref_coords.shape[0] # Length of reference (potentially sliced for backbone)

    if args.backbone_only:
         # Predicted coords must have enough atoms to cover all backbone indices
         max_bb_idx = max(backbone_indices)
         if pred_len <= max_bb_idx:
               raise ValueError(f"Predicted coords have {pred_len} atoms, but backbone indices require at least {max_bb_idx + 1} atoms.")
         # We will slice the predicted coordinates later inside the loop
    else:
        # All-atom comparison: lengths must match exactly
        if pred_len != ref_len_effective:
            raise ValueError(f"Predicted structure length ({pred_len}) does not match reference structure length ({ref_len_effective}).")


    # If sampling:
    if args.max_samples > 0 and args.max_samples < N_total:
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed) # Also seed numpy for consistency if used elsewhere
            torch.manual_seed(args.seed) # Seed torch as well
        subset_indices = random.sample(range(N_total), args.max_samples)
        print(f"Sampling {args.max_samples} models (seed={args.seed}).")
    else:
        subset_indices = range(N_total)
        print(f"Processing all {N_total} models.")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ref_coords = ref_coords.to(device)
    # Keep full pred_coords on CPU/GPU as needed, slice inside loop
    pred_coords_device = pred_coords.to(device)

    lddt_values = []
    for count, i_model in enumerate(subset_indices, start=1):
        # Get the full coordinates for the current model
        model_xyz_full = pred_coords_device[i_model]

        # Slice if backbone-only mode
        if args.backbone_only:
            # Ensure indices are valid for this specific model structure's length
            if max(backbone_indices) >= model_xyz_full.shape[0]:
                 print(f"Warning: Skipping model index {i_model}. Max backbone index {max(backbone_indices)} exceeds model length {model_xyz_full.shape[0]}.")
                 continue # Skip this model if backbone indices are out of bounds
            model_xyz_effective = model_xyz_full[backbone_indices, :]
        else:
            model_xyz_effective = model_xyz_full

        # Ensure shapes match before alignment and lDDT calculation
        if model_xyz_effective.shape[0] != ref_coords.shape[0]:
             print(f"Warning: Skipping model index {i_model}. Shape mismatch after potential slicing.")
             print(f"  Reference shape: {ref_coords.shape}")
             print(f"  Model effective shape: {model_xyz_effective.shape}")
             continue # Skip if shapes don't match (shouldn't happen with earlier checks, but safety first)

        # Align
        try:
            aligned = kabsch_alignment_torch(ref_coords, model_xyz_effective)
        except Exception as e:
             print(f"Warning: Kabsch alignment failed for model index {i_model}. Error: {e}. Skipping.")
             continue # Skip if alignment fails

        # lDDT
        try:
            val, _ = calculate_lddt_global_torch(ref_coords, aligned,
                                                 cutoff=args.cutoff,
                                                 seq_sep=args.seq_sep)
            lddt_values.append(val)
        except Exception as e:
             print(f"Warning: lDDT calculation failed for model index {i_model} after alignment. Error: {e}. Skipping.")
             continue # Skip if lDDT calculation fails

        if count % 100 == 0: # Print progress periodically
             print(f"  Processed {count}/{len(subset_indices)} models...")


    # Print metrics
    if not lddt_values:
         print("\n=====================================================")
         print("No models were successfully processed. Check warnings and input data.")
         print("=====================================================")
    else:
         arr_np = np.array(lddt_values)
         mean_val = arr_np.mean()
         std_val  = arr_np.std() # Standard deviation of the sample lDDTs
         num_processed = len(lddt_values)
         num_requested = len(subset_indices)

         print("\n=====================================================")
         print(f"Number of models requested = {num_requested}")
         print(f"Number of models successfully processed = {num_processed}")
         if num_processed < num_requested:
              print(f"({num_requested - num_processed} models skipped due to errors or warnings)")
         print(f"lDDT mean = {mean_val:.4f}, stdev = {std_val:.4f}")
         print("=====================================================")


if __name__ == "__main__":
    main()
