  GNU nano 5.6.1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 calc_tm.py                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
#!/usr/bin/env python3
"""
calc_tm_sample.py
=================

Compute TM-score for predicted structures in an HDF5 file against a reference structure (X_ref.npy or .pt).
Supports:
  - All-atom mode (directly use reference).
  - Backbone-only mode (parse backbone indices from a PDB to slice reference).
  - Random subsampling of models (via --max_samples and --seed).
  - Kabsch alignment, then TM-score calculation.
  - Automatic HDF5 key detection if --key is not provided.

Usage Examples:
  1) All-atom, sampling 100 models (explicit key):
     python calc_tm.py --h5file full_coords.h5  --xref X_ref.npy --max_samples 1000

  2) All-atom, sampling 100 models (auto-detect key):
     python calc_tm.py --h5file full_coords.h5 --xref X_ref.npy --max_samples 1000

  3) Backbone-only (auto-detect key):
     python calc_tm.py --h5file full_coords.h5 --xref X_ref.npy --pdb heavy_chain.pdb \
       --backbone_only --max_samples 100

  4) All-atom (diffused, explicit key):
     python calc_tm.py --h5file full_coords_diff.h5 --xref X_ref.npy --max_samples 1000

  5) Backbone-only (diffused, auto-detect key):
     python calc_tm.py --h5file full_coords_diff.h5 --xref X_ref.npy --pdb heavy_chain.pdb \
       --backbone_only --max_samples 100
"""

import argparse
import h5py
import torch
import numpy as np
import random
import math # For TM-score calculation

###############################################################################
# (A) LOADING FUNCTIONS
###############################################################################
'''
def load_h5_structure(h5file, key):
    """
    Load predicted structures from an HDF5 file under dataset 'key'.
    Returns a torch.FloatTensor shape [N, L, 3].
    """
    with h5py.File(h5file, "r") as f:
        if key not in f:
            keys_avail = list(f.keys())
            # This error should ideally not be reached if key auto-detection works
            raise KeyError(f"Key '{key}' not found in '{h5file}'. Available keys: {keys_avail}")
        data = f[key][:]
    return torch.from_numpy(data).float()
'''

def load_h5_structure(h5file, key):
    with h5py.File(h5file, "r") as f:
        if key not in f:
            avail_keys = list(f.keys())
            raise KeyError(f"Key '{key}' not found in '{h5file}'. Available keys: {avail_keys}")
        # Load data into a NumPy array
        data = f[key][:]

    # Store original shape for logging/debugging if needed
    original_shape = data.shape

    if data.ndim == 2:
        # User clarification: If data is 2D, its shape is (N_samples * 2191, 3).
        # The target shape is (N_samples, 2191, 3).

        # Define the expected structure components for the 2D array
        expected_middle_dim_size = 2191  # This is the '2191' part in the target (N_samples, 2191, 3)
        expected_last_dim_size = 3	 # This is the '3' part, which is data.shape[1] for the 2D input

        if data.shape[1] == expected_last_dim_size:
            # The last dimension of the 2D array matches (it's 3).
            # The first dimension (data.shape[0]) should be N_samples * expected_middle_dim_size.
            
            if data.shape[0] % expected_middle_dim_size == 0:
                # The first dimension is divisible by expected_middle_dim_size (2191).
                # This allows us to calculate N_samples.
                N_samples_calculated = data.shape[0] // expected_middle_dim_size
                
                # Reshape the data
                try:
                    data = data.reshape((N_samples_calculated, expected_middle_dim_size, expected_last_dim_size))
                except ValueError as e:
                    # This safeguard should ideally not be reached if previous checks are correct.
                    raise ValueError(
                        f"Error reshaping 2D data. Original shape: {original_shape}, "
                        f"attempted reshape to ({N_samples_calculated}, {expected_middle_dim_size}, {expected_last_dim_size}). "
                        f"NumPy error: {e}"
                    )
            else:
                # The first dimension is not a multiple of expected_middle_dim_size.
                raise ValueError(
                    f"Cannot reshape 2D data. Original shape: {original_shape}. "
                    f"The first dimension ({data.shape[0]}) is not a multiple of {expected_middle_dim_size}. "
                    f"Expected 2D input shape like (N_samples * {expected_middle_dim_size}, {expected_last_dim_size}) "
                    f"to reshape to (N_samples, {expected_middle_dim_size}, {expected_last_dim_size})."
                )
        else:
            # The second dimension of the 2D array is not expected_last_dim_size (3).
            raise ValueError(
                f"Cannot reshape 2D data. Original shape: {original_shape}. "
                f"The second dimension ({data.shape[1]}) is not {expected_last_dim_size}. "
                f"Expected 2D input shape like (N_samples * {expected_middle_dim_size}, {expected_last_dim_size}) "
                f"to reshape to (N_samples, {expected_middle_dim_size}, {expected_last_dim_size})."
            )
    elif data.ndim == 3:
        # Data is already 3D. Assume it's in the target format (N_samples, 2191, 3).
        # Optionally, add checks for the specific dimensions if they are strictly required.
        # For example, to ensure the inner dimensions are indeed 2191 and 3:
        if data.shape[1] != 2191 or data.shape[2] != 3:
            print(f"Warning: Loaded 3D data has shape {original_shape}. "
                  f"While it's 3D, the inner dimensions are not (..., 2191, 3). Proceeding as is.")
        # No reshape needed based on the primary request for 2D data.
        pass
    else:
        # Data is not 2D or 3D.
        raise ValueError(
            f"Unsupported data dimensionality. Original shape: {original_shape} (ndim={data.ndim}). "
            f"This function expects 2D or 3D data."
        )

    return torch.from_numpy(data).float()





def load_xref(xref_file):
    """Loads reference structure from .npy or .pt file."""
    if xref_file.endswith('.pt'):
        return torch.load(xref_file)
    else:
	arr = np.load(xref_file)
        return torch.from_numpy(arr).float()

def parse_backbone_indices_from_pdb(pdb_file):
    """
    Extract backbone atom indices (N, CA, C, O, OXT) from the given PDB.
    Returns a sorted list of 0-based backbone indices.
    """
    allowed = {"N", "CA", "C", "O", "OXT"}
    backbone_indices = []
    current_atom_index = 0

    try:
        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom_name = line[12:16].strip()
                    if atom_name in allowed:
                        backbone_indices.append(current_atom_index)
                    current_atom_index += 1
    except FileNotFoundError:
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    except Exception as e:
        raise IOError(f"Error reading PDB file {pdb_file}: {e}")

    if not backbone_indices:
        print(f"Warning: No backbone atoms (N, CA, C, O, OXT) found in {pdb_file}.")

    return sorted(backbone_indices)

###############################################################################
# (B) KABSCH ALIGNMENT + UTILS
###############################################################################

def kabsch_alignment_torch(P, Q):
    """
    Align Q onto P using Kabsch algorithm (in torch).
    P, Q => shape (L,3).
    Returns Q_aligned => shape (L,3).
    Uses torch.linalg.svd for modern PyTorch compatibility.
    """
    if P.shape != Q.shape:
         raise ValueError(f"Kabsch alignment requires P and Q to have the same shape. Got {P.shape} and {Q.shape}")
    if P.shape[0] < 3:
        # SVD may not be stable or meaningful with fewer than 3 points
        print("Warning: Kabsch alignment with fewer than 3 points might be unstable.")
        # Return Q untransformed or handle as appropriate for the use case
        return Q # Or raise an error

    centroid_P = P.mean(dim=0)
    centroid_Q = Q.mean(dim=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    C = torch.mm(Q_centered.t(), P_centered)

    try:
        U, S, Vt = torch.linalg.svd(C)
    except Exception as e:
         raise RuntimeError(f"torch.linalg.svd failed during Kabsch alignment: {e}")

    # Check for reflection
    d = torch.det(torch.mm(Vt.t(), U.t()))
    if d < 0:
	Vt[-1, :] = -Vt[-1, :] # Adjust the last row of Vt

    R = torch.mm(Vt.t(), U.t())
    Q_aligned = torch.mm(Q_centered, R) + centroid_P
    return Q_aligned

###############################################################################
# (C) TM-SCORE CALCULATION
###############################################################################

def compute_tm_score_torch(coords1, coords2):
    """
    Compute TM-score (torch version), given two sets of coords (L,3).
    Assumes coords1 and coords2 are ALREADY ALIGNED.
    Uses the standard TM-score formula.
    """
    if coords1.shape != coords2.shape:
        raise ValueError(f"TM-score calculation requires inputs to have the same shape. Got {coords1.shape} and {coords2.shape}")
    if coords1.shape[0] == 0:
         return 0.0 # Or raise error, TM-score undefined for empty structures

    L = coords1.shape[0]
    device = coords1.device

    dist_sq = torch.sum((coords1 - coords2)**2, dim=1) # Squared distances, shape (L,)

    # Calculate d0 - length-dependent scaling factor
    L_int = int(L)
    if L_int > 15:
        # Use math.cbrt for cube root or pow(x, 1/3)
        d0 = 1.24 * math.pow(L_int - 15, 1/3) - 1.8
    else:
	d0 = 0.5
    # Ensure d0 is positive, TM-score formula requires d0 > 0
    d0 = max(d0, 0.1) # Use a small positive floor like 0.1 if formula gives <= 0

    d0_sq = d0 * d0

    # Compute TM score terms: 1 / (1 + (dist_i / d0)^2) = 1 / (1 + dist_sq_i / d0_sq)
    score_terms = 1.0 / (1.0 + dist_sq / d0_sq)

    tm_score = score_terms.mean().item()
    return tm_score

###############################################################################
# (D) MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Compute TM-score for predicted structures (in an HDF5) vs. reference. Optional backbone-only mode and auto key detection."
    )
    parser.add_argument("--h5file", required=True,
                        help="HDF5 file containing predicted structures.")
    # Make --key optional
    parser.add_argument("--key", required=False, default=None,
                        help="Dataset key in the HDF5 file. If not provided, uses the first key found.")
    parser.add_argument("--xref", required=True,
                        help="Path to reference structure file (e.g., X_ref.npy or X_ref.pt).")

    parser.add_argument("--backbone_only", action="store_true",
                        help="If set, compute TM-score only on backbone atoms. Must also supply --pdb.")
    parser.add_argument("--pdb", type=str, default=None,
                        help="PDB file used to extract backbone indices (required if --backbone_only).")

    # optional sampling
    parser.add_argument("--max_samples", type=int, default=0,
                        help="If >0, randomly sample that many models from the dataset. If 0, use all models.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility if sampling is used.")

    args = parser.parse_args()

    # --- Determine HDF5 key ---
    h5_key_to_use = args.key
    if h5_key_to_use is None:
        try:
            with h5py.File(args.h5file, "r") as f:
                keys = list(f.keys())
                if not keys:
                    raise ValueError(f"HDF5 file '{args.h5file}' contains no datasets.")
                h5_key_to_use = keys[0]
                print(f"No --key provided. Automatically using first key found: '{h5_key_to_use}'")
        except FileNotFoundError:
             raise FileNotFoundError(f"HDF5 file not found: {args.h5file}")
        except Exception as e:
             raise IOError(f"Error opening or reading keys from HDF5 file '{args.h5file}': {e}")

    # --- Load Reference ---
    try:
        ref_all = load_xref(args.xref)  # shape (L_all,3)
        print(f"Loaded reference coords from '{args.xref}', shape={tuple(ref_all.shape)}")
    except Exception as e:
        raise IOError(f"Failed to load reference file '{args.xref}': {e}")

    # --- Handle Backbone Mode ---
    backbone_indices = None
    if args.backbone_only:
        if not args.pdb:
            raise ValueError("Must provide --pdb file path when using --backbone_only.")
        backbone_indices = parse_backbone_indices_from_pdb(args.pdb)
        if not backbone_indices:
             raise ValueError(f"Could not extract backbone indices from PDB '{args.pdb}'. Check file and atom names.")

        # Validate indices against reference structure length
        max_ref_idx = ref_all.shape[0] - 1
        if max(backbone_indices) > max_ref_idx:
             raise ValueError(f"Max backbone index ({max(backbone_indices)}) from PDB exceeds reference length ({ref_all.shape[0]}).")

        ref_coords = ref_all[backbone_indices, :]
        print(f"Backbone-only mode: Using {len(backbone_indices)} backbone atoms defined by '{args.pdb}'. Reference shape: {ref_coords.shape}")
    else:
	ref_coords = ref_all
        print(f"All-atom mode. Reference shape: {ref_coords.shape}")


    # --- Load Predicted Coords ---
    try:
        pred_coords_all = load_h5_structure(args.h5file, h5_key_to_use)  # shape (N, L_pred, 3)
        N_total = pred_coords_all.shape[0]
        L_pred = pred_coords_all.shape[1]
        print(f"Loaded predicted coords from '{args.h5file}' (key='{h5_key_to_use}'), shape={tuple(pred_coords_all.shape)}")
    except Exception as e:
         raise IOError(f"Failed to load predicted coordinates from '{args.h5file}' with key '{h5_key_to_use}': {e}")

    # --- Validate Shapes for Comparison ---
    ref_len_effective = ref_coords.shape[0] # Length of reference being used (all or backbone)

    if args.backbone_only:
         # Predicted coords must have enough atoms to cover all backbone indices
         max_bb_idx = max(backbone_indices)
         if L_pred <= max_bb_idx:
               raise ValueError(f"Predicted structures have length {L_pred}, but backbone indices require at least {max_bb_idx + 1} atoms.")
         # Slicing happens inside the loop
    else:
        # All-atom comparison: lengths must match exactly
        if L_pred != ref_len_effective:
            raise ValueError(f"Predicted structure length ({L_pred}) does not match reference structure length ({ref_len_effective}).")

    # --- Determine Subset for Processing ---
    if args.max_samples > 0 and args.max_samples < N_total:
        if args.seed is not None:
            print(f"Setting random seed to {args.seed} for sampling.")
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
        subset_indices = random.sample(range(N_total), args.max_samples)
        print(f"Randomly sampling {args.max_samples} models out of {N_total}.")
    else:
	subset_indices = range(N_total)
        print(f"Processing all {N_total} models.")

    # --- Set Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ref_coords = ref_coords.to(device)
    # Keep full predicted coords on host, move model by model or batch to device
    # This prevents OOM for very large HDF5 files
    # pred_coords_all = pred_coords_all.to(device) # Optionally move all if memory allows

    # --- Main Calculation Loop ---
    tm_scores = []
    num_processed = 0
    num_skipped = 0

    for count, i_model in enumerate(subset_indices, start=1):
        try:
            # Get full coordinates for the current model and move to device
            model_xyz_full = pred_coords_all[i_model].to(device)

            # Slice if backbone-only mode
            if args.backbone_only:
                # Double check indices are valid for this model (already checked against L_pred globally)
                model_xyz_effective = model_xyz_full[backbone_indices, :]
            else:
                model_xyz_effective = model_xyz_full

            # Final shape check before alignment
            if model_xyz_effective.shape[0] != ref_coords.shape[0]:
                 raise ValueError(f"Shape mismatch for model {i_model}: Ref={ref_coords.shape}, Model={model_xyz_effective.shape}")

            # Kabsch alignment: align model onto reference
            aligned_model = kabsch_alignment_torch(ref_coords, model_xyz_effective)

            # Compute TM-score between reference and the *aligned* model
            tm_val = compute_tm_score_torch(ref_coords, aligned_model)
            tm_scores.append(tm_val)
            num_processed += 1

        except Exception as e:
            print(f"Warning: Skipping model index {i_model} due to error: {e}")
            num_skipped += 1
            continue # Skip to the next model

        if count % 100 == 0 or count == len(subset_indices): # Print progress
             print(f"  Processed {count}/{len(subset_indices)} models...")


    # --- Summarize Results ---
    print("\n================= Summary =================")
    print(f"Models requested for processing: {len(subset_indices)}")
    print(f"Models successfully processed: {num_processed}")
    if num_skipped > 0:
        print(f"Models skipped due to errors: {num_skipped}")

    if not tm_scores:
         print("No TM-scores were calculated.")
    else:
         arr_np = np.array(tm_scores)
         mean_val = arr_np.mean()
         std_val  = arr_np.std() # Standard deviation of the calculated scores
         print(f"TM-score Mean = {mean_val:.4f}")
         print(f"TM-score Stdev = {std_val:.4f}")
    print("==========================================")


if __name__ == "__main__":
    main()




