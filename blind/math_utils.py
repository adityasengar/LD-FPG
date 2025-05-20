import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Any

# --- Dihedral Utilities ---
@torch.jit.script # For potential performance improvement via TorchScript
def compute_dihedral(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """
    Computes the dihedral angle from four 3D points (p0, p1, p2, p3).
    Points are Tensors of shape [B, 3] or [3] for single calculation.
    Formula from Blondel, F., & Karplus, M. (1996). New formulation for derivatives of torsion angles...
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    # Normal vectors to the planes
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    # Normalize normal vectors (add epsilon for stability)
    n1_normalized = F.normalize(n1, p=2.0, dim=-1, eps=1e-8)
    n2_normalized = F.normalize(n2, p=2.0, dim=-1, eps=1e-8)

    # Vector orthogonal to n1 and b2 (in the plane of b1, b2 and perp to b2)
    b2_normalized = F.normalize(b2, p=2.0, dim=-1, eps=1e-8)
    m1 = torch.cross(n1_normalized, b2_normalized, dim=-1) # Should be normalized if n1 and b2 are orthonormal

    # x = n1_normalized ⋅ n2_normalized
    # y = m1 ⋅ n2_normalized
    x_val = (n1_normalized * n2_normalized).sum(dim=-1)
    y_val = (m1 * n2_normalized).sum(dim=-1)

    return torch.atan2(y_val, x_val)

def compute_all_dihedrals_vectorized(coords_batch: torch.Tensor,
                                     dihedral_info: Dict[str, Dict[str, Any]],
                                     num_residues: int,
                                     logger: logging.Logger) -> Dict[str, torch.Tensor]:
    """
    Computes all specified dihedral angles for a batch of coordinate sets.
    coords_batch: Tensor of shape [B, N_atoms, 3].
    dihedral_info: Dict mapping angle_name to {'indices': [t_idx0, t_idx1, t_idx2, t_idx3], 'res_idx': t_res_ids}.
                   Indices are 1D Tensors of atom indices for each angle instance.
                   res_idx is a 1D Tensor of residue indices (0 to N_res-1) for each angle.
    num_residues: Total number of residues in the protein.
    Returns a dict mapping angle_name to a Tensor of angles [B, N_res].
    """
    if coords_batch.ndim != 3:
        raise ValueError(f"Expected coords_batch to be [B, N_atoms, 3], but got shape {coords_batch.shape}")

    B, N_atoms_in_coords, _ = coords_batch.shape
    device = coords_batch.device
    all_computed_angles: Dict[str, torch.Tensor] = {}

    for angle_name, angle_data in dihedral_info.items():
        atom_indices_list = angle_data.get('indices') # List of 4 Tensors
        residue_indices_for_angle = angle_data.get('res_idx') # Tensor

        # Initialize output tensor for this angle type (filled with zeros for non-existent angles)
        angles_output_tensor = torch.zeros(B, num_residues, device=device, dtype=coords_batch.dtype)

        if atom_indices_list is not None and residue_indices_for_angle is not None and \
           all(t.numel() > 0 for t in atom_indices_list if t is not None) and residue_indices_for_angle.numel() > 0:
            try:
                # Move indices to the correct device
                p_indices_device = [idx_tensor.to(device) for idx_tensor in atom_indices_list]
                res_idx_device = residue_indices_for_angle.to(device)

                # Validate atom indices before gathering coordinates
                max_atom_idx_needed = max(idx_tensor.max() for idx_tensor in p_indices_device)
                if max_atom_idx_needed >= N_atoms_in_coords:
                    raise IndexError(f"Atom index {max_atom_idx_needed} out of bounds for N_atoms={N_atoms_in_coords} when computing {angle_name}.")
                max_res_idx_needed = res_idx_device.max()
                if max_res_idx_needed >= num_residues:
                     raise IndexError(f"Residue index {max_res_idx_needed} out of bounds for N_res={num_residues} when computing {angle_name}.")


                # Gather coordinates for p0, p1, p2, p3 using advanced indexing
                # p_coords will be a list of 4 tensors, each [B, num_angles_of_this_type, 3]
                p_coords = [coords_batch[:, p_idx_tensor, :] for p_idx_tensor in p_indices_device]

                # Compute dihedrals: output shape [B, num_angles_of_this_type]
                computed_values = compute_dihedral(p_coords[0], p_coords[1], p_coords[2], p_coords[3])

                # Scatter these values into the [B, N_res] output tensor
                # Create batch indices for scattering: [B, 1]
                batch_indices_for_scatter = torch.arange(B, device=device).unsqueeze(1)
                # Expand residue indices for scattering: [1, num_angles_of_this_type] -> [B, num_angles_of_this_type]
                # No, res_idx_device is already [num_angles_of_this_type]
                # Need to map B x num_angles_of_this_type to B x N_res
                angles_output_tensor[batch_indices_for_scatter, res_idx_device.unsqueeze(0)] = computed_values

            except IndexError as e:
                logger.error(f"Indexing error while computing dihedral angle '{angle_name}': {e}", exc_info=False) # exc_info=False to avoid spamming logs
            except Exception as e:
                logger.error(f"Unexpected error computing dihedral angle '{angle_name}': {e}", exc_info=True)
        all_computed_angles[angle_name] = angles_output_tensor
    return all_computed_angles

# --- Distribution Divergence Metrics ---
def compute_angle_kl_div(pred_angles: torch.Tensor, target_angles: torch.Tensor,
                         num_bins: int = 36, angle_range: Tuple[float, float] = (-np.pi, np.pi)) -> torch.Tensor:
    """Computes KL divergence between distributions of predicted and target angles."""
    pred_detached, target_detached = pred_angles.detach(), target_angles.detach()
    device = pred_angles.device

    if pred_detached.numel() == 0 or target_detached.numel() == 0: # Handle empty inputs
        return torch.tensor(0.0, device=device)

    # Create histograms
    pred_hist = torch.histc(pred_detached, bins=num_bins, min=angle_range[0], max=angle_range[1])
    target_hist = torch.histc(target_detached, bins=num_bins, min=angle_range[0], max=angle_range[1])

    # Normalize to distributions (add epsilon for stability)
    epsilon = 1e-10
    pred_dist = pred_hist / (pred_hist.sum() + epsilon)
    target_dist = target_hist / (target_hist.sum() + epsilon)

    # KL divergence: sum(target_dist * (log(target_dist) - log(pred_dist)))
    # PyTorch F.kl_div expects log input for the first argument if log_target=False
    log_pred_dist = torch.log(pred_dist + epsilon) # log q(x)
    # kl_div(log_pred, target_dist) computes sum(target_dist * (log(target_dist) - log_pred_dist))
    # This is D_KL(target || pred). The paper calculates D_KL(P_pred || P_true)
    # So we need F.kl_div(log_target_dist, pred_dist, reduction='sum', log_target=False)
    # Or, more commonly, D_KL(P_pred || P_true) = sum P_pred * (log P_pred - log P_true)
    log_target_dist = torch.log(target_dist + epsilon) # log p(x)
    # To compute D_KL(pred_dist || target_dist):
    return F.kl_div(log_pred_dist, target_dist, reduction='sum', log_target=False)


def compute_angle_js_div(pred_angles: torch.Tensor, target_angles: torch.Tensor,
                         num_bins: int = 36, angle_range: Tuple[float, float] = (-np.pi, np.pi)) -> torch.Tensor:
    """Computes Jensen-Shannon divergence between distributions of predicted and target angles."""
    pred_detached, target_detached = pred_angles.detach(), target_angles.detach()
    device = pred_angles.device

    if pred_detached.numel() == 0 or target_detached.numel() == 0:
        return torch.tensor(0.0, device=device)

    # Create histograms
    pred_hist = torch.histc(pred_detached, bins=num_bins, min=angle_range[0], max=angle_range[1])
    target_hist = torch.histc(target_detached, bins=num_bins, min=angle_range[0], max=angle_range[1])

    # Normalize to distributions
    epsilon = 1e-10
    P_dist = pred_hist / (pred_hist.sum() + epsilon)  # Distribution P
    Q_dist = target_hist / (target_hist.sum() + epsilon)  # Distribution Q

    # Mixture distribution M
    M_dist = 0.5 * (P_dist + Q_dist)
    log_M_dist = torch.log(M_dist + epsilon)

    # JS Div = 0.5 * [KL(P || M) + KL(Q || M)]
    # KL(P || M) = sum(P * (logP - logM))
    # PyTorch: F.kl_div(log_M_dist (as log_target), P_dist (as input), reduction='sum', log_target=True) means P * (logP - logM)
    # OR: F.kl_div(torch.log(P_dist+eps), M_dist, reduction='sum', log_target=False)
    kl_P_M = F.kl_div(torch.log(P_dist + epsilon), M_dist, reduction='sum', log_target=False)
    kl_Q_M = F.kl_div(torch.log(Q_dist + epsilon), M_dist, reduction='sum', log_target=False)
    
    # Ensure non-negative result due to numerical precision
    return F.relu(0.5 * (kl_P_M + kl_Q_M))


def compute_angle_wasserstein(pred_angles: torch.Tensor, target_angles: torch.Tensor,
                              num_bins: int = 36, angle_range: Tuple[float, float] = (-np.pi, np.pi)) -> torch.Tensor:
    """Computes 1-Wasserstein distance (Earth Mover's Distance) between 1D distributions."""
    pred_detached, target_detached = pred_angles.detach(), target_angles.detach()
    device = pred_angles.device

    if pred_detached.numel() == 0 or target_detached.numel() == 0:
        return torch.tensor(0.0, device=device)

    pred_hist = torch.histc(pred_detached, bins=num_bins, min=angle_range[0], max=angle_range[1])
    target_hist = torch.histc(target_detached, bins=num_bins, min=angle_range[0], max=angle_range[1])

    epsilon = 1e-10
    pred_dist = pred_hist / (pred_hist.sum() + epsilon)
    target_dist = target_hist / (target_hist.sum() + epsilon)

    # Compute CDFs
    pred_cdf = torch.cumsum(pred_dist, dim=0)
    target_cdf = torch.cumsum(target_dist, dim=0)

    # Wasserstein-1 distance for 1D is the L1 norm of the difference between CDFs
    return torch.sum(torch.abs(pred_cdf - target_cdf))

# --- MSE Utilities ---
def compute_bb_sc_mse(predictions: torch.Tensor, targets: torch.Tensor,
                      backbone_indices: torch.Tensor, sidechain_indices: torch.Tensor,
                      logger: logging.Logger) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Overall, Backbone, and Sidechain MSE.
    Assumes predictions and targets are flat [N_total_atoms_in_batch, 3] or [B*N_atoms_per_struct, 3].
    Indices must correspond to the flattened structure if batching.
    If inputs are per-structure [N_atoms, 3], indices should be for a single structure.
    The function expects backbone_indices and sidechain_indices to be on the same device as predictions.
    """
    criterion = nn.MSELoss()
    targets_casted = targets.to(predictions.dtype) # Ensure dtype match
    overall_mse = criterion(predictions, targets_casted)

    backbone_mse = torch.tensor(0.0, device=predictions.device)
    sidechain_mse = torch.tensor(0.0, device=predictions.device)

    try:
        if backbone_indices.numel() > 0:
            backbone_mse = criterion(predictions[backbone_indices], targets_casted[backbone_indices])
        if sidechain_indices.numel() > 0:
            sidechain_mse = criterion(predictions[sidechain_indices], targets_casted[sidechain_indices])
    except IndexError as e:
        # This typically happens if indices are out of bounds for the predictions/targets tensor.
        # This can occur if batching is handled incorrectly with indices.
        # The provided code structure seems to pass global bb_idx/sc_idx to compute_bb_sc_mse,
        # which implies that predictions/targets within train/eval loops are per-structure (Data.x, Data.y).
        logger.error(f"MSE Indexing Error: {e}. Pred shape: {predictions.shape}, Target shape: {targets_casted.shape}, "
                     f"BB_idx_max: {backbone_indices.max() if backbone_indices.numel() > 0 else 'N/A'}, "
                     f"SC_idx_max: {sidechain_indices.max() if sidechain_indices.numel() > 0 else 'N/A'}", exc_info=False)

    return overall_mse, backbone_mse, sidechain_mse

# --- MLP Builder ---
def build_decoder_mlp(input_dim: int, output_dim: int, num_hidden_layers: int,
                      hidden_layer_dim: int = 128) -> nn.Sequential:
    """
    Builds a Multi-Layer Perceptron (MLP) with BatchNorm1d and ReLU activations.
    num_hidden_layers: Total number of layers in MLP (e.g., 2 means input -> hidden -> output).
                       The original code's N_layers seems to be this total number.
    """
    layers: List[nn.Module] = []
    current_dim = input_dim

    if num_hidden_layers <= 0:
        raise ValueError("Number of MLP layers must be at least 1 (for a direct linear map).")
    elif num_hidden_layers == 1: # Direct linear layer
        layers.append(nn.Linear(current_dim, output_dim))
    else:
        # First hidden layer
        layers.extend([
            nn.Linear(current_dim, hidden_layer_dim),
            nn.BatchNorm1d(hidden_layer_dim),
            nn.ReLU()
        ])
        current_dim = hidden_layer_dim

        # Intermediate hidden layers (num_hidden_layers - 2 of these)
        for _ in range(num_hidden_layers - 2):
            layers.extend([
                nn.Linear(current_dim, hidden_layer_dim),
                nn.BatchNorm1d(hidden_layer_dim),
                nn.ReLU()
            ])
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

    return nn.Sequential(*layers)
