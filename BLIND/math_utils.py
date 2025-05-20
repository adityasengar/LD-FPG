# math_utils.py
# Description: Mathematical utilities including dihedral calculations,
# MSE computation, and distribution divergence metrics.

from common_imports import (
    torch, nn, F, np, logging, # logging might not be used here directly unless for debug
    Dict, List, Tuple, Any
)

# --- Dihedral Utilities ---
@torch.jit.script
def compute_dihedral(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """Computes dihedral angle from four 3D points [B, 3] or [3]."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    n1_normalized = F.normalize(n1, p=2.0, dim=-1, eps=1e-8)
    n2_normalized = F.normalize(n2, p=2.0, dim=-1, eps=1e-8)
    b2_normalized = F.normalize(b2, p=2.0, dim=-1, eps=1e-8)
    m1 = torch.cross(n1_normalized, b2_normalized, dim=-1)
    x_val = (n1_normalized * n2_normalized).sum(dim=-1)
    y_val = (m1 * n2_normalized).sum(dim=-1)
    return torch.atan2(y_val, x_val)

def compute_all_dihedrals_vectorized(coords_batch: torch.Tensor,
                                     dihedral_info: Dict[str, Dict[str, Any]],
                                     num_residues: int,
                                     logger: logging.Logger) -> Dict[str, torch.Tensor]:
    """Computes all specified dihedrals for a batch [B, N_atoms, 3]."""
    if coords_batch.ndim != 3:
        raise ValueError(f"Expected coords_batch [B, N, 3], got {coords_batch.shape}")
    B, N_atoms_in_coords, _ = coords_batch.shape
    device = coords_batch.device
    all_computed_angles: Dict[str, torch.Tensor] = {}

    for angle_name, angle_data in dihedral_info.items():
        atom_indices_list = angle_data.get('indices')
        residue_indices_for_angle = angle_data.get('res_idx')
        angles_output_tensor = torch.zeros(B, num_residues, device=device, dtype=coords_batch.dtype)

        if atom_indices_list is not None and residue_indices_for_angle is not None and \
           all(t is not None and t.numel() > 0 for t in atom_indices_list) and \
           residue_indices_for_angle.numel() > 0: # Added None check for tensors in list
            try:
                p_indices_device = [idx_tensor.to(device) for idx_tensor in atom_indices_list]
                res_idx_device = residue_indices_for_angle.to(device)
                max_atom_idx_needed = max(idx_tensor.max() for idx_tensor in p_indices_device)
                if max_atom_idx_needed >= N_atoms_in_coords:
                    raise IndexError(f"Atom index {max_atom_idx_needed} >= {N_atoms_in_coords} for {angle_name}.")
                max_res_idx_needed = res_idx_device.max()
                if max_res_idx_needed >= num_residues:
                     raise IndexError(f"Residue index {max_res_idx_needed} >= {num_residues} for {angle_name}.")

                p_coords = [coords_batch[:, p_idx_tensor, :] for p_idx_tensor in p_indices_device]
                computed_values = compute_dihedral(p_coords[0], p_coords[1], p_coords[2], p_coords[3])
                batch_indices_for_scatter = torch.arange(B, device=device).unsqueeze(1)
                angles_output_tensor.scatter_(1, res_idx_device.unsqueeze(0).expand(B, -1), computed_values) # Use scatter_ for in-place
            except IndexError as e:
                logger.error(f"Indexing error computing dihedral '{angle_name}': {e}", exc_info=False)
            except Exception as e:
                logger.error(f"Error computing dihedral '{angle_name}': {e}", exc_info=True)
        all_computed_angles[angle_name] = angles_output_tensor
    return all_computed_angles

# --- Distribution Divergence Metrics ---
def compute_angle_kl_div(pred_angles: torch.Tensor, target_angles: torch.Tensor,
                         num_bins: int = 36, angle_range: Tuple[float, float] = (-np.pi, np.pi)) -> torch.Tensor:
    pred_detached, target_detached = pred_angles.detach(), target_angles.detach()
    device = pred_angles.device
    if pred_detached.numel() == 0 or target_detached.numel() == 0:
        return torch.tensor(0.0, device=device)
    pred_hist = torch.histc(pred_detached, bins=num_bins, min=angle_range[0], max=angle_range[1])
    target_hist = torch.histc(target_detached, bins=num_bins, min=angle_range[0], max=angle_range[1])
    epsilon = 1e-10
    pred_dist = pred_hist / (pred_hist.sum() + epsilon)
    target_dist = target_hist / (target_hist.sum() + epsilon)
    log_pred_dist = torch.log(pred_dist + epsilon)
    return F.kl_div(log_pred_dist, target_dist, reduction='sum', log_target=False)

def compute_angle_js_div(pred_angles: torch.Tensor, target_angles: torch.Tensor,
                         num_bins: int = 36, angle_range: Tuple[float, float] = (-np.pi, np.pi)) -> torch.Tensor:
    pred_detached, target_detached = pred_angles.detach(), target_angles.detach()
    device = pred_angles.device
    if pred_detached.numel() == 0 or target_detached.numel() == 0:
        return torch.tensor(0.0, device=device)
    pred_hist = torch.histc(pred_detached, bins=num_bins, min=angle_range[0], max=angle_range[1])
    target_hist = torch.histc(target_detached, bins=num_bins, min=angle_range[0], max=angle_range[1])
    epsilon = 1e-10
    P_dist = pred_hist / (pred_hist.sum() + epsilon)
    Q_dist = target_hist / (target_hist.sum() + epsilon)
    M_dist = 0.5 * (P_dist + Q_dist)
    # For JS = 0.5 * (KL(P||M) + KL(Q||M))
    # F.kl_div(log_input, target) where log_input is log(P) and target is M
    kl_P_M = F.kl_div(torch.log(P_dist + epsilon), M_dist, reduction='sum', log_target=False)
    kl_Q_M = F.kl_div(torch.log(Q_dist + epsilon), M_dist, reduction='sum', log_target=False)
    return F.relu(0.5 * (kl_P_M + kl_Q_M))

def compute_angle_wasserstein(pred_angles: torch.Tensor, target_angles: torch.Tensor,
                              num_bins: int = 36, angle_range: Tuple[float, float] = (-np.pi, np.pi)) -> torch.Tensor:
    pred_detached, target_detached = pred_angles.detach(), target_angles.detach()
    device = pred_angles.device
    if pred_detached.numel() == 0 or target_detached.numel() == 0:
        return torch.tensor(0.0, device=device)
    pred_hist = torch.histc(pred_detached, bins=num_bins, min=angle_range[0], max=angle_range[1])
    target_hist = torch.histc(target_detached, bins=num_bins, min=angle_range[0], max=angle_range[1])
    epsilon = 1e-10
    pred_dist = pred_hist / (pred_hist.sum() + epsilon)
    target_dist = target_hist / (target_hist.sum() + epsilon)
    pred_cdf = torch.cumsum(pred_dist, dim=0)
    target_cdf = torch.cumsum(target_dist, dim=0)
    return torch.sum(torch.abs(pred_cdf - target_cdf))

# --- MSE Utilities ---
def compute_bb_sc_mse(predictions: torch.Tensor, targets: torch.Tensor,
                      backbone_indices: torch.Tensor, sidechain_indices: torch.Tensor,
                      logger: logging.Logger) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    criterion = nn.MSELoss() # Make sure nn is imported as torch.nn
    targets_casted = targets.to(predictions.dtype)
    overall_mse = criterion(predictions, targets_casted)
    backbone_mse = torch.tensor(0.0, device=predictions.device)
    sidechain_mse = torch.tensor(0.0, device=predictions.device)
    try:
        if backbone_indices.numel() > 0:
            backbone_mse = criterion(predictions[backbone_indices], targets_casted[backbone_indices])
        if sidechain_indices.numel() > 0:
            sidechain_mse = criterion(predictions[sidechain_indices], targets_casted[sidechain_indices])
    except IndexError as e:
        logger.error(f"MSE Indexing Error: {e}. Pred shape: {predictions.shape}, Target shape: {targets_casted.shape}, "
                     f"BB_idx_max: {backbone_indices.max() if backbone_indices.numel() > 0 else 'N/A'}, "
                     f"SC_idx_max: {sidechain_indices.max() if sidechain_indices.numel() > 0 else 'N/A'}", exc_info=False)
    return overall_mse, backbone_mse, sidechain_mse

# --- MLP Builder ---
def build_decoder_mlp(input_dim: int, output_dim: int, num_total_layers: int, # Renamed from num_hidden_layers for clarity
                      hidden_layer_dim: int = 128) -> nn.Sequential:
    layers: List[nn.Module] = []
    current_dim = input_dim
    if num_total_layers <= 0:
        raise ValueError("Number of MLP layers must be at least 1.")
    elif num_total_layers == 1:
        layers.append(nn.Linear(current_dim, output_dim))
    else:
        # Input layer to first hidden layer
        layers.extend([
            nn.Linear(current_dim, hidden_layer_dim),
            nn.BatchNorm1d(hidden_layer_dim), # Original code had BN here
            nn.ReLU()
        ])
        current_dim = hidden_layer_dim
        # Intermediate hidden layers (num_total_layers - 2 of these)
        for _ in range(num_total_layers - 2): # Corrected loop range
            layers.extend([
                nn.Linear(current_dim, hidden_layer_dim),
                nn.BatchNorm1d(hidden_layer_dim),
                nn.ReLU()
            ])
        # Final hidden layer to output layer
        layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)
