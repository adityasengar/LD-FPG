import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader # Renamed to avoid conflict if PyG DataLoader is also imported
import logging
import random
from typing import Dict, Optional, List, Tuple, Any

# Assuming these are in their respective new files
from models import HNO, ProteinStateReconstructor2D
from checkpoint_utils import load_checkpoint, save_checkpoint
from math_utils import compute_bb_sc_mse, compute_all_dihedrals_vectorized, \
                       compute_angle_kl_div, compute_angle_js_div, compute_angle_wasserstein

logger_trainers = logging.getLogger(__name__) # Module-specific logger

# --- Train HNO Encoder ---
def train_hno_model(
    model: HNO,
    train_loader: TorchDataLoader,
    val_loader: TorchDataLoader,
    backbone_atom_indices: torch.Tensor, # Global indices for a single structure
    sidechain_atom_indices: torch.Tensor, # Global indices for a single structure
    num_epochs: int,
    learning_rate: float,
    checkpoint_path: str,
    save_interval: int,
    device: torch.device,
    logger: logging.Logger # Pass logger instance
):
    model = model.to(device)
    # Ensure indices are on the correct device for MSE calculation within the loop
    # These are global indices, used when data.x/y are for single structures [N_atoms, 3]
    bb_idx_dev = backbone_atom_indices.to(device)
    sc_idx_dev = sidechain_atom_indices.to(device)

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate) if trainable_params else None

    model, optimizer, completed_epochs = load_checkpoint(model, optimizer, checkpoint_path, device, logger)
    start_epoch_num = completed_epochs + 1 # Training starts from the next epoch

    logger.info(f"Starting HNO encoder training from epoch {start_epoch_num}/{num_epochs}. Learning Rate: {learning_rate}")

    if completed_epochs >= num_epochs:
        logger.info(f"Loaded checkpoint indicates epoch {completed_epochs} already completed. Target epochs: {num_epochs}. Skipping HNO training.")
        return model

    if not optimizer:
        logger.warning("No optimizer initialized for HNO (likely no trainable parameters). Skipping training.")
        return model

    for epoch in range(start_epoch_num, num_epochs + 1):
        model.train()
        train_total_mse, train_bb_mse, train_sc_mse = 0.0, 0.0, 0.0
        num_train_batches = len(train_loader)

        for batch_idx, data_batch in enumerate(train_loader):
            data_batch = data_batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            # The HNO model's forward pass directly predicts coordinates
            # log_shapes_once = (epoch == start_epoch_num and batch_idx == 0 and logger.isEnabledFor(logging.DEBUG))
            # Original code had log_debug in HNO forward, I removed direct call to simplify trainer. HNO internal logging is fine.
            predicted_coords = model(data_batch.x, data_batch.edge_index) # data.x is [N_atoms_in_batch, 3]

            # MSE computation assumes data.y (target) is also [N_atoms_in_batch, 3]
            # and bb_idx_dev/sc_idx_dev are for a single structure, applied if batch is effectively N_atoms.
            # If PyG DataLoader unnests batch to N_atoms_in_batch, indices need care.
            # The original code's compute_bb_sc_mse took flat preds/targets, and indices were global.
            # This implies that data.x and data.y are effectively [N_atoms, 3] if batch_size=1,
            # or they need to be handled carefully if PyG combines graphs in a batch.
            # For ChebNet, typically operate on full graphs, so data.x is likely [B*N, 3]
            # Let's assume data.x from PyG loader for HNO is [N_atoms_in_batch, 3] where batch means
            # data.ptr is used to delineate individual graphs.
            # The HNO model as written processes a single graph [N, dim] or a batch [B*N, dim] if using PyG batching.
            # The compute_bb_sc_mse in original code works on flattened tensors,
            # it expects bb_idx/sc_idx to correspond to indices within that flattened tensor.
            # This setup is a bit tricky with PyG batching. Assuming for now it's correct as per original.

            loss_all, loss_bb, loss_sc = compute_bb_sc_mse(predicted_coords, data_batch.y, bb_idx_dev, sc_idx_dev, logger)
            total_loss = loss_all
            total_loss.backward()
            optimizer.step()

            train_total_mse += loss_all.item()
            train_bb_mse += loss_bb.item()
            train_sc_mse += loss_sc.item()

        avg_train_mse = [val / num_train_batches if num_train_batches > 0 else 0 for val in [train_total_mse, train_bb_mse, train_sc_mse]]

        # Validation phase
        model.eval()
        val_total_mse, val_bb_mse, val_sc_mse = 0.0, 0.0, 0.0
        num_val_batches = len(val_loader)
        with torch.no_grad():
            for data_batch in val_loader:
                data_batch = data_batch.to(device)
                predicted_coords = model(data_batch.x, data_batch.edge_index)
                loss_all, loss_bb, loss_sc = compute_bb_sc_mse(predicted_coords, data_batch.y, bb_idx_dev, sc_idx_dev, logger)
                val_total_mse += loss_all.item()
                val_bb_mse += loss_bb.item()
                val_sc_mse += loss_sc.item()

        avg_val_mse = [val / num_val_batches if num_val_batches > 0 else 0 for val in [val_total_mse, val_bb_mse, val_sc_mse]]

        logger.info(f"[HNO Enc] Epoch {epoch}/{num_epochs} | "
                    f"Train MSE: All={avg_train_mse[0]:.5f} (BB={avg_train_mse[1]:.5f}, SC={avg_train_mse[2]:.5f}) | "
                    f"Val MSE: All={avg_val_mse[0]:.5f} (BB={avg_val_mse[1]:.5f}, SC={avg_val_mse[2]:.5f})")

        if optimizer and (epoch % save_interval == 0 or epoch == num_epochs):
            save_checkpoint({
                "epoch": epoch, # Save the epoch number that was just completed
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, checkpoint_path, logger)

    logger.info(f"Finished HNO encoder training. Final checkpoint saved to: {checkpoint_path}")
    return model


# --- Train Decoder2 ---
def train_decoder2_model(
    model: ProteinStateReconstructor2D,
    train_loader: TorchDataLoader, # Expects Data(x=embeddings, y=true_coords)
    val_loader: TorchDataLoader,
    backbone_atom_indices: torch.Tensor, # Global indices for a single structure
    sidechain_atom_indices: torch.Tensor, # Global indices for a single structure
    conditioner_tensor: torch.Tensor,  # Shared conditioner [N_atoms, cond_dim]
    num_total_atoms: int,              # N for a single protein
    num_epochs: int,
    learning_rate: float,
    checkpoint_path: str,
    save_interval: int,
    device: torch.device,
    logger: logging.Logger, # Pass logger instance
    base_coord_loss_weight: float = 1.0,
    use_dihedral_loss_terms: bool = False,
    dihedral_angle_info: Optional[Dict] = None,
    dihedral_valid_mask: Optional[torch.Tensor] = None, # [N_res, num_angle_types]
    num_total_residues: Optional[int] = None,
    lambda_dihedral_divergence: float = 0.0,
    lambda_dihedral_mse: float = 0.0,
    dihedral_divergence_type: str = "KL",
    fraction_batches_for_dihedral_loss: float = 0.1
):
    model = model.to(device)
    # Ensure indices and conditioner are on the correct device
    bb_idx_dev = backbone_atom_indices.to(device)
    sc_idx_dev = sidechain_atom_indices.to(device)
    # Conditioner is per-structure, not per-batch item, but needs to be on device
    # It will be expanded within the model's forward pass.
    # The model's forward expects conditioner_per_atom for a single structure.
    # conditioner_tensor_dev = conditioner_tensor.to(device)

    mask_on_device = dihedral_valid_mask.to(device) if dihedral_valid_mask is not None else None

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate) if trainable_params else None

    model, optimizer, completed_epochs = load_checkpoint(model, optimizer, checkpoint_path, device, logger)
    start_epoch_num = completed_epochs + 1
    best_validation_loss = float("inf") # Or use training loss as in original code for saving best

    logger.info(f"Starting Decoder2 training from epoch {start_epoch_num}/{num_epochs}. LR: {learning_rate}, BaseCoordLossW: {base_coord_loss_weight}")

    divergence_function = None
    is_dihedral_setup_valid = False
    if use_dihedral_loss_terms:
        if dihedral_divergence_type.upper() == "JS": divergence_function = compute_angle_js_div
        elif dihedral_divergence_type.upper() == "WASSERSTEIN": divergence_function = compute_angle_wasserstein
        elif dihedral_divergence_type.upper() == "KL": divergence_function = compute_angle_kl_div
        else:
            logger.warning(f"Unknown dihedral_divergence_type '{dihedral_divergence_type}', defaulting to KL.")
            dihedral_divergence_type = "KL" # Ensure consistency
            divergence_function = compute_angle_kl_div

        if dihedral_angle_info and mask_on_device is not None and num_total_residues and divergence_function:
            is_dihedral_setup_valid = True
            logger.info(f"  Dihedral Loss Terms ENABLED: Type={dihedral_divergence_type.upper()}, "
                        f"Lambda_Div={lambda_dihedral_divergence}, Lambda_MSE={lambda_dihedral_mse}, "
                        f"Fraction_Batches={fraction_batches_for_dihedral_loss}")
        else:
            logger.warning("Dihedral components (info, mask, N_res, div_func) missing or invalid. Disabling dihedral loss terms.")
    else:
        logger.info("  Dihedral Loss Terms DISABLED by configuration.")

    angle_names_list = ['phi', 'psi', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5']

    if completed_epochs >= num_epochs:
        logger.info(f"Loaded checkpoint indicates epoch {completed_epochs} already completed. Target epochs: {num_epochs}. Skipping Decoder2 training.")
        return model

    if not optimizer:
        logger.warning("No optimizer initialized for Decoder2. Skipping training.")
        return model

    for epoch in range(start_epoch_num, num_epochs + 1):
        model.train()
        # Metrics: [total_loss, coord_mse, bb_mse, sc_mse, dihedral_divergence_loss, dihedral_angle_mse_loss]
        train_metrics_sum = [0.0] * 6
        num_train_batches = len(train_loader)

        for batch_idx, data_batch in enumerate(train_loader):
            # data_batch.x should be [B*N_atoms, embedding_dim], data_batch.y is [B*N_atoms, 3]
            data_batch = data_batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Forward pass: model input is [B*N_atoms, embedding_dim] and conditioner [N_atoms, cond_dim]
            # The model handles batching internally for the conditioner.
            predicted_coords_flat = model(data_batch.x, data_batch.batch, conditioner_tensor) # Pass global conditioner

            coord_mse_all, coord_mse_bb, coord_mse_sc = compute_bb_sc_mse(
                predicted_coords_flat, data_batch.y, bb_idx_dev, sc_idx_dev, logger
            )
            current_loss = coord_mse_all * base_coord_loss_weight

            current_dihedral_div_loss = torch.tensor(0.0, device=device)
            current_dihedral_angle_mse_loss = torch.tensor(0.0, device=device)

            # Stochastic application of dihedral loss
            apply_dihedral_this_batch = use_dihedral_loss_terms and is_dihedral_setup_valid and \
                                      (random.random() < fraction_batches_for_dihedral_loss)

            if apply_dihedral_this_batch:
                batch_size_this_iter = data_batch.num_graphs # Number of graphs in this PyG batch
                # Reshape predictions and targets to [B, N_atoms, 3] for dihedral calculation
                pred_coords_3d = predicted_coords_flat.view(batch_size_this_iter, num_total_atoms, 3)
                true_coords_3d = data_batch.y.view(batch_size_this_iter, num_total_atoms, 3)

                try:
                    pred_angles_dict = compute_all_dihedrals_vectorized(pred_coords_3d, dihedral_angle_info, num_total_residues, logger)
                    true_angles_dict = compute_all_dihedrals_vectorized(true_coords_3d, dihedral_angle_info, num_total_residues, logger)

                    for angle_name_idx, angle_name_str in enumerate(angle_names_list):
                        pred_angle_values = pred_angles_dict.get(angle_name_str) # [B, N_res]
                        true_angle_values = true_angles_dict.get(angle_name_str) # [B, N_res]

                        if pred_angle_values is not None and true_angle_values is not None:
                            # Get the mask for this specific angle type: [N_res]
                            current_angle_mask = mask_on_device[:, angle_name_idx] # boolean
                            if current_angle_mask.any(): # If any angles of this type are valid
                                # Expand mask for batch: [B, N_res]
                                expanded_mask_for_batch = current_angle_mask.view(1, -1).expand(batch_size_this_iter, -1)

                                # Select valid angles using the expanded mask
                                valid_pred_angles = pred_angle_values[expanded_mask_for_batch] # Flat tensor
                                valid_true_angles = true_angle_values[expanded_mask_for_batch] # Flat tensor

                                if valid_pred_angles.numel() > 0:
                                    current_dihedral_angle_mse_loss += F.mse_loss(valid_pred_angles, valid_true_angles)
                                    if divergence_function:
                                      current_dihedral_div_loss += divergence_function(valid_pred_angles, valid_true_angles)
                    current_loss += lambda_dihedral_mse * current_dihedral_angle_mse_loss + \
                                    lambda_dihedral_divergence * current_dihedral_div_loss
                except Exception as e:
                    logger.error(f"Error during dihedral loss computation for batch {batch_idx}: {e}", exc_info=False)


            if current_loss.requires_grad: # Ensure loss requires grad before backward (e.g. if all weights are 0)
                current_loss.backward()
                optimizer.step()

            # Accumulate metrics
            current_batch_metrics = [
                current_loss.item(), coord_mse_all.item(), coord_mse_bb.item(), coord_mse_sc.item(),
                current_dihedral_div_loss.item(), current_dihedral_angle_mse_loss.item()
            ]
            for k in range(6): train_metrics_sum[k] += current_batch_metrics[k]
        # End of training epoch loop

        avg_train_metrics = [val / num_train_batches if num_train_batches > 0 else 0 for val in train_metrics_sum]

        # Validation phase
        model.eval()
        val_metrics_sum = [0.0] * 6
        num_val_batches = len(val_loader)
        with torch.no_grad():
            for data_batch in val_loader:
                data_batch = data_batch.to(device)
                predicted_coords_flat = model(data_batch.x, data_batch.batch, conditioner_tensor)
                coord_mse_all, coord_mse_bb, coord_mse_sc = compute_bb_sc_mse(
                    predicted_coords_flat, data_batch.y, bb_idx_dev, sc_idx_dev, logger
                )
                current_val_loss = coord_mse_all * base_coord_loss_weight

                current_val_dihedral_div_loss = torch.tensor(0.0, device=device)
                current_val_dihedral_angle_mse_loss = torch.tensor(0.0, device=device)

                if use_dihedral_loss_terms and is_dihedral_setup_valid: # Usually apply to all val batches
                    batch_size_this_iter = data_batch.num_graphs
                    pred_coords_3d = predicted_coords_flat.view(batch_size_this_iter, num_total_atoms, 3)
                    true_coords_3d = data_batch.y.view(batch_size_this_iter, num_total_atoms, 3)
                    try:
                        pred_angles_dict = compute_all_dihedrals_vectorized(pred_coords_3d, dihedral_angle_info, num_total_residues, logger)
                        true_angles_dict = compute_all_dihedrals_vectorized(true_coords_3d, dihedral_angle_info, num_total_residues, logger)
                        for angle_name_idx, angle_name_str in enumerate(angle_names_list):
                            # ... (similar dihedral calculation as in training loop) ...
                            pred_angle_values = pred_angles_dict.get(angle_name_str)
                            true_angle_values = true_angles_dict.get(angle_name_str)
                            if pred_angle_values is not None and true_angle_values is not None:
                                current_angle_mask = mask_on_device[:, angle_name_idx]
                                if current_angle_mask.any():
                                    expanded_mask_for_batch = current_angle_mask.view(1, -1).expand(batch_size_this_iter, -1)
                                    valid_pred_angles = pred_angle_values[expanded_mask_for_batch]
                                    valid_true_angles = true_angle_values[expanded_mask_for_batch]
                                    if valid_pred_angles.numel() > 0:
                                        current_val_dihedral_angle_mse_loss += F.mse_loss(valid_pred_angles, valid_true_angles)
                                        if divergence_function:
                                          current_val_dihedral_div_loss += divergence_function(valid_pred_angles, valid_true_angles)
                        current_val_loss += lambda_dihedral_mse * current_val_dihedral_angle_mse_loss + \
                                            lambda_dihedral_divergence * current_val_dihedral_div_loss
                    except Exception: pass # Ignore errors in validation dihedral calculation

                current_val_batch_metrics = [
                    current_val_loss.item(), coord_mse_all.item(), coord_mse_bb.item(), coord_mse_sc.item(),
                    current_val_dihedral_div_loss.item(), current_val_dihedral_angle_mse_loss.item()
                ]
                for k in range(6): val_metrics_sum[k] += current_val_batch_metrics[k]
        # End of validation loop

        avg_val_metrics = [val / num_val_batches if num_val_batches > 0 else 0 for val in val_metrics_sum]

        logger.info(
            f"[Dec2] Ep {epoch} | "
            f"TR: Tot={avg_train_metrics[0]:.4f} Coord={avg_train_metrics[1]:.4f} "
            f"(BB={avg_train_metrics[2]:.4f},SC={avg_train_metrics[3]:.4f}) "
            f"Dih(MSE={avg_train_metrics[5]:.4f},{dihedral_divergence_type.upper()}={avg_train_metrics[4]:.4f}) | "
            f"VAL: Tot={avg_val_metrics[0]:.4f} Coord={avg_val_metrics[1]:.4f} "
            f"(BB={avg_val_metrics[2]:.4f},SC={avg_val_metrics[3]:.4f}) "
            f"Dih(MSE={avg_val_metrics[5]:.4f},{dihedral_divergence_type.upper()}={avg_val_metrics[4]:.4f})"
        )

        # Save checkpoint logic (original saved based on training loss improvement)
        if optimizer:
            current_total_train_loss = avg_train_metrics[0]
            # Using validation loss to save best model is more standard
            current_total_val_loss = avg_val_metrics[0]
            if current_total_val_loss < best_validation_loss: # Save based on validation loss
                best_validation_loss = current_total_val_loss
                save_checkpoint({
                    "epoch": epoch, # Epoch just completed
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_validation_loss
                }, checkpoint_path, logger)
                logger.info(f"[Dec2] Saved best model checkpoint at epoch {epoch} (ValLoss={best_validation_loss:.5f})")
            elif epoch % save_interval == 0 or epoch == num_epochs: # Also save periodically or at the end
                 # Save a different file for periodic saves if desired, e.g. f"{checkpoint_path}_epoch{epoch}"
                 # For simplicity, overwriting the main checkpoint path if not best but periodic
                 save_checkpoint({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "current_val_loss": current_total_val_loss # Store current val loss for info
                 }, f"{os.path.splitext(checkpoint_path)[0]}_epoch{epoch}.pth", logger) # Save periodic separately
                 logger.info(f"[Dec2] Saved periodic checkpoint at epoch {epoch}")


    logger.info(f"Finished Decoder2 training. Best validation loss: {best_validation_loss:.5f}. Checkpoint at: {checkpoint_path}")
    # Load the best model before returning
    if os.path.isfile(checkpoint_path):
        best_checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        logger.info(f"Loaded best model from {checkpoint_path} for returning.")
    model.eval()
    return model
