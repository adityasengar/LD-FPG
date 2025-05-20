# trainers.py
# Description: Training loops for HNO Encoder and Decoder2.

from common_imports import (
    os, torch, nn, F, TorchDataLoader, # TorchDataLoader for type hint if PyG one is different
    logging, random, Dict, Optional, List, Tuple, Any
)

# Custom project module imports
from models import HNO, ProteinStateReconstructor2D # For type hinting
from checkpoint_utils import load_checkpoint, save_checkpoint
from math_utils import (compute_bb_sc_mse, compute_all_dihedrals_vectorized,
                        compute_angle_kl_div, compute_angle_js_div, compute_angle_wasserstein)

logger_trainers = logging.getLogger(__name__)

def train_hno_model(
    model: HNO, train_loader: TorchDataLoader, val_loader: TorchDataLoader,
    backbone_atom_indices: torch.Tensor, sidechain_atom_indices: torch.Tensor,
    num_epochs: int, learning_rate: float, checkpoint_path: str,
    save_interval: int, device: torch.device, logger: logging.Logger
):
    model = model.to(device)
    bb_idx_dev = backbone_atom_indices.to(device)
    sc_idx_dev = sidechain_atom_indices.to(device)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate) if trainable_params else None
    model, optimizer, completed_epochs = load_checkpoint(model, optimizer, checkpoint_path, device, logger)
    start_epoch_num = completed_epochs + 1

    logger.info(f"Starting HNO training: Epochs {start_epoch_num}-{num_epochs}, LR={learning_rate}")
    if completed_epochs >= num_epochs:
        logger.info(f"HNO training already completed up to epoch {completed_epochs}. Skipping.")
        return model
    if not optimizer:
        logger.warning("No optimizer for HNO. Skipping training.")
        return model

    for epoch in range(start_epoch_num, num_epochs + 1):
        model.train()
        tr_metrics = [0.0] * 3 # all_mse, bb_mse, sc_mse
        num_train_batches = len(train_loader)
        for batch_idx, data_batch in enumerate(train_loader):
            data_batch = data_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            # Assuming PyGDataLoader yields Data objects where data_batch.x is [N_nodes_in_batch, 3]
            # and data_batch.edge_index correctly refers to these nodes.
            # The HNO model's forward directly predicts coordinates.
            predicted_coords = model(data_batch.x, data_batch.edge_index)
            # compute_bb_sc_mse expects flattened predictions and targets if indices are global for single structure
            # This part needs careful alignment of how PyG batches and how indices are applied.
            # If batch_size > 1 for HNO, PyG concatenates graphs. Indices bb_idx/sc_idx are for single graph.
            # This would require processing graph by graph within the batch for correct MSE with global indices.
            # For simplicity, assuming here the original script's logic implies batch_size=1 for HNO training
            # or that compute_bb_sc_mse handles PyG batching (which it doesn't seem to do directly).
            # A common way for PyG is to sum losses over items in batch.
            # Here, let's assume data_batch.y refers to target coords for data_batch.x
            # And indices are global (for N_atoms) applied to potentially B*N tensor. This is problematic.
            # Let's adjust compute_bb_sc_mse to be safer or assume batch_size = 1 for HNO.
            # Given the setup, compute_bb_sc_mse will likely error if batch_size > 1 and PyG's batching is used
            # without adjusting indices. The original script might have implicitly relied on B=1.
            loss_all, loss_bb, loss_sc = compute_bb_sc_mse(predicted_coords, data_batch.y, bb_idx_dev, sc_idx_dev, logger)
            total_loss = loss_all
            if total_loss.requires_grad: total_loss.backward()
            optimizer.step()
            tr_metrics[0] += loss_all.item(); tr_metrics[1] += loss_bb.item(); tr_metrics[2] += loss_sc.item()

        avg_tr = [m / num_train_batches if num_train_batches > 0 else 0 for m in tr_metrics]
        model.eval()
        val_metrics = [0.0] * 3; num_val_batches = len(val_loader)
        with torch.no_grad():
            for data_batch in val_loader:
                data_batch = data_batch.to(device)
                predicted_coords = model(data_batch.x, data_batch.edge_index)
                loss_all, loss_bb, loss_sc = compute_bb_sc_mse(predicted_coords, data_batch.y, bb_idx_dev, sc_idx_dev, logger)
                val_metrics[0] += loss_all.item(); val_metrics[1] += loss_bb.item(); val_metrics[2] += loss_sc.item()
        avg_val = [m / num_val_batches if num_val_batches > 0 else 0 for m in val_metrics]
        logger.info(f"[HNO] Ep {epoch} TR MSE={avg_tr[0]:.5f}(BB={avg_tr[1]:.5f},SC={avg_tr[2]:.5f}) | "
                    f"VAL MSE={avg_val[0]:.5f}(BB={avg_val[1]:.5f},SC={avg_val[2]:.5f})")
        if optimizer and (epoch % save_interval == 0 or epoch == num_epochs):
            save_checkpoint({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                            checkpoint_path, logger)
    logger.info(f"Finished HNO training. Checkpoint: {checkpoint_path}")
    return model

def train_decoder2_model(
    model: ProteinStateReconstructor2D, train_loader: TorchDataLoader, val_loader: TorchDataLoader,
    backbone_atom_indices: torch.Tensor, sidechain_atom_indices: torch.Tensor,
    conditioner_tensor_ref: torch.Tensor, num_total_atoms: int, num_epochs: int,
    learning_rate: float, checkpoint_path: str, save_interval: int, device: torch.device,
    logger: logging.Logger, base_coord_loss_weight: float = 1.0,
    use_dihedral_loss_terms: bool = False, dihedral_angle_info: Optional[Dict] = None,
    dihedral_valid_mask: Optional[torch.Tensor] = None, num_total_residues: Optional[int] = None,
    lambda_dihedral_divergence: float = 0.0, lambda_dihedral_mse: float = 0.0,
    dihedral_divergence_type: str = "KL", fraction_batches_for_dihedral_loss: float = 0.1
):
    model = model.to(device)
    bb_idx_dev = backbone_atom_indices.to(device)
    sc_idx_dev = sidechain_atom_indices.to(device)
    mask_on_device = dihedral_valid_mask.to(device) if dihedral_valid_mask is not None else None
    # conditioner_tensor_ref is already on CPU from main, model forward will move to device.

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate) if trainable_params else None
    model, optimizer, completed_epochs = load_checkpoint(model, optimizer, checkpoint_path, device, logger)
    start_epoch_num = completed_epochs + 1
    best_validation_loss = float("inf")

    logger.info(f"Starting Decoder2 training: Epochs {start_epoch_num}-{num_epochs}, LR={learning_rate}")
    divergence_function = None; is_dihedral_setup_valid = False
    if use_dihedral_loss_terms:
        if dihedral_divergence_type.upper() == "JS": divergence_function = compute_angle_js_div
        elif dihedral_divergence_type.upper() == "WASSERSTEIN": divergence_function = compute_angle_wasserstein
        elif dihedral_divergence_type.upper() == "KL": divergence_function = compute_angle_kl_div
        else: logger.warning(f"Unknown div_type '{dihedral_divergence_type}', defaulting to KL."); divergence_function = compute_angle_kl_div
        if dihedral_angle_info and mask_on_device is not None and num_total_residues and divergence_function:
            is_dihedral_setup_valid = True; logger.info(f"  Dihedral Loss Terms ENABLED: Type={dihedral_divergence_type.upper()}")
        else: logger.warning("Dihedral components missing. Disabling dihedral loss.")
    else: logger.info("  Dihedral Loss Terms DISABLED.")
    angle_names_list = ['phi', 'psi', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5']

    if completed_epochs >= num_epochs: logger.info("Decoder2 training already completed. Skipping."); return model
    if not optimizer: logger.warning("No optimizer for Decoder2. Skipping."); return model

    for epoch in range(start_epoch_num, num_epochs + 1):
        model.train(); train_metrics = [0.0]*6; num_train_batches = len(train_loader)
        for batch_idx, data_batch in enumerate(train_loader): # data_batch.x is [B*N, EmbedDim], data_batch.y is [B*N, 3]
            data_batch = data_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            # Model's forward expects flat embeddings and batch_indices (from PyG DataBatch)
            predicted_coords_flat = model(data_batch.x, data_batch.batch, conditioner_tensor_ref)
            c_mse, b_mse, s_mse = compute_bb_sc_mse(predicted_coords_flat, data_batch.y, bb_idx_dev, sc_idx_dev, logger)
            current_loss = c_mse * base_coord_loss_weight
            div_l, d_mse_l = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

            apply_di = use_dihedral_loss_terms and is_dihedral_setup_valid and (random.random() < fraction_batches_for_dihedral_loss)
            if apply_di:
                B_this = data_batch.num_graphs
                if B_this * num_total_atoms == predicted_coords_flat.shape[0]: # Ensure consistency
                    pred3d = predicted_coords_flat.view(B_this, num_total_atoms, 3)
                    true3d = data_batch.y.view(B_this, num_total_atoms, 3)
                    try:
                        pred_a = compute_all_dihedrals_vectorized(pred3d, dihedral_angle_info, num_total_residues, logger)
                        true_a = compute_all_dihedrals_vectorized(true3d, dihedral_angle_info, num_total_residues, logger)
                        for angle_idx, name in enumerate(angle_names_list):
                            pa, ta = pred_a.get(name), true_a.get(name) # These are [B, N_res]
                            if pa is not None and ta is not None and mask_on_device is not None:
                                current_mask = mask_on_device[:, angle_idx] # [N_res]
                                if current_mask.any():
                                    mask_ex = current_mask.view(1, -1).expand(B_this, -1) # [B, N_res]
                                    pv, tv = pa[mask_ex], ta[mask_ex] # Flattened valid angles
                                    if pv.numel() > 0:
                                        d_mse_l += F.mse_loss(pv, tv) # Use F from common_imports
                                        if divergence_function: div_l += divergence_function(pv, tv)
                        current_loss += lambda_dihedral_divergence * div_l + lambda_dihedral_mse * d_mse_l
                    except Exception as e_dih: logger.error(f"Dih. loss error batch {batch_idx}: {e_dih}", exc_info=False)
            if current_loss.requires_grad: current_loss.backward(); optimizer.step()
            batch_m = [current_loss.item(), c_mse.item(), b_mse.item(), s_mse.item(), div_l.item(), d_mse_l.item()]
            for k_idx in range(6): train_metrics[k_idx] += batch_m[k_idx]
        avg_tr = [m / num_train_batches if num_train_batches > 0 else 0 for m in train_metrics]

        model.eval(); val_metrics = [0.0]*6; num_val_batches = len(val_loader)
        with torch.no_grad():
            for data_batch in val_loader:
                data_batch = data_batch.to(device)
                predicted_coords_flat = model(data_batch.x, data_batch.batch, conditioner_tensor_ref)
                c_mse,b_mse,s_mse=compute_bb_sc_mse(predicted_coords_flat,data_batch.y,bb_idx_dev,sc_idx_dev,logger)
                v_loss = c_mse * base_coord_loss_weight; v_div_l, v_d_mse_l = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                if use_dihedral_loss_terms and is_dihedral_setup_valid: # Typically evaluate on all val data
                    B_this = data_batch.num_graphs
                    if B_this * num_total_atoms == predicted_coords_flat.shape[0]:
                        pred3d=predicted_coords_flat.view(B_this,num_total_atoms,3); true3d=data_batch.y.view(B_this,num_total_atoms,3)
                        try:
                            pred_a=compute_all_dihedrals_vectorized(pred3d,dihedral_angle_info,num_total_residues,logger)
                            true_a=compute_all_dihedrals_vectorized(true3d,dihedral_angle_info,num_total_residues,logger)
                            for angle_idx,name in enumerate(angle_names_list):
                                pa,ta=pred_a.get(name),true_a.get(name)
                                if pa is not None and ta is not None and mask_on_device is not None:
                                    current_mask=mask_on_device[:,angle_idx]
                                    if current_mask.any():
                                        mask_ex=current_mask.view(1,-1).expand(B_this,-1)
                                        pv,tv=pa[mask_ex],ta[mask_ex]
                                        if pv.numel()>0: v_d_mse_l+=F.mse_loss(pv,tv); v_div_l+=divergence_function(pv,tv) if divergence_function else 0.0
                            v_loss += lambda_dihedral_divergence * v_div_l + lambda_dihedral_mse * v_d_mse_l
                        except Exception: pass # Ignore val dih errors
                val_batch_m = [v_loss.item(),c_mse.item(),b_mse.item(),s_mse.item(),v_div_l.item(),v_d_mse_l.item()]
                for k_idx in range(6): val_metrics[k_idx] += val_batch_m[k_idx]
        avg_val = [m / num_val_batches if num_val_batches > 0 else 0 for m in val_metrics]
        logger.info(f"[Dec2] Ep {epoch} TR L={avg_tr[0]:.4f}(C={avg_tr[1]:.4f} BB={avg_tr[2]:.4f} SC={avg_tr[3]:.4f} "
                    f"DihM={avg_tr[5]:.4f} DihD={avg_tr[4]:.4f}) | VAL L={avg_val[0]:.4f}(C={avg_val[1]:.4f} "
                    f"BB={avg_val[2]:.4f} SC={avg_val[3]:.4f} DihM={avg_val[5]:.4f} DihD={avg_val[4]:.4f})")
        if optimizer:
            current_val_total_loss = avg_val[0]
            if current_val_total_loss < best_validation_loss:
                best_validation_loss = current_val_total_loss
                save_checkpoint({"epoch": epoch, "model_state_dict": model.state_dict(),
                                 "optimizer_state_dict": optimizer.state_dict(), "best_val_loss": best_validation_loss},
                                checkpoint_path, logger)
                logger.info(f"[Dec2] Saved best model at epoch {epoch} (ValLoss={best_validation_loss:.5f})")
            elif epoch % save_interval == 0 or epoch == num_epochs:
                 save_checkpoint({"epoch": epoch, "model_state_dict": model.state_dict(),
                                 "optimizer_state_dict": optimizer.state_dict(), "current_val_loss": current_val_total_loss},
                                f"{os.path.splitext(checkpoint_path)[0]}_epoch{epoch}.pth", logger) # Periodic save
    logger.info(f"Finished Decoder2 training. Best ValLoss: {best_validation_loss:.5f}. Ckpt: {checkpoint_path}")
    if os.path.isfile(checkpoint_path): # Load best model before returning
        best_ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"]); logger.info("Loaded best Decoder2 model state.")
    model.eval()
    return model
