# checkpoint_utils.py
# Description: Helper functions for saving and loading model checkpoints.

from common_imports import (
    os, sys, torch, nn, logging,
    Dict, Optional, Tuple, Any # from typing
)

def save_checkpoint(state: Dict[str, Any], filename: str, logger: logging.Logger):
    """Saves model and optimizer state to a file."""
    try:
        torch.save(state, filename)
        logger.debug(f"Checkpoint successfully saved to: {filename}")
    except IOError as e:
        logger.error(f"Error saving checkpoint to {filename}: {e}")
    sys.stdout.flush()

def load_checkpoint(model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    filename: str,
                    device: torch.device,
                    logger: logging.Logger) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int]:
    """
    Loads model and optimizer state from a checkpoint file.
    Returns model, optimizer, and last completed epoch number (0 if no checkpoint).
    """
    start_epoch = 0
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint from: '{filename}'")
        try:
            checkpoint = torch.load(filename, map_location=device)
            start_epoch = checkpoint.get("epoch", 0)
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError as e:
                logger.warning(f"Could not load model state_dict strictly: {e}. Attempting non-strict load.")
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if optimizer and "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    logger.info("Optimizer state loaded successfully.")
                    for state_val in optimizer.state.values():
                        for k, v_tensor in state_val.items():
                            if isinstance(v_tensor, torch.Tensor):
                                state_val[k] = v_tensor.to(device)
                except Exception as e_opt:
                    logger.warning(f"Could not load optimizer state: {e_opt}. Optimizer will be re-initialized.")
            elif optimizer:
                logger.warning("Optimizer state_dict not found in checkpoint.")
            model.to(device)
            logger.info(f"Checkpoint loaded. Training will resume after completed epoch {start_epoch}.")
        except Exception as e_load:
            logger.error(f"Error loading checkpoint from {filename}: {e_load}", exc_info=True)
            start_epoch = 0
            logger.warning("Checkpoint loading failed. Model will train from scratch.")
    else:
        logger.info(f"No checkpoint found at '{filename}'. Model will train from scratch.")
        model.to(device)
    return model, optimizer, start_epoch
