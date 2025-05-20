import os
import sys
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, Tuple

def save_checkpoint(state: Dict, filename: str, logger: logging.Logger):
    """Saves model and optimizer state to a file."""
    try:
        torch.save(state, filename)
        logger.debug(f"Checkpoint successfully saved to: {filename}")
    except IOError as e:
        logger.error(f"Error saving checkpoint to {filename}: {e}")
    sys.stdout.flush() # Ensure log messages are flushed

def load_checkpoint(model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    filename: str,
                    device: torch.device,
                    logger: logging.Logger) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int]:
    """
    Loads model and optimizer state from a checkpoint file.
    Returns the model, optimizer, and the epoch number *after which training should resume* (i.e., last completed epoch).
    """
    start_epoch = 0 # Default: start from epoch 0 (meaning training for epoch 1)
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint from: '{filename}'")
        try:
            checkpoint = torch.load(filename, map_location=device)
            start_epoch = checkpoint.get("epoch", 0) # Get the epoch number that was COMPLETED

            # Load model state dictionary
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError as e: # Handle cases where keys might not perfectly match (e.g. DDP wrapper)
                logger.warning(f"Could not load model state_dict strictly: {e}. Attempting non-strict load.")
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                logger.warning("Model state_dict loaded non-strictly. Review if this was expected.")

            # Load optimizer state dictionary
            if optimizer and "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    logger.info("Optimizer state loaded successfully.")
                    # Move optimizer state to the correct device
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}. Optimizer will be re-initialized.")
            elif optimizer:
                logger.warning("Optimizer state_dict not found in checkpoint. Optimizer will be re-initialized.")

            model.to(device) # Ensure model is on the correct device after loading
            logger.info(f"Checkpoint loaded. Model will resume training after completed epoch {start_epoch}.")
        except Exception as e:
            logger.error(f"Error loading checkpoint from {filename}: {e}", exc_info=True)
            start_epoch = 0 # Reset epoch if loading failed
            logger.warning("Checkpoint loading failed. Model will train from scratch.")
    else:
        logger.info(f"No checkpoint found at '{filename}'. Model will train from scratch.")
        model.to(device) # Ensure model is on device even if not loading checkpoint

    return model, optimizer, start_epoch
