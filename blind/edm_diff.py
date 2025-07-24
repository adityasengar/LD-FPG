#!/usr/bin/env python3
import os
import sys
import h5py
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional

# ======================================================================================
# (A) Logging Setup
# ======================================================================================
def setup_logger(log_file="edm_latent_training.log"):
    """Sets up a logger that outputs to a file and the console."""
    logger = logging.getLogger("EDM_Latent")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='a') # Use append mode to resume logging
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    
    return logger

# ======================================================================================
# (B) EDM Model Definitions
# ======================================================================================

class DenoisingNetwork(nn.Module):
    """A simple MLP to act as the core denoising network F_theta."""
    def __init__(self, latent_dim: int, hidden_dim: int = 1024, num_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, sigma):
        return self.net(x)

class EDMWrapper(nn.Module):
    """
    A wrapper for the DenoisingNetwork that applies the EDM preconditioning.
    """
    def __init__(self, network: DenoisingNetwork, sigma_data: float):
        super().__init__()
        self.network = network
        self.sigma_data = sigma_data

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.view(-1, 1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (sigma**2 + self.sigma_data**2).sqrt()
        network_input = c_in * x
        network_output = self.network(network_input, sigma)
        denoised_x = c_skip * x + c_out * network_output
        return denoised_x

# ======================================================================================
# (C) Data Handling
# ======================================================================================

class H5Dataset(Dataset):
    """A simple PyTorch Dataset for loading data from an HDF5 file."""
    def __init__(self, h5_path: str, key: str):
        self.h5_path = h5_path
        self.key = key
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f[self.key].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            data = f[self.key][idx]
        return torch.from_numpy(data.astype(np.float32))

# ======================================================================================
# (D) Checkpoint Utilities
# ======================================================================================

def save_checkpoint(state: dict, filename: str, logger: logging.Logger):
    """Saves model and optimizer state."""
    try:
        torch.save(state, filename)
        logger.debug(f"Checkpoint saved to {filename}")
    except IOError as e:
        logger.error(f"Error saving checkpoint to {filename}: {e}")

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, filename: str, device: torch.device, logger: logging.Logger) -> Tuple[int, Optional[float]]:
    """Loads model and optimizer state and returns the completed epoch number."""
    start_epoch = 0
    best_loss = None
    if not os.path.isfile(filename):
        logger.info(f"No checkpoint found at '{filename}'. Starting from scratch.")
        return start_epoch, best_loss

    try:
        logger.info(f"Loading checkpoint from '{filename}'...")
        checkpoint = torch.load(filename, map_location=device)
        
       	model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss') # Use .get for backward compatibility

        # Move optimizer state to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        logger.info(f"Successfully loaded checkpoint. Resuming training from epoch {start_epoch + 1}.")
        if best_loss is not None:
            logger.info(f"Best recorded loss was: {best_loss:.6f}")

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}. Starting from scratch.")
        start_epoch = 0
        best_loss = None

    return start_epoch, best_loss

# ======================================================================================
# (E) EDM Sampler (Heun's 2nd Order ODE Solver)
# ======================================================================================

@torch.no_grad()
def edm_sampler(
    model: EDMWrapper,
    latents_shape: tuple,
    device: torch.device,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    num_steps: int = 18,
    rho: float = 7.0,
    logger: logging.Logger = None,
) -> torch.Tensor:
    """Implements the EDM deterministic sampler using Heun's 2nd order method."""
    if logger: logger.info(f"Starting EDM sampling with {num_steps} steps...")
    
    step_indices = torch.arange(num_steps, device=device)
    t_steps = (sigma_max**(1/rho) + step_indices / (num_steps - 1) * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    x_next = torch.randn(latents_shape, device=device) * t_steps[0]

    for i in range(num_steps):
        t_cur, t_next = t_steps[i], t_steps[i+1]
        x_cur = x_next
        
       	denoised = model(x_cur, t_cur * torch.ones(latents_shape[0], device=device))
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        if i < num_steps - 1:
            denoised_next = model(x_next, t_next * torch.ones(latents_shape[0], device=device))
            d_next = (x_next - denoised_next) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_next)

        if logger and (i + 1) % 5 == 0:
            logger.info(f"  Sampler step {i+1}/{num_steps}, sigma={t_cur.item():.3f}")

    if logger: logger.info("Sampling finished.")
    return x_next

# ======================================================================================
# (F) Score Function (for future Langevin Dynamics)
# ======================================================================================

@torch.no_grad()
def get_score(model: EDMWrapper, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Explicitly calculates the score function Nabla_x log p(x; sigma)."""
    denoised_x = model(x, sigma)
    return (denoised_x - x) / (sigma.view(-1, 1)**2)

# ======================================================================================
# (G) Main Training and Execution Block
# ======================================================================================

def main(args):
    logger = setup_logger(args.log_file)
    logger.info("--- Initializing EDM Training for Latent Space ---")
    
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading latent embeddings from '{args.data_file}' with key '{args.data_key}'")
    try:
        dataset = H5Dataset(args.data_file, args.data_key)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        latent_dim = dataset[0].shape[0]
        logger.info(f"Dataset loaded. Samples: {len(dataset)}, Latent dim: {latent_dim}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}"); sys.exit(1)

    f_theta = DenoisingNetwork(latent_dim, args.hidden_dim, args.num_layers)
    model = EDMWrapper(f_theta, args.sigma_data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info(f"Models initialized. Trainable params: {sum(p.numel() for p in model.parameters())}")

    # --- Load Checkpoint to Resume ---
    start_epoch, best_loss = load_checkpoint(model, optimizer, args.output_model_path, device, logger)
    if best_loss is None:
        best_loss = float('inf')

    if start_epoch >= args.epochs:
        logger.info("Training already completed. Skipping training phase.")
    else:
	logger.info(f"Starting training from epoch {start_epoch + 1} to {args.epochs}.")
        start_time = time.time()
        
       	for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_loss = 0.0
            for i, clean_x in enumerate(dataloader):
                clean_x = clean_x.to(device)
                optimizer.zero_grad()

                rnd_normal = torch.randn([clean_x.shape[0]], device=device)
                sigma = (rnd_normal * args.P_std + args.P_mean).exp()
                noise = torch.randn_like(clean_x)
                noisy_x = clean_x + noise * sigma.view(-1, 1)
                pred_x = model(noisy_x, sigma)
                weight = (sigma**2 + args.sigma_data**2) / (sigma * args.sigma_data)**2
                loss = (weight * F.mse_loss(pred_x, clean_x, reduction='none').mean(dim=-1)).mean()

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_epoch_loss:.6f}")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                logger.info(f"New best loss: {best_loss:.6f}. Saving checkpoint.")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                }, args.output_model_path, logger)

        total_time = time.time() - start_time
        logger.info(f"--- Training Finished in {total_time/3600:.2f} hours ---")

    # --- Generate Samples After Training ---
    # Reload the best model for generation
    logger.info(f"Loading best model from '{args.output_model_path}' for final sampling.")
    final_checkpoint = torch.load(args.output_model_path, map_location=device)
    model.load_state_dict(final_checkpoint['model_state_dict'])
    model.eval()

    logger.info("--- Generating Samples with Trained Model ---")
    generated_latents = edm_sampler(
        model, latents_shape=(args.num_samples, latent_dim), device=device,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        num_steps=args.sampler_steps, rho=args.rho, logger=logger
    )
    
    os.makedirs(os.path.dirname(args.output_samples_path), exist_ok=True)
    with h5py.File(args.output_samples_path, 'w') as f:
        f.create_dataset('generated_latents', data=generated_latents.cpu().numpy())
    logger.info(f"Saved {args.num_samples} generated samples to '{args.output_samples_path}'")
    
    logger.info("--- Example Score Calculation for Langevin ---")
    clean_sample = generated_latents[0:1] 
    example_sigma = torch.tensor([args.sigma_min], device=device) 
    score_vector = get_score(model, clean_sample, example_sigma)
    logger.info(f"Example score vector for a clean sample (at sigma={args.sigma_min}):")
    logger.info(f"Shape: {score_vector.shape}, First 5 values: {score_vector.flatten()[:5].cpu().numpy()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an EDM on Latent Space Embeddings.")
    
    # File Paths
    parser.add_argument("--data_file", type=str, required=True, help="Path to the HDF5 file containing latent embeddings.")
    parser.add_argument("--data_key", type=str, default="pooled_embedding", help="Key for the dataset within the HDF5 file.")
    parser.add_argument("--output_model_path", type=str, default="checkpoints/edm_latent_model.pth", help="Path to save/load the model checkpoint.")
    parser.add_argument("--output_samples_path", type=str, default="generated/edm_latents.h5", help="Path to save generated latent samples.")
    parser.add_argument("--log_file", type=str, default="logs/edm_latent_training.log", help="Path for the log file.")

    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device index to use.")

    # Model Architecture
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension of the denoising MLP.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the denoising MLP.")

    # EDM & Sampler Hyperparameters
    parser.add_argument("--sigma_data", type=float, default=0.5, help="Expected standard deviation of the training data.")
    parser.add_argument("--sigma_min", type=float, default=0.002, help="Minimum noise level for the sampler.")
    parser.add_argument("--sigma_max", type=float, default=80.0, help="Maximum noise level for the sampler and training.")
    parser.add_argument("--P_mean", type=float, default=-1.2, help="Mean of the log-normal distribution for sigma sampling.")
    parser.add_argument("--P_std", type=float, default=1.2, help="Standard deviation of the log-normal distribution for sigma sampling.")
    parser.add_argument("--rho", type=float, default=7.0, help="Time step exponent for the sampler.")
    parser.add_argument("--sampler_steps", type=int, default=40, help="Number of steps for the EDM sampler.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate after training.")
    
    # NOTE: save_interval is removed, as we now save based on best validation loss
    
    args = parser.parse_args()
    
    for path in [args.output_model_path, args.output_samples_path, args.log_file]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
    main(args)
