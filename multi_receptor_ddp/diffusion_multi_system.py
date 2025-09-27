#!/usr/bin/env python3
import argparse
import os
import yaml
import logging
import torch
import numpy as np
import h5py
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pathlib

# DDP and AMP Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

# ===================================================================
# (A) Argument Parsing & DDP Setup
# ===================================================================

parser = argparse.ArgumentParser(description="Multi-System Diffusion Model Runner (DDP/AMP)")
parser.add_argument('--config', type=str, required=True, help='Path to YAML file.')
parser.add_argument('--system_id', type=int, required=True, help='System ID to train on.')
parser.add_argument('--exp_idx', type=int, default=1, help='Experiment index for this run.')
parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
args = parser.parse_args()

# --- DDP Setup ---
dist.init_process_group(backend="nccl")
RANK = dist.get_rank()
WORLD_SIZE = dist.get_world_size()
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(LOCAL_RANK)
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")

# ===================================================================
# (B) Logging Setup
# ===================================================================
log_level = logging.DEBUG if args.debug else logging.INFO
log_format = f'[%(asctime)s RANK {RANK}] %(levelname)s - %(message)s'
logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler()])
logger = logging.getLogger()
if RANK != 0:
    logger.setLevel(logging.WARNING) # Only main rank logs INFO

logger.info("Diffusion Runner Script Started (DDP/AMP Multi-System Version)")
logger.info(f"Running with arguments: {args}")
logger.info(f"World Size: {WORLD_SIZE}, Device: {DEVICE}")

# ===================================================================
# (C) Configuration Loading
# ===================================================================

config_path = pathlib.Path(args.config)
if not config_path.is_file():
    logger.error(f"Config file not found: {args.config}"); exit(1)
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
logger.info(f"Loaded configuration from {config_path}")

params = config.get('parameters', {})
H5_FILE_PATH = pathlib.Path(config['h5_file_path'])
GROUP_TEMPLATE = config['dataset_group_key_template']
DATASET_NAME = config['dataset_name_in_group']
OUTPUT_DIR = pathlib.Path(config['output_dir'])

BATCH_SIZE = params.get('batch_size', 64)
NUM_EPOCHS = params.get('num_epochs', 50000)
LEARNING_RATE = params.get('learning_rate', 1e-5)
NUM_GENERATE = params.get('num_gen', 5000)
SAVE_INTERVAL = params.get('save_interval', 1000)

if RANK == 0:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUTPUT_DIR / f'checkpoints_sys_{args.system_id}'
    checkpoint_dir.mkdir(exist_ok=True)
    logger.info(f"Output dir: {OUTPUT_DIR.resolve()}")
    logger.info(f"Checkpoint dir: {checkpoint_dir.resolve()}")

# ===================================================================
# (D) Data Loading
# ===================================================================

if not H5_FILE_PATH.is_file():
    logger.error(f"HDF5 file not found: {H5_FILE_PATH}"); exit(1)

try:
    with h5py.File(H5_FILE_PATH, 'r') as f:
        group_name = GROUP_TEMPLATE.format(args.system_id)
        data_for_system = f[group_name][DATASET_NAME][:]
    logger.info(f"Data for system {args.system_id} loaded. Shape: {data_for_system.shape}")
except Exception as e:
    logger.error(f"Failed to read HDF5 file: {e}", exc_info=True); exit(1)

data_mean = data_for_system.mean(); data_std = data_for_system.std(); epsilon = 1e-9
normalized_data = (data_for_system - data_mean) / (data_std + epsilon)

class EmbeddingDataset(Dataset):
    def __init__(self, data): self.data = data.astype(np.float32)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return torch.from_numpy(self.data[idx])

dataset = EmbeddingDataset(normalized_data)
sampler = DistributedSampler(dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)

# ===================================================================
# (E) Diffusion Model & Utilities
# ===================================================================

betas, alphas, alphas_cumprod = [None] * 3 # Will be set in main
current_diffusion_steps = None

def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps, device=DEVICE)

def save_checkpoint(state, filename):
    torch.save(state, filename)
    logger.info(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, scaler, filename):
    start_epoch = 0
    if filename.is_file():
        logger.info(f"Loading checkpoint: '{filename}'")
        ckpt = torch.load(filename, map_location=DEVICE)
        model.module.load_state_dict(ckpt['model_state_dict'])
        if optimizer: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scaler: scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        logger.info(f"Resuming from epoch {start_epoch + 1}")
    return start_epoch

class DiffusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, input_dim))
    def forward(self, x, t):
        t_norm = t.float().unsqueeze(1) / current_diffusion_steps
        return self.net(torch.cat([x, t_norm], dim=1))

def q_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None: noise = torch.randn_like(x_0)
    return (sqrt_alphas_cumprod[t].view(-1, 1) * x_0 +
            sqrt_one_minus_alphas_cumprod[t].view(-1, 1) * noise)

@torch.no_grad()
def p_sample_loop(model, shape):
    logger.info(f"Starting sampling for shape: {shape}")
    x_t = torch.randn(shape, device=DEVICE)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    for t in reversed(range(current_diffusion_steps)):
        with autocast():
            predicted_noise = model(x_t, torch.full((shape[0],), t, device=DEVICE))
        
        model_mean = sqrt_recip_alphas[t] * (x_t - betas[t] * predicted_noise / sqrt_one_minus_alphas_cumprod[t])
        if t > 0:
            x_t = model_mean + torch.sqrt(betas[t]) * torch.randn_like(x_t)
        else:
            x_t = model_mean
    logger.info("Sampling finished.")
    return x_t

# ===================================================================
# (F) Training Loop
# ===================================================================

def train(model, optimizer, scaler, checkpoint_path):
    model.train()
    start_epoch = load_checkpoint(model, optimizer, scaler, checkpoint_path) if checkpoint_path.is_file() else 0
    if start_epoch >= NUM_EPOCHS:
        logger.warning(f"Loaded epoch ({start_epoch}) >= target ({NUM_EPOCHS}). Skipping training.")
        return

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    criterion = nn.MSELoss()

    logger.info(f"Starting training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        sampler.set_epoch(epoch)
        for batch_data in dataloader:
            x0 = batch_data.to(DEVICE); optimizer.zero_grad()
            t = torch.randint(0, current_diffusion_steps, (x0.shape[0],), device=DEVICE)
            noise = torch.randn_like(x0)
            x_t = q_sample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise)
            
            with autocast():
                predicted_noise = model(x_t, t)
                loss = criterion(predicted_noise, noise)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if RANK == 0 and ((epoch + 1) % 100 == 0 or (epoch + 1) == NUM_EPOCHS):
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.6f}")

        if RANK == 0 and ((epoch + 1) % SAVE_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS):
            save_checkpoint({
                'epoch': epoch + 1, 'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(),
                'loss': loss.item(), 'params': {k:v for k,v in params.items() if isinstance(v, (int,float,str,bool))}
            }, checkpoint_path)
    logger.info("Training finished.")

# ===================================================================
# (G) Main Execution
# ===================================================================

def main():
    global betas, alphas, alphas_cumprod, current_diffusion_steps
    
    # --- Setup Diffusion Schedule ---
    current_diffusion_steps = int(params['diffusion_steps'])
    betas = linear_beta_schedule(current_diffusion_steps, float(params['beta_start']), float(params['beta_end']))
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # --- Instantiate Model ---
    model = DiffusionMLP(input_dim=data_for_system.shape[1], hidden_dim=int(params['hidden_dim'])).to(DEVICE)
    model = DDP(model, device_ids=[LOCAL_RANK])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    
    # --- Train Model ---
    checkpoint_path = OUTPUT_DIR / f'checkpoints_sys_{args.system_id}' / f"diffusion_checkpoint_exp{args.exp_idx}.pth"
    train(model, optimizer, scaler, checkpoint_path)

    # --- Generate and Save Samples (Rank 0 only) ---
    if RANK == 0:
        model.eval()
        generation_shape = (NUM_GENERATE, data_for_system.shape[1])
        generated_samples_norm = p_sample_loop(model.module, generation_shape).cpu().numpy()
        
        generated_samples_unnorm = generated_samples_norm * (data_std + epsilon) + data_mean
        logger.info(f"Un-normalized samples shape: {generated_samples_unnorm.shape}")
        
        save_path = OUTPUT_DIR / f"generated_embeddings_sys{args.system_id}_exp{args.exp_idx}.h5"
        with h5py.File(save_path, 'w') as f:
            group = f.create_group(GROUP_TEMPLATE.format(args.system_id))
            dset = group.create_dataset(DATASET_NAME, data=generated_samples_unnorm)
            dset.attrs['source_system_id'] = args.system_id
        logger.info(f"Saved {NUM_GENERATE} generated embeddings to: {save_path}")

    dist.destroy_process_group()
    logger.info("Script finished.")

if __name__ == "__main__":
    main()
