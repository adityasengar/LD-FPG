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
from torch.nn.utils.rnn import pad_sequence
import pathlib

# ===================================================================
# (A) Argument Parsing & Setup
# ===================================================================

parser = argparse.ArgumentParser(description="Robust Conditional Diffusion Model Runner")
parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file.')
parser.add_argument('--debug', action='store_true', help='Enable debug level logging.')
parser.add_argument('--log_file', type=str, default="conditional_diffusion_robust.log", help='Path to log file.')
args = parser.parse_args()

# Setup logging
log_level = logging.DEBUG if args.debug else logging.INFO
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler()])
logger = logging.getLogger()
file_handler = logging.FileHandler(args.log_file, mode='w')
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

logger.info("Robust Conditional Diffusion Runner Script Started")

# ===================================================================
# (B) Configuration Loading
# ===================================================================

config_path = pathlib.Path(args.config)
if not config_path.is_file():
    logger.error(f"Configuration file not found: {args.config}"); exit(1)

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    logger.info(f"Loaded configuration from {config_path}")

# Extract parameters
params = config.get('parameters', {})
data_cfg = config.get('data', {})
USE_ENSEMBLE = params.get('use_conditioner_ensemble', False)

EMBEDDINGS_H5_PATH = pathlib.Path(data_cfg['embeddings_h5_path'])
CONDITIONERS_H5_PATH = pathlib.Path(data_cfg['conditioners_h5_path'])
GROUP_TEMPLATE = data_cfg['group_key_template']
EMBEDDING_DSET_NAME = data_cfg['embedding_dataset_name']
OUTPUT_DIR = pathlib.Path(config['output_dir'])

if USE_ENSEMBLE:
    CONDITIONER_DSET_NAME = data_cfg.get('conditioner_ensemble_dataset_name', 'z_ref_ensemble')
    logger.info("Configuration set to use CONDITIONER ENSEMBLE training.")
else:
    CONDITIONER_DSET_NAME = data_cfg.get('conditioner_dataset_name', 'z_ref')
    logger.info("Configuration set to use SINGLE conditioner training.")


# Global training variables
BATCH_SIZE = params.get('batch_size', 64)
NUM_EPOCHS = params.get('num_epochs', 50000)
LEARNING_RATE = float(params.get('learning_rate', 1e-5)) # Explicitly cast to float
NUM_GENERATE = params.get('num_gen', 1000)
SAVE_INTERVAL = params.get('save_interval', 1000)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Setup output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_dir = OUTPUT_DIR / 'checkpoints'
checkpoint_dir.mkdir(exist_ok=True)
logger.info(f"Output directory: {OUTPUT_DIR.resolve()}")

# ===================================================================
# (C) Data Loading (Robust Conditional w/ Ensemble Support)
# ===================================================================

class ConditionalEmbeddingDataset(Dataset):
    def __init__(self, embeddings_path, conditioners_path, group_template, emb_dset, cond_dset, use_ensemble, logger):
        self.logger = logger
        self.use_ensemble = use_ensemble
        self.data_pairs = []
        self.system_ids = {} # Maps index to original SID

        if not embeddings_path.is_file() or not conditioners_path.is_file():
            raise FileNotFoundError(f"HDF5 file not found at {embeddings_path} or {conditioners_path}")

        with h5py.File(embeddings_path, 'r') as femb, h5py.File(conditioners_path, 'r') as fcond:
            common_groups = sorted(list(set(femb.keys()) & set(fcond.keys())))
            self.logger.info(f"Found {len(common_groups)} common system groups to load.")

            for group_name in common_groups:
                sid = int(group_name.split('_')[-1])
                if emb_dset not in femb[group_name] or cond_dset not in fcond[group_name]:
                    self.logger.warning(f"Dataset '{emb_dset}' or '{cond_dset}' missing in group {group_name}, skipping.")
                    continue

                embs = torch.from_numpy(femb[group_name][emb_dset][:].astype(np.float32))
                cond_data = torch.from_numpy(fcond[group_name][cond_dset][:].astype(np.float32))
                
                for i in range(embs.shape[0]):
                    idx = len(self.data_pairs)
                    # Store the dynamic embedding and the *entire* conditioner data (either single or ensemble)
                    self.data_pairs.append((embs[i], cond_data))
                    self.system_ids[idx] = sid
        
        self.logger.info(f"Loaded {len(self.data_pairs)} total data points (frames).")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        embedding, conditioner_data = self.data_pairs[idx]
        
        # If using ensemble and the conditioner is indeed an ensemble (3D tensor)
        if self.use_ensemble and conditioner_data.ndim == 3:
            rand_idx = torch.randint(0, conditioner_data.shape[0], (1,)).item()
            final_conditioner = conditioner_data[rand_idx]
        else:
            # Otherwise, just use the data as is (it's already a single 2D z_ref)
            final_conditioner = conditioner_data
            
        return embedding, final_conditioner

def collate_fn(batch):
    """Pads variable-length conditioners to the max length in a batch."""
    embeddings, conditioners = zip(*batch)
    padded_conditioners = pad_sequence(conditioners, batch_first=True, padding_value=0.0)
    embedding_batch = torch.stack(embeddings, 0)
    return embedding_batch, padded_conditioners

try:
    dataset = ConditionalEmbeddingDataset(
        EMBEDDINGS_H5_PATH, CONDITIONERS_H5_PATH, GROUP_TEMPLATE,
        EMBEDDING_DSET_NAME, CONDITIONER_DSET_NAME, USE_ENSEMBLE, logger
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
except Exception as e:
    logger.error(f"Failed to create dataset: {e}", exc_info=True); exit(1)

# ===================================================================
# (D) Conditional Diffusion Model & Utilities
# ===================================================================

# Diffusion schedule
T = params.get('diffusion_steps', 1000)
beta_start = params.get('beta_start', 0.0001)
beta_end = params.get('beta_end', 0.02)
betas = torch.linspace(beta_start, beta_end, T, device=DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def q_sample(x_0, t, noise=None):
    if noise is None: noise = torch.randn_like(x_0)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
    return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

# --- Model Architecture ---
class ConditionerEncoder(nn.Module):
    """Encodes a variable-length 2D conditioner to a fixed-size vector."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1), # Global Max Pooling
            nn.Flatten(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        # Input x: [Batch, SeqLen, Features] -> e.g., [64, 450, 16]
        # Conv1d expects [Batch, Features, SeqLen]
        x = x.permute(0, 2, 1)
        return self.net(x)

class DiffusionModel(nn.Module):
    """Main diffusion model that uses the conditioner encoder."""
    def __init__(self, data_dim, cond_input_dim, cond_encoded_dim, hidden_dim):
        super().__init__()
        self.encoder = ConditionerEncoder(cond_input_dim, cond_encoded_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1 + cond_encoded_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, data_dim)
        )
    
    def forward(self, x, t, cond_2d):
        t_norm = t.float().unsqueeze(1) / T
        encoded_cond = self.encoder(cond_2d)
        net_input = torch.cat([x, t_norm, encoded_cond], dim=1)
        return self.net(net_input)

# --- Sampling ---
@torch.no_grad()
def p_sample_loop(model, shape, conditioner_2d):
    x_t = torch.randn(shape, device=DEVICE)
    # The conditioner is the same for all samples in this batch
    conditioner_batch = conditioner_2d.unsqueeze(0).repeat(shape[0], 1, 1)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=DEVICE, dtype=torch.long)
        predicted_noise = model(x_t, t_batch, conditioner_batch)
        
        alpha_t = alphas[t]
        beta_t = betas[t]
        sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)
        
        model_mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alphas_cumprod[t]) * predicted_noise)
        
        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = model_mean + torch.sqrt(beta_t) * noise
        else:
            x_t = model_mean
    return x_t

# ===================================================================
# (E) Training Loop
# ===================================================================

def load_checkpoint(model, optimizer, filename, device, logger):
    """Loads a checkpoint to resume training."""
    start_epoch = 0
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint: '{filename}'")
        try:
            ckpt = torch.load(filename, map_location=device)
            start_epoch = ckpt.get('epoch', 0)
            model.load_state_dict(ckpt['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
    else:
        logger.info("No checkpoint found. Starting from scratch.")
    return model, optimizer, start_epoch

def train():
    data_dim = dataset[0][0].shape[0]
    cond_input_dim = dataset[0][1].shape[1]
    cond_encoded_dim = params.get('conditioner_encoded_dim', 128)
    
    model = DiffusionModel(
        data_dim, cond_input_dim, cond_encoded_dim, 
        params.get('hidden_dim', 1024)
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # --- Load Checkpoint to Resume ---
    checkpoint_path = checkpoint_dir / "cond_diffusion_latest.pth"
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, DEVICE, logger)

    if start_epoch >= NUM_EPOCHS:
        logger.info(f"Training has already been completed to epoch {start_epoch}. Skipping.")
        return model

    logger.info(f"Starting training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for i, (x0_batch, cond_batch) in enumerate(dataloader):
            x0_batch, cond_batch = x0_batch.to(DEVICE), cond_batch.to(DEVICE)
            optimizer.zero_grad()
            
            t = torch.randint(0, T, (x0_batch.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(x0_batch)
            x_t = q_sample(x0_batch, t, noise=noise)
            
            predicted_noise = model(x_t, t, cond_batch)
            loss = criterion(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Avg Loss: {avg_epoch_loss:.6f}")

        # Save a single, consistently named checkpoint file
        if (epoch + 1) % SAVE_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    logger.info("Training finished.")
    return model

# ===================================================================
# (F) Generation (No longer called automatically)
# ===================================================================

def generate(model):
    logger.info("Starting generation...")
    model.eval()
    
    # Get unique system IDs from the original dataset mapping
    unique_sids = sorted(list(set(dataset.system_ids.values())))
    logger.info(f"Will generate {NUM_GENERATE} samples for each of the {len(unique_sids)} unique systems found.")

    # For generation, we always use the clean, single z_ref, not the ensemble.
    gen_cond_dset_name = config.get('data', {}).get('conditioner_dataset_name', 'z_ref')

    with h5py.File(CONDITIONERS_H5_PATH, 'r') as fcond:
        for sid in unique_sids:
            group_name = GROUP_TEMPLATE.format(sid)
            logger.info(f"--- Generating for System ID: {sid} ---")
            
            try:
                if group_name not in fcond or gen_cond_dset_name not in fcond[group_name]:
                    logger.warning(f"Clean conditioner '{gen_cond_dset_name}' not found for SID {sid}. Skipping generation.")
                    continue

                cond_np = fcond[group_name][gen_cond_dset_name][:]
                conditioner_2d = torch.from_numpy(cond_np).float().to(DEVICE)
                
                gen_shape = (NUM_GENERATE, dataset[0][0].shape[0])
                
                generated_samples = p_sample_loop(model, gen_shape, conditioner_2d).cpu().numpy()
                
                save_path = OUTPUT_DIR / f"generated_embeddings_cond_on_sys_{sid}.h5"
                with h5py.File(save_path, 'w') as f_out:
                    f_out.create_dataset("generated_embeddings", data=generated_samples)
                    f_out.create_dataset("conditioner_z_ref", data=cond_np)
                    logger.info(f"Saved generated samples for system {sid} to {save_path}")

            except Exception as e:
                logger.error(f"Failed to generate for system {sid}: {e}", exc_info=True)


if __name__ == "__main__":
    trained_model = train()
    # The generate() function is no longer called automatically.
    # Generation is now handled by a dedicated inference script.
    logger.info("Script finished training.")
