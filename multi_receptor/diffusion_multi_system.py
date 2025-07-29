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

# ===================================================================
# (A) Argument Parsing & Setup
# ===================================================================

parser = argparse.ArgumentParser(description="Multi-System Diffusion Model Runner")
parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file.')
parser.add_argument('--system_id', type=int, default=None, help='Override the system_to_train from the config file.')
parser.add_argument('--exp_idx', type=int, default=None, help='For grid search: run a single specific experiment index (1-based).')
parser.add_argument('--instance_id', type=int, default=0, help='For grid search: instance ID for splitting experiments.')
parser.add_argument('--debug', action='store_true', help='Enable debug level logging.')
parser.add_argument('--log_file', type=str, default="diffusion_runner.log", help='Path to log file.')
args = parser.parse_args()

# Setup logging
log_level = logging.DEBUG if args.debug else logging.INFO
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

# Configure root logger
logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler()])
logger = logging.getLogger()

# Add file handler
file_handler = logging.FileHandler(args.log_file, mode='w')
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

logger.info("Diffusion Runner Script Started (Multi-System Version)")
logger.info(f"Running with arguments: {args}")

# ===================================================================
# (B) Configuration Loading
# ===================================================================

config_path = pathlib.Path(args.config)
if not config_path.is_file():
    logger.error(f"Configuration file not found: {args.config}"); exit(1)

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    logger.info(f"Loaded configuration from {config_path}")

# Determine which system to train on (command line overrides YAML)
SYSTEM_ID_TO_TRAIN = args.system_id if args.system_id is not None else config.get('system_to_train')
if SYSTEM_ID_TO_TRAIN is None:
    logger.error("Must specify 'system_to_train' in YAML or via --system_id argument."); exit(1)
logger.info(f"--- TARGET SYSTEM ID FOR THIS RUN: {SYSTEM_ID_TO_TRAIN} ---")

# Extract parameters
params = config.get('parameters', {})
RUN_MODE = config.get('run_mode', 'user_defined')
H5_FILE_PATH = pathlib.Path(config['h5_file_path'])
GROUP_TEMPLATE = config['dataset_group_key_template']
DATASET_NAME = config['dataset_name_in_group']
OUTPUT_DIR = pathlib.Path(config['output_dir'])

# Global training variables from params
BATCH_SIZE = params.get('batch_size', 64)
NUM_EPOCHS = params.get('num_epochs', 50000)
LEARNING_RATE = params.get('learning_rate', 1e-5)
NUM_GENERATE = params.get('num_gen', 5000)
SAVE_INTERVAL = params.get('save_interval', 1000)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Setup output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_dir = OUTPUT_DIR / f'checkpoints_sys_{SYSTEM_ID_TO_TRAIN}'
checkpoint_dir.mkdir(exist_ok=True)
logger.info(f"Output directory for this run: {OUTPUT_DIR.resolve()}")
logger.info(f"Checkpoint directory for this run: {checkpoint_dir.resolve()}")


# ===================================================================
# (C) Data Loading for Specified System
# ===================================================================

logger.info(f"Loading data from: {H5_FILE_PATH}")
if not H5_FILE_PATH.is_file():
    logger.error(f"Input HDF5 file not found: {H5_FILE_PATH}"); exit(1)

try:
    with h5py.File(H5_FILE_PATH, 'r') as f:
        group_name = GROUP_TEMPLATE.format(SYSTEM_ID_TO_TRAIN)
        if group_name not in f:
            logger.error(f"Group '{group_name}' not found in {H5_FILE_PATH}. Available groups: {list(f.keys())}"); exit(1)
        
        if DATASET_NAME not in f[group_name]:
            logger.error(f"Dataset '{DATASET_NAME}' not found in group '{group_name}'."); exit(1)
            
        # Shape is expected to be (N_frames, EmbeddingDim) e.g. (N, 100)
        data_for_system = f[group_name][DATASET_NAME][:]
        logger.info(f"Successfully loaded data for system {SYSTEM_ID_TO_TRAIN}. Shape: {data_for_system.shape}")

except Exception as e:
    logger.error(f"Failed to read HDF5 file: {e}", exc_info=True); exit(1)

# Normalize data
data_mean = data_for_system.mean()
data_std = data_for_system.std()
epsilon = 1e-9
normalized_data = (data_for_system - data_mean) / (data_std + epsilon)
logger.info(f"Data normalization stats: mean={data_mean:.6f}, std={data_std:.6f}")

# Create Dataset and DataLoader
class EmbeddingDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor.astype(np.float32)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

dataset_obj = EmbeddingDataset(normalized_data)
dataloader = DataLoader(dataset_obj, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

# ===================================================================
# (D) Diffusion Model & Utilities
# ===================================================================

# --- Diffusion schedule variables (will be updated per experiment) ---
betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = [None] * 5
current_diffusion_steps = None
current_exp_params = {} # Global placeholder for current experiment params

def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps, device=DEVICE)

# --- Checkpoint helpers ---
def save_checkpoint(state, filename):
    try:
        torch.save(state, filename)
        logger.info(f"Checkpoint saved: {filename}")
    except Exception as e:
        logger.error(f"Error saving checkpoint {filename}: {e}")

def load_checkpoint(model, optimizer, filename):
    start_epoch = 0
    if filename.is_file():
        logger.info(f"Loading checkpoint: '{filename}'")
        try:
            checkpoint = torch.load(filename, map_location=DEVICE)
            start_epoch = checkpoint.get('epoch', 0)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.to(DEVICE)
            logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}")
        except Exception as e:
            logger.error(f"Error loading checkpoint {filename}: {e}. Training from scratch.", exc_info=False)
            start_epoch = 0
    return model, optimizer, start_epoch

# --- Diffusion Model Architectures ---
class DiffusionMLPBase(nn.Module):
    def _prepare_input(self, x, t):
        t_norm = (t.float().unsqueeze(1) / current_diffusion_steps)
        return torch.cat([x, t_norm], dim=1)

class DiffusionMLP_v2(DiffusionMLPBase):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x, t):
        return self.net(self._prepare_input(x, t))

# --- Forward/Reverse Process ---
def q_sample(x_0, t, noise=None):
    if noise is None: noise = torch.randn_like(x_0)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
    return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

@torch.no_grad()
def p_sample_loop(model, shape):
    logger.info(f"Starting sampling process for shape: {shape}")
    x_t = torch.randn(shape, device=DEVICE)
    for t in reversed(range(current_diffusion_steps)):
        if t % (current_diffusion_steps // 10) == 0:
            logger.debug(f"Sampling step {t}/{current_diffusion_steps}")
        t_batch = torch.full((shape[0],), t, device=DEVICE, dtype=torch.long)
        predicted_noise = model(x_t, t_batch)
        
        alpha_t = alphas[t]
        beta_t = betas[t]
        
        sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)
        model_mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alphas_cumprod[t]) * predicted_noise)
        
        if t > 0:
            noise = torch.randn_like(x_t)
            posterior_variance = beta_t 
            x_t = model_mean + torch.sqrt(posterior_variance) * noise
        else:
            x_t = model_mean
    logger.info("Sampling finished.")
    return x_t

# ===================================================================
# (E) Training Loop
# ===================================================================

def train_diffusion_model(model, optimizer, num_epochs_target, checkpoint_path):
    criterion = nn.MSELoss()
    model.train()
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    if start_epoch >= num_epochs_target:
        logger.warning(f"Loaded checkpoint epoch ({start_epoch}) >= target epochs ({num_epochs_target}). Skipping training.")
        return 0.0

    logger.info(f"Starting training from epoch {start_epoch + 1} up to {num_epochs_target}...")
    last_avg_epoch_loss = 0.0
    for epoch in range(start_epoch, num_epochs_target):
        epoch_loss = 0.0
        for i, batch_data in enumerate(dataloader):
            x0 = batch_data.to(DEVICE)
            optimizer.zero_grad()
            
            t = torch.randint(0, current_diffusion_steps, (x0.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(x0)
            x_t = q_sample(x0, t, noise=noise)
            
            predicted_noise = model(x_t, t)
            loss = criterion(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        last_avg_epoch_loss = avg_epoch_loss
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs_target:
             logger.info(f"Epoch {epoch+1}/{num_epochs_target}, Avg Loss: {avg_epoch_loss:.6f}")

        if (epoch + 1) % SAVE_INTERVAL == 0 or (epoch + 1) == num_epochs_target:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'params': {k: v for k, v in current_exp_params.items() if isinstance(v, (int, float, str, bool))}
            }, checkpoint_path)
    
    logger.info(f"Training finished. Final Avg Loss: {last_avg_epoch_loss:.6f}")
    return last_avg_epoch_loss

# ===================================================================
# (F) Main Experiment Execution
# ===================================================================

def main():
    """Main function to run the experiment(s)."""
    global betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, current_diffusion_steps, current_exp_params

    experiments_to_run = []
    current_run_start_idx = 0

    if RUN_MODE == 'grid_search':
        logger.info("Building curated hyperparameter grid for grid_search mode.")
        grid_params = config.get('grid_search_space', {})
        curated_experiments = []
        
        # Fixed parameters for the grid
        fixed_lr = grid_params.get('learning_rate', 1e-5)
        fixed_epochs = grid_params.get('num_epochs', 50000)
        fixed_model = grid_params.get('model_type', 'mlp_v2')
        fixed_hidden = grid_params.get('hidden_dim', 1024)

        # Group 1: Explore beta_end and diffusion_steps around a small beta_start
        group1_params = [
            {"diffusion_steps": 1200, "beta_end": 0.02}, {"diffusion_steps": 1400, "beta_end": 0.03},
            {"diffusion_steps": 1500, "beta_end": 0.03}, {"diffusion_steps": 1400, "beta_end": 0.02},
            {"diffusion_steps": 1400, "beta_end": 0.04}, {"diffusion_steps": 1600, "beta_end": 0.03},
        ]
        for exp in group1_params:
            curated_experiments.append({
                'learning_rate': fixed_lr, 'num_epochs': fixed_epochs, 'hidden_dim': fixed_hidden,
                'model_type': fixed_model, 'beta_start': 5e-6, 'beta_end': exp["beta_end"],
                'scheduler': "linear", 'diffusion_steps': exp["diffusion_steps"]
            })

        # Group 2: Explore diffusion_steps around a larger beta range
        for steps in np.linspace(450, 550, 5, dtype=int):
            curated_experiments.append({
                'learning_rate': fixed_lr, 'num_epochs': fixed_epochs, 'hidden_dim': fixed_hidden,
                'model_type': fixed_model, 'beta_start': 0.005, 'beta_end': 0.1,
                'scheduler': "linear", 'diffusion_steps': int(steps)
            })
        
        experiments_all = curated_experiments
        num_total_experiments = len(experiments_all)
        logger.info(f"Total curated experiments generated: {num_total_experiments}")

        if args.exp_idx is not None:
            if not 1 <= args.exp_idx <= num_total_experiments:
                logger.error(f"Invalid --exp_idx {args.exp_idx}; valid range is 1 to {num_total_experiments}."); exit(1)
            experiments_to_run = [experiments_all[args.exp_idx - 1]]
            current_run_start_idx = args.exp_idx - 1
        else:
            num_instances = params.get('num_instances', 1)
            base_size = num_total_experiments // num_instances
            remainder = num_total_experiments % num_instances
            sizes = [base_size + 1 if i < remainder else base_size for i in range(num_instances)]
            starts = [sum(sizes[:i]) for i in range(num_instances)]
            ends = [sum(sizes[:i+1]) for i in range(num_instances)]
            
            current_run_start_idx = starts[args.instance_id]
            current_run_end_idx = ends[args.instance_id]
            experiments_to_run = experiments_all[current_run_start_idx:current_run_end_idx]
            logger.info(f"[Instance {args.instance_id}] Running {len(experiments_to_run)} experiments (Indices {current_run_start_idx} to {current_run_end_idx - 1})")

    elif RUN_MODE == 'user_defined':
        experiments_to_run = [params]
    else:
        logger.error(f"Unknown run_mode: {RUN_MODE}. Choose 'grid_search' or 'user_defined'."); exit(1)

    logger.info(f"Number of experiments to execute in this run: {len(experiments_to_run)}")
    experiment_results = []

    # --- Experiment Loop ---
    for loop_idx, exp_params in enumerate(experiments_to_run):
        global_exp_index = current_run_start_idx + loop_idx + 1
        current_exp_params = exp_params # Set for checkpointing

        logger.info(f"========== Starting Experiment {global_exp_index} for System {SYSTEM_ID_TO_TRAIN} ==========")
        logger.info(f"Parameters: {exp_params}")

        # --- Setup Diffusion Schedule ---
        current_diffusion_steps = int(exp_params['diffusion_steps'])
        betas = linear_beta_schedule(current_diffusion_steps, float(exp_params['beta_start']), float(exp_params['beta_end']))
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # --- Instantiate Model ---
        model_input_dim = data_for_system.shape[1] # e.g., 100
        model_type = exp_params.get('model_type', 'mlp_v2')
        if model_type == 'mlp_v2':
            model_instance = DiffusionMLP_v2(input_dim=model_input_dim, hidden_dim=int(exp_params['hidden_dim']))
        else:
            logger.error(f"Unsupported model_type '{model_type}' in this version."); continue
        
        model_instance.to(DEVICE)
        optimizer_instance = optim.Adam(model_instance.parameters(), lr=float(exp_params['learning_rate']))
        
        # --- Train Model ---
        checkpoint_filename = checkpoint_dir / f"diffusion_checkpoint_exp{global_exp_index}.pth"
        final_loss = train_diffusion_model(
            model=model_instance,
            optimizer=optimizer_instance,
            num_epochs_target=int(exp_params.get('num_epochs', NUM_EPOCHS)),
            checkpoint_path=checkpoint_filename
        )

        # --- Generate and Save Samples ---
        model_instance.eval()
        generation_shape = (NUM_GENERATE, model_input_dim)
        logger.info(f"Generating {NUM_GENERATE} samples with shape {generation_shape}...")
        generated_samples_norm = p_sample_loop(model_instance, generation_shape).cpu().numpy()
        
        logger.info("Un-normalizing generated samples...")
        generated_samples_unnorm = generated_samples_norm * (data_std + epsilon) + data_mean
        logger.info(f"Final generated embeddings shape: {generated_samples_unnorm.shape}")
        
        save_path = OUTPUT_DIR / f"generated_embeddings_sys{SYSTEM_ID_TO_TRAIN}_exp{global_exp_index}.h5"
        try:
            with h5py.File(save_path, 'w') as f:
                group = f.create_group(GROUP_TEMPLATE.format(SYSTEM_ID_TO_TRAIN))
                dset = group.create_dataset(DATASET_NAME, data=generated_samples_unnorm)
                dset.attrs['source_system_id'] = SYSTEM_ID_TO_TRAIN
                dset.attrs['source_h5_file'] = str(H5_FILE_PATH)
                for key, val in exp_params.items():
                    if isinstance(val, (int, float, str, bool)):
                        dset.attrs[key] = val
            logger.info(f"Saved {NUM_GENERATE} generated embeddings to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save generated embeddings to {save_path}: {e}")

        experiment_results.append({
            'exp_idx': global_exp_index,
            'params': exp_params,
            'final_loss': final_loss,
            'save_path': str(save_path)
        })
        logger.info(f"========== Finished Experiment {global_exp_index} ==========")

    # --- Final Summary ---
    logger.info("All specified experiments for this run are complete.")
    print("\n========== Run Summary ==========")
    for result in experiment_results:
        summary_line = f"System {SYSTEM_ID_TO_TRAIN}, Exp {result['exp_idx']}: Loss={result.get('final_loss', 'N/A'):.6f}, Output='{result.get('save_path', 'N/A')}'"
        print(summary_line)
        logger.info(f"Summary - {summary_line}")

if __name__ == "__main__":
    main()
