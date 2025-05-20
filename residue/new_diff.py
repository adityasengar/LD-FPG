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
import itertools # For generating more grid options if needed
import shutil # For copying files

# -----------------------------
# Command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Diffusion Experiment Grid Runner (MLP-Only) with YAML Config, Debug Logging, and N_out Snapshots")
parser.add_argument('--instance_id', type=int, default=0,
                    help='Instance ID for splitting experiments (if grid_search)')
parser.add_argument('--exp_idx', type=int, default=None,
                    help='Global experiment index to run (if provided, only that experiment is run)')
parser.add_argument('--num_epochs_override', type=int, default=None,
                    help='Override the default number of epochs for training')
parser.add_argument('--config', type=str, default=None, # Keep optional for backward compatibility
                    help='Path to YAML config file with hyperparameters')
parser.add_argument('--debug', action='store_true',
                    help='Turn on debug mode (logs debug info)')
parser.add_argument('--log_file', type=str, default="diffusion_debug.log",
                    help='Path to log file for debug output')
parser.add_argument('--N_out', type=int, default=0,
                    help='Interval for saving epoch snapshots (checkpoints and generated embeddings). If 0, disabled.')

args = parser.parse_args()

# -----------------------------
# Setup logging
# -----------------------------
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
if args.debug:
    logging.basicConfig(filename=args.log_file,
                        filemode='w',
                        level=logging.DEBUG,
                        format=log_format)
    logging.debug("Debug mode is enabled.")
else:
    logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger("DiffusionRunner") # Use a named logger
logger.info(f"Running instance_id: {args.instance_id}")
if args.N_out > 0:
    logger.info(f"N_out snapshotting enabled every {args.N_out} epochs.")

# -----------------------------
# Default global parameters (can be overridden via YAML)
# -----------------------------
default_params = {
    'batch_size': 64,
    'num_epochs': 25000,
    'learning_rate': 1e-5,
    'num_gen': 5000,
    'save_interval': 50,
    'num_instances': 30, # Default number of instances for splitting grid search
    'hidden_dim': 1024, # Default hidden dim for MLPs
    'model_type': "mlp_v2", # Default model
    'beta_start': 5e-6,
    'beta_end': 0.03,
    'scheduler': "linear",
    'diffusion_steps': 1400,
    'h5_file_path': '/path/to/your/data/decoder2_pooled_latent_embeddings.h5',
    'dataset_key': 'pooled_latent',
    'output_dir': '/path/to/your/output/diff_out_mlp_only' # Changed output dir name
}

# -----------------------------
# Load YAML config if provided
# -----------------------------
config = {}
if args.config is not None:
    if os.path.exists(args.config):
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded config from {args.config}")
    else:
        logger.warning(f"Config file {args.config} not found. Using defaults.")
else:
    logger.info("No config file provided. Using default parameters.")

params = default_params.copy()
if 'parameters' in config:
    params.update(config['parameters'])

run_mode = config.get('run_mode', 'grid_search')
logger.info(f"Run mode: {run_mode}")

if args.num_epochs_override is not None:
    params['num_epochs'] = args.num_epochs_override
    logger.info(f"Overriding num_epochs to {args.num_epochs_override} via command line.")
    print(f"Overriding num_epochs to {args.num_epochs_override} via command line.")

# -----------------------------
# Global training parameters from params
# -----------------------------
batch_size = params['batch_size']
num_gen = params['num_gen']
save_interval = params['save_interval']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
print("Using device:", device)

# -----------------------------
# Output directory and checkpoint directory setup
# -----------------------------
output_dir = params['output_dir']
os.makedirs(output_dir, exist_ok=True)
checkpoint_dir = os.path.join(output_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
logger.info(f"Output directory: {output_dir}")
logger.info(f"Checkpoint directory: {checkpoint_dir}")

# -----------------------------
# Load pooled embeddings from HDF5 file
# -----------------------------
h5_file_path = params['h5_file_path']
dataset_key = params['dataset_key']
try:
    with h5py.File(h5_file_path, 'r') as f:
        all_pooled = f[dataset_key][:]
    logger.info(f"Loaded dataset '{dataset_key}' from {h5_file_path} with shape {all_pooled.shape}")
    print(f"Loaded dataset '{dataset_key}' with shape:", all_pooled.shape)
except Exception as e:
    logger.error(f"Failed to load data from {h5_file_path}, key {dataset_key}: {e}")
    raise

if all_pooled.ndim != 3:
    logger.error(f"Expected 3 dimensions in HDF5 data (N, num_residues, pooling_dim), but got {all_pooled.ndim}. Check dataset key '{dataset_key}'.")
    raise ValueError("Incorrect data dimensions in HDF5 file.")

num_samples_total, num_residues_total, pooling_dim = all_pooled.shape
logger.info(f"Detected total residues: {num_residues_total}, pooling dimension: {pooling_dim}")

default_residues = list(range(num_residues_total))
selected_residues_indices = config.get('selected_residues', default_residues)

if not selected_residues_indices:
    logger.warning("selected_residues list is empty. Using all residues.")
    selected_residues_indices = default_residues
elif not all(isinstance(i, int) and 0 <= i < num_residues_total for i in selected_residues_indices):
    logger.error(f"Invalid residue index found in selected_residues. Max index allowed: {num_residues_total - 1}")
    raise ValueError("Invalid residue index.")

residue_embeddings = all_pooled[:, selected_residues_indices, :]
num_sel_residues = len(selected_residues_indices)
logger.info(f"Extracted {num_sel_residues} residues (indices: {selected_residues_indices[:5]}...{selected_residues_indices[-5:]}), shape: {residue_embeddings.shape}")
print(f"Extracted {num_sel_residues} residues embeddings, shape:", residue_embeddings.shape)

data_shape_H = num_sel_residues
data_shape_W = pooling_dim

# -----------------------------
# Build hyperparameter grid or use user-defined parameters
# -----------------------------
experiments = []
start_idx = 0

if run_mode == "grid_search":
    logger.info("Building grid search experiments (MLP-Only)...")
    base_lr = params['learning_rate']
    base_num_epochs = params['num_epochs']
    base_beta_start = params['beta_start']
    base_hidden_dim = params['hidden_dim'] # For MLPs

    curated_experiments = []

    # --- Group 1 (from script 2, adapted, MLP_v2 only) ---
    group1_params = [
        {"diffusion_steps": 1200, "beta_end": 0.02},
        {"diffusion_steps": 1400, "beta_end": 0.03}, # Base combo
        {"diffusion_steps": 1500, "beta_end": 0.03},
        {"diffusion_steps": 1400, "beta_end": 0.02},
        {"diffusion_steps": 1400, "beta_end": 0.04},
        {"diffusion_steps": 1600, "beta_end": 0.03},
    ]
    for exp_g1 in group1_params:
        curated_experiments.append({
            'learning_rate': base_lr, 'num_epochs': base_num_epochs,
            'hidden_dim': base_hidden_dim,
            'model_type': "mlp_v2",
            'beta_start': base_beta_start, 'beta_end': exp_g1["beta_end"],
            'scheduler': "linear", 'diffusion_steps': exp_g1["diffusion_steps"]
        }) # 6 mlp_v2 experiments

    # --- Group 2 (from script 2, adapted, MLP_v2 only) ---
    # Script 1 originally had np.linspace(450, 550, 3, dtype=int) -> [450, 500, 550]
    # Script 2 logic had np.linspace(450, 550, 5, dtype=int) -> [450, 475, 500, 525, 550]
    # We will use the original 3 from script 1 here and add the missing 2 later.
    for steps in np.linspace(450, 550, 3, dtype=int):
        curated_experiments.append({
            'learning_rate': base_lr, 'num_epochs': base_num_epochs,
            'hidden_dim': base_hidden_dim,
            'model_type': "mlp_v2",
            'beta_start': 0.005, 'beta_end': 0.1,
            'scheduler': "linear", 'diffusion_steps': int(steps)
        }) # Total Curated: 6 + 3 = 9

    # --- Group 3 (from script 2, adapted, MLP_v2 only) ---
    for bstart in [0.004, 0.006]:
        for bend in [0.09, 0.11]:
            curated_experiments.append({
                'learning_rate': base_lr, 'num_epochs': base_num_epochs,
                'hidden_dim': base_hidden_dim,
                'model_type': "mlp_v2",
                'beta_start': bstart, 'beta_end': bend,
                'scheduler': "linear", 'diffusion_steps': 500
            }) # Total Curated: 9 + 4 = 13

    additional_experiments = []

    # Additions Set 1: Vary MLP hidden dims more (Base schedule) (+3 = 16 total curated+added)
    for hd in [512, 768, 2048]: # Added 768
        additional_experiments.append({
            'learning_rate': base_lr, 'num_epochs': base_num_epochs,
            'hidden_dim': hd, 'model_type': "mlp_v2",
            'beta_start': base_beta_start, 'beta_end': 0.03, 'scheduler': "linear", 'diffusion_steps': 1400
        })

    # Additions Set 2: REMOVED Conv2D hidden channels variations

    # Additions Set 3: Try other MLP models (Base schedule) (+2 = 18 total)
    for mt in ["mlp", "mlp_v3"]:
        additional_experiments.append({
            'learning_rate': base_lr, 'num_epochs': base_num_epochs,
            'hidden_dim': base_hidden_dim, 'model_type': mt,
            'beta_start': base_beta_start, 'beta_end': 0.03, 'scheduler': "linear", 'diffusion_steps': 1400
        })

    # Additions Set 4: Different Learning Rates (Base schedule, mlp_v2 only) (+2 = 20 total)
    for lr_add in [5e-5, 1e-6]:
        additional_experiments.append({
            'learning_rate': lr_add, 'num_epochs': base_num_epochs,
            'hidden_dim': base_hidden_dim, 'model_type': "mlp_v2",
            'beta_start': base_beta_start, 'beta_end': 0.03, 'scheduler': "linear", 'diffusion_steps': 1400
        })

    # Additions Set 5: Different Beta Schedules / Steps (mlp_v2 only) (+2 = 22 total)
    additional_experiments.append({ # Longer steps, small beta range, mlp_v2
        'learning_rate': base_lr, 'num_epochs': base_num_epochs,
        'hidden_dim': base_hidden_dim, 'model_type': "mlp_v2",
        'beta_start': 1e-6, 'beta_end': 0.01, 'scheduler': "linear", 'diffusion_steps': 2000
    })
    additional_experiments.append({ # Shorter steps, wider beta range, mlp_v2
        'learning_rate': base_lr, 'num_epochs': base_num_epochs,
        'hidden_dim': base_hidden_dim, 'model_type': "mlp_v2",
        'beta_start': 1e-4, 'beta_end': 0.1, 'scheduler': "linear", 'diffusion_steps': 800
    })

    # Additions Set 6: REMOVED (was combining schedule variations with Conv2D)

    # Additions Set 7: More Diffusion Step variations (Base schedule, mlp_v2 only) (+2 = 24 total)
    for ds_add in [1000, 1800]:
        additional_experiments.append({ # mlp_v2
            'learning_rate': base_lr, 'num_epochs': base_num_epochs,
            'hidden_dim': base_hidden_dim, 'model_type': "mlp_v2",
            'beta_start': base_beta_start, 'beta_end': 0.03, 'scheduler': "linear", 'diffusion_steps': ds_add
        })

    # Additions Set 8: More Beta End variations (Base schedule, ds=1400, mlp_v2 only) (+2 = 26 total)
    for be_add in [0.01, 0.05]:
        additional_experiments.append({ # mlp_v2
            'learning_rate': base_lr, 'num_epochs': base_num_epochs,
            'hidden_dim': base_hidden_dim, 'model_type': "mlp_v2",
            'beta_start': base_beta_start, 'beta_end': be_add, 'scheduler': "linear", 'diffusion_steps': 1400
        })

    # Additions Set 9: Combine different LR with different Hidden Dim (mlp_v2 only) (+2 = 28 total)
    additional_experiments.append({ # Low LR, Large MLP Dim
        'learning_rate': 1e-6, 'num_epochs': base_num_epochs,
        'hidden_dim': 2048, 'model_type': "mlp_v2",
        'beta_start': base_beta_start, 'beta_end': 0.03, 'scheduler': "linear", 'diffusion_steps': 1400
    })
    additional_experiments.append({ # Mid LR, Small MLP Dim
        'learning_rate': 5e-5, 'num_epochs': base_num_epochs,
        'hidden_dim': 512, 'model_type': "mlp_v2",
        'beta_start': base_beta_start, 'beta_end': 0.03, 'scheduler': "linear", 'diffusion_steps': 1400
    })

    # Add missing mlp_v2 experiments from Script 2's Group 2 logic
    # (beta_start: 0.005, beta_end: 0.1, hidden_dim: 1024 (base_hidden_dim))
    missing_steps_from_s2_g2 = [475, 525]
    for missing_step in missing_steps_from_s2_g2:
        additional_experiments.append({
            'learning_rate': base_lr, 'num_epochs': base_num_epochs,
            'hidden_dim': base_hidden_dim, 'model_type': "mlp_v2",
            'beta_start': 0.005, 'beta_end': 0.1,
            'scheduler': "linear", 'diffusion_steps': missing_step
        }) # Total curated + additional should be 28 + 2 = 30

    experiments_all = curated_experiments + additional_experiments
    experiments_all = [dict(t) for t in {tuple(sorted(d.items())) for d in experiments_all}]
    experiments_all = sorted(experiments_all, key=lambda x: str(x))

    total_exps = len(experiments_all)
    logger.info(f"Total unique MLP experiments generated: {total_exps}")
    print(f"Total unique MLP experiments generated: {total_exps}")

    if args.exp_idx is not None:
        if args.exp_idx < 1 or args.exp_idx > total_exps:
            raise ValueError(f"Invalid --exp_idx {args.exp_idx}; valid range is 1 to {total_exps}.")
        experiments = [experiments_all[args.exp_idx - 1]]
        start_idx = args.exp_idx - 1
        logger.info(f"Running specific experiment index: {args.exp_idx}")
        print(f"Running specific experiment index: {args.exp_idx}")
    else:
        num_instances = params.get('num_instances', 4)
        if args.instance_id >= num_instances:
            raise ValueError(f"Instance ID {args.instance_id} is >= number of instances {num_instances}")
        group_size = total_exps // num_instances
        remainder = total_exps % num_instances
        if args.instance_id < remainder:
            start_idx = args.instance_id * (group_size + 1)
            end_idx = start_idx + (group_size + 1)
        else:
            remainder_offset = remainder * (group_size + 1)
            start_idx = remainder_offset + (args.instance_id - remainder) * group_size
            end_idx = start_idx + group_size
        start_idx = min(start_idx, total_exps)
        end_idx = min(end_idx, total_exps)
        experiments = experiments_all[start_idx:end_idx]
        if start_idx < end_idx:
            logger.info(f"Instance {args.instance_id}/{num_instances}: Running experiments indices {start_idx} to {end_idx - 1} ({len(experiments)} experiments)")
            print(f"[Instance {args.instance_id}/{num_instances}] Running experiments indices {start_idx} to {end_idx - 1} ({len(experiments)} experiments)")
        else:
            logger.warning(f"Instance {args.instance_id}/{num_instances}: No experiments assigned (total experiments = {total_exps}).")

else:  # user_defined mode
    logger.info("Running in user_defined mode.")
    if params['model_type'] == "conv2d":
        logger.error("Conv2D model type is not supported in MLP-Only grid search mode via YAML. Please use an MLP model type.")
        raise ValueError("Conv2D model type not supported in this script version for user_defined mode if it was in YAML.")
    experiments = [{
        'learning_rate': params['learning_rate'],
        'num_epochs': params['num_epochs'],
        'hidden_dim': params.get('hidden_dim', default_params['hidden_dim']),
        'model_type': params['model_type'], # Must be an MLP type
        'beta_start': params['beta_start'],
        'beta_end': params['beta_end'],
        'scheduler': params.get('scheduler', "linear"),
        'diffusion_steps': params['diffusion_steps']
    }]
    start_idx = 0


logger.info(f"Experiments to run in this instance: {len(experiments)}")
if experiments:
    logger.debug("Hyperparameter settings for this instance:")
    for i, exp in enumerate(experiments):
        effective_exp_idx = args.exp_idx if args.exp_idx is not None else (start_idx + i + 1)
        logger.debug(f"Exp {effective_exp_idx}: {exp}")
        print(f"Exp {effective_exp_idx}: {exp}")

# -----------------------------
# Diffusion Schedule Function (linear only)
# -----------------------------
def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = max(float(beta_start), 1e-7)
    beta_end = max(float(beta_end), beta_start + 1e-7)
    return torch.linspace(beta_start, beta_end, int(timesteps), dtype=torch.float32)

# -----------------------------
# Dataset Class
# -----------------------------
class ResidueEmbeddingDataset(Dataset):
    def __init__(self, data):
        # Data should be pre-shaped correctly for MLP (N, F)
        self.data = data.astype(np.float32)
        logger.debug(f"Dataset initialized with data shape: {self.data.shape}")
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

# -----------------------------
# Checkpoint helper functions
# -----------------------------
def save_checkpoint(state, filename):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
        logger.info(f"Checkpoint saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint {filename}: {e}")

def load_checkpoint(model, optimizer, filename):
    start_epoch = 0
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint from {filename}")
        print(f"Loading checkpoint from {filename}")
        try:
            checkpoint = torch.load(filename, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                logger.warning(f"Optimizer state not found in checkpoint {filename}. Initializing optimizer from scratch.")
            start_epoch = checkpoint.get('epoch', 0)
            logger.info(f"Resumed from epoch {start_epoch}")
            print(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {filename}: {e}. Starting from scratch.")
            start_epoch = 0
    else:
        logger.info(f"No checkpoint found at {filename}. Starting from scratch.")
    return model, optimizer, start_epoch

# -----------------------------
# Define Diffusion Model Variants (MLP Only)
# -----------------------------
class DiffusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(DiffusionMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x, t):
        t_norm = (t.float() / current_diffusion_steps).unsqueeze(1)
        x_in = torch.cat([x, t_norm], dim=1)
        return self.net(x_in)

class DiffusionMLP_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(DiffusionMLP_v2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x, t):
        t_norm = (t.float() / current_diffusion_steps).unsqueeze(1)
        x_in = torch.cat([x, t_norm], dim=1)
        return self.net(x_in)

class DiffusionMLP_v3(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.2):
        super(DiffusionMLP_v3, self).__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, input_dim)
    def forward(self, x, t):
        t_norm = (t.float() / current_diffusion_steps).unsqueeze(1)
        x_in = torch.cat([x, t_norm], dim=1)
        out = self.fc1(x_in)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc3(out)

# Removed DiffusionConv2D model

# -----------------------------
# Define forward diffusion process
# -----------------------------
betas = None
alphas = None
alphas_cumprod = None
sqrt_alphas_cumprod = None
sqrt_one_minus_alphas_cumprod = None
current_diffusion_steps = None

def q_sample(x_0, t, noise=None):
    global sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
    if sqrt_alphas_cumprod is None or sqrt_one_minus_alphas_cumprod is None:
        raise ValueError("Diffusion schedule parameters not initialized for q_sample.")
    if noise is None:
        noise = torch.randn_like(x_0)
    B = x_0.shape[0]
    shape_rest = [1] * (x_0.dim() - 1)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(B, *shape_rest).to(x_0.dtype)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(B, *shape_rest).to(x_0.dtype)
    return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

# -----------------------------
# Reverse diffusion sampling
# -----------------------------
@torch.no_grad()
def p_sample_loop(model, shape, current_model_device):
    global betas, alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, current_diffusion_steps
    if not all([betas is not None, alphas is not None, sqrt_alphas_cumprod is not None,
                sqrt_one_minus_alphas_cumprod is not None, current_diffusion_steps is not None]):
        raise ValueError("Diffusion schedule parameters not initialized for p_sample_loop.")

    model.eval()
    batch_size = shape[0]
    x = torch.randn(shape, device=current_model_device, dtype=torch.float32)
    logger.info(f"Starting sampling loop for {current_diffusion_steps} steps with shape {shape}...")

    for t_idx in reversed(range(current_diffusion_steps)):
        t_batch = torch.full((batch_size,), t_idx, device=current_model_device, dtype=torch.long)
        noise_pred = model(x, t_batch)

        beta_t = betas[t_idx].to(current_model_device)
        alpha_t = alphas[t_idx].to(current_model_device)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t_idx].to(current_model_device)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)

        shape_rest = [1] * (x.dim() - 1)
        beta_t_ = beta_t.view(*shape_rest)
        sqrt_one_minus_alphas_cumprod_t_ = sqrt_one_minus_alphas_cumprod_t.view(*shape_rest)
        sqrt_recip_alpha_t_ = sqrt_recip_alpha_t.view(*shape_rest)

        model_mean = sqrt_recip_alpha_t_ * (x - (beta_t_ / sqrt_one_minus_alphas_cumprod_t_) * noise_pred)

        if t_idx > 0:
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(beta_t_) * noise # beta_t_ already broadcasted
        else:
            x = model_mean
        if t_idx % 100 == 0:
            logger.debug(f"Sampling step {t_idx} completed.")
    logger.info("Sampling loop finished.")
    return x

# -----------------------------
# N_out Snapshot Helper Function
# -----------------------------
def generate_and_save_samples_at_epoch(
    model_for_sampling, epoch_num, global_exp_idx, current_model_type_str,
    num_samples_to_gen, mlp_input_dim_for_gen,
    reshape_H, reshape_W,
    data_mean_val, data_std_val, epsilon_val,
    artifact_dir, sampling_device):

    logger.info(f"N_out: Generating {num_samples_to_gen} samples at epoch {epoch_num} for exp {global_exp_idx}...")

    shape_for_generation = (num_samples_to_gen, mlp_input_dim_for_gen) # MLP only

    generated_samples_tensor = p_sample_loop(model_for_sampling, shape_for_generation, sampling_device)
    generated_samples_np = generated_samples_tensor.cpu().numpy()
    logger.info(f"N_out: Generated samples raw shape: {generated_samples_np.shape}")

    generated_unnormalized_np = generated_samples_np * (data_std_val + epsilon_val) + data_mean_val

    # Reshape MLP output: (num_gen, num_sel_residues * pooling_dim) -> (num_gen, num_sel_residues, pooling_dim)
    generated_reshaped_np = generated_unnormalized_np.reshape(num_samples_to_gen, reshape_H, reshape_W)

    logger.info(f"N_out: Generated & Reshaped embeddings final shape: {generated_reshaped_np.shape}")

    save_path_h5 = os.path.join(artifact_dir, f"generated_embeddings.h5")
    try:
        with h5py.File(save_path_h5, 'w') as f_h5:
            f_h5.create_dataset('generated_embeddings', data=generated_reshaped_np)
        logger.info(f"N_out: Saved epoch {epoch_num} generated embeddings to: {save_path_h5}")
    except Exception as e_save:
        logger.error(f"N_out: Failed to save epoch {epoch_num} generated embeddings for exp {global_exp_idx}: {e_save}")


# -----------------------------
# Training loop for diffusion model with checkpointing and N_out snapshots
# -----------------------------
def train_diffusion_model(
    model, dataloader, optimizer, num_epochs_total,
    main_checkpoint_path, regular_save_interval,
    # N_out parameters
    n_out_interval, current_global_exp_idx, base_output_dir,
    num_gen_for_snapshot, mlp_input_dim_for_snapshot,
    data_H_for_snapshot_reshape, data_W_for_snapshot_reshape,
    mean_for_snapshot, std_for_snapshot, eps_for_snapshot,
    model_type_str_for_snapshot, training_device
    ):
    criterion = nn.MSELoss()
    model, optimizer, start_epoch_num = load_checkpoint(model, optimizer, main_checkpoint_path)
    logger.info(f"Starting training from epoch {start_epoch_num} for {num_epochs_total} total epochs.")

    for epoch_iter in range(start_epoch_num, num_epochs_total):
        model.train()
        epoch_loss_sum = 0.0
        num_batches_processed = len(dataloader)
        if num_batches_processed == 0:
            logger.warning(f"Epoch {epoch_iter+1}/{num_epochs_total}: DataLoader is empty. Skipping epoch.")
            continue

        for i, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(training_device)
            B_size = batch_data.shape[0]

            if current_diffusion_steps is None:
                raise ValueError("current_diffusion_steps not set for training loop.")
            t_timesteps = torch.randint(0, current_diffusion_steps, (B_size,), device=training_device, dtype=torch.long)

            noise_target = torch.randn_like(batch_data)
            try:
                x_t_noised = q_sample(batch_data, t_timesteps, noise_target)
            except ValueError as e_qsample:
                logger.error(f"Error in q_sample during training epoch {epoch_iter+1}: {e_qsample}. Skipping batch.")
                continue

            noise_predicted = model(x_t_noised, t_timesteps)
            loss = criterion(noise_predicted, noise_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item()

            if (i + 1) % 100 == 0:
                logger.debug(f"Epoch {epoch_iter+1}/{num_epochs_total}, Batch {i+1}/{num_batches_processed}, Batch Loss: {loss.item():.6f}")

        avg_epoch_loss = epoch_loss_sum / num_batches_processed if num_batches_processed > 0 else 0.0
        logger.info(f"Epoch {epoch_iter+1}/{num_epochs_total} completed. Average Loss: {avg_epoch_loss:.6f}")
        print(f"Epoch {epoch_iter+1}/{num_epochs_total}, Loss: {avg_epoch_loss:.6f}")

        # Regular checkpointing for resumability
        if (epoch_iter + 1) % regular_save_interval == 0:
            checkpoint_state_resume = {
                'epoch': epoch_iter + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss
            }
            save_checkpoint(checkpoint_state_resume, main_checkpoint_path)

        # N_out snapshotting (if enabled)
        if n_out_interval > 0 and (epoch_iter + 1) % n_out_interval == 0:
            logger.info(f"N_out: Processing snapshot for epoch {epoch_iter + 1}, experiment {current_global_exp_idx}.")
            snapshot_base_dir = os.path.join(base_output_dir, f"exp{current_global_exp_idx}_epoch_snapshots")
            epoch_artifact_dir = os.path.join(snapshot_base_dir, f"epoch_{epoch_iter + 1}")
            os.makedirs(epoch_artifact_dir, exist_ok=True)

            snapshot_model_state = {
                'epoch': epoch_iter + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), # Save optimizer too for potential fine-tuning
                'loss': avg_epoch_loss
            }
            snapshot_ckpt_file = os.path.join(epoch_artifact_dir, f"diffusion_checkpoint.pth")
            save_checkpoint(snapshot_model_state, snapshot_ckpt_file)

            # Original model was already in train mode here, switch to eval for sampling
            model.eval()
            generate_and_save_samples_at_epoch(
                model_for_sampling=model,
                epoch_num=epoch_iter + 1,
                global_exp_idx=current_global_exp_idx,
                current_model_type_str=model_type_str_for_snapshot,
                num_samples_to_gen=num_gen_for_snapshot,
                mlp_input_dim_for_gen=mlp_input_dim_for_snapshot,
                reshape_H=data_H_for_snapshot_reshape,
                reshape_W=data_W_for_snapshot_reshape,
                data_mean_val=mean_for_snapshot,
                data_std_val=std_for_snapshot,
                epsilon_val=eps_for_snapshot,
                artifact_dir=epoch_artifact_dir,
                sampling_device=training_device
            )
            model.train() # Crucial: set model back to training mode

    logger.info(f"Training finished after {num_epochs_total} epochs.")
    return avg_epoch_loss


# =============================
# Main Experiment Loop
# =============================
results = []
if not experiments:
    logger.warning("No experiments selected to run for this instance/configuration.")
else:
    for exp_i, exp_params_dict in enumerate(experiments):
        if args.exp_idx is not None:
            global_exp_index_val = args.exp_idx
        else:
            global_exp_index_val = start_idx + exp_i + 1

        logger.info(f"================== Starting Experiment {global_exp_index_val} ==================")
        logger.info(f"Hyperparameters: {exp_params_dict}")
        print(f"\n=============== Experiment {global_exp_index_val} ================")
        print(f"Hyperparameters: {exp_params_dict}")

        try:
            lr_val = float(exp_params_dict['learning_rate'])
            hidden_dim_val = int(exp_params_dict.get('hidden_dim', default_params['hidden_dim']))
            model_type_val = exp_params_dict['model_type']
            beta_start_val_sched = float(exp_params_dict['beta_start'])
            beta_end_val_sched = float(exp_params_dict['beta_end'])
            current_diffusion_steps = int(exp_params_dict['diffusion_steps']) # Sets global
            current_num_epochs_val = int(exp_params_dict.get('num_epochs', params['num_epochs']))

            if model_type_val not in ["mlp", "mlp_v2", "mlp_v3"]:
                logger.error(f"Unsupported model_type '{model_type_val}' in experiment {global_exp_index_val}. Skipping.")
                continue

            logger.info(f"Effective parameters: LR={lr_val}, H_Dim={hidden_dim_val}, Model={model_type_val}, "
                        f"Beta=[{beta_start_val_sched:.1E}, {beta_end_val_sched:.1E}], Steps={current_diffusion_steps}, Epochs={current_num_epochs_val}")
        except KeyError as e_key:
            logger.error(f"Missing parameter {e_key} in experiment {global_exp_index_val}. Skipping.")
            continue
        except ValueError as e_val:
            logger.error(f"Invalid parameter value in experiment {global_exp_index_val}: {e_val}. Skipping.")
            continue

        # MLP data preparation
        current_input_dim_mlp = num_sel_residues * pooling_dim
        if residue_embeddings.shape[1:] != (num_sel_residues, pooling_dim):
            logger.error(f"Residue embedding shape {residue_embeddings.shape} mismatch for MLP. Expected ({num_sel_residues}, {pooling_dim})")
            continue
        final_data_unnormalized_np = residue_embeddings.reshape(residue_embeddings.shape[0], -1)
        logger.info(f"MLP mode: Flattened data to shape {final_data_unnormalized_np.shape}, Input Dim: {current_input_dim_mlp}")

        data_mean_norm = final_data_unnormalized_np.mean()
        data_std_norm = final_data_unnormalized_np.std()
        epsilon_norm = 1e-9
        logger.info(f"Normalization parameters: mean={data_mean_norm:.6f}, std={data_std_norm:.6f}")
        print(f"Normalization parameters: mean={data_mean_norm:.6f}, std={data_std_norm:.6f}")
        normalized_data_np = (final_data_unnormalized_np - data_mean_norm) / (data_std_norm + epsilon_norm)

        try:
            dataset_obj_torch = ResidueEmbeddingDataset(normalized_data_np)
            dataloader_torch = DataLoader(dataset_obj_torch, batch_size=batch_size, shuffle=True, drop_last=True)
            logger.info(f"Created DataLoader with batch size {batch_size}. Number of batches: {len(dataloader_torch)}")
        except Exception as e_data:
            logger.error(f"Failed to create Dataset/DataLoader for experiment {global_exp_index_val}: {e_data}. Skipping.")
            continue

        try:
            # Set global diffusion schedule parameters for this experiment
            betas = linear_beta_schedule(current_diffusion_steps, beta_start_val_sched, beta_end_val_sched).to(device)
            alphas = torch.clamp(1.0 - betas, min=1e-7)
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            sqrt_alphas_cumprod = torch.sqrt(torch.clamp(alphas_cumprod, min=0.0))
            sqrt_one_minus_alphas_cumprod = torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=0.0))
            logger.debug(f"Diffusion Schedule: Betas [{betas.min():.2E} - {betas.max():.2E}], Alphas [{alphas.min():.2E} - {alphas.max():.2E}]")
        except Exception as e_sched:
            logger.error(f"Failed to create diffusion schedule for exp {global_exp_index_val}: {e_sched}. Resetting globals and skipping.")
            betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, current_diffusion_steps = None, None, None, None, None, None
            continue

        try:
            if model_type_val == "mlp":
                model_instance_torch = DiffusionMLP(input_dim=current_input_dim_mlp, hidden_dim=hidden_dim_val)
            elif model_type_val == "mlp_v2":
                model_instance_torch = DiffusionMLP_v2(input_dim=current_input_dim_mlp, hidden_dim=hidden_dim_val)
            elif model_type_val == "mlp_v3":
                model_instance_torch = DiffusionMLP_v3(input_dim=current_input_dim_mlp, hidden_dim=hidden_dim_val)
            else: # Should be caught earlier
                raise ValueError(f"Invalid model_type '{model_type_val}'")

            model_instance_torch = model_instance_torch.to(device)
            optimizer_instance_torch = optim.Adam(model_instance_torch.parameters(), lr=lr_val)
            logger.info(f"Instantiated model '{model_type_val}' with hidden_dim={hidden_dim_val}.")
        except Exception as e_model:
            logger.error(f"Failed to instantiate model or optimizer for exp {global_exp_index_val}: {e_model}. Skipping.")
            continue

        checkpoint_path_main = os.path.join(checkpoint_dir, f"diffusion_checkpoint_exp{global_exp_index_val}.pth")
        logger.info(f"Main checkpoint path: {checkpoint_path_main}")

        logger.info(f"Starting training for {current_num_epochs_val} epochs...")
        print(f"Training model for {current_num_epochs_val} epochs...")
        final_loss_val = float('inf')
        try:
            final_loss_val = train_diffusion_model(
                model=model_instance_torch,
                dataloader=dataloader_torch,
                optimizer=optimizer_instance_torch,
                num_epochs_total=current_num_epochs_val,
                main_checkpoint_path=checkpoint_path_main,
                regular_save_interval=save_interval,
                # N_out parameters
                n_out_interval=args.N_out,
                current_global_exp_idx=global_exp_index_val,
                base_output_dir=output_dir, #params['output_dir']
                num_gen_for_snapshot=num_gen,
                mlp_input_dim_for_snapshot=current_input_dim_mlp,
                data_H_for_snapshot_reshape=data_shape_H, # num_sel_residues
                data_W_for_snapshot_reshape=data_shape_W, # pooling_dim
                mean_for_snapshot=data_mean_norm,
                std_for_snapshot=data_std_norm,
                eps_for_snapshot=epsilon_norm,
                model_type_str_for_snapshot=model_type_val,
                training_device=device
            )
            logger.info(f"Training completed. Final average loss: {final_loss_val:.6f}")
            print(f"Final training loss: {final_loss_val:.6f}")
        except Exception as e_train:
            logger.error(f"Training failed for experiment {global_exp_index_val}: {e_train}", exc_info=args.debug)

        save_path_final_embeddings = None
        if final_loss_val != float('inf') and not np.isnan(final_loss_val):
            logger.info(f"Generating {num_gen} final samples...")
            print(f"Generating {num_gen} final samples...")
            try:
                shape_for_final_gen = (num_gen, current_input_dim_mlp) # MLP only

                generated_samples_final = p_sample_loop(model_instance_torch, shape_for_final_gen, device)
                generated_samples_final_np = generated_samples_final.cpu().numpy()
                logger.info(f"Generated final samples raw shape: {generated_samples_final_np.shape}")

                generated_unnormalized_final_np = generated_samples_final_np * (data_std_norm + epsilon_norm) + data_mean_norm
                generated_reshaped_final_np = generated_unnormalized_final_np.reshape(num_gen, data_shape_H, data_shape_W)
                logger.info(f"Generated & Reshaped final embeddings shape: {generated_reshaped_final_np.shape}")

                save_path_final_embeddings = os.path.join(output_dir, f"generated_embeddings_exp{global_exp_index_val}.h5")
                with h5py.File(save_path_final_embeddings, 'w') as f_h5_final:
                    f_h5_final.create_dataset('generated_embeddings', data=generated_reshaped_final_np)
                logger.info(f"Saved final generated residue embeddings to: {save_path_final_embeddings}")
                print(f"Saved final generated residue embeddings to: {save_path_final_embeddings}")
            except Exception as e_gen_final:
                logger.error(f"Final sample generation or saving failed for experiment {global_exp_index_val}: {e_gen_final}", exc_info=args.debug)
                save_path_final_embeddings = None
        else:
            logger.warning(f"Skipping final sample generation for experiment {global_exp_index_val} due to training failure or NaN loss.")

        results.append({
            'exp_idx': global_exp_index_val,
            'params': exp_params_dict,
            'final_loss': final_loss_val if final_loss_val != float('inf') and not np.isnan(final_loss_val) else 'failed/nan',
            'checkpoint_path': checkpoint_path_main,
            'save_path': save_path_final_embeddings if save_path_final_embeddings else 'failed/skipped'
        })
        logger.info(f"================== Finished Experiment {global_exp_index_val} ==================\n")

        betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, current_diffusion_steps = None, None, None, None, None, None


# =============================
# Final Summary
# =============================
logger.info("All experiments for this instance completed. Summary:")
print("\n================== Final Summary ==================")
if not results:
    print("No experiments were successfully run or summarized.")
else:
    for r_item in results:
        param_str_summary = str(r_item['params'])
        if len(param_str_summary) > 150:
            param_str_summary = param_str_summary[:147] + "..."
        summary_str_line = (f"Exp {r_item['exp_idx']}: Loss={r_item['final_loss']}, "
                            f"Checkpoint='{os.path.basename(r_item['checkpoint_path'])}', "
                            f"Output='{os.path.basename(str(r_item['save_path']))}', "
                            f"Params={param_str_summary}")
        logger.info(summary_str_line)
        print(summary_str_line)
print("===================================================")
