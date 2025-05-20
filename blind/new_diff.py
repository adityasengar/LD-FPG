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
import pathlib # Use pathlib for path handling

# -----------------------------
# Command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Diffusion Experiment Runner (Revised)")
parser.add_argument('--instance_id', type=int, default=0,
                    help='Instance ID (0, 1, 2, …) for splitting grid search experiments')
parser.add_argument('--exp_idx', type=int, default=None,
                    help='Global experiment index (1-based) to run a single specific experiment from the grid')
parser.add_argument('--num_epochs_override', type=int, default=None,
                    help='Override the number of training epochs specified in the config')
parser.add_argument('--config', type=str, default=None, required=True, # Make config required
                    help='Path to YAML config file with hyperparameters')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug level logging')
parser.add_argument('--log_file', type=str, default="diffusion_runner.log", # Default log file name
                    help='Path to log file')

args = parser.parse_args()

# -----------------------------
# Setup logging
# -----------------------------
log_level = logging.DEBUG if args.debug else logging.INFO
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(filename=args.log_file,
                    filemode='w', # Overwrite log file each time
                    level=log_level,
                    format=log_format)
# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_handler.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(console_handler)

logging.info("Diffusion Runner Script Started")
logging.info(f"Running with arguments: {args}")

# -----------------------------
# Default global parameters
# -----------------------------
default_params = {
    'batch_size': 64,
    'num_epochs': 50000,
    'learning_rate': 1e-5,
    'num_gen': 5000, # Number of samples to generate after training
    'save_interval': 1000, # Save checkpoint frequency
    'hidden_dim': 1024, # Default hidden dim for MLP models
    'model_type': "mlp_v2", # Default model type
    'beta_start': 5e-6,
    'beta_end': 0.03,
    'diffusion_steps': 1400,
    'scheduler': "linear", # Only linear supported currently
    'num_instances': 6, # For grid search partitioning
    'h5_file_path': None, # Must be provided in config
    'dataset_key': None, # Must be provided in config
    'output_dir': 'diffusion_output', # Default relative output dir
    'pooling': 'blind', # 'blind' or 'selected' (or other identifier)
    'selected_residues': [0], # Default for 'blind', overridden if pooling != 'blind'
    'conv2d_hidden_channels': 64, # Hidden channels for Conv2D model
    'decoder2_settings': { # Example shape info needed if model_type='conv2d' and pooling='blind'
        'output_height': 50,
        'output_width': 2
    }
}

# -----------------------------
# Load YAML config and merge with defaults
# -----------------------------
config_path = pathlib.Path(args.config)
if not config_path.is_file():
    logging.error(f"Configuration file not found: {args.config}")
    exit(1)

with open(config_path, 'r') as file:
    yaml_config = yaml.safe_load(file)
    logging.info(f"Loaded configuration from {config_path}")

# Merge parameters: YAML overrides defaults
params = default_params.copy()
if 'parameters' in yaml_config:
    params.update(yaml_config['parameters'])
    logging.info("Updated parameters from YAML file.")

# Determine run mode (default is grid_search from YAML, fallback to user_defined if not specified)
run_mode = yaml_config.get('run_mode', 'user_defined')
logging.info(f"Run mode determined: {run_mode}")

# --- Validate essential parameters ---
if not params.get('h5_file_path') or not params.get('dataset_key'):
    logging.error("Missing required parameters in config: 'h5_file_path' or 'dataset_key'")
    exit(1)

# -----------------------------
# Apply Command-Line Overrides
# -----------------------------
if args.num_epochs_override is not None:
    params['num_epochs'] = args.num_epochs_override
    logging.info(f"Overrode num_epochs to {params['num_epochs']} via command line.")

# -----------------------------
# Set Global Training Variables from Params
# -----------------------------
BATCH_SIZE = params['batch_size']
NUM_EPOCHS = params['num_epochs']
LEARNING_RATE = params['learning_rate'] # Base LR, might be overwritten by grid search exp
NUM_GENERATE = params['num_gen']
SAVE_INTERVAL = params['save_interval']
POOLING_MODE = params['pooling']
DEFAULT_MODEL_TYPE = params['model_type'] # Can be overridden by grid search exp
DECODER2_SETTINGS = params.get('decoder2_settings', {}) # For conv2d shape if blind
H5_FILE_PATH = pathlib.Path(params['h5_file_path'])
DATASET_KEY = params['dataset_key']
OUTPUT_DIR = pathlib.Path(params['output_dir'])
SELECTED_RESIDUES = params['selected_residues'] if POOLING_MODE != 'blind' else [0] # Use [0] for blind

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")
logging.info(f"Effective parameters: {params}") # Log all final params

# -----------------------------
# Output directory and checkpoint directory setup
# -----------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_dir = OUTPUT_DIR / 'checkpoints'
checkpoint_dir.mkdir(exist_ok=True)
logging.info(f"Output directory set to: {OUTPUT_DIR.resolve()}")
logging.info(f"Checkpoint directory set to: {checkpoint_dir.resolve()}")

# -----------------------------
# Build curated hyperparameter grid (if applicable)
# -----------------------------
# (Grid search logic remains the same as the original script)
# This section defines `experiments_all` if run_mode is 'grid_search'
if run_mode == "grid_search":
    logging.info("Building curated hyperparameter grid for grid_search mode.")
    curated_experiments = []
    fixed_lr_grid = params['learning_rate'] # Use LR from config for grid
    num_epochs_grid = params['num_epochs'] # Use epochs from config (or override)
    model_type_grid = params['model_type'] # Use model type from config for all grid exps
    hidden_dim_grid = params['hidden_dim'] # Use hidden dim from config

    # Group 1: Around beta_start = 5e-6, base (beta_end=0.03, diffusion_steps=1400)
    group1_params = [
        {"diffusion_steps": 1200, "beta_end": 0.02}, {"diffusion_steps": 1400, "beta_end": 0.03},
        {"diffusion_steps": 1500, "beta_end": 0.03}, {"diffusion_steps": 1400, "beta_end": 0.02},
        {"diffusion_steps": 1400, "beta_end": 0.04}, {"diffusion_steps": 1600, "beta_end": 0.03},
    ]
    for exp in group1_params:
        curated_experiments.append({
            'learning_rate': fixed_lr_grid, 'num_epochs': num_epochs_grid, 'hidden_dim': hidden_dim_grid,
            'model_type': model_type_grid, 'beta_start': 5e-6, 'beta_end': exp["beta_end"],
            'scheduler': "linear", 'diffusion_steps': exp["diffusion_steps"]
        })

    # Group 2: Fix beta_start=0.005, beta_end=0.1; vary diffusion_steps (5 values from 450 to 550)
    for steps in np.linspace(450, 550, 5, dtype=int):
        curated_experiments.append({
            'learning_rate': fixed_lr_grid, 'num_epochs': num_epochs_grid, 'hidden_dim': hidden_dim_grid,
            'model_type': model_type_grid, 'beta_start': 0.005, 'beta_end': 0.1,
            'scheduler': "linear", 'diffusion_steps': int(steps)
        })

    # Group 3: diffusion_steps=500; vary beta_start in [0.004, 0.006], beta_end in [0.09, 0.11]
    for bstart in [0.004, 0.006]:
        for bend in [0.09, 0.11]:
            curated_experiments.append({
                'learning_rate': fixed_lr_grid, 'num_epochs': num_epochs_grid, 'hidden_dim': hidden_dim_grid,
                'model_type': model_type_grid, 'beta_start': bstart, 'beta_end': bend,
                'scheduler': "linear", 'diffusion_steps': 500
            })

    experiments_all = curated_experiments
    num_total_experiments = len(experiments_all)
    logging.info(f"Total curated experiments generated: {num_total_experiments}")

    # --- Partition experiments based on instance_id or select specific exp_idx ---
    if args.exp_idx is not None:
        # Run only the specified experiment index
        if not 1 <= args.exp_idx <= num_total_experiments:
            logging.error(f"Invalid --exp_idx {args.exp_idx}; valid range is 1 to {num_total_experiments}.")
            exit(1)
        experiments_to_run = [experiments_all[args.exp_idx - 1]]
        current_run_start_idx = args.exp_idx - 1 # 0-based index for internal use
        logging.info(f"Running single specified experiment index: {args.exp_idx} (Internal index: {current_run_start_idx})")
    else:
        # Partition the grid search for the current instance_id
        num_instances = params.get('num_instances', 1)
        if not 0 <= args.instance_id < num_instances:
             logging.error(f"Invalid --instance_id {args.instance_id}; must be between 0 and {num_instances-1}.")
             exit(1)

        logging.info(f"Partitioning {num_total_experiments} experiments across {num_instances} instances.")
        base_size = num_total_experiments // num_instances
        remainder = num_total_experiments % num_instances
        sizes = [base_size + 1 if i < remainder else base_size for i in range(num_instances)]
        starts = [sum(sizes[:i]) for i in range(num_instances)]
        ends = [sum(sizes[:i+1]) for i in range(num_instances)]

        current_run_start_idx = starts[args.instance_id]
        current_run_end_idx = ends[args.instance_id]
        experiments_to_run = experiments_all[current_run_start_idx:current_run_end_idx]
        logging.info(f"[Instance {args.instance_id}] Running {len(experiments_to_run)} experiments (Indices {current_run_start_idx} to {current_run_end_idx - 1})")

elif run_mode == 'user_defined':
    # Run a single experiment based on the main parameters section
    logging.info("Running in user_defined mode (single experiment from main parameters).")
    experiments_to_run = [{
        'learning_rate': params['learning_rate'], 'num_epochs': params['num_epochs'],
        'hidden_dim': params['hidden_dim'], 'model_type': params['model_type'],
        'beta_start': params['beta_start'], 'beta_end': params['beta_end'],
        'scheduler': params.get('scheduler', "linear"), 'diffusion_steps': params['diffusion_steps']
    }]
    current_run_start_idx = 0 # Only one experiment, starts at index 0
else:
    logging.error(f"Unknown run_mode: {run_mode}. Choose 'grid_search' or 'user_defined'.")
    exit(1)

logging.info(f"Number of experiments to execute in this run: {len(experiments_to_run)}")

# -----------------------------
# Diffusion schedule function
# -----------------------------
def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps, device=DEVICE) # Create directly on device

# --- Global Diffusion Schedule Variables (will be updated per experiment) ---
# These need to be accessible by the model and diffusion functions
betas = None
alphas = None
alphas_cumprod = None
sqrt_alphas_cumprod = None
sqrt_one_minus_alphas_cumprod = None
current_diffusion_steps = None

# -----------------------------
# Load and Prepare Data
# -----------------------------
logging.info(f"Loading data from: {H5_FILE_PATH}")
if not H5_FILE_PATH.is_file():
    logging.error(f"Input HDF5 file not found: {H5_FILE_PATH}")
    exit(1)

with h5py.File(H5_FILE_PATH, 'r') as f:
    if DATASET_KEY not in f:
        logging.error(f"Dataset key '{DATASET_KEY}' not found in {H5_FILE_PATH}")
        exit(1)
    all_pooled_raw = f[DATASET_KEY][:] # Shape (N, ...)
logging.info(f"Loaded raw pooled embeddings with shape: {all_pooled_raw.shape}")

# --- Reshape for Blind Pooling Compatibility ---
if POOLING_MODE == 'blind':
    if all_pooled_raw.ndim == 2:
        # Expected input from chebnet_blind.py: (N_frames, final_pool_dim)
        logging.info("Blind pooling mode: Reshaping 2D data (N, D) to 3D (N, 1, D).")
        all_pooled_processed = all_pooled_raw.reshape(all_pooled_raw.shape[0], 1, all_pooled_raw.shape[1])
    elif all_pooled_raw.ndim == 3 and all_pooled_raw.shape[1] == 1:
        # Data is already in (N, 1, D) format
        logging.info("Blind pooling mode: Data already has shape (N, 1, D).")
        all_pooled_processed = all_pooled_raw
    else:
        logging.error(f"Blind pooling mode requires 2D (N, D) or 3D (N, 1, D) data, but got shape {all_pooled_raw.shape}")
        exit(1)
    # For blind pooling, we only care about the single "residue" at index 0
    selected_residue_indices = [0]
    logging.info(f"Using selected residue indices for blind pooling: {selected_residue_indices}")
    # Extracting the data (shape N, 1, D)
    data_to_process = all_pooled_processed[:, selected_residue_indices, :]

elif POOLING_MODE == 'selected':
     # Expects 3D input: (N, num_total_residues, D_per_residue)
     if all_pooled_raw.ndim != 3:
         logging.error(f"Selected pooling mode requires 3D data (N, num_residues, D), but got shape {all_pooled_raw.shape}")
         exit(1)
     all_pooled_processed = all_pooled_raw
     selected_residue_indices = params.get('selected_residues', list(range(all_pooled_processed.shape[1]))) # Default to all if not specified
     logging.info(f"Using selected residue indices for selected pooling: {selected_residue_indices}")
     # Check if indices are valid
     max_available_idx = all_pooled_processed.shape[1] - 1
     if not all(0 <= idx <= max_available_idx for idx in selected_residue_indices):
         logging.error(f"Invalid residue indices in 'selected_residues'. Max available index: {max_available_idx}, Got: {selected_residue_indices}")
         exit(1)
     # Extract the selected residues/segments (shape N, num_selected, D)
     data_to_process = all_pooled_processed[:, selected_residue_indices, :]
else:
    logging.error(f"Unknown pooling mode: {POOLING_MODE}. Choose 'blind' or 'selected'.")
    exit(1)


logging.info(f"Data shape after residue selection: {data_to_process.shape}") # Shape (N, num_selected, D)
N_samples = data_to_process.shape[0]
num_selected_residues = data_to_process.shape[1]
embedding_dim_per_residue = data_to_process.shape[2]

# -----------------------------
# Determine Model Input Shape and Prepare Final Data Tensor
# -----------------------------
# This depends on the model type used in the *current* experiment loop iteration
# We prepare the base data here, final flattening/reshaping happens before training/sampling

# For MLP models, we'll flatten the selected residue data later.
# For Conv2D models, we need to reshape to (N, 1, H, W).
# We store the base 3D selected data first.
base_selected_data = data_to_process # Shape (N, num_selected_residues, embedding_dim_per_residue)

# Calculate normalization stats on this base selected data
data_mean = base_selected_data.mean()
data_std = base_selected_data.std()
logging.info(f"Normalization calculated on selected data: mean={data_mean:.6f}, std={data_std:.6f}")
epsilon = 1e-9 # Avoid division by zero if std is very small
normalized_base_data = (base_selected_data - data_mean) / (data_std + epsilon)

# -----------------------------
# Create Dataset
# -----------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, data_tensor):
        # Expects data tensor shape (N, num_selected, D)
        self.data = data_tensor.astype(np.float32)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        # Return the data for one sample (shape num_selected, D)
        return torch.from_numpy(self.data[idx])

dataset_obj = EmbeddingDataset(normalized_base_data)
dataloader = DataLoader(dataset_obj, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True) # Set num_workers=0 if issues arise
logging.info(f"Created DataLoader with batch size {BATCH_SIZE}")

# -----------------------------
# Checkpoint helper functions
# -----------------------------
def save_checkpoint(state, filename):
    """Saves checkpoint safely."""
    try:
        torch.save(state, filename)
        logging.info(f"Checkpoint saved: {filename}")
    except Exception as e:
        logging.error(f"Error saving checkpoint {filename}: {e}")

def load_checkpoint(model, optimizer, filename):
    """Loads checkpoint if it exists."""
    start_epoch = 0
    if filename.is_file():
        logging.info(f"Loading checkpoint: '{filename}'")
        try:
            checkpoint = torch.load(filename, map_location=DEVICE)
            start_epoch = checkpoint.get('epoch', 0) # Default to 0 if epoch key missing
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer:
                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.to(DEVICE) # Ensure model is on correct device after loading
            logging.info(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch + 1}")
        except Exception as e:
            logging.error(f"Error loading checkpoint {filename}: {e}. Training from scratch.", exc_info=False)
            start_epoch = 0 # Reset epoch if loading fails
    else:
        logging.info(f"No checkpoint found at '{filename}'. Training from scratch.")
    return model, optimizer, start_epoch

# -----------------------------
# Diffusion Model Variants
# -----------------------------
# --- MLP Models ---
class DiffusionMLPBase(nn.Module):
    """Base class for MLP models to handle time embedding."""
    def __init__(self):
        super().__init__()

    def _prepare_input(self, x, t):
        # x shape: (B, FlatInputDim)
        # t shape: (B,)
        # Assumes current_diffusion_steps is globally available
        t_norm = (t.float().unsqueeze(1) / current_diffusion_steps) # Shape (B, 1)
        # Concatenate along the feature dimension
        x_in = torch.cat([x, t_norm], dim=1) # Shape (B, FlatInputDim + 1)
        return x_in

class DiffusionMLP(DiffusionMLPBase):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x, t): # x is expected to be flattened (B, D_flat)
        x_in = self._prepare_input(x, t)
        return self.net(x_in)

class DiffusionMLP_v2(DiffusionMLPBase):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x, t): # x is expected to be flattened (B, D_flat)
        x_in = self._prepare_input(x, t)
        return self.net(x_in)

class DiffusionMLP_v3(DiffusionMLPBase):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, input_dim)
    def forward(self, x, t): # x is expected to be flattened (B, D_flat)
        x_in = self._prepare_input(x, t)
        out = self.relu(self.fc1(x_in))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        return self.fc3(out)

# --- Conv2D Model ---
class DiffusionConv2D(nn.Module):
    """ Conv2D denoiser. Input x expected shape (B, 1, H, W). """
    def __init__(self, input_channels=1, hidden_channels=64):
        super().__init__()
        # Input: (B, input_channels + 1, H, W) after time concat
        self.conv1 = nn.Conv2d(input_channels + 1, hidden_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1) # Output matches input channels

    def forward(self, x, t):
        # x shape: (B, C, H, W), C=input_channels (usually 1)
        # t shape: (B,)
        B, C, H_in, W_in = x.shape
        # Assumes current_diffusion_steps is globally available
        # Create time map (B, 1, H, W)
        t_norm = (t.float().view(B, 1, 1, 1) / current_diffusion_steps)
        t_map = t_norm.expand(B, 1, H_in, W_in) # Shape (B, 1, H, W)
        # Concatenate input features and time map along channel dimension
        x_in = torch.cat([x, t_map], dim=1) # Shape (B, C+1, H, W)

        h = self.relu1(self.conv1(x_in))
        h = self.relu2(self.conv2(h))
        out = self.conv3(h) # Shape (B, C, H, W)
        return out

# -----------------------------
# Forward diffusion process (q_sample)
# -----------------------------
def q_sample(x_0, t, noise=None):
    """ Adds noise to data x_0 according to timestep t. """
    if noise is None:
        noise = torch.randn_like(x_0)

    # Get precomputed schedule values for the batch of timesteps t
    # Ensure shapes are broadcastable: (B,) -> (B, 1, ...)
    batch_size = t.shape[0]
    shape_suffix = (1,) * (x_0.dim() - 1) # e.g., (1,) for MLP, (1,1,1) for Conv2D
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(batch_size, *shape_suffix)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(batch_size, *shape_suffix)

    # Apply noise: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
    x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
    return x_t

# -----------------------------
# Training loop function
# -----------------------------
def train_diffusion_model(model, model_type, dataloader, optimizer, num_epochs_target, checkpoint_path):
    """ Trains the diffusion model. """
    criterion = nn.MSELoss()
    model.train()
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    logging.info(f"Starting training from epoch {start_epoch + 1} up to {num_epochs_target}...")

    if start_epoch >= num_epochs_target:
        logging.warning(f"Loaded checkpoint epoch ({start_epoch}) is >= target epochs ({num_epochs_target}). Skipping training.")
        # Return last loss from checkpoint if available, else 0
        # Note: Checkpoint format would need to include last loss for this.
        # For simplicity, just return 0 or indicate skipped.
        return 0.0 # Or None

    # Determine input shape transformation needed based on model type
    requires_flatten = model_type.startswith("mlp")
    requires_conv2d_shape = model_type == "conv2d"

    # --- Get H, W for Conv2D reshaping ---
    conv2d_h, conv2d_w = -1, -1
    if requires_conv2d_shape:
        # If blind, use decoder2_settings; otherwise, use selected data shape
        if POOLING_MODE == 'blind':
            conv2d_h = DECODER2_SETTINGS.get('output_height')
            conv2d_w = DECODER2_SETTINGS.get('output_width')
            if not conv2d_h or not conv2d_w or (conv2d_h * conv2d_w != embedding_dim_per_residue):
                 logging.error(f"Conv2D blind pooling shape mismatch: decoder2_settings ({conv2d_h}x{conv2d_w}) product != embedding dim ({embedding_dim_per_residue})")
                 raise ValueError("Conv2D shape configuration error for blind pooling")
        else: # 'selected' pooling
            conv2d_h = num_selected_residues
            conv2d_w = embedding_dim_per_residue
        logging.info(f"Conv2D target input shape (H, W): ({conv2d_h}, {conv2d_w})")


    last_epoch_loss = 0.0
    for epoch in range(start_epoch, num_epochs_target):
        epoch_loss = 0.0
        num_batches = len(dataloader)
        for i, batch_data_3d in enumerate(dataloader): # Data is (B, num_selected, D)
            batch_data_3d = batch_data_3d.to(DEVICE)
            B = batch_data_3d.shape[0]

            # --- Reshape batch data based on model type ---
            if requires_flatten:
                # Flatten (B, num_selected, D) -> (B, num_selected * D)
                x0 = batch_data_3d.view(B, -1)
            elif requires_conv2d_shape:
                # Reshape (B, num_selected, D) -> (B, 1, H, W)
                # Note: num_selected = H, D = W for selected pooling
                # Note: num_selected = 1, D = H*W for blind pooling (needs reshape inside)
                if POOLING_MODE == 'blind':
                    # (B, 1, D) -> (B, 1, H, W)
                     x0 = batch_data_3d.view(B, 1, conv2d_h, conv2d_w)
                else: # selected
                     # (B, num_selected, D) -> (B, 1, num_selected, D)
                     x0 = batch_data_3d.unsqueeze(1) # Add channel dim
            else:
                x0 = batch_data_3d # Should not happen with current models

            # --- Forward diffusion and prediction ---
            t = torch.randint(0, current_diffusion_steps, (B,), device=DEVICE).long()
            noise = torch.randn_like(x0) # Noise matches shape of x0 (flattened or 4D)
            x_t = q_sample(x0, t, noise=noise)
            predicted_noise = model(x_t, t) # Model takes appropriate input shape

            # --- Loss Calculation ---
            loss = criterion(predicted_noise, noise) # Compare predicted noise to original noise
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        last_epoch_loss = avg_epoch_loss # Store loss from the actual last epoch run

        logging.debug(f"Epoch {epoch+1}/{num_epochs_target}, Avg Loss: {avg_epoch_loss:.6f}")
        # Log less frequently at INFO level
        if (epoch + 1) % (SAVE_INTERVAL // 10 if SAVE_INTERVAL > 10 else 1) == 0:
             print(f"Epoch {epoch+1}/{num_epochs_target}, Avg Loss: {avg_epoch_loss:.6f}")

        # Save checkpoint periodically and at the end
        if (epoch + 1) % SAVE_INTERVAL == 0 or (epoch + 1) == num_epochs_target:
            checkpoint_state = {
                'epoch': epoch + 1, # Save completed epoch number
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss, # Save last avg loss
                # Include relevant params for reproducibility?
                'params': {k: v for k, v in current_exp_params.items() if isinstance(v, (int, float, str, bool))}
            }
            save_checkpoint(checkpoint_state, checkpoint_path)

    logging.info(f"Training finished at epoch {num_epochs_target}. Final Avg Loss: {last_epoch_loss:.6f}")
    return last_epoch_loss

# -----------------------------
# Reverse diffusion sampling (p_sample_loop)
# -----------------------------
@torch.no_grad()
def p_sample_loop(model, shape_for_gen):
    """ Generates samples starting from noise. """
    logging.info(f"Starting sampling process for shape: {shape_for_gen}")
    logging.info(f"Using {current_diffusion_steps} diffusion steps for sampling.")

    B = shape_for_gen[0] # Number of samples to generate
    # Start with random noise matching the target generation shape
    x_t = torch.randn(shape_for_gen, device=DEVICE)

    for t in reversed(range(current_diffusion_steps)):
        # Create a batch of timesteps
        t_batch = torch.full((B,), t, device=DEVICE, dtype=torch.long)

        # Predict noise using the model
        predicted_noise = model(x_t, t_batch)

        # Get schedule constants for timestep t
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t) # 1 / sqrt(alpha_t)

        # Calculate model mean using the DDPM formula:
        # model_mean = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_cumprod_t)) * predicted_noise)
        # Ensure broadcasting works for different shapes (MLP vs Conv2D)
        shape_suffix = (1,) * (x_t.dim() - 1)
        model_mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)

        # Add noise for steps t > 0
        if t > 0:
            noise = torch.randn_like(x_t)
            # Variance is beta_t
            # Posterior variance calculation can be more complex (e.g., using beta_tilde)
            # Using beta_t is a simpler choice often used.
            model_variance_sqrt = torch.sqrt(beta_t)
            x_t = model_mean + model_variance_sqrt * noise
        else:
            # Final step t=0, output is the model mean
            x_t = model_mean

        if t % (current_diffusion_steps // 10) == 0:
             logging.debug(f"Sampling step {t}/{current_diffusion_steps}")


    logging.info("Sampling finished.")
    # x_t at the end holds the generated samples x_0
    return x_t

# -----------------------------
# Main Experiment Loop
# -----------------------------
experiment_results = []
current_exp_params = {} # Global placeholder for current experiment params

for loop_idx, exp_params in enumerate(experiments_to_run):
    # Calculate the global experiment index (1-based)
    global_exp_index = current_run_start_idx + loop_idx + 1
    current_exp_params = exp_params # Store globally for checkpointing

    logging.info(f"========== Starting Experiment {global_exp_index} ==========")
    logging.info(f"Parameters: {exp_params}")
    print(f"\n========== Experiment {global_exp_index} ==========")
    print(f"Parameters: {exp_params}")

    # Extract experiment-specific parameters
    lr_exp = float(exp_params['learning_rate'])
    hidden_dim_exp = int(exp_params['hidden_dim'])
    model_type_exp = exp_params['model_type']
    beta_start_exp = float(exp_params['beta_start'])
    beta_end_exp = float(exp_params['beta_end'])
    diffusion_steps_exp = int(exp_params['diffusion_steps'])

    logging.info(f"Setting diffusion steps for this experiment: {diffusion_steps_exp}")
    current_diffusion_steps = diffusion_steps_exp # Update global variable

    # --- Compute Diffusion Schedule for this experiment ---
    betas = linear_beta_schedule(diffusion_steps_exp, beta_start_exp, beta_end_exp)
    alphas = torch.clamp(1.0 - betas, min=1e-9) # Prevent alpha=0
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # Ensure values don't become exactly 0 or 1 for numerical stability
    alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-9, max=1.0-1e-9)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # --- Determine Model Input Dimension based on Type ---
    if model_type_exp.startswith("mlp"):
        # MLP uses flattened input: num_selected_residues * embedding_dim_per_residue
        model_input_dim = num_selected_residues * embedding_dim_per_residue
        logging.info(f"MLP model type selected. Input dimension (flattened): {model_input_dim}")
    elif model_type_exp == "conv2d":
        # Conv2D uses (C, H, W). C=1 here.
        # H, W depend on pooling mode.
        if POOLING_MODE == 'blind':
             conv2d_h = DECODER2_SETTINGS.get('output_height')
             conv2d_w = DECODER2_SETTINGS.get('output_width')
             if not conv2d_h or not conv2d_w:
                  logging.error(f"Missing decoder2_settings for Conv2D blind pooling.")
                  raise ValueError("Missing shape config for Conv2D blind")
        else: # Selected pooling
             conv2d_h = num_selected_residues
             conv2d_w = embedding_dim_per_residue
        model_input_dim = (1, conv2d_h, conv2d_w) # Store shape tuple
        logging.info(f"Conv2D model type selected. Input shape (C, H, W): {model_input_dim}")
    else:
        logging.error(f"Unknown model type specified: {model_type_exp}")
        raise ValueError(f"Unknown model type: {model_type_exp}")


    # --- Instantiate Model ---
    if model_type_exp == "mlp":
        model_instance = DiffusionMLP(input_dim=model_input_dim, hidden_dim=hidden_dim_exp)
    elif model_type_exp == "mlp_v2":
        model_instance = DiffusionMLP_v2(input_dim=model_input_dim, hidden_dim=hidden_dim_exp)
    elif model_type_exp == "mlp_v3":
        model_instance = DiffusionMLP_v3(input_dim=model_input_dim, hidden_dim=hidden_dim_exp)
    elif model_type_exp == "conv2d":
        model_instance = DiffusionConv2D(input_channels=model_input_dim[0], # C=1
                                         hidden_channels=params['conv2d_hidden_channels'])
    else: # Should not happen due to earlier check
         raise ValueError(f"Unhandled model type: {model_type_exp}")

    model_instance = model_instance.to(DEVICE)
    optimizer_instance = optim.Adam(model_instance.parameters(), lr=lr_exp)

    # --- Train Model ---
    checkpoint_filename = checkpoint_dir / f"diffusion_checkpoint_exp{global_exp_index}.pth"
    logging.info(f"Starting training for Experiment {global_exp_index}...")
    final_loss = train_diffusion_model(
        model=model_instance,
        model_type=model_type_exp,
        dataloader=dataloader,
        optimizer=optimizer_instance,
        num_epochs_target=NUM_EPOCHS, # Use the potentially overridden value
        checkpoint_path=checkpoint_filename
    )
    logging.info(f"Training completed for Experiment {global_exp_index}. Final Loss: {final_loss:.6f}")
    print(f"Training completed for Experiment {global_exp_index}. Final Loss: {final_loss:.6f}")


    # --- Generate Samples ---
    model_instance.eval() # Set model to evaluation mode for sampling
    # Determine the shape needed for generation based on model type
    if model_type_exp.startswith("mlp"):
        # MLP generates flattened data (NumGen, FlatDim)
        shape_for_gen = (NUM_GENERATE, model_input_dim)
    elif model_type_exp == "conv2d":
        # Conv2D generates 4D data (NumGen, C, H, W)
        shape_for_gen = (NUM_GENERATE,) + model_input_dim # e.g., (NumGen, 1, H, W)
    else: # Should not happen
         raise ValueError(f"Cannot determine generation shape for model type {model_type_exp}")

    logging.info(f"Generating {NUM_GENERATE} samples with shape {shape_for_gen}...")
    generated_samples_norm = p_sample_loop(model_instance, shape_for_gen) # Normalized samples
    generated_samples_norm = generated_samples_norm.cpu().numpy()

    # --- Un-normalize Samples ---
    logging.info("Un-normalizing generated samples...")
    generated_samples_unnorm = generated_samples_norm * (data_std + epsilon) + data_mean

    # --- Reshape Output to Consistent Format ---
    # Aim for final output shape: (N_gen, num_selected_residues, embedding_dim_per_residue)
    logging.info("Reshaping generated samples to consistent output format...")
    if model_type_exp.startswith("mlp"):
        # Input was (N_gen, num_selected * D)
        try:
            final_generated_embeddings = generated_samples_unnorm.reshape(
                NUM_GENERATE, num_selected_residues, embedding_dim_per_residue
            )
        except ValueError as e:
             logging.error(f"Error reshaping MLP output: {e}. Expected product {num_selected_residues * embedding_dim_per_residue}, got {generated_samples_unnorm.shape[1]}")
             # Fallback: save as is? Or skip saving?
             final_generated_embeddings = generated_samples_unnorm # Save flattened
    elif model_type_exp == "conv2d":
        # Input was (N_gen, 1, H, W)
        # We need to reshape back to (N_gen, H, W) if channel=1, then maybe further if blind
        if generated_samples_unnorm.shape[1] == 1:
             generated_hw = generated_samples_unnorm.squeeze(1) # Shape (N_gen, H, W)
        else: # Should not happen if input_channels=1
             generated_hw = generated_samples_unnorm
             logging.warning("Conv2D output had more than 1 channel, using as is.")

        # If pooling was blind, HxW = D, need (N_gen, 1, D)
        # If pooling was selected, H = num_selected, W = D, need (N_gen, num_selected, D)
        if POOLING_MODE == 'blind':
             # Reshape (N_gen, H, W) -> (N_gen, 1, H*W=D)
             final_generated_embeddings = generated_hw.reshape(NUM_GENERATE, 1, -1)
        else: # Selected pooling, generated_hw is already (N_gen, H=num_selected, W=D)
             final_generated_embeddings = generated_hw
    else: # Should not happen
         final_generated_embeddings = generated_samples_unnorm # Save as is

    logging.info(f"Final generated embeddings shape for saving: {final_generated_embeddings.shape}")
    logging.info(f"Example generated sample (unnormalized, reshaped): {final_generated_embeddings[0,:,:]}")


    # --- Save Generated Samples to HDF5 ---
    save_path = OUTPUT_DIR / f"generated_embeddings_exp{global_exp_index}.h5"
    try:
        with h5py.File(save_path, 'w') as f:
            dset = f.create_dataset('generated_embeddings', data=final_generated_embeddings)
            # Add metadata attributes
            dset.attrs['experiment_index'] = global_exp_index
            dset.attrs['model_type'] = model_type_exp
            dset.attrs['diffusion_steps'] = diffusion_steps_exp
            dset.attrs['beta_start'] = beta_start_exp
            dset.attrs['beta_end'] = beta_end_exp
            dset.attrs['pooling_mode'] = POOLING_MODE
            dset.attrs['source_h5_file'] = str(H5_FILE_PATH)
            dset.attrs['source_dataset_key'] = DATASET_KEY
            # Add selected residues if not blind
            if POOLING_MODE != 'blind':
                 dset.attrs['selected_residues'] = np.array(selected_residue_indices)

        logging.info(f"Saved {NUM_GENERATE} generated embeddings to: {save_path}")
        print(f"Saved generated embeddings to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to save generated embeddings to {save_path}: {e}")

    experiment_results.append({
        'exp_idx': global_exp_index,
        'params': exp_params,
        'final_loss': final_loss,
        'checkpoint_path': str(checkpoint_filename),
        'save_path': str(save_path)
    })
    logging.info(f"========== Finished Experiment {global_exp_index} ==========")


# -----------------------------
# Final Summary
# -----------------------------
logging.info("All specified experiments completed.")
print("\n========== Run Summary ==========")
for result in experiment_results:
    print(f"Exp {result['exp_idx']}: Loss={result.get('final_loss', 'N/A'):.6f}, Output='{result.get('save_path', 'N/A')}'")
    logging.info(f"Summary - Exp {result['exp_idx']}: Loss={result.get('final_loss', 'N/A'):.6f}, Output='{result.get('save_path', 'N/A')}'")
logging.info("Diffusion Runner Script Finished")
