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

# -----------------------------
# Command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Diffusion Training on Pooled Embeddings (Backbone or Sidechain)")
parser.add_argument('--instance_id', type=int, default=0,
                    help='Instance ID for splitting experiments (if grid_search)')
parser.add_argument('--exp_idx', type=int, default=None,
                    help='Global experiment index to run (if provided, only that experiment is run)')
parser.add_argument('--num_epochs_override', type=int, default=None,
                    help='Override the default number of epochs for training')
parser.add_argument('--config', type=str, required=True,
                    help='Path to YAML config file with hyperparameters')
parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
parser.add_argument('--log_file', type=str, default="diffusion_debug.log",
                    help='Path to log file for debug output')
args = parser.parse_args()

# -----------------------------
# Setup logging
# -----------------------------
if args.debug:
    logging.basicConfig(filename=args.log_file, filemode='w', level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug("Debug mode is enabled.")
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger("Diffusion")
logger.info(f"Running instance_id: {args.instance_id}")

# -----------------------------
# Load YAML config & merge defaults
# -----------------------------
default_params = {
    'batch_size': 64,
    'num_epochs': 25000,
    'learning_rate': 1e-5,
    'num_gen': 5000,
    'save_interval': 50,
    'model_type': "mlp_v2",  # Options: mlp, mlp_v2, mlp_v3, conv2d
    'beta_start': 5e-6,
    'beta_end': 0.03,
    'diffusion_steps': 1400,
    'num_instances': 3,
    # File and dataset parameters:
    'h5_file_path': 'latent_reps/backbone_pooled.h5',  # or sidechain_pooled.h5
    'dataset_key': 'backbone_pooled',                 # e.g., "backbone_pooled" or "sidechain_pooled"
    'output_dir': 'latent_reps/diff_out',
    # Pooling dimensions are taken from the YAML:
    'pooling_dim': [30, 1]  # For backbone_pooled; for sidechain_pooled use e.g. [10, 3]
}
config = {}
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
logger.info(f"Loaded config from {args.config}")
params = default_params.copy()
if 'parameters' in config:
    params.update(config['parameters'])

# If exp_idx override:
if args.num_epochs_override is not None:
    params['num_epochs'] = args.num_epochs_override
    logger.info(f"Overriding num_epochs to {args.num_epochs_override}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
print("Using device:", device)

checkpoint_dir = os.path.join(params['output_dir'], "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# -----------------------------
# Grid Search Setup
# -----------------------------
fixed_lr = params['learning_rate']
num_epochs = params['num_epochs']
model_type = params['model_type']

curated_experiments = []

# Group 1
group1 = [
    {"diffusion_steps": 1200, "beta_end": 0.02},
    {"diffusion_steps": 1400, "beta_end": 0.03},  # base combination
    {"diffusion_steps": 1500, "beta_end": 0.03},
    {"diffusion_steps": 1400, "beta_end": 0.02},
    {"diffusion_steps": 1400, "beta_end": 0.04},
    {"diffusion_steps": 1600, "beta_end": 0.03},
]
for exp in group1:
    curated_experiments.append({
        'learning_rate': fixed_lr,
        'num_epochs': num_epochs,
        'hidden_dim': 1024,
        'model_type': model_type,
        'beta_start': 5e-6,
        'beta_end': exp["beta_end"],
        'scheduler': "linear",
        'diffusion_steps': exp["diffusion_steps"]
    })

# Group 2
for steps in np.linspace(450, 550, 5, dtype=int):
    curated_experiments.append({
        'learning_rate': fixed_lr,
        'num_epochs': num_epochs,
        'hidden_dim': 1024,
        'model_type': model_type,
        'beta_start': 0.005,
        'beta_end': 0.1,
        'scheduler': "linear",
        'diffusion_steps': int(steps)
    })

# Group 3
for bstart in [0.004, 0.006]:
    for bend in [0.09, 0.11]:
        curated_experiments.append({
            'learning_rate': fixed_lr,
            'num_epochs': num_epochs,
            'hidden_dim': 1024,
            'model_type': model_type,
            'beta_start': bstart,
            'beta_end': bend,
            'scheduler': "linear",
            'diffusion_steps': 500
        })

experiments_all = curated_experiments
total_exps = len(experiments_all)
logger.info(f"Total curated experiments: {total_exps}")
print(f"Total curated experiments: {total_exps}")

if args.exp_idx is not None:
    if args.exp_idx < 1 or args.exp_idx > total_exps:
        raise ValueError(f"Invalid --exp_idx {args.exp_idx}; valid range is 1 to {total_exps}.")
    experiments = [experiments_all[args.exp_idx - 1]]
    start_idx = args.exp_idx - 1
else:
    num_instances = params.get('num_instances', 3)
    group_size = total_exps // num_instances
    remainder = total_exps % num_instances
    if args.instance_id < remainder:
        start_idx = args.instance_id * (group_size + 1)
        end_idx = start_idx + (group_size + 1)
    else:
        start_idx = remainder * (group_size + 1) + (args.instance_id - remainder) * group_size
        end_idx = start_idx + group_size
    experiments = experiments_all[start_idx:end_idx]
    logger.info(f"Instance {args.instance_id}: Running experiments indices {start_idx} to {end_idx - 1}")
    print(f"[Instance {args.instance_id}] Running experiments indices {start_idx} to {end_idx - 1}")

# -----------------------------
# Data Loading
# -----------------------------
with h5py.File(params['h5_file_path'], 'r') as f:
    all_data = f[params['dataset_key']][:]
logger.info(f"Loaded dataset '{params['dataset_key']}' from {params['h5_file_path']} with shape {all_data.shape}")
print(f"Loaded dataset '{params['dataset_key']}' with shape:", all_data.shape)

# For MLP, if data is (N, H*W) then reshape to (N, H, W); for conv2d, if data is (N, H*W) we reshape
pool_H, pool_W = params['pooling_dim'][0], params['pooling_dim'][1]
N = all_data.shape[0]

if model_type == "conv2d":
    if all_data.ndim == 2 and all_data.shape[1] == pool_H * pool_W:
        data_2d = all_data.reshape(N, pool_H, pool_W)
    elif all_data.ndim == 3 and all_data.shape[1] == pool_H and all_data.shape[2] == pool_W:
        data_2d = all_data
    else:
        raise ValueError(f"Data shape {all_data.shape} does not match expected for conv2d with pooling_dim {pool_H}x{pool_W}.")
    final_data = data_2d  # (N, H, W)
    input_dim = pool_H * pool_W
else:
    # For MLP, we flatten any (N, H, W) to (N, H*W)
    if all_data.ndim == 3 and all_data.shape[1] == pool_H and all_data.shape[2] == pool_W:
        final_data = all_data.reshape(N, pool_H * pool_W)
    else:
        final_data = all_data
    input_dim = final_data.shape[1]

# -----------------------------
# Normalization
# -----------------------------
data_mean = final_data.mean()
data_std  = final_data.std()
epsilon = 1e-9
norm_data = (final_data - data_mean) / (data_std + epsilon)
logger.info(f"Normalization: mean={data_mean:.6f}, std={data_std:.6f}")
print(f"Normalization: mean={data_mean:.6f}, std={data_std:.6f}")

# -----------------------------
# Dataset and DataLoader
# -----------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, data_array):
        self.data = data_array.astype(np.float32)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])
        
dataset_obj = EmbeddingDataset(norm_data)
dataloader = DataLoader(dataset_obj, batch_size=params['batch_size'], shuffle=True)

# -----------------------------
# Diffusion Model Definitions
# -----------------------------
class DiffusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):
        super(DiffusionMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x, t):
        t_norm = t.float().unsqueeze(1) / current_diffusion_steps
        x_in = torch.cat([x, t_norm], dim=1)
        return self.net(x_in)

class DiffusionMLP_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):
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
        t_norm = t.float().unsqueeze(1) / current_diffusion_steps
        x_in = torch.cat([x, t_norm], dim=1)
        return self.net(x_in)

class DiffusionMLP_v3(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):
        super(DiffusionMLP_v3, self).__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, input_dim)
    def forward(self, x, t):
        t_norm = t.float().unsqueeze(1) / current_diffusion_steps
        x_in = torch.cat([x, t_norm], dim=1)
        out = self.fc1(x_in)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc3(out)

class DiffusionConv2D(nn.Module):
    def __init__(self, hidden_channels=64):
        super(DiffusionConv2D, self).__init__()
        self.conv1 = nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
    def forward(self, x, t):
        B, C, H, W = x.shape
        t_norm = (t.float() / current_diffusion_steps).view(B,1,1,1)
        t_map = t_norm.expand(B,1,H,W)
        x_in = torch.cat([x, t_map], dim=1)
        h = self.conv1(x_in)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.relu2(h)
        return self.conv3(h)

# -----------------------------
# Forward Diffusion Process
# -----------------------------
def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    B = x_0.shape[0]
    shape_rest = [1]*(x_0.dim()-1)
    alpha_t = sqrt_alphas_cumprod[t].view(B, *shape_rest)
    one_minus_t = sqrt_one_minus_alphas_cumprod[t].view(B, *shape_rest)
    return alpha_t * x_0 + one_minus_t * noise

# -----------------------------
# Training Loop
# -----------------------------
def train_diffusion_model(model, dataloader, optimizer, num_epochs, checkpoint_path):
    criterion = nn.MSELoss()
    model, optimizer, start_epoch = load_ckpt(model, optimizer, checkpoint_path)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            B = batch.shape[0]
            t = torch.randint(0, current_diffusion_steps, (B,), device=device).long()
            noise = torch.randn_like(batch)
            x_t = q_sample(batch, t, noise)
            noise_pred = model(x_t, t)
            loss = criterion(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % params['save_interval'] == 0:
            avg_loss = epoch_loss / len(dataloader)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
            ckpt_state = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(ckpt_state, checkpoint_path)
    return loss.item()

def load_ckpt(model, optimizer, filename):
    start_epoch = 0
    if os.path.isfile(filename):
        logging.info(f"Loading checkpoint from {filename}")
        ckpt = torch.load(filename, map_location=device)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        logging.info(f"Resumed from epoch {start_epoch}")
    return model, optimizer, start_epoch

# -----------------------------
# Reverse Diffusion Sampling
# -----------------------------
@torch.no_grad()
def p_sample_loop(model, shape):
    x = torch.randn(shape, device=device)
    for t in reversed(range(current_diffusion_steps)):
        t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        noise_pred = model(x, t_batch)
        beta_t = betas[t]
        sqrt_one_minus_t = sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha = torch.sqrt(1.0 / alphas[t])
        shape_rest = [1]*(x.dim()-1)
        beta_t_ = beta_t.view(*shape_rest)
        sqrt_1m_ = sqrt_one_minus_t.view(*shape_rest)
        model_mean = sqrt_recip_alpha*(x - (beta_t_/sqrt_1m_)*noise_pred)
        if t > 0:
            x = model_mean + torch.sqrt(beta_t)*torch.randn_like(x)
        else:
            x = model_mean
    return x

# -----------------------------
# Main Experiment Loop
# -----------------------------
results = []
for exp_idx, exp_params in enumerate(experiments):
    global_idx = args.exp_idx if args.exp_idx is not None else (start_idx + exp_idx + 1)
    logging.info(f"Experiment {global_idx} with parameters: {exp_params}")
    print(f"\nExperiment {global_idx} with parameters: {exp_params}")

    local_lr         = float(exp_params['learning_rate'])
    local_num_epochs = int(exp_params['num_epochs'])
    local_beta_start = float(exp_params['beta_start'])
    local_beta_end   = float(exp_params['beta_end'])
    local_diff_steps = int(exp_params['diffusion_steps'])
    
    current_diffusion_steps = local_diff_steps
    betas = torch.linspace(local_beta_start, local_beta_end, local_diff_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    if exp_params['model_type'] == "mlp":
        net = DiffusionMLP(input_dim, hidden_dim=1024).to(device)
    elif exp_params['model_type'] == "mlp_v2":
        net = DiffusionMLP_v2(input_dim, hidden_dim=1024).to(device)
    elif exp_params['model_type'] == "mlp_v3":
        net = DiffusionMLP_v3(input_dim, hidden_dim=1024).to(device)
    elif exp_params['model_type'] == "conv2d":
        net = DiffusionConv2D(hidden_channels=64).to(device)
    else:
        raise ValueError(f"Unknown model type: {exp_params['model_type']}")
    
    optimizer_instance = optim.Adam(net.parameters(), lr=local_lr)
    ckpt_path = os.path.join(checkpoint_dir, f"diffusion_exp{global_idx}.pth")
    final_loss = train_diffusion_model(net, dataloader, optimizer_instance, local_num_epochs, ckpt_path)
    logging.info(f"Final training loss: {final_loss:.6f}")
    print(f"Final training loss: {final_loss:.6f}")
    
    num_gen = params['num_gen']
    if exp_params['model_type'] == "conv2d":
        shape_for_gen = (num_gen, 1, pool_H, pool_W)
    else:
        shape_for_gen = (num_gen, input_dim)
    
    generated = p_sample_loop(net, shape_for_gen).cpu().numpy()
    generated_un = generated * (data_std + epsilon) + data_mean
    if exp_params['model_type'] == "conv2d":
        output_data = generated_un.reshape(num_gen, pool_H, pool_W)
    else:
        output_data = generated_un.reshape(num_gen, pool_H, pool_W)
    
    out_fname = os.path.join(params['output_dir'], f"generated_diff_exp{global_idx}.h5")
    with h5py.File(out_fname, 'w') as f:
        f.create_dataset('generated_diffusion', data=output_data)
    logging.info(f"Saved generated samples to {out_fname}")
    print(f"Saved generated samples to {out_fname}")
    
    results.append({
        'exp_idx': global_idx,
        'params': exp_params,
        'final_loss': final_loss,
        'checkpoint_path': ckpt_path,
        'output_file': out_fname
    })

logging.info("All experiments completed. Summary:")
print("\nAll experiments completed. Summary:")
for r in results:
    logging.info(r)
    print(r)
