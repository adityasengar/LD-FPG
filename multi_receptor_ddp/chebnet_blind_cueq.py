################################################################################
# %% Imports
################################################################################
import os
import sys
import json
import yaml
import argparse
import logging
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import List, Tuple

# DDP and AMP Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_cluster import knn_graph
from sklearn.model_selection import train_test_split

# Equivariant imports
from .cueq_encoder import CUEQ_Encoder

################################################################################
# (A) Argument Parsing
################################################################################
parser = argparse.ArgumentParser(
    description="Protein Encoding: CUEquivariance Encoder (Multi-System DDP/AMP Version)"
)
parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
parser.add_argument("--debug", action='store_true', help="Enable debug logging.")
args = parser.parse_args()

################################################################################
# (B) Logging & DDP Setup
################################################################################
logger = logging.getLogger("CUEQ_Encoder_DDP_AMP")

def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s - (Rank %(rank)s) - %(message)s")
        if rank == 0:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG if args.debug else logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    adapter = logging.LoggerAdapter(logger, {'rank': rank})
    adapter.info(f"DDP Initialized. Rank: {rank}/{world_size}, Device: {device}")
    return rank, world_size, device, adapter

def cleanup_ddp():
    dist.destroy_process_group()

################################################################################
# (C) Utility Functions
################################################################################

def load_heavy_atom_coords_from_json(json_file: str, logger: logging.Logger) -> Tuple[List[torch.Tensor], int]:
    logger.info(f"Loading coordinates from JSON: {json_file}")
    try:
        with open(json_file, "r") as f: data = json.load(f)
        keys_str = sorted(data.keys(), key=int)
        n_frames = len(data[keys_str[0]]["heavy_atom_coords_per_frame"])
        if n_frames == 0: logger.warning("JSON contains 0 frames."); return [], 0
        coords_frames, n_atoms_check = [], -1
        for i in range(n_frames):
            frame_coords = np.concatenate([np.array(data[k]["heavy_atom_coords_per_frame"][i], dtype=np.float32) for k in keys_str])
            if i == 0: n_atoms_check = frame_coords.shape[0]
            assert frame_coords.shape[0] == n_atoms_check, f"Inconsistent atom count on frame {i}"
            coords_frames.append(torch.from_numpy(frame_coords))
        logger.info(f"Loaded {n_frames} frames with {n_atoms_check} atoms each.")
        return coords_frames, n_atoms_check
    except Exception as e:
        logger.error(f"Error reading or processing JSON {json_file}: {e}", exc_info=True); return [], -1

def build_graph_dataset(coords: List[torch.Tensor], knn: int, sid: int, log: logging.Logger, dev: torch.device) -> List[Data]:
    dset = []
    for c in coords:
        pos = c.to(dev)
        # Using a simple scalar '1' for each node feature as atom types are not available
        x = torch.ones(pos.shape[0], 1).to(dev)
        edge_index = knn_graph(pos, k=knn, loop=False)
        dset.append(Data(x=x, edge_index=edge_index, pos=pos, y=pos, system_id=torch.tensor([sid])))
    log.info(f"Built graph dataset for system {sid} with {len(dset)} frames for CUEQ.")
    return dset

def save_checkpoint(state: dict, filename: str, logger: logging.Logger):
    try:
        torch.save(state, filename)
        logger.debug(f"Checkpoint saved: {filename}")
    except IOError as e:
        logger.error(f"Error saving checkpoint {filename}: {e}")

def load_checkpoint(model, optimizer, scaler, filename, device, logger):
    start_epoch = 0
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint: '{filename}'")
        ckpt = torch.load(filename, map_location=device)
        start_epoch = ckpt.get("epoch", 0)
        state_dict = ckpt["model_state_dict"]
        if isinstance(model, DDP): model.module.load_state_dict(state_dict)
        else:
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        if optimizer and "optimizer_state_dict" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and "scaler_state_dict" in ckpt: scaler.load_state_dict(ckpt["scaler_state_dict"])
        logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}")
    else:
        logger.info(f"No checkpoint found at '{filename}'. Starting fresh.")
    return model, optimizer, scaler, start_epoch

################################################################################
# (D) Training Function
################################################################################

def train_cueq_model(model, tr_loader, N_epochs, lr, ckpt_path, save_int, dev, log, rank, world_size):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    model, opt, scaler, start_ep = load_checkpoint(model, opt, scaler, ckpt_path, dev, log)

    if start_ep >= N_epochs:
        if rank == 0: log.info("CUEQ Encoder training already completed."); return model.module

    for ep in range(start_ep, N_epochs):
        model.train()
        tr_loader.sampler.set_epoch(ep)
        for data in tr_loader:
            data = data.to(dev)
            opt.zero_grad()
            with autocast():
                # The CUEQ encoder model expects pos as a separate argument
                pred_latent = model(data.x, data.edge_index, data.pos)
                # The target is the latent representation of the final coordinates.
                # Here, we simplify and use the coordinates themselves as a target for autoencoding.
                # A more advanced setup might encode data.y as well.
                loss = F.mse_loss(pred_latent, data.y)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        if rank == 0 and ((ep + 1) % save_int == 0 or (ep + 1) == N_epochs):
            save_checkpoint({
                "epoch": ep + 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }, ckpt_path, log)
            log.info(f"Epoch {ep+1}/{N_epochs} | Loss: {loss.item():.6f}")

    if rank == 0: log.info("Finished CUEQ Encoder training.")
    return model.module

################################################################################
# (E) Main Execution
################################################################################

def main():
    rank, world_size, device, logger = setup_ddp()
    start_time = time.time()
    if rank == 0: logger.info("================ CUEQ Blind Script Starting ================")
    
    with open(args.config, "r") as f: config = yaml.safe_load(f)
    cueq_cfg = config["cueq_encoder"] # New config section
    out_cfg = config["output_directories"]
    if rank == 0:
        os.makedirs(out_cfg['checkpoint_dir'], exist_ok=True)
        os.makedirs(out_cfg['latent_dir'], exist_ok=True)

    # Data loading (all ranks)
    full_dset = []
    for i, sys_cfg in enumerate(config["data"]["systems"]):
        coords, _ = load_heavy_atom_coords_from_json(sys_cfg["json_path"], logger)
        if coords:
            full_dset.extend(build_graph_dataset(coords, config["graph"]["knn_value"], i, logger, device))

    # Train/Test Split and Samplers
    tr_dset, te_dset = train_test_split(full_dset, test_size=0.1, random_state=42)
    tr_sampler = DistributedSampler(tr_dset, num_replicas=world_size, rank=rank)
    load_tr = DataLoader(tr_dset, cueq_cfg['batch_size'], sampler=tr_sampler, num_workers=config.get("num_workers",0), pin_memory=True)
    
    # CUEQ Encoder Training
    model = DDP(CUEQ_Encoder(
        in_irreps=cueq_cfg['in_irreps'],
        hidden_irreps_1=cueq_cfg['hidden_irreps_1'],
        hidden_irreps_2=cueq_cfg['hidden_irreps_2'],
        out_irreps=cueq_cfg['out_irreps'],
        mlp_h_dim=cueq_cfg['mlp_hidden_dim']
    ).to(device), device_ids=[rank])
    
    ckpt_path = os.path.join(out_cfg['checkpoint_dir'], "cueq_encoder_blind_checkpoint.pth")
    
    trained_encoder = train_cueq_model(model, load_tr, cueq_cfg['num_epochs'], cueq_cfg['learning_rate'], ckpt_path, cueq_cfg['save_interval'], device, logger, rank, world_size)

    # Export latent spaces (Rank 0 only)
    if rank == 0:
        logger.info("--- Starting Latent Space Export ---")
        trained_encoder.eval()
        # This part would typically use the full dataset loader without shuffling
        full_loader = DataLoader(full_dset, batch_size=cueq_cfg['batch_size'])
        
        # Aggregate representations by system_id
        all_latents = {}
        with torch.no_grad():
            for data in full_loader:
                data = data.to(device)
                with autocast():
                    latents = trained_encoder.forward_representation(data.x, data.edge_index, data.pos)
                
                for i in range(data.num_graphs):
                    sid = data.system_id[i].item()
                    if sid not in all_latents:
                        all_latents[sid] = []
                    
                    # Get the latents for the current graph in the batch
                    graph_latents = latents[data.batch == i]
                    all_latents[sid].append(graph_latents.cpu().numpy())
        
        # Save aggregated latents to an HDF5 file
        output_h5_path = os.path.join(out_cfg['latent_dir'], 'cueq_latents.h5')
        with h5py.File(output_h5_path, 'w') as f:
            for sid, latent_list in all_latents.items():
                if latent_list:
                    sys_group = f.create_group(f"system_{sid}")
                    sys_group.create_dataset("latents", data=np.concatenate(latent_list, axis=0))
                    logger.info(f"Saved latents for system {sid}")
        logger.info(f"--- Finished Latent Space Export to {output_h5_path} ---")

    cleanup_ddp()
    if rank == 0: logger.info(f"================ Script Finished ({time.time() - start_time:.2f}s) ================")

if __name__ == "__main__":
    main()
