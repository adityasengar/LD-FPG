# Protein Structure Autoencoder Training (LD-FPG Step 1 Variant)

This part of the project focuses on training a graph-based autoencoder for protein structures. It corresponds to **Step 1** of a larger pipeline aimed at generative modeling of full-atom protein conformations using latent diffusion. This specific script trains a Chebyshev Graph Autoencoder (HNO Encoder + Decoder2 MLP) to learn a compressed latent representation of protein structures, particularly focusing on a **blind pooling** strategy by default.

## Overview

The autoencoder consists of:
1.  **HNO Encoder**: A ChebNet-based graph neural network that encodes input protein coordinates (from MD snapshots) into atom-wise latent embeddings ($Z$).
2.  **Pooling Layer**: The atom-wise embeddings $Z$ are processed by a pooling strategy (configured as "blind" in the default YAML) to create a compact global latent representation ($\mathbf{h}_0$).
3.  **Decoder2 MLP**: A multi-layer perceptron that reconstructs the full-atom Cartesian coordinates from the pooled latent representation $\mathbf{h}_0$, conditioned on reference information (either reference coordinates $X_{ref}$ or reference latent embeddings $Z_{ref}$).

The script handles data loading, preprocessing (PDB parsing, coordinate alignment), model training for both encoder and decoder stages, and exporting key outputs.

## File Structure

The code is organized into the following Python modules:

* `main_autoencoder.py`: The main executable script that orchestrates the entire training workflow for this autoencoder step.
* `models.py`: Contains the PyTorch `nn.Module` definitions for the `HNO` encoder and the `ProteinStateReconstructor2D` (Decoder2) model.
* `trainers.py`: Includes the training loop functions (`train_hno_model`, `train_decoder2_model`).
* `data_utils.py`: Utility functions for data loading (PDB, JSON), coordinate alignment (Kabsch), and PyTorch Geometric graph dataset construction.
* `math_utils.py`: Functions for mathematical operations like dihedral angle computation, MSE loss calculation, and distribution divergence metrics (KL, JS, Wasserstein).
* `checkpoint_utils.py`: Helper functions for saving and loading model checkpoints.
* `export_utils.py`: Functionality to export the final ground truth data, model reconstructions, and learned latent representations to HDF5 files.
* `configs/`: Directory containing YAML configuration files.
    * `blind_pooling_config.yaml`: Default configuration file for this autoencoder training script.

## Prerequisites

* Python 3.x
* PyTorch
* torch_geometric (PyG)
* h5py
* PyYAML
* NumPy
* scikit-learn (for `train_test_split`)

You can install most dependencies using pip:
```bash
pip install torch torchvision torchaudio
pip install torch_geometric
pip install h5py pyyaml numpy scikit-learn
