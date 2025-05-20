# Project Title: Generative Modeling of Full-Atom Protein Conformations (LD-FPG Sequential Pooling Implementation)

## 🧬 Overview

This repository provides a Python-based implementation of the **Latent Diffusion for Full Protein Generation (LD-FPG)** framework, specifically focusing on the **Sequential Pooling** strategy. The goal is to generate diverse, all-atom conformational ensembles of proteins by learning from Molecular Dynamics (MD) simulation data. This work is intended for submission to NeurIPS 2025 and is based on the methodology described in the accompanying paper, "Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings."

The sequential pooling pipeline involves three main stages:

1. **Stage 1: Autoencoder Component Training (`chebnet_seq.py`)**:
   * A Chebyshev Graph Neural Network (ChebNet) based **HNO Encoder** is trained to learn atom-wise latent embeddings from input protein conformations.
   * A **Backbone Decoder** is trained to take these HNO embeddings, pool the backbone-specific latents ($\mathbf{h}_{0,bb}$), and reconstruct backbone coordinates. During this training, the script saves the $\mathbf{h}_{0,bb}$ latents derived from the input dataset.
   * A **Sidechain Decoder** is trained to take HNO embeddings, the *predicted backbone coordinates* from the Backbone Decoder, and a conditioner. It pools sidechain-specific latents ($\mathbf{h}_{0,sc}$) and reconstructs sidechain coordinates to form the full atom structure. Similarly, it saves the $\mathbf{h}_{0,sc}$ latents from the input dataset.

2. **Stage 2: Separate Latent Diffusion Model Training (`new_diff.py`, run twice)**:
   * One Denoising Diffusion Probabilistic Model (DDPM) is trained on the distribution of **backbone pooled latents** ($\mathbf{h}_{0,bb}$) saved from Stage 1.
   * A second, independent DDPM is trained on the distribution of **sidechain pooled latents** ($\mathbf{h}_{0,sc}$) saved from Stage 1.

3. **Stage 3: Structure Reconstruction with Diffused Latents (`chebnet_seq.py` with override flags)**:
   * The trained HNO Encoder, Backbone Decoder, and Sidechain Decoder from Stage 1 are loaded.
   * *New* backbone latents ($\mathbf{h}_{0,bb}^{\text{gen}}$) and sidechain latents ($\mathbf{h}_{0,sc}^{\text{gen}}$) are sampled from their respective trained diffusion models (Stage 2).
   * `chebnet_seq.py` is run with special command-line flags (`--use_diffusion`, `--diffused_backbone_h5`, `--diffused_sidechain_h5`) to feed these novel diffused latents directly into the decoder stages, bypassing their internal pooling mechanisms, to generate full-atom protein structures.

---

## 🛠️ Methodology Highlights (Sequential Pooling)

* **Decoupled Latent Spaces:** Backbone and sidechain dynamics are captured in separate, specialized pooled latent spaces ($\mathbf{h}_{0,bb}$ and $\mathbf{h}_{0,sc}$).
* **Conditioned Generation:** The Sidechain Decoder's predictions are conditioned on the output of the Backbone Decoder, ensuring structural consistency.
* **Two-Step Diffusion:** Independent diffusion models learn the distributions of backbone and sidechain pooled latents, allowing for modular sampling.
* **Override Mechanism:** The main script `chebnet_seq.py` is versatile. It handles the initial training and data generation (Stage 1) and also performs the final reconstruction (Stage 3) by using command-line flags to accept externally generated (diffused) pooled latents.

---

## 📂 Repository Structure

```
.
├── chebnet_seq.py            # Script for Stage 1 (Autoencoder training) & Stage 3 (Reconstruction)
├── param.yaml                # Configuration for chebnet_seq.py
├── new_diff.py               # Script for Stage 2: Latent diffusion model training (used twice)
├── diffusion_backbone.yaml   # Config for new_diff.py (backbone latents)
├── diffusion_sidechain.yaml  # Config for new_diff.py (sidechain latents)
├── README.md                 # This file
│
├── helper/                   # Directory for input data files
│   ├── heavy_chain.pdb       # Example reference PDB structure
│   ├── my_protein.json       # Example MD trajectory data (or link to Zenodo)
│   └── condensed_residues.json # Example dihedral angle definitions (optional)
│
├── checkpoints/              # Default output directory for trained model weights
│   ├── hno_model.pth
│   ├── decoder_backbone.pth
│   ├── decoder_sidechain.pth
│   │
│   ├── diff_backbone/        # Checkpoints for backbone diffusion model
│   │   └── diffusion_exp_bb.pth
│   └── diff_sidechain/       # Checkpoints for sidechain diffusion model
│       └── diffusion_exp_sc.pth
│
├── structures/               # Default output directory for coordinate files
│   ├── X_ref_coords.pt
│   ├── ground_truth_aligned.h5
│   ├── hno_reconstructions.h5
│   ├── backbone_coords.h5    # Backbone reconstructions (from Stage 1 training data)
│   ├── full_coords.h5        # Full reconstructions (from Stage 1 training data)
│   │
│   ├── backbone_coords_diff.h5 # Backbone reconstructions (from Stage 3 using diffused latents)
│   └── full_coords_diff.h5   # Final generated structures (from Stage 3 using diffused latents)
│
└── latent_reps/              # Default output directory for latent embeddings
    ├── z_ref_embedding.pt
    ├── hno_embeddings.h5
    ├── backbone_pooled.h5    # Output of Stage 1 (BackboneDecoder), input to Stage 2A
    ├── sidechain_pooled.h5   # Output of Stage 1 (SidechainDecoder), input to Stage 2B
    │
    ├── diff_backbone/        # Outputs from backbone diffusion model
    │   └── generated_diff_exp.h5
    └── diff_sidechain/       # Outputs from sidechain diffusion model
        └── generated_diff_exp.h5
```

Log files (e.g., `my_logfile.log` from `param.yaml`, `diffusion_debug.log` from `new_diff.py`) will also be created.

---

## ⚙️ Prerequisites

* **Python:** 3.8+
* **Core Libraries:**
    * PyTorch (version compatible with PyG)
    * PyTorch Geometric (PyG)
    * h5py
    * PyYAML
    * NumPy
    * scikit-learn
    ```bash
    # Example installation (adjust for your PyTorch/CUDA version)
    # See: https://pytorch.org/get-started/locally/
    # See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
    pip install torch torchvision torchaudio
    pip install torch_geometric
    pip install h5py pyyaml numpy scikit-learn
    ```
* **CUDA:** Recommended for GPU acceleration.
* **Input Data Files** (place in `helper/` or update paths in YAML configurations):
    * **Reference PDB file:** (e.g., `helper/heavy_chain.pdb`).
    * **Molecular Dynamics Trajectory Data (JSON):** (e.g., `helper/my_protein.json`).
    * **Dihedral Angle Definition File (Optional):** (e.g., `helper/condensed_residues.json`). Required if dihedral losses are enabled in `param.yaml`.

---

## 🚀 Workflow: Generating Protein Conformations (Sequential Model)

### Step 1: Training Autoencoder Components & Generating Pooled Latents (`chebnet_seq.py`)

* **Purpose:**
    1. Train the HNO encoder.
    2. Train the Backbone Decoder (predicts backbone coordinates from HNO latents + reference).
    3. Train the Sidechain Decoder (predicts sidechain coordinates using HNO latents, predicted backbone, and reference).
    4. During the final export phase of this script (when run without diffusion override flags), it saves the pooled backbone embeddings (`backbone_pooled.h5`) and pooled sidechain embeddings (`sidechain_pooled.h5`) derived from your input MD dataset. These serve as training data for the diffusion models.

* **Configuration (`param.yaml`):**
    * **Input/Output Paths:** `json_path`, `pdb_filename`, `torsion_info_path`, `output_directories`.
    * **Model Hyperparameters:** `knn_value`, `cheb_order`, `hidden_dim` (for HNO), `pooling_dim_backbone`, `pooling_dim_sidechain`.
    * **`hno_training`**: Parameters for HNO encoder training.
    * **`decoderB_training` (Backbone Decoder):** Training parameters, `decoder_depth`, `mlp_hidden_dim`. Optional dihedral loss settings (`use_dihedral`, `lambda_1`, `lambda_2`, `divergence_type`) specifically for φ/ψ angles.
    * **`decoderSC_training` (Sidechain Decoder):** Training parameters, `arch_type` (determines how conditioner and backbone info are used), `decoder_depth`, `mlp_hidden_dim`. Optional dihedral loss settings (`use_dihedral_sc`, `lambda_1_sc`, `lambda_2_sc`, `divergence_type_sc`) for χ angles.
    * **Important:** For this stage, ensure the diffusion override flags (`--use_diffusion`, etc.) are *NOT* used with `chebnet_seq.py`.

* **Execution:**
    ```bash
    python chebnet_seq.py --config param.yaml
    ```
    Add `--debug` for verbose logging if specified in `param.yaml` or via CLI.

* **Key Outputs for this step:**
    * `checkpoints/hno_model.pth`
    * `checkpoints/decoder_backbone.pth`
    * `checkpoints/decoder_sidechain.pth`
    * `structures/X_ref_coords.pt` (reference coordinates)
    * `latent_reps/z_ref_embedding.pt` (reference latent embedding)
    * **`latent_reps/backbone_pooled.h5`**: Pooled backbone latents from input data (Dataset key: `backbone_pooled`). **Input for Step 2A.**
    * **`latent_reps/sidechain_pooled.h5`**: Pooled sidechain latents from input data (Dataset key: `sidechain_pooled`). **Input for Step 2B.**
    * Other analytical files (e.g., `full_coords.h5`, `hno_reconstructions.h5`).

### Step 2A: Training Latent Diffusion Model for Backbone Embeddings

* **Purpose:** Train a DDPM on the `backbone_pooled.h5` embeddings generated in Step 1.
* **Script:** `new_diff.py`
* **Configuration (`diffusion_backbone.yaml`):**
    * `run_mode`: e.g., `"grid_search"` or `"user_defined"`.
    * `parameters.h5_file_path`: Path to `latent_reps/backbone_pooled.h5`.
    * `parameters.dataset_key`: `"backbone_pooled"`.
    * `parameters.output_dir`: e.g., `latent_reps/diff_backbone`. Checkpoints and generated latents will be saved here.
    * `parameters.pooling_dim`: Must match the dimensions of the embeddings in `backbone_pooled.h5` (derived from `param.yaml`'s `pooling_dim_backbone`).
    * Other diffusion model parameters (model_type, schedule, epochs, etc.).

* **Execution:**
    ```bash
    python new_diff.py --config diffusion_backbone.yaml
    # Or for a specific experiment from grid:
    # python new_diff.py --config diffusion_backbone.yaml --exp_idx 1
    ```
    Use --instance_id for parallel grid search execution. Add --debug for verbose logs.

* **Key Outputs (in `latent_reps/diff_backbone/`):**
    * `checkpoints/diffusion_exp_bb.pth` (or similar, name includes experiment index)
    * `generated_diff_exp.h5` (containing diffused backbone latents $\mathbf{h}_{0,bb}^{\text{gen}}$)

### Step 2B: Training Latent Diffusion Model for Sidechain Embeddings

* **Purpose:** Train a DDPM on the `sidechain_pooled.h5` embeddings generated in Step 1.
* **Script:** `new_diff.py`
* **Configuration (`diffusion_sidechain.yaml`):**
    * `run_mode`: e.g., `"grid_search"` or `"user_defined"`.
    * `parameters.h5_file_path`: Path to `latent_reps/sidechain_pooled.h5`.
    * `parameters.dataset_key`: `"sidechain_pooled"`.
    * `parameters.output_dir`: e.g., `latent_reps/diff_sidechain`.
    * `parameters.pooling_dim`: Must match the dimensions of the embeddings in `sidechain_pooled.h5` (derived from `param.yaml`'s `pooling_dim_sidechain`).
    * Other diffusion model parameters.

* **Execution:**
    ```bash
    python new_diff.py --config diffusion_sidechain.yaml
    # Or for a specific experiment from grid:
    # python new_diff.py --config diffusion_sidechain.yaml --exp_idx 1
    ```

* **Key Outputs (in `latent_reps/diff_sidechain/`):**
    * `checkpoints/diffusion_exp_sc.pth` (or similar, name includes experiment index)
    * `generated_diff_exp.h5` (containing diffused sidechain latents $\mathbf{h}_{0,sc}^{\text{gen}}$)

### Step 3: Reconstructing Structures with Diffused Latents (`chebnet_seq.py` with override flags)

* **Purpose:** Generate novel full-atom protein structures by using the trained autoencoder components (HNO, BackboneDecoder, SidechainDecoder from Step 1) but feeding them the *newly sampled* backbone and sidechain pooled latents generated by the diffusion models in Step 2A and 2B.
* **Script:** `chebnet_seq.py` (the same script as Stage 1, but used with different command-line arguments).
* **Configuration:**
    * The script still loads model architectures and checkpoint paths for HNO, BackboneDecoder, and SidechainDecoder based on `param.yaml`. Ensure these paths in `param.yaml` point to the models trained in Step 1.
    * The crucial part is using the command-line override flags.

* **Execution:**
    ```bash
    python chebnet_seq.py --config param.yaml \
        --use_diffusion \
        --diffused_backbone_h5 latent_reps/diff_backbone/generated_diff_exp.h5 \
        --diffused_sidechain_h5 latent_reps/diff_sidechain/generated_diff_exp.h5
    ```
    * Replace paths with the desired experiment indices from your diffusion runs (Step 2A and 2B).
    * `--config param.yaml`: Provides paths to trained autoencoder component checkpoints and their architectural details.
    * `--use_diffusion`: This flag signals the script to use the external HDF5 files for pooled latents.
    * `--diffused_backbone_h5`: Path to the HDF5 file containing diffused backbone pooled latents from Step 2A.
    * `--diffused_sidechain_h5`: Path to the HDF5 file containing diffused sidechain pooled latents from Step 2B.

* **Key Inputs:**
    * `param.yaml` (for loading trained autoencoder component architectures and their checkpoint paths).
    * Trained `hno_model.pth`, `decoder_backbone.pth`, `decoder_sidechain.pth` (loaded via paths in `param.yaml`).
    * `structures/X_ref_coords.pt` and/or `latent_reps/z_ref_embedding.pt` (for conditioning, loaded via `param.yaml`).
    * HDF5 file with diffused backbone latents from Step 2A.
    * HDF5 file with diffused sidechain latents from Step 2B.

* **Key Outputs:**
    * `structures/backbone_coords_diff.h5`: Backbone coordinates reconstructed using the diffused backbone latents.
    * `structures/full_coords_diff.h5`: Final all-atom generated structures.

---

## ✨ Example Usage (Illustrative)

**Step 1: Train Autoencoder Components**
```bash
# Configure param.yaml with paths and training settings for HNO, BackboneDecoder, SidechainDecoder
python chebnet_seq.py --config param.yaml
```
This generates `latent_reps/backbone_pooled.h5` and `latent_reps/sidechain_pooled.h5`.

**Step 2A: Train Backbone Diffusion Model**
```bash
# Configure diffusion_backbone.yaml to point to backbone_pooled.h5 and set output dir
# Example: Running as experiment 1 (user_defined or first in grid)
python new_diff.py --config diffusion_backbone.yaml --exp_idx 1
```
This generates, e.g., `latent_reps/diff_backbone/generated_diff_exp1.h5`.

**Step 2B: Train Sidechain Diffusion Model**
```bash
# Configure diffusion_sidechain.yaml to point to sidechain_pooled.h5 and set output dir
# Example: Running as experiment 1 (user_defined or first in grid)
python new_diff.py --config diffusion_sidechain.yaml --exp_idx 1
```
This generates, e.g., `latent_reps/diff_sidechain/generated_diff_exp1.h5`.

**Step 3: Reconstruct Structures from Diffused Latents**
```bash
# Ensure param.yaml points to the correct trained autoencoder component checkpoints from Step 1
python chebnet_seq.py --config param.yaml \
    --use_diffusion \
    --diffused_backbone_h5 latent_reps/diff_backbone/generated_diff_exp1.h5 \
    --diffused_sidechain_h5 latent_reps/diff_sidechain/generated_diff_exp1.h5
```
This generates `structures/full_coords_diff.h5` containing novel structures.

---

## 🔑 Key Outputs Explained & File Naming

**Autoencoder Checkpoints** (from Stage 1, in `checkpoints/`):
* `hno_model.pth`: Trained HNO encoder.
* `decoder_backbone.pth`: Trained Backbone Decoder.
* `decoder_sidechain.pth`: Trained Sidechain Decoder.

**Pooled Latents from Training Data** (from Stage 1, in `latent_reps/`):
* `backbone_pooled.h5`: Pooled backbone latents ($\mathbf{h}_{0,bb}$) from the input dataset. Training data for the backbone diffusion model. (Dataset key: `backbone_pooled`).
* `sidechain_pooled.h5`: Pooled sidechain latents ($\mathbf{h}_{0,sc}$) from the input dataset. Training data for the sidechain diffusion model. (Dataset key: `sidechain_pooled`).

**Diffusion Model Outputs** (from Stage 2, in e.g., `latent_reps/diff_backbone/` and `latent_reps/diff_sidechain/`):
* `checkpoints/diffusion_exp_bb.pth` (or similar): Checkpoint for a backbone diffusion model experiment.
* `generated_diff_exp.h5`: Newly generated backbone pooled latents ($\mathbf{h}_{0,bb}^{\text{gen}}$) from experiment. (Dataset key: `generated_diffusion`).
* `checkpoints/diffusion_exp_sc.pth` (or similar): Checkpoint for a sidechain diffusion model experiment.
* `generated_diff_exp.h5`: Newly generated sidechain pooled latents ($\mathbf{h}_{0,sc}^{\text{gen}}$) from experiment. (Dataset key: `generated_diffusion`).

**Choosing experiments**: Evaluate your diffusion experiments based on training loss and qualitative assessment of generated structures (by running a few samples through Stage 3). Select the best performing experiment for backbone and sidechain.

**Final Reconstructed Structures** (from Stage 3, in `structures/`):
* `backbone_coords_diff.h5`: Backbone structures reconstructed using the diffused backbone latents.
* `full_coords_diff.h5`: Final all-atom generated structures using diffused backbone and sidechain latents. The names of these files do not currently include experiment indices from the diffused inputs; you may want to manage outputs from different diffused latents by saving them to different directories or renaming them manually.

---

## 🔧 Customization and Advanced Use

* **Dihedral Losses**: `param.yaml` allows for separate configuration of dihedral losses (MSE and distribution divergence) for the Backbone Decoder (φ, ψ angles) and the Sidechain Decoder (χ angles) during Stage 1 training.
* **Sidechain Decoder Architecture** (`arch_type`): The `decoderSC_training.arch_type` in `param.yaml` controls how the Sidechain Decoder combines information (e.g., predicted backbone coordinates, reference latents).
* **Pooling Dimensions**: `pooling_dim_backbone` and `pooling_dim_sidechain` in `param.yaml` control the size of the respective pooled latent vectors. These dimensions must be consistent with the `pooling_dim` setting in the corresponding diffusion YAML files (`diffusion_backbone.yaml`, `diffusion_sidechain.yaml`).
* **Diffusion Hyperparameters**: `new_diff.py` and its YAML configurations offer extensive control over the diffusion model training (architecture, schedule, learning rate, epochs).

---

## 📄 Citing this Work

If you use this code or the LD-FPG methodology in your research, please cite our NeurIPS 2025 paper:

[Placeholder for NeurIPS Paper Citation - To be added upon acceptance/publication]

Title: Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings  
Authors: [Author One, Author Two, et al.]  
Conference: Advances in Neural Information Processing Systems (NeurIPS) 2025.

---

## 🐛 Troubleshooting

* **Dimension Mismatches**: This is critical in the sequential pipeline.
  * Ensure the output dimensions of `backbone_pooled.h5` and `sidechain_pooled.h5` (determined by `param.yaml`'s `pooling_dim_backbone` and `pooling_dim_sidechain`) match the `pooling_dim` expected by `new_diff.py` in `diffusion_backbone.yaml` and `diffusion_sidechain.yaml`.
  * When using `--use_diffusion` with `chebnet_seq.py`, ensure the HDF5 files with diffused latents have the correct dimensions expected by the BackboneDecoder and SidechainDecoder's pooling layers.
* **File Not Found**: Verify all paths in YAML configurations and command-line arguments.
* **Checkpoint Loading**: Ensure that paths to pre-trained model components (`hno_model.pth`, `decoder_backbone.pth`, `decoder_sidechain.pth`) in `param.yaml` are correct when running Stage 3.
* **Log Files**: Check `my_logfile.log` (from `chebnet_seq.py`) and `diffusion_debug.log` (from `new_diff.py`, if `--debug` is used) for detailed error messages.
