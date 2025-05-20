# Project Title: Generative Modeling of Full-Atom Protein Conformations (LD-FPG Implementation)

## 🧬 Overview

This repository provides a Python-based implementation of the **Latent Diffusion for Full Protein Generation (LD-FPG)** framework, focusing on the **Blind Pooling** strategy. The goal is to generate diverse, all-atom conformational ensembles of proteins by learning from Molecular Dynamics (MD) simulation data. This work is intended for submission to NeurIPS 2025 and is based on the methodology described in the accompanying paper, "Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings."

The pipeline consists of three main stages:

1. **Autoencoder Training (`chebnet_blind.py`):** Trains a Chebyshev Graph Neural Network (ChebNet)-based autoencoder.
2. **Latent Diffusion Model Training (`new_diff.py`):** Trains a DDPM on latent embeddings.
3. **Structure Reconstruction (`chebnet_diff.py`):** Decodes latent embeddings into full-atom protein coordinates.

---

## 🛠️ Methodology Highlights

* **ChebNet Autoencoder:** Processes protein structures into latent vectors using global pooling.
* **Latent Diffusion Model:** Trains a DDPM to generate new latent samples.
* **Structure Reconstruction:** Converts latent vectors back to protein coordinates.

---

## 📂 Repository Structure

```
.
├── chebnet_blind.py
├── param.yaml
├── new_diff.py
├── param_diff.yaml
├── chebnet_diff.py
├── README.md
├── helper/
│   ├── heavy_chain.pdb
│   ├── my_protein.json
│   └── condensed_residues.json
├── checkpoints/
│   ├── hno_checkpoint.pth
│   ├── decoder2_checkpoint.pth
│   └── diffusion_checkpoint_exp<N>.pth
├── structures/
│   ├── X_ref_coords.pt
│   ├── ground_truth_aligned.h5
│   ├── hno_reconstructions.h5
│   ├── full_coords.h5
│   └── full_coords_diff.h5
└── latent_reps/
    ├── z_ref_embedding.pt
    ├── hno_embeddings.h5
    ├── pooled_embedding.h5
    └── generated_embeddings_exp<N>.h5
```

---

## ⚙️ Prerequisites

* **Python:** 3.8+
* **Core Libraries:** PyTorch, PyTorch Geometric, h5py, PyYAML, NumPy, scikit-learn

```bash
pip install torch torchvision torchaudio
torch_geometric h5py pyyaml numpy scikit-learn
```

* **CUDA:** Recommended for GPU acceleration.

---

## 🚀 Workflow

### Step 1: Autoencoder Training

```bash
python chebnet_blind.py --config param.yaml
```

### Step 2: Latent Diffusion Training

```bash
python new_diff.py --config param_diff.yaml --exp_idx 1
```

### Step 3: Structure Reconstruction

```bash
python chebnet_diff.py \
    --config param.yaml \
    --decoder2_ckpt checkpoints/decoder2_checkpoint.pth \
    --diff_emb_file latent_reps/generated_embeddings_exp1.h5 \
    --conditioner_x_ref_pt structures/X_ref_coords.pt \
    --conditioner_z_ref_pt latent_reps/z_ref_embedding.pt \
    --output_file structures/full_coords_diff_exp1.h5
```

---

## 🔑 Key Outputs

* **Autoencoder Weights:** `hno_checkpoint.pth`, `decoder2_checkpoint.pth`
* **Latent Embeddings:** `pooled_embedding.h5`, `generated_embeddings_exp<N>.h5`
* **Structures:** `X_ref_coords.pt`, `full_coords_diff_exp<N>.h5`

---

## 📄 Citing this Work

If using this implementation, please cite:

```
[Placeholder for NeurIPS Paper Citation - To be added upon acceptance/publication]
Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings
Authors: [Author One, Author Two, et al.]
Conference: Advances in Neural Information Processing Systems (NeurIPS) 2025.
```

---

## 🐛 Troubleshooting

* **Dimension Mismatches:** Verify pooled dimensions.
* **File Paths:** Check YAML configuration paths.
* **CUDA Errors:** Adjust batch sizes or verify installations.

---

## 📜 License

Specify your chosen license here (e.g., MIT, Apache 2.0).
