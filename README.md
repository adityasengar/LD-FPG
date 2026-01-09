# Generative Modeling of Full-Atom Protein Conformations (LD-FPG)

## 🧬 Overview

This repository provides the Python-based implementation for the **Latent Diffusion for Full Protein Generation (LD-FPG)** framework, as described in our NeurIPS 2025 paper submission, "Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings."

The primary goal of this project is to generate diverse, all-atom conformational ensembles of proteins. This is achieved by learning from Molecular Dynamics (MD) simulation data through a multi-stage pipeline:

1. **Autoencoder Training:** A Chebyshev Graph Neural Network (ChebNet) based autoencoder learns a compressed latent representation of protein conformations. Different pooling strategies are explored to create this representation.
2. **Latent Diffusion Model Training:** A Denoising Diffusion Probabilistic Model (DDPM) is trained on the distribution of these learned latent embeddings.
3. **Structure Reconstruction:** New latent embeddings sampled by the trained diffusion model are decoded back into full-atom 3D protein coordinates.

This repository contains implementations for three distinct pooling strategies within the LD-FPG framework:

- **Blind Pooling:** Uses a global pooling mechanism over all atom embeddings.
- **Residue-based Pooling:** Focuses on residue-level deformations and contexts.
- **Sequential Pooling:** Decodes the protein structure in stages, typically backbone first, then sidechains.

Implementations for each strategy, including the necessary scripts and configuration files, are organized into their respective dedicated directories. Shared utility Python files for common tasks like data processing, model components, and training helpers are also provided.

Detailed instructions for running each specific pooling strategy can be found in the `README.md` file within its corresponding directory.

---

## ⚙️ Prerequisites & Dependencies

### Software Requirements

- **Python:** Version 3.8 or higher is recommended
- **CUDA:** For GPU acceleration, ensure you have a compatible CUDA toolkit installed. The PyTorch and PyTorch Geometric versions should match your CUDA version

### Core Python Libraries

The primary dependencies include:

- **PyTorch:** For tensor computations and neural network building (Version compatible with your PyTorch Geometric and CUDA versions)
- **PyTorch Geometric (PyG):** For graph neural network functionalities
- **h5py:** For reading and writing HDF5 files (used for storing large datasets like coordinates and embeddings)
- **PyYAML:** For parsing YAML configuration files
- **NumPy:** For numerical operations
- **scikit-learn:** For utilities like data splitting

### Installation

You can install these libraries using pip:

```bash
# 1. Install PyTorch (visit https://pytorch.org/get-started/locally/ for specific command)
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Install PyTorch Geometric (ensure compatibility with your PyTorch & CUDA version)
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch_geometric

# Optional: For CUDA accelerated operations if not automatically included
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
# (Replace ${TORCH} and ${CUDA} with your PyTorch and CUDA versions, e.g., 2.0.0 and cu118)

# 3. Install other dependencies
pip install h5py pyyaml numpy scikit-learn
```

Please refer to the official installation guides for PyTorch and PyTorch Geometric to ensure compatibility with your specific hardware and CUDA setup.

## 📁 Input Data Files

The scripts require specific input data files, typically placed in a `helper/` directory (or paths configured in the YAML files):

### Reference PDB File
- **Description:** A Protein Data Bank (PDB) file containing the heavy atom structure of your target protein. This is used for initial atom indexing and as a structural reference.
- **Example:** `helper/heavy_chain.pdb`

### Molecular Dynamics Trajectory Data
- **Description:** A JSON file containing per-frame heavy atom coordinates from your MD simulation. The format should be compatible with the data loading functions used in the scripts.
- **Example:** `helper/my_protein.json`

> **Note:** For large trajectory datasets, consider hosting them on a data repository like Zenodo and downloading them as needed. The D2R-MD dataset used in the paper is available at: https://zenodo.org/records/15479781

### Dihedral Angle Definition File (Optional)
- **Description:** A JSON file defining the atom quadruplets for standard dihedral angle calculations (e.g., φ, ψ, χ₁-χ₅). This is required if you enable dihedral angle-based loss terms during autoencoder training. The atom indexing in this file must correspond to the indexing derived from your reference PDB file.
- **Example:** `helper/condensed_residues.json`

Ensure these files are correctly formatted and accessible to the scripts by placing them in the expected locations or updating the paths in the respective `param.yaml` configuration files.

## 🚀 General Workflow

While each pooling strategy's directory (`blind/`, `residue/`, `sequential/`) has its specific README.md with detailed execution commands, the general workflow involves three main script executions:

### 1. Autoencoder Training
- Trains the ChebNet encoder and the specific pooling-based decoder
- Saves trained model checkpoints (encoder and decoder)
- Generates and saves the pooled latent embeddings (h₀) of the input MD data. This output is crucial for the next stage

### 2. Latent Diffusion Model Training
- Trains a DDPM on the distribution of pooled latent embeddings (h₀) from the previous stage
- Saves the trained diffusion model checkpoint
- Generates and saves new latent embeddings (h₀ᵍᵉⁿ) sampled from the trained DDPM

### 3. Structure Reconstruction
- Loads the trained decoder from Stage 1
- Loads the novel latent embeddings (h₀ᵍᵉⁿ) from Stage 2
- Loads the appropriate conditioner (e.g., reference structure's latent embedding Z_ref)
- Decodes the new latent embeddings to produce the final all-atom 3D protein coordinates

Please refer to the `README.md` inside each specific strategy folder for detailed command-line examples and configuration guidance.

## 🤝 Contributing

We welcome contributions to improve the LD-FPG framework! Please feel free to:

- Report bugs or issues
- Suggest new features or improvements
- Submit pull requests


## 📜 License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

This means you are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use
- **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits

For more details, see the [LICENSE](LICENSE) file or visit [Creative Commons CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

**⭐ If you find this work useful, please consider starring this repository!**

# Updated on 2026-01-09
