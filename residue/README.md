# Generative Modeling of Full-Atom Protein Conformations (LD-FPG Residue Pooling Implementation)

## 🧬 Overview

This repository provides a Python-based implementation of the **Latent Diffusion for Full Protein Generation (LD-FPG)** framework, specifically focusing on the **Residue Pooling** strategy. The goal is to generate diverse, all-atom conformational ensembles of proteins by learning from Molecular Dynamics (MD) simulation data. This work is intended for submission to NeurIPS 2025 and is based on the methodology described in the accompanying paper.

The residue pooling pipeline involves three main stages:

1. **Stage 1: Autoencoder Training & Per-Residue Pooled Latent Generation (`chebnet_res.py`)**:
   * A Chebyshev Graph Neural Network (ChebNet) based **HNO Encoder** is trained to learn atom-wise latent embeddings.
   * The *`ProteinStateReconstructor2D`* (Decoder2) is trained, configured for **residue-level pooling**. It takes HNO embeddings and a conditioner to reconstruct full-atom coordinates.
   * During its export phase (controlled by *`save_decoder2_pooled_latent: true`* in *`param.yaml`* or if *`recon_output`* is not 1), this script saves the *per-residue pooled latent embeddings* ($\mathbf{h}_{0,R_j}$) derived from the input MD dataset. These have a shape like `(N_frames, N_residues, pooled_dim_per_residue)`.

2. **Stage 2: Latent Diffusion Model Training on Per-Residue Embeddings (`new_diff.py`)**:
   * A Denoising Diffusion Probabilistic Model (DDPM) is trained on the distribution of these per-residue pooled latent embeddings saved from Stage 1.
   * The `param_diff.yaml` configuration allows specifying which subset of residue latents to train the diffusion model on via the `selected_residues` key.
   * This stage outputs newly sampled *diffused per-residue pooled latents* ($\mathbf{h}_{0,R_j}^{\text{gen}}$).

3. **Stage 3: Structure Reconstruction with Diffused Per-Residue Latents (`chebnet_res.py`)**:
   * The same `chebnet_res.py` script is used, loading the trained HNO Encoder and `ProteinStateReconstructor2D` from Stage 1.
   * The key difference is that `param.yaml` is modified to point `decoder2_settings.special_res_file` to the HDF5 file containing the diffused per-residue latents from Stage 2.
   * The `decoder2_settings.override_residues` list in `param.yaml` must specify which residue latents should be replaced by those from the `special_res_file`.
   * The script then reconstructs full-atom structures using these (partially or fully) overridden diffused latents.

---

## 🛠️ Methodology Highlights (Residue Pooling)

* **Localized Latent Representations:** Each residue (or segment) of the protein gets its own pooled latent vector, aiming to capture local conformational dynamics more directly.
* **Structured Latent Space for Diffusion:** The diffusion model learns the joint distribution of these per-residue latent vectors.
* **Targeted Override for Reconstruction:** The `ProteinStateReconstructor2D` can selectively use diffused latents for specified residues, allowing for flexible generation or analysis.

---

## 📂 Repository Structure

```
.
├── chebnet_res.py            # Script for Stage 1 (Autoencoder) & Stage 3 (Reconstruction)
├── param.yaml                # Configuration for chebnet_res.py
├── new_diff.py               # Script for Stage 2: Latent diffusion model training
├── param_diff.yaml           # Configuration for new_diff.py (for residue latents)
├── README.md                 # This file
│
├── helper/                   # Directory for input data files
│   ├── heavy_chain.pdb       # Example reference PDB structure
│   ├── my_protein.json       # Example MD trajectory data (JSON from extract_residues.py)
│   └── condensed_residues.json # Dihedral angle definitions (from condense_residues.py)
│
├── checkpoints/              # Default output directory for trained model weights
│   ├── hno_checkpoint.pth
│   ├── decoder2_residue.pth  # Or decoder2_residue_2level.pth
│   └── <exp_label>epoch_snapshots/epoch<E>/diffusion_checkpoint.pth # From new_diff.py N_out
│       # Or checkpoints/diffusion_exp<N>.pth if N_out=0
│
├── structures/               # Default output directory for coordinate files
│   ├── X_ref_coords.npy      # Reference coordinates for conditioning
│   ├── hno_reconstructed_coords.h5
│   └── decoder2_output_residue_z_ref.h5 # Contains reconstructions from Stage 1 & Stage 3
│       ├── reconstructions_with_override
│       └── reconstructions_no_override
│
└── latent_reps/              # Default output directory for latent embeddings
    ├── z_ref_embedding.npy
    ├── hno_latent_embeddings.h5
    ├── decoder2_pooled_latent_embeddings.h5 # Output of Stage 1, input to Stage 2
    │
    └── <exp_label>epoch_snapshots/    # Output structure from new_diff.py with N_out > 0
        └── epoch<E>/
            └── generated_embeddings.h5 # Diffused per-residue latents
    # Or latent_reps/generated_embeddings_exp<N>.h5 if N_out=0 in new_diff.py
```

Log files (e.g., `logfile_optimized.log` from `chebnet_res.py`, `diffusion_debug.log` from `new_diff.py`) will also be created.

---

## ⚙️ Prerequisites

* **Python:** 3.8+
* **Core Libraries:**
    * PyTorch, PyTorch Geometric (PyG), h5py, PyYAML, NumPy, scikit-learn.
* **Input Data Files** (typically in `helper/`):
    * **Original Trajectory Data (JSON):** A JSON file in the format expected by `chebnet_res.py`'s `load_heavy_atom_coords_from_json` function (e.g., `helper/my_protein.json`). This file should be the output of a script like `extract_residues.py` followed by `condense_residues.py` if the remapping to contiguous indices is done by `condense_residues.py`. The `chebnet_res.py` itself performs a remapping step internally based on `original_residue_atom_indices` loaded from the JSON. *Ensure your input JSON for *`chebnet_res.py`* correctly provides *`heavy_atom_indices`* per residue for this internal remapping.*
    * **(Optional but recommended for `residue` pooling):** A PDB file for context if needed, though `chebnet_res.py` relies heavily on the input JSON's structure.

---

## 🚀 Workflow: Generating Protein Conformations (Residue Pooling Model)

### Step 1: Training Autoencoder & Generating Per-Residue Pooled Latents (`chebnet_res.py`)

* **Purpose:**
    1. Train the HNO encoder.
    2. Train the `ProteinStateReconstructor2D` (Decoder2) configured for residue-level pooling.
    3. Save the per-residue pooled latent embeddings ($\mathbf{h}_{0,R_j}$) derived from your input MD dataset. This is crucial for training the diffusion model.

* **Configuration (`param.yaml`):**
    * `json_path`: Path to your processed MD trajectory JSON data.
    * `pooling_type`: Must be set to `"residue"`.
    * *`output_size_per_segment`*: Defines the pooling dimensions (H, W) for each residue's latent vector. The product *`H × W`* is the `pooled_dim_per_residue`.
    * `use_second_level_pooling` (optional): If `true`, applies another pooling step over all residue latents.
    * `save_decoder2_pooled_latent: true` (add this to config if not present): Ensure this is enabled to save `decoder2_pooled_latent_embeddings.h5`.
    * `conditioner_mode`: `"z_ref"` or `"X_ref"`.
    * Training parameters for HNO (`decoder1`) and Decoder2 (`decoder2`).
    * For this stage, ensure `decoder2_settings.special_res_file` is `null` or not set, and `decoder2_settings.override_residues` is empty.

* **Execution:**
    ```bash
    python chebnet_res.py --config param.yaml
    ```

* **Key Outputs for this step:**
    * `checkpoints/hno_checkpoint.pth`
    * `checkpoints/decoder2_residue.pth` (or `decoder2_residue_2level.pth` if 2nd level pooling used)
    * `structures/X_ref.npy` (Reference coordinates from the first frame)
    * `latent_reps/z_ref_embedding.npy` (If `conditioner_mode: "z_ref"`)
    * **`latent_reps/decoder2_pooled_latent_embeddings.h5`**: Contains per-residue pooled latents. Shape `(N_frames, N_residues, pooled_dim_per_residue)`. **Input for Step 2.** (Dataset key: `pooled_latent`).

### Step 2: Training Latent Diffusion Model on Per-Residue Embeddings (`new_diff.py`)

* **Purpose:** Train a DDPM on the `decoder2_pooled_latent_embeddings.h5` (per-residue latents) from Step 1.

* **Configuration (`param_diff.yaml` for residue pooling):**
    * `run_mode`: e.g., `"grid_search"` or `"user_defined"`.
    * `parameters.h5_file_path`: Path to `latent_reps/decoder2_pooled_latent_embeddings.h5`.
    * `parameters.dataset_key`: `"pooled_latent"`.
    * `parameters.selected_residues`: A list of 0-based residue indices to train the diffusion model on. For example, `[0, 1, 2, ..., N_residues-1]` to train on all, or a subset like `[10, 15, 20]` for focused diffusion. The `new_diff.py` script will slice the HDF5 data according to these indices.
    * `parameters.output_dir`: e.g., `latent_reps/my_residue_diffusion_exp1`. This directory will store checkpoints and generated latents.
    * `parameters.pooling_dim`: This should be `[num_selected_residues, pooled_dim_per_residue]`. The script uses this for internal reshaping if `model_type` were `conv2d`, but for MLP models (which `new_diff.py` is focused on), the data is flattened. The crucial part is that `residue_embeddings` loaded will have shape `(N_samples_total, num_sel_residues, pooling_dim_per_residue)`.
    * `N_out` (command-line or in YAML): If > 0, saves epoch snapshots (checkpoints and generated embeddings) every `N_out` epochs into subfolders like `output_dir/<exp_label>_epoch_snapshots/epoch_<E>/`.

* **Execution:**
    ```bash
    # Example for a specific experiment from grid search (defined in new_diff.py script itself)
    # The YAML primarily provides data paths and general settings for this script.
    python new_diff.py --config param_diff.yaml --exp_idx 1 --N_out 10000
    
    # Or to run a specific instance_id for parallel grid search:
    # python new_diff.py --config param_diff.yaml --instance_id 0 --N_out 10000
    ```

* **Key Outputs (in the specified `output_dir` from `param_diff.yaml`):**
    * If `N_out > 0`:
        * `<output_dir>/<exp_label_from_script_logic>_epoch_snapshots/epoch_<E>/diffusion_checkpoint.pth`
        * `<output_dir>/<exp_label_from_script_logic>_epoch_snapshots/epoch_<E>/generated_embeddings.h5`
    * If `N_out = 0` (or at the end of training):
        * `<output_dir>/checkpoints/diffusion_checkpoint_exp<N>.pth`
        * `<output_dir>/generated_embeddings_exp<N>.h5`
    * The `generated_embeddings.h5` files will contain diffused per-residue latents with shape `(N_generated_samples, N_selected_residues, pooled_dim_per_residue)`.

### Step 3: Reconstructing Structures with Diffused Per-Residue Latents (`chebnet_res.py`)

* **Purpose:** Generate novel full-atom protein structures using the trained HNO and Decoder2 from Stage 1, but with the per-residue pooled latents for specified residues overridden by the diffused latents generated in Stage 2.

* **Configuration:**
    1. **Modify `param.yaml` (the one used for `chebnet_res.py`):**
        * `decoder2_settings.special_res_file`: Set this to the full path of the `generated_embeddings.h5` file from Stage 2 that you want to use for reconstruction (e.g., `latent_reps/my_residue_diffusion_exp1_epoch_snapshots/epoch_7500000/generated_embeddings.h5`).
        * `decoder2_settings.override_residues`: This list must contain the 0-based indices of the residues whose pooled latents you want to replace with those from `special_res_file`.
            * If your *`generated_embeddings.h5`* (from diffusion) contains latents for *all* residues (i.e., `selected_residues` in `param_diff.yaml` covered all residues), then `override_residues` should be `[0, 1, ..., num_total_residues-1]`.
            * If *`generated_embeddings.h5`* contains latents for a *subset* of residues (e.g., those specified in `selected_residues` in `param_diff.yaml`), then `override_residues` in `param.yaml` must be *exactly that same list of residue indices*. The order matters.
        * Ensure `recon_output: 1` in `param.yaml` to trigger the export that uses this override logic.
    2. Ensure paths to trained model checkpoints (`hno_checkpoint.pth`, `decoder2_residue.pth`) in `param.yaml` are correct.

* **Execution:**
    The current `chebnet_res.py` doesn't take the diffused H5 path via CLI directly for this override, it uses `special_res_file` from its `param.yaml`. The CLI args `--exp_label`, `--embedding_base_folder`, `--epoch_folder` are for organizing outputs of `chebnet_res.py` itself if it were run multiple times, not for specifying the *input* diffused H5.
    
    **Therefore, after modifying `param.yaml` as described above:**
    ```bash
    python chebnet_res.py --config param.yaml
    ```

* **Key Inputs:**
    * Modified `param.yaml` (pointing `special_res_file` to diffused latents and setting `override_residues`).
    * Trained `hno_checkpoint.pth`, `decoder2_residue.pth` (loaded via paths in `param.yaml`).
    * `structures/X_ref.npy` and/or `latent_reps/z_ref_embedding.npy` (for conditioning).

* **Key Outputs:**
    * The primary output of interest will be in the HDF5 file specified by `recon_output: 1` in `param.yaml` (e.g., `structures/decoder2_output_residue_z_ref.h5`).
    * Inside this HDF5, the dataset `reconstructions_with_override` will contain the final generated structures using the diffused per-residue latents.
    * `reconstructions_no_override` shows reconstructions using the original autoencoder's internally pooled latents.

---

## ✨ Example Usage (Illustrative)

**Step 1: Train Autoencoder & Generate Pooled Latents for Diffusion Training**

```bash
# Configure param.yaml:
# - json_path, pdb_filename
# - pooling_type: "residue"
# - output_size_per_segment: [1, 3] # Example: each residue pooled to 3 dimensions
# - save_decoder2_pooled_latent: true
# - Ensure special_res_file is null or commented out, override_residues is empty.
# - recon_output: 0 (or any value other than 1 if you only want to generate the pooled latents for diffusion)
#   (If recon_output=1, it will also do a self-reconstruction pass, ensure save_decoder2_pooled_latent is true)

python chebnet_res.py --config param.yaml
```

This generates `latent_reps/decoder2_pooled_latent_embeddings.h5` with shape `(N_frames, N_residues, 3)`.

**Step 2: Train Latent Diffusion Model on Per-Residue Embeddings**

```bash
# Configure param_diff.yaml:
# - parameters.h5_file_path: "latent_reps/decoder2_pooled_latent_embeddings.h5"
# - parameters.dataset_key: "pooled_latent"
# - parameters.selected_residues: [0, 1, 2, ..., N_residues-1] # To train on all residues
# - parameters.output_dir: "latent_reps/my_residue_diff_run"
# - parameters.pooling_dim: [N_residues, 3] # N_residues from your protein, 3 from above
# - parameters.N_out: 50000 # Example: save snapshot every 50k epochs

python new_diff.py --config param_diff.yaml --exp_idx 1
```

This generates, e.g., `latent_reps/my_residue_diff_run/exp1_epoch_snapshots/epoch_7500000/generated_embeddings.h5` (if num_epochs=7.5M, N_out=50k). Shape: `(N_gen, N_residues, 3)`.

**Step 3: Reconstruct Structures from Diffused Per-Residue Latents**

```bash
# 1. Modify param.yaml:
#    decoder2_settings:
#      ...
#      special_res_file: "latent_reps/my_residue_diff_run/exp1_epoch_snapshots/epoch_7500000/generated_embeddings.h5"
#      override_residues: [0, 1, 2, ..., N_residues-1] # Match selected_residues from diffusion
#      ...
#    recon_output: 1 # Ensure this is set to trigger export with override
#
# 2. Run chebnet_res.py again:

python chebnet_res.py --config param.yaml
```

This uses the diffused latents specified in param.yaml to generate `structures/decoder2_output_residue_z_ref.h5`, where the `reconstructions_with_override` dataset contains the novel structures.

---

## 🔑 Key Outputs Explained & File Naming

**Autoencoder Checkpoints** (from Stage 1, in `checkpoints/`):
- `hno_checkpoint.pth`
- `decoder2_residue.pth` (or `decoder2_residue_2level.pth`)

**Pooled Latents for Diffusion Training** (from Stage 1, in `latent_reps/`):
- `decoder2_pooled_latent_embeddings.h5`: Contains per-residue pooled latents from the input MD. Shape: `(N_frames, N_residues, pooled_dim_per_residue)`. Dataset key: `pooled_latent`.

**Diffusion Model Outputs** (from Stage 2, in `output_dir` specified in `param_diff.yaml`):
- If `N_out > 0`: Snapshots are saved in `<output_dir>/<exp_label>_epoch_snapshots/epoch_<E>/`.
  - `diffusion_checkpoint.pth`: Checkpoint of the diffusion model at epoch `<E>`.
  - `generated_embeddings.h5`: Diffused per-residue latents from this snapshot. Shape: `(N_gen_samples, N_selected_residues_for_diffusion, pooled_dim_per_residue)`.
- If `N_out = 0` (or for the final model):
  - `<output_dir>/checkpoints/diffusion_exp<N>.pth`
  - `<output_dir>/generated_embeddings_exp<N>.h5`
- `selected_residues` in `param_diff.yaml`: This list determines which residues' latents (and how many, `N_selected_residues_for_diffusion`) are included in the `generated_embeddings.h5` from the diffusion model. This is critical for Stage 3.

**Final Reconstructed Structures** (from Stage 3, e.g., in `structures/`):
- `decoder2_output_residue_z_ref.h5` (filename depends on `param.yaml` settings for pooling and conditioner):
  - `reconstructions_with_override`: Structures generated using the diffused per-residue latents loaded via `special_res_file`.
  - `reconstructions_no_override`: Structures generated using the autoencoder's internally pooled latents from the input data (for comparison).

---

## 🔧 Customization and Advanced Use

- **`pooling_type: "residue"`**: Must be set in `param.yaml` for this workflow.
- **`output_size_per_segment`**: Controls the dimension of each per-residue pooled latent vector.
- **`use_second_level_pooling`**: If `true` in `param.yaml`, an additional pooling layer is applied over all the per-residue latents, resulting in a single global latent vector that is then repeated for each residue before MLP input. This changes the nature of `decoder2_pooled_latent_embeddings.h5` and how diffusion should be approached. The provided `new_diff.py` is designed for per-residue latents or global latents, so ensure consistency. The current example workflow assumes `use_second_level_pooling: false` for per-residue diffusion.
- **`selected_residues` (in `param_diff.yaml`)**: Allows training the diffusion model on only a subset of residue latents, which can be useful for very large proteins or for focusing on specific dynamic regions. Ensure the `override_residues` list in `param.yaml` for Stage 3 matches these selected indices.
- **`N_out` in `new_diff.py`**: Useful for saving intermediate snapshots of diffused latents during a long diffusion training run, allowing for evaluation at different training stages.

---

## 📄 Citing this Work

If you use this code or the LD-FPG methodology in your research, please cite our NeurIPS 2025 paper:

[Placeholder for NeurIPS Paper Citation - To be added upon acceptance/publication]

**Title:** Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings  
**Authors:** [Author One, Author Two, et al.]  
**Conference:** Advances in Neural Information Processing Systems (NeurIPS) 2025.

---

## 🐛 Troubleshooting

**Dimension Mismatches:**
- The `pooled_dim_per_residue` (product of `output_size_per_segment` in `param.yaml`) must be consistent with the feature dimension expected by `new_diff.py`.
- The `generated_embeddings.h5` from `new_diff.py` will have `N_selected_residues` as its second dimension. The `override_residues` list in `param.yaml` for Stage 3 must match these selected residues and their order.
- The `special_res_file` HDF5 must have an embedding dimension that matches what the `ProteinStateReconstructor2D` expects after its internal pooling (i.e., `pooled_dim_per_segment` if `use_second_level_pooling` is false, or `final_effective_pooled_dim` if true).

**`special_res_file` and `override_residues`**: These must be correctly configured in `param.yaml` for Stage 3 to correctly load and use the diffused latents. Ensure the file path is accurate and the list of residue indices aligns with the content of the HDF5 file from diffusion.

**Log Files**: Check script-specific log files for detailed error messages.

---

## 📜 License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

This means you are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
