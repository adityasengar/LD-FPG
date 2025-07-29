# Multi-System Protein Conformation Generation Pipeline

This repository contains a suite of Python scripts that form a complete deep learning pipeline for analyzing, modeling, and generating novel 3D conformations of multiple protein systems.

The core workflow is a three-stage **Encode -> Generate -> Decode** process, followed by a final conversion step to produce PDB files for visualization.

## Core Components

The pipeline consists of four main scripts, each with its own YAML configuration file. They are designed to be run in sequence.

### 1. Encoder: `chebnet_multi_final.py`
- **Configuration:** `param_new.yaml`
- **Role:** This is the primary learning stage. The script trains a graph-based autoencoder on simulation data from multiple protein systems. It learns a shared, low-dimensional "language" of protein structure.
- **Inputs:**
    - PDB files for multiple protein structures.
    - JSON files containing coordinate trajectories for each system.
- **Key Outputs:**
    - `checkpoints/decoder2_checkpoint.pth`: The trained decoder model, essential for the final stage.
    - `latent_reps/pooled_embeddings.h5`: The compressed (latent) representation for every frame of the input simulations.
    - `latent_reps/reference_conditioners.h5`: The static latent representation (`z_ref`) for the reference structure of each system.

### 2. Generator: `diffusion_multi_system.py`
- **Configuration:** `param_diff.yaml`
- **Role:** This script trains a generative diffusion model. For each protein system, it learns the specific statistical distribution of its latent embeddings. It can then generate new, artificial embeddings from scratch that conform to this learned distribution.
- **Inputs:**
    - `latent_reps/pooled_embeddings.h5`: The latent vectors produced by the encoder.
- **Key Outputs:**
    - `diffusion_output/generated_embeddings_sys<ID>_exp<N>.h5`: HDF5 files containing new, artificially generated latent embeddings. It creates one file per experiment (`exp<N>`) for each system (`sys<ID>`).

### 3. Decoder: `chebnet_gen_multi.py`
- **Configuration:** `param_gen_multi.yaml`
- **Role:** This script translates the artificial latent embeddings from the diffusion model back into full 3D structures. It acts as the bridge between the abstract latent space and physical coordinates.
- **Inputs:**
    - `checkpoints/decoder2_checkpoint.pth`: The trained decoder model from Stage 1.
    - `latent_reps/reference_conditioners.h5`: The static `z_ref` for each system.
    - The `diffusion_output/` directory containing all generated embeddings.
- **Key Outputs:**
    - `generated_structures/multi_system_generated_coords.h5`: A single, organized HDF5 file containing the full 3D coordinates of the newly generated structures. The file has a nested structure to separate results by system and experiment (e.g., `system_0/exp_1/coords`).

### 4. Converter: `h5_to_pdb_multi.py`
- **Configuration:** `param_h5_to_pdb_multi.yaml`
- **Role:** The final utility step. This script converts the generated 3D coordinates from the HDF5 file into standard PDB files that can be viewed in molecular visualization software.
- **Inputs:**
    - `generated_structures/multi_system_generated_coords.h5`: The final coordinate file from Stage 3.
    - The original PDB files for each system, which are used as templates for atom/residue information.
- **Key Outputs:**
    - `generated_pdbs/`: A directory containing the final, viewable PDB files, organized into sub-directories by system and experiment.

## Workflow: How to Run the Pipeline

Execute the scripts in the following order. Ensure you have configured the corresponding YAML file before each step.

**Step 1: Train the Encoder**
```bash
python chebnet_multi_final.py --config param_new.yaml
```

**Step 2: Train Diffusion Models and Generate Latent Embeddings**
This step must be run **once for each system** you want to model. The `--system_id` flag specifies which system's data to use from the encoder's output.

```bash
# Run for System 0
python diffusion_multi_system.py --config param_diff.yaml --system_id 0

# Run for System 1
python diffusion_multi_system.py --config param_diff.yaml --system_id 1

# ...and so on for all other systems.
```

**Step 3: Decode Generated Embeddings into 3D Coordinates**
This script finds all the generated embeddings from the previous step and processes them in one go.
```bash
python chebnet_gen_multi.py --config param_gen_multi.yaml
```

**Step 4: Convert Final Coordinates to PDB Files**
This final step converts the HDF5 coordinate data into viewable PDB files.
```bash
python h5_to_pdb_multi.py --config param_h5_to_pdb_multi.yaml
```

After completing these steps, the `generated_pdbs/` directory will contain your final results.
