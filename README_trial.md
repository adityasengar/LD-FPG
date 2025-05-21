# Molecular Dynamics Simulation Data for D2R Project

This archive contains selected Molecular Dynamics (MD) simulation data for the human Dopamine D2 receptor (D2R), related to the work described in the paper "Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings."

This dataset provides raw and semi-processed files for the GROMACS simulations of specific replicas. The primary processed JSON file used as input for the ML pipeline described in the paper is provided in the `Processed_ML_Input_JSON/` directory.

## Archived Simulation Replicas:

The `MD_simulation_data/` directory contains data for the following independent simulation replicas:
- **run1/**
- **run6/**

These two runs are provided to allow users to explore simulation variability or specific effects. We have additional simulation replicas available; interested users can contact the corresponding authors for access.

## Directory Structure (for each run, e.g., MD_simulation_data/run1/):

- **input_files/**: Contains GROMACS input files for the simulation run.
  - `system_begin.pdb`: The PDB file of the full solvated system used as input for GROMACS (derived from `step5_input.pdb` in the original run directory).
  - `protein_initial.pdb`: The initial protein structure (e.g., `prot.pdb` from the run directory).
  - `topol.top`: Topology file.
  - `index.ndx`: Index file, if used.
  - `*.mdp`: MD parameter files for minimization, equilibration steps, and production. (Note: Please verify the exact name of the 'production.mdp' file provided, as it's based on a common naming convention).
  - `toppar/`: Contains custom force field parameters, if any were used for this specific run (otherwise, standard CHARMM36m was employed).
- **production_run/**: Contains files from the production MD phase.
  - `production_run.tpr`: GROMACS TPR (portable run input) file for the production simulation.
  - `traj_protein_noPBC.xtc`: Processed trajectory containing only protein heavy-atoms, with periodic boundary conditions removed and likely centered/fitted. This is suitable for many analyses of protein dynamics.

## Notes on Trajectory Data and ML Pipeline Input:

1.  **Full System Trajectory (`step7_1.xtc`):** The full production trajectory for each run, which includes membrane, solvent, and ions (typically named `step7_1.xtc` in the original run directories and can be very large, e.g., 14-15 GB), is **not included** in this archive due to its size. Users requiring this full trajectory for detailed environment analysis can contact the corresponding authors.
    * If you obtain the full `step7_1.xtc` and the `extract_residues.py` script (available from the authors), you would typically use the full system PDB provided here as `input_files/system_begin.pdb` as the reference for processing.

2.  **Processing Provided Trajectories for ML Input:**
    * The provided `production_run/traj_protein_noPBC.xtc` is a processed, protein-only trajectory.
    * To convert this (or a similar protein-only trajectory) into the JSON format used by our LD-FPG pipeline, one would typically use a processing script (like `process_trajectory_parallel.py`, available from authors) along with a reference PDB of the protein heavy atoms (often named `heavy_chain.pdb` in the project's context, also available from authors).

3.  **Main ML Input JSON:**
    * The primary processed JSON file (`my_protein.json` as referred to in the paper, e.g., `final_combined.json`) which was used to train the LD-FPG model **is provided in the `Processed_ML_Input_JSON/` directory** of this archive. (Note to uploader: Please ensure you manually copy the correct JSON file into this folder).

## Simulation Details Overview:

- Protein: Human Dopamine D2 receptor (D2R)
- Starting structure basis: PDB ID 6CM4, with ICL3 remodeled.
- Force Field: CHARMM36m (unless `toppar/` for a specific run indicates modifications).
- Software: GROMACS 2024.2
- General protocol: Each run involved energy minimization, a multi-step equilibration protocol, and a 2 µs production phase. The `traj_protein_noPBC.xtc` is derived from this production phase.

Please refer to the main paper for full methodological details.

(TODO: Add any other specific notes about these runs or the data if necessary)
