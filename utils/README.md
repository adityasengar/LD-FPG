# Protein Structure Analysis & Utility Toolkit 🔬📊

## Overview

This repository contains a collection of Python scripts designed for in-depth analysis, comparison, and format conversion of protein structure data. These tools are particularly useful for processing outputs from Molecular Dynamics (MD) simulations or generative models like LD-FPG. The toolkit allows for:

* Detailed dihedral angle distribution analysis and comparison across multiple datasets.
* Calculation of structural similarity scores (lDDT, TM-score).
* Extraction of coordinate and dihedral data from trajectories.
* Conversion between various file formats (HDF5, PDB, XTC, NPY, JSON).

## Scripts Overview

Here's a brief summary of each script:

* **extract_residues.py**: **NEW v2**: Processes PDB and MD trajectory files (DCD, XTC) to extract per-residue heavy atom coordinates and calculate dihedral angles over time, saving to a detailed JSON format. Now includes optional frame sub-sampling (`--fraction`), automatic export of a heavy-atom-only PDB (`heavy_chain.pdb`), and **optionally writes a condensed JSON** with contiguous atom indices (`--condensed_out`) suitable for direct input to comparison scripts.
* **condense_residues.py**: (legacy/optional) Takes the JSON output from older versions of extract_residues.py and creates a "condensed" JSON file with remapped, contiguous 0-based atom indices. **Not required if you use `--condensed_out` in extract_residues.py v2**.
* **CompareDihedrals_csv.py**: Performs comprehensive dihedral angle analysis by comparing three HDF5 coordinate ensembles. It generates global and per-residue dihedral distribution plots (scatter plots, 1D/2D histograms, heatmaps), calculates various statistical metrics (KL divergence, JS divergence, Wasserstein distance) for pairwise comparisons, and **exports data for every generated plot into corresponding CSV files**. It also compiles plots into a summary PDF.
* **CompareDihedrals.py**: Similar to CompareDihedrals_csv.py, this script also performs dihedral analysis comparing three HDF5 files, including global plots, per-residue KL analysis, and calculation of 1D/2D metrics. It's recommended to use CompareDihedrals_csv.py for the most comprehensive output including per-plot CSV data.
* **calc_lddt.py**: Calculates Local Distance Difference Test (lDDT) scores between a set of predicted structures (from an HDF5 file) and a reference structure. Supports backbone-only calculations and random subsampling of structures.
* **calc_tm.py**: Calculates TM-scores between predicted structures (HDF5) and a reference structure. Also supports backbone-only mode and subsampling.
* **h5_to_pdb.py**: Converts ensembles of protein structures stored in HDF5 format into individual PDB files, using a template PDB for atom information.
* **json_to_pdb.py**: Extracts the first frame of heavy atom data from the JSON output of extract_residues.py and writes it as a PDB file and a simple .dat file (listing atom information and coordinates).
* **npy_to_pdb.py**: Converts coordinate data stored in a NumPy .npy file into multiple PDB files, using a template PDB.
* **npy_to_xtc.py**: Converts coordinate data from a .npy file into an XTC trajectory file, using a PDB file for topology information.
* **pdbs_to_xtc.py**: Creates a single XTC trajectory file from a numerically sorted series of individual PDB files.

---

## ⚙️ General Prerequisites

* **Python 3.x**
* **Core Libraries:**
    * NumPy
    * SciPy (for 1D Wasserstein distance)
    * h5py (for HDF5 file I/O)
    * PyTorch
    * MDAnalysis (for extract_residues.py, npy_to_xtc.py)
    * Matplotlib (for plotting)
    * Seaborn (for enhanced plotting)
    * img2pdf (for compiling plots into PDF)
    * POT (Python Optimal Transport) library (optional, for 2D Wasserstein distance in CompareDihedrals_csv.py and CompareDihedrals.py). Install with pip install pot.
    
```bash
# Example installation
pip install numpy scipy h5py torch mdanalysis matplotlib seaborn img2pdf
pip install pot  # Optional, for 2D Wasserstein
```

* **Input Data:** Specific scripts will require input data such as PDB files, trajectory files (DCD, XTC), HDF5 files with coordinates, JSON files from other scripts in this toolkit, or NumPy arrays.

---

## 🛠️ Detailed Script Descriptions and Usage

### 1. extract_residues.py

* **Purpose:** Processes PDB and MD trajectory files (DCD/XTC) to extract per-residue heavy atom coordinates and dihedral angles over time, saving to a detailed JSON format. **v2** now also:
    * Supports **optional frame sub-sampling** (`--fraction`)
    * Automatically exports a **heavy-atom-only PDB** (`heavy_chain.pdb`)
    * **Optionally writes a condensed JSON** with contiguous atom indices (`--condensed_out`), suitable for direct input to comparison scripts

* **Key Features:**
    * **Frame sub-sampling**: `--fraction <F>` keeps roughly F × 100% of frames (uniform stride).
    * **Automatic residue detection**: No hard-coded residue range.
    * **Heavy-atom PDB export**: Outputs a clean heavy-only PDB for use in further analysis.
    * **Condensed output**: Add `--condensed_out condensed.json` to create a compact, contiguous-index JSON (previously required a separate script).
    * **Side-chain χ-angles**: χ1-χ5 calculated when possible.
    * **Flexible output names**: `--json_out`, `--pdb_out`, `--condensed_out`.

* **Inputs:**
    * `--pdb`: PDB file (topology).
    * `--traj`: Trajectory file (.xtc, .dcd, etc.).
    * `--fraction` (optional, default 1.0): Fraction of frames to keep.
    * `--json_out` (optional): Output JSON file (default: residues_data.json).
    * `--pdb_out` (optional): Output heavy atom PDB (default: heavy_chain.pdb).
    * `--condensed_out` (optional): Output condensed JSON (for comparison/analysis).

* **Outputs:**
    * `heavy_chain.pdb` — heavy-atom-only PDB
    * `residues_data.json` — full time-series JSON with per-frame heavy atom coords and torsions
    * `condensed.json` (optional) — **compact, remapped atom index JSON** (for dihedral comparison)

* **Usage Example:**
    ```bash
    python extract_residues.py \
        --pdb prot.pdb \
        --traj traj.xtc \
        --fraction 0.05 \
        --json_out residues_full.json \
        --pdb_out heavy_chain.pdb \
        --condensed_out condensed.json
    ```
    **Note:** You no longer need `condense_residues.py` for new workflows—just use the `--condensed_out` flag!

### 2. condense_residues.py (legacy/optional)

* **Purpose:** For legacy workflows, converts JSON from older versions of extract_residues.py into a "condensed" JSON file with remapped, contiguous 0-based atom indices. **Not required if you use `--condensed_out` in extract_residues.py v2**.

* **Key Inputs:**
    * `input_json`: Path to JSON file generated by older versions of extract_residues.py.
    * `output_json`: Path for the condensed JSON file.

* **Key Output:**
    * Condensed JSON file with remapped, contiguous 0-based atom indices.

* **Usage Example:**
    ```bash
    python condense_residues.py residues_data_active_full.json condensed_residues.json
    ```

### 3. CompareDihedrals_csv.py (Recommended for Dihedral Analysis)

* **Purpose:** Provides a comprehensive comparison of dihedral angle distributions from three different structural ensembles (provided as HDF5 files containing coordinates). It performs:
    * Global dihedral distribution plotting (individual and overlaid scatter plots, 1D histograms for φ, ψ, χn, and 2D Ramachandran plots).
    * Pairwise per-residue Kullback-Leibler (KL) divergence analysis, visualized as heatmaps and summary bar plots.
    * Calculation of additional 1D metrics (JS divergence, Wasserstein distance) and 2D Ramachandran metrics (KL, JS, and optional 2D Wasserstein if POT is installed) for global pairwise comparisons.
    * **Crucially, exports the data underlying every generated plot into a corresponding CSV file**, facilitating further analysis and re-plotting.
    * Compiles plots into summary PDFs.

* **Key Inputs:**
    * `--condensed_json`: Path to the condensed_residues.json file (output of condense_residues.py).
    * `--h5_1`, `--h5_2`, `--h5_3`: Paths to three HDF5 files containing coordinate ensembles (e.g., ground truth, model A predictions, model B predictions). Each HDF5 file should contain a dataset of coordinates with shape (N_frames, N_atoms, 3).
    * `--labels`: Three labels for the HDF5 datasets, used in plot legends and filenames.
    * `--out_dir`: Output directory for results.
    * `--device` (optional): Use 'cuda' if available for faster processing.

* **Key Outputs (within `--out_dir`):**
    * Subdirectories for global comparisons (global_compare/individual/, global_compare/combined/) containing PNG plots and their associated CSV data files.
    * A compiled PDF (global_compare/Global_Distributions.pdf).
    * Subdirectories for each pairwise comparison (e.g., KL_Set1_vs_Set2/) containing:
        * Per-residue KL divergence heatmaps (PNGs + CSVs).
        * Summary plots of top differing residues/angles (PNGs + CSVs).
        * Detailed overlay plots for top differing residues (PNGs + CSVs).
        * A CSV file (kl_data.csv) with the full per-residue, per-angle KL matrix.
        * A compiled PDF for the pairwise comparison (e.g., KL_Set1_vs_Set2.pdf).
    * CSV files (Global_Metrics_LabelA_vs_LabelB.csv) summarizing global 1D and 2D dihedral metrics for each pair.

* **Usage Example:**
    ```bash
    python CompareDihedrals_csv.py \
        --condensed_json helper/condensed_residues.json \
        --h5_1 structures/ground_truth.h5 \
        --h5_2 structures/model_A_coords.h5 \
        --h5_3 structures/model_B_coords.h5 \
        --labels "GroundTruth" "ModelA" "ModelB" \
        --out_dir dihedral_analysis_results \
        --device cuda  # Optional: use 'cuda' if available
    ```

* **Note on CompareDihedrals.py**: This script offers similar analytical capabilities to CompareDihedrals_csv.py for plotting and metric calculation. However, CompareDihedrals_csv.py is generally recommended due to its explicit design for exporting CSV data alongside every generated plot, enhancing data accessibility and reproducibility.

### 4. calc_lddt.py

* **Purpose:** Calculates the Local Distance Difference Test (lDDT) score, a measure of local structural similarity, by comparing predicted structures to a reference structure.

* **Key Inputs:**
    * `--h5file`: Path to an HDF5 file containing predicted coordinates (shape (N_structures, N_atoms, 3) or (N_structures * N_atoms, 3) which will be reshaped).
    * `--xref`: Path to the reference structure coordinates (.npy or .pt file, shape (N_atoms, 3)).
    * `--key` (optional): Specific dataset key within the HDF5 file. If not provided, the script attempts to use the first key found.
    * `--backbone_only` (flag): If set, calculates lDDT on backbone atoms only. Requires `--pdb`.
    * `--pdb` (optional): Path to a PDB file used to define backbone atom indices if `--backbone_only` is active.
    * `--max_samples` (optional): If greater than 0, randomly samples this many structures from the HDF5 file for lDDT calculation.
    * `--seed` (optional): Random seed for reproducible sampling.

* **Key Output:**
    * Prints the mean and standard deviation of the lDDT scores calculated for the processed structures.

* **Usage Example (All-atom, sampling 100 models):**
    ```bash
    python calc_lddt.py \
        --h5file structures/generated_coords.h5 \
        --xref structures/X_ref_coords.pt \
        --max_samples 100 \
        --seed 42
    ```

* **Usage Example (Backbone-only):**
    ```bash
    python calc_lddt.py \
        --h5file structures/generated_coords.h5 \
        --xref structures/X_ref_coords.pt \
        --backbone_only \
        --pdb helper/heavy_chain.pdb
    ```

### 5. calc_tm.py

* **Purpose:** Calculates the TM-score, a measure of global structural similarity, comparing predicted structures to a reference.

* **Key Inputs:**
    * `--h5file`: Path to an HDF5 file containing predicted coordinates.
    * `--xref`: Path to the reference structure coordinates (.npy or .pt file).
    * `--key` (optional): Specific dataset key within the HDF5 file.
    * `--backbone_only` (flag): If set, calculates TM-score on backbone atoms only. Requires `--pdb`.
    * `--pdb` (optional): Path to a PDB file used to define backbone atom indices if `--backbone_only` is active.
    * `--max_samples` (optional): If greater than 0, randomly samples this many structures.
    * `--seed` (optional): Random seed for reproducible sampling.

* **Key Output:**
    * Prints the mean and standard deviation of the TM-scores.

* **Usage Example (All-atom):**
    ```bash
    python calc_tm.py \
        --h5file structures/generated_coords.h5 \
        --xref structures/X_ref_coords.pt
    ```

* **Usage Example (Backbone-only with sampling):**
    ```bash
    python calc_tm.py \
        --h5file structures/generated_coords.h5 \
        --xref structures/X_ref_coords.pt \
        --backbone_only \
        --pdb helper/heavy_chain.pdb \
        --max_samples 50
    ```

### 6. h5_to_pdb.py

* **Purpose:** Converts multiple protein structures stored in an HDF5 file into individual PDB files.

* **Key Inputs:**
    * `--hno_file` (optional): Path to HDF5 file from HNO reconstructions.
    * `--decoder2_file` (optional): Path to HDF5 file from Decoder2 reconstructions (expects specific keys like reconstructions_with_override).
    * `--pdb_file`: Path to a template PDB file whose atom information (names, residue info, etc.) will be used for formatting the output PDBs. The number of atoms in this template should match the structures in the HDF5.
    * `--output_dir`: Root directory where subfolders for PDB files will be created.
    * `--num_files`: Number of PDB files to generate from each input HDF5 dataset.

* **Key Output:**
    * PDB files saved in subdirectories within the specified `--output_dir`.

* **Usage Example:**
    ```bash
    python h5_to_pdb.py \
        --decoder2_file structures/full_coords_diff_exp1.h5 \
        --pdb_file helper/heavy_chain.pdb \
        --output_dir generated_pdbs \
        --num_files 50
    ```

### 7. json_to_pdb.py

* **Purpose:** Extracts the heavy atom coordinates from the **first frame** of a JSON file (typically one generated by extract_residues.py) and outputs them as a single PDB file. It also creates a simple .dat file listing residue number, residue name, atom index, atom name, and X, Y, Z coordinates.

* **Key Inputs:**
    * `json_path`: Path to the input JSON file (e.g., residues_data_active_full.json).
    * `dat_path`: Path for the output .dat file.
    * `pdb_path`: Path for the output .pdb file.

* **Key Outputs:**
    * A .dat file with tabular atom information.
    * A .pdb file containing the first frame coordinates.

* **Usage Example:**
    ```bash
    python json_to_pdb.py helper/residues_data_active_full.json output/first_frame.dat output/first_frame.pdb
    ```

### 8. npy_to_pdb.py

* **Purpose:** Converts protein structure coordinates stored in a NumPy .npy file into multiple individual PDB files. Assumes the .npy file contains a 2D array that can be reshaped into (N_structures, N_atoms, 3).

* **Key Inputs (modify hardcoded paths in script):**
    * `npy_file`: Path to the input .npy file.
    * `pdb_file`: Path to a template PDB file.
    * `output_directory`: Directory to save the generated PDB files.
    * `num_files`: Number of PDBs to generate.

* **Key Output:**
    * PDB files saved in the specified output directory.

* **Usage:** Modify the hardcoded paths in the script and run:
    ```bash
    python npy_to_pdb.py
    ```

### 9. npy_to_xtc.py

* **Purpose:** Converts protein structure coordinates from a NumPy .npy file into an XTC trajectory file, using a PDB file to provide the necessary topology information.

* **Key Inputs (modify hardcoded paths in script):**
    * `npy_file`: Path to the input .npy file (reshapable to (N_frames, N_atoms, 3)).
    * `pdb_file`: Path to the PDB topology file.
    * `output_xtc`: Path for the output XTC file.

* **Key Output:**
    * An .xtc trajectory file.

* **Usage:** Modify the hardcoded paths in the script and run:
    ```bash
    python npy_to_xtc.py
    ```

### 10. pdbs_to_xtc.py

* **Purpose:** Combines a series of individual PDB files (each representing one frame) into a single, continuous XTC trajectory file. It automatically sorts the input files numerically to ensure the correct frame order.

* **Key Inputs:**
    * `-p, --pattern`: The input PDB file pattern (e.g., 'generated_*.pdb'). **Use quotes** to ensure the pattern is passed correctly.
    * `-o, --output`: Path for the output XTC file (e.g., 'full_trajectory.xtc').

* **Key Output:**
    * An .xtc trajectory file.

* **Usage Example:**
    ```bash
    python pdbs_to_xtc.py --pattern "path/to/your_pdbs/generated_*.pdb" --output "trajectory.xtc"
    ```

* **Note:** The script assumes that all input PDB files contain the exact same atoms in the same order.

---

## 💡 Workflow Examples

1. **Detailed Dihedral Analysis of MD Trajectories:**
    * Use extract_residues.py to process your PDB and DCD/XTC trajectory into a detailed JSON. Use `--condensed_out` to also generate the condensed JSON directly.
    * (Or, if using extract_residues.py v2, simply use the `--condensed_out` option to generate this directly instead of running condense_residues.py separately.)
    * If you have multiple trajectories (e.g., wild-type vs. mutant, or different simulation conditions) that you've processed into HDF5 coordinate files, you can then use CompareDihedrals_csv.py with the condensed JSON to perform a thorough comparison.

2. **Evaluating Generated Structures (from LD-FPG or other models):**
    * Assume your generative model produces an HDF5 file of coordinates (e.g., generated_structures.h5).
    * Use calc_lddt.py and calc_tm.py to compare these generated structures against a native/reference structure (X_ref_coords.pt or .npy).
    * Use h5_to_pdb.py to convert a subset of these generated HDF5 structures into PDB format for visualization or further analysis with other tools.
    * Use CompareDihedrals_csv.py to compare the dihedral distributions of your generated ensemble against a ground truth MD ensemble (both in HDF5 coordinate format) and a set of experimentally determined structures (if available and converted to HDF5).
    * Note: If using extract_residues.py v2 with `--condensed_out`, you can reference the condensed JSON output directly in CompareDihedrals_csv.py.

---

## 📝 Notes

* **Data Formats:** Pay close attention to the expected input and output data formats for each script, especially the shapes of coordinate arrays in HDF5 files and the structure of JSON files.
* **File Paths:** Many scripts use command-line arguments for file paths. Ensure these are correct. Some utility scripts (npy_to_pdb.py, npy_to_xtc.py) have hardcoded paths that you'll need to modify directly in the script.
* **Dependencies:** Ensure all required libraries are installed. MDAnalysis is a key dependency for trajectory processing, and POT is optional but enhances CompareDihedrals_csv.py.

---

## 📜 License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

This means you are free to:
* **Share** — copy and redistribute the material in any medium or format
* **Adapt** — remix, transform, and build upon the material

for any purpose, even commercially.

Under the following terms:
* **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
