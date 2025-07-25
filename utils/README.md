# Protein Structure Analysis & Utility Toolkit 🔬📊

## Overview

This repository provides a set of Python utilities for analysing, comparing, and converting protein-structure data. The tools are optimised for datasets originating from molecular-dynamics (MD) simulations and generative models such as LD-FPG.

### Key Capabilities

- Per-residue heavy-atom extraction and dihedral analysis
- Structural similarity metrics (lDDT, TM-score)
- Multi-ensemble dihedral-distribution comparisons with rich visualisation and CSV export
- Format conversion between HDF5, PDB, XTC, NPY and JSON

## Script Summary

| Script | Purpose |
|--------|---------|
| `extract_residues.py` | **NEW v2**: Extract heavy atoms & dihedral angles from a trajectory with optional frame sub-sampling (`--fraction`) and automatic export of a heavy-atom-only PDB (`heavy_chain.pdb`) |
| `condense_residues.py` | Remap atom indices in the JSON from `extract_residues.py` to a contiguous 0-based scheme for downstream analysis |
| `CompareDihedrals_csv.py` | Comprehensive three-way dihedral comparison (plots + metrics + per-plot CSV) |
| `CompareDihedrals.py` | Legacy variant of the above (plots & metrics, no CSV) |
| `calc_lddt.py` | Compute lDDT between predicted structures (HDF5) and a reference |
| `calc_tm.py` | Compute TM-score between predicted structures and a reference |
| `h5_to_pdb.py` | Convert coordinate ensembles in HDF5 to individual PDBs |
| `json_to_pdb.py` | Convert the first frame of a detailed JSON to PDB + DAT |
| `npy_to_pdb.py` | Convert NumPy coordinate arrays to PDBs |
| `npy_to_xtc.py` | Convert NumPy coordinate arrays to an XTC trajectory |
| `pdbs_to_xtc.py` | Stitch a series of PDB files into a single XTC trajectory |

## Detailed Descriptions & Usage

### 1. extract_residues.py (version 2)

**Purpose**: Extract per-residue heavy-atom coordinates and backbone/side-chain dihedral angles from a PDB + trajectory pair, saving the result as structured JSON and writing a hydrogen-stripped PDB (`heavy_chain.pdb`).

#### Key Features

| Feature | Notes |
|---------|-------|
| Frame sub-sampling | `--fraction <F>` keeps roughly F × 100% of frames (uniform stride). `--fraction 1.0` ⇒ keep all frames |
| Automatic residue detection | All residues present in the PDB are processed (no hard-coded range) |
| Heavy-atom PDB export | A clean heavy-only topology (`heavy_chain.pdb`) is produced and can be reused by other scripts |
| Side-chain χ-angles | χ1-χ5 calculated when defined for the residue type |
| JSON output | Default `residues_data.json`; customizable via `--json_out` |

#### CLI Usage

```bash
python extract_residues.py \
    --pdb  system.pdb \
    --traj system.xtc \
    --fraction 0.20            # keep 20% of frames
    --json_out residues.json    # optional
    --pdb_out  heavy_chain.pdb  # optional
```

#### Outputs

- `heavy_chain.pdb` – heavy atoms only, original serial numbers preserved
- `residues.json` – hierarchical JSON with:
  - Heavy-atom indices & time-series coordinates
  - φ / ψ / χn dihedral lists synchronised with the selected frames

> **Tip**: A smaller JSON can be generated instantly by adjusting `--fraction`, e.g. `--fraction 0.05` (≈ every 20th frame).

### 2. condense_residues.py

**Purpose**: Convert the detailed JSON from `extract_residues.py` into a compact representation with contiguous atom indices. Essential for consistent input to `CompareDihedrals_csv.py` / `CompareDihedrals.py`.

```bash
python condense_residues.py residues.json condensed_residues.json
```

### 3. CompareDihedrals_csv.py (recommended)

Comprehensive three-way comparison of dihedral distributions with per-plot CSV export, KL/JS/Wasserstein metrics and publication-ready figures. See inline `--help` for the extensive list of options.

```bash
python CompareDihedrals_csv.py \
    --condensed_json condensed_residues.json \
    --h5_1 native.h5  --h5_2 model_A.h5  --h5_3 model_B.h5 \
    --labels Native ModelA ModelB \
    --out_dir dihedral_results
```

*Note: `CompareDihedrals.py` provides similar functionality without per-plot CSVs.*

### 4-10. Other Utilities

- `calc_lddt.py` – local similarity metric (supports backbone-only mode)
- `calc_tm.py` – global TM-score metric (supports backbone-only mode)
- `h5_to_pdb.py` – write PDB snapshots from an HDF5 coordinate ensemble
- `json_to_pdb.py` – dump the first JSON frame to PDB/DAT
- `npy_to_pdb.py` – write PDB snapshots from a NumPy array
- `npy_to_xtc.py` – turn a NumPy array into an XTC trajectory
- `pdbs_to_xtc.py` – merge a series of PDBs into an XTC trajectory

Each script has an inline `--help` with examples.

## Prerequisites

```bash
pip install numpy scipy h5py torch mdanalysis matplotlib seaborn img2pdf
# Optional (2D Wasserstein in dihedral comparison)
pip install pot
```

Python ≥ 3.8 is recommended.

## Example Workflow 🚀

1. **Extract heavy-atom data & dihedrals**
   ```bash
   python extract_residues.py --pdb prot.pdb --traj traj.xtc --fraction 0.1
   ```

2. **Condense indices**
   ```bash
   python condense_residues.py residues.json condensed.json
   ```

3. **Compare three coordinate ensembles**
   ```bash
   python CompareDihedrals_csv.py \
       --condensed_json condensed.json \
       --h5_1 truth.h5 --h5_2 modelA.h5 --h5_3 modelB.h5 \
       --labels Truth A B --out_dir analysis
   ```

4. **Quality metrics (optional)**
   ```bash
   python calc_lddt.py --h5file modelA.h5 --xref ref.npy
   python calc_tm.py   --h5file modelA.h5 --xref ref.npy
   ```

## License

Distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence.

**You are free to:**
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

**Under the following terms:**
- **Attribution** — You must give appropriate credit, provide a link to the licence, and indicate if changes were made
