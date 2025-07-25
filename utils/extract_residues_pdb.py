#!/usr/bin/env python3
"""
Heavy‑atom extractor and torsion‑angle calculator

Usage:
    python extract_residues.py \
        --pdb system.pdb \
        --traj system.xtc \
        --fraction 0.20      # keep 20 % of the frames
"""

import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

try:
    import MDAnalysis as mda
except ImportError:
    print("Please install MDAnalysis (e.g., pip install MDAnalysis)")
    sys.exit(1)



# ----------------------------------------------------------------------
# Helper utilities: residue ↔ atom‑index maps and χ‑angle templates
# ----------------------------------------------------------------------
def build_resseq_to_indices(pdb):
    """Return {resseq: numpy‑array(atom_indices)} for all residues."""
    mapping = {}
    for idx, resseq in enumerate(pdb.resseq):
        mapping.setdefault(resseq, []).append(idx)
    return {k: np.asarray(v, dtype=int) for k, v in mapping.items()}


def select_atom(pdb, indices, atom_name):
    """First index in *indices* whose atom name matches *atom_name*."""
    mask = pdb.name[indices] == atom_name
    return int(indices[mask][0]) if mask.any() else None


def get_phi_atoms(pdb, resmap, resnum):
    if (resnum - 1) in resmap:
        C_prev = select_atom(pdb, resmap[resnum - 1], "C")
        N_i = select_atom(pdb, resmap[resnum], "N")
        CA_i = select_atom(pdb, resmap[resnum], "CA")
        C_i = select_atom(pdb, resmap[resnum], "C")
        if None not in (C_prev, N_i, CA_i, C_i):
            return [C_prev, N_i, CA_i, C_i]
    return None


def get_psi_atoms(pdb, resmap, resnum):
    if (resnum + 1) in resmap:
        N_i = select_atom(pdb, resmap[resnum], "N")
        CA_i = select_atom(pdb, resmap[resnum], "CA")
        C_i = select_atom(pdb, resmap[resnum], "C")
        N_next = select_atom(pdb, resmap[resnum + 1], "N")
        if None not in (N_i, CA_i, C_i, N_next):
            return [N_i, CA_i, C_i, N_next]
    return None


def get_side_chain_atoms(resname):
    """Lookup table for χ‑angle definitions (truncated to common residues)."""
    return {
	"ARG": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "CD"]},
        "ASN": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "OD1"]},
        "ASP": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "OD1"]},
        "CYS": {"chi1": "SG"},
        "GLN": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "CD"]},
        "GLU": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "CD"]},
        "HIS": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "ND1"]},
        "ILE": {"chi1": "CG1", "chi2": ["CA", "CB", "CG1", "CD1"]},
        "LEU": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "CD1"]},
        "LYS": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "CD"]},
        "MET": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "SD"]},
        "PHE": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "CD1"]},
        "PRO": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "CD"]},
        "SER": {"chi1": "OG"},
        "THR": {"chi1": "OG1"},
        "TRP": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "CD1"]},
        "TYR": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "CD1"]},
        "VAL": {"chi1": "CG1"},
    }.get(resname, None)


def get_chi_atoms(pdb, resname, res_indices, higher_order="all"):
    templ = get_side_chain_atoms(resname)
    if templ is None:
        return {}
    chi = {}

    # Chi1 is always defined by N‑CA‑CB‑X
    N = select_atom(pdb, res_indices, "N")
    CA = select_atom(pdb, res_indices, "CA")
    CB = select_atom(pdb, res_indices, "CB")
    X1 = select_atom(pdb, res_indices, templ["chi1"])
    chi["chi1"] = [N, CA, CB, X1] if None not in (N, CA, CB, X1) else None

    if higher_order != "all":
        return chi

    # Higher χ angles (χ2‑χ5) if template lists them
    for k in ("chi2", "chi3", "chi4", "chi5"):
        if k in templ:
            idx = [select_atom(pdb, res_indices, a) for a in templ[k]]
            chi[k] = idx if None not in idx else None
    return chi


# ----------------------------------------------------------------------
# Basic PDB parsing
# ----------------------------------------------------------------------
class PDB:
    def __init__(self, pdb_file: str):
        serial, name, resname, chainID, resseq, x, y, z, element = (
            [], [], [], [], [], [], [], [], []
        )
	with open(pdb_file, "r") as fh:
            for line in fh:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                serial.append(int(line[6:11]))
                name.append(line[12:16].strip())
                resname.append(line[17:20].strip())
                chainID.append(line[21].strip())
                resseq.append(int(line[22:26]))
                x.append(float(line[30:38]))
                y.append(float(line[38:46]))
                z.append(float(line[46:54]))
                elem = line[76:78].strip()
                element.append(elem if elem else line[12:14].strip())

        self.serial = np.array(serial)
        self.name = np.array(name)
        self.resname = np.array(resname)
        self.chainID = np.array(chainID)
        self.resseq = np.array(resseq)
        self.coords = np.column_stack((x, y, z))
        self.element = np.array(element)


def is_heavy(element: str) -> bool:
    return element != "H"


def write_heavy_chain_pdb(pdb_obj: PDB, out_path: str):
    """Write heavy atoms only, preserving original serial numbers."""
    with open(out_path, "w") as fh:
        fh.write("HEADER    HEAVY ATOMS ONLY\n")
        for i in range(len(pdb_obj.serial)):
            if not is_heavy(pdb_obj.element[i]):
                continue
            line = (
                f"ATOM  {pdb_obj.serial[i]:5d} {pdb_obj.name[i]:>4} "
                f"{pdb_obj.resname[i]:>3} {pdb_obj.chainID[i]:>1}"
                f"{pdb_obj.resseq[i]:4d}    "
                f"{pdb_obj.coords[i,0]:8.3f}{pdb_obj.coords[i,1]:8.3f}"
                f"{pdb_obj.coords[i,2]:8.3f}  1.00  0.00           "
                f"{pdb_obj.element[i]:>2}\n"
            )
            fh.write(line)
    print(f"Wrote heavy‑atom PDB → {out_path}")


# ----------------------------------------------------------------------
# Geometry helpers (unchanged from your original script)
# ----------------------------------------------------------------------
def calc_dihedral(coords):
    b1 = coords[:, 1, :] - coords[:, 0, :]
    b2 = coords[:, 2, :] - coords[:, 1, :]
    b3 = coords[:, 3, :] - coords[:, 2, :]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_norm = b2 / np.linalg.norm(b2, axis=1)[:, None]
    m1 = np.cross(n1, b2_norm)
    x = (n1 * n2).sum(axis=1)
    y = (m1 * n2).sum(axis=1)
    return np.arctan2(y, x)


def calc_dihedral_angles(coords):
    return calc_dihedral(coords) * 180.0 / np.pi


# ----------------------------------------------------------------------
# Residue container (simplified to heavy‑atom coords + dihedrals)
# ----------------------------------------------------------------------
class Residue:
    def __init__(self, res_num, resname, heavy_atom_indices, torsion_atoms):
        self.res_num = res_num
        self.resname = resname
        self.heavy_atom_indices = np.array(heavy_atom_indices)
        self.heavy_atom_coords_per_frame = []
        self.torsion_atoms = torsion_atoms
        self.dihedral_angles = {"phi": [], "psi": [], "chi": {}}

    def calculate_dihedrals(self, coords):
        for angle_name in ("phi", "psi"):
            atom_idx = self.torsion_atoms.get(angle_name)
            if atom_idx and None not in atom_idx:
                angle = calc_dihedral_angles(coords[atom_idx].reshape(1, 4, 3))[0]
                self.dihedral_angles[angle_name].append(float(angle))
            else:
                self.dihedral_angles[angle_name].append(None)

        for chi_name, atom_idx in self.torsion_atoms.get("chi", {}).items():
            if atom_idx and None not in atom_idx:
                angle = calc_dihedral_angles(coords[atom_idx].reshape(1, 4, 3))[0]
                self.dihedral_angles["chi"].setdefault(chi_name, []).append(float(angle))
            else:
                self.dihedral_angles["chi"].setdefault(chi_name, []).append(None)

        self.heavy_atom_coords_per_frame.append(
            coords[self.heavy_atom_indices].copy()
        )


# ----------------------------------------------------------------------
# Helpers that map residue → atom indices
# (functions build_resseq_to_indices, select_atom, get_phi_atoms, etc.)
# They are identical to your earlier code; paste them here unchanged.
# ----------------------------------------------------------------------
# ... (omitted for brevity, keep original implementations)


# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------
def process_trajectory(
    pdb_file: str,
    traj_file: str,
    fraction: float,
    higher_order: str = "all",
):
    pdb = PDB(pdb_file)
    resseq_to_indices = build_resseq_to_indices(pdb)
    residues = {}

    for res_num, res_indices in resseq_to_indices.items():
        resname = pdb.resname[res_indices[0]]
        heavy_idx = res_indices[~np.isin(pdb.element[res_indices], ["H"])]
        torsion_atoms = {
            "phi": get_phi_atoms(pdb, resseq_to_indices, res_num),
            "psi": get_psi_atoms(pdb, resseq_to_indices, res_num),
            "chi": get_chi_atoms(pdb, resname, res_indices, higher_order) or {},
        }
	residues[res_num] = Residue(
            res_num, resname, heavy_idx, torsion_atoms
        )

    # Load trajectory
    u = mda.Universe(pdb_file, traj_file)
    n_frames = len(u.trajectory)
    stride = max(int(round(1.0 / fraction)), 1)
    selected_frames = range(0, n_frames, stride)

    for ts in tqdm(u.trajectory, desc="Processing frames", total=len(selected_frames)):
        if ts.frame not in selected_frames:
            continue
        coords = ts.positions
        for res in residues.values():
            res.calculate_dihedrals(coords)

    return residues, pdb


def serialize_residues(residues):
    serial = {}
    for res in residues.values():
        serial[str(res.res_num)] = {
            "resname": res.resname,
            "heavy_atom_indices": res.heavy_atom_indices.tolist(),
            "heavy_atom_coords_per_frame": [
                c.tolist() for c in res.heavy_atom_coords_per_frame
            ],
            "dihedral_angles": res.dihedral_angles,
        }
    return serial


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True, help="Topology PDB file")
    parser.add_argument("--traj", required=True, help="Trajectory (xtc/dcd)")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of frames to export (uniform stride)",
    )
    parser.add_argument(
        "--json_out", default="residues_data.json", help="Output JSON filename"
    )
    parser.add_argument(
        "--pdb_out", default="heavy_chain.pdb", help="Output heavy‑atom PDB"
    )
    args = parser.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        parser.error("--fraction must be in (0, 1]")

    residues, pdb_obj = process_trajectory(
        pdb_file=args.pdb,
        traj_file=args.traj,
        fraction=args.fraction,
        higher_order="all",
    )

    # Heavy‑atom PDB
    write_heavy_chain_pdb(pdb_obj, args.pdb_out)

    # JSON
    with open(args.json_out, "w") as fh:
        json.dump(serialize_residues(residues), fh, indent=2)
    print(f"Wrote JSON → {args.json_out}")


if __name__ == "__main__":
    main()
