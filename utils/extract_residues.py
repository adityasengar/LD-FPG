"""extract_residues.py
======================

End‑to‑end utility for
1.  extracting heavy‑atom data + dihedral angles from a PDB + trajectory
2.  optional frame sub‑sampling via uniform stride (``--fraction``)
3.  writing a heavy‑atom‑only PDB (``--pdb_out``)
4.  serialising a **detailed JSON** of per‑residue time‑series data
5.  **optionally** creating a *condensed* JSON with contiguous atom indices
   (``--condensed_out``) suitable for dihedral‑comparison utilities.

Dependencies
------------
* numpy
* MDAnalysis
* tqdm

Install with::

    pip install numpy mdanalysis tqdm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from tqdm import tqdm

try:
    import MDAnalysis as mda
except ImportError:  # pragma: no cover – user environment
    sys.exit("ERROR: MDAnalysis not installed.  pip install mdanalysis")

# -----------------------------------------------------------------------------
# Basic PDB parsing (minimal, self‑contained – no BioPython dependence)
# -----------------------------------------------------------------------------
class PDB:  # noqa: D101
    def __init__(self, pdb_path: str | Path):
        serial: List[int] = []
        name: List[str] = []
        resname: List[str] = []
        chain: List[str] = []
        resseq: List[int] = []
        coords: List[List[float]] = []
        element: List[str] = []

        with open(pdb_path, "r") as fh:
            for line in fh:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                serial.append(int(line[6:11]))
                name.append(line[12:16].strip())
                resname.append(line[17:20].strip())
                chain.append(line[21].strip())
                resseq.append(int(line[22:26]))
                coords.append(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )
                elem = line[76:78].strip()
                element.append(elem if elem else line[12:14].strip())

        self.serial = np.asarray(serial, dtype=int)
        self.name = np.asarray(name)
        self.resname = np.asarray(resname)
        self.chain = np.asarray(chain)
        self.resseq = np.asarray(resseq, dtype=int)
        self.coords = np.asarray(coords, dtype=float)
        self.element = np.asarray(element)

    # -------- helper ---------------------------------------------------------
    def residues(self) -> Dict[int, np.ndarray]:
        """Map residue number → atom indices (numpy array)."""
        res_map: Dict[int, List[int]] = {}
        for idx, res_id in enumerate(self.resseq):
            res_map.setdefault(int(res_id), []).append(idx)
        return {k: np.asarray(v, dtype=int) for k, v in res_map.items()}


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _dihedral(coords: np.ndarray) -> float:
    """Return dihedral angle in **radians** for a (4, 3) coordinate array."""
    b1, b2, b3 = coords[1] - coords[0], coords[2] - coords[1], coords[3] - coords[2]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(np.arctan2(y, x))


def dihedral_deg(coords: np.ndarray) -> float:
    return np.degrees(_dihedral(coords))


# -----------------------------------------------------------------------------
# χ‑angle templates
# -----------------------------------------------------------------------------
_CHI_TEMPL = {
    "ARG": {
        "chi1": "CG",
        "chi2": ["CA", "CB", "CG", "CD"],
        "chi3": ["CB", "CG", "CD", "NE"],
        "chi4": ["CG", "CD", "NE", "CZ"],
        "chi5": ["CD", "NE", "CZ", "NH1"],
    },
    "ASN": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "OD1"]},
    "ASP": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "OD1"]},
    "CYS": {"chi1": "SG"},
    "GLN": {
        "chi1": "CG",
        "chi2": ["CA", "CB", "CG", "CD"],
        "chi3": ["CB", "CG", "CD", "OE1"],
    },
    "GLU": {
        "chi1": "CG",
        "chi2": ["CA", "CB", "CG", "CD"],
        "chi3": ["CB", "CG", "CD", "OE1"],
    },
    "HIS": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "ND1"]},
    "ILE": {"chi1": "CG1", "chi2": ["CA", "CB", "CG1", "CD1"]},
    "LEU": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "CD1"]},
    "LYS": {
        "chi1": "CG",
        "chi2": ["CA", "CB", "CG", "CD"],
        "chi3": ["CB", "CG", "CD", "CE"],
        "chi4": ["CG", "CD", "CE", "NZ"],
    },
    "MET": {
        "chi1": "CG",
        "chi2": ["CA", "CB", "CG", "SD"],
        "chi3": ["CB", "CG", "SD", "CE"],
    },
    "PHE": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "CD1"]},
    "PRO": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "CD"]},
    "SER": {"chi1": "OG"},
    "THR": {"chi1": "OG1"},
    "TRP": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "CD1"]},
    "TYR": {"chi1": "CG", "chi2": ["CA", "CB", "CG", "CD1"]},
    "VAL": {"chi1": "CG1"},
}


# -----------------------------------------------------------------------------
# Residue container
# -----------------------------------------------------------------------------
class ResidueData:  # noqa: D101
    def __init__(self, res_id: int, resname: str, pdb: PDB, atom_indices: np.ndarray):
        self.res_id = res_id
        self.resname = resname
        self.atom_indices = atom_indices  # heavy atoms only
        # boolean mask determines backbone vs side‑chain
        bb_mask = np.isin(pdb.name[atom_indices], ["N", "CA", "C", "O", "OXT"])
        self.backbone_idx = atom_indices[bb_mask]
        self.sidechain_idx = atom_indices[~bb_mask]
        # torsion templates (indices in original numbering, may contain None)
        self.torsion = {"phi": None, "psi": None, "chi": {}}

    # ------- setters ---------------------------------------------------------
    def set_phi(self, idx: Sequence[int]):
        self.torsion["phi"] = list(idx)

    def set_psi(self, idx: Sequence[int]):
        self.torsion["psi"] = list(idx)

    def set_chi(self, name: str, idx: Sequence[int]):
        self.torsion["chi"][name] = list(idx)


# -----------------------------------------------------------------------------
# Helpers to find named atoms inside a residue
# -----------------------------------------------------------------------------

def _find_atom(pdb: PDB, res_indices: np.ndarray, target: str | Sequence[str]):
    """Return **first** atom index matching *target* (name string or list)."""
    if isinstance(target, str):
        target = [target]
    mask = np.isin(pdb.name[res_indices], target)  # type: ignore[arg-type]
    return int(res_indices[mask][0]) if mask.any() else None


# -----------------------------------------------------------------------------
# Main extraction routine
# -----------------------------------------------------------------------------

def extract(pdb_path: str | Path, traj_path: str | Path, fraction: float):
    pdb = PDB(pdb_path)
    res_map = pdb.residues()

    # ---------- build per‑residue metadata ----------------------------------
    residues: Dict[int, ResidueData] = {}
    for res_id, atom_idx in res_map.items():
        # heavy atoms = non‑hydrogen
        heavy_idx = atom_idx[pdb.element[atom_idx] != "H"]
        res = ResidueData(res_id, pdb.resname[atom_idx][0], pdb, heavy_idx)
        residues[res_id] = res

    # ---------- backbone torsions (φ/ψ) -------------------------------------
    for res_id, res in residues.items():
        # φ  = C(i‑1)‑N‑CA‑C
        if (res_id - 1) in residues:
            c_prev = _find_atom(pdb, residues[res_id - 1].atom_indices, "C")
            n_i = _find_atom(pdb, res.atom_indices, "N")
            ca_i = _find_atom(pdb, res.atom_indices, "CA")
            c_i = _find_atom(pdb, res.atom_indices, "C")
            if None not in (c_prev, n_i, ca_i, c_i):
                res.set_phi([c_prev, n_i, ca_i, c_i])
        # ψ  = N‑CA‑C‑N(i+1)
        if (res_id + 1) in residues:
            n_i = _find_atom(pdb, res.atom_indices, "N")
            ca_i = _find_atom(pdb, res.atom_indices, "CA")
            c_i = _find_atom(pdb, res.atom_indices, "C")
            n_next = _find_atom(pdb, residues[res_id + 1].atom_indices, "N")
            if None not in (n_i, ca_i, c_i, n_next):
                res.set_psi([n_i, ca_i, c_i, n_next])

    # ---------- side‑chain χ torsions ---------------------------------------
    for res in residues.values():
        templ = _CHI_TEMPL.get(res.resname, {})
        if not templ:
            continue
        n = _find_atom(pdb, res.atom_indices, "N")
        ca = _find_atom(pdb, res.atom_indices, "CA")
        cb = _find_atom(pdb, res.atom_indices, "CB")
        # χ1
        x1 = _find_atom(pdb, res.atom_indices, templ["chi1"])
        if None not in (n, ca, cb, x1):
            res.set_chi("chi1", [n, ca, cb, x1])
        # higher χn (template gives explicit list of 4 atoms)
        for key in ("chi2", "chi3", "chi4", "chi5"):
            if key not in templ:
                continue
            idx = [_find_atom(pdb, res.atom_indices, name) for name in templ[key]]
            if None not in idx:
                res.set_chi(key, idx)  # type: ignore[arg-type]

    # ---------- trajectory processing ---------------------------------------
    u = mda.Universe(pdb_path, traj_path)
    stride = max(int(round(1.0 / fraction)), 1)
    selected = set(range(0, len(u.trajectory), stride))

    # container: res_id → list[frame][coords]
    coords_store: Dict[int, List[np.ndarray]] = {rid: [] for rid in residues}

    for ts in tqdm(u.trajectory, desc="Frames", total=len(selected)):
        if ts.frame not in selected:
            continue
        xyz = ts.positions  # (n_atoms, 3)
        for rid, res in residues.items():
            coords_store[rid].append(xyz[res.atom_indices])

    # ---------- JSON serialisation -----------------------------------------
    detailed_json: Dict[str, dict] = {}
    for rid, res in residues.items():
        detailed_json[str(rid)] = {
            "resname": res.resname,
            "heavy_atom_indices": res.atom_indices.tolist(),
            "heavy_atom_coords_per_frame": [c.tolist() for c in coords_store[rid]],
            "heavy_atoms": [  # metadata per atom
                {
                    "index": int(idx),
                    "name": str(pdb.name[idx]),
                    "is_backbone": bool(idx in res.backbone_idx),
                }
                for idx in res.atom_indices
            ],
            "torsion_atoms": res.torsion,
        }
    return pdb, detailed_json


# -----------------------------------------------------------------------------
# Heavy‑atom PDB writer (preserve serial numbers)
# -----------------------------------------------------------------------------

def write_heavy_pdb(pdb: PDB, path: str | Path):
    with open(path, "w") as fh:
        fh.write("HEADER    HEAVY ATOMS ONLY\n")
        for i in range(len(pdb.serial)):
            if pdb.element[i] == "H":
                continue
            fh.write(
                f"ATOM  {pdb.serial[i]:5d} {pdb.name[i]:>4} {pdb.resname[i]:>3} "
                f"{pdb.chain[i]:>1}{pdb.resseq[i]:4d}    "
                f"{pdb.coords[i,0]:8.3f}{pdb.coords[i,1]:8.3f}{pdb.coords[i,2]:8.3f}"
                f"  1.00  0.00           {pdb.element[i]:>2}\n"
            )
    print(f"[✓] Heavy‑atom PDB written → {path}")


# -----------------------------------------------------------------------------
# Condensing functions (old‑>new remap)
# -----------------------------------------------------------------------------

def _build_old2new(original: dict) -> Dict[int, int]:
    unique = sorted(
        {
            idx for res in original.values() for idx in res["heavy_atom_indices"]
        }
    )
    return {old: new for new, old in enumerate(unique)}


def _condense(original: dict) -> dict:
    old2new = _build_old2new(original)
    condensed: Dict[str, dict] = {}
    for new_rid, old_rid_str in enumerate(sorted(original, key=int)):
        res = original[old_rid_str]
        bb_old, sc_old = [], []
        for atom in res["heavy_atoms"]:
            (bb_old if atom["is_backbone"] else sc_old).append(atom["index"])
        bb_new = [old2new[i] for i in bb_old]
        sc_new = [old2new[i] for i in sc_old]

        tors_new = {"phi": None, "psi": None, "chi": {}}
        t_old = res.get("torsion_atoms", {})
        for key in ("phi", "psi"):
            old_q = t_old.get(key)
            tors_new[key] = [old2new[i] for i in old_q] if old_q and None not in old_q else None
        for chi_name, chi_atoms in (t_old.get("chi", {}) or {}).items():
            tors_new["chi"][chi_name] = (
                [old2new[i] for i in chi_atoms] if chi_atoms and None not in chi_atoms else None
            )

        condensed[str(new_rid)] = {
            "new_res_id": new_rid,
            "old_res_id": int(old_rid_str),
            "resname": res["resname"],
            "backbone": bb_new,
            "sidechain": sc_new,
            "torsion_atoms": tors_new,
        }
    return condensed


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def cli():  # noqa: D401
    """Parse args & run extraction."""
    p = argparse.ArgumentParser(description="Extract heavy atoms + dihedrals")
    p.add_argument("--pdb", required=True, help="Topology PDB file")
    p.add_argument("--traj", required=True, help="Trajectory (XTC/DCD)")
    p.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of frames to keep (uniform stride)",
    )
    p.add_argument("--json_out", default="residues_data.json", help="Detailed JSON")
    p.add_argument(
        "--pdb_out", default="heavy_chain.pdb", help="Heavy‑atom PDB filename"
    )
    p.add_argument(
        "--condensed_out",
        default=None,
        help="Write condensed JSON with contiguous atom indices",
    )
    args = p.parse_args()

    if not (0 < args.fraction <= 1):
        p.error("--fraction must be > 0 and ≤ 1")

    pdb, detailed = extract(args.pdb, args.traj, args.fraction)

    # write files
    write_heavy_pdb(pdb, args.pdb_out)
    Path(args.json_out).write_text(json.dumps(detailed, indent=2))
    print(f"[✓] Detailed JSON written → {args.json_out}")

    if args.condensed_out:
        condensed = _condense(detailed)
        Path(args.condensed_out).write_text(json.dumps(condensed, indent=2))
        print(f"[✓] Condensed JSON written → {args.condensed_out}")


if __name__ == "__main__":
    cli()
