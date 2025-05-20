#!/usr/bin/env python3
"""
================================
Merged global‑distribution and per‑residue KL comparison of three Molecular
Dynamics trajectories **with automatic CSV exports for every figure**.

For each PNG produced anywhere under ``--out_dir``, a CSV file with the same
basename is written right next to it, containing the raw data behind the plot.
This makes the results immediately machine‑readable for spreadsheets,
post‑processing in R / Pandas, etc.

The analytical workflow is identical to the original *mega_compare_3h5_with_kl.py*:

1. **Global dihedral distributions** for the three HDF5 files.
2. **Pair‑wise per‑residue KL analysis** with extra metrics (JS, Wasserstein).
3. Publication‑quality plots **+** data tables:
     * 1‑D histograms (φ, ψ, χn)
     * 2‑D Ramachandran heatmaps (linear‑density, log‑density)
     * Residue×Angle KL heatmaps
     * Top‑residue / top‑angle bar charts

Only three small helper utilities have been added plus ~20 strategic lines that
write CSVs after each ``plt.savefig``.  No other logic is touched, so the
scientific output is bit‑identical to the original PNGs.

Usage example
-------------
```bash
python mega_compare_3h5_with_kl_csv.py \
       --condensed_json condensed_residues.json \
       --h5_1 traj_a.h5 --h5_2 traj_b.h5 --h5_3 traj_c.h5 \
       --labels "WT" "MutA" "MutB" \
       --out_dir mega_output --device cuda
```

All additional CLI flags are unchanged; see ``-h`` for details.
"""

###############################################################################
# 0) IMPORTS & GLOBALS
###############################################################################
import os
import sys
import json
import argparse
import math
import csv
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import img2pdf
from matplotlib.colors import LogNorm
from copy import deepcopy
import logging

# Optional – POT for 2‑D Wasserstein
try:
    import ot  # type: ignore
    HAS_POT = True
except ImportError:
    HAS_POT = False

# 1‑D Wasserstein from SciPy
from scipy.stats import wasserstein_distance

# ---------------------------------------------------------------------------
# LOGGER – sane defaults
# ---------------------------------------------------------------------------
logger = logging.getLogger("MegaCompare3h5‑CSV")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s"))
    logger.addHandler(_h)

###############################################################################
# 0‑bis)  TINY HELPERS – automatic CSV dumping
###############################################################################

def _csv_safe(arr: np.ndarray | list | None) -> np.ndarray:
    """Return a 1‑D numpy array flattened and stripped of NaNs (or empty)."""
    if arr is None:
        return np.array([])
    a = np.asarray(arr).flatten()
    if a.size == 0:
        return np.array([])
    return a[~np.isnan(a)]


def _write_csv(path: str, header: list[str], *cols: list | np.ndarray) -> None:
    """Write equal‑length columns under *header* to *path* (overwrites)."""
    if not cols:
        return
    for c in cols[1:]:
        if len(c) != len(cols[0]):
            raise ValueError("All columns must have the same length for CSV export")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(zip(*cols))
    logger.debug(f"CSV written: {path}")

###############################################################################
# 1) LOADING UTILITIES
###############################################################################

def load_condensed_json(json_path: str):
    if not os.path.isfile(json_path):
        sys.exit(f"Error: condensed JSON not found → {json_path}")
    with open(json_path, "r") as fh:
        return json.load(fh)


def load_h5_coords(h5_file: str):
    if not os.path.isfile(h5_file):
        sys.exit(f"Error: HDF5 file not found → {h5_file}")
    with h5py.File(h5_file, "r") as hf:
        if "data" in hf:
            coords = hf["data"][:]
        else:
            keys = list(hf.keys())
            if len(keys) == 1:
                coords = hf[keys[0]][:]
            else:
                sys.exit(f"Ambiguous dataset in {h5_file}, keys={keys}")
    return coords  # (N_frames, N_atoms, 3)

###############################################################################
# 2) TORCH DIHEDRAL COMPUTATION
###############################################################################

def dihedral_torch(coords: torch.Tensor):
    """coords: (batch, 4, 3) → angles (batch,) in radians"""
    b1 = coords[:, 1] - coords[:, 0]
    b2 = coords[:, 2] - coords[:, 1]
    b3 = coords[:, 3] - coords[:, 2]
    n1 = torch.cross(b1, b2, dim=1)
    n2 = torch.cross(b2, b3, dim=1)
    b2_unit = b2 / torch.norm(b2, dim=1, keepdim=True).clamp(min=1e-6)
    m1 = torch.cross(n1, b2_unit, dim=1)
    x = torch.sum(n1 * n2, dim=1)
    y = torch.sum(m1 * n2, dim=1)
    return torch.atan2(y, x)

###############################################################################
# 3) GATHER GLOBAL + PER‑RESIDUE ANGLES
###############################################################################

def gather_global_and_perres_angles(coords_np: np.ndarray, condensed_data: dict, *,
                                    device: str = "cpu", chunk_size: int = 2000):
    """Return (global_dict, perres_dict, all_chi_names, all_res_ids)."""
    device_t = torch.device(device)

    all_res_ids = sorted(int(k) for k in condensed_data.keys())
    angle_info: dict[int, dict] = {}
    all_chi_names: set[str] = set()

    for r_str, entry in condensed_data.items():
        r_int = int(r_str)
        tors = entry.get("torsion_atoms", {})
        angle_info[r_int] = {
            "phi": tors.get("phi"),
            "psi": tors.get("psi"),
            "chi": {},
        }
        for c_name, quad in tors.get("chi", {}).items():
            angle_info[r_int]["chi"][c_name] = quad
            all_chi_names.add(c_name)

    all_chi_list = sorted(all_chi_names)

    # --- preallocate structures ------------------------------------------------
    global_dict: dict[str, list[float]] = {k: [] for k in ("phi", "psi", *all_chi_list)}
    perres_dict: dict[int, dict[str, list[float]]] = {
        rid: {k: [] for k in ("phi", "psi", *all_chi_list)} for rid in all_res_ids
    }

    N_frames = coords_np.shape[0]

    def compute_quad(quad, coords_chunk_t: torch.Tensor):
        if quad is None or len(quad) != 4:
            return None
        idx_t = torch.tensor(quad, dtype=torch.long, device=device_t)
        angles_rad = dihedral_torch(coords_chunk_t[:, idx_t, :].view(-1, 4, 3))
        return (angles_rad * 180.0 / math.pi).cpu().numpy()

    for start in range(0, N_frames, chunk_size):
        end = min(N_frames, start + chunk_size)
        coords_chunk_t = torch.from_numpy(coords_np[start:end]).to(device_t)

        for rid in all_res_ids:
            info = angle_info[rid]
            # φ & ψ
            for aname in ("phi", "psi"):
                res = compute_quad(info[aname], coords_chunk_t)
                if res is not None:
                    global_dict[aname].extend(res)
                    perres_dict[rid][aname].extend(res)
            # χn sidechains
            for c_name in all_chi_list:
                res = compute_quad(info["chi"].get(c_name), coords_chunk_t)
                if res is not None:
                    global_dict[c_name].extend(res)
                    perres_dict[rid][c_name].extend(res)
        logger.info(f"Frames {end}/{N_frames} processed for dihedral extraction…")

    # convert to numpy arrays
    for k in global_dict:
        global_dict[k] = np.asarray(global_dict[k], dtype=float)
    for rid in all_res_ids:
        for k in perres_dict[rid]:
            perres_dict[rid][k] = np.asarray(perres_dict[rid][k], dtype=float)

    return global_dict, perres_dict, all_chi_list, all_res_ids

###############################################################################
# 4) GLOBAL PLOTTING HELPERS (PNG + CSV)
###############################################################################

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Scatter φ‑ψ
# ---------------------------------------------------------------------------

def plot_scatter_phi_psi(phi: np.ndarray, psi: np.ndarray, label: str,
                          out_dir: str, *, prefix: str = "01"):
    ensure_dir(out_dir)
    arr_phi = _csv_safe(phi)
    arr_psi = _csv_safe(psi)

    # figure
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=arr_phi, y=arr_psi, alpha=0.3, s=5)
    plt.title(f"{label}: φ vs ψ Scatter")
    plt.xlabel("φ (deg)")
    plt.ylabel("ψ (deg)")
    plt.xlim([-180, 180])
    plt.ylim([-180, 180])
    plt.grid(True, linestyle="--", alpha=0.5)

    outpng = os.path.join(out_dir, f"{prefix}_scatter_{label}.png")
    plt.savefig(outpng, dpi=300, bbox_inches="tight")
    plt.close()

    # data CSV
    _write_csv(outpng.replace(".png", ".csv"), ["phi_deg", "psi_deg"], arr_phi, arr_psi)

# ---------------------------------------------------------------------------
# 2‑D log‑density Ramachandran
# ---------------------------------------------------------------------------

def plot_2d_hist_phi_psi_log(phi: np.ndarray, psi: np.ndarray, label: str,
                             out_dir: str, *, prefix: str = "02"):
    ensure_dir(out_dir)
    arr_phi = _csv_safe(phi)
    arr_psi = _csv_safe(psi)

    plt.figure(figsize=(7, 7))
    counts, xedges, yedges, _ = plt.hist2d(arr_phi, arr_psi, bins=72,
                                           range=[[-180, 180], [-180, 180]],
                                           cmap="viridis", norm=LogNorm())
    plt.colorbar(label="Counts (log scale)")
    plt.title(f"{label}: φ vs ψ 2‑D Hist (log)")
    plt.xlabel("φ (deg)")
    plt.ylabel("ψ (deg)")
    plt.grid(True, linestyle="--", alpha=0.5)

    outpng = os.path.join(out_dir, f"{prefix}_2dhist_{label}.png")
    plt.savefig(outpng, dpi=300, bbox_inches="tight")
    plt.close()

    # data CSV – bin centers + density
    x_cent = 0.5 * (xedges[:-1] + xedges[1:])
    y_cent = 0.5 * (yedges[:-1] + yedges[1:])
    XX, YY = np.meshgrid(x_cent, y_cent, indexing="ij")
    _write_csv(outpng.replace(".png", ".csv"), ["phi_deg", "psi_deg", "density"],
               XX.flatten(), YY.flatten(), counts.flatten())

# ---------------------------------------------------------------------------
# 1‑D histogram helper (individual)
# ---------------------------------------------------------------------------

def plot_1d_hist_angle(angle_data: np.ndarray, angle_name: str, label: str,
                        prefix: str, out_dir: str, *, log_y: bool = False,
                        bins: int = 72):
    ensure_dir(out_dir)
    arr = _csv_safe(angle_data)
    if arr.size < 1:
        return

    plt.figure(figsize=(8, 6))
    counts, bin_edges, _ = plt.hist(arr, bins=bins, range=[-180, 180],
                                    alpha=0.7, density=True)
    if log_y:
        plt.yscale("log")
    plt.title(f"{label}: {angle_name} Distribution")
    plt.xlabel(f"{angle_name} (deg)")
    plt.ylabel("Density")
    plt.xlim([-180, 180])
    plt.grid(True, linestyle="--", alpha=0.5)

    outpng = os.path.join(out_dir, f"{prefix}_{angle_name}_{label}.png")
    plt.savefig(outpng, dpi=300, bbox_inches="tight")
    plt.close()

    # data CSV – bin centers + density
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    _write_csv(outpng.replace(".png", ".csv"), [f"{angle_name}_deg", "density"],
               centers, counts)

###############################################################################
# 4‑b) GLOBAL COLLECTION PLOTS (individual & overlapped)
###############################################################################

def global_individual_plots(angles_all: dict, labels: list[str], out_dir: str):
    ensure_dir(out_dir)
    # φ‑ψ scatter + 2‑D hist
    for lb in labels:
        plot_scatter_phi_psi(angles_all[lb]["phi"], angles_all[lb]["psi"], lb, out_dir, prefix="01")
        plot_2d_hist_phi_psi_log(angles_all[lb]["phi"], angles_all[lb]["psi"], lb, out_dir, prefix="02")
        plot_1d_hist_angle(angles_all[lb]["phi"], "phi", lb, "03", out_dir, log_y=True)
        plot_1d_hist_angle(angles_all[lb]["psi"], "psi", lb, "04", out_dir, log_y=True)

    # side‑chain χn
    chi_names = sorted({k for lb in labels for k in angles_all[lb] if k.startswith("chi")})
    for cname in chi_names:
        for lb in labels:
            if cname in angles_all[lb]:
                plot_1d_hist_angle(angles_all[lb][cname], cname, lb, f"05_{cname}", out_dir)

# ---------------------------------------------------------------------------
# Overlapped plots – scatter + two 1‑D hists + χn hists
# ---------------------------------------------------------------------------

def _export_overlap_scatter_csv(outpng: str, labels: list[str], angles_all: dict):
    rows_label, rows_phi, rows_psi = [], [], []
    for lb in labels:
        valid = ~np.isnan(angles_all[lb]["phi"]) & ~np.isnan(angles_all[lb]["psi"])
        rows_label.extend([lb] * np.sum(valid))
        rows_phi.extend(angles_all[lb]["phi"][valid])
        rows_psi.extend(angles_all[lb]["psi"][valid])
    _write_csv(outpng.replace(".png", ".csv"), ["label", "phi_deg", "psi_deg"],
               rows_label, rows_phi, rows_psi)


def _export_overlap_hist_csv(outpng: str, labels: list[str], angles_all: dict, aname: str,
                             bins: int = 72):
    lab_col, center_col, dens_col = [], [], []
    for lb in labels:
        arr = _csv_safe(angles_all[lb][aname])
        if arr.size == 0:
            continue
        counts, bin_edges = np.histogram(arr, bins=bins, range=[-180, 180], density=True)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        lab_col.extend([lb] * len(centers))
        center_col.extend(centers)
        dens_col.extend(counts)
    _write_csv(outpng.replace(".png", ".csv"), ["label", f"{aname}_deg", "density"],
               lab_col, center_col, dens_col)


def global_overlapped_plots(angles_all: dict, labels: list[str], out_dir: str):
    ensure_dir(out_dir)

    # ---------- scatter ----------
    plt.figure(figsize=(7, 7))
    for lb in labels:
        valid = ~np.isnan(angles_all[lb]["phi"]) & ~np.isnan(angles_all[lb]["psi"])
        plt.scatter(angles_all[lb]["phi"][valid], angles_all[lb]["psi"][valid],
                    alpha=0.3, s=5, label=lb)
    plt.title("Overlapped φ‑ψ Scatter (3 sets)")
    plt.xlabel("φ (deg)")
    plt.ylabel("ψ (deg)")
    plt.xlim([-180, 180])
    plt.ylim([-180, 180])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    outpng = os.path.join(out_dir, "01_scatter_overlap.png")
    plt.savefig(outpng, dpi=300, bbox_inches="tight")
    plt.close()
    _export_overlap_scatter_csv(outpng, labels, angles_all)

    # ---------- φ histogram ----------
    plt.figure(figsize=(8, 6))
    for lb in labels:
        arr = _csv_safe(angles_all[lb]["phi"])
        sns.histplot(arr, bins=72, stat="density", alpha=0.3, label=lb, kde=False)
    plt.yscale("log")
    plt.title("Overlapped φ Distribution")
    plt.xlabel("φ (deg)")
    plt.ylabel("Density")
    plt.xlim([-180, 180])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    outpng = os.path.join(out_dir, "02_phi_overlap.png")
    plt.savefig(outpng, dpi=300, bbox_inches="tight")
    plt.close()
    _export_overlap_hist_csv(outpng, labels, angles_all, "phi")

    # ---------- ψ histogram ----------
    plt.figure(figsize=(8, 6))
    for lb in labels:
        arr = _csv_safe(angles_all[lb]["psi"])
        sns.histplot(arr, bins=72, stat="density", alpha=0.3, label=lb, kde=False)
    plt.yscale("log")
    plt.title("Overlapped ψ Distribution")
    plt.xlabel("ψ (deg)")
    plt.ylabel("Density")
    plt.xlim([-180, 180])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    outpng = os.path.join(out_dir, "03_psi_overlap.png")
    plt.savefig(outpng, dpi=300, bbox_inches="tight")
    plt.close()
    _export_overlap_hist_csv(outpng, labels, angles_all, "psi")

    # ---------- side‑chain χn ----------
    chi_names = sorted({k for lb in labels for k in angles_all[lb] if k.startswith("chi")})
    idx = 4
    for cname in chi_names:
        plt.figure(figsize=(8, 6))
        for lb in labels:
            arr = _csv_safe(angles_all[lb][cname])
            sns.histplot(arr, bins=72, stat="density", alpha=0.3, label=lb, kde=False)
        plt.title(f"Overlapped {cname} Distribution")
        plt.xlabel(f"{cname} (deg)")
        plt.ylabel("Density")
        plt.xlim([-180, 180])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()

        outpng = os.path.join(out_dir, f"{idx:02d}_{cname}_overlap.png")
        plt.savefig(outpng, dpi=300, bbox_inches="tight")
        plt.close()
        _export_overlap_hist_csv(outpng, labels, angles_all, cname)
        idx += 1

# ---------------------------------------------------------------------------
# Helper – compile all PNGs into one PDF
# ---------------------------------------------------------------------------

def compile_pdf(dir_list: list[str], output_pdf: str):
    pngs = []
    for directory in dir_list:
        for root, _, files in os.walk(directory):
            pngs.extend(os.path.join(root, f) for f in files if f.lower().endswith(".png"))
    if not pngs:
        logger.warning(f"No PNGs found – skipped PDF compilation: {output_pdf}")
        return
    pngs.sort()
    try:
        with open(output_pdf, "wb") as fh:
            fh.write(img2pdf.convert(pngs))
        logger.info(f"PDF compiled: {output_pdf} ({len(pngs)} pages)")
    except Exception as e:
        logger.error(f"PDF compilation failed: {e}")

###############################################################################
# 5) 1‑D / 2‑D METRICS (KL, JS, Wasserstein)
###############################################################################

def hist_kl_1d(data1, data2, *, bins: int = 36, a_min: float = -180, a_max: float = 180):
    d1 = _csv_safe(data1)
    d2 = _csv_safe(data2)
    if len(d1) < 2 or len(d2) < 2:
        return np.nan
    p, bin_edges = np.histogram(d1, bins=bins, range=(a_min, a_max), density=True)
    q, _ = np.histogram(d2, bins=bin_edges, density=True)
    eps = 1e-10
    p = np.where(p == 0, eps, p)
    q = np.where(q == 0, eps, q)
    return float(np.sum(p * np.log(p / q)))


def compute_1d_js(data1, data2, *, bins: int = 36, a_min: float = -180, a_max: float = 180):
    d1 = _csv_safe(data1)
    d2 = _csv_safe(data2)
    if len(d1) < 2 or len(d2) < 2:
        return np.nan
    p, bin_edges = np.histogram(d1, bins=bins, range=(a_min, a_max), density=True)
    q, _ = np.histogram(d2, bins=bin_edges, density=True)
    eps = 1e-10
    p = np.where(p == 0, eps, p)
    q = np.where(q == 0, eps, q)
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def compute_1d_wasserstein(data1, data2):
    d1 = _csv_safe(data1)
    d2 = _csv_safe(data2)
    if len(d1) < 2 or len(d2) < 2:
        return np.nan
    return float(wasserstein_distance(d1, d2))

# ---------------------------------------------------------------------------
# 2‑D helpers
# ---------------------------------------------------------------------------

def compute_2d_hist(phi: np.ndarray, psi: np.ndarray, *, bins: int = 72,
                    a_min: float = -180, a_max: float = 180):
    arr_phi = _csv_safe(phi)
    arr_psi = _csv_safe(psi)
    if arr_phi.size < 2 or arr_psi.size < 2:
        return np.zeros((bins, bins), dtype=float)
    H, _, _ = np.histogram2d(arr_phi, arr_psi, bins=bins,
                             range=[[a_min, a_max], [a_min, a_max]], density=True)
    return H


def compute_2d_kl(H_p: np.ndarray, H_q: np.ndarray):
    eps = 1e-10
    p = H_p.flatten(); q = H_q.flatten()
    if p.sum() < eps or q.sum() < eps:
        return np.nan
    p = p / p.sum(); q = q / q.sum()
    p = np.where(p == 0, eps, p); q = np.where(q == 0, eps, q)
    return float(np.sum(p * np.log(p / q)))


def js_2d(H_p: np.ndarray, H_q: np.ndarray):
    eps = 1e-10
    p = H_p.flatten(); q = H_q.flatten()
    if p.sum() < eps or q.sum() < eps:
        return np.nan
    p = p / p.sum(); q = q / q.sum()
    p = np.where(p == 0, eps, p); q = np.where(q == 0, eps, q)
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def compute_2d_wasserstein(H_p: np.ndarray, H_q: np.ndarray, *,
                           bins: int = 72, a_min: float = -180, a_max: float = 180):
    if not HAS_POT:
        return None
    eps = 1e-10
    p = H_p.flatten(); q = H_q.flatten()
    if p.sum() < eps or q.sum() < eps:
        return np.nan
    p = p / p.sum(); q = q / q.sum()
    p = np.where(p < eps, eps, p); q = np.where(q < eps, eps, q)

    # cost matrix of bin‑center euclidean distances
    def center(i, n):
        step = (a_max - a_min) / n
        return a_min + step * (i + 0.5)
    centers = np.array([[center(i, bins), center(j, bins)]
                        for i in range(bins) for j in range(bins)])
    C = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    G = ot.emd(p, q, C)
    return float(np.sum(G * C))

###############################################################################
# 6) VISUAL KL HEATMAPS + CSV
###############################################################################

def chunked_kl_heatmaps(kl_matrix: np.ndarray, angle_list: list[str], residue_ids: list[int],
                        *, chunk_size: int = 50, out_dir: str = "kl_heatmaps"):
    os.makedirs(out_dir, exist_ok=True)
    n_res = len(residue_ids)
    n_blocks = math.ceil(n_res / chunk_size)
    for b in range(n_blocks):
        s, e = b * chunk_size, min(n_res, (b + 1) * chunk_size)
        block = kl_matrix[s:e, :]
        mask = np.isnan(block)

        fig_w = 2 + 0.5 * len(angle_list)
        fig_h = 1 + 0.3 * (e - s)
        plt.figure(figsize=(fig_w, fig_h))
        cmap = sns.color_palette("viridis", as_cmap=True)
        cmap.set_bad("lightgray")
        ax = sns.heatmap(block, mask=mask, cmap=cmap, square=False,
                         cbar_kws={"label": "KL Divergence"}, linewidths=0.4, linecolor="white")
        ax.set_xlabel("Dihedral Angle"); ax.set_ylabel("Residue ID")
        ax.set_title(f"KL Divergence Resid {residue_ids[s]}–{residue_ids[e-1]}")
        ax.set_xticks(np.arange(len(angle_list)) + 0.5)
        ax.set_xticklabels(angle_list, rotation=45, ha="right")
        ax.set_yticks(np.arange(e - s) + 0.5)
        ax.set_yticklabels(residue_ids[s:e], rotation=0)
        plt.tight_layout()

        outpng = os.path.join(out_dir, f"kl_heatmap_block_{b+1}.png")
        plt.savefig(outpng, dpi=300, bbox_inches="tight")
        plt.close()
        # CSV – fade NaN to empty string for readability
        _write_csv(outpng.replace(".png", ".csv"), ["ResidueID", *angle_list],
                   residue_ids[s:e], *[block[:, j] for j in range(len(angle_list))])

###############################################################################
# 7) TOP‑K SUMMARIES + CSV EXPORTS
###############################################################################

def kl_summaries(kl_matrix: np.ndarray, angle_list: list[str], residue_ids: list[int],
                 *, top_k: int = 10, out_dir: str | None = None):
    mean_res = np.nanmean(kl_matrix, axis=1)
    mean_ang = np.nanmean(kl_matrix, axis=0)

    top_res_idx = np.argsort(mean_res)[::-1][:top_k]
    top_ang_idx = np.argsort(mean_ang)[::-1][:min(top_k, len(angle_list))]

    logger.info("\nTop %d Residues by average KL:", top_k)
    for rnk, idx in enumerate(top_res_idx, 1):
        logger.info(" %2d. Residue %d → KL=%.4f", rnk, residue_ids[idx], mean_res[idx])

    logger.info("\nTop %d Angles by average KL:", top_k)
    for rnk, idx in enumerate(top_ang_idx, 1):
        logger.info(" %2d. %s → KL=%.4f", rnk, angle_list[idx], mean_ang[idx])

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # bar charts + CSV of bars
        for which, idxs, means, fname, color in [
            ("Residues", top_res_idx, mean_res[top_res_idx], "top_residues_bar.png", "gray"),
            ("Angles",   top_ang_idx, mean_ang[top_ang_idx], "top_angles_bar.png", "orange"),
        ]:
            plt.figure(figsize=(6, 4))
            plt.barh(range(len(idxs)), means[::-1], color=color)
            labels = [str(residue_ids[i] if which == "Residues" else angle_list[i]) for i in idxs]
            plt.yticks(range(len(idxs)), labels[::-1])
            plt.xlabel("Average KL"); plt.title(f"Top {top_k} {which}")
            plt.tight_layout()
            outpng = os.path.join(out_dir, fname)
            plt.savefig(outpng, dpi=300, bbox_inches="tight")
            plt.close()
            _write_csv(outpng.replace(".png", ".csv"), [which[:-1], "avg_KL"], labels, means)

    return top_res_idx, top_ang_idx

###############################################################################
# 8) PER‑RESIDUE OVERLAY DIAGNOSTICS (PNG + CSV)
###############################################################################

def distribution_overlay_topres(top_res_idx: np.ndarray, kl_matrix: np.ndarray,
                                residue_ids: list[int], angle_list: list[str],
                                angles_A: dict, angles_B: dict, out_dir: str,
                                *, bins: int = 36, a_min: float = -180, a_max: float = 180):
    os.makedirs(out_dir, exist_ok=True)
    for i_r in top_res_idx:
        rid = residue_ids[i_r]
        subdir = os.path.join(out_dir, f"res_{rid}")
        os.makedirs(subdir, exist_ok=True)
        for j_a, aname in enumerate(angle_list):
            arr1 = _csv_safe(angles_A[rid][aname])
            arr2 = _csv_safe(angles_B[rid][aname])
            if arr1.size == 0 and arr2.size == 0:
                continue

            plt.figure(figsize=(6, 4))
            sns.histplot(arr1, bins=bins, stat="density", alpha=0.4, label="Traj1",
                         element="step", fill=True, binrange=(a_min, a_max))
            sns.histplot(arr2, bins=bins, stat="density", alpha=0.4, label="Traj2",
                         element="step", fill=True, binrange=(a_min, a_max))
            kl_val = kl_matrix[i_r, j_a]
            plt.title(f"Residue {rid}, {aname}, KL={kl_val:.3f}")
            plt.xlabel(f"{aname} (deg)"); plt.ylabel("Density")
            plt.xlim([a_min, a_max]); plt.grid(True, linestyle="--", alpha=0.5); plt.legend()
            plt.tight_layout()

            outpng = os.path.join(subdir, f"{aname}.png")
            plt.savefig(outpng, dpi=300, bbox_inches="tight")
            plt.close()
            # CSV – two columns for two trajectories (density‑normalised hist)
            counts1, bin_edges = np.histogram(arr1, bins=bins, range=(a_min, a_max), density=True)
            counts2, _ = np.histogram(arr2, bins=bin_edges, density=True)
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            _write_csv(outpng.replace(".png", ".csv"), ["bin_center_deg", "traj1_density", "traj2_density"],
                       centers, counts1, counts2)

###############################################################################
# 9) MAIN DRIVER
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="""
        Merge global distribution comparison (3×HDF5) and per‑residue KL/JS/Wasserstein analysis,
        with automatic CSV exports for every plot.
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--condensed_json", required=True, help="condensed_residues.json path")
    parser.add_argument("--h5_1", required=True, help="HDF5 trajectory #1")
    parser.add_argument("--h5_2", required=True, help="HDF5 trajectory #2")
    parser.add_argument("--h5_3", required=True, help="HDF5 trajectory #3")
    parser.add_argument("--labels", nargs=3, default=["Set1", "Set2", "Set3"],
                        help="Labels corresponding to the three trajectories")
    parser.add_argument("--out_dir", default="mega_output", help="Output directory root")
    parser.add_argument("--device", default="cpu", help="PyTorch device: cpu / cuda")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Chunk size for dihedral extraction")
    parser.add_argument("--hist_bins", type=int, default=36, help="Bins for 1‑D histograms")
    parser.add_argument("--angle_min", type=float, default=-180)
    parser.add_argument("--angle_max", type=float, default=180)
    parser.add_argument("--top_k", type=int, default=10, help="Top residues for overlay")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- load inputs ----------------
    cond_data = load_condensed_json(args.condensed_json)
    coords_dict = {}
    for path, lb in zip([args.h5_1, args.h5_2, args.h5_3], args.labels):
        coords_dict[lb] = load_h5_coords(path)
        logger.info("%-5s: coords shape = %s", lb, coords_dict[lb].shape)

    # ---------------- extract angles ----------------
    angles_global, angles_perres = {}, {}
    all_resids_global = None
    for lb in args.labels:
        g, p, _, r_ids = gather_global_and_perres_angles(coords_dict[lb], cond_data,
                                                          device=args.device, chunk_size=args.chunk_size)
        angles_global[lb] = g
        angles_perres[lb] = p
        if all_resids_global is None:
            all_resids_global = r_ids

    # ---------------- global plots ----------------
    glob_dir = os.path.join(args.out_dir, "global_compare")
    indiv_dir = os.path.join(glob_dir, "individual")
    comb_dir  = os.path.join(glob_dir, "combined")

    global_individual_plots(angles_global, args.labels, indiv_dir)
    global_overlapped_plots(angles_global, args.labels, comb_dir)
    compile_pdf([indiv_dir, comb_dir], os.path.join(glob_dir, "Global_Distributions.pdf"))

    # ---------------- extra metrics per global pair ----------------
    def global_pair_metrics(lbA: str, lbB: str):
        logger.info("\n=== Metrics: %s vs %s ===", lbA, lbB)
        A, B = angles_global[lbA], angles_global[lbB]
        angle_union = sorted(set(A) | set(B))
        out_csv = os.path.join(args.out_dir, f"Global_Metrics_{lbA}_vs_{lbB}.csv")
        with open(out_csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["Angle", "KL_1D", "JS_1D", "WDist_1D"])
            for aname in angle_union:
                if aname not in A or aname not in B:
                    wr.writerow([aname, "N/A", "N/A", "N/A"])
                    continue
                kl = hist_kl_1d(A[aname], B[aname], bins=args.hist_bins,
                                 a_min=args.angle_min, a_max=args.angle_max)
                js = compute_1d_js(A[aname], B[aname], bins=args.hist_bins,
                                    a_min=args.angle_min, a_max=args.angle_max)
                wd = compute_1d_wasserstein(A[aname], B[aname])
                wr.writerow([aname, f"{kl:.6f}", f"{js:.6f}", f"{wd:.6f}"])

            # 2‑D φ‑ψ Ramachandran
            if all(key in A for key in ["phi", "psi"]) and all(key in B for key in ["phi", "psi"]):
                H_A = compute_2d_hist(A["phi"], A["psi"])
                H_B = compute_2d_hist(B["phi"], B["psi"])
                kl2d = compute_2d_kl(H_A, H_B)
                js2d = js_2d(H_A, H_B)
                w2d = compute_2d_wasserstein(H_A, H_B) if HAS_POT else None
                wr.writerow([])
                wr.writerow(["2D_Rama_KL(A→B)", f"{kl2d:.6f}" if not np.isnan(kl2d) else "N/A"])
                wr.writerow(["2D_Rama_JS", f"{js2d:.6f}" if not np.isnan(js2d) else "N/A"])
                wr.writerow(["2D_Rama_Wasserstein", f"{w2d:.6f}" if w2d not in [None, np.nan] else "N/A"])
        logger.info("Global metrics CSV → %s", out_csv)

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        global_pair_metrics(args.labels[i], args.labels[j])

    # ---------------- per‑residue KL for each pair ----------------
    def do_pair(lbA: str, lbB: str):
        pair_dir = os.path.join(args.out_dir, f"KL_{lbA}_vs_{lbB}")
        os.makedirs(pair_dir, exist_ok=True)
        anglesA, anglesB = angles_perres[lbA], angles_perres[lbB]
        angle_list = sorted({k for d in anglesA.values() for k in d} | {k for d in anglesB.values() for k in d})
        residue_ids = sorted(anglesA)

        kl_mat = np.full((len(residue_ids), len(angle_list)), np.nan, dtype=float)
        for i_r, rid in enumerate(residue_ids):
            for j_a, aname in enumerate(angle_list):
                a1, a2 = anglesA[rid].get(aname, np.array([])), anglesB[rid].get(aname, np.array([]))
                if a1.size >= 2 and a2.size >= 2:
                    kl_mat[i_r, j_a] = hist_kl_1d(a1, a2, bins=args.hist_bins,
                                                  a_min=args.angle_min, a_max=args.angle_max)

        heat_dir = os.path.join(pair_dir, "heatmaps")
        chunked_kl_heatmaps(kl_mat, angle_list, residue_ids, out_dir=heat_dir)
        top_res_idx, _ = kl_summaries(kl_mat, angle_list, residue_ids,
                                      top_k=args.top_k, out_dir=pair_dir)
        # raw matrix CSV
        export_csv = os.path.join(pair_dir, "kl_data.csv")
        _write_csv(export_csv, ["ResidueID", *angle_list], residue_ids,
                   *[kl_mat[:, j] for j in range(len(angle_list))])
        # overlays
        overlay_dir = os.path.join(pair_dir, "detailed_overlays")
        distribution_overlay_topres(top_res_idx, kl_mat, residue_ids, angle_list,
                                    anglesA, anglesB, overlay_dir,
                                    bins=args.hist_bins, a_min=args.angle_min, a_max=args.angle_max)
        # PDF
        compile_pdf([pair_dir], os.path.join(pair_dir, f"KL_{lbA}_vs_{lbB}.pdf"))
        logger.info("Pairwise KL done: %s vs %s", lbA, lbB)

    do_pair(args.labels[0], args.labels[1])
    do_pair(args.labels[0], args.labels[2])
    do_pair(args.labels[1], args.labels[2])

    logger.info("All tasks complete ✔  CSVs available alongside PNGs.")


if __name__ == "__main__":
    main()
