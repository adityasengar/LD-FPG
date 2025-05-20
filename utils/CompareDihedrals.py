#!/usr/bin/env python3
"""
Perform both:
  1) Global dihedral distribution plotting for 3 .h5 files (like compare_3h5_dihedrals_ordered).
  2) Per-residue KL analysis (like kl_dihedrals_h5) for all pairwise combos:
     (h5_1 vs. h5_2), (h5_1 vs. h5_3), (h5_2 vs. h5_3).
  3) Additional metrics for each pair, stored in Global_Metrics_*.csv:
     - 1D KL, 1D JS, 1D Wasserstein (for each angle).
     - 2D KL (Ramachandran), 2D JS, and optional 2D Wasserstein for the φ,ψ histograms.
Removed the older 2d_rama_coverage approach.

Usage:
  python mega_compare_3h5_with_kl.py \
      --condensed_json condensed_residues.json \
      --h5_1 file1.h5 \
      --h5_2 file2.h5 \
      --h5_3 file3.h5 \
      --labels "Set1" "Set2" "Set3" \
      --out_dir "mega_output" \
      --device "cuda" \
      --chunk_size 500 \
      --hist_bins 36

Install pot (for 2D Wasserstein):
  pip install pot
"""

import os
import sys
import json
import argparse
import math
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import img2pdf
from matplotlib.colors import LogNorm
from copy import deepcopy
import logging

try:
    import ot  # pip install pot
    HAS_POT = True
except ImportError:
    HAS_POT = False

# For 1D Wasserstein distance (scipy)
from scipy.stats import wasserstein_distance

logger = logging.getLogger("MegaCompare3h5")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s"))
    logger.addHandler(ch)

###############################################################################
# 1) LOADING UTILS
###############################################################################
def load_condensed_json(json_path):
    if not os.path.isfile(json_path):
        sys.exit(f"Error: condensed JSON not found => {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def load_h5_coords(h5_file):
    if not os.path.isfile(h5_file):
        sys.exit(f"Error: HDF5 file not found => {h5_file}")
    with h5py.File(h5_file, "r") as hf:
        if "data" in hf:
            coords = hf["data"][:]
        else:
            keys = list(hf.keys())
            if len(keys) == 1:
                coords = hf[keys[0]][:]
            else:
                sys.exit(f"Ambiguous dataset in {h5_file}, keys={keys}")
    return coords  # shape => (N_frames, N_atoms, 3)

###############################################################################
# 2) TORCH DIHEDRAL
###############################################################################
def dihedral_torch(coords):
    """
    coords => (batch, 4,3)
    returns => (batch,) in radians
    """
    b1 = coords[:,1] - coords[:,0]
    b2 = coords[:,2] - coords[:,1]
    b3 = coords[:,3] - coords[:,2]
    n1 = torch.cross(b1,b2,dim=1)
    n2 = torch.cross(b2,b3,dim=1)
    b2_len = torch.norm(b2,dim=1,keepdim=True).clamp(min=1e-6)
    b2_unit = b2 / b2_len
    m1 = torch.cross(n1,b2_unit,dim=1)
    x = torch.sum(n1*n2,dim=1)
    y = torch.sum(m1*n2,dim=1)
    angles = torch.atan2(y,x)
    return angles

###############################################################################
# 3) GATHER BOTH GLOBAL ANGLES + PER-RES ANGLES
###############################################################################
def gather_global_and_perres_angles(coords_np, condensed_data, device="cpu", chunk_size=2000):
    """
    - "Global angles" => single distribution across ALL residues for each angle (phi, psi, each chi).
    - "Per-res angles" => angles_dict[res_id][angle_name] => array of shape (#frames,).
    Returns (global_dict, perres_dict, all_chi_names, all_res_ids).
    """
    device = torch.device(device)
    all_res_ids = sorted([int(k) for k in condensed_data.keys()])
    angle_info = {}
    all_chi_names = set()

    for r_str in condensed_data.keys():
        r_int = int(r_str)
        torsions = condensed_data[r_str].get("torsion_atoms",{})
        phi_quad = torsions.get("phi", None)
        psi_quad = torsions.get("psi", None)
        chi_dict = torsions.get("chi",{})
        angle_info[r_int] = {
            "phi": phi_quad,
            "psi": psi_quad,
            "chi": {}
        }
        for c_name, c_quad in chi_dict.items():
            all_chi_names.add(c_name)
            angle_info[r_int]["chi"][c_name] = c_quad

    all_chi_list = sorted(list(all_chi_names))

    global_dict = {
        "phi": [],
        "psi": []
    }
    for c_name in all_chi_list:
        global_dict[c_name] = []

    perres_dict = {}
    for rid in all_res_ids:
        perres_dict[rid] = {"phi":[], "psi":[]}
        for c_name in all_chi_list:
            perres_dict[rid][c_name] = []

    N_frames = coords_np.shape[0]

    def compute_quad(quad, coords_chunk):
        if quad is None or len(quad)!=4:
            return None
        idx_t = torch.tensor(quad,dtype=torch.long,device=device)
        gather_coords = coords_chunk[:, idx_t, :]
        angles_rad = dihedral_torch(gather_coords.view(-1,4,3))
        angles_deg = angles_rad*(180.0/math.pi)
        return angles_deg.cpu().numpy()

    frame_start=0
    while frame_start<N_frames:
        frame_end=min(N_frames, frame_start+chunk_size)
        chunk_np=coords_np[frame_start:frame_end]
        coords_chunk=torch.from_numpy(chunk_np).to(device)
        csize=coords_chunk.shape[0]

        for rid in all_res_ids:
            info=angle_info[rid]
            # phi
            phi_res=compute_quad(info["phi"], coords_chunk)
            if phi_res is not None:
                global_dict["phi"].extend(phi_res.tolist())
                perres_dict[rid]["phi"].extend(phi_res.tolist())
            # psi
            psi_res=compute_quad(info["psi"], coords_chunk)
            if psi_res is not None:
                global_dict["psi"].extend(psi_res.tolist())
                perres_dict[rid]["psi"].extend(psi_res.tolist())
            # chi
            for c_name in all_chi_list:
                c_quad=info["chi"].get(c_name,None)
                c_res=compute_quad(c_quad, coords_chunk)
                if c_res is not None:
                    global_dict[c_name].extend(c_res.tolist())
                    perres_dict[rid][c_name].extend(c_res.tolist())

        logger.info(f"Frames {frame_end}/{N_frames} done for angle extraction.")
        frame_start=frame_end

    for k in global_dict.keys():
        global_dict[k]=np.array(global_dict[k],dtype=float)
    for rid in all_res_ids:
        for k in perres_dict[rid].keys():
            perres_dict[rid][k]=np.array(perres_dict[rid][k],dtype=float)

    return global_dict, perres_dict, all_chi_list, all_res_ids

###############################################################################
# 4) GLOBAL PLOTTING (3-FILES)
###############################################################################
def ensure_dir(d):
    os.makedirs(d,exist_ok=True)

def plot_scatter_phi_psi(phi, psi, label, out_dir, prefix="01"):
    ensure_dir(out_dir)
    arr_phi=phi[~np.isnan(phi)]
    arr_psi=psi[~np.isnan(psi)]
    plt.figure(figsize=(7,7))
    sns.scatterplot(x=arr_phi, y=arr_psi, alpha=0.3)
    plt.title(f"{label}: φ vs. ψ Scatter")
    plt.xlabel("φ (deg)")
    plt.ylabel("ψ (deg)")
    plt.xlim([-180,180])
    plt.ylim([-180,180])
    plt.grid(True,linestyle='--',alpha=0.5)
    outpng=os.path.join(out_dir,f"{prefix}_scatter_{label}.png")
    plt.savefig(outpng,dpi=300,bbox_inches='tight')
    plt.close()

def plot_2d_hist_phi_psi_log(phi, psi, label, out_dir, prefix="02"):
    ensure_dir(out_dir)
    arr_phi=phi[~np.isnan(phi)]
    arr_psi=psi[~np.isnan(psi)]
    plt.figure(figsize=(7,7))
    plt.hist2d(arr_phi, arr_psi, bins=72, range=[[-180,180],[-180,180]],
               cmap="viridis", norm=LogNorm())
    plt.colorbar(label="Counts (log scale)")
    plt.title(f"{label}: φ vs. ψ 2D Hist (log)")
    plt.xlabel("φ (deg)")
    plt.ylabel("ψ (deg)")
    plt.grid(True,linestyle='--',alpha=0.5)
    outpng=os.path.join(out_dir,f"{prefix}_2dhist_{label}.png")
    plt.savefig(outpng,dpi=300,bbox_inches='tight')
    plt.close()

def plot_1d_hist_angle(angle_data, angle_name, label, prefix, out_dir, log_y=False):
    ensure_dir(out_dir)
    arr=angle_data[~np.isnan(angle_data)]
    if arr.size<1:
        return
    plt.figure(figsize=(8,6))
    sns.histplot(arr,bins=72,stat='density',alpha=0.7)
    if log_y:
        plt.yscale("log")
    plt.title(f"{label}: {angle_name} Distribution")
    plt.xlabel(f"{angle_name} (deg)")
    plt.ylabel("Density")
    plt.xlim([-180,180])
    plt.grid(True,linestyle='--',alpha=0.5)
    outpng=os.path.join(out_dir,f"{prefix}_{angle_name}_{label}.png")
    plt.savefig(outpng,dpi=300,bbox_inches='tight')
    plt.close()

def global_individual_plots(angles_all, labels, out_dir):
    ensure_dir(out_dir)
    # scatter
    for lb in labels:
        phi=angles_all[lb]["phi"]
        psi=angles_all[lb]["psi"]
        plot_scatter_phi_psi(phi,psi,lb,out_dir,prefix="01")
    # 2D hist
    for lb in labels:
        phi=angles_all[lb]["phi"]
        psi=angles_all[lb]["psi"]
        plot_2d_hist_phi_psi_log(phi,psi,lb,out_dir,prefix="02")
    # 1D hist φ, log
    for lb in labels:
        plot_1d_hist_angle(angles_all[lb]["phi"],"phi",lb,"03",out_dir,log_y=True)
    # 1D hist ψ, log
    for lb in labels:
        plot_1d_hist_angle(angles_all[lb]["psi"],"psi",lb,"04",out_dir,log_y=True)
    # sidechain
    all_chi=set()
    for lb in labels:
        for k in angles_all[lb].keys():
            if k.startswith("chi"):
                all_chi.add(k)
    all_chi_list=sorted(list(all_chi))
    for c_name in all_chi_list:
        for lb in labels:
            if c_name in angles_all[lb]:
                plot_1d_hist_angle(angles_all[lb][c_name],c_name,lb,f"05_{c_name}",out_dir,log_y=False)

def global_overlapped_plots(angles_all, labels, out_dir):
    ensure_dir(out_dir)
    # overlapped scatter
    plt.figure(figsize=(7,7))
    for lb in labels:
        arr_phi=angles_all[lb]["phi"]
        arr_psi=angles_all[lb]["psi"]
        valid=~np.isnan(arr_phi)&~np.isnan(arr_psi)
        arr_phi=arr_phi[valid]
        arr_psi=arr_psi[valid]
        plt.scatter(arr_phi,arr_psi,alpha=0.3,s=5,label=lb)
    plt.title("Overlapped φ-ψ Scatter (3 sets)")
    plt.xlabel("φ (deg)")
    plt.ylabel("ψ (deg)")
    plt.xlim([-180,180])
    plt.ylim([-180,180])
    plt.grid(True,linestyle='--',alpha=0.5)
    plt.legend(loc="upper right")
    outpng=os.path.join(out_dir,"01_scatter_overlap.png")
    plt.savefig(outpng,dpi=300,bbox_inches='tight')
    plt.close()

    # overlapped φ
    plt.figure(figsize=(8,6))
    for lb in labels:
        arr=angles_all[lb]["phi"]
        arr=arr[~np.isnan(arr)]
        sns.histplot(arr,bins=72,stat='density',alpha=0.3,label=lb,kde=False)
    plt.title("Overlapped φ Distribution")
    plt.xlabel("φ (deg)")
    plt.ylabel("Density")
    plt.xlim([-180,180])
    plt.yscale("log")
    plt.grid(True,linestyle='--',alpha=0.5)
    plt.legend(loc="upper right")
    outpng=os.path.join(out_dir,"02_phi_overlap.png")
    plt.savefig(outpng,dpi=300,bbox_inches='tight')
    plt.close()

    # overlapped ψ
    plt.figure(figsize=(8,6))
    for lb in labels:
        arr=angles_all[lb]["psi"]
        arr=arr[~np.isnan(arr)]
        sns.histplot(arr,bins=72,stat='density',alpha=0.3,label=lb,kde=False)
    plt.title("Overlapped ψ Distribution")
    plt.xlabel("ψ (deg)")
    plt.ylabel("Density")
    plt.xlim([-180,180])
    plt.yscale("log")
    plt.grid(True,linestyle='--',alpha=0.5)
    plt.legend(loc="upper right")
    outpng=os.path.join(out_dir,"03_psi_overlap.png")
    plt.savefig(outpng,dpi=300,bbox_inches='tight')
    plt.close()

    # sidechains normal y scale
    all_chi=set()
    for lb in labels:
        for k in angles_all[lb].keys():
            if k.startswith("chi"):
                all_chi.add(k)
    all_chi_list=sorted(list(all_chi))
    idx=4
    for c_name in all_chi_list:
        plt.figure(figsize=(8,6))
        for lb in labels:
            arr=angles_all[lb][c_name]
            arr=arr[~np.isnan(arr)]
            sns.histplot(arr,bins=72,stat='density',alpha=0.3,label=lb,kde=False)
        plt.title(f"Overlapped {c_name} Distribution")
        plt.xlabel(f"{c_name} (deg)")
        plt.ylabel("Density")
        plt.xlim([-180,180])
        plt.grid(True,linestyle='--',alpha=0.5)
        plt.legend(loc="upper right")
        outpng=os.path.join(out_dir,f"{idx:02d}_{c_name}_overlap.png")
        idx+=1
        plt.savefig(outpng,dpi=300,bbox_inches='tight')
        plt.close()

def compile_pdf(dir_list,output_pdf):
    png_files=[]
    for d in dir_list:
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".png"):
                    png_files.append(os.path.join(root,f))
    if not png_files:
        logger.warning(f"No PNG found for PDF => {output_pdf}")
        return
    png_files=sorted(png_files)
    try:
        with open(output_pdf,"wb") as of:
            of.write(img2pdf.convert(png_files))
        logger.info(f"Compiled {len(png_files)} images => {output_pdf}")
    except Exception as e:
        logger.error(f"PDF compilation failed => {e}")

###############################################################################
# 5) 1D KL, 2D KL, 1D & 2D JS, WASS
###############################################################################
def hist_kl_1d(data1,data2,bins=36,angle_min=-180,angle_max=180):
    d1=data1[~np.isnan(data1)]
    d2=data2[~np.isnan(data2)]
    if len(d1)<2 or len(d2)<2:
        return np.nan
    p_hist, bin_edges=np.histogram(d1,bins=bins,range=(angle_min,angle_max),density=True)
    q_hist, _=np.histogram(d2,bins=bin_edges,density=True)
    eps=1e-10
    p_hist=np.where(p_hist==0,eps,p_hist)
    q_hist=np.where(q_hist==0,eps,q_hist)
    kl_val=np.sum(p_hist*np.log(p_hist/q_hist))
    return kl_val

def compute_1d_js(data1, data2, bins=36, angle_min=-180, angle_max=180):
    d1=data1[~np.isnan(data1)]
    d2=data2[~np.isnan(data2)]
    if len(d1)<2 or len(d2)<2:
        return np.nan
    p_hist, bin_edges=np.histogram(d1,bins=bins,range=(angle_min,angle_max),density=True)
    q_hist, _=np.histogram(d2,bins=bin_edges,density=True)
    eps=1e-10
    p_hist=np.where(p_hist==0,eps,p_hist)
    q_hist=np.where(q_hist==0,eps,q_hist)
    m=0.5*(p_hist+q_hist)
    def kl_div(p,q):
        return np.sum(p*np.log(p/q))
    js=0.5*kl_div(p_hist,m)+0.5*kl_div(q_hist,m)
    return js

def compute_1d_wasserstein(data1, data2):
    d1=data1[~np.isnan(data1)]
    d2=data2[~np.isnan(data2)]
    if len(d1)<2 or len(d2)<2:
        return np.nan
    return wasserstein_distance(d1,d2)

def compute_2d_hist(data_phi,data_psi,bins=72,angle_min=-180,angle_max=180):
    arr_phi=data_phi[~np.isnan(data_phi)]
    arr_psi=data_psi[~np.isnan(data_psi)]
    if arr_phi.size<2 or arr_psi.size<2:
        return np.zeros((bins,bins),dtype=float)
    h2d, _, _=np.histogram2d(arr_phi, arr_psi,bins=bins,
                             range=[[angle_min,angle_max],[angle_min,angle_max]],
                             density=True)
    return h2d

def compute_2d_kl(histA, histB):
    """
    2D KL divergence p->q.
    flatten, sum p_i log(p_i / q_i).
    not symmetrical. If you want symmetrical, do 0.5*(KL(A->B)+KL(B->A)).
    """
    eps=1e-10
    p=histA.flatten()
    q=histB.flatten()
    # sum p => pSum, sum q => qSum
    pSum=np.sum(p)
    qSum=np.sum(q)
    if pSum<eps or qSum<eps:
        return np.nan
    # normalize
    p/=pSum
    q/=qSum
    p=np.where(p==0,eps,p)
    q=np.where(q==0,eps,q)
    kl_2d=np.sum(p*np.log(p/q))
    return kl_2d

def js_2d(histA,histB):
    eps=1e-10
    p=histA.flatten()
    q=histB.flatten()
    sp=np.sum(p)
    sq=np.sum(q)
    if sp<eps or sq<eps:
        return np.nan
    p/=sp
    q/=sq
    p=np.where(p==0,eps,p)
    q=np.where(q==0,eps,q)
    m=0.5*(p+q)
    def kl(r,m):
        return np.sum(r*np.log(r/m))
    return 0.5*kl(p,m)+0.5*kl(q,m)

def compute_2d_wasserstein(histA, histB):
    """
    Attempt 2D EMD using the pot (Python OT) library.
    We'll assume the same bin shape and flatten them. We'll build a cost matrix
    for bin centers in 2D. This can be slow for big bins. 
    """
    if not HAS_POT:
        return None  # or np.nan, indicating not available
    eps=1e-10
    p=histA.flatten()
    q=histB.flatten()
    sp=p.sum()
    sq=q.sum()
    if sp<eps or sq<eps:
        return np.nan
    p/=sp
    q/=sq
    # build cost matrix from bin centers
    binsX,binsY=histA.shape
    # create (binsX*binsY,2) array of bin centers
    coords=[]
    stepX=1.0/binsX  # not exactly correct if we want real degrees,
    # better approach: we know the range is angle_min..angle_max, 
    # so binWidth=(angle_max-angle_min)/binsX
    # let's do the simpler approach
    # We'll do a real approach:
    # rangeSize=angle_max-angle_min
    # binWidth=rangeSize/binsX
    # center_i= angle_min+binWidth*(i+0.5)
    # We'll do that for X and Y.
    # Then cost=EuclDist between centers. 
    # We'll disclaim it might be slow.
    # Implementation below:
    rangeSizeX=180-(-180)  # but better to use arguments. let's do so.
    # we'll just use global
    # Let's do argument or just do 72 bins => step= (angle_max-angle_min)/binsX
    # We'll define a function for the center of i
    def centerCoord(i, bins, amin, amax):
        step=(amax-amin)/bins
        return amin + step*(i+0.5)

    costMat=np.zeros((binsX*binsY,binsX*binsY),dtype=float)
    centers=[]
    for i in range(binsX):
        cx=centerCoord(i,binsX,-180,180)
        for j in range(binsY):
            cy=centerCoord(j,binsY,-180,180)
            centers.append([cx,cy])
    centers=np.array(centers)
    # now cost = eucl dist
    for idxA in range(binsX*binsY):
        dx=centers[idxA]
        for idxB in range(binsX*binsY):
            dy=centers[idxB]
            costMat[idxA,idxB]=np.sqrt((dx[0]-dy[0])**2+(dx[1]-dy[1])**2)

    import ot
    # run EMD
    # p,q => shape => (binsX*binsY,)
    # costMat => shape => (binsX*binsY, binsX*binsY)
    # result => optimal plan => shape => same
    G=ot.emd(p,q,costMat)
    w2d=np.sum(G*costMat)  # total cost
    return w2d

###############################################################################
# CHUNKED KL HEATMAPS, SUMMARIES, ETC
###############################################################################
def chunked_kl_heatmaps(kl_matrix, angle_list, residue_ids, chunk_size=50, out_dir="kl_heatmaps"):
    os.makedirs(out_dir, exist_ok=True)
    import math
    n_res=len(residue_ids)
    n_angles=len(angle_list)
    n_blocks=math.ceil(n_res/chunk_size)
    for block_i in range(n_blocks):
        start_r=block_i*chunk_size
        end_r=min(n_res,(block_i+1)*chunk_size)
        block_data=kl_matrix[start_r:end_r,:]
        block_res=residue_ids[start_r:end_r]
        mask=np.isnan(block_data)
        fig_w=2+0.5*n_angles
        fig_h=1+0.3*(end_r-start_r)
        plt.figure(figsize=(fig_w,fig_h))
        cmap=sns.color_palette("viridis",as_cmap=True)
        cmap.set_bad("lightgray")
        ax=sns.heatmap(block_data,cmap=cmap,mask=mask,square=False,
                       cbar_kws={"label":"KL Divergence"},linewidths=0.4,linecolor="white")
        ax.set_xlabel("Dihedral Angle",fontsize=12)
        ax.set_ylabel("Residue ID",fontsize=12)
        block_title=f"KL Divergence for Resid {block_res[0]}..{block_res[-1]}"
        ax.set_title(block_title,fontsize=14)
        ax.set_xticks(np.arange(n_angles)+0.5)
        ax.set_xticklabels(angle_list,rotation=45,ha='right',fontsize=10)
        block_size=(end_r-start_r)
        ax.set_yticks(np.arange(block_size)+0.5)
        ax.set_yticklabels(block_res,rotation=0,fontsize=10)
        plt.tight_layout()
        outpng=os.path.join(out_dir,f"kl_heatmap_block_{block_i+1}.png")
        plt.savefig(outpng,dpi=300,bbox_inches='tight')
        plt.close()

def kl_summaries(kl_matrix, angle_list, residue_ids, top_k=10, out_dir=None):
    mean_res=np.nanmean(kl_matrix,axis=1)
    mean_ang=np.nanmean(kl_matrix,axis=0)
    sort_res_idx=np.argsort(mean_res)[::-1]
    top_res_idx=sort_res_idx[:top_k]
    logger.info(f"\nTop {top_k} Residues by average KL:")
    for rank,i_r in enumerate(top_res_idx):
        logger.info(f" {rank+1}. Residue {residue_ids[i_r]} => KL={mean_res[i_r]:.4f}")

    sort_ang_idx=np.argsort(mean_ang)[::-1]
    top_ang_idx=sort_ang_idx[:min(top_k,len(angle_list))]
    logger.info(f"\nTop {top_k} Angles by average KL:")
    for rank,i_a in enumerate(top_ang_idx):
        logger.info(f" {rank+1}. {angle_list[i_a]} => KL={mean_ang[i_a]:.4f}")

    if out_dir:
        os.makedirs(out_dir,exist_ok=True)
        # bar plot for residues
        plt.figure(figsize=(6,4))
        vals=mean_res[top_res_idx]
        labs=[str(residue_ids[r]) for r in top_res_idx]
        plt.barh(range(len(vals)),vals[::-1],color="gray")
        plt.yticks(range(len(vals)),labs[::-1])
        plt.xlabel("Average KL")
        plt.title(f"Top {top_k} Residues")
        plt.tight_layout()
        outpng=os.path.join(out_dir,"top_residues_bar.png")
        plt.savefig(outpng,dpi=300,bbox_inches='tight')
        plt.close()

        # bar plot angles
        plt.figure(figsize=(6,4))
        avals=mean_ang[top_ang_idx]
        alabs=[angle_list[a] for a in top_ang_idx]
        plt.barh(range(len(avals)),avals[::-1],color='orange')
        plt.yticks(range(len(avals)),alabs[::-1])
        plt.xlabel("Average KL")
        plt.title(f"Top {top_k} Angles")
        plt.tight_layout()
        outpng=os.path.join(out_dir,"top_angles_bar.png")
        plt.savefig(outpng,dpi=300,bbox_inches='tight')
        plt.close()

    return top_res_idx, top_ang_idx

def export_kl_csv(kl_matrix, angle_list, residue_ids, out_csv):
    import csv
    with open(out_csv,"w",newline="") as f:
        writer=csv.writer(f)
        header=["ResidueID"]+angle_list
        writer.writerow(header)
        for i_r,rid in enumerate(residue_ids):
            rowvals=[]
            for j_a in range(len(angle_list)):
                val=kl_matrix[i_r,j_a]
                if np.isnan(val):
                    rowvals.append("")
                else:
                    rowvals.append(f"{val:.6g}")
            writer.writerow([rid]+rowvals)
    logger.info(f"KL data => {out_csv}")

def distribution_overlay_topres(
    top_res_idx, kl_matrix, residue_ids, angle_list,
    angles_perres_1, angles_perres_2,
    out_dir,
    bins=36,
    angle_min=-180,
    angle_max=180
):
    os.makedirs(out_dir,exist_ok=True)
    for i_r in top_res_idx:
        r_id=residue_ids[i_r]
        r_subdir=os.path.join(out_dir,f"res_{r_id}")
        os.makedirs(r_subdir,exist_ok=True)
        for j_a, aname in enumerate(angle_list):
            arr1=angles_perres_1[r_id][aname]
            arr2=angles_perres_2[r_id][aname]
            if arr1.size<1 and arr2.size<1:
                continue
            kl_val=kl_matrix[i_r,j_a]
            plt.figure(figsize=(6,4))
            arr1n=arr1[~np.isnan(arr1)]
            arr2n=arr2[~np.isnan(arr2)]
            sns.histplot(arr1n,bins=bins,stat='density',alpha=0.4,label='Traj1',
                         element='step',fill=True,binrange=(angle_min,angle_max))
            sns.histplot(arr2n,bins=bins,stat='density',alpha=0.4,label='Traj2',
                         element='step',fill=True,binrange=(angle_min,angle_max))
            plt.title(f"Residue {r_id}, {aname}, KL={kl_val:.3f}")
            plt.xlabel(f"{aname} (deg)")
            plt.ylabel("Density")
            plt.xlim([angle_min,angle_max])
            plt.grid(True,linestyle='--',alpha=0.5)
            plt.legend()
            plt.tight_layout()
            outpng=os.path.join(r_subdir,f"{aname}.png")
            plt.savefig(outpng,dpi=300,bbox_inches='tight')
            plt.close()

###############################################################################
# 8) MAIN
###############################################################################
def main():
    parser=argparse.ArgumentParser(description=(
        "Single script merging global distribution comparison (3 .h5) and pairwise per-res KL, "
        "with advanced 2D + 1D metrics: 1D KL, 1D JS, 1D WDist, 2D KL, 2D JS, optionally 2D WDist if pot installed."
    ))
    parser.add_argument("--condensed_json",type=str,required=True,
                        help="Path to condensed_residues.json")
    parser.add_argument("--h5_1",type=str,required=True,help="HDF5 file #1")
    parser.add_argument("--h5_2",type=str,required=True,help="HDF5 file #2")
    parser.add_argument("--h5_3",type=str,required=True,help="HDF5 file #3")
    parser.add_argument("--labels",type=str,nargs=3,default=["Set1","Set2","Set3"],
                        help="Labels for the 3 HDF5 datasets.")
    parser.add_argument("--out_dir",type=str,default="mega_output",help="Output directory.")
    parser.add_argument("--device",type=str,default="cpu",help="Device for PyTorch: 'cpu' or 'cuda'.")
    parser.add_argument("--chunk_size",type=int,default=500,help="Chunk size for angle extraction.")
    parser.add_argument("--hist_bins",type=int,default=36,help="Bins for 1D hist.")
    parser.add_argument("--angle_min",type=float,default=-180,help="Minimum angle range.")
    parser.add_argument("--angle_max",type=float,default=180,help="Maximum angle range.")
    parser.add_argument("--top_k",type=int,default=10,help="Number of top residues for distribution overlays.")
    args=parser.parse_args()

    os.makedirs(args.out_dir,exist_ok=True)

    # 1) Load condensed JSON
    cond_data=load_condensed_json(args.condensed_json)

    # 2) Load coords
    coords_dict={}
    h5_paths=[args.h5_1,args.h5_2,args.h5_3]
    for i, path in enumerate(h5_paths):
        lb=args.labels[i]
        coords_dict[lb]=load_h5_coords(path)
        logger.info(f"{lb}: shape => {coords_dict[lb].shape}")

    # 3) Gather angles
    angles_global={}
    angles_perres={}
    all_resids_global=None
    for lb in args.labels:
        g_dict, p_dict, c_list, r_ids = gather_global_and_perres_angles(
            coords_dict[lb], cond_data, device=args.device, chunk_size=args.chunk_size)
        angles_global[lb] = g_dict
        angles_perres[lb]= p_dict
        if all_resids_global is None:
            all_resids_global=r_ids

    # 4) Global distribution plotting
    global_compare_dir=os.path.join(args.out_dir,"global_compare")
    os.makedirs(global_compare_dir,exist_ok=True)
    indiv_dir=os.path.join(global_compare_dir,"individual")
    os.makedirs(indiv_dir,exist_ok=True)
    global_individual_plots(angles_global, args.labels, indiv_dir)
    comb_dir=os.path.join(global_compare_dir,"combined")
    os.makedirs(comb_dir,exist_ok=True)
    global_overlapped_plots(angles_global, args.labels, comb_dir)
    global_pdf=os.path.join(global_compare_dir,"Global_Distributions.pdf")
    compile_pdf([indiv_dir,comb_dir],global_pdf)
    logger.info(f"Global distribution plotting => {global_pdf}")

    ############################################################################
    # Extra 2D KL, 2D JS, optional 2D WDist
    ############################################################################
    def compute_2d_wasserstein(histA, histB, bins=72, amin=-180, amax=180):
        """
        2D Earth Mover's distance using pot (if installed).
        We build a cost matrix for bin centers in a (bins x bins) grid.
        """
        if not HAS_POT:
            return None  # or np.nan
        eps=1e-10
        p=histA.flatten()
        q=histB.flatten()
        sp=p.sum()
        sq=q.sum()
        if sp<eps or sq<eps:
            return np.nan
        p/=sp
        q/=sq
        p=np.where(p<eps,eps,p)
        q=np.where(q<eps,eps,q)

        # build cost matrix
        # for index i, j => centerCoord
        def centerCoord(i, bins, amin, amax):
            step=(amax-amin)/bins
            return amin+step*(i+0.5)
        coords=[]
        for i in range(bins):
            cx=centerCoord(i,bins,amin,amax)
            for j in range(bins):
                cy=centerCoord(j,bins,amin,amax)
                coords.append([cx,cy])
        coords=np.array(coords)
        costMat=np.zeros((bins*bins,bins*bins),dtype=float)
        for idxA in range(bins*bins):
            dx=coords[idxA]
            for idxB in range(bins*bins):
                dy=coords[idxB]
                costMat[idxA,idxB]=np.sqrt((dx[0]-dy[0])**2+(dx[1]-dy[1])**2)
        import ot
        G=ot.emd(p,q,costMat)
        w2d=np.sum(G*costMat)
        return w2d

    def global_pairwise_metrics(lbA, lbB):
        logger.info(f"\n=== Extra metrics for {lbA} vs {lbB} ===")
        arrA=angles_global[lbA]
        arrB=angles_global[lbB]
        angle_union=set(arrA.keys()).union(set(arrB.keys()))
        angle_list=sorted(list(angle_union))

        # We'll store results in CSV => "Global_Metrics_lbA_vs_lbB.csv"
        out_csv = os.path.join(args.out_dir, f"Global_Metrics_{lbA}_vs_{lbB}.csv")
        import csv
        with open(out_csv,"w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["Angle","KL_1D","JS_1D","WDist_1D"])
            for aname in angle_list:
                if aname not in arrA or aname not in arrB:
                    writer.writerow([aname,"N/A","N/A","N/A"])
                    continue
                dA=arrA[aname]
                dB=arrB[aname]
                kl_val=hist_kl_1d(dA,dB,args.hist_bins,args.angle_min,args.angle_max)
                js_val=compute_1d_js(dA,dB,args.hist_bins,args.angle_min,args.angle_max)
                wd_val=compute_1d_wasserstein(dA,dB)
                def fm(x):
                    return f"{x:.6f}" if not np.isnan(x) else "N/A"
                writer.writerow([aname,fm(kl_val),fm(js_val),fm(wd_val)])
        
        # Now 2D hist for phi,psi => do 2D KL, 2D JS, optional 2D WDist
        if "phi" in arrA and "psi" in arrA and "phi" in arrB and "psi" in arrB:
            # build 2D hist
            h2dA=compute_2d_hist(arrA["phi"], arrA["psi"], 72, args.angle_min,args.angle_max)
            h2dB=compute_2d_hist(arrB["phi"], arrB["psi"], 72, args.angle_min,args.angle_max)
            kl2d=compute_2d_kl(h2dA,h2dB)  # p->q
            js2d=js_2d(h2dA,h2dB)
            if HAS_POT:
                w2d=compute_2d_wasserstein(h2dA,h2dB,72,args.angle_min,args.angle_max)
            else:
                w2d=None
            # append to the CSV
            with open(out_csv,"a",newline="") as f2:
                wr=csv.writer(f2)
                wr.writerow([])
                wr.writerow(["2D_Rama_KL(A->B)",f"{kl2d:.6f}" if not np.isnan(kl2d) else "N/A"])
                wr.writerow(["2D_Rama_JS",f"{js2d:.6f}" if not np.isnan(js2d) else "N/A"])
                if w2d is not None:
                    wr.writerow(["2D_Rama_Wasserstein", f"{w2d:.6f}" if not np.isnan(w2d) else "N/A"])
                else:
                    wr.writerow(["2D_Rama_Wasserstein","N/A"])

            logger.info(f"   2D KL(A->B)={kl2d:.6f}, 2D JS={js2d:.6f}, 2D Wass={w2d}")

        logger.info(f"Global metrics for {lbA} vs {lbB} => {out_csv}")

    pairs=[(0,1),(0,2),(1,2)]
    for iA,iB in pairs:
        lbA=args.labels[iA]
        lbB=args.labels[iB]
        global_pairwise_metrics(lbA, lbB)

    # 6) Pairwise KL detailed
    def do_kl_for_pair(lbA, lbB):
        pair_dir=os.path.join(args.out_dir,f"KL_{lbA}_vs_{lbB}")
        os.makedirs(pair_dir,exist_ok=True)
        anglesA=angles_perres[lbA]
        anglesB=angles_perres[lbB]
        angle_union=set()
        for rid in anglesA.keys():
            angle_union.update(list(anglesA[rid].keys()))
        for rid in anglesB.keys():
            angle_union.update(list(anglesB[rid].keys()))
        angle_list=sorted(list(angle_union))
        residue_ids=sorted(list(anglesA.keys()))

        kl_matrix=np.zeros((len(residue_ids), len(angle_list)),dtype=float)
        kl_matrix.fill(np.nan)

        for i_r, rid in enumerate(residue_ids):
            for j_a, aname in enumerate(angle_list):
                arr1=anglesA[rid].get(aname,np.array([]))
                arr2=anglesB[rid].get(aname,np.array([]))
                if arr1.size<2 or arr2.size<2:
                    kl_matrix[i_r,j_a]=np.nan
                else:
                    val=hist_kl_1d(arr1,arr2,
                                   bins=args.hist_bins,
                                   angle_min=args.angle_min,
                                   angle_max=args.angle_max)
                    kl_matrix[i_r,j_a]=val

        # chunked heatmap
        heat_dir=os.path.join(pair_dir,"heatmaps")
        chunked_kl_heatmaps(kl_matrix,angle_list,residue_ids,args.chunk_size,heat_dir)
        # summaries
        top_res_idx, top_ang_idx=kl_summaries(kl_matrix,angle_list,residue_ids,
                                              top_k=args.top_k,out_dir=pair_dir)
        # csv
        kl_csv=os.path.join(pair_dir,"kl_data.csv")
        export_kl_csv(kl_matrix,angle_list,residue_ids,kl_csv)
        # distribution overlay
        detail_dir=os.path.join(pair_dir,"detailed_overlays")
        distribution_overlay_topres(top_res_idx, kl_matrix, residue_ids, angle_list,
                                    anglesA, anglesB, detail_dir,
                                    bins=args.hist_bins,
                                    angle_min=args.angle_min,
                                    angle_max=args.angle_max)
        # compile => single PDF
        pdf_path=os.path.join(pair_dir,f"KL_{lbA}_vs_{lbB}.pdf")
        compile_pdf([pair_dir],pdf_path)
        logger.info(f"Finished pairwise KL => {lbA} vs {lbB}, PDF => {pdf_path}")

    do_kl_for_pair(args.labels[0],args.labels[1])
    do_kl_for_pair(args.labels[0],args.labels[2])
    do_kl_for_pair(args.labels[1],args.labels[2])

    logger.info("All tasks complete. Merged functionality with 2D KL, 2D JS, 2D WDist done.")

if __name__=="__main__":
    main()
