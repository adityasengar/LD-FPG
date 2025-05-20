#!/usr/bin/env python

import sys
import os
import json
import numpy as np
from tqdm import tqdm

try:
    import MDAnalysis as mda
except ImportError:
    print("Please install MDAnalysis (e.g., pip install MDAnalysis)")
    sys.exit(1)


class Residue:
    def __init__(self, res_num, resname, heavy_atoms, torsion_atoms):
        """
        Initialize a Residue instance with static information.

        Parameters:
        - res_num: Residue number.
        - resname: Residue name.
        - heavy_atoms: List of heavy atoms with indices, names, backbone flags, and coordinates.
        - torsion_atoms: Dictionary containing atom indices for each torsion angle.
        """
        self.res_num = res_num
        self.resname = resname
        self.heavy_atoms = heavy_atoms
        self.heavy_atom_indices = np.array([atom['index'] for atom in heavy_atoms])
        self.heavy_atom_coords_per_frame = []  # Will store coordinates per frame
        self.torsion_atoms = torsion_atoms
        self.dihedral_angles = {
            'phi': [],
            'psi': [],
            'chi': {}
        }

    def calculate_dihedrals(self, coords):
        """
        Calculate dihedral angles for the residue given atom coordinates.

        Parameters:
        - coords: Atom coordinates for the entire system at the current time step.
        """
        # Backbone dihedrals
        for angle_name in ['phi', 'psi']:
            atom_indices = self.torsion_atoms.get(angle_name)
            if atom_indices and None not in atom_indices:
                atom_coords = coords[atom_indices].reshape(1, 4, 3)
                angle = calc_dihedral_angles(atom_coords)
                self.dihedral_angles[angle_name].append(angle[0])
            else:
                self.dihedral_angles[angle_name].append(np.nan)

        # Side-chain dihedrals
        chi_angles = self.torsion_atoms.get('chi') or {}
        for chi_name, atom_indices in chi_angles.items():
            if atom_indices and None not in atom_indices:
                atom_coords = coords[atom_indices].reshape(1, 4, 3)
                angle = calc_dihedral_angles(atom_coords)
                if chi_name not in self.dihedral_angles['chi']:
                    self.dihedral_angles['chi'][chi_name] = []
                self.dihedral_angles['chi'][chi_name].append(angle[0])
            else:
                if chi_name not in self.dihedral_angles['chi']:
                    self.dihedral_angles['chi'][chi_name] = []
                self.dihedral_angles['chi'][chi_name].append(np.nan)

        # Store heavy atom coordinates for the current frame
        heavy_atom_coords = coords[self.heavy_atom_indices]
        self.heavy_atom_coords_per_frame.append(heavy_atom_coords.copy())


def process_trajectory(pdb_file, traj_file, rotamers, higher_order='Non'):
    """
    Process the trajectory file and calculate dihedral angles for specified residues.

    Parameters:
    - pdb_file: Path to the PDB file.
    - traj_file: Path to the trajectory file (.dcd, .xtc, etc.).
    - rotamers: List of residue numbers to process.
    - higher_order: If 'all', calculates higher-order side-chain dihedrals.

    Returns:
    - residues: Dictionary of Residue instances with calculated dihedral angles.
    """
    # Load PDB data
    pdb = PDB(pdb_file)

    # Extract static information
    residues = {}
    resseq_to_indices = build_resseq_to_indices(pdb)

    for res_num in rotamers:
        if res_num not in resseq_to_indices:
            continue

        res_indices = resseq_to_indices[res_num]
        resname = pdb.resname[res_indices[0]]

        # Extract heavy atoms
        heavy_atoms = extract_heavy_atoms(pdb, res_indices)

        # Extract torsion atoms
        torsion_atoms = {}
        # Backbone torsion atoms
        phi_atoms = get_phi_atoms(pdb, resseq_to_indices, res_num)
        psi_atoms = get_psi_atoms(pdb, resseq_to_indices, res_num)
        torsion_atoms['phi'] = phi_atoms
        torsion_atoms['psi'] = psi_atoms

        # Side-chain torsion atoms
        chi_atoms = get_chi_atoms(pdb, resname, res_indices, higher_order)
        torsion_atoms['chi'] = chi_atoms if chi_atoms is not None else {}

        # Create Residue instance
        residue = Residue(res_num, resname, heavy_atoms, torsion_atoms)
        residues[res_num] = residue

    # Process trajectory
    # Load trajectory data
    traj_coords = load_trajectory(pdb_file, traj_file)

    # Iterate over frames
    num_frames = traj_coords.shape[0]
    for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
        coords = traj_coords[frame_idx]

        # Calculate dihedrals for each residue
        for residue in residues.values():
            residue.calculate_dihedrals(coords)

    return residues


# Helper functions

def build_resseq_to_indices(pdb):
    """
    Build a mapping from residue sequence numbers to their atom indices in the pdb arrays.
    """
    resseq_to_indices = {}
    unique_resseq = np.unique(pdb.resseq)
    for resseq in unique_resseq:
        indices = np.where(pdb.resseq == resseq)[0]
        resseq_to_indices[resseq] = indices
    return resseq_to_indices

def extract_heavy_atoms(pdb, res_indices):
    """
    Extracts heavy atoms (non-hydrogen atoms) for a given residue.
    """
    heavy_atom_mask = ~np.isin(pdb.element[res_indices], ['H'])  # Exclude hydrogen atoms
    heavy_atom_indices = res_indices[heavy_atom_mask]
    heavy_atom_names = pdb.name[heavy_atom_indices]
    heavy_atom_coords = pdb.coords[heavy_atom_indices]
    backbone_atoms = {'N', 'CA', 'C', 'O', 'OXT'}
    backbone_flags = np.isin(heavy_atom_names, list(backbone_atoms))
    heavy_atoms = []
    for idx, name, is_backbone, coord in zip(heavy_atom_indices, heavy_atom_names, backbone_flags, heavy_atom_coords):
        heavy_atoms.append({
            'index': idx,
            'name': name,
            'is_backbone': is_backbone,
            'coords': coord
        })
    return heavy_atoms

def get_phi_atoms(pdb, resseq_to_indices, res_num):
    """
    Get atom indices for phi angle calculation.
    """
    if (res_num - 1) in resseq_to_indices:
        prev_res_indices = resseq_to_indices[res_num - 1]
        C_prev = select_atom(pdb, prev_res_indices, 'C')
        N_i = select_atom(pdb, resseq_to_indices[res_num], 'N')
        CA_i = select_atom(pdb, resseq_to_indices[res_num], 'CA')
        C_i = select_atom(pdb, resseq_to_indices[res_num], 'C')
        if None not in [C_prev, N_i, CA_i, C_i]:
            return [C_prev, N_i, CA_i, C_i]
    return None

def get_psi_atoms(pdb, resseq_to_indices, res_num):
    """
    Get atom indices for psi angle calculation.
    """
    if (res_num + 1) in resseq_to_indices:
        N_i = select_atom(pdb, resseq_to_indices[res_num], 'N')
        CA_i = select_atom(pdb, resseq_to_indices[res_num], 'CA')
        C_i = select_atom(pdb, resseq_to_indices[res_num], 'C')
        N_next = select_atom(pdb, resseq_to_indices[res_num + 1], 'N')
        if None not in [N_i, CA_i, C_i, N_next]:
            return [N_i, CA_i, C_i, N_next]
    return None

def get_chi_atoms(pdb, resname, res_indices, higher_order):
    """
    Get atom indices for chi angle calculations.
    """
    chi_atoms = {}
    side_chain_atoms = get_side_chain_atoms(resname)
    if side_chain_atoms is None:
        return {}
    # Chi1
    N = select_atom(pdb, res_indices, 'N')
    CA = select_atom(pdb, res_indices, 'CA')
    CB = select_atom(pdb, res_indices, 'CB')
    side_atom = select_atom(pdb, res_indices, side_chain_atoms['chi1'])
    if None not in [N, CA, CB, side_atom]:
        chi_atoms['chi1'] = [N, CA, CB, side_atom]
    else:
        chi_atoms['chi1'] = None

    # Higher-order chis
    if higher_order == 'all':
        for chi_num in range(2, 6):
            chi_key = f'chi{chi_num}'
            atom_names = side_chain_atoms.get(chi_key)
            if atom_names:
                atom_indices = []
                for atom_name in atom_names:
                    atom_idx = select_atom(pdb, res_indices, atom_name)
                    if atom_idx is None:
                        break
                    atom_indices.append(atom_idx)
                else:
                    chi_atoms[chi_key] = atom_indices
                    continue
            chi_atoms[chi_key] = None

    return chi_atoms

def load_trajectory(pdb_file, traj_file):
    """
    Load trajectory data from a .dcd or .xtc file.

    Parameters:
    - pdb_file: Path to the PDB file.
    - traj_file: Path to the trajectory file.

    Returns:
    - traj_coords: NumPy array of shape (num_frames, num_atoms, 3)
    """
    # Create an MDAnalysis Universe
    u = mda.Universe(pdb_file, traj_file)
    num_frames = len(u.trajectory)
    num_atoms = len(u.atoms)
    traj_coords = np.zeros((num_frames, num_atoms, 3), dtype=np.float32)

    for ts in tqdm(u.trajectory, desc="Loading trajectory"):
        # ts.frame starts at 0 in MDAnalysis, so we use ts.frame directly
        traj_coords[ts.frame] = ts.positions

    return traj_coords

def select_atom(pdb, indices, atom_name):
    """
    Selects the index of an atom by name within given indices.
    """
    atom_sel = pdb.name[indices] == atom_name
    if np.any(atom_sel):
        return indices[atom_sel][0]
    else:
        return None

def calc_dihedral_angles(coords):
    """
    Calculates dihedral angles from coordinates.

    Parameters:
    - coords: Array of shape (1, 4, 3)

    Returns:
    - angles: Array of dihedral angles in degrees
    """
    angles = calc_dihedral(coords) * 180.0 / np.pi
    return angles

def calc_dihedral(coords):
    """
    Calculate dihedral angles for a set of coordinates.

    Parameters:
    - coords: Array of shape (1, 4, 3)

    Returns:
    - angles: Array of dihedral angles in radians
    """
    b1 = coords[:, 1, :] - coords[:, 0, :]
    b2 = coords[:, 2, :] - coords[:, 1, :]
    b3 = coords[:, 3, :] - coords[:, 2, :]

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    b2_norm = b2 / np.linalg.norm(b2, axis=1)[:, np.newaxis]

    m1 = np.cross(n1, b2_norm)

    x = np.einsum('ij,ij->i', n1, n2)
    y = np.einsum('ij,ij->i', m1, n2)

    angles = np.arctan2(y, x)
    return angles

def get_side_chain_atoms(resname):
    """
    Returns a dictionary of side-chain atom names for chi angles based on residue name.
    """
    chi_atoms = {
        # Definitions for residues with side-chain chi angles
        'ARG': {
            'chi1': 'CG',
            'chi2': ['CA', 'CB', 'CG', 'CD'],
            'chi3': ['CB', 'CG', 'CD', 'NE'],
            'chi4': ['CG', 'CD', 'NE', 'CZ'],
            'chi5': ['CD', 'NE', 'CZ', 'NH1']
        },
        'ASN': {
            'chi1': 'CG',
            'chi2': ['CA', 'CB', 'CG', 'OD1']
        },
        'ASP': {
            'chi1': 'CG',
            'chi2': ['CA', 'CB', 'CG', 'OD1']
        },
        'CYS': {
            'chi1': 'SG'
        },
        'GLN': {
            'chi1': 'CG',
            'chi2': ['CA', 'CB', 'CG', 'CD'],
            'chi3': ['CB', 'CG', 'CD', 'OE1']
        },
        'GLU': {
            'chi1': 'CG',
            'chi2': ['CA', 'CB', 'CG', 'CD'],
            'chi3': ['CB', 'CG', 'CD', 'OE1']
        },
        'HIS': {
            'chi1': 'CG',
            'chi2': ['CA', 'CB', 'CG', 'ND1']
        },
        'ILE': {
            'chi1': 'CG1',
            'chi2': ['CA', 'CB', 'CG1', 'CD1']
        },
        'LEU': {
            'chi1': 'CG',
            'chi2': ['CA', 'CB', 'CG', 'CD1']
        },
        'LYS': {
            'chi1': 'CG',
            'chi2': ['CA', 'CB', 'CG', 'CD'],
            'chi3': ['CB', 'CG', 'CD', 'CE'],
            'chi4': ['CG', 'CD', 'CE', 'NZ']
        },
        'MET': {
            'chi1': 'CG',
            'chi2': ['CA', 'CB', 'CG', 'SD'],
            'chi3': ['CB', 'CG', 'SD', 'CE']
        },
        'PHE': {
            'chi1': 'CG',
            'chi2': ['CA', 'CB', 'CG', 'CD1']
        },
        'PRO': {
            'chi1': 'CG',
            'chi2': ['CA', 'CB', 'CG', 'CD']
        },
        'SER': {
            'chi1': 'OG'
        },
        'THR': {
            'chi1': 'OG1'
        },
        'TRP': {
            'chi1': 'CG',
            'chi2': ['CA', 'CB', 'CG', 'CD1']
        },
        'TYR': {
            'chi1': 'CG',
            'chi2': ['CA', 'CB', 'CG', 'CD1']
        },
        'VAL': {
            'chi1': 'CG1'
        }
    }
    return chi_atoms.get(resname, None)


class PDB:
    def __init__(self, pdb_file):
        """
        Initialize PDB object by parsing a PDB file.

        Parameters:
        - pdb_file: Path to the PDB file.
        """
        serial = []
        name = []
        resname = []
        chainID = []
        resseq = []
        x = []
        y = []
        z = []
        element = []

        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    serial.append(int(line[6:11]))
                    name.append(line[12:16].strip())
                    resname.append(line[17:20].strip())
                    chainID.append(line[21].strip())
                    resseq.append(int(line[22:26]))
                    x.append(float(line[30:38]))
                    y.append(float(line[38:46]))
                    z.append(float(line[46:54]))
                    elem = line[76:78].strip()
                    if elem:
                        element.append(elem)
                    else:
                        # fallback: use the first characters from line[12:14]
                        element.append(line[12:14].strip())

        self.serial = np.array(serial)
        self.name = np.array(name)
        self.resname = np.array(resname)
        self.chainID = np.array(chainID)
        self.resseq = np.array(resseq)
        self.coords = np.column_stack((x, y, z))
        self.element = np.array(element)


def serialize_residues(residues):
    """
    Serialize the residues data into a JSON-serializable format.

    Parameters:
    - residues: Dictionary of Residue instances.

    Returns:
    - serialized_data: Dictionary with serialized data.
    """
    def convert_numpy_types(obj):
        """
        Recursively convert NumPy types to native Python types.
        """
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return convert_numpy_types(obj.tolist())
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif obj is np.nan:
            return None
        else:
            return obj

    serialized_data = {}
    for res_num, residue in residues.items():
        serialized_residue = {
            'res_num': int(residue.res_num),
            'resname': residue.resname,
            'heavy_atoms': [],
            'heavy_atom_indices': residue.heavy_atom_indices.tolist(),
            'heavy_atom_coords_per_frame': [
                coords.tolist() for coords in residue.heavy_atom_coords_per_frame
            ],
            'torsion_atoms': residue.torsion_atoms,  # Atom indices involved in torsion angles
            'dihedral_angles': residue.dihedral_angles
        }

        # Serialize heavy_atoms
        for atom in residue.heavy_atoms:
            serialized_atom = {
                'index': int(atom['index']),
                'name': atom['name'],
                'is_backbone': bool(atom['is_backbone']),
                'coords': [float(c) for c in atom['coords']]
            }
            serialized_residue['heavy_atoms'].append(serialized_atom)

        # Convert the entire serialized_residue to native Python types
        serialized_residue = convert_numpy_types(serialized_residue)
        # Store under string key (JSON keys must be strings)
        serialized_data[str(res_num)] = serialized_residue

    return serialized_data


if __name__ == "__main__":
    # Simple command-line argument handling
    if len(sys.argv) < 3:
        print("Usage: python extract_residues.py <pdb_file> <traj_file>")
        sys.exit(1)

    pdb_file = sys.argv[1]
    traj_file = sys.argv[2]

    # Output filename in the same directory:
    output_json = "residues_data_active_full.json"

    # You can adjust the residue range as needed
    # For example, here we pick 1 to 278, inclusive
    rotamers = list(range(1, 279))

    # Process the trajectory and compute dihedrals
    residues = process_trajectory(
        pdb_file=pdb_file,
        traj_file=traj_file,
        rotamers=rotamers,
        higher_order='all'  # 'all' for side-chain dihedrals up to chi5
    )

    # Serialize
    serialized_residues = serialize_residues(residues)

    # Write to JSON in the same folder
    with open(output_json, 'w') as f:
        json.dump(serialized_residues, f, indent=2)

    print(f"Finished! Wrote JSON data to {output_json}")
