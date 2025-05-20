import numpy as np
import MDAnalysis as mda
import os

def read_and_reshape_npy(file_path, reshape_dim=(2191, 3)):
    """
    Load a .npy file and reshape it to (N, 2191, 3).

    Parameters:
    - file_path (str): Path to the .npy file.
    - reshape_dim (tuple): Desired shape for each frame (default: (2191, 3)).

    Returns:
    - reshaped_data (numpy.ndarray): Reshaped data array.
    """
    data = np.load(file_path)
    total_atoms = reshape_dim[0] * reshape_dim[1]
    if data.shape[0] != total_atoms:
        # Calculate N based on the total number of atoms
        N = data.shape[0] // reshape_dim[0]
        if data.shape[0] % reshape_dim[0] != 0:
            raise ValueError(f"Total number of atoms ({data.shape[0]}) is not divisible by {reshape_dim[0]}")
        reshaped_data = data.reshape(N, *reshape_dim)
    else:
        reshaped_data = data.reshape(1, *reshape_dim)
    return reshaped_data

def load_pdb(pdb_path):
    """
    Load the PDB file using MDAnalysis.

    Parameters:
    - pdb_path (str): Path to the PDB file.

    Returns:
    - u (MDAnalysis.Universe): MDAnalysis Universe object.
    """
    u = mda.Universe(pdb_path)
    return u

def generate_xtc(npy_file, pdb_file, output_xtc, reshape_dim=(2191, 3)):
    """
    Generate an XTC file from a reshaped .npy file and a PDB topology.

    Parameters:
    - npy_file (str): Path to the .npy file.
    - pdb_file (str): Path to the PDB topology file.
    - output_xtc (str): Path to the output XTC file.
    - reshape_dim (tuple): Shape to reshape the .npy data into (default: (2191, 3)).
    """
    # Step 1: Read and reshape the .npy file
    reshaped_data = read_and_reshape_npy(npy_file, reshape_dim)
    print(f"Reshaped data shape: {reshaped_data.shape}")  # (N, 2191, 3)

    # Step 2: Load the PDB file
    u = load_pdb(pdb_file)
    num_atoms = len(u.atoms)
    expected_atoms = reshape_dim[0]
    if num_atoms != expected_atoms:
        raise ValueError(f"Number of atoms in PDB ({num_atoms}) does not match reshape_dim ({expected_atoms})")
    print(f"PDB loaded with {num_atoms} atoms.")

    # Step 3: Prepare the writer for the XTC file
    with mda.Writer(output_xtc, n_atoms=num_atoms, multiframe=True) as writer:
        for i, frame in enumerate(reshaped_data):
            # Update atom positions
            u.atoms.positions = frame

            # Write the current frame to the XTC file
            writer.write(u.atoms)

            # Optional: Print progress every 100 frames
            if (i + 1) % 100 == 0:
                print(f"Written {i + 1} frames...")

    print(f"XTC file saved to: {output_xtc}")

# Example Usage
if __name__ == "__main__":
    npy_file = "input_test.npy"       # Path to the .npy file
    pdb_file = "heavy_chain.pdb"               # Path to the topology PDB file
    output_xtc = "test_input2.xtc"     # Path to the output XTC file

    # Ensure output directory exists
    output_dir = os.path.dirname(output_xtc)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate the XTC file
    generate_xtc(npy_file, pdb_file, output_xtc)
