import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
import os


def create_pca_animation(
    mean_pdb_path,
    pca_eigenvectors_csv,
    pca_projections_csv,
    pc_number,
    output_pdb_path,
    frames=30,
):
    """
    Generates a multi-model PDB file animating the motion along a principal component,
    with the amplitude determined by the most extreme structures in the dataset. It also
    preserves HELIX and SHEET records from the mean PDB file.

    Args:
        mean_pdb_path (str): Path to the PDB file representing the mean structure.
                             Your original target PDB file is a good choice.
        pca_eigenvectors_csv (str): Path to the pca_components.csv file (the eigenvectors).
        pca_projections_csv (str): Path to the principal_components.csv file (the projection data).
        pc_number (int): The principal component to visualize (e.g., 1 for PC1).
        output_pdb_path (str): Path to save the output multi-model PDB animation.
        frames (int): Number of frames to generate in the animation.
    """
    print(f"--- Starting PCA Animation Generation for PC{pc_number} ---")

    # --- 1. Load Data ---
    print(f"Loading mean structure from: {mean_pdb_path}")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("mean_structure", mean_pdb_path)

    # Isolate backbone atoms which were used in the PCA
    backbone_atoms = [
        atom for atom in structure.get_atoms() if atom.name in ["N", "CA", "C", "O"]
    ]
    if not backbone_atoms:
        print("Error: No backbone atoms (N, CA, C, O) found in the mean PDB file.")
        return

    mean_coords = np.array([atom.get_coord() for atom in backbone_atoms])

    print(f"Loading eigenvectors from: {pca_eigenvectors_csv}")
    # The first row is the header, so we skip it
    eigenvectors = np.loadtxt(pca_eigenvectors_csv, delimiter=",", skiprows=1)
    pc_eigenvector = eigenvectors[pc_number - 1]

    # Reshape the flat eigenvector to a per-atom (N_atoms, 3) array
    pc_eigenvector_reshaped = pc_eigenvector.reshape(len(backbone_atoms), 3)

    # --- 2. Determine Data-Driven Animation Range ---
    print(f"Loading PCA projections from: {pca_projections_csv}")
    projections_df = pd.read_csv(pca_projections_csv)
    pc_column = f"PC{pc_number}"

    if pc_column not in projections_df.columns:
        print(
            f"Error: {pc_column} not found in {pca_projections_csv}. Please check the PC number."
        )
        return

    min_projection = projections_df[pc_column].min()
    max_projection = projections_df[pc_column].max()
    print(
        f"Animation range for PC{pc_number} determined from data: [{min_projection:.2f}, {max_projection:.2f}]"
    )

    # --- 3. Preserve Secondary Structure Records ---
    print("Extracting HELIX and SHEET records from mean PDB...")
    ss_records = []
    with open(mean_pdb_path, "r") as f:
        for line in f:
            if line.startswith(("HELIX", "SHEET")):
                ss_records.append(line)

    # --- 4. Generate the Animation PDB File ---
    # Define a selection class to write only the backbone atoms, ensuring consistency
    class BackboneSelect(Select):
        def accept_atom(self, atom):
            return atom.name in ["N", "CA", "C", "O"]

    io = PDBIO()
    io.set_structure(structure)

    print(f"Generating {frames}-frame PDB movie at: {output_pdb_path}")
    with open(output_pdb_path, "w") as pdb_file:
        # Write the preserved secondary structure records to the header
        if ss_records:
            pdb_file.writelines(ss_records)

        # Create a smooth interpolation between the most extreme points
        for i, factor in enumerate(np.linspace(min_projection, max_projection, frames)):
            pdb_file.write(f"MODEL        {i+1}\n")

            # Calculate new coordinates for this frame
            new_coords = mean_coords + factor * pc_eigenvector_reshaped

            # Update the coordinates of the actual Biopython Atom objects
            for j, atom in enumerate(backbone_atoms):
                atom.set_coord(new_coords[j])

            # Save the current state of the structure as a model
            io.save(pdb_file, BackboneSelect())
            pdb_file.write("ENDMDL\n")

    print("--- Animation generation complete! ---")


# --- Example Usage ---
if __name__ == "__main__":
    # Directory where your PCA output was saved
    RUN_OUTPUT_DIR = "PCA_output\\Multi_Align_Data=all_02"

    # Path to the reference PDB (used to define the starting structure and SS elements)
    MEAN_PDB = "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Best_PDB_Base\\wood_frog_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_000.pdb"
    PC_NUMBER = 1  # Change this to visualize a different principal component
    # --- Generate an animation for PC1 ---
    create_pca_animation(
        mean_pdb_path=MEAN_PDB,
        pca_eigenvectors_csv=os.path.join(
            RUN_OUTPUT_DIR, "raw_data/pca_components.csv"
        ),
        pca_projections_csv=os.path.join(
            RUN_OUTPUT_DIR, "raw_data/principal_components.csv"
        ),
        pc_number=PC_NUMBER,
        output_pdb_path=os.path.join(
            RUN_OUTPUT_DIR,
            f"graphs/PC1_animation_PC_{PC_NUMBER}_{RUN_OUTPUT_DIR[11:].split('_')[0]}.pdb",
        ),
        frames=30,
    )
