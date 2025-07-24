import os
import multiprocessing
import logging
from Bio.PDB import PDBParser, Superimposer, PPBuilder
from Bio.PDB.PDBExceptions import PDBConstructionException
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# Configure a basic logger for console output.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("PDB_PCA_Script")

### --- HELPER FUNCTIONS FOR ROBUST ATOM MATCHING --- ###

def get_backbone_identifiers(structure):
    """
    Generates a list of stable identifiers (chain, residue, atom name) for backbone atoms.
    """
    identifiers = []
    for model in structure:
        for chain in model:
            if chain.id.isalpha() or ' ' in chain.id:
                for residue in chain:
                    if residue.get_id()[0] == " ":
                        for atom_name in ["N", "CA", "C", "O"]:
                            if atom_name in residue:
                                identifiers.append((chain.id, residue.id, atom_name))
    return identifiers


def get_atoms_from_identifiers(structure, identifiers):
    """
    Extracts a list of Atom objects from a structure based on a list of identifiers.
    """
    try:
        atom_lookup = {
            c.id: {
                r.id: {a.name: a for a in r} for r in c
            } for c in structure[0]
        }
    except Exception:
         raise ValueError(f"Could not build atom lookup table for {structure.id}. Ensure it is a valid structure.")

    selected_atoms = []
    missing_identifiers = []
    for chain_id, res_id, atom_name in identifiers:
        try:
            atom = atom_lookup[chain_id][res_id][atom_name]
            selected_atoms.append(atom)
        except KeyError:
            missing_identifiers.append((chain_id, res_id, atom_name))

    if missing_identifiers:
        missing_str = ", ".join([f"Chain {c} Res {r[1]} Atom {a}" for c, r, a in missing_identifiers[:5]])
        if len(missing_identifiers) > 5:
            missing_str += "..."
        raise ValueError(
            f"{len(missing_identifiers)} alignment atom(s) not found in {structure.id}. Examples: {missing_str}"
        )

    return selected_atoms

### --- CORE SCRIPT FUNCTIONS --- ###

def get_backbone_atoms(structure):
    """Extracts N, CA, C, O atoms for each residue in a protein structure."""
    backbone_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":
                    for atom_name in ["N", "CA", "C", "O"]:
                        if atom_name in residue:
                            backbone_atoms.append(residue[atom_name])
    return backbone_atoms


def get_dihedral_features(structure):
    """Extracts phi and psi dihedral angles from a protein structure."""
    ppb = PPBuilder()
    dihedrals = []
    for model in structure:
        for pp in ppb.build_peptides(model):
            phi_psi_list = pp.get_phi_psi_list()
            for phi, psi in phi_psi_list:
                dihedrals.append(phi if phi is not None else 0.0)
                dihedrals.append(psi if psi is not None else 0.0)
    return np.array(dihedrals)


def process_single_pdb_for_pca(args):
    """
    Processes a single PDB file for alignment and feature extraction.
    """
    (
        target_structure_path,
        pdb_file_path,
        directory_group,
        alignment_identifiers,
        coordinate_type,
    ) = args
    parser = PDBParser()
    pdb_name = os.path.basename(pdb_file_path)

    result = {
        "PDB Name": pdb_name,
        "Directory Group": directory_group,
        "Aligned Features": None,
        "Notes": "",
    }

    try:
        target_structure = parser.get_structure("target", target_structure_path)
        moving_structure = parser.get_structure(pdb_name, pdb_file_path)

        target_core_atoms = get_atoms_from_identifiers(target_structure, alignment_identifiers)
        moving_core_atoms = get_atoms_from_identifiers(moving_structure, alignment_identifiers)
        
        superimposer_core = Superimposer()
        superimposer_core.set_atoms(target_core_atoms, moving_core_atoms)
        superimposer_core.apply(moving_structure.get_atoms())

        if coordinate_type == "cartesian":
            transformed_backbone_atoms = get_backbone_atoms(moving_structure)
            aligned_features = np.array(
                [atom.get_coord() for atom in transformed_backbone_atoms]
            ).flatten()
            feature_type_note = "Cartesian backbone coordinates"
        elif coordinate_type == "dihedral":
            aligned_features = get_dihedral_features(moving_structure)
            feature_type_note = "Dihedral angles (phi, psi)"
        else:
            result["Notes"] = f"Error: Invalid coordinate_type '{coordinate_type}'."
            return result

        if aligned_features.size == 0:
            result["Notes"] = f"Error: Could not extract any {feature_type_note}."
            return result

        result["Aligned Features"] = aligned_features
        result["Notes"] = f"Successfully extracted aligned {feature_type_note}."
        return result

    except (PDBConstructionException, ValueError, Exception) as e:
        result["Notes"] = f"Error processing {pdb_name}: {e}"
        return result


def run_pca_script(
    output_base_directory,
    run_name,
    pdb_directories,
    target_pdb_path,
    coordinate_type="cartesian",
):
    """Main function to orchestrate the PDB structural PCA."""
    logger.info("Starting PDB Structural PCA script for run: %s", run_name)
    run_output_dir = os.path.join(output_base_directory, run_name)
    raw_data_dir = os.path.join(run_output_dir, "raw_data")
    graphs_dir = os.path.join(run_output_dir, "graphs")
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    logger.info("Output will be saved to: %s", run_output_dir)

    parser = PDBParser()
    try:
        logger.info("Generating stable backbone identifiers from target PDB: %s", target_pdb_path)
        target_structure = parser.get_structure("target_for_indices", target_pdb_path)
        alignment_identifiers = get_backbone_identifiers(target_structure)
        if not alignment_identifiers:
            logger.error("Could not extract any backbone identifiers from target PDB. Aborting.")
            return
        logger.info("Generated %d backbone atom identifiers for alignment.", len(alignment_identifiers))
    except Exception as e:
        logger.error(f"Failed to process target PDB for identifiers: {e}")
        return

    tasks = []
    all_pdb_files = set()
    for directory in pdb_directories:
        dir_name = os.path.basename(os.path.normpath(directory))
        for filename in os.listdir(directory):
            if filename.lower().endswith((".pdb", ".ent")):
                full_path = os.path.join(directory, filename)
                if full_path not in all_pdb_files:
                    tasks.append((target_pdb_path, full_path, dir_name, alignment_identifiers, coordinate_type))
                    all_pdb_files.add(full_path)
    
    target_dir_name = "Target"
    if target_pdb_path not in all_pdb_files:
        tasks.append((target_pdb_path, target_pdb_path, target_dir_name, alignment_identifiers, coordinate_type))

    logger.info("Found %d unique PDB files to process for PCA.", len(tasks))

    collected_pca_data = []
    try:
        with multiprocessing.Pool() as pool:
            processed_results = pool.map(process_single_pdb_for_pca, tasks)
        for res in processed_results:
            if res["Aligned Features"] is not None:
                collected_pca_data.append((res["PDB Name"], res["Directory Group"], res["Aligned Features"]))
            else:
                logger.warning("Skipping %s due to error: %s", res["PDB Name"], res["Notes"])
    except Exception as e:
        logger.error("An error occurred during multiprocessing: %s", e)
        return

    if len(collected_pca_data) < 2:
        logger.error("Not enough valid PDBs (%d) for PCA.", len(collected_pca_data))
        return

    pdb_names_for_pca = [item[0] for item in collected_pca_data]
    directory_groups_for_pca = [item[1] for item in collected_pca_data]
    pca_input_matrix = np.array([item[2] for item in collected_pca_data])

    logger.info("PCA input matrix shape: %s", pca_input_matrix.shape)
    pca = PCA(n_components=None)
    principal_components = pca.fit_transform(pca_input_matrix)
    logger.info("PCA performed successfully.")

    # --- Save Raw Data ---
    # Save transformed data (principal components for each PDB)
    pc_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(principal_components.shape[1])])
    pc_df["PDB_Name"] = pdb_names_for_pca
    pc_df["Source_Directory"] = directory_groups_for_pca
    pc_output_path = os.path.join(raw_data_dir, "principal_components.csv")
    pc_df.to_csv(pc_output_path, index=False)
    logger.info("Principal components saved to: %s", pc_output_path)

    # **FIXED**: Save the other raw data files
    # Save explained variance ratio
    explained_variance_path = os.path.join(raw_data_dir, "explained_variance_ratio.csv")
    np.savetxt(
        explained_variance_path,
        pca.explained_variance_ratio_,
        delimiter=",",
        header="Explained Variance Ratio",
        comments=""  # Prevents '#' from being added to the header
    )
    logger.info(f"Explained variance ratio saved to: {explained_variance_path}")

    # Save PCA components (eigenvectors) - CRUCIAL for the analysis script
    components_path = os.path.join(raw_data_dir, "pca_components.csv")
    np.savetxt(
        components_path,
        pca.components_,
        delimiter=",",
        header="PCA Components (Eigenvectors)",
        comments=""
    )
    logger.info(f"PCA components (eigenvectors) saved to: {components_path}")

    # --- Generate and Save Plots ---
    logger.info("Generating plots...")
    plot_df = pc_df.copy()
    plot_df["Is_Target"] = plot_df["PDB_Name"] == os.path.basename(target_pdb_path)
    
    # 2D Interactive Plot
    if principal_components.shape[1] >= 2:
        fig_2d = px.scatter(
            plot_df, x="PC1", y="PC2",
            color="Source_Directory", symbol="Is_Target",
            symbol_map={True: "star", False: "circle"},
            hover_name="PDB_Name",
            labels={"PC1": f"PC 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)", "PC2": f"PC 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)"},
            title="Interactive PCA: PC1 vs PC2",
        )
        fig_2d.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
        fig_2d.update_traces(marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')), selector=dict(symbol='star'))
        fig_2d.write_html(os.path.join(graphs_dir, "interactive_pc1_vs_pc2.html"))
        logger.info("Saved interactive 2D plot.")

    # 3D Interactive Plot
    if principal_components.shape[1] >= 3:
        fig_3d = px.scatter_3d(
            plot_df, x="PC1", y="PC2", z="PC3",
            color="Source_Directory", symbol="Is_Target",
            symbol_map={True: "diamond", False: "circle"},
            hover_name="PDB_Name",
            labels={"PC1": f"PC 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)", "PC2": f"PC 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)", "PC3": f"PC 3 ({pca.explained_variance_ratio_[2]*100:.2f}%)"},
            title="Interactive PCA: PC1 vs PC2 vs PC3",
        )
        fig_3d.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig_3d.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')), selector=dict(symbol='diamond'))
        fig_3d.write_html(os.path.join(graphs_dir, "interactive_pc1_pc2_pc3.html"))
        logger.info("Saved interactive 3D plot.")

    logger.info("Script finished successfully.")

if __name__ == "__main__":
    # --- Configuration ---
    # Base directory where all output folders will be created
    pca_output_base_dir = "PCA_output"

    # A unique name for this specific run, a subfolder will be created with this name
    # species abbreviation + coordinate type + method of alignment + description of what directories are included
    current_run_name = "NP_CT_DO_PDB_02"

    # --- List of Input Directories ---
    # The script will process PDBs from all directories in this list.
    # Each directory will be assigned a different color in the interactive plots.
    pdb_directory_paths = [
        # "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Penguin",
        # "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Penguin_w_dropout",
        # "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Bos_mutus_w_dropout",
        # "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Camelus_ferus_w_dropout",
        # "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Chrysemys_picta_bellii_w_dropout",
        # "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Dryophytes_japonicus_w_dropout",
        # "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Jaculus_jaculus_w_dropout",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Nanorana_parkeri_w_dropout",
    ]

    # Path to the reference structure for alignment.
    # This structure will be marked with a star in the plots.
    target_pdb_file = "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Best_PDB_Base\\Nanorana_parkeri_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_000.pdb"

    # --- PCA Settings ---
    # Choose coordinate type: 'cartesian' or 'dihedral'
    coordinate_type_for_pca = "cartesian"

    # --- Run Script ---
    # `align_on_backbone=True` is the key new feature. It automatically finds
    # the backbone atoms of the target_pdb_file and uses them for alignment.
    # You no longer need to manually specify atom indices for this common use case.
    run_pca_script(
        output_base_directory=pca_output_base_dir,
        run_name=current_run_name,
        pdb_directories=pdb_directory_paths,
        target_pdb_path=target_pdb_file,
        coordinate_type=coordinate_type_for_pca,

    )
