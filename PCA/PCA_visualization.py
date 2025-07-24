import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Atom, Polypeptide
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from scipy.interpolate import splev, splprep
from multi_gene_PCA import RunParams

# --- Configuration ---

# Set up a logger instance. Will be configured in the main class.
logger = logging.getLogger("PCAVisualizer")

@dataclass
class ClusteringParams:
    """Parameters for clustering the PCA data."""

    range_pcs_for_clustering: List[int]
    min_cluster_size: int = 15

    def __post_init__(self):
        """Validates and clamps the PC range for spline interpolation compatibility."""
        if (
            not isinstance(self.range_pcs_for_clustering, list)
            or len(self.range_pcs_for_clustering) != 2
        ):
            raise ValueError(
                "range_pcs_for_clustering must be a list of two integers, e.g., [1, 5]."
            )

        start_pc, end_pc = self.range_pcs_for_clustering
        num_pcs = (end_pc - start_pc) + 1

        if num_pcs > 10:
            new_end_pc = start_pc + 9
            logger.warning(
                f"Spline interpolation is limited to 10 dimensions, but {num_pcs} were requested. "
                f"Clamping range for spline to PC{start_pc}-PC{new_end_pc}."
            )
            self.range_pcs_for_clustering[1] = new_end_pc

@dataclass
class VisualizationParams:
    """Parameters for creating graphs and animations."""

    pcs_to_plot_2d: Optional[List[List[int]]] = None
    pcs_to_plot_3d: Optional[List[int]] = None
    range_frames: tuple[int, int] = (50, 200)
    k_neighbors: int = 15
    cluster_start: int = 0
    cluster_end: int = 1

    def __post_init__(self):
        if self.pcs_to_plot_2d is None and self.pcs_to_plot_3d is None:
            raise ValueError(
                "At least one of pcs_to_plot_2d or pcs_to_plot_3d must be set."
            )

# --- Main Visualizer Class ---

class PCAVisualizer:
    def __init__(
        self,
        run_params: RunParams,
        clustering_params: ClusteringParams,
        viz_params: VisualizationParams,
    ):
        self.run_params = run_params
        self.clustering_params = clustering_params
        self.viz_params = viz_params

        self.run_path = Path(self.run_params.output_dir) / self.run_params.run_name
        self.raw_data_path = self.run_path / "raw_data"
        self.visuals_path = self.run_path / "visualizations"
        self.visuals_path.mkdir(exist_ok=True)

        self._setup_logging()

        self.pdb_path_map = self._create_pdb_path_map()

        self.is_data_loaded = False
        self.projections_df: pd.DataFrame = pd.DataFrame()
        self.eigenvectors: np.ndarray = np.array([])
        self.mean_coords: np.ndarray = np.array([])
        self.explained_variance: np.ndarray = np.array([])
        self.processing_metadata: Dict[str, Any] = {}
        self.ref_structure: Optional[Structure.Structure] = None

    def _setup_logging(self):
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        log_path = self.run_path / f"{self.run_params.run_name}.log"
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    def _create_pdb_path_map(self) -> Dict[str, str]:
        """Creates a map of PDB filenames to their full paths."""
        path_map = {}
        for directory in self.run_params.pdb_dirs:
            for p in Path(directory).glob("*.pdb"):
                path_map[p.name] = str(p)
        return path_map

    def _load_data(self):
        if self.is_data_loaded:
            return
        logger.info("Loading data from PCA run...")
        clustered_csv_path = (
            self.raw_data_path / "principal_components_with_clusters.csv"
        )
        base_csv_path = self.raw_data_path / "principal_components.csv"
        run_params_path = self.run_path / "run_parameters.json"

        try:
            self.projections_df = pd.read_csv(
                clustered_csv_path if clustered_csv_path.exists() else base_csv_path
            )
            self.eigenvectors = np.loadtxt(
                self.raw_data_path / "pca_components.csv", delimiter=","
            )
            self.explained_variance = np.loadtxt(
                self.raw_data_path / "explained_variance_ratio.csv", delimiter=","
            )
            self.mean_coords = np.load(self.raw_data_path / "mean_coords.npy")
            with open(self.raw_data_path / "processing_metadata.json", "r") as f:
                self.processing_metadata = json.load(f)

            # Load the original run parameters to get the atom selection criteria
            with open(run_params_path, "r") as f:
                run_params_data = json.load(f)
                self.atom_selection = run_params_data.get("alignment_params", {}).get(
                    "atom_selection", "ca"
                )
                logger.info(
                    f"Loaded atom selection criteria from original run: '{self.atom_selection}'"
                )

            self._generate_source_species_column()
            self.is_data_loaded = True
            logger.info("Data loaded successfully.")
        except FileNotFoundError as e:
            logger.error(
                f"Failed to load data: {e}. Ensure pca_calculator.py ran successfully.",
                exc_info=True,
            )
            raise

    def _generate_source_species_column(self):
        if "PDB_Name" not in self.projections_df.columns:
            return
        logger.info("Generating 'Source_Species' column from PDB names...")
        self.projections_df["Source_Species"] = (
            self.projections_df["PDB_Name"].str.split("_unrelaxed", n=1).str[0]
        )
        species_counts = self.projections_df["Source_Species"].value_counts()
        small_species = species_counts[species_counts < 100].index
        if not small_species.empty:
            logger.info(f"Grouping {len(small_species)} small species into 'misc'.")
            self.projections_df.loc[
                self.projections_df["Source_Species"].isin(small_species),
                "Source_Species",
            ] = "misc"

    def run_clustering(self):
        # This method is unchanged
        if not self.is_data_loaded:
            self._load_data()
        logger.info("Performing HDBSCAN clustering...")
        start_pc, end_pc = self.clustering_params.range_pcs_for_clustering
        pc_cols = [f"PC{i}" for i in range(start_pc, end_pc + 1)]
        clusterer = HDBSCAN(min_cluster_size=self.clustering_params.min_cluster_size)
        self.projections_df["cluster"] = clusterer.fit_predict(
            self.projections_df[pc_cols].values
        )
        n_clusters = len(set(self.projections_df["cluster"])) - (
            1 if -1 in self.projections_df["cluster"].unique() else 0
        )
        logger.info(f"Clustering complete. Found {n_clusters} clusters.")
        self.projections_df.to_csv(
            self.raw_data_path / "principal_components_with_clusters.csv", index=False
        )
        logger.info("Saved clustering results.")

    def generate_plots(self):
        # This method is unchanged
        if not self.is_data_loaded:
            self._load_data()
        if "cluster" not in self.projections_df.columns:
            self.run_clustering()
        plot_df = self.projections_df.copy()
        logger.info("--- Generating plots colored by CLUSTER ---")
        plot_df["Cluster_Label"] = [
            f"Cluster {l}" if l != -1 else "Noise" for l in plot_df["cluster"]
        ]
        self._create_plots_from_dataframe(
            plot_df, color_by="Cluster_Label", suffix="_by_cluster"
        )
        if "Source_Species" in plot_df.columns:
            logger.info("--- Generating plots colored by SOURCE SPECIES ---")
            self._create_plots_from_dataframe(
                plot_df, color_by="Source_Species", suffix="_by_species"
            )

    def _create_plots_from_dataframe(
        self, plot_df: pd.DataFrame, color_by: str, suffix: str
    ):
        # This method is unchanged
        if self.viz_params.pcs_to_plot_3d:
            pcs = self.viz_params.pcs_to_plot_3d
            pc_x, pc_y, pc_z = f"PC{pcs[0]}", f"PC{pcs[1]}", f"PC{pcs[2]}"
            title = f"3D PCA colored by {color_by.replace('_', ' ').title()}"
            fig = px.scatter_3d(
                plot_df,
                x=pc_x,
                y=pc_y,
                z=pc_z,
                color=color_by,
                hover_name="PDB_Name",
                title=title,
                labels={
                    pc_x: f"{pc_x} ({self.explained_variance[pcs[0]-1]:.2%})",
                    pc_y: f"{pc_y} ({self.explained_variance[pcs[1]-1]:.2%})",
                    pc_z: f"{pc_z} ({self.explained_variance[pcs[2]-1]:.2%})",
                },
            )
            fig.write_html(
                self.visuals_path / f"3d_plot_pc{pcs[0]}_{pcs[1]}_{pcs[2]}{suffix}.html"
            )
        if self.viz_params.pcs_to_plot_2d:
            for pcs in self.viz_params.pcs_to_plot_2d:
                pc_x, pc_y = f"PC{pcs[0]}", f"PC{pcs[1]}"
                title = f"2D PCA ({pc_x} vs {pc_y}) colored by {color_by.replace('_', ' ').title()}"
                fig = px.scatter(
                    plot_df,
                    x=pc_x,
                    y=pc_y,
                    color=color_by,
                    hover_name="PDB_Name",
                    title=title,
                    labels={
                        pc_x: f"{pc_x} ({self.explained_variance[pcs[0]-1]:.2%})",
                        pc_y: f"{pc_y} ({self.explained_variance[pcs[1]-1]:.2%})",
                    },
                )
                fig.write_html(
                    self.visuals_path / f"2d_plot_pc{pcs[0]}_{pcs[1]}{suffix}.html"
                )

    # --- NEW ANIMATION LOGIC ---

    def generate_animation(self):
        """Generates a full-atom PDB animation using a spline path and reconstruction."""
        if not self.is_data_loaded:
            self._load_data()
        assert self.projections_df is not None and not self.projections_df.empty
        if "cluster" not in self.projections_df.columns:
            logger.error(
                "Cannot animate without clustering data. Run `run_clustering()` first."
            )
            return

        # Step 1: Find non-linear path and interpolate with splines
        logger.info("Step 1: Finding non-linear path and interpolating with splines...")
        keyframe_projections = self._find_path_between_clusters()
        if not keyframe_projections:
            return

        num_frames = np.random.randint(
            self.viz_params.range_frames[0], self.viz_params.range_frames[1] + 1
        )
        interpolated_projections = self._interpolate_path_with_spline(
            keyframe_projections, num_frames
        )

        # Step 2: Reconstruct target core coordinates for each frame
        logger.info(
            f"Step 2: Reconstructing target core coordinates for {num_frames} frames..."
        )
        target_core_coords = self._reconstruct_coords_from_pcs(interpolated_projections)

        # Step 3: Build full-atom frames using reconstruction
        logger.info("Step 3: Building full-atom frames via reconstruction...")
        animation_frames = self._build_full_atom_frames(target_core_coords)

        # Step 4: Write animation to PDB
        logger.info("Step 4: Writing animation to PDB file...")
        self._write_animation_pdb(animation_frames)
        logger.info("Animation generation complete.")

    def _find_path_between_clusters(self) -> Optional[List[np.ndarray]]:
        # This method is largely unchanged, just used to get keyframes
        pc_cols = [f"PC{i}" for i in range(1, self.eigenvectors.shape[0] + 1)]
        all_projections = self.projections_df[pc_cols].values

        def find_centroid_index(cluster_id):
            indices = self.projections_df.index[
                self.projections_df["cluster"] == cluster_id
            ]
            if len(indices) == 0:
                return None
            cluster_points = all_projections[indices]
            centroid = np.mean(cluster_points, axis=0)
            return indices[cdist([centroid], cluster_points).argmin()]

        start_idx = find_centroid_index(self.viz_params.cluster_start)
        end_idx = find_centroid_index(self.viz_params.cluster_end)
        if start_idx is None or end_idx is None:
            logger.error(
                f"Could not find centroids for clusters {self.viz_params.cluster_start} or {self.viz_params.cluster_end}."
            )
            return None

        # Load the reference structure, which is the PDB for the starting centroid
        ref_pdb_name = self.projections_df.loc[start_idx, "PDB_Name"]
        parser = PDBParser(QUIET=True)
        self.ref_structure = parser.get_structure(
            "ref", self.pdb_path_map[ref_pdb_name]
        )

        nn = NearestNeighbors(n_neighbors=self.viz_params.k_neighbors).fit(
            all_projections
        )
        knn_graph = nn.kneighbors_graph(mode="distance")
        distances, predecessors = dijkstra(
            csgraph=knn_graph,
            directed=False,
            indices=start_idx,
            return_predecessors=True,
        )
        if np.isinf(distances[end_idx]):
            logger.error("No path found. Try increasing k_neighbors.")
            return None
        path_indices = []
        curr = end_idx
        while curr != start_idx and curr != -9999:
            path_indices.append(curr)
            curr = predecessors[curr]
        path_indices.append(start_idx)
        path_indices.reverse()
        return [all_projections[i] for i in path_indices]

    def _interpolate_path_with_spline(
        self, path_projections: List[np.ndarray], num_frames: int
    ) -> np.ndarray:
        """
        Uses B-spline interpolation for the PCs specified in clustering_params and
        linear interpolation for the rest.
        """
        path_projections = np.array(path_projections)
        num_keyframes, num_total_pcs = path_projections.shape

        # Get the range of PCs for spline from the validated parameters
        start_pc_1based, end_pc_1based = self.clustering_params.range_pcs_for_clustering

        # Convert from 1-based PC number to 0-based index for array slicing
        spline_indices = list(range(start_pc_1based - 1, end_pc_1based))
        all_indices = list(range(num_total_pcs))
        linear_indices = [i for i in all_indices if i not in spline_indices]

        # Create a parameter for interpolation based on cumulative distance along the path
        distances = np.linalg.norm(np.diff(path_projections, axis=0), axis=1)
        u_keyframes = np.insert(np.cumsum(distances), 0, 0)
        if u_keyframes[-1] == 0:
            u_keyframes[-1] = 1.0  # Avoid division by zero if path is a single point
        u_keyframes /= u_keyframes[-1]  # Normalize to 0-1 range

        u_interp = np.linspace(0, 1, num_frames)
        interp_path = np.zeros((num_frames, num_total_pcs))

        # --- Spline interpolation for the selected PCs ---
        if spline_indices:
            projections_spline = path_projections[:, spline_indices].T
            # The number of keyframes must be > k for splines
            k = min(3, num_keyframes - 1)
            if k > 0:
                # s=0 ensures the spline passes through all points
                tck, _ = splprep(projections_spline, u=u_keyframes, s=0, k=k)
                interp_spline_T = splev(u_interp, tck)
                interp_path[:, spline_indices] = np.array(interp_spline_T).T
            else:  # Fallback to linear interpolation if not enough points for a spline
                logger.warning(
                    "Not enough keyframes for spline interpolation, falling back to linear."
                )
                for col_idx in spline_indices:
                    interp_path[:, col_idx] = np.interp(
                        u_interp, u_keyframes, path_projections[:, col_idx]
                    )

        # --- Linear interpolation for all other PCs ---
        if linear_indices:
            for col_idx in linear_indices:
                interp_path[:, col_idx] = np.interp(
                    u_interp, u_keyframes, path_projections[:, col_idx]
                )

        return interp_path

    def _reconstruct_coords_from_pcs(self, projections: np.ndarray) -> np.ndarray:
        reconstructed_dev = projections @ self.eigenvectors
        flat_mean = self.mean_coords.flatten()
        reconstructed_flat = flat_mean + reconstructed_dev
        return reconstructed_flat.reshape(-1, len(flat_mean) // 3, 3)

    def _kabsch_transform(
        self, P: np.ndarray, Q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        H = P_centered.T @ Q_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_Q - R @ centroid_P
        return R, t

    def _build_full_atom_frames(
        self, target_core_coords: np.ndarray
    ) -> List[Structure.Structure]:
        """Builds a list of full-atom Structure objects for the animation."""
        assert self.ref_structure is not None

        ref_chain = next(self.ref_structure.get_models()).get_list()[0]
        full_sequence_residues = [
            res for res in ref_chain.get_residues() if Polypeptide.is_aa(res)
        ]

        # Get the correct map from aligned index -> original index
        ref_sequence_str = "".join(
            [
                Polypeptide.three_to_one(res.get_resname())
                for res in full_sequence_residues
            ]
        )
        residue_map = self.processing_metadata["residue_maps"].get(ref_sequence_str)
        if not residue_map:
            logger.error(
                f"Could not find sequence for reference structure {self.ref_structure.id} in residue maps."
            )
            return []
        aligned_to_orig_map = {v: int(k) for k, v in residue_map.items()}

        # Get the list of core residues from the reference structure in the correct order
        core_indices = sorted(list(self.processing_metadata["core_filter_indices"]))
        core_residues_in_ref = []
        for aligned_idx in core_indices:
            original_residue_idx = aligned_to_orig_map.get(aligned_idx)
            if original_residue_idx is not None and original_residue_idx < len(
                full_sequence_residues
            ):
                core_residues_in_ref.append(
                    full_sequence_residues[original_residue_idx]
                )

        # Determine which atoms to use based on the original PCA run
        atom_list = ["CA"] if self.atom_selection == "ca" else ["N", "CA", "C", "O"]

        # Gather the original coordinates for the core atoms from the reference structure
        original_core_coords_list = []
        for res in core_residues_in_ref:
            for atom_name in atom_list:
                if atom_name in res:
                    original_core_coords_list.append(res[atom_name].get_coord())
                else:
                    logger.error(
                        f"Core atom {atom_name} missing from residue {res.get_id()} in reference PDB. Cannot build animation."
                    )
                    return []
        original_core_coords = np.array(original_core_coords_list)

        animation_frames = []
        for frame_idx in range(target_core_coords.shape[0]):
            new_frame = Structure.Structure(f"frame_{frame_idx}")
            new_model = Model.Model(frame_idx)
            new_chain = Chain.Chain("A")
            new_model.add(new_chain)
            new_frame.add(new_model)

            target_core_for_frame = target_core_coords[frame_idx]

            # Ensure the shapes match before Kabsch alignment
            if original_core_coords.shape != target_core_for_frame.shape:
                logger.error(
                    f"Shape mismatch for Kabsch alignment in frame {frame_idx}. "
                    f"Original core: {original_core_coords.shape}, "
                    f"Target core: {target_core_for_frame.shape}. Skipping animation."
                )
                return []

            R_global, t_global = self._kabsch_transform(
                original_core_coords, target_core_for_frame
            )

            for res in full_sequence_residues:
                new_res = res.copy()
                for atom in new_res.get_atoms():
                    atom.transform(R_global, t_global)
                new_chain.add(new_res)

            animation_frames.append(new_frame)
        return animation_frames

    def _write_animation_pdb(self, animation_frames: List[Structure.Structure]):
        """Writes a list of Structure objects into a multi-MODEL PDB file."""
        filename = f"animation_c{self.viz_params.cluster_start}_to_c{self.viz_params.cluster_end}.pdb"
        filepath = self.visuals_path / filename

        pdb_io = PDBIO()
        with open(filepath, "w") as f:
            for i, frame_struct in enumerate(animation_frames):
                f.write(f"MODEL        {i+1}\n")
                pdb_io.set_structure(frame_struct)
                pdb_io.save(f)
                f.write("ENDMDL\n")
        logger.info(f"Animation saved to {filepath}")
