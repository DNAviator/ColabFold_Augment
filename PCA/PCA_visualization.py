import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
from Bio.PDB import PDBIO, Structure, Model, Chain, Atom
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from multi_gene_PCA import RunParams

# --- Configuration ---

# Set up a logger instance. Will be configured in the main class.
logger = logging.getLogger("PCAVisualizer")

@dataclass
class ClusteringParams:
    """Parameters for clustering the PCA data."""

    # Use a list [start_pc, end_pc], e.g., [1, 3] for PC1, PC2, PC3
    range_pcs_for_clustering: List[int]
    min_cluster_size: int = 15


@dataclass
class VisualizationParams:
    """Parameters for creating graphs and animations."""

    # e.g., [[1, 2], [1, 3]] for two 2D plots
    pcs_to_plot_2d: Optional[List[List[int]]] = None
    # e.g., [1, 2, 3] for one 3D plot
    pcs_to_plot_3d: Optional[List[int]] = None
    # Min/max frames for the animation
    range_frames: tuple[int, int] = (50, 200)
    # For pathfinding algorithm
    k_neighbors: int = 15
    # Cluster IDs for animation path
    cluster_start: int = 0
    cluster_end: int = 1

    def __post_init__(self):
        if self.pcs_to_plot_2d is None and self.pcs_to_plot_3d is None:
            raise ValueError("At least one of pcs_to_plot_2d or pcs_to_plot_3d must be set.")


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

        # --- Path Setup ---
        self.run_path = Path(self.run_params.output_dir) / self.run_params.run_name
        self.raw_data_path = self.run_path / "raw_data"
        self.visuals_path = self.run_path / "visualizations"
        self.visuals_path.mkdir(exist_ok=True)

        self._setup_logging()

        # --- Loaded Data ---
        self.is_data_loaded = False

        projections_df, eigenvectors, mean_coords, explained_variance = self._load_data()
        self.projections_df = projections_df
        self.eigenvectors = eigenvectors
        self.mean_coords = mean_coords
        self.core_atom_metadata = None
        self.explained_variance = explained_variance

    def _setup_logging(self):
        """Configures logging to stream to console and save to the run's log file."""
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Use the existing log file from the first script
        log_path = self.run_path / f"{self.run_params.run_name}.log"
        fh = logging.FileHandler(log_path, mode="a")  # Append to the log
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    def _load_data(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Loads all necessary data from the PCA calculation step."""
        assert self.is_data_loaded is False, "Data is already loaded. Call load_data() only once."
        try:
            logger.info("Loading data from PCA run...")
            projections_df = pd.read_csv(
                self.raw_data_path / "principal_components.csv"
            )
            eigenvectors = np.loadtxt(
                self.raw_data_path / "pca_components.csv", delimiter=","
            )
            explained_variance = np.loadtxt(
                self.raw_data_path / "explained_variance_ratio.csv", delimiter=","
            )

            # Load the consolidated metadata file
            with open(self.raw_data_path / "processing_metadata.json", "r") as f:
                metadata = json.load(f)
            # This assumes core atom metadata is needed. If so, it should be saved by the first script.
            # For now, we will assume it's not needed for the simplified animation.

            # This is a critical piece for animation. We need to ensure it's saved by pca_calculator.py
            mean_coords = np.load(self.raw_data_path / "mean_coords.npy")

            self.is_data_loaded = True
            logger.info("Data loaded successfully.")
            return (projections_df, eigenvectors, mean_coords, explained_variance)
        except FileNotFoundError as e:
            logger.error(
                f"Failed to load data: {e}. Make sure the pca_calculator script ran successfully."
            )
            raise

    def run_clustering(self):
        """Performs HDBSCAN clustering on the PCA data."""
        if not self.is_data_loaded:
            self._load_data()

        logger.info("Performing HDBSCAN clustering...")
        if not self.clustering_params.range_pcs_for_clustering:
            logger.warning("range_pcs_for_clustering not set. Defaulting to PC1-PC3.")
            self.clustering_params.range_pcs_for_clustering = [1, 3]

        start_pc, end_pc = self.clustering_params.range_pcs_for_clustering
        pc_cols = [f"PC{i}" for i in range(start_pc, end_pc + 1)]

        cluster_data = self.projections_df[pc_cols].values
        clusterer = HDBSCAN(min_cluster_size=self.clustering_params.min_cluster_size)

        self.projections_df["cluster"] = clusterer.fit_predict(cluster_data)

        n_clusters = len(set(self.projections_df["cluster"])) - (
            1 if -1 in self.projections_df["cluster"].unique() else 0
        )
        logger.info(f"Clustering complete. Found {n_clusters} clusters.")

        # Save the updated dataframe
        self.projections_df.to_csv(
            self.raw_data_path / "principal_components_with_clusters.csv", index=False
        )
        logger.info(
            "Saved clustering results to principal_components_with_clusters.csv"
        )

    def generate_plots(self):
        """Generates 2D and 3D scatter plots of the PCA data."""
        if not self.is_data_loaded:
            self._load_data()
        if "cluster" not in self.projections_df.columns:
            logger.warning(
                "No cluster data found. Running clustering with default parameters."
            )
            self.run_clustering()

        plot_df = self.projections_df.copy()
        plot_df["Display_Label"] = [
            f"Cluster {l}" if l != -1 else "Noise" for l in plot_df["cluster"]
        ]

        # Generate 3D plot
        if self.viz_params.pcs_to_plot_3d:
            pcs = self.viz_params.pcs_to_plot_3d
            pc_x, pc_y, pc_z = f"PC{pcs[0]}", f"PC{pcs[1]}", f"PC{pcs[2]}"
            logger.info(f"Generating 3D plot for {pc_x}, {pc_y}, {pc_z}...")

            fig = px.scatter_3d(
                plot_df,
                x=pc_x,
                y=pc_y,
                z=pc_z,
                color="Display_Label",
                hover_name="PDB_Name",
                title="3D PCA Clustering",
                labels={
                    pc_x: f"{pc_x} ({self.explained_variance[pcs[0]-1]:.2%})",
                    pc_y: f"{pc_y} ({self.explained_variance[pcs[1]-1]:.2%})",
                    pc_z: f"{pc_z} ({self.explained_variance[pcs[2]-1]:.2%})",
                },
            )
            fig.write_html(
                self.visuals_path / f"3d_plot_pc{pcs[0]}_{pcs[1]}_{pcs[2]}.html"
            )

        # Generate 2D plots
        if self.viz_params.pcs_to_plot_2d:
            for pcs in self.viz_params.pcs_to_plot_2d:
                pc_x, pc_y = f"PC{pcs[0]}", f"PC{pcs[1]}"
                logger.info(f"Generating 2D plot for {pc_x}, {pc_y}...")

                fig = px.scatter(
                    plot_df,
                    x=pc_x,
                    y=pc_y,
                    color="Display_Label",
                    hover_name="PDB_Name",
                    title=f"2D PCA Clustering ({pc_x} vs {pc_y})",
                    labels={
                        pc_x: f"{pc_x} ({self.explained_variance[pcs[0]-1]:.2%})",
                        pc_y: f"{pc_y} ({self.explained_variance[pcs[1]-1]:.2%})",
                    },
                )
                fig.write_html(self.visuals_path / f"2d_plot_pc{pcs[0]}_{pcs[1]}.html")

    def generate_animation(self):
        """Generates a PDB animation of the transition between two clusters."""
        if not self.is_data_loaded:
            self._load_data()
        if "cluster" not in self.projections_df.columns:
            logger.error(
                "Cannot generate animation without clustering data. Run `run_clustering()` first."
            )
            return

        # 1. Find path between cluster centroids in PC space
        logger.info("Finding shortest path between cluster centroids...")
        path_projections = self._find_path_between_clusters()
        if not path_projections:
            logger.error("Could not find a path. Aborting animation.")
            return

        # 2. Interpolate frames along the path
        logger.info("Interpolating frames for animation...")
        if not self.viz_params.range_frames:
            num_frames = 100  # Default
        else:
            num_frames = np.random.randint(
                self.viz_params.range_frames[0], self.viz_params.range_frames[1] + 1
            )

        interpolated_path = self._interpolate_path(path_projections, num_frames)

        # 3. Reconstruct 3D coordinates for each frame
        logger.info(
            f"Reconstructing 3D coordinates for {len(interpolated_path)} frames..."
        )
        animation_coords = self._reconstruct_coords_from_pcs(interpolated_path)

        # 4. Write the animation to a multi-MODEL PDB file
        logger.info("Writing animation to PDB file...")
        self._write_animation_pdb(animation_coords)
        logger.info("Animation generation complete.")

    def _find_path_between_clusters(self) -> Optional[List[np.ndarray]]:
        """Finds a path of projections between start and end cluster centroids."""
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
            closest_point_idx_in_cluster = cdist([centroid], cluster_points).argmin()
            return indices[closest_point_idx_in_cluster]

        start_idx = find_centroid_index(self.viz_params.cluster_start)
        end_idx = find_centroid_index(self.viz_params.cluster_end)

        if start_idx is None or end_idx is None:
            logger.error(
                f"Could not find centroids for clusters {self.viz_params.cluster_start} or {self.viz_params.cluster_end}."
            )
            return None

        # Build graph for pathfinding
        nn = NearestNeighbors(n_neighbors=self.viz_params.k_neighbors).fit(
            all_projections
        )
        knn_graph = nn.kneighbors_graph(mode="distance")

        # Find shortest path
        distances, predecessors = dijkstra(
            csgraph=knn_graph,
            directed=False,
            indices=start_idx,
            return_predecessors=True,
        )

        if np.isinf(distances[end_idx]):
            logger.error("No path found between centroids. Try increasing k_neighbors.")
            return None

        # Reconstruct path
        path_indices = []
        curr = end_idx
        while curr != start_idx and curr != -9999:
            path_indices.append(curr)
            curr = predecessors[curr]
        path_indices.append(start_idx)
        path_indices.reverse()

        return [all_projections[i] for i in path_indices]

    def _interpolate_path(
        self, path_projections: List[np.ndarray], num_frames: int
    ) -> np.ndarray:
        """Linearly interpolates between keyframes in a path to get a smooth trajectory."""
        path_projections = np.array(path_projections)
        distances = np.linalg.norm(np.diff(path_projections, axis=0), axis=1)
        cumulative_dist = np.insert(np.cumsum(distances), 0, 0)

        total_dist = cumulative_dist[-1]
        interp_dist = np.linspace(0, total_dist, num_frames)

        interp_path = np.zeros((num_frames, path_projections.shape[1]))
        for i in range(path_projections.shape[1]):  # Interpolate each PC
            interp_path[:, i] = np.interp(
                interp_dist, cumulative_dist, path_projections[:, i]
            )

        return interp_path

    def _reconstruct_coords_from_pcs(self, projections: np.ndarray) -> np.ndarray:
        """Converts points in PC space back to 3D Cartesian coordinates."""
        # Formula: coords = mean + (projection @ eigenvectors)
        reconstructed_dev = projections @ self.eigenvectors
        # Reshape the mean and add the deviation for each frame
        flat_mean = self.mean_coords.flatten()
        reconstructed_flat = flat_mean + reconstructed_dev
        return reconstructed_flat.reshape(-1, len(flat_mean) // 3, 3)

    def _write_animation_pdb(self, animation_coords: np.ndarray):
        """Writes the generated coordinates into a multi-MODEL PDB file."""
        filename = f"animation_c{self.viz_params.cluster_start}_to_c{self.viz_params.cluster_end}.pdb"
        filepath = self.visuals_path / filename

        with open(filepath, "w") as f:
            for frame_idx, coords in enumerate(animation_coords):
                f.write(f"MODEL        {frame_idx + 1}\n")
                # Create a simple structure for the core atoms
                for atom_idx, atom_coord in enumerate(coords):
                    # Using generic atom info since we simplified.
                    # This could be enhanced with core_atom_metadata.
                    line = (
                        f"ATOM  {atom_idx+1:>5}  CA  ALA A{atom_idx+1:>4}    "
                        f"{atom_coord[0]:8.3f}{atom_coord[1]:8.3f}{atom_coord[2]:8.3f}"
                        "  1.00  0.00           C  \n"
                    )
                    f.write(line)
                f.write("ENDMDL\n")
        logger.info(f"Animation saved to {filepath}")
