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

    range_pcs_for_clustering: List[int]
    min_cluster_size: int = 15


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

        # --- Path Setup ---
        self.run_path = Path(self.run_params.output_dir) / self.run_params.run_name
        self.raw_data_path = self.run_path / "raw_data"
        self.visuals_path = self.run_path / "visualizations"
        self.visuals_path.mkdir(exist_ok=True)

        self._setup_logging()

        # --- Loaded Data ---
        # Initialize attributes to their empty types to avoid Optional typing issues
        self.is_data_loaded = False
        self.projections_df: pd.DataFrame = pd.DataFrame()
        self.eigenvectors: np.ndarray = np.array([])
        self.mean_coords: np.ndarray = np.array([])
        self.explained_variance: np.ndarray = np.array([])

    def _setup_logging(self):
        """Configures logging to stream to console and save to the run's log file."""
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

    def _load_data(self):
        """Loads all necessary data from the PCA calculation step."""
        if self.is_data_loaded:
            return
        logger.info("Loading data from PCA run...")

        # Try to load the file with cluster data first, fall back to the original
        clustered_csv_path = (
            self.raw_data_path / "principal_components_with_clusters.csv"
        )
        base_csv_path = self.raw_data_path / "principal_components.csv"

        try:
            if clustered_csv_path.exists():
                self.projections_df = pd.read_csv(clustered_csv_path)
            else:
                self.projections_df = pd.read_csv(base_csv_path)

            self.eigenvectors = np.loadtxt(
                self.raw_data_path / "pca_components.csv", delimiter=","
            )
            self.explained_variance = np.loadtxt(
                self.raw_data_path / "explained_variance_ratio.csv", delimiter=","
            )
            self.mean_coords = np.load(self.raw_data_path / "mean_coords.npy")

            self._generate_source_species_column()

            self.is_data_loaded = True
            logger.info("Data loaded successfully.")
        except FileNotFoundError as e:
            logger.error(
                f"Failed to load data: {e}. Make sure the pca_calculator script ran successfully."
            )
            raise

    def _generate_source_species_column(self):
        """Derives a 'Source_Species' column from PDB names and groups small sets into 'misc'."""
        if "PDB_Name" not in self.projections_df.columns:
            logger.warning(
                "Cannot generate Source_Species column, 'PDB_Name' not found."
            )
            return

        logger.info("Generating 'Source_Species' column from PDB names...")
        # Extract species name by splitting at '_unrelaxed'
        self.projections_df["Source_Species"] = (
            self.projections_df["PDB_Name"].str.split("_unrelaxed", n=1).str[0]
        )

        # Count occurrences of each species
        species_counts = self.projections_df["Source_Species"].value_counts()

        # Find species with fewer than 100 members
        small_species = species_counts[species_counts < 100].index

        if not small_species.empty:
            logger.info(f"Grouping {len(small_species)} small species into 'misc'.")
            # Re-label small species as 'misc'
            self.projections_df.loc[
                self.projections_df["Source_Species"].isin(small_species),
                "Source_Species",
            ] = "misc"

    def run_clustering(self):
        """Performs HDBSCAN clustering on the PCA data."""
        if not self.is_data_loaded:
            self._load_data()

        logger.info("Performing HDBSCAN clustering...")
        start_pc, end_pc = self.clustering_params.range_pcs_for_clustering
        pc_cols = [f"PC{i}" for i in range(start_pc, end_pc + 1)]

        cluster_data = self.projections_df[pc_cols].values
        clusterer = HDBSCAN(min_cluster_size=self.clustering_params.min_cluster_size)
        self.projections_df["cluster"] = clusterer.fit_predict(cluster_data)

        n_clusters = len(set(self.projections_df["cluster"])) - (
            1 if -1 in self.projections_df["cluster"].unique() else 0
        )
        logger.info(f"Clustering complete. Found {n_clusters} clusters.")
        self.projections_df.to_csv(
            self.raw_data_path / "principal_components_with_clusters.csv", index=False
        )
        logger.info("Saved clustering results.")

    def generate_plots(self):
        """Generates 2D and 3D scatter plots, colored by cluster and by source species."""
        if not self.is_data_loaded:
            self._load_data()
        if "cluster" not in self.projections_df.columns:
            logger.warning("No cluster data found. Running clustering now.")
            self.run_clustering()

        plot_df = self.projections_df.copy()

        # --- Plotting by Cluster ---
        logger.info("--- Generating plots colored by CLUSTER ---")
        plot_df["Cluster_Label"] = [
            f"Cluster {l}" if l != -1 else "Noise" for l in plot_df["cluster"]
        ]
        self._create_plots_from_dataframe(
            plot_df, color_by="Cluster_Label", suffix="_by_cluster"
        )

        # --- Plotting by Source Species ---
        if "Source_Species" in plot_df.columns:
            logger.info("--- Generating plots colored by SOURCE SPECIES ---")
            self._create_plots_from_dataframe(
                plot_df, color_by="Source_Species", suffix="_by_species"
            )

    def _create_plots_from_dataframe(
        self, plot_df: pd.DataFrame, color_by: str, suffix: str
    ):
        """Helper function to generate and save 2D and 3D plots."""
        # Generate 3D plot
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
            filename = f"3d_plot_pc{pcs[0]}_{pcs[1]}_{pcs[2]}{suffix}.html"
            fig.write_html(self.visuals_path / filename)

        # Generate 2D plots
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
                filename = f"2d_plot_pc{pcs[0]}_{pcs[1]}{suffix}.html"
                fig.write_html(self.visuals_path / filename)

    def generate_animation(self):
        """Generates a PDB animation of the transition between two clusters."""
        if not self.is_data_loaded:
            self._load_data()
        if "cluster" not in self.projections_df.columns:
            logger.error(
                "Cannot animate without clustering data. Run `run_clustering()` first."
            )
            return

        logger.info("Finding shortest path between cluster centroids...")
        path_projections = self._find_path_between_clusters()
        if not path_projections:
            logger.error("Could not find a path. Aborting animation.")
            return

        logger.info("Interpolating frames for animation...")
        num_frames = np.random.randint(
            self.viz_params.range_frames[0], self.viz_params.range_frames[1] + 1
        )
        interpolated_path = self._interpolate_path(path_projections, num_frames)

        logger.info(
            f"Reconstructing 3D coordinates for {len(interpolated_path)} frames..."
        )
        animation_coords = self._reconstruct_coords_from_pcs(interpolated_path)

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
            logger.error("No path found between centroids. Try increasing k_neighbors.")
            return None

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
        interp_dist = np.linspace(0, cumulative_dist[-1], num_frames)
        interp_path = np.zeros((num_frames, path_projections.shape[1]))
        for i in range(path_projections.shape[1]):
            interp_path[:, i] = np.interp(
                interp_dist, cumulative_dist, path_projections[:, i]
            )
        return interp_path

    def _reconstruct_coords_from_pcs(self, projections: np.ndarray) -> np.ndarray:
        """Converts points in PC space back to 3D Cartesian coordinates."""
        reconstructed_dev = projections @ self.eigenvectors
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
                for atom_idx, atom_coord in enumerate(coords):
                    line = (
                        f"ATOM  {atom_idx+1:>5}  CA  ALA A{atom_idx+1:>4}    "
                        f"{atom_coord[0]:8.3f}{atom_coord[1]:8.3f}{atom_coord[2]:8.3f}"
                        "  1.00  0.00           C  \n"
                    )
                    f.write(line)
                f.write("ENDMDL\n")
        logger.info(f"Animation saved to {filepath}")
