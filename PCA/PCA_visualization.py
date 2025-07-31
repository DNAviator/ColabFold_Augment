import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Bio.PDB import (
    PDBParser,
    PDBIO,
    Structure,
    Model,
    Chain,
    Atom,
    Polypeptide,
    Superimposer,
)
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
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
    num_frames: int = 100
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
        self.visuals_path.mkdir(exist_ok=True, parents=True)

        self._setup_logging()

        self.pdb_path_map = self._create_pdb_path_map()

        self.is_data_loaded = False
        self.projections_df: pd.DataFrame = pd.DataFrame()
        self.eigenvectors: np.ndarray = np.array([])
        self.mean_coords: np.ndarray = np.array([])
        self.explained_variance: np.ndarray = np.array([])
        self.processing_metadata: Dict[str, Any] = {}
        self.ref_structure: Optional[Structure.Structure] = None
        self.atom_selection: str = "ca"

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
        path_map = {}
        for directory in self.run_params.pdb_dirs:
            for p in Path(directory).glob("*.pdb"):
                path_map[p.name] = str(p)
        return path_map

    def _load_data(self):
        if self.is_data_loaded:
            return
        logger.info("Loading data from PCA run...")
        try:
            clustered_csv_path = (
                self.raw_data_path / "principal_components_with_clusters.csv"
            )
            base_csv_path = self.raw_data_path / "principal_components.csv"
            run_params_path = self.run_path / "run_parameters.json"

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

    def run_clustering(self):
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

    def generate_plots(self):
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

    def generate_animation(self):
        if not self.is_data_loaded:
            self._load_data()
        if "cluster" not in self.projections_df.columns:
            logger.error("Clustering data not found. Run clustering first.")
            return

        logger.info("Step 1: Finding path between clusters using noise bridge.")
        path_indices, keyframe_projections = self._find_path_between_clusters()
        if not path_indices or not keyframe_projections:
            logger.error("Could not find a path between the specified clusters.")
            return

        logger.info(f"Found path with {len(path_indices)} keyframes.")
        logger.info(
            f"Step 2: Interpolating path to {self.viz_params.num_frames} frames."
        )
        interpolated_projections = self._interpolate_path_with_spline(
            keyframe_projections, self.viz_params.num_frames
        )

        logger.info("Step 3: Visualizing the animation path in PC space.")
        self._plot_animation_path(path_indices, interpolated_projections)

        logger.info("Step 4: Reconstructing 3D coordinates from interpolated path.")
        target_core_coords = self._reconstruct_coords_from_pcs(interpolated_projections)

        logger.info("Step 5: Building full-atom frames for the animation.")
        animation_frames = self._build_full_atom_frames(target_core_coords)

        if not animation_frames:
            logger.error("Failed to build animation frames. Aborting.")
            return

        logger.info("Step 6: Writing animation to PDB file.")
        self._write_animation_pdb(animation_frames)
        logger.info("Animation generation complete.")

    def _find_path_between_clusters(
        self,
    ) -> tuple[Optional[List[int]], Optional[List[np.ndarray]]]:
        pc_cols = [f"PC{i}" for i in range(1, self.eigenvectors.shape[0] + 1)]
        all_projections = self.projections_df[pc_cols].values

        def find_centroid_index(cluster_id):
            indices = self.projections_df.index[
                self.projections_df["cluster"] == cluster_id
            ].tolist()
            if not indices:
                logger.warning(f"No points found for cluster {cluster_id}.")
                return None
            cluster_projections = all_projections[indices]
            centroid = cluster_projections.mean(axis=0)
            closest_point_idx_in_cluster = cdist(
                [centroid], cluster_projections
            ).argmin()
            return indices[closest_point_idx_in_cluster]

        start_idx = find_centroid_index(self.viz_params.cluster_start)
        end_idx = find_centroid_index(self.viz_params.cluster_end)

        if start_idx is None or end_idx is None:
            logger.error("Could not find centroids for start and/or end clusters.")
            return None, None

        ref_pdb_name = self.projections_df.loc[start_idx, "PDB_Name"]
        self.ref_structure = PDBParser(QUIET=True).get_structure(
            "ref", self.pdb_path_map[ref_pdb_name]
        )
        logger.info(f"Set reference structure to {ref_pdb_name}.")

        logger.info("Building targeted graph using start, end, and noise clusters...")
        start_cluster_indices = self.projections_df[
            self.projections_df["cluster"] == self.viz_params.cluster_start
        ].index
        end_cluster_indices = self.projections_df[
            self.projections_df["cluster"] == self.viz_params.cluster_end
        ].index
        bridge_indices = self.projections_df[self.projections_df["cluster"] == -1].index

        subset_indices = sorted(
            list(
                set(start_cluster_indices)
                | set(end_cluster_indices)
                | set(bridge_indices)
            )
        )
        if not subset_indices:
            logger.error("No points found for start, end, or bridge clusters.")
            return None, None

        logger.info(
            f"Constructing graph from {len(subset_indices)} points (start/end/noise)."
        )
        subset_projections = all_projections[subset_indices]

        original_to_subset_map = {
            orig_idx: new_idx for new_idx, orig_idx in enumerate(subset_indices)
        }

        tree = cKDTree(subset_projections)
        nn_distances, _ = tree.query(subset_projections, k=2)

        valid_distances = nn_distances[:, 1][np.isfinite(nn_distances[:, 1])]
        if len(valid_distances) == 0:
            logger.error("Could not determine a connection radius.")
            return None, None
        avg_nn_dist = valid_distances.mean()
        radius = 2.5 * avg_nn_dist
        logger.info(f"Calculated connection radius for subset graph: {radius:.3f}")

        pairs = tree.query_pairs(r=radius)
        graph = lil_matrix((len(subset_indices), len(subset_indices)))
        for i, j in pairs:
            dist = np.linalg.norm(subset_projections[i] - subset_projections[j])
            graph[i, j] = dist
            graph[j, i] = dist

        graph = graph.tocsr()
        logger.info(f"Subset graph constructed with {graph.nnz // 2} edges.")

        start_node_subset = original_to_subset_map[start_idx]
        end_node_subset = original_to_subset_map[end_idx]

        distances, predecessors = dijkstra(
            csgraph=graph,
            directed=False,
            indices=start_node_subset,
            return_predecessors=True,
        )

        if np.isinf(distances[end_node_subset]):
            logger.error(
                f"No path found between start and end points even with noise bridge."
            )
            return None, None

        path_subset_indices = []
        curr = end_node_subset
        while curr != start_node_subset and curr != -9999:
            path_subset_indices.append(curr)
            curr = predecessors[curr]
        path_subset_indices.append(start_node_subset)
        path_subset_indices.reverse()

        path_original_indices = [subset_indices[i] for i in path_subset_indices]

        return path_original_indices, [
            all_projections[i] for i in path_original_indices
        ]

    def _interpolate_path_with_spline(
        self, path_projections: List[np.ndarray], num_frames: int
    ) -> np.ndarray:
        path = np.array(path_projections)
        start_pc, end_pc = self.clustering_params.range_pcs_for_clustering
        spline_indices = list(range(start_pc - 1, end_pc))
        linear_indices = [i for i in range(path.shape[1]) if i not in spline_indices]
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        u_k = np.insert(np.cumsum(distances), 0, 0)
        if u_k[-1] == 0:
            u_k[-1] = 1.0
        u_k /= u_k[-1]
        u_i = np.linspace(0, 1, num_frames)
        interp_path = np.zeros((num_frames, path.shape[1]))
        if spline_indices:
            k = min(3, len(u_k) - 1)
            if k > 0:
                tck, _ = splprep(path[:, spline_indices].T, u=u_k, s=0, k=k)
                interp_path[:, spline_indices] = np.array(splev(u_i, tck)).T
            else:
                for col_idx in spline_indices:
                    interp_path[:, col_idx] = np.interp(u_i, u_k, path[:, col_idx])
        if linear_indices:
            for col_idx in linear_indices:
                interp_path[:, col_idx] = np.interp(u_i, u_k, path[:, col_idx])
        return interp_path

    def _plot_animation_path(
        self, path_indices: List[int], interpolated_projections: np.ndarray
    ):
        logger.info("Generating animation path visualization plot...")
        path_df = self.projections_df.loc[path_indices]
        interp_df = pd.DataFrame(
            interpolated_projections,
            columns=[f"PC{i+1}" for i in range(interpolated_projections.shape[1])],
        )
        if self.viz_params.pcs_to_plot_3d:
            pcs = self.viz_params.pcs_to_plot_3d
            pc_x, pc_y, pc_z = f"PC{pcs[0]}", f"PC{pcs[1]}", f"PC{pcs[2]}"
            fig = go.Figure()
            fig.add_trace(
                go.Scatter3d(
                    x=self.projections_df[pc_x],
                    y=self.projections_df[pc_y],
                    z=self.projections_df[pc_z],
                    mode="markers",
                    marker=dict(size=2, opacity=0.3),
                    name="All Points",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=path_df[pc_x],
                    y=path_df[pc_y],
                    z=path_df[pc_z],
                    mode="markers",
                    marker=dict(size=6, color="orange", symbol="diamond"),
                    name="Path Keyframes",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=interp_df[pc_x],
                    y=interp_df[pc_y],
                    z=interp_df[pc_z],
                    mode="lines",
                    line=dict(color="red", width=5),
                    name="Spline Path",
                )
            )
            fig.update_layout(
                title="Animation Path Visualization (3D)",
                scene=dict(xaxis_title=pc_x, yaxis_title=pc_y, zaxis_title=pc_z),
            )
            fig.write_html(self.visuals_path / "animation_path_visualization_3d.html")

    def _reconstruct_coords_from_pcs(self, projections: np.ndarray) -> np.ndarray:
        """Reconstructs full 3D coordinates from PC projections."""
        n_atoms = self.mean_coords.shape[0]
        n_features_expected = n_atoms * 3
        n_features_actual = self.eigenvectors.shape[1]

        if n_features_actual != n_features_expected:
            raise ValueError(
                f"Shape mismatch in input data for reconstruction. The number of features in "
                f"the eigenvectors file (pca_components.csv) does not match the number of "
                f"coordinates in the mean structure (mean_coords.npy).\n"
                f" - Expected eigenvector columns: {n_features_expected} ({n_atoms} atoms * 3 coords)\n"
                f" - Found eigenvector columns: {n_features_actual}\n"
                f"Please verify that the pca_calculator.py script is saving the full, flattened (3N) components."
            )

        flat_mean = self.mean_coords.flatten()
        reconstructed_dev = projections @ self.eigenvectors
        reconstructed_flat = flat_mean + reconstructed_dev
        return reconstructed_flat.reshape(-1, n_atoms, 3)

    def _kabsch_transform(
        self, P: np.ndarray, Q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        centroid_P, centroid_Q = P.mean(axis=0), Q.mean(axis=0)
        P_c, Q_c = P - centroid_P, Q - centroid_Q
        H = P_c.T @ Q_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_Q - (R @ centroid_P)
        return R, t

    def _build_full_atom_frames(
        self, target_core_coords: np.ndarray
    ) -> List[Structure.Structure]:
        assert self.ref_structure is not None
        ref_chain = next(self.ref_structure.get_models()).get_list()[0]
        full_seq_res = [
            res for res in ref_chain.get_residues() if Polypeptide.is_aa(res)
        ]

        ref_seq_str = "".join(
            [Polypeptide.three_to_one(res.get_resname()) for res in full_seq_res]
        )
        residue_map = self.processing_metadata["residue_maps"].get(ref_seq_str)
        if not residue_map:
            logger.error(
                f"Could not find residue map for reference structure '{self.ref_structure.id}'."
            )
            return []

        core_indices = set(self.processing_metadata["core_filter_indices"])
        atom_list_full = (
            ["CA"] if self.atom_selection == "ca" else ["N", "CA", "C", "O"]
        )
        canonical_core_map = [
            {"msa_index": msa_idx, "atom_name": atom_name}
            for msa_idx in sorted(list(core_indices))
            for atom_name in atom_list_full
        ]

        mean_coords_map = {
            (info["msa_index"], info["atom_name"]): self.mean_coords[i]
            for i, info in enumerate(canonical_core_map)
        }

        animation_frames = []
        for frame_idx in range(target_core_coords.shape[0]):
            new_frame = Structure.Structure(f"frame_{frame_idx}")
            new_model = Model.Model(frame_idx)
            new_chain = Chain.Chain("A")
            new_model.add(new_chain)
            new_frame.add(new_model)

            target_coords_this_frame = target_core_coords[frame_idx]
            target_coords_map = {
                (info["msa_index"], info["atom_name"]): target_coords_this_frame[i]
                for i, info in enumerate(canonical_core_map)
            }

            mean_core_flat = np.array(list(mean_coords_map.values()))
            target_core_flat = np.array(list(target_coords_map.values()))

            if mean_core_flat.size == 0 or target_core_flat.size == 0:
                logger.error(f"Frame {frame_idx}: Core coordinate set is empty.")
                return []

            R_global, t_global = self._kabsch_transform(
                mean_core_flat, target_core_flat
            )

            for seq_idx, res_template in enumerate(full_seq_res):
                new_res = res_template.copy()
                msa_index = residue_map.get(str(seq_idx))
                is_core_res = msa_index is not None and msa_index in core_indices

                if is_core_res:
                    atom_list_bb = ["N", "CA", "C"]
                    mean_bb_coords, target_bb_coords, ref_bb_coords = [], [], []
                    for atom_name in atom_list_bb:
                        key = (msa_index, atom_name)
                        if (
                            key in mean_coords_map
                            and key in target_coords_map
                            and atom_name in res_template
                        ):
                            mean_bb_coords.append(mean_coords_map[key])
                            target_bb_coords.append(target_coords_map[key])
                            ref_bb_coords.append(res_template[atom_name].get_coord())

                    use_local_transform = (
                        self.atom_selection != "ca" and len(target_bb_coords) == 3
                    )

                    if use_local_transform:
                        R_local, _ = self._kabsch_transform(
                            np.array(mean_bb_coords), np.array(target_bb_coords)
                        )
                        ref_bb_centroid = np.array(ref_bb_coords).mean(axis=0)
                        target_bb_centroid = np.array(target_bb_coords).mean(axis=0)

                        for atom in new_res:
                            key = (msa_index, atom.get_name())
                            if key in target_coords_map:
                                atom.set_coord(target_coords_map[key])
                            else:  # Sidechain atom
                                original_coord = res_template[
                                    atom.get_name()
                                ].get_coord()
                                relative_pos = original_coord - ref_bb_centroid
                                atom.set_coord(
                                    R_local @ relative_pos + target_bb_centroid
                                )
                    else:  # Fallback for CA-only or incomplete backbone
                        for atom in new_res:
                            key = (msa_index, atom.get_name())
                            if key in target_coords_map:
                                atom.set_coord(target_coords_map[key])
                            else:  # Sidechain atom
                                original_coord = res_template[
                                    atom.get_name()
                                ].get_coord()
                                atom.transform(R_global, t_global)
                else:  # Loop Residue
                    for atom in new_res:
                        atom.transform(R_global, t_global)
                new_chain.add(new_res)
            animation_frames.append(new_frame)
        return animation_frames

    def _write_animation_pdb(self, animation_frames: List[Structure.Structure]):
        if not animation_frames:
            logger.warning("No animation frames were generated to write.")
            return
        filename = f"animation_c{self.viz_params.cluster_start}_to_c{self.viz_params.cluster_end}.pdb"
        filepath = self.visuals_path / filename
        logger.info(
            f"Saving animation with {len(animation_frames)} frames to {filepath}..."
        )
        pdb_io = PDBIO()
        with open(filepath, "w") as f:
            for i, frame_struct in enumerate(animation_frames):
                f.write(f"MODEL        {i+1}\n")
                pdb_io.set_structure(frame_struct)
                pdb_io.save(f, write_end=False)
                f.write("ENDMDL\n")
        logger.info("Successfully wrote animation PDB.")

    # --- REVISED PUTTY MODEL GENERATION ---

    def generate_putty_models(self):
        """
        Generates a 'putty' PDB model for each cluster by aligning the original
        PDB files and calculating per-residue RMSF.
        """
        if not self.is_data_loaded: self._load_data()
        if "cluster" not in self.projections_df.columns:
            logger.error("Clustering data not found. Run clustering first."); return

        cluster_ids = sorted([c for c in self.projections_df['cluster'].unique() if c != -1])
        if not cluster_ids:
            logger.warning("No clusters found to generate putty models for.")
            return

        logger.info(f"Found {len(cluster_ids)} clusters. Generating putty model for each.")

        for cluster_id in cluster_ids:
            self._create_putty_model_for_cluster(cluster_id)

    def _create_putty_model_for_cluster(self, cluster_id: int):
        """
        Creates a putty model for a single cluster based on all-atom RMSF
        from the original, aligned PDB files.
        """
        logger.info(f"--- Processing Cluster {cluster_id} ---")
        
        # 1. Get PDB files for the current cluster
        cluster_pdb_names = self.projections_df[self.projections_df["cluster"] == cluster_id]["PDB_Name"].tolist()
        if not cluster_pdb_names:
            logger.warning(f"Cluster {cluster_id} is empty. Skipping.")
            return
            
        parser = PDBParser(QUIET=True)
        structures = [parser.get_structure(name, self.pdb_path_map[name]) for name in cluster_pdb_names]
        logger.info(f"Loaded {len(structures)} PDB files for cluster {cluster_id}.")

        # 2. Select a reference and align all structures to it
        ref_structure = structures[0]
        ref_chain = next(ref_structure.get_models()).get_list()[0]
        ref_residues = list(ref_chain.get_residues())
        ref_seq_str = "".join([Polypeptide.three_to_one(res.get_resname()) for res in ref_residues if Polypeptide.is_aa(res)])
        ref_res_map = self.processing_metadata["residue_maps"].get(ref_seq_str)
        if not ref_res_map:
            logger.error(f"Could not find residue map for reference PDB {ref_structure.id}. Skipping cluster {cluster_id}."); return

        aligned_structures = [ref_structure.copy()] # Start with the reference
        
        for moving_structure in structures[1:]:
            moving_chain = next(moving_structure.get_models()).get_list()[0]
            moving_residues = list(moving_chain.get_residues())
            moving_seq_str = "".join([Polypeptide.three_to_one(res.get_resname()) for res in moving_residues if Polypeptide.is_aa(res)])
            moving_res_map = self.processing_metadata["residue_maps"].get(moving_seq_str)
            if not moving_res_map:
                logger.warning(f"Skipping {moving_structure.id}, residue map not found.")
                continue

            # Find conserved backbone atoms for alignment
            ref_atoms, moving_atoms = [], []
            for ref_seq_idx, ref_msa_idx in ref_res_map.items():
                for moving_seq_idx, moving_msa_idx in moving_res_map.items():
                    if ref_msa_idx == moving_msa_idx:
                        try:
                            ref_res = ref_residues[int(ref_seq_idx)]
                            moving_res = moving_residues[int(moving_seq_idx)]
                            if ref_res.get_resname() == moving_res.get_resname():
                                for atom_name in ["N", "CA", "C"]:
                                    if atom_name in ref_res and atom_name in moving_res:
                                        ref_atoms.append(ref_res[atom_name])
                                        moving_atoms.append(moving_res[atom_name])
                        except (IndexError, KeyError):
                            continue
            
            if len(ref_atoms) < 3:
                logger.warning(f"Skipping {moving_structure.id}, not enough conserved atoms to align."); continue
            
            # Align (superimpose) the moving structure onto the reference
            super_imposer = Superimposer()
            super_imposer.set_atoms(ref_atoms, moving_atoms)
            super_imposer.apply(moving_structure.get_atoms())
            aligned_structures.append(moving_structure)

        logger.info(f"Aligned {len(aligned_structures)} structures.")

        # 3. Calculate per-residue RMSF
        # Collect all coordinates for each atom of each residue
        coords_by_residue = {} # Key: res_id, Value: {atom_name: [coord1, coord2, ...]}
        for structure in aligned_structures:
            for res in structure.get_residues():
                if not Polypeptide.is_aa(res): continue
                res_id = res.get_id()
                
                # Check if residue exists in reference before processing
                if res_id in ref_chain:
                    if res_id not in coords_by_residue:
                        coords_by_residue[res_id] = {}
                    
                    # Only add coords if the residue type matches the reference
                    if res.get_resname() == ref_chain[res_id].get_resname():
                        for atom in res:
                            if atom.get_name() not in coords_by_residue[res_id]:
                                coords_by_residue[res_id][atom.get_name()] = []
                            coords_by_residue[res_id][atom.get_name()].append(atom.get_coord())

        # Calculate RMSF for each residue
        rmsf_map = {}
        mean_structure_coords = {}
        for res_id, atom_coords_dict in coords_by_residue.items():
            # Skip if residue has no atoms (should not happen with is_aa check, but safe)
            if not atom_coords_dict: continue

            mean_res_coords = {} # Mean coords for each atom in this residue
            
            # Find the number of structures that contributed to this residue
            num_structures_for_res = len(next(iter(atom_coords_dict.values())))
            if num_structures_for_res < 2: continue # Cannot calculate variance with < 2 points

            for atom_name, coords_list in atom_coords_dict.items():
                coords_array = np.array(coords_list)
                mean_coord = coords_array.mean(axis=0)
                mean_res_coords[atom_name] = mean_coord

            mean_structure_coords[res_id] = mean_res_coords

            # Calculate RMSD of the residue from its mean for each structure
            rmsd_values = []
            for i in range(num_structures_for_res):
                res_coords_this_struct = np.array([atom_coords_dict[atom_name][i] for atom_name in mean_res_coords])
                mean_coords_this_res = np.array(list(mean_res_coords.values()))
                
                # Ensure we have the same number of atoms for comparison
                if res_coords_this_struct.shape != mean_coords_this_res.shape: continue

                rmsd_sq = np.sum((res_coords_this_struct - mean_coords_this_res)**2) / len(mean_coords_this_res)
                rmsd_values.append(np.sqrt(rmsd_sq))

            # RMSF is the root mean square of the RMSD values
            if rmsd_values:
                rmsf = np.sqrt(np.mean(np.square(rmsd_values)))
                rmsf_map[res_id] = rmsf

        # 4. Create final PDB with RMSF in B-factor column
        putty_model = ref_structure.copy()
        for res in putty_model.get_residues():
            res_id = res.get_id()
            rmsf_val = rmsf_map.get(res_id, 0.0)
            for atom in res:
                if res_id in mean_structure_coords and atom.get_name() in mean_structure_coords[res_id]:
                    atom.set_coord(mean_structure_coords[res_id][atom.get_name()])
                atom.set_bfactor(rmsf_val)
        
        # 5. Write the PDB file
        output_filename = self.visuals_path / f"putty_model_cluster_{cluster_id}.pdb"
        pdb_io = PDBIO()
        pdb_io.set_structure(putty_model)
        pdb_io.save(str(output_filename))
        logger.info(f"Successfully wrote putty model to {output_filename}")
