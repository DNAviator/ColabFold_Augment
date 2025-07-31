import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from multi_gene_PCA import RunParams

# --- Configuration ---

# Use a specific logger for this script
logger = logging.getLogger("MovementAnalyzer")

@dataclass
class AnalysisParams:
    """Parameters to control the movement analysis."""

    # Which PCs to report in the detailed raw data log
    pcs_for_raw_report: List[int] = field(default_factory=lambda: [1, 2, 3])
    # Which PCs to report in the high-level summary log
    pcs_for_summary_report: List[int] = field(default_factory=lambda: [1])
    # Ignore any atomic movements smaller than this threshold (in Angstroms)
    movement_threshold: float = 0.001


# --- Main Analyzer Class ---


class MovementAnalyzer:
    def __init__(self, run_params: RunParams, analysis_params: AnalysisParams):
        self.run_params = run_params
        self.analysis_params = analysis_params

        # --- Path Setup ---
        self.run_path = Path(self.run_params.output_dir) / self.run_params.run_name
        self.raw_data_path = self.run_path / "raw_data"
        self.analysis_path = self.run_path / "analysis"
        self.analysis_path.mkdir(exist_ok=True)

        # --- Loaded Data ---
        self.projections_df: pd.DataFrame = pd.DataFrame()
        self.eigenvectors: np.ndarray = np.array([])
        self.processing_metadata: Dict[str, Any] = {}
        self.atom_selection: str = "ca"
        self.canonical_atom_map: List[Dict[str, Any]] = []

    def _setup_logging(self, log_name: str) -> logging.Logger:
        """Sets up a dedicated logger for a specific output file."""
        file_logger = logging.getLogger(log_name)
        file_logger.setLevel(logging.INFO)
        # Prevent logs from propagating to the root logger
        file_logger.propagate = False

        # Clear existing handlers to avoid duplicate logs
        if file_logger.hasHandlers():
            file_logger.handlers.clear()

        formatter = logging.Formatter("%(message)s")  # Simple format for clean logs
        fh = logging.FileHandler(self.analysis_path / log_name, mode="w")
        fh.setFormatter(formatter)
        file_logger.addHandler(fh)

        return file_logger

    def _load_data(self):
        """Loads all necessary data from the PCA calculation step."""
        logger.info("Loading data from PCA run...")
        try:
            # Load the CSV which must have cluster data
            clustered_csv_path = self.raw_data_path / "principal_components_with_clusters.csv"
            if not clustered_csv_path.exists():
                logger.error(f"Missing required file: {clustered_csv_path}. Please run clustering first.")
                raise FileNotFoundError
            self.projections_df = pd.read_csv(clustered_csv_path)

            self.eigenvectors = np.loadtxt(self.raw_data_path / "pca_components.csv", delimiter=",")

            with open(self.raw_data_path / "processing_metadata.json", "r") as f:
                self.processing_metadata = json.load(f)

            with open(self.run_path / "run_parameters.json", "r") as f:
                run_params_data = json.load(f)
                self.atom_selection = run_params_data.get("alignment_params", {}).get("atom_selection", "ca")

            logger.info("Data loaded successfully.")
            return True
        except FileNotFoundError as e:
            logger.error(f"Failed to load data. Ensure pca_calculator.py and pca_visualizer.py (clustering) ran successfully.", exc_info=True)
            return False

    def _reconstruct_coords_from_pcs(self, projections: np.ndarray) -> np.ndarray:
        """Converts a point in PC space back to 3D Cartesian coordinates."""
        # For this analysis, we only care about the *difference* between two reconstructions,
        # so the mean term (Mean_B - Mean_A) cancels out. We can use a zero-mean.
        reconstructed_dev = projections @ self.eigenvectors
        return reconstructed_dev.reshape(-1, 3)

    def _create_canonical_atom_map(self):
        """Creates a list mapping each atom index to its residue and atom name."""
        logger.info("Creating canonical atom map...")
        atom_list = ["CA"] if self.atom_selection == "ca" else ["N", "CA", "C", "O"]
        # Ensure core_filter_indices is a list of integers
        core_indices = sorted([int(i) for i in self.processing_metadata['core_filter_indices']])

        for msa_idx in core_indices:
            for atom_name in atom_list:
                self.canonical_atom_map.append({'msa_index': msa_idx, 'atom_name': atom_name})
        logger.info(f"Canonical map created for {len(self.canonical_atom_map)} atoms.")

    def analyze(self):
        """Main method to run the entire movement analysis workflow."""
        if not self._load_data():
            return

        self._create_canonical_atom_map()

        # 1. Get coordinates and PDB names for each cluster centroid
        logger.info("Reconstructing coordinates for all cluster centroids...")
        cluster_ids = sorted([c for c in self.projections_df['cluster'].unique() if c != -1])
        pc_cols = [f"PC{i+1}" for i in range(self.eigenvectors.shape[0])]

        centroid_coords = {}
        centroid_pdb_names = {}
        for cid in cluster_ids:
            cluster_df = self.projections_df[self.projections_df['cluster'] == cid]

            # Find the projection vector for the point closest to the geometric center
            cluster_projections = cluster_df[pc_cols].values
            centroid_proj_vector = cluster_projections.mean(axis=0)

            # Find the index of the actual data point closest to this geometric center
            closest_point_index_in_cluster = cdist([centroid_proj_vector], cluster_projections).argmin()

            # Get the projection values for that closest point directly from the numpy array
            actual_centroid_projections = cluster_projections[closest_point_index_in_cluster]

            # Get the PDB name using the original dataframe index
            centroid_df_index = cluster_df.index[closest_point_index_in_cluster]
            centroid_pdb_names[cid] = self.projections_df.loc[centroid_df_index, 'PDB_Name']

            # Reconstruct coordinates using the clean numpy array
            centroid_coords[cid] = self._reconstruct_coords_from_pcs(actual_centroid_projections)

        # 2. Calculate all pairwise movements
        logger.info("Calculating all pairwise centroid movements...")
        all_movements = {}
        for c1, c2 in combinations(cluster_ids, 2):
            coords1 = centroid_coords[c1]
            coords2 = centroid_coords[c2]

            total_movement_vectors = coords2 - coords1
            total_distances = np.linalg.norm(total_movement_vectors, axis=1)

            pc_movements = {}
            total_movement_flat = total_movement_vectors.flatten()

            all_pcs_to_analyze = set(self.analysis_params.pcs_for_raw_report + self.analysis_params.pcs_for_summary_report)
            for pc_num in all_pcs_to_analyze:
                pc_idx = pc_num - 1
                eigenvector = self.eigenvectors[pc_idx]
                movement_along_pc_flat = np.dot(total_movement_flat, eigenvector) * eigenvector
                movement_along_pc_reshaped = movement_along_pc_flat.reshape(-1, 3)
                pc_distances = np.linalg.norm(movement_along_pc_reshaped, axis=1)
                pc_movements[pc_num] = pc_distances

            all_movements[(c1, c2)] = {
                'total_distances': total_distances,
                'pc_distances': pc_movements
            }

        # 3. Generate log files
        logger.info("Generating analysis logs...")
        self._write_raw_log(all_movements, cluster_ids)
        self._write_summary_log(all_movements, cluster_ids, centroid_pdb_names)
        logger.info(f"Analysis complete. Logs saved to: {self.analysis_path}")

    def _format_movement_line(self, atom_idx: int, distance: float) -> str:
        """Formats a single line for the log files."""
        atom_info = self.canonical_atom_map[atom_idx]
        return f"Residue at MSA index {atom_info['msa_index']:<4} (Atom {atom_info['atom_name']:<2}) moved {distance:.4f} Å"

    def _write_raw_log(self, all_movements: Dict, cluster_ids: List[int]):
        """Writes the detailed raw data log file."""
        raw_logger = self._setup_logging("movement_analysis_raw.log")

        for c1, c2 in combinations(cluster_ids, 2):
            raw_logger.info(f"--- C{c1} vs C{c2} ---\n")
            pair_data = all_movements[(c1, c2)]

            raw_logger.info("Top 10 Total Movements:")
            total_distances = pair_data['total_distances']
            sorted_indices = np.argsort(total_distances)[::-1]
            count = 0
            for i in sorted_indices:
                if total_distances[i] >= self.analysis_params.movement_threshold and count < 10:
                    raw_logger.info(f"  {count+1}. {self._format_movement_line(i, total_distances[i])}")
                    count += 1
            raw_logger.info("")

            for pc_num in self.analysis_params.pcs_for_raw_report:
                raw_logger.info(f"Top 10 Movements along PC{pc_num}:")
                pc_distances = pair_data['pc_distances'][pc_num]
                sorted_indices = np.argsort(pc_distances)[::-1]
                count = 0
                for i in sorted_indices:
                    if pc_distances[i] >= self.analysis_params.movement_threshold and count < 10:
                        raw_logger.info(f"  {count+1}. {self._format_movement_line(i, pc_distances[i])}")
                        count += 1
                raw_logger.info("")
            raw_logger.info("\n" + "="*60 + "\n")

    def _write_summary_log(self, all_movements: Dict, cluster_ids: List[int], centroid_pdb_names: Dict[int, str]):
        """Writes the high-level summary log file."""
        summary_logger = self._setup_logging("movement_analysis_summary.log")
        
        summary_logger.info("Cluster Centroids:\n")
        for i, centroid in centroid_pdb_names.items():
            summary_logger.info(f"Cluster {i}: {centroid}")

        best_partners = {}
        for c1 in cluster_ids:
            max_dist = -1
            best_partner = -1
            for c2 in cluster_ids:
                if c1 == c2: continue
                pair = tuple(sorted((c1, c2)))
                dist = np.max(all_movements[pair]['total_distances'])
                if dist > max_dist:
                    max_dist = dist
                    best_partner = c2
            best_partners[c1] = best_partner

        for c1 in sorted(best_partners.keys()):
            c2 = best_partners[c1]
            pair = tuple(sorted((c1, c2)))

            summary_logger.info(f"Cluster {c1} had the greatest difference with Cluster {c2}:\n")
            pair_data = all_movements[pair]

            for pc_num in self.analysis_params.pcs_for_summary_report:
                summary_logger.info(f"  Top 5 Movements along PC{pc_num}:")
                pc_distances = pair_data['pc_distances'][pc_num]
                sorted_indices = np.argsort(pc_distances)[::-1]
                count = 0
                for i in sorted_indices:
                    if pc_distances[i] >= self.analysis_params.movement_threshold and count < 5:
                        summary_logger.info(f"    {count+1}. {self._format_movement_line(i, pc_distances[i])}")
                        count += 1
                summary_logger.info("")
            summary_logger.info("\n" + "="*60 + "\n")
