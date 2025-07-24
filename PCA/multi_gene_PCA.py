import os
import json
import logging
import multiprocessing
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
from Bio import AlignIO
from sklearn.decomposition import PCA

# --- Configuration & Setup ---

# Set up a logger instance. Configuration will be done in the main class.
logger = logging.getLogger("PCACalculator")


@dataclass
class RunParams:
    """Parameters for the overall PCA run execution."""

    pdb_dirs: List[str]
    msa_path: str
    output_dir: str
    run_name: str


@dataclass
class AlignmentParams:
    """Parameters defining the alignment strategy."""

    atom_selection: str = "ca"
    num_iterations: int = 3
    residues_to_exclude: Optional[List[int]] = None

    def __post_init__(self):
        self.atom_selection = self.atom_selection.lower()
        if self.atom_selection not in ["ca", "backbone"]:
            raise ValueError("atom_selection must be 'ca' or 'backbone'.")


# --- Worker Functions for Multiprocessing ---


def _init_extraction_worker(residue_maps_data, core_indices_data, atom_list_data):
    """Initializes PDB extraction worker with necessary data."""
    global residue_maps, core_indices, atom_list
    residue_maps = residue_maps_data
    core_indices = core_indices_data
    atom_list = atom_list_data


def _pdb_extraction_worker(
    pdb_path: Path,
) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Worker to process a single PDB file based on pre-computed maps.
    Returns the PDB name, its full atomic coordinates, and its filtered (core) coordinates.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.name, pdb_path)
        all_residues = [
            res for res in structure.get_residues() if is_aa(res, standard=True)
        ]
        pdb_sequence = "".join(
            [three_to_one(res.get_resname()) for res in all_residues]
        )

        if pdb_sequence not in residue_maps:
            # Using logger which will be configured in the main process
            logging.warning(
                f"Sequence for {pdb_path.name} not found in MSA map. Skipping."
            )
            return None

        residue_map = residue_maps[pdb_sequence]
        filtered_coords = []
        for res_initial_idx, res_aligned_idx in residue_map.items():
            if res_aligned_idx in core_indices:
                residue = all_residues[res_initial_idx]
                for atom in residue.get_atoms():
                    if atom.name in atom_list:
                        filtered_coords.append(atom.get_coord())

        expected_len = len(core_indices) * len(atom_list)
        if len(filtered_coords) != expected_len:
            logging.warning(
                f"Atom filtering mismatch for {pdb_path.name}. "
                f"Expected {expected_len}, found {len(filtered_coords)}. Skipping."
            )
            return None

        all_coords = np.array(
            [atom.get_coord() for res in all_residues for atom in res.get_atoms()]
        )
        return pdb_path.name, all_coords, np.array(filtered_coords)

    except Exception as e:
        logging.error(f"Failed to process {pdb_path.name}: {e}")
        return None


def _alignment_worker(
    args: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs alignment using the Kabsch algorithm to find the optimal rotation and translation.
    """
    moving_coords, reference_coords = args
    centroid_moving = np.mean(moving_coords, axis=0)
    centroid_reference = np.mean(reference_coords, axis=0)
    moving_centered = moving_coords - centroid_moving
    reference_centered = reference_coords - centroid_reference
    H = moving_centered.T @ reference_centered
    U, S, Vt = np.linalg.svd(H)
    rotation = Vt.T @ U.T

    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    translation = centroid_reference - rotation @ centroid_moving
    return rotation, translation


# --- Main Calculator Class ---


class PCACalculator:
    def __init__(self, run_params: RunParams, alignment_params: AlignmentParams):
        self.run_params = run_params
        self.align_params = alignment_params

        self.output_path = Path(self.run_params.output_dir) / self.run_params.run_name
        # Create the new raw_data directory for all data outputs
        self.raw_data_path = self.output_path / "raw_data"
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.raw_data_path.mkdir(exist_ok=True)

        self.atom_list = (
            ["CA"]
            if self.align_params.atom_selection == "ca"
            else ["N", "CA", "C", "O"]
        )

        # This will be configured in the run() method.
        self.logger = logging.getLogger("PCACalculator")

    def _setup_logging(self):
        """Configures logging to stream to console and save to a file."""
        self.logger.setLevel(logging.INFO)
        # Clear existing handlers to prevent duplicate logs on re-runs
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler (saves to the main run directory)
        log_path = self.output_path / f"{self.run_params.run_name}.log"
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(f"Logger configured. Log file will be saved to {log_path}")

    def _create_sequence_and_residue_maps(
        self, msa: AlignIO.MultipleSeqAlignment
    ) -> Dict[str, Dict[int, int]]:
        """Processes the MSA to create a map from each unique ungapped sequence to its residue index mapping."""
        residue_maps = {}
        for record in msa:
            ungapped_seq = str(record.seq).replace("-", "")
            if ungapped_seq not in residue_maps:
                res_map = {}
                ungapped_idx = 0
                for aligned_idx, aa in enumerate(record.seq):
                    if aa != "-":
                        res_map[ungapped_idx] = aligned_idx
                        ungapped_idx += 1
                residue_maps[ungapped_seq] = res_map
        return residue_maps

    def _create_core_filter(self, msa: AlignIO.MultipleSeqAlignment) -> set[int]:
        """Creates a filter of common residue indices as a set for fast lookups."""
        common_indices = []
        alignment_length = msa.get_alignment_length()
        excluded_set = set(self.align_params.residues_to_exclude or [])

        for col_idx in range(alignment_length):
            if col_idx in excluded_set:
                continue
            if all(record.seq[col_idx] != "-" for record in msa):
                common_indices.append(col_idx)

        core_indices_set = set(common_indices)
        self.logger.info(
            f"Identified {len(core_indices_set)} common core residue indices."
        )
        return core_indices_set

    def run(self):
        """Executes the full PCA calculation workflow."""
        self._setup_logging()
        self.logger.info(f"Starting PCA run: '{self.run_params.run_name}'")

        # Save run parameters to the top-level run directory
        params_to_save = {
            "run_params": asdict(self.run_params),
            "alignment_params": asdict(self.align_params),
        }
        with open(self.output_path / "run_parameters.json", "w") as f:
            json.dump(params_to_save, f, indent=4)

        # 1. Load MSA, create maps and filter
        msa = AlignIO.read(self.run_params.msa_path, "clustal")
        residue_maps = self._create_sequence_and_residue_maps(msa)
        core_indices = self._create_core_filter(msa)

        # 2. Gather PDB files
        pdb_files = [p for d in self.run_params.pdb_dirs for p in Path(d).glob("*.pdb")]
        self.logger.info(f"Found {len(pdb_files)} PDB files to process.")

        # 3. Extract coordinates via multiprocessing
        init_args = (residue_maps, core_indices, self.atom_list)
        with multiprocessing.Pool(
            initializer=_init_extraction_worker, initargs=init_args
        ) as pool:
            extraction_results = pool.map(_pdb_extraction_worker, pdb_files)

        pdb_data = {
            name: {"all_coords": all_c, "filtered_coords": filt_c}
            for name, all_c, filt_c in extraction_results
            if name is not None
        }

        if not pdb_data:
            self.logger.error("No PDB files could be processed. Aborting.")
            return

        # 4. Iterative Alignment
        initial_filtered_coords = [
            data["filtered_coords"] for data in pdb_data.values()
        ]
        mean_coords = np.mean(np.array(initial_filtered_coords), axis=0)
        pdb_names_sorted = sorted(pdb_data.keys())

        for i in range(self.align_params.num_iterations):
            self.logger.info(
                f"Alignment iteration {i+1}/{self.align_params.num_iterations}..."
            )
            tasks = [
                (pdb_data[name]["filtered_coords"], mean_coords)
                for name in pdb_names_sorted
            ]

            with multiprocessing.Pool() as pool:
                transformations = pool.map(_alignment_worker, tasks)

            aligned_filtered_coords_list = []
            for i, name in enumerate(pdb_names_sorted):
                rotation, translation = transformations[i]
                aligned_filtered = (
                    rotation @ pdb_data[name]["filtered_coords"].T
                ).T + translation
                aligned_filtered_coords_list.append(aligned_filtered)
            mean_coords = np.mean(np.array(aligned_filtered_coords_list), axis=0)

        self.logger.info("Finalizing transformations.")
        final_transformations = {
            name: {"rotation": rot.tolist(), "translation": trans.tolist()}
            for name, (rot, trans) in zip(pdb_names_sorted, transformations)
        }

        np.save(
            self.raw_data_path / "mean_coords.npy",
            mean_coords.flatten(),
        )
        # 5. Run PCA on the final aligned filtered coordinates
        self.logger.info("Running final PCA...")
        pca_input_matrix = np.array(
            [coords.flatten() for coords in aligned_filtered_coords_list]
        )
        pca = PCA(n_components=None)
        principal_components = pca.fit_transform(pca_input_matrix)

        # 6. Save all data outputs to the 'raw_data' directory
        self.logger.info(f"Saving all data to: {self.raw_data_path}")

        # Save PCA results
        pc_df = pd.DataFrame(
            principal_components,
            columns=[f"PC{i+1}" for i in range(principal_components.shape[1])],
        )
        pc_df["PDB_Name"] = pdb_names_sorted
        pc_df.to_csv(self.raw_data_path / "principal_components.csv", index=False)

        np.savetxt(
            self.raw_data_path / "explained_variance_ratio.csv",
            pca.explained_variance_ratio_,
            delimiter=",",
        )
        np.savetxt(
            self.raw_data_path / "pca_components.csv", pca.components_, delimiter=","
        )

        # 7. Consolidate and save processing metadata into a single JSON file
        processing_metadata = {
            "residue_maps": residue_maps,
            "core_filter_indices": sorted(list(core_indices)),
            "final_transformations": final_transformations,
        }
        with open(self.raw_data_path / "processing_metadata.json", "w") as f:
            json.dump(processing_metadata, f, indent=4)

        self.logger.info("PCA calculation complete!")
