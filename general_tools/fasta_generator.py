import os
import logging
import random
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.Polypeptide import three_to_one, standard_aa_names
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import glob

# Configure a basic logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("FastaSubsetGenerator")


class FastaSubsetGenerator:
    """
    Extracts sequences from PDB files and creates a representative subset
    in a single FASTA file, ready for alignment by an external tool.
    """

    def __init__(self, pdb_directories, base_pdb_filenames):
        """
        Initializes the generator.

        Args:
            pdb_directories (list): A list of paths to directories containing PDB files.
            base_pdb_filenames (list): A list of filenames for the "base" PDB of each species.
                                       These will always be included in the subset.
        """
        if not pdb_directories:
            raise ValueError("PDB directories list cannot be empty.")
        self.pdb_directories = pdb_directories
        self.base_pdb_filenames = base_pdb_filenames
        self.all_pdb_paths = self._get_all_pdb_paths()

    def _get_all_pdb_paths(self):
        """Gathers all PDB file paths from the specified directories."""
        paths = {}
        for directory in self.pdb_directories:
            if not os.path.isdir(directory):
                logger.warning(f"Directory not found, skipping: {directory}")
                continue
            for filename in os.listdir(directory):
                if filename.lower().endswith((".pdb", ".ent")):
                    paths[filename] = os.path.join(directory, filename)
        return paths

    def _extract_sequence_from_pdb(self, pdb_path):
        """Extracts the amino acid sequence from a single PDB file."""
        parser = PDBParser(QUIET=True)
        pdb_id = os.path.basename(pdb_path)
        try:
            structure = parser.get_structure(pdb_id, pdb_path)
            ppb = PPBuilder()
            polypeptides = ppb.build_peptides(structure)
            if not polypeptides:
                logger.warning(f"No polypeptides found in {pdb_id}.")
                return None

            sequence = "".join(
                [
                    three_to_one(res.get_resname())
                    for res in polypeptides[0]
                    if res.get_resname() in standard_aa_names
                ]
            )
            return sequence
        except (PDBConstructionException, IndexError, KeyError) as e:
            logger.error(f"Could not process {pdb_id}: {e}")
            return None

    def create_subset_fasta(self, num_variants_per_species, output_fasta_file):
        """
        Creates a FASTA file with base PDBs and a random sample of variants.

        Args:
            num_variants_per_species (int): Number of random variants to include for each species.
            output_fasta_file (str): Path to save the output FASTA file.
        """
        logger.info("Creating representative subset for alignment...")

        # Always include the base PDBs
        subset_filenames = set(self.base_pdb_filenames)

        # Add random variants, excluding the base PDBs already added
        other_pdb_filenames = [
            name for name in self.all_pdb_paths.keys() if name not in subset_filenames
        ]
        if len(other_pdb_filenames) > num_variants_per_species:
            subset_filenames.update(
                random.sample(other_pdb_filenames, num_variants_per_species)
            )
        else:
            subset_filenames.update(other_pdb_filenames)

        logger.info(
            f"Selected {len(subset_filenames)} sequences for the representative FASTA file."
        )

        sequences_to_write = []
        for filename in sorted(list(subset_filenames)):
            if filename in self.all_pdb_paths:
                path = self.all_pdb_paths[filename]
                sequence = self._extract_sequence_from_pdb(path)
                if sequence:
                    seq_record = SeqRecord(Seq(sequence), id=filename, description="")
                    sequences_to_write.append(seq_record)
            else:
                logger.warning(f"Base PDB specified but not found: {filename}")

        if not sequences_to_write:
            logger.error("Failed to extract any sequences for the subset.")
            return

        logger.info(
            f"Saving {len(sequences_to_write)} sequences to {output_fasta_file}"
        )
        SeqIO.write(sequences_to_write, output_fasta_file, "fasta")
        print(f"\nFASTA file '{output_fasta_file}' created successfully.")
        print(
            "Please upload this file to an alignment web service (e.g., Clustal Omega) to generate the scaffold alignment."
        )


if __name__ == "__main__":
    # --- Configuration ---
    # 1. List of directories containing ALL your PDB files (non-speech_AF)
    pdb_dirs = [
        # "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Penguin",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Penguin_w_dropout",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Bos_mutus_w_dropout",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Camelus_ferus_w_dropout",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Chrysemys_picta_bellii_w_dropout",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Dryophytes_japonicus_w_dropout",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Jaculus_jaculus_w_dropout",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Nanorana_parkeri_w_dropout",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Wood_Frog_w_dropout_all\\Wood_Frog_w_dropout_1",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Wood_Frog_w_dropout_all\\Wood_Frog_w_dropout_2",
    ]

    # 2. List of the "base" PDB filenames, one for each species. These will
    #    form the foundation of the alignment.
    base_pdbs = [
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Best_PDB_Base\\Aptenodytes_forsteri_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_000.pdb",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Best_PDB_Base\\Bos_mutus_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_000.pdb",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Best_PDB_Base\\Camelus_ferus_unrelaxed_rank_001_alphafold2_ptm_model_4_seed_000.pdb",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Best_PDB_Base\\Chrysemys_picta_bellii_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_000.pdb",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Best_PDB_Base\\Dryophytes_japonicus_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_000.pdb",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Best_PDB_Base\\Jaculus_jaculus_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_000.pdb",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Best_PDB_Base\\Nanorana_parkeri_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_000.pdb",
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\Best_PDB_Base\\wood_frog_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_000.pdb",
    ]

    # 3. How many additional random variants to include in the alignment.
    #    A small number (e.g., 50-200) is usually sufficient.
    num_additional_variants = 200

    # 4. Name of the output FASTA file to be created.
    output_fasta = "representative_sequences.fasta"

    # --- Run Script ---
    generator = FastaSubsetGenerator(
        pdb_directories=pdb_dirs, base_pdb_filenames=base_pdbs
    )
    generator.create_subset_fasta(
        num_variants_per_species=num_additional_variants, output_fasta_file=output_fasta
    )
