import os
import numpy as np


def analyze_amino_acid_conservation(filepath, amino_acid_code):
    """
    Analyzes the conservation of a specific amino acid at a given position
    within sequences in an .a3m file, and provides the probability distribution
    of all characters at that position.

    Args:
        filepath (str): The full path to the .a3m file.
        amino_acid_code (str): The amino acid and its position (e.g., 'H3', 'W10').
                                The amino acid should be a single uppercase letter.
    Returns:
        dict: A dictionary containing conservation statistics, or None if an error occurs.
    """
    if not (
        2 <= len(amino_acid_code) <= 4
        and amino_acid_code[0].isalpha()
        and amino_acid_code[0].isupper()
        and amino_acid_code[1:].isdigit()
    ):
        print(
            "Error: Invalid amino_acid_code format. Expected format like 'H3' or 'W10'."
        )
        return None

    target_aa = amino_acid_code[0]
    target_position = int(amino_acid_code[1:]) - 1  # Convert to 0-based index

    sequences = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith(">"):
                    continue  # Skip header lines
                else:
                    sequences.append(line.strip())
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading or parsing {filepath}: {e}")
        return None

    if not sequences:
        print(f"No sequences found in {filepath}.")
        return None

    total_sequences = len(sequences)
    sequences_with_aa = 0
    sequences_without_aa = 0
    sequences_totally_omitted = 0
    amino_acid_counts_at_position = (
        {}
    )  # New dictionary to store counts of all characters at the position

    # Determine the maximum length among all sequences to handle varying lengths
    max_len = max(len(s) for s in sequences)

    if target_position >= max_len:
        print(
            f"Warning: Target position {target_position + 1} is beyond the length of all sequences in the file."
        )
        sequences_totally_omitted = total_sequences
        # Initialize counts for this case
        for char_code in "ACDEFGHIKLMNPQRSTVWXY-":  # Common amino acids and gap
            amino_acid_counts_at_position[char_code] = 0
    else:
        for seq in sequences:
            if target_position < len(seq):
                char_at_position = seq[target_position]
                amino_acid_counts_at_position[char_at_position] = (
                    amino_acid_counts_at_position.get(char_at_position, 0) + 1
                )

                if char_at_position == target_aa:
                    sequences_with_aa += 1
                else:
                    sequences_without_aa += 1
            else:
                # If the sequence is shorter than the target_position, the amino acid is "omitted" at that position
                sequences_totally_omitted += 1
                sequences_without_aa += (
                    1  # Also count as 'without' since it's not present
                )
                amino_acid_counts_at_position["<OMITTED>"] = (
                    amino_acid_counts_at_position.get("<OMITTED>", 0) + 1
                )  # Special key for omitted

    # Calculate probabilities for the distribution
    probability_distribution = {}
    total_observed_chars = sum(amino_acid_counts_at_position.values())

    if total_observed_chars > 0:
        for char, count in amino_acid_counts_at_position.items():
            probability_distribution[char] = count / total_observed_chars

    # Ensure all standard amino acids are in the distribution, even if count is 0
    standard_amino_acids = "ACDEFGHIKLMNPQRSTVWXY"  # 20 standard AA + Selenocysteine (U) and Pyrrolysine (O) not typically in A3M, X for unknown
    for aa in standard_amino_acids:
        if aa not in probability_distribution:
            probability_distribution[aa] = 0.0
    if "-" not in probability_distribution:  # Gap character
        probability_distribution["-"] = 0.0

    # Sort the probability distribution for consistent output
    sorted_probability_distribution = dict(sorted(probability_distribution.items()))

    # Calculate probabilities for target AA presence/absence based on the counts of the specific AA only
    prob_finding_aa = sequences_with_aa / total_sequences if total_sequences > 0 else 0
    prob_not_finding_aa = (
        sequences_without_aa / total_sequences if total_sequences > 0 else 0
    )

    results = {
        "filepath": filepath,
        "target_amino_acid": target_aa,
        "target_position": target_position + 1,  # Convert back to 1-based for output
        "total_sequences": total_sequences,
        "sequences_with_amino_acid": sequences_with_aa,
        "sequences_without_amino_acid": sequences_without_aa,
        "sequences_totally_omitted_at_position": sequences_totally_omitted,
        "probability_of_finding_amino_acid": prob_finding_aa,
        "probability_of_not_finding_amino_acid": prob_not_finding_aa,
        "amino_acid_probability_distribution": sorted_probability_distribution,  # Added distribution
    }
    return results


def main():
    """
    Main function to get user input and display amino acid conservation.
    """
    msa_file = "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result\\MSA_Base\\Nanorana_parkeri.a3m"
    amino_acid_input = "H283"

    conservation_data = analyze_amino_acid_conservation(msa_file, amino_acid_input)

    if conservation_data:
        print("\n--- Amino Acid Conservation Analysis Results ---")
        print(f"  File: {conservation_data['filepath']}")
        print(
            f"  Target Amino Acid: **{conservation_data['target_amino_acid']}** at Position **{conservation_data['target_position']}**"
        )
        print(f"  Total Sequences Analyzed: {conservation_data['total_sequences']} 📊")
        print("-" * 40)
        print(
            f"  Sequences with '{conservation_data['target_amino_acid']}' at position {conservation_data['target_position']}: **{conservation_data['sequences_with_amino_acid']}**"
        )
        print(
            f"  Sequences without '{conservation_data['target_amino_acid']}' at position {conservation_data['target_position']}: **{conservation_data['sequences_without_amino_acid']}**"
        )
        print(
            f"    (Specifically, sequences where the position is totally omitted: **{conservation_data['sequences_totally_omitted_at_position']}**)"
        )
        print("-" * 40)
        print(
            f"  Probability of finding '{conservation_data['target_amino_acid']}' at this position: **{conservation_data['probability_of_finding_amino_acid']:.4f}**"
        )
        print(
            f"  Probability of **not** finding '{conservation_data['target_amino_acid']}' at this position: **{conservation_data['probability_of_not_finding_amino_acid']:.4f}**"
        )
        print("\n--- Probability Distribution at Target Position ---")
        for aa, prob in conservation_data[
            "amino_acid_probability_distribution"
        ].items():
            print(f"  {aa}: {prob:.4f}")
        print("\n--- End of Analysis ---")


if __name__ == "__main__":
    main()
