import os
from itertools import combinations
import numpy as np
from scipy.stats import entropy


def parse_all_characters_and_counts(filepath):
    """
    Parses an .a3m file and counts the occurrences of all characters
    (amino acids, gaps, lowercase insertions) across all sequences.

    Args:
        filepath (str): The full path to the .a3m file.

    Returns:
        dict: A dictionary where keys are characters and values are their counts.
              Returns an empty dict if the file is not found or no sequences are parsed.
    """
    char_counts = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith(">"):
                    continue  # Skip header lines
                else:
                    # Iterate through all characters in the sequence line
                    for char in line.strip():
                        char_counts[char] = char_counts.get(char, 0) + 1
        return char_counts
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return {}
    except Exception as e:
        print(f"Error reading or parsing {filepath}: {e}")
        return {}


def calculate_jsd(dist1_counts, dist2_counts):
    """
    Calculates the Jensen-Shannon Divergence (JSD) between two amino acid
    frequency distributions.

    Args:
        dist1_counts (dict): Character counts for the first MSA.
        dist2_counts (dict): Character counts for the second MSA.

    Returns:
        float: The Jensen-Shannon Divergence (0.0 to ~0.693 for natural log, or 0 to 1 for log2).
               Returns 0.0 if both distributions are empty.
    """
    if not dist1_counts and not dist2_counts:
        return 0.0

    # Create a unified alphabet from all characters present in both distributions
    all_chars = set(dist1_counts.keys()).union(set(dist2_counts.keys()))
    sorted_chars = sorted(list(all_chars))  # Ensure consistent order

    # Convert counts to probability arrays, adding a small pseudocount
    # to avoid log(0) and ensure all characters are represented.
    pseudocount = 1e-9  # A very small number to handle zero probabilities

    total1 = sum(dist1_counts.values()) + len(sorted_chars) * pseudocount
    total2 = sum(dist2_counts.values()) + len(sorted_chars) * pseudocount

    p1 = np.array(
        [(dist1_counts.get(char, 0) + pseudocount) / total1 for char in sorted_chars]
    )
    p2 = np.array(
        [(dist2_counts.get(char, 0) + pseudocount) / total2 for char in sorted_chars]
    )

    # Calculate the average distribution M
    m = 0.5 * (p1 + p2)

    # Calculate KL Divergence from P1 to M and P2 to M
    # entropy(pk, qk) calculates sum(pk * log(pk / qk))
    kl_p1_m = entropy(p1, m, base=2)  # Using base 2 for log, JSD ranges from 0 to 1
    kl_p2_m = entropy(p2, m, base=2)

    # Jensen-Shannon Divergence
    jsd = 0.5 * (kl_p1_m + kl_p2_m)
    return jsd


def main(directory_path=None):
    """
    Main function to find .a3m files, parse all characters and their counts,
    calculate pairwise Jensen-Shannon Divergence, and output the results in a matrix format
    with shortened filenames.
    """
    if directory_path is None:
        directory_path = input("Enter the directory path containing .a3m files: ").strip()

    # Validate directory path
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return

    # Find all .a3m files in the specified directory
    a3m_files = [f for f in os.listdir(directory_path) if f.endswith(".a3m")]

    if not a3m_files:
        print(
            f"No .a3m files found in '{directory_path}'. Please ensure files end with '.a3m'."
        )
        return

    print(
        f"\nFound {len(a3m_files)} .a3m files. Parsing all characters and their counts from each..."
    )

    # Dictionary to store {original_filename: character_counts_dict}
    all_msa_char_counts = {}
    for filename in a3m_files:
        filepath = os.path.join(directory_path, filename)
        char_counts = parse_all_characters_and_counts(filepath)
        if char_counts:
            all_msa_char_counts[filename] = char_counts
            total_chars = sum(char_counts.values())
            print(
                f"  - Parsed {total_chars} characters from {filename} ({len(char_counts)} unique characters)"
            )
        else:
            print(
                f"Warning: No valid characters found in {filename}. Skipping this file for comparison."
            )

    # Filter out files that had no characters and sort them
    valid_filenames = sorted(list(all_msa_char_counts.keys()))

    if len(valid_filenames) < 2:
        print(
            "Not enough valid .a3m files with extractable characters to perform comparisons."
        )
        return

    # Create a mapping for shortened filenames for display
    shortened_names_map = {
        original: original.split(".")[0][:6].ljust(
            6
        )  # Pad with spaces if shorter than 6
        for original in valid_filenames
    }
    shortened_valid_filenames = [shortened_names_map[f] for f in valid_filenames]

    # Initialize the difference matrix
    difference_matrix = {
        name: {other_name: 0.0 for other_name in valid_filenames}
        for name in valid_filenames
    }

    # Calculate pairwise differences (JSD)
    print("\n--- Calculating Pairwise MSA Differences (Jensen-Shannon Divergence) ---")
    file_pairs = list(combinations(valid_filenames, 2))

    for file1, file2 in file_pairs:
        counts1 = all_msa_char_counts[file1]
        counts2 = all_msa_char_counts[file2]

        jsd_value = calculate_jsd(counts1, counts2)
        # JSD is already a "distance" or "difference" metric, no need to convert to percentage
        # It ranges from 0 (identical distributions) to 1 (maximally different distributions)

        # Store the difference symmetrically
        difference_matrix[file1][file2] = jsd_value
        difference_matrix[file2][file1] = jsd_value

    print("\n--- MSA Jensen-Shannon Divergence Matrix ---")
    print(" (Metric: Jensen-Shannon Divergence. Ranges from 0.0 to 1.0.)")
    print(
        " (0.0 means identical amino acid distributions; 1.0 means maximally different.)"
    )
    print(" (All characters (uppercase, lowercase, gaps) are considered.)")
    print(" (Filenames are truncated to 6 characters for display.)")

    # Determine column width for formatting
    column_width = 10  # Sufficient for "X.XXXX" and 6 char names

    # Print the header row for the matrix
    header_format = "{:<{width}}".format("", width=column_width) + "".join(
        [
            "{:>{width}}".format(name, width=column_width)
            for name in shortened_valid_filenames
        ]
    )
    print(header_format)
    print("-" * (column_width * (len(valid_filenames) + 1)))  # Separator line

    # Print each row of the matrix
    for row_original_file in valid_filenames:
        row_short_name = shortened_names_map[row_original_file]
        row_string = "{:<{width}}".format(row_short_name, width=column_width)
        for col_original_file in valid_filenames:
            value = difference_matrix[row_original_file][col_original_file]
            row_string += "{:>{width}.4f}".format(
                value, width=column_width
            )  # Format to 4 decimal places for JSD
        print(row_string)

    print("\n--- Analysis Complete ---")
    print(
        "The matrix shows the Jensen-Shannon Divergence between the overall amino acid (and gap/insertion) distributions of each MSA pair."
    )
    print(
        "A higher JSD value indicates greater dissimilarity in the 'information' content of the MSAs, reflecting differences in their overall character composition and variability."
    )
    print(
        "This metric is more sensitive to the positional frequencies of characters than the previous Jaccard Distance."
    )


DIRECTORY_PATH = "SPEECH_AF_Output"
main(DIRECTORY_PATH)
