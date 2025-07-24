import re
import os
import argparse

def split_fasta_by_species(input_filepath, output_dir="."):
    """
    Reads a FASTA file and splits it into multiple FASTA files based on the
    species name in the description line.

    The species name is expected to be enclosed in square brackets, like [Species name].
    Entries without a species name in brackets or without a sequence are ignored.

    Args:
        input_filepath (str): The path to the input FASTA file.
        output_dir (str): The directory where the output files will be saved.
                          Defaults to the current directory.
    """
    # --- Input Validation ---
    if not os.path.exists(input_filepath):
        print(f"Error: The file '{input_filepath}' was not found.")
        return

    # --- Create Output Directory if it doesn't exist ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output will be saved to: '{os.path.abspath(output_dir)}'")
    except OSError as e:
        print(f"Error creating output directory '{output_dir}': {e}")
        return

    print(f"Processing file: {input_filepath}")

    try:
        # --- Read the entire file content ---
        with open(input_filepath, 'r') as f:
            content = f.read()
    except IOError as e:
        print(f"Error reading file: {e}")
        return

    # --- Split the file into individual FASTA entries ---
    # The split is done on '>' which marks the start of a new sequence description.
    # The first element of the split will be empty if the file starts with '>', so we slice it off.
    fasta_entries = content.split('>')[1:]

    if not fasta_entries:
        print("No FASTA entries found in the file.")
        return

    # --- Process each entry ---
    sequences_written = 0
    for entry in fasta_entries:
        # Split the entry into lines and remove any empty lines
        lines = [line for line in entry.strip().split('\n') if line.strip()]
        
        # An entry must have at least a header and one sequence line
        if len(lines) < 2:
            # This handles entries with a missing sequence
            print(f"Skipping entry due to missing sequence: '{lines[0] if lines else 'N/A'}'")
            continue

        header = lines[0]
        sequence = "".join(lines[1:])

        # --- Extract species name using a regular expression ---
        # This looks for any characters inside square brackets []
        match = re.search(r'\[(.*?)\]', header)

        if match:
            # The actual matched group is retrieved with group(1)
            species_name = match.group(1).strip()
            
            # --- Sanitize the species name to create a valid filename ---
            # Replace spaces with underscores and remove characters not allowed in filenames
            safe_filename_base = re.sub(r'[\\/*?:"<>|]', "", f"{species_name}.fasta")
            output_filepath = os.path.join(output_dir, safe_filename_base)

            try:
                # --- Write the new FASTA file ---
                with open(output_filepath, 'w') as output_file:
                    # Write the header with the '>' prefix
                    output_file.write(f">{header}\n")
                    # Write the sequence
                    output_file.write(f"{sequence}\n")
                
                print(f"Successfully created file: '{output_filepath}'")
                sequences_written += 1
            except IOError as e:
                print(f"Error writing file '{output_filepath}': {e}")
        else:
            # This handles entries without a species name in brackets
            print(f"Skipping entry due to missing species name in brackets: '{header}'")

    print(f"\nProcessing complete. A total of {sequences_written} files were created.")

# --- Command-line Interface ---
parser = argparse.ArgumentParser(
    description="Split a multi-sequence FASTA file into separate files for each species."
)
parser.add_argument(
    "input_file", 
    help="The path to the input FASTA file."
)
parser.add_argument(
    "-o", "--output_dir", 
    default=".", 
    help="The directory to save the output files. Defaults to the current directory."
)

args = parser.parse_args()

# Call the function with the parsed command-line arguments
split_fasta_by_species(args.input_file, args.output_dir)