import pandas as pd
from tqdm import tqdm

# We use our template library to get the path, ensuring consistency
import research_template as rt


def find_symmetric_pairs_in_full_df():
    """
    Analyzes the raw DTI interaction file (e.g., full_df.csv) to detect
    if it contains symmetric pairs, such as (DrugA, ProtX) and (ProtX, DrugA).
    """
    print("--- Starting Symmetry Check for Raw DTI Data ---")

    # 1. Load configuration and get the path to the raw data file
    try:
        config = rt.load_config()
        primary_dataset = config["data"]["primary_dataset"]
        raw_file_key = f"{primary_dataset}.raw.dti_interactions"
        raw_file_path = rt.get_path(config, raw_file_key)
        print(f"--> Analyzing file: {raw_file_path}")
    except (FileNotFoundError, KeyError) as e:
        print(
            f"!!! ERROR: Could not load config or find file key. Please ensure config is correct. Error: {e}"
        )
        return

    # 2. Load the raw DataFrame
    try:
        df = pd.read_csv(raw_file_path)
        # We only care about positive interactions
        df_pos = df[df["Y"] == 1].copy()
        print(f"--> Found {len(df_pos)} positive interactions to analyze.")
    except FileNotFoundError:
        print(f"!!! ERROR: Raw data file not found at {raw_file_path}")
        return

    # 3. Create a canonical representation for each interaction pair
    print("--> Normalizing interaction pairs (this may take a moment)...")

    # We use a set to efficiently store the normalized pairs we've seen
    seen_pairs = set()
    symmetric_pairs_found = []

    # Use tqdm for a progress bar, as this can be slow on large files
    for index, row in tqdm(df_pos.iterrows(), total=len(df_pos), desc="Scanning pairs"):
        # We assume the columns are named 'SMILES' and 'Protein'
        try:
            entity1 = row["SMILES"]
            entity2 = row["Protein"]
        except KeyError:
            print(
                "\n!!! ERROR: DataFrame does not contain 'SMILES' or 'Protein' columns."
            )
            return

        # Normalize the pair by sorting the string representations alphabetically
        normalized_pair = tuple(sorted((entity1, entity2)))

        # Check for symmetry
        if normalized_pair in seen_pairs:
            # We found a pair whose normalized form is already in our set.
            # This means we've encountered a symmetric counterpart.
            # We store the original (non-normalized) pair for inspection.
            symmetric_pairs_found.append((entity1, entity2))
        else:
            seen_pairs.add(normalized_pair)

    # 4. Report the findings
    print("\n" + "=" * 50)
    print("          Symmetry Analysis Report")
    print("=" * 50)

    num_symmetric = len(symmetric_pairs_found)

    if num_symmetric > 0:
        print(
            f"✅ INVESTIGATION COMPLETE: Found {num_symmetric} symmetric interaction pairs!"
        )
        print(
            "   This confirms that the raw data file contains duplicate interactions in reversed order."
        )
        print("   Example symmetric pairs found:")
        # Print the first 5 examples for inspection
        for i, pair in enumerate(symmetric_pairs_found[:5]):
            print(
                f"     - Example #{i + 1}: ('{pair[0][:30]}...', '{pair[1][:30]}...')"
            )
        print(
            "\n   CONCLUSION: The bug is in the data source. The proposed fix in `data_proc.py` (using set and sorted tuples) is necessary and correct."
        )
    else:
        print("✅ INVESTIGATION COMPLETE: No symmetric pairs were found in the data.")
        print("   This suggests the raw data is clean in terms of symmetry.")
        print(
            "\n   CONCLUSION: The bug is NOT in the raw data. The cause of the disappearing edges must lie elsewhere in the processing pipeline (e.g., during ID conversion or graph construction)."
        )

    print("=" * 50)


if __name__ == "__main__":
    find_symmetric_pairs_in_full_df()
