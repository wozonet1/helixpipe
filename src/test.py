import pandas as pd
import research_template as rt


def investigate_dataframe():
    """
    Performs a deep inspection of the loaded DataFrame to diagnose
    why the positive sample filtering is failing.
    """
    print("--- Starting Deep DataFrame Investigation ---")

    # 1. Load config and get data path
    try:
        config = rt.load_config()
        primary_dataset = config["data"]["primary_dataset"]
        raw_file_key = f"{primary_dataset}.raw.dti_interactions"
        raw_file_path = rt.get_path(config, raw_file_key)
        print(f"--> Investigating file: {raw_file_path}")
    except Exception as e:
        print(f"!!! ERROR: Failed to load config or get path. Error: {e}")
        return

    # 2. Load the DataFrame
    try:
        df = pd.read_csv(raw_file_path)
    except FileNotFoundError:
        print(f"!!! ERROR: File not found at {raw_file_path}")
        return

    print("\n" + "=" * 50)
    print("1. DataFrame Info & Head")
    print("=" * 50)
    print("--> df.info():")
    df.info(verbose=True, show_counts=True)
    print("\n--> df.head():")
    print(df.head())

    print("\n" + "=" * 50)
    print("2. Column Name Inspection")
    print("=" * 50)

    # [KEY DEBUG 1] Print column names with surrounding characters to reveal hidden spaces
    column_reprs = [repr(col) for col in df.columns]
    print(f"--> Column names (raw representation): {column_reprs}")

    label_col_name = "Y"  # The name we expect
    if label_col_name not in df.columns:
        print(f"❌ FAILURE: Column '{label_col_name}' NOT FOUND in DataFrame columns.")
        # Let's try to find a similar column
        for col in df.columns:
            if label_col_name.lower() == col.strip().lower():
                print(
                    f"    - HINT: Found a similar column: {repr(col)}. It might have different case or hidden spaces."
                )
    else:
        print(f"✅ SUCCESS: Column '{label_col_name}' found.")

    print("\n" + "=" * 50)
    print("3. Label Column ('Y') Content Inspection")
    print("=" * 50)

    if label_col_name in df.columns:
        # [KEY DEBUG 2] Check the data type of the column
        label_dtype = df[label_col_name].dtype
        print(f"--> Data type of column '{label_col_name}': {label_dtype}")

        # [KEY DEBUG 3] Check the unique values and their counts
        value_counts = df[label_col_name].value_counts()
        print(f"\n--> Unique values in '{label_col_name}' and their counts:")
        print(value_counts)

        # [KEY DEBUG 4] Explicitly test the filtering condition
        try:
            # We first try to convert to a numeric type, in case it's a string
            numeric_labels = pd.to_numeric(df[label_col_name], errors="coerce")

            num_pos_samples_int = (numeric_labels == 1).sum()
            num_pos_samples_float = (numeric_labels == 1.0).sum()

            print(
                f"\n--> [TEST 1] Result of `(numeric_labels == 1).sum()`: {num_pos_samples_int}"
            )
            print(
                f"--> [TEST 2] Result of `(numeric_labels == 1.0).sum()`: {num_pos_samples_float}"
            )

            if num_pos_samples_int > 0 or num_pos_samples_float > 0:
                print(
                    "✅ SUCCESS: Filtering condition seems to work after converting to numeric."
                )
                print(
                    "   CONCLUSION: The 'Y' column is likely being read as a non-integer type (e.g., float or string)."
                )
            else:
                print(
                    "❌ FAILURE: Even after converting to numeric, no positive samples (value 1) were found."
                )
                print(
                    "   CONCLUSION: The raw data file might not contain any interactions labeled with the integer 1."
                )

        except Exception as e:
            print(f"\n!!! ERROR during filtering test: {e}")

    else:
        print(
            f"--> Skipping content inspection because column '{label_col_name}' was not found."
        )

    print("=" * 50)


if __name__ == "__main__":
    investigate_dataframe()
