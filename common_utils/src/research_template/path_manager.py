from pathlib import Path
import shutil
import sys


def setup_dataset_directories(config: dict) -> None:
    """
    Checks and creates the necessary directory structure for the primary dataset.

    This function ensures that `raw/`, `processed/baseline/`, and `processed/gtopdb/`
    directories exist under the specified primary_dataset path. It also creates
    the `indexes` subdirectories within the processed variants.

    Args:
        config (dict): The loaded YAML configuration dictionary.
    """
    print("--- [Setup] Verifying and setting up dataset directories... ---")

    data_config = config["data"]
    root_path = Path(data_config["root"])
    runtime_config = config.get("runtime", {})
    primary_dataset = data_config.get("primary_dataset")

    if not primary_dataset:
        raise ValueError("'primary_dataset' not defined in the data configuration.")

    dataset_path = root_path / primary_dataset

    # 1. Define all required subdirectories
    raw_dir = dataset_path / data_config["subfolders"]["raw"]
    processed_dir = dataset_path / data_config["subfolders"]["processed"]
    use_gtopdb = data_config.get("use_gtopdb", False)
    variant_folder_name = "gtopdb" if use_gtopdb else "baseline"
    target_dir_to_clean = processed_dir / variant_folder_name

    # 2. Execute targeted deletion if force_restart is True
    force_restart = runtime_config.get("force_restart", False)
    if force_restart and target_dir_to_clean.exists():
        print(
            f"!!! WARNING: `force_restart` is True for the '{variant_folder_name}' variant."
        )
        print(f"    Deleting directory: {target_dir_to_clean}")
        try:
            shutil.rmtree(target_dir_to_clean)
            print(f"--> Successfully deleted old '{variant_folder_name}' directory.")
        except OSError as e:
            print(
                f"--> ERROR: Failed to delete directory {target_dir_to_clean}. Error: {e}"
            )
            sys.exit(1)

    baseline_dir = processed_dir / "baseline"
    gtopdb_dir = processed_dir / "gtopdb"

    # Also include the 'indexes' sub-subdirectories
    baseline_indexes_dir = baseline_dir / "indexes"
    gtopdb_indexes_dir = gtopdb_dir / "indexes"

    baseline_sim_dir = baseline_dir / "sim_matrixes"
    gtopdb_sim_dir = gtopdb_dir / "sim_matrixes"

    dirs_to_create = [
        raw_dir,
        processed_dir,
        baseline_dir,
        gtopdb_dir,
        baseline_indexes_dir,
        gtopdb_indexes_dir,
        baseline_sim_dir,
        gtopdb_sim_dir,
    ]

    # 2. Loop through and create them if they don't exist
    created_count = 0
    for directory in dirs_to_create:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"-> Created directory: {directory}")
            created_count += 1

    if created_count == 0:
        print("-> All necessary directories already exist. Setup complete.")
    else:
        print(
            f"-> Successfully created {created_count} new directories. Setup complete."
        )

    # [Optional but Recommended] A helpful message for the user
    if not any(raw_dir.iterdir()):
        print(f"-> WARNING: The 'raw' directory at '{raw_dir}' is empty.")
        print(
            "   Please make sure to place your raw data files (e.g., full.csv) inside it."
        )


def get_path(config: dict, file_key: str) -> Path:
    """
    Constructs the full path for any file defined in the config.
    This is the single source of truth for all file paths in the project.

    It understands the nested dataset structure (e.g., DrugBank/raw)
    and the `baseline/gtopdb` variant for processed files of the primary dataset.

    Args:
        config (dict): The loaded YAML configuration dictionary.
        file_key (str): A dot-separated key pointing to the filename in the config.
                        e.g., "DrugBank.raw.dti_interactions",
                        "DrugBank.processed.nodes_metadata",
                        "gtopdb.processed.interactions"

    Returns:
        Path: The complete Path object for the requested file.
    """
    data_config = config["data"]
    root_path = Path(data_config["root"])

    # 1. Parse the key
    key_parts = file_key.split(".")
    if len(key_parts) < 3:
        raise ValueError(
            f"Invalid file_key '{file_key}'. Must have at least 3 parts (e.g., 'DataSet.raw.fileName')."
        )

    dataset_name = key_parts[0]
    data_type = key_parts[1]  # 'raw' or 'processed'

    # 2. Determine the base directory
    subfolder_name = data_config["subfolders"].get(data_type)
    if not subfolder_name:
        raise KeyError(
            f"Data type '{data_type}' not defined in data.subfolders config."
        )

    base_dir = root_path / dataset_name / subfolder_name

    # 3. Handle the special case for 'processed' files of the PRIMARY dataset
    # They need an additional 'baseline' or 'gtopdb' sub-subfolder.
    primary_dataset_name = data_config.get("primary_dataset")
    if data_type == "processed" and dataset_name == primary_dataset_name:
        variant_folder = (
            "gtopdb" if data_config.get("use_gtopdb", False) else "baseline"
        )
        base_dir = base_dir / variant_folder

    # 4. Retrieve the filename from the config
    filename_dict_level = data_config["files"]
    try:
        for part in key_parts:
            filename_dict_level = filename_dict_level[part]
        filename = filename_dict_level
    except KeyError:
        raise KeyError(f"File key '{file_key}' not found in the config.yaml structure.")

    # Make sure the parent directory exists, creating it if necessary.
    # This is a good practice to avoid errors when writing files.
    base_dir.mkdir(parents=True, exist_ok=True)

    return base_dir / filename


def check_files_exist(config: dict, *file_keys: str) -> bool:
    """
    Checks if all specified data files (referenced by their config keys) exist.
    (This function can remain largely the same, as it relies on get_path)
    """
    for key in file_keys:
        try:
            filepath = get_path(config, key)
            if not filepath.exists():
                print(
                    f"File check FAILED: '{key}' not found at expected path: {filepath}"
                )
                return False
        except (KeyError, TypeError, ValueError) as e:
            print(f"File check FAILED: Could not resolve key '{key}'. Error: {e}")
            return False

    primary_dataset_name = config["data"].get("primary_dataset")
    variant_folder = "gtopdb" if config["data"].get("use_gtopdb", False) else "baseline"
    print(
        f"File check PASSED: All requested files exist for primary dataset '{primary_dataset_name}' (variant: '{variant_folder}')."
    )
    return True
