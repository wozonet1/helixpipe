from pathlib import Path
import shutil
import sys
from omegaconf import DictConfig


def ensure_path_exists(filepath: Path):
    """
    Ensures that the parent directory of a given file path exists.
    If the directory does not exist, it is created.

    Args:
        filepath (Path): The Path object representing the full file path.
    """
    # .parent gets the directory containing the file
    # e.g., for Path('/path/to/my/file.csv'), .parent is Path('/path/to/my')
    parent_directory = filepath.parent

    # .mkdir() creates the directory.
    # `parents=True` means it will create any necessary parent directories as well.
    #   (e.g., if neither '/path' nor '/path/to' exist, it creates both)
    # `exist_ok=True` means it will NOT raise an error if the directory already exists.
    parent_directory.mkdir(parents=True, exist_ok=True)


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


def get_relations_suffix(cfg: DictConfig) -> str:
    """
    Generates a human-readable suffix based on the `relations` config group.
    """
    try:
        # The relations config is now a top-level component.
        abbrs = cfg.data.files.processed.abbr
        relations_flags = cfg.relations.flags
        suffix_parts = [
            abbrs[key]
            for key in sorted(abbrs.keys())
            if relations_flags.get(key, False)
        ]

        if not suffix_parts:
            return "no_relations"
        return "-".join(suffix_parts)

    except Exception as e:
        print(
            f"Error getting relations suffix. Check 'conf/relations/*' structure. Error: {e}"
        )
        return "config_error"


def get_path(cfg: DictConfig, file_key: str) -> Path:
    """
    Constructs the full path for any data file, understanding the difference
    between primary dataset files and auxiliary data source files.

    Args:
        cfg (DictConfig): The Hydra configuration object.
        file_key (str): A dot-separated key (e.g., "processed.nodes_metadata"
                        or "gtopdb.processed.ligands").

    Returns:
        Path: The complete, absolute Path object to the data file.
    """
    from hydra.utils import get_original_cwd

    project_root = Path(get_original_cwd())
    # Assuming your top-level data folder is named 'data' in the project root
    data_root = project_root / "data"

    key_parts = file_key.split(".")

    base_dir = None

    # --- [NEW CONTEXT-AWARE LOGIC] ---
    if key_parts[0] == "gtopdb":
        # CONTEXT 1: Auxiliary Data Source (gtopdb)
        # Path: data/gtopdb/raw/ or data/gtopdb/processed/
        # key_parts[0] is 'gtopdb', key_parts[1] is 'raw' or 'processed'
        data_source_name = key_parts[0]
        subfolder_name = key_parts[1]
        base_dir = data_root / data_source_name / subfolder_name

    else:
        # CONTEXT 2: Primary Dataset (davis, drugbank, etc.)
        # Path: data/davis/processed/baseline/
        primary_dataset = cfg.data.primary_dataset  # Correctly access dataset name
        subfolder_name = key_parts[0]  # 'raw' or 'processed'
        base_dir = data_root / primary_dataset / subfolder_name

        # The 'variant' sub-folder only applies to 'processed' data of the primary dataset
        if subfolder_name == "processed":
            variant = "gtopdb" if cfg.data.use_gtopdb else "baseline"
            base_dir = base_dir / variant

    # --- Filename lookup and rendering logic remains the same ---

    # 1. Retrieve the filename template by navigating the `files` block
    current_level = cfg.data.files
    for key in key_parts:
        current_level = current_level[key]
    filename_template = current_level

    # 2. Render the filename template if it's dynamic
    final_filename = filename_template
    if "{relations_suffix}" in str(filename_template):
        relations_suffix = get_relations_suffix(cfg)
        final_filename = filename_template.format(relations_suffix=relations_suffix)

    return base_dir / final_filename


def check_files_exist(config: DictConfig, *file_keys: str) -> bool:
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
