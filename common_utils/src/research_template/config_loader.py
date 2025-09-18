# In config_loader.py

import yaml
import argparse
from pathlib import Path


def load_config_with_override(config_path: str = "config.yaml") -> dict:
    """
    Loads a YAML config file and allows overriding its values with command-line arguments.

    Command-line arguments should be in the format:
    `--config key1.subkey=value key2=value2`

    e.g., `python script.py --config data.use_gtopdb=true training.k_folds=10`

    Args:
        config_path (str, optional): Path to the base YAML config file.

    Returns:
        dict: The final configuration dictionary after applying overrides.
    """
    # 1. Load the base configuration from the YAML file
    project_root = Path.cwd()
    config_file_path = project_root / config_path
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Set up argparse to parse a special '--config' argument
    parser = argparse.ArgumentParser(
        description="Run experiment with YAML config and CLI overrides."
    )
    # 'nargs='*'' allows us to accept multiple key=value pairs
    parser.add_argument(
        "--config",
        nargs="*",
        default=[],
        help="Override config values, e.g., key.subkey=value",
    )

    args, _ = (
        parser.parse_known_args()
    )  # Use parse_known_args to ignore other potential args

    # 3. Apply the overrides to the loaded config dictionary
    for override in args.config:
        try:
            key_str, value = override.split("=", 1)
            keys = key_str.split(".")

            # Navigate the dictionary to the target key
            d = config
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}  # Create nested dicts if they don't exist
                d = d[key]

            # Try to automatically convert the value to the correct type
            final_key = keys[-1]
            try:
                # Attempt to convert to int, then float, then bool, otherwise keep as string
                if "." in value:
                    d[final_key] = float(value)
                elif value.lower() in ["true", "false"]:
                    d[final_key] = value.lower() == "true"
                else:
                    d[final_key] = int(value)
            except ValueError:
                d[final_key] = value  # Keep as string if conversion fails

        except ValueError:
            print(
                f"--> WARNING: Invalid override format '{override}'. Skipping. Use 'key=value' format."
            )

    return config
