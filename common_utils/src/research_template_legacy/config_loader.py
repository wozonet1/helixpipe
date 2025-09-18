from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads the YAML config file from the current working directory.

    This function is designed to be called from a main script located in
    the project's root or `src` directory. It assumes `config.yaml` is in
    the project root.

    Args:
        config_path (str, optional): The name of the config file.
                                     Defaults to "config.yaml".

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    # [MODIFIED] Use Path.cwd() to get the current working directory
    # This is the directory from where you run `python src/main.py`
    project_root = Path.cwd()

    config_file_path = project_root / config_path

    if not config_file_path.exists():
        raise FileNotFoundError(
            f"Config file not found at expected path: {config_file_path}\n"
            f"Please ensure you are running your script from the project's root directory "
            f"(e.g., '/e/zhucj/nasnet/'), and that '{config_path}' exists there."
        )

    with open(config_file_path, "r") as f:
        return yaml.safe_load(f)
