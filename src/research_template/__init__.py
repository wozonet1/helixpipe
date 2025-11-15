"""
A common utility library for reproducible research projects.
This __init__.py file serves as the public API for the research_template package.
"""

# From config_loader.py
from .caching import run_cached_operation
from .config_loader import load_config_with_override as load_config
from .errors import ConfigPathError, SchemaRegistrationError

# From graph_utils.py
from .graph_utils import (
    get_canonical_relation,
)

# From metrics.py
from .metrics import accuracy

# From path_manager.py
from .path_manager import (
    check_paths_exist,
    ensure_path_exists,
    get_project_root,
)

# From reproducibility.py
from .reproducibility import set_seeds
from .tracking import MLflowTracker

# --- You can also define a version for your library ---
__version__ = "0.1.0"

# --- Control what `from research_template import *` does (optional but good practice) ---
__all__ = [
    # List all the functions you want to be importable with '*'
    "load_config",
    "check_paths_exist",
    "set_seeds",
    "accuracy",
    "MLflowTracker",
    "ensure_path_exists",
    "get_project_root",
    "get_canonical_relation",
    "ConfigPathError",
    "SchemaRegistrationError",
    "run_cached_operation",
]
