"""
A common utility library for reproducible research projects.
This __init__.py file serves as the public API for the research_template package.
"""

# --- Import key functions from submodules to expose them at the top level ---

# From config_loader.py
from .config_loader import load_config_with_override as load_config

# From path_manager.py
from .path_manager import (
    get_path,
    check_files_exist,
    setup_dataset_directories,
    ensure_path_exists,
    get_relations_suffix,
)

# From reproducibility.py
from .reproducibility import set_seeds

# From graph_utils.py
from .graph_utils import (
    sparse_mx_to_torch_sparse_tensor,
    normalize_sparse_matrix,
)

# From metrics.py
from .metrics import accuracy

from .tracking import MLflowTracker

# --- You can also define a version for your library ---
__version__ = "0.1.0"

# --- Control what `from research_template import *` does (optional but good practice) ---
__all__ = [
    # List all the functions you want to be importable with '*'
    "load_config",
    "get_path",
    "check_files_exist",
    "setup_dataset_directories",
    "set_seeds",
    "sparse_mx_to_torch_sparse_tensor",
    "normalize_sparse_matrix",
    "accuracy",
    "MLflowTracker",
    "ensure_path_exists",
    "get_relations_suffix",
]
