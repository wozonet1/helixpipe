import hydra
from omegaconf import DictConfig, OmegaConf
import research_template as rt

# Import all your core logic building blocks
# (We need to refactor data_proc.py and main.py into functions)
from data_processing.main_pipeline import process_data
from train import train
from data_utils.debug_utils import is_config_valid


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig):
    """
    Main entry point for the experiment, driven by Hydra.
    'cfg' is the fully composed configuration object.
    """
    # --- 1. Print and Setup ---
    print("--- Fully Composed Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    if not is_config_valid(cfg):
        return
    config_dict = cfg
    rt.register_hydra_resolvers()
    rt.set_seeds(config_dict.runtime.seed)
    rt.setup_dataset_directories(config_dict)  # Handles directory creation and cleaning
    if not config_dict.runtime.get("skip_data_proc", False):
        process_data(config_dict)
    train(config_dict)


if __name__ == "__main__":
    run_experiment()
