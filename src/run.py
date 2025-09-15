import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import research_template as rt

# Import all your core logic building blocks
# (We need to refactor data_proc.py and main.py into functions)
from data_proc import process_data
from train import train


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig):
    """
    Main entry point for the experiment, driven by Hydra.
    'cfg' is the fully composed configuration object.
    """
    # --- 1. Print and Setup ---
    print("--- Fully Composed Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    config_dict = cfg

    rt.set_seeds(config_dict.runtime.seed)
    rt.setup_dataset_directories(config_dict)  # Handles directory creation and cleaning
    device = torch.device(
        config_dict.runtime.seed if torch.cuda.is_available() else "cpu"
    )

    # --- 2. Data Processing Stage ---
    # The logic from data_proc.py is called here, passing the config.
    # It needs to be refactored to not be a standalone script.
    process_data(config_dict)

    # --- 3. Training/Evaluation Stage ---
    # The logic from main.py is called here.
    # MLflow logging would happen inside this function.

    # train(config_dict, device)


if __name__ == "__main__":
    run_experiment()
