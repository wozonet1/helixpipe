import hydra
from omegaconf import DictConfig, OmegaConf
import research_template as rt

# Import all your core logic building blocks
# (We need to refactor data_proc.py and main.py into functions)
from data_processing.main_pipeline import process_data
from train import train


def is_config_valid(config: DictConfig) -> bool:
    """
    Checks if the experiment configuration is logically valid.

    Rules:
    1. If gtopdb is NOT used (`use_gtopdb: false`), then relations involving 'l'
       (ligand) are FORBIDDEN.
    2. If gtopdb IS used (`use_gtopdb: true`), then relations involving 'l'
       are MANDATORY (at least one 'l' relation must be enabled).

    Args:
        config (DictConfig): The fully composed Hydra configuration object.

    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    try:
        # --- Rule 1: Check for forbidden relations when gtopdb is off ---
        use_gtopdb = config.data.use_gtopdb

        # Access the dictionary of relation switches
        # The actual config group is 'relations', which contains a 'params' sub-key,
        # which in turn contains 'include_relations'. Let's use a robust access method.
        # UPDATE: Based on your provided config structure, the path is simpler.
        include_relations = config.relations.flags

        # Define which relation keys are considered 'ligand-related'
        ligand_related_keys = ["lp_interaction", "ll_similarity", "dl_similarity"]

        # Check if any ligand-related relation is enabled
        any_l_relation_enabled = any(
            include_relations.get(key, False) for key in ligand_related_keys
        )

        if not use_gtopdb:
            if any_l_relation_enabled:
                print(
                    "[CONFIG INVALID] Run skipped: Ligand relations (e.g., lp, ll, dl) are enabled, but 'use_gtopdb' is false."
                )
                return False

        # --- Rule 2: Check for mandatory relations when gtopdb is on ---
        else:  # This means use_gtopdb is True
            if not any_l_relation_enabled:
                print(
                    "[CONFIG INVALID] Run skipped: 'use_gtopdb' is true, but no ligand-related relations are enabled in the config."
                )
                return False

    except Exception as e:
        # This will catch errors if the config structure is unexpected (e.g., 'include_relations' is missing)
        print(
            f"[CONFIG CHECK FAILED] Could not validate config due to an error: {e}. Skipping run."
        )
        return False

    # If all checks pass, the configuration is valid
    return True


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

    rt.set_seeds(config_dict.runtime.seed)
    rt.setup_dataset_directories(config_dict)  # Handles directory creation and cleaning
    if not config_dict.runtime.get("skip_data_proc", False):
        process_data(config_dict)
    train(config_dict)


if __name__ == "__main__":
    run_experiment()
