import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from typing import List
from data_utils.debug_utils import is_config_valid
import research_template as rt
from data_processing.main_pipeline import process_data
from train import train

rt.register_hydra_resolvers()


def load_data_according_to_strategy(cfg: DictConfig) -> List[pd.DataFrame]:
    """
    【新增】一个专门负责根据配置加载和组合数据集的策略函数。
    """
    print("--- [Data Loading Strategy] Executing... ---")
    dataframes_to_process = []

    # 1. 加载主数据集
    #    主数据集的配置由 `data_structure` 组决定
    print(f"--> Loading PRIMARY dataset: '{cfg.data_structure.name}'")
    primary_dti_path = rt.get_path(cfg, "data_structure.paths.raw.authoritative_dti")
    try:
        primary_df = pd.read_csv(primary_dti_path)
        dataframes_to_process.append(primary_df)
        print(f"    - Loaded {len(primary_df)} interactions.")
    except FileNotFoundError:
        print(f"❌ FATAL: Primary DTI file not found at '{primary_dti_path}'.")
        raise

    # 2. 按需加载并合并“辅助”数据集

    aux_datasets_to_load = cfg.data_params.get("auxiliary_datasets", [])
    if aux_datasets_to_load:
        for dataset_name in aux_datasets_to_load:
            aux_df = rt.load_auxiliary_dataset(cfg, dataset_name)
            if aux_df is not None:
                dataframes_to_process.append(aux_df)

    return dataframes_to_process


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
