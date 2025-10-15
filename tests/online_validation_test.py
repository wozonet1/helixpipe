# 文件: tests/online_validation_test.py (还原后的简洁版)

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import research_template as rt
from data_utils.debug_utils import run_online_validation

rt.register_hydra_resolvers()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("--- [Online Validation] Starting ---")

    # 即使在这里立即实例化，也不会再引起路径解析的错误

    cfg = rt.instantiate_data_config(cfg=cfg)
    print("\n--- Current Data Structure Config ---")
    print(OmegaConf.to_yaml(cfg.data_structure))
    print("\n--- Current Data Params Config ---")
    print(OmegaConf.to_yaml(cfg.data_params))

    final_dti_path = rt.get_path(cfg, "data_structure.paths.raw.authoritative_dti")

    # ... 后续逻辑不变 ...
    main_df = pd.read_csv(final_dti_path)
    run_online_validation(
        main_df, n_samples=200, n_jobs=6, random_state=cfg.runtime.seed
    )

    print("\n--- [Online Validation] Finished ---")


if __name__ == "__main__":
    main()
