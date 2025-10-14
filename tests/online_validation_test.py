import pandas as pd
from data_utils.debug_utils import run_online_validation
import research_template as rt
import hydra
from omegaconf import OmegaConf

rt.register_hydra_resolvers()


# 1. 加载您最终生成的权威DTI文件
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    print("--- [Online Validation] Starting ---")

    # 打印出当前正在验证的数据集配置，以便确认
    print("\n--- Current Data Structure Config ---")
    print(OmegaConf.to_yaml(cfg.data_structure))
    print("-" * 35)

    # 1. 加载您最终生成的权威DTI文件
    #    get_path 会根据传入的 cfg 动态地构建正确的路径
    print(
        f"--> Attempting to load authoritative DTI file for dataset: '{cfg.data_structure.primary_dataset}'"
    )
    final_dti_path = rt.get_path(cfg, "data_structure.paths.raw.authoritative_dti")

    try:
        main_df = pd.read_csv(final_dti_path)
        print(f"--> Successfully loaded file: {final_dti_path}")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{final_dti_path}'.")
        print(
            "   Please make sure you have run the corresponding data processing pipeline first."
        )
        return  # 优雅地退出

    # 2. 运行在线验证
    print("\n--> Starting online validation...")
    run_online_validation(
        main_df, n_samples=200, n_jobs=6, random_state=cfg.runtime.seed
    )
    print("\n--- [Online Validation] Finished ---")


if __name__ == "__main__":
    # 现在 __main__ 块只负责调用由 @hydra.main 装饰的 main 函数。
    # 所有初始化逻辑都移到了 main 函数内部或文件顶部。
    main()
