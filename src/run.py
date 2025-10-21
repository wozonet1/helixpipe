import hydra
import research_template as rt
from omegaconf import OmegaConf

from configs.register_schemas import AppConfig, register_all_schemas
from data_processing.id_validator import generate_id_whitelists
from data_processing.main_pipeline import process_data
from data_utils.data_loader_strategy import (
    load_datasets,
)

# <--- 导入我们的新策略函数
from data_utils.debug_utils import validate_config_with_data
from train import train  # noqa: F401

# from src.train import train # (暂时注释)
register_all_schemas()
rt.register_hydra_resolvers()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run_experiment(cfg: AppConfig):
    """
    项目主实验流程的顶层入口。
    """
    print(OmegaConf.to_yaml(cfg))
    generate_id_whitelists(cfg)
    # --- 阶段 1: 调用策略函数加载数据 ---
    # 这一行代码取代了所有之前的数据加载逻辑
    base_df, extra_dfs = load_datasets(cfg)

    # --- 阶段 2: 验证配置与加载的数据是否逻辑一致 ---
    if not validate_config_with_data(cfg, base_df, extra_dfs):
        print("❌ Halting execution due to configuration inconsistency.")
        return

    # --- 阶段 3: 运行下游数据处理流水线 (图构建等) ---
    if not cfg.runtime.skip_data_proc:
        process_data(cfg, base_df=base_df, extra_dfs=extra_dfs)
    else:
        print("\n⚠️  Skipping data processing as per 'runtime.skip_data_proc' flag.")

    # --- 阶段 4: 运行模型训练 ---
    # train(cfg)


if __name__ == "__main__":
    run_experiment()
