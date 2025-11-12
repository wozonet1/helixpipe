import hydra
import research_template as rt

from nasnet.configs import AppConfig, register_all_schemas
from nasnet.data_loader_strategy import load_datasets
from nasnet.pipelines import process_data

# from nasnet.train import train  # noqa: F401
# <--- 导入我们的新策略函数
from nasnet.utils import register_hydra_resolvers

register_all_schemas()
register_hydra_resolvers()

project_root = rt.get_project_root()
config_path = f"{project_root}/conf"


@hydra.main(config_path=f"{config_path}", config_name="config", version_base=None)
def run_experiment(cfg: AppConfig):
    """
    项目主实验流程的顶层入口。
    """
    # --- 阶段 1: 调用策略函数加载数据 ---
    # 这一行代码取代了所有之前的数据加载逻辑
    processor_outputs = load_datasets(cfg)

    # --- 阶段 2: 验证配置与加载的数据是否逻辑一致 ---
    # if not validate_config_with_data(cfg, base_df, extra_dfs):
    #     print("❌ Halting execution due to configuration inconsistency.")
    #     return

    # --- 阶段 3: 运行下游数据处理流水线 (图构建等) ---
    if not cfg.runtime.skip_data_proc:
        process_data(cfg, processor_outputs)
    else:
        print("\n⚠️  Skipping data processing as per 'runtime.skip_data_proc' flag.")

    # --- 阶段 4: 运行模型训练 ---
    # train(cfg)


if __name__ == "__main__":
    run_experiment()
