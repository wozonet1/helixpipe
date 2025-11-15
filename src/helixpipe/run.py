import logging

import hydra

import helixlib as hx
from helixpipe.configs import register_all_schemas
from helixpipe.data_loader_strategy import load_datasets
from helixpipe.pipelines import process_data
from helixpipe.typing import AppConfig

# from helixpipe.train import train  # noqa: F401
# <--- 导入我们的新策略函数
from helixpipe.utils import register_hydra_resolvers, setup_logging

register_all_schemas()
register_hydra_resolvers()

project_root = hx.get_project_root()
config_path = f"{project_root}/conf"
logger = logging.getLogger(__name__)


@hydra.main(config_path=f"{config_path}", config_name="config", version_base=None)
def run_experiment(cfg: AppConfig) -> None:
    """
    项目主实验流程的顶层入口。
    """
    setup_logging(cfg)
    # --- 阶段 1: 调用策略函数加载数据 ---
    # 这一行代码取代了所有之前的数据加载逻辑
    processor_outputs = load_datasets(cfg)

    process_data(cfg, processor_outputs)


if __name__ == "__main__":
    run_experiment()
