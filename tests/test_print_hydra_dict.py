import hydra
from omegaconf import DictConfig, OmegaConf
import traceback

# 导入我们的注册函数
from configs.register_schemas import register_all_schemas

# 在所有Hydra操作之前，执行注册
try:
    register_all_schemas()
except Exception as e:
    # 捕获在注册过程中发生的任何错误
    # 并用一种更简洁的方式打印出来
    print("\n" + "=" * 80)
    print("❌ FATAL: Failed to register Hydra schemas.")
    print(f"   Error Type: {type(e).__name__}")
    print(f"   Error Message: {e}")
    print("   Please check your dataclass definitions in 'src/configs/'.")
    print("=" * 80)
    # 可以在这里选择退出，而不是打印长长的堆栈
    # 【核心修正】打印完整的、详细的堆栈跟踪信息
    print("--- Full Stack Trace ---")
    traceback.print_exc()
    print("------------------------")
    import sys

    sys.exit(1)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):  # <-- cfg 仍然是 DictConfig
    # 【不再需要任何instantiate调用！】
    # 因为YAML继承了Schema，Hydra在加载时就已经自动完成了类型检查和默认值填充！

    print("--- Fully Composed and Validated Config ---")
    print(OmegaConf.to_yaml(cfg))
    print(f"Dataset name: {cfg.data_structure.primary_dataset}")
    print(
        f"Internal schema for molecule: {cfg.data_structure.schema.internal.molecule_id}"
    )


if __name__ == "__main__":
    main()
