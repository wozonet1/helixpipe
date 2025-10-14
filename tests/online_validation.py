import pandas as pd
from data_utils.debug_utils import run_online_validation
import research_template as rt
import hydra


# 1. 加载您最终生成的权威DTI文件
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config):
    final_dti_path = rt.get_path(config, "data_structure.paths.raw.authoritative_dti")
    main_df = pd.read_csv(final_dti_path)
    run_online_validation(
        main_df, n_samples=200, n_jobs=6, random_state=config.runtime.seed
    )


# 2. 运行在线验证
#    它会从你的DataFrame中随机抽取200个样本进行验证
#    并使用4个并行的worker。

if __name__ == "__main__":
    rt.register_hydra_resolvers()
    main()
