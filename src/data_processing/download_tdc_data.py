# 文件: src/data_processing/download_tdc_data.py

import sys
from pathlib import Path

# 将项目根目录添加到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from tdc.resource import PrimeKG
import research_template as rt
from omegaconf import OmegaConf


def download_and_extract_dti_from_tdc(config):
    """
    从TDC的PrimeKG知识图谱中，提取高质量的、经过验证的DTI数据，
    并将其保存为我们项目流水线可以消费的 "raw/dti_interactions.csv" 格式。
    """
    print("--- [TDC Downloader] Starting... ---")

    # 定义我们想要保存的原始交互文件的路径
    # 我们将覆盖旧的、来源可疑的 full.csv
    output_path = rt.get_path(config, "raw.dti_interactions")

    if output_path.exists():
        print(
            f"--> Found existing interaction file at '{output_path}'. Skipping download."
        )
        print("    To force re-download, please delete the file manually.")
        return

    print("--> Downloading and processing PrimeKG from Therapeutics Data Commons...")
    # TDC会自动将数据下载并缓存到 'data/' 目录下
    data = PrimeKG(path="data/")

    # PrimeKG 是一个巨大的知识图谱，我们只提取其中与DTI相关的部分
    # TDC的文档告诉我们，'drug-protein' 这个表格，主要就是来自DrugBank
    dti_df = data.get_data(table="drug-protein")

    print(f"--> Extracted {len(dti_df)} DTI entries from PrimeKG.")

    # --- [关键] 将TDC的DataFrame，转换为我们项目期望的格式 ---
    # TDC PrimeKG的列名可能是 'drug_id', 'relation', 'protein_id'
    # 我们的旧 full.csv 格式可能是 'SMILES', 'Protein', 'Y'
    # 我们需要进行转换和扩充

    # 假设TDC提供的是DrugBank ID和UniProt ID，我们需要获取对应的SMILES和序列
    # 这是一个复杂的过程，一个更简单的起点是，我们先创建一个只包含ID的文件

    # 为了与我们即将进行的“权威ID”重构完美衔接，我们输出一个包含ID和Label的文件
    # 列名: 'PubChem_CID', 'UniProt_ID', 'Label'

    # 这是一个简化的例子，假设TDC的 'drug_id' 是PubChem CID
    # 并且 'Y' (标签) 都是1

    # 真实的TDC PrimeKG可能需要更复杂的ID映射
    # 但核心思想是，我们将它处理成一个干净的、基于ID的交互列表

    # 假设 dti_df 的列是 ['drug_id', 'relation', 'protein_id']
    # drug_id 是 DrugBank ID, protein_id 是 UniProt ID
    # 我们需要一个 DrugBank ID -> PubChem CID 的映射
    # TDC 同样提供了这个！
    from tdc.metadata import DrugMap

    drug_map = DrugMap(map_from="DrugBank", map_to="PubChem CID")
    dti_df["PubChem_CID"] = dti_df["drug_id"].map(drug_map.get_map())

    # 筛选并重命名列
    output_df = dti_df[["PubChem_CID", "protein_id"]].copy()
    output_df.rename(columns={"protein_id": "UniProt_ID"}, inplace=True)
    output_df["Label"] = 1  # 假设所有已知的交互都是正样本

    # 删除任何映射失败的行
    output_df.dropna(inplace=True)

    print(
        f"--> Successfully mapped to PubChem CIDs. Final dataset size: {len(output_df)}"
    )

    # 保存为我们自己的、权威的原始交互文件
    rt.ensure_path_exists(output_path)
    output_df.to_csv(output_path, index=False)

    print(
        f"✅ Successfully created a new, authoritative interaction file at: '{output_path}'"
    )


if __name__ == "__main__":
    # 允许我们独立运行这个脚本
    # 加载一个简化的、仅包含必要路径的配置
    # 或者直接硬编码路径

    # 更优雅的方式是使用我们之前的Hydra compose API
    from hydra import initialize, compose

    with initialize(config_path="../../conf", job_name="tdc_download"):
        cfg = compose(config_name="config")

    download_and_extract_dti_from_tdc(cfg)
