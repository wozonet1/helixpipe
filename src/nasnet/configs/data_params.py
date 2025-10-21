# 文件: src/configs/data_params.py (完整修正版)

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# 嵌套的 Dataclasses (这部分保持不变)
# --------------------------------------------------------------------------


@dataclass
class FeatureExtractorConfig:
    """定义单个特征提取器的配置。"""

    extractor_function: str
    model_name: str
    batch_size: int


@dataclass
class FilteringConfig:
    """定义分子理化性质过滤的规则。"""

    enabled: bool = False
    molecular_weight: Optional[Dict[str, float]] = None
    logp: Optional[Dict[str, float]] = None


# --------------------------------------------------------------------------
# 主 Dataclass (核心修正与补全)
# --------------------------------------------------------------------------


@dataclass
class DataParamsConfig:
    """
    【结构化配置 - 完整版】定义数据处理的“风味”或“参数集”。
    它现在精确地反映了 baseline.yaml 和其他变体中的所有字段。
    """

    # --- 字段定义 ---
    name: str = "baseline"

    # 相似度阈值
    similarity_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "protein_protein": 0.8,
            "drug_drug": 0.988,
            "ligand_ligand": 0.988,
            "drug_ligand": 0.7,
        }
    )

    # 【已补全】最大蛋白质-蛋白质边数量
    max_pp_edges: int = 500

    # 负采样策略
    negative_sampling_strategy: str = "popular"

    # 【修正】通用亲和力阈值，默认值与 baseline.yaml 一致
    affinity_threshold_nM: int = 10000

    # 要加载的辅助数据集列表
    auxiliary_datasets: List[str] = field(default_factory=list)

    # 特征提取器配置 (字典形式)
    # 我们可以在这里为 feature_extractors 提供一个更完整的默认结构
    feature_extractors: Dict[str, FeatureExtractorConfig] = field(
        default_factory=lambda: {
            "protein": FeatureExtractorConfig(
                extractor_function="extract_esm_protein_embeddings",
                model_name="facebook/esm2_t30_150M_UR50D",
                batch_size=32,
            ),
            "molecule": FeatureExtractorConfig(
                extractor_function="extract_chemberta_molecule_embeddings",
                model_name="seyonec/ChemBERTa-zinc-base-v1",
                batch_size=64,
            ),
        }
    )

    # 分子属性过滤配置 (默认是关闭的)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
