# 文件: src/configs/data_params.py (完整修正版)

from dataclasses import dataclass, field
from typing import Dict, Optional, Union


# --------------------------------------------------------------------------
# 嵌套的 Dataclasses (这部分保持不变)
# --------------------------------------------------------------------------
@dataclass
class DownstreamSamplingConfig:
    """
    定义下游交互对采样的策略。
    """

    # 总开关
    enabled: bool = False

    # 【方案1: 统一采样】对所有正样本进行统一比例的随机采样
    # fraction: 采样比例 (0.0 to 1.0) 或 采样数量 (int)
    fraction: Optional[Union[float, int]] = 1.0

    # 【方案2: 分层采样】根据 Drug vs Ligand 的比例进行采样
    # drug_to_ligand_ratio: 'drug'交互数量与'ligand'交互数量的比例
    # 例如: 1.0 表示 1:1, 10.0 表示 10:1
    drug_to_ligand_ratio: Optional[float] = None


@dataclass
class FeatureExtractorConfig:
    """定义单个特征提取器的配置。"""

    extractor_function: str
    model_name: str
    batch_size: int


@dataclass
class RangeFilterConfig:
    """定义一个范围过滤器 (min, max)。"""

    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class MinValueFilterConfig:
    """定义一个最小值过滤器 (value >= min)。"""

    min: float = 0.0


@dataclass
class MaxValueFilterConfig:
    """定义一个最大值过滤器 (value <= max)。"""

    max: float = 10.0


# --- 【核心修改】重构顶层的 FilteringConfig ---


@dataclass
class FilteringConfig:
    """
    【V2 重构版】定义分子理化性质过滤的所有规则。
    每个字段都是一个嵌套的dataclass，提供了清晰的结构和默认值。
    """

    # 总开关
    enabled: bool = False

    # --- 阶段一: 结构警报 ---
    apply_pains_filter: bool = False
    apply_bms_filter: bool = False  # 为未来预留

    # --- 阶段二: 类药性范围 ---
    # 使用 field(default=None) 表示这些规则是可选的
    molecular_weight: Optional[RangeFilterConfig] = None
    logp: Optional[RangeFilterConfig] = None
    h_bond_donors: Optional[MaxValueFilterConfig] = None
    h_bond_acceptors: Optional[MaxValueFilterConfig] = None

    # --- 阶段三: 高级评分 ---
    qed: Optional[MinValueFilterConfig] = None
    sa_score: Optional[MaxValueFilterConfig] = None


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
    similarity_top_k: int = 10

    # 【已补全】最大蛋白质-蛋白质边数量
    max_pp_edges: int = 500

    # 负采样策略
    negative_sampling_strategy: str = "popular"

    # 【修正】通用亲和力阈值，默认值与 baseline.yaml 一致
    affinity_threshold_nM: int = 10000
    km_threshold_nM: int = 10000  # 为 Brenda 数据集预留

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

    sampling: DownstreamSamplingConfig = field(default_factory=DownstreamSamplingConfig)
