# src/helixpipe/configs/data_params.py

from dataclasses import dataclass, field

# 'InteractionSelectorConfig' 依赖于 'training' 模块，我们需要正确导入它
# 为了避免循环依赖，我们使用 if TYPE_CHECKING
from typing import Any, Dict, List, Optional

from .selectors import InteractionSelectorConfig

# ==============================================================================
# 1. 基础配置构建块 (保持不变)
# ==============================================================================
# 这些是构成 DataParamsConfig 的、独立的、可复用的配置单元。


# ---采样相关 (Sampling)---
@dataclass
class StratumConfig:
    """定义一个采样层的配置规则。"""

    name: str
    selector: "InteractionSelectorConfig"  # 使用字符串前向引用
    fraction: Optional[float] = 1.0
    ratio_to: Optional[str] = None
    ratio: Optional[float] = 1.0


@dataclass
class StratifiedSamplingConfig:
    """定义分层采样的所有规则。"""

    enabled: bool = False
    strata: List[StratumConfig] = field(default_factory=list)


@dataclass
class UniformSamplingConfig:
    enabled: bool = False
    fraction: float = 1.0


@dataclass
class DownstreamSamplingConfig:
    enabled: bool = False
    stratified_sampling: StratifiedSamplingConfig = field(
        default_factory=StratifiedSamplingConfig
    )
    uniform_sampling: UniformSamplingConfig = field(
        default_factory=UniformSamplingConfig
    )


# ---特征提取 (Feature Extraction)---
@dataclass
class FeatureExtractorConfig:
    extractor_function: str
    model_name: str
    batch_size: int


# ---分子过滤 (Molecule Filtering)---
@dataclass
class RangeFilterConfig:
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class MinValueFilterConfig:
    min: float = 0.0


@dataclass
class MaxValueFilterConfig:
    max: float = 10.0


@dataclass
class FilteringConfig:
    enabled: bool = False
    apply_pains_filter: bool = False
    apply_bms_filter: bool = False
    molecular_weight: Optional[RangeFilterConfig] = None
    logp: Optional[RangeFilterConfig] = None
    h_bond_donors: Optional[MaxValueFilterConfig] = None
    h_bond_acceptors: Optional[MaxValueFilterConfig] = None
    qed: Optional[MinValueFilterConfig] = None
    sa_score: Optional[MaxValueFilterConfig] = None


# ---相似度阈值 (Similarity Thresholds)---
@dataclass
class SimilarityThresholdsConfig:
    drug_drug: float = 0.9
    drug_ligand: float = 0.9
    ligand_ligand: float = 0.9
    protein_protein: float = 0.99


# ==============================================================================
# 2. Processor 专属参数块 (保持不变)
# ==============================================================================
# 每个 dataclass 封装了一个 Processor 所需的、特有的参数。


@dataclass
class BindingdbParams:
    affinity_threshold_nM: int = 10000
    filtering: Optional[FilteringConfig] = None


@dataclass
class BrendaParams:
    affinity_threshold_nM: int = 10000
    km_threshold_nM: int = 10000
    filtering: Optional[FilteringConfig] = None


@dataclass
class GtopdbParams:
    affinity_threshold_nM: int = 10000
    filtering: Optional[FilteringConfig] = None


# 可以在这里为 stringdb 等其他 processor 预留一个空的 dataclass
@dataclass
class StringParams:
    pass


# ==============================================================================
# 3. 最终的、可组合的顶层 DataParamsConfig (核心重构)
# ==============================================================================


@dataclass
class DataParamsConfig:
    """
    【V3 - 可组合的配置组模式】
    这是一个容器，通过其 `defaults` 列表，动态地、模块化地加载每个部分的配置。
    """

    # --- 1. [核心修改] Defaults 列表 ---
    # 这个列表定义了如何从配置组 (Config Groups) 中构建完整的 DataParamsConfig。
    # 每个条目 `{"<group_name>": "<config_name>"}` 会加载对应的 YAML 文件。
    defaults: List[Any] = field(
        default_factory=lambda: [
            # 为每个 Processor 参数块加载一个默认配置
            {"bindingdb": "default"},
            {"gtopdb": "default"},
            {"brenda": "default"},
            {"stringdb": "default"},
            {"sampling": "default"},
            {"filtering": "disabled"},
            # `_self_` 是 Hydra 的一个关键字，允许我们在 YAML 文件中直接覆盖其他值
            "_self_",
        ]
    )

    # --- 2. 通用参数 (保持不变) ---
    # 这些参数通常在主 data_params 文件中定义，或通过命令行覆盖。
    name: str = "default"
    similarity_thresholds: SimilarityThresholdsConfig = field(
        default_factory=SimilarityThresholdsConfig
    )
    similarity_top_k: int = 10
    max_pp_edges: int = 500
    negative_sampling_strategy: str = "popular"
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
    sampling: DownstreamSamplingConfig = field(default_factory=DownstreamSamplingConfig)
    filtering: FilteringConfig = field(
        default_factory=FilteringConfig
    )  # 注意：这个是全局的 filtering 配置
    # --- 3. [核心修改] Processor 专属参数块 ---
    # 字段名必须与 defaults 列表中的组名完全匹配。
    # 类型注解为 Optional，因为它们的实例将由 Hydra 在运行时根据 defaults 列表动态创建和填充。
    bindingdb: Optional[BindingdbParams] = None
    brenda: Optional[BrendaParams] = None
    gtopdb: Optional[GtopdbParams] = None
    stringdb: Optional[StringParams] = None
