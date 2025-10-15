# 文件: src/configs/data_structure.py

from dataclasses import dataclass, field
from typing import Dict, Any

# --------------------------------------------------------------------------
# 这些是嵌套在主 DataStructureConfig 内部的、更小的配置块
# --------------------------------------------------------------------------


@dataclass
class RawFilenames:
    """定义 'raw' 目录下所有文件的文件名模板。"""

    authoritative_dti: str = "default_interactions.csv"
    raw_tsv: str = "default_raw.tsv"
    dummy_file_to_get_dir: str = "dummy.txt"
    interactions: str = "interactions_default.csv"
    ligands: str = "ligands_default.csv"


@dataclass
class SimilarityMatrixFilenames:
    """定义 'processed/common/sim_matrixes' 目录下的文件名模板。"""

    molecule: str = "sim_matrixes/dl_similarity_matrix.pkl"
    protein: str = "sim_matrixes/prot_similarity_matrix.pkl"


@dataclass
class CommonProcessedFilenames:
    """定义 'processed/common' 目录下所有文件的文件名模板。"""

    nodes_metadata: str = "nodes.csv"
    node_features: str = "node_features.npy"
    similarity_matrices: SimilarityMatrixFilenames = field(
        default_factory=SimilarityMatrixFilenames
    )


@dataclass
class SpecificProcessedFilenames:
    """定义 'processed/specific' 目录下所有文件的文件名模板。"""

    graph_template: str = "{prefix}_graph_{suffix}.csv"
    labels_template: str = "{prefix}_labels_{suffix}.csv"


@dataclass
class ProcessedFilenames:
    """组织所有 'processed' 子目录的文件名。"""

    common: CommonProcessedFilenames = field(default_factory=CommonProcessedFilenames)
    specific: SpecificProcessedFilenames = field(
        default_factory=SpecificProcessedFilenames
    )


@dataclass
class CacheFilenames:
    """定义全局特征缓存的文件名模板。"""

    feature_template: str = "{entity_type}/{model_name}/{authoritative_id}.pt"


@dataclass
class FilenamesConfig:
    """顶层的 Filenames 配置块。"""

    raw: RawFilenames = field(default_factory=RawFilenames)
    processed: ProcessedFilenames = field(default_factory=ProcessedFilenames)
    cache: CacheFilenames = field(default_factory=CacheFilenames)


# --- Paths相关


@dataclass
class RawPaths:
    """定义 'raw' 目录下所有文件的【路径插值模板】。"""

    authoritative_dti: str = "${path:raw.authoritative_dti}"
    raw_tsv: str = "${path:raw.raw_tsv}"
    dummy_file_to_get_dir: str = "${path:raw.dummy_file_to_get_dir}"
    interactions: str = "${path:raw.interactions}"
    ligands: str = "${path:raw.ligands}"


@dataclass
class SimilarityMatrixPaths:
    """定义 'sim_matrixes' 目录下所有文件的【路径插值模板】。"""

    molecule: str = "${path:processed.common.similarity_matrices.molecule}"
    protein: str = "${path:processed.common.similarity_matrices.protein}"


@dataclass
class CommonProcessedPaths:
    """定义 'processed/common' 目录下所有文件的【路径插值模板】。"""

    nodes_metadata: str = "${path:processed.common.nodes_metadata}"
    node_features: str = "${path:processed.common.node_features}"
    similarity_matrices: SimilarityMatrixPaths = field(
        default_factory=SimilarityMatrixPaths
    )


@dataclass
class SpecificProcessedPaths:
    """定义 'processed/specific' 目录下所有文件的【路径插值模板】。"""

    graph_template: str = "${path:processed.specific.graph_template}"
    labels_template: str = "${path:processed.specific.labels_template}"


@dataclass
class ProcessedPaths:
    """组织所有 'processed' 子目录的路径。"""

    common: CommonProcessedPaths = field(default_factory=CommonProcessedPaths)
    specific: SpecificProcessedPaths = field(default_factory=SpecificProcessedPaths)


@dataclass
class CachePaths:
    """定义全局特征缓存的【路径插值模板】。"""

    feature_template: str = "${path:cache.feature_template}"


@dataclass
class PathsConfig:
    """顶层的 Paths 配置块。"""

    raw: RawPaths = field(default_factory=RawPaths)
    processed: ProcessedPaths = field(default_factory=ProcessedPaths)
    cache: CachePaths = field(default_factory=CachePaths)


# --- Schema相关的Dataclasses ---


@dataclass
class InternalSchema:
    """定义项目内部黄金标准DataFrame的列名。"""

    molecule_id: str = "PubChem_CID"
    protein_id: str = "UniProt_ID"
    molecule_sequence: str = "SMILES"
    protein_sequence: str = "Sequence"
    label: str = "Label"


@dataclass
class SchemaConfig:
    """顶层的 Schema 配置块。"""

    internal: InternalSchema = field(default_factory=InternalSchema)
    # external 是一个开放的字典，因为每个数据源的schema都不同
    # 我们用 Dict[str, Any] 来表示它可以包含任何键值对
    external: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 这是最终的、将被Hydra实例化的主 Dataclass
# --------------------------------------------------------------------------


@dataclass
class DataStructureConfig:
    """
    【结构化配置】定义一个数据集的完整“蓝图”。
    它描述了文件结构、文件名模板和数据列名(schema)，但与具体的数据内容或处理参数无关。
    这个 dataclass 对应于 `conf/data_structure/base.yaml`。
    """

    # Hydra将使用这个target来实例化一个DataStructureConfig对象
    _target_: str = "src.configs.data_structure.DataStructureConfig"

    # --- 字段定义 ---
    # 这些字段直接对应于 base.yaml 中的键
    name: str = "baseline"
    primary_dataset: str = "default_dataset"

    # 嵌套的 dataclasses
    filenames: FilenamesConfig = field(default_factory=FilenamesConfig)

    paths: PathsConfig = field(default_factory=PathsConfig)

    schema: SchemaConfig = field(default_factory=SchemaConfig)
