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
    similarity_matrices: SimilarityMatrixFilenames = SimilarityMatrixFilenames()


@dataclass
class SpecificProcessedFilenames:
    """定义 'processed/specific' 目录下所有文件的文件名模板。"""

    graph_template: str = "{prefix}_graph_{suffix}.csv"
    labels_template: str = "{prefix}_labels_{suffix}.csv"


@dataclass
class ProcessedFilenames:
    """组织所有 'processed' 子目录的文件名。"""

    common: CommonProcessedFilenames = CommonProcessedFilenames()
    specific: SpecificProcessedFilenames = SpecificProcessedFilenames()


@dataclass
class CacheFilenames:
    """定义全局特征缓存的文件名模板。"""

    feature_template: str = "{entity_type}/{model_name}/{authoritative_id}.pt"


@dataclass
class FilenamesConfig:
    """顶层的 Filenames 配置块。"""

    raw: RawFilenames = RawFilenames()
    processed: ProcessedFilenames = ProcessedFilenames()
    cache: CacheFilenames = CacheFilenames()


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
    similarity_matrices: SimilarityMatrixPaths = SimilarityMatrixPaths()


@dataclass
class SpecificProcessedPaths:
    """定义 'processed/specific' 目录下所有文件的【路径插值模板】。"""

    graph_template: str = "${path:processed.specific.graph_template}"
    labels_template: str = "${path:processed.specific.labels_template}"


@dataclass
class ProcessedPaths:
    common: CommonProcessedPaths = CommonProcessedPaths()
    specific: SpecificProcessedPaths = SpecificProcessedPaths()


@dataclass
class CachePaths:
    """定义全局特征缓存的【路径插值模板】。"""

    feature_template: str = "${path:cache.feature_template}"


@dataclass
class PathsConfig:
    raw: RawPaths = RawPaths()
    processed: ProcessedPaths = ProcessedPaths()
    cache: CachePaths = CachePaths()


# --- Schema相关的Dataclasses ---


@dataclass
class AuthoritativeDTISchema:
    """定义项目内部黄金标准DataFrame的列名。(重命名，更清晰)"""

    molecule_id: str = "PubChem_CID"
    protein_id: str = "UniProt_ID"
    molecule_sequence: str = "SMILES"
    protein_sequence: str = "Sequence"
    label: str = "Label"


@dataclass
class GraphOutputSchema:
    """【新增】定义图结构CSV文件的列名。"""

    source_node: str = "source"
    target_node: str = "target"
    edge_type: str = "edge_type"


@dataclass
class LabeledEdgesSchema:
    """【新增】定义标签文件CSV的列名。"""

    source_node: str = "source"
    target_node: str = "target"
    label: str = "label"


@dataclass
class InternalSchemaConfig:  # <--- 新增一个层级
    """组织所有项目内部使用的Schema定义。"""

    authoritative_dti: AuthoritativeDTISchema = AuthoritativeDTISchema()
    graph_output: GraphOutputSchema = GraphOutputSchema()
    labeled_edges: LabeledEdgesSchema = LabeledEdgesSchema()


@dataclass
class SchemaConfig:
    """顶层的 Schema 配置块。"""

    internal: InternalSchemaConfig = InternalSchemaConfig()  # <--- 引用新的组织类
    external: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 这是最终的、将被Hydra实例化的主 Dataclass
# --------------------------------------------------------------------------


@dataclass
class DataStructureConfig:
    name: str = "baseline"
    primary_dataset: str = "default_dataset"

    # 【核心修正】
    filenames: FilenamesConfig = FilenamesConfig()
    paths: PathsConfig = PathsConfig()
    schema: SchemaConfig = SchemaConfig()
