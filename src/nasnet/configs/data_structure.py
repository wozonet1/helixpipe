# 文件: src/configs/data_structure.py

from dataclasses import dataclass, field
from typing import Any, Dict

# --------------------------------------------------------------------------
# 这些是嵌套在主 DataStructureConfig 内部的、更小的配置块
# --------------------------------------------------------------------------


@dataclass
class RawFilenames:
    """定义 'raw' 目录下所有文件的文件名模板。"""

    authoritative_dti: str = "default_interactions.csv"
    raw_tsv: str = "default_raw.tsv"
    raw_json: str = "default_raw.json"
    dummy_file_to_get_dir: str = "dummy.txt"
    interactions: str = "interactions_default.csv"
    ligands: str = "ligands_default.csv"


@dataclass
class SimilarityMatrixFilenames:
    """定义 'processed/common/sim_matrixes' 目录下的文件名模板。"""

    molecule: str = "dl_similarity_matrix.pkl"
    protein: str = "prot_similarity_matrix.pkl"


@dataclass
class CommonProcessedFilenames:
    """定义 'processed/common' 目录下所有文件的文件名模板。"""

    nodes_metadata: str = "nodes.csv"
    node_features: str = "node_features.npy"
    similarity_matrices: SimilarityMatrixFilenames = SimilarityMatrixFilenames()
    uniprot_whitelist: str = "uniprot_whitelist.txt"
    cid_whitelist: str = "cid_whitelist.txt"


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
class CacheFeaturesFilenames:
    """定义 cache/features 下的文件名模板。"""

    template: str = "{entity_type}/{model_name}/{authoritative_id}.pt"


@dataclass
class CacheIdsFilenames:
    """定义 cache/ids 下的文件名。"""

    uniprot_whitelist: str = "human_uniprot_whitelist.txt"
    cid_whitelist: str = "pubchem_cid_whitelist.txt"
    brenda_name_to_cid: str = "pubchem_name_to_cid.pkl"
    enriched_protein_sequences: str = "enriched_protein_sequences.pkl"
    enriched_molecule_smiles: str = "enriched_molecule_smiles.pkl"


@dataclass
class CacheFilenames:
    """组织所有 'cache' 子目录的文件名。"""

    features: CacheFeaturesFilenames = field(default_factory=CacheFeaturesFilenames)
    ids: CacheIdsFilenames = field(default_factory=CacheIdsFilenames)


@dataclass
class AssetsFilenames:
    """定义 'assets' 目录下所有文件的文件名模板。"""

    uniprot_proteome_tsv: str = "uniprotkb_proteome_UP000005640.tsv"


@dataclass
class FilenamesConfig:
    """顶层的 Filenames 配置块。"""

    raw: RawFilenames = RawFilenames()
    processed: ProcessedFilenames = ProcessedFilenames()
    cache: CacheFilenames = CacheFilenames()
    assets: AssetsFilenames = AssetsFilenames()


# --- Paths相关


@dataclass
class RawPaths:
    """定义 'raw' 目录下所有文件的【路径插值模板】。"""

    authoritative_dti: str = "${path:raw.authoritative_dti}"
    raw_tsv: str = "${path:raw.raw_tsv}"
    raw_json: str = "${path:raw.raw_json}"
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
    uniprot_whitelist: str = "${path:processed.common.uniprot_whitelist}"
    cid_whitelist: str = "${path:processed.common.cid_whitelist}"


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
class CacheFeaturesPaths:
    """定义 cache/features 下的路径插值模板。"""

    template: str = "${path:cache.features.template}"


@dataclass
class CacheIdsPaths:
    """定义 cache/ids 下的路径插值模板。"""

    uniprot_whitelist: str = "${path:cache.ids.uniprot_whitelist}"
    cid_whitelist: str = "${path:cache.ids.cid_whitelist}"
    brenda_name_to_cid: str = "${path:cache.ids.brenda_name_to_cid}"
    enriched_protein_sequences: str = "${path:cache.ids.enriched_protein_sequences}"
    enriched_molecule_smiles: str = "${path:cache.ids.enriched_molecule_smiles}"


@dataclass
class CachePaths:
    """组织所有 'cache' 子目录的路径插值模板。"""

    features: CacheFeaturesPaths = field(default_factory=CacheFeaturesPaths)
    ids: CacheIdsPaths = field(default_factory=CacheIdsPaths)


@dataclass
class AssetPaths:
    """定义 'assets' 目录下所有文件的路径插值模板。"""

    uniprot_proteome_tsv: str = "${path:assets.uniprot_proteome_tsv}"


@dataclass
class PathsConfig:
    raw: RawPaths = RawPaths()
    processed: ProcessedPaths = ProcessedPaths()
    cache: CachePaths = CachePaths()
    assets: AssetPaths = field(default_factory=AssetPaths)


# --- Schema相关的Dataclasses ---


@dataclass
class AuthoritativeDTISchema:
    """定义项目内部黄金标准DataFrame的列名。(重命名，更清晰)"""

    molecule_id: str = "PubChem_CID"
    protein_id: str = "UniProt_ID"
    molecule_sequence: str = "SMILES"
    protein_sequence: str = "Sequence"
    label: str = "Label"
    relation_type: str = "relation_type"


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
class NodesOutputSchema:
    """【新增】定义调试文件 nodes.csv 的列名。"""

    global_id: str = "global_id"
    node_type: str = "node_type"
    authoritative_id: str = "authoritative_id"
    structure: str = "sequence_or_smiles"


@dataclass
class InternalSchemaConfig:  # <--- 新增一个层级
    """组织所有项目内部使用的Schema定义。"""

    authoritative_dti: AuthoritativeDTISchema = AuthoritativeDTISchema()
    graph_output: GraphOutputSchema = GraphOutputSchema()
    labeled_edges_output: LabeledEdgesSchema = LabeledEdgesSchema()
    nodes_output: NodesOutputSchema = field(default_factory=NodesOutputSchema)


@dataclass
class SchemaConfig:
    """顶层的 Schema 配置块。"""

    internal: InternalSchemaConfig = InternalSchemaConfig()
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
