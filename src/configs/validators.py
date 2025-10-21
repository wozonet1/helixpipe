# 文件: src/configs/validators.py (全新)

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# 嵌套的 Dataclasses，用于组织参数
# --------------------------------------------------------------------------


@dataclass
class UniProtValidatorConfig:
    """定义 UniProt ID 验证器的参数。"""

    # UniProt ID Mapping API 的基础URL
    api_url: str = "https://rest.uniprot.org/idmapping"

    # 每次批量提交给API的ID数量
    batch_size: int = 500

    # 我们只关心人类的蛋白质。9606是人类的NCBI物种分类ID。
    taxon_id: int = 9606


@dataclass
class PubChemValidatorConfig:
    """定义 PubChem CID 验证器的参数。"""

    # PubChem PUG REST API 的基础URL
    api_url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    # 每次批量查询的ID数量
    batch_size: int = 100


# --------------------------------------------------------------------------
# 主 Dataclass
# --------------------------------------------------------------------------


@dataclass
class ValidatorsConfig:
    """
    【结构化配置】定义所有ID验证步骤相关的参数。
    """

    # 变体名称，用于识别
    name: str = "default"

    # 嵌套的配置块
    uniprot: UniProtValidatorConfig = field(default_factory=UniProtValidatorConfig)
    pubchem: PubChemValidatorConfig = field(default_factory=PubChemValidatorConfig)
