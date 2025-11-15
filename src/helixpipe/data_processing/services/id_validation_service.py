import logging
from typing import TYPE_CHECKING, Union

import pandas as pd

from helixpipe.typing import AppConfig
from helixpipe.utils import get_path

if TYPE_CHECKING:
    from helixpipe.typing import CID, PID, AppConfig

_local_human_uniprot_whitelist: Union[set["PID"], None] = None


logger = logging.getLogger(__name__)


def _load_local_human_whitelist(config: "AppConfig") -> set["PID"]:
    """
    【核心新函数】从本地下载的蛋白质组TSV文件中加载并筛选ID。
    结果会被缓存在内存中。
    """
    global _local_human_uniprot_whitelist
    if _local_human_uniprot_whitelist is not None:
        return _local_human_uniprot_whitelist

    logger.info(
        "--> [ID Validator] Loading and filtering local human proteome TSV for the first time..."
    )

    # 1. 从配置获取路径
    proteome_tsv_path = get_path(config, "assets.uniprot_proteome_tsv")

    if not proteome_tsv_path.exists():
        raise FileNotFoundError(
            f"Human proteome TSV file not found at '{proteome_tsv_path}'.\n"
            "Please download it from UniProt (proteome:UP000005640, format=tsv, columns=Entry,Reviewed,Organism (ID)) "
            "and save it to the 'data/assets/' directory."
        )

    # 2. 加载并筛选
    try:
        df = pd.read_csv(
            proteome_tsv_path, sep="\t", usecols=["Entry", "Reviewed", "Organism (ID)"]
        )

        # a. 筛选物种ID为9606 (人类)
        df_human = df[df["Organism (ID)"] == 9606]

        # b. 筛选状态为 'reviewed' (即Swiss-Prot)
        df_reviewed_human = df_human[df_human["Reviewed"] == "reviewed"]

        # c. 提取Entry列并转换为集合
        _local_human_uniprot_whitelist = set(df_reviewed_human["Entry"].unique())

        logger.info(
            f"--> Successfully loaded and filtered. Found {len(_local_human_uniprot_whitelist)} reviewed human UniProt IDs."
        )

    except Exception as e:
        logger.error(f"❌ ERROR: Failed to read or process the proteome TSV file: {e}")
        # 如果文件处理失败，返回一个空集合，避免整个程序崩溃
        _local_human_uniprot_whitelist = set()

    return _local_human_uniprot_whitelist


# --- 公共接口函数 ---


def get_human_uniprot_whitelist(
    ids_to_check: set["PID"], config: "AppConfig"
) -> set["PID"]:
    """
    【V5 纯离线版】通过与本地权威的人类蛋白质组ID列表进行比对来验证ID。
    """
    # 1. 加载权威的本地白名单
    #    这个函数内部有内存缓存，所以多次调用开销很小
    local_whitelist = _load_local_human_whitelist(config)

    # 3. 返回输入ID与本地白名单的交集
    valid_ids = local_whitelist

    if config.runtime.verbose > 0:
        logger.info(
            f"    - Validated {len(ids_to_check)} input IDs against local list."
        )
        logger.info(f"    - Found {len(valid_ids)} valid reviewed human protein IDs.")

    return valid_ids


def get_valid_pubchem_cids(cids_to_check: set["CID"], config: AppConfig) -> set["CID"]:
    """
    【公共接口】获取一个经过本地验证的PubChem CID白名单。
    目前只进行格式和类型检查。
    """
    logger.info(
        f"    - Performing local validation for {len(cids_to_check)} PubChem CIDs..."
    )

    valid_cids = set()
    for cid in cids_to_check:
        try:
            # 尝试将CID转换为整数，如果失败则跳过
            int_cid = int(cid)
            # 确保CID是正数
            if int_cid > 0:
                valid_cids.add(int_cid)
        except (ValueError, TypeError):
            continue

    logger.info(
        f"    - Local validation complete. Found {len(valid_cids)} valid PubChem CIDs."
    )
    return valid_cids
