from torch_geometric.data import HeteroData
import torch
import pandas as pd
from project_types import AppConfig
from rdkit import Chem
import re
from tqdm import tqdm
from typing import Optional
import time
import requests
from collections import namedtuple, Counter
from joblib import Parallel, delayed

# run.py 或 utils/config_validator.py
from typing import List


def validate_config_with_data(
    cfg: AppConfig, base_df: pd.DataFrame, extra_dfs: List[pd.DataFrame]
) -> bool:
    """
    【V3 重构版】在数据加载后，对配置和数据进行深度的逻辑一致性检查。
    """
    print("\n--- [Config & Data Validation] Running logic consistency checks... ---")
    checks_passed = True

    # --- 检查 1: 辅助数据集与图关系的依赖关系 ---
    # 这个检查与之前基本相同，但现在我们可以基于 `extra_dfs` 的存在来判断

    # a. 如果我们加载了扩展数据（意味着启用了GtoPdb等）...
    if extra_dfs:
        relation_flags = cfg.relations.flags
        ligand_related_relations = [
            "ligand_protein_interaction",
            "ligand_ligand_similarity",
            "drug_ligand_similarity",
        ]
        is_any_ligand_relation_enabled = any(
            relation_flags.get(rel, False) for rel in ligand_related_relations
        )

        if not is_any_ligand_relation_enabled:
            print(
                "❌ VALIDATION FAILED: Auxiliary datasets were loaded, but no ligand-related relations are enabled in the `relations` config."
            )
            print(
                "   - This implies 'ligand' nodes would be created but never used in the graph."
            )
            checks_passed = False

    # b. 反向检查：如果启用了ligand关系，但没有加载扩展数据...
    relation_flags = cfg.relations.flags
    ligand_related_relations = [
        "ligand_protein_interaction",
        "ligand_ligand_similarity",
        "drug_ligand_similarity",
    ]
    is_any_ligand_relation_enabled = any(
        relation_flags.get(rel, False) for rel in ligand_related_relations
    )

    if is_any_ligand_relation_enabled and not extra_dfs:
        print(
            "⚠️ VALIDATION WARNING: Ligand-related relations are enabled, but no auxiliary datasets were loaded."
        )
        print(
            "   - This is valid only if the base dataset itself contains entities you consider 'ligands'."
        )

    # --- 检查 2: 基于DataFrame内容的检查 ---
    # 我们可以检查 DataFrame 是否包含我们期望的实体类型

    # a. 检查 `extra_dfs` 是否真的引入了新的分子
    if extra_dfs:
        schema = cfg.data_structure.schema.internal.authoritative_dti
        base_cids = set(base_df[schema.molecule_id].unique())
        extra_cids = set()
        for df in extra_dfs:
            extra_cids.update(df[schema.molecule_id].unique())

        pure_ligand_cids = extra_cids - base_cids
        if not pure_ligand_cids:
            print(
                "⚠️ VALIDATION WARNING: Auxiliary datasets were loaded, but they did not introduce any new molecules (pure ligands)."
            )
            print(
                "   - The distinction between 'drug' and 'ligand' nodes will be meaningless in this run."
            )

    # --- 总结 ---
    if checks_passed:
        print("✅ All configuration and data logic checks passed.")
    else:
        print("\n--- [Validation] Finished with errors. Halting execution. ---")

    return checks_passed


def run_optional_diagnostics(hetero_graph: HeteroData):
    """
    Runs a suite of OPTIONAL but recommended diagnostic checks.
    Call this during development to ensure data integrity.
    """
    print("\n--- [OPTIONAL DIAGNOSTIC] Running full diagnostic suite ---")

    # Check 1: Official PyG validation
    hetero_graph.validate(raise_on_error=True)
    print("✅ (1/3) Official PyG validation successful.")

    # Check 2 & 3: Deep health checks
    if not (
        diagnose_hetero_data(hetero_graph) and diagnose_node_features(hetero_graph)
    ):
        raise ValueError("HeteroData object failed deep health checks.")
    print("✅ (2/3 & 3/3) Deep health checks successful.")

    print("--- [OPTIONAL DIAGNOSTIC] All checks passed. ---\n")


# 放置在 train.py 的顶部
def diagnose_hetero_data(data: HeteroData):
    """一个详细的诊断函数，彻查HeteroData对象的健康状况。"""
    print("\n--- [DIAGNOSTIC 2] Performing deep health check on HeteroData object...")
    is_healthy = True

    # 检查1: 所有edge_index都必须是torch.long
    for edge_type in data.edge_types:
        if data[edge_type].edge_index.dtype != torch.long:
            print(
                f"❌ DTYPE_ERROR for edge_type '{edge_type}': edge_index is {data[edge_type].edge_index.dtype}, but MUST be torch.long!"
            )
            is_healthy = False

    if is_healthy:
        print("✅ All edge_index tensors have correct dtype (torch.long).")

    # 检查2: 检查所有边索引是否越界
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type

        # 检查源节点
        if data[edge_type].edge_index.numel() > 0:  # 仅在有边的情况下检查
            max_src_id = data[edge_type].edge_index[0].max().item()
            num_src_nodes = data[src_type].num_nodes
            if max_src_id >= num_src_nodes:
                print(
                    f"❌ BOUNDS_ERROR for edge_type '{edge_type}': Max source ID is {max_src_id}, but node_type '{src_type}' only has {num_src_nodes} nodes!"
                )
                is_healthy = False

            # 检查目标节点
            max_dst_id = data[edge_type].edge_index[1].max().item()
            num_dst_nodes = data[dst_type].num_nodes
            if max_dst_id >= num_dst_nodes:
                print(
                    f"❌ BOUNDS_ERROR for edge_type '{edge_type}': Max destination ID is {max_dst_id}, but node_type '{dst_type}' only has {num_dst_nodes} nodes!"
                )
                is_healthy = False

    if is_healthy:
        print(
            "✅ All edge indices are within the bounds of their respective node stores."
        )

    print("--- Deep health check finished. ---")
    return is_healthy


def diagnose_node_features(data: HeteroData) -> bool:
    """
    Performs a deep analysis of node features in a HeteroData object.

    Checks for the presence of dangerous NaN (Not a Number) or Infinity
    values, which are common culprits for Segmentation Faults in C++/CUDA extensions.

    Args:
        data (HeteroData): The graph data object to diagnose.

    Returns:
        bool: True if all features are clean, False otherwise.
    """
    print("\n--- [DIAGNOSTIC] Analyzing node features for invalid values (NaN/inf)...")
    is_clean = True

    for node_type in data.node_types:
        # Check if the node type has features assigned
        if "x" not in data[node_type]:
            print(
                f"⚠️  INFO: Node type '{node_type}' has no features ('x' attribute). Skipping."
            )
            continue

        features = data[node_type].x

        # Check for NaN values
        nan_mask = torch.isnan(features)
        if nan_mask.any():
            num_nan = nan_mask.sum().item()
            print(
                f"❌ FATAL: Found {num_nan} NaN value(s) in features for node_type '{node_type}'!"
            )
            is_clean = False

        # Check for Infinity values
        inf_mask = torch.isinf(features)
        if inf_mask.any():
            num_inf = inf_mask.sum().item()
            print(
                f"❌ FATAL: Found {num_inf} Infinity value(s) in features for node_type '{node_type}'!"
            )
            is_clean = False

    if is_clean:
        print("✅ All node features are clean (no NaN/inf found).")

    return is_clean


def sanitize_for_loader(data: HeteroData) -> HeteroData:
    """
    Performs a final, deep sanitization of a HeteroData object to ensure
    all its tensors have a contiguous memory layout before being passed
    to a C++-backed loader.

    Args:
        data (HeteroData): The graph object to sanitize.

    Returns:
        HeteroData: The sanitized graph object.
    """
    print(
        "\n--- [FINAL SANITIZATION] Forcing contiguous memory layout for all tensors..."
    )

    for store in data.stores:
        for key, value in store.items():
            if torch.is_tensor(value):
                # .contiguous() returns a new tensor with contiguous memory if the
                # original is not; otherwise, it returns the original tensor.
                # This is a very cheap operation if the tensor is already contiguous.
                store[key] = value.contiguous()

    print("✅ All tensors are now memory-contiguous.")
    return data


def validate_entity_list_and_index(
    entity_list: list, entity_to_index_map: dict, entity_type: str, start_index: int = 0
) -> bool:
    """
    【关键诊断】验证一个实体列表和其索引字典之间的顺序和内容是否严格一致。

    本函数执行两个核心检查：
    1.  内容一致性：列表中的所有实体，是否与字典的键完全相同。
    2.  顺序一致性：列表中第 i 个实体，其在字典中对应的ID，是否精确地等于 i + start_index。

    Args:
        entity_list (list): 实体的有序列表 (例如 final_proteins_list)。
        entity_to_index_map (dict): 从实体映射到其全局ID的字典 (例如 prot2index)。
        entity_type (str): 实体的名称，用于打印清晰的日志信息 (例如 "Protein")。
        start_index (int): 该类型实体的全局ID起始编号。对于drug/ligand是0，
                           对于protein，是drug+ligand的总数。

    Returns:
        bool: 如果验证通过，返回True，否则返回False并打印详细错误。
    """
    print(f"--> [DIAGNOSTIC] Validating consistency for '{entity_type}' entities...")

    # 1. 内容一致性检查
    list_set = set(entity_list)
    dict_keys_set = set(entity_to_index_map.keys())

    if list_set != dict_keys_set:
        print(f"❌ VALIDATION FAILED for '{entity_type}': Content Mismatch!")
        missing_in_list = dict_keys_set - list_set
        missing_in_dict = list_set - dict_keys_set
        if missing_in_list:
            print(
                f"    - {len(missing_in_list)} items are in the dictionary but NOT in the list."
            )
        if missing_in_dict:
            print(
                f"    - {len(missing_in_dict)} items are in the list but NOT in the dictionary."
            )
        return False

    # 2. 顺序一致性检查
    all_ids = sorted(entity_to_index_map.values())
    expected_ids = list(range(start_index, start_index + len(entity_list)))
    if all_ids != expected_ids:
        print(
            f"❌ VALIDATION FAILED for '{entity_type}': Index values are not contiguous!"
        )
        print(
            f"    - Expected ID range: {start_index} to {start_index + len(entity_list) - 1}"
        )
        print(f"    - Actual IDs found (first 10): {all_ids[:10]}")  # 可选的debug输出
        return False

    for i, entity in enumerate(entity_list):
        expected_id = i + start_index
        actual_id = entity_to_index_map[entity]

        if actual_id != expected_id:
            print(f"❌ VALIDATION FAILED for '{entity_type}': Order Mismatch!")
            print(
                f"    - At list index {i}, for entity '{str(entity)[:50]}...'"
            )  # 打印实体的前50个字符
            print(f"    - Expected global ID: {expected_id}")
            print(f"    - Actual ID found in dictionary: {actual_id}")
            return False

    print(
        f"✅ Validation PASSED for '{entity_type}': Content and order are perfectly consistent."
    )
    return True


def validate_embedding_consistency(
    embedding_tensor: torch.Tensor,
    entity_list: list,
    entity_to_index_map: dict,
    entity_type: str,
) -> bool:
    """
    【关键诊断】验证一个预计算的嵌入张量，与其对应的实体列表和索引字典
    在维度、内容和顺序上是否完全一致。

    本函数执行三个核心检查：
    1.  维度一致性：嵌入张量的行数，是否与实体列表的长度完全相等。
    2.  内容一致性 (间接)：实体列表中的内容，是否与索引字典的键完全一致。
    3.  顺序一致性 (间接)：通过抽样检查，确保列表中的实体顺序，与其在
        索引字典中ID所对应的嵌入行号，是严格匹配的。

    Args:
        embedding_tensor (torch.Tensor): 预计算的嵌入张量 (例如 protein_embeddings)。
        entity_list (list): 实体的有序列表 (例如 final_proteins_list)。
        entity_to_index_map (dict): 从实体映射到其【全局】ID的字典 (例如 prot2index)。
        entity_type (str): 实体的名称，用于打印清晰的日志信息 (例如 "Protein")。

    Returns:
        bool: 如果验证通过，返回True，否则返回False并打印详细错误。
    """
    print(
        f"--> [DIAGNOSTIC] Validating consistency for '{entity_type}' PRE-COMPUTED EMBEDDINGS..."
    )

    # 1. 维度一致性检查
    num_embeddings = embedding_tensor.shape[0]
    num_entities_in_list = len(entity_list)

    if num_embeddings != num_entities_in_list:
        print(f"❌ VALIDATION FAILED for '{entity_type}': Dimension Mismatch!")
        print(f"    - Number of rows in embedding tensor: {num_embeddings}")
        print(f"    - Number of items in entity list: {num_entities_in_list}")
        return False

    # 我们借用之前的函数来完成内容和顺序的内部检查
    # 注意：这里的start_index必须是0，因为我们是在比较“局部”的列表和嵌入
    is_list_and_index_ok = validate_entity_list_and_index(
        entity_list=entity_list,
        entity_to_index_map={
            k: v - min(entity_to_index_map.values())
            for k, v in entity_to_index_map.items()
        },
        entity_type=f"{entity_type} (internal list vs. index)",
        start_index=0,
    )

    if not is_list_and_index_ok:
        print(
            f"❌ VALIDATION FAILED for '{entity_type}': Internal list/index inconsistency detected."
        )
        return False

    print(
        f"✅ Validation PASSED for '{entity_type}': Dimensions and internal consistency are correct."
    )
    return True


def validate_data_pipeline_integrity(
    *,  # 使用星号强制所有后续参数都必须是关键字参数，增加可读性
    final_smiles_list: list = None,
    final_proteins_list: list = None,
    dl2index: dict = None,
    prot2index: dict = None,
    molecule_embeddings: torch.Tensor = None,
    protein_embeddings: torch.Tensor = None,
):
    """
    【顶层诊断包装器】一个高级函数，用于在数据处理流水线的不同阶段，
    验证所有相关数据结构之间的完整性和一致性。

    它会根据传入的参数，自动执行所有相关的低级诊断函数。
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "RUNNING DATA PIPELINE INTEGRITY VALIDATION")
    print("=" * 80)

    # --- 验证阶段 1: 实体列表 vs 索引字典 ---
    # 仅当相关参数被提供时，才执行此检查
    if final_smiles_list is not None and dl2index is not None:
        if not validate_entity_list_and_index(
            entity_list=final_smiles_list,
            entity_to_index_map=dl2index,
            entity_type="Molecule (Drug/Ligand)",
            start_index=0,
        ):
            raise ValueError(
                "Stage 1 (Molecules) output failed consistency validation."
            )

    if (
        final_proteins_list is not None
        and prot2index is not None
        and dl2index is not None
    ):
        protein_start_index = len(dl2index)
        if not validate_entity_list_and_index(
            entity_list=final_proteins_list,
            entity_to_index_map=prot2index,
            entity_type="Protein",
            start_index=protein_start_index,
        ):
            raise ValueError("Stage 1 (Proteins) output failed consistency validation.")

    # --- 验证阶段 2: 嵌入 vs 列表/索引 ---
    # 仅当相关参数被提供时，才执行此检查
    if (
        molecule_embeddings is not None
        and final_smiles_list is not None
        and dl2index is not None
    ):
        if not validate_embedding_consistency(
            embedding_tensor=molecule_embeddings,
            entity_list=final_smiles_list,
            entity_to_index_map=dl2index,
            entity_type="Molecule (Drug/Ligand) Embeddings",
        ):
            raise ValueError(
                "Stage 2 (Molecule Embeddings) failed consistency validation."
            )

    if (
        protein_embeddings is not None
        and final_proteins_list is not None
        and prot2index is not None
    ):
        if not validate_embedding_consistency(
            embedding_tensor=protein_embeddings,
            entity_list=final_proteins_list,
            entity_to_index_map=prot2index,
            entity_type="Protein Embeddings",
        ):
            raise ValueError(
                "Stage 2 (Protein Embeddings) failed consistency validation."
            )

    print("=" * 80)
    print(" " * 22 + "✅ PIPELINE INTEGRITY VALIDATION PASSED ✅")
    print("=" * 80 + "\n")


# 可以在文件顶部定义一些颜色，让输出更醒目
class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def validate_authoritative_dti_file(
    config: AppConfig, df: Optional[pd.DataFrame] = None, verbose: int = 1
):
    """
    一个通用的、严格的、带分级日志的验证函数。
    """
    # 从配置中获取verbose级别，如果不存在则默认为1
    verbose = config.runtime.get("verbose", 1)

    # 【核心逻辑】verbose=0时，直接跳过所有验证
    if verbose == 0:
        return

    # --- 打印标题 ---
    if verbose > 0:
        print("\n" + "=" * 80)
        print(
            f"{bcolors.HEADER}{bcolors.BOLD}"
            + " " * 18
            + "开始执行权威DTI文件质量检验流程"
            + f"{bcolors.ENDC}"
        )
        print("=" * 80)

    # --- 1. 加载数据 ---
    if df is None:
        # ... (加载数据的逻辑不变)
        pass

    if verbose > 1:
        print(f"--> 使用已传入的DataFrame进行检验 (共 {len(df)} 行)。")

    # --- 2. 模式和结构验证 ---
    if verbose > 1:
        print("\n" + "-" * 30 + " 1. 模式与结构验证 " + "-" * 29)

    required_columns = {"PubChem_CID", "UniProt_ID", "SMILES", "Sequence", "Label"}
    actual_columns = set(df.columns)
    assert required_columns.issubset(actual_columns), (
        f"验证失败: 文件缺少必需的列。需要: {required_columns}, 实际: {actual_columns}"
    )
    if verbose > 1:
        print("  ✅ 列完整性: 所有必需列均存在。")

    assert pd.api.types.is_integer_dtype(df["PubChem_CID"]), (
        "验证失败: 'PubChem_CID' 列应为整数类型。"
    )
    assert pd.api.types.is_integer_dtype(df["Label"]), (
        "验证失败: 'Label' 列应为整数类型。"
    )
    if verbose > 1:
        print("  ✅ 数据类型: 关键列的数据类型正确。")

    print(f"✅ {bcolors.OKGREEN}模式与结构: 通过。{bcolors.ENDC}")

    # --- 3. 数据唯一性验证 ---
    if verbose > 1:
        print("\n" + "-" * 30 + " 2. 数据唯一性验证 " + "-" * 29)

    duplicates = df.duplicated(subset=["PubChem_CID", "UniProt_ID"]).sum()
    assert duplicates == 0, (
        f"验证失败: 在 ('PubChem_CID', 'UniProt_ID') 上发现 {duplicates} 条重复记录。"
    )
    if verbose > 1:
        print("  ✅ 交互对唯一性: 所有 (药物, 靶点) 对都是唯一的。")

    print(f"✅ {bcolors.OKGREEN}数据唯一性: 通过。{bcolors.ENDC}")

    # --- 4. 内容有效性验证 ---
    if verbose > 1:
        print("\n" + "-" * 30 + " 3. 内容有效性验证 " + "-" * 30)

    # a. SMILES 有效性
    sample_size = min(len(df), 5000)
    # 只有在verbose>1时才显示tqdm进度条
    disable_tqdm = verbose <= 1
    sampled_smiles = df["SMILES"].sample(
        n=sample_size, random_state=config.runtime.seed
    )

    invalid_smiles_count = 0
    # 使用一个生成器表达式，更高效
    invalid_smiles_count = sum(
        1
        for smiles in tqdm(
            sampled_smiles, desc="  检验SMILES有效性", disable=disable_tqdm
        )
        if Chem.MolFromSmiles(smiles) is None
    )

    invalidation_rate = (
        (invalid_smiles_count / sample_size) * 100 if sample_size > 0 else 0
    )
    assert invalidation_rate < 0.1, (
        f"验证失败: SMILES有效性过低。在{sample_size}个样本中发现 {invalid_smiles_count} ({invalidation_rate:.2f}%) 个无效SMILES。"
    )
    if verbose > 1:
        print(
            f"  ✅ SMILES有效性: 在{sample_size}个样本中，无效比例为 {invalidation_rate:.2f}% (通过)。"
        )

    # b. UniProt ID 格式
    uniprot_pattern = re.compile(r"^[A-Z0-9]+$")
    invalid_uniprot_ids = df[~df["UniProt_ID"].astype(str).str.match(uniprot_pattern)]
    assert len(invalid_uniprot_ids) == 0, (
        f"验证失败: 发现 {len(invalid_uniprot_ids)} 个不符合标准格式的UniProt ID。例如: {invalid_uniprot_ids['UniProt_ID'].head().tolist()}"
    )
    if verbose > 1:
        print("  ✅ UniProt ID格式: 所有ID均符合标准格式。")

    # c. 蛋白质序列内容
    amino_acids = "ACDEFGHIKLMNPQRSTVWYU"  # 使用我们最终确定的严格标准
    invalid_char_pattern = f"[^{amino_acids}]"
    invalid_seq_df = df[
        df["Sequence"]
        .str.upper()
        .str.contains(invalid_char_pattern, regex=True, na=False)
    ]

    if not invalid_seq_df.empty:
        num_invalid = len(invalid_seq_df)
        # 失败时的详细信息总是要打印的
        print(
            f"❌ {bcolors.FAIL}验证失败: 发现 {num_invalid} 条蛋白质序列包含非法字符。{bcolors.ENDC}"
        )
        print("--- 无效序列样本 (前5条): ---")
        print(invalid_seq_df[["Sequence"]].head().to_string())
        print("-" * 30)
        raise ValueError(f"数据集中存在 {num_invalid} 条无效的蛋白质序列。")
    if verbose > 1:
        print("  ✅ 蛋白质序列内容: 所有序列均由合法的氨基酸字符组成。")

    print(f"✅ {bcolors.OKGREEN}内容有效性: 通过。{bcolors.ENDC}")

    # --- 5. 最终总结 ---

    print(
        f"{bcolors.OKGREEN}{bcolors.BOLD}"
        + " " * 25
        + "✅ 所有验证项目均已通过 ✅"
        + f"{bcolors.ENDC}"
    )


# 定义一个结构化的返回类型，让结果更清晰
ValidationResult = namedtuple("ValidationResult", ["status", "message"])

# --- 核心验证函数 ---


def validate_pubchem_entry(cid: int, local_smiles: str) -> ValidationResult:
    """
    通过PubChem PUG REST API查询给定的CID，并将其规范SMILES与本地SMILES进行比较。

    Args:
        cid (int): PubChem Compound ID.
        local_smiles (str): 数据集中与该CID关联的SMILES字符串。

    Returns:
        ValidationResult: 包含 'MATCH', 'MISMATCH', 或 'API_ERROR' 状态的结果。
    """
    # PubChem 要求每秒请求不超过5次。在每个请求前暂停一下来遵守规则。
    time.sleep(0.25)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/TXT"
    # url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/TXT"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # 检查HTTP错误 (如 404, 500)

        api_smiles = response.text.strip()

        local_mol = Chem.MolFromSmiles(local_smiles)
        api_mol = Chem.MolFromSmiles(api_smiles)

        if not local_mol:
            return ValidationResult("LOCAL_INVALID", "Local SMILES is invalid.")
        if not api_mol:
            return ValidationResult("API_INVALID", "API SMILES is invalid.")

        # --- 【核心修复】 ---
        # 直接将分子对象 local_mol 和 api_mol 传给指纹函数
        local_fp = Chem.RDKFingerprint(local_mol)
        api_fp = Chem.RDKFingerprint(api_mol)

        if local_fp == api_fp:
            return ValidationResult("MATCH", "Molecules are chemically equivalent.")
        else:
            # 只有在不匹配时，我们才为了生成报告而创建规范SMILES字符串
            local_canonical_str = Chem.MolToSmiles(local_mol, canonical=True)
            api_canonical_str = Chem.MolToSmiles(api_mol, canonical=True)
            msg = f"Molecules are different. Local: '{local_canonical_str}' vs API: '{api_canonical_str}'"
            return ValidationResult("MISMATCH", msg)

    except requests.exceptions.RequestException as e:
        return ValidationResult("API_ERROR", str(e))


def validate_uniprot_entry(pid: str, local_sequence: str) -> ValidationResult:
    """
    通过UniProt API查询给定的蛋白质ID，并将其序列与本地序列进行比较。

    Args:
        pid (str): UniProt Primary Accession ID.
        local_sequence (str): 数据集中与该PID关联的蛋白质序列。

    Returns:
        ValidationResult: 包含 'MATCH', 'MISMATCH', 或 'API_ERROR' 状态的结果。
    """
    url = f"https://rest.uniprot.org/uniprotkb/{pid}.fasta"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        # 解析FASTA格式
        lines = response.text.strip().split("\n")
        api_sequence = "".join(lines[1:])
        # 【核心修复】: 在比较前对两个字符串都使用 .strip()
        local_seq_clean = local_sequence.strip().upper()
        api_seq_clean = api_sequence.strip().upper()

        if local_seq_clean == api_seq_clean:
            return ValidationResult("MATCH", "Sequences successfully matched.")
        else:
            # 现在，任何不匹配都是真实的内容差异
            import difflib

            if len(local_seq_clean) == len(api_seq_clean):
                # 使用difflib找出具体字符差异
                diff = list(difflib.ndiff(local_seq_clean, api_seq_clean))
                # 过滤出有差异的部分，并使其更可读
                diff_parts = [
                    d for d in diff if d.startswith("+ ") or d.startswith("- ")
                ]
                # 将例如 ['- A', '+ B', '- C', '+ D'] 变成 "A->B, C->D"
                readable_diff = []
                # 假设差异总是成对出现
                for i in range(0, len(diff_parts), 2):
                    if i + 1 < len(diff_parts):
                        readable_diff.append(
                            f"{diff_parts[i][-1]}->{diff_parts[i + 1][-1]}"
                        )
                msg = f"Content mismatch: {', '.join(readable_diff)[:100]}"
            elif local_seq_clean in api_seq_clean or api_seq_clean in local_seq_clean:
                msg = f"Local sequence is a SUBSTRING of API sequence (or vice versa). Local={len(local_seq_clean)}, API={len(api_seq_clean)}"
                return ValidationResult("MATCH_SUBSTRING", msg)
            else:
                msg = f"Length mismatch: Local={len(local_seq_clean)}, API={len(api_seq_clean)}"

            return ValidationResult("MISMATCH", msg)

    except requests.exceptions.RequestException as e:
        return ValidationResult("API_ERROR", str(e))


# --- 主协调与报告函数 ---


def _print_validation_report(
    title: str, results: list[ValidationResult], sample_df: pd.DataFrame, id_col: str
):
    """一个辅助函数，用于打印格式化的报告。"""
    print("\n" + "=" * 30)
    print(f" {title} Validation Report")
    print("=" * 30)

    counts = Counter(r.status for r in results)
    total = len(results)

    print(f"  - Total Samples: {total}")
    for status, count in counts.items():
        percentage = (count / total) * 100
        print(f"  - {status:<10}: {count:>4} ({percentage:.1f}%)")

    mismatches = [
        (res.message, row[id_col])
        for res, (_, row) in zip(results, sample_df.iterrows())
        if res.status == "MISMATCH"
    ]

    if mismatches:
        print("\n--- Mismatch Details (up to 5) ---")
        for i, (msg, item_id) in enumerate(mismatches[:5]):
            print(f"{i + 1}. ID: {item_id}")
            print(f"   Reason: {msg}")
    print("=" * 30)


def run_online_validation(
    df: pd.DataFrame, n_samples: int = 200, n_jobs: int = 4, random_state: int = 42
):
    """
    从给定的DataFrame中抽样，并行执行对PubChem和UniProt的在线验证，并打印总结报告。

    Args:
        df (pd.DataFrame): 包含 'PubChem_CID', 'SMILES', 'UniProt_ID', 'Sequence' 列的DataFrame。
        n_samples (int): 要随机抽取的样本数量。
        n_jobs (int): 用于并行API请求的作业数。
                       注意：PubChem有速率限制，过高的n_jobs可能无益。
        random_state (int): 用于抽样的随机种子，以保证结果可复现。
    """
    print(f"\n🚀 Starting online validation for {n_samples} random samples...")

    if n_samples > len(df):
        print(
            f"Warning: n_samples ({n_samples}) is larger than DataFrame size ({len(df)}). Validating all entries."
        )
        n_samples = len(df)

    sample_df = df.sample(n=n_samples, random_state=random_state)

    with Parallel(n_jobs=n_jobs) as parallel:
        # --- PubChem Validation ---
        print("\n[Phase 1/2] Querying PubChem API for SMILES validation...")
        pubchem_results = parallel(
            delayed(validate_pubchem_entry)(row.PubChem_CID, row.SMILES)
            for _, row in tqdm(
                sample_df.iterrows(), total=len(sample_df), desc="PubChem Checks"
            )
        )

        # --- UniProt Validation ---
        print("\n[Phase 2/2] Querying UniProt API for Sequence validation...")
        uniprot_results = parallel(
            delayed(validate_uniprot_entry)(row.UniProt_ID, row.Sequence)
            for _, row in tqdm(
                sample_df.iterrows(), total=len(sample_df), desc="UniProt Checks"
            )
        )

    # --- Generate Reports ---
    _print_validation_report(
        "PubChem CID vs SMILES", pubchem_results, sample_df, "PubChem_CID"
    )
    _print_validation_report(
        "UniProt ID vs Sequence", uniprot_results, sample_df, "UniProt_ID"
    )
