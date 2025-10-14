from torch_geometric.data import HeteroData
import torch
import research_template as rt
import pandas as pd
from omegaconf import DictConfig
from rdkit import Chem
import re
from tqdm import tqdm
from typing import Optional


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
    config: DictConfig, df: Optional[pd.DataFrame] = None
):
    """
    一个通用的、严格的验证函数，用于审查由任何数据处理流水线生成的
    最终权威DTI交互文件(例如 full.csv)。

    Args:
        config (DictConfig): 实验的完整配置对象。
        df (Optional[pd.DataFrame]): 一个可选的、已加载的DataFrame。
                                      如果为None,函数将从config指定的路径加载。

    Raises:
        AssertionError: 如果检测到任何严重的数据质量问题。
    """
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
        try:
            file_path = rt.get_path(config, "raw.dti_interactions")
            print(f"正在加载文件: {file_path}")
            df = pd.read_csv(file_path)
            print(
                f"--> {bcolors.OKGREEN}文件加载成功。共 {len(df)} 行记录。{bcolors.ENDC}"
            )
        except FileNotFoundError:
            print(
                f"❌ {bcolors.FAIL}致命错误: 找不到指定的DTI文件: {file_path}{bcolors.ENDC}"
            )
            raise
    else:
        print("--> 使用已传入的DataFrame进行检验。")

    # --- 2. 模式和结构验证 ---
    print("\n" + "-" * 30 + " 1. 模式与结构验证 " + "-" * 29)
    required_columns = {"PubChem_CID", "UniProt_ID", "SMILES", "Sequence", "Label"}
    actual_columns = set(df.columns)
    assert required_columns.issubset(actual_columns), (
        f"❌ {bcolors.FAIL}验证失败: 文件缺少必需的列。需要: {required_columns}, 实际: {actual_columns}{bcolors.ENDC}"
    )
    print(f"✅ {bcolors.OKGREEN}列完整性: 所有必需列均存在。{bcolors.ENDC}")

    assert pd.api.types.is_integer_dtype(df["PubChem_CID"]), (
        f"❌ {bcolors.FAIL}验证失败: 'PubChem_CID' 列应为整数类型。{bcolors.ENDC}"
    )
    assert pd.api.types.is_integer_dtype(df["Label"]), (
        f"❌ {bcolors.FAIL}验证失败: 'Label' 列应为整数类型。{bcolors.ENDC}"
    )
    print(f"✅ {bcolors.OKGREEN}数据类型: 关键列的数据类型正确。{bcolors.ENDC}")

    # --- 3. 数据唯一性验证 ---
    print("\n" + "-" * 30 + " 2. 数据唯一性验证 " + "-" * 29)
    duplicates = df.duplicated(subset=["PubChem_CID", "UniProt_ID"]).sum()
    assert duplicates == 0, (
        f"❌ {bcolors.FAIL}验证失败: 在 ('PubChem_CID', 'UniProt_ID') 上发现 {duplicates} 条重复记录。{bcolors.ENDC}"
    )
    print(
        f"✅ {bcolors.OKGREEN}交互对唯一性: 所有 (药物, 靶点) 对都是唯一的。{bcolors.ENDC}"
    )

    # --- 4. 内容有效性验证 ---
    print("\n" + "-" * 30 + " 3. 内容有效性验证 " + "-" * 30)

    # a. SMILES 有效性
    sample_size = min(len(df), 5000)  # 最多检查5000个样本，避免大数据集上耗时过长
    invalid_smiles_count = 0
    sampled_smiles = df["SMILES"].sample(
        n=sample_size, random_state=config.runtime.seed
    )
    for smiles in tqdm(sampled_smiles, desc="检验SMILES有效性"):
        if Chem.MolFromSmiles(smiles) is None:
            invalid_smiles_count += 1

    invalidation_rate = (invalid_smiles_count / sample_size) * 100
    assert invalidation_rate < 0.1, (
        f"❌ {bcolors.FAIL}验证失败: SMILES有效性过低。在{sample_size}个样本中发现 {invalid_smiles_count} ({invalidation_rate:.2f}%) 个无效SMILES。{bcolors.ENDC}"
    )
    print(
        f"✅ {bcolors.OKGREEN}SMILES有效性: 在{sample_size}个样本中，无效比例为 {invalidation_rate:.2f}% (通过)。{bcolors.ENDC}"
    )

    # b. UniProt ID 格式
    uniprot_pattern = re.compile(
        r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})"
    )
    invalid_uniprot_ids = df[~df["UniProt_ID"].astype(str).str.match(uniprot_pattern)]
    assert len(invalid_uniprot_ids) == 0, (
        f"❌ {bcolors.FAIL}验证失败: 发现 {len(invalid_uniprot_ids)} 个不符合标准格式的UniProt ID。例如: {invalid_uniprot_ids['UniProt_ID'].head().tolist()}{bcolors.ENDC}"
    )
    print(f"✅ {bcolors.OKGREEN}UniProt ID格式: 所有ID均符合标准格式。{bcolors.ENDC}")

    # a. 定义合法的氨基酸字符集
    amino_acids = "ACDEFGHIKLMNPQRSTVWYU"

    # b. 构建查找非法字符的正则表达式
    invalid_char_pattern = f"[^{amino_acids}]"

    # c. 找出所有包含非法字符的序列的DataFrame
    invalid_seq_df = df[
        df["Sequence"]
        .str.upper()
        .str.contains(invalid_char_pattern, regex=True, na=False)
    ]

    # d. 检查是否存在无效序列
    if not invalid_seq_df.empty:
        num_invalid = len(invalid_seq_df)
        print(
            f"❌ {bcolors.FAIL}验证失败: 发现 {num_invalid} 条蛋白质序列包含非法字符。{bcolors.ENDC}"
        )
        print("--- 无效序列样本 (前5条): ---")
        # 使用 .to_string() 保证打印内容对齐
        print(invalid_seq_df[["Sequence"]].head().to_string())
        print("-" * 30)
        # 抛出更明确的错误
        raise ValueError(f"数据集中存在 {num_invalid} 条无效的蛋白质序列。")
    else:
        print(
            f"✅ {bcolors.OKGREEN}蛋白质序列内容: 所有序列均由合法的氨基酸字符组成。{bcolors.ENDC}"
        )

    # --- 5. 最终总结 ---
    print("\n" + "=" * 80)
    print(
        f"{bcolors.OKGREEN}{bcolors.BOLD}"
        + " " * 25
        + "✅ 所有验证项目均已通过 ✅"
        + f"{bcolors.ENDC}"
    )
    print("=" * 80)
