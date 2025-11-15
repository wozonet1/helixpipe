# src/helixpipe/utils/diagnostics.py

import logging
import time
from collections import Counter
from typing import Any, NamedTuple, cast, no_type_check

import pandas as pd
import requests
import torch
from joblib import Parallel, delayed
from rdkit import Chem
from torch_geometric.data import HeteroData
from tqdm import tqdm

from helixpipe.typing import CID, PID, SMILES, ProteinSequence

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. PyG HeteroData Health Checks
# ==============================================================================
# 这些函数用于检查最终喂给模型的图对象的底层健康状况。


@no_type_check
def run_graph_diagnostics(hetero_graph: HeteroData, deep_check: bool = True) -> None:
    """
    对一个 HeteroData 对象运行一个诊断套件。

    Args:
        hetero_graph (HeteroData): 要检查的图对象。
        deep_check (bool): 是否执行更耗时的深度检查（如 NaN/inf）。
    """
    header = "Running Graph Diagnostics"
    logger.info(f"\n{'=' * 20} {header} {'=' * 20}")

    try:
        # 检查 1: PyG 官方验证
        hetero_graph.validate(raise_on_error=True)
        logger.info("✅ (1/3) PyG official validation successful.")

        # 检查 2: 边索引边界检查
        if not _diagnose_edge_indices(hetero_graph):
            raise ValueError("Edge index bounds check failed.")
        logger.info("✅ (2/3) Edge index bounds check successful.")

        # 检查 3: 节点特征 NaN/inf 检查 (可选)
        if deep_check:
            if not _diagnose_node_features(hetero_graph):
                raise ValueError("Node feature health check failed.")
            logger.info("✅ (3/3) Node feature health check successful.")
        else:
            logger.info("ℹ️ (3/3) Skipped deep node feature check.")

    except Exception as e:
        logger.error(f"❌ Graph diagnostics failed: {e}")
        raise  # 重新抛出异常，让调用者知道检查失败

    logger.info(f"{'=' * (42 + len(header))}\n")


@no_type_check
def _diagnose_edge_indices(data: HeteroData) -> bool:
    """检查所有边索引是否在其对应节点类型的边界内。"""
    is_healthy = True
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = data[edge_type].edge_index

        if edge_index.numel() > 0:
            max_src_id = edge_index[0].max().item()
            num_src_nodes = data[src_type].num_nodes
            if max_src_id >= num_src_nodes:
                logger.error(
                    f"❌ BOUNDS_ERROR for edge_type '{edge_type}': "
                    f"Max source ID is {max_src_id}, but node_type '{src_type}' "
                    f"only has {num_src_nodes} nodes!"
                )
                is_healthy = False

            max_dst_id = edge_index[1].max().item()
            num_dst_nodes = data[dst_type].num_nodes
            if max_dst_id >= num_dst_nodes:
                logger.error(
                    f"❌ BOUNDS_ERROR for edge_type '{edge_type}': "
                    f"Max destination ID is {max_dst_id}, but node_type '{dst_type}' "
                    f"only has {num_dst_nodes} nodes!"
                )
                is_healthy = False
    return is_healthy


@no_type_check
def _diagnose_node_features(data: HeteroData) -> bool:
    """检查所有节点特征中是否存在 NaN 或 inf 值。"""
    is_clean = True
    for node_type in data.node_types:
        if "x" not in data[node_type]:
            logger.warning(
                f"⚠️  Node type '{node_type}' has no features ('x' attribute)."
            )
            continue

        features = data[node_type].x
        if torch.isnan(features).any():
            num_nan = torch.isnan(features).sum().item()
            logger.error(
                f"❌ Found {num_nan} NaN value(s) in features for node_type '{node_type}'!"
            )
            is_clean = False
        if torch.isinf(features).any():
            num_inf = torch.isinf(features).sum().item()
            logger.error(
                f"❌ Found {num_inf} Infinity value(s) in features for node_type '{node_type}'!"
            )
            is_clean = False
    return is_clean


# ==============================================================================
# 2. Online Data Consistency Validation
# ==============================================================================
# 这些函数用于抽样检查我们的本地数据与权威在线数据源的一致性。


class ValidationResult(NamedTuple):
    status: str
    message: str


def cast_result(result: Any) -> list[ValidationResult]:
    if result is None or not isinstance(result, list):
        raise TypeError(
            f"Expected a list of results from parallel execution, but got {type(result).__name__}"
        )
    results_for_report: list[ValidationResult] = cast(list[ValidationResult], result)
    if result and not all(isinstance(res, ValidationResult) for res in result):
        logger.warning(
            "Some items in parallel results are not of type ValidationResult."
        )
    return results_for_report


def run_online_validation(
    nodes_df: pd.DataFrame,
    n_samples: int = 100,
    n_jobs: int = 4,
    random_state: int = 42,
) -> None:
    """
    从 nodes.csv 生成的 DataFrame 中抽样，并行执行在线验证。
    """
    header = f"Starting Online Validation for {n_samples} Samples"
    logger.info(f"\n{'=' * 20} {header} {'=' * 20}")

    if n_samples > len(nodes_df):
        logger.warning(
            f"n_samples ({n_samples}) is larger than DataFrame size ({len(nodes_df)}). "
            "Validating all entries."
        )
        n_samples = len(nodes_df)

    sample_df = nodes_df.sample(n=n_samples, random_state=random_state)

    # 分离出需要验证的分子和蛋白质
    mols_to_validate = sample_df[
        sample_df["node_type"].isin(["drug", "ligand", "molecule"])
    ]
    prots_to_validate = sample_df[sample_df["node_type"] == "protein"]

    with Parallel(n_jobs=n_jobs) as parallel:
        # --- PubChem Validation ---
        if not mols_to_validate.empty:
            logger.info(
                f"Querying PubChem API for {len(mols_to_validate)} molecules..."
            )
            pubchem_results = parallel(
                delayed(_validate_pubchem_entry)(row.authoritative_id, row.structure)
                for _, row in tqdm(
                    mols_to_validate.iterrows(),
                    total=len(mols_to_validate),
                    desc="PubChem Checks",
                )
            )

            _print_validation_report(
                "PubChem CID vs SMILES",
                cast_result(pubchem_results),
                mols_to_validate,
                "authoritative_id",
            )

        # --- UniProt Validation ---
        if not prots_to_validate.empty:
            logger.info(
                f"Querying UniProt API for {len(prots_to_validate)} proteins..."
            )
            uniprot_results = parallel(
                delayed(_validate_uniprot_entry)(row.authoritative_id, row.structure)
                for _, row in tqdm(
                    prots_to_validate.iterrows(),
                    total=len(prots_to_validate),
                    desc="UniProt Checks",
                )
            )
            _print_validation_report(
                "UniProt ID vs Sequence",
                cast_result(uniprot_results),
                prots_to_validate,
                "authoritative_id",
            )

    logger.info(f"{'=' * (42 + len(header))}\n")


def _validate_pubchem_entry(cid: CID, local_smiles: SMILES) -> ValidationResult:
    """通过 PubChem API 验证 CID 和 SMILES 的一致性。"""
    time.sleep(0.25)  # Respect API rate limits
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/TXT"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        api_smiles = response.text.strip()

        local_mol = Chem.MolFromSmiles(local_smiles)
        api_mol = Chem.MolFromSmiles(api_smiles)

        if not local_mol:
            return ValidationResult("LOCAL_INVALID", "Local SMILES is invalid.")
        if not api_mol:
            return ValidationResult("API_INVALID", "API returned an invalid SMILES.")

        if Chem.MolToInchiKey(local_mol) == Chem.MolToInchiKey(api_mol):
            return ValidationResult(
                "MATCH", "Molecules are chemically equivalent (InChIKey match)."
            )
        else:
            return ValidationResult("MISMATCH", "InChIKey mismatch. Local vs API.")

    except requests.exceptions.RequestException as e:
        return ValidationResult("API_ERROR", str(e))


def _validate_uniprot_entry(
    pid: PID, local_sequence: ProteinSequence
) -> ValidationResult:
    """通过 UniProt API 验证 PID 和序列的一致性。"""
    url = f"https://rest.uniprot.org/uniprotkb/{pid}.fasta"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        lines = response.text.strip().split("\n")
        if len(lines) < 2:
            return ValidationResult(
                "API_INVALID", "API did not return a valid FASTA format."
            )

        api_sequence = "".join(lines[1:])

        if local_sequence.strip().upper() == api_sequence.strip().upper():
            return ValidationResult("MATCH", "Sequences are identical.")
        else:
            return ValidationResult(
                "MISMATCH",
                f"Sequence content differs. Local len={len(local_sequence)}, API len={len(api_sequence)}",
            )

    except requests.exceptions.RequestException as e:
        return ValidationResult("API_ERROR", str(e))


def _print_validation_report(
    title: str, results: list[ValidationResult], sample_df: pd.DataFrame, id_col: str
) -> None:
    """打印格式化的验证报告。"""
    report_header = f" {title} Validation Report "
    logger.info(f"\n{report_header:=^50}")

    counts = Counter(r.status for r in results)
    total = len(results)

    for status, count in sorted(counts.items()):
        percentage = (count / total) * 100
        logger.info(f"  - {status:<15}: {count:>4} ({percentage:.1f}%)")

    mismatches = [
        (res.message, row[id_col])
        for res, (_, row) in zip(results, sample_df.iterrows())
        if "MISMATCH" in res.status
    ]
    if mismatches:
        logger.warning("--- Mismatch Details (up to 5) ---")
        for i, (msg, item_id) in enumerate(mismatches[:5]):
            logger.warning(f"  {i + 1}. ID: {item_id} | Reason: {msg}")
    logger.info(f"{'=' * 50}")
