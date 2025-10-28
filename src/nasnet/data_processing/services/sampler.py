# 文件: src/nasnet/data_processing/services/sampler.py (最终正确顺序版)

import math
import random
from typing import List, Set, Tuple

from nasnet.configs import AppConfig

from .id_mapper import IDMapper


def sample_interactions(
    all_positive_pairs: List[Tuple[int, int, str]],
    id_mapper: IDMapper,
    config: AppConfig,
) -> Tuple[List[Tuple[int, int, str]], Set[Tuple[int, int]]]:
    """
    【V4 - 正确顺序串联版】
    根据配置，以流水线方式对输入的交互对列表执行采样。

    正确执行顺序:
    1. (如果启用) 首先进行分层采样 (drug_to_ligand_ratio)。
    2. (如果启用) 然后在第一步的结果上，进行统一采样 (fraction)。
    """
    sampling_cfg = config.data_params.sampling

    if not sampling_cfg.enabled:
        original_set = {(u, v) for u, v, _ in all_positive_pairs}
        return all_positive_pairs, original_set

    # 初始化工作数据集
    working_pairs = all_positive_pairs

    # --- 流水线步骤1: 分层采样 (Stratified Sampling) ---
    if sampling_cfg.get("drug_to_ligand_ratio", None) is not None:
        print(
            f"\n--- [Sampler] Applying STRATIFIED sampling with D:L ratio of {sampling_cfg.drug_to_ligand_ratio}:1... ---"
        )

        drug_pairs = [p for p in working_pairs if p[0] < id_mapper.num_drugs]
        ligand_pairs = [p for p in working_pairs if p[0] >= id_mapper.num_drugs]

        n_ligand_pairs = len(ligand_pairs)
        n_drug_pairs = len(drug_pairs)

        if n_ligand_pairs == 0 and n_drug_pairs > 0:
            print(
                "    - WARNING: No Ligand pairs found for stratified sampling. Using all available Drug pairs."
            )
            working_pairs = drug_pairs  # 更新工作集
        elif n_ligand_pairs > 0:
            print(
                f"    - Found {n_drug_pairs} Drug pairs and {n_ligand_pairs} Ligand pairs."
            )

            n_drug_to_sample = int(n_ligand_pairs * sampling_cfg.drug_to_ligand_ratio)
            n_drug_to_sample = min(n_drug_to_sample, n_drug_pairs)

            print(
                f"    - Sub-sampling {n_drug_to_sample} Drug pairs to match {n_ligand_pairs} Ligand pairs."
            )

            random.seed(config.runtime.seed)
            sampled_drug_pairs = random.sample(drug_pairs, n_drug_to_sample)

            # 更新工作集为分层采样后的结果
            working_pairs = sampled_drug_pairs + ligand_pairs
        else:
            working_pairs = []

    # --- 流水线步骤2: 统一采样 (Uniform Sampling) ---
    # 这一步作用于上一步(分层)的结果 `working_pairs` 上
    if sampling_cfg.get("fraction", None) is not None and sampling_cfg.fraction < 1.0:
        print(
            f"\n--- [Sampler] Applying UNIFORM FRACTION ({sampling_cfg.fraction}) to each stratum..."
        )

        sample_frac = sampling_cfg.fraction
        if not (isinstance(sample_frac, float) and 0 < sample_frac < 1.0):
            print(
                f"    - WARNING: Invalid fraction '{sample_frac}'. It must be a float between 0 and 1. Skipping."
            )
        else:
            # 1. 再次分离出经过第一步处理后的 drug 和 ligand 组
            drug_pairs_stratum = [
                p for p in working_pairs if p[0] < id_mapper.num_drugs
            ]
            ligand_pairs_stratum = [
                p for p in working_pairs if p[0] >= id_mapper.num_drugs
            ]

            # 2. 计算每一组需要采样的数量

            n_drug_to_sample = math.ceil(len(drug_pairs_stratum) * sample_frac)
            n_ligand_to_sample = math.ceil(len(ligand_pairs_stratum) * sample_frac)

            print(
                f"    - Sub-sampling {n_drug_to_sample} Drug pairs from {len(drug_pairs_stratum)}."
            )
            print(
                f"    - Sub-sampling {n_ligand_to_sample} Ligand pairs from {len(ligand_pairs_stratum)}."
            )

            # 3. 分别进行采样
            random.seed(config.runtime.seed)
            sampled_drug_pairs = random.sample(drug_pairs_stratum, n_drug_to_sample)
            sampled_ligand_pairs = random.sample(
                ligand_pairs_stratum, n_ligand_to_sample
            )

            # 4. 合并，更新工作集
            working_pairs = sampled_drug_pairs + sampled_ligand_pairs

    # --- 最终处理 (保持不变) ---
    final_sampled_pairs = working_pairs
    random.seed(config.runtime.seed)
    random.shuffle(final_sampled_pairs)

    sampled_pairs_set = {(u, v) for u, v, _ in final_sampled_pairs}
    print(
        f"\n--- [Sampler] Sampling complete. Total pairs for downstream processing: {len(final_sampled_pairs)} ---"
    )

    return final_sampled_pairs, sampled_pairs_set
