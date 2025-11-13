import hydra
import numpy as np
import pandas as pd
import research_template as rt
import torch

# 导入所有需要的项目内部模块
from helixpipe.configs import AppConfig, register_all_schemas
from helixpipe.data_processing import (
    GraphBuildContext,
    IDMapper,
)
from helixpipe.data_processing.services.graph_builder import HeteroGraphBuilder
from helixpipe.utils import get_path, register_hydra_resolvers

# 导入绘图工具
from .plot_utils import plot_pos_neg_similarity_kde, plot_similarity_distributions

# 在所有Hydra操作之前，执行全局注册
register_all_schemas()
register_hydra_resolvers()


def analyze_pos_neg_similarities(
    sampled_pairs: list, embeddings: torch.Tensor, id_mapper: IDMapper
) -> pd.DataFrame:
    """
    (可选)分析正负样本对中，实体间的嵌入相似度。
    """
    print("\n--- [Analysis] Analyzing Positive vs. Negative Pair Similarities...")

    # 1. 准备正样本
    pos_pairs = [(u, v) for u, v, _ in sampled_pairs]

    # 2. 准备负样本 (随机采样)
    neg_pairs = []
    all_mol_ids = list(id_mapper.molecule_to_id.values())
    all_prot_ids = list(id_mapper.protein_to_id.values())
    pos_pairs_set = set(pos_pairs)

    if not all_mol_ids or not all_prot_ids:
        return pd.DataFrame()

    rng = np.random.default_rng(123)  # 使用固定的种子
    while len(neg_pairs) < len(pos_pairs):
        mol_id = rng.choice(all_mol_ids)
        prot_id = rng.choice(all_prot_ids)
        if (mol_id, prot_id) not in pos_pairs_set:
            neg_pairs.append((mol_id, prot_id))

    # 3. 计算相似度
    results = []
    # 使用 PyTorch 的高效索引和计算
    embeddings = embeddings.float()  # 确保是 float

    for label, pairs in [("Positive", pos_pairs), ("Negative", neg_pairs)]:
        if not pairs:
            continue

        mol_indices = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        # 将蛋白质全局ID转换为0-based索引
        prot_indices = torch.tensor(
            [p[1] - id_mapper.num_molecules for p in pairs], dtype=torch.long
        )

        mol_embs = embeddings[mol_indices]
        prot_embs = embeddings[id_mapper.num_molecules :][prot_indices]

        # 归一化后计算点积，即余弦相似度
        mol_embs = torch.nn.functional.normalize(mol_embs, p=2, dim=1)
        prot_embs = torch.nn.functional.normalize(prot_embs, p=2, dim=1)

        similarities = (mol_embs * prot_embs).sum(dim=1).cpu().numpy()

        for sim in similarities:
            results.append({"similarity": sim, "label": label})

    return pd.DataFrame(results)


config_path = rt.get_project_root() / "conf"


@hydra.main(config_path=str(config_path), config_name="config", version_base=None)
def main(cfg: AppConfig):
    """
    一个独立的、由Hydra驱动的脚本，用于在全局相关实体上进行相似度分布的
    探索性数据分析 (EDA)，以指导阈值设定。
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "STARTING SIMILARITY DISTRIBUTION ANALYSIS SCRIPT")
    print("=" * 80)

    try:
        # --- 步骤 1: 模拟 main_pipeline 的早期数据加载和采样 ---
        print("--- [Step 1/4] Loading data and simulating pre-splitting stage...")

        # [MODIFIED] 使用 nodes.csv 来正确地初始化 IDMapper
        # a. 加载 nodes.csv，这是我们所有实体信息的“户籍簿”
        nodes_df = pd.read_csv(get_path(cfg, "processed.common.nodes_metadata"))

        # b. 准备一个最小化的、只包含ID的DataFrame，以符合IDMapper的初始化契约
        schema = cfg.data_structure.schema.internal.authoritative_dti

        mol_nodes = nodes_df[nodes_df["node_type"] != "protein"]
        prot_nodes = nodes_df[nodes_df["node_type"] == "protein"]

        # 创建一个足够让IDMapper收集所有ID的DataFrame
        simulated_df = pd.DataFrame(
            {
                schema.molecule_id: mol_nodes["authoritative_id"],
                schema.protein_id: prot_nodes["authoritative_id"],
            }
        )

        # c. 使用这个模拟的DataFrame来正确初始化IDMapper
        id_mapper = IDMapper([simulated_df], cfg)

        # d. 注入“drug”的身份定义
        drug_cids = set(
            nodes_df[nodes_df["node_type"] == "drug"]["authoritative_id"].astype(int)
        )
        id_mapper.set_drug_cids(drug_cids)

        # e. 最终化IDMapper，使其内部状态（如num_molecules）被正确计算
        id_mapper.finalize_mappings()

        # f. 加载特征嵌入
        features_np = np.load(get_path(cfg, "processed.common.node_features"))
        features_tensor = torch.from_numpy(features_np)

        # g. 安全地【读取】num_molecules属性，并分离特征
        num_molecules = id_mapper.num_molecules
        molecule_embeddings = features_tensor[:num_molecules]
        protein_embeddings = features_tensor[num_molecules:]

        # h. 加载一个有代表性的交互对样本
        # 我们加载 Fold 1 的训练标签作为代表，来定义本次分析的“相关实体”范围
        train_labels_df = pd.read_csv(
            get_path(
                cfg,
                "processed.specific.labels_template",
                prefix="fold_1",
                suffix="train",
            )()
        )
        # 简单地给一个默认的 relation_type
        sampled_pairs_with_type = [
            (row.source, row.target, "interacts_with")
            for row in train_labels_df.itertuples()
        ]

        # --- 步骤 2: 创建全局分析上下文 ---
        print("\n--- [Step 2/4] Creating global analysis context...")
        # 注意：这里的ID已经是全局逻辑ID了
        relevant_mol_ids = {u for u, v, _ in sampled_pairs_with_type}
        relevant_prot_ids = {v for u, v, _ in sampled_pairs_with_type}

        context = GraphBuildContext(
            fold_idx=0,  # 0 在这里表示这是一个“全局”分析上下文
            global_id_mapper=id_mapper,
            global_mol_embeddings=molecule_embeddings,
            global_prot_embeddings=protein_embeddings,
            relevant_mol_ids=relevant_mol_ids,
            relevant_prot_ids=relevant_prot_ids,
            config=cfg,
        )

        # --- 步骤 3: 调用 Builder 的分析方法 ---
        print("\n--- [Step 3/4] Running builder in analysis-only mode...")
        builder = HeteroGraphBuilder(
            config=cfg,
            context=context,
            molecule_embeddings=context.local_mol_embeddings,
            protein_embeddings=context.local_prot_embeddings,
        )

        all_similarities_df = builder.analyze_similarities()

        # --- 步骤 4: 可视化与保存 ---
        print("\n--- [Step 4/4] Generating and saving plots...")

        # a. 准备输出目录
        output_dir = (
            rt.get_project_root()
            / "analysis_outputs"
            / cfg.dataset_collection.name
            / cfg.data_params.name
            / cfg.relations.name
        )
        rt.ensure_path_exists(output_dir / "dummy.txt")

        # b. 绘制相似度分布图
        plot_similarity_distributions(
            df=all_similarities_df, output_dir=output_dir, config=cfg
        )

        # c. (可选) 绘制正负样本相似度对比图
        pos_neg_sim_df = analyze_pos_neg_similarities(
            sampled_pairs_with_type, features_tensor, id_mapper
        )
        if not pos_neg_sim_df.empty:
            plot_pos_neg_similarity_kde(
                df=pos_neg_sim_df,
                output_path=output_dir / "positive_vs_negative_pair_similarity.png",
                config=cfg,
            )

        print("\n" + "=" * 80)
        print(f"✅ ANALYSIS COMPLETE. All plots saved to: {output_dir}")
        print("=" * 80)

    except FileNotFoundError as e:
        print(
            f"❌ FATAL: A required data file was not found: {e.filename if hasattr(e, 'filename') else e}"
        )
        print(
            "   Please ensure you have successfully run the main data processing pipeline (`run.py`) first to generate all necessary files."
        )
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
