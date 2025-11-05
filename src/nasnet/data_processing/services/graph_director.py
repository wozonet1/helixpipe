# 文件: src/nasnet/data_processing/services/graph_director.py (全新)

from typing import List, Tuple

from nasnet.configs import AppConfig

# 我们需要从 graph_builder 模块导入 GraphBuilder 的抽象基类定义
# (我们将在下一步创建它)
from .graph_builder import GraphBuilder


class GraphDirector:
    """
    【指挥者 (Director)】
    根据 Builder 设计模式，这个类负责根据传入的配置（蓝图），
    按预设顺序调用一个具体的图生成器(Builder)的方法来构建图。

    它将图的【构建流程】与具体的【构建实现】相分离。
    """

    def __init__(self, config: AppConfig):
        """
        初始化指挥者。

        Args:
            config (AppConfig): 完整的Hydra配置对象，作为构建的“蓝图”。
        """
        self.config = config
        self.verbose = config.runtime.verbose

    def construct(self, builder: GraphBuilder, train_pairs: List[Tuple[int, int, str]]):
        """
        执行图的构建。

        这个方法会读取 config.relations.flags，并根据每个标志位的真假，
        来决定是否调用 builder 对应的构建方法。

        Args:
            builder (GraphBuilder): 一个实现了 GraphBuilder 接口的具体生成器实例。
            train_pairs (List[Tuple[int, int, str]]):
                训练集中的交互对，包含 (u, v, final_edge_type)。
                这是构建交互边的必需“原料”。
        """
        if self.verbose > 0:
            print("\n--- [GraphDirector] Starting graph construction process... ---")

        # 从配置中获取所有关系类型的开关
        flags = self.config.relations.flags

        # --- 按预设的、逻辑清晰的顺序指挥构建 ---

        # 1. 构建交互边 (DTI, LPI, inhibits, catalyzes...)
        #    我们不需要在这里区分具体的交互类型，因为 add_interaction_edges
        #    方法会处理所有类型的交互。我们只需要一个总开关。
        #    一个合理的约定是：只要有任何最终关系类型被启用，就调用这个方法。
        #    （另一种策略是检查 train_pairs 是否为空，但检查配置更符合设计）
        if train_pairs:  # 只要有交互数据，就尝试构建交互边
            if self.verbose > 0:
                print("  -> Instructing builder to add interaction edges...")
            builder.add_interaction_edges(train_pairs)

        # 2. 构建分子-分子相似性边 (Drug-Drug, Ligand-Ligand, Drug-Ligand)
        #    只要有任何一种分子间的相似性被启用，就调用这个方法。
        if (
            flags.get("drug_drug_similarity", False)
            or flags.get("ligand_ligand_similarity", False)
            or flags.get("drug_ligand_similarity", False)
        ):
            if self.verbose > 0:
                print("  -> Instructing builder to add molecule similarity edges...")
            builder.add_molecule_similarity_edges()

        # 3. 构建蛋白质-蛋白质相似性边 (Protein-Protein)
        if flags.get("protein_protein_similarity", False):
            if self.verbose > 0:
                print("  -> Instructing builder to add protein similarity edges...")
            builder.add_protein_similarity_edges()

        # 4. 【为未来扩展预留】构建蛋白质-蛋白质交互边 (PPI from STRING)
        if flags.get(
            "associated_with", False
        ):  # 假设最终的PPI边类型是 'associated_with'
            if self.verbose > 0:
                print("  -> Instructing builder to add PPI edges...")
            # builder.add_ppi_edges() # <-- 调用未来的方法

        if self.config.training.coldstart.strictness == "strict":
            if self.verbose > 0:
                print("  -> Instructing builder to apply strict cold-start filter...")
            builder.filter_background_edges_for_strict_mode()
        if self.verbose > 0:
            print("--- [GraphDirector] Graph construction process finished. ---")
