### **ADR-003: 重构内部数据模型为规范化交互格式**

**状态：** 被 ADR-006 取代 (Superseded by ADR-006)

**日期：** 2025-11-04

#### **背景 (Context)**

在尝试实现 **ADR-002（集成 PPI 信息）** 的过程中，我们发现现有的内部数据模型存在一个核心的架构瓶颈。该模型隐性地假设了所有交互都遵循“药物-靶点相互作用（DTI）”的模式，即 `(分子ID, 蛋白质ID)`。这导致了以下问题：

1.  **硬编码依赖：** 多个核心服务（如 `BaseProcessor._filter_by_whitelist`, `InteractionStore`）被硬编码为查找特定的列名（如 `'UniProt_ID'`, `'PubChem_CID'`）。
2.  **缺乏扩展性：** 当引入不符合 DTI 模式的新数据类型（如 `protein-protein` 交互）时，会导致 `KeyError` 和逻辑混乱，需要对多个模块进行侵入式修改。
3.  **未来隐患：** 如果未来需要引入更多异构实体（如基因、疾病、通路），当前架构将难以为继，每次都需要大量的“打补丁”式修改。

为了优雅地支持 ADR-002 并为项目的长远发展奠定基础，我们需要设计一个更通用、更具扩展性的内部数据模型。

#### **决策 (Decision)**

我们将废弃现有的、基于特定列名的“DTI-like”内部 DataFrame 格式，转而采用一种更高级的抽象——**“规范化交互三元组”**格式。

从现在起，所有 `*Processor` 的最终产出，都必须是一个遵循以下标准结构的 DataFrame：

| `source_id` | `source_type` | `target_id` | `target_type` | `relation_type` | ... (可选列) |
| :---------- | :------------ | :---------- | :------------ | :-------------- | :----------- |
| (Any)       | (str)         | (Any)       | (str)         | (str)           | ...          |

**核心架构变更：**

1.  **`*Processor` 职责变更：** 每个 `Processor` 的核心职责将是**“翻译”**——将任何特定格式的原始数据，翻译成上述的规范化交互格式。对 ID 的白名单验证等领域特定逻辑，将**下沉**到各自的 `Processor` 内部实现。
2.  **`BaseProcessor` 简化：** `_filter_by_whitelist` 等具有硬编码假设的通用方法将被移除，使得 `BaseProcessor` 的流水线更加通用。
3.  **核心服务重构 (`IDMapper`, `InteractionStore`)：**
    - `InteractionStore` 将被简化为一个纯粹的“数据仓库”，只负责拼接所有 `Processor` 产出的规范化 DataFrame。
    - `IDMapper` 将被重构，使其能够遍历 `(source_id, source_type)` 和 `(target_id, target_type)` 列，来自动发现和管理所有类型的实体，无论它们是蛋白质、分子、基因还是疾病。

#### **备选方案 (Considered Options)**

**1. “打补丁”式修复**

- **方案描述：** 逐个修复当前遇到的 `KeyError`。例如，修改 `_filter_by_whitelist` 使其能处理 `protein1_id`, `protein2_id`；修改 `InteractionStore` 以特殊逻辑处理 PPI 数据。
- **优点：** 短期内工作量最小，能最快地让 PPI 数据跑起来。
- **缺点：**
  - **技术债务：** 每引入一种新数据类型，都需要增加一堆 `if/else` 特殊逻辑，代码会迅速变得复杂、脆弱且难以维护。
  - **治标不治本：** 没有解决问题的根源——数据模型与业务逻辑耦合过紧。
- **决策理由：** 这是一种短视的、会损害项目长期健康度的做法，予以否决。

**2. 为每种交互类型创建独立的流水线**

- **方案描述：** 为 DTI、PPI 等不同交互类型创建完全独立的 `InteractionStore` 和 `IDMapper` 实例，在最后阶段再将它们构建的图合并。
- **优点：** 避免了修改现有核心服务。
- **缺点：**
  - **ID 空间不统一：** 会产生多个独立的逻辑 ID 空间，在最后合并图时需要进行复杂的 ID 重映射，极易出错。
  - **实体信息割裂：** 一个在 DTI 和 PPI 中都出现的蛋白质，可能会在两个独立的 `IDMapper` 中被当作两个不同的实体，导致信息无法有效聚合。
- **决策理由：** 该方案会引入巨大的复杂性和潜在的数据一致性问题，予以否决。

#### **后果 (Consequences)**

**正面影响：**

- **极高的可扩展性：** 项目现在拥有了一个可以轻松容纳任何新实体和关系类型的“即插即用”数据底层。
- **彻底的解耦：** 核心服务（`IDMapper`, `InteractionStore`）与具体的业务数据类型（DTI, PPI）完全解耦，变得更加稳定和通用。
- **职责清晰：** 每个 `Processor` 的职责变得更加单一和明确，易于开发和测试。
- **支持回归预测：** 规范化格式中的 `label` 列天然支持从`int`（分类）平滑过渡到`float`（回归），只需在 `Processor` 和 `Trainer` 中进行相应修改即可。

**负面影响/风险：**

- **一次性的重构成本：** 需要对 `BaseProcessor`, `BindingdbProcessor`, `GtopdbProcessor`, `InteractionStore`, `IDMapper` 等多个现有核心模块进行一次性的、中等规模的重构。
- **抽象层级的增加：** 引入了一个新的抽象层，对于新加入的开发者，需要先理解“规范化交互格式”这个核心概念。然而，我们认为这个抽象是良性的，它带来的长期收益远大于初期的学习成本。

---

**附录：实施过程中的架构精炼 (Appendix: Architectural Refinement during Implementation)**

在重构各个 \*Processor 以产出“规范化交互格式”的过程中，我们发现实体类型（如'molecule'）和原始关系类型（如'physical_association'）都属于知识图谱的“Schema 定义”范畴。

为了追求极致的概念完整性，我们做出了以下**精炼决策**：

- **将 RelationNames dataclass 从 RelationsConfig 移动到 KnowledgeGraphConfig 中，并重命名为 RelationTypeNames。**

**理由：**

- **职责统一：** KnowledgeGraphConfig 现在成为定义图谱“词汇表”（所有节点类型和边类型）的唯一真理之源。
- **职责纯化：** RelationsConfig 的职责被纯化为“配置”在一次特定实验中如何使用这些关系（例如，通过 flags 启用或禁用）。

**取代原因**
本决策中提出的‘规范化交互模型’和‘中心化校验’的思想，已被 ADR-004 中提出的、更根本的‘实体元数据中心’架构所继承和完善。
