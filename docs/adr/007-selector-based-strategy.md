### **ADR-007: 引入基于选择器的策略定义以实现配置正交性**

**状态：** 已实施 (Implemented)

**日期：** 2025-11-05

#### **背景 (Context)**

随着我们架构的演进（ADR-006），`IDMapper` 成为了一个强大的实体元数据中心。然而，下游服务如 `DataSplitter` 和 `SupervisionFileManager` 依然依赖于硬编码的字符串（如 `coldstart.mode: "molecule"`）来触发特定的行为。这些字符串的**语义**需要在 Python 代码内部进行**“解释”**，造成了**配置与实现之间的隐性耦合**。

例如，`DataSplitter` 必须知道字符串 `"all_molecules"` 意味着它要去调用 `id_mapper.is_molecule()`。这种设计使得新增一个划分策略（如“只划分内源性配体”）需要修改 Python 代码，降低了系统的灵活性和可扩展性。

#### **决策 (Decision)**

我们将引入一种**“基于标签选择器的策略定义（Tag Selector-Based Strategy Definition）”**的新设计模式，以实现配置的完全正交性和声明式。

1.  **引入 `EntitySelectorConfig`:** 创建一个新的`dataclass`，`EntitySelectorConfig`。它能结构化地描述一组实体筛选规则，例如 `entity_types`, `meta_types`, `from_sources`。

2.  **重构 `ColdstartConfig`:**

    - 将原有的基于字符串的 `pool_scope` 和 `evaluation_scope` 字段，替换为基于 `EntitySelectorConfig` 的结构化配置。
    - `pool_scope` 将直接定义用于冷启动划分的实体池。
    - `evaluation_scope` 将由一对 `EntitySelectorConfig` 组成，分别定义一条有效评估边的源节点和目标节点所需满足的条件。

3.  **增强 `IDMapper`:** 为`IDMapper`新增一个 `get_ids_by_selector(selector: EntitySelectorConfig)` 方法，使其能够解析并执行这些选择器规则，返回一个逻辑 ID 列表。

4.  **“无知化”下游服务：**
    - 重构 `DataSplitter`，使其不再“解释”任何字符串，而是直接将 `pool_scope` 选择器对象传递给 `IDMapper` 以获取实体池。
    - 重构 `SupervisionFileManager`，使其不再硬编码评估逻辑，而是遍历测试交互，并使用 `evaluation_scope` 中的选择器规则和 `IDMapper` 的元信息查询接口来动态地判断每条交互是否应被纳入最终评估。

#### **备选方案 (Considered Options)**

**1. 保持现状（字符串驱动）**

- **方案描述：** 继续使用 `"drug_only"`, `"protagonist_dti_only"` 等字符串，并在 `DataSplitter` 和 `SupervisionFileManager` 中维护一个不断增长的 `if/elif/else` 逻辑块来解释这些字符串。
- **优点：** 无需立即进行重构。
- **缺点：** **扩展性差。** 每增加一种新的实验维度（例如，按新的亚型划分），都需要修改核心的 Python 代码，违背了配置驱动的初衷，并会迅速积累技术债务。
- **决策理由：** 该方案无法满足我们对系统长期灵活性和可维护性的要求，予以否决。

#### **后果 (Consequences)**

**正面影响：**

1.  **实现了配置的终极正交性：** 数据源（`dataset_collection`）、语义粒度（`knowledge_graph`）、划分策略（`pool_scope`）和评估范围（`evaluation_scope`）现在是四个完全独立的“控制旋钮”，可以任意组合。
2.  **将策略定义权完全交给用户：** 研究者现在可以通过编写**YAML 文件**来**“发明”**全新的、极其复杂的划分和评估策略，而**无需修改任何一行 Python 代码**。这极大地提升了实验的灵活性和迭代速度。
3.  **代码的终极解耦：** `DataSplitter` 和 `SupervisionFileManager` 变成了纯粹的、**“无知”的执行引擎**。它们不再关心策略的“是什么”和“为什么”，只关心“如何执行”一个被清晰定义的 `EntitySelectorConfig` 对象。
4.  **配置即文档：** 实验的设置（Setup）变得完全**自文档化**。通过阅读 `coldstart` 配置块，任何人都能清晰、无歧义地理解该实验的划分和评估方式。

**负面影响/权衡：**

- **增加了配置的“冗长度”：** 相比于一个简单的字符串，结构化的选择器配置会更长。然而，我们认为这种“冗长度”换来的是**无歧义的清晰性**，是完全值得的。
- **一次性的重构成本：** 需要对 `IDMapper`, `DataSplitter`, `SupervisionFileManager` 及其测试进行一次性的、中等规模的重构。

**关联的 ADR**

- 本决策是 `ADR-006 (IDMapper中心化)` 的一次直接且强大的应用，它利用了`IDMapper`作为元数据中心的能力，来解锁更高级的配置驱动能力。
