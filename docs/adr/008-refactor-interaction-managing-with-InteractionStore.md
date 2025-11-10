# ADR-008: 将交互处理重构为以 `InteractionStore` 为中心的统一关系管理模型

- **状态：** 已接受 (Accepted)
- **日期：** 2025-11-10

## 背景 (Context)

在 `ADR-006` 成功将 `IDMapper` 确立为“实体元数据中心”后，我们解决了所有与**节点（Node）**相关的管理问题。然而，我们发现对于**交互（Interaction/Edge）**的处理，其逻辑变得高度**碎片化和分散**，形成了一个新的架构瓶颈。

当前的交互数据流如下：

1.  **原始聚合：** 所有 `Processor` 的输出被简单地 `pd.concat` 成一个巨大的、临时的 DataFrame。
2.  **匿名传递：** 这个 DataFrame 或者一个 `List[Tuple]` 被作为“二等公民”，在 `main_pipeline`, `sampler`, `splitter` 等多个服务之间传来传去。
3.  **逻辑重复：** 复杂的交互筛选逻辑（如根据源/目标实体的属性进行过滤）需要在 `sampler` 和 `splitter` 中被**重复实现**，违反了“不要重复自己”（DRY）原则。
4.  **职责不清：** `main_pipeline` 承担了过多的交互过滤职责，而 `sampler` 和 `splitter` 则混合了策略执行和底层数据操作，导致了高耦合和低内聚。

问题的根源在于，我们为“实体”建立了中央管理服务，却**缺少一个与之对应的“交互”中央管理服务**。

## 决策 (Decision)

我们将引入一个新的核心服务——`InteractionStore`，并围绕它对数据处理流水线进行重构，从而建立一个**节点（`IDMapper`）和边（`InteractionStore`）双核心驱动**的架构模型。

`InteractionStore` 将成为项目中所有**交互关系的唯一事实来源（Single Source of Truth）**，负责统一管理、查询和过滤。

### 核心架构变更：

1.  **`InteractionStore` 服务诞生：**
    创建一个新的 `InteractionStore` 类，其职责是：

    - a. 在初始化时，聚合所有 `Processor` 的输出，并为每条交互记录打上其**来源数据集**的标签。
    - b. 提供一个核心的 `filter_by_entities(valid_entity_ids)`方法，用一个纯净的实体 ID 集来过滤自身，并返回一个**新的、纯净的 `InteractionStore` 实例**（不可变模式）。
    - c. 提供一个强大的 `query(selector: InteractionSelectorConfig, id_mapper: IDMapper)` API，允许下游服务根据复杂的边规则（源、目标、关系类型）进行声明式查询。
    - d. 提供 `get_mapped_positive_pairs(id_mapper)`方法，将内部的权威 ID 对，高效地映射为下游任务所需的**逻辑 ID 对**。

2.  **`main_pipeline` 流程重构：**

    - 在 `stage1`，`IDMapper` 和 `InteractionStore` 将被**同时初始化**，并行存在。
    - 在 `stage5`，`IDMapper` 和 `InteractionStore` 将**同时被“最终化”**：
      - `IDMapper` 接收 `valid_entity_ids` 来分配逻辑 ID。
      - `InteractionStore` 调用 `filter_by_entities(valid_entity_ids)`来生成一个 `pure_interaction_store` 实例。
    - 所有下游阶段（`stage7` 及以后）将接收这个 `pure_interaction_store` 对象，而不是临时的 DataFrame 或列表。

3.  **下游服务（`sampler`, `splitter`）重构：**

    - `sampler` 和 `splitter` 不再直接操作 `List[Tuple]`。它们将接收 `InteractionStore` 对象作为输入。
    - 它们的核心逻辑将被简化为：调用 `interaction_store.query(...)`来获取满足特定策略的交互子集，然后在这些子集（也是 `InteractionStore` 对象）上执行采样或划分，最后再合并结果。
    - 所有底层的、重复的交互匹配逻辑，将从 `sampler` 和 `splitter` 中移除，统一收归到 `InteractionStore.query` 方法中。

4.  **引入 `selectors.py` 模块：**
    - 创建一个新的 `src/nasnet/configs/selectors.py` 文件。
    - `EntitySelectorConfig` 将从 `training.py` 移动到此文件中。
    - 新增 `InteractionSelectorConfig` 的定义，用于 `sampler` 和 `splitter` 的声明式配置。

## 备选方案 (Considered Options)

### 1. 保持现状，将匹配逻辑抽象为工具函数

- **方案描述：** 不引入 `InteractionStore` 类。创建一个通用的工具函数 `match_interaction(pair, selector, id_mapper)`，然后让 `sampler` 和 `splitter` 在内部循环中调用这个函数。
- **优点：** 改动范围最小，无需引入新类。
- **缺点：**
  - **性能低下：** 在 Python 中对大型列表进行逐项循环匹配，效率远低于在 Pandas DataFrame 上执行的向量化操作。
  - **数据流不清晰：** 交互数据仍然以匿名的 `List[Tuple]`形式在系统中流转，认知负担没有降低。
  - **治标不治本：** 没有解决“交互”作为“二等公民”的根本问题。
- **决策理由：** 这是一种战术上的小优化，而非战略上的架构升级，无法从根本上解决逻辑分散和数据流混乱的问题，予以否决。

### 2. 将 `InteractionStore` 作为 `IDMapper` 的下游

- **方案描述：** 先完成 `IDMapper` 的全部流程（包括最终化和逻辑 ID 分配），然后再用最终的 `IDMapper` 去初始化 `InteractionStore`，使其内部直接存储逻辑 ID。
- **优点：** `InteractionStore` 内部只处理逻辑 ID，可能更简单。
- **缺点：**
  - **破坏了信息流：** `IDMapper` 的初始化依赖于从所有交互中提取实体 ID。如果 `InteractionStore` 在 `IDMapper` 之后，会导致数据流的“回头”，逻辑混乱。
  - **耦合过紧：** 使得 `InteractionStore` 强依赖于 `IDMapper` 的最终状态，降低了其作为独立关系管理器的通用性。
- **决策理由：** “并行协作”模式比“串行依赖”模式更能体现两大服务对等的核心地位，也使得数据流更加单向和清晰，故否决此方案。

## 后果 (Consequences)

### 正面影响：

- **架构的二元对称性：** 形成了 `IDMapper`（节点中心）和 `InteractionStore`（边中心）的完美对称，使得整个系统的概念模型极度清晰和完整。
- **逻辑的高度内聚：** 所有与“边”相关的查询、过滤、匹配逻辑，现在都内聚在 `InteractionStore` 这一个地方，彻底解决了逻辑分散和代码重复的问题。
- **下游服务的极致简化：** `sampler` 和 `splitter` 被重构为纯粹的“策略执行引擎”，其内部实现变得极其简单、优雅且易于测试。它们只负责编排对 `InteractionStore` API 的调用。
- **性能提升：** 将交互数据保留在 Pandas DataFrame 中，使得所有查询和过滤操作都可以利用向量化的优势，性能优于在 Python 原生列表上的循环。
- **增强的可读性与可维护性：** 在 `main_pipeline` 中传递一个有明确 API 的 `InteractionStore` 对象，比传递一个匿名的 `List[Tuple]`，大大提高了代码的可读性和长期可维护性。

### 负面影响/风险：

- **一次性的重构成本：** 需要对 `main_pipeline`, `sampler.py`, `splitter.py` 进行一次性的、中等规模的重构。然而，这次重构是为了消除技术债务，其长期收益远大于短期成本。
- **增加了一个核心抽象：** 引入了 `InteractionStore` 这个新的核心抽象。但我们认为这是良性的抽象，它补全了系统缺失的一环，降低了整体的认知复杂度。

## 关联的 ADR

- **完善 (Complements):** 本 ADR 是 `ADR-006` (IDMapper 中心化)在“关系”维度的直接对应和补充，两者共同构成了项目的核心数据模型。
