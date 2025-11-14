### **ADR-010: 将 `data_params` 配置重构为按模块划分的命名空间结构**

**状态：** 已接受 (Accepted)

**日期：** 2025-11-13

#### **背景 (Context)**

随着 `HelixPipe` 框架接入的数据源 (`Processor`) 越来越多，我们发现 `conf/data_params/` 的配置结构开始出现“坏味道”。当前的`DataParamsConfig`采用一种**扁平化**的结构，将所有数据处理相关的参数都放在同一个层级。

例如，`affinity_threshold_nM` (专用于 `BindingdbProcessor`)、`km_threshold_nM` (专用于 `BrendaProcessor`) 与 `similarity_top_k` (通用参数) 被混合在一起。这导致了以下问题：

1.  **配置混杂与职责不清：** 很难从配置文件中一眼看出哪个参数属于哪个模块。当多个 `Processor` 同时运行时，参数的作用域变得模糊，存在误用的风险。
2.  **可读性差：** 随着系统扩展，`data_params` 的根命名空间会变得越来越拥挤和混乱。
3.  **扩展性受限：** 每当为一个新的 `Processor` 添加专属参数时，都必须污染顶层的命名空间，这使得配置的维护成本越来越高。

为了保持配置系统的清晰、可扩展和高内聚，我们需要一种更结构化的方式来组织这些参数。

#### **决策 (Decision)**

我们将对`DataParamsConfig`的`dataclass`定义进行重构，从**扁平化结构**演进为**按模块划分的命名空间结构**。

**核心架构变更：**

1.  **引入嵌套的`dataclass`：**

    - 我们将为每个拥有专属参数的`Processor`，创建一个对应的嵌套`dataclass`。例如：

      ```python
      @dataclass
      class BindingdbParams:
          affinity_threshold_nM: int = 10000

      @dataclass
      class BrendaParams:
          km_threshold_nM: int = 10000
      ```

2.  **重构`DataParamsConfig`：**

    - 顶层的`DataParamsConfig`将包含这些嵌套的`dataclass`作为其字段，从而形成清晰的命名空间。
      ```python
      @dataclass
      class DataParamsConfig:
          name: str
          # --- 通用参数 ---
          similarity_top_k: int
          # --- 按 Processor 划分的参数块 ---
          bindingdb: BindingdbParams = field(default_factory=BindingdbParams)
          brenda: BrendaParams = field(default_factory=BrendaParams)
          # --- 通用功能块 ---
          filtering: FilteringConfig = field(default_factory=FilteringConfig)
          sampling: DownstreamSamplingConfig = field(default_factory=DownstreamSamplingConfig)
      ```

3.  **更新代码中的配置访问路径：**

    - 每个`Processor`现在将从其专属的、类型安全的命名空间中读取配置，确保了参数的正确归属。
      ```python
      # BrendaProcessor.py
      # 旧: self.config.data_params.km_threshold_nM
      # 新: self.config.data_params.brenda.km_threshold_nM
      ```

4.  **调整`YAML`文件结构：**
    - `YAML`配置文件将镜像新的`dataclass`结构，通过嵌套的字典来定义不同模块的参数，极大地提升了可读性。

#### **备选方案 (Considered Options)**

**1. 保持现状，通过命名约定来区分**

- **方案描述：** 不改变`dataclass`结构，而是通过为参数添加前缀来进行区分，例如`bindingdb_affinity_threshold_nM`。
- **优点：** 无需重构代码。
- **缺点：**
  - **治标不治本：** 命名空间仍然是扁平的，可读性问题没有得到根本解决。
  - **违反 DRY 原则：** `bindingdb_`这样的前缀会在多个参数中重复。
  - **无法利用 Hydra 的结构化优势：** 无法对`bindingdb`的所有参数进行整体的覆盖或组合。
- **决策理由：** 这是一种临时的“补丁”，而非长期的、优雅的架构解决方案，予以否决。

**2. 在`conf/data_params/`下创建子目录**

- **方案描述：** 物理上将`bindingdb.yaml`, `brenda.yaml`等文件放入不同的子目录，然后在主配置文件中通过`defaults`列表来组合它们。
- **优点：** 实现了文件的物理隔离。
- **缺点：**
  - **未解决核心问题：** 即使文件被分开了，它们最终仍然会被 Hydra 合并到同一个扁平的`DataParamsConfig`对象中。代码层面的命名空间问题依然存在。
  - **增加了配置的复杂性：** `defaults`列表会变得更长，更难以管理。
- **决策理由：** 该方案只解决了表面问题，未能从根本上改善代码和配置的结构，予以否决。

#### **后果 (Consequences)**

**正面影响：**

- **高内聚，低耦合：** 所有与特定`Processor`相关的参数，现在都内聚在其自己的配置块中。`Processor`之间以及`Processor`与通用配置之间实现了完全的解耦。
- **清晰的命名空间：** `config.data_params.bindingdb.affinity_threshold_nM`这样的访问路径清晰、无歧义，极大地增强了代码的可读性和可维护性。
- **极佳的可扩展性：** 当未来添加`DrugBankProcessor`时，只需新增一个`DrugBankParams` dataclass 和对应的`drugbank`字段即可，完全不影响现有代码和配置。
- **更强的组合能力：** 在`YAML`中，我们可以轻松地对整个`bindingdb`或`filtering`块进行整体覆盖或复用，充分发挥了 Hydra 的强大能力。

**负面影响/风险：**

- **一次性的重构成本：** 需要对`DataParamsConfig`的定义、所有`Processor`中获取配置的代码，以及现有的`data_params` `YAML`文件进行一次性的、中等规模的重构。然而，这次重构是为了未来的清晰和可扩展性，是必要的前期投资。

---

**关联的 ADR**

- **补充 (Supplements)**: 本 ADR 是`ADR-009`（引入`SelectorExecutor`）在**配置层面**的一个重要补充，它将“职责分离”的原则从代码架构贯彻到了配置架构中。
