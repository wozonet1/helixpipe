# HelixPipe 数据引擎正确性审计

审计日期: 2026-05-11
审计范围: `src/helixpipe/data_processing/services/` 全部核心服务模块
审计目标: 检查数据引擎部分的正确性与准确性（不含训练模块）

---

## 一、逻辑 Bug（会导致错误结果）

### BUG-01: `InteractionStore.difference()` — 去重键包含内部列导致差集泄漏 ✅ 已修复

**文件**: `interaction_store.py:204-236`

**现状**:
```python
current_set = set(self._df.itertuples(index=False, name=None))
other_set = set(other._df.itertuples(index=False, name=None))
difference_set = current_set - other_set
```

**问题**: `itertuples` 将每行的**所有列**纳入元组比较，包括 `source_dataset` 内部元数据列。`difference()` 被调用时，`other`（evaluable_store）通过 `query()` → `_from_dataframe()` 创建，其 `_df` 保留了 `source_dataset` 列。同一条交互如果来源不同（BindingDB vs GtopDB），元组就不等，差集结果就多出了本应被减掉的行。

**影响**: `_presplit_by_evaluation_scope()` 中 `self.store.difference(evaluable_store)` 计算的 background_store 可能包含本应属于 evaluable 的交互，导致数据泄漏到训练集中。

**严重度**: 高

**修复** (2026-05-11): 将 `source_dataset` 从 DataFrame 中移出，改为独立的 `_source_tags` 字典。`difference()` 改为仅基于 canonical 列 (source_id, target_id, relation_type) 构建匹配键，使用布尔索引而非元组集合运算。

**修订** (2026-05-13): 将 `source_dataset` 改回 DataFrame 列，但作为正式的 schema 字段（`CanonicalInteractionSchema.source_dataset`），由 `BaseProcessor._finalize_columns()` 在输出时盖章。`difference()` 仅基于 canonical 三列比较，不纳入 `source_dataset`，因此列的存在不影响差集正确性。同时删除了 `_source_tags` 字典及其所有维护代码（`_rebuild_source_tags`、`_filter_source_tags`、`get_source_for_interaction`）。

---

### BUG-02: `InteractionStore._from_dataframe()` — 缺失 canonicalize

**文件**: `interaction_store.py:121-129`

**现状**:
```python
@classmethod
def _from_dataframe(cls, df, config):
    store = cls.__new__(cls)
    store._df = df  # 直接赋值，不做 canonicalize
    return store
```

**问题**: 构造函数执行了 `_canonicalize_interactions()`，但工厂方法 `_from_dataframe` 跳过了这个步骤。经过 `filter_by_entities()`、`query()`、`sample()` 返回的新 store 中，source/target 的排序可能已被用户数据打乱（非规范化），但没有重新 canonicalize。

**影响**: 虽然 `_from_dataframe` 主要用于已规范化的子集，但如果上游数据有方向不一致的交互进入，后续依赖 canonicalize 顺序的逻辑（如 `global_positive_set` 的构建）可能出错。

**严重度**: 中（当前流程中子集操作不会引入新的方向问题，但 concat 会）

---

### BUG-03: `InteractionStore.concat()` — 缺失 canonicalize 和去重

**文件**: `interaction_store.py:131-161`

**现状**:
```python
concatenated_df = pd.concat(dataframes_to_concat, ignore_index=True)
return cls._from_dataframe(concatenated_df, config)
```

**问题**: 多个 store 的 DataFrame 被 concat 后可能产生重复行或 source/target 排序不一致的行，`_from_dataframe` 不做任何规范化或去重处理。

**影响**: 在 `_split_data()` 中，`train_graph_stores_to_concat` 通过 `InteractionStore.concat()` 合并 train_labels 和 background，如果两者有重复交互（尤其来自不同 source 的同一交互），合并后会有重复边，图构建时会添加重复边。

**严重度**: 中

---

### BUG-04: `GraphBuildContext` 蛋白质嵌入索引偏移风险

**文件**: `graph_context.py:86-89`

**现状**:
```python
prot_indices_0_based = torch.tensor(
    [gid - global_id_mapper.num_molecules for gid in sorted_relevant_prots],
    dtype=torch.long,
)
self.local_prot_embeddings = global_prot_embeddings[prot_indices_0_based]
```

**问题**: 假设蛋白质的全局 LogicID 从 `num_molecules` 开始连续排列。但这依赖于 `IDMapper.finalize_with_valid_entities()` 按 entity type 的 `priority` 依次分配 ID。如果未来 `knowledge_graph.entity_meta` 配置中加入 disease 等类型且优先级插在 molecule 和 protein 之间，蛋白质的起始 LogicID 就不再是 `num_molecules`，索引计算会偏移。

**影响**: 当前配置只有 molecule 和 protein 两种 metatype 不会触发，但扩展实体类型时会导致特征向量与节点错位。

**严重度**: 中（当前安全，扩展时必错）

---

### BUG-05: 负采样全局正样本集合的方向性问题

**文件**: `label_generator.py:223-227`

**现状**:
```python
if (source_id, target_id) not in self.global_positive_set and \
   (target_id, source_id) not in self.global_positive_set:
    negative_pairs_auth.append((source_id, target_id))
```

**问题**: `global_positive_set` 由 `sampled_store` 构建，经过 canonicalize 后每条交互都是 source priority <= target priority 排列（如 `(drug, protein)`）。负采样时 source 来自 molecule pool、target 来自 protein pool，生成的也是 `(drug, protein)` 方向，所以反向检查永远不会命中。这不是 bug，但当框架扩展到蛋白质-蛋白质等同质交互场景时，反向检查会漏掉同质正样本，导致生成假负样本。

**影响**: 当前无实际问题，但扩展时需要重写碰撞检查逻辑。

**严重度**: 低（当前安全）

---

### BUG-06: CID float/int 类型漂移

**文件**: `id_mapper.py:64-80` 和 `interaction_store.py:93-96`

**现状**: Pandas 从 TSV 读取数据时，整数列可能被推断为 `float64`（如有缺失值）。`InteractionStore._canonicalize_interactions()` 中 `df[s_id_col].astype(str)` 会把 `2244` 和 `2244.0` 变成不同的字符串，影响规范化排序。`IDMapper.finalize_with_valid_entities` 中 `sorted(self.entities_by_type[entity_type])` 在类型混合时会抛 `TypeError`。

**影响**: 可能导致交互方向的规范化结果不一致，或在 IDMapper finalize 阶段崩溃。

**严重度**: 高

---

## 二、数据精度 / 静默错误问题

### ISSUE-07: `source_dataset` 列泄漏到整个流水线 ✅ 已修复

**文件**: `interaction_store.py:47`, `base_processor.py:218`

**现状**:
```python
df_copy["__source_dataset__"] = source_dataset
```

**问题**: `source_dataset` 在聚合时被添加到每条交互上，此后再也没有被移除。它会在 `difference()` 的元组比较中造成错误（BUG-01），也会在 `concat()` 中传播。`get_mapped_positive_pairs()` 只取三列所以不受影响，但任何直接操作 `dataframe` 属性的代码都会看到这个多余的列。

**影响**: 数据泄漏 + BUG-01 的根因。

**严重度**: 中（与 BUG-01 联动则为高）

**修复** (2026-05-11): 不再将 `source_dataset` 作为 DataFrame 列，改为存储在独立的 `_source_tags: dict[tuple, str]` 字典中。关键字为 `(str(source_id), str(target_id), relation_type)` 三元组。`difference()` 改为只基于 canonical 列比较。新增 `_rebuild_source_tags()` 处理规范化后的 key 重建，`_filter_source_tags()` 处理子集操作后的 tags 过滤，`get_source_for_interaction()` 提供公共查询接口。

**修订** (2026-05-13): 将 `source_dataset` 改回 DataFrame 列，升级为 `CanonicalInteractionSchema` 的正式字段。由 `BaseProcessor._finalize_columns()` 在输出阶段自动盖章（`df[schema.source_dataset] = self.source_name`），不再由 `InteractionStore.__init__` 手动添加。`difference()` 仅基于 canonical 三列比较，列的存在不影响正确性。删除了 `_source_tags` 字典及所有维护代码，`InteractionStore.__init__` 简化为纯 `pd.concat`。列名从 `__source_dataset__` 改为 `source_dataset`（去除双下划线）。

---

### ISSUE-08: `_validate_molecules` 静默丢弃计算失败的行

**文件**: `entity_validator.py:224-244`

**现状**: `calculate_molecular_properties` 内部过滤掉了 MW/LogP 为 NaN 的行，这些行既不进入 `props_df` 也不进入 `criteria_df`，最终在 `final_mask` 中默认为 `False`。没有任何日志记录有多少分子因计算失败被丢弃。

**影响**: 用户无法知道有多少分子因 RDKit 解析失败而被静默过滤掉，难以排查数据质量问题。

**严重度**: 中

---

### ISSUE-09: 蛋白质序列验证过于宽松

**文件**: `purifiers.py:66`

**现状**:
```python
VALID_SEQ_CHARS = "ACDEFGHIKLMNOPQRSTUVWXYZ"
```

**问题**: 包含了整个大写字母表 A-Z，任何大写字母串（如 `"ZZZZZ"`）都会通过验证，失去了过滤作用。

**影响**: 几乎不起到蛋白序列质量校验的作用。

**严重度**: 低

---

## 三、配置安全 / 副作用问题

### ISSUE-10: `DataSplitter` 直接修改传入的配置对象

**文件**: `splitter.py:95` 和 `splitter.py:124`

**现状**:
```python
self.coldstart_cfg.pool_scope.meta_types = ["molecule"]
# 以及
self.coldstart_cfg.evaluation_scope = InteractionSelectorConfig(...)
```

**问题**: 直接修改了 Hydra/OmegaConf 的结构化配置对象。如果 `DataSplitter` 被多次实例化（如 multirun/sweep 模式），这些修改会累积而不是每次重新初始化。

**影响**: 在 Hydra sweep 场景下，第一次实验的配置修改会影响后续所有实验。

**严重度**: 高

---

## 三-B、已排除项（审查时发现但确认非 Bug）

### EXCLUDED-01: `DataSplitter.__next__` 中 fold_idx 与 sklearn KFold 编号不同步

**文件**: `splitter.py:364-407`

**发现**: `fold_idx` 从 1 开始，但 sklearn 的 `KFold.split()` 产生的是 0-based 的 numpy index 数组。两者用途不同：`fold_idx` 仅用于文件命名和日志，sklearn index 用于实际数据切分。

**结论**: 不构成 bug。`_split_data()` 中冷启动分支正确区分了 `num_folds > 1` 和 `num_folds <= 1` 两条路径，`split_result` 与 `fold_idx` 各司其职。

### EXCLUDED-02: `_apply_stratified_sampling` 中不同 stratum 共享同一个 `rng`

**文件**: `interaction_store.py:367`

**发现**: `apply_sampling_strategy` 将同一个 `rng` 传入 `_apply_stratified_sampling`，不同 stratum 的 `.sample(random_state=rng)` 调用共享 PRNG 状态，使得各层采样统计相关。

**结论**: 不构成 bug。分层采样中各层共享 PRNG 状态是常见做法，且代码中 `rng` 本身就来自 `np.random.default_rng(seed)`——如果需要完全独立随机性，可以为每层创建独立 `rng`，但当前行为不影响正确性。

---

## 四、健壮性 / 设计问题

### ISSUE-11: `IDMapper.get_meta_by_auth_id()` 返回内部可变状态的引用

**文件**: `id_mapper.py:272-277`

**现状**:
```python
return self._final_entity_map.get(auth_id)
```

**问题**: 返回的是内部字典 `{type, sources: set(...)}` 的直接引用，调用者可以意外修改 IDMapper 的内部状态。

**影响**: 理论上的封装被打破，可能导致难以追踪的 bug。

**严重度**: 低

---

### ISSUE-12: `canonicalize_smiles` 函数返回类型标注错误

**文件**: `canonicalizer.py:18`

**现状**:
```python
def canonicalize_smiles(smiles: SMILES) -> None:
```

**问题**: 返回类型标注为 `None`，但实际返回 `str | None`。类型检查器会误报。

**影响**: 对调用者类型推断有误导。

**严重度**: 低

---

### ISSUE-13: `entity_validator.py` 硬编码的数据源名称映射

**文件**: `entity_validator.py:62-73`

**现状**:
```python
source_configs = {
    "bindingdb": ...,
    "brenda": ...,
    "gtopdb": ...,
}
```

**问题**: 硬编码了数据源名称，与项目的"插件化数据源接入"设计哲学矛盾。新增 Processor 的过滤配置不会被自动纳入动态放宽逻辑。

**影响**: 新数据源的实体校验只能使用全局默认配置，无法享受来源感知的放宽策略。

**严重度**: 中

---

### ISSUE-14: `InteractionStore.difference()` 性能问题

**文件**: `interaction_store.py:225-226`

**现状**:
```python
current_set = set(self._df.itertuples(index=False, name=None))
other_set = set(other._df.itertuples(index=False, name=None))
```

**问题**: 对大型数据集（几百万行），将整个 DataFrame 转为元组集合的内存和计算开销极大。

**影响**: 大规模数据下性能显著下降。

**严重度**: 低

---

### ISSUE-15: `InteractionStore` 的不可变性保证薄弱

**问题**: 文档注释说遵循不可变模式，但 `_from_dataframe` 可以接受任意 DataFrame，且 `dataframe` 属性返回 `.copy()` 并不能阻止对 `_df` 的直接内部修改。外部调用者如果绕过公共 API 直接操作 `_df`，不可变性就被打破。

**严重度**: 低

---

## 优先修复建议

1. **BUG-01 + ISSUE-07** (高): ✅ 已修复 (2026-05-11, 修订 2026-05-13)。`difference()` 仅基于 canonical 三列比较。`source_dataset` 升级为 `CanonicalInteractionSchema` 正式字段，由 `BaseProcessor` 在输出时盖章，`InteractionStore` 做纯 `pd.concat`。

2. **BUG-06** (高): 在 `Processor` 输出阶段或 `InteractionStore.__init__` 阶段，对 CID 列强制转为 `int` 类型。在 `_canonicalize_interactions` 中也做类型规范化。

3. **ISSUE-10** (高): 使用 `copy.deepcopy()` 或 OmegaConf 的 `OmegaConf.merge()` 创建 coldstart_cfg 的本地副本，避免修改原始配置。

4. **BUG-02 + BUG-03** (中): 在 `concat()` 中增加去重和 canonicalize 步骤，或在所有返回新 store 的方法中保证规范化。

5. **BUG-04** (中): 将蛋白质嵌入索引计算改为基于 `id_mapper.auth_id_to_logic_id_map` 做查找，而非依赖整数偏移假设。

6. **ISSUE-08** (中): 在 `_validate_molecules` 中记录被 `calculate_molecular_properties` 静默丢弃的行数。