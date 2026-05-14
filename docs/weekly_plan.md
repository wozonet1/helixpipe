# HelixPipe 数据引擎一周工作计划

制定日期: 2026-05-11
修订日期: 2026-05-12
目标: 最小化修复阻塞性 bug + 接入 PPI 数据 + 输出可解释性模型所需数据
前提: 项目为短期科研工具，完成当前课题后弃置，不追求长期可维护性，不重构架构

---

## 修订说明

原计划 5 天修 6 个 bug + 加 PPI + 加 provenance。修订后压缩 bug 修复范围至**仅修阻塞管线跑通的 2 个 bug**，其余 bug 视为"跑通后再看"。多留一天缓冲。

核心原则：**这 5 天是修管道+接水管，不是装修房子。代码能跑出正确结果就行。**

---

## 背景与约束

- 引用 `docs/bug_audit.md` 中的审计结论作为 bug 修复依据
- PPI 数据源为 STRING 数据库，`StringProcessor` 已编写完成但被注释
- 下游模型为可解释性模型，需根据知识图谱中的边与通路解释链接预测结果
- 当前图输出缺少边级 provenance（`source_dataset`）和置信度分数

### Bug 修复优先级重排

| Bug/Issue | 严重度 | 是否阻塞管线 | 修订后处理 |
|-----------|--------|-------------|-----------|
| BUG-01 + ISSUE-07 (difference 数据泄漏) | 高 | — | **已修复**，跳过 |
| BUG-06 (CID float/int 漂移) | 高 | **是** — 会导致 finalize 崩溃 | **必修** |
| ISSUE-10 (DataSplitter 配置突变) | 高 | 仅 multirun | 单次实验不触发，**跳过** |
| BUG-02 (_from_dataframe 缺 canonicalize) | 中 | 看情况 | 先跑一次，出问题再修 |
| BUG-03 (concat 缺去重) | 中 | 看情况 | 先跑一次，重复边对 GNN 影响不大 |
| BUG-04 (蛋白质嵌入索引偏移) | 中 | 当前配置安全 | **跳过** — 只有 molecule + protein |
| BUG-05 (负采样方向性) | 低 | 当前安全 | **跳过** |
| ISSUE-08 (静默丢弃分子) | 中 | 不影响结果 | **跳过** |
| ISSUE-09 (蛋白质序列验证宽松) | 低 | 不影响结果 | **跳过** |

---

## Day 1: 修 BUG-06 + 管线冒烟测试

### BUG-06: CID float/int 类型漂移

**文件**: `interaction_store.py`, `id_mapper.py`, `base_processor.py`

- 在 `InteractionStore.__init__` 的聚合阶段，对 CID 列强制 `pd.to_numeric(..., errors='coerce').astype('Int64')`
- 在 `BaseProcessor._finalize_columns` 中确保 `_smart_clean_id_column` 正确处理（已实现，验证即可）
- 在 `IDMapper.finalize_with_valid_entities` 中增加类型一致性断言
- 预估改动: ~20 行

### 管线冒烟测试

- 用 `tests/fake_data_v3/` 中的假数据跑一次完整管线
- 确认不崩溃、输出文件存在、行数 > 0
- 如果崩了，当场修；如果不崩，Day 2 直接开始 PPI

**Day 1 交付**: 管线能在假数据上跑通，CID 类型问题修复

---

## Day 2: PPI 功能接入

### 启用 StringProcessor

- 取消注释 `src/helixpipe/data_processing/datasets/__init__.py` 中的 `StringProcessor` import
- 创建 `conf/data_params/stringdb/default.yaml` 配置文件
- 在 `conf/dataset_collection/` 的主配置中将 stringdb 注册为辅助数据集

### PPI 规模控制（关键）

- STRING 数据库极大，必须在 `StringProcessor._standardize_ids` 或 `_filter_data` 中加 `combined_score` 阈值过滤
- 建议阈值: `combined_score >= 700`（高置信度交互）
- 如果仍然太大，进一步收严到 900

### 实现 PPI 边入图

**文件**: `graph_builder.py`, `graph_director.py`

- 在 `GraphBuilder` 接口中添加 `add_ppi_edges` 抽象方法
- 在 `HeteroGraphBuilder` 中实现：从 InteractionStore 的 train pairs 中筛选 `relation_type == "physical_association"` 的交互，转化为图边（与 DTI 边处理方式一致）
- 在 `GraphDirector.construct()` 中取消注释 `builder.add_ppi_edges()` 调用（`:95` 行）
- 在 `conf/relations/` 的 flags 配置中启用 `associated_with`

**Day 2 交付**: STRING PPI 数据可通过管线处理并进入图构建

---

## Day 3: 可解释性输出扩展

### 图边列表追加 source_dataset 列

**文件**: `graph_builder.py`

- 修改 `get_graph()` 输出格式，从 3 列 `(source, target, edge_type)` 扩展为 4 列 `(source, target, edge_type, source_dataset)`
- 在 `_edges` 列表中追加第 4 个元素
- 交互边的 source_dataset 从 InteractionStore 的 source_tags 获取
- 相似性边的 source_dataset 标记为 `"computed"`
- PPI 边标记为 `"stringdb"`

### 追加 score 列（可选）

- 交互边的 score 来自原始数据集（BindingDB affinity、STRING combined_score）
- 相似性边的 score 为余弦相似度值
- 如果下游模型需要，在 `_edges` 中追加第 5 个元素
- 与模型端确认是否需要，不需要就不加

### 更新配置

- 更新 `GraphOutputSchema` 以反映新列
- 确保 `nodes.csv` 中 PPI 相关蛋白质的 `sources` 字段正确包含 `stringdb`

**Day 3 交付**: 图输出文件包含边级 provenance 和可选的置信度分数

---

## Day 4: 端到端验证与模型对接

### 假数据验证（先跑）

- 用 `tests/fake_data_v3/` 跑一次包含 PPI 的完整管线
- 检查点:
  - [ ] PPI 边在图输出中正确出现
  - [ ] `source_dataset` 列值正确（`bindingdb`, `gtopdb`, `stringdb`, `computed`）
  - [ ] 无重复边
  - [ ] CID 类型一致（无 float 混入）
  - [ ] 冷启动划分正确（训练集无 evaluable 交互泄漏）

### 真数据验证

- 使用小规模真实数据集跑一次完整管线（DTI + PPI）
- 检查图规模是否在合理范围内（节点 < 50K，边 < 500K 为佳）
- 如果 PPI 导致图暴增，提高 combined_score 阈值

### 模型对接

- 将输出文件提供给可解释性模型端
- 确认模型能正确读取:
  - 节点特征 `node_features.npy`
  - 图边列表（含 source_dataset）
  - 标签文件
  - 节点元数据 `nodes.csv`
- 如有格式不匹配，当天调整

**Day 4 交付**: 全管线端到端验证通过，模型可消费产出数据

---

## Day 5: 缓冲日

**预留用于**:

- 前 4 天任何超时任务的收尾
- 真实数据跑出的意外问题排查
- 可解释性模型端反馈的格式微调
- 如果一切顺利：整理管线运行文档，记录关键配置参数

---

## 风险评估

| 风险 | 概率 | 应对 |
|------|------|------|
| PPI 数据量过大导致内存/时间问题 | 高 | combined_score 阈值从 700 提到 900；必要时只取人类子集 |
| Day 4 端到端跑出格式问题 | 中 | Day 3 结束前用假数据先跑一次，提前暴露 |
| 可解释性模型要求的输出格式差异大 | 低 | Day 2 同步与模型端确认格式需求 |
| BUG-02/03 在真实数据上触发 | 低 | 出问题时现场修，5-10 行改动 |

---

## 明确排除的项

以下项在本次迭代中不做，即使它们是"正确的"：

- 架构简化（砍 GraphDirector、SupervisionFileManager 等）
- 类型系统重构（`NewType` 化）
- `_split_data` 方法拆分重构
- 测试补充
- ADR 更新
- 死代码清理
- SelectorExecutor 空选择器检测优化
- ISSUE-10 (DataSplitter 配置突变) — 单次实验不触发
- BUG-04 (蛋白质嵌入索引偏移) — 当前配置安全
- ISSUE-08 (静默丢弃分子) — 不影响结果
- ISSUE-09 (蛋白质序列验证宽松) — 不影响结果
