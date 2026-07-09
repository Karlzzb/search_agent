# EDRE 复用 search_agent 的边界：检索 adapter，而非完整子图

## 背景

现有 `search_agent` 是一个 LangGraph 子图 `decompose → fan_out_search → consolidate`，对外契约为 `{"task"} → {"consolidated", "citations"}`。
EDRE 需要在其之上做 Evidence 驱动的研究编排，但 PRD 的 Principle 2（Planner/QueryGenerator 与检索解耦）和"不做 Answer Synthesis"要求 EDRE 拿到的是去重后的原始文档，而不是被 LLM 消化过的 answer。

## 决定

EDRE 不调用完整子图。
把检索侧重构出一个纯函数 adapter `search_many(queries) → 带 query 归属、已 URL 去重的文档集合`，内部复用现有 `fan_out_search`（并发/retry/provider registry/tracing）与 `_merge_responses`（去重与 citation 编号）。
- `decompose`：由 EDRE 的 Query Generator 顶替。
- `consolidate`：**不使用**其"把一个 query 的所有结果合并成单一 answer 字符串"的输出形状（会破坏逐文档结构与 citation→文档映射，与 EDRE 的逐 (文档,claim) 打分和可追溯冲突）；但其 LLM 规范化能力（`use_llm=True` 的 prompt：dense factual + 保留 citation + 冲突标注，显式面向"grounding context for another LLM"）被**改造粒度后复用**——搬进 EDRE 一个**独立的逐文档规范化节点**，蒸馏每篇幸存文档为 dense 事实片段，喂给下游 claim 打分。冲突标注天然对接有符号 support/证伪（见 ADR-0005）。

现有 `build_search_subgraph` 完整子图作为 search_agent 的独立用法保留，EDRE 不走它。

## 权衡

- 接完整子图（A）：改动最小，但造成 Query Generator 与 decompose 双重拆解，且被迫吃下不想要的 answer。
- 接 provider 层（B）：干净但丢掉 fan-out 并发与 URL 去重的复用。
- 抽 adapter（本决定，C）：唯一同时满足"复用底层"与"解耦"。

## 影响

去重逻辑当前与 `consolidate` 的 answer 合成耦合在同一节点，需拆分：去重（`_merge_responses`）进 adapter，规范化能力抽成独立节点，query 合并式 answer 输出弃用。
adapter 返回结构保留 query 归属（一篇文档可记多个命中 query），以支撑 SearchRound 记录与"每个 Evidence 可追溯"的非功能要求。
规范化节点只跑 reranker 幸存文档，成本有界；reranker 粗筛用原始文本即可，不依赖规范化。
