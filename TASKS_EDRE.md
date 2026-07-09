# EDRE 实现任务清单（v1）

Evidence-Driven Research Engine 的可独立认领任务集，按**垂直切片（tracer bullet）**拆分。
每个切片贯穿整张图（`plan → … → finalize`），可在最高缝 `ResearchInput → ResearchOutput` 上独立验证。

**权威来源**：需求以 `PRD.md` 为准，术语以 `CONTEXT.md` 为准，架构取舍以 `docs/adr/0001~0005` 为准。
本文件不重述论证，只做任务编排与状态跟踪。

**测试哲学**（贯穿所有切片）：只测外部行为，不测内部节点中间形状。
在整图边界 `ResearchInput → ResearchOutput` 上断言语义（终止态、各 claim verdict、覆盖、可追溯引用），
把 `search_many` adapter、LLM 节点（plan / normalize / score_claims）、reranker 以可注入的假实现替换，使断言可复现。

---

## 状态跟踪

状态取值：`TODO`（未开始） / `WIP`（进行中） / `BLOCKED`（被依赖阻塞） / `REVIEW`（待评审） / `DONE`（完成并测试通过）。
认领任务：把状态改为 `WIP` 并在 Owner 填名字。完成后改 `DONE`，并勾选该切片下全部验收项。

| # | 切片 | 状态 | Owner | 依赖 | 覆盖的 User Stories |
|---|------|------|-------|------|---------------------|
| A | 预重构：`search_many` 检索 adapter（唯一集成缝） | `DONE` | Claude | 无 | 15, 21 |
| B | 行走骨架：扁平 EDRE 全图 + 数据模型 + 循环与基础终止 | `DONE` | Claude | A | 1, 22, 23 |
| C | Planner：task → 3~6 EvidenceClaim + importance 护栏 | `DONE` | Claude | B | 2, 3, 18 |
| D | Query Generator：只为未解决 claim 换新角度生成 query | `DONE` | Claude | B | 4, 19 |
| E | 第一层门控：本地 cross-encoder reranker（Task/Query Match） | `DONE` | Claude | B | 5, 20 |
| F | 文档规范化：逐文档蒸馏为带引用/冲突标注的事实片段 | `DONE` | Claude | E | 24 |
| G | 第二层评估：有符号 support ∈ [-1,1] + 证据更新 + 证伪 | `DONE` | Claude | B | 6, 7, 8, 25 |
| H | 终止诚实性：ABANDONED / DONE vs EXHAUSTED / 不拖垮 | `DONE` | Claude | D, G | 9, 10, 13, 17 |
| I | 确定性输出快照 + 全量配置 + 可观测 + 真实 E2E | `DONE` | Claude | C, F, H | 11, 12, 14, 16 |

**依赖关系图**

```
A ──▶ B ──┬──▶ C ─────────────────────┐
          ├──▶ D ──┐                   │
          ├──▶ E ──┼──▶ F ─────────────┼──▶ I
          └──▶ G ──┴──▶ H ─────────────┘
```

A、B 为串行地基；C / D / E / G 在 B 之后可并行；F 在 E 之后；H 在 D+G 之后；I 收口在 C+F+H 之后。

---

## Slice A — 预重构：`search_many` 检索 adapter

**状态**：`DONE` ・ **依赖**：无 ・ **User Stories**：15, 21

### 要构建什么

把检索侧重构出一个纯函数 adapter，作为 EDRE 与检索层之间**唯一**的缝。
adapter 输入一组 query，返回**带 query 归属、已按 URL 去重**的文档集合；provider 故障转移内置于内部，不上升为编排决策。
复用现有 `fan_out_search` 的并发 / retry / provider registry / tracing 与 `_merge_responses` 的去重与 citation 编号能力；
把去重逻辑从 `consolidate` 节点拆出并入 adapter，弃用其"把一个 query 的所有结果合并成单一 answer 字符串"的输出形状。
一篇文档可记录多个命中它的 query（保留归属），以支撑后续 SearchRound 与"每个 Evidence 可追溯"。

现有 `build_search_subgraph` 完整子图**保持不变**，供其它调用方独立使用；EDRE 不走它。
这是"先让改动变简单，再做简单改动"的预重构，必须最先完成。

### 验收标准

- [x] 存在纯函数 adapter：输入多个 query，输出带 query 归属、已 URL 去重的文档集合，无 LLM answer 合成。
- [x] 去重跨 fan-out 路径生效；无 URL 的结果始终保留；citation 编号连续。
- [x] 同一文档被多个 query 命中时，其 query 归属完整保留（可追溯到全部命中 query）。
- [x] provider 故障转移在 adapter 内部完成，单 provider 故障不抛给上层。
- [x] 单路 query 失败被跳过而非致命，幸存路径仍产出文档。
- [x] adapter 自身的第二测试缝有覆盖（去重 / query 归属 / 故障转移），不依赖网络。
- [x] 现有 subgraph 测试全绿，`build_search_subgraph` 行为不变。

### 依赖

无 —— 可立即开始。

---

## Slice B — 行走骨架：扁平 EDRE 全图 + 数据模型

**状态**：`DONE` ・ **依赖**：A ・ **User Stories**：1, 22, 23

### 要构建什么

搭起 EDRE 的"行走骨架"：一张单层扁平 LangGraph，把全部节点按顺序接通并跑通循环与终止，产出一个真实的 `ResearchOutput`。
本切片中各智能节点只做**最小可注入的假实现/直通逻辑**（真正的智能在 C~G 逐个加深），但**脊柱是真的**：claim 集固定、循环回边、终止态计算、确定性输出拼装。

数据模型（全部落地）：
- `EvidenceClaim`：`id, hypothesis, importance(CRITICAL|OPTIONAL), confidence∈[-1,1], supporting_documents(带有符号 support), search_attempts`；`status` 为派生只读（`NOT_STARTED / PARTIAL / VERIFIED / REFUTED / ABANDONED`），不独立存储。
- `SearchRound`：`loop_index, queries, documents(带 source_query), rerank_scores, support_scores, provider, duration, errors`。
- `ResearchState`：`task, evidence_plan, search_history, loop_count, terminal(None|DONE|EXHAUSTED), config`；仅在内存。
- `EDREConfig`：注入式配置容器（本切片先放最小必需项，I 切片补全）。
- 接口契约：输入 `ResearchInput { task: str }`；输出 `ResearchOutput { research_summary, evidence, citations, loop_count }`。

图拓扑（扁平、无嵌套子图）：
`START → plan → generate_queries → search → rerank → normalize → score_claims → update_evidence → control`，
`control` 条件路由到 `generate_queries`（CONTINUE）或 `finalize`（DONE/EXHAUSTED），`finalize → END`。
`search` 节点即调用 Slice A 的 `search_many`（普通节点，非嵌套子图）。
循环回边由 `max_loops`（= LangGraph `recursion_limit`）兜底。全图全异步。

确立**最高单一测试缝**：`ResearchInput → ResearchOutput`，把 `search_many` 打桩为返回受控文档集，plan/score/reranker 用可注入假实现。

### 验收标准

- [x] 提交 `ResearchInput { task }` 能驱动全图跑通并返回结构完整的 `ResearchOutput`。
- [x] 全部节点按 PRD 拓扑接通；`search` 节点通过 `search_many` adapter 取数（非完整子图）。
- [x] claim 集在 plan 后固定，循环中不增删。
- [x] `EvidenceClaim.status` 完全由 `confidence + search_attempts` 派生，不独立存储。
- [x] `control` 能计算并区分 `DONE` 与 `EXHAUSTED` 两种终止态，并据此路由。
- [x] `recursion_limit` 作为 max_loops 安全网生效，循环不会无限。
- [x] 图为全异步，检索/评估可并发驱动。
- [x] 最高缝测试就位：注入 fake `search_many` + fake LLM/reranker，可复现地断言 `ResearchOutput`。
- [x] `finalize` 为纯拼装、无 LLM。

### 依赖

- Slice A（`search` 节点依赖 `search_many` adapter）。

---

## Slice C — Planner：task → EvidenceClaim 集

**状态**：`DONE` ・ **依赖**：B ・ **User Stories**：2, 3, 18

### 要构建什么

把骨架里直通的 plan 节点换成真正的 Planner：一次性 LLM 调用，把 task 拆成 **3~6 个可判真伪的 EvidenceClaim**（是 hypothesis，不是 Topic / 搜索词），每个标 `importance = CRITICAL | OPTIONAL`。
护栏：至少 1 个 critical 且 critical 应为少数；claim 总数被约束在配置的下限(3)/上限(6)。
Planner 作为可替换组件，替换它不影响下游节点（便于实验不同拆解策略）。

### 验收标准

- [x] 给定 task，Planner 产出 3~6 个 EvidenceClaim，每个是可判真伪的 hypothesis 且带 importance。
- [x] 护栏生效：至少 `min_critical`(默认 1) 个 critical，critical 为少数；总数越界时被夹到 [下限, 上限]。
- [x] Planner 可替换：注入假 Planner LLM 即可在最高缝断言 claim 集属性（数量、importance 约束）。
- [x] claim 集产出后固定，被后续循环消费，不再变更。

### 依赖

- Slice B（骨架图与数据模型）。

---

## Slice D — Query Generator：只为未解决 claim 生成 query

**状态**：`DONE` ・ **依赖**：B ・ **User Stories**：4, 19

### 要构建什么

把骨架里直通的 generate_queries 换成真正的 Query Generator：只为**未解决**（`NOT_STARTED / PARTIAL`）的 claim 生成查询，每个 claim 生成 `queries_per_claim`（默认 2~3）条；
v1 只做"换新角度重试"策略（每轮换新角度，不细分改写 vs 换新）。已解决（VERIFIED / REFUTED）或已放弃（ABANDONED）的 claim 不再生成 query，不浪费预算。
Query Generator 作为可替换组件，便于独立优化检索策略。

### 验收标准

- [x] 只为 `NOT_STARTED / PARTIAL` 的 claim 生成 query；已解决/已放弃的 claim 不再产出 query。
- [x] 每个未解决 claim 生成约 `queries_per_claim` 条查询，随轮次换新角度。
- [x] 在最高缝可观测：已解决 claim 不再触发新检索（预算不浪费在已搞定的部分）。
- [x] Query Generator 可替换，替换不影响上下游。

### 依赖

- Slice B（骨架图与 claim status 派生）。

---

## Slice E — 第一层门控：本地 cross-encoder reranker

**状态**：`DONE` ・ **依赖**：B ・ **User Stories**：5, 20

### 要构建什么

把骨架里直通的 rerank 换成真正的第一层评估门控：本地 cross-encoder 对每篇文档打 **Task Match**（文档↔原始 task）与 **Query Match**（文档↔当前 query），
低于门控阈值的文档直接丢弃（Precision First 第一道闸）。Task Match 与 Query Match 复用同一模型，仅左侧输入不同；粗筛用原始文本，不依赖规范化。
封装为可替换的 `Reranker` 协议 `rerank(left_text, docs) → scores`（本地/托管可换，模型更换不影响上层）。
门控权重 `w1(TaskMatch) / w2(QueryMatch)` 可配置。

### 验收标准

- [x] 存在 `Reranker` 协议 `rerank(left_text, docs) → scores`；Task Match 与 Query Match 复用同一模型仅换左侧输入。
- [x] 低于门控阈值的文档在进入下游评估前被丢弃；仅幸存文档继续。
- [x] 在最高缝可观测：明显跑题的文档被第一层挡下，不进入证据。
- [x] reranker 可替换：注入假打分器即可复现地断言门控行为。
- [x] 门控用原始文本，不依赖规范化节点的输出。

### 依赖

- Slice B（骨架图）。

---

## Slice F — 文档规范化：逐文档蒸馏

**状态**：`DONE` ・ **依赖**：E ・ **User Stories**：24

### 要构建什么

把骨架里直通的 normalize 换成真正的文档规范化：**仅对 reranker 幸存文档**，逐篇蒸馏为 dense、保留 citation、带**冲突标注**的事实片段，作为下游 claim 打分的 grounding。
改造复用现有 `AnswerConsolidator` 的 LLM 规范化 prompt（dense factual + 保留 citation + 冲突标注），但粒度从"对一个 query 的合并整体"改为"**逐文档**"。
冲突标注天然对接下游有符号 support / 证伪。

### 验收标准

- [x] 只对 reranker 幸存文档运行，逐篇产出 dense 事实片段，成本有界。
- [x] 规范化片段保留 citation 归属，并带冲突标注。
- [x] 复用/改造自 consolidate 的规范化能力，但为逐文档粒度（不产 query 合并式 answer）。
- [x] 规范化结果作为 grounding 传给 score_claims；注入假 LLM 可复现地断言。

### 依赖

- Slice E（规范化只跑第一层门控的幸存文档）。

---

## Slice G — 第二层评估：有符号 support 与证伪

**状态**：`DONE` ・ **依赖**：B ・ **User Stories**：6, 7, 8, 25

### 要构建什么

把骨架里直通的 score_claims + update_evidence 换成真正的第二层评估：
对每篇幸存文档发**一次** LLM 调用，一次性返回它对**当前全部 claim** 的**有符号 support ∈ [-1,1]**（正=支持，负=反驳/证伪）。
`update_evidence` 为每个被命中的 claim：更新 `confidence`（取该 claim 所有证据中**绝对值最大**的有符号 support，v1 纯 max、不做多来源累积）、
累加 `search_attempts`、把命中文档连同其有符号 support 记入 `supporting_documents`。
据此 confidence 的符号驱动 claim 走向 `VERIFIED`（confidence ≥ verify_threshold）或 `REFUTED`（confidence ≤ −refute_threshold）。
证伪是一个**成功的发现**，不得塌成"未验证"。

### 验收标准

- [x] 每篇幸存文档一次 LLM 调用，返回对全部 claim 的有符号 support 向量 ∈ [-1,1]。
- [x] `confidence` = 该 claim 所有证据中绝对值最大的有符号 support；纯 max，无多来源加成。
- [x] 强正 support 使关键 claim 走向 `VERIFIED`；强负 support 使其走向 `REFUTED`。
- [x] 每条证据记录其有符号 support，`supporting_documents` 可追溯到具体文档。
- [x] 在最高缝可观测："支持 / 反驳 / 没查到"三态清晰可分（REFUTED ≠ ABANDONED）。
- [x] 注入假打分器可复现地断言 verdict 走向。

### 依赖

- Slice B（骨架图与 confidence/status 派生脊柱）。

---

## Slice H — 终止诚实性与每 claim 收敛

**状态**：`DONE` ・ **依赖**：D, G ・ **User Stories**：9, 10, 13, 17

### 要构建什么

加深薄 Controller 的终止语义与每 claim 预算收敛，确保系统**绝不把"没搞定"伪装成"完成"**：
- claim「已解决」= `VERIFIED` 或 `REFUTED`；「未解决」= `ABANDONED`。
- 单条 claim 在 `search_attempts ≥ max_attempts` 且既未 VERIFIED 也未 REFUTED 时派生为 `ABANDONED`（不新增字段），不再为它检索，且不拖垮其余 claim。
- **DONE** = 所有 critical claim 已解决（VERIFIED 或 REFUTED）。
- **EXHAUSTED** = 撞 `max_loops` 或无进展退出，但仍有 critical claim 未解决。
- 每个终止决策可解释（基于哪些 claim 的状态得出）。

Controller 仍是薄的：决策空间 = `{CONTINUE, 自动终止}`，不做 LLM 决策，不做 REPLAN / CHANGE_PROVIDER。

### 验收标准

- [x] 全部 critical 被验证 → `DONE`。
- [x] 关键前提被证伪 → 仍 `DONE` 且该 claim 标 `REFUTED`。
- [x] 关键 claim 查不到 → `EXHAUSTED` 且该 claim 如实标 `ABANDONED`；绝不伪装成 DONE。
- [x] 单条查不到的 claim 在 `max_attempts` 后被放弃，不再检索，也不拖垮其余 claim 的收敛。
- [x] 无进展或撞 `max_loops` 时退出为 `EXHAUSTED`（仍有 critical 未解决时）。
- [x] 终止决策可解释：可追溯到是哪些 claim 的状态触发了该终止态（`research_summary.blocking_claim_ids` 命名触发终止的未解决 critical claim）。

### 依赖

- Slice D（未解决判定驱动 generate_queries，支撑"不再检索/不拖垮"）。
- Slice G（真实 verdict 驱动 DONE/REFUTED/ABANDONED 判定）。

---

## Slice I — 确定性输出、全量配置、可观测与真实 E2E

**状态**：`DONE` ・ **依赖**：C, F, H ・ **User Stories**：11, 12, 14, 16

### 要构建什么

收口输出、配置与可观测，交付可用的 v1：
- `finalize` 产出**确定性结果快照（无 LLM）**：终止态、claim verdict 计数（VERIFIED / REFUTED / ABANDONED）、critical 是否全解决、coverage（已解决/总数，纯展示）、`loop_count`；
  `evidence` 为每 claim 的 verdict + 支持/反驳文档；`citations` 引用列表；`loop_count`。不含成文叙述答案、不含 controller_trace。
- 证据排序用 `doc_relevance = w1·TaskMatch + w2·QueryMatch + w3·max|support|`。
- 每轮记一条 `SearchRound`，支撑"每个 Evidence 可追溯"。
- 全量 `EDREConfig` 接入并可注入：`claim 数下限(3)/上限(6)`、`min_critical(1)`、`verify_threshold`、`refute_threshold`、`rerank 门控阈值`、门控权重 `w1/w2`、`max_attempts`、`max_loops`、`queries_per_claim`、`doc_relevance` 权重 `w1/w2/w3`。
- 复用现有 Langfuse 回调实现逐节点、逐 LLM 生成的追踪。
- 提供真实 E2E 冒烟脚本（参照 `e2e_smoke.py`），跑通一次真实研究。

### 验收标准

- [x] `research_summary` 为确定性快照（无 LLM），计数与终止态一致。
- [x] 输出区分 VERIFIED / REFUTED / ABANDONED 三类 verdict，REFUTED 与 ABANDONED 清晰可分。
- [x] 每条结论可追溯到具体文档引用（citations + supporting_documents）。
- [x] 全部配置项进 `EDREConfig` 并可注入，可在精度/深度/成本间调节（阈值、尝试上限、最大轮数、claim 数量等）。
- [x] 每轮、每节点、每次 LLM 生成可在 Langfuse 追踪。
- [x] 真实 E2E 脚本可跑通一次完整研究并打印结果快照与引用（`edre_e2e_smoke.py`）。
- [x] `doc_relevance` 排序按 `w1·TaskMatch + w2·QueryMatch + w3·max|support|` 生效。

### 依赖

- Slice C（真实 claim 集）、Slice F（规范化 grounding）、Slice H（终止诚实性）。

---

## 明确的非目标（Out of Scope，v1 不做）

- Answer Synthesis / 成文叙述答案与研究报告（R7）；v1 的 `research_summary` 仅为确定性快照。
- REPLAN 与循环中可变的 claim 集。
- CHANGE_PROVIDER 作为编排层决策（v1 仅在 adapter 内做故障转移）。
- 状态可恢复 / checkpointer / 崩溃续跑（ResearchState 仅在内存）。
- 多来源累积式 confidence 与同一 claim 内证据冲突的特殊处理（v1 取绝对值最大 support）。
- Query 改写 vs 换新的区分（v1 仅换新角度重试）。
- controller_trace 富决策轨迹。
- Search Provider 实现、Web Crawling、Embedding Index、Long-term Memory、Citation 渲染、UI。
