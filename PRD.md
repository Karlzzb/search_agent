# EDRE — Evidence-Driven Research Engine (v1)

一个位于 `search_agent` 检索层之上的研究编排引擎。
它围绕用户问题规划一组待验证的知识断言（EvidenceClaim），多轮检索、评估、更新证据，直至所有关键断言被结论性地验证或证伪。

本 PRD 是自洽的：新会话可仅凭本文件 + 仓库内 `CONTEXT.md`（术语表）+ `docs/adr/0001~0005`（架构决策）开始实现。
所有术语以 `CONTEXT.md` 为准；所有架构取舍的理由以对应 ADR 为准，本文件不重复论证。

---

## Problem Statement

作为一个需要高质量答案的使用者，我把问题丢给普通搜索/RAG 时，最终答案经常被低质量或跑题的检索结果污染。
系统"搜到了东西"就直接合成答案，但没有人判断这些证据到底支不支持我真正要回答的问题。
它也不知道自己"够不够"——要么搜一轮就停，要么无脑多搜却始终没对齐我的问题。
更糟的是，当某个关键前提其实是**错的**时，系统只会表现为"没找到"，而不会告诉我"这个前提被证据推翻了"。

## Solution

作为使用者，我提交一个问题，EDRE 先把它拆成若干条**需要被验证的知识断言**（而不是搜索词），并标出哪些是回答问题的关键。
然后它围绕这些断言持续检索：每轮生成查询、检索、先用轻量模型挡掉跑题结果、再用 LLM 判断每篇文档对每条断言是**支持还是反驳**，据此更新每条断言的置信度。
当所有关键断言都被结论性地解决（验证为真或证伪为假）时，它停下并给出一份**确定性的结果快照**，明确告诉我哪些断言成立、哪些被推翻、哪些没查到，以及全部可追溯的引用。
它绝不把"没搞定"伪装成"完成"。

---

## Architecture & Agent Design

### 与 search_agent 的关系（唯一集成缝）

EDRE 不调用现有 `build_search_subgraph` 完整子图。
它通过一个纯函数 adapter `search_many` 复用检索底层能力（fan-out 并发、retry、provider registry、URL 去重、tracing），跳过子图的 `decompose` 与 query 合并式 answer 输出（见 ADR-0001）。
`search_many` 是 EDRE 与检索层之间**唯一**的缝。

### 智能体设计图（扁平 LangGraph，单图无嵌套子图，见 ADR-0004）

```
                          ResearchInput { task }
                                   │
                                   ▼
                            ┌────────────┐
                            │    plan    │  Planner (LLM, 一次性)
                            └────────────┘  task → Evidence Plan: 3~6 个 EvidenceClaim（标 critical/optional）
                                   │
   ┌───────────────────────────────┼──────────  Research Loop  ─────────────────────┐
   │                               ▼                                                 │
   │                      ┌──────────────────┐                                       │
   │                      │ generate_queries │  Query Generator：为未解决 claim 生成 query
   │                      └──────────────────┘                                       │
   │                               ▼                                                 │
   │                      ┌──────────────────┐                                       │
   │                      │      search      │  search_many adapter                  │
   │                      └──────────────────┘  queries → 去重、带 query 归属的文档   │
   │                               ▼                                                 │
   │                      ┌──────────────────┐                                       │
   │                      │      rerank      │  第一层门控：本地 cross-encoder       │
   │                      └──────────────────┘  Task Match + Query Match，粗筛挡跑题  │
   │                               ▼  （仅幸存文档继续）                             │
   │                      ┌──────────────────┐                                       │
   │                      │    normalize     │  文档规范化：逐篇蒸馏为 dense 事实片段 │
   │                      └──────────────────┘  保留 citation + 冲突标注             │
   │                               ▼                                                 │
   │                      ┌──────────────────┐                                       │
   │                      │   score_claims   │  第二层：LLM，逐文档一次调用          │
   │                      └──────────────────┘  输出对全部 claim 的有符号 support∈[-1,1]
   │                               ▼                                                 │
   │                      ┌──────────────────┐                                       │
   │                      │  update_evidence │  confidence=最具决定性 support；      │
   │                      └──────────────────┘  search_attempts++；派生 status       │
   │                               ▼                                                 │
   │                      ┌──────────────────┐                                       │
   │                      │     control      │  薄 Controller：计算终止态           │
   │                      └──────────────────┘                                       │
   │                          │           │                                         │
   │                 CONTINUE │           │ DONE / EXHAUSTED                         │
   └──────────────────────────┘           ▼                                         │
                                    ┌──────────────┐
                                    │   finalize   │  组装 ResearchOutput（确定性，无 LLM）
                                    └──────────────┘
                                           ▼
                                          END
```

### 节点（Nodes）

1. **plan** — Planner。LLM 一次性把 task 拆成 3~6 个 EvidenceClaim，每个标 `importance = critical | optional`。claim 集在整个循环中固定不变（见 ADR-0003）。
2. **generate_queries** — Query Generator。只为未解决（NOT_STARTED / PARTIAL）的 claim 生成查询；v1 采用"换新角度重试"策略。
3. **search** — `search_many` adapter。输入 query 列表，返回已 URL 去重、保留 query 归属的文档集合；provider 故障转移内置于此，不上升为编排决策。
4. **rerank** — 第一层评估门控。本地 cross-encoder 对每篇文档打 Task Match（文档↔原始 task）与 Query Match（文档↔当前 query），低于门控阈值的文档直接丢弃（Precision First 第一道闸，见 ADR-0002）。
5. **normalize** — 文档规范化。仅对幸存文档，逐篇蒸馏为 dense、保留 citation、带冲突标注的事实片段，作为下游 grounding（改造复用 consolidate 的 LLM 规范化能力，见 ADR-0001）。
6. **score_claims** — 第二层评估。对每篇幸存文档发一次 LLM 调用，一次性返回它对**当前全部 claim** 的有符号 support ∈ [-1,1]（正=支持，负=反驳/证伪，见 ADR-0005）。
7. **update_evidence** — Evidence Updater。为每个被命中的 claim 更新 confidence（取所有证据中绝对值最大的 support）、累加 search_attempts、把命中文档记入 supporting_documents。
8. **control** — Controller（薄）。仅计算终止态并在 CONTINUE 与终止之间路由；不做 LLM 决策（见 ADR-0003）。
9. **finalize** — 组装 ResearchOutput，纯拼装终止态，无 LLM。

### 边（Edges）

- `START → plan`
- `plan → generate_queries`
- `generate_queries → search → rerank → normalize → score_claims → update_evidence → control`（顺序边）
- `control → generate_queries`（条件边：CONTINUE，回到循环起点）
- `control → finalize`（条件边：终止态 DONE 或 EXHAUSTED）
- `finalize → END`
- 循环回边受 `max_loops`（= LangGraph `recursion_limit`）保护，作为兜底安全网。

### 状态模型（State）

**ResearchState**（即扁平图的图状态，仅在内存，v1 不做续跑，见 ADR-0004）
- `task: str`
- `evidence_plan: list[EvidenceClaim]` — plan 后固定
- `search_history: list[SearchRound]`
- `loop_count: int`
- `terminal: None | DONE | EXHAUSTED`
- `config: EDREConfig` — 注入

**EvidenceClaim**
- `id: str`
- `hypothesis: str` — 一条可判真伪的断言（不是 Topic）
- `importance: CRITICAL | OPTIONAL`
- `confidence: float ∈ [-1, 1]` — 最具决定性的有符号 support；唯一进度真相
- `supporting_documents: list[DocumentRef]` — 每条带其有符号 support
- `search_attempts: int`
- `status`（派生只读，不落库）：
  - `NOT_STARTED`：attempts == 0
  - `PARTIAL`：已尝试但未达结论且未超预算
  - `VERIFIED`：confidence ≥ verify_threshold（强正）
  - `REFUTED`：confidence ≤ −refute_threshold（强负，证伪是成功发现）
  - `ABANDONED`：attempts ≥ max_attempts 且既未 VERIFIED 也未 REFUTED

**SearchRound**（每轮一条，支撑"每个 Evidence 可追溯"）
- `loop_index, queries, documents(带 source_query), rerank_scores, support_scores, provider, duration, errors`

### 终止语义

- claim「已解决」= VERIFIED 或 REFUTED；「未解决」= ABANDONED。
- **DONE** = 所有 critical claim 已解决。
- **EXHAUSTED** = 撞 max_loops 或无进展退出，但仍有 critical claim 未解决。
- 输出必须区分二者，绝不把 EXHAUSTED 伪装成 DONE。

### ResearchOutput（v1）

- `research_summary` — **确定性结果快照，无 LLM**：终止态、claim verdict 计数（VERIFIED/REFUTED/ABANDONED）、critical 是否全解决、coverage（已解决/总数，纯展示）、loop_count。
- `evidence` — 每个 claim 的 verdict + 支持/反驳文档。
- `citations` — 引用列表。
- `loop_count`。

### 配置（全部可配，对应非功能要求 Loop/Score 可配置）

`claim 数下限(3)/上限(6)`、`min_critical(1)`、`verify_threshold`、`refute_threshold`、`rerank 门控阈值`、`门控权重 w1(TaskMatch)/w2(QueryMatch)`、`max_attempts(每 claim)`、`max_loops`、`queries_per_claim(默认 2~3)`、证据排序用 `doc_relevance = w1·TaskMatch + w2·QueryMatch + w3·max|support|`。

---

## User Stories

1. 作为使用者，我想提交一个自然语言问题，从而无需自己设计搜索词就能启动一次研究。
2. 作为使用者，我想让系统把我的问题拆成若干条明确的、可判真伪的知识断言，从而清楚它到底要验证什么。
3. 作为使用者，我想看到每条断言被标为"关键"或"可选"，从而知道哪些是回答问题的骨架。
4. 作为使用者，我想让系统只围绕未解决的断言继续检索，从而不浪费预算在已经搞定的部分。
5. 作为使用者，我想让明显跑题的检索结果在进入评估前就被挡掉，从而最终结论不被低质量内容污染。
6. 作为使用者，我想让系统判断每篇文档是"支持"还是"反驳"某条断言，从而区分"真""假""没查到"。
7. 作为使用者，当一个关键前提其实是错的时，我想被明确告知它被证据"证伪"，而不是笼统地说"未找到"。
8. 作为使用者，我想让每条断言的置信度随证据累积而更新，从而进度是基于证据而非搜索次数。
9. 作为使用者，我想让系统在所有关键断言都被结论性解决时自动停下，从而不多做无用轮次。
10. 作为使用者，我想让系统在明显无法取得进展时停下并诚实标注缺口，从而不无限循环也不假装完成。
11. 作为使用者，我想拿到一份确定性的结果快照（各断言结论、覆盖情况、轮数），从而快速判断答案的完整度。
12. 作为使用者，我想让每条结论都能追溯到具体文档引用，从而自行核验。
13. 作为使用者，我想让单条查不到的断言在有限次尝试后被放弃，从而它不拖垮整轮研究。
14. 作为使用者，我想调节关键阈值（验证/证伪阈值、每断言尝试上限、最大轮数、断言数量），从而在精度、深度与成本间权衡。
15. 作为使用者，我想让检索层某个 provider 故障时自动切换，从而单点故障不打断研究。
16. 作为运维者，我想让每轮、每个节点、每次 LLM 生成都可追踪（Langfuse），从而排查与观测。
17. 作为运维者，我想让每个终止决策可解释（基于哪些断言的状态），从而信任系统的停止行为。
18. 作为开发者，我想让 Planner 可替换，从而实验不同的断言拆解策略而不动下游。
19. 作为开发者，我想让 Query Generator 可替换，从而独立优化检索策略。
20. 作为开发者，我想让第一层 reranker 通过统一协议可替换（本地/托管），从而更换模型不影响上层。
21. 作为开发者，我想让检索缝收敛为单一 `search_many` adapter，从而 EDRE 与检索层解耦、便于打桩测试。
22. 作为开发者，我想在整图 `ResearchInput → ResearchOutput` 这一最高缝上测试，从而验证外部行为而非实现细节。
23. 作为开发者，我想让 EDRE 全异步，从而检索与评估并发、吞吐可接受。
24. 作为使用者，我想让文档在评估前被规范化为带冲突标注的事实片段，从而对断言的判断更干净、更省 token。
25. 作为使用者，我想让证伪结论（REFUTED）与查不到（ABANDONED）在输出中清晰区分，从而不误读研究结果。

---

## Implementation Decisions

- **集成缝（ADR-0001）**：新增纯函数 adapter `search_many(queries) → 带 query 归属、已 URL 去重的文档集合`，内部复用现有 `fan_out_search` 与 `_merge_responses`。将去重逻辑从 `consolidate` 节点中拆出并入 adapter；`consolidate` 的 query 合并式 answer 输出弃用。provider 故障转移落在 adapter 内。
- **文档规范化节点（ADR-0001）**：改造复用 `AnswerConsolidator` 的 LLM 规范化 prompt（dense factual + 保留 citation + 冲突标注），但从"对一个 query 的合并整体"改为"逐文档"，仅对 reranker 幸存文档运行。
- **两层评估（ADR-0002）**：废弃 PRD 原"四项相加"Final Score。第一层 = 本地 cross-encoder reranker 计算 Task Match + Query Match 做门控；第二层 = LLM 逐文档输出对全 claim 的有符号 support 向量。Coverage 不参与文档打分。
- **第一层 reranker**：本地 cross-encoder（如 bge-reranker / ms-marco MiniLM），封装为可替换的 `Reranker` 协议 `rerank(left_text, docs) → scores`；Task Match 与 Query Match 复用同一模型，仅左侧输入不同（原始 task vs 当前 query）。粗筛用原始文本，不依赖规范化。
- **有符号 support 与证伪（ADR-0005）**：第二层打分范围为 `[-1, 1]`，负值表示证据反驳该 claim。claim 终止态因此含 REFUTED。v1 不特殊处理同一 claim 内证据互相矛盾（取绝对值最大的 support 决定走向）。
- **confidence 聚合**：取该 claim 所有证据中绝对值最大的有符号 support（v1 纯 max，不做多来源累积加成）。status 完全由 confidence + search_attempts 派生，不独立存储。
- **薄 Controller 与固定 claim 集（ADR-0003）**：Controller 决策空间收敛为 `{CONTINUE, 自动终止}`。FINISH/STOP 由停止条件计算为 DONE/EXHAUSTED，非主动选择。REPLAN、CHANGE_PROVIDER、query 策略细分均移出：REPLAN 推迟到后续；CHANGE_PROVIDER 下沉 adapter；改写 vs 换新归 Query Generator 内部（v1 只做换新）。claim 集 plan 后固定。
- **每 claim 收敛**：复用 EvidenceClaim 既有的 `search_attempts` 字段配合 `max_attempts` 上限；ABANDONED 为派生态，不新增字段。
- **编排载体（ADR-0004）**：EDRE 为单张扁平 LangGraph，`search_many` 是普通节点而非嵌套子图；`recursion_limit` 兼作 max_loops。复用现有 Langfuse 回调实现逐节点追踪。
- **无续跑（ADR-0004）**：v1 有意识放弃 PRD 非功能要求中的"状态可恢复"；ResearchState 仅在内存，不接 checkpointer。
- **停止条件**：DONE = 所有 critical claim 已 VERIFIED 或 REFUTED；EXHAUSTED = 撞 max_loops 或无进展仍有 critical 未解决。
- **输出**：`research_summary` 为确定性快照（无 LLM）；不含成文叙述答案、不含 controller_trace。
- **Planner 护栏**：LLM 标 importance 时约束"至少 1 个 critical 且 critical 应为少数"；claim 总数约束在 3~6。
- **可配置项**：见上文"配置"清单，全部进 `EDREConfig` 注入。
- **接口契约**：输入 `ResearchInput { task: str }`；输出 `ResearchOutput { research_summary, evidence, citations, loop_count }`。

## Testing Decisions

- **测试哲学**：只测外部行为，不测实现细节。断言 ResearchOutput 的语义（终止态、各 claim verdict、覆盖、可追溯引用），不断言内部节点的中间数据形状。
- **首选缝（最高、单一）**：在整图边界 `ResearchInput → ResearchOutput` 测试 EDRE，把 `search_many` adapter 打桩为返回受控文档集的假实现。这是理想的单缝，避免为每个内部节点单独造缝。
- **第二缝（必要时）**：`search_many` adapter 自身（复用/改造了现有 `fan_out_search` + `_merge_responses`），验证去重、query 归属、provider 故障转移。
- **确定性**：LLM 节点（plan / score_claims）与 reranker 在整图测试中以可注入的假模型/假打分器替换，使断言可复现。
- **需覆盖的行为**：全部 critical 验证 → DONE；关键前提被证伪 → 仍 DONE 且标 REFUTED；关键断言查不到 → EXHAUSTED 且如实标 ABANDONED；单死 claim 在 max_attempts 后被放弃而不拖垮其余；跑题文档被第一层门控挡下不进证据；research_summary 计数与终止态一致。
- **既有范式（Prior art）**：仓库 `tests/` 已有针对 subgraph 的图级测试与 `e2e_smoke.py`，新测试沿用其"构造 config + 注入假 LLM/provider + 断言输出契约"的模式。

## Out of Scope

- Answer Synthesis / 成文叙述答案与研究报告（PRD 原 Phase R7）；v1 的 research_summary 仅为确定性快照。
- REPLAN 与循环中可变的 claim 集。
- CHANGE_PROVIDER 作为编排层决策（v1 仅在 adapter 内做故障转移）。
- 状态可恢复 / checkpointer / 崩溃续跑。
- 多来源累积式 confidence 与同一 claim 内证据冲突的特殊处理。
- Query 改写 vs 换新的区分（v1 仅换新角度重试）。
- controller_trace 富决策轨迹。
- Search Provider 实现、Web Crawling、Embedding Index、Long-term Memory、Citation 渲染、UI。

## Further Notes

- 现有 `build_search_subgraph` 完整子图保持不变，供其它调用方独立使用；EDRE 不走它。
- 术语一律以 `CONTEXT.md` 为准；实现时如需引入新领域术语，先更新该 glossary。
- 各架构取舍的完整理由见 `docs/adr/0001~0005`，本 PRD 不重述。
- 后续 Phase 的自然候选：Answer Synthesis（R7）、REPLAN 与动态 claim 集、多来源交叉验证的 confidence、崩溃续跑、托管 reranker 选项、改写策略。
