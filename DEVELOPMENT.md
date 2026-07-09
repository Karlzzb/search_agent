# 后续开发手册（Development Guide）

本手册指导在 `search_agent` 仓库上做后续开发：架构、扩展点、必须保持的不变量，以及待办演进方向。
仓库含两层，本手册分两部分覆盖：

- **Part A — 检索子图（search_agent）**：自足的、可编译为 LangGraph 子图的 map-reduce 检索包（§1–§10）。
- **Part B — EDRE 研究引擎（search_agent.edre）**：叠在检索层之上的证据驱动研究编排引擎（§11–§16）。

配套文档：`PRD.md`（需求与设计定稿）、`CONTEXT.md`（EDRE 术语表）、`TASKS_EDRE.md`（EDRE 切片进度）、`docs/adr/0001~0005`（EDRE 架构取舍）、`TESTING.md`（如何安装与跑测试）。

---

# Part A — 检索子图（search_agent）

## 1. 这个包是什么

一个**自足的、可编译为 LangGraph 子图**的检索包，从 deeptutor monolith 抽取而来，**零 `deeptutor.*` 依赖**。
对外只暴露一个工厂：

```python
from search_agent import build_search_subgraph, SearchConfig

# 包在 import 时自动加载同目录 .env（LLM_* / SEARXNG_BASE_URL / LANGFUSE_*）。
# base_url 缺省即取 SEARXNG_BASE_URL；LLM 与 Langfuse 也由 .env 驱动。
graph = build_search_subgraph(SearchConfig(consolidation_use_llm=True))
out = await graph.ainvoke({"task": "对比 A 和 B 的优劣"})
# out == {"consolidated": str, "citations": list[Citation]}
```

它是**确定性 map-reduce 检索子图**，不是自主 agent：不决定“要不要搜”，不做充分性反思，不循环重搜——这些由上层父图负责。

---

## 2. 拓扑与数据流

```
{"task": str}
     │  (SearchInput，父图只见此)
     ▼
 decompose(LLM)         task → subqueries: list[str]        无 LLM 时退化为 [task]
     ▼
 fan_out_search(并发)    每个 subquery 并发检索 → raw_results  单路失败被跳过
     ▼
 consolidate(模板|LLM)   合并 + 去重 → consolidated + citations
     ▼
{"consolidated": str, "citations": list[Citation]}
     │  (SearchOutput，父图只见此)
```

内部 State（`SearchSubgraphState`，`subgraph.py`）比对外契约多出中间键，由 `input_schema` / `output_schema` 隔离，父图不感知：

```python
class SearchSubgraphState(TypedDict):
    task: str
    subqueries: list[str]
    raw_results: list[WebSearchResponse]
    consolidated: str
    citations: list[Citation]
```

三个节点都是 `async def`，在 LangGraph 里各自可见（可流式 / checkpoint）。

---

## 3. 模块地图

| 文件 | 职责 | 关键出口 |
| --- | --- | --- |
| `subgraph.py` | 子图定义与工厂 | `build_search_subgraph`、`SearchSubgraphState`、`SearchInput/Output` |
| `config.py` | 注入式运行配置 | `SearchConfig`（dataclass；`base_url` 缺省取 `SEARXNG_BASE_URL`） |
| `consolidation.py` | 结果整合（模板/LLM） | `AnswerConsolidator`（LLM 路径吃 LangChain `BaseChatModel`） |
| `llm.py` | 内置默认 LangChain chat model | `default_chat_model()`、`normalize_base_url()` |
| `tracing.py` | Langfuse 接入 | `get_langfuse_callback()` |
| `env.py` | `.env` 自动加载与读取 | `load_env()`、`env_str()` |
| `providers/__init__.py` | provider 注册表 | `register_provider`、`get_provider`、`list_providers` |
| `providers/*.py` | 各 provider 实现 | searxng / duckduckgo / brave / tavily / jina / perplexity / serper |
| `base.py` | provider 抽象基类 | `BaseSearchProvider` |
| `contracts.py` | 数据契约（勿改） | `Citation`、`SearchResult`、`WebSearchResponse` |
| `retrieval.py` | EDRE 的唯一检索缝（见 Part B） | `search_many`、`RetrievedDocument` |
| `edre/` | EDRE 研究引擎（见 Part B） | `build_research_graph`、`EDREConfig` |

---

## 4. 配置（SearchConfig）

所有运行期开关都在 `SearchConfig`，无隐式全局状态：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `provider` | `"searxng"` | 主 provider |
| `base_url` | `SEARXNG_BASE_URL` env | SearXNG 必需；缺省取环境变量，显式传参优先；构建时校验（见 §7） |
| `api_key` | `None` | SearXNG/DuckDuckGo 不需要；其它 provider 需要 |
| `proxy` | `None` | 透传给 provider |
| `max_results` | `5` | provider 内部硬上限仍为 10 |
| `consolidation_use_llm` | `False` | True 走 LLM 整合，False 走 Jinja2 模板 |
| `consolidation_custom_template` | `None` | 覆盖 provider 默认模板 |
| `consolidation_llm_model` | `None` | 仅覆盖 consolidate 步的 model；decompose 保持默认 |

**最小可用锚点**：`SearchConfig()` + `consolidation_use_llm=False`，无 LLM key（`LLM_KEY` 未设）时子图走无 LLM 路径，是最低依赖起步组合。

---

## 5. LLM 注入策略（原生 LangChain）

整个包已**原生接入 LangChain/LangGraph**：LLM 不再是裸 `openai` SDK 适配器，而是 LangChain `ChatOpenAI`（`BaseChatModel`）。这样 decompose/consolidate 的模型调用会被 Langfuse callback 自动追踪（generation span 自动嵌套在节点 span 下）。

`build_search_subgraph(config, llm=...)` 接受一个可选 **LangChain chat model**（`BaseChatModel`）：

- **注入** → decompose 与 consolidate **默认共用**该 chat model。节点内部直接 `await chat.ainvoke([SystemMessage(...), HumanMessage(...)])`，读 `.content`。
- **未注入** → `default_chat_model()` 尝试构建内置默认模型：
  - 读 `LLM_KEY`（必需以激活）、`LLM_BASE_URL`、`LLM_MODEL`（默认 DashScope 兼容模式 `qwen-plus`）。
  - `LLM_BASE_URL` 末尾的 `/chat/completions` 会被 `normalize_base_url()` 去掉，以匹配 SDK 期望的 `.../v1` base（`.env` 原样写完整 completions URL 即可）。
  - `LLM_KEY` 未设 → 返回 `None`，子图退回无 LLM 路径（decompose = `[task]`）。
- **consolidate 端 model 覆盖**：`consolidation_llm_model` 提供时只覆盖 consolidate 步——注入场景用 `chat.bind(model=...)`，默认场景用 `default_chat_model(model=...)`；decompose 始终用基础模型。

任何 `BaseChatModel`（`ChatOpenAI`、其它 LangChain chat model、或 duck-type 出 `ainvoke(...) -> AIMessage` 的测试 fake）都可注入。

**硬约束**：全程 `await chat.ainvoke(...)`，禁止同步调用——LangGraph 节点在事件循环内运行。

---

## 5b. Langfuse 追踪

`tracing.get_langfuse_callback()` 依据 `.env` 里的 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` 构建 LangChain `CallbackHandler`，并在 `build_search_subgraph` **编译期**通过 `graph.with_config({"callbacks":[handler]})` bake 进图。

- 因此任意 `graph.ainvoke({"task": ...})` 都会被自动追踪，无需在调用处传 callbacks。
- **非致命**：`LANGFUSE_*` 缺失或 `langfuse` 未安装 → 返回 `None`，子图照常运行、不追踪、不报错。
- 由于 LLM 已原生化，节点（decompose/fan_out/consolidate）与其内部的 LLM generation 会一并出现在同一条 trace 里。

---

## 6. 扩展点

### 6.1 新增一个 provider

1. 在 `providers/` 新建 `myprovider.py`，继承 `BaseSearchProvider`。
2. 用 `@register_provider("myprovider")` 装饰。
3. 实现 `search(self, query, base_url="", max_results=5, timeout=20, **kwargs) -> WebSearchResponse`。
4. 设置类属性：`requires_api_key`、`supports_answer`、`display_name`、`description`、`API_KEY_ENV_VARS`。
5. 在 `providers/__init__.py` 的 `_register_builtin_providers()` 里加入 import（触发注册）。
6. 在 `tests/test_providers.py` 补注册表用例；如需端到端，参照 slice 测试注入 fake。

约定：
- 同步 `requests.get` 会被子图用 `asyncio.to_thread` 包裹，provider 内保持同步即可。
- `supports_answer = True` 的 provider（自带 answer）**不会**触发 LLM consolidation；`False`（如 SearXNG）才触发。
- `max_results` 上限硬编码 10，勿放开。

### 6.2 调整整合模板

`consolidation.py` 的 `PROVIDER_TEMPLATES` + `PROVIDER_TEMPLATE_MAP` 管理 provider 专用模板；未命中者走 `_format_simple_results` 通用兜底（SearXNG / DuckDuckGo 走这条）。
Jinja2 环境 `autoescape=True`，勿关闭（见 §7）。

### 6.3 接入自己的 LLM

实现 `LLMCompleter` 协议后，`build_search_subgraph(config, llm=your_llm)` 即可，无需改包内代码。

---

## 7. 必须保持的不变量（改动前先读）

- **零 monolith 依赖**：包内任何文件不得 `import deeptutor.*`。`tests/test_independence.py` 用 AST 静态护栏，破坏即红。
- **数据契约稳定**：`contracts.py` 的 `Citation` / `SearchResult` / `WebSearchResponse` 及 `to_dict()` 字段是下游契约，勿删改字段。引用需保留 `reference`（形如 `[1]`）/ `url` / `title` / `snippet`，供可点击溯源。
- **父图契约收窄**：对外只进 `{"task": str}`、只出 `{"consolidated": str, "citations": list[Citation]}`；中间键不得泄漏到 output schema。
- **异步纯净**：节点全 `async`，LLM 调用全 `await chat.ainvoke(...)`，并发用 `asyncio.gather`，同步 IO 用 `asyncio.to_thread`。
- **配置来自 `.env`（刻意放宽的既有不变量）**：包在 import 时通过 `env.load_env()` 自动加载**同目录 `.env`**，`SearchConfig.base_url` 缺省取 `SEARXNG_BASE_URL`，`default_chat_model()` / Langfuse 读各自环境变量。这**放宽了原“无隐式全局状态”约束**——刻意为之，以适配项目的 `.env` 配置风格。约束仍在：`load_dotenv(override=False)` 永不覆盖已存在的进程环境变量；单测套件用 `tests/conftest.py` 的 autouse fixture 清空这些变量以保持 hermetic。
- **环境变量命名**：模型用 `LLM_KEY` / `LLM_BASE_URL` / `LLM_MODEL`，搜索用 `SEARXNG_BASE_URL`，追踪用 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL`。（已弃用旧的 `OPENAI_*` 命名。）
- **SearXNG 回退**：`provider="searxng"` 且无 `base_url` → 回退 `duckduckgo`（`_effective_provider`）。
- **降级不崩**：单路 fan-out 失败 → 记录并跳过，不整体抛出（`asyncio.gather(..., return_exceptions=True)`）；全失败 → `consolidated` 为“无结果”占位、`citations` 为空。
- **引用去重**：合并时按 `url` 简单去重、`reference` 连续编号；不做语义融合 / rerank / embedding。
- **安全姿态**：SearXNG `base_url` 在 `build_search_subgraph` 构建期用 `_validate_base_url` 校验（非法 scheme / 缺 host 直接抛，而非被降级吞掉）；Jinja2 `autoescape=True` 保持。
  - 注意既有校验的归一化行为：裸主机名（如 `not-a-url`）与 `http://` 会被补全为合法 host，只有明确非法 scheme（如 `ftp://`、`gopher://`）会被拒绝——这是刻意保留的现有姿态。

---

## 8. 切片进度（检索子图）

- Slice 1 — 切断 monolith 依赖、建立注入 seam 与打包 ✅
- Slice 2 — 最小可用子图（无 LLM）✅
- Slice 3 — LLM decompose + 并发 fan-out + 默认适配器 ✅
- Slice 4 — LLM consolidation + consolidate 端 model 覆盖 ✅
- Slice 5 — 兜底与降级路径（回退 / 单路降级 / 全失败 / 去重 / base_url 校验）✅
- Slice 6 — 原生化 + 配置接入：`.env` 自动加载、`LLM_*`/`SEARXNG_*` 命名、LLM 改 LangChain `ChatOpenAI`、Langfuse 全量追踪、真实 E2E（`e2e_smoke.py`）✅

EDRE 引擎（Slice A~I）的进度见 §16 与 `TASKS_EDRE.md`。

**当前测试基线：全量 87 passed（hermetic，无网络）**；检索层真实 E2E 由 `e2e_smoke.py` 单独跑，EDRE 真实 E2E 由 `edre_e2e_smoke.py` 单独跑（见 `TESTING.md`）。

---

## 9. 明确的非目标（Out of Scope）

以下**不在本包**内实现（见 `PRD.md` Out of Scope）：
- 上层“是否搜索 / 是否再搜一轮”的决策、反思与循环——父图负责。
- rerank / 向量化 / 分块 / 跨源语义融合——当前仅做基于 url 的引用去重。
- 新增 SearXNG 之外的 provider 能力增强；废弃 provider（exa / baidu / openrouter）不纳入。
- deeptutor 侧集成（`WebSearchTool`、`model_catalog.json` UI、settings 路由）。
- 把子图升级为自主 research agent（多轮反思、自主终止）——属未来演进。
- 持久化 / 落盘。

---

## 10. 演进方向（可选，非承诺）

- decompose 与 consolidate 完全独立配模型：需在 `SearchConfig` 增加 decompose 侧 model 字段。
- 引用层跨源语义去重 / rerank：当前只按 url 去重，若要语义融合需新增 reduce 步。
- provider 级重试 / 超时策略：当前单路失败即跳过，可按需加退避重试。

---

# Part B — EDRE 研究引擎（search_agent.edre）

## 11. EDRE 是什么

**EDRE（Evidence-Driven Research Engine）** 是叠在检索层之上的**证据驱动研究编排引擎**。
它围绕用户问题规划一组待验证的知识点（`EvidenceClaim`），多轮收集证据、评估完整性，直至所有关键 claim 都被**验证或证伪**，或预算耗尽而**诚实地**退出。
核心状态是 **`EvidenceClaim`**（一个可判真伪的 hypothesis），而非 Query 或 Document（术语以 `CONTEXT.md` 为准）。

对外只暴露一个工厂：

```python
from search_agent.config import SearchConfig
from search_agent.edre import EDREConfig, build_research_graph

graph = build_research_graph(EDREConfig(search=SearchConfig(provider="searxng", max_results=5)))
result = await graph.ainvoke({"task": "研究 X 的发展趋势"})
out = result["output"]          # ResearchOutput
# out.research_summary  确定性结果快照（终止态 / verdict 计数 / coverage / loop_count）
# out.evidence          list[ClaimVerdict]，每 claim 的 verdict + 支持/反驳文档
# out.citations         list[Citation]，可点击溯源
```

与检索子图的关系：EDRE **不**驱动 `build_search_subgraph` 完整子图，而是只通过 `retrieval.search_many` 这**唯一的检索缝**取数（复用 fan-out / 去重 / 故障转移，跳过其 decompose 与 query 合并式 answer；见 ADR-0001）。`build_search_subgraph` 保持不变，供其它调用方独立使用。

它**不是**自主 research agent：Controller 是薄的（决策空间只有 `{CONTINUE, 自动终止}`，无 LLM 决策、无 REPLAN / CHANGE_PROVIDER）；claim 集在 plan 后固定；无持久化 / checkpointer（ResearchState 仅在内存，见 ADR-0004）。

---

## 12. 拓扑与数据流（扁平图，无嵌套子图）

```
{"task": str}   (ResearchInput，调用方只见此)
     ▼
 plan(LLM)              task → 3~6 个 EvidenceClaim（带 importance）；claim 集自此固定
     ▼
 ┌─▶ generate_queries(LLM)  只为未解决 claim 换新角度生成 query（loop_count += 1）
 │        ▼
 │   search               扁平化 query → search_many → 带 query 归属、URL 去重的文档
 │        ▼
 │   rerank               第一层门控：本地 cross-encoder，低于阈值的文档丢弃
 │        ▼
 │   normalize(LLM)       逐文档蒸馏为 dense、带引用/冲突标注的事实片段（只跑幸存文档）
 │        ▼
 │   score_claims(LLM)    每篇文档一次调用 → 对全部 claim 的有符号 support ∈[-1,1]
 │        ▼
 │   update_evidence      confidence = |support| 最大者；累加 search_attempts；记 SearchRound
 │        ▼
 │   control              计算终止态：DONE / EXHAUSTED / None
 └────────┤ None(CONTINUE)
          ▼ DONE|EXHAUSTED
 finalize                纯拼装、无 LLM → ResearchOutput
     ▼
{"output": ResearchOutput}   (ResearchResult，调用方只见此)
```

全图全异步。循环回边由 `max_loops` 语义兜底（`control` 在 `loop_count >= max_loops` 时返回 EXHAUSTED），LangGraph `recursion_limit` 是原始安全网（`max_loops * 7 + 10`）。

`status` 是**派生只读**：`derive_status(claim, config)` 由 `confidence + search_attempts` 对阈值计算，从不独立存储：

| status | 条件 |
| --- | --- |
| `VERIFIED` | `confidence >= verify_threshold` |
| `REFUTED` | `confidence <= -refute_threshold`（证伪是成功发现，≠ ABANDONED） |
| `NOT_STARTED` | 尚未检索（`search_attempts == 0`） |
| `ABANDONED` | `search_attempts >= max_attempts` 且仍未决定性 |
| `PARTIAL` | 其余（检索过但未决定性、预算未尽） |

已解决 = `VERIFIED | REFUTED`；未解决 = `NOT_STARTED | PARTIAL`；`ABANDONED` 是“尽力未成”，绝不伪装成 DONE。

---

## 13. EDRE 模块地图

| 文件 | 职责 | 关键出口 |
| --- | --- | --- |
| `edre/graph.py` | 扁平图定义与工厂；节点解析（注入 > 真实 LLM/reranker > inert 兜底） | `build_research_graph` |
| `edre/models.py` | 全部数据模型 + 派生逻辑（勿改契约字段） | `EvidenceClaim`、`EDREConfig`、`ClaimStatus`、`Terminal`、`DocumentRef`、`SearchRound`、`ResearchInput/Output/Result/State`、`derive_status`、`doc_relevance` |
| `edre/planner.py` | Planner：task → 3~6 falsifiable claim + importance 护栏 | `make_llm_planner` |
| `edre/query_generator.py` | 只为未解决 claim 换新角度生成 query | `make_llm_query_generator` |
| `edre/reranker.py` | 第一层门控：本地 cross-encoder（懒加载 torch） | `make_local_reranker`、`make_rerank_gate`、`default_local_reranker` |
| `edre/normalizer.py` | 逐文档蒸馏为带引用/冲突标注的事实片段 | `make_llm_normalizer` |
| `edre/scorer.py` | 第二层评估：每文档一次 LLM → 全 claim 有符号 support | `make_llm_scorer` |
| `retrieval.py` | `search_many` adapter（Part A 里的唯一检索缝） | `search_many`、`RetrievedDocument` |

`graph.py` 里的每个智能节点都遵循同一套**三级解析**（`_resolve_*`）：**注入的实现优先 → 否则构建真实 LLM/cross-encoder 实现 → 否则退化为 inert 兜底/passthrough**（`default_chat_model()` / `default_local_reranker()` 返回 `None` 时）。这让 hermetic 测试与真实运行共用同一张图。

---

## 14. EDREConfig（全量注入式配置）

所有精度/深度/成本旋钮都在 `EDREConfig`（`edre/models.py`），无隐式全局状态：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `search` | `SearchConfig()` | 透传给 `search_many` 的检索配置（provider / base_url / max_results 等） |
| `verify_threshold` | `0.7` | `confidence >= 此值` → VERIFIED |
| `refute_threshold` | `0.7` | `confidence <= -此值` → REFUTED |
| `max_attempts` | `3` | 单 claim 检索尝试上限；超过仍不决定性 → ABANDONED |
| `max_loops` | `6` | 研究循环轮数上限（语义 EXHAUSTED 闸；`recursion_limit` 据此推导） |
| `queries_per_claim` | `2` | 每个未解决 claim 每轮生成的 query 数 |
| `min_claims` / `max_claims` | `3` / `6` | Planner 产出的 claim 数量下限/上限（上限硬夹） |
| `min_critical` | `1` | 至少多少个 CRITICAL；且 CRITICAL 保持少数 |
| `rerank_threshold` | `0.3` | 第一层门控阈值：`w1·TaskMatch + w2·QueryMatch >= 此值` 才幸存 |
| `rerank_w1` / `rerank_w2` | `0.5` / `0.5` | 门控权重（TaskMatch / QueryMatch） |
| `doc_relevance_w1/w2/w3` | `1.0` 各 | 证据排序权重：`w1·TaskMatch + w2·QueryMatch + w3·\|support\|` |

reranker 模型可用 `EDRE_RERANK_MODEL` 环境变量覆盖（默认 `cross-encoder/ms-marco-MiniLM-L-6-v2`）。

---

## 15. EDRE 必须保持的不变量（改动前先读）

- **claim 集固定**：`plan` 之后 claim 集不增删（v1 无 REPLAN）；循环只更新每个 claim 的 confidence / search_attempts / 证据。
- **status 派生只读**：`ClaimStatus` 永远由 `derive_status(confidence, search_attempts)` 现算，绝不作为字段存储。改阈值语义只改 `derive_status` 一处。
- **证伪是成功发现**：强负 support → `REFUTED`，必须与 `ABANDONED`（预算耗尽仍不决定性）清晰可分；绝不把二者混同。
- **confidence 取绝对值最大**：`confidence` = 该 claim 所有证据中 `|signed support|` 最大者，符号决定走向；v1 纯 max，不做多来源累积、不特殊处理证据互相矛盾。
- **终止诚实性**：`DONE` = 所有 critical claim 已解决；`EXHAUSTED` = 撞 `max_loops` / 无可尝试工作但仍有 critical 未解决。`research_summary.blocking_claim_ids` 命名触发 EXHAUSTED 的未解决 critical claim，使决策可解释。
- **薄 Controller**：`control` 决策空间仅 `{CONTINUE, 自动终止}`，无 LLM 决策、无 REPLAN / CHANGE_PROVIDER（故障转移只在 `search_many` adapter 内部做，见 ADR-0003）。
- **唯一检索缝**：EDRE 只经 `search_many` 取数，不直接调用 `build_search_subgraph`；一篇文档保留全部命中它的 query 归属（可追溯）。
- **两层评估分离**：rerank 只判**相关性**（第一层门控，用原始文本，不依赖 normalize），score_claims 才判**有符号 support**（第二层）；勿把 relevance 与 support 混同（ADR-0002 / 0005）。
- **逐文档规范化**：normalize 是**逐文档**蒸馏（保留 citation + 冲突标注），不产 query 合并式 answer；只跑 reranker 幸存文档以约束成本（ADR-0001）。
- **finalize 无 LLM**：`research_summary` 是**确定性快照**（纯拼装），不是成文叙述答案（Answer Synthesis 属 R7，v1 非目标）。
- **契约收窄**：对外只进 `{"task": str}`（`ResearchInput`）、只出 `{"output": ResearchOutput}`（`ResearchResult`）；中间键（queries / documents / doc_scores）不泄漏到 output schema。
- **异步纯净**：全节点 `async`，LLM 调用全 `await llm.ainvoke(...)`，per-claim / per-doc 并发用 `asyncio.gather`。
- **零 monolith 依赖**：与 Part A 同一硬约束，`tests/test_independence.py` 全树 AST 护栏。

---

## 16. EDRE 切片进度（TASKS_EDRE.md 摘要）

| 切片 | 内容 | 状态 |
| --- | --- | --- |
| A | `search_many` 检索 adapter（唯一集成缝） | ✅ DONE |
| B | 行走骨架：扁平全图 + 数据模型 + 循环与基础终止 | ✅ DONE |
| C | Planner：task → 3~6 EvidenceClaim + importance 护栏 | ✅ DONE |
| D | Query Generator：只为未解决 claim 换新角度生成 query | ✅ DONE |
| E | 第一层门控：本地 cross-encoder reranker | ✅ DONE |
| F | 文档规范化：逐文档蒸馏为带引用/冲突标注的事实片段 | ✅ DONE |
| G | 第二层评估：有符号 support ∈[-1,1] + 证据更新 + 证伪 | ✅ DONE |
| H | 终止诚实性：ABANDONED / DONE vs EXHAUSTED / 不拖垮 | ✅ DONE |
| I | 确定性输出快照 + 全量配置 + 可观测 + 真实 E2E | ✅ DONE |

v1 全部切片 DONE。真实 E2E 由 `edre_e2e_smoke.py` 单独跑（见 `TESTING.md` §9）。

**v1 非目标**（见 `TASKS_EDRE.md` 末节）：Answer Synthesis / 成文答案（R7）、REPLAN 与可变 claim 集、编排层 CHANGE_PROVIDER、checkpointer / 崩溃续跑、多来源累积式 confidence、query 改写 vs 换新的区分。
