# 操作测试手册（Testing Guide）

本手册说明如何在本机安装、运行并扩展 `search_agent` 包的测试。
适用对象：需要跑通/新增子图与 provider 测试的开发者。

---

## 1. 环境准备

要求 Python ≥ 3.11。本仓库使用名为 `search_agent` 的 conda 环境（Python 3.11）。

本包的 `pyproject.toml` 把包名 `search_agent` 映射到当前目录（`package-dir: search_agent = "."`），
且包目录名本身就是 `search_agent`（其父目录即为“上级目录”）。因此必须先以可编辑方式安装，测试才能 `import search_agent`。

```bash
# 激活（或创建）conda 环境
conda activate search_agent          # 如未创建：conda create -n search_agent python=3.11 && conda activate search_agent

# 在仓库根目录以可编辑方式安装本包及测试工具链
cd <仓库根>/search_agent             # 例如 ~/repos/work/search_agent
pip install -e ".[dev]"              # 安装本包 + pytest / pytest-asyncio
```

依赖说明（全部来自 `pyproject.toml`，`pip install -e ".[dev]"` 一次装齐）：
- core：`langgraph` / `langchain` / `langchain-openai`（原生 LLM 与图）、`jinja2` / `requests` / `ddgs`（模板与检索）、`python-dotenv`（`.env` 自动加载）、`langfuse`（追踪）、`sentence-transformers`（EDRE 第一层本地 cross-encoder，会引入 `torch`，首次安装体积较大）。
- `[dev]`：`pytest`、`pytest-asyncio`（测试工具链）。

> **首次运行 EDRE reranker 会联网下载模型**（默认 `cross-encoder/ms-marco-MiniLM-L-6-v2`）。这只影响真实 E2E（§9）；hermetic 单测由 `conftest.py` 把本地 cross-encoder 置空为 passthrough，从不加载模型、不触网。

`.gitignore` 已覆盖 `__pycache__/`、`*.egg-info/`、`.pytest_cache/`、`.env`（含密钥，勿提交）、`.idea/`、HF/模型缓存等；`.env.example` 是可提交的模板（复制为 `.env` 并填真值）。

---

## 2. 运行测试

**约定：从上一级目录运行 `pytest`，让包以已安装模块的形式被导入。**
早期版本包内有一个与 stdlib 同名的 `types.py`，在包目录内运行时会遮蔽标准库 `types` 并导致启动崩溃；该文件已重命名为 `contracts.py`，此坑已消除。
保留从父目录运行的约定，是为了让导入路径与生产部署（已安装的 `search_agent` 包）保持一致：

```bash
cd <仓库根的上一级>          # 即包含 search_agent/ 目录的父目录，例如 ~/repos/work
python -m pytest search_agent -q
```

`pyproject.toml` 已固定两项 pytest 配置：
- `addopts = "--import-mode=importlib"`：以已安装包的方式导入测试，避免 sys.path 注入。
- `testpaths = ["tests"]`：只收集 `tests/` 下的用例。

常用命令：

```bash
# 全量
python -m pytest search_agent -q

# 单文件
python -m pytest search_agent/tests/test_subgraph_slice5.py -q

# 单用例
python -m pytest search_agent/tests/test_subgraph_slice5.py::test_single_subquery_failure_is_skipped_not_fatal -q

# 关键字过滤
python -m pytest search_agent -k "fallback or dedup" -q

# 只跑 EDRE 相关
python -m pytest search_agent -k "edre or search_many" -q
```

当前基线：全量 **87 passed**（hermetic，无网络；含检索子图与 EDRE 全部切片）。真实 E2E 见 §8（检索层）与 §9（EDRE）。

---

## 3. 异步测试约定

子图节点是 `async def`，但测试**不依赖 pytest-asyncio 插件**（`pyproject.toml` 未配置 `asyncio_mode`）。
所有异步调用都用普通同步测试函数包一层 `asyncio.run(...)` 驱动：

```python
out = asyncio.run(graph.ainvoke({"task": "..."}))
```

新增子图测试请沿用此写法，保持与现有用例一致，避免引入插件配置。

---

## 4. 测试 seam 与注入的 fake

唯一、最高层的测试 seam 是**编译后的子图对象**：

```python
graph = build_search_subgraph(config, llm=fake_llm)
out = asyncio.run(graph.ainvoke({"task": "..."}))
```

在此 seam 断言外部可观察的输出（`out["consolidated"]` / `out["citations"]`），**不**断言内部函数调用次数或私有字段。

三类 fake（均无真实网络）：

### 4.1 Fake chat model（LangChain 形状）

LLM 已原生化为 LangChain `BaseChatModel`，fake 只需 duck-type 出 `ainvoke(messages) -> AIMessage`：

```python
from langchain_core.messages import AIMessage

class FakeChat:
    def __init__(self, subqueries):
        self._subqueries = subqueries
    async def ainvoke(self, messages, config=None, **kwargs):
        return AIMessage(content="\n".join(self._subqueries))
```

若需区分 decompose 与 consolidate 调用，依据 system message（`m.type == "system"`）内容是否含 `"consolidat"` 判断；consolidate 端的 model 覆盖走 `chat.bind(model=...)`，fake 实现一个记录 `bound_model` 的 `bind` 即可断言（见 `tests/test_subgraph_slice4.py` 的 `RecordingChat`）。

### 4.2 Fake SearXNG（patch `requests.get`）

SearXNG provider 内部走 `requests.get`，用 monkeypatch 拦截：

```python
def _patch_query_aware_searxng(monkeypatch, per_query):
    def _get(url, timeout=20, **kwargs):
        query = kwargs["params"]["q"]
        if query not in per_query:
            raise RuntimeError(f"searxng boom for {query}")   # 用于降级测试
        payload = {"results": per_query[query]}
        return SimpleNamespace(status_code=200, json=lambda: payload, text="")
    monkeypatch.setattr("search_agent.providers.searxng.requests.get", _get)
```

让某个 query 抛错，即可覆盖“单路失败被跳过”“全失败降级”的路径。

### 4.3 Fake provider（换进注册表）

两种方式：

- **永久注册探针**（模块级，进程内长期存在）：用 `@register_provider("probe_name")` 装饰一个 `BaseSearchProvider` 子类，`requires_api_key = False`。适合并发探针（见 slice3 的 `_BarrierProbeProvider`）与自带 answer 的 provider（见 slice4 的 `_AnswerProvider`，`supports_answer = True`）。
- **临时替换已注册项**（推荐用于 fallback 测试，可逆）：

```python
import search_agent.providers as providers_pkg
monkeypatch.setitem(providers_pkg._PROVIDERS, "duckduckgo", _FakeDDG)
```

用它把 `duckduckgo` 换成 fake，即可在 `provider="searxng"` 且无 `base_url` 时验证回退，而不触碰真实 `ddgs`。

---

## 5. 现有测试清单

全量 **87 passed**，跨 19 个文件。

**检索子图（search_agent）**

| 文件 | 覆盖范围 |
| --- | --- |
| `tests/test_config.py` | `SearchConfig` 默认值与覆盖 |
| `tests/test_types_contract.py` | `Citation` / `SearchResult` / `WebSearchResponse.to_dict()` 契约稳定性 |
| `tests/test_providers.py` | 注册表：keyless / keyed / deprecated / unknown |
| `tests/test_consolidation.py` | `AnswerConsolidator` 模板与 LLM（chat model）路径 |
| `tests/test_llm_adapter.py` | `normalize_base_url()` 与 `default_chat_model()` |
| `tests/test_independence.py` | AST 护栏：全树无 `import deeptutor.*` |
| `tests/test_subgraph.py` | Slice 2：无 LLM 最小子图 |
| `tests/test_subgraph_slice3.py` | Slice 3：LLM decompose + 并发 fan-out + 默认适配器 |
| `tests/test_subgraph_slice4.py` | Slice 4：LLM consolidation + model 覆盖 |
| `tests/test_subgraph_slice5.py` | Slice 5：回退 / 单路降级 / 全失败 / 去重 / base_url 校验 / autoescape |

**EDRE 研究引擎（search_agent.edre）**

| 文件 | 覆盖范围（切片） |
| --- | --- |
| `tests/test_search_many.py` | Slice A：`search_many` 去重 / query 归属 / 故障转移 |
| `tests/test_edre_skeleton.py` | Slice B：行走骨架全图 `ResearchInput → ResearchOutput` |
| `tests/test_edre_planner.py` | Slice C：Planner 数量夹取 / importance 护栏 |
| `tests/test_edre_query_generator.py` | Slice D：只为未解决 claim 换新角度生成 query |
| `tests/test_edre_reranker.py` | Slice E：第一层门控（真 gate + 假 `Reranker`） |
| `tests/test_edre_normalizer.py` | Slice F：逐文档蒸馏 grounding |
| `tests/test_edre_scorer.py` | Slice G：有符号 support + 证据更新 + 证伪 |
| `tests/test_edre_termination.py` | Slice H：DONE / EXHAUSTED / REFUTED / ABANDONED 诚实性 |
| `tests/test_edre_output.py` | Slice I：确定性快照 + `doc_relevance` 排序 |

`tests/conftest.py` 放一个 autouse 的 `_hermetic_env` fixture：因为包在 import 时会自动加载 `.env`，该 fixture 为每条用例清空 `LLM_*` / `SEARXNG_*` / `LANGFUSE_*`，保证套件离线确定（`default_chat_model()` 与 `get_langfuse_callback()` 都退化为 `None`）。**它还把 `search_agent.edre.graph.default_local_reranker` 打桩为返回 `None`**，使 EDRE 第一层门控退化为 passthrough，单测永不加载重型 cross-encoder / 触网。fake 与其余辅助函数仍在各测试文件内内联。

---

## 6. 新增一条子图测试的步骤

1. 从 `tests/` 里挑一个最接近的文件（多数情况是 `test_subgraph_slice*.py`）复制 fake 辅助函数。
2. 构造 `SearchConfig(...)`，注入 `FakeChat` 或 patch `requests.get`。
3. `graph = build_search_subgraph(config, llm=fake)`。
4. `out = asyncio.run(graph.ainvoke({"task": ...}))`。
5. 只断言 `out["consolidated"]` / `out["citations"]` 等外部可观察行为。
6. 从上级目录跑：`cd <仓库根的上一级> && python -m pytest search_agent -q`。

> 新增一条 **EDRE** 测试同理：在最高缝 `ResearchInput → ResearchOutput` 上断言语义（终止态、各 claim verdict、覆盖、可追溯引用），把 `search_fn`、`planner` / `query_generator` / `normalizer` / `scorer`（LLM 节点）、`reranker`（gate）以可注入的假实现替换。`build_research_graph(config, search_fn=..., planner=..., scorer=..., ...)` 的每个组件都可注入；参照 `tests/test_edre_*.py`。

---

## 7. 独立性回归（不可回退的护栏）

`tests/test_independence.py` 用 AST 静态断言包内**任何文件都不出现** `import deeptutor.*`。
任何重新引入 monolith 依赖的改动都会让它变红——这是本包“可独立拷贝运行”的硬约束，改动 provider / consolidation / subgraph 时务必保持它绿。

---

## 8. 真实 E2E（`e2e_smoke.py`）

单测套件是 hermetic 的（不打网络）。要验证**真实**链路——真 SearXNG + 真 qwen-plus + 真 Langfuse——跑顶层脚本 `e2e_smoke.py`。它读 `.env`、用正常工厂构图（自动解析默认 `ChatOpenAI` 并 bake Langfuse handler）、跑一个真实 task、打印 `consolidated` / `citations` 与 Langfuse trace 链接。

从**上级目录**以模块方式运行（同 §2，让包以已安装模块形式导入）；已激活 `search_agent` conda 环境时直接用 `python` 即可：

```bash
cd <仓库根的上一级>          # 例如 ~/repos/work
python -m search_agent.e2e_smoke
```

前置：`.env` 需含 `SEARXNG_BASE_URL` / `LLM_KEY` / `LLM_BASE_URL`（缺失即报错退出）；`LANGFUSE_*` 缺失时脚本照跑，只是不追踪。运行结束会打印形如 `http://<langfuse-host>/project/<id>/traces/<trace-id>` 的链接，节点与 LLM generation 都在同一条 trace 内。

---

## 9. EDRE 真实 E2E（`edre_e2e_smoke.py`）

验证**整条研究链路**——真 SearXNG（经 `search_many`）+ 真 qwen-plus（plan / query / normalize / score）+ 真本地 cross-encoder（第一层门控）+ 真 Langfuse——跑顶层脚本 `edre_e2e_smoke.py`。它用正常工厂 `build_research_graph(EDREConfig(...))` 构图、跑一次真实研究 task，打印**确定性结果快照**（终止态 / verdict 计数 / coverage / loop_count）、每 claim 的 verdict 与最相关支持文档、引用列表，最后给出 Langfuse trace 链接。它不做断言，是人工检视的冒烟脚本。

```bash
cd <仓库根的上一级>          # 例如 ~/repos/work
python -m search_agent.edre_e2e_smoke
```

前置：同 §8（`.env` 需含 `SEARXNG_BASE_URL` / `LLM_KEY` / `LLM_BASE_URL`；`LANGFUSE_*` 可缺）。
额外：**首次运行会联网下载 cross-encoder 模型**（默认 `cross-encoder/ms-marco-MiniLM-L-6-v2`，可用 `EDRE_RERANK_MODEL` 覆盖），之后走本地缓存。整条链路会按“每轮 / 每节点 / 每次 LLM 生成”出现在同一条 Langfuse trace 内。
