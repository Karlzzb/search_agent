"""Slice 4 — LLM consolidation + consolidate-side model override.

Tests observe behavior only at the agreed seam: the compiled subgraph via
``build_search_subgraph(config, llm=fake).ainvoke({"task": ...})``. A single
fake chat model serves both decompose and consolidate; it distinguishes the two
by their system prompt and records the bound model per call so we can assert the
model override lands only on the consolidate call (the subgraph binds it via
``chat.bind(model=...)``). A fake SearXNG (``requests.get`` patched) supplies
deterministic results. No real network or real LLM is hit. Async graphs are
driven with ``asyncio.run``.
"""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from search_agent.base import BaseSearchProvider
from search_agent.config import SearchConfig
from search_agent.providers import register_provider
from search_agent.subgraph import build_search_subgraph
from search_agent.contracts import Citation, SearchResult, WebSearchResponse


class RecordingChat:
    """Deterministic chat model that plays both decompose and consolidate.

    Decompose calls (identified by their system prompt) return a fixed subquery
    list; consolidate calls return a fixed grounding string. ``bind(model=...)``
    returns a clone sharing the recording list, so tests can assert which model
    reached each call as ``(kind, bound_model)``.
    """

    def __init__(self, subqueries, grounding, *, bound_model=None, calls=None):
        self._subqueries = subqueries
        self._grounding = grounding
        self.bound_model = bound_model
        self.calls = calls if calls is not None else []  # list of (kind, bound_model)

    def bind(self, **kwargs):
        return RecordingChat(
            self._subqueries,
            self._grounding,
            bound_model=kwargs.get("model"),
            calls=self.calls,
        )

    async def ainvoke(self, messages, config=None, **kwargs):
        system = next((m.content for m in messages if getattr(m, "type", "") == "system"), "")
        is_consolidate = "consolidat" in system.lower()
        kind = "consolidate" if is_consolidate else "decompose"
        self.calls.append((kind, self.bound_model))
        content = self._grounding if is_consolidate else "\n".join(self._subqueries)
        return AIMessage(content=content)

    def calls_of(self, kind):
        return [c for c in self.calls if c[0] == kind]


def _patch_query_aware_searxng(monkeypatch, per_query):
    def _get(url, timeout=20, **kwargs):
        query = kwargs["params"]["q"]
        payload = {"results": per_query.get(query, [])}
        return SimpleNamespace(status_code=200, json=lambda: payload, text="")

    monkeypatch.setattr("search_agent.providers.searxng.requests.get", _get)


def test_llm_consolidation_uses_injected_llm_output(monkeypatch):
    grounding = "Summary of python.\n- fact one [1]\n- fact two [2]\nSources:\n[1] a\n[2] b"
    fake_llm = RecordingChat(["python history", "python usage"], grounding)
    _patch_query_aware_searxng(
        monkeypatch,
        {
            "python history": [
                {"title": "History", "url": "https://hist.example", "content": "born 1991"},
            ],
            "python usage": [
                {"title": "Usage", "url": "https://usage.example", "content": "web and ml"},
            ],
        },
    )
    config = SearchConfig(
        provider="searxng",
        base_url="http://searx.local",
        consolidation_use_llm=True,
    )
    graph = build_search_subgraph(config, llm=fake_llm)

    out = asyncio.run(graph.ainvoke({"task": "tell me about python"}))

    # consolidate output is exactly what the injected LLM returned.
    assert out["consolidated"] == grounding
    # The consolidate LLM call actually happened.
    assert fake_llm.calls_of("consolidate"), "LLM consolidation should call the injected LLM"
    # Citations survive with contiguous [1][2] references, consistent with the grounding.
    assert [c.reference for c in out["citations"]] == ["[1]", "[2]"]
    assert {c.url for c in out["citations"]} == {
        "https://hist.example",
        "https://usage.example",
    }


def test_consolidation_model_override_targets_consolidate_only(monkeypatch):
    fake_llm = RecordingChat(["sub-a"], "grounding [1]")
    _patch_query_aware_searxng(
        monkeypatch,
        {"sub-a": [{"title": "A", "url": "https://a.example", "content": "c"}]},
    )
    config = SearchConfig(
        provider="searxng",
        base_url="http://searx.local",
        consolidation_use_llm=True,
        consolidation_llm_model="big-cheap-longctx",
    )
    graph = build_search_subgraph(config, llm=fake_llm)

    asyncio.run(graph.ainvoke({"task": "anything"}))

    # The override model reaches the consolidate call...
    (consolidate_call,) = fake_llm.calls_of("consolidate")
    assert consolidate_call[1] == "big-cheap-longctx"
    # ...but decompose keeps the default (no per-call model override).
    (decompose_call,) = fake_llm.calls_of("decompose")
    assert decompose_call[1] is None


@register_provider("answer_probe")
class _AnswerProvider(BaseSearchProvider):
    """A provider that already produces its own answer (supports_answer=True)."""

    requires_api_key = False
    supports_answer = True

    def search(self, query, base_url="", max_results=5, **kwargs):
        return WebSearchResponse(
            query=query,
            answer="provider's own answer",
            provider="answer_probe",
            citations=[Citation(id=1, reference="[1]", url=f"https://x/{query}")],
            search_results=[SearchResult(title=query, url=f"https://x/{query}", snippet="s")],
        )


def test_supports_answer_provider_skips_llm_consolidation():
    # Provider already returns an answer -> LLM consolidation must NOT run,
    # even though consolidation_use_llm=True.
    fake_llm = RecordingChat(["only-sub"], "should not appear")
    config = SearchConfig(provider="answer_probe", consolidation_use_llm=True)
    graph = build_search_subgraph(config, llm=fake_llm)

    out = asyncio.run(graph.ainvoke({"task": "task"}))

    assert fake_llm.calls_of("consolidate") == [], "answer-capable provider must skip LLM consolidation"
    assert out["consolidated"] != "should not appear"
    # decompose still ran on the injected LLM.
    assert fake_llm.calls_of("decompose")


def test_non_answer_provider_triggers_llm_consolidation(monkeypatch):
    # SearXNG has supports_answer=False -> LLM consolidation runs.
    fake_llm = RecordingChat(["only-sub"], "llm grounding [1]")
    _patch_query_aware_searxng(
        monkeypatch,
        {"only-sub": [{"title": "T", "url": "https://t.example", "content": "c"}]},
    )
    config = SearchConfig(
        provider="searxng",
        base_url="http://searx.local",
        consolidation_use_llm=True,
    )
    graph = build_search_subgraph(config, llm=fake_llm)

    out = asyncio.run(graph.ainvoke({"task": "task"}))

    assert fake_llm.calls_of("consolidate"), "searxng must trigger LLM consolidation"
    assert out["consolidated"] == "llm grounding [1]"
