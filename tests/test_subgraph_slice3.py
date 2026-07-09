"""Slice 3 — LLM decompose + parallel fan-out.

Tests observe behavior only at the agreed seam: the compiled subgraph via
``build_search_subgraph(config, llm=fake).ainvoke({"task": ...})``. Two fakes are
injected: a deterministic LangChain-shaped chat model (decompose returns a fixed
subquery set) and a query-aware fake SearXNG (``requests.get`` patched). No real
network or real LLM is hit. Async graphs are driven with ``asyncio.run``.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from search_agent.base import BaseSearchProvider
from search_agent.config import SearchConfig
from search_agent.providers import register_provider
from search_agent.subgraph import build_search_subgraph
from search_agent.contracts import Citation, SearchResult, WebSearchResponse


class FakeChat:
    """Deterministic chat model: decompose returns a fixed newline list."""

    def __init__(self, subqueries):
        self._subqueries = subqueries
        self.prompts = []

    async def ainvoke(self, messages, config=None, **kwargs):
        self.prompts.append(messages[-1].content)
        return AIMessage(content="\n".join(self._subqueries))


def _patch_query_aware_searxng(monkeypatch, per_query):
    """Fake SearXNG that returns a different payload per query string."""

    def _get(url, timeout=20, **kwargs):
        query = kwargs["params"]["q"]
        payload = {"results": per_query.get(query, [])}
        return SimpleNamespace(status_code=200, json=lambda: payload, text="")

    monkeypatch.setattr("search_agent.providers.searxng.requests.get", _get)


def test_llm_decompose_fans_out_and_aggregates_all_paths(monkeypatch):
    fake_llm = FakeChat(["python history", "python usage"])
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
    config = SearchConfig(provider="searxng", base_url="http://searx.local")
    graph = build_search_subgraph(config, llm=fake_llm)

    out = asyncio.run(graph.ainvoke({"task": "tell me about python"}))

    # decompose used the injected LLM.
    assert fake_llm.prompts, "decompose should call the injected LLM"

    # Both subquery paths are aggregated into the citations.
    urls = {c.url for c in out["citations"]}
    assert urls == {"https://hist.example", "https://usage.example"}
    # References are renumbered contiguously across the merged paths.
    assert [c.reference for c in out["citations"]] == ["[1]", "[2]"]


# A shared barrier proves the two subquery searches run concurrently: each call
# blocks until *both* have arrived. If fan-out were sequential, the first call
# would wait forever and trip the barrier's timeout.
_CONCURRENCY_BARRIER = threading.Barrier(2, timeout=5)


@register_provider("barrier_probe")
class _BarrierProbeProvider(BaseSearchProvider):
    requires_api_key = False

    def search(self, query, base_url="", max_results=5, **kwargs):
        _CONCURRENCY_BARRIER.wait()
        return WebSearchResponse(
            query=query,
            answer="",
            provider="barrier_probe",
            citations=[Citation(id=1, reference="[1]", url=f"https://x/{query}")],
            search_results=[SearchResult(title=query, url=f"https://x/{query}", snippet="")],
        )


def test_fan_out_runs_subqueries_concurrently():
    _CONCURRENCY_BARRIER.reset()
    fake_llm = FakeChat(["q-alpha", "q-beta"])
    config = SearchConfig(provider="barrier_probe")
    graph = build_search_subgraph(config, llm=fake_llm)

    # Both paths must be in-flight simultaneously for the barrier to release.
    out = asyncio.run(graph.ainvoke({"task": "concurrent task"}))

    urls = {c.url for c in out["citations"]}
    assert urls == {"https://x/q-alpha", "https://x/q-beta"}


def test_no_injected_llm_resolves_default_adapter(monkeypatch):
    # When no llm is injected, the subgraph pulls the built-in default adapter.
    fake_default = FakeChat(["sub-one", "sub-two"])
    monkeypatch.setattr("search_agent.subgraph.default_chat_model", lambda: fake_default)
    _patch_query_aware_searxng(
        monkeypatch,
        {
            "sub-one": [{"title": "One", "url": "https://one.example", "content": "c1"}],
            "sub-two": [{"title": "Two", "url": "https://two.example", "content": "c2"}],
        },
    )
    config = SearchConfig(provider="searxng", base_url="http://searx.local")
    graph = build_search_subgraph(config, llm=None)

    out = asyncio.run(graph.ainvoke({"task": "anything"}))

    assert fake_default.prompts, "default adapter should drive decompose"
    assert {c.url for c in out["citations"]} == {
        "https://one.example",
        "https://two.example",
    }


def test_no_llm_and_no_default_degrades_to_single_query(monkeypatch):
    # No injected llm and no default available -> decompose degrades to [task].
    monkeypatch.setattr("search_agent.subgraph.default_chat_model", lambda: None)
    _patch_query_aware_searxng(
        monkeypatch,
        {"just the task": [{"title": "T", "url": "https://t.example", "content": "c"}]},
    )
    config = SearchConfig(provider="searxng", base_url="http://searx.local")
    graph = build_search_subgraph(config, llm=None)

    out = asyncio.run(graph.ainvoke({"task": "just the task"}))

    assert {c.url for c in out["citations"]} == {"https://t.example"}
