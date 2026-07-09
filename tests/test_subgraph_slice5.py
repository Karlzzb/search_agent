"""Slice 5 — fallbacks & degradation.

Tests observe behavior only at the agreed seams:
  * the compiled subgraph via ``build_search_subgraph(config, llm=fake).ainvoke``
    (provider fallback, single-path degradation, all-fail degradation, dedup),
  * ``build_search_subgraph(config)`` raising at build time on an invalid
    ``base_url``.

Fakes are injected without touching the network: a deterministic chat model for
decompose, a query-aware fake SearXNG (``requests.get`` patched), and a fake
``duckduckgo`` provider swapped into the registry. Async graphs are driven with
``asyncio.run`` so no pytest plugin is required.
"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import search_agent.providers as providers_pkg
from search_agent.base import BaseSearchProvider
from search_agent.config import SearchConfig
from search_agent.subgraph import build_search_subgraph
from search_agent.contracts import Citation, SearchResult, WebSearchResponse


class FakeChat:
    """Deterministic chat model: decompose returns a fixed newline list."""

    def __init__(self, subqueries):
        self._subqueries = subqueries

    async def ainvoke(self, messages, config=None, **kwargs):
        return AIMessage(content="\n".join(self._subqueries))


def _patch_query_aware_searxng(monkeypatch, per_query):
    """Fake SearXNG returning a payload per query; missing keys raise."""

    def _get(url, timeout=20, **kwargs):
        query = kwargs["params"]["q"]
        if query not in per_query:
            raise RuntimeError(f"searxng boom for {query}")
        payload = {"results": per_query[query]}
        return SimpleNamespace(status_code=200, json=lambda: payload, text="")

    monkeypatch.setattr("search_agent.providers.searxng.requests.get", _get)


def _swap_duckduckgo(monkeypatch, rows):
    """Replace the registered ``duckduckgo`` provider with a fake."""

    class _FakeDDG(BaseSearchProvider):
        requires_api_key = False

        def search(self, query, base_url="", max_results=5, **kwargs):
            results = [
                SearchResult(title=r["title"], url=r["url"], snippet=r.get("content", ""))
                for r in rows
            ]
            return WebSearchResponse(
                query=query,
                answer="",
                provider="duckduckgo",
                search_results=results,
            )

    monkeypatch.setitem(providers_pkg._PROVIDERS, "duckduckgo", _FakeDDG)


def test_searxng_without_base_url_falls_back_to_duckduckgo(monkeypatch):
    _swap_duckduckgo(
        monkeypatch,
        [{"title": "DDG", "url": "https://ddg.example", "content": "from duckduckgo"}],
    )
    # provider is searxng but no base_url -> subgraph must fall back to duckduckgo.
    config = SearchConfig(provider="searxng", base_url=None)
    graph = build_search_subgraph(config, llm=FakeChat(["the task"]))

    out = asyncio.run(graph.ainvoke({"task": "the task"}))

    assert {c.url for c in out["citations"]} == {"https://ddg.example"}


def test_single_subquery_failure_is_skipped_not_fatal(monkeypatch):
    # "good" returns a result; "bad" raises. The subgraph must survive and keep
    # the surviving path's citations.
    _patch_query_aware_searxng(
        monkeypatch,
        {"good": [{"title": "Good", "url": "https://good.example", "content": "ok"}]},
    )
    config = SearchConfig(provider="searxng", base_url="http://searx.local")
    graph = build_search_subgraph(config, llm=FakeChat(["good", "bad"]))

    out = asyncio.run(graph.ainvoke({"task": "mixed"}))

    assert {c.url for c in out["citations"]} == {"https://good.example"}
    assert [c.reference for c in out["citations"]] == ["[1]"]


def test_all_subqueries_failing_degrades_to_no_results(monkeypatch):
    # Every query raises -> no path survives.
    _patch_query_aware_searxng(monkeypatch, {})
    config = SearchConfig(provider="searxng", base_url="http://searx.local")
    graph = build_search_subgraph(config, llm=FakeChat(["a", "b"]))

    out = asyncio.run(graph.ainvoke({"task": "everything fails"}))

    assert out["citations"] == []
    assert "No results" in out["consolidated"]


def test_duplicate_urls_are_deduped_with_contiguous_references(monkeypatch):
    # Two subqueries surface the same page plus one unique page each.
    dup = {"title": "Shared", "url": "https://dup.example", "content": "same page"}
    _patch_query_aware_searxng(
        monkeypatch,
        {
            "q1": [dup, {"title": "A", "url": "https://a.example", "content": "a"}],
            "q2": [dup, {"title": "B", "url": "https://b.example", "content": "b"}],
        },
    )
    config = SearchConfig(provider="searxng", base_url="http://searx.local")
    graph = build_search_subgraph(config, llm=FakeChat(["q1", "q2"]))

    out = asyncio.run(graph.ainvoke({"task": "dedup"}))

    urls = [c.url for c in out["citations"]]
    assert urls.count("https://dup.example") == 1
    assert set(urls) == {"https://dup.example", "https://a.example", "https://b.example"}
    # References stay contiguous after dedup.
    assert [c.reference for c in out["citations"]] == ["[1]", "[2]", "[3]"]
    assert [c.id for c in out["citations"]] == [1, 2, 3]


@pytest.mark.parametrize("bad_url", ["ftp://searx.local", "gopher://searx.local"])
def test_invalid_searxng_base_url_is_rejected_at_build_time(bad_url):
    # An invalid base_url must surface as a clear error at build time, not be
    # silently swallowed by the per-path degradation.
    config = SearchConfig(provider="searxng", base_url=bad_url)
    with pytest.raises(ValueError):
        build_search_subgraph(config, llm=FakeChat(["x"]))


def test_consolidator_keeps_jinja_autoescape_enabled():
    # Security posture regression: rendering must keep autoescape on.
    from search_agent.consolidation import AnswerConsolidator

    consolidator = AnswerConsolidator(use_llm=False)
    assert consolidator.jinja_env.autoescape is True
