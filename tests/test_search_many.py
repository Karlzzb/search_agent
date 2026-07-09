"""Task A — the ``search_many`` retrieval adapter (EDRE's sole seam).

Tests observe behavior only at the adapter's public boundary:
``search_many(config, queries) -> list[RetrievedDocument]``. The network is
faked without a real LLM/HTTP call: a query-aware fake SearXNG (``requests.get``
patched) and a fake ``duckduckgo`` provider swapped into the registry. Async is
driven with ``asyncio.run`` so no pytest plugin is required.
"""

import asyncio
from types import SimpleNamespace

import search_agent.providers as providers_pkg
from search_agent.base import BaseSearchProvider
from search_agent.config import SearchConfig
from search_agent.contracts import SearchResult, WebSearchResponse
from search_agent.retrieval import RetrievedDocument, search_many


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
                query=query, answer="", provider="duckduckgo", search_results=results
            )

    monkeypatch.setitem(providers_pkg._PROVIDERS, "duckduckgo", _FakeDDG)


def test_fans_out_queries_and_dedupes_documents(monkeypatch):
    # Two queries surface the same shared page plus one unique page each.
    dup = {"title": "Shared", "url": "https://dup.example", "content": "same"}
    _patch_query_aware_searxng(
        monkeypatch,
        {
            "q1": [dup, {"title": "A", "url": "https://a.example", "content": "a"}],
            "q2": [dup, {"title": "B", "url": "https://b.example", "content": "b"}],
        },
    )
    config = SearchConfig(provider="searxng", base_url="http://searx.local")

    docs = asyncio.run(search_many(config, ["q1", "q2"]))

    assert all(isinstance(d, RetrievedDocument) for d in docs)
    urls = [d.result.url for d in docs]
    # URL dedup across fan-out paths: the shared page appears exactly once.
    assert urls.count("https://dup.example") == 1
    assert set(urls) == {"https://dup.example", "https://a.example", "https://b.example"}
    # Contiguous citation numbering after dedup; no LLM answer synthesis here.
    assert [d.citation.reference for d in docs] == ["[1]", "[2]", "[3]"]
    assert [d.citation.id for d in docs] == [1, 2, 3]


def test_query_attribution_records_every_hitting_query(monkeypatch):
    # The shared page is surfaced by both queries; the unique pages by one each.
    dup = {"title": "Shared", "url": "https://dup.example", "content": "same"}
    _patch_query_aware_searxng(
        monkeypatch,
        {
            "q1": [dup, {"title": "A", "url": "https://a.example", "content": "a"}],
            "q2": [dup, {"title": "B", "url": "https://b.example", "content": "b"}],
        },
    )
    config = SearchConfig(provider="searxng", base_url="http://searx.local")

    docs = asyncio.run(search_many(config, ["q1", "q2"]))
    by_url = {d.result.url: d for d in docs}

    # The shared page traces back to *every* query that surfaced it.
    assert by_url["https://dup.example"].source_queries == ["q1", "q2"]
    assert by_url["https://a.example"].source_queries == ["q1"]
    assert by_url["https://b.example"].source_queries == ["q2"]


def test_results_without_url_are_always_kept(monkeypatch):
    # Two url-less results from different queries must both survive (no dedup key).
    _patch_query_aware_searxng(
        monkeypatch,
        {
            "q1": [{"title": "NoUrlA", "url": "", "content": "a"}],
            "q2": [{"title": "NoUrlB", "url": "", "content": "b"}],
        },
    )
    config = SearchConfig(provider="searxng", base_url="http://searx.local")

    docs = asyncio.run(search_many(config, ["q1", "q2"]))

    titles = sorted(d.result.title for d in docs)
    assert titles == ["NoUrlA", "NoUrlB"]
    assert [d.citation.reference for d in docs] == ["[1]", "[2]"]


def test_single_query_failure_is_skipped_not_fatal(monkeypatch):
    # "good" returns a result; "bad" raises. The adapter must survive and keep
    # the surviving path's documents.
    _patch_query_aware_searxng(
        monkeypatch,
        {"good": [{"title": "Good", "url": "https://good.example", "content": "ok"}]},
    )
    config = SearchConfig(provider="searxng", base_url="http://searx.local")

    docs = asyncio.run(search_many(config, ["good", "bad"]))

    assert {d.result.url for d in docs} == {"https://good.example"}
    assert docs[0].source_queries == ["good"]


def test_searxng_without_base_url_fails_over_to_duckduckgo(monkeypatch):
    # provider is searxng but no base_url -> the adapter must fail over to
    # duckduckgo internally, without the caller making a provider decision.
    _swap_duckduckgo(
        monkeypatch,
        [{"title": "DDG", "url": "https://ddg.example", "content": "from duckduckgo"}],
    )
    config = SearchConfig(provider="searxng", base_url=None)

    docs = asyncio.run(search_many(config, ["anything"]))

    assert {d.result.url for d in docs} == {"https://ddg.example"}
