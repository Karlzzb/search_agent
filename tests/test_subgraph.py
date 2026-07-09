"""Slice 2 — minimal no-LLM search subgraph.

Tests observe behavior only at the agreed seam: the compiled subgraph via
``build_search_subgraph(config, llm=None).ainvoke({"task": ...})``. SearXNG is
faked by patching ``requests.get``; no real network is hit. Async graphs are
driven with ``asyncio.run`` so no pytest plugin is required.
"""

import asyncio
from types import SimpleNamespace

from search_agent.config import SearchConfig
from search_agent.subgraph import build_search_subgraph


def _patch_searxng(monkeypatch, payload):
    def _get(url, timeout=20, **kwargs):
        return SimpleNamespace(status_code=200, json=lambda: payload, text="")

    monkeypatch.setattr("search_agent.providers.searxng.requests.get", _get)


def test_minimal_subgraph_returns_consolidated_and_citations(monkeypatch):
    _patch_searxng(
        monkeypatch,
        {
            "results": [
                {
                    "title": "Python",
                    "url": "https://python.org",
                    "content": "The official Python site",
                    "engine": "google",
                },
                {
                    "title": "Wikipedia",
                    "url": "https://en.wikipedia.org/Python",
                    "content": "encyclopedia entry",
                    "engine": "bing",
                },
            ]
        },
    )
    config = SearchConfig(
        provider="searxng",
        base_url="http://searx.local",
        consolidation_use_llm=False,
    )
    graph = build_search_subgraph(config, llm=None)

    out = asyncio.run(graph.ainvoke({"task": "what is python"}))

    assert isinstance(out["consolidated"], str)
    assert "python.org" in out["consolidated"]

    citations = out["citations"]
    assert [c.url for c in citations] == [
        "https://python.org",
        "https://en.wikipedia.org/Python",
    ]
    assert citations[0].reference == "[1]"
    assert citations[1].reference == "[2]"
    assert citations[0].title == "Python"
    assert citations[0].snippet == "The official Python site"


def test_output_schema_hides_internal_keys(monkeypatch):
    _patch_searxng(
        monkeypatch,
        {"results": [{"title": "T", "url": "https://a.example", "content": "c"}]},
    )
    config = SearchConfig(provider="searxng", base_url="http://searx.local")
    graph = build_search_subgraph(config, llm=None)

    out = asyncio.run(graph.ainvoke({"task": "q"}))

    # Parent graph only sees the contract, never the intermediate keys.
    assert set(out.keys()) == {"consolidated", "citations"}
    assert "subqueries" not in out
    assert "raw_results" not in out
    assert "task" not in out


def test_no_llm_path_uses_template_format(monkeypatch):
    _patch_searxng(
        monkeypatch,
        {
            "results": [
                {"title": "First", "url": "https://one.example", "content": "snippet one"},
            ]
        },
    )
    config = SearchConfig(
        provider="searxng",
        base_url="http://searx.local",
        consolidation_use_llm=False,
    )
    graph = build_search_subgraph(config, llm=None)

    out = asyncio.run(graph.ainvoke({"task": "topic"}))

    consolidated = out["consolidated"]
    assert '### Search Results for "topic"' in consolidated
    assert "**[1] First**" in consolidated
    assert "snippet one" in consolidated


def test_subgraph_nodes_are_individually_visible():
    config = SearchConfig(provider="searxng", base_url="http://searx.local")
    graph = build_search_subgraph(config, llm=None)

    nodes = set(graph.get_graph().nodes)
    assert {"decompose", "fan_out_search", "consolidate"} <= nodes


def test_factory_is_exported_from_package():
    import search_agent

    assert search_agent.build_search_subgraph is build_search_subgraph
