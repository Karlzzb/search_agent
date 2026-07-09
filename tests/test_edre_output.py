"""EDRE Slice I — deterministic output snapshot + doc_relevance evidence sorting.

Observed only at the highest seam ``ResearchInput -> ResearchOutput``. The rerank
gate is the *real* ``make_rerank_gate`` wrapped around a fake cross-encoder so the
per-document TaskMatch / QueryMatch that feed ``doc_relevance`` are exercised
deterministically and offline; ``search_fn`` and ``scorer`` are fakes.
"""

from __future__ import annotations

import asyncio

from search_agent.contracts import Citation, SearchResult
from search_agent.edre.graph import build_research_graph
from search_agent.edre.models import EDREConfig, EvidenceClaim, Importance
from search_agent.edre.reranker import make_rerank_gate
from search_agent.retrieval import RetrievedDocument


def _doc(url: str, title: str, cid: int) -> RetrievedDocument:
    return RetrievedDocument(
        result=SearchResult(title=title, url=url, snippet="snippet"),
        citation=Citation(
            id=cid, reference=f"[{cid}]", url=url, title=title, snippet="snippet"
        ),
        source_queries=["q"],
    )


def _cross_encoder(table):
    """A fake ``Reranker``: look up ``(left_text, url) -> score``, default 0.0."""

    def rerank(left_text, docs):
        return [table.get((left_text, d.citation.url), 0.0) for d in docs]

    return rerank


def _one_critical_planner():
    async def planner(_task, _config):
        return [EvidenceClaim(id="c1", hypothesis="h1", importance=Importance.CRITICAL)]

    return planner


# Doc A: low task/query match, high signed support.
# Doc B: high task/query match, low signed support.
_MATCH_TABLE = {
    ("t", "https://a.example"): 0.2,
    ("q", "https://a.example"): 0.2,
    ("t", "https://b.example"): 0.95,
    ("q", "https://b.example"): 0.95,
}
_SUPPORT_TABLE = {"https://a.example": 0.9, "https://b.example": 0.3}


async def _search_two(_config, _queries):
    # Insertion order is A then B.
    return [_doc("https://a.example", "A", 1), _doc("https://b.example", "B", 2)]


async def _scorer_by_url(claims, docs):
    return [{c.id: _SUPPORT_TABLE[d.citation.url] for c in claims} for d in docs]


def _run(config):
    graph = build_research_graph(
        config,
        planner=_one_critical_planner(),
        search_fn=_search_two,
        reranker=make_rerank_gate(_cross_encoder(_MATCH_TABLE), config),
        scorer=_scorer_by_url,
    )
    return asyncio.run(graph.ainvoke({"task": "t"}))["output"]


def test_evidence_sorted_by_doc_relevance_not_insertion_or_support():
    # Weights 1/1/1: match terms dominate for B (0.95+0.95+0.3=2.2) over A
    # (0.2+0.2+0.9=1.3), so B ranks first — the reverse of both insertion order
    # (A,B) and support-only order (A has the stronger |support|).
    config = EDREConfig(
        rerank_threshold=0.05,
        rerank_w1=0.5,
        rerank_w2=0.5,
        doc_relevance_w1=1.0,
        doc_relevance_w2=1.0,
        doc_relevance_w3=1.0,
        verify_threshold=0.7,
        max_loops=3,
    )
    out = _run(config)

    verdict = out.evidence[0]
    assert verdict.status == "VERIFIED"
    ordered_urls = [ref.citation.url for ref in verdict.supporting_documents]
    assert ordered_urls == ["https://b.example", "https://a.example"]
    # Citations follow the same relevance ordering.
    assert [c.url for c in out.citations] == [
        "https://b.example",
        "https://a.example",
    ]


def test_doc_relevance_weights_are_configurable():
    # Same evidence, but w3 dominates: A's strong |support| (10*0.9=9.0) outranks
    # B's strong match, flipping the order back to A first. Proves the support
    # term participates and the weights are injectable.
    config = EDREConfig(
        rerank_threshold=0.05,
        rerank_w1=0.5,
        rerank_w2=0.5,
        doc_relevance_w1=1.0,
        doc_relevance_w2=1.0,
        doc_relevance_w3=10.0,
        verify_threshold=0.7,
        max_loops=3,
    )
    out = _run(config)

    ordered_urls = [ref.citation.url for ref in out.evidence[0].supporting_documents]
    assert ordered_urls == ["https://a.example", "https://b.example"]


def test_research_summary_is_deterministic_and_counts_match_verdicts():
    config = EDREConfig(
        rerank_threshold=0.05,
        rerank_w1=0.5,
        rerank_w2=0.5,
        verify_threshold=0.7,
        max_loops=3,
    )
    first = _run(config)
    second = _run(config)

    assert first.research_summary == second.research_summary

    summary = first.research_summary
    statuses = [v.status for v in first.evidence]
    assert summary.verified == statuses.count("VERIFIED")
    assert summary.refuted == statuses.count("REFUTED")
    assert summary.abandoned == statuses.count("ABANDONED")
