"""Slice B — EDRE walking skeleton, tested at the single highest seam.

Tests observe behavior only at ``ResearchInput { task } -> ResearchOutput``. The
intelligent components (Planner, ``search_many``, scorer) are replaced by
injectable fakes so the whole flat graph — plan, the research loop, control's
DONE/EXHAUSTED routing, and deterministic finalize — is exercised reproducibly
without any network or LLM. Async is driven with ``asyncio.run`` (no plugin),
matching the repo's existing test convention.
"""

import asyncio

from search_agent.contracts import Citation, SearchResult
from search_agent.edre import (
    EDREConfig,
    EvidenceClaim,
    Importance,
    build_research_graph,
)
from search_agent.retrieval import RetrievedDocument


def _doc(url: str, title: str, cid: int) -> RetrievedDocument:
    return RetrievedDocument(
        result=SearchResult(title=title, url=url, snippet="snippet"),
        citation=Citation(
            id=cid, reference=f"[{cid}]", url=url, title=title, snippet="snippet"
        ),
        source_queries=["q"],
    )


async def _search_returns_one(_search_config, _queries):
    return [_doc("https://a.example", "A", 1)]


def _two_claims_planner():
    async def planner(_task, _config):
        return [
            EvidenceClaim(id="c1", hypothesis="h1", importance=Importance.CRITICAL),
            EvidenceClaim(id="c2", hypothesis="h2", importance=Importance.OPTIONAL),
        ]

    return planner


def test_all_critical_verified_terminates_done():
    # A scorer that returns strong positive support for every claim on every doc
    # should verify the critical claim in the first loop and terminate DONE.
    async def scorer_verifies(claims, docs):
        return [{c.id: 0.9 for c in claims} for _ in docs]

    config = EDREConfig(verify_threshold=0.7, refute_threshold=0.7, max_loops=5)
    graph = build_research_graph(
        config,
        planner=_two_claims_planner(),
        search_fn=_search_returns_one,
        scorer=scorer_verifies,
    )

    out = asyncio.run(graph.ainvoke({"task": "does X hold?"}))["output"]

    # Terminal state is the successful, deterministic DONE snapshot.
    assert out.research_summary.terminal == "DONE"
    assert out.research_summary.critical_all_resolved is True
    # Resolved in a single loop; the loop did not spin.
    assert out.loop_count == 1
    assert out.research_summary.loop_count == 1
    # The fixed claim set (2 planned) is fully reflected in the evidence.
    assert len(out.evidence) == 2
    by_id = {v.claim_id: v for v in out.evidence}
    assert by_id["c1"].status == "VERIFIED"
    assert by_id["c1"].importance == "CRITICAL"
    # Every conclusion is traceable to at least one citation.
    assert out.citations


def test_unresolvable_critical_exhausts_and_abandons_without_hanging():
    # A scorer that never produces support cannot resolve the critical claim.
    # After max_attempts the claim is ABANDONED and the loop stops as EXHAUSTED
    # — well before the (higher) max_loops backstop, and it must not hang.
    async def scorer_silent(_claims, docs):
        return [{} for _ in docs]

    config = EDREConfig(max_attempts=2, max_loops=25)
    graph = build_research_graph(
        config,
        planner=_two_claims_planner(),
        search_fn=_search_returns_one,
        scorer=scorer_silent,
    )

    out = asyncio.run(graph.ainvoke({"task": "unknowable?"}))["output"]

    assert out.research_summary.terminal == "EXHAUSTED"
    assert out.research_summary.critical_all_resolved is False
    by_id = {v.claim_id: v for v in out.evidence}
    # The critical claim is honestly ABANDONED (not faked VERIFIED).
    assert by_id["c1"].status == "ABANDONED"
    assert out.research_summary.abandoned >= 1
    # Bounded by per-claim max_attempts, not the far larger max_loops.
    assert out.loop_count == 2


def test_refuted_critical_is_done_and_distinct_from_abandoned():
    # Strong *negative* support falsifies the critical claim. Falsification is a
    # successful discovery: the claim is REFUTED (not ABANDONED) and, since the
    # critical claim is now resolved, the research terminates DONE.
    async def scorer_refutes(claims, docs):
        return [{c.id: -0.9 for c in claims} for _ in docs]

    config = EDREConfig(verify_threshold=0.7, refute_threshold=0.7, max_loops=5)
    graph = build_research_graph(
        config,
        planner=_two_claims_planner(),
        search_fn=_search_returns_one,
        scorer=scorer_refutes,
    )

    out = asyncio.run(graph.ainvoke({"task": "is the false premise true?"}))["output"]

    assert out.research_summary.terminal == "DONE"
    assert out.research_summary.critical_all_resolved is True
    by_id = {v.claim_id: v for v in out.evidence}
    assert by_id["c1"].status == "REFUTED"
    # REFUTED and ABANDONED are reported as distinct verdict categories.
    assert out.research_summary.refuted >= 1
    assert out.research_summary.abandoned == 0
    # The refuting document is traceable with its signed (negative) support.
    assert by_id["c1"].supporting_documents[0].support == -0.9


def test_search_node_drives_search_many_and_claim_set_is_fixed():
    # The search node must take data through the search_many seam, driven by
    # queries derived from the planned claims. A 3-claim plan must surface as
    # exactly 3 verdicts — the claim set is fixed at plan time, never grown or
    # pruned by the loop.
    calls: list[list[str]] = []

    async def recording_search(_search_config, queries):
        calls.append(list(queries))
        return [_doc("https://a.example", "A", 1)]

    async def three_claims(_task, _config):
        return [
            EvidenceClaim(id="c1", hypothesis="alpha", importance=Importance.CRITICAL),
            EvidenceClaim(id="c2", hypothesis="beta", importance=Importance.OPTIONAL),
            EvidenceClaim(id="c3", hypothesis="gamma", importance=Importance.OPTIONAL),
        ]

    async def scorer_verifies(claims, docs):
        return [{c.id: 0.9 for c in claims} for _ in docs]

    config = EDREConfig(max_loops=5)
    graph = build_research_graph(
        config,
        planner=three_claims,
        search_fn=recording_search,
        scorer=scorer_verifies,
    )

    out = asyncio.run(graph.ainvoke({"task": "three things"}))["output"]

    # search_many seam was actually driven, with queries from the claims.
    assert calls, "search node did not call the search_many seam"
    assert set(calls[0]) == {"alpha", "beta", "gamma"}
    # Fixed claim set: exactly the 3 planned claims appear as verdicts.
    assert [v.claim_id for v in out.evidence] == ["c1", "c2", "c3"]
