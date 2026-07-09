"""Slice H — termination honesty and per-claim convergence, at the highest seam.

Every test observes only ``ResearchInput { task } -> ResearchOutput``. The thin
Controller must never dress "not solved" up as "done": it emits DONE only when
all critical claims are resolved (VERIFIED or REFUTED), EXHAUSTED otherwise, and
the terminal decision is *explainable* — traceable to exactly which critical
claims were still unresolved when it stopped (``blocking_claim_ids``). A single
unfindable claim is abandoned after its budget and must not drag the rest of the
research down. Fakes are injected so the whole flat graph runs reproducibly with
no network or LLM; async is driven with ``asyncio.run`` per the repo convention.
"""

import asyncio

from search_agent.contracts import Citation, SearchResult
from search_agent.edre import (
    ClaimStatus,
    EDREConfig,
    EvidenceClaim,
    Importance,
    build_research_graph,
    derive_status,
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


def _critical_pair_planner():
    async def planner(_task, _config):
        return [
            EvidenceClaim(id="c1", hypothesis="h1", importance=Importance.CRITICAL),
            EvidenceClaim(id="c2", hypothesis="h2", importance=Importance.OPTIONAL),
        ]

    return planner


def test_exhausted_termination_names_the_blocking_critical_claim():
    # An unresolvable critical claim forces an honest EXHAUSTED. The decision must
    # be explainable: blocking_claim_ids names exactly the critical claim(s) whose
    # unresolved status triggered the terminal state — here the abandoned c1.
    async def scorer_silent(_claims, docs):
        return [{} for _ in docs]

    config = EDREConfig(max_attempts=2, max_loops=25)
    graph = build_research_graph(
        config,
        planner=_critical_pair_planner(),
        search_fn=_search_returns_one,
        scorer=scorer_silent,
    )

    out = asyncio.run(graph.ainvoke({"task": "unknowable?"}))["output"]

    assert out.research_summary.terminal == "EXHAUSTED"
    assert out.research_summary.critical_all_resolved is False
    # The termination is traceable to the unresolved critical claim.
    assert out.research_summary.blocking_claim_ids == ["c1"]
    by_id = {v.claim_id: v for v in out.evidence}
    assert by_id["c1"].status == "ABANDONED"


def test_done_termination_has_no_blocking_claims():
    # When every critical claim resolves, DONE is not "triggered by" any claim:
    # blocking_claim_ids is empty. This is the honest counterpart to EXHAUSTED —
    # an empty offender list is what distinguishes a real DONE from a faked one.
    async def scorer_verifies(claims, docs):
        return [{c.id: 0.9 for c in claims} for _ in docs]

    config = EDREConfig(verify_threshold=0.7, refute_threshold=0.7, max_loops=5)
    graph = build_research_graph(
        config,
        planner=_critical_pair_planner(),
        search_fn=_search_returns_one,
        scorer=scorer_verifies,
    )

    out = asyncio.run(graph.ainvoke({"task": "does X hold?"}))["output"]

    assert out.research_summary.terminal == "DONE"
    assert out.research_summary.critical_all_resolved is True
    assert out.research_summary.blocking_claim_ids == []


def _recording_query_generator(record: list[list[str]]):
    # Mirrors the real generator's "only unresolved claims" filter while recording
    # which claim ids were asked for each loop, so a test can observe that a
    # resolved/abandoned claim stops consuming search budget.
    unresolved = {ClaimStatus.NOT_STARTED, ClaimStatus.PARTIAL}

    async def generator(state, config):
        asked = [
            c.id
            for c in state["evidence_plan"]
            if derive_status(c, config) in unresolved
        ]
        record.append(asked)
        return {cid: ["q"] for cid in asked}

    return generator


def test_abandoned_claim_does_not_drag_down_a_converging_claim():
    # Two critical claims: c1 is unfindable, c2 is findable. c1 must be abandoned
    # after its per-claim budget and stop consuming search budget, while c2
    # converges VERIFIED on its own. The loop must stop promptly at EXHAUSTED,
    # far below max_loops, blaming only c1 — the dead claim never drags c2 down.
    async def scorer_only_c2(_claims, docs):
        return [{"c2": 0.9} for _ in docs]

    async def two_critical(_task, _config):
        return [
            EvidenceClaim(id="c1", hypothesis="h1", importance=Importance.CRITICAL),
            EvidenceClaim(id="c2", hypothesis="h2", importance=Importance.CRITICAL),
        ]

    record: list[list[str]] = []
    config = EDREConfig(max_attempts=2, max_loops=25)
    graph = build_research_graph(
        config,
        planner=two_critical,
        query_generator=_recording_query_generator(record),
        search_fn=_search_returns_one,
        scorer=scorer_only_c2,
    )

    out = asyncio.run(graph.ainvoke({"task": "one dead, one alive"}))["output"]

    assert out.research_summary.terminal == "EXHAUSTED"
    by_id = {v.claim_id: v for v in out.evidence}
    # c2 converged independently; c1 was honestly abandoned.
    assert by_id["c2"].status == "VERIFIED"
    assert by_id["c1"].status == "ABANDONED"
    # Only the dead claim is blamed for stopping.
    assert out.research_summary.blocking_claim_ids == ["c1"]
    # Bounded by c1's max_attempts, nowhere near the far larger max_loops.
    assert out.loop_count == 2
    # Loop 1 searched both; once c2 verified it dropped out and only c1 was
    # retried — the abandoned claim never forced re-search of the resolved one.
    assert record == [["c1", "c2"], ["c1"]]


def test_max_loops_cap_exhausts_with_partial_critical_claim():
    # With a generous per-claim budget, a claim that never resolves stays PARTIAL
    # rather than ABANDONED — so it is the loop cap, not the per-claim budget,
    # that stops the research. It stops honestly as EXHAUSTED at exactly max_loops,
    # blaming the still-unresolved critical claim (PARTIAL counts as a blocker).
    async def scorer_silent(_claims, docs):
        return [{} for _ in docs]

    config = EDREConfig(max_attempts=25, max_loops=3)
    graph = build_research_graph(
        config,
        planner=_critical_pair_planner(),
        search_fn=_search_returns_one,
        scorer=scorer_silent,
    )

    out = asyncio.run(graph.ainvoke({"task": "slow to answer?"}))["output"]

    assert out.research_summary.terminal == "EXHAUSTED"
    # The loop cap fired at exactly max_loops (not the per-claim budget).
    assert out.loop_count == 3
    by_id = {v.claim_id: v for v in out.evidence}
    # Still-searchable but unresolved: PARTIAL, not ABANDONED.
    assert by_id["c1"].status == "PARTIAL"
    assert out.research_summary.abandoned == 0
    # A PARTIAL critical claim is a legitimate blocker for the terminal decision.
    assert out.research_summary.blocking_claim_ids == ["c1"]
