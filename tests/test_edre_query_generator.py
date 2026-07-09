"""Slice D — Query Generator, tested at the single highest seam.

The real Query Generator is an LLM call that turns each *unresolved*
``EvidenceClaim`` into ``queries_per_claim`` fresh-angle search queries, skipping
resolved (VERIFIED/REFUTED) and abandoned claims so budget is never wasted on
work already settled. We observe only ``ResearchInput -> ResearchOutput``: inject
``query_generator=make_llm_query_generator(FakeQueryChat())`` plus a *recording*
search seam and a scorer that decides which claims resolve, then assert which
queries reach ``search_many`` across loops. Async is driven with ``asyncio.run``
(no plugin), matching the repo's existing convention.
"""

import asyncio
import re

from langchain_core.messages import AIMessage

from search_agent.contracts import Citation, SearchResult
from search_agent.edre import EDREConfig, EvidenceClaim, Importance, build_research_graph
from search_agent.edre.query_generator import make_llm_query_generator
from search_agent.retrieval import RetrievedDocument


class FakeQueryChat:
    """Fake query LLM: returns claim- and round-specific queries, one per line.

    Extracts the hypothesis and retry round embedded in the prompt so the queries
    it emits are traceable to the originating claim and loop. Always returns more
    lines than any sane ``queries_per_claim`` so the generator's own truncation is
    what's under test.
    """

    def __init__(self):
        self.prompts: list[str] = []

    async def ainvoke(self, messages, config=None, **kwargs):
        content = messages[-1].content
        self.prompts.append(content)
        hypothesis = content.split("Hypothesis:\n", 1)[-1].strip()
        match = re.search(r"round (\d+)", content)
        rnd = match.group(1) if match else "1"
        lines = "\n".join(f"{hypothesis} r{rnd} a{i}" for i in range(1, 6))
        return AIMessage(content=lines)


def _doc(url: str, title: str, cid: int) -> RetrievedDocument:
    return RetrievedDocument(
        result=SearchResult(title=title, url=url, snippet="snippet"),
        citation=Citation(
            id=cid, reference=f"[{cid}]", url=url, title=title, snippet="snippet"
        ),
        source_queries=["q"],
    )


def _recording_search(calls: list[list[str]]):
    async def search(_search_config, queries):
        calls.append(list(queries))
        return [_doc("https://a.example", "A", 1)]

    return search


def test_generates_queries_per_claim_for_each_unresolved_claim():
    # Two unresolved claims, queries_per_claim=3: the first search round must be
    # driven by exactly 3 fresh-angle queries per claim (6 total), proving the
    # generator honors queries_per_claim rather than echoing a single hypothesis.
    calls: list[list[str]] = []

    async def planner(_task, _config):
        return [
            EvidenceClaim(id="c1", hypothesis="alpha", importance=Importance.CRITICAL),
            EvidenceClaim(id="c2", hypothesis="beta", importance=Importance.OPTIONAL),
        ]

    async def scorer_silent(_claims, docs):
        return [{} for _ in docs]

    # max_attempts=1 so the run terminates after a single observable round.
    config = EDREConfig(queries_per_claim=3, max_attempts=1, max_loops=5)
    graph = build_research_graph(
        config,
        planner=planner,
        query_generator=make_llm_query_generator(FakeQueryChat()),
        search_fn=_recording_search(calls),
        scorer=scorer_silent,
    )

    asyncio.run(graph.ainvoke({"task": "two things"}))

    assert calls, "search node was never driven"
    first = calls[0]
    alpha_qs = [q for q in first if q.startswith("alpha")]
    beta_qs = [q for q in first if q.startswith("beta")]
    assert len(alpha_qs) == 3
    assert len(beta_qs) == 3
    assert len(first) == 6


def test_resolved_claim_stops_triggering_searches():
    # c1 (optional) resolves on the first round; c2 (critical) never does. The
    # loop must continue for the unresolved critical claim, but from round two on
    # only c2's queries reach search — no budget is wasted re-searching the
    # already-VERIFIED c1.
    calls: list[list[str]] = []

    async def planner(_task, _config):
        return [
            EvidenceClaim(id="c1", hypothesis="alpha", importance=Importance.OPTIONAL),
            EvidenceClaim(id="c2", hypothesis="beta", importance=Importance.CRITICAL),
        ]

    async def scorer_resolves_c1(claims, docs):
        # Strong support for c1 only; c2 stays indecisive and keeps the loop alive.
        return [{"c1": 0.9} for _ in docs]

    config = EDREConfig(
        queries_per_claim=2, verify_threshold=0.7, max_attempts=2, max_loops=5
    )
    graph = build_research_graph(
        config,
        planner=planner,
        query_generator=make_llm_query_generator(FakeQueryChat()),
        search_fn=_recording_search(calls),
        scorer=scorer_resolves_c1,
    )

    out = asyncio.run(graph.ainvoke({"task": "two things"}))["output"]

    assert len(calls) == 2, "expected exactly two research rounds"
    # Round one searches both claims; round two only the still-unresolved c2.
    assert any(q.startswith("alpha") for q in calls[0])
    assert any(q.startswith("beta") for q in calls[0])
    assert all(not q.startswith("alpha") for q in calls[1])
    assert all(q.startswith("beta") for q in calls[1])
    # Honest terminal state: c1 VERIFIED, critical c2 ABANDONED -> EXHAUSTED.
    by_id = {v.claim_id: v for v in out.evidence}
    assert by_id["c1"].status == "VERIFIED"
    assert by_id["c2"].status == "ABANDONED"
    assert out.research_summary.terminal == "EXHAUSTED"


def test_retries_use_fresh_angles_each_round():
    # A single unresolved claim searched across two rounds must get *new-angle*
    # queries on the retry — v1's only retry strategy. The two rounds must not
    # repeat a single query verbatim.
    calls: list[list[str]] = []

    async def planner(_task, _config):
        return [
            EvidenceClaim(id="c1", hypothesis="gamma", importance=Importance.CRITICAL),
        ]

    async def scorer_silent(_claims, docs):
        return [{} for _ in docs]

    config = EDREConfig(queries_per_claim=2, max_attempts=2, max_loops=5)
    graph = build_research_graph(
        config,
        planner=planner,
        query_generator=make_llm_query_generator(FakeQueryChat()),
        search_fn=_recording_search(calls),
        scorer=scorer_silent,
    )

    asyncio.run(graph.ainvoke({"task": "one thing"}))

    assert len(calls) == 2, "expected two retry rounds"
    round_one, round_two = set(calls[0]), set(calls[1])
    assert round_one and round_two
    assert round_one.isdisjoint(round_two), "retry reused a prior-round query verbatim"


def test_query_generator_is_replaceable():
    # The generator is a swappable seam: a bespoke (non-LLM) generator drives the
    # exact queries it emits through search, unchanged, without disturbing any
    # up/downstream node.
    calls: list[list[str]] = []

    async def planner(_task, _config):
        return [
            EvidenceClaim(id="c1", hypothesis="alpha", importance=Importance.CRITICAL),
        ]

    async def custom_generator(state, config):
        from search_agent.edre import ClaimStatus, derive_status

        unresolved = {ClaimStatus.NOT_STARTED, ClaimStatus.PARTIAL}
        return {
            claim.id: ["bespoke-query"]
            for claim in state["evidence_plan"]
            if derive_status(claim, config) in unresolved
        }

    async def scorer_verifies(claims, docs):
        return [{c.id: 0.9 for c in claims} for _ in docs]

    config = EDREConfig(max_loops=5)
    graph = build_research_graph(
        config,
        planner=planner,
        query_generator=custom_generator,
        search_fn=_recording_search(calls),
        scorer=scorer_verifies,
    )

    out = asyncio.run(graph.ainvoke({"task": "one thing"}))["output"]

    assert calls[0] == ["bespoke-query"]
    assert out.research_summary.terminal == "DONE"
