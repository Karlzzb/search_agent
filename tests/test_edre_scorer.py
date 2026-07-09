"""Slice G — second-layer signed-support scoring, tested at the single highest seam.

The real scorer issues one LLM call per reranker-surviving document and returns
that document's *signed* support for every current claim (support in [-1, 1]:
positive supports, negative refutes/falsifies, ~0 = no opinion; see ADR-0005).
We observe only ``ResearchInput -> ResearchOutput``: the scoring LLM is a
deterministic injected ``FakeChat`` returning canned JSON, so the real prompt
build + JSON parse + evidence-update + status derivation all run through the
full pipeline. Async is driven with ``asyncio.run`` (no plugin), matching the
repo convention.
"""

import json

import asyncio

from langchain_core.messages import AIMessage

from search_agent.contracts import Citation, SearchResult
from search_agent.edre import EDREConfig, EvidenceClaim, Importance, build_research_graph
from search_agent.edre.scorer import make_llm_scorer
from search_agent.retrieval import RetrievedDocument


class FakeScoringChat:
    """Chat model that replays per-document canned support JSON.

    ``by_url`` maps a URL substring to the support dict the LLM "returns" for the
    document whose prompt contains that URL; the first matching URL wins, else an
    empty object (no opinion). Records prompts for call-count/scoping assertions.
    """

    def __init__(self, by_url):
        self.by_url = by_url
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    async def ainvoke(self, messages, config=None, **kwargs):
        prompt = messages[-1].content
        self.system_prompts.append(messages[0].content)
        self.prompts.append(prompt)
        payload: dict = {}
        for url, scores in self.by_url.items():
            if url in prompt:
                payload = scores
                break
        return AIMessage(content=json.dumps(payload))


def _doc(url, title, cid, query, snippet="snippet", content=""):
    return RetrievedDocument(
        result=SearchResult(title=title, url=url, snippet=snippet, content=content),
        citation=Citation(
            id=cid, reference=f"[{cid}]", url=url, title=title, snippet=snippet
        ),
        source_queries=[query],
    )


def _fixed_search(docs):
    async def search(_search_config, _queries):
        return [
            RetrievedDocument(
                result=d.result,
                citation=d.citation,
                source_queries=list(d.source_queries),
            )
            for d in docs
        ]

    return search


def _fixed_query_generator(query_by_claim):
    async def generator(state, config):
        from search_agent.edre import ClaimStatus, derive_status

        unresolved = {ClaimStatus.NOT_STARTED, ClaimStatus.PARTIAL}
        return {
            claim.id: list(query_by_claim[claim.id])
            for claim in state["evidence_plan"]
            if derive_status(claim, config) in unresolved
        }

    return generator


def _planner(claims):
    async def planner(_task, _config):
        return [
            EvidenceClaim(id=c[0], hypothesis=c[1], importance=c[2]) for c in claims
        ]

    return planner


def test_strong_positive_support_verifies_critical_claim():
    # Tracer bullet: a document the LLM scores +0.9 for a critical claim drives it
    # to VERIFIED, terminating DONE, with the signed support traceable to the doc.
    task = "does X hold?"
    doc = _doc("https://a.example", "A", 1, "q")
    chat = FakeScoringChat({"https://a.example": {"c1": 0.9}})

    config = EDREConfig(verify_threshold=0.7, refute_threshold=0.7, max_loops=5)
    graph = build_research_graph(
        config,
        planner=_planner([("c1", "h1", Importance.CRITICAL)]),
        query_generator=_fixed_query_generator({"c1": ["q"]}),
        search_fn=_fixed_search([doc]),
        scorer=make_llm_scorer(chat),
    )

    out = asyncio.run(graph.ainvoke({"task": task}))["output"]

    assert out.research_summary.terminal == "DONE"
    by_id = {v.claim_id: v for v in out.evidence}
    assert by_id["c1"].status == "VERIFIED"
    assert by_id["c1"].confidence == 0.9
    assert by_id["c1"].supporting_documents[0].support == 0.9
    assert by_id["c1"].supporting_documents[0].citation.url == "https://a.example"


def test_strong_negative_support_refutes_and_is_distinct_from_abandoned():
    # A refuted critical premise is a *successful* discovery: strong-negative
    # support drives REFUTED, still terminates DONE (the critical claim is
    # resolved), and is counted as refuted -- never as abandoned.
    task = "is the false premise true?"
    doc = _doc("https://a.example", "A", 1, "q")
    chat = FakeScoringChat({"https://a.example": {"c1": -0.9}})

    config = EDREConfig(verify_threshold=0.7, refute_threshold=0.7, max_loops=5)
    graph = build_research_graph(
        config,
        planner=_planner([("c1", "h1", Importance.CRITICAL)]),
        query_generator=_fixed_query_generator({"c1": ["q"]}),
        search_fn=_fixed_search([doc]),
        scorer=make_llm_scorer(chat),
    )

    out = asyncio.run(graph.ainvoke({"task": task}))["output"]

    assert out.research_summary.terminal == "DONE"
    assert out.research_summary.critical_all_resolved is True
    by_id = {v.claim_id: v for v in out.evidence}
    assert by_id["c1"].status == "REFUTED"
    assert by_id["c1"].confidence == -0.9
    assert by_id["c1"].supporting_documents[0].support == -0.9
    assert out.research_summary.refuted == 1
    assert out.research_summary.abandoned == 0


def test_verified_refuted_and_abandoned_are_three_distinct_states():
    # One run surfaces all three verdicts from signed support: c1 (+0.9) VERIFIED,
    # c2 (-0.9) REFUTED, c3 (no key => no opinion) ABANDONED after its single
    # allowed attempt. REFUTED (a finding) must stay clearly separate from
    # ABANDONED (budget spent, indecisive), and the un-opinioned claim accrues no
    # supporting document.
    task = "sort the premises"
    doc = _doc("https://a.example", "A", 1, "q")
    chat = FakeScoringChat({"https://a.example": {"c1": 0.9, "c2": -0.9}})

    config = EDREConfig(
        verify_threshold=0.7, refute_threshold=0.7, max_attempts=1, max_loops=5
    )
    graph = build_research_graph(
        config,
        planner=_planner(
            [
                ("c1", "supported premise", Importance.CRITICAL),
                ("c2", "false premise", Importance.OPTIONAL),
                ("c3", "unfindable premise", Importance.OPTIONAL),
            ]
        ),
        query_generator=_fixed_query_generator(
            {"c1": ["q"], "c2": ["q"], "c3": ["q"]}
        ),
        search_fn=_fixed_search([doc]),
        scorer=make_llm_scorer(chat),
    )

    out = asyncio.run(graph.ainvoke({"task": task}))["output"]

    # Only c1 is critical and it is verified, so the run terminates DONE.
    assert out.research_summary.terminal == "DONE"
    by_id = {v.claim_id: v for v in out.evidence}
    assert by_id["c1"].status == "VERIFIED"
    assert by_id["c2"].status == "REFUTED"
    assert by_id["c3"].status == "ABANDONED"
    assert (out.research_summary.verified, out.research_summary.refuted,
            out.research_summary.abandoned) == (1, 1, 1)
    # No opinion => no fabricated evidence for the abandoned claim.
    assert by_id["c3"].supporting_documents == []


def test_one_llm_call_per_document_scoring_all_claims_at_once():
    # The second layer is per-*document*, not per-claim: two surviving docs yield
    # exactly two LLM calls, each prompt scoped to one document yet listing every
    # current claim, so a single call returns that doc's support over all claims.
    task = "does X hold?"
    a = _doc("https://a.example", "Alpha", 1, "q")
    b = _doc("https://b.example", "Beta", 2, "q")
    # Doc A supports c1; doc B refutes c2. Both claims resolve from one call each.
    chat = FakeScoringChat(
        {
            "https://a.example": {"c1": 0.9, "c2": 0.0},
            "https://b.example": {"c1": 0.0, "c2": -0.9},
        }
    )

    config = EDREConfig(verify_threshold=0.7, refute_threshold=0.7, max_loops=5)
    graph = build_research_graph(
        config,
        planner=_planner(
            [
                ("c1", "premise one", Importance.CRITICAL),
                ("c2", "premise two", Importance.CRITICAL),
            ]
        ),
        query_generator=_fixed_query_generator({"c1": ["q"], "c2": ["q"]}),
        search_fn=_fixed_search([a, b]),
        scorer=make_llm_scorer(chat),
    )

    out = asyncio.run(graph.ainvoke({"task": task}))["output"]

    assert len(chat.prompts) == 2  # one call per document, not per claim
    scoped = sorted(
        ("https://a.example" in p, "https://b.example" in p) for p in chat.prompts
    )
    assert scoped == [(False, True), (True, False)]  # each prompt one document
    # Every single-document call lists all current claims (scores them at once).
    for prompt in chat.prompts:
        assert "c1" in prompt and "c2" in prompt
        assert "premise one" in prompt and "premise two" in prompt
    by_id = {v.claim_id: v for v in out.evidence}
    assert by_id["c1"].status == "VERIFIED"  # from doc A's vector
    assert by_id["c2"].status == "REFUTED"   # from doc B's vector
