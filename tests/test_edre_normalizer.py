"""Slice F — per-document normalization, tested at the single highest seam.

The real normalizer distills each *reranker-surviving* document into a dense,
citation-preserving, conflict-annotated fact fragment (adapting the consolidate
LLM prompt to per-document granularity), and binds that fragment to the document
as grounding for the second-layer claim scorer. We observe only
``ResearchInput -> ResearchOutput``: the normalizer LLM is a deterministic
injected ``FakeChat`` and the scorer is a fake that keys its support off the
distilled grounding, so "the fragment reached scoring" is provable at the seam.
Async is driven with ``asyncio.run`` (no plugin), matching the repo convention.
"""

import asyncio

from langchain_core.messages import AIMessage

from search_agent.contracts import Citation, SearchResult
from search_agent.edre import EDREConfig, EvidenceClaim, Importance, build_research_graph
from search_agent.edre.normalizer import make_llm_normalizer
from search_agent.edre.reranker import make_rerank_gate
from search_agent.retrieval import RetrievedDocument


class FakeChat:
    """Minimal chat model: replays a canned response, records prompts."""

    def __init__(self, content: str = "DISTILLED [1] fact."):
        self._content = content
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    async def ainvoke(self, messages, config=None, **kwargs):
        self.system_prompts.append(messages[0].content)
        self.prompts.append(messages[-1].content)
        return AIMessage(content=self._content)


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
        # Fresh copies each loop so per-run mutation never leaks across the loop.
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


def _cross_encoder(table):
    def rerank(left_text, docs):
        return [table.get((left_text, d.citation.url), 0.0) for d in docs]

    return rerank


async def _one_critical_planner(_task, _config):
    return [EvidenceClaim(id="c1", hypothesis="h1", importance=Importance.CRITICAL)]


def test_normalized_grounding_reaches_claim_scoring():
    # Tracer bullet: the normalizer distills the document via the injected fake
    # LLM, and the scorer only credits support when it sees that distilled
    # grounding on the document. A VERIFIED critical claim therefore proves the
    # per-document fragment was produced and threaded into claim scoring.
    task = "does X hold?"
    fragment = "DISTILLED-FRAGMENT [1] fact"
    doc = _doc("https://a.example", "A", 1, "q")

    async def scorer_reads_grounding(claims, docs):
        return [
            {
                c.id: (0.9 if getattr(d, "normalized", None) == fragment else 0.0)
                for c in claims
            }
            for d in docs
        ]

    config = EDREConfig(max_attempts=1, max_loops=5)
    graph = build_research_graph(
        config,
        planner=_one_critical_planner,
        query_generator=_fixed_query_generator({"c1": ["q"]}),
        search_fn=_fixed_search([doc]),
        normalizer=make_llm_normalizer(FakeChat(fragment)),
        scorer=scorer_reads_grounding,
    )

    out = asyncio.run(graph.ainvoke({"task": task}))["output"]

    assert out.research_summary.terminal == "DONE"
    by_id = {v.claim_id: v for v in out.evidence}
    assert by_id["c1"].status == "VERIFIED"
    assert by_id["c1"].supporting_documents[0].citation.url == "https://a.example"


def test_normalization_is_per_document_one_llm_call_each():
    # Granularity is per-document, not a query-merged answer: two surviving docs
    # yield two separate LLM calls, and each call's prompt is scoped to exactly
    # one document (its own URL, never the other's).
    task = "does X hold?"
    a = _doc("https://a.example", "Alpha", 1, "q")
    b = _doc("https://b.example", "Beta", 2, "q")
    chat = FakeChat("DISTILLED [1] fact")

    async def scorer_verifies(claims, docs):
        return [{c.id: 0.9 for c in claims} for _ in docs]

    config = EDREConfig(max_attempts=1, max_loops=5)
    graph = build_research_graph(
        config,
        planner=_one_critical_planner,
        query_generator=_fixed_query_generator({"c1": ["q"]}),
        search_fn=_fixed_search([a, b]),
        normalizer=make_llm_normalizer(chat),
        scorer=scorer_verifies,
    )

    asyncio.run(graph.ainvoke({"task": task}))

    assert len(chat.prompts) == 2  # one call per document, not one merged call
    scoped = sorted(
        ("https://a.example" in p, "https://b.example" in p) for p in chat.prompts
    )
    # Each prompt mentions exactly one of the two documents.
    assert scoped == [(False, True), (True, False)]


def test_only_reranker_survivors_are_normalized():
    # Cost is bounded to survivors: the first-layer gate drops the off-topic doc
    # *before* normalization, so the normalizer LLM is never invoked for it — the
    # dropped URL appears in no normalization prompt.
    task = "is the sky blue?"
    query = "sky color"
    on = _doc("https://on.example", "On", 1, query)
    off = _doc("https://off.example", "Off", 2, query)
    table = {
        (task, "https://on.example"): 0.9,
        (query, "https://on.example"): 0.9,
        (task, "https://off.example"): 0.05,
        (query, "https://off.example"): 0.05,
    }
    chat = FakeChat("DISTILLED [1] fact")

    async def scorer_verifies(claims, docs):
        return [{c.id: 0.9 for c in claims} for _ in docs]

    config = EDREConfig(
        rerank_threshold=0.3, rerank_w1=0.5, rerank_w2=0.5, max_attempts=1, max_loops=5
    )
    graph = build_research_graph(
        config,
        planner=_one_critical_planner,
        query_generator=_fixed_query_generator({"c1": [query]}),
        search_fn=_fixed_search([on, off]),
        reranker=make_rerank_gate(_cross_encoder(table), config),
        normalizer=make_llm_normalizer(chat),
        scorer=scorer_verifies,
    )

    asyncio.run(graph.ainvoke({"task": task}))

    assert len(chat.prompts) == 1  # only the survivor was distilled
    assert all("https://off.example" not in p for p in chat.prompts)
    assert "https://on.example" in chat.prompts[0]


def test_normalization_preserves_citation_and_flags_conflicts():
    # The per-document fragment must stay traceable and conflict-aware: the prompt
    # carries the document's citation marker so distilled facts keep their
    # attribution, and the normalizer is instructed to flag conflicting/hedged
    # statements (the seam the downstream signed-support refutation relies on).
    task = "does X hold?"
    doc = _doc("https://a.example", "A", 7, "q")  # citation reference "[7]"
    chat = FakeChat("DISTILLED [7] fact")

    async def scorer_verifies(claims, docs):
        return [{c.id: 0.9 for c in claims} for _ in docs]

    config = EDREConfig(max_attempts=1, max_loops=5)
    graph = build_research_graph(
        config,
        planner=_one_critical_planner,
        query_generator=_fixed_query_generator({"c1": ["q"]}),
        search_fn=_fixed_search([doc]),
        normalizer=make_llm_normalizer(chat),
        scorer=scorer_verifies,
    )

    asyncio.run(graph.ainvoke({"task": task}))

    assert "[7]" in chat.prompts[0]  # citation marker travels into the prompt
    assert "conflict" in chat.system_prompts[0].lower()  # refutation-aware distillation
