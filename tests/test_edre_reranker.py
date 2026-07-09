"""Slice E — first-layer reranker gate, tested at the single highest seam.

The real first-layer gate is a local cross-encoder that scores every document on
Task Match (document vs the original task) and Query Match (document vs the
current query), combines them with weights ``w1``/``w2``, and drops any document
below ``rerank_threshold`` — the Precision First gate, before any normalization
or claim scoring. We observe only ``ResearchInput -> ResearchOutput``: the gate
logic (the weighting + threshold drop) is the *real* code under test, while the
cross-encoder itself is a deterministic injected fake, so gating is reproducible
and offline. Async is driven with ``asyncio.run`` (no plugin), matching the
repo's existing convention.
"""

import asyncio

from search_agent.contracts import Citation, SearchResult
from search_agent.edre import EDREConfig, EvidenceClaim, Importance, build_research_graph
from search_agent.edre.reranker import make_rerank_gate
from search_agent.retrieval import RetrievedDocument


def _doc(url: str, title: str, cid: int, query: str, snippet: str = "snippet"):
    return RetrievedDocument(
        result=SearchResult(title=title, url=url, snippet=snippet),
        citation=Citation(
            id=cid, reference=f"[{cid}]", url=url, title=title, snippet=snippet
        ),
        source_queries=[query],
    )


def _fixed_search(docs):
    async def search(_search_config, _queries):
        # Return fresh copies so per-run mutation never leaks across the loop.
        return [
            RetrievedDocument(
                result=d.result, citation=d.citation, source_queries=list(d.source_queries)
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
    """Deterministic fake cross-encoder: score = table[(left_text, url)], default 0.

    Mirrors the ``Reranker`` protocol ``rerank(left_text, docs) -> [scores]``. The
    same fake is called with the task as ``left_text`` (Task Match) and with each
    query as ``left_text`` (Query Match) — only the left side changes.
    """

    def rerank(left_text, docs):
        return [table.get((left_text, d.citation.url), 0.0) for d in docs]

    return rerank


def test_offtopic_document_is_gated_out_before_evidence():
    # Two documents surface for the critical claim's query: one on-topic, one
    # off-topic. The off-topic doc scores near zero on both Task and Query Match
    # and must be dropped by the gate — so it never reaches claim scoring and
    # never appears as evidence. Only the on-topic doc backs the verdict.
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

    async def scorer_sees(claims, docs):
        return [{c.id: 0.9 for c in claims} for _ in docs]

    async def planner(_task, _config):
        return [EvidenceClaim(id="c1", hypothesis="h1", importance=Importance.CRITICAL)]

    config = EDREConfig(rerank_threshold=0.3, rerank_w1=0.5, rerank_w2=0.5, max_loops=5)
    gate = make_rerank_gate(_cross_encoder(table), config)
    graph = build_research_graph(
        config,
        planner=planner,
        query_generator=_fixed_query_generator({"c1": [query]}),
        search_fn=_fixed_search([on, off]),
        reranker=gate,
        scorer=scorer_sees,
    )

    out = asyncio.run(graph.ainvoke({"task": task}))["output"]

    assert out.research_summary.terminal == "DONE"
    urls = {c.url for c in out.citations}
    # The on-topic document backs the verdict; the off-topic one was gated out.
    assert "https://on.example" in urls
    assert "https://off.example" not in urls
    by_id = {v.claim_id: v for v in out.evidence}
    supporting_urls = {r.citation.url for r in by_id["c1"].supporting_documents}
    assert supporting_urls == {"https://on.example"}


def test_threshold_gates_all_documents_reproducibly():
    # The gate is threshold-driven: with rerank_threshold raised above every
    # document's combined score, even the otherwise-relevant document is dropped.
    # No evidence reaches the critical claim, which is honestly ABANDONED after
    # its attempts are spent — the run ends EXHAUSTED, not a faked DONE.
    task = "is the sky blue?"
    query = "sky color"
    doc = _doc("https://on.example", "On", 1, query)
    table = {
        (task, "https://on.example"): 0.6,
        (query, "https://on.example"): 0.6,
    }

    async def scorer_sees(claims, docs):
        return [{c.id: 0.9 for c in claims} for _ in docs]

    async def planner(_task, _config):
        return [EvidenceClaim(id="c1", hypothesis="h1", importance=Importance.CRITICAL)]

    # Combined score is 0.6; a threshold of 0.9 gates it out entirely.
    config = EDREConfig(
        rerank_threshold=0.9, rerank_w1=0.5, rerank_w2=0.5, max_attempts=1, max_loops=5
    )
    gate = make_rerank_gate(_cross_encoder(table), config)
    graph = build_research_graph(
        config,
        planner=planner,
        query_generator=_fixed_query_generator({"c1": [query]}),
        search_fn=_fixed_search([doc]),
        reranker=gate,
        scorer=scorer_sees,
    )

    out = asyncio.run(graph.ainvoke({"task": task}))["output"]

    assert out.research_summary.terminal == "EXHAUSTED"
    assert not out.citations
    by_id = {v.claim_id: v for v in out.evidence}
    assert by_id["c1"].status == "ABANDONED"
    assert by_id["c1"].supporting_documents == []


def _run_with_weights(w1, w2, table, task, query, doc):
    async def scorer_sees(claims, docs):
        return [{c.id: 0.9 for c in claims} for _ in docs]

    async def planner(_task, _config):
        return [EvidenceClaim(id="c1", hypothesis="h1", importance=Importance.CRITICAL)]

    config = EDREConfig(
        rerank_threshold=0.5, rerank_w1=w1, rerank_w2=w2, max_attempts=1, max_loops=5
    )
    gate = make_rerank_gate(_cross_encoder(table), config)
    graph = build_research_graph(
        config,
        planner=planner,
        query_generator=_fixed_query_generator({"c1": [query]}),
        search_fn=_fixed_search([doc]),
        reranker=gate,
        scorer=scorer_sees,
    )
    return asyncio.run(graph.ainvoke({"task": task}))["output"]


def test_task_and_query_match_weights_decide_survival():
    # One document that matches the task strongly but the query weakly. The gate
    # scores it against the *same* model twice, changing only the left input
    # (task vs query). Weighting Task Match alone lets it through; weighting Query
    # Match alone gates it out — proving both signals are computed and the
    # w1/w2 knobs actually steer the decision.
    task = "is the sky blue?"
    query = "sky color"
    doc = _doc("https://d.example", "D", 1, query)
    table = {
        (task, "https://d.example"): 0.9,   # strong Task Match
        (query, "https://d.example"): 0.1,  # weak Query Match
    }

    # w1=1, w2=0 -> combined = 0.9 >= 0.5 -> survives, claim verified, DONE.
    task_only = _run_with_weights(1.0, 0.0, table, task, query, doc)
    assert task_only.research_summary.terminal == "DONE"
    assert {c.url for c in task_only.citations} == {"https://d.example"}

    # w1=0, w2=1 -> combined = 0.1 < 0.5 -> gated out, no evidence, EXHAUSTED.
    query_only = _run_with_weights(0.0, 1.0, table, task, query, doc)
    assert query_only.research_summary.terminal == "EXHAUSTED"
    assert not query_only.citations


def test_gate_reuses_one_model_on_raw_text_changing_only_left_input():
    # A single recording cross-encoder is the only scorer the gate uses. It must
    # be invoked with the *task* as left input (Task Match) and with the *query*
    # as left input (Query Match) — same model, different left side. And it must
    # see the *raw* document text: the gate runs before normalization, so no
    # normalized/distilled text can reach it.
    task = "is the sky blue?"
    query = "sky color"
    raw_snippet = "RAW-SNIPPET-distinct-body"
    doc = _doc("https://d.example", "RAW-TITLE", 1, query, snippet=raw_snippet)

    left_inputs: list[str] = []
    right_bodies: list[str] = []

    def recording(left_text, docs):
        left_inputs.append(left_text)
        right_bodies.extend(d.result.snippet for d in docs)
        return [0.9 for _ in docs]  # pass everything through

    async def scorer_sees(claims, docs):
        return [{c.id: 0.9 for c in claims} for _ in docs]

    async def planner(_task, _config):
        return [EvidenceClaim(id="c1", hypothesis="h1", importance=Importance.CRITICAL)]

    config = EDREConfig(rerank_threshold=0.3, rerank_w1=0.5, rerank_w2=0.5, max_loops=5)
    gate = make_rerank_gate(recording, config)
    graph = build_research_graph(
        config,
        planner=planner,
        query_generator=_fixed_query_generator({"c1": [query]}),
        search_fn=_fixed_search([doc]),
        reranker=gate,
        scorer=scorer_sees,
    )

    asyncio.run(graph.ainvoke({"task": task}))

    # Same model, two left inputs: the raw task and the raw query, verbatim.
    assert set(left_inputs) == {task, query}
    # The gate scored the raw document body — never a normalized derivative.
    assert all(body == raw_snippet for body in right_bodies)
