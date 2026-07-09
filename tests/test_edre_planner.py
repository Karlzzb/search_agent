"""Slice C — Planner, tested at the single highest seam.

The real Planner is a one-shot LLM call that turns a task into 3..6 falsifiable
``EvidenceClaim`` hypotheses tagged CRITICAL/OPTIONAL, with guardrails: at least
``min_critical`` critical, critical kept a minority, and the count clamped to
``[min_claims, max_claims]``. We observe only ``ResearchInput -> ResearchOutput``:
inject ``planner=make_llm_planner(FakeChat(...))`` plus fake search/scorer and
assert the resulting claim set via ``out.evidence``. Async is driven with
``asyncio.run`` (no plugin), matching the repo's existing convention.
"""

import asyncio
import json

from langchain_core.messages import AIMessage

from search_agent.contracts import Citation, SearchResult
from search_agent.edre import EDREConfig, build_research_graph
from search_agent.edre.planner import make_llm_planner
from search_agent.retrieval import RetrievedDocument


class FakeChat:
    """Minimal chat model: replays a canned response, records prompts."""

    def __init__(self, content: str):
        self._content = content
        self.prompts: list[str] = []

    async def ainvoke(self, messages, config=None, **kwargs):
        self.prompts.append(messages[-1].content)
        return AIMessage(content=self._content)


def _claims_json(items) -> str:
    return json.dumps(
        [{"hypothesis": h, "importance": imp} for h, imp in items]
    )


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


async def _scorer_verifies(claims, docs):
    # Verify every claim so the loop terminates deterministically in one pass.
    return [{c.id: 0.9 for c in claims} for _ in docs]


def _run(llm_content: str, config: EDREConfig | None = None):
    config = config or EDREConfig()
    graph = build_research_graph(
        config,
        planner=make_llm_planner(FakeChat(llm_content)),
        search_fn=_search_returns_one,
        scorer=_scorer_verifies,
    )
    return asyncio.run(graph.ainvoke({"task": "does X hold?"}))["output"]


def test_planner_decomposes_task_into_falsifiable_claims():
    # A well-formed 4-claim plan (1 critical, 3 optional) is carried through the
    # graph verbatim as 4 verdicts with their hypotheses and importance intact.
    items = [
        ("X increases Y", "CRITICAL"),
        ("Z was released in 2020", "OPTIONAL"),
        ("A depends on B", "OPTIONAL"),
        ("C is faster than D", "OPTIONAL"),
    ]
    out = _run(_claims_json(items))

    assert len(out.evidence) == 4
    hypotheses = [v.hypothesis for v in out.evidence]
    assert hypotheses == [h for h, _ in items]
    importances = [v.importance for v in out.evidence]
    assert importances.count("CRITICAL") == 1


def test_planner_clamps_claim_count_to_upper_bound():
    # An over-eager LLM returning more than max_claims is truncated to the cap.
    items = [(f"claim {n}", "OPTIONAL") for n in range(8)]
    items[0] = ("claim 0", "CRITICAL")
    config = EDREConfig(min_claims=3, max_claims=6)
    out = _run(_claims_json(items), config)

    assert len(out.evidence) == config.max_claims


def test_planner_guarantees_min_critical_when_llm_marks_none():
    # An LLM that tags nothing CRITICAL still yields at least min_critical, so a
    # research run always has a critical spine to terminate against.
    items = [
        ("first claim", "OPTIONAL"),
        ("second claim", "OPTIONAL"),
        ("third claim", "OPTIONAL"),
    ]
    config = EDREConfig(min_claims=3, max_claims=6, min_critical=1)
    out = _run(_claims_json(items), config)

    criticals = [v for v in out.evidence if v.importance == "CRITICAL"]
    assert len(criticals) >= config.min_critical


def test_planner_keeps_critical_a_minority_when_llm_marks_all():
    # An LLM that tags everything CRITICAL is demoted so CRITICAL stays a strict
    # minority — importance must actually discriminate the answer's spine.
    items = [(f"claim {n}", "CRITICAL") for n in range(5)]
    config = EDREConfig(min_claims=3, max_claims=6, min_critical=1)
    out = _run(_claims_json(items), config)

    critical_count = sum(1 for v in out.evidence if v.importance == "CRITICAL")
    total = len(out.evidence)
    assert critical_count >= config.min_critical
    assert critical_count < total - critical_count  # strict minority


def test_planner_degrades_to_nonempty_plan_on_unusable_output():
    # If the LLM returns nothing usable, the Planner must not emit an empty plan
    # (which would leave the loop with nothing to research). It degrades to a
    # single critical claim echoing the task.
    out = _run("   ")

    assert len(out.evidence) == 1
    assert out.evidence[0].importance == "CRITICAL"
    assert out.evidence[0].hypothesis == "does X hold?"
