"""AnswerConsolidator: injectable LLM, template path works without any LLM.

The deeptutor ``get_llm_client`` import is gone; the module must import with no
monolith present (guaranteed by test_independence), and the template path must
run with no LLM injected at all.
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage

from search_agent import AnswerConsolidator
from search_agent.contracts import Citation, SearchResult, WebSearchResponse


def _response() -> WebSearchResponse:
    return WebSearchResponse(
        query="what is langgraph",
        answer="",
        provider="searxng",
        search_results=[
            SearchResult(title="LangGraph", url="https://ex.com/lg", snippet="a graph lib"),
            SearchResult(title="Docs", url="https://ex.com/docs", snippet="the docs"),
        ],
        citations=[Citation(id=1, reference="[1]", url="https://ex.com/lg", title="LangGraph")],
    )


def test_template_path_needs_no_llm():
    consolidator = AnswerConsolidator(use_llm=False)
    out = asyncio.run(consolidator.consolidate(_response()))
    # Generic fallback formatting for searxng: a titled results list.
    assert 'Search Results for "what is langgraph"' in out.answer
    assert "LangGraph" in out.answer
    assert "https://ex.com/lg" in out.answer


def test_llm_is_injected_not_imported():
    class FakeChat:
        async def ainvoke(self, messages, config=None, **kwargs):
            return AIMessage(content="fake grounding [1]")

    fake = FakeChat()
    consolidator = AnswerConsolidator(use_llm=True, llm=fake)
    assert consolidator.llm is fake
    # The injected chat model drives consolidation; its content becomes the answer.
    out = asyncio.run(consolidator.consolidate(_response()))
    assert out.answer == "fake grounding [1]"


def test_llm_path_requires_injected_llm():
    # With use_llm=True but no llm injected, consolidate must fail loudly rather
    # than silently reaching for a monolith client.
    consolidator = AnswerConsolidator(use_llm=True)
    with pytest.raises(ValueError):
        asyncio.run(consolidator.consolidate(_response()))
