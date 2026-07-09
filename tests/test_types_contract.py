"""Data contract stability: Citation / SearchResult / WebSearchResponse.to_dict().

Downstream consumers depend on these field names and the to_dict() shape; this
locks them so the extraction did not silently alter the contract.
"""

from search_agent.contracts import Citation, SearchResult, WebSearchResponse


def test_citation_fields():
    c = Citation(id=1, reference="[1]", url="https://ex.com", title="T", snippet="S")
    assert (c.id, c.reference, c.url, c.title, c.snippet) == (1, "[1]", "https://ex.com", "T", "S")


def test_to_dict_shape():
    resp = WebSearchResponse(
        query="q",
        answer="a",
        provider="searxng",
        model="m",
        citations=[Citation(id=1, reference="[1]", url="https://ex.com", title="T", snippet="S")],
        search_results=[SearchResult(title="T", url="https://ex.com", snippet="S")],
    )
    d = resp.to_dict()
    assert d["query"] == "q"
    assert d["answer"] == "a"
    assert d["provider"] == "searxng"
    assert d["model"] == "m"
    assert d["response"]["content"] == "a"
    assert d["response"]["role"] == "assistant"
    assert d["response"]["finish_reason"] == "stop"

    assert d["citations"][0] == {
        "id": 1,
        "reference": "[1]",
        "url": "https://ex.com",
        "title": "T",
        "snippet": "S",
        "date": "",
        "source": "",
        "content": "",
        "type": "web",
        "icon": "",
        "website": "",
        "web_anchor": "",
    }
    assert d["search_results"][0]["title"] == "T"
    assert d["search_results"][0]["url"] == "https://ex.com"
    assert d["search_results"][0]["snippet"] == "S"
