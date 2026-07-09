"""SearchConfig contract (PRD:108-119)."""

from search_agent import SearchConfig


def test_defaults_match_spec():
    cfg = SearchConfig()
    assert cfg.provider == "searxng"
    assert cfg.base_url is None
    assert cfg.api_key is None
    assert cfg.proxy is None
    assert cfg.max_results == 5
    assert cfg.consolidation_use_llm is False
    assert cfg.consolidation_custom_template is None
    assert cfg.consolidation_llm_model is None


def test_overrides_are_stored():
    cfg = SearchConfig(
        provider="brave",
        base_url="https://searx.example.com",
        api_key="secret",
        proxy="http://proxy:8080",
        max_results=8,
        consolidation_use_llm=True,
        consolidation_custom_template="{{ query }}",
        consolidation_llm_model="big-model",
    )
    assert cfg.provider == "brave"
    assert cfg.base_url == "https://searx.example.com"
    assert cfg.api_key == "secret"
    assert cfg.proxy == "http://proxy:8080"
    assert cfg.max_results == 8
    assert cfg.consolidation_use_llm is True
    assert cfg.consolidation_custom_template == "{{ query }}"
    assert cfg.consolidation_llm_model == "big-model"
