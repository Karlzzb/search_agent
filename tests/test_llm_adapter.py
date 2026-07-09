"""Built-in default chat model (LangChain ``ChatOpenAI``).

Seam under test: ``normalize_base_url`` (pure) and the ``default_chat_model``
factory. No real network call is made; only construction and env resolution are
checked. The autouse ``_hermetic_env`` fixture (conftest) clears ``LLM_*`` so
each test controls its own environment.
"""

import pytest
from langchain_openai import ChatOpenAI

from search_agent.llm import default_chat_model, normalize_base_url


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://host/compatible-mode/v1/chat/completions", "https://host/compatible-mode/v1"),
        ("https://host/compatible-mode/v1/chat/completions/", "https://host/compatible-mode/v1"),
        ("https://host/compatible-mode/v1", "https://host/compatible-mode/v1"),
        ("https://host/compatible-mode/v1/", "https://host/compatible-mode/v1"),
        ("  https://host/v1  ", "https://host/v1"),
        (None, None),
        ("", None),
    ],
)
def test_normalize_base_url_strips_completions_suffix(raw, expected):
    assert normalize_base_url(raw) == expected


def test_default_chat_model_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("LLM_KEY", raising=False)
    assert default_chat_model() is None


def test_default_chat_model_defaults_to_qwen_plus_on_dashscope(monkeypatch):
    monkeypatch.setenv("LLM_KEY", "k")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    model = default_chat_model()

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "qwen-plus"
    assert model.openai_api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_default_chat_model_reads_env_and_normalizes_base_url(monkeypatch):
    monkeypatch.setenv("LLM_KEY", "env-key")
    # A full completions URL (as in the project's .env) must be normalized.
    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.example/compatible-mode/v1/chat/completions")
    monkeypatch.setenv("LLM_MODEL", "qwen-max")

    model = default_chat_model()

    assert model.model_name == "qwen-max"
    assert model.openai_api_base == "https://dashscope.example/compatible-mode/v1"


def test_default_chat_model_model_override_targets_only_that_call(monkeypatch):
    monkeypatch.setenv("LLM_KEY", "k")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    overridden = default_chat_model(model="big-cheap-longctx")

    assert overridden.model_name == "big-cheap-longctx"
    # The env/default is unchanged for a call without the override.
    assert default_chat_model().model_name == "qwen-plus"
