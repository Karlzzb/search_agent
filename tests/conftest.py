"""Shared test fixtures.

Because the package auto-loads a ``.env`` on import (populating ``LLM_*`` /
``SEARXNG_*`` / ``LANGFUSE_*``), the autouse ``_hermetic_env`` fixture strips
those variables for every test. This keeps the unit suite deterministic and
offline regardless of the developer's ``.env``: ``default_chat_model()`` returns
``None`` (no ``LLM_KEY``) and ``get_langfuse_callback()`` returns ``None`` (no
Langfuse creds), so nothing touches the network unless a fake is injected.
"""

from __future__ import annotations

import pytest

_HERMETIC_VARS = (
    "LLM_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "SEARXNG_BASE_URL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Strip env that the package auto-loads from ``.env`` so tests are offline."""
    for name in _HERMETIC_VARS:
        monkeypatch.delenv(name, raising=False)
    # Neutralize the real local cross-encoder default so the graph never loads a
    # heavy model (or touches the network) in the unit suite: the reranker
    # degrades to a passthrough unless a test injects its own gate.
    monkeypatch.setattr(
        "search_agent.edre.graph.default_local_reranker",
        lambda config: None,
        raising=False,
    )
