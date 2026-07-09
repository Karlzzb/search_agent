"""Langfuse tracing wiring.

``get_langfuse_callback`` builds a LangChain ``CallbackHandler`` from the
``LANGFUSE_*`` environment (loaded from ``.env``). The subgraph bakes this
handler into the compiled graph, so every node and LLM generation is traced
with no per-call arguments. It is non-fatal by design: when credentials or the
``langfuse`` package are absent, it returns ``None`` and the subgraph runs
untraced.

Env:
- ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` — required to activate
- ``LANGFUSE_BASE_URL``                            — Langfuse host
"""

from __future__ import annotations

import logging

from .env import env_str

logger = logging.getLogger(__name__)

_client_configured = False


def get_langfuse_callback():
    """Return a Langfuse ``CallbackHandler``, or ``None`` if not configured."""
    public_key = env_str("LANGFUSE_PUBLIC_KEY")
    secret_key = env_str("LANGFUSE_SECRET_KEY")
    if not (public_key and secret_key):
        return None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except Exception as exc:  # pragma: no cover - optional integration guard
        logger.warning("Langfuse configured but import failed; tracing disabled: %s", exc)
        return None

    global _client_configured
    if not _client_configured:
        # Configures the process-wide Langfuse client that CallbackHandler uses.
        Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=env_str("LANGFUSE_BASE_URL"),
        )
        _client_configured = True

    return CallbackHandler()


__all__ = ["get_langfuse_callback"]
