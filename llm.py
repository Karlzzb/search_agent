"""Built-in default chat model (LangChain ``ChatOpenAI``).

Used when the caller does not inject their own chat model. It targets any
OpenAI-compatible endpoint, defaulting to DashScope's compatible-mode
``qwen-plus``. Credentials/overrides come from the environment (loaded from
``.env`` on package import):

- ``LLM_KEY``       — required to activate the default model
- ``LLM_BASE_URL``  — optional; a trailing ``/chat/completions`` is stripped so
                      the value in ``.env`` (a full completions URL) works as-is
- ``LLM_MODEL``     — optional, defaults to ``qwen-plus``

Because the model is a native LangChain ``Runnable``, LLM generations are traced
automatically by the Langfuse callback handler wired into the subgraph.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .env import env_str

_DEFAULT_MODEL = "qwen-plus"
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_COMPLETIONS_SUFFIX = "/chat/completions"


def normalize_base_url(url: str | None) -> str | None:
    """Normalize an OpenAI-compatible base URL for the SDK.

    The SDK appends ``/chat/completions`` itself, so a base URL that already
    ends in it (as configured in ``.env``) would double up. Strip that suffix
    and any trailing slash. Returns ``None`` for empty input.
    """
    if not url:
        return None
    normalized = url.strip().rstrip("/")
    if normalized.endswith(_COMPLETIONS_SUFFIX):
        normalized = normalized[: -len(_COMPLETIONS_SUFFIX)].rstrip("/")
    return normalized or None


def default_chat_model(*, model: str | None = None) -> BaseChatModel | None:
    """Return the built-in default chat model, or ``None`` if it can't be used.

    Returns ``None`` when ``LLM_KEY`` is unset so the subgraph can fall back to
    the no-LLM path (decompose degrades to a single subquery).

    Args:
        model: Optional model-name override (used by the consolidate step).
    """
    api_key = env_str("LLM_KEY")
    if not api_key:
        return None
    base_url = normalize_base_url(env_str("LLM_BASE_URL")) or _DEFAULT_BASE_URL
    resolved_model = model or env_str("LLM_MODEL") or _DEFAULT_MODEL
    return ChatOpenAI(
        model=resolved_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


__all__ = ["default_chat_model", "normalize_base_url"]
