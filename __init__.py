"""Self-contained SearXNG search retrieval package.

This package was extracted from the deeptutor monolith and has no dependency on
it. It exposes the stable data contracts (``Citation`` / ``SearchResult`` /
``WebSearchResponse``), the provider registry, the injectable ``SearchConfig``,
the ``AnswerConsolidator``, the default LangChain chat model, the Langfuse
callback factory, and the LangGraph subgraph factory ``build_search_subgraph``.

A package-local ``.env`` (``LLM_*`` / ``SEARXNG_BASE_URL`` / ``LANGFUSE_*``) is
auto-loaded on import; process environment variables always take precedence.
"""

from .env import load_env

load_env()

from .config import SearchConfig  # noqa: E402
from .consolidation import AnswerConsolidator  # noqa: E402
from .llm import default_chat_model, normalize_base_url  # noqa: E402
from .providers import (  # noqa: E402
    get_available_providers,
    get_provider,
    get_providers_info,
    list_providers,
    register_provider,
)
from .subgraph import build_search_subgraph  # noqa: E402
from .tracing import get_langfuse_callback  # noqa: E402
from .contracts import Citation, SearchResult, WebSearchResponse  # noqa: E402

__all__ = [
    "SearchConfig",
    "AnswerConsolidator",
    "default_chat_model",
    "normalize_base_url",
    "get_langfuse_callback",
    "Citation",
    "SearchResult",
    "WebSearchResponse",
    "register_provider",
    "get_provider",
    "list_providers",
    "get_available_providers",
    "get_providers_info",
    "build_search_subgraph",
]
