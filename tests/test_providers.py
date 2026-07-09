"""Providers construct from constructor params only — no global config resolution.

These assert the observable behavior after cutting the ``resolve_search_runtime_config``
seam: a key-requiring provider is available iff an api_key was injected, and the
keyless providers (searxng/duckduckgo) are always constructible and available.
"""

import pytest

from search_agent import get_provider
from search_agent.providers import _DEPRECATED_UNSUPPORTED


def test_searxng_constructs_without_api_key():
    provider = get_provider("searxng")
    assert provider.requires_api_key is False
    assert provider.supports_answer is False
    assert provider.is_available() is True


def test_duckduckgo_constructs_without_api_key():
    provider = get_provider("duckduckgo")
    assert provider.requires_api_key is False
    assert provider.is_available() is True


def test_key_requiring_provider_unavailable_without_key():
    # Constructing a key-requiring provider with no key must not reach any global
    # config; _get_api_key raises, so construction fails cleanly.
    with pytest.raises(ValueError):
        get_provider("brave")


def test_key_requiring_provider_available_with_injected_key():
    provider = get_provider("brave", api_key="injected-key")
    assert provider.api_key == "injected-key"
    assert provider.is_available() is True


def test_proxy_is_taken_from_constructor_kwargs():
    provider = get_provider("searxng", proxy="http://proxy:8080")
    assert provider.proxy == "http://proxy:8080"


@pytest.mark.parametrize("name", sorted(_DEPRECATED_UNSUPPORTED))
def test_deprecated_providers_raise(name):
    with pytest.raises(ValueError):
        get_provider(name)


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_provider("does-not-exist")


def test_registry_lists_only_supported_providers():
    from search_agent import list_providers

    names = set(list_providers())
    assert "searxng" in names
    assert "duckduckgo" in names
    assert names.isdisjoint(_DEPRECATED_UNSUPPORTED)
