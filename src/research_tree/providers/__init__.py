"""Model and retrieval provider adapters."""

from .base import (
    ChatProvider,
    ProviderResponse,
    SearchOptions,
    SearchProvider,
    SearchResponse,
    SearchResult,
)
from .openrouter import OpenRouterClient, resolve_openrouter_key
from .perplexity import PerplexitySearchClient, resolve_perplexity_key

__all__ = [
    "ChatProvider",
    "OpenRouterClient",
    "PerplexitySearchClient",
    "ProviderResponse",
    "SearchOptions",
    "SearchProvider",
    "SearchResponse",
    "SearchResult",
    "resolve_openrouter_key",
    "resolve_perplexity_key",
]
