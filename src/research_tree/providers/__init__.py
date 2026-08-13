"""Model provider adapters."""

from .openrouter import OpenRouterClient, ProviderResponse, resolve_openrouter_key

__all__ = ["OpenRouterClient", "ProviderResponse", "resolve_openrouter_key"]
