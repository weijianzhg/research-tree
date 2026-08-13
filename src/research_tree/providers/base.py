"""Provider-neutral request and response contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ProviderResponse:
    """One structured chat completion plus its provider provenance."""

    content: str
    requested_model: str
    resolved_model: str
    annotations: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    response_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    provider_name: str = "openrouter"


@dataclass(frozen=True)
class SearchOptions:
    """Portable subset of real-time web-search controls."""

    max_results: int = 8
    context_size: str | None = "high"
    country: str | None = None
    languages: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    recency: str | None = None
    published_after: str | None = None
    published_before: str | None = None
    updated_after: str | None = None
    updated_before: str | None = None
    max_tokens_per_page: int | None = None


@dataclass(frozen=True)
class SearchResult:
    """Normalized ranked result returned by a retrieval provider."""

    title: str
    url: str
    snippet: str
    published_at: str | None = None
    last_updated: str | None = None


@dataclass
class SearchResponse:
    """A raw retrieval response and its normalized results."""

    query: str
    results: list[SearchResult]
    request: dict[str, Any]
    request_id: str = ""
    server_time: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    provider_name: str = ""


class ChatProvider(Protocol):
    provider_name: str

    def chat(self, **kwargs: Any) -> ProviderResponse: ...


class SearchProvider(Protocol):
    provider_name: str

    def search(self, *, query: str, options: SearchOptions | None = None) -> SearchResponse: ...
