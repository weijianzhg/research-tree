"""Perplexity Search API adapter with frozen ranked-result provenance."""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from ..errors import ConfigurationError, ProviderError, ValidationError
from .base import SearchOptions, SearchResponse, SearchResult

DEFAULT_SEARCH_URL = "https://api.perplexity.ai/search"
RECENCY_VALUES = {"hour", "day", "week", "month", "year"}
DATE_RE = re.compile(r"^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}$")
CODE_RE = re.compile(r"^[A-Za-z]{2}$")
CONTEXT_VALUES = {"low", "medium", "high"}
ESTIMATED_REQUEST_COST_USD = 0.005
PRICING_SNAPSHOT_DATE = "2026-08-13"
RETRYABLE_STATUS_CODES = {408, 409, 429}


class _SameOriginHTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward a bearer credential to a different origin or plaintext transport."""

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        old = urllib.parse.urlparse(request.full_url)
        new = urllib.parse.urlparse(new_url)
        if new.scheme != "https" or (old.hostname, old.port) != (new.hostname, new.port):
            raise urllib.error.HTTPError(
                new_url,
                code,
                "Perplexity Search refused a cross-origin or non-HTTPS redirect",
                headers,
                file_pointer,
            )
        return super().redirect_request(request, file_pointer, code, message, headers, new_url)


_SAFE_OPENER = urllib.request.build_opener(_SameOriginHTTPSRedirectHandler())


def _key_from_research_config() -> str | None:
    path = Path.home() / ".config" / "research-tree" / "config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("perplexity_api_key")
    return value.strip() if isinstance(value, str) and value.strip() else None


def resolve_perplexity_key() -> str:
    """Resolve a Search API credential without persisting it in a graph."""
    key = os.environ.get("PERPLEXITY_API_KEY", "").strip() or _key_from_research_config()
    if not key:
        raise ConfigurationError(
            "no Perplexity credential found. Set PERPLEXITY_API_KEY or configure "
            "~/.config/research-tree/config.json"
        )
    return key


def _validate_options(options: SearchOptions) -> None:
    if (
        isinstance(options.max_results, bool)
        or not isinstance(options.max_results, int)
        or not 1 <= options.max_results <= 20
    ):
        raise ValidationError("Perplexity max_results must be between 1 and 20")
    if options.country is not None and (
        not isinstance(options.country, str) or not CODE_RE.fullmatch(options.country)
    ):
        raise ValidationError("Perplexity country must be a two-letter ISO code")
    if options.context_size is not None and (
        not isinstance(options.context_size, str) or options.context_size not in CONTEXT_VALUES
    ):
        raise ValidationError("Perplexity context_size must be low, medium, or high")
    if options.context_size and options.max_tokens_per_page is not None:
        raise ValidationError("Perplexity context_size cannot be combined with max_tokens_per_page")
    if (
        not isinstance(options.languages, tuple)
        or len(options.languages) > 20
        or any(
            not isinstance(language, str) or not CODE_RE.fullmatch(language)
            for language in options.languages
        )
    ):
        raise ValidationError("Perplexity languages must be up to 20 two-letter ISO codes")
    if not isinstance(options.domains, tuple):
        raise ValidationError("Perplexity domains must be a tuple of strings")
    domain_modes = {domain.startswith("-") for domain in options.domains if isinstance(domain, str)}
    if (
        len(options.domains) > 20
        or len(domain_modes) > 1
        or any(
            not isinstance(domain, str)
            or domain.startswith("--")
            or not domain.lstrip("-")
            or len(domain) > 253
            or "://" in domain
            or any(character.isspace() for character in domain)
            for domain in options.domains
        )
    ):
        raise ValidationError(
            "Perplexity domains must be up to 20 all-include or all-exclude host/path values"
        )
    if options.recency is not None and (
        not isinstance(options.recency, str) or options.recency not in RECENCY_VALUES
    ):
        raise ValidationError("Perplexity recency must be one of hour, day, week, month, or year")
    for label, value in (
        ("published_after", options.published_after),
        ("published_before", options.published_before),
        ("updated_after", options.updated_after),
        ("updated_before", options.updated_before),
    ):
        if value is not None and (not isinstance(value, str) or not DATE_RE.fullmatch(value)):
            raise ValidationError(f"Perplexity {label} must use MM/DD/YYYY")
    if options.max_tokens_per_page is not None and (
        isinstance(options.max_tokens_per_page, bool)
        or not isinstance(options.max_tokens_per_page, int)
        or not 1 <= options.max_tokens_per_page <= 1_000_000
    ):
        raise ValidationError("Perplexity max_tokens_per_page must be between 1 and 1000000")


def _request_payload(query: str, options: SearchOptions) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValidationError("search query cannot be empty")
    _validate_options(options)
    payload: dict[str, Any] = {"query": query, "max_results": options.max_results}
    if options.context_size:
        payload["search_context_size"] = options.context_size
    if options.country:
        payload["country"] = options.country.upper()
    if options.languages:
        payload["search_language_filter"] = [item.lower() for item in options.languages]
    if options.domains:
        payload["search_domain_filter"] = list(options.domains)
    if options.recency:
        payload["search_recency_filter"] = options.recency
    if options.published_after:
        payload["search_after_date_filter"] = options.published_after
    if options.published_before:
        payload["search_before_date_filter"] = options.published_before
    if options.updated_after:
        payload["last_updated_after_filter"] = options.updated_after
    if options.updated_before:
        payload["last_updated_before_filter"] = options.updated_before
    if options.max_tokens_per_page is not None:
        payload["max_tokens_per_page"] = options.max_tokens_per_page
    return payload


class PerplexitySearchClient:
    """Direct client for Perplexity's results-oriented Search API."""

    provider_name = "perplexity"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str | None = None,
        timeout: int = 90,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
        opener: Any = None,
    ):
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValidationError("Perplexity max_retries must be a non-negative integer")
        self.api_key = api_key.strip() if isinstance(api_key, str) else None
        self.url = (
            url or os.environ.get("RESEARCH_TREE_PERPLEXITY_URL") or DEFAULT_SEARCH_URL
        ).strip()
        parsed_url = urllib.parse.urlparse(self.url)
        if parsed_url.scheme != "https" or not parsed_url.hostname:
            raise ValidationError("Perplexity Search URL must be a valid HTTPS URL")
        self.timeout = timeout
        self.max_retries = max_retries
        self._sleep = sleep
        self._random = random_source
        self._open = (opener or _SAFE_OPENER).open

    def _retry_delay(self, headers: Any, attempt: int) -> float:
        try:
            milliseconds = headers.get("retry-after-ms")
            seconds = headers.get("Retry-After")
        except AttributeError:
            milliseconds = seconds = None
        try:
            if milliseconds is not None:
                return min(8.0, max(0.0, float(milliseconds) / 1000))
            if seconds is not None:
                return min(8.0, max(0.0, float(seconds)))
        except (TypeError, ValueError):
            pass
        return min(8.0, 0.5 * (2**attempt) + self._random() * 0.25)

    @staticmethod
    def _http_error_message(exc: urllib.error.HTTPError) -> str:
        try:
            body = exc.read(8192).decode("utf-8", errors="replace")
        except OSError:
            body = ""
        try:
            detail = json.loads(body)
        except json.JSONDecodeError:
            detail = body.strip() or exc.reason
        if isinstance(detail, dict):
            error = detail.get("error")
            if isinstance(error, dict):
                detail = error.get("message") or error
            else:
                detail = error or detail.get("message") or detail.get("detail") or detail
        rendered = json.dumps(detail, ensure_ascii=False) if not isinstance(detail, str) else detail
        return rendered[:2000]

    def search(self, *, query: str, options: SearchOptions | None = None) -> SearchResponse:
        options = options or SearchOptions()
        payload = _request_payload(query, options)
        key = self.api_key or resolve_perplexity_key()
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        data: Any = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._open(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code in RETRYABLE_STATUS_CODES or exc.code >= 500
                if retryable and attempt < self.max_retries:
                    self._sleep(self._retry_delay(exc.headers, attempt))
                    continue
                message = self._http_error_message(exc).replace(key, "[redacted]")
                raise ProviderError(
                    f"Perplexity Search request failed ({exc.code}): {message}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt < self.max_retries:
                    self._sleep(self._retry_delay(None, attempt))
                    continue
                raise ProviderError(f"Perplexity Search request failed: {exc}") from exc
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ProviderError("Perplexity Search returned invalid JSON") from exc

        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ProviderError("Perplexity Search response has no results array")
        results: list[SearchResult] = []
        for index, item in enumerate(data["results"]):
            if not isinstance(item, dict):
                raise ProviderError(f"Perplexity Search result {index + 1} is not an object")
            title, url, snippet = item.get("title"), item.get("url"), item.get("snippet")
            if not isinstance(title, str) or not title.strip():
                raise ProviderError(f"Perplexity Search result {index + 1} has no title")
            if not isinstance(url, str) or not url.strip():
                raise ProviderError(f"Perplexity Search result {index + 1} has an invalid URL")
            normalized_url = url.strip()
            parsed_result_url = urllib.parse.urlparse(normalized_url)
            if parsed_result_url.scheme in {"http", "https"} and (
                not parsed_result_url.hostname
                or any(character.isspace() for character in normalized_url)
            ):
                raise ProviderError(f"Perplexity Search result {index + 1} has an invalid URL")
            if not isinstance(snippet, str):
                raise ProviderError(f"Perplexity Search result {index + 1} has no snippet")
            published_at = item.get("date")
            last_updated = item.get("last_updated")
            if published_at is not None and not isinstance(published_at, str):
                raise ProviderError(f"Perplexity Search result {index + 1} has an invalid date")
            if last_updated is not None and not isinstance(last_updated, str):
                raise ProviderError(
                    f"Perplexity Search result {index + 1} has an invalid last_updated"
                )
            results.append(
                SearchResult(
                    title=title.strip(),
                    url=normalized_url,
                    snippet=snippet.strip(),
                    published_at=published_at or None,
                    last_updated=last_updated or None,
                )
            )
        request_id = data.get("id")
        server_time = data.get("server_time")
        if not isinstance(request_id, str) or not request_id:
            raise ProviderError("Perplexity Search response has an invalid id")
        if server_time is not None and not isinstance(server_time, str):
            raise ProviderError("Perplexity Search response has an invalid server_time")
        return SearchResponse(
            query=payload["query"],
            results=results,
            request=payload,
            request_id=request_id,
            server_time=server_time,
            usage={
                "search_calls": 1,
                "query_units": 1,
                "estimated_cost_usd": ESTIMATED_REQUEST_COST_USD,
                "pricing_snapshot_date": PRICING_SNAPSHOT_DATE,
            },
            raw=data,
            provider_name=self.provider_name,
        )
