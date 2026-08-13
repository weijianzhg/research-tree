"""Small OpenRouter client with model-controlled web search and provenance capture."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..errors import ConfigurationError, ProviderError

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class ProviderResponse:
    content: str
    requested_model: str
    resolved_model: str
    annotations: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    response_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def _key_from_research_config() -> str | None:
    path = Path.home() / ".config" / "research-tree" / "config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("openrouter_api_key")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _key_from_fluff_cutter() -> str | None:
    path = Path.home() / ".fluff-cutter" / "config.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    value = data.get("openrouter_api_key")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _key_from_pi(model: str) -> str | None:
    try:
        result = subprocess.run(
            [
                "pi",
                "auth",
                "print-api-key",
                "--provider",
                "openrouter",
                "--model",
                model,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    key = result.stdout.strip()
    return key if result.returncode == 0 and key else None


def resolve_openrouter_key(model: str) -> str:
    """Resolve credentials without ever printing or persisting them."""
    key = (
        os.environ.get("OPENROUTER_API_KEY", "").strip()
        or _key_from_research_config()
        or _key_from_pi(model)
        or _key_from_fluff_cutter()
    )
    if not key:
        raise ConfigurationError(
            "no OpenRouter credential found. Set OPENROUTER_API_KEY, configure "
            "~/.fluff-cutter/config.yaml, or sign into OpenRouter in Pi"
        )
    return key


class OpenRouterClient:
    provider_name = "openrouter"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 900,
        app_title: str = "Research Tree",
    ):
        self.api_key = api_key
        self.base_url = (
            base_url or os.environ.get("RESEARCH_TREE_OPENROUTER_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = timeout
        self.app_title = app_title

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        web: bool = False,
        reasoning_effort: str | None = "high",
        response_schema: dict[str, Any] | None = None,
        max_tokens: int = 8000,
        max_search_results: int = 8,
    ) -> ProviderResponse:
        key = self.api_key or resolve_openrouter_key(model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort, "exclude": True}
        if web:
            payload["tools"] = [
                {
                    "type": "openrouter:web_search",
                    "parameters": {
                        "max_results": min(max_search_results, 10),
                        "max_total_results": max_search_results,
                        "search_context_size": "high",
                    },
                },
                {"type": "openrouter:web_fetch"},
            ]
        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "research_result",
                    "strict": True,
                    "schema": response_schema,
                },
            }

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "X-Title": self.app_title,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("error", {})
                message = detail.get("message") or str(detail)
            except (json.JSONDecodeError, AttributeError):
                message = exc.reason
            raise ProviderError(f"OpenRouter request failed ({exc.code}): {message}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"OpenRouter request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError("OpenRouter returned invalid JSON") from exc

        try:
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if not isinstance(content, str):
                raise TypeError("message content is not text")
        except (KeyError, IndexError, TypeError) as exc:
            error = data.get("error") if isinstance(data, dict) else None
            raise ProviderError(f"OpenRouter response has no answer: {error or exc}") from exc

        return ProviderResponse(
            content=content,
            requested_model=model,
            resolved_model=str(data.get("model") or model),
            annotations=list(message.get("annotations") or []),
            usage=dict(data.get("usage") or {}),
            response_id=str(data.get("id") or ""),
            raw=data,
        )
