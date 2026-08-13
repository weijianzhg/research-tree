from __future__ import annotations

import io
import json
import urllib.error

import pytest

from research_tree.errors import ConfigurationError, ProviderError, ValidationError
from research_tree.providers import perplexity
from research_tree.providers.base import SearchOptions
from research_tree.providers.perplexity import PerplexitySearchClient, resolve_perplexity_key


class FakeHTTPResponse:
    def __init__(self, body: bytes):
        self.body = body

    @classmethod
    def json(cls, value):
        return cls(json.dumps(value).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def install_response(monkeypatch, value, *, captured=None):
    response = value if isinstance(value, FakeHTTPResponse) else FakeHTTPResponse.json(value)

    def fake_urlopen(request, timeout):
        if captured is not None:
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["headers"] = {key.lower(): item for key, item in request.header_items()}
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
        return response

    monkeypatch.setattr(perplexity._SAFE_OPENER, "open", fake_urlopen)


def ok_response(*, results=None, request_id="search-1", server_time=None):
    return {
        "id": request_id,
        "results": [] if results is None else results,
        "server_time": server_time,
    }


def perform_search(monkeypatch, *, options=None, response=None):
    install_response(monkeypatch, response or ok_response())
    return PerplexitySearchClient(api_key="test-key").search(
        query="test query",
        options=options,
    )


def test_credential_resolution_prefers_environment(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", " environment-key ")
    monkeypatch.setattr(perplexity, "_key_from_research_config", lambda: "config-key")

    assert resolve_perplexity_key() == "environment-key"


def test_credential_resolution_uses_config_and_reports_missing(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setattr(perplexity, "_key_from_research_config", lambda: "config-key")
    assert resolve_perplexity_key() == "config-key"

    monkeypatch.setattr(perplexity, "_key_from_research_config", lambda: None)
    with pytest.raises(ConfigurationError, match="PERPLEXITY_API_KEY"):
        resolve_perplexity_key()


def test_search_sends_exact_filter_payload_and_headers(monkeypatch):
    captured = {}
    install_response(monkeypatch, ok_response(), captured=captured)
    client = PerplexitySearchClient(
        api_key="not-a-real-key",
        url="https://search.test/search",
        timeout=37,
    )

    client.search(
        query="  current model releases  ",
        options=SearchOptions(
            max_results=20,
            context_size=None,
            country="gb",
            languages=("EN", "fr"),
            domains=("huggingface.co", "github.com/org"),
            recency="month",
            published_after="01/02/2026",
            published_before="08/13/2026",
            updated_after="02/03/2026",
            updated_before="08/12/2026",
            max_tokens_per_page=4096,
        ),
    )

    assert captured == {
        "url": "https://search.test/search",
        "method": "POST",
        "headers": {
            "authorization": "Bearer not-a-real-key",
            "accept": "application/json",
            "content-type": "application/json",
        },
        "payload": {
            "query": "current model releases",
            "max_results": 20,
            "country": "GB",
            "search_language_filter": ["en", "fr"],
            "search_domain_filter": ["huggingface.co", "github.com/org"],
            "search_recency_filter": "month",
            "search_after_date_filter": "01/02/2026",
            "search_before_date_filter": "08/13/2026",
            "last_updated_after_filter": "02/03/2026",
            "last_updated_before_filter": "08/12/2026",
            "max_tokens_per_page": 4096,
        },
        "timeout": 37,
    }


def test_default_request_uses_high_context_and_explicit_key(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        perplexity,
        "resolve_perplexity_key",
        lambda: pytest.fail("explicit key should avoid credential resolution"),
    )
    install_response(monkeypatch, ok_response(), captured=captured)

    PerplexitySearchClient(api_key="explicit-key").search(query="question")

    assert captured["payload"] == {
        "query": "question",
        "max_results": 8,
        "search_context_size": "high",
    }
    assert captured["headers"]["authorization"] == "Bearer explicit-key"


def test_normal_response_preserves_ranked_results_and_nullable_metadata(monkeypatch):
    payload = ok_response(
        request_id="request-123",
        server_time="2026-08-13T12:00:00Z",
        results=[
            {
                "title": " Official release ",
                "url": " https://example.com/release ",
                "snippet": "\nDetailed evidence.\n",
                "date": "2026-08-01",
                "last_updated": "2026-08-12",
                "future_field": {"kept": "in raw"},
            },
            {
                "title": "Undated result",
                "url": "https://example.org/undated",
                "snippet": "",
                "date": None,
            },
        ],
    )
    install_response(monkeypatch, payload)

    response = PerplexitySearchClient(api_key="test-key").search(query=" evidence ")

    assert response.provider_name == "perplexity"
    assert response.query == "evidence"
    assert response.request_id == "request-123"
    assert response.server_time == "2026-08-13T12:00:00Z"
    assert [result.url for result in response.results] == [
        "https://example.com/release",
        "https://example.org/undated",
    ]
    assert response.results[0].title == "Official release"
    assert response.results[0].snippet == "Detailed evidence."
    assert response.results[0].published_at == "2026-08-01"
    assert response.results[0].last_updated == "2026-08-12"
    assert response.results[1].snippet == ""
    assert response.results[1].published_at is None
    assert response.results[1].last_updated is None
    assert response.raw == payload
    assert response.usage == {
        "search_calls": 1,
        "query_units": 1,
        "estimated_cost_usd": 0.005,
        "pricing_snapshot_date": "2026-08-13",
    }


def test_successful_empty_response_is_valid_and_records_cost(monkeypatch):
    install_response(monkeypatch, {"id": "empty-1", "results": []})

    response = PerplexitySearchClient(api_key="test-key").search(query="obscure query")

    assert response.results == []
    assert response.server_time is None
    assert response.usage["estimated_cost_usd"] == 0.005


def test_non_http_result_is_preserved_for_provenance(monkeypatch):
    install_response(
        monkeypatch,
        ok_response(
            results=[
                {
                    "title": "Unexpected scheme",
                    "url": "ftp://archive.example.org/model.txt",
                    "snippet": "Provider-returned metadata must remain inspectable.",
                    "date": None,
                    "last_updated": None,
                }
            ]
        ),
    )

    response = PerplexitySearchClient(api_key="test-key").search(query="model archive")

    assert response.results[0].url == "ftp://archive.example.org/model.txt"
    assert response.raw["results"][0]["url"] == "ftp://archive.example.org/model.txt"


def _http_error(status, body, headers=None):
    return urllib.error.HTTPError(
        "https://api.perplexity.ai/search",
        status,
        "failed",
        headers or {},
        io.BytesIO(json.dumps(body).encode()),
    )


def test_rate_limit_honors_retry_after_then_succeeds(monkeypatch):
    attempts = [
        _http_error(429, {"message": "slow down"}, {"Retry-After": "0.25"}),
        FakeHTTPResponse.json(ok_response(request_id="retry-ok")),
    ]
    delays = []
    monkeypatch.setattr(
        perplexity._SAFE_OPENER,
        "open",
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(attempts.pop(0))
            if isinstance(attempts[0], Exception)
            else attempts.pop(0)
        ),
    )

    response = PerplexitySearchClient(
        api_key="test-key", sleep=delays.append, random_source=lambda: 0
    ).search(query="question")

    assert response.request_id == "retry-ok"
    assert delays == [0.25]
    assert attempts == []


def test_validation_http_error_is_not_retried_and_key_is_not_exposed(monkeypatch):
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise _http_error(
            422,
            {"detail": [{"loc": ["body", "max_results"], "msg": "too large"}]},
        )

    monkeypatch.setattr(perplexity._SAFE_OPENER, "open", fail)
    with pytest.raises(ProviderError, match="too large") as caught:
        PerplexitySearchClient(api_key="super-secret").search(query="question")
    assert calls == 1
    assert "super-secret" not in str(caught.value)


def test_transient_transport_failure_retries_only_to_configured_limit(monkeypatch):
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(perplexity._SAFE_OPENER, "open", fail)
    with pytest.raises(ProviderError, match="connection reset"):
        PerplexitySearchClient(
            api_key="test-key",
            max_retries=2,
            sleep=lambda _: None,
            random_source=lambda: 0,
        ).search(query="question")
    assert calls == 3


def test_invalid_client_transport_configuration_is_rejected():
    with pytest.raises(ValidationError, match="max_retries"):
        PerplexitySearchClient(api_key="x", max_retries=True)
    with pytest.raises(ValidationError, match="HTTP"):
        PerplexitySearchClient(api_key="x", url="file:///tmp/search")


def test_custom_endpoint_must_be_https_and_redirect_handler_blocks_other_origins():
    with pytest.raises(ValidationError, match="HTTPS"):
        PerplexitySearchClient(api_key="x", url="http://localhost/search")
    handler = perplexity._SameOriginHTTPSRedirectHandler()
    request = perplexity.urllib.request.Request(
        "https://api.perplexity.ai/search",
        headers={"Authorization": "Bearer secret"},
    )
    with pytest.raises(perplexity.urllib.error.HTTPError, match="refused"):
        handler.redirect_request(
            request,
            None,
            307,
            "redirect",
            {},
            "https://attacker.example/search",
        )


@pytest.mark.parametrize(
    "options",
    [
        SearchOptions(max_results=1.5),
        SearchOptions(context_size=123),
        SearchOptions(country=123),
        SearchOptions(languages=["en"]),
        SearchOptions(domains=["example.com"]),
        SearchOptions(recency=123),
        SearchOptions(published_after=123),
        SearchOptions(context_size=None, max_tokens_per_page=1.5),
    ],
)
def test_wrong_option_types_raise_validation_error(monkeypatch, options):
    with pytest.raises(ValidationError):
        perform_search(monkeypatch, options=options)


def test_non_object_config_is_treated_as_missing(monkeypatch, tmp_path):
    config = tmp_path / ".config" / "research-tree" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(perplexity.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="PERPLEXITY_API_KEY"):
        resolve_perplexity_key()


def test_invalid_utf8_success_body_is_a_provider_error(monkeypatch):
    install_response(monkeypatch, FakeHTTPResponse(b"\xff"))
    with pytest.raises(ProviderError, match="invalid JSON"):
        PerplexitySearchClient(api_key="x").search(query="question")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"not-json", "invalid JSON"),
        (json.dumps([]).encode(), "no results array"),
        (json.dumps({"id": "x"}).encode(), "no results array"),
        (json.dumps({"id": "x", "results": ["bad"]}).encode(), "not an object"),
        (
            json.dumps(
                {
                    "id": "x",
                    "results": [{"title": "", "url": "https://x.test", "snippet": "x"}],
                }
            ).encode(),
            "has no title",
        ),
        (
            json.dumps(
                {
                    "id": "x",
                    "results": [{"title": "x", "url": "", "snippet": "x"}],
                }
            ).encode(),
            "invalid URL",
        ),
        (
            json.dumps(
                {
                    "id": "x",
                    "results": [{"title": "x", "url": "https://", "snippet": "x"}],
                }
            ).encode(),
            "invalid URL",
        ),
        (
            json.dumps(
                {
                    "id": "x",
                    "results": [{"title": "x", "url": "https://x.test"}],
                }
            ).encode(),
            "has no snippet",
        ),
        (
            json.dumps(
                {
                    "id": "x",
                    "results": [
                        {
                            "title": "x",
                            "url": "https://x.test",
                            "snippet": "x",
                            "date": 123,
                        }
                    ],
                }
            ).encode(),
            "invalid date",
        ),
        (json.dumps({"id": "", "results": []}).encode(), "invalid id"),
        (
            json.dumps({"id": "x", "results": [], "server_time": 123}).encode(),
            "invalid server_time",
        ),
    ],
)
def test_malformed_responses_raise_provider_error(monkeypatch, body, message):
    install_response(monkeypatch, FakeHTTPResponse(body))

    with pytest.raises(ProviderError, match=message):
        PerplexitySearchClient(api_key="test-key").search(query="question")


@pytest.mark.parametrize("value", [1, 20])
def test_max_results_accepts_documented_boundaries(monkeypatch, value):
    response = perform_search(monkeypatch, options=SearchOptions(max_results=value))
    assert response.request["max_results"] == value


@pytest.mark.parametrize("value", [0, 21, True])
def test_max_results_rejects_values_outside_documented_boundaries(monkeypatch, value):
    monkeypatch.setattr(
        perplexity._SAFE_OPENER,
        "open",
        lambda *args, **kwargs: pytest.fail("invalid options must fail before network I/O"),
    )
    with pytest.raises(ValidationError, match="max_results"):
        PerplexitySearchClient(api_key="test-key").search(
            query="question", options=SearchOptions(max_results=value)
        )


@pytest.mark.parametrize("value", [1, 1_000_000])
def test_token_budget_accepts_documented_boundaries(monkeypatch, value):
    response = perform_search(
        monkeypatch,
        options=SearchOptions(context_size=None, max_tokens_per_page=value),
    )
    assert response.request["max_tokens_per_page"] == value


@pytest.mark.parametrize("value", [0, 1_000_001, True])
def test_token_budget_rejects_values_outside_documented_boundaries(monkeypatch, value):
    with pytest.raises(ValidationError, match="max_tokens_per_page"):
        perform_search(
            monkeypatch,
            options=SearchOptions(context_size=None, max_tokens_per_page=value),
        )


def test_context_size_and_explicit_token_budget_are_mutually_exclusive(monkeypatch):
    with pytest.raises(ValidationError, match="cannot be combined"):
        perform_search(
            monkeypatch,
            options=SearchOptions(context_size="high", max_tokens_per_page=4096),
        )


@pytest.mark.parametrize(
    "options",
    [
        SearchOptions(country="G"),
        SearchOptions(languages=("en",) * 21),
        SearchOptions(languages=("english",)),
        SearchOptions(domains=tuple(f"example{index}.com" for index in range(21))),
        SearchOptions(domains=("example.com", "-spam.example")),
        SearchOptions(domains=("https://example.com",)),
        SearchOptions(domains=("--example.com",)),
        SearchOptions(domains=("a" * 254,)),
        SearchOptions(recency="decade"),
        SearchOptions(published_after="2026-08-13"),
        SearchOptions(context_size="huge"),
    ],
)
def test_invalid_filter_boundaries_fail_before_network_io(monkeypatch, options):
    monkeypatch.setattr(
        perplexity._SAFE_OPENER,
        "open",
        lambda *args, **kwargs: pytest.fail("invalid options must fail before network I/O"),
    )
    with pytest.raises(ValidationError):
        PerplexitySearchClient(api_key="test-key").search(query="question", options=options)


def test_maximum_filter_counts_and_lengths_are_accepted(monkeypatch):
    longest_domain = "a" * 249 + ".com"
    response = perform_search(
        monkeypatch,
        options=SearchOptions(
            country="US",
            languages=("en",) * 20,
            domains=(longest_domain,) * 20,
            recency="hour",
        ),
    )

    assert len(response.request["search_language_filter"]) == 20
    assert len(response.request["search_domain_filter"]) == 20
    assert len(response.request["search_domain_filter"][0]) == 253


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_fails_before_credential_or_network_resolution(monkeypatch, query):
    monkeypatch.setattr(
        perplexity,
        "resolve_perplexity_key",
        lambda: pytest.fail("invalid query must fail before credential resolution"),
    )
    monkeypatch.setattr(
        perplexity._SAFE_OPENER,
        "open",
        lambda *args, **kwargs: pytest.fail("invalid query must fail before network I/O"),
    )

    with pytest.raises(ValidationError, match="query cannot be empty"):
        PerplexitySearchClient().search(query=query)
