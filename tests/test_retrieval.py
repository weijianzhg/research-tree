from __future__ import annotations

import json
import threading

import pytest

import research_tree.research as research_module
from research_tree.doctor import inspect_graph
from research_tree.errors import ProviderError, ValidationError
from research_tree.providers import ProviderResponse, SearchOptions, SearchResponse, SearchResult
from research_tree.research import (
    ANSWER_SCHEMA,
    COUNCIL_SCHEMA,
    EVIDENCE_RESULT_PROMPT_LIMIT,
    REVIEW_SCHEMA,
    ask_question,
    resolve_search_provider,
    run_council,
    search_question,
)
from research_tree.store import GraphStore

SEARCH_URL = "https://example.com/original-model-card"
SEARCH_SNIPPET = (
    "The original model card reports that sparse routing selects two experts per token, "
    "with the benchmark configuration and limitations documented in the release artifact."
)


def _answer_payload(label: str = "answer", *, council: bool = False) -> dict:
    payload = {
        "answer_markdown": f"{label}: Sparse routing activates two experts per token.",
        "confidence": 0.82,
        "claims": [
            {
                "text": "The released model activates two experts per token.",
                "confidence": 0.86,
                "source_urls": [SEARCH_URL],
            }
        ],
        "uncertainties": ["The model card does not independently reproduce throughput claims."],
        "follow_up_questions": [],
    }
    if council:
        payload.update(
            {
                "consensus": ["The release uses sparse expert routing."],
                "disagreements": ["The end-to-end speed benefit remains workload-dependent."],
            }
        )
    return payload


class RecordingRetriever:
    provider_name = "perplexity"

    def __init__(self):
        self.calls: list[dict] = []

    def search(self, *, query: str, options: SearchOptions | None = None) -> SearchResponse:
        self.calls.append({"query": query, "options": options})
        request = {
            "query": query,
            "max_results": options.max_results if options else 8,
        }
        return SearchResponse(
            query=query,
            results=[
                SearchResult(
                    title="Original model card",
                    url=SEARCH_URL,
                    snippet=SEARCH_SNIPPET,
                    published_at="2026-08-01",
                    last_updated="2026-08-10",
                )
            ],
            request=request,
            request_id="search-request-1",
            server_time="2026-08-13T10:00:00Z",
            usage={"search_calls": 1},
            raw={"id": "search-request-1", "results": [{"url": SEARCH_URL}]},
            provider_name=self.provider_name,
        )


class RecordingChatClient:
    provider_name = "openrouter"

    def __init__(self):
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def chat(self, **kwargs) -> ProviderResponse:
        with self.lock:
            self.calls.append(kwargs)
        schema = kwargs["response_schema"]
        if schema == REVIEW_SCHEMA:
            payload = {
                "strongest_response": "Response A",
                "evidence_strengths": ["The response cites the supplied model card."],
                "unsupported_or_weak_claims": [],
                "disagreements": ["Throughput depends on the workload."],
                "missing_questions": ["Was the result independently reproduced?"],
            }
        elif schema == COUNCIL_SCHEMA:
            payload = _answer_payload("synthesis", council=True)
        else:
            assert schema == ANSWER_SCHEMA
            payload = _answer_payload(kwargs["model"])
        return ProviderResponse(
            content=json.dumps(payload),
            requested_model=kwargs["model"],
            resolved_model=kwargs["model"] + "-resolved",
            # Deliberately omit provider annotations. The cited claim must map back to the
            # richer frozen Perplexity snapshot rather than an empty URL-only source.
            annotations=[],
            usage={"cost": 0.01},
            raw={"model": kwargs["model"], "payload": payload},
            provider_name=self.provider_name,
        )


class StageOneFailureChatClient(RecordingChatClient):
    def chat(self, **kwargs) -> ProviderResponse:
        if kwargs["response_schema"] == ANSWER_SCHEMA and kwargs["model"] == "model/b":
            with self.lock:
                self.calls.append(kwargs)
            raise ProviderError("member unavailable")
        return super().chat(**kwargs)


class InventedUrlChatClient(RecordingChatClient):
    def chat(self, **kwargs):
        response = super().chat(**kwargs)
        payload = json.loads(response.content)
        payload["claims"][0]["source_urls"] = ["https://invented.example/not-retrieved"]
        response.content = json.dumps(payload)
        response.raw["payload"] = payload
        return response


def test_standalone_search_is_durable_without_answering_question(store):
    retriever = RecordingRetriever()
    before = store.load_node("root")

    outcome = search_question(
        store,
        "root",
        client=retriever,
        options=SearchOptions(max_results=4, recency="month"),
    )

    assert len(retriever.calls) == 1
    assert outcome.query == before.title
    assert outcome.run.mode == "search"
    assert outcome.run.provider == "perplexity"
    assert outcome.run.requested_models == outcome.run.resolved_models == []
    assert outcome.run.raw["request_id"] == "search-request-1"
    assert outcome.run.raw["ranked_results"] == [
        {
            "rank": 1,
            "source_id": outcome.sources[0].id,
            "title": "Original model card",
            "url": SEARCH_URL,
            "published_at": "2026-08-01",
            "last_updated": "2026-08-10",
        }
    ]

    persisted_question = store.load_node("root")
    assert persisted_question.status == before.status
    assert outcome.run.id in persisted_question.run_ids
    assert outcome.sources[0].id in persisted_question.source_ids
    assert store.load_run(outcome.run.id).to_dict() == outcome.run.to_dict()
    persisted_source = store.load_source(outcome.sources[0].id)
    assert persisted_source.excerpt == SEARCH_SNIPPET
    assert persisted_source.metadata == {
        "via": "perplexity_search",
        "provider": "perplexity",
        "evidence_scope": "search_excerpt",
    }
    assert inspect_graph(GraphStore(store.root)).healthy


def test_doctor_detects_corrupt_search_result_manifest(store):
    outcome = search_question(store, "root", client=RecordingRetriever())
    path = store.run_path(outcome.run.id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["raw"]["ranked_results"][0]["rank"] = 2
    path.write_text(json.dumps(data), encoding="utf-8")
    report = inspect_graph(store)
    assert not report.healthy
    assert any("invalid result ranking" in error for error in report.errors)


def test_search_rejects_source_id_collision_with_existing_snapshot(store, monkeypatch):
    first = search_question(store, "root", client=RecordingRetriever()).sources[0]
    monkeypatch.setattr(research_module, "stable_source_id", lambda url, excerpt: first.id)

    class CollidingRetriever(RecordingRetriever):
        def search(self, *, query, options=None):
            response = super().search(query=query, options=options)
            response.results[0] = SearchResult(
                title="Different source",
                url="https://different.example/source",
                snippet="Different content",
            )
            return response

    with pytest.raises(ValidationError, match="ID collision"):
        search_question(store, "root", client=CollidingRetriever())


def test_search_rejects_source_id_collision_inside_one_response(store, monkeypatch):
    monkeypatch.setattr(research_module, "stable_source_id", lambda url, excerpt: "s_aaaaaaaaaaaa")

    class CollidingRetriever(RecordingRetriever):
        def search(self, *, query, options=None):
            response = super().search(query=query, options=options)
            response.results.append(
                SearchResult(
                    title="Different source",
                    url="https://different.example/source",
                    snippet="Different content",
                )
            )
            return response

    with pytest.raises(ValidationError, match="ID collision"):
        search_question(store, "root", client=CollidingRetriever())


def test_search_rejects_explicit_blank_query_before_retrieval(store):
    retriever = RecordingRetriever()
    with pytest.raises(ValidationError, match="query cannot be empty"):
        search_question(store, "root", client=retriever, query="   ")
    assert retriever.calls == []


def test_retrieved_excerpt_is_fully_stored_but_bounded_in_model_prompt(store):
    long_snippet = "evidence " * (EVIDENCE_RESULT_PROMPT_LIMIT // 4)

    class LongRetriever(RecordingRetriever):
        def search(self, *, query, options=None):
            response = super().search(query=query, options=options)
            original = response.results[0]
            response.results[0] = SearchResult(
                title=original.title,
                url=original.url,
                snippet=long_snippet,
            )
            return response

    retriever = LongRetriever()
    chat = RecordingChatClient()
    outcome = ask_question(
        store,
        "root",
        client=chat,
        retriever=retriever,
        model="model/author",
        search_provider="perplexity",
    )
    prompt = chat.calls[0]["messages"][-1]["content"]
    assert "[excerpt truncated]" in prompt
    source = next(source for source in outcome.sources if source.url == SEARCH_URL)
    assert source.excerpt == long_snippet


def test_perplexity_only_does_not_materialize_model_invented_urls(store):
    outcome = ask_question(
        store,
        "root",
        client=InventedUrlChatClient(),
        retriever=RecordingRetriever(),
        model="model/author",
        search_provider="perplexity",
    )
    assert outcome.answer.source_ids == []
    assert outcome.claims[0].source_ids == []
    assert all(source.url != "https://invented.example/not-retrieved" for source in outcome.sources)
    assert inspect_graph(store).healthy


def test_claim_only_url_is_marked_unfetched_outside_frozen_retrieval(store):
    outcome = ask_question(
        store,
        "root",
        client=InventedUrlChatClient(),
        model="model/author",
        search_provider="none",
    )
    source = next(
        source
        for source in outcome.sources
        if source.url == "https://invented.example/not-retrieved"
    )
    assert source.metadata == {
        "via": "model_claim_url",
        "provider": "openrouter",
        "evidence_scope": "unfetched_claim_url",
    }


def test_legacy_web_flags_and_new_provider_selection_are_compatible():
    legacy = {"web_search": True}
    assert resolve_search_provider(legacy) == "openrouter"
    assert resolve_search_provider({"web_search": False}) == "none"
    assert resolve_search_provider({"web_search": True, "search_provider": "none"}) == "none"
    assert (
        resolve_search_provider({"web_search": True, "search_provider": "none"}, web=True)
        == "openrouter"
    )
    with pytest.raises(ValidationError, match="cannot be combined"):
        resolve_search_provider(legacy, web=False, search_provider="perplexity")
    with pytest.raises(ValidationError, match="cannot be combined"):
        resolve_search_provider(legacy, web=True, search_provider="perplexity")


@pytest.mark.parametrize(
    ("search_provider", "expected_web"),
    [("perplexity", False), ("both", True)],
)
def test_ask_uses_one_shared_search_and_correct_native_web_mode(
    store, search_provider, expected_web
):
    retriever = RecordingRetriever()
    chat = RecordingChatClient()

    outcome = ask_question(
        store,
        "root",
        client=chat,
        retriever=retriever,
        model="model/author",
        search_provider=search_provider,
        search_options=SearchOptions(max_results=3),
    )

    assert len(retriever.calls) == 1
    assert len(chat.calls) == 1
    assert chat.calls[0]["web"] is expected_web
    prompt = chat.calls[0]["messages"][-1]["content"]
    assert "Frozen real-time search results:" in prompt
    assert SEARCH_URL in prompt
    assert SEARCH_SNIPPET in prompt
    assert "published 2026-08-01" in prompt

    source_id = outcome.sources[0].id
    assert store.load_source(source_id).excerpt == SEARCH_SNIPPET
    assert outcome.answer.source_ids == [source_id]
    assert outcome.claims[0].source_ids == [source_id]
    assert outcome.run.raw["search_provider"] == search_provider
    retrieval_run_id = outcome.run.raw["retrieval"]["run_id"]
    assert {retrieval_run_id, outcome.run.id}.issubset(outcome.answer.run_ids)
    assert store.load_run(retrieval_run_id).mode == "search"
    assert inspect_graph(store).healthy


def test_council_searches_once_and_shares_frozen_evidence_across_every_stage(store):
    retriever = RecordingRetriever()
    chat = RecordingChatClient()

    outcome = run_council(
        store,
        "root",
        client=chat,
        retriever=retriever,
        models=["model/a", "model/b"],
        chairman_model="model/chair",
        search_provider="perplexity",
    )

    assert len(retriever.calls) == 1
    assert len(chat.calls) == 5  # two answers, two blind reviews, one chairman
    assert all(call["web"] is False for call in chat.calls)
    for call in chat.calls:
        prompt = call["messages"][-1]["content"]
        assert "Frozen real-time search results:" in prompt
        assert SEARCH_URL in prompt
        assert SEARCH_SNIPPET in prompt

    source_id = next(source.id for source in outcome.sources if source.url == SEARCH_URL)
    assert outcome.answer.source_ids == [source_id]
    assert outcome.claims[0].source_ids == [source_id]
    assert all(perspective.source_ids == [source_id] for perspective in outcome.perspectives)
    assert store.load_source(source_id).excerpt == SEARCH_SNIPPET
    assert outcome.run.raw["retrieval"]["provider"] == "perplexity"
    assert inspect_graph(store).healthy


def test_failed_council_retains_shared_retrieval_and_failed_attempt_provenance(store):
    retriever = RecordingRetriever()
    chat = StageOneFailureChatClient()

    with pytest.raises(ProviderError, match="preserved attempt as"):
        run_council(
            store,
            "root",
            client=chat,
            retriever=retriever,
            models=["model/a", "model/b"],
            chairman_model="model/chair",
            search_provider="perplexity",
        )

    assert len(retriever.calls) == 1
    runs = [store.load_run(path.stem) for path in store.runs_dir.glob("*.json")]
    assert {run.mode for run in runs} == {"search", "council"}
    search_run = next(run for run in runs if run.mode == "search")
    failed_run = next(run for run in runs if run.mode == "council")
    assert failed_run.raw["status"] == "failed_before_review"
    assert failed_run.raw["retrieval"] == {
        "run_id": search_run.id,
        "provider": "perplexity",
        "source_ids": search_run.source_ids,
    }
    assert set(search_run.source_ids).issubset(failed_run.source_ids)
    question = store.load_node("root")
    assert {search_run.id, failed_run.id}.issubset(question.run_ids)
    assert inspect_graph(store).healthy
