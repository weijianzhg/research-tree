from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from research_tree.doctor import inspect_graph
from research_tree.errors import ProviderError
from research_tree.providers import ProviderResponse
from research_tree.research import (
    ANSWER_SCHEMA,
    COUNCIL_SCHEMA,
    REVIEW_SCHEMA,
    VERIFY_SCHEMA,
    ask_question,
    parse_json_content,
    record_manual_answer,
    run_council,
    validate_answer,
    validate_verification,
    verify_claims,
)


def answer_payload(label="single", confidence=0.82):
    return {
        "answer_markdown": (
            f"{label}: Sparse MoE activates only a subset of parameters per token "
            "while retaining a larger total capacity [paper](https://example.com/paper)."
        ),
        "confidence": confidence,
        "claims": [
            {
                "text": "Only selected experts run for each token.",
                "confidence": 0.9,
                "source_urls": ["https://example.com/paper"],
            }
        ],
        "uncertainties": ["Hardware communication can erase theoretical gains."],
        "follow_up_questions": [
            {
                "question": "When does communication dominate MoE inference?",
                "rationale": "It tests whether sparse FLOPs translate into wall-clock gains.",
                "priority": 1,
            }
        ],
    }


def provider_response(model, payload):
    return ProviderResponse(
        content=json.dumps(payload),
        requested_model=model,
        resolved_model=model + "-resolved",
        annotations=[
            {
                "type": "url_citation",
                "url_citation": {
                    "url": "https://example.com/paper",
                    "title": "Primary MoE paper",
                    "content": "A sparse gate selects a subset of experts for each token.",
                },
            }
        ],
        usage={"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01},
        raw={"model": model + "-resolved", "content": payload},
    )


class FakeSingleClient:
    def chat(self, **kwargs):
        return provider_response(kwargs["model"], answer_payload())


class TwoClaimClient:
    def chat(self, **kwargs):
        payload = answer_payload()
        payload["claims"].append(
            {
                "text": "Communication overhead can dominate sparse inference.",
                "confidence": 0.75,
                "source_urls": ["https://example.com/paper"],
            }
        )
        return provider_response(kwargs["model"], payload)


class MultipleSnapshotsClient(FakeSingleClient):
    def chat(self, **kwargs):
        response = super().chat(**kwargs)
        response.annotations.append(
            {
                "type": "url_citation",
                "url_citation": {
                    "url": "https://example.com/paper",
                    "title": "Primary MoE paper, alternate excerpt",
                    "content": "A second provider excerpt for the same URL.",
                },
            }
        )
        return response


class FakeCouncilClient:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = []

    def chat(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
        model = kwargs["model"]
        schema = kwargs.get("response_schema")
        if schema == REVIEW_SCHEMA:
            payload = {
                "strongest_response": "Response A",
                "evidence_strengths": ["Response A cites a primary paper."],
                "unsupported_or_weak_claims": [],
                "disagreements": ["The answers weight communication costs differently."],
                "missing_questions": ["What hardware topology was assumed?"],
            }
        elif schema == COUNCIL_SCHEMA:
            payload = {
                **answer_payload("synthesis", confidence=0.88),
                "consensus": ["Sparse activation reduces per-token arithmetic."],
                "disagreements": ["End-to-end speed depends on communication."],
            }
        else:
            assert schema == ANSWER_SCHEMA
            payload = answer_payload(model)
        return provider_response(model, payload)


class FakeVerifyClient:
    def chat(self, **kwargs):
        assert kwargs["response_schema"] == VERIFY_SCHEMA
        prompt = kwargs["messages"][-1]["content"]
        claim_ids = sorted(set(re.findall(r"c_[a-f0-9]{12}", prompt)))
        source_ids = sorted(set(re.findall(r"s_[a-f0-9]{12}", prompt)))
        payload = {
            "verdicts": [
                {
                    "claim_id": claim_id,
                    "verdict": "supported",
                    "confidence": 0.84,
                    "explanation": "The captured excerpt directly states the routing behavior.",
                    "supporting_source_ids": source_ids,
                    "evidence_quality": "primary",
                    "quality_explanation": "The excerpt is from the original technical paper.",
                    "missing_evidence": [],
                }
                for claim_id in claim_ids
            ],
            "overall_assessment": "The stored source supports the scoped claim.",
            "follow_up_questions": [],
        }
        return provider_response(kwargs["model"], payload)


class ConcurrentVerifyClient(FakeVerifyClient):
    def __init__(self, barrier):
        self.barrier = barrier

    def chat(self, **kwargs):
        self.barrier.wait(timeout=5)
        return super().chat(**kwargs)


class FixedVerdictClient(FakeVerifyClient):
    def __init__(self, verdict):
        self.verdict = verdict

    def chat(self, **kwargs):
        response = super().chat(**kwargs)
        payload = json.loads(response.content)
        for item in payload["verdicts"]:
            item["verdict"] = self.verdict
            item["explanation"] = f"Fixed {self.verdict} verdict for aggregation testing."
        response.content = json.dumps(payload)
        response.raw["content"] = payload
        return response


class ConcurrentClient:
    def __init__(self, barrier):
        self.barrier = barrier

    def chat(self, **kwargs):
        self.barrier.wait(timeout=5)
        model = kwargs["model"]
        url = f"https://example.com/{model}"
        payload = answer_payload(model)
        payload["claims"][0]["source_urls"] = [url]
        payload["follow_up_questions"] = []
        response = provider_response(model, payload)
        response.annotations[0]["url_citation"]["url"] = url
        return response


class MalformedCouncilClient(FakeCouncilClient):
    def chat(self, **kwargs):
        if kwargs["model"] == "model/b" and kwargs.get("response_schema") == ANSWER_SCHEMA:
            with self.lock:
                self.calls.append(kwargs)
            return provider_response(kwargs["model"], {})
        return super().chat(**kwargs)


class MalformedSingleClient:
    def chat(self, **kwargs):
        value = answer_payload()
        value["follow_up_questions"] = [None]
        return provider_response(kwargs["model"], value)


class DistinctCouncilSourcesClient(FakeCouncilClient):
    def chat(self, **kwargs):
        response = super().chat(**kwargs)
        if kwargs.get("response_schema") == ANSWER_SCHEMA:
            response.annotations[0]["url_citation"]["content"] = (
                f"Evidence captured specifically by {kwargs['model']}."
            )
        return response


class StageOneFailureClient(FakeCouncilClient):
    def chat(self, **kwargs):
        if kwargs["model"] == "model/b" and kwargs.get("response_schema") == ANSWER_SCHEMA:
            raise ProviderError("member unavailable")
        return super().chat(**kwargs)


class ChairmanFailureClient(FakeCouncilClient):
    def chat(self, **kwargs):
        if kwargs.get("response_schema") == COUNCIL_SCHEMA:
            raise ProviderError("chairman unavailable")
        return super().chat(**kwargs)


class UrlOnlyChairmanFailureClient(ChairmanFailureClient):
    def chat(self, **kwargs):
        response = super().chat(**kwargs)
        if kwargs.get("response_schema") == ANSWER_SCHEMA:
            response.annotations = []
        return response


def test_parse_json_content_accepts_fenced_json():
    value = parse_json_content('```json\n{"answer": 1}\n```')
    assert value == {"answer": 1}


@pytest.mark.parametrize("target", ["answer", "claim"])
def test_answer_rejects_boolean_confidence(target):
    value = answer_payload()
    if target == "answer":
        value["confidence"] = True
        message = "model confidence is not numeric"
    else:
        value["claims"][0]["confidence"] = True
        message = "model claim confidence is not numeric"
    with pytest.raises(ProviderError, match=message):
        validate_answer(value)


def test_ask_persists_answer_claim_source_run_and_followup(store):
    outcome = ask_question(
        store,
        "root",
        client=FakeSingleClient(),
        model="test/model",
        followups=3,
    )
    assert outcome.answer.type == "answer"
    assert len(outcome.claims) == 1
    assert len(outcome.followups) == 1
    assert len(outcome.sources) == 1
    assert outcome.run.resolved_models == ["test/model-resolved"]
    assert outcome.run.usage["total_cost"] == 0.01
    assert store.load_node("root").status == "answered"
    assert inspect_graph(store).healthy


def test_openrouter_keeps_distinct_snapshots_for_the_same_url(store):
    outcome = ask_question(
        store,
        "root",
        client=MultipleSnapshotsClient(),
        model="test/model",
    )
    snapshots = [source for source in outcome.sources if source.url == "https://example.com/paper"]
    assert len(snapshots) == 2
    assert len(outcome.run.source_ids) == 2
    assert set(outcome.answer.source_ids) == {source.id for source in snapshots}


def test_duplicate_followup_is_not_added_twice(store):
    ask_question(store, "root", client=FakeSingleClient(), model="test/model")
    second = ask_question(store, "root", client=FakeSingleClient(), model="test/model")
    assert second.followups == []
    assert store.load_node("root").status == "answered"


def test_malformed_nested_answer_preserves_paid_attempt_without_semantic_nodes(store):
    with pytest.raises(ProviderError, match="preserved attempt as.*follow-up"):
        ask_question(store, "root", client=MalformedSingleClient(), model="model/malformed")
    assert len(store.list_nodes()) == 1
    runs = [store.load_run(path.stem) for path in store.runs_dir.glob("*.json")]
    assert len(runs) == 1
    assert runs[0].raw["status"] == "failed_validation"
    assert runs[0].usage["total_cost"] == pytest.approx(0.01)
    assert store.load_node("root").run_ids == [runs[0].id]
    assert inspect_graph(store).healthy


def test_io_failure_rolls_back_the_whole_research_outcome(store, monkeypatch):
    original_save = store.save_node
    calls = 0

    def fail_after_one_node(node):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated disk failure")
        return original_save(node)

    monkeypatch.setattr(store, "save_node", fail_after_one_node)
    with pytest.raises(OSError, match="simulated disk failure"):
        ask_question(store, "root", client=FakeSingleClient(), model="test/model")
    monkeypatch.setattr(store, "save_node", original_save)
    assert [node.type for node in store.list_nodes()] == ["question"]
    assert store.load_node("root").run_ids == []
    assert list(store.sources_dir.glob("*.json")) == []
    assert list(store.runs_dir.glob("*.json")) == []
    assert inspect_graph(store).healthy


def test_concurrent_answers_merge_question_provenance(store):
    barrier = threading.Barrier(2)

    def research(model):
        return ask_question(
            store,
            "root",
            client=ConcurrentClient(barrier),
            model=model,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(research, ["model/a", "model/b"]))
    question = store.load_node("root")
    assert len(question.run_ids) == 2
    assert len(question.source_ids) == 2
    assert {outcome.run.id for outcome in outcomes} == set(question.run_ids)
    assert inspect_graph(store).healthy


def test_manual_answer_updates_question_without_model_run(store):
    answer = record_manual_answer(store, "root", "An answer based on my own notes.")
    assert answer.tags == ["manual"]
    assert store.load_node("root").status == "answered"
    assert list(store.runs_dir.glob("*.json")) == []


def test_council_persists_individual_views_reviews_and_synthesis(store):
    client = FakeCouncilClient()
    outcome = run_council(
        store,
        "root",
        client=client,
        models=["model/a", "model/b"],
        chairman_model="model/chair",
        followups=2,
    )
    assert outcome.answer.type == "synthesis"
    assert len(outcome.perspectives) == 2
    assert {node.tags[0] for node in outcome.perspectives} == {"council-perspective"}
    assert outcome.run.mode == "council"
    assert len(client.calls) == 5  # 2 answers + 2 blind reviews + chairman
    assert "label_mapping" in outcome.run.raw
    assert inspect_graph(store).healthy


def test_council_continues_when_one_successful_response_is_malformed(store):
    client = MalformedCouncilClient()
    outcome = run_council(
        store,
        "root",
        client=client,
        models=["model/a", "model/b", "model/c"],
        chairman_model="model/chair",
    )
    assert len(outcome.perspectives) == 2
    assert "member-2" in outcome.run.raw["stage1_errors"]
    assert outcome.run.raw["attempts"]["stage1"]["member-2"] == {
        "requested_model": "model/b",
        "status": "invalid",
        "error": "model response is missing fields: answer_markdown, confidence, claims, "
        "uncertainties, follow_up_questions",
    }
    assert len(outcome.run.requested_models) == len(outcome.run.resolved_models)
    assert outcome.run.usage["total_cost"] == pytest.approx(0.06)


def test_council_perspectives_keep_their_own_source_snapshots(store):
    outcome = run_council(
        store,
        "root",
        client=DistinctCouncilSourcesClient(),
        models=["model/a", "model/b"],
        chairman_model="model/chair",
    )
    excerpts = [
        {store.load_source(source_id).excerpt for source_id in perspective.source_ids}
        for perspective in outcome.perspectives
    ]
    assert excerpts == [
        {"Evidence captured specifically by model/a."},
        {"Evidence captured specifically by model/b."},
    ]


def test_failed_stage_one_council_preserves_paid_calls_and_errors(store):
    with pytest.raises(ProviderError, match="preserved attempt as"):
        run_council(
            store,
            "root",
            client=StageOneFailureClient(),
            models=["model/a", "model/b"],
            chairman_model="model/chair",
        )
    runs = [store.load_run(path.stem) for path in store.runs_dir.glob("*.json")]
    assert len(runs) == 1
    assert runs[0].raw["status"] == "failed_before_review"
    assert runs[0].raw["attempts"]["stage1"]["member-2"]["requested_model"] == "model/b"
    assert runs[0].usage["total_cost"] == 0.01
    assert runs[0].id in store.load_node("root").run_ids


def test_failed_chairman_preserves_stage_one_and_review_calls(store):
    with pytest.raises(ProviderError, match="preserved attempt as"):
        run_council(
            store,
            "root",
            client=ChairmanFailureClient(),
            models=["model/a", "model/b"],
            chairman_model="model/chair",
        )
    runs = [store.load_run(path.stem) for path in store.runs_dir.glob("*.json")]
    assert len(runs) == 1
    assert runs[0].raw["status"] == "failed_chairman"
    assert runs[0].raw["attempts"]["chairman"]["error"] == "chairman unavailable"
    assert len(runs[0].usage["calls"]) == 4
    assert runs[0].usage["total_cost"] == pytest.approx(0.04)
    assert len(runs[0].requested_models) == len(runs[0].resolved_models) == 4


def test_failed_council_preserves_claim_urls_without_provider_annotations(store):
    with pytest.raises(ProviderError, match="preserved attempt as"):
        run_council(
            store,
            "root",
            client=UrlOnlyChairmanFailureClient(),
            models=["model/a", "model/b"],
            chairman_model="model/chair",
        )
    run = next(store.load_run(path.stem) for path in store.runs_dir.glob("*.json"))
    assert run.source_ids
    assert all(
        store.load_source(source_id).url == "https://example.com/paper"
        for source_id in run.source_ids
    )


def test_verify_checks_frozen_sources_and_updates_claim_status(store):
    research = ask_question(store, "root", client=FakeSingleClient(), model="test/author")
    outcome = verify_claims(
        store,
        research.answer.id,
        client=FakeVerifyClient(),
        model="test/verifier",
    )
    assert outcome.note.type == "note"
    assert outcome.note.status == "answered"
    assert outcome.run.mode == "verify"
    assert outcome.run.resolved_models == ["test/verifier-resolved"]
    verified_claim = store.load_node(research.claims[0].id)
    assert "verdict-supported" in verified_claim.tags
    assert "unverified" not in verified_claim.tags
    verified_answer = store.load_node(research.answer.id)
    assert "verified" in verified_answer.tags
    assert inspect_graph(store).healthy


def test_individual_claim_verification_aggregates_all_sibling_state(store):
    research = ask_question(store, "root", client=TwoClaimClient(), model="test/author")
    first, second = research.claims
    verify_claims(
        store,
        first.id,
        client=FixedVerdictClient("contradicted"),
        model="test/contradictor",
    )
    assert store.load_node("root").status == "contested"
    assert "unverified" in store.load_node(research.answer.id).tags

    verify_claims(
        store,
        second.id,
        client=FixedVerdictClient("supported"),
        model="test/supporter",
    )
    assert store.load_node("root").status == "contested"
    answer = store.load_node(research.answer.id)
    assert answer.status == "contested"
    assert "verified" in answer.tags
    assert inspect_graph(store).healthy


def test_new_answer_does_not_erase_a_contested_question(store):
    research = ask_question(store, "root", client=FakeSingleClient(), model="test/first")
    verify_claims(
        store,
        research.claims[0].id,
        client=FixedVerdictClient("contradicted"),
        model="test/contradictor",
    )
    ask_question(store, "root", client=FakeSingleClient(), model="test/second")
    assert store.load_node("root").status == "contested"


def test_new_answer_does_not_erase_historical_verification_uncertainty(store):
    research = ask_question(store, "root", client=FakeSingleClient(), model="test/first")
    verify_claims(
        store,
        research.claims[0].id,
        client=FixedVerdictClient("partially_supported"),
        model="test/partial",
    )
    ask_question(store, "root", client=FakeSingleClient(), model="test/second")
    assert store.load_node("root").status == "uncertain"


def test_conflicting_reverification_preserves_contested_state(store):
    research = ask_question(store, "root", client=FakeSingleClient(), model="test/author")
    claim = research.claims[0]
    verify_claims(
        store,
        claim.id,
        client=FixedVerdictClient("contradicted"),
        model="test/contradictor",
    )
    verify_claims(
        store,
        claim.id,
        client=FixedVerdictClient("supported"),
        model="test/supporter",
    )
    persisted = store.load_node(claim.id)
    assert {"verdict-contradicted", "verdict-supported"}.issubset(persisted.tags)
    assert persisted.status == "contested"
    assert store.load_node(research.answer.id).status == "contested"
    assert store.load_node("root").status == "contested"


def test_concurrent_verifications_merge_claim_provenance(store):
    research = ask_question(store, "root", client=FakeSingleClient(), model="test/author")
    barrier = threading.Barrier(2)

    def verify(model):
        return verify_claims(
            store,
            research.answer.id,
            client=ConcurrentVerifyClient(barrier),
            model=model,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(verify, ["test/verifier-a", "test/verifier-b"]))
    claim = store.load_node(research.claims[0].id)
    assert {outcome.run.id for outcome in outcomes}.issubset(set(claim.run_ids))
    assert inspect_graph(store).healthy


def test_io_failure_rolls_back_verification(store, monkeypatch):
    research = ask_question(store, "root", client=FakeSingleClient(), model="test/author")
    claim_before = store.node_path(research.claims[0].id).read_text(encoding="utf-8")
    original_save_run = store.save_run

    def fail_save_run(run):
        raise OSError("simulated verification disk failure")

    monkeypatch.setattr(store, "save_run", fail_save_run)
    with pytest.raises(OSError, match="simulated verification disk failure"):
        verify_claims(
            store,
            research.answer.id,
            client=FakeVerifyClient(),
            model="test/verifier",
        )
    monkeypatch.setattr(store, "save_run", original_save_run)
    assert store.node_path(research.claims[0].id).read_text(encoding="utf-8") == claim_before
    assert len(list(store.runs_dir.glob("*.json"))) == 1
    assert len([node for node in store.list_nodes() if node.type == "note"]) == 0
    assert inspect_graph(store).healthy


def test_verifier_cannot_use_another_claims_source_and_downgrades_missing_evidence():
    claim_id = "c_111111111111"
    source_id = "s_222222222222"
    value = {
        "verdicts": [
            {
                "claim_id": claim_id,
                "verdict": "supported",
                "confidence": 0.8,
                "explanation": "Mostly supported.",
                "supporting_source_ids": [source_id],
                "evidence_quality": "primary",
                "quality_explanation": "An original technical artifact.",
                "missing_evidence": ["A direct measurement is missing."],
            }
        ],
        "overall_assessment": "Partial evidence.",
        "follow_up_questions": [],
    }
    validated = validate_verification(value, {claim_id: {source_id}})
    assert validated["verdicts"][0]["verdict"] == "partially_supported"

    value["verdicts"][0]["supporting_source_ids"] = ["s_333333333333"]
    with pytest.raises(ProviderError, match="not attached"):
        validate_verification(value, {claim_id: {source_id}})


@pytest.mark.parametrize("field", ["verdict", "evidence_quality"])
def test_verification_rejects_non_text_enums_as_provider_errors(field):
    claim_id = "c_111111111111"
    value = {
        "verdicts": [
            {
                "claim_id": claim_id,
                "verdict": "supported",
                "confidence": 0.8,
                "explanation": "Supported.",
                "supporting_source_ids": [],
                "evidence_quality": "unknown",
                "quality_explanation": "No captured evidence.",
                "missing_evidence": [],
            }
        ],
        "overall_assessment": "Assessment.",
        "follow_up_questions": [],
    }
    value["verdicts"][0][field] = []
    with pytest.raises(ProviderError, match="unknown verification"):
        validate_verification(value, {claim_id: set()})


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("confidence", True, "confidence is not numeric"),
        ("missing_evidence", [{}], "missing_evidence must contain only strings"),
    ],
)
def test_verification_rejects_malformed_scalar_and_list_values(field, bad_value, message):
    claim_id = "c_111111111111"
    value = {
        "verdicts": [
            {
                "claim_id": claim_id,
                "verdict": "unknown",
                "confidence": 0.5,
                "explanation": "Unknown.",
                "supporting_source_ids": [],
                "evidence_quality": "unknown",
                "quality_explanation": "No evidence.",
                "missing_evidence": [],
            }
        ],
        "overall_assessment": "Assessment.",
        "follow_up_questions": [],
    }
    value["verdicts"][0][field] = bad_value
    with pytest.raises(ProviderError, match=message):
        validate_verification(value, {claim_id: set()})


def test_source_less_claim_cannot_be_verified_as_supported():
    claim_id = "c_111111111111"
    value = {
        "verdicts": [
            {
                "claim_id": claim_id,
                "verdict": "supported",
                "confidence": 0.9,
                "explanation": "It sounds plausible.",
                "supporting_source_ids": [],
                "evidence_quality": "unknown",
                "quality_explanation": "There is no captured source.",
                "missing_evidence": [],
            }
        ],
        "overall_assessment": "No captured evidence.",
        "follow_up_questions": [],
    }
    validated = validate_verification(value, {claim_id: set()})
    assert validated["verdicts"][0]["verdict"] == "unsupported"
    assert validated["verdicts"][0]["evidence_quality"] == "unknown"


def test_secondary_entailment_is_not_treated_as_primary_verification():
    claim_id = "c_111111111111"
    source_id = "s_222222222222"
    value = {
        "verdicts": [
            {
                "claim_id": claim_id,
                "verdict": "supported",
                "confidence": 0.9,
                "explanation": "The roundup repeats the number.",
                "supporting_source_ids": [source_id],
                "evidence_quality": "secondary",
                "quality_explanation": "This is an unaffiliated roundup, not raw benchmark data.",
                "missing_evidence": [],
            }
        ],
        "overall_assessment": "Entailed only by secondary evidence.",
        "follow_up_questions": [],
    }
    validated = validate_verification(value, {claim_id: {source_id}})
    assert validated["verdicts"][0]["verdict"] == "partially_supported"
