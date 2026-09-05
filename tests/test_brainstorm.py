from __future__ import annotations

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from research_tree.brainstorm import prepare_brainstorm, run_brainstorm, validate_idea
from research_tree.cli import EXIT_PROVIDER, EXIT_VALIDATION, main
from research_tree.doctor import inspect_graph
from research_tree.errors import ModelOutputError, ProviderError, ValidationError
from research_tree.providers import ProviderResponse
from research_tree.render import frontier
from research_tree.research import record_manual_answer


def idea_payload(index=1, *, method="ssot", question=None):
    return {
        **({"random_string": f"a8!Kz4#pY2$vR9&mX6@bT3{index:03d}"} if method == "ssot" else {}),
        "question": question or f"Can intervention {index} improve expert routing?",
        "angle": f"Intervention {index}: causal analysis of routing at inference time.",
        "rationale": "Separates routing effects from model capacity.",
        "first_step": "Fix the weights and compare the intervention to the original router.",
        "assumptions": ["The router can be changed independently of the weights."],
    }


class FakeClient:
    def __init__(self, outputs=None):
        self.requests = []
        self.outputs = iter(outputs) if outputs is not None else None

    def chat(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        value = next(self.outputs) if self.outputs is not None else idea_payload(len(self.requests))
        if isinstance(value, Exception):
            raise value
        content = value if isinstance(value, str) else json.dumps(value)
        return ProviderResponse(
            content=content,
            requested_model=kwargs["model"],
            resolved_model="resolved/model",
            usage={"cost": 0.01, "completion_tokens": 80},
            raw={"choices": [{"message": {"content": content}}]},
        )


def canonical_files(store):
    return {
        str(path.relative_to(store.root)): path.read_bytes()
        for path in store.root.rglob("*")
        if path.is_file() and ".state" not in path.parts
    }


def test_independent_ssot_calls_create_proposals_with_provenance(store):
    question = store.load_node("root")
    question.status = "contested"
    store.update_node(question)
    focused = store.add_question("Existing branch?", parent="root")
    plan = prepare_brainstorm(store, "root", model="test/model", ideas=3)
    client = FakeClient()
    outcome = run_brainstorm(store, plan, client=client)

    assert len(outcome.ideas) == 3
    assert client.requests == [plan.request] * 3
    assert client.requests[0]["web"] is False
    assert client.requests[0]["temperature"] == 0.6
    assert client.requests[0]["reasoning_effort"] == "high"
    assert "Existing branch?" in plan.request["messages"][1]["content"]
    for node in outcome.ideas:
        assert node.type == "question"
        assert node.parent_id == question.id
        assert node.status == "proposed"
        assert node.confidence is None
        assert node.source_ids == []
        assert node.run_ids == [outcome.run.id]
        assert "First step" in node.body
        assert "Assumptions to test" in node.body
    assert store.load_node("root").status == "contested"
    assert store.get_focus() == focused.id
    assert store.load_node("root").run_ids == [outcome.run.id]
    run = store.load_run(outcome.run.id)
    assert run.mode == "brainstorm"
    assert run.raw["method"] == "ssot"
    assert run.raw["status"] == "completed"
    assert run.requested_models == ["test/model"] * 3
    assert run.resolved_models == ["resolved/model"] * 3
    assert run.usage["total_cost"] == pytest.approx(0.03)
    assert run.response_node_ids == [node.id for node in outcome.ideas]
    assert [sample["parsed"]["random_string"] for sample in run.raw["samples"]] == [
        idea_payload(i)["random_string"] for i in range(1, 4)
    ]
    assert {node.id for node in outcome.ideas} <= {
        node.id for node in frontier(store, start="root")
    }
    assert inspect_graph(store).healthy


def test_direct_method_has_no_seed_contract_and_keeps_same_sampling_controls(store):
    client = FakeClient([idea_payload(method="direct")])
    plan = prepare_brainstorm(store, method="direct", ideas=1, temperature=0.7)
    outcome = run_brainstorm(store, plan, client=client)
    assert "random_string" not in json.dumps(plan.request)
    assert "String Seed of Thought" not in json.dumps(plan.request)
    assert plan.request["temperature"] == 0.7
    assert outcome.run.raw["method"] == "direct"
    assert "direct" in outcome.ideas[0].tags


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("question", "  "),
        ("question", []),
        ("angle", None),
        ("rationale", 10),
        ("first_step", False),
        ("assumptions", "assume"),
        ("assumptions", [4]),
        ("assumptions", [""]),
        ("random_string", 17),
        ("random_string", "short"),
        ("random_string", "a" * 129),
        ("random_string", "a" * 24 + "\n"),
        ("random_string", "\u4e2d" * 24),
    ],
)
def test_invalid_idea_fields_are_rejected(field, bad_value):
    value = idea_payload()
    value[field] = bad_value
    with pytest.raises(ModelOutputError):
        validate_idea(value, "ssot")


def test_ssot_requires_seed_before_idea_and_exact_fields():
    value = idea_payload()
    value["random_string"] = value.pop("random_string")
    with pytest.raises(ModelOutputError, match="before"):
        validate_idea(value, "ssot")
    for value in ({**idea_payload(), "extra": 1}, idea_payload(method="direct")):
        with pytest.raises(ModelOutputError, match="fields"):
            validate_idea(value, "ssot")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ideas": 0},
        {"ideas": 21},
        {"ideas": True},
        {"ideas": 1.5},
        {"method": "unknown"},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": -1},
        {"temperature": 3},
        {"temperature": True},
        {"cursor": "../bad"},
    ],
)
def test_invalid_plan_options_do_not_change_graph(store, kwargs):
    before = canonical_files(store)
    with pytest.raises(ValidationError):
        prepare_brainstorm(store, "A new topic", **kwargs)
    assert canonical_files(store) == before


def test_duplicate_titles_are_skipped_but_every_candidate_is_recorded(store):
    existing = store.add_question("Can an intervention improve routing?", parent="root")
    client = FakeClient(
        [
            idea_payload(1, question="  CAN  AN intervention improve routing! "),
            idea_payload(2),
            idea_payload(3, question=idea_payload(2)["question"].upper()),
            idea_payload(4, question=store.load_node("root").title),
        ]
    )
    outcome = run_brainstorm(store, prepare_brainstorm(store, "root", ideas=4), client=client)
    assert len(outcome.ideas) == 1
    assert len(client.requests) == 4  # no paid retries to replace duplicates
    assert outcome.run.raw["duplicates_skipped"] == 3
    samples = outcome.run.raw["samples"]
    assert samples[0]["node_id"] == existing.id
    assert samples[1]["node_id"] == samples[2]["node_id"] == outcome.ideas[0].id
    assert samples[3]["node_id"] == outcome.question.id
    assert all("parsed" in sample for sample in samples)
    assert inspect_graph(store).healthy


def test_invalid_output_retries_once_and_preserves_both_paid_attempts(store):
    client = FakeClient(["not JSON", idea_payload()])
    outcome = run_brainstorm(store, prepare_brainstorm(store, ideas=1), client=client)
    assert len(client.requests) == 2
    assert client.requests[0] == client.requests[1]
    assert len(outcome.ideas) == 1
    attempts = outcome.run.raw["samples"][0]["attempts"]
    assert attempts[0]["content"] == "not JSON"
    assert "error" in attempts[0]
    assert outcome.run.usage["total_cost"] == pytest.approx(0.02)


@pytest.mark.parametrize(
    "outputs,error,status,node_count,paid_calls",
    [
        (["bad", "bad"], ModelOutputError, "failed_validation", 0, 2),
        ([idea_payload(), "bad", "bad"], ModelOutputError, "partial", 1, 3),
        ([idea_payload(), ProviderError("network failure")], ProviderError, "partial", 1, 1),
        ([ProviderError("unavailable")], ProviderError, "failed_provider", 0, 0),
    ],
)
def test_failed_batch_stops_and_preserves_completed_work(
    store, outputs, error, status, node_count, paid_calls
):
    client = FakeClient(outputs)
    with pytest.raises(error, match="saved.*run r_"):
        run_brainstorm(store, prepare_brainstorm(store, ideas=5), client=client)
    assert len(client.requests) == len(outputs)
    run = store.load_run(next(store.runs_dir.glob("*.json")).stem)
    assert run.raw["status"] == status
    assert len(run.response_node_ids) == node_count
    assert len(run.usage["calls"]) == paid_calls
    if paid_calls:
        assert run.usage["total_cost"] == pytest.approx(0.01 * paid_calls)
    else:
        assert "total_cost" not in run.usage
    assert store.load_node("root").status == "open"
    assert inspect_graph(store).healthy


def test_transaction_failure_rolls_back_new_topic_and_ideas(store, monkeypatch):
    before = canonical_files(store)
    focus = store.get_focus()
    plan = prepare_brainstorm(store, "A new topic", ideas=2)
    save_run = store.save_run

    def fail(run):
        raise OSError("disk failure")

    monkeypatch.setattr(store, "save_run", fail)
    with pytest.raises(OSError, match="disk failure"):
        run_brainstorm(store, plan, client=FakeClient())
    assert canonical_files(store) == before
    assert store.get_focus() == focus
    assert inspect_graph(store).healthy
    # Retrying after a rollback must not link the new topic to an uncommitted run.
    assert plan.question.run_ids == []
    monkeypatch.setattr(store, "save_run", save_run)
    outcome = run_brainstorm(store, plan, client=FakeClient())
    assert outcome.question.run_ids == [outcome.run.id]
    assert inspect_graph(store).healthy
    client = FakeClient()
    with pytest.raises(ValidationError, match="already created"):
        run_brainstorm(store, plan, client=client)
    assert client.requests == []


def test_concurrent_brainstorms_merge_provenance_and_deduplicate_under_lock(store):
    barrier = threading.Barrier(2)

    class ConcurrentClient(FakeClient):
        def chat(self, **kwargs):
            barrier.wait(timeout=10)
            return super().chat(**kwargs)

    plans = [prepare_brainstorm(store, ideas=1) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(lambda p: run_brainstorm(store, p, client=ConcurrentClient()), plans)
        )
    assert sum(len(outcome.ideas) for outcome in outcomes) == 1
    assert set(store.load_node("root").run_ids) == {o.run.id for o in outcomes}
    assert inspect_graph(store).healthy


def test_context_snapshot_does_not_overwrite_later_question_changes(store):
    plan = prepare_brainstorm(store, ideas=1)
    question = store.load_node("root")
    question.status = "answered"
    question.body = "New evidence arrived while brainstorming."
    store.update_node(question)
    run_brainstorm(store, plan, client=FakeClient())
    current = store.load_node("root")
    assert current.status == "answered"
    assert current.body.strip() == question.body


def test_cli_dry_run_new_topic_is_offline_and_leaves_cursors_unchanged(store, monkeypatch, capsys):
    def forbidden():
        raise AssertionError("dry run must not construct a client")

    monkeypatch.setattr("research_tree.cli.OpenRouterClient", forbidden)
    before = canonical_files(store)
    focus = store.get_focus()
    code = main(
        [
            "--root",
            str(store.root),
            "--cursor",
            "agent",
            "brainstorm",
            "New topic",
            "--ideas",
            "2",
            "--model",
            "test/model",
            "--effort",
            "low",
            "--dry-run",
            "--json",
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["creates_question"] is True
    assert data["question"]["title"] == "New topic"
    assert data["planned_calls"] == 2
    assert data["max_calls"] == 4
    assert data["request"]["model"] == "test/model"
    assert data["request"]["reasoning_effort"] == "low"
    assert canonical_files(store) == before
    assert store.get_focus() == focus
    assert not store.cursor_path("agent").exists()


def test_cli_new_topic_branches_from_named_cursor_and_supports_followup_research(
    store, monkeypatch, capsys
):
    parent = store.add_question("Agent topic", parent="root", focus=False)
    store.set_focus(parent.id, cursor="agent")
    default_focus = store.get_focus()
    client = FakeClient()
    monkeypatch.setattr("research_tree.cli.OpenRouterClient", lambda: client)
    assert (
        main(
            [
                "--root",
                str(store.root),
                "--cursor",
                "agent",
                "brainstorm",
                "New topic",
                "--ideas",
                "1",
                "--json",
            ]
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)["data"]
    topic = data["question"]
    assert topic["title"] == "New topic"
    assert topic["parent_id"] == parent.id
    assert store.get_focus("agent") == topic["id"]
    assert store.get_focus() == default_focus
    idea = data["ideas"][0]
    assert idea["parent_id"] == topic["id"]
    answer = record_manual_answer(store, idea["id"], "Follow-up investigation results.")
    assert answer.question_id == idea["id"]
    assert inspect_graph(store).healthy


@pytest.mark.parametrize(
    "outputs,exit_code",
    [(["bad", "bad"], EXIT_VALIDATION), ([ProviderError("offline")], EXIT_PROVIDER)],
)
def test_cli_failure_envelope_has_stable_exit_code_and_saved_run(
    store, monkeypatch, capsys, outputs, exit_code
):
    monkeypatch.setattr("research_tree.cli.OpenRouterClient", lambda: FakeClient(outputs))
    assert main(["--root", str(store.root), "brainstorm", "--ideas", "1", "--json"]) == exit_code
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["exit_code"] == exit_code
    assert "run r_" in error["error"]
    assert inspect_graph(store).healthy


def test_cli_invalid_options_fail_before_graph_creation_or_client_use(store, monkeypatch, capsys):
    def forbidden():
        raise AssertionError("invalid options must not construct a client")

    monkeypatch.setattr("research_tree.cli.OpenRouterClient", forbidden)
    before = canonical_files(store)
    assert (
        main(["--root", str(store.root), "brainstorm", "New topic", "--ideas", "0", "--json"])
        == EXIT_VALIDATION
    )
    assert json.loads(capsys.readouterr().err)["exit_code"] == EXIT_VALIDATION
    assert canonical_files(store) == before


def test_cli_human_output_shows_usable_ideas_and_next_action(store, monkeypatch, capsys):
    monkeypatch.setattr("research_tree.cli.OpenRouterClient", FakeClient)
    assert main(["--root", str(store.root), "brainstorm", "--ideas", "1"]) == 0
    captured = capsys.readouterr()
    assert "proposed branches" in captured.out
    assert "Angle:" in captured.out
    assert "First step:" in captured.out
    assert "Next: research-tree" in captured.out
    assert "one call each" in captured.err
