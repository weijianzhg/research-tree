from __future__ import annotations

import json

from research_tree.cli import EXIT_NOT_FOUND, EXIT_VALIDATION, main
from research_tree.models import Node, new_id, utc_now
from research_tree.providers import ProviderResponse
from research_tree.research import ask_question
from research_tree.store import GraphStore


def test_cli_json_can_appear_after_subcommand(tmp_path, capsys):
    root = tmp_path / "graph"
    assert main(["init", str(root), "What should we research?", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["data"]["root_question"]["type"] == "question"


def test_where_json_includes_path_children_and_answers(tmp_path, capsys):
    store, root = GraphStore.create(tmp_path / "graph", "A root question?")
    child = store.add_question("A child question?", parent="root", focus=False)
    assert main(["--root", str(store.root), "where", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)["data"]
    assert [node["id"] for node in output["path"]] == [root.id]
    assert [node["id"] for node in output["children"]] == [child.id]
    assert output["answers"] == []


def test_cli_missing_root_has_stable_error_code_and_json(capsys, tmp_path):
    code = main(["--root", str(tmp_path / "missing"), "where", "--json"])
    captured = capsys.readouterr()
    assert code == EXIT_NOT_FOUND
    error = json.loads(captured.err)
    assert error["ok"] is False
    assert error["exit_code"] == EXIT_NOT_FOUND


def test_unhealthy_doctor_json_uses_error_envelope(capsys, tmp_path):
    store, _ = GraphStore.create(tmp_path / "graph", "A root question?")
    created = utc_now()
    store.save_node(
        Node(
            id=new_id("question"),
            type="question",
            title="An orphan",
            status="open",
            created_at=created,
            updated_at=created,
        )
    )
    code = main(["--root", str(store.root), "doctor", "--json"])
    captured = capsys.readouterr()
    assert code == EXIT_VALIDATION
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["ok"] is False
    assert error["data"]["healthy"] is False


def test_promote_blocks_unverified_model_output(tmp_path, capsys):
    class Client:
        def chat(self, **kwargs):
            return ProviderResponse(
                content=json.dumps(
                    {
                        "answer_markdown": "A model answer.",
                        "confidence": 0.8,
                        "claims": [],
                        "uncertainties": [],
                        "follow_up_questions": [],
                    }
                ),
                requested_model=kwargs["model"],
                resolved_model=kwargs["model"],
            )

    store, _ = GraphStore.create(tmp_path / "graph", "A root question?")
    outcome = ask_question(store, "root", client=Client(), model="test/model")
    target = tmp_path / "notes.md"
    code = main(
        ["--root", str(store.root), "promote", outcome.answer.id, "--to", str(target), "--json"]
    )
    assert code == EXIT_VALIDATION
    assert not target.exists()
    error = json.loads(capsys.readouterr().err)
    assert "has not been verified" in error["error"]


def test_promote_rejects_non_answer_node(tmp_path, capsys):
    store, root = GraphStore.create(tmp_path / "graph", "A root question?")
    code = main(
        [
            "--root",
            str(store.root),
            "promote",
            root.id,
            "--to",
            str(tmp_path / "notes.md"),
            "--json",
        ]
    )
    assert code == EXIT_VALIDATION
    error = json.loads(capsys.readouterr().err)
    assert "expects an answer or synthesis node" in error["error"]


def test_next_defaults_to_whole_tree_not_focus(tmp_path, capsys):
    store, root = GraphStore.create(tmp_path / "graph", "A root question?")
    left = store.add_question("Left", parent="root", focus=False)
    store.add_question("Right", parent="root", focus=False)
    store.set_focus(left.id)
    code = main(["--root", str(store.root), "next", "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)["data"]
    ids = {node["id"] for node in data}
    assert root.id in ids
    assert left.id in ids
    assert any(node["title"] == "Right" for node in data)


def test_ask_accepts_free_text_question(tmp_path, capsys, monkeypatch):
    class FakeClient:
        def chat(self, **kwargs):
            return ProviderResponse(
                content=json.dumps(
                    {
                        "answer_markdown": "Bearer tokens alone cannot attribute users.",
                        "confidence": 0.8,
                        "claims": [],
                        "uncertainties": [],
                        "follow_up_questions": [],
                    }
                ),
                requested_model=kwargs["model"],
                resolved_model=kwargs["model"],
            )

    import research_tree.cli as cli

    monkeypatch.setattr(cli, "OpenRouterClient", lambda: FakeClient())

    store, root = GraphStore.create(tmp_path / "graph", "A root question?")
    code = main(["--root", str(store.root), "ask", "How do I secure a call?", "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)["data"]

    questions = [
        node
        for node in store.list_nodes(node_type="question")
        if node.title == "How do I secure a call?"
    ]
    assert len(questions) == 1
    question = questions[0]
    assert question.parent_id == root.id
    assert data["answer"]["type"] == "answer"
    assert data["answer"]["question_id"] == question.id
    assert store.get_focus() == question.id


def test_ask_still_resolves_node_reference(tmp_path, capsys, monkeypatch):
    class FakeClient:
        def chat(self, **kwargs):
            return ProviderResponse(
                content=json.dumps(
                    {
                        "answer_markdown": "A direct answer.",
                        "confidence": 0.7,
                        "claims": [],
                        "uncertainties": [],
                        "follow_up_questions": [],
                    }
                ),
                requested_model=kwargs["model"],
                resolved_model=kwargs["model"],
            )

    import research_tree.cli as cli

    monkeypatch.setattr(cli, "OpenRouterClient", lambda: FakeClient())

    store, root = GraphStore.create(tmp_path / "graph", "A root question?")
    child = store.add_question("A follow-up?", parent="root", focus=False)
    code = main(["--root", str(store.root), "ask", child.id, "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["answer"]["question_id"] == child.id
