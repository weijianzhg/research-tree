from __future__ import annotations

import json
import stat

from research_tree.cli import main
from research_tree.store import GraphStore


class _FakeStdin:
    def __init__(self, text: str):
        self._text = text

    def read(self) -> str:
        return self._text

    def isatty(self) -> bool:
        return False


def _fake_editor(tmp_path, body: str):
    script = tmp_path / "fake-editor.sh"
    script.write_text(f"#!/bin/sh\nprintf '%s' {body!r} > \"$1\"\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def test_record_free_form_branches_and_answers(tmp_path, capsys):
    store, _ = GraphStore.create(tmp_path / "graph", "A root question?")
    code = main(
        [
            "--root",
            str(store.root),
            "record",
            "What is a monad?",
            "--text",
            "A monad is a design pattern for sequencing computations.",
        ]
    )
    assert code == 0
    questions = [n for n in store.list_nodes() if n.type == "question"]
    answers = [n for n in store.list_nodes() if n.type == "answer"]
    assert len(questions) == 2
    assert len(answers) == 1
    new_question = next(n for n in questions if n.title == "What is a monad?")
    assert new_question.parent_id == store.load_project().root_question_id
    assert new_question.title == "What is a monad?"
    assert answers[0].question_id == new_question.id
    assert "manual" in answers[0].tags
    assert "design pattern" in answers[0].body
    assert store.load_node(new_question.id).status == "answered"


def test_record_existing_node_reference(tmp_path, capsys):
    store, _ = GraphStore.create(tmp_path / "graph", "A root question?")
    code = main(
        [
            "--root",
            str(store.root),
            "record",
            "root",
            "--text",
            "The root answer.",
        ]
    )
    assert code == 0
    answers = [n for n in store.list_nodes() if n.type == "answer"]
    assert len(answers) == 1
    assert answers[0].question_id == store.load_project().root_question_id
    assert store.load_node(store.load_project().root_question_id).status == "answered"


def test_record_reads_piped_stdin(tmp_path, monkeypatch, capsys):
    store, _ = GraphStore.create(tmp_path / "graph", "A root question?")
    monkeypatch.setattr("sys.stdin", _FakeStdin("Answer from stdin."))
    code = main(["--root", str(store.root), "record", "Piped question?", "--json"])
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["data"]["answer"]["body"].strip() == "Answer from stdin."
    assert output["data"]["question"]["title"] == "Piped question?"


def test_record_empty_answer_keeps_question_open(tmp_path, monkeypatch, capsys):
    store, _ = GraphStore.create(tmp_path / "graph", "A root question?")
    monkeypatch.setattr("sys.stdin", _FakeStdin("   "))
    code = main(["--root", str(store.root), "record", "A question?", "--json"])
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["data"]["answer"] is None
    question = store.load_node(output["data"]["question"]["id"])
    assert question.status == "open"
    assert not [n for n in store.list_nodes() if n.type == "answer"]


def test_record_with_file(tmp_path, capsys):
    store, _ = GraphStore.create(tmp_path / "graph", "A root question?")
    notes = tmp_path / "answer.md"
    notes.write_text("# Notes\n\nAnswer body from a file.")
    code = main(["--root", str(store.root), "record", "root", "--file", str(notes)])
    assert code == 0
    answers = [n for n in store.list_nodes() if n.type == "answer"]
    assert len(answers) == 1
    assert "Answer body from a file." in answers[0].body


def test_answer_edit_uses_editor(tmp_path, monkeypatch, capsys):
    store, _ = GraphStore.create(tmp_path / "graph", "A root question?")
    monkeypatch.setenv("EDITOR", _fake_editor(tmp_path, "Edited answer body."))
    code = main(["--root", str(store.root), "answer", "root", "--edit"])
    assert code == 0
    answers = [n for n in store.list_nodes() if n.type == "answer"]
    assert len(answers) == 1
    assert answers[0].body.strip() == "Edited answer body."


def test_record_edit_uses_editor(tmp_path, monkeypatch, capsys):
    store, _ = GraphStore.create(tmp_path / "graph", "A root question?")
    monkeypatch.setenv("EDITOR", _fake_editor(tmp_path, "Edited via record."))
    code = main(["--root", str(store.root), "record", "An edited question?", "--edit"])
    assert code == 0
    answers = [n for n in store.list_nodes() if n.type == "answer"]
    assert len(answers) == 1
    assert answers[0].body.strip() == "Edited via record."


def test_record_repl_branches_answers_and_quits(tmp_path, monkeypatch, capsys):
    store, _ = GraphStore.create(tmp_path / "graph", "A root question?")
    responses = iter(
        [
            "What is X?",  # branch a question (focuses it)
            "a",  # record an answer
            "X is Y.",  # answer body
            ".",  # terminator
            "quit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
    code = main(["--root", str(store.root), "record"])
    assert code == 0
    questions = [n for n in store.list_nodes() if n.type == "question"]
    answers = [n for n in store.list_nodes() if n.type == "answer"]
    assert len(questions) == 2
    assert len(answers) == 1
    new_question = next(n for n in questions if n.title == "What is X?")
    assert answers[0].question_id == new_question.id
    assert "X is Y." in answers[0].body


def test_record_repl_show_resolves_aliases_with_named_cursor(tmp_path, monkeypatch, capsys):
    store, root = GraphStore.create(tmp_path / "graph", "Root?")
    child = store.add_question("Child?", parent="root", focus=False)
    grandchild = store.add_question("Grandchild?", parent=child.id, focus=False)
    store.set_focus(grandchild.id, cursor="foo")
    assert root.id == store.load_project().root_question_id
    responses = iter(["s ..", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
    code = main(["--root", str(store.root), "--cursor", "foo", "record"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Child?" in out  # `..` from the named cursor resolves to the parent
    assert "Root?" not in out
