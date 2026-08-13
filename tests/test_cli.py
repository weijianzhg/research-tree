from __future__ import annotations

import json

import pytest

import research_tree.cli as cli_module
from research_tree.cli import EXIT_NOT_FOUND, EXIT_VALIDATION, main
from research_tree.models import Node, new_id, utc_now
from research_tree.providers import ProviderResponse, SearchResponse, SearchResult
from research_tree.research import ask_question
from research_tree.store import GraphStore


def _cli_answer(model):
    return ProviderResponse(
        content=json.dumps(
            {
                "answer_markdown": "A current answer.",
                "confidence": 0.8,
                "claims": [],
                "uncertainties": [],
                "follow_up_questions": [],
            }
        ),
        requested_model=model,
        resolved_model=model,
        provider_name="openrouter",
    )


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


def test_search_cli_persists_ranked_json_and_forwards_filters(monkeypatch, tmp_path, capsys):
    store, _ = GraphStore.create(tmp_path / "graph", "What changed this week?")
    calls = []

    class Retriever:
        provider_name = "perplexity"

        def search(self, *, query, options):
            calls.append((query, options))
            return SearchResponse(
                query=query,
                request={"query": query, "max_results": options.max_results},
                request_id="req-cli",
                results=[
                    SearchResult(
                        title="Official release",
                        url="https://example.com/release",
                        snippet="The project released a new model.",
                    )
                ],
                raw={"id": "req-cli", "results": []},
                provider_name="perplexity",
            )

    monkeypatch.setattr(cli_module, "PerplexitySearchClient", Retriever)
    code = main(
        [
            "--root",
            str(store.root),
            "search",
            "root",
            "--query",
            "latest release",
            "--max-results",
            "4",
            "--recency",
            "week",
            "--domain",
            "example.com",
            "--json",
        ]
    )
    assert code == 0
    output = json.loads(capsys.readouterr().out)["data"]
    assert output["run"]["mode"] == "search"
    assert output["ranked_results"][0]["rank"] == 1
    assert output["sources"][0]["excerpt"] == "The project released a new model."
    assert calls[0][0] == "latest release"
    assert calls[0][1].max_results == 4
    assert calls[0][1].recency == "week"
    assert calls[0][1].domains == ("example.com",)


def test_ask_cli_forwards_max_results_to_openrouter(monkeypatch, tmp_path, capsys):
    store, _ = GraphStore.create(tmp_path / "graph", "What is current?")
    calls = []

    class ChatClient:
        provider_name = "openrouter"

        def chat(self, **kwargs):
            calls.append(kwargs)
            return _cli_answer(kwargs["model"])

    monkeypatch.setattr(cli_module, "OpenRouterClient", ChatClient)
    code = main(
        [
            "--root",
            str(store.root),
            "ask",
            "root",
            "--search-provider",
            "openrouter",
            "--max-results",
            "3",
            "--json",
        ]
    )
    assert code == 0
    json.loads(capsys.readouterr().out)
    assert calls[0]["web"] is True
    assert calls[0]["max_search_results"] == 3


@pytest.mark.parametrize(("legacy_flag", "expected_web"), [("--web", True), ("--no-web", False)])
def test_ask_cli_preserves_legacy_web_flags(
    monkeypatch, tmp_path, capsys, legacy_flag, expected_web
):
    store, _ = GraphStore.create(tmp_path / "graph", "What is current?")
    calls = []

    class ChatClient:
        provider_name = "openrouter"

        def chat(self, **kwargs):
            calls.append(kwargs)
            return _cli_answer(kwargs["model"])

    monkeypatch.setattr(cli_module, "OpenRouterClient", ChatClient)
    assert main(["--root", str(store.root), "ask", "root", legacy_flag, "--json"]) == 0
    json.loads(capsys.readouterr().out)
    assert calls[0]["web"] is expected_web


def test_cli_rejects_conflicting_search_flags_before_provider_call(monkeypatch, tmp_path, capsys):
    store, _ = GraphStore.create(tmp_path / "graph", "What is current?")
    monkeypatch.setattr(
        cli_module,
        "OpenRouterClient",
        lambda: (_ for _ in ()).throw(AssertionError("provider must not be constructed")),
    )
    code = main(
        [
            "--root",
            str(store.root),
            "ask",
            "root",
            "--no-web",
            "--search-provider",
            "perplexity",
            "--json",
        ]
    )
    assert code == EXIT_VALIDATION
    assert "cannot be combined" in json.loads(capsys.readouterr().err)["error"]


def test_config_can_select_default_search_provider(tmp_path, capsys):
    store, _ = GraphStore.create(tmp_path / "graph", "What is current?")
    assert (
        main(
            [
                "--root",
                str(store.root),
                "config",
                "--search-provider",
                "perplexity",
                "--json",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)["data"]
    assert output["search_provider"] == "perplexity"
    assert store.load_project().settings["search_provider"] == "perplexity"


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
