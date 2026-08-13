"""Command-line interface for Research Tree."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .doctor import inspect_graph
from .errors import (
    ConfigurationError,
    ModelOutputError,
    NotFoundError,
    ProviderError,
    ResearchTreeError,
    ValidationError,
)
from .models import Source, content_hash, stable_source_id, utc_now
from .providers import OpenRouterClient
from .render import (
    STATUS_LEGEND,
    frontier,
    graph_data,
    render_graph,
    render_node,
    render_tree,
    render_where,
    write_overview,
)
from .research import (
    ask_question,
    record_manual_answer,
    run_council,
    synthesize_answers,
    verify_claims,
)
from .store import GraphStore, _atomic_write, load_store

EXIT_NOT_FOUND = 3
EXIT_PROVIDER = 4
EXIT_VALIDATION = 5


def _node_json(node) -> dict[str, Any]:
    return {**node.frontmatter(), "body": node.body}


def _outcome_json(outcome) -> dict[str, Any]:
    return {
        "answer": _node_json(outcome.answer),
        "claims": [_node_json(node) for node in outcome.claims],
        "followups": [_node_json(node) for node in outcome.followups],
        "perspectives": [_node_json(node) for node in outcome.perspectives],
        "run": outcome.run.to_dict(),
        "sources": [source.to_dict() for source in outcome.sources],
    }


def _verification_json(outcome) -> dict[str, Any]:
    return {
        "note": _node_json(outcome.note),
        "claims": [_node_json(node) for node in outcome.claims],
        "run": outcome.run.to_dict(),
        "verdicts": outcome.verdicts,
    }


def emit(data: Any, *, as_json: bool, human: str | None = None) -> None:
    if as_json:
        print(json.dumps({"ok": True, "data": data}, indent=2, ensure_ascii=False, sort_keys=True))
    elif human is not None:
        print(human)
    elif isinstance(data, str):
        print(data)
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))


def _store(args) -> GraphStore:
    return load_store(args.root)


def cmd_init(args):
    settings = {"default_model": args.model} if args.model else None
    store, question = GraphStore.create(
        args.directory, args.question, title=args.title, settings=settings
    )
    if args.cursor != "default":
        store.set_focus(question.id, cursor=args.cursor)
    write_overview(store, cursor=args.cursor)
    data = {
        "root": str(store.root),
        "project": store.load_project().to_dict(),
        "root_question": _node_json(question),
    }
    emit(
        data,
        as_json=args.json,
        human=(
            f"Created research tree at {store.root}\n"
            f"Root question: {question.id} — {question.title}\n"
            f'Next: research-tree --root {store.root} branch "a useful follow-up"'
        ),
    )


def cmd_where(args):
    store = _store(args)
    focus = store.load_node(store.get_focus(args.cursor))
    nodes = store.list_nodes()
    path = []
    current = focus
    if current.type != "question" and current.question_id:
        current = store.load_node(current.question_id)
    seen = set()
    while current.id not in seen:
        path.append(current)
        seen.add(current.id)
        if not current.parent_id:
            break
        current = store.load_node(current.parent_id)
    path.reverse()
    children = sorted(
        [node for node in nodes if node.type == "question" and node.parent_id == focus.id],
        key=lambda node: (node.created_at, node.id),
    )
    answers = [
        node
        for node in nodes
        if node.question_id == focus.id and node.type in {"answer", "synthesis"}
    ]
    emit(
        {
            "focus": _node_json(focus),
            "project": store.load_project().to_dict(),
            "path": [_node_json(node) for node in path],
            "children": [_node_json(node) for node in children],
            "answers": [_node_json(node) for node in answers],
        },
        as_json=args.json,
        human=render_where(store, cursor=args.cursor),
    )


def cmd_focus(args):
    store = _store(args)
    with store.locked():
        node = store.set_focus(
            store.resolve_node_id(args.node, cursor=args.cursor), cursor=args.cursor
        )
    emit(
        {"cursor": args.cursor, "focus": _node_json(node)},
        as_json=args.json,
        human=f"Focused {args.cursor} on {node.id} — {node.title}",
    )


def cmd_branch(args):
    store = _store(args)
    with store.locked():
        node = store.add_question(
            args.question,
            parent=args.from_node,
            status="proposed" if args.proposed else "open",
            body=(f"# {args.question}\n\n{args.note.strip()}\n" if args.note else ""),
            priority=args.priority,
            cursor=args.cursor,
            focus=not args.stay,
        )
        project = store.load_project()
        store.save_project(project)
        write_overview(store, cursor=args.cursor)
    human = f"Branched to {node.id} [{node.status}] — {node.title} (under {node.parent_id})"
    if not args.stay:
        human += f"\nFocused {args.cursor} on {node.id}."
    emit(_node_json(node), as_json=args.json, human=human)


def cmd_answer(args):
    store = _store(args)
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    node = record_manual_answer(store, args.node, text, cursor=args.cursor)
    emit(_node_json(node), as_json=args.json, human=f"Saved {node.id} — {node.title}")


def cmd_ask(args):
    store = _store(args)
    question_id = store.resolve_node_id(args.node, cursor=args.cursor)
    settings = store.load_project().settings
    chosen = args.model or settings["default_model"]
    use_web = settings.get("web_search", True) if args.web is None else args.web
    if not args.json:
        print(
            f"Researching {question_id} with {chosen}"
            + (" + web evidence..." if use_web else "..."),
            file=sys.stderr,
        )
    outcome = ask_question(
        store,
        question_id,
        client=OpenRouterClient(),
        model=args.model,
        web=args.web,
        reasoning_effort=args.effort,
        followups=args.followups,
        cursor=args.cursor,
    )
    human = render_node(store, outcome.answer)
    if outcome.followups:
        human += "\n\nSuggested branches:\n" + "\n".join(
            f"  {node.id}  {node.title}" for node in outcome.followups
        )
    if outcome.run.usage.get("total_cost") is not None:
        human += f"\n\nRecorded cost: ${outcome.run.usage['total_cost']:.4f}"
    emit(_outcome_json(outcome), as_json=args.json, human=human)


def cmd_council(args):
    store = _store(args)
    models = args.models or store.load_project().settings["council_models"]
    if not args.json:
        print(
            f"Running evidence council: {len(models)} first opinions, "
            "peer reviews, then synthesis...",
            file=sys.stderr,
        )
    outcome = run_council(
        store,
        args.node,
        client=OpenRouterClient(),
        models=args.models,
        chairman_model=args.chairman,
        web=args.web,
        reasoning_effort=args.effort,
        followups=args.followups,
        cursor=args.cursor,
    )
    human = render_node(store, outcome.answer)
    human += "\n\nPerspectives:\n" + "\n".join(
        f"  {node.id}  {node.title}" for node in outcome.perspectives
    )
    if outcome.followups:
        human += "\n\nSuggested branches:\n" + "\n".join(
            f"  {node.id}  {node.title}" for node in outcome.followups
        )
    if outcome.run.usage.get("total_cost") is not None:
        human += f"\n\nRecorded cost: ${outcome.run.usage['total_cost']:.4f}"
    emit(_outcome_json(outcome), as_json=args.json, human=human)


def cmd_verify(args):
    store = _store(args)
    chosen = args.model or store.load_project().settings["default_model"]
    if not args.json:
        print(
            f"Verifying captured citation support with {chosen}...",
            file=sys.stderr,
        )
    outcome = verify_claims(
        store,
        args.node,
        client=OpenRouterClient(),
        model=args.model,
        reasoning_effort=args.effort,
        cursor=args.cursor,
    )
    human = render_node(store, outcome.note)
    if outcome.run.usage.get("total_cost") is not None:
        human += f"\n\nRecorded cost: ${outcome.run.usage['total_cost']:.4f}"
    emit(_verification_json(outcome), as_json=args.json, human=human)


def cmd_synthesize(args):
    store = _store(args)
    chosen = args.model or store.load_project().settings["default_model"]
    if not args.json:
        print(
            f"Synthesizing answers under {args.node} with {chosen}...",
            file=sys.stderr,
        )
    outcome = synthesize_answers(
        store,
        args.node,
        client=OpenRouterClient(),
        model=args.model,
        reasoning_effort=args.effort,
        cursor=args.cursor,
    )
    human = render_node(store, outcome.node)
    if outcome.aggregated:
        human += "\n\nAggregated answers:\n" + "\n".join(
            f"  {node_id}" for node_id in outcome.aggregated
        )
    if outcome.run.usage.get("total_cost") is not None:
        human += f"\n\nRecorded cost: ${outcome.run.usage['total_cost']:.4f}"
    data = {
        "node": _node_json(outcome.node),
        "run": outcome.run.to_dict(),
        "aggregated": outcome.aggregated,
    }
    emit(data, as_json=args.json, human=human)


def cmd_tree(args):
    store = _store(args)
    text = render_tree(store, start=args.from_node, depth=args.depth, cursor=args.cursor)
    emit(graph_data(store), as_json=args.json, human=text + "\n\n" + STATUS_LEGEND)


def cmd_next(args):
    store = _store(args)
    start_id = store.resolve_node_id(args.from_node, cursor=args.cursor)
    candidates = frontier(store, start=start_id, cursor=args.cursor)[: args.limit]
    if args.focus and candidates:
        with store.locked():
            store.set_focus(candidates[0].id, cursor=args.cursor)
    scope = (
        "the whole tree"
        if start_id == store.load_project().root_question_id
        else f"branch {start_id}"
    )
    if not candidates:
        human = f"No unanswered questions in {scope}."
    else:
        human = f"Unanswered questions in {scope}:\n" + "\n".join(
            f"{index}. [{node.status}] {node.id} — {node.title}"
            for index, node in enumerate(candidates, 1)
        )
    if args.focus and candidates:
        human += f"\n\nFocused on {candidates[0].id}."
    data = [_node_json(node) for node in candidates]
    emit(data, as_json=args.json, human=human)


def cmd_show(args):
    store = _store(args)
    node = store.load_node(store.resolve_node_id(args.node, cursor=args.cursor))
    emit(_node_json(node), as_json=args.json, human=render_node(store, node))


def cmd_graph(args):
    store = _store(args)
    text = render_graph(store, args.format)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        _atomic_write(output, text + "\n")
        emit(
            {"output": str(output), "format": args.format},
            as_json=args.json,
            human=f"Wrote {args.format} graph to {output}",
        )
    else:
        emit(graph_data(store) if args.format == "json" else text, as_json=args.json, human=text)


def cmd_overview(args):
    store = _store(args)
    with store.locked():
        path = write_overview(store, cursor=args.cursor)
    emit({"path": str(path)}, as_json=args.json, human=f"Updated {path}")


def cmd_doctor(args):
    store = _store(args)
    report = inspect_graph(store)
    data = {
        "healthy": report.healthy,
        "errors": report.errors,
        "warnings": report.warnings,
        "stats": report.stats,
    }
    human = ["Research tree is healthy." if report.healthy else "Research tree has errors."]
    human.extend(f"ERROR: {item}" for item in report.errors)
    human.extend(f"WARNING: {item}" for item in report.warnings)
    human.append("  " + "  ".join(f"{key}: {value}" for key, value in report.stats.items()))
    if not report.healthy:
        if args.json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "research graph failed integrity checks",
                        "exit_code": EXIT_VALIDATION,
                        "data": data,
                    },
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        else:
            print("\n".join(human))
        return EXIT_VALIDATION
    emit(data, as_json=args.json, human="\n".join(human))
    return 0


def _promotion_markdown(store: GraphStore, node) -> str:
    sources = [store.load_source(source_id) for source_id in node.source_ids]
    lines = [
        f"## {node.title}",
        "",
        f"Research node: `{node.id}`",
        "",
        node.body.strip(),
    ]
    if sources and "## Sources captured" not in node.body:
        lines.extend(["", "### Sources", ""])
        lines.extend(f"- [{source.title}]({source.url})" for source in sources)
    lines.extend(["", "---", ""])
    return "\n".join(lines)


def cmd_promote(args):
    store = _store(args)
    node = store.load_node(store.resolve_node_id(args.node, cursor=args.cursor))
    if node.type not in {"answer", "synthesis"}:
        raise ValidationError(
            f"promote expects an answer or synthesis node, got {node.type}: {node.id}. "
            "Pass a bare node ID such as a_xxxx or y_xxxx (run `show` for details)."
        )
    if (
        node.type in {"answer", "synthesis"}
        and "manual" not in node.tags
        and "verified" not in node.tags
        and not args.allow_unverified
    ):
        raise ResearchTreeError(
            f"{node.id} is model output that has not been verified; run `verify {node.id}` "
            "or pass --allow-unverified deliberately"
        )
    if node.status in {"uncertain", "contested"} and not args.allow_uncertain:
        raise ResearchTreeError(
            f"{node.id} is {node.status}; resolve its evidence gaps or pass --allow-uncertain"
        )
    target = Path(args.to).expanduser().resolve()
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    separator = "" if not existing or existing.endswith("\n") else "\n"
    _atomic_write(target, existing + separator + "\n" + _promotion_markdown(store, node))
    emit(
        {"node_id": node.id, "target": str(target)},
        as_json=args.json,
        human=f"Promoted {node.id} into {target}",
    )


def cmd_open(args):
    store = _store(args)
    if args.node:
        node_id = store.resolve_node_id(args.node, cursor=args.cursor)
        target = store.node_path(node_id)
    else:
        target = store.views_dir / "overview.md"
    editor = os.environ.get("RESEARCH_EDITOR") or os.environ.get("EDITOR")
    if not editor:
        editor = next(
            (name for name in ("cursor", "code", "ghostwriter", "open") if shutil.which(name)), None
        )
    if not editor:
        raise ConfigurationError("no editor found; set RESEARCH_EDITOR or EDITOR")
    command = [*editor.split(), str(target)]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    emit(
        {"path": str(target), "editor": editor},
        as_json=args.json,
        human=f"Opened {target} in {editor}",
    )


def cmd_config(args):
    store = _store(args)
    with store.locked():
        project = store.load_project()
        changed = False
        if args.model:
            project.settings["default_model"] = args.model
            changed = True
        if args.chairman:
            project.settings["chairman_model"] = args.chairman
            changed = True
        if args.council_model:
            project.settings["council_models"] = args.council_model
            changed = True
        if args.effort:
            project.settings["reasoning_effort"] = args.effort
            changed = True
        if args.max_search_results:
            project.settings["max_search_results"] = args.max_search_results
            changed = True
        if args.web is not None:
            project.settings["web_search"] = args.web
            changed = True
        if changed:
            store.save_project(project)
    emit(
        project.settings,
        as_json=args.json,
        human=json.dumps(project.settings, indent=2, ensure_ascii=False),
    )


def cmd_source_add(args):
    store = _store(args)
    content = args.excerpt or ""
    source = Source(
        id=stable_source_id(args.url, content),
        url=args.url,
        title=args.title or args.url,
        retrieved_at=utc_now(),
        content_hash=content_hash(content),
        excerpt=content,
        metadata={"via": "manual"},
    )
    with store.locked():
        store.save_source(source)
    emit(source.to_dict(), as_json=args.json, human=f"Saved {source.id} — {source.title}")


def cmd_source_list(args):
    store = _store(args)
    sources = store.list_sources()
    emit(
        [source.to_dict() for source in sources],
        as_json=args.json,
        human="\n".join(f"{source.id}  {source.title}\n  {source.url}" for source in sources)
        or "No sources captured.",
    )


def cmd_run_list(args):
    store = _store(args)
    runs = [store.load_run(path.stem) for path in sorted(store.runs_dir.glob("*.json"))]
    emit(
        [run.to_dict() for run in runs],
        as_json=args.json,
        human="\n".join(
            f"{run.id}  {run.mode}  {run.created_at}  {', '.join(run.resolved_models)}"
            for run in runs
        )
        or "No model runs recorded.",
    )


def cmd_run_show(args):
    store = _store(args)
    run_id = args.run
    if not run_id.startswith("r_"):
        matches = [path.stem for path in store.runs_dir.glob(f"{run_id}*.json")]
        if len(matches) != 1:
            raise NotFoundError(f"run prefix is missing or ambiguous: {run_id}")
        run_id = matches[0]
    run = store.load_run(run_id)
    emit(
        run.to_dict(),
        as_json=args.json,
        human=json.dumps(run.to_dict(), indent=2, ensure_ascii=False),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-tree",
        description="Traverse a Git-native tree of questions with evidence and model provenance.",
    )
    parser.add_argument("--root", help="research directory (auto-discovered when omitted)")
    parser.add_argument("--cursor", default="default", help="local focus cursor name")
    parser.add_argument("--json", action="store_true", help="stable machine-readable output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("init", help="create a research graph")
    command.add_argument("directory")
    command.add_argument("question")
    command.add_argument("--title")
    command.add_argument("--model")
    command.set_defaults(func=cmd_init)

    command = sub.add_parser("where", help="show the current inquiry location")
    command.set_defaults(func=cmd_where)

    command = sub.add_parser("focus", help="move a local cursor to a node")
    command.add_argument("node")
    command.set_defaults(func=cmd_focus)

    command = sub.add_parser("branch", help="add a child question")
    command.add_argument("question")
    command.add_argument("--from", dest="from_node", default="focus")
    command.add_argument("--note")
    command.add_argument("--proposed", action="store_true")
    command.add_argument("--priority", type=int, choices=range(1, 6))
    command.add_argument("--stay", action="store_true", help="do not focus the new branch")
    command.set_defaults(func=cmd_branch)

    command = sub.add_parser("answer", help="record a manual answer")
    command.add_argument("node", nargs="?", default="focus")
    group = command.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--file")
    command.set_defaults(func=cmd_answer)

    command = sub.add_parser("ask", help="research a question with one model")
    command.add_argument("node", nargs="?", default="focus")
    command.add_argument("--model")
    command.add_argument("--effort", choices=["minimal", "low", "medium", "high", "xhigh", "max"])
    web_group = command.add_mutually_exclusive_group()
    web_group.add_argument("--web", dest="web", action="store_true")
    web_group.add_argument("--no-web", dest="web", action="store_false")
    command.set_defaults(web=None)
    command.add_argument("--followups", type=int, default=4)
    command.set_defaults(func=cmd_ask)

    command = sub.add_parser(
        "council", help="run independent answers, blind reviews, and synthesis"
    )
    command.add_argument("node", nargs="?", default="focus")
    command.add_argument("--model", dest="models", action="append", help="council model (repeat)")
    command.add_argument("--chairman")
    command.add_argument("--effort", choices=["minimal", "low", "medium", "high", "xhigh", "max"])
    web_group = command.add_mutually_exclusive_group()
    web_group.add_argument("--web", dest="web", action="store_true")
    web_group.add_argument("--no-web", dest="web", action="store_false")
    command.set_defaults(web=None)
    command.add_argument("--followups", type=int, default=5)
    command.set_defaults(func=cmd_council)

    command = sub.add_parser("verify", help="check claim entailment against captured sources")
    command.add_argument("node", nargs="?", default="focus")
    command.add_argument("--model", help="independent verifier model")
    command.add_argument("--effort", choices=["minimal", "low", "medium", "high", "xhigh", "max"])
    command.set_defaults(func=cmd_verify)

    command = sub.add_parser("synthesize", help="merge answered questions into one synthesis node")
    command.add_argument("node", nargs="?", default="focus")
    command.add_argument("--model")
    command.add_argument("--effort", choices=["minimal", "low", "medium", "high", "xhigh", "max"])
    command.set_defaults(func=cmd_synthesize)

    command = sub.add_parser("tree", help="render the question hierarchy")
    command.add_argument("--from", dest="from_node", default="root")
    command.add_argument("--depth", type=int)
    command.set_defaults(func=cmd_tree)

    command = sub.add_parser("next", help="rank unanswered questions")
    command.add_argument("--from", dest="from_node", default="root")
    command.add_argument("--limit", type=int, default=8)
    command.add_argument("--focus", action="store_true", help="focus the first result")
    command.set_defaults(func=cmd_next)

    command = sub.add_parser("show", help="show one node")
    command.add_argument("node", nargs="?", default="focus")
    command.set_defaults(func=cmd_show)

    command = sub.add_parser("graph", help="export the wider typed graph")
    command.add_argument("--format", choices=["mermaid", "dot", "json"], default="mermaid")
    command.add_argument("--output")
    command.set_defaults(func=cmd_graph)

    command = sub.add_parser("overview", help="regenerate the Markdown overview")
    command.set_defaults(func=cmd_overview)

    command = sub.add_parser("doctor", help="validate graph integrity")
    command.set_defaults(func=cmd_doctor)

    command = sub.add_parser("promote", help="append a research node to a writing file")
    command.add_argument("node", nargs="?", default="focus")
    command.add_argument("--to", required=True)
    command.add_argument("--allow-unverified", action="store_true")
    command.add_argument("--allow-uncertain", action="store_true")
    command.set_defaults(func=cmd_promote)

    command = sub.add_parser("open", help="open the overview or a node in an editor")
    command.add_argument("node", nargs="?")
    command.set_defaults(func=cmd_open)

    command = sub.add_parser("config", help="show or update project model settings")
    command.add_argument("--model")
    command.add_argument("--chairman")
    command.add_argument("--council-model", action="append")
    command.add_argument("--effort", choices=["minimal", "low", "medium", "high", "xhigh", "max"])
    command.add_argument("--max-search-results", type=int, choices=range(1, 21))
    web_group = command.add_mutually_exclusive_group()
    web_group.add_argument("--web", dest="web", action="store_true")
    web_group.add_argument("--no-web", dest="web", action="store_false")
    command.set_defaults(web=None)
    command.set_defaults(func=cmd_config)

    source = sub.add_parser("source", help="inspect or add source snapshots")
    source_sub = source.add_subparsers(dest="source_command", required=True)
    source_add = source_sub.add_parser("add")
    source_add.add_argument("url")
    source_add.add_argument("--title")
    source_add.add_argument("--excerpt")
    source_add.set_defaults(func=cmd_source_add)
    source_list = source_sub.add_parser("list")
    source_list.set_defaults(func=cmd_source_list)

    run = sub.add_parser("run", help="inspect immutable model runs")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    run_list = run_sub.add_parser("list")
    run_list.set_defaults(func=cmd_run_list)
    run_show = run_sub.add_parser("show")
    run_show.add_argument("run")
    run_show.set_defaults(func=cmd_run_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    json_anywhere = "--json" in raw_args
    if json_anywhere:
        raw_args = [item for item in raw_args if item != "--json"]
    args = parser.parse_args(raw_args)
    args.json = args.json or json_anywhere
    try:
        result = args.func(args)
        return result if isinstance(result, int) else 0
    except NotFoundError as exc:
        code = EXIT_NOT_FOUND
        error_message = str(exc)
    except ModelOutputError as exc:
        code = EXIT_VALIDATION
        error_message = str(exc)
    except (ConfigurationError, ProviderError) as exc:
        code = EXIT_PROVIDER
        error_message = str(exc)
    except ResearchTreeError as exc:
        code = EXIT_VALIDATION
        error_message = str(exc)
    if args.json:
        print(
            json.dumps({"ok": False, "error": error_message, "exit_code": code}),
            file=sys.stderr,
        )
    else:
        print(f"error: {error_message}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
