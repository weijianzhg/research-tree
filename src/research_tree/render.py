"""Terminal, Markdown, Mermaid, DOT, and JSON graph views."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .models import Node
from .store import GraphStore

STATUS_MARKS = {
    "proposed": "·",
    "open": "?",
    "researching": "~",
    "answered": "✓",
    "uncertain": "!",
    "contested": "≠",
    "parked": "×",
}


def _recorded_priority(node: Node) -> int:
    for tag in node.tags:
        if tag.startswith("priority-") and tag.removeprefix("priority-").isdigit():
            return int(tag.removeprefix("priority-"))
    return 99


def _question_maps(nodes: Iterable[Node]):
    questions = {node.id: node for node in nodes if node.type == "question"}
    children: dict[str | None, list[Node]] = defaultdict(list)
    for node in questions.values():
        children[node.parent_id].append(node)
    for group in children.values():
        group.sort(key=lambda node: (_recorded_priority(node), node.created_at, node.id))
    return questions, children


def breadcrumb(store: GraphStore, node: Node) -> list[Node]:
    nodes = {item.id: item for item in store.list_nodes()}
    current = node
    if current.type != "question" and current.question_id:
        current = nodes[current.question_id]
    path = [current]
    seen = {current.id}
    while current.parent_id:
        if current.parent_id in seen or current.parent_id not in nodes:
            break
        current = nodes[current.parent_id]
        path.append(current)
        seen.add(current.id)
    return list(reversed(path))


def render_where(store: GraphStore, *, cursor: str = "default") -> str:
    focus = store.load_node(store.get_focus(cursor))
    trail = breadcrumb(store, focus)
    lines = [f"Project: {store.load_project().title}", f"Cursor: {cursor}", ""]
    lines.append(" / ".join(f"{node.id} {node.title}" for node in trail))
    lines.append(f"\nFocus: [{focus.status}] {focus.id} — {focus.title}")

    all_nodes = store.list_nodes()
    children = [
        node for node in all_nodes if node.type == "question" and node.parent_id == focus.id
    ]
    answers = [
        node
        for node in all_nodes
        if node.question_id == focus.id and node.type in {"answer", "synthesis"}
    ]
    if answers:
        lines.append(f"Answers: {len(answers)}")
    if children:
        lines.append("Branches:")
        for child in sorted(children, key=lambda item: (item.created_at, item.id)):
            lines.append(f"  {STATUS_MARKS.get(child.status, ' ')} {child.id}  {child.title}")
    return "\n".join(lines)


def render_tree(
    store: GraphStore,
    *,
    start: str = "root",
    depth: int | None = None,
    cursor: str = "default",
    mark_focus: bool = True,
) -> str:
    nodes = store.list_nodes()
    questions, children = _question_maps(nodes)
    start_node = store.load_node(store.resolve_node_id(start, cursor=cursor))
    if start_node.type != "question" and start_node.question_id:
        start_node = questions[start_node.question_id]
    focus_id = store.get_focus(cursor)
    answer_counts = Counter(
        node.question_id
        for node in nodes
        if node.type in {"answer", "synthesis"} and node.question_id
    )

    lines: list[str] = []

    def walk(node: Node, prefix: str, connector: str, level: int) -> None:
        marker = "→" if mark_focus and node.id == focus_id else STATUS_MARKS.get(node.status, " ")
        answer_note = (
            f" ({answer_counts[node.id]} answer{'s' if answer_counts[node.id] != 1 else ''})"
            if answer_counts[node.id]
            else ""
        )
        lines.append(f"{prefix}{connector}{marker} {node.id}  {node.title}{answer_note}")
        if depth is not None and level >= depth:
            if children.get(node.id):
                lines.append(f"{prefix}{'   ' if not connector else '│  '}…")
            return
        group = children.get(node.id, [])
        for index, child in enumerate(group):
            last = index == len(group) - 1
            next_connector = "└─ " if last else "├─ "
            next_prefix = prefix + ("   " if connector == "└─ " else "│  " if connector else "")
            walk(child, next_prefix, next_connector, level + 1)

    walk(start_node, "", "", 0)
    return "\n".join(lines)


def frontier(store: GraphStore, *, start: str = "focus", cursor: str = "default") -> list[Node]:
    nodes = store.list_nodes()
    questions, children = _question_maps(nodes)
    answers = {
        node.question_id
        for node in nodes
        if node.type in {"answer", "synthesis"} and node.question_id is not None
    }
    start_node = store.load_node(store.resolve_node_id(start, cursor=cursor))
    if start_node.type != "question" and start_node.question_id:
        start_node = questions[start_node.question_id]

    descendants: set[str] = set()
    stack = [start_node.id]
    while stack:
        current = stack.pop()
        descendants.add(current)
        stack.extend(node.id for node in children.get(current, []))

    candidates = [
        node
        for node in questions.values()
        if node.id in descendants
        and node.status in {"open", "proposed", "uncertain", "contested"}
        and node.id not in answers
    ]
    status_rank = {"open": 0, "uncertain": 1, "contested": 2, "proposed": 3}

    return sorted(
        candidates,
        key=lambda n: (status_rank.get(n.status, 9), _recorded_priority(n), n.created_at, n.id),
    )


def graph_data(store: GraphStore) -> dict:
    nodes = store.list_nodes()
    sources = store.list_sources()
    result_nodes = [
        {
            "id": node.id,
            "type": node.type,
            "title": node.title,
            "status": node.status,
            "confidence": node.confidence,
            "tags": node.tags,
        }
        for node in nodes
    ]
    edges = []
    for node in nodes:
        if node.parent_id:
            edges.append({"source": node.parent_id, "target": node.id, "type": "decomposes_into"})
        if node.question_id:
            edges.append({"source": node.id, "target": node.question_id, "type": "answers"})
        for edge in node.edges:
            edges.append({"source": node.id, "target": edge.target, "type": edge.type})
        for source_id in node.source_ids:
            edges.append({"source": source_id, "target": node.id, "type": "cited_by"})
    result_nodes.extend(
        {"id": source.id, "type": "source", "title": source.title, "status": "snapshot"}
        for source in sources
    )
    return {"project": store.load_project().to_dict(), "nodes": result_nodes, "edges": edges}


def render_graph(store: GraphStore, format_name: str) -> str:
    data = graph_data(store)
    if format_name == "json":
        return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
    if format_name == "mermaid":
        lines = ["flowchart TD"]
        for node in data["nodes"]:
            label = str(node["title"]).replace('"', "'")
            lines.append(f'  {node["id"]}["{label}"]')
        for edge in data["edges"]:
            lines.append(f"  {edge['source']} -->|{edge['type']}| {edge['target']}")
        return "\n".join(lines)
    if format_name == "dot":
        lines = ["digraph research_tree {", "  rankdir=LR;"]
        for node in data["nodes"]:
            label = str(node["title"]).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'  "{node["id"]}" [label="{label}"];')
        for edge in data["edges"]:
            lines.append(f'  "{edge["source"]}" -> "{edge["target"]}" [label="{edge["type"]}"];')
        lines.append("}")
        return "\n".join(lines)
    raise ValueError(f"unknown graph format: {format_name}")


def render_node(store: GraphStore, node: Node) -> str:
    lines = [
        f"{node.id}  [{node.type} / {node.status}]",
        node.title,
        f"Created: {node.created_at}",
    ]
    if node.confidence is not None:
        lines.append(f"Confidence: {node.confidence:.0%}")
    if node.tags:
        lines.append(f"Tags: {', '.join(node.tags)}")
    if node.parent_id:
        lines.append(f"Parent: {node.parent_id}")
    if node.question_id:
        lines.append(f"Question: {node.question_id}")
    if node.source_ids:
        lines.append(f"Sources: {', '.join(node.source_ids)}")
    if node.run_ids:
        lines.append(f"Runs: {', '.join(node.run_ids)}")
    if node.body.strip():
        lines.extend(["", node.body.rstrip()])
    return "\n".join(lines)


def write_overview(store: GraphStore, *, cursor: str = "default") -> Path:
    nodes = store.list_nodes()
    counts = Counter(node.type for node in nodes)
    candidates = frontier(store, start="root", cursor=cursor)
    project = store.load_project()
    lines = [
        f"# Research map — {project.title}",
        "",
        f"Updated: {project.updated_at}",
        "",
        "## Progress",
        "",
        " | ".join(f"{kind}: {counts[kind]}" for kind in sorted(counts)) or "No nodes",
        "",
        "## Inquiry tree",
        "",
        "```text",
        render_tree(store, start="root", cursor=cursor, mark_focus=False),
        "```",
        "",
        "## Research frontier",
        "",
    ]
    if candidates:
        for node in candidates[:20]:
            lines.append(f"- [{node.title}](../nodes/{node.id}.md) — `{node.status}`")
    else:
        lines.append("No unanswered questions in the current graph.")
    lines.append("")
    path = store.views_dir / "overview.md"
    store._write_text(path, "\n".join(lines))
    return path
