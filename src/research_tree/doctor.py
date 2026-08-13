"""Graph integrity checks."""

from __future__ import annotations

from dataclasses import dataclass, field

from .store import GraphStore


@dataclass
class DoctorReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return not self.errors


def inspect_graph(store: GraphStore) -> DoctorReport:
    with store.read_locked():
        return _inspect_graph_locked(store)


def _inspect_graph_locked(store: GraphStore) -> DoctorReport:
    report = DoctorReport()
    try:
        project = store.load_project()
    except Exception as exc:
        report.errors.append(f"cannot load project: {exc}")
        report.stats = {"nodes": 0, "questions": 0, "claims": 0, "sources": 0, "runs": 0}
        return report

    nodes = []
    for path in sorted(store.nodes_dir.glob("*.md")):
        try:
            nodes.append(store.load_node(path.stem))
        except Exception as exc:
            report.errors.append(f"cannot load node {path.stem}: {exc}")

    sources = []
    for path in sorted(store.sources_dir.glob("*.json")):
        try:
            sources.append(store.load_source(path.stem))
        except Exception as exc:
            report.errors.append(f"cannot load source {path.stem}: {exc}")
    node_map = {node.id: node for node in nodes}
    source_ids = {source.id for source in sources}
    run_ids = {path.stem for path in store.runs_dir.glob("*.json")}

    report.stats = {
        "nodes": len(nodes),
        "questions": sum(node.type == "question" for node in nodes),
        "claims": sum(node.type == "claim" for node in nodes),
        "sources": len(sources),
        "runs": len(run_ids),
    }

    if project.root_question_id not in node_map:
        report.errors.append(f"project root question is missing: {project.root_question_id}")
    elif node_map[project.root_question_id].type != "question":
        report.errors.append("project root_question_id does not point to a question")
    elif node_map[project.root_question_id].parent_id:
        report.errors.append("project root question must not have a parent")

    for node in nodes:
        if (
            node.type == "question"
            and node.id != project.root_question_id
            and node.parent_id is None
        ):
            report.errors.append(f"question {node.id} is disconnected from the configured root")
        if node.parent_id and node.parent_id not in node_map:
            report.errors.append(f"{node.id} has missing parent {node.parent_id}")
        if node.question_id and node.question_id not in node_map:
            report.errors.append(f"{node.id} has missing question {node.question_id}")
        for edge in node.edges:
            if edge.target not in node_map:
                report.errors.append(f"{node.id} edge {edge.type} targets missing {edge.target}")
        for source_id in node.source_ids:
            if source_id not in source_ids:
                report.errors.append(f"{node.id} cites missing source {source_id}")
        for run_id in node.run_ids:
            if run_id not in run_ids:
                report.errors.append(f"{node.id} references missing run {run_id}")

    # Question parentage must be a DAG even though the wider knowledge structure is a graph.
    for node in nodes:
        if node.type != "question":
            continue
        seen = {node.id}
        current = node
        while current.parent_id:
            if current.parent_id in seen:
                report.errors.append(f"question parent cycle detected at {current.parent_id}")
                break
            seen.add(current.parent_id)
            parent = node_map.get(current.parent_id)
            if not parent:
                break
            current = parent

    for run_id in sorted(run_ids):
        try:
            run = store.load_run(run_id)
        except Exception as exc:
            report.errors.append(f"cannot load run {run_id}: {exc}")
            continue
        if run.question_id not in node_map:
            report.errors.append(f"run {run_id} targets missing question {run.question_id}")
        for node_id in run.response_node_ids:
            if node_id not in node_map:
                report.errors.append(f"run {run_id} references missing response node {node_id}")
        for source_id in run.source_ids:
            if source_id not in source_ids:
                report.errors.append(f"run {run_id} references missing source {source_id}")

    if not (store.root / ".gitignore").exists():
        report.warnings.append(".gitignore is missing; local cursor state may be committed")
    if not (store.views_dir / "overview.md").exists():
        report.warnings.append("views/overview.md is missing; run `research-tree overview`")
    return report
