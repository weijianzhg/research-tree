from __future__ import annotations

import json

from research_tree.doctor import inspect_graph
from research_tree.models import Node, new_id, utc_now
from research_tree.render import (
    frontier,
    render_graph,
    render_node,
    render_tree,
    render_where,
    write_overview,
)


def test_tree_marks_focus_and_status(store):
    child = store.add_question("How are experts selected?")
    store.add_question("What is the auxiliary loss?", parent="root", status="proposed", focus=False)
    text = render_tree(store)
    assert "→" in text
    assert child.id in text
    assert "auxiliary loss" in text
    assert "├─" in text


def test_where_includes_breadcrumb_and_children(store):
    child = store.add_question("How are experts selected?")
    grandchild = store.add_question("What does top-2 routing mean?", parent=child.id, focus=False)
    text = render_where(store)
    assert store.load_project().root_question_id in text
    assert child.id in text
    assert grandchild.id in text


def test_tracked_overview_does_not_contain_local_cursor_focus(store):
    child = store.add_question("A local focus")
    path = write_overview(store, cursor="default")
    before = path.read_text(encoding="utf-8")
    store.set_focus("root", cursor="other-session")
    write_overview(store, cursor="other-session")
    assert path.read_text(encoding="utf-8") == before
    assert "Current focus:" not in before
    assert "→" not in before
    assert child.id in before


def test_frontier_is_scoped_to_subtree(store):
    left = store.add_question("Left branch", parent="root")
    left_child = store.add_question("Left child", parent=left.id, focus=False)
    store.add_question("Right branch", parent="root", focus=False)
    items = frontier(store, start=left.id)
    assert {item.id for item in items} == {left.id, left_child.id}


def test_frontier_uses_recorded_followup_priority(store):
    low = store.add_question("Low priority", parent="root", status="proposed", focus=False)
    high = store.add_question("High priority", parent="root", status="proposed", focus=False)
    low.tags = ["priority-4"]
    high.tags = ["priority-1"]
    store.update_node(low)
    store.update_node(high)
    proposed = [node.id for node in frontier(store, start="root") if node.status == "proposed"]
    assert proposed == [high.id, low.id]


def test_graph_exports_mermaid_dot_and_json(store):
    node = store.add_question("How are experts selected?", priority=1)
    assert render_graph(store, "mermaid").startswith("flowchart TD")
    assert render_graph(store, "dot").startswith("digraph research_tree")
    assert '"decomposes_into"' in render_graph(store, "json")
    assert '"priority-1"' in render_graph(store, "json")
    assert "Tags: priority-1" in render_node(store, node)


def test_doctor_detects_question_cycle(store):
    first = store.add_question("First")
    second = store.add_question("Second", parent=first.id)
    first.parent_id = second.id
    store.update_node(first)
    report = inspect_graph(store)
    assert not report.healthy
    assert any("cycle" in error for error in report.errors)


def test_doctor_detects_an_extra_root_question(store):
    created = utc_now()
    orphan = Node(
        id=new_id("question"),
        type="question",
        title="Disconnected question",
        status="open",
        created_at=created,
        updated_at=created,
    )
    store.save_node(orphan)
    report = inspect_graph(store)
    assert not report.healthy
    assert any("disconnected" in error for error in report.errors)


def test_doctor_reports_healthy_graph(store):
    store.add_question("A branch")
    report = inspect_graph(store)
    assert report.healthy
    assert report.stats["questions"] == 2


def test_doctor_detects_tampered_source_snapshot(store):
    source_path = store.sources_dir / "s_111111111111.json"
    source_path.write_text(
        json.dumps(
            {
                "id": "s_111111111111",
                "url": "https://example.com/source",
                "title": "Source",
                "retrieved_at": "2026-01-01T00:00:00Z",
                "content_hash": "not-the-real-hash",
                "excerpt": "captured text",
                "source_type": "web",
                "published_at": None,
                "authors": [],
                "metadata": {},
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    report = inspect_graph(store)
    assert not report.healthy
    assert any("content hash" in error for error in report.errors)
