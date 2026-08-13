from __future__ import annotations

import pytest

from research_tree.errors import ValidationError
from research_tree.models import Edge, Node, content_hash, new_id, stable_source_id, utc_now


def test_ids_have_type_prefix_and_validate_through_node():
    timestamp = utc_now()
    node = Node(
        id=new_id("question"),
        type="question",
        title="Why?",
        status="open",
        created_at=timestamp,
        updated_at=timestamp,
    )
    node.validate()
    assert node.id.startswith("q_")


def test_node_rejects_invalid_status():
    timestamp = utc_now()
    node = Node(
        id=new_id("question"),
        type="question",
        title="Why?",
        status="done-ish",
        created_at=timestamp,
        updated_at=timestamp,
    )
    with pytest.raises(ValidationError, match="status"):
        node.validate()


def test_only_questions_can_have_parents():
    timestamp = utc_now()
    node = Node(
        id=new_id("answer"),
        type="answer",
        title="An answer",
        status="answered",
        created_at=timestamp,
        updated_at=timestamp,
        parent_id=new_id("question"),
    )
    with pytest.raises(ValidationError, match="only question"):
        node.validate()


def test_edge_types_are_closed_and_targets_are_safe():
    Edge("supports", new_id("claim")).validate()
    with pytest.raises(ValidationError, match="edge type"):
        Edge("vibes_with", new_id("claim")).validate()
    with pytest.raises(ValidationError, match="invalid object ID"):
        Edge("supports", "../claim").validate()


def test_source_id_is_snapshot_deterministic():
    assert stable_source_id("https://example.com", "v1") == stable_source_id(
        "https://example.com", "v1"
    )
    assert stable_source_id("https://example.com", "v1") != stable_source_id(
        "https://example.com", "v2"
    )
    assert len(content_hash("hello")) == 64
