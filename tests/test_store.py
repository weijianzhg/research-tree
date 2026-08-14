from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import threading
import time

import pytest

from research_tree.doctor import inspect_graph
from research_tree.errors import NotFoundError, ValidationError
from research_tree.models import Node, new_id, utc_now
from research_tree.store import GraphStore


def test_create_writes_git_friendly_layout(store):
    assert (store.root / "project.json").is_file()
    assert (store.root / "nodes").is_dir()
    assert (store.root / "sources").is_dir()
    assert (store.root / "runs").is_dir()
    assert (store.root / "views" / "overview.md").is_file()
    assert (store.root / ".gitignore").read_text() == ".state/\n"
    assert store.load_project().root_question_id.startswith("q_")


def test_create_refuses_to_overwrite_project(store):
    with pytest.raises(ValidationError, match="already exists"):
        GraphStore.create(store.root, "A replacement root?")


def test_create_refuses_a_nonempty_directory_without_changing_it(tmp_path):
    root = tmp_path / "graph"
    root.mkdir()
    (root / ".gitignore").write_text("private-notes/\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="non-empty"):
        GraphStore.create(root, "What should we inspect?")
    assert (root / ".gitignore").read_text() == "private-notes/\n"


def test_create_refuses_existing_canonical_directories(tmp_path):
    root = tmp_path / "graph"
    (root / "nodes").mkdir(parents=True)
    with pytest.raises(ValidationError, match="non-empty"):
        GraphStore.create(root, "What should we inspect?")


def _concurrent_create_worker(root, question, ready, start, results):
    ready.put(True)
    start.wait(timeout=5)
    try:
        GraphStore.create(root, question)
    except Exception as exc:
        results.put(("error", str(exc)))
    else:
        results.put(("ok", question))


def _crashing_create_worker(root):
    import research_tree.store as store_module

    original_atomic_write = store_module._atomic_write

    def write_then_crash(path, text):
        original_atomic_write(path, text)
        if path.name == "project.json":
            os._exit(91)

    store_module._atomic_write = write_then_crash
    store_module.GraphStore.create(root, "Interrupted root?")


def test_concurrent_create_has_one_winner_and_a_healthy_graph(tmp_path):
    root = tmp_path / "graph"
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    results = context.Queue()
    start = context.Event()
    processes = [
        context.Process(
            target=_concurrent_create_worker,
            args=(root, question, ready, start, results),
        )
        for question in ("First root?", "Second root?")
    ]
    for process in processes:
        process.start()
    ready.get(timeout=10)
    ready.get(timeout=10)
    start.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    outcomes = [results.get(timeout=5), results.get(timeout=5)]
    assert sorted(status for status, _ in outcomes) == ["error", "ok"]
    store = GraphStore(root)
    assert len(store.list_nodes(node_type="question")) == 1
    assert inspect_graph(store).healthy


def test_interrupted_create_never_publishes_a_partial_graph_and_can_be_retried(tmp_path):
    root = tmp_path / "graph"
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_crashing_create_worker, args=(root,))
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 91
    assert not root.exists()

    store, _ = GraphStore.create(root, "Healthy retry?")
    assert inspect_graph(store).healthy


def test_node_markdown_round_trip_preserves_metadata_and_body(store):
    timestamp = utc_now()
    node = Node(
        id=new_id("concept"),
        type="concept",
        title="Expert routing",
        status="open",
        created_at=timestamp,
        updated_at=timestamp,
        body="# Expert routing\n\nTokens are assigned to experts.\n",
        tags=["moe", "routing"],
    )
    store.save_node(node)
    loaded = store.load_node(node.id)
    assert loaded.frontmatter() == node.frontmatter()
    assert loaded.body == node.body


def test_branch_and_independent_cursors(store):
    first = store.add_question("How does top-k routing work?")
    second = store.add_question(
        "What is expert parallelism?", parent="root", cursor="other", focus=True
    )
    assert store.get_focus() == first.id
    assert store.get_focus("other") == second.id
    assert first.parent_id == store.load_project().root_question_id
    assert second.parent_id == store.load_project().root_question_id


def test_manual_branch_can_record_priority(store):
    node = store.add_question("A ranked branch", priority=2)
    assert node.tags == ["priority-2"]


def test_resolve_unique_prefix_and_reject_ambiguous(store):
    node = store.add_question("A branch")
    assert store.resolve_node_id(node.id[:8]) == node.id
    with pytest.raises(NotFoundError):
        store.resolve_node_id("q_deadbeef")
    with pytest.raises(ValidationError, match="invalid node reference"):
        store.resolve_node_id("../*")


def test_resolve_parent_reference(store):
    root = store.load_project().root_question_id
    child = store.add_question("Child", parent="root")
    grandchild = store.add_question("Grandchild", parent=child.id)
    assert store.get_focus() == grandchild.id
    assert store.resolve_node_id("..") == child.id
    store.set_focus(child.id)
    assert store.resolve_node_id("..") == root
    store.set_focus(root)
    assert store.resolve_node_id("..") == root  # root has no parent; stays put


def test_is_node_reference_distinguishes_text(store):
    assert store.is_node_reference("focus")
    assert store.is_node_reference("root")
    assert store.is_node_reference("..")
    assert store.is_node_reference("q_abc123")
    assert not store.is_node_reference("How do I secure a call?")
    assert not store.is_node_reference("  ")


def test_cursor_name_cannot_traverse_directories(store):
    with pytest.raises(ValidationError, match="cursor names"):
        store.cursor_path("../../secrets")


def test_corrupt_metadata_is_not_silently_replaced(store):
    node_id = store.load_project().root_question_id
    store.node_path(node_id).write_text("not frontmatter", encoding="utf-8")
    with pytest.raises(ValidationError, match="frontmatter"):
        store.load_node(node_id)


def test_node_filename_must_match_metadata_id(store):
    node_id = store.load_project().root_question_id
    text = store.node_path(node_id).read_text(encoding="utf-8")
    other_id = new_id("question")
    store.node_path(node_id).write_text(text.replace(node_id, other_id), encoding="utf-8")
    with pytest.raises(ValidationError, match="does not match metadata ID"):
        store.load_node(node_id)


def test_discover_finds_nested_research_directory(store, monkeypatch, tmp_path):
    topic = tmp_path / "topic"
    topic.mkdir()
    graph, _ = GraphStore.create(topic / "research", "A root question?")
    monkeypatch.delenv("RESEARCH_TREE_ROOT", raising=False)
    assert GraphStore.discover(topic).root == graph.root


def test_project_json_is_valid_and_versioned(store):
    data = json.loads((store.root / "project.json").read_text())
    assert data["schema_version"] == 1
    assert data["settings"]["web_search"] is True


def test_transaction_rolls_back_partial_writes(store):
    created = utc_now()
    first = Node(
        id=new_id("note"),
        type="note",
        title="First",
        status="open",
        created_at=created,
        updated_at=created,
    )
    second = Node(
        id=new_id("note"),
        type="note",
        title="Second",
        status="open",
        created_at=created,
        updated_at=created,
    )
    with pytest.raises(OSError, match="simulated failure"):
        with (
            store.locked(),
            store.transaction([store.node_path(first.id), store.node_path(second.id)]),
        ):
            store.save_node(first)
            raise OSError("simulated failure")
    assert not store.node_path(first.id).exists()
    assert not store.node_path(second.id).exists()


def test_lock_recovers_an_interrupted_transaction(store):
    node = store.load_node("root")
    original = store.node_path(node.id).read_text(encoding="utf-8")
    transaction_id = "1" * 32
    transaction = store.root / ".state" / "transactions" / transaction_id
    transaction.mkdir(parents=True)
    (transaction / "backup-0").write_text(original, encoding="utf-8")
    entry = {
        "relative": f"nodes/{node.id}.md",
        "existed": True,
        "backup_sha256": hashlib.sha256(original.encode()).hexdigest(),
    }
    manifest = {
        "schema_version": 2,
        "transaction_id": transaction_id,
        "paths": [entry],
        "manifest_hmac": store._transaction_manifest_hmac(transaction_id, [entry]),
    }
    expected_hash = hashlib.sha256(original.encode()).hexdigest()
    content_hash_value = hashlib.sha256(b"partial").hexdigest()
    marker_payload = {
        "transaction_id": transaction_id,
        "entry": entry,
        "expected_sha256": expected_hash,
        "content_sha256": content_hash_value,
    }
    (transaction / "written-0.json").write_text(
        json.dumps(
            {
                "transaction_id": transaction_id,
                "expected_sha256": expected_hash,
                "content_sha256": content_hash_value,
                "marker_hmac": store._journal_hmac(marker_payload),
            }
        ),
        encoding="utf-8",
    )
    (transaction / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    store.node_path(node.id).write_text("partial", encoding="utf-8")
    with store.locked():
        pass
    assert store.node_path(node.id).read_text(encoding="utf-8") == original
    assert not transaction.exists()


def test_recovery_preserves_journal_when_expected_backup_is_missing(store):
    node = store.load_node("root")
    transaction_id = "2" * 32
    transaction = store.root / ".state" / "transactions" / transaction_id
    transaction.mkdir(parents=True)
    entry = {
        "relative": f"nodes/{node.id}.md",
        "existed": True,
        "backup_sha256": "0" * 64,
    }
    current_hash = hashlib.sha256(store.node_path(node.id).read_bytes()).hexdigest()
    marker_payload = {
        "transaction_id": transaction_id,
        "entry": entry,
        "expected_sha256": current_hash,
        "content_sha256": current_hash,
    }
    (transaction / "written-0.json").write_text(
        json.dumps(
            {
                "transaction_id": transaction_id,
                "expected_sha256": current_hash,
                "content_sha256": current_hash,
                "marker_hmac": store._journal_hmac(marker_payload),
            }
        ),
        encoding="utf-8",
    )
    (transaction / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "transaction_id": transaction_id,
                "paths": [entry],
                "manifest_hmac": store._transaction_manifest_hmac(transaction_id, [entry]),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="missing backup"):
        GraphStore(store.root)
    assert transaction.is_dir()


def test_corrupt_journal_cannot_delete_an_existing_canonical_file(store):
    node = store.load_node("root")
    original = store.node_path(node.id).read_text(encoding="utf-8")
    transaction_id = "3" * 32
    transaction = store.root / ".state" / "transactions" / transaction_id
    transaction.mkdir(parents=True)
    entry = {
        "relative": f"nodes/{node.id}.md",
        "existed": False,
        "backup_sha256": None,
    }
    (transaction / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "transaction_id": transaction_id,
                "paths": [entry],
                "manifest_hmac": store._transaction_manifest_hmac(transaction_id, [entry]),
            }
        ),
        encoding="utf-8",
    )
    GraphStore(store.root)
    assert store.node_path(node.id).read_text(encoding="utf-8") == original
    assert not transaction.exists()


def test_transaction_rejects_an_outside_path_without_hanging(store, tmp_path):
    with pytest.raises(ValidationError, match="escapes"):
        with store.transaction([tmp_path / "outside.md"]):
            pass


def test_transaction_takes_its_own_lock_and_rolls_back(store):
    node = store.load_node("root")
    original = store.node_path(node.id).read_text(encoding="utf-8")
    with pytest.raises(OSError, match="stop"):
        with store.transaction([store.node_path(node.id)]):
            node.title = "Partial update"
            store.save_node(node)
            raise OSError("stop")
    assert store.node_path(node.id).read_text(encoding="utf-8") == original


def test_transaction_rejects_undeclared_store_write(store):
    node = store.add_question("A declared branch")
    another = store.add_question("An undeclared branch")
    with pytest.raises(ValidationError, match="undeclared write"):
        with store.transaction([store.node_path(node.id)]):
            store.save_node(another)


def test_nested_transactions_are_rejected(store):
    node = store.load_node("root")
    with store.transaction([store.node_path(node.id)]):
        with pytest.raises(ValidationError, match="nested"):
            with store.transaction([store.project_path]):
                pass


def test_first_transaction_migrates_an_existing_graph_without_a_journal_key(store):
    key = store.root / ".state" / "transaction.key"
    key.unlink()
    node = store.load_node("root")
    with store.transaction([store.node_path(node.id)]):
        node.title = "Migrated graph"
        store.save_node(node)
    assert key.is_file()
    assert store.load_node(node.id).title == "Migrated graph"


def test_reader_waits_for_multi_file_transaction(store):
    entered = threading.Event()
    release = threading.Event()
    observed = []

    def writer():
        with store.locked(), store.transaction([store.node_path("q_000000000001")]):
            entered.set()
            release.wait(timeout=5)

    def reader():
        entered.wait(timeout=5)
        other = GraphStore(store.root)
        observed.append(other.load_project().id)

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    reader_thread.start()
    assert entered.wait(timeout=5)
    time.sleep(0.05)
    assert observed == []
    release.set()
    writer_thread.join(timeout=5)
    reader_thread.join(timeout=5)
    assert observed == [store.load_project().id]


def test_canonical_symlink_directory_is_rejected(store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    real_nodes = store.root / "real-nodes"
    store.nodes_dir.rename(real_nodes)
    (store.root / "nodes").symlink_to(outside)
    with pytest.raises(ValidationError, match="canonical research path escapes"):
        GraphStore(store.root)
    assert list(outside.iterdir()) == []


def test_symlinked_transaction_directory_is_rejected_without_deleting_target(store, tmp_path):
    outside = tmp_path / "valuable"
    outside.mkdir()
    keep = outside / "keep.txt"
    keep.write_text("do not delete", encoding="utf-8")
    transactions = store.root / ".state" / "transactions"
    transactions.symlink_to(outside)
    with pytest.raises(ValidationError, match="symlinked directory"):
        GraphStore(store.root)
    assert keep.read_text(encoding="utf-8") == "do not delete"


def test_symlinked_lock_file_cannot_touch_an_outside_target(store, tmp_path):
    outside = tmp_path / "outside-lock"
    lock = store.root / ".state" / "lock"
    lock.unlink()
    lock.symlink_to(outside)
    with pytest.raises(ValidationError, match="symlink"):
        GraphStore(store.root)
    assert not outside.exists()
