"""Filesystem-backed graph store with safe paths, locking, and atomic writes."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

from .errors import NotFoundError, ValidationError
from .models import ModelRun, Node, Project, Source, new_id, utc_now, validate_id

PROJECT_FILE = "project.json"
CURSOR_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
NODE_REFERENCE_RE = re.compile(r"^(?:q|a|c|k|y|n)_[a-f0-9]{1,12}$")
TRANSACTION_ID_RE = re.compile(r"^[a-f0-9]{32}$")
TRANSACTION_SCHEMA_VERSION = 2

DEFAULT_SETTINGS = {
    "default_model": "~openai/gpt-latest",
    "reasoning_effort": "high",
    "web_search": True,
    "max_search_results": 8,
    "council_models": [
        "~openai/gpt-latest",
        "~anthropic/claude-sonnet-latest",
        "~google/gemini-pro-latest",
    ],
    "chairman_model": "~openai/gpt-latest",
}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes where the local filesystem supports it."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        _fsync_directory(directory)


def _json_text(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


class GraphStore:
    """Canonical Git-friendly graph rooted at one directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self._thread_lock = threading.RLock()
        self._lock_state = threading.local()
        if not (self.root / PROJECT_FILE).is_file():
            raise NotFoundError(
                f"no research tree at {self.root}; initialize it with `research-tree init`"
            )
        for relative in ("nodes", "sources", "runs", "views", ".state"):
            path = self.root / relative
            if path.is_symlink() or (path.exists() and self.root not in path.resolve().parents):
                raise ValidationError(f"canonical research path escapes the root: {path}")
        # A reader may be the first process to open the graph after an interrupted writer.
        with self.locked():
            pass

    def _active_transaction(self) -> dict[str, Any] | None:
        return getattr(self._lock_state, "transaction", None)

    def _write_text(self, path: Path, text: str) -> None:
        """Write canonical text, journaling the intended content before it can change."""
        lexical = self._contained_path(path)
        transaction = self._active_transaction()
        if transaction is None:
            _atomic_write(lexical, text)
            return
        relative = str(lexical.relative_to(self.root))
        try:
            index = transaction["path_indexes"][relative]
        except KeyError as exc:
            raise ValidationError(f"transaction attempted an undeclared write: {relative}") from exc
        intended_hash = hashlib.sha256(text.encode()).hexdigest()
        expected_hash = (
            hashlib.sha256(lexical.read_bytes()).hexdigest() if lexical.is_file() else None
        )
        marker_payload = {
            "transaction_id": transaction["transaction_id"],
            "entry": transaction["entries"][index],
            "expected_sha256": expected_hash,
            "content_sha256": intended_hash,
        }
        _atomic_write(
            transaction["directory"] / f"written-{index}.json",
            _json_text(
                {
                    "transaction_id": transaction["transaction_id"],
                    "expected_sha256": expected_hash,
                    "content_sha256": intended_hash,
                    "marker_hmac": self._journal_hmac(marker_payload),
                }
            ),
        )
        _atomic_write(lexical, text)

    @classmethod
    def create(
        cls,
        root: str | Path,
        root_question: str,
        *,
        title: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> tuple["GraphStore", Node]:
        root_path = Path(root).expanduser().resolve()
        question = root_question.strip()
        if not question:
            raise ValidationError("root question cannot be empty")
        root_path.parent.mkdir(parents=True, exist_ok=True)
        lock_name = hashlib.sha256(str(root_path).encode()).hexdigest()[:20]
        lock_dir = Path(tempfile.gettempdir()) / f"research-tree-create-{os.getuid()}"
        if lock_dir.is_symlink():
            raise ValidationError(f"refusing symlinked create-lock directory: {lock_dir}")
        lock_dir.mkdir(mode=0o700, exist_ok=True)
        lock_path = lock_dir / f"{lock_name}.lock"
        with lock_path.open("a+") as create_lock:
            fcntl.flock(create_lock.fileno(), fcntl.LOCK_EX)
            stage: Path | None = None
            try:
                if (root_path / PROJECT_FILE).exists():
                    raise ValidationError(f"a research tree already exists at {root_path}")
                existing = (
                    sorted(path.name for path in root_path.iterdir()) if root_path.exists() else []
                )
                if existing:
                    raise ValidationError(
                        f"refusing to initialize non-empty directory {root_path}: "
                        + ", ".join(existing[:8])
                    )
                if root_path.exists():
                    root_path.rmdir()

                # Build the complete graph beside its destination, then publish it with one
                # atomic rename. A hard crash can therefore leave an orphaned staging directory,
                # but never a half-created graph at the requested path.
                stage = Path(
                    tempfile.mkdtemp(
                        prefix=f".{root_path.name}.research-tree-init-",
                        dir=root_path.parent,
                    )
                )
                for directory in ("nodes", "sources", "runs", "views", ".state/cursors"):
                    (stage / directory).mkdir(parents=True, exist_ok=True)
                _atomic_write(stage / ".gitignore", ".state/\n")
                transaction_key = stage / ".state" / "transaction.key"
                _atomic_write(transaction_key, os.urandom(32).hex() + "\n")
                transaction_key.chmod(0o600)

                created = utc_now()
                root_node = Node(
                    id=new_id("question"),
                    type="question",
                    title=question,
                    status="open",
                    created_at=created,
                    updated_at=created,
                    body=f"# {question}\n\n",
                )
                project = Project(
                    id=new_id("project"),
                    title=(title or question).strip(),
                    root_question_id=root_node.id,
                    created_at=created,
                    updated_at=created,
                    settings={**DEFAULT_SETTINGS, **(settings or {})},
                )
                metadata = yaml.safe_dump(
                    root_node.frontmatter(),
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                ).rstrip()
                _atomic_write(
                    stage / "nodes" / f"{root_node.id}.md",
                    f"---\n{metadata}\n---\n\n{root_node.body.rstrip()}\n",
                )
                _atomic_write(
                    stage / ".state" / "cursors" / "default.json",
                    _json_text(
                        {
                            "cursor": "default",
                            "focus_id": root_node.id,
                            "updated_at": created,
                        }
                    ),
                )
                _atomic_write(stage / PROJECT_FILE, _json_text(project.to_dict()))
                staged_store = cls(stage)
                if (
                    staged_store.load_project().root_question_id != root_node.id
                    or staged_store.load_node(root_node.id).id != root_node.id
                    or staged_store.get_focus() != root_node.id
                ):
                    raise ValidationError("staged research tree failed initialization checks")
                _fsync_tree(stage)
                os.replace(stage, root_path)
                _fsync_directory(root_path.parent)
                stage = None
                store = cls(root_path)
                return store, root_node
            finally:
                if stage is not None:
                    shutil.rmtree(stage, ignore_errors=True)
                fcntl.flock(create_lock.fileno(), fcntl.LOCK_UN)

    @classmethod
    def discover(cls, start: str | Path | None = None) -> "GraphStore":
        configured = os.environ.get("RESEARCH_TREE_ROOT")
        if configured:
            return cls(configured)
        current = Path(start or Path.cwd()).expanduser().resolve()
        for candidate in (current, *current.parents):
            if (candidate / PROJECT_FILE).is_file():
                return cls(candidate)
            nested = candidate / "research"
            if (nested / PROJECT_FILE).is_file():
                return cls(nested)
        raise NotFoundError(
            "could not find project.json; pass `--root`, set RESEARCH_TREE_ROOT, "
            "or run inside a research tree"
        )

    @property
    def project_path(self) -> Path:
        return self._contained_path(self.root / PROJECT_FILE)

    @property
    def nodes_dir(self) -> Path:
        return self._contained_path(self.root / "nodes")

    @property
    def sources_dir(self) -> Path:
        return self._contained_path(self.root / "sources")

    @property
    def runs_dir(self) -> Path:
        return self._contained_path(self.root / "runs")

    @property
    def views_dir(self) -> Path:
        return self._contained_path(self.root / "views")

    @contextmanager
    def _file_locked(self, *, exclusive: bool) -> Iterator[None]:
        with self._thread_lock:
            depth = getattr(self._lock_state, "depth", 0)
            if depth:
                if exclusive and getattr(self._lock_state, "mode", "shared") != "exclusive":
                    raise RuntimeError("cannot upgrade a shared research lock to exclusive")
                self._lock_state.depth = depth + 1
                try:
                    yield
                finally:
                    self._lock_state.depth -= 1
                return

            state_dir = self._contained_path(self.root / ".state")
            state_dir.mkdir(parents=True, exist_ok=True)
            lock_mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            lock_path = self._contained_path(state_dir / "lock")
            if lock_path.is_symlink():
                raise ValidationError(f"research lock cannot be a symlink: {lock_path}")
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                lock_fd = os.open(lock_path, flags, 0o600)
            except OSError as exc:
                raise ValidationError(
                    f"cannot safely open research lock {lock_path}: {exc}"
                ) from exc
            with os.fdopen(lock_fd, "a+") as lock_file:
                fcntl.flock(lock_file.fileno(), lock_mode)
                self._lock_state.depth = 1
                self._lock_state.mode = "exclusive" if exclusive else "shared"
                try:
                    if exclusive:
                        self._recover_transactions()
                    yield
                finally:
                    self._lock_state.depth = 0
                    self._lock_state.mode = None
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self._file_locked(exclusive=True):
            yield

    @contextmanager
    def read_locked(self) -> Iterator[None]:
        # Graphs are small. An exclusive read lock closes the recovery check race and guarantees
        # one coherent multi-file snapshot; parallel read optimization can come later.
        with self._file_locked(exclusive=True):
            yield

    def _contained_path(self, path: Path) -> Path:
        path = Path(os.path.abspath(path.expanduser()))
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValidationError(f"canonical path escapes the research root: {path}") from exc
        current = path
        while current != self.root:
            if current.is_symlink():
                raise ValidationError(f"canonical path contains a symlink: {path}")
            parent = current.parent
            if parent == current:
                raise ValidationError(f"canonical path escapes the research root: {path}")
            current = parent
        resolved_parent = path.parent.resolve()
        if resolved_parent != self.root and self.root not in resolved_parent.parents:
            raise ValidationError(f"canonical path escapes the research root: {path}")
        return path

    def _safe_transaction_path(self, relative: str) -> Path:
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValidationError(f"invalid transaction path: {relative!r}")
        parts = Path(relative).parts
        allowed = False
        if relative == PROJECT_FILE or relative == "views/overview.md":
            allowed = True
        elif len(parts) == 2 and parts[0] == "nodes" and parts[1].endswith(".md"):
            validate_id(parts[1][:-3])
            allowed = True
        elif len(parts) == 2 and parts[0] == "sources" and parts[1].endswith(".json"):
            validate_id(parts[1][:-5], prefixes={"s"})
            allowed = True
        elif len(parts) == 2 and parts[0] == "runs" and parts[1].endswith(".json"):
            validate_id(parts[1][:-5], prefixes={"r"})
            allowed = True
        elif (
            len(parts) == 3
            and parts[:2] == (".state", "cursors")
            and parts[2].endswith(".json")
            and CURSOR_RE.fullmatch(parts[2][:-5])
        ):
            allowed = True
        if not allowed:
            raise ValidationError(f"transaction path is not canonical: {relative}")
        return self._contained_path(self.root / relative)

    def _transactions_dir(self) -> Path:
        transactions = self.root / ".state" / "transactions"
        if transactions.is_symlink():
            raise ValidationError(
                f"refusing transaction recovery through symlinked directory: {transactions}"
            )
        return self._contained_path(transactions)

    def _transaction_cleanup_dir(self) -> Path:
        cleanup = self.root / ".state" / "transaction-cleanup"
        if cleanup.is_symlink():
            raise ValidationError(
                f"refusing transaction cleanup through symlinked directory: {cleanup}"
            )
        return self._contained_path(cleanup)

    def _discard_settled_transaction(self, directory: Path) -> None:
        """Atomically remove a settled journal from recovery scope before recursive cleanup."""
        transactions = self._transactions_dir()
        cleanup = self._transaction_cleanup_dir()
        cleanup.mkdir(parents=True, exist_ok=True)
        tombstone = cleanup / directory.name
        if tombstone.exists() or tombstone.is_symlink():
            raise ValidationError(
                f"transaction cleanup target already exists: {tombstone}; journal preserved"
            )
        os.replace(directory, tombstone)
        _fsync_directory(transactions)
        _fsync_directory(cleanup)
        shutil.rmtree(tombstone)
        _fsync_directory(cleanup)

    def _remove_stale_transaction_cleanup(self) -> None:
        cleanup = self._transaction_cleanup_dir()
        if not cleanup.is_dir():
            return
        for tombstone in sorted(cleanup.iterdir()):
            if tombstone.is_symlink():
                raise ValidationError(
                    f"refusing transaction cleanup through symlinked entry: {tombstone}"
                )
            if tombstone.is_dir():
                if self.root not in tombstone.resolve().parents:
                    raise ValidationError(
                        f"transaction cleanup entry escapes the research root: {tombstone}"
                    )
                shutil.rmtree(tombstone)
            else:
                tombstone.unlink()
            _fsync_directory(cleanup)

    def _transaction_key(self, *, create: bool) -> bytes:
        path = self._contained_path(self.root / ".state" / "transaction.key")
        if path.is_symlink():
            raise ValidationError(f"transaction key cannot be a symlink: {path}")
        if not path.exists():
            if not create:
                raise ValidationError(
                    "transaction key is missing; pending journal preserved for manual recovery"
                )
            _atomic_write(path, os.urandom(32).hex() + "\n")
            path.chmod(0o600)
        try:
            key_text = path.read_text(encoding="ascii").strip()
        except OSError as exc:
            raise ValidationError(f"cannot read transaction key: {exc}") from exc
        if not re.fullmatch(r"[a-f0-9]{64}", key_text):
            raise ValidationError("transaction key is malformed")
        return bytes.fromhex(key_text)

    def _journal_hmac(self, payload: Any, *, create_key: bool = False) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hmac.new(
            self._transaction_key(create=create_key), encoded, hashlib.sha256
        ).hexdigest()

    def _transaction_manifest_hmac(self, transaction_id: str, entries: list[dict[str, Any]]) -> str:
        return self._journal_hmac({"transaction_id": transaction_id, "paths": entries})

    def _transaction_commit_hmac(self, transaction_id: str) -> str:
        return self._journal_hmac({"transaction_id": transaction_id, "status": "committed"})

    def _validated_transaction_manifest(
        self, directory: Path, manifest: Any
    ) -> list[dict[str, Any]]:
        if not isinstance(manifest, dict):
            raise ValidationError(f"transaction {directory.name} manifest is not an object")
        if manifest.get("schema_version") != TRANSACTION_SCHEMA_VERSION:
            raise ValidationError(f"transaction {directory.name} has an unsupported schema")
        transaction_id = manifest.get("transaction_id")
        if (
            not isinstance(transaction_id, str)
            or transaction_id != directory.name
            or not TRANSACTION_ID_RE.fullmatch(transaction_id)
        ):
            raise ValidationError(f"transaction {directory.name} has an invalid identity")
        entries = manifest.get("paths")
        if not isinstance(entries, list) or not entries:
            raise ValidationError(f"transaction {directory.name} has no valid path list")
        manifest_hmac = manifest.get("manifest_hmac")
        if not isinstance(manifest_hmac, str) or not re.fullmatch(r"[a-f0-9]{64}", manifest_hmac):
            raise ValidationError(f"transaction {directory.name} has no valid manifest HMAC")
        seen: set[str] = set()
        validated: list[dict[str, Any]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValidationError(f"transaction {directory.name} path {index} is not an object")
            relative = entry.get("relative")
            existed = entry.get("existed")
            backup_hash = entry.get("backup_sha256")
            if not isinstance(relative, str) or not isinstance(existed, bool):
                raise ValidationError(
                    f"transaction {directory.name} path {index} has invalid fields"
                )
            self._safe_transaction_path(relative)
            if relative in seen:
                raise ValidationError(f"transaction {directory.name} repeats path {relative}")
            seen.add(relative)
            if existed:
                if not isinstance(backup_hash, str) or not re.fullmatch(
                    r"[a-f0-9]{64}", backup_hash
                ):
                    raise ValidationError(
                        f"transaction {directory.name} path {index} has no backup hash"
                    )
            elif backup_hash is not None:
                raise ValidationError(
                    f"transaction {directory.name} new path {index} has a backup hash"
                )
            validated.append(entry)
        expected_hmac = self._transaction_manifest_hmac(transaction_id, validated)
        if not hmac.compare_digest(manifest_hmac, expected_hmac):
            raise ValidationError(f"transaction {directory.name} manifest failed authentication")
        return validated

    def _validated_written_indexes(
        self, directory: Path, transaction_id: str, entries: list[dict[str, Any]]
    ) -> set[int]:
        written: set[int] = set()
        for marker in directory.iterdir():
            if not marker.name.startswith("written-"):
                continue
            if marker.is_symlink() or not marker.is_file():
                raise ValidationError(
                    f"transaction {directory.name} has an invalid mutation marker"
                )
            match = re.fullmatch(r"written-(\d+)\.json", marker.name)
            if not match:
                raise ValidationError(
                    f"transaction {directory.name} has an invalid mutation marker name"
                )
            index = int(match.group(1))
            if index >= len(entries) or index in written:
                raise ValidationError(
                    f"transaction {directory.name} has an invalid mutation marker index"
                )
            try:
                data = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValidationError(
                    f"transaction {directory.name} has a bad mutation marker"
                ) from exc
            expected_hash = data.get("expected_sha256") if isinstance(data, dict) else None
            content_hash_value = data.get("content_sha256") if isinstance(data, dict) else None
            marker_payload = {
                "transaction_id": transaction_id,
                "entry": entries[index],
                "expected_sha256": expected_hash,
                "content_sha256": content_hash_value,
            }
            expected_marker_hmac = self._journal_hmac(marker_payload)
            if (
                not isinstance(data, dict)
                or data.get("transaction_id") != transaction_id
                or (
                    expected_hash is not None
                    and (
                        not isinstance(expected_hash, str)
                        or not re.fullmatch(r"[a-f0-9]{64}", expected_hash)
                    )
                )
                or not isinstance(content_hash_value, str)
                or not re.fullmatch(r"[a-f0-9]{64}", content_hash_value)
                or not isinstance(data.get("marker_hmac"), str)
                or not hmac.compare_digest(data["marker_hmac"], expected_marker_hmac)
            ):
                raise ValidationError(
                    f"transaction {directory.name} mutation marker failed authentication"
                )
            path = self._safe_transaction_path(entries[index]["relative"])
            if not path.exists():
                if not entries[index]["existed"]:
                    # Either the write never landed or an earlier rollback already deleted it.
                    continue
                # The process stopped after removing an existing path, which tracked writes never
                # do, or the canonical file disappeared independently. Preserve the journal.
                if expected_hash is not None:
                    raise ValidationError(
                        f"transaction {directory.name} lost an existing marked path; "
                        "journal preserved"
                    )
                continue
            try:
                current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise ValidationError(
                    f"transaction {directory.name} marked path is unreadable; journal preserved"
                ) from exc
            if entries[index]["existed"] and current_hash == entries[index]["backup_sha256"]:
                # The original snapshot is already present: the write did not land, or rollback
                # completed before the process died.
                continue
            if current_hash == content_hash_value:
                written.add(index)
                continue
            if expected_hash is not None and current_hash == expected_hash:
                # The process stopped after arming this write but before replacing it.
                original_hash = (
                    entries[index]["backup_sha256"] if entries[index]["existed"] else None
                )
                if expected_hash != original_hash:
                    # An earlier write in this same transaction did land, so the original
                    # snapshot still needs to be restored.
                    written.add(index)
                continue
            raise ValidationError(
                f"transaction {directory.name} marked path failed integrity check; "
                "journal preserved"
            )
        return written

    def _rollback_transaction(self, directory: Path, manifest: dict[str, Any]) -> None:
        entries = self._validated_transaction_manifest(directory, manifest)
        written = self._validated_written_indexes(directory, manifest["transaction_id"], entries)
        backups: dict[int, str] = {}
        # Validate the complete rollback plan before changing any canonical file. If one
        # backup or marker is damaged, recovery preserves the whole journal and graph state.
        for index, entry in enumerate(entries):
            if index not in written or not entry["existed"]:
                continue
            backup = directory / f"backup-{index}"
            if backup.is_symlink() or not backup.is_file():
                raise ValidationError(
                    f"transaction {directory.name} is missing backup-{index}; "
                    "journal preserved for manual recovery"
                )
            try:
                backup_bytes = backup.read_bytes()
            except OSError as exc:
                raise ValidationError(
                    f"cannot read transaction backup {backup}: {exc}; journal preserved"
                ) from exc
            if hashlib.sha256(backup_bytes).hexdigest() != entry["backup_sha256"]:
                raise ValidationError(
                    f"transaction {directory.name} backup-{index} failed integrity check; "
                    "journal preserved for manual recovery"
                )
            try:
                backups[index] = backup_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValidationError(
                    f"transaction {directory.name} backup-{index} is not UTF-8; "
                    "journal preserved for manual recovery"
                ) from exc
        for index, entry in reversed(list(enumerate(entries))):
            if index not in written:
                continue
            path = self._safe_transaction_path(entry["relative"])
            if entry["existed"]:
                _atomic_write(path, backups[index])
            else:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                else:
                    _fsync_directory(path.parent)

    def _recover_transactions(self) -> None:
        self._remove_stale_transaction_cleanup()
        transactions = self._transactions_dir()
        if transactions.exists():
            resolved = transactions.resolve()
            if self.root not in resolved.parents:
                raise ValidationError(
                    f"transaction directory escapes the research root: {resolved}"
                )
        if not transactions.is_dir():
            return
        for directory in sorted(transactions.iterdir()):
            if directory.is_symlink():
                raise ValidationError(
                    f"refusing transaction recovery through symlinked entry: {directory}"
                )
            if not directory.is_dir():
                continue
            if self.root not in directory.resolve().parents:
                raise ValidationError(f"transaction entry escapes the research root: {directory}")
            manifest_path = directory / "manifest.json"
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise ValidationError(
                    f"transaction {directory.name} has no trustworthy manifest; journal preserved"
                )
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValidationError(
                    f"cannot recover transaction {directory.name}: {exc}"
                ) from exc
            self._validated_transaction_manifest(directory, manifest)
            committed = directory / "committed.json"
            if committed.exists():
                if committed.is_symlink() or not committed.is_file():
                    raise ValidationError(
                        f"transaction {directory.name} has an invalid commit marker"
                    )
                try:
                    committed_data = json.loads(committed.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValidationError(
                        f"transaction {directory.name} has an invalid commit marker"
                    ) from exc
                if not isinstance(committed_data, dict):
                    raise ValidationError(
                        f"transaction {directory.name} commit marker is malformed"
                    )
                expected_commit_hmac = self._transaction_commit_hmac(directory.name)
                if (
                    committed_data.get("transaction_id") != directory.name
                    or committed_data.get("status") != "committed"
                    or not isinstance(committed_data.get("commit_hmac"), str)
                    or not hmac.compare_digest(committed_data["commit_hmac"], expected_commit_hmac)
                ):
                    raise ValidationError(
                        f"transaction {directory.name} commit marker does not match"
                    )
            else:
                self._rollback_transaction(directory, manifest)
            self._discard_settled_transaction(directory)

    @contextmanager
    def transaction(self, paths: list[Path]) -> Iterator[None]:
        """Make a multi-file mutation crash-recoverable under one exclusive graph lock."""
        with self.locked():
            if self._active_transaction() is not None:
                raise ValidationError("nested research transactions are not supported")
            unique: dict[str, Path] = {}
            for candidate in paths:
                lexical = self._contained_path(candidate)
                relative = str(lexical.relative_to(self.root))
                self._safe_transaction_path(relative)
                unique[relative] = lexical
            if not unique:
                raise ValidationError("a transaction needs at least one canonical path")

            transaction_id = uuid.uuid4().hex
            transactions = self._transactions_dir()
            transactions.mkdir(parents=True, exist_ok=True)
            # Existing v0.1 graphs predate the private journal key. It is safe to create one
            # here, before publishing a journal; recovery never creates a missing key.
            self._transaction_key(create=True)
            preparing = Path(
                tempfile.mkdtemp(prefix=".transaction-preparing-", dir=self.root / ".state")
            )
            directory = transactions / transaction_id
            try:
                entries: list[dict[str, Any]] = []
                for relative, path in sorted(unique.items()):
                    existed = path.is_file()
                    backup_hash = None
                    if existed:
                        backup_bytes = path.read_bytes()
                        backup_hash = hashlib.sha256(backup_bytes).hexdigest()
                    entries.append(
                        {
                            "relative": relative,
                            "existed": existed,
                            "backup_sha256": backup_hash,
                        }
                    )
                manifest = {
                    "schema_version": TRANSACTION_SCHEMA_VERSION,
                    "transaction_id": transaction_id,
                    "paths": entries,
                    "manifest_hmac": self._transaction_manifest_hmac(transaction_id, entries),
                }
                for index, entry in enumerate(entries):
                    if entry["existed"]:
                        _atomic_write(
                            preparing / f"backup-{index}",
                            self._safe_transaction_path(entry["relative"]).read_text(
                                encoding="utf-8"
                            ),
                        )
                _atomic_write(preparing / "manifest.json", _json_text(manifest))
                os.replace(preparing, directory)
                _fsync_directory(transactions)
                preparing = None
                self._lock_state.transaction = {
                    "transaction_id": transaction_id,
                    "directory": directory,
                    "path_indexes": {
                        entry["relative"]: index for index, entry in enumerate(entries)
                    },
                    "entries": entries,
                }
                try:
                    yield
                except BaseException:
                    self._rollback_transaction(directory, manifest)
                    self._discard_settled_transaction(directory)
                    raise
                else:
                    _atomic_write(
                        directory / "committed.json",
                        _json_text(
                            {
                                "transaction_id": transaction_id,
                                "status": "committed",
                                "commit_hmac": self._transaction_commit_hmac(transaction_id),
                            }
                        ),
                    )
                    self._discard_settled_transaction(directory)
            finally:
                self._lock_state.transaction = None
                if preparing is not None:
                    shutil.rmtree(preparing, ignore_errors=True)

    def load_project(self) -> Project:
        with self.read_locked():
            try:
                data = json.loads(self.project_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValidationError(f"cannot read {self.project_path}: {exc}") from exc
            return Project.from_dict(data)

    def save_project(self, project: Project) -> None:
        with self.locked():
            project.updated_at = utc_now()
            self._write_text(self.project_path, _json_text(project.to_dict()))

    def node_path(self, node_id: str) -> Path:
        validate_id(node_id)
        return self._contained_path(self.nodes_dir / f"{node_id}.md")

    def source_path(self, source_id: str) -> Path:
        validate_id(source_id, prefixes={"s"})
        return self._contained_path(self.sources_dir / f"{source_id}.json")

    def run_path(self, run_id: str) -> Path:
        validate_id(run_id, prefixes={"r"})
        return self._contained_path(self.runs_dir / f"{run_id}.json")

    def save_node(self, node: Node) -> None:
        with self.locked():
            node.validate()
            metadata = yaml.safe_dump(
                node.frontmatter(), sort_keys=False, allow_unicode=True, default_flow_style=False
            ).rstrip()
            body = node.body.rstrip() + "\n" if node.body.strip() else ""
            self._write_text(self.node_path(node.id), f"---\n{metadata}\n---\n\n{body}")

    def load_node(self, reference: str) -> Node:
        with self.read_locked():
            node_id = self.resolve_node_id(reference)
            path = self.node_path(node_id)
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise NotFoundError(f"node not found: {reference}") from exc
            if not text.startswith("---\n"):
                raise ValidationError(f"node {node_id} is missing YAML frontmatter")
            try:
                metadata_text, body = text[4:].split("\n---\n", 1)
                metadata = yaml.safe_load(metadata_text)
            except (ValueError, yaml.YAMLError) as exc:
                raise ValidationError(f"cannot parse node {node_id}: {exc}") from exc
            if not isinstance(metadata, dict):
                raise ValidationError(f"node {node_id} frontmatter is not a mapping")
            node = Node.from_parts(metadata, body.lstrip("\n"))
            if node.id != node_id:
                raise ValidationError(
                    f"node filename {node_id}.md does not match metadata ID {node.id}"
                )
            return node

    def list_nodes(self, *, node_type: str | None = None) -> list[Node]:
        with self.read_locked():
            nodes: list[Node] = []
            for path in sorted(self.nodes_dir.glob("*.md")):
                node = self.load_node(path.stem)
                if node_type is None or node.type == node_type:
                    nodes.append(node)
            return nodes

    def resolve_node_id(self, reference: str, *, cursor: str = "default") -> str:
        with self.read_locked():
            ref = reference.strip()
            if ref in {"focus", "current", "."}:
                return self.get_focus(cursor)
            if ref == "root":
                return self.load_project().root_question_id
            if not NODE_REFERENCE_RE.fullmatch(ref):
                raise ValidationError(
                    f"invalid node reference {reference!r}; use a node ID or unambiguous ID prefix"
                )
            exact = self.nodes_dir / f"{ref}.md"
            if exact.is_file():
                validate_id(ref)
                return ref
            matches = [path.stem for path in self.nodes_dir.glob(f"{ref}*.md")]
            if not matches:
                raise NotFoundError(f"node not found: {reference}")
            if len(matches) > 1:
                raise ValidationError(
                    f"ambiguous node prefix {reference!r}: {', '.join(sorted(matches)[:6])}"
                )
            return matches[0]

    def save_source(self, source: Source) -> None:
        with self.locked():
            source.validate()
            path = self.source_path(source.id)
            if path.exists():  # source snapshots are immutable
                existing = self.load_source(source.id)
                if (existing.url, existing.excerpt, existing.content_hash) != (
                    source.url,
                    source.excerpt,
                    source.content_hash,
                ):
                    raise ValidationError(f"source snapshot ID collision: {source.id}")
                return
            self._write_text(path, _json_text(source.to_dict()))

    def load_source(self, source_id: str) -> Source:
        with self.read_locked():
            try:
                data = json.loads(self.source_path(source_id).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise NotFoundError(f"source not found or unreadable: {source_id}") from exc
            source = Source.from_dict(data)
            if source.id != source_id:
                raise ValidationError(
                    f"source filename {source_id}.json does not match object ID {source.id}"
                )
            return source

    def list_sources(self) -> list[Source]:
        with self.read_locked():
            result = []
            for path in sorted(self.sources_dir.glob("*.json")):
                result.append(self.load_source(path.stem))
            return result

    def save_run(self, run: ModelRun) -> None:
        with self.locked():
            run.validate()
            path = self.run_path(run.id)
            if path.exists():
                raise ValidationError(f"model run is immutable and already exists: {run.id}")
            self._write_text(path, _json_text(run.to_dict()))

    def load_run(self, run_id: str) -> ModelRun:
        with self.read_locked():
            try:
                data = json.loads(self.run_path(run_id).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise NotFoundError(f"run not found or unreadable: {run_id}") from exc
            run = ModelRun.from_dict(data)
            if run.id != run_id:
                raise ValidationError(
                    f"run filename {run_id}.json does not match object ID {run.id}"
                )
            return run

    def cursor_path(self, cursor: str) -> Path:
        if not CURSOR_RE.fullmatch(cursor):
            raise ValidationError(
                "cursor names may contain only letters, digits, dot, dash, underscore"
            )
        return self._contained_path(self.root / ".state" / "cursors" / f"{cursor}.json")

    def set_focus(self, node_reference: str, *, cursor: str = "default") -> Node:
        with self.locked():
            node = (
                self.load_node(node_reference)
                if (self.nodes_dir / f"{node_reference}.md").exists()
                else None
            )
            if node is None:
                # During initialization, save_node has already run; otherwise resolve short refs.
                node = self.load_node(self.resolve_node_id(node_reference, cursor=cursor))
            data = {"cursor": cursor, "focus_id": node.id, "updated_at": utc_now()}
            self._write_text(self.cursor_path(cursor), _json_text(data))
            return node

    def get_focus(self, cursor: str = "default") -> str:
        with self.read_locked():
            path = self.cursor_path(cursor)
            if not path.exists():
                return self.load_project().root_question_id
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                focus_id = data["focus_id"]
                self.node_path(focus_id)
            except (OSError, json.JSONDecodeError, KeyError, ValidationError) as exc:
                raise ValidationError(f"invalid cursor {cursor!r}: {exc}") from exc
            if not self.node_path(focus_id).is_file():
                raise ValidationError(f"cursor {cursor!r} points to missing node {focus_id}")
            return focus_id

    def add_question(
        self,
        title: str,
        *,
        parent: str = "focus",
        status: str = "open",
        body: str = "",
        priority: int | None = None,
        cursor: str = "default",
        focus: bool = True,
    ) -> Node:
        with self.locked():
            question = title.strip()
            if not question:
                raise ValidationError("question cannot be empty")
            if priority is not None and not 1 <= priority <= 5:
                raise ValidationError("question priority must be between 1 and 5")
            parent_node = self.load_node(self.resolve_node_id(parent, cursor=cursor))
            if parent_node.type != "question":
                if parent_node.question_id:
                    parent_node = self.load_node(parent_node.question_id)
                else:
                    raise ValidationError(
                        f"cannot branch from {parent_node.type} node {parent_node.id}"
                    )
            created = utc_now()
            node = Node(
                id=new_id("question"),
                type="question",
                title=question,
                status=status,
                created_at=created,
                updated_at=created,
                parent_id=parent_node.id,
                tags=[f"priority-{priority}"] if priority is not None else [],
                body=body or f"# {question}\n\n",
            )
            self.save_node(node)
            if focus:
                self.set_focus(node.id, cursor=cursor)
            return node

    def update_node(self, node: Node) -> None:
        with self.locked():
            if not self.node_path(node.id).is_file():
                raise NotFoundError(f"cannot update missing node {node.id}")
            node.updated_at = utc_now()
            self.save_node(node)


def load_store(root: str | Path | None) -> GraphStore:
    return GraphStore(root) if root else GraphStore.discover()
