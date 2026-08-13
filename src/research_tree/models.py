"""Versioned data models for the canonical Markdown/JSON graph."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .errors import ValidationError

SCHEMA_VERSION = 1
NODE_TYPES = {"question", "answer", "claim", "concept", "synthesis", "note"}
NODE_STATUSES = {
    "proposed",
    "open",
    "researching",
    "answered",
    "uncertain",
    "contested",
    "parked",
}
EDGE_TYPES = {
    "answers",
    "supports",
    "contradicts",
    "depends_on",
    "explains",
    "related_to",
    "derived_from",
    "supersedes",
}
RUN_MODES = {"ask", "council", "verify"}
ID_RE = re.compile(r"^(p|q|a|c|k|y|n|s|r)_[a-f0-9]{12}$")

PREFIXES = {
    "project": "p",
    "question": "q",
    "answer": "a",
    "claim": "c",
    "concept": "k",
    "synthesis": "y",
    "note": "n",
    "source": "s",
    "run": "r",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def new_id(kind: str) -> str:
    try:
        prefix = PREFIXES[kind]
    except KeyError as exc:
        raise ValidationError(f"unknown ID kind: {kind}") from exc
    return f"{prefix}_{secrets.token_hex(6)}"


def stable_source_id(url: str, content: str = "") -> str:
    digest = hashlib.sha256((url.strip() + "\n" + content.strip()).encode()).hexdigest()[:12]
    return f"s_{digest}"


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def validate_id(value: str, *, prefixes: set[str] | None = None) -> str:
    if not ID_RE.fullmatch(value):
        raise ValidationError(f"invalid object ID: {value!r}")
    if prefixes and value[0] not in prefixes:
        expected = ", ".join(sorted(prefixes))
        raise ValidationError(f"ID {value!r} has the wrong type; expected prefix: {expected}")
    return value


@dataclass
class Edge:
    type: str
    target: str
    note: str = ""

    def validate(self) -> None:
        if self.type not in EDGE_TYPES:
            raise ValidationError(f"unknown edge type: {self.type}")
        validate_id(self.target)


@dataclass
class Node:
    id: str
    type: str
    title: str
    status: str
    created_at: str
    updated_at: str
    body: str = ""
    parent_id: str | None = None
    question_id: str | None = None
    edges: list[Edge] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    confidence: float | None = None
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(
                f"node {self.id} uses schema {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        validate_id(self.id, prefixes={PREFIXES[self.type]} if self.type in PREFIXES else None)
        if self.type not in NODE_TYPES:
            raise ValidationError(f"unknown node type: {self.type}")
        if not self.title.strip():
            raise ValidationError(f"node {self.id} has an empty title")
        if self.status not in NODE_STATUSES:
            raise ValidationError(f"unknown node status: {self.status}")
        if self.parent_id:
            validate_id(self.parent_id, prefixes={"q"})
            if self.type != "question":
                raise ValidationError("only question nodes can have parent_id")
        if self.question_id:
            validate_id(self.question_id, prefixes={"q"})
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValidationError("confidence must be between 0 and 1")
        for edge in self.edges:
            edge.validate()
        for source_id in self.source_ids:
            validate_id(source_id, prefixes={"s"})
        for run_id in self.run_ids:
            validate_id(run_id, prefixes={"r"})

    def frontmatter(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.parent_id:
            data["parent_id"] = self.parent_id
        if self.question_id:
            data["question_id"] = self.question_id
        if self.edges:
            data["edges"] = [asdict(edge) for edge in self.edges]
        if self.source_ids:
            data["source_ids"] = self.source_ids
        if self.run_ids:
            data["run_ids"] = self.run_ids
        if self.tags:
            data["tags"] = self.tags
        if self.confidence is not None:
            data["confidence"] = self.confidence
        return data

    @classmethod
    def from_parts(cls, data: dict[str, Any], body: str) -> "Node":
        try:
            edges = [Edge(**edge) for edge in data.get("edges", [])]
            node = cls(
                id=data["id"],
                type=data["type"],
                title=data["title"],
                status=data["status"],
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                body=body,
                parent_id=data.get("parent_id"),
                question_id=data.get("question_id"),
                edges=edges,
                source_ids=list(data.get("source_ids", [])),
                run_ids=list(data.get("run_ids", [])),
                tags=list(data.get("tags", [])),
                confidence=data.get("confidence"),
                schema_version=int(data.get("schema_version", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed node metadata: {exc}") from exc
        node.validate()
        return node


@dataclass
class Project:
    id: str
    title: str
    root_question_id: str
    created_at: str
    updated_at: str
    settings: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        validate_id(self.id, prefixes={"p"})
        validate_id(self.root_question_id, prefixes={"q"})
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(
                f"project uses schema {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if not self.title.strip():
            raise ValidationError("project title cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        try:
            project = cls(**data)
        except TypeError as exc:
            raise ValidationError(f"malformed project.json: {exc}") from exc
        project.validate()
        return project


@dataclass
class Source:
    id: str
    url: str
    title: str
    retrieved_at: str
    content_hash: str
    excerpt: str = ""
    source_type: str = "web"
    published_at: str | None = None
    authors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        validate_id(self.id, prefixes={"s"})
        if not self.url.startswith(("http://", "https://", "file:")):
            raise ValidationError(f"source URL is not supported: {self.url}")
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(f"source {self.id} uses an unsupported schema")
        if self.content_hash != content_hash(self.excerpt):
            raise ValidationError(f"source {self.id} excerpt does not match its content hash")
        if self.id != stable_source_id(self.url, self.excerpt):
            raise ValidationError(f"source {self.id} does not match its URL/excerpt snapshot ID")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Source":
        try:
            source = cls(**data)
        except TypeError as exc:
            raise ValidationError(f"malformed source: {exc}") from exc
        source.validate()
        return source


@dataclass
class ModelRun:
    id: str
    mode: str
    question_id: str
    created_at: str
    provider: str
    requested_models: list[str]
    resolved_models: list[str]
    prompt_hash: str
    response_node_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        validate_id(self.id, prefixes={"r"})
        validate_id(self.question_id, prefixes={"q"})
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(f"run {self.id} uses an unsupported schema")
        if self.mode not in RUN_MODES:
            raise ValidationError(f"run {self.id} has unknown mode: {self.mode}")
        if not self.provider.strip():
            raise ValidationError(f"run {self.id} has no provider")
        if len(self.requested_models) != len(self.resolved_models):
            raise ValidationError(
                f"run {self.id} requested/resolved model lists have different lengths"
            )
        for node_id in self.response_node_ids:
            validate_id(node_id)
        for source_id in self.source_ids:
            validate_id(source_id, prefixes={"s"})

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRun":
        try:
            run = cls(**data)
        except TypeError as exc:
            raise ValidationError(f"malformed model run: {exc}") from exc
        run.validate()
        return run


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
