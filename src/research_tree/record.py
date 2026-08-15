"""Interactive recorder: grow a research tree with your own questions and answers.

`research-tree record` is the free, offline path for recording research by hand:
type a question (it becomes a branch) and then paste or edit an answer. The
model-powered workflows (`ask`, `verify`, `council`, `synthesize`) can be mixed
into the same tree later, or a tree can stay entirely hand-recorded.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .errors import ConfigurationError, ResearchTreeError
from .render import STATUS_LEGEND, frontier, render_node, render_tree, render_where
from .research import record_manual_answer
from .store import GraphStore

ANSWER_TERMINATOR = "."
ANSWER_CANCEL = ":cancel"


def _resolve_editor() -> list[str]:
    editor = os.environ.get("RESEARCH_EDITOR") or os.environ.get("EDITOR")
    if not editor:
        editor = next(
            (name for name in ("cursor", "code", "vim", "nano") if shutil.which(name)), None
        )
    if not editor:
        raise ConfigurationError("no editor found; set RESEARCH_EDITOR or EDITOR")
    return [*editor.split()]


def edit_answer_text(*, hint: str = "") -> str:
    """Open $EDITOR on a scratch file and return its trimmed contents."""
    fd, path = tempfile.mkstemp(prefix="research-tree-answer-", suffix=".md")
    os.close(fd)
    scratch = Path(path)
    try:
        if hint:
            scratch.write_text(hint, encoding="utf-8")
        code = subprocess.call([*_resolve_editor(), str(scratch)])
        if code != 0:
            raise ConfigurationError(f"editor exited with code {code}")
        return scratch.read_text(encoding="utf-8").strip()
    finally:
        scratch.unlink(missing_ok=True)


def prompt_answer_text(*, prompt: str = "") -> str:
    """Collect a multi-line answer from the terminal.

    Paste freely; finish with a line containing only `.`, Ctrl-D (EOF), or write
    `:cancel` to abort. Returns "" when cancelled.
    """
    if not prompt:
        prompt = (
            "Record an answer (paste text; finish with a line containing only '.', "
            "Ctrl-D, or ':cancel' to abort):"
        )
    print(prompt, file=sys.stderr)
    lines: list[str] = []
    try:
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == ANSWER_CANCEL:
                return ""
            if line.strip() == ANSWER_TERMINATOR:
                break
            lines.append(line)
    except KeyboardInterrupt:
        return ""
    return "\n".join(lines).strip()


def read_answer_text(
    *,
    text: str | None = None,
    file: str | None = None,
    edit: bool = False,
) -> str:
    """Resolve answer text from the strongest source the caller offered.

    Priority: --text, --file, --edit, piped stdin, interactive prompt.
    """
    if text is not None:
        return text.strip()
    if file:
        return Path(file).expanduser().read_text(encoding="utf-8").strip()
    if edit:
        return edit_answer_text()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return prompt_answer_text()


def _branch(store: GraphStore, text: str, *, cursor: str) -> None:
    question = store.add_question(text, parent="focus", status="open", cursor=cursor, focus=True)
    print(f"Branched to {question.id} — {question.title}", file=sys.stderr)
    print("Type `a` to record an answer.", file=sys.stderr)


def _record_answer(store: GraphStore, *, cursor: str) -> None:
    focus_id = store.get_focus(cursor)
    question = store.load_node(focus_id)
    if question.type != "question":
        print(
            f"Focus {question.id} is a {question.type} node, not a question.",
            file=sys.stderr,
        )
        return
    text = prompt_answer_text()
    if not text:
        print("No answer recorded.", file=sys.stderr)
        return
    answer = record_manual_answer(store, focus_id, text, cursor=cursor)
    print(f"Saved {answer.id} for {question.id} — {question.title}", file=sys.stderr)


def _edit_focus(store: GraphStore, *, cursor: str) -> None:
    focus_id = store.get_focus(cursor)
    code = subprocess.call([*_resolve_editor(), str(store.node_path(focus_id))])
    if code != 0:
        raise ConfigurationError(f"editor exited with code {code}")


def _print_children(store: GraphStore, *, cursor: str) -> None:
    focus_id = store.get_focus(cursor)
    children = sorted(
        [
            node
            for node in store.list_nodes()
            if node.type == "question" and node.parent_id == focus_id
        ],
        key=lambda node: (node.created_at, node.id),
    )
    if not children:
        print(f"No child questions under {focus_id}.", file=sys.stderr)
        return
    for node in children:
        print(f"{node.id}  {node.title}", file=sys.stderr)


def _print_next(store: GraphStore, *, cursor: str) -> None:
    candidates = frontier(store, start="root", cursor=cursor)[:8]
    if not candidates:
        print("No unanswered questions in the tree.", file=sys.stderr)
        return
    for index, node in enumerate(candidates, 1):
        print(f"{index}. {node.id}  {node.title}", file=sys.stderr)
    print("Focus one with `f <id>`.", file=sys.stderr)


_HELP = """\
Commands:
  <text>         branch a new question under focus (and focus it)
  a              record an answer for the focused question
  e              edit the focused node in $EDITOR
  f <ref>        focus a node (ID, prefix, root, ..)
  ..             move focus to the parent question
  s <ref>        show a node's content (default: focus)
  t              render the question tree
  w              show the current location
  c              list child questions of the focus
  n              list unanswered questions (whole tree)
  help           this help
  quit, exit     leave record mode (the tree is saved as you go)"""


def record_repl(store: GraphStore, *, cursor: str = "default") -> int:
    """Run an interactive record-questions-and-answers session."""
    print(
        "Research Tree recorder — type a question to branch it, `a` to record an\n"
        "answer, `help` for all commands, `quit` to stop.",
        file=sys.stderr,
    )
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            print(file=sys.stderr)
            return 0
        except KeyboardInterrupt:
            print(file=sys.stderr)
            return 0
        if not line:
            continue
        try:
            if line in {"quit", "exit", "done"}:
                return 0
            if line in {"h", "help"}:
                print(_HELP)
            elif line in {"t", "tree"}:
                print(render_tree(store, cursor=cursor) + "\n\n" + STATUS_LEGEND)
            elif line in {"w", "where"}:
                print(render_where(store, cursor=cursor))
            elif line in {"c", "children"}:
                _print_children(store, cursor=cursor)
            elif line in {"n", "next"}:
                _print_next(store, cursor=cursor)
            elif line in {"..", "up"}:
                node = store.set_focus("..", cursor=cursor)
                print(f"Focused {cursor} on {node.id} — {node.title}", file=sys.stderr)
            elif line in {"a", "ans", "answer"}:
                _record_answer(store, cursor=cursor)
            elif line in {"e", "edit"}:
                _edit_focus(store, cursor=cursor)
            elif line in {"s", "show"}:
                focus_id = store.get_focus(cursor)
                print(render_node(store, store.load_node(focus_id)))
            elif line.startswith(("f ", "focus ")):
                node = store.set_focus(line.split(maxsplit=1)[1], cursor=cursor)
                print(f"Focused {cursor} on {node.id} — {node.title}", file=sys.stderr)
            elif line.startswith(("s ", "show ")):
                node_id = store.resolve_node_id(line.split(maxsplit=1)[1], cursor=cursor)
                print(render_node(store, store.load_node(node_id)))
            else:
                _branch(store, line, cursor=cursor)
        except (ResearchTreeError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
