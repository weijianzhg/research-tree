# Research Tree

Research Tree is a terminal-first, Git-native workspace for branching research. It keeps the
question you are exploring, the paths you have opened, the answers you have received, the claims
those answers make, their evidence, and the exact model runs that produced them.

It is designed to sit underneath an agent such as [Pi](https://pi.dev/).
The agent is the conversational interface; Research Tree is the durable, inspectable memory.

## Why a tree *and* a graph?

Questions have a natural hierarchy: a root problem breaks into follow-up questions. That hierarchy
is what `tree`, `where`, and `next` show. Research itself is not a strict tree, though: one source can
support several claims, claims can contradict one another, and answers can inform several branches.
Research Tree therefore stores a typed property graph and treats question parentage as one useful
view of it.

```text
Pi conversation tree                 Research Tree
(interaction history)                (durable semantic state)

message → fork → message             question → child question
                                          ↓
                                      answer → claim → source
                                          ↘ contradiction ↗
```

## Install

Python 3.10 or newer is required.

```bash
uv tool install --editable /path/to/research-tree
# or: pipx install --editable /path/to/research-tree
```

Research calls use OpenRouter. Credentials are resolved without copying them into a project, in
this order:

1. `OPENROUTER_API_KEY`
2. `~/.config/research-tree/config.json`
3. the active OpenRouter login already held by Pi
4. an existing `~/.fluff-cutter/config.yaml`

All graph-management commands work offline.

## First research session

```bash
research-tree init ./research \
  "What makes sparse mixture-of-experts models efficient?" \
  --title "Understanding MoE"

research-tree --root ./research branch \
  "How does top-k expert routing work?" --priority 1

research-tree --root ./research where
research-tree --root ./research tree
research-tree --root ./research ask focus
research-tree --root ./research verify a_abc123 --model '~anthropic/claude-sonnet-4.5:latest'
research-tree --root ./research next --from root
```

`ask` uses high reasoning and model-controlled web search by default. It stores the answer, atomic
claims, citation snapshots, uncertainties, suggested branches, usage/cost data, and the raw provider
response. Use `--no-web`, `--effort`, or `--model` to override a run. `verify` then freezes the
captured excerpts and asks a verifier—ideally a different model—whether each excerpt actually
entails its claim. It records source authority separately, so an aggregator repeating a number is
not mistaken for a primary benchmark artifact. Verification can mark claims supported, partial,
unsupported, contradicted, or unknown without rewriting the original answer.

For questions where disagreement matters:

```bash
research-tree --root ./research council focus \
  --model '~openai/gpt-5.6-sol-pro' \
  --model '~anthropic/claude-opus-5' \
  --model '~x-ai/grok-4.6'
```

Council mode runs independent evidence searches, anonymized peer reviews, and a chairman synthesis.
It preserves each model's answer and the minority views; consensus is recorded as a signal, not
treated as truth. A three-model council makes seven paid completions (three answers, three reviews,
one synthesis), so it is intentionally explicit rather than automatic.

The council idea is inspired by [Andrej Karpathy's LLM Council](https://github.com/karpathy/llm-council).

## Navigation

```bash
research-tree --root ./research where
research-tree --root ./research focus q_abc123
research-tree --root ./research tree --depth 3
research-tree --root ./research next --from focus --limit 5
research-tree --root ./research show q_abc123
research-tree --root ./research graph --format mermaid
research-tree --root ./research graph --format dot --output map.dot
research-tree --root ./research graph --format json --json
```

IDs can be shortened to any unambiguous prefix. `root`, `focus`, `current`, and `.` are accepted as
node references. Focus is local state under `.state/` and is ignored by Git. Named cursors keep
simultaneous agent sessions independent:

```bash
research-tree --root ./research --cursor pi-session-42 focus q_abc123
research-tree --root ./research --cursor pi-session-42 where
```

Every command supports `--json` for agents and scripts. Expected failures use stable exit codes:
`3` not found, `4` provider/configuration, and `5` validation/integrity.

## Writing workflow

Promote a useful answer or synthesis into an article's research notes:

```bash
research-tree --root ./research promote y_abc123 --to ../research.md
```

Model-generated answers must be verified and non-contested before promotion. Deliberate escape
hatches (`--allow-unverified`, `--allow-uncertain`) are available for exploratory notes, but are
never automatic.

Research Tree does not commit or push on its own. This is deliberate: one investigation can create
many related files, and a writing agent should batch them into one meaningful sync.

## Canonical layout

```text
research/
├── project.json              project identity, root question, model settings
├── nodes/
│   ├── q_….md                questions
│   ├── a_….md                answers and council perspectives
│   ├── c_….md                atomic claims
│   ├── k_….md                concepts
│   └── y_….md                council syntheses
├── sources/s_….json          immutable URL + retrieved excerpt snapshots
├── runs/r_….json             immutable prompts, outputs, models, usage, cost
├── views/overview.md         generated human-readable map
└── .state/                   ignored locks and named cursors
```

Markdown/YAML and JSON are the source of truth. There is no required graph server or committed
binary database. See [the format contract](docs/format.md) for entity and relation details.

## Commands

| Command | Purpose |
| --- | --- |
| `init` | Create a project and root question |
| `where` / `focus` | Inspect or change a named cursor |
| `branch` | Add an explicit child question |
| `answer` | Record a human/manual answer |
| `ask` | Run one evidence-aware model |
| `council` | Compare models through blind review and synthesis |
| `verify` | Check claim-level citation support against frozen excerpts |
| `tree` / `next` | See the inquiry hierarchy and research frontier |
| `show` | Inspect a node with provenance |
| `graph` | Export Mermaid, DOT, or JSON |
| `source` / `run` | Inspect source snapshots and immutable model runs |
| `promote` | Append a node to writing notes |
| `doctor` | Check references, schemas, missing artifacts, and parent cycles |

## Design lineage

The council protocol borrows the pattern—not code—from [Andrej Karpathy's LLM
Council](https://github.com/karpathy/llm-council), then changes the judging target from polished prose
to claim support and source quality. The branching, human-steerable research experience is closest
to [Co-STORM](https://github.com/stanford-oval/storm). OpenRouter's
[web-search server tool](https://openrouter.ai/docs/guides/features/server-tools/web-search) supplies
model-controlled current evidence through a common API.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The project is experimental (`0.1.x`). The on-disk format is versioned and validated before use.
