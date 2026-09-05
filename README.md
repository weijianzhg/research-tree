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

## Recording research by hand

Research Tree works as a pure recorder with no model calls at all. Add questions and
answers yourself, then grow the tree interactively or one command at a time:

```bash
# one-shot: branch a question and record its answer in a single step
research-tree --root ./research record "What is top-k routing?" --text "The router scores experts..."

# pipe an answer in (heredoc, file, or editor)
printf '%s\n' 'Top-k routing keeps the k highest-gated experts.' | research-tree --root ./research record focus
research-tree --root ./research record focus --file answer.md
research-tree --root ./research record focus --edit

# interactive recorder: type questions, then `a` to paste an answer
research-tree --root ./research record
```

Inside `record` (interactive mode), plain text branches a question, `a` records a
multi-line answer for the focused question (finish with a line containing only `.`,
or Ctrl-D), and `t`/`w`/`n`/`c`/`f`/`..`/`e`/`s` navigate, edit, and inspect the tree.
`quit` leaves the session; every step is saved immediately. Manual answers are tagged
`manual`, so they promote without model verification.

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

## Brainstorm research directions

```bash
research-tree --root ./research brainstorm focus --ideas 5
research-tree --root ./research brainstorm "How could expert routing adapt at inference time?" \
  --model 'deepseek/deepseek-v4-pro-0813' --ideas 5 --json
research-tree --root ./research ask q_abc123
```

`brainstorm` defaults to **String Seed of Thought (SSoT)**, adapted from
[Sakana AI's research](https://pub.sakana.ai/ssot/). Each independent model call first generates
its own random string and uses it to select a perspective, method, and scope before developing
one idea. All calls share a frozen topic/context; candidates do not see one another's outputs.
This is a prompting technique, not the provider's numeric decoding seed. The
[paper's external-randomness comparison](https://arxiv.org/html/2510.21150v3#A4.SS6) is why the
default generates the string inside the model rather than injecting a string from Python.

Each idea becomes a `proposed` child question with an angle, rationale, first step, and assumptions
to test. Brainstorming does not search the web or mark the topic answered. Use `ask` on a promising
branch to investigate it with evidence. Existing question status and focus are preserved; a
free-form topic creates and focuses a new question under the current focus. Seeds, full prompts,
model IDs, candidate outputs (including duplicates), and usage are saved in the immutable run.

The default is five candidates, at temperature `0.6`, using the project's model and reasoning
effort. `--model`, `--effort`, `--temperature`, and `--ideas` (1–20) override those choices. Five
candidates normally cost five completions. Invalid output is retried once per candidate; exhausted
validation or provider failure stops the batch, preserves completed calls and valid ideas, and
returns the usual error code. Repeated question titles are skipped without extra replacement calls,
so the number of new branches can be smaller than the candidate count.

Agents can inspect the exact request without credentials, paid calls, or graph changes:

```bash
research-tree --root ./research brainstorm focus --ideas 5 --dry-run --json
```

Use `--method direct` for an independent-call baseline without the seed instruction. Compare methods
on copies of the same starting graph, with the same topic, model, effort, temperature, and candidate
count, so earlier generated branches do not bias the second run. Assess distinct research directions
and usefulness, not just different wording. Title deduplication does not measure semantic diversity.
This JSON adaptation has not been benchmarked across models; SSoT does not guarantee novel or better
ideas, and its benefit depends on the model. Recorded seeds provide provenance, not deterministic
replay of the model's output.

## Navigation

```bash
research-tree --root ./research where
research-tree --root ./research focus q_abc123
research-tree --root ./research tree --depth 3
research-tree --root ./research next --limit 5
research-tree --root ./research next --from focus --limit 5   # scope to one subtree
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
`3` not found, `4` provider/configuration, and `5` validation/integrity (including model output
that fails to parse or validate).

## Writing workflow

Promote a useful answer or synthesis into an article's research notes:

```bash
research-tree --root ./research promote y_abc123 --to ../research.md
```

To turn a whole subtree of answered questions into a single synthesis without the cost of a
council, run `synthesize` (one model call, no web search):

```bash
research-tree --root ./research synthesize root
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
│   ├── y_….md                syntheses (council or synthesize)
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
| `record` | Record questions and answers yourself (free, offline; interactive) |
| `ask` | Run one evidence-aware model |
| `brainstorm` | Generate independent research directions with SSoT or direct prompting |
| `council` | Compare models through blind review and synthesis |
| `verify` | Check claim-level citation support against frozen excerpts |
| `synthesize` | Merge answered questions into one `y_` synthesis node |
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
