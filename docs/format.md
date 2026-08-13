# Research Tree format, schema version 1

The canonical store is a small set of Markdown/YAML and JSON files. Implementations must reject
unsupported `schema_version` values and malformed data rather than silently replacing them. A
future migration command will be responsible for preserving fields it understands explicitly.

## IDs

Objects use an immutable type prefix plus 12 lowercase hexadecimal characters:

| Prefix | Entity |
| --- | --- |
| `p_` | project |
| `q_` | question |
| `a_` | answer |
| `c_` | claim |
| `k_` | concept |
| `y_` | synthesis |
| `n_` | note |
| `s_` | source snapshot |
| `r_` | model run |

Source IDs are deterministic hashes of the URL and retrieved excerpt. This deduplicates identical
snapshots while preserving a later, changed version as a new object.

## Question hierarchy

Only `question` nodes may carry `parent_id`. Following those links must form a directed acyclic
graph with exactly one configured root. The displayed `decomposes_into` edge is derived from this
child-owned parent field, so adding a branch requires one canonical file write.

Question statuses are `proposed`, `open`, `researching`, `answered`, `uncertain`, `contested`, or
`parked`.

## Wider graph

Nodes may contain typed outgoing `edges`:

- `answers`
- `supports`
- `contradicts`
- `depends_on`
- `explains`
- `related_to`
- `derived_from`
- `supersedes`

`question_id` links an answer, synthesis, claim, concept, or note to the question it informs.
`source_ids` and `run_ids` link semantic nodes to immutable evidence and execution provenance.

## Source snapshots

A source stores its URL, title, retrieval time, excerpt content hash, optional publication metadata,
and retrieval mechanism. Search snippets are capped before persistence. A source snapshot is
immutable; changed content receives a new ID.

Source existence is not evidence of claim support. Each claim separately records the snapshots it
relies upon, allowing later citation-entailment verification.

Verification records both the entailment verdict and evidence quality (`primary`, `mixed`,
`secondary`, or `unknown`). A first-party release page is primary evidence for what its publisher
claims, but not independent reproduction of performance. Secondary-only support is never promoted
to a fully supported verdict by deterministic validation.

## Model runs

A run stores:

- mode (`ask`, `council`, or `verify`)
- question ID
- requested and resolved model IDs
- prompt hash and full prompts
- raw provider responses and parsed structures
- response node and source IDs
- tokens and provider-reported cost
- timestamp and schema version

Runs are immutable. A later re-evaluation creates another run and another answer; it never overwrites
the historical result.

## Local state and generated views

`.state/` contains file locks and named focus cursors. It is ignored by Git and is never canonical
research evidence. `views/overview.md` is derived from the graph and may be rebuilt at any time.
