# Changelog

All notable changes to Research Tree are documented here. The package follows semantic
versioning; the on-disk graph format is versioned separately (`schema_version` on each object).

## [0.3.0] - 2026-08-14

### Added

- `ask` now accepts free-form question text in addition to node references. When the argument is
  not a node reference, `ask` creates a question node with the exact wording under the current
  focus and answers it in one step, so asking a brand-new question no longer requires a separate
  `branch` first.
- `..` resolves to the parent of the current focus (a question's parent, or the owning question
  for non-question nodes; root stays put), so `focus ..`, `branch --from ..`, and `ask ..` can
  navigate up and branch at an ancestor without typing full node IDs.

## [0.2.0] - 2026-08-13

### Added

- `synthesize` command: merge every answer under a question (and its descendants) into a single
  `y_` synthesis with one model call. The synthesis links back to each aggregated answer and
  inherits their source snapshots, so provenance is preserved without a full council.
- `ask` now retries once when a provider returns unparseable or schema-invalid output. If every
  attempt fails, an immutable `failed_validation` run is persisted with the prompts, raw
  responses, and cost, so a paid call is never invisible.

### Changed

- `next` ranks unanswered questions from the whole tree by default (`--from root`) instead of the
  focused subtree, and reports which scope it searched.
- `branch` prints the parent of the new question and tells you when the cursor moved to it.
- `tree` prints a legend for its status markers.
- `promote` rejects non-answer/synthesis nodes with a clear message instead of promoting a
  question's body.

### Fixed

- Model-output validation failures now raise `ModelOutputError` and exit with code `5`
  (validation), distinguishing "the model returned garbage — try another model" from "the
  provider could not be reached or the key is wrong" (code `4`).

## [0.1.0] - 2026-08-13

- Initial release: Git-native question tree, evidence-aware `ask`, `verify`, and multi-model
  `council` workflows, source snapshots, and immutable model runs.
