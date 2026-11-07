# Architecture and adding a search method

LipoQ separates formulation evaluation from candidate selection. `common` has no
dependency on `vqa` or `random_search`. Each method supplies a search function and
may supply a postprocessing function. The CLI parses and validates inputs, then
delegates to the shared experiment runner.

## Responsibilities

| Module | Responsibility |
|---|---|
| `common.encoding` | Composition identity, feasible ratio grid, bit decoding and uniform composition indexing |
| `common.comet` | Frozen upstream inference, input validation and model provenance |
| `common.evaluation` | Score requests, deduplication, cache accounting, candidate log and best candidate |
| `common.experiment` | Run creation, model/preflight, cache lifetime, timings, status and failure handling |
| `common.results` | Versioned artifact locations, atomic JSON writes and legacy path lookup |
| `common.archive` | Completed-run discovery and validation, independent of a search algorithm |
| `common.comparison` | Matched-budget comparison tables and discovery curves for any two methods |
| `vqa.method` / `random_search.method` | Adapt each algorithm to the shared search contract |
| `vqa.optimizer` / `random_search.search` | Candidate selection and method-specific progress |
| Method plotting/reporting modules | Read saved results; never run inference or train |

Qiskit imports remain in the VQA package. The local simulator's 1–20-qubit limit
is checked by the VQA configuration validator, not the shared composition model.
Configuration validation remains method-specific; new methods can introduce
their own settings without modifying the shared configuration parser.

## New result layout

```text
results/lnp_search/
  vqa/seed-21/run-<timestamp>-<id>/
  random/seed-21/run-<timestamp>-<id>/
  <new_method>/seed-21/run-<timestamp>-<id>/
```

Every new run uses schema version 2:

```text
run-<timestamp>-<id>/
  summary.json       Running/completed/failed/interrupted status and common metrics
  best.json          Expanded best formulation, updated during search
  candidates.jsonl   One compact record per distinct composition
  metadata/          Config, encoding, environment, model, catalog, preflight, reference
  metrics/           History, objective evaluations, method progress, comparison reports
  state/             Circuit and optimizer parameter/checkpoint files
  logs/              Model log and failure traceback
  plots/             PNG/SVG exports and plotting status
```

Only applicable files are created. Probe runs have no candidate or optimizer
files. A failed search retains its partial candidate log and best candidate;
`summary.json` is the authoritative status. Python exceptions and keyboard
interrupts are recorded. A force-killed process can retain a `running` status;
automatic resume is not implemented.

Search completion is saved before postprocessing. A comparison/report failure
is recorded in `metrics/analysis_status.json`, and a plot failure in
`plots/plot_status.json`. Neither changes a completed search into a failed one.
Search timings exclude analysis/plot generation. Both methods share the same
timing fields, preflight accounting, and cache policy labels.

Candidate records contain lipid IDs, fraction ticks/denominator, canonical key,
score, cache status, discovery order, running best, and elapsed time. Molecular
names and SMILES are stored once in `metadata/catalog.json`. `best.json` and the
summary retain readable component names and fractions.

Readers use `artifact_path(directory, filename)` to accept new structured runs
and old flat archives. Existing archives are never migrated in place. Raw model
and output paths in archived configurations refer to the machine that ran the
experiment; moving an archive does not automatically relocate its model assets.

## Add a method

1. Create `src/<method>/__init__.py` and a method-specific configuration validator. Use
   `load_common_config()` for the shared sections. Keep the algorithm in this
   package; no registration in `common` is required.
2. Implement `search(context: SearchContext) -> dict`. Select candidates and call
   `context.evaluator.score(compositions)` or
   `context.evaluator.evaluate_counts(counts, context.encoder)`. Calling COMET
   directly would bypass shared caching, accounting and candidate records.
3. Return method-specific metrics, such as completed steps or a stop reason.
   Common fields (`status`, `method`, `seed`, `best`, `counts`, `timing`, etc.)
   are owned by the runner and cannot be overwritten.
4. Call the shared runner from the CLI:

```python
from common.experiment import ExperimentSpec, SearchContext, run_experiment
from common.results import output_path, append_csv

def search(context: SearchContext) -> dict:
    # Replace this loop with the candidate-selection rule for the new method.
    for step, composition in enumerate(your_candidate_generator(context)):
        score = context.evaluator.score([composition])[0]
        append_csv(output_path(context.directory, "metrics/steps.csv"),
                   {"step": step, "score": score})
    return {"termination_reason": "budget_reached"}

result = run_experiment(
    ExperimentSpec(method="new_method", config=validated_config),
    search,
)
```

The two placeholder names above are supplied by the new method. No model-loading,
failure-handling, cache-closing or summary-writing code belongs in that loop.
The default cache is fresh in-memory. Persistent reuse must be explicitly
selected with `ExperimentSpec(cache_path=...)` and accounted for in comparisons.

Use `output_path(directory, "state/<file>")` for custom state, or
`output_path(directory, "metrics/<file>")` for method-specific records. An
optional `analyze(directory, summary)` callback runs only after search completion.
Use `common.comparison` to compare against a completed archive of any method;
it checks the composition budget, model fingerprint, search space and conditions.
New packages with `__init__.py` are discovered automatically under `src/`. Add a
command entry point in `pyproject.toml` only if a separate CLI is needed, then
reinstall the editable package to refresh its commands.

## Validation and compatibility

Tests cover an additional test-only method using the same lifecycle, a failing
postprocessor, search failure/interruption, compact candidate archives, and a
valid VQA fixed point with unchanged parameters. Algorithms have focused tests
for their own behavior. Test-only oracles are injected inside tests; no synthetic
backend is selectable from production CLIs.

The archived seed 21 pair and submitted notebook are immutable evidence of the
earlier implementation. They are read for compatibility checks, not rewritten
by a refactor. The COMET adapter bytes remain unchanged, preserving the existing
prediction/cache fingerprint. Source changes elsewhere are represented by the
new archive schema and the repository history. `metadata/environment.json` also
records SHA256 hashes of shared and method-package Python source files, so later
method revisions can be distinguished even when their COMET fingerprint matches.
