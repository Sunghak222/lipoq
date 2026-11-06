# Benchmark protocol

Run `.venv/bin/lipoq benchmark --config configs/benchmark.yaml` after the standard COMET setup. All methods evaluate the original frozen checkpoint on CPU. No learned surrogate or synthetic experiment backend is used.

## Finite-space reference and exact learning analysis

The base configuration contains 16 molecular identity combinations and eight molar-fraction tuples, giving 128 unique formulations. All 128 are evaluated directly with COMET and stored in `exhaustive/candidates.jsonl`. The maximum is a certified maximum only over this finite space, with this fixed checkpoint, cell condition, and preprocessing seed.

For initial and final circuit parameters, Qiskit's statevector provides every computational-basis probability. Bitstrings are decoded, aliases are combined, and the exact expectation is computed as `sum(composition_probability * reference_score)`. The probability of obtaining a grid maximum in one shot is the sum of the probabilities assigned to all maximizers. A separate field counts scores within the configured floating-point tolerance. These calculations remove shot-sampling noise; they do not remove COMET prediction uncertainty or numerical precision limits.

Reference scores never enter VQA training, random selection, or continuous refinement. The historical run option checks its saved model fingerprint, condition, and composition grid before analyzing its saved angles. Differences between its sampled estimates and exact means are expected.

## Matched discrete evaluation budgets

Seeds are declared before execution: `0, 1, 2, 3, 4`. Each VQA run uses 128 shots and 12 SPSA updates. For a particular space and seed, let `N` be the number of distinct formulations encountered by all VQA objective evaluations, including perturbations. Random search evaluates exactly `N` different formulations, sampled uniformly without replacement from the same composition space.

This is a paired, observed-budget comparison. `N` can vary between seeds because VQA sampling can revisit candidates. It is not a prospective common stopping threshold across all seeds. Both methods have the same unique evaluation count within each pair. Random sampling is uniform over formulations, not over aliased bitstrings.

Every method/seed receives a new in-memory score cache. Neither the exhaustive table nor earlier methods' caches are used for search. Duplicates within a run use its own cache. `candidates.jsonl` records each new composition once, its stage, its cumulative evaluation count, score, elapsed time, and best score so far.

## Continuous refinement

For each method, rank its evaluated discrete formulations by score. Choose the top three distinct molecular identity combinations and start each refinement at the best observed fractions for those identities. Apply the same bounded [SciPy Powell optimizer](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-powell.html), with at most 60 objective calls per start, `xtol=0.001`, and `ftol=0.00001`.

Three unit-interval coordinates map continuously onto the bounded four-component simplex. Each coordinate reserves sufficient fraction for the remaining components; the fourth fraction is the residual. All COMET inputs satisfy the configured bounds and sum to one. The discrete grid step is ignored. Cache identity uses 12 decimal places to merge numerical duplicates, far below COMET's float32 input precision; this does not impose the discrete experiment grid on optimizer proposals.

The base refinement keeps the original bounds: ionizable 0.35–0.40, phospholipid 0.15–0.20, sterol 0.38–0.49, and PEG 0.01–0.02. Thus a gain over the base grid reflects newly accessible ratios for the selected identities, rather than wider bounds or different molecules. This does not certify the continuous global optimum.

The best candidate is retained across all evaluations, including the discrete search, even if Powell returns a worse final point. Individual solver termination messages are saved: reaching `maxfev` is an evaluation-budget stop, not proof of convergence.

Both methods receive the same solver call limits, but their actual **new unique COMET evaluations** can differ due to cached starting points, repeated proposals, or early termination. The report therefore shows objective requests, additional unique evaluations, total unique evaluations, and total wall time. It also compares post-hoc prefixes at the smaller total unique count in each pair. Prefix timing charges an entire batch when it contains the cutoff; it is not a separately executed early-stopping policy.

Set `continuous.enabled: false` to disable this stage. Changing `top_k`, `maxfev_per_start`, or optimizer tolerances changes its cost and potentially its result.

## Expanded search

| Group | Candidate IDs | Number | Fraction bounds | Discrete step |
|---|---|---:|---|---:|
| Ionizable | 0, 4, 5, 6, 7, 8, 9 | 7 | 0.30–0.55 | 0.05 |
| Phospholipid | 1, 13 | 2 | 0.10–0.25 | 0.05 |
| Sterol | 2, 10, 12 | 3 | 0.15–0.59 | 0.01 |
| PEG | 3, 11 | 2 | 0.01–0.03 | 0.01 |

These are all 14 molecules in the pinned COMET lipid library. Phospholipid and PEG counts remain two because there are no additional candidates of those types in that library. Molecules without a verified display name retain their COMET type and ID; exact SMILES are saved in `catalog.json`.

The expanded configuration yields 84 identity combinations and 72 feasible fraction tuples: **6,048 formulations**, encoded in 14 qubits. Sum-to-one constraints make some nominal component bounds unreachable; for example, the feasible sterol minimum is 0.17. All feasible fractions remain within the observed marginal data envelope, which does not guarantee joint in-domain predictive accuracy.

Non-power-of-two fields use the existing modulo encoding, so formulations do not all have the same number of bitstring aliases. This is part of the evaluated VQA encoding; random search remains uniform over unique formulations. Only the base grid is exhaustively evaluated. Expanded-grid regret and continuous global-optimum gaps are not asserted.

## Runtime and outputs

The model is loaded once, real input-pipeline parity is verified, and all selected molecule features are warmed before timing methods. Methods run sequentially; VQA runs first to determine each random budget. Both use the same model batch-size setting. Random search can fill batches efficiently, whereas VQA issues batches as its measurement calls discover new compositions and Powell evaluates single proposals. This difference is included in observed runtime.

Method wall times include circuit simulation, optimization, COMET evaluation, cache lookup, and result recording. They exclude shared model loading, preflight, feature warmup, the exhaustive reference, and post-hoc exact analysis. COMET time is recorded separately. The suite also records its complete elapsed time and every actual COMET input evaluation, including setup and analysis. Repeated timings on a laptop can vary with thermal state and background work.

- `summary.json`: protocol, all trials, aggregate statistics, setup costs, and total costs.
- `comparison.csv`: one row per method, seed, and stage.
- `report.md`: portable English report generated from measured values.
- `exhaustive/`: all base-grid predictions and optimum.
- `archived_run_analysis.json`: optional historical-run exact analysis and source-file hashes.
- `<space>/seed-<seed>/<method>/`: discrete result, continuous result, all candidate evaluations, and final summary.
- VQA subdirectories additionally contain measured bitstrings, initial/final parameters, circuit QASM, history, and base-grid exact learning analysis.
- Model provenance, both search encodings, data bounds, complete configurations, source hashes, and package versions are saved at the suite root.

Audit without new COMET calls:

```bash
.venv/bin/python scripts/verify_benchmark.py results/<benchmark-directory>
```

Five seeds provide a small descriptive comparison. Higher COMET scores do not establish higher measured delivery efficacy, and these experiments do not establish quantum advantage.
