# Experimental protocol

## Frozen objective

COMET is a fixed pretrained evaluator. Only the circuit angles change. For a measured bitstring `b`, the decoder produces composition `c(b)`. The optimizer minimizes

```text
loss(theta) = -sum_b count(b) * COMET(c(b)) / shots
```

Counts are aggregated after decoding, so bitstring aliases share a prediction while retaining their shot weights. The objective is not the unweighted mean of distinct candidates. Every uncached formulation uses the original model; no quadratic fit or other surrogate is used.

## Encoding and optimizer

Qiskit prints bitstrings as `q[n-1]...q[0]`. Fields are decoded from the least significant bits in the order ionizable lipid, phospholipid, sterol, PEG lipid, and fraction tuple. Integer fraction tuples sum to the configured denominator. Extra binary codes use modulo indexing, giving non-power-of-two fields unequal numbers of aliases.

The circuit contains `depth + 1` RY layers separated by `depth` linear CX chains. Initialization called `uniform` prepares a uniform distribution over bitstrings, not over formulations. The simulator computes a statevector and samples finite shots; optimizer objective calls do not use exact expectations.

First-order SPSA evaluates positive and negative perturbations for each of `r` directions, then monitors the updated circuit. Optional paired sampling uses the same simulator seed within each perturbation pair. With `K` completed updates, total calls are `(2*r + 1)*K + 2`, including initial and final measurements. The returned parameters average the configured number of recent updates; their fresh final measurement is recorded separately from the last monitoring row.

Plateau stopping compares nonoverlapping windows of monitoring scores. Improvement must exceed the minimum delta plus a sampling-error margin. After the minimum update count, consecutive unsuccessful windows increase the patience counter. Single-shot observations cannot provide the required standard error and do not trigger a noise-aware plateau stop. The returned reason is `plateau` or `iteration_limit`; neither proves convergence.

## Random baseline

The standalone baseline reads a completed VQA archive. The archive's distinct-composition count determines its budget, including all perturbation, monitoring, final, and persistent-cache candidates. Random samples the same number of compositions uniformly without replacement. It does not use VQA scores or rankings to choose candidates.

Both methods require identical cell/process conditions, composition space, and model fingerprint. Candidate-log consistency is checked before loading the model. The fingerprint includes source/checkpoint/data identities, preprocessing settings, and adapter bytes. Moving an unchanged adapter file preserves compatibility; changing its contents intentionally changes the fingerprint.

This protocol compares discovery at a matched number of distinct formulations. It does not match circuit shots, optimizer calls, or execution time. Random has a fresh in-memory score cache, while a VQA archive may have used a persistent cache. Timings are retained as descriptive measurements, not presented as a controlled speed comparison. Float32 score differences can arise from batch layout; comparison metadata records a tolerance of `1e-5`.

## Accounting and figures

Each run records unique encountered compositions, newly inferred compositions, persistent-cache hits, and elapsed search/model time. VQA additionally records sampled and duplicate shots. The three real-COMET preflight evaluations are separate from the search budget. Initial asset download and environment installation are excluded. Figure generation has a separate status and duration.

New archives use schema version 2 and a symmetric `<method>/seed-<seed>/run-<id>` layout. Shared metadata, metrics, optimizer state, logs, and plots have separate directories. Compact candidate records store molecule IDs and fractions; molecular structures are held once in the catalog. Readers accept the earlier flat archives without rewriting them. See [architecture](architecture.md) for the file contract and method extension interface.

Learning curves show finite-shot mean scores and best-so-far scores. Smoothing is a trailing average with no future samples. Shading is one sampling standard error, not a confidence interval for biological efficacy. Discovery curves follow the saved order of unique candidate records; within a batch, this is log order rather than a sequence of independent timed model calls.

Combined learning figures select the latest timestamped run per seed without ranking results by performance. Stopped runs are not padded. Incomplete records are labeled as snapshots. Archived raw records are the source of truth; plotting never invokes COMET or updates the optimizer.

## Model and data scope

The deployment checkpoint is COMET's lipid-only fold 0. The adapter keeps upstream source unmodified, verifies its revision and checkpoint SHA256 before loading the trusted legacy checkpoint, and loads weights with `strict=True`. On macOS it uses upstream PyTorch CPU kernels, `include_excipients=False`, frozen float32 weights, preprocessing seed 0 by default, and molecule-dataset epoch 1. Molecule IDs, SMILES, and conformers come from the pinned upstream library.

The configured DC2.4 target is `in_house_lnp_DC24_luc`; the checkpoint also supports `in_house_lnp_B16F10_luc`. Default process settings are N/P = 5.287338420806728 and aqueous:organic mixing ratio 1:1. These are distinct from the four lipid molar fractions.

The upstream deployment dataset has 3,119 records, including 2,069 four-component formulations. The adapter checks configured lipid types and fraction bounds against these inputs. Marginal coverage is not joint-domain validation. Scores are contrastive ranking predictions; absolute scales should not be compared across cells. One fold and one deterministic conformer do not constitute an ensemble or a model-uncertainty estimate.

The project provides no claim of measured delivery efficacy, global optimality, or quantum advantage. Such claims require evidence beyond these local simulator experiments.
