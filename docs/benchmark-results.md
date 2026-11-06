# LipoQ benchmark results

All scores below are frozen pretrained COMET predictions for DC2.4, not measured efficacy.
Results describe this checkpoint, preprocessing seed, search bounds, and 5 configured search seeds.

Executed on 2026-11-06 on macOS arm64, Python 3.12, PyTorch CPU float32. The full suite used seeds 0–4; the previously saved seed-7 run was analyzed separately.

The historical VQA best is the exact maximum of the 128-formulation grid. Across the five new seeds, exact expected score improved in all five runs, but the probability of the single grid maximum decreased in one. Grid-optimum discovery succeeded in 4/5 VQA runs and 5/5 random runs. Thus improving the trained expectation did not establish better best-candidate discovery.

Random search had a higher average best score and lower runtime in both discrete spaces. Continuous refinement improved both methods; their base-space refined means were almost identical. In the expanded space, random plus refinement retained a higher average best score and lower runtime. These are descriptive results for the fixed configuration, not a claim about all VQA implementations.

## Protocol

- Independent empty score caches per method and seed; refinement reuses only its own search cache.
- Random search samples unique compositions uniformly without replacement and matches the VQA unique count for each seed.
- Method times exclude shared model loading, preflight, warmup, exhaustive enumeration, and exact statevector analysis.
- Both methods use identical Powell settings for refinement of their top 3 distinct lipid identities. Actual unique calls can differ due to caching and termination.
- Exact statistics use the full base score table only after search. Float32 batch-layout differences are handled with a 1e-05 score tolerance.

## Exhaustive base grid

Evaluated **128** unique formulations. Grid maximum: **7.800375938**.

| Component | Molecule | Molar fraction |
|---|---|---:|
| ionizable | SM-102 | 0.350000 |
| phospholipid | DOPE | 0.200000 |
| sterol | Cholesterol | 0.440000 |
| peg | C18-PEG (COMET representation) | 0.010000 |

## Previously saved VQA run

Seed 7; gap between its best sampled composition and the grid optimum: **0.000000000**.

| Exact statistic | Before training | After training |
|---|---:|---:|
| Mean COMET score | 2.636940350 | 5.200967144 |
| Probability of the grid maximum | 0.019006% | 0.235213% |

## Exact learning diagnostics for 5 seeds (base grid)

| Seed | Exact initial mean | Exact final mean | Initial optimum probability | Final optimum probability | Grid regret |
|---|---:|---:|---:|---:|---:|
| 0 | 4.344756 | 4.980167 | 0.00368939% | 0.000147699% | 0.000000 |
| 1 | 3.047664 | 5.771743 | 0.80341% | 6.68862% | 0.000000 |
| 2 | 0.255808 | 4.475296 | 0.043176% | 1.82574% | 0.000000 |
| 3 | -2.722327 | -0.803827 | 5.52347e-06% | 0.00629012% | 0.000000 |
| 4 | 0.097890 | 2.418480 | 9.71541e-06% | 0.0265572% | 0.227236 |

## Matched discrete search and continuous refinement

Values are mean ± sample standard deviation over the configured seeds. Continuous rows include discrete search cost.

| Space | Stage | Method | Best score | Unique COMET evaluations | Wall time (s) | COMET time (s) |
|---|---|---|---:|---:|---:|---:|
| base | discrete | vqa | 7.754929 ± 0.101623 | 105.8 ± 5.1 | 3.60 ± 0.39 | 3.49 ± 0.38 |
| base | discrete | random | 7.800376 ± 0.000000 | 105.8 ± 5.1 | 1.31 ± 0.07 | 1.30 ± 0.07 |
| base | with_continuous | vqa | 7.960546 ± 0.000028 | 281.8 ± 5.1 | 19.90 ± 1.08 | 19.72 ± 1.07 |
| base | with_continuous | random | 7.960533 ± 0.000000 | 281.8 ± 5.1 | 17.32 ± 0.77 | 17.25 ± 0.77 |
| expanded | discrete | vqa | 7.292186 ± 1.262008 | 1152.0 ± 251.1 | 43.74 ± 8.09 | 34.79 ± 8.07 |
| expanded | discrete | random | 8.314327 ± 0.163908 | 1152.0 ± 251.1 | 33.68 ± 7.36 | 33.57 ± 7.34 |
| expanded | with_continuous | vqa | 8.047533 ± 0.888315 | 1329.0 ± 251.1 | 59.77 ± 8.37 | 50.77 ± 8.35 |
| expanded | with_continuous | random | 8.495229 ± 0.045258 | 1324.4 ± 258.6 | 49.15 ± 8.16 | 48.97 ± 8.13 |

| Space | Stage | VQA wins | Ties | Random wins |
|---|---|---:|---:|---:|
| base | discrete | 0 | 4 | 1 |
| base | with_continuous | 1 | 4 | 0 |
| expanded | discrete | 2 | 0 | 3 |
| expanded | with_continuous | 1 | 1 | 3 |

## Per-seed measurements

| Space | Seed | Method | Grid evaluations | Grid best | Grid time (s) | Total evaluations | Final best | Total time (s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 0 | vqa | 104 | 7.800376 | 3.77 | 280 | 7.960533 | 21.61 |
| base | 0 | random | 104 | 7.800376 | 1.38 | 280 | 7.960533 | 18.64 |
| base | 1 | vqa | 107 | 7.800376 | 3.73 | 283 | 7.960533 | 20.05 |
| base | 1 | random | 107 | 7.800376 | 1.27 | 283 | 7.960533 | 17.08 |
| base | 2 | vqa | 114 | 7.800376 | 3.05 | 290 | 7.960533 | 18.91 |
| base | 2 | random | 114 | 7.800376 | 1.40 | 290 | 7.960533 | 17.26 |
| base | 3 | vqa | 103 | 7.800376 | 4.07 | 279 | 7.960533 | 19.86 |
| base | 3 | random | 103 | 7.800376 | 1.28 | 279 | 7.960533 | 16.60 |
| base | 4 | vqa | 101 | 7.573140 | 3.41 | 277 | 7.960596 | 19.05 |
| base | 4 | random | 101 | 7.800376 | 1.24 | 277 | 7.960533 | 17.03 |
| expanded | 0 | vqa | 1394 | 5.684404 | 50.72 | 1571 | 6.473986 | 67.26 |
| expanded | 0 | random | 1394 | 8.141047 | 40.53 | 1571 | 8.492804 | 56.12 |
| expanded | 1 | vqa | 828 | 6.304700 | 34.30 | 1005 | 8.227434 | 50.30 |
| expanded | 1 | random | 828 | 8.464256 | 24.36 | 982 | 8.521954 | 38.10 |
| expanded | 2 | vqa | 1215 | 7.543312 | 47.85 | 1392 | 8.492336 | 63.50 |
| expanded | 2 | random | 1215 | 8.435085 | 35.15 | 1392 | 8.521954 | 50.97 |
| expanded | 3 | vqa | 956 | 8.464256 | 35.64 | 1133 | 8.521954 | 51.19 |
| expanded | 3 | random | 956 | 8.399100 | 27.84 | 1133 | 8.417482 | 43.56 |
| expanded | 4 | vqa | 1367 | 8.464256 | 50.17 | 1544 | 8.521954 | 66.62 |
| expanded | 4 | random | 1367 | 8.132148 | 40.53 | 1544 | 8.521954 | 56.97 |

## Equal-total-evaluation prefixes after refinement

Post-hoc prefixes use the smaller total unique count in each pair; a batched evaluation is charged its full elapsed time. These are diagnostics, not separately rerun stopping policies.

| Space | Seed | Equal total evaluations | VQA best | Random best | VQA time to prefix (s) | Random time to prefix (s) |
|---|---:|---:|---:|---:|---:|---:|
| base | 0 | 280 | 7.960533 | 7.960533 | 21.61 | 18.64 |
| base | 1 | 283 | 7.960533 | 7.960533 | 20.05 | 17.08 |
| base | 2 | 290 | 7.960533 | 7.960533 | 18.91 | 17.26 |
| base | 3 | 279 | 7.960533 | 7.960533 | 19.86 | 16.60 |
| base | 4 | 277 | 7.960596 | 7.960533 | 19.05 | 17.03 |
| expanded | 0 | 1571 | 6.473986 | 8.492804 | 67.26 | 56.13 |
| expanded | 1 | 982 | 6.688238 | 8.521954 | 48.18 | 38.10 |
| expanded | 2 | 1392 | 8.492336 | 8.521954 | 63.50 | 50.98 |
| expanded | 3 | 1133 | 8.521954 | 8.417482 | 51.19 | 43.56 |
| expanded | 4 | 1544 | 8.521954 | 8.521954 | 66.62 | 56.97 |

## Best formulations found after continuous refinement

These are the best found across the recorded trials; they are not certified continuous optima.

| Space | Best score | Ionizable | Phospholipid | Sterol | PEG | Molar fractions (same order) |
|---|---:|---|---|---|---|---|
| base | 7.960596085 | SM-102 | DOPE | Cholesterol | C18-PEG (COMET representation) | 0.3500000000 / 0.1715282656 / 0.4684677347 / 0.0100039996 |
| expanded | 8.521953583 | SM-102 | DOPE | Cholesterol | C18-PEG (COMET representation) | 0.3000999510 / 0.1709441950 / 0.5189558540 / 0.0100000000 |

1 of 60 Powell starts reported convergence; 59 ended without convergence. See each saved termination message before interpreting a solution.

## Scope and accounting

- Base space: 128 formulations; expanded space: 6048 formulations.
- Only the base discrete grid was exhaustively evaluated. Expanded and continuous maxima are best found, not certified global optima.
- Powell is capped by its configured evaluation budget; individual success/termination messages are retained in continuous.json.
- More optimizer calls, different budgets, folds, conformers, or seeds can change the outcome.
- Expanded VQA encoding uses modulo aliases for non-power-of-two fields; random composition sampling is uniform.
- Methods run sequentially on a shared, prewarmed CPU model. VQA runs first to establish the paired budget; times are not hardware-independent estimates.
- Entire suite: 737.37 s; actual COMET input evaluations including setup: 16223.
- Setup evaluations: {'preflight': 3, 'warmup': 7, 'exhaustive': 128}; total COMET time: 684.15 s.
- Checkpoint/model fingerprint: `dbee523a78c0f38d68a65f0e4444df8fabe978d00a5cafd54ece4721d56c1bba`.
- These comparisons do not establish quantum advantage, statistical superiority, or biological efficacy.
