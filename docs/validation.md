# Validation record

This record summarizes an actual local execution on **2026-11-06**, using Python 3.12, macOS arm64, PyTorch 2.6.0 on CPU, Qiskit 2.4.2, and Qiskit Algorithms 0.4.0. Machine-specific paths and raw runtime logs are not distributed.

## Model provenance

- COMET commit: `eee99d2055bad3b7d253c436b3eb7603550b37c5`
- Checkpoint: official lipid-only deployment fold 0
- SHA256: `8de532194feead3662a33d62855801ebe55b02014eb05e3eedbeb27ee60f7335`
- All checkpoint weights matched with `strict=True`; model parameters were frozen.
- Preprocessing seed: 0; molecule-dataset epoch: 1; one original conformer per molecule.
- Target: `in_house_lnp_DC24_luc`; N/P: 5.287338420806728; mixing ratio: 1:1.

For upstream example formulation 0, the original input pipeline produced **4.0856547355651855** and the adapter produced **4.085655212402344**. The absolute difference was **4.76837158203125e-7**. Repeated adapter inputs produced the same score. This tests input-pipeline parity and deterministic inference, not agreement with experimental measurements.

## Search and results

The configuration is [`configs/smoke.yaml`](../configs/smoke.yaml): 7 qubits, one linear CX layer between two RY layers, 14 angles, 128 shots, first-order SPSA with 12 updates, and seed 7.

| Metric | Empty-cache run |
|---|---:|
| Objective evaluations | 38 |
| Sampled shots | 4,864 |
| Unique sampled compositions / new COMET predictions | 108 / 108 |
| Avoided duplicate evaluations | 4,756 |
| Initial mean score | 2.5137177873402834 |
| Mean after update 12 | 5.240544 |
| Final additional measurement mean | 5.574887666851282 |
| Best sampled score | 7.800375938415527 |
| Total execution time | 8.5218 s |
| Model loading | 4.4103 s |
| COMET evaluation, including preflight | 3.6064 s |
| COMET evaluation during search | 3.3831 s |

Best sampled composition:

| Group | Molecule | COMET ID | Molar fraction |
|---|---|---:|---:|
| Ionizable | SM-102 | 6 | 0.35 |
| Helper | DOPE | 1 | 0.20 |
| Sterol | Cholesterol | 2 | 0.44 |
| PEG | C18-PEG, original COMET representation | 11 | 0.01 |

A separate process with the same seed reproduced all measured counts, final angles, and the best candidate exactly. It reused all 108 cached search predictions, made zero new search predictions, and completed in 4.1552 seconds. Preflight still performed real inference. The 14 unit tests and saved-run arithmetic audits passed.

The final additional measurement uses the final parameters with fresh shots, so its mean differs from the last monitored iteration. The search sampled 108 of 128 possible formulations; the recorded best is not asserted to be the global optimum. Runtime varies with hardware, system load, and cache state.

## Data scope

The original lipid-only deployment dataset contains 3,119 records. Of these, 2,069 contain exactly one molecule from each of the four component groups. Their observed marginal fraction bounds are:

| Group | Observed minimum | Observed maximum | Default search range |
|---|---:|---:|---:|
| Ionizable | 0.07079829 | 0.8 | 0.35–0.40 |
| Helper | 0.034177 | 0.811 | 0.15–0.20 |
| Sterol | 0.03846185 | 0.88378476 | 0.38–0.49 |
| PEG | 0.0050577 | 0.16208907 | 0.01–0.02 |

Known molecules and in-range marginal fractions do not guarantee that a joint composition is in-distribution. This experiment uses one cell, one fold, and one preprocessing seed. It establishes that the implementation runs with the actual pretrained model; it does not establish statistical optimization superiority, quantum advantage, or biological efficacy.
