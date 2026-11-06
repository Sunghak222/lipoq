# LipoQ

**Black-box variational quantum search for lipid nanoparticle compositions.**

LipoQ is an experimental Python codebase for an IBM Quantum Qiskit Fall Fest project. It uses a shallow Qiskit circuit to sample lipid nanoparticle (LNP) formulations, scores them with a frozen pretrained [COMET](https://github.com/alvinchangw/COMET) model, and updates the circuit parameters with first-order SPSA.

```text
Parameterized circuit → measured bitstrings → lipid identities + molar fractions
          ↑                                           ↓
     SPSA update ← negative mean score ← pretrained COMET inference
```

The search jointly selects one molecule from each of four groups—ionizable lipid, phospholipid, sterol, and PEG lipid—and their molar fractions. Every decoded formulation sums exactly to one. Only the circuit angles are trained; the COMET weights remain frozen.

**COMET scores are model predictions, not measured delivery efficacy, probabilities, or efficacy percentages.** This project does not demonstrate quantum advantage. It does not use a quadratic surrogate, QUBO, or QAOA.

## Quick start

Tested on macOS arm64 with Python 3.12, PyTorch CPU float32, and the local Qiskit statevector simulator. Install [uv](https://docs.astral.sh/uv/) and Git first. Internet access is needed for initial installation and the model download; experiments then run locally.

```bash
git clone https://github.com/Sunghak222/lipoq.git
cd lipoq

uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements.lock
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
.venv/bin/python scripts/setup_comet.py
```

Setup downloads the pinned upstream source and official lipid-only fold-0 checkpoint (approximately 614 MB), then verifies its SHA256. Downloaded source, data, and weights stay in the ignored `vendor/` and `artifacts/` directories. COMET's upstream license is preserved at `vendor/COMET/license.pdf`.

Run an experiment with one configuration file:

```bash
.venv/bin/python -m lnp_vqa run --config configs/smoke.yaml
```

The installed CLI is also available as `.venv/bin/lipoq`. To check real pretrained inference without optimization:

```bash
.venv/bin/lipoq probe --config configs/smoke.yaml
```

Every run checks real COMET inference and compares the adapter output with the original input pipeline before optimization. Missing assets, invalid checksums, unsupported cells, invalid compositions, or inference errors stop the run. There is no synthetic-score fallback.

## Default experiment

| Setting | Default |
|---|---|
| Cell | DC2.4, a mouse dendritic cell line; COMET target `in_house_lnp_DC24_luc` |
| Ionizable lipid | ALC-0315 or SM-102; COMET molecule IDs 0, 6 |
| Phospholipid | DOPE or DSPC; IDs 1, 13 |
| Sterol | Cholesterol or beta-sitosterol; IDs 2, 10 |
| PEG lipid | C14-PEG or C18-PEG; IDs 3, 11 |
| Molar fractions | Ionizable 0.35–0.40, phospholipid 0.15–0.20, sterol 0.38–0.49, PEG 0.01–0.02 |
| Fixed process conditions | N/P = 5.287338420806728; aqueous:organic mixing ratio = 1:1 |
| Search space | 16 lipid combinations × 8 feasible fraction tuples = 128 formulations |
| Circuit | 7 qubits, RY → linear CX chain → RY; 14 parameters |
| Optimization | 128 shots, 12 SPSA iterations, seed 7 |

The executed search used the **DC2.4 prediction target**; no cell experiments were performed. The same checkpoint also supports B16-F10, which was not used in this run. See the [DC2.4 cell-line reference](https://www.cellosaurus.org/CVCL_J409).

For this small execution check, the configured ranges and steps generate exactly eight feasible fraction tuples. This list stays fixed during a run; SPSA learns the circuit's sampling probabilities over lipid identities and these tuples. Fractions are therefore searched on a discrete grid, not optimized as continuous variables. The eight tuples are not a COMET restriction or a list of experimentally validated recipes.

| Tuple | Ionizable (mol%) | Phospholipid (mol%) | Sterol (mol%) | PEG (mol%) |
|---|---:|---:|---:|---:|
| 1 | 35 | 15 | 48 | 2 |
| 2 | 35 | 15 | 49 | 1 |
| 3 | 35 | 20 | 43 | 2 |
| 4 | 35 | 20 | 44 | 1 |
| 5 | 40 | 15 | 43 | 2 |
| 6 | 40 | 15 | 44 | 1 |
| 7 | 40 | 20 | 38 | 2 |
| 8 | 40 | 20 | 39 | 1 |

Every row sums to 100 mol%. Ionizable lipid takes 35 or 40 mol%, phospholipid 15 or 20 mol%, and PEG lipid 1 or 2 mol%; sterol fills the remainder within its configured 38–49 mol% range. The best sampled formulation used tuple 4. These lipid molar fractions are separate from the fixed aqueous:organic process mixing ratio of 1:1.

Copy and edit [`configs/smoke.yaml`](configs/smoke.yaml) to change the cell, candidate molecule IDs, fraction ranges and resolution, circuit depth, shots, iterations, SPSA coefficients, or seeds. Relative paths are resolved against the YAML file's directory. Qubit count is derived from the encoding; an explicit count must match it. The local simulator is limited to 20 qubits.

Molecules, SMILES, and 3D conformers come directly from the pinned COMET lipid-only dataset. PEG lipids retain COMET's shortened molecular representations. Display names are labels; the recorded molecule IDs and SMILES define the actual model input. Arbitrary new SMILES are not supported. The generated `catalog.json` lists available upstream molecules and their groups.

Default fraction ranges lie inside the marginal ranges of the 2,069 four-component formulations in the 3,119-record deployment dataset. Process conditions come from example record 0. These bounds do not establish that every new combination is within the model's validated joint domain. See [validation and scope](docs/validation.md).

## Encoding and objective

Fractions use an integer grid. Only tuples whose integer sum equals the denominator are encoded, so every bitstring maps to a valid composition without a penalty term.

Qiskit bitstrings are interpreted as `q[n-1]…q[0]`. From the least significant bit, fields select the ionizable lipid, phospholipid, sterol, PEG lipid, and a feasible fraction tuple. Non-power-of-two field sizes use modulo mapping, which can give some candidates more aliases. The default configuration has no aliases; caching always deduplicates by composition.

Configuration and output use the group key `phospholipid`. The COMET adapter preserves the upstream type identifier `HL` when reading the original dataset.

```text
loss(theta) = -sum(measurement_count × COMET_score) / shots
```

SPSA minimizes this shot-weighted loss. Each iteration uses two perturbation evaluations and a measurement of the updated circuit. Including initial and final measurements, the default run makes 38 objective calls. Finite-shot noise means the mean need not increase at every iteration. The best candidate is retained across **all** sampled evaluations, including perturbations.

## Outputs and caching

Each experiment creates a new `results/run-<timestamp>-<id>/` directory.

| Output | Contents |
|---|---|
| `summary.json`, `best.json` | Best score, four lipids and fractions, initial/final means, timing, and evaluation counts |
| `history.csv` | Initial and per-iteration mean, sampling standard error, and best-so-far score |
| `evaluations.jsonl` | Every objective call's counts, angles, sampling seed, and scores |
| `candidates.jsonl` | Unique sampled formulations, predictions, and persistent-cache status |
| `config.json`, `encoding.json` | Resolved configuration, fraction grid, and bit allocation |
| `model.json`, `preflight.json`, `domain.json`, `catalog.json` | Provenance, inference parity, data bounds, and molecular inputs |
| `initial_parameters.npy`, `final_parameters.npy`, `optimizer_state.json`, `circuit.qasm` | Circuit and optimization state |
| `environment.json`, `comet_load.log` | Runtime versions and upstream loading output |

Timing separates model loading, COMET input preparation and inference, forward passes, preflight, and search. Total time covers the CLI execution body through optimization, excluding installation and downloads. The three preflight input evaluations are reported separately from search evaluations.

The SQLite cache is namespaced by model/data hashes, preprocessing seed, cell, N/P, mixing ratio, and adapter version. Repeated runs can reuse scores while repeating sampling and SPSA updates. Set a new cache path to measure inference with an empty cache. On interruption, completed cache entries, candidates, progress, best results, and `failure.json` remain available; automatic optimizer resume is not implemented.

## Verified local run

The default experiment was executed with the **real pretrained checkpoint** on macOS arm64 CPU:

| Metric | Observed result |
|---|---:|
| Unique formulations evaluated | 108 |
| Initial mean COMET score | 2.513718 |
| Mean after the 12th update | 5.240544 |
| Final additional measurement mean | 5.574888 |
| Best sampled score | 7.800376 |
| Total time, empty cache | 8.52 s |
| COMET evaluation time, including preflight | 3.61 s |

The best sampled formulation was **SM-102 / DOPE / Cholesterol / C18-PEG**, with molar fractions **0.35 / 0.20 / 0.44 / 0.01**. A repeat reproduced measured counts and final parameters exactly and reused all 108 search predictions from cache. This is a small, single-seed execution check, not a statistical benchmark or evidence of biological efficacy. See the [validation record](docs/validation.md) for provenance and limitations.

## Tests and code

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_run.py results/<run-directory>
```

All 14 unit tests passed. The saved-run audit checks shot-weighted means, composition validity, best-candidate consistency, evaluation counts, and parameter updates. Test spies are confined to unit tests; they are not selectable experiment backends.

```text
configs/smoke.yaml       Experiment settings
src/lnp_vqa/
  comet.py              Frozen original COMET and its input pipeline
  encoding.py           Lipid/fraction encoding and decoding
  vqa.py                Qiskit circuit, sampling, and first-order SPSA
  cache.py              Composition cache and shot-weighted evaluation
  config.py             Configuration validation
  results.py, cli.py    Output recording and command-line execution
scripts/setup_comet.py   Pinned source/checkpoint setup
scripts/verify_run.py    Audit saved experiment results
tests/                  Encoding, sampling, cache, and failure tests
```

The internal module remains `lnp_vqa`; the project and installed command are named **LipoQ**.

## Compatibility and limitations

- Uses COMET's original PyTorch CPU fallback instead of CUDA fused kernels. Upstream `fused_* is not installed` messages are expected on this path.
- Verifies the checkpoint hash before loading the legacy checkpoint with `weights_only=False`; loads all model weights with `strict=True`.
- Sets `include_excipients=False` for this lipid-only checkpoint and fixes the molecule-dataset epoch to 1 for reproducible conformer selection. Upstream source is unmodified.
- Uses one pretrained fold and one deterministic conformer per molecule, not an ensemble or an average over 11 conformers. Changing the preprocessing seed can change scores.
- Outputs contrastive ranking scores. Do not interpret them as measured luciferase values or compare absolute scales across cells.
- Uses a circuit-defined distribution over formulations. It is not a reproduction of the reference paper's complete QITE/VarQFS algorithm. No classical optimizer comparison, notebook, presentation, or separate UI is included.

## References

- [COMET source and pretrained assets](https://github.com/alvinchangw/COMET)
- [Designing lipid nanoparticles using a transformer-based neural network](https://www.nature.com/articles/s41565-025-01975-4)
- [Variational quantum algorithm for unconstrained black box binary optimization](https://quantum-journal.org/papers/q-2023-01-26-909/)
- [Qiskit SPSA documentation](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.optimizers.SPSA.html)
