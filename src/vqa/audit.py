"""Validate VQA accounting from saved records, without model or circuit execution."""
import csv
import math
import numpy as np

from common.archive import load_archive
from common.results import artifact_path, read_jsonl


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path):
    archive = load_archive(path, method="vqa")
    path, summary, cfg, encoder = (archive[k] for k in ("path", "summary", "config", "encoder"))
    scores = {c["composition_key"]: c["comet_score"] for c in archive["records"]}
    evaluations = read_jsonl(artifact_path(path, "evaluations.jsonl"))
    with artifact_path(path, "history.csv").open() as stream:
        history = list(csv.DictReader(stream))
    iterations = summary["iterations"]
    _require(1 <= iterations <= cfg["optimizer"]["iterations"], "Invalid completed iteration count")
    _require([int(r["iteration"]) for r in history] == list(range(iterations + 1)), "Invalid history sequence")
    expected_calls = (2 * cfg["optimizer"]["resamplings"] + 1) * iterations + 2
    _require(len(evaluations) == summary["objective_calls"] == expected_calls, "Invalid objective-call count")
    parameter_count = encoder.num_qubits * (cfg["circuit"]["depth"] + 1)
    for row in evaluations:
        counts = row["counts"]
        _require(bool(counts) and all(type(n) is int and n > 0 for n in counts.values()), "Invalid shot counts")
        _require(sum(counts.values()) == cfg["circuit"]["shots"], "Shot budget mismatch")
        expected = sum(n * scores[encoder.decode(bits).key] for bits, n in counts.items()) / sum(counts.values())
        _require(math.isclose(expected, row["mean_comet_score"], rel_tol=1e-12, abs_tol=1e-12), "Objective mean mismatch")
        _require(len(row["theta"]) == parameter_count and all(math.isfinite(x) for x in row["theta"]), "Invalid parameters")
    for file, row in (("initial_parameters.npy", evaluations[0]), ("final_parameters.npy", evaluations[-1])):
        parameters = np.load(artifact_path(path, file), allow_pickle=False)
        _require(parameters.shape == (parameter_count,) and np.array_equal(parameters, row["theta"]), f"{file} mismatch")
    # Equal initial/final parameters are valid, e.g. a zero SPSA gradient.
    _require(summary["counts"]["sampled_shots"] == len(evaluations) * cfg["circuit"]["shots"], "Total shot mismatch")
    counts = summary["counts"]
    _require(counts["unique_comet_evaluations"] + counts["unique_loaded_from_persistent_cache"] == len(scores), "Cache accounting mismatch")
    return {"status": "verified", "run": str(path), "candidates": len(scores), "objective_calls": len(evaluations)}
