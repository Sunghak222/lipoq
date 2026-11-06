"""Audit a completed real run using only its saved candidates and measurements."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lnp_vqa.encoding import CompositionEncoder


def verify(path):
    summary = json.loads((path / "summary.json").read_text())
    cfg = json.loads((path / "config.json").read_text())
    encoder = CompositionEncoder(cfg["search"])
    candidates = [json.loads(line) for line in (path / "candidates.jsonl").read_text().splitlines()]
    scores = {c["composition_key"]: c["comet_score"] for c in candidates}
    evaluations = [json.loads(line) for line in (path / "evaluations.jsonl").read_text().splitlines()]
    history = list(csv.DictReader((path / "history.csv").open()))
    assert summary["status"] == "completed"
    assert summary["best"]["comet_score"] == max(scores.values())
    assert len(scores) == len(candidates) == summary["counts"]["unique_sampled_compositions"]
    assert len(history) == cfg["optimizer"]["iterations"] + 1
    assert len(evaluations) == summary["objective_calls"] == 3*cfg["optimizer"]["iterations"] + 2
    for row in evaluations:
        counts = row["counts"]
        assert sum(counts.values()) == cfg["circuit"]["shots"]
        expected = sum(n*scores[encoder.decode(bits).key] for bits, n in counts.items()) / sum(counts.values())
        assert math.isclose(expected, row["mean_comet_score"], rel_tol=1e-12, abs_tol=1e-12)
    for c in candidates:
        assert sum(c["fraction_ticks"]) == c["denominator"]
        assert len(c["components"]) == 4
    assert evaluations[0]["theta"] != evaluations[-1]["theta"]
    assert summary["counts"]["sampled_shots"] == len(evaluations)*cfg["circuit"]["shots"]
    assert summary["counts"]["unique_comet_evaluations"] + summary["counts"]["unique_loaded_from_persistent_cache"] == len(scores)
    return {"status": "verified", "run": str(path), "candidates": len(candidates), "objective_calls": len(evaluations)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_directory), indent=2))

