"""The shared lifecycle works for new methods without copying either existing CLI."""
import json
import csv
import sqlite3

import pytest

from common.archive import load_archive
from common.comparison import write_comparison
from common.experiment import ExperimentSpec, run_experiment
from common.results import artifact_path
from vqa.audit import verify
from vqa.method import search as vqa_search


class FlatOracle:
    fingerprint = "TEST-ONLY"
    provenance = {"source": "TEST-ONLY"}
    evaluation_seconds = 0.0
    compositions_evaluated = 0
    load_seconds = 0.0

    def __init__(self, config, condition, log_path):
        self.catalog = {i: {"name": str(i), "smiles": "TEST-ONLY"} for i in range(14)}

    def validate_search(self, encoder):
        return {"status": "test-only"}

    def preflight(self):
        self.compositions_evaluated += 3
        return {"status": "test-only"}

    def predict(self, compositions):
        self.compositions_evaluated += len(compositions)
        return [1.0] * len(compositions)


def test_new_method_uses_shared_contract_and_analysis_failure_is_separate(cfg, tmp_path):
    cfg["output"]["directory"] = str(tmp_path)

    def other_method(context):
        a, b = [context.encoder.composition_at(i) for i in range(2)]
        context.evaluator.score([a, b, a])
        return {"custom_steps": 1}

    def broken_analysis(*args):
        raise RuntimeError("test-only analysis failure")

    result = run_experiment(ExperimentSpec("another_method", cfg), other_method,
                            analyze=broken_analysis, oracle_factory=FlatOracle)
    assert result.directory.parent == tmp_path / "another_method" / "seed-7"
    assert result.summary["status"] == "completed"
    assert result.summary["custom_steps"] == 1
    assert result.summary["counts"]["unique_comet_evaluations"] == 2
    assert result.summary["preflight_composition_evaluations"] == 3
    assert result.analysis["status"] == "failed"
    assert (result.directory / "logs/analysis_traceback.txt").is_file()
    assert not (result.directory / "logs/traceback.txt").exists()
    archive = load_archive(result.directory, method="another_method")
    assert archive["budget"] == 2
    assert all("components" not in row and "smiles" not in row for row in archive["records"])
    assert result.summary["best"]["components"][0]["name"]
    assert (result.directory / "metadata/catalog.json").is_file()
    # Generic comparison does not assume either side is VQA or random search.
    other = run_experiment(ExperimentSpec("third_method", cfg), other_method, oracle_factory=FlatOracle)
    for _ in range(2):
        report = write_comparison(archive, other.summary, other.directory)
    assert [row["method"] for row in report["rows"]] == ["another_method", "third_method"]
    with artifact_path(other.directory, "comparison.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2  # Regeneration is idempotent.


@pytest.mark.parametrize("exception,status", [(RuntimeError, "failed"), (KeyboardInterrupt, "interrupted")])
def test_failure_preserves_partial_candidate_and_closes_cache(cfg, tmp_path, exception, status):
    cfg["output"]["directory"] = str(tmp_path)
    caches = []

    def broken_search(context):
        caches.append(context.evaluator.cache)
        context.evaluator.score([context.encoder.composition_at(0)])
        raise exception("test-only failure")

    with pytest.raises(exception):
        run_experiment(ExperimentSpec("another_method", cfg), broken_search, oracle_factory=FlatOracle)
    summary_path = next(tmp_path.rglob("summary.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == status
    assert summary["counts"]["unique_comet_evaluations"] == 1
    assert summary["best"] == json.loads((summary_path.parent / "best.json").read_text())
    assert (summary_path.parent / "logs/traceback.txt").is_file()
    with pytest.raises(sqlite3.ProgrammingError):
        caches[0].get("closed")


def test_fixed_point_is_valid_and_new_layout_can_be_audited(cfg, tmp_path):
    cfg["output"]["directory"] = str(tmp_path)
    cfg["circuit"]["shots"] = 8
    cfg["optimizer"].update(iterations=1, last_avg=1, resamplings=1)
    result = run_experiment(ExperimentSpec("vqa", cfg), vqa_search, oracle_factory=FlatOracle)
    assert verify(result.directory)["status"] == "verified"
    assert (result.directory / "metrics/history.csv").is_file()
    assert (result.directory / "state/final_parameters.npy").is_file()
    # Tampering with the numerical record must still be detected.
    evaluations = artifact_path(result.directory, "evaluations.jsonl")
    rows = [json.loads(line) for line in evaluations.read_text().splitlines()]
    rows[0]["mean_comet_score"] = 1000.0
    evaluations.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="mean mismatch"):
        verify(result.directory)
