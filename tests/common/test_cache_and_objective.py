"""Isolated arithmetic tests use a spy; no synthetic oracle exists in production."""
import math
import pytest
from common.evaluation import Evaluator
from common.cache import ScoreCache
from common.encoding import CompositionEncoder


class OracleSpy:
    def __init__(self):
        self.calls = []

    def predict(self, compositions):
        self.calls.extend(compositions)
        return [float(c.lipid_ids[0]) for c in compositions]


def test_weighted_mean_deduplication_persistence_and_namespace(cfg, tmp_path):
    encoder = CompositionEncoder(cfg["search"])
    catalog = {i: {"name": str(i), "smiles": "test-only"} for i in range(14)}
    spy = OracleSpy()
    cache = ScoreCache(tmp_path / "scores.sqlite", "model-cell-A")
    evaluator = Evaluator(spy, cache, catalog, tmp_path / "candidates.jsonl")
    metrics = evaluator.evaluate_counts({"0000000": 3, "0000001": 1}, encoder)
    assert metrics["mean_comet_score"] == 1.5  # (3*0 + 1*6)/4, NOT (0+6)/2
    assert evaluator.best["comet_score"] == 6
    evaluator.evaluate_counts({"0000001": 5}, encoder)
    assert len(spy.calls) == 2
    assert evaluator.stats()["duplicate_shots"] == 7
    assert len((tmp_path / "candidates.jsonl").read_text().splitlines()) == 2
    cache.close()
    cache = ScoreCache(tmp_path / "scores.sqlite", "model-cell-A")
    assert cache.get(encoder.decode("0000001").key) == 6
    cache.close()
    other = ScoreCache(tmp_path / "scores.sqlite", "model-cell-B")
    assert other.get(encoder.decode("0000001").key) is None
    with pytest.raises(ValueError):
        other.put_many([("bad", math.nan)])
    other.close()


def test_aliases_are_evaluated_once(cfg, tmp_path):
    cfg["search"]["candidates"]["ionizable"] = [0, 4, 6]
    encoder = CompositionEncoder(cfg["search"])
    spy = OracleSpy()
    catalog = {i: {"name": str(i), "smiles": "test-only"} for i in range(14)}
    cache = ScoreCache(tmp_path / "scores.sqlite", "test")
    evaluator = Evaluator(spy, cache, catalog, tmp_path / "candidates.jsonl")
    evaluator.evaluate_counts({"00000000": 2, "00000011": 3}, encoder)
    assert len(spy.calls) == 1
    assert evaluator.stats()["unique_sampled_compositions"] == 1
    cache.close()

