"""Method-independent readers for completed, versioned or legacy run archives."""
import json
import math
from pathlib import Path

from .comet import sha256
from .config import load_common_config
from .encoding import Composition, CompositionEncoder
from .results import SCHEMA_VERSION, artifact_path, method_name, read_jsonl


def find_completed(source, method=None):
    source = Path(source).resolve()
    if (source / "summary.json").is_file():
        paths = [source]
    else:
        paths = [p.parent for p in source.rglob("summary.json")]
    matched = []
    for path in paths:
        summary = json.loads((path / "summary.json").read_text())
        if summary.get("status") == "completed" and (method is None or method_name(summary) == method):
            matched.append(path)
    if not matched:
        raise ValueError(f"No completed {method or 'experiment'} run found under {source}")
    return max(matched, key=lambda p: (p.name, str(p)))


def composition_from_record(record):
    """Decode compact records and legacy records with expanded components."""
    ids = record.get("lipid_ids")
    if ids is None:
        ids = [c["mol_id"] for c in record["components"]]
    return Composition(tuple(ids), tuple(record["fraction_ticks"]), record["denominator"])


def load_archive(source, *, method=None):
    path = find_completed(source, method)
    summary = json.loads((path / "summary.json").read_text())
    if summary.get("schema_version", 1) not in (1, SCHEMA_VERSION):
        raise ValueError("Unsupported run schema version")
    cfg = load_common_config(artifact_path(path, "config.json"))
    encoder = CompositionEncoder(cfg["search"])
    records = read_jsonl(path / "candidates.jsonl")
    keys = [r["composition_key"] for r in records]
    budget = summary["counts"]["unique_sampled_compositions"]
    if type(budget) is not int or not 1 <= budget <= encoder.cardinality or len(set(keys)) != len(keys) or len(keys) != budget:
        raise ValueError("Candidate log does not match its unique-composition budget")
    ratios = set(encoder.ratios)
    for record in records:
        composition = composition_from_record(record)
        if (composition.key != record["composition_key"] or composition.denominator != encoder.denominator
                or composition.ticks not in ratios
                or any(mid not in options for mid, options in zip(composition.lipid_ids, encoder.candidates))
                or not math.isfinite(float(record["comet_score"]))):
            raise ValueError("Archive contains invalid compositions or nonfinite scores")
    best = summary["best"]
    if not math.isclose(max(r["comet_score"] for r in records), best["comet_score"], rel_tol=0, abs_tol=1e-7):
        raise ValueError("Archive best score does not match its candidate log")
    if "composition_key" in best:
        scores = {r["composition_key"]: r["comet_score"] for r in records}
        if best["composition_key"] not in scores or not math.isclose(scores[best["composition_key"]], best["comet_score"], rel_tol=0, abs_tol=1e-7):
            raise ValueError("Archive best composition does not match its candidate log")
    if not summary.get("model_fingerprint"):
        raise ValueError("Archive has no model fingerprint")
    return {"path": path, "config": cfg, "summary": summary, "encoder": encoder,
            "records": records, "budget": budget,
            "hashes": {name: sha256(artifact_path(path, name))
                       for name in ("config.json", "summary.json", "candidates.jsonl")}}
