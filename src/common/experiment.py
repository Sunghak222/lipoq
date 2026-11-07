"""Shared experiment lifecycle. Search methods supply only their search function.

No quantum implementation is imported here. Model setup, cache lifetime,
provenance, failure handling and timing follow one contract for every method.
"""
from dataclasses import dataclass
from datetime import datetime
import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time
import traceback
from typing import Callable, Protocol, Any
from zoneinfo import ZoneInfo

from .cache import ScoreCache
from .comet import CometOracle, SCORE_DESCRIPTION
from .config import integer
from .encoding import CompositionEncoder
from .evaluation import Evaluator
from .results import SCHEMA_VERSION, create_run_dir, output_path, write_json


@dataclass(frozen=True)
class ExperimentSpec:
    method: str
    config: dict
    cache_path: str = ":memory:"
    expected_fingerprint: str | None = None
    reference: dict | None = None

    @property
    def output_root(self) -> Path:
        return Path(self.config["output"]["directory"]) / self.method / f"seed-{self.config['seed']}"


@dataclass(frozen=True)
class SearchContext:
    config: dict
    encoder: CompositionEncoder
    evaluator: Evaluator
    directory: Path


class SearchMethod(Protocol):
    def __call__(self, context: SearchContext) -> dict[str, Any]:
        """Search through context.evaluator and return method-specific metrics."""
        ...


@dataclass(frozen=True)
class ExperimentResult:
    directory: Path
    summary: dict
    analysis: dict | None


def _environment(search):
    roots = {Path(__file__).resolve().parent}
    standalone_files = []
    if search is not None:
        module = sys.modules.get(getattr(search, "__module__", ""))
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).is_file():
            path = Path(filename).resolve()
            package = path.parent
            if (package / "__init__.py").is_file():
                while (package.parent / "__init__.py").is_file():
                    package = package.parent
                roots.add(package)
            else:
                # Do not recursively scan a user script's environment/data directory.
                standalone_files.append(path)
    source_hashes = {
        f"{root.name}/{path.relative_to(root).as_posix()}": hashlib.sha256(path.read_bytes()).hexdigest()
        for root in sorted(roots) for path in sorted(root.rglob("*.py"))
    }
    source_hashes.update({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in standalone_files})
    return {
        "started_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "source_sha256": source_hashes,
    }


def _analyze(directory, summary, analyze):
    """Analysis failure is recorded separately from a completed search."""
    started = time.perf_counter()
    try:
        status = {"status": "completed", "result": analyze(directory, summary)}
    except Exception as error:
        status = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
        output_path(directory, "logs/analysis_traceback.txt").write_text(traceback.format_exc())
        print(f"Experiment saved; analysis failed: {error}", file=sys.stderr, flush=True)
    status["seconds"] = time.perf_counter() - started
    write_json(output_path(directory, "analysis_status.json"), status)
    return status


def run_experiment(
    spec: ExperimentSpec,
    search: SearchMethod | None,
    *,
    analyze: Callable[[Path, dict], dict] | None = None,
    oracle_factory=None,
) -> ExperimentResult:
    """Run one method, save a versioned archive, and always close its cache.

    ``search=None`` performs a real-model probe only. Oracle injection is used
    by isolated tests; production CLIs always use the original CometOracle.
    """
    started = time.perf_counter()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", spec.method):
        raise ValueError("Method names must use lowercase letters, digits and underscores")
    integer(spec.config["seed"], "seed", 0, 2**32 - 1)
    encoder = CompositionEncoder(spec.config["search"])
    directory = create_run_dir(spec.output_root, "run")
    base = {"schema_version": SCHEMA_VERSION, "method": spec.method,
            "seed": spec.config["seed"], "score_description": SCORE_DESCRIPTION,
            "search_space_size": encoder.cardinality}
    if spec.reference:
        base["reference_run"] = spec.reference["path"]
    oracle = cache = evaluator = None
    preflight_seconds = preflight_evaluations = 0
    search_started = None

    def timing():
        return {"model_load_seconds": getattr(oracle, "load_seconds", 0.0),
                "preflight_comet_seconds": preflight_seconds,
                "search_comet_seconds": (getattr(oracle, "evaluation_seconds", 0.0) - preflight_seconds
                                         if search_started is not None else 0.0),
                "comet_evaluation_seconds": getattr(oracle, "evaluation_seconds", 0.0),
                "comet_forward_seconds": getattr(oracle, "forward_seconds", 0.0),
                "search_seconds": time.perf_counter() - search_started if search_started is not None else 0.0,
                "total_seconds": time.perf_counter() - started}

    try:
        print(f"Results: {directory}", flush=True)
        write_json(directory / "summary.json", {**base, "status": "running"})
        for name, value in (("config.json", spec.config), ("encoding.json", encoder.metadata()),
                            ("environment.json", _environment(search))):
            write_json(output_path(directory, name), value)
        if spec.reference:
            write_json(output_path(directory, "reference.json"), spec.reference)
        print("Loading verified pretrained COMET on CPU ...", flush=True)
        factory = CometOracle if oracle_factory is None else oracle_factory
        oracle = factory(spec.config["comet"], spec.config["condition"], output_path(directory, "comet_load.log"))
        if spec.expected_fingerprint and oracle.fingerprint != spec.expected_fingerprint:
            raise ValueError("COMET fingerprint differs from the reference archive")
        write_json(output_path(directory, "model.json"), oracle.provenance)
        write_json(output_path(directory, "catalog.json"), oracle.catalog)
        write_json(output_path(directory, "domain.json"), oracle.validate_search(encoder))
        preflight = oracle.preflight()
        write_json(output_path(directory, "preflight.json"), preflight)
        preflight_evaluations = oracle.compositions_evaluated
        preflight_seconds = oracle.evaluation_seconds
        print("Real COMET preflight passed.", flush=True)
        metrics = {}
        if search is not None:
            cache = ScoreCache(spec.cache_path, oracle.fingerprint)
            evaluator = Evaluator(oracle, cache, oracle.catalog, directory / "candidates.jsonl",
                                  best_path=directory / "best.json")
            search_started = time.perf_counter()
            metrics = search(SearchContext(spec.config, encoder, evaluator, directory))
            if evaluator.best is None:
                raise ValueError("A completed search must evaluate at least one composition")
        summary = {**base, "status": "completed", "preflight": preflight,
                   "model_fingerprint": oracle.fingerprint,
                   "preflight_composition_evaluations": preflight_evaluations,
                   "comet_forward_batches": getattr(oracle, "forward_batches", 0),
                   "cache_policy": "fresh_memory" if spec.cache_path == ":memory:" else "persistent",
                   "best": evaluator.best if evaluator else None,
                   "counts": evaluator.stats() if evaluator else {}, "timing": timing()}
        overlap = summary.keys() & metrics.keys()
        if overlap:
            raise ValueError(f"Method metrics overwrite common summary fields: {sorted(overlap)}")
        summary.update(metrics)
        write_json(directory / "summary.json", summary)
    except (Exception, KeyboardInterrupt) as error:
        failure = {**base, "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   "error_type": type(error).__name__, "error": str(error), "synthetic_fallback": False,
                   "best": evaluator.best if evaluator else None,
                   "counts": evaluator.stats() if evaluator else {}, "timing": timing()}
        write_json(directory / "summary.json", failure)
        output_path(directory, "traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        if cache is not None:
            cache.close()
    analysis = _analyze(directory, summary, analyze) if analyze is not None else None
    return ExperimentResult(directory, summary, analysis)


def cli_guard(action: Callable[[], int]) -> int:
    """Report CLI errors, including validation failures before a run exists."""
    try:
        return action()
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          "error_type": type(error).__name__, "error": str(error),
                          "synthetic_fallback": False}, ensure_ascii=False), file=sys.stderr)
        return 1
