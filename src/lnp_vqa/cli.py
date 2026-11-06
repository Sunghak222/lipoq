"""A single configuration and command run a real COMET-backed experiment."""
import argparse
from datetime import datetime
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from zoneinfo import ZoneInfo

from .cache import Evaluator, ScoreCache
from .comet import CometOracle, SCORE_DESCRIPTION
from .config import load_config
from .encoding import CompositionEncoder
from .results import create_run_dir, write_json


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "probe", "benchmark"), help="run an experiment, check inference, or run the comparison suite")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-run", help="optional saved run to audit with benchmark's exhaustive score table")
    args = parser.parse_args()
    if args.command == "benchmark":
        from .benchmark import run_benchmark
        return run_benchmark(args.config, args.reference_run)
    if args.reference_run:
        parser.error("--reference-run is only supported with benchmark")
    run_dir = None
    cache = None
    oracle = None
    evaluator = None
    try:
        cfg = load_config(args.config)
        encoder = CompositionEncoder(cfg["search"])
        if cfg["circuit"]["qubits"] not in ("auto", encoder.num_qubits):
            raise ValueError(f"circuit.qubits must equal encoding size {encoder.num_qubits} or 'auto'")
        run_dir = create_run_dir(cfg["output"]["directory"], args.command)
        print(f"Results: {run_dir}", flush=True)
        write_json(run_dir / "config.json", cfg)
        write_json(run_dir / "encoding.json", encoder.metadata())
        write_json(run_dir / "environment.json", {
            "started_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
            "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        })
        print("Loading verified pretrained COMET on CPU ...", flush=True)
        oracle = CometOracle(cfg["comet"], cfg["condition"], run_dir / "comet_load.log")
        write_json(run_dir / "model.json", oracle.provenance)
        write_json(run_dir / "catalog.json", oracle.catalog)
        write_json(run_dir / "domain.json", oracle.validate_search(encoder))
        preflight = oracle.preflight()
        write_json(run_dir / "preflight.json", preflight)
        print(f"Real COMET preflight: score={preflight['adapter_score']:.6f}, "
              f"upstream parity error={preflight['absolute_error']:.3g}", flush=True)
        preflight_evaluations = oracle.compositions_evaluated
        preflight_seconds = oracle.evaluation_seconds
        summary = {"status": "completed", "command": args.command, "score_description": SCORE_DESCRIPTION,
                   "preflight": preflight, "model_fingerprint": oracle.fingerprint}
        if args.command == "run":
            from .vqa import optimize
            cache = ScoreCache(cfg["output"]["cache"], oracle.fingerprint)
            evaluator = Evaluator(oracle, cache, oracle.catalog, run_dir / "candidates.jsonl")
            summary.update(optimize(cfg, encoder, evaluator, run_dir))
            summary.update({"best": evaluator.best, "counts": evaluator.stats()})
        summary["timing"] = {
            "model_load_seconds": oracle.load_seconds,
            "comet_evaluation_seconds": oracle.evaluation_seconds,
            "comet_forward_seconds": oracle.forward_seconds,
            "preflight_comet_seconds": preflight_seconds,
            "search_comet_seconds": oracle.evaluation_seconds-preflight_seconds,
            "total_seconds": time.perf_counter()-start,
        }
        summary["preflight_composition_evaluations"] = preflight_evaluations
        summary["comet_forward_batches"] = oracle.forward_batches
        write_json(run_dir / "summary.json", summary)
        print(json.dumps({"status": "completed", "result": str(run_dir / "summary.json"),
                          "best_score": summary.get("best", {}).get("comet_score"),
                          "timing": summary["timing"]}, indent=2), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        failure = {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   "error_type": type(error).__name__, "error": str(error),
                   "total_seconds": time.perf_counter()-start, "synthetic_fallback": False}
        if evaluator is not None:
            failure.update({"best": evaluator.best, "counts": evaluator.stats()})
        if oracle is not None:
            failure["comet_evaluation_seconds"] = oracle.evaluation_seconds
        if run_dir is not None:
            write_json(run_dir / "failure.json", failure)
            (run_dir / "traceback.txt").write_text(traceback.format_exc())
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    finally:
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    sys.exit(main())

