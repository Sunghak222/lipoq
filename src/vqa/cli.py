"""VQA command parsing; shared machinery owns model and output lifetime."""
import argparse
import json

from common.config import integer
from common.encoding import CompositionEncoder
from common.experiment import ExperimentSpec, cli_guard, run_experiment
from .config import load_config


def run_vqa(config, *, seed=None, probe=False, dry_run=False):
    def execute():
        cfg = load_config(config)
        if seed is not None:
            integer(seed, "seed", 0, 2**32 - 1)
            cfg["seed"] = seed
        spec = ExperimentSpec("probe" if probe else "vqa", cfg, cfg["output"]["cache"])
        if dry_run:
            encoder = CompositionEncoder(cfg["search"])
            print(json.dumps({"status": "dry_run", "method": spec.method, "seed": cfg["seed"],
                              "cell": cfg["condition"]["cell"], "qubits": encoder.num_qubits,
                              "compositions": encoder.cardinality, "shots": cfg["circuit"]["shots"],
                              "maximum_iterations": cfg["optimizer"]["iterations"],
                              "output_root": str(spec.output_root)}, indent=2))
            return 0
        if probe:
            result = run_experiment(spec, None)
        else:
            from .method import search, analyze
            result = run_experiment(spec, search, analyze=analyze)
        print(json.dumps({"status": "completed", "result": str(result.directory / "summary.json"),
                          "best_score": (result.summary.get("best") or {}).get("comet_score"),
                          "timing": result.summary["timing"], "analysis": result.analysis}, indent=2))
        return 0
    return cli_guard(execute)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "probe"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run_vqa(args.config, seed=args.seed, probe=args.command == "probe", dry_run=args.dry_run)
