"""Run random search alone, matching the budget of a completed VQA archive."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from common.config import integer
from common.experiment import ExperimentSpec, cli_guard, run_experiment
from common.plotting import export_automatically
from common.comparison import plot_comparison, write_comparison
from .method import search
from .reference import load_reference


def run_random(source, output=None, seed=None, dry_run=False):
    def execute():
        reference = load_reference(source)
        cfg = deepcopy(reference["config"])
        cfg["seed"] = cfg["seed"] if seed is None else seed
        integer(cfg["seed"], "random seed", 0, 2**32 - 1)
        # A random experiment retains only shared inputs and its own settings.
        cfg.pop("circuit", None)
        cfg.pop("optimizer", None)
        cfg["random_search"] = {"budget": reference["budget"]}
        cfg["output"]["cache"] = ":memory:"
        # Older archives stored the seed directory as output.directory.
        # New runs always store the experiment root above method and seed.
        root = Path(cfg["output"]["directory"])
        if reference["summary"].get("schema_version", 1) == 1 and root.name == f"seed-{reference['config']['seed']}":
            cfg["output"]["directory"] = str(root.parent)
        if output is not None:
            cfg["output"]["directory"] = str(Path(output).resolve())
        spec = ExperimentSpec("random", cfg, expected_fingerprint=reference["summary"]["model_fingerprint"],
                              reference={"path": str(reference["path"]), "sha256": reference["hashes"]})
        if dry_run:
            print(json.dumps({"status": "dry_run", "method": "random", "seed": cfg["seed"],
                              "reference_run": str(reference["path"]), "cell": cfg["condition"]["cell"],
                              "search_space_size": reference["encoder"].cardinality,
                              "unique_budget": reference["budget"], "output_root": str(spec.output_root),
                              "cache": "fresh in-memory cache"}, indent=2))
            return 0

        def analyze(directory, summary):
            comparison = write_comparison(reference, summary, directory)
            plots = export_automatically(directory, lambda: plot_comparison(reference, directory))
            return {"reference_minus_candidate_best": comparison["reference_minus_candidate_best"], "plots": plots}

        result = run_experiment(spec, search, analyze=analyze)
        print(json.dumps({"status": "completed", "result": str(result.directory / "summary.json"),
                          "best_score": result.summary["best"]["comet_score"],
                          "matched_unique_budget": reference["budget"],
                          "timing_directly_comparable": False, "analysis": result.analysis}, indent=2))
        return 0
    return cli_guard(execute)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True,
                        help="Completed VQA run or directory containing runs; selects latest completed VQA.")
    parser.add_argument("--output", type=Path, help="Experiment root; appends random/seed-<seed>/run-<id>.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run_random(args.reference_run, args.output, args.seed, args.dry_run)
