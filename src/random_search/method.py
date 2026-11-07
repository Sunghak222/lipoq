"""Adapt uniform random search to the common method contract."""
import time

from common.experiment import SearchContext
from common.results import append_csv, output_path
from .search import random_search


def search(context: SearchContext) -> dict:
    settings = context.config["random_search"]
    budget = settings["budget"]
    started = time.perf_counter()
    comet_before = context.evaluator.oracle.evaluation_seconds
    next_progress = 0

    def on_batch(done, state):
        nonlocal next_progress
        append_csv(output_path(context.directory, "progress.csv"), {
            "unique_compositions": done, "best_so_far_score": state.best["comet_score"],
            "elapsed_search_seconds": time.perf_counter() - started,
            "comet_seconds": state.oracle.evaluation_seconds - comet_before})
        if done >= next_progress or done == budget:
            print(f"Random seed={context.config['seed']}: {done}/{budget} compositions, "
                  f"best={state.best['comet_score']:.6f}", flush=True)
            next_progress = done + max(context.config["comet"]["batch_size"], budget // 10)

    result = random_search(context.encoder, context.evaluator, budget, context.config["seed"],
                           context.config["comet"]["batch_size"], on_batch)
    if len(context.evaluator.seen) != budget:
        raise RuntimeError("Random search did not complete its distinct-composition budget")
    return {"search": result}
