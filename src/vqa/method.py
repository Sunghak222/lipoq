"""Adapt the VQA optimizer to the common search-method contract."""
from common.experiment import SearchContext
from common.plotting import export_automatically
from .optimizer import optimize
from .plotting import plot_completed_run


def search(context: SearchContext) -> dict:
    return optimize(context.config, context.encoder, context.evaluator, context.directory)


def analyze(directory, summary) -> dict:
    return {"plots": export_automatically(directory, lambda: plot_completed_run(directory))}
