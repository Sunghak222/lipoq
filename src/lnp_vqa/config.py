"""Read and validate experiment settings without silently accepting typos."""
from pathlib import Path
import math
import yaml

from .encoding import GROUPS


def keys(mapping, expected, label):
    if not isinstance(mapping, dict) or set(mapping) != set(expected):
        raise ValueError(f"{label} requires exactly these keys: {', '.join(expected)}")


def integer(value, label, low=1, high=1_000_000):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    keys(cfg, ("seed", "comet", "condition", "search", "circuit", "optimizer", "output"), "config")
    keys(cfg["comet"], ("repo", "checkpoint", "preprocessing_seed", "batch_size", "cpu_threads"), "comet")
    keys(cfg["condition"], ("cell", "np_ratio", "volumetric_ratio"), "condition")
    keys(cfg["search"], ("candidates", "ratios"), "search")
    keys(cfg["search"]["candidates"], GROUPS, "search.candidates")
    keys(cfg["search"]["ratios"], ("denominator", "bounds"), "ratios")
    keys(cfg["search"]["ratios"]["bounds"], GROUPS, "ratios.bounds")
    for group in GROUPS:
        keys(cfg["search"]["ratios"]["bounds"][group], ("min", "max", "step"), group)
    keys(cfg["circuit"], ("qubits", "depth", "shots"), "circuit")
    keys(cfg["optimizer"], ("iterations", "learning_rate", "perturbation", "alpha", "gamma", "stability_constant"), "optimizer")
    keys(cfg["output"], ("directory", "cache"), "output")
    integer(cfg["seed"], "seed", 0, 2**32-1)
    integer(cfg["comet"]["preprocessing_seed"], "preprocessing_seed", 0, 2**32-1)
    integer(cfg["comet"]["batch_size"], "batch_size", 1, 128)
    integer(cfg["comet"]["cpu_threads"], "cpu_threads", 1, 16)
    integer(cfg["circuit"]["depth"], "depth", 1, 20)
    integer(cfg["circuit"]["shots"], "shots")
    if cfg["circuit"]["qubits"] != "auto":
        integer(cfg["circuit"]["qubits"], "qubits", 1, 20)
    integer(cfg["optimizer"]["iterations"], "iterations", 1, 10000)
    for key in ("learning_rate", "perturbation", "alpha", "gamma"):
        positive(cfg["optimizer"][key], key)
    a = cfg["optimizer"]["stability_constant"]
    if type(a) not in (int, float) or not math.isfinite(a) or a < 0:
        raise ValueError("stability_constant must be finite and nonnegative")
    positive(cfg["condition"]["np_ratio"], "np_ratio")
    # All relative paths are relative to the YAML file, not the shell's cwd.
    for section, name in (("comet", "repo"), ("comet", "checkpoint"), ("output", "directory"), ("output", "cache")):
        p = Path(cfg[section][name]).expanduser()
        cfg[section][name] = str((path.parent / p).resolve())
    return cfg

