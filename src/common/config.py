"""Validation and path resolution for settings shared by both search methods."""
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
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")


def load_common_config(path):
    """Read YAML or an archived JSON config; resolve paths relative to that file."""
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a mapping')
    required = {'seed', 'comet', 'condition', 'search', 'output'}
    if not required <= cfg.keys():
        raise ValueError(f'Missing shared settings: {sorted(required-cfg.keys())}')
    keys(cfg['comet'], ('repo', 'checkpoint', 'preprocessing_seed', 'batch_size', 'cpu_threads'), 'comet')
    keys(cfg['condition'], ('cell', 'np_ratio', 'volumetric_ratio'), 'condition')
    keys(cfg['search'], ('candidates', 'ratios'), 'search')
    keys(cfg['search']['candidates'], GROUPS, 'search.candidates')
    keys(cfg['search']['ratios'], ('denominator', 'bounds'), 'ratios')
    keys(cfg['search']['ratios']['bounds'], GROUPS, 'ratios.bounds')
    for group in GROUPS:
        keys(cfg['search']['ratios']['bounds'][group], ('min', 'max', 'step'), group)
    keys(cfg['output'], ('directory', 'cache'), 'output')
    integer(cfg['seed'], 'seed', 0, 2**32-1)
    integer(cfg['comet']['preprocessing_seed'], 'preprocessing_seed', 0, 2**32-1)
    integer(cfg['comet']['batch_size'], 'batch_size', 1, 128)
    integer(cfg['comet']['cpu_threads'], 'cpu_threads', 1, 16)
    positive(cfg['condition']['np_ratio'], 'np_ratio')
    for section, name in (('comet', 'repo'), ('comet', 'checkpoint'), ('output', 'directory'), ('output', 'cache')):
        if (section, name) == ('output', 'cache') and cfg[section][name] == ':memory:':
            continue
        cfg[section][name] = str((path.parent / Path(cfg[section][name]).expanduser()).resolve())
    return cfg
