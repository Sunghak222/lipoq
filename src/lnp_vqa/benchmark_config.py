"""Configuration for exhaustive analysis, matched random trials, and refinement."""
from pathlib import Path
import yaml

from .config import integer, keys, load_config, positive
from .encoding import CompositionEncoder


def load_benchmark_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    keys(cfg, ('base_config', 'expanded_config', 'seeds', 'output_directory',
               'exhaustive_max_compositions', 'score_tolerance', 'continuous'), 'benchmark')
    if not isinstance(cfg['seeds'], list) or not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Use distinct benchmark seeds')
    for seed in cfg['seeds']:
        integer(seed, 'seed', 0, 2**32-1)
    integer(cfg['exhaustive_max_compositions'], 'exhaustive_max_compositions', 1, 1_000_000)
    positive(cfg['score_tolerance'], 'score_tolerance')
    settings = cfg['continuous']
    keys(settings, ('enabled', 'top_k', 'maxfev_per_start', 'xtol', 'ftol'), 'continuous')
    if type(settings['enabled']) is not bool:
        raise ValueError('continuous.enabled must be a boolean')
    integer(settings['top_k'], 'top_k', 1, 100)
    integer(settings['maxfev_per_start'], 'maxfev_per_start', 4, 10000)
    positive(settings['xtol'], 'xtol')
    positive(settings['ftol'], 'ftol')
    for name in ('base_config', 'expanded_config', 'output_directory'):
        cfg[name] = str((path.parent / cfg[name]).resolve())
    experiments = {name: load_config(cfg[f'{name}_config']) for name in ('base', 'expanded')}
    for name, experiment in experiments.items():
        encoder = CompositionEncoder(experiment['search'])
        if experiment['circuit']['qubits'] not in ('auto', encoder.num_qubits):
            raise ValueError(f'{name}: qubits do not match encoding')
    if CompositionEncoder(experiments['base']['search']).cardinality > cfg['exhaustive_max_compositions']:
        raise ValueError('Base exhaustive space exceeds configured safety limit')
    for name in ('comet', 'condition'):
        if experiments['base'][name] != experiments['expanded'][name]:
            raise ValueError(f'Base and expanded experiments must use identical {name} settings')
    return cfg, experiments
