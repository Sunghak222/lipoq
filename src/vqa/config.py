"""Strict configuration for the supported VQA workflow."""
import math

from common.config import integer, keys, load_common_config, positive
from common.encoding import CompositionEncoder
from .early_stopping import DEFAULTS as EARLY_STOP_FIELDS


def load_config(path):
    cfg = load_common_config(path)
    keys(cfg, ('seed', 'comet', 'condition', 'search', 'circuit', 'optimizer', 'output'), 'config')
    circuit, settings = cfg['circuit'], cfg['optimizer']
    keys(circuit, ('qubits', 'depth', 'shots'), 'circuit')
    keys(settings, ('iterations', 'learning_rate', 'perturbation', 'alpha', 'gamma', 'stability_constant',
                    'resamplings', 'paired_sampling', 'trust_region', 'last_avg', 'initialization',
                    'early_stopping'), 'optimizer')
    integer(circuit['depth'], 'circuit.depth', 1, 20)
    integer(circuit['shots'], 'circuit.shots')
    encoder = CompositionEncoder(cfg['search'])
    if not 1 <= encoder.num_qubits <= 20:
        raise ValueError('The local VQA simulator requires 1–20 qubits; adjust the search space')
    if circuit['qubits'] != 'auto':
        integer(circuit['qubits'], 'circuit.qubits', 1, 20)
        if circuit['qubits'] != encoder.num_qubits:
            raise ValueError(f'circuit.qubits must equal encoding size {encoder.num_qubits}')
    integer(settings['iterations'], 'optimizer.iterations', 1, 10000)
    integer(settings['resamplings'], 'optimizer.resamplings', 1, 64)
    integer(settings['last_avg'], 'optimizer.last_avg', 1, settings['iterations'])
    for key in ('paired_sampling', 'trust_region'):
        if type(settings[key]) is not bool:
            raise ValueError(f'optimizer.{key} must be a boolean')
    if settings['initialization'] not in ('random', 'uniform'):
        raise ValueError('optimizer.initialization must be random or uniform')
    for key in ('learning_rate', 'perturbation', 'alpha', 'gamma'):
        positive(settings[key], f'optimizer.{key}')
    a = settings['stability_constant']
    if type(a) not in (int, float) or not math.isfinite(a) or a < 0:
        raise ValueError('optimizer.stability_constant must be finite and nonnegative')
    stop = settings['early_stopping']
    keys(stop, EARLY_STOP_FIELDS, 'optimizer.early_stopping')
    if type(stop['enabled']) is not bool:
        raise ValueError('early_stopping.enabled must be a boolean')
    for key in ('min_iterations', 'patience', 'window'):
        integer(stop[key], f'early_stopping.{key}', 1, 10000)
    if stop['min_iterations'] < stop['window']:
        raise ValueError('early_stopping.min_iterations must be >= window')
    if stop['enabled'] and stop['min_iterations'] > settings['iterations']:
        raise ValueError('early_stopping.min_iterations exceeds the iteration budget')
    for key in ('min_delta', 'confidence_z'):
        if type(stop[key]) not in (int, float) or not math.isfinite(stop[key]) or stop[key] < 0:
            raise ValueError(f'early_stopping.{key} must be finite and nonnegative')
    return cfg
