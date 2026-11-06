"""Accounting/math tests only; these spies cannot be selected by the CLI."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from lnp_vqa.benchmark_metrics import TrialEvaluator, exact_distribution, random_search
from lnp_vqa.continuous import BoundedSimplex, ContinuousComposition, refine
from lnp_vqa.encoding import CompositionEncoder


class CountingOracle:
    def __init__(self, encoder):
        self.catalog = {i: {'name': str(i), 'smiles': 'test-only'} for ids in encoder.candidates for i in ids}
        self.inputs = []

    def predict(self, compositions):
        self.inputs.extend(compositions)
        return [-sum((a-b)**2 for a, b in zip(c.fractions, (0.377, 0.181, 0.427, 0.015))) for c in compositions]


def test_enumeration_and_uniform_random_ignore_bit_aliases(cfg, tmp_path):
    cfg['search']['candidates']['ionizable'] = [0, 6, 4]
    encoder = CompositionEncoder(cfg['search'])
    enumerated = {c.key for c in encoder.compositions()}
    bit_keys = {encoder.decode(format(i, f'0{encoder.num_qubits}b')).key for i in range(2**encoder.num_qubits)}
    assert len(enumerated) == encoder.cardinality == 192
    assert enumerated == bit_keys
    oracle = CountingOracle(encoder)
    evaluator = TrialEvaluator(oracle, tmp_path/'one')
    random_search(encoder, evaluator, 73, 42, 16)
    assert len(evaluator.seen) == len(oracle.inputs) == 73
    evaluator.score(list(evaluator.compositions.values()))
    assert len(oracle.inputs) == 73
    other = TrialEvaluator(oracle, tmp_path/'two')
    random_search(encoder, other, 73, 42, 16)
    assert list(evaluator.scores) == list(other.scores)
    assert len(oracle.inputs) == 146  # No cross-method/seed cache sharing.


def test_exact_statistics_endianness_and_alias_probability(cfg):
    encoder = CompositionEncoder(cfg['search'])
    scores = {c.key: float(i) for i, c in enumerate(encoder.compositions())}
    zero = np.zeros(encoder.num_qubits*2)
    before = exact_distribution(encoder, 1, zero, scores, 1e-5)
    after_angles = zero.copy()
    after_angles[0] = np.pi  # q0 then the CX chain prepares |1111111>.
    after = exact_distribution(encoder, 1, after_angles, scores, 1e-5)
    assert before['mean_comet_score'] == 0
    assert after['mean_comet_score'] == pytest.approx(127)
    assert after['probability_of_grid_maximum'] == pytest.approx(1)
    cfg['search']['candidates']['ionizable'] = [0, 6, 4]
    encoder = CompositionEncoder(cfg['search'])
    scores = {c.key: float(c.lipid_ids[0] == 0) for c in encoder.compositions()}
    theta = np.r_[np.full(encoder.num_qubits, np.pi/2), np.zeros(encoder.num_qubits)]
    result = exact_distribution(encoder, 1, theta, scores, 1e-5)
    assert result['mean_comet_score'] == pytest.approx(0.5)  # Two codes out of four alias to ID 0.
    assert sum(result['probabilities'].values()) == pytest.approx(1)


@pytest.mark.parametrize('expanded', [False, True])
def test_continuous_simplex_roundtrip_and_boundaries(cfg, expanded):
    if expanded:
        from lnp_vqa.config import load_config
        cfg = load_config('configs/expanded.yaml')
    encoder = CompositionEncoder(cfg['search'])
    space = BoundedSimplex(cfg['search']['ratios']['bounds'])
    for coordinates in list(product((0., 1.), repeat=3)) + list(np.random.default_rng(1).random((100, 3))):
        f = space.decode(coordinates)
        assert sum(f) == pytest.approx(1)
        assert np.all(np.array(f) >= space.low-1e-12)
        assert np.all(np.array(f) <= space.high+1e-12)
    for c in encoder.compositions():
        decoded = space.decode(space.encode(c.fractions))
        assert decoded == pytest.approx(c.fractions, abs=1e-12)
        assert ContinuousComposition(c.lipid_ids, decoded).key == c.key


def test_refinement_reuses_cache_keeps_bounds_and_never_loses_best(cfg, tmp_path):
    encoder = CompositionEncoder(cfg['search'])
    oracle = CountingOracle(encoder)
    evaluator = TrialEvaluator(oracle, tmp_path)
    evaluator.score(list(encoder.compositions())[:24])
    count_before, best_before = len(evaluator.seen), evaluator.best['comet_score']
    settings = {'enabled': True, 'top_k': 2, 'maxfev_per_start': 60, 'xtol': .001, 'ftol': 1e-5}
    result = refine(evaluator, cfg['search']['ratios']['bounds'], settings, tmp_path)
    assert result['objective_requests'] <= 120
    assert result['additional_unique_comet_evaluations'] == len(oracle.inputs)-count_before
    assert evaluator.best['comet_score'] > best_before
    assert result['discrete_best']['comet_score'] == best_before
    space = BoundedSimplex(cfg['search']['ratios']['bounds'])
    for composition in oracle.inputs:
        space.validate(composition.fractions)
    assert any(abs(c.fractions[0]/.05-round(c.fractions[0]/.05)) > 1e-4 for c in oracle.inputs[count_before:])


def test_nonfinite_or_infeasible_continuous_inputs_rejected(cfg):
    with pytest.raises(ValueError):
        ContinuousComposition((0, 1, 2, 3), (.3, .2, .5, float('nan')))
    bad = deepcopy(cfg['search']['ratios']['bounds'])
    bad['ionizable']['min'] = 0.99
    with pytest.raises(ValueError):
        BoundedSimplex(bad)
