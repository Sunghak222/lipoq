"""Uniform sampling and cache accounting, with test-only oracle values."""
from common.cache import ScoreCache
from common.evaluation import Evaluator
from common.encoding import CompositionEncoder
from random_search.search import random_search


class CountingOracle:
    def __init__(self, encoder):
        self.catalog = {i: {'name': str(i), 'smiles': 'test-only'} for ids in encoder.candidates for i in ids}
        self.inputs = []

    def predict(self, compositions):
        self.inputs.extend(compositions)
        return [float(c.lipid_ids[0]) for c in compositions]


def test_enumeration_and_uniform_random_ignore_bit_aliases(cfg, tmp_path):
    cfg['search']['candidates']['ionizable'] = [0, 6, 4]
    encoder = CompositionEncoder(cfg['search'])
    enumerated = {c.key for c in encoder.compositions()}
    bit_keys = {encoder.decode(format(i, f'0{encoder.num_qubits}b')).key for i in range(2**encoder.num_qubits)}
    assert len(enumerated) == encoder.cardinality == 192
    assert enumerated == bit_keys
    oracle = CountingOracle(encoder)
    cache = ScoreCache(':memory:', 'test')
    evaluator = Evaluator(oracle, cache, oracle.catalog, tmp_path/'one.jsonl')
    random_search(encoder, evaluator, 73, 42, 16)
    assert len(evaluator.seen) == len(oracle.inputs) == 73
    evaluator.score(oracle.inputs.copy())
    assert len(oracle.inputs) == 73
    other_cache = ScoreCache(':memory:', 'test')
    other = Evaluator(oracle, other_cache, oracle.catalog, tmp_path/'two.jsonl')
    random_search(encoder, other, 73, 42, 16)
    assert (tmp_path/'one.jsonl').read_text().count('composition_key') == 73
    assert oracle.inputs[:73] == oracle.inputs[73:]
    assert len(oracle.inputs) == 146  # No cross-method/seed cache sharing.

    cache.close()
    other_cache.close()
