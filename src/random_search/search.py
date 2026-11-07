"""Uniform sampling of unique compositions, without bitstring alias bias."""
import numpy as np

from common.config import integer


def random_search(encoder, evaluator, budget, seed, batch_size, on_batch=None):
    integer(budget, 'random budget', 1, encoder.cardinality)
    integer(seed, 'random seed', 0, 2**32-1)
    integer(batch_size, 'random batch size', 1, 128)
    indices = np.random.default_rng(seed).choice(encoder.cardinality, size=budget, replace=False)
    for offset in range(0, budget, batch_size):
        batch = indices[offset:offset+batch_size]
        evaluator.score([encoder.composition_at(i) for i in batch])
        if on_batch is not None:
            on_batch(offset+len(batch), evaluator)
    return {'sampling': 'uniform unique compositions without replacement',
            'seed': seed, 'unique_budget': budget}
