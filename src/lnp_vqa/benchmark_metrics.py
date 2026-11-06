"""Fresh-cache measurements and exact, post-hoc finite-space diagnostics."""
import math
from pathlib import Path
import time

import numpy as np

from .results import append_jsonl
from .vqa import QuantumSampler


class TrialEvaluator:
    """Independent score cache per method/seed; no reference-table access."""
    def __init__(self, oracle, directory):
        self.oracle = oracle
        self.catalog = oracle.catalog
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.scores, self.compositions = {}, {}
        self.seen = set()
        self.best = None
        self.unique_comet_evaluations = 0
        self.sampled_shots = 0
        self.query_requests = 0
        self.comet_seconds = 0.0
        self.stage_counts = {}
        self.started = time.perf_counter()

    def score(self, compositions, stage='discrete'):
        compositions = list(compositions)
        self.query_requests += len(compositions)
        pending = {c.key: c for c in compositions if c.key not in self.scores}
        if pending:
            start = time.perf_counter()
            predicted = self.oracle.predict(list(pending.values()))
            self.comet_seconds += time.perf_counter()-start
            if len(predicted) != len(pending) or any(not math.isfinite(s) for s in predicted):
                raise RuntimeError('Invalid real COMET prediction batch')
            for (key, composition), value in zip(pending.items(), predicted):
                self.scores[key] = float(value)
                self.compositions[key] = composition
                self.seen.add(key)
                self.unique_comet_evaluations += 1
                self.stage_counts[stage] = self.stage_counts.get(stage, 0)+1
                entry = {'composition_key': key, 'comet_score': float(value),
                         **composition.as_dict(self.catalog)}
                if self.best is None or value > self.best['comet_score']:
                    self.best = entry
                append_jsonl(self.directory/'candidates.jsonl', {
                    **entry, 'stage': stage, 'unique_evaluations': len(self.seen),
                    'best_so_far_score': self.best['comet_score'],
                    'elapsed_seconds': time.perf_counter()-self.started})
        return [self.scores[c.key] for c in compositions]

    def evaluate_counts(self, counts, encoder):
        if not counts or any(type(n) is not int or n <= 0 for n in counts.values()):
            raise ValueError('Counts must be positive integers')
        compositions, frequencies = {}, {}
        for bits, count in sorted(counts.items()):
            c = encoder.decode(bits)
            compositions[c.key] = c
            frequencies[c.key] = frequencies.get(c.key, 0)+count
        previous_count = len(self.seen)
        self.score(compositions.values())
        shots = sum(frequencies.values())
        self.sampled_shots += shots
        mean = sum(n*self.scores[k] for k, n in frequencies.items())/shots
        variance = sum(n*(self.scores[k]-mean)**2 for k, n in frequencies.items())/shots
        return {'mean_comet_score': mean,
                'shot_standard_error': math.sqrt(variance/(shots-1)) if shots > 1 else None,
                'batch_best_score': max(self.scores[k] for k in frequencies),
                'best_so_far_score': self.best['comet_score'], 'unique_in_batch': len(compositions),
                'new_comet_evaluations': len(self.seen)-previous_count,
                'sampled_shots': shots, 'total_unique_sampled': len(self.seen)}

    def stats(self):
        return {'unique_comet_evaluations': len(self.seen), 'query_requests': self.query_requests,
                'cached_requests': self.query_requests-len(self.seen),
                'sampled_shots': self.sampled_shots, 'by_stage': dict(self.stage_counts)}


def random_search(encoder, evaluator, budget, seed, batch_size):
    if not 1 <= budget <= encoder.cardinality:
        raise ValueError('Random budget outside search cardinality')
    # Uniform over unique compositions, NOT encoded bitstrings with modulo aliases.
    indices = np.random.default_rng(seed).choice(encoder.cardinality, size=budget, replace=False)
    for offset in range(0, budget, batch_size):
        evaluator.score([encoder.composition_at(i) for i in indices[offset:offset+batch_size]])
    return {'sampling': 'uniform unique compositions without replacement',
            'seed': seed, 'unique_budget': budget}


def exact_distribution(encoder, depth, theta, reference_scores, tolerance):
    if len(reference_scores) != encoder.cardinality:
        raise ValueError('Exact expectation requires the complete reference score table')
    sampler = QuantumSampler(encoder.num_qubits, depth, 0)
    probabilities = sampler.probabilities(theta)
    by_composition = {}
    for index, probability in enumerate(probabilities):
        key = encoder.decode(format(index, f'0{encoder.num_qubits}b')).key
        by_composition[key] = by_composition.get(key, 0.0)+float(probability)
    if set(by_composition) != set(reference_scores) or not math.isclose(sum(by_composition.values()), 1, abs_tol=1e-10):
        raise RuntimeError('Probability/score support mismatch')
    optimum = max(reference_scores.values())
    exact_keys = [k for k, s in reference_scores.items() if s == optimum]
    near_keys = [k for k, s in reference_scores.items() if optimum-s <= tolerance]
    return {'mean_comet_score': math.fsum(by_composition[k]*s for k, s in reference_scores.items()),
            'probability_of_grid_maximum': math.fsum(by_composition[k] for k in exact_keys),
            'probability_within_tolerance_of_optimum': math.fsum(by_composition[k] for k in near_keys),
            'optimal_composition_count': len(exact_keys), 'near_optimal_composition_count': len(near_keys),
            'score_tolerance': tolerance, 'probabilities': by_composition}


def exact_learning(encoder, cfg, theta0, theta1, reference_scores, best_key, tolerance):
    before = exact_distribution(encoder, cfg['circuit']['depth'], theta0, reference_scores, tolerance)
    after = exact_distribution(encoder, cfg['circuit']['depth'], theta1, reference_scores, tolerance)
    optimum = max(reference_scores.values())
    return {'definition': 'Exact statevector expectation over every grid composition; finite-precision COMET scores',
            'initial': before, 'final': after,
            'mean_improvement': after['mean_comet_score']-before['mean_comet_score'],
            'optimum_probability_change': after['probability_of_grid_maximum']-before['probability_of_grid_maximum'],
            'grid_optimum': optimum, 'found_best_reference_score': reference_scores[best_key],
            'gap_to_grid_optimum': optimum-reference_scores[best_key],
            'found_grid_optimum_within_tolerance': optimum-reference_scores[best_key] <= tolerance}
