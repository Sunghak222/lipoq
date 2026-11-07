"""Direct COMET evaluation, composition caching, and shot-weighted objectives."""
import math
from pathlib import Path
import time

from .results import append_jsonl_many, write_json


class Evaluator:
    def __init__(self, oracle, cache, catalog, candidates_path, best_path=None):
        self.oracle, self.cache, self.catalog = oracle, cache, catalog
        self.candidates_path = Path(candidates_path)
        self.best_path = Path(best_path) if best_path is not None else None
        self.best = None
        self.seen = set()
        self.sampled_shots = 0
        self.composition_requests = 0
        self.unique_comet_evaluations = 0
        self.persistent_cache_hits = 0
        self.started = time.perf_counter()

    def score(self, compositions):
        """Evaluate unique inputs in first-encounter order; cache every valid score."""
        compositions = list(compositions)
        if not compositions:
            raise ValueError('An evaluation batch must not be empty')
        self.composition_requests += len(compositions)
        unique = {c.key: c for c in compositions}
        scores, pending = {}, []
        for key, composition in unique.items():
            value = self.cache.get(key)
            if value is None:
                pending.append(composition)
            else:
                scores[key] = value
                if key not in self.seen:
                    self.persistent_cache_hits += 1
        if pending:
            predicted = self.oracle.predict(pending)
            if len(predicted) != len(pending):
                raise RuntimeError('COMET prediction count mismatch')
            values = [(c.key, float(score)) for c, score in zip(pending, predicted)]
            self.cache.put_many(values)  # Rejects nonfinite scores before recording.
            scores.update(values)
            self.unique_comet_evaluations += len(values)
        pending_keys = {c.key for c in pending}
        records = []
        for key, composition in unique.items():
            if self.best is None or scores[key] > self.best['comet_score']:
                self.best = {'composition_key': key, 'comet_score': scores[key],
                             **composition.as_dict(self.catalog)}
            if key not in self.seen:
                self.seen.add(key)
                # Molecular structures live once in metadata/catalog.json.
                # Keep the best candidate expanded for convenient reading.
                records.append({
                    'composition_key': key, 'comet_score': scores[key],
                    'lipid_ids': composition.lipid_ids, 'fraction_ticks': composition.ticks,
                    'denominator': composition.denominator,
                    'from_persistent_cache': key not in pending_keys,
                    'unique_evaluations': len(self.seen), 'best_so_far_score': self.best['comet_score'],
                    'elapsed_seconds': time.perf_counter()-self.started})
        if records:
            append_jsonl_many(self.candidates_path, records)
        if self.best_path is not None:
            write_json(self.best_path, self.best)
        return [scores[c.key] for c in compositions]

    def evaluate_counts(self, counts, encoder):
        if not counts or any(type(n) is not int or n <= 0 for n in counts.values()):
            raise ValueError('Counts must contain positive integer frequencies')
        compositions, frequencies = {}, {}
        for bits, count in sorted(counts.items()):
            composition = encoder.decode(bits)
            compositions[composition.key] = composition
            frequencies[composition.key] = frequencies.get(composition.key, 0)+count
        previous_evaluations = self.unique_comet_evaluations
        predicted = self.score(compositions.values())
        scores = dict(zip(compositions, predicted))
        shots = sum(frequencies.values())
        self.sampled_shots += shots
        mean = sum(frequencies[key]*scores[key] for key in frequencies)/shots
        variance = sum(frequencies[key]*(scores[key]-mean)**2 for key in frequencies)/shots
        return {'mean_comet_score': mean,
                'shot_standard_error': math.sqrt(variance/(shots-1)) if shots > 1 else None,
                'batch_best_score': max(scores.values()), 'best_so_far_score': self.best['comet_score'],
                'unique_in_batch': len(compositions),
                'new_comet_evaluations': self.unique_comet_evaluations-previous_evaluations,
                'sampled_shots': shots, 'total_unique_sampled': len(self.seen)}

    def stats(self):
        result = {'unique_sampled_compositions': len(self.seen),
                  'unique_comet_evaluations': self.unique_comet_evaluations,
                  'unique_loaded_from_persistent_cache': self.persistent_cache_hits,
                  'composition_requests': self.composition_requests}
        if self.sampled_shots:
            result.update({'sampled_shots': self.sampled_shots,
                           'duplicate_shots': self.sampled_shots-len(self.seen),
                           'avoided_comet_evaluations': self.sampled_shots-self.unique_comet_evaluations})
        return result
