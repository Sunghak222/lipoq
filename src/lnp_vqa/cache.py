"""SQLite score cache namespaced by exact model, input data and cell conditions."""
import json
import math
from pathlib import Path
import sqlite3


class ScoreCache:
    def __init__(self, path, namespace):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (namespace TEXT, composition TEXT, score REAL NOT NULL, PRIMARY KEY(namespace, composition))")
        self.namespace = namespace

    def get(self, key):
        row = self.db.execute("SELECT score FROM scores WHERE namespace=? AND composition=?", (self.namespace, key)).fetchone()
        if row is not None and not math.isfinite(row[0]):
            raise ValueError("Nonfinite cached score")
        return None if row is None else row[0]

    def put_many(self, values):
        if any(not math.isfinite(score) for _, score in values):
            raise ValueError("Refusing to cache a nonfinite score")
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?, ?)",
                                [(self.namespace, key, score) for key, score in values])

    def close(self):
        self.db.close()


class Evaluator:
    def __init__(self, oracle, cache, catalog, candidates_path):
        self.oracle, self.cache, self.catalog = oracle, cache, catalog
        self.candidates_path = Path(candidates_path)
        self.best = None
        self.seen = set()
        self.sampled_shots = 0
        self.unique_comet_evaluations = 0
        self.persistent_cache_hits = 0

    def evaluate_counts(self, counts, encoder):
        if not counts or any(type(n) is not int or n <= 0 for n in counts.values()):
            raise ValueError("Counts must contain positive integer frequencies")
        compositions, frequencies = {}, {}
        for bits, count in sorted(counts.items()):
            comp = encoder.decode(bits)
            compositions[comp.key] = comp
            frequencies[comp.key] = frequencies.get(comp.key, 0) + count
        scores, pending = {}, []
        for key, comp in compositions.items():
            score = self.cache.get(key)
            if score is None:
                pending.append(comp)
            else:
                scores[key] = score
                if key not in self.seen:
                    self.persistent_cache_hits += 1
        if pending:
            predicted = self.oracle.predict(pending)
            if len(predicted) != len(pending):
                raise RuntimeError("COMET prediction count mismatch")
            values = [(c.key, float(s)) for c, s in zip(pending, predicted)]
            self.cache.put_many(values)
            scores.update(values)
            self.unique_comet_evaluations += len(values)
        pending_keys = {c.key for c in pending}
        with self.candidates_path.open("a") as stream:
            for key, comp in compositions.items():
                entry = {"composition_key": key, "comet_score": scores[key], **comp.as_dict(self.catalog)}
                if key not in self.seen:
                    stream.write(json.dumps({**entry, "from_persistent_cache": key not in pending_keys}, allow_nan=False) + "\n")
                if self.best is None or entry["comet_score"] > self.best["comet_score"]:
                    self.best = entry
        self.seen.update(compositions)
        shots = sum(frequencies.values())
        self.sampled_shots += shots
        # Each observed shot contributes; NOT an unweighted mean over unique candidates.
        mean = sum(frequencies[key] * scores[key] for key in frequencies) / shots
        variance = sum(frequencies[key] * (scores[key]-mean)**2 for key in frequencies) / shots
        return {"mean_comet_score": mean, "shot_standard_error": math.sqrt(variance/(shots-1)) if shots > 1 else None,
                "batch_best_score": max(scores.values()), "best_so_far_score": self.best["comet_score"],
                "unique_in_batch": len(compositions), "new_comet_evaluations": len(pending),
                "sampled_shots": shots, "total_unique_sampled": len(self.seen)}

    def stats(self):
        return {"sampled_shots": self.sampled_shots, "unique_sampled_compositions": len(self.seen),
                "unique_comet_evaluations": self.unique_comet_evaluations,
                "unique_loaded_from_persistent_cache": self.persistent_cache_hits,
                "duplicate_shots": self.sampled_shots-len(self.seen),
                "avoided_comet_evaluations": self.sampled_shots-self.unique_comet_evaluations}

