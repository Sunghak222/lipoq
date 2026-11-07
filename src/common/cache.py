"""SQLite score cache namespaced by exact model, input data and cell conditions."""
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
