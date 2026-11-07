"""Incremental, machine-readable outputs; never overwrite another experiment."""
from datetime import datetime
import csv
import json
import os
from pathlib import Path
import secrets
from zoneinfo import ZoneInfo

SCHEMA_VERSION = 2
_SECTIONS = {
    "metadata": {"config.json", "encoding.json", "environment.json", "model.json",
                 "catalog.json", "domain.json", "preflight.json", "reference.json"},
    "metrics": {"history.csv", "evaluations.jsonl", "early_stopping.jsonl", "progress.csv",
                "comparison.json", "comparison.csv", "analysis_status.json"},
    "state": {"circuit.qasm", "initial_parameters.npy", "final_parameters.npy", "optimizer_state.json"},
    "logs": {"comet_load.log", "traceback.txt"},
    "plots": {"plot_status.json"},
}


def _relative_artifact(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Artifact names must stay inside their run directory")
    if len(path.parts) == 1:
        for section, names in _SECTIONS.items():
            if path.name in names:
                return Path(section) / path
    return path


def output_path(directory, name):
    """Path for a new artifact; custom method files may use e.g. state/model.npz."""
    path = Path(directory) / _relative_artifact(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(directory, name):
    """Read structured runs and older flat archives without moving either."""
    root = Path(directory)
    current = root / _relative_artifact(name)
    legacy = root / name
    return current if current.exists() or not legacy.exists() else legacy


def method_name(summary):
    return summary.get("method") or {"run": "vqa", "random": "random", "probe": "probe"}.get(summary.get("command"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def snapshot_lines(path):
    """Return only fully appended CSV/JSONL lines, plus their SHA256."""
    import hashlib
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw = raw.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in raw else b""
    return raw.decode(), hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def create_run_dir(root, prefix):
    stamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%d-%H%M%S")
    path = Path(root) / f"{prefix}-{stamp}-{secrets.token_hex(3)}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def append_jsonl(path, row):
    append_jsonl_many(path, [row])


def append_jsonl_many(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path, rows):
    """Replace a derived table atomically; regenerating reports never appends duplicates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError("A report table must have at least one row")
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)
