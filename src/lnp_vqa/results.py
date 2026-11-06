"""Incremental, machine-readable outputs; never overwrite another experiment."""
from datetime import datetime
import csv
import json
import os
from pathlib import Path
import secrets
from zoneinfo import ZoneInfo


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def create_run_dir(root, prefix):
    stamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%d-%H%M%S")
    path = Path(root) / f"{prefix}-{stamp}-{secrets.token_hex(3)}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def append_jsonl(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def append_csv(path, row):
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)

