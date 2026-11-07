from copy import deepcopy
import json
import subprocess
import sys
import yaml
import pytest
from common.comet import CometOracle
from vqa.config import load_config


def test_unsupported_cell_and_missing_assets_fail(cfg, tmp_path):
    condition = {**cfg["condition"], "cell": "not-trained"}
    with pytest.raises(ValueError, match="Unsupported cell"):
        CometOracle(cfg["comet"], condition, tmp_path / "log")
    cfg["comet"]["checkpoint"] = str(tmp_path / "absent.pt")
    with pytest.raises(FileNotFoundError):
        CometOracle(cfg["comet"], cfg["condition"], tmp_path / "log")


def test_config_typo_and_negative_shots_rejected(cfg, tmp_path):
    for mutation in (lambda x: x.update({"typo": 1}), lambda x: x["circuit"].update({"shots": -1})):
        bad = deepcopy(cfg)
        mutation(bad)
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError):
            load_config(path)


def test_memory_cache_setting_remains_a_sqlite_memory_database(cfg, tmp_path):
    cfg['output']['cache'] = ':memory:'
    path = tmp_path/'memory.yaml'
    path.write_text(yaml.safe_dump(cfg))
    assert load_config(path)['output']['cache'] == ':memory:'


def test_cli_missing_checkpoint_records_failure_without_score(cfg, tmp_path):
    cfg["comet"]["checkpoint"] = str(tmp_path / "missing.pt")
    cfg["output"]["directory"] = str(tmp_path / "results")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = subprocess.run([sys.executable, "-m", "vqa", "run", "--config", str(path)], capture_output=True, text=True)
    assert result.returncode == 1
    failure = next((tmp_path / "results").rglob("summary.json"))
    content = json.loads(failure.read_text())
    assert content["synthetic_fallback"] is False
    assert content["error_type"] == "FileNotFoundError"
    assert content["status"] == "failed"
    assert (failure.parent / "logs/traceback.txt").is_file()
    assert not (failure.parent / "best.json").exists()

