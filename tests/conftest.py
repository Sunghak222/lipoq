from pathlib import Path
import pytest
from vqa.config import load_config


@pytest.fixture
def cfg():
    """Small test-only grid for fast arithmetic and circuit checks."""
    config = load_config(Path(__file__).resolve().parents[1] / "configs/vqa.yaml")
    config["seed"] = 7
    config["optimizer"].update(resamplings=1, last_avg=1)
    config["optimizer"]["early_stopping"]["enabled"] = False
    config["search"] = {'candidates': {'ionizable': [0, 6],
                    'phospholipid': [1, 13],
                    'sterol': [2, 10],
                    'peg': [3, 11]},
     'ratios': {'denominator': 1000,
                'bounds': {'ionizable': {'min': 0.35, 'max': 0.4, 'step': 0.05},
                           'phospholipid': {'min': 0.15, 'max': 0.2, 'step': 0.05},
                           'sterol': {'min': 0.38, 'max': 0.49, 'step': 0.01},
                           'peg': {'min': 0.01, 'max': 0.02, 'step': 0.01}}}}
    return config
