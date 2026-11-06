from pathlib import Path
import pytest
from lnp_vqa.config import load_config


@pytest.fixture
def cfg():
    return load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")

