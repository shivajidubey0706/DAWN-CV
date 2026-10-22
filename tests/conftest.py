import os
import sys
import warnings
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SGERUP_PRELOAD_DETECTOR", "false")
os.environ.setdefault("SGERUP_DEVICE", "cpu")
warnings.filterwarnings("ignore", category=UserWarning)

WEIGHTS = ROOT / "weights" / "yolov3u.pt"
DAWN = ROOT / "data" / "DAWN"
needs_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="detector weights not downloaded")
needs_dawn = pytest.mark.skipif(not (DAWN / "Fog").exists(), reason="DAWN not extracted")


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(0)


@pytest.fixture
def img(rng):
    return torch.rand(2, 3, 64, 80, generator=rng)
