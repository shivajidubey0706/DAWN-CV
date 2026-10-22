"""Reproducibility helpers: seeding, environment record, JSON I/O."""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def environment_record() -> dict:
    rec = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
        "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    try:
        import ultralytics
        rec["ultralytics"] = ultralytics.__version__
    except ImportError:
        rec["ultralytics"] = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rec["gpu"] = p.name
        rec["gpu_memory_gb"] = round(p.total_memory / 2**30, 2)
    rec["cpu"] = platform.processor()
    return rec


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")
    tmp.replace(p)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
