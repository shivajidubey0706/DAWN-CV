"""SG-ERUP: spatially gated, saturation-aware image-adaptive processing (research prototype).

Tensor convention used throughout the package: float32, NCHW, RGB channel order,
values in [0, 1]. Every module accepts tensors on any device and keeps outputs on
the input's device.

Status: the method is a research hypothesis (see the paper, Sec. VI-VII). Nothing
in this package demonstrates that SG-ERUP improves detection; only measured
results written by ``scripts/evaluate.py`` may be cited.
"""

import os

# Ultralytics may pip-install missing extras at runtime; keep the environment reproducible.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")

__version__ = "0.1.0"

LUMA_WEIGHTS = (0.27, 0.67, 0.06)  # w in Sec. III-E, the weights used by ERUP-YOLO
