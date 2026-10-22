"""Train an SG-ERUP variant (front end end-to-end through the detection loss).

Examples:
    python scripts/train.py --config configs/smoke.yaml --variant sg_full
    python scripts/train.py --config configs/short_real.yaml --variant b2_erup --seed 1
    python scripts/train.py --config configs/paper_full.yaml --variant sg_full --resume
    python scripts/train.py --config configs/smoke.yaml --variant sg_full train.epochs=2 loss.lambda_s=5
"""
import argparse

import _common  # noqa: F401

from sgerup.config import make_config, parse_overrides
from sgerup.engine import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*", help="key=value overrides, e.g. train.epochs=3")
    a = ap.parse_args()
    ov = parse_overrides(a.overrides)
    if a.seed is not None:
        ov.setdefault("train", {})["seed"] = a.seed
    cfg = make_config(a.config, a.variant, ov)
    out = train(cfg, resume=a.resume, device=a.device)
    print(f"run directory: {out}")


if __name__ == "__main__":
    main()
