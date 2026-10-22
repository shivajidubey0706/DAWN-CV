"""Evaluate a trained run (or an untrained variant) on a dataset; writes a JSON result.

Examples:
    python scripts/evaluate.py --run runs/short_real/sg_full/seed0 --dataset dawn
    python scripts/evaluate.py --variant b0_no_filter --dataset dawn --exp baseline_check
    python scripts/evaluate.py --variant b0_no_filter --dataset voc --split test --limit 500
"""
import argparse
from pathlib import Path

import _common  # noqa: F401

from sgerup.config import ROOT, make_config
from sgerup.engine import evaluate
from sgerup.utils import read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run directory containing best.pt and config.json")
    ap.add_argument("--variant", help="evaluate an untrained variant (identity-initialised front end)")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--dataset", default="dawn")
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--exp", default=None, help="experiment name for results/experiments/<exp>/")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if not (a.run or a.variant):
        ap.error("give --run or --variant")
    if a.run:
        run = Path(a.run) if Path(a.run).is_absolute() else ROOT / a.run
        c = read_json(run / "config.json")
        cfg = make_config(None, c["variant"], {k: v for k, v in c.items() if k != "variant"})
        ckpt = run / "best.pt"
        exp = a.exp or cfg.name
    else:
        cfg = make_config(a.config, a.variant)
        ckpt = None
        exp = a.exp or "untrained"
    res = evaluate(cfg, ckpt, a.dataset, a.split, a.limit, a.device, a.bootstrap)
    tag = f"{a.dataset}{'_' + a.split if a.split else ''}{'_n' + str(a.limit) if a.limit else ''}"
    out = ROOT / "results" / "experiments" / exp / f"{cfg.variant}__seed{cfg.train.seed}__{tag}.json"
    write_json(out, res)
    o = res["overall"]
    print(f"{cfg.variant}: mAP@0.5={o['map50']:.4f} mAP@[.5:.95]={o['map50_95']:.4f} "
          f"P={o['precision']} R={o['recall']} NSR={o['mechanism']['nsr']:.4f} -> {out}")


if __name__ == "__main__":
    main()
