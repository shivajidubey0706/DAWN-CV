"""Train + evaluate a set of variants over paired seeds, then write summary tables and plots.

    python scripts/run_ablation.py --config configs/short_real.yaml
    python scripts/run_ablation.py --config configs/smoke.yaml --variants b2_erup sg_full --seeds 0
    python scripts/run_ablation.py --config configs/short_real.yaml --summarize-only

Training is skipped when a finished run (status 'done') already exists, so the script
can be re-run to resume an interrupted ablation.
"""
import argparse
import time

import _common  # noqa: F401

from sgerup.config import ROOT, load_yaml, make_config
from sgerup.detector import YOLODetector
from sgerup.engine import evaluate, run_dir, train
from sgerup.report import summarize_experiment
from sgerup.utils import read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--limit", type=int, default=None, help="evaluate on at most N images per dataset")
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    exp_cfg = load_yaml(a.config)
    variants = a.variants or exp_cfg.get("ablation_variants") or ["b0_no_filter", "b2_erup", "sg_full"]
    seeds = a.seeds if a.seeds is not None else exp_cfg.get("seeds", [0])
    base = make_config(a.config, variants[0])
    datasets = a.datasets or base.eval_datasets
    bootstrap = a.bootstrap if a.bootstrap is not None else exp_cfg.get("bootstrap", 0)
    exp_dir = ROOT / "results" / "experiments" / base.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    if not a.summarize_only:
        det = None
        for seed in seeds:
            for v in variants:
                cfg = make_config(a.config, v, {"train": {"seed": seed}})
                rd = run_dir(cfg)
                st = read_json(rd / "status.json") if (rd / "status.json").exists() else {}
                if st.get("state") != "done":
                    print(f"=== train {v} seed {seed}")
                    train(cfg, resume=(rd / "last.pt").exists())
                for ds in datasets:
                    out = exp_dir / f"{v}__seed{seed}__{ds.replace(':', '-')}.json"
                    if out.exists():
                        continue
                    print(f"=== evaluate {v} seed {seed} on {ds}")
                    if det is None:
                        det = YOLODetector(cfg.detector_weights)
                    res = evaluate(cfg, rd / "best.pt", ds, limit=a.limit, bootstrap=bootstrap, det=det)
                    res["train_status"] = read_json(rd / "status.json")
                    hist = rd / "history.json"
                    res["train_history_epochs"] = len(read_json(hist).get("epochs", [])) if hist.exists() else 0
                    write_json(out, res)
    summary = summarize_experiment(exp_dir, reference="b2_erup")
    summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary["config_file"] = a.config
    write_json(exp_dir / "summary.json", summary)
    print(f"summary written to {exp_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
