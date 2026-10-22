"""E7 cost protocol: parameters, FLOPs, latency (FP32/FP16), FPS, peak memory at 448x448, batch 1.

    python scripts/benchmark_cost.py                       # paper protocol: 50 warm-up, 500 timed
    python scripts/benchmark_cost.py --warmup 10 --iters 50 # quick
"""
import argparse

import _common  # noqa: F401
import torch

from sgerup.config import ROOT, make_config
from sgerup.cost import count_flops, time_fn
from sgerup.detector import YOLODetector
from sgerup.frontend import SGERUPFrontEnd, count_parameters
from sgerup.utils import environment_record, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["b0_no_filter", "b2_erup", "b3_kbl_only", "b4_global_gate", "sg_full"])
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    det = YOLODetector(make_config().detector_weights, dev)
    det.eval_mode()
    x = torch.rand(1, 3, a.imgsz, a.imgsz, device=dev)
    det_flops = count_flops(det.forward, x)
    rows = []
    for v in a.variants:
        cfg = make_config(variant=v)
        fe = SGERUPFrontEnd(cfg.frontend).to(dev).eval()
        fe_flops = count_flops(fe, x) if cfg.frontend.enabled else 0
        full = lambda t, fe=fe: det.forward(fe(t)["out"])  # noqa: E731
        row = {
            "variant": v, "params_frontend": count_parameters(fe), "params_detector": det.n_params,
            "params_total": count_parameters(fe) + det.n_params,
            "gflops_frontend": None if fe_flops is None else fe_flops / 1e9,
            "gflops_detector": None if det_flops is None else det_flops / 1e9,
            "gflops_total": None if None in (fe_flops, det_flops) else (fe_flops + det_flops) / 1e9,
            "latency_fp32": time_fn(full, x, a.warmup, a.iters, half=False),
            "latency_fp16": time_fn(full, x, a.warmup, a.iters, half=True) if dev.type == "cuda" else None,
            "latency_frontend_fp32": time_fn(lambda t, fe=fe: fe(t), x, a.warmup, a.iters) if cfg.frontend.enabled else None,
        }
        rows.append(row)
        print(f"{v}: params {row['params_total']:,}  GFLOPs {row['gflops_total']:.2f}  "
              f"median {row['latency_fp32']['median_ms']:.2f} ms fp32")
    out = {"imgsz": a.imgsz, "batch": 1, "device": str(dev), "environment": environment_record(),
           "flops_convention": "torch.utils.flop_counter: 2 FLOPs per MAC for conv/matmul; element-wise, pooling, "
                               "interpolation and searchsorted not counted",
           "latency_protocol": f"{a.warmup} warm-up + {a.iters} timed iterations, CUDA synchronised, median / p90",
           "note": "Detector is Ultralytics YOLOv3u (not original Darknet YOLOv3).", "rows": rows}
    write_json(ROOT / "results" / "cost" / "cost.json", out)
    print("written results/cost/cost.json")


if __name__ == "__main__":
    main()
