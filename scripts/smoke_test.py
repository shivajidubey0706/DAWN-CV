"""Final reproducible smoke test: image -> SG-ERUP processing -> real detection, with checks.

    python scripts/smoke_test.py [--device cpu] [--image data/DAWN/Sand/sand_storm-151.jpg]

Writes logs/smoke_test.json and outputs/smoke/*.jpg. Exit code 1 if any check fails.
"""
import argparse
import json
import time

import _common  # noqa: F401
import numpy as np
import torch

from sgerup.config import ROOT, make_config
from sgerup.data.datasets import load_samples
from sgerup.detector import YOLODetector
from sgerup.draw import draw_detections
from sgerup.imaging import encode_jpeg, load_image
from sgerup.metrics.detection import box_iou
from sgerup.pipeline import Pipeline
from sgerup.utils import environment_record, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--image", default="data/DAWN/Sand/sand_storm-151.jpg")
    a = ap.parse_args()
    seed_everything(0)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""))

    img_path = ROOT / a.image
    img = load_image(img_path)
    check("image loads", img.ndim == 3 and img.shape[2] == 3, f"{img.shape}")
    det = YOLODetector(make_config().detector_weights, a.device)
    check("detector weights load", det.n_params > 1e8, f"{det.n_params:,} params on {det.device}")
    pipe = Pipeline(det, 448)
    out_dir = ROOT / "outputs" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    sid = f"{img_path.parent.name}/{img_path.stem}"
    gt = next((s for s in load_samples("dawn") if s.id == sid), None)
    ckpt = ROOT / "runs" / "short_real" / "sg_full" / "seed0" / "best.pt"
    runs = [("b0_no_filter", None), ("sg_full", None)] + ([("sg_full", str(ckpt))] if ckpt.exists() else [])
    summary = {}
    for variant, ck in runs:
        t0 = time.perf_counter()
        r = pipe.run(img, variant, ck, conf=0.25, iou=0.6, class_filter="protocol5", intermediates=True)
        dt = (time.perf_counter() - t0) * 1e3
        tag = f"{variant}{'_trained' if ck else ''}"
        boxes = np.array([d["box"] for d in r["detections"]]).reshape(-1, 4)
        h, w = img.shape[:2]
        check(f"[{tag}] processed image shape/range", r["processed"].shape == img.shape)
        check(f"[{tag}] boxes inside image, scores in [0.25,1]",
              bool(len(boxes) == 0 or ((boxes[:, 0] >= 0) & (boxes[:, 2] <= w) & (boxes[:, 1] >= 0) & (boxes[:, 3] <= h)
                                       & (boxes[:, 2] > boxes[:, 0])).all()) and all(0.25 <= d["score"] <= 1 for d in r["detections"]),
              f"{len(boxes)} detections")
        gate = r["intermediates"]["gate"]
        check(f"[{tag}] gate in [0,1]", float(gate.min()) >= 0 and float(gate.max()) <= 1, f"mean {gate.mean():.3f}")
        recall = None
        if gt is not None and len(boxes):
            keep = [i for i, lab in enumerate(gt.labels) if lab != "__ignore__"]
            iou = box_iou(gt.boxes[keep], boxes)
            recall = float((iou.max(1) >= 0.5).mean()) if len(keep) else None
        (out_dir / f"{tag}.jpg").write_bytes(encode_jpeg(draw_detections(r["processed"], r["detections"])))
        summary[tag] = {"n_detections": len(boxes), "time_ms": round(dt, 1), "mechanism": r["mechanism"],
                        "frontend": r["frontend"], "recall_vs_gt_iou50": recall, "classes": sorted({d["class_name"] for d in r["detections"]})}
        print(f"  {tag}: {len(boxes)} dets, recall vs GT {recall}, {dt:.0f} ms, NSR {r['mechanism']['nsr']:.4f}")
    check("detector finds GT objects on the test image (b0)", (summary["b0_no_filter"]["recall_vs_gt_iou50"] or 0) > 0.3,
          f"recall {summary['b0_no_filter']['recall_vs_gt_iou50']}")
    rep = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "image": a.image, "device": a.device, "checks": checks,
           "runs": summary, "environment": environment_record(), "torch_threads": torch.get_num_threads()}
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "smoke_test.json").write_text(json.dumps(rep, indent=2, default=str))
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    print(f"{len(checks) - n_fail}/{len(checks)} checks passed -> logs/smoke_test.json, images in outputs/smoke/")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
