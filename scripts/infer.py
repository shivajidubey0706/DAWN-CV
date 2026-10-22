"""Single-image or batch inference. Writes annotated images, processed images and a JSON of detections.

    python scripts/infer.py data/DAWN/Fog/foggy-001.jpg
    python scripts/infer.py data/DAWN/Sand --variant b2_erup --checkpoint runs/short_real/b2_erup/seed0/best.pt
    python scripts/infer.py my_images/ --conf 0.3 --classes protocol5 --out outputs/infer
"""
import argparse
from pathlib import Path

import _common  # noqa: F401

from sgerup.config import ROOT, make_config
from sgerup.detector import YOLODetector
from sgerup.draw import draw_detections
from sgerup.imaging import IMG_EXTS_SET, encode_jpeg, load_image
from sgerup.pipeline import Pipeline
from sgerup.utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="image file or directory")
    ap.add_argument("--variant", default="b0_no_filter")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--classes", default="all", choices=["all", "protocol5"])
    ap.add_argument("--out", default="outputs/infer")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    src = Path(a.source)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS_SET) if src.is_dir() else [src]
    if not files:
        raise SystemExit(f"no images found at {src}")
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(YOLODetector(make_config().detector_weights, a.device))
    all_res = []
    for f in files:
        img = load_image(f)
        r = pipe.run(img, a.variant, a.checkpoint, a.conf, a.iou, a.classes)
        (out / f"{f.stem}_annotated.jpg").write_bytes(encode_jpeg(draw_detections(r["processed"], r["detections"])))
        (out / f"{f.stem}_processed.jpg").write_bytes(encode_jpeg(r["processed"]))
        all_res.append({"image": str(f), "detections": r["detections"], "timing_ms": r["timing_ms"],
                        "mechanism": r["mechanism"], "frontend": r["frontend"]})
        print(f"{f.name}: {len(r['detections'])} detections, {r['timing_ms']['total']:.1f} ms")
    write_json(out / "detections.json", {"variant": a.variant, "checkpoint": a.checkpoint, "conf": a.conf,
                                         "results": all_res})
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
