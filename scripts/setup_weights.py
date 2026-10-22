"""Download the pretrained detector weights (Ultralytics YOLOv3u, COCO) into weights/.

    python scripts/setup_weights.py            # yolov3u.pt (~198 MB)
    python scripts/setup_weights.py --tiny     # also yolov3-tinyu.pt (~24 MB) for quick experiments
"""
import argparse
import os

import _common  # noqa: F401

from sgerup.config import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    wdir = ROOT / "weights"
    wdir.mkdir(exist_ok=True)
    os.chdir(wdir)  # Ultralytics downloads into the working directory
    from ultralytics import YOLO
    for name in ["yolov3u.pt"] + (["yolov3-tinyu.pt"] if a.tiny else []):
        m = YOLO(name)
        print(f"{name}: OK, {sum(p.numel() for p in m.model.parameters()):,} parameters, {len(m.names)} classes")


if __name__ == "__main__":
    main()
