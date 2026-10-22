"""Bounding-box rendering for downloads and the CLI (the web UI draws its own overlay)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [(34, 211, 238), (167, 139, 250), (52, 211, 153), (251, 191, 36), (248, 113, 113), (96, 165, 250)]


def draw_detections(img: np.ndarray, detections: list[dict], gt: list[dict] | None = None) -> np.ndarray:
    im = Image.fromarray(img.copy())
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(im.size) / 400))
    try:
        font = ImageFont.truetype("arial.ttf", max(12, lw * 6))
    except OSError:
        font = ImageFont.load_default()
    for g in gt or []:
        d.rectangle(g["box"], outline=(255, 255, 255), width=max(1, lw - 1))
    for det in detections:
        c = PALETTE[det["class_id"] % len(PALETTE)]
        x1, y1, x2, y2 = det["box"]
        d.rectangle([x1, y1, x2, y2], outline=c, width=lw)
        label = f"{det['class_name']} {det['score']:.2f}"
        tb = d.textbbox((x1, y1), label, font=font)
        d.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=c)
        d.text((x1, y1), label, fill=(10, 14, 26), font=font)
    return np.asarray(im)
