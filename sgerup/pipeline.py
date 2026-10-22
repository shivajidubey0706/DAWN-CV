"""End-to-end inference: letterbox -> SG-ERUP front end -> YOLO detector -> boxes in original pixels."""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import ROOT, FrontEndConfig, RunConfig, make_config
from .data.class_maps import CLASS_SETS, COCO_IDS
from .detector import YOLODetector
from .frontend import SGERUPFrontEnd, count_parameters
from .imaging import crop_letterbox, letterbox, to_tensor, to_uint8, unletterbox_boxes
from .metrics.mechanism import mechanism_metrics


class CheckpointError(RuntimeError):
    pass


def load_frontend_checkpoint(path: str | Path, device="cpu") -> tuple[dict, dict]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise CheckpointError(f"Front-end checkpoint not found: {p}")
    try:
        ck = torch.load(p, map_location=device, weights_only=True)
    except Exception as e:  # noqa: BLE001
        raise CheckpointError(f"Could not read checkpoint {p.name}: {e}") from e
    for key in ("frontend", "config"):
        if key not in ck:
            raise CheckpointError(f"Checkpoint {p.name} is missing '{key}'")
    return ck["frontend"], ck


def file_sha(path: Path, n: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:16]


class Pipeline:
    """Owns one detector and builds/caches a front end per (variant, checkpoint)."""

    def __init__(self, detector: YOLODetector, imgsz: int = 448):
        self.detector = detector
        self.device = detector.device
        self.imgsz = imgsz
        self._frontends: dict[str, tuple[SGERUPFrontEnd, dict]] = {}

    def frontend(self, variant: str, checkpoint: str | None = None,
                 gate_override: str | None = None) -> tuple[SGERUPFrontEnd, dict]:
        """gate_override replaces the gate mode (e.g. 'fixed' to visualise B5's analytic gate on top of
        a checkpoint's BPW curve); weights that no longer apply are dropped."""
        key = f"{variant}|{checkpoint}|{gate_override}"
        if key not in self._frontends:
            info = {"variant": variant, "checkpoint": checkpoint, "trained": False, "gate_override": gate_override}
            if checkpoint:
                state, ck = load_frontend_checkpoint(checkpoint, self.device)
                fe_cfg = FrontEndConfig(**ck["config"]["frontend"])
                if gate_override:
                    fe_cfg.gate_mode = gate_override
                fe = SGERUPFrontEnd(fe_cfg)
                fe.load_state_dict(state, strict=not gate_override)
                info.update(trained=bool(ck.get("trained", True)), epoch=ck.get("epoch"), run=ck.get("run_name"),
                            seed=ck["config"]["train"]["seed"], variant=ck["config"]["variant"])
            else:
                fe_cfg = make_config(variant=variant).frontend
                if gate_override:
                    fe_cfg.gate_mode = gate_override
                fe = SGERUPFrontEnd(fe_cfg)
            fe.to(self.device).eval()
            info["params"] = count_parameters(fe)
            self._frontends[key] = (fe, info)
        return self._frontends[key]

    @torch.no_grad()
    def run(self, img: np.ndarray, variant: str = "b0_no_filter", checkpoint: str | None = None,
            conf: float = 0.25, iou: float = 0.6, class_filter: str = "all", intermediates: bool = False,
            theta_b: torch.Tensor | None = None, gate_override: str | None = None) -> dict:
        h, w = img.shape[:2]
        fe, info = self.frontend(variant, checkpoint, gate_override)
        t0 = time.perf_counter()
        lb, r, pad = letterbox(img, self.imgsz)
        x = to_tensor(lb, self.device)
        sync = torch.cuda.synchronize if self.device.type == "cuda" else (lambda: None)
        sync()
        t1 = time.perf_counter()
        fo = fe(x, theta_b=None if theta_b is None else theta_b.to(self.device).view(1, 12))
        sync()
        t2 = time.perf_counter()
        classes = None
        if class_filter == "protocol5":
            classes = [COCO_IDS[c] for c in CLASS_SETS["protocol5"]]
        det = self.detector.predict(fo["out"], conf=conf, iou=iou, classes=classes)[0].cpu().numpy()
        sync()
        t3 = time.perf_counter()
        boxes = unletterbox_boxes(det[:, :4], r, pad, w, h) if len(det) else np.zeros((0, 4))
        detections = [
            {"box": [round(float(v), 1) for v in b], "score": round(float(d[4]), 4), "class_id": int(d[5]),
             "class_name": self.detector.names.get(int(d[5]), str(int(d[5])))}
            for b, d in zip(boxes, det)
        ]
        out = {
            "detections": detections,
            "processed": crop_letterbox(to_uint8(fo["out"]), r, pad, w, h),
            "frontend": info,
            "timing_ms": {"preprocess": round((t1 - t0) * 1e3, 2), "frontend": round((t2 - t1) * 1e3, 2),
                          "detector_and_nms": round((t3 - t2) * 1e3, 2), "total": round((t3 - t0) * 1e3, 2)},
            "device": str(self.device), "imgsz": self.imgsz,
        }
        cfg = fe.cfg
        out["mechanism"] = {k: float(v) for k, v in mechanism_metrics(
            x, fo["blended"], fo["out"], cfg.cue_radius, cfg.tau_b, cfg.s, cfg.tau_hi).items()}
        if intermediates:
            crop = lambda t: crop_letterbox(to_uint8(t), r, pad, w, h)  # noqa: E731
            gate = fo["gate"][0, 0].float().cpu().numpy()
            cues = fo["cues"][0].float().cpu().numpy()
            out["intermediates"] = {
                "bpw": crop(fo["bpw"]), "blended": crop(fo["blended"]),
                "gate": _crop_map(gate, r, pad, w, h), "luminance": _crop_map(cues[0], r, pad, w, h),
                "dark_channel": _crop_map(cues[1], r, pad, w, h), "saturation": _crop_map(cues[2], r, pad, w, h),
                "gate_lowres": fo["gate_lowres"][0, 0].float().cpu().numpy(),
            }
            if "theta_b" in fo:
                out["theta_b"] = fo["theta_b"][0].float().cpu().tolist()
        return out


def _crop_map(m: np.ndarray, r: float, pad, w: int, h: int) -> np.ndarray:
    from PIL import Image
    nh, nw = max(1, round(h * r)), max(1, round(w * r))
    crop = m[pad[1] : pad[1] + nh, pad[0] : pad[0] + nw]
    return np.asarray(Image.fromarray(crop.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR))


def config_summary(cfg: RunConfig) -> dict:
    return asdict(cfg)


def manual_bpw_params(strength: float) -> torch.Tensor:
    """Illustrative curve for the web UI (NOT a model output): strength > 0 brightens, < 0 darkens.
    Both control points of every channel move by `strength` towards (0, 1) or (1, 0)."""
    s = max(-1.0, min(1.0, float(strength)))
    return torch.tensor([[-s, s] * 6], dtype=torch.float32)
