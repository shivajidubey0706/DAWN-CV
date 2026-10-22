"""YOLOv3 detector wrapper (Ultralytics YOLOv3u weights).

Deviation from the paper: [8] uses the original YOLOv3 (anchor-based Darknet head and
loss). The detector here is Ultralytics' YOLOv3u: the YOLOv3 Darknet-53 backbone
and neck with Ultralytics' anchor-free head and v8 detection loss (TAL assignment,
BCE + CIoU + DFL), COCO-pretrained. It is used both for inference and, through
``loss()``, as L_det when training the front end end-to-end.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .config import ROOT


class DetectorError(RuntimeError):
    pass


def resolve_weights(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


class YOLODetector:
    def __init__(self, weights: str | Path, device: str | torch.device | None = None):
        p = resolve_weights(weights)
        if not p.exists():
            raise DetectorError(
                f"Detector weights not found at {p}. Download with: python scripts/setup_weights.py "
                "(fetches yolov3u.pt from the Ultralytics GitHub release)."
            )
        if p.stat().st_size < 1_000_000:
            raise DetectorError(f"Detector weights at {p} look truncated ({p.stat().st_size} bytes).")
        try:
            from ultralytics import YOLO
            yolo = YOLO(str(p))
        except ImportError as e:
            raise DetectorError("The 'ultralytics' package is not installed (pip install -r requirements.txt).") from e
        except Exception as e:  # noqa: BLE001
            raise DetectorError(f"Could not load detector weights {p.name}: {e}") from e
        if getattr(yolo, "task", None) != "detect":
            raise DetectorError(f"{p.name} is a '{yolo.task}' model, not a detection model.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = yolo.model.to(self.device).eval()
        self.names: dict[int, str] = dict(self.model.names)
        self.weights = p
        self.stride = int(self.model.stride.max())
        for q in self.model.parameters():
            q.requires_grad_(False)
        self._criterion_ready = False

    @property
    def n_params(self) -> int:
        return sum(q.numel() for q in self.model.parameters())

    def train_mode(self, train_detector: bool = False) -> None:
        """Detect head in training-output mode. When frozen, BatchNorm keeps its running stats."""
        self.model.train()
        for q in self.model.parameters():
            q.requires_grad_(train_detector)
        if not train_detector:
            for m in self.model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()

    def eval_mode(self) -> None:
        self.model.eval()

    def forward(self, img: torch.Tensor):
        return self.model(img)

    def loss(self, preds, batch: dict) -> tuple[torch.Tensor, dict]:
        """Ultralytics v8 detection loss, divided by batch size so it is a per-image mean."""
        if not self._criterion_ready:
            from ultralytics.cfg import get_cfg
            self.model.args = get_cfg()
            self.model.criterion = None
            self._criterion_ready = True
        loss, items = self.model.loss(batch, preds)
        bs = batch["img"].shape[0]
        return loss.sum() / bs, {k: float(v) for k, v in items.items()}

    @torch.no_grad()
    def predict(self, img: torch.Tensor, conf: float = 0.25, iou: float = 0.6, classes: list[int] | None = None,
                max_det: int = 300) -> list[torch.Tensor]:
        """img (B,3,S,S) in [0,1] -> per image (N, 6) [x1, y1, x2, y2, score, cls] in letterbox pixels."""
        from ultralytics.utils.nms import non_max_suppression
        self.model.eval()
        preds = self.model(img)
        return non_max_suppression(preds, conf_thres=conf, iou_thres=iou, classes=classes, max_det=max_det)
