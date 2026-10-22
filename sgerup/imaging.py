"""Image I/O and letterboxing shared by training, evaluation, the CLI and the API."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

PAD_VALUE = 114  # Ultralytics letterbox grey
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
IMG_EXTS_SET = {".jpg", ".jpeg", ".png", ".webp"}


class ImageError(ValueError):
    """Raised for unreadable, corrupted or unsupported images (mapped to HTTP 400 by the API)."""


def load_image(src: str | Path | bytes) -> np.ndarray:
    """Return an RGB uint8 array (H, W, 3). EXIF orientation is applied."""
    try:
        im = Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
        im.load()
    except Exception as e:  # noqa: BLE001
        raise ImageError(f"Could not decode image: {e.__class__.__name__}: {e}") from e
    im = ImageOps.exif_transpose(im)
    return np.asarray(im.convert("RGB"))


def validate_upload(data: bytes, max_bytes: int) -> str:
    """Check size, decodability and format of uploaded bytes. Returns the PIL format name."""
    if not data:
        raise ImageError("Empty file.")
    if len(data) > max_bytes:
        raise ImageError(f"File too large ({len(data) / 1e6:.1f} MB); limit is {max_bytes / 1e6:.0f} MB.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
    except Exception as e:  # noqa: BLE001
        raise ImageError("File is not a readable image or is corrupted.") from e
    if fmt not in ALLOWED_FORMATS:
        raise ImageError(f"Unsupported format {fmt}; use JPG, PNG or WebP.")
    return fmt


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio so the long side is `size`, pad to size x size."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = max(1, round(h * r)), max(1, round(w * r))
    resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    out = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top : top + nh, left : left + nw] = resized
    return out, r, (left, top)


def to_tensor(img: np.ndarray, device: torch.device | str = "cpu") -> torch.Tensor:
    """uint8 HWC RGB -> float32 1x3xHxW in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """1x3xHxW or 3xHxW float in [0,1] -> HWC uint8."""
    if t.ndim == 4:
        t = t[0]
    return (t.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def unletterbox_boxes(boxes: np.ndarray, r: float, pad: tuple[int, int], w: int, h: int) -> np.ndarray:
    b = boxes.copy()
    b[:, [0, 2]] = ((b[:, [0, 2]] - pad[0]) / r).clip(0, w)
    b[:, [1, 3]] = ((b[:, [1, 3]] - pad[1]) / r).clip(0, h)
    return b


def crop_letterbox(img: np.ndarray, r: float, pad: tuple[int, int], w: int, h: int) -> np.ndarray:
    """Remove the padding from a letterboxed array and resize back to the original size."""
    nh, nw = max(1, round(h * r)), max(1, round(w * r))
    crop = img[pad[1] : pad[1] + nh, pad[0] : pad[0] + nw]
    return np.asarray(Image.fromarray(crop).resize((w, h), Image.BILINEAR))


def colormap(x: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """Map a 2-D float array to an RGB uint8 heatmap (matplotlib 'magma' if available)."""
    v = np.clip((x - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    try:
        from matplotlib import colormaps
        return (colormaps["magma"](v)[..., :3] * 255).astype(np.uint8)
    except Exception:  # noqa: BLE001
        g = (v * 255).astype(np.uint8)
        return np.stack([g, g, g], -1)


def encode_png(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
