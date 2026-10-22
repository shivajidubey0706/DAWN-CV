"""Kernel-based local (KBL) filter, Eq. (2) of the paper / Eq. 15 of ERUP-YOLO.

    K_theta(J)_c = clip(J_c * (k1_c conv J_c) + k2_c conv J_c + J_c, 0, 1)

with per-image, per-channel 9x9 kernels k1, k2 in [-1, 1]. The clipping is the
paper's addition (the original does not say how out-of-range values are handled).

Documented assumptions: reflect padding (not specified), and an optional
``kernel_scale`` multiplier (default 1.0 = paper's stated range). With all 81 taps
free in [-1, 1] the convolution can reach +-81; ERUP-YOLO's constraint mechanism
is unspecified (paper Table II, Gap F), so the scale is exposed for ablation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kbl_filter(img: torch.Tensor, kernels: torch.Tensor, kernel_scale: float = 1.0) -> torch.Tensor:
    """img (B, 3, H, W) in [0,1]; kernels (B, 2, 3, k, k) in [-1, 1] with k odd."""
    if img.ndim != 4 or img.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W) RGB image, got {tuple(img.shape)}")
    b, c, h, w = img.shape
    if kernels.ndim != 5 or kernels.shape[:3] != (b, 2, c) or kernels.shape[3] != kernels.shape[4]:
        raise ValueError(f"kernels must be (B, 2, 3, k, k), got {tuple(kernels.shape)}")
    k = kernels.shape[-1]
    if k % 2 == 0:
        raise ValueError("kernel size must be odd")
    kern = kernels.clamp(-1, 1).to(img.dtype) * kernel_scale
    pad = k // 2
    mode = "reflect" if min(h, w) > pad else "replicate"
    x = F.pad(img.reshape(1, b * c, h, w), (pad, pad, pad, pad), mode=mode)
    k1 = kern[:, 0].reshape(b * c, 1, k, k)
    k2 = kern[:, 1].reshape(b * c, 1, k, k)
    c1 = F.conv2d(x, k1, groups=b * c).view(b, c, h, w)
    c2 = F.conv2d(x, k2, groups=b * c).view(b, c, h, w)
    return (img * c1 + c2 + img).clamp(0, 1)
