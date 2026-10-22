"""Parameter-free cue maps, Eqs. (3)-(5).

    mu(x)    = (A_r * Y)(x)                          local luminance
    delta(x) = min_{y in Omega_r(x)} min_c I_c(y)     dark channel [He et al.]
    rho(x)   = (A_r * 1[max_c I_c > tau_hi])(x)       local near-saturated fraction

A_r is a box filter of radius r (window (2r+1)^2). Borders: the box filter averages
only over in-image pixels (count_include_pad=False); the min-filter ignores
out-of-image pixels. The radius r and tau_hi are unspecified in the paper; defaults
live in configs/ and are hyperparameters to fix on the validation split.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import LUMA_WEIGHTS


def luminance(img: torch.Tensor) -> torch.Tensor:
    """Y = w^T I with w = (0.27, 0.67, 0.06). (B,3,H,W) -> (B,1,H,W)."""
    w = torch.tensor(LUMA_WEIGHTS, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    return (img * w).sum(dim=1, keepdim=True)


def box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return x
    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


def local_luminance(img: torch.Tensor, r: int) -> torch.Tensor:
    return box_filter(luminance(img), r)


def dark_channel(img: torch.Tensor, r: int) -> torch.Tensor:
    m = img.min(dim=1, keepdim=True).values
    if r <= 0:
        return m
    return -F.max_pool2d(-m, 2 * r + 1, stride=1, padding=r)


def saturation_fraction(img: torch.Tensor, r: int, tau_hi: float) -> torch.Tensor:
    ind = (img.max(dim=1, keepdim=True).values > tau_hi).to(img.dtype)
    return box_filter(ind, r)


def cue_maps(img: torch.Tensor, r: int = 7, tau_hi: float = 0.95) -> torch.Tensor:
    """Stack [mu; delta; rho] at full resolution: (B, 3, H, W). No learnable parameters."""
    with torch.no_grad():
        return torch.cat(
            [local_luminance(img, r), dark_channel(img, r), saturation_fraction(img, r, tau_hi)], dim=1
        )


def pool_to_grid(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """P_{h x w}: average-pool a map to the gate grid."""
    return F.adaptive_avg_pool2d(x, size)
