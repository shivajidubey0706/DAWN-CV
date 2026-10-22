"""Mechanism metrics, Eqs. (13)-(14).

NSR = 1/HW sum_x 1[Y~(x) > tau_hi and Y(x) <= tau_hi]       newly-saturated ratio
BRB = sum_x beta(x) ReLU(Y~(x) - Y(x)) / sum_x beta(x)        bright-region brightening

The paper defines both on I~ (the blended image, before KBL). We also report them on
the final processed image I^ (suffix "_final"), labelled separately.
"""
from __future__ import annotations

import torch

from ..cues import luminance
from ..losses import bright_weight


def newly_saturated_ratio(img: torch.Tensor, processed: torch.Tensor, tau_hi: float = 0.95) -> torch.Tensor:
    y, yt = luminance(img), luminance(processed)
    return ((yt > tau_hi) & (y <= tau_hi)).float().mean(dim=(1, 2, 3))


def bright_region_brightening(img: torch.Tensor, processed: torch.Tensor, r: int = 7, tau_b: float = 0.6,
                              s: float = 0.05) -> torch.Tensor:
    y, yt = luminance(img), luminance(processed)
    beta = bright_weight(img, r, tau_b, s)
    return (beta * torch.relu(yt - y)).sum(dim=(1, 2, 3)) / beta.sum(dim=(1, 2, 3)).clamp_min(1e-8)


@torch.no_grad()
def mechanism_metrics(img, blended, final, r=7, tau_b=0.6, s=0.05, tau_hi=0.95) -> dict[str, float]:
    """Batch-mean NSR / BRB on I~ (paper definition) and on I^."""
    return {
        "nsr": newly_saturated_ratio(img, blended, tau_hi).mean().item(),
        "brb": bright_region_brightening(img, blended, r, tau_b, s).mean().item(),
        "nsr_final": newly_saturated_ratio(img, final, tau_hi).mean().item(),
        "brb_final": bright_region_brightening(img, final, r, tau_b, s).mean().item(),
        "bright_weight_mean": bright_weight(img, r, tau_b, s).mean().item(),
    }
