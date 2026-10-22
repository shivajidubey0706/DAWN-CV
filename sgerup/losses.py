"""Training losses, Eqs. (9)-(11).

L_sat = 1/HW sum_x [ beta(x) ReLU(Y~(x) - Y(x))^2 + eta ReLU(Y~(x) - tau_hi) 1[Y(x) <= tau_hi] ]
beta(x) = sigmoid((mu(x) - tau_b) / s)
L_tv  = 1/(hw) sum_{i,j} |g_{i+1,j} - g_{i,j}| + |g_{i,j+1} - g_{i,j}|
L     = L_det + lambda_s L_sat + lambda_tv L_tv

beta and the indicator depend only on the input I, so L_sat is differentiable with
respect to theta_B and g (through Y~).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .cues import box_filter, luminance


def bright_weight(img: torch.Tensor, r: int, tau_b: float, s: float) -> torch.Tensor:
    """beta(x) = sigmoid((mu(x) - tau_b) / s), computed from the input only (no gradient)."""
    with torch.no_grad():
        return torch.sigmoid((box_filter(luminance(img), r) - tau_b) / s)


def saturation_loss(img: torch.Tensor, processed: torch.Tensor, r: int = 7, tau_b: float = 0.6,
                    s: float = 0.05, tau_hi: float = 0.95, eta: float = 1.0) -> torch.Tensor:
    y = luminance(img).detach()
    yt = luminance(processed)
    beta = bright_weight(img, r, tau_b, s)
    term1 = beta * F.relu(yt - y) ** 2
    term2 = eta * F.relu(yt - tau_hi) * (y <= tau_hi).to(yt.dtype)
    return (term1 + term2).mean()


def gate_tv_loss(g: torch.Tensor) -> torch.Tensor:
    """g: (B, 1, h, w). Returns batch mean of Eq. (10); zero for a 1x1 gate."""
    h, w = g.shape[-2:]
    dv = (g[..., 1:, :] - g[..., :-1, :]).abs().sum(dim=(-1, -2, -3))
    dh = (g[..., :, 1:] - g[..., :, :-1]).abs().sum(dim=(-1, -2, -3))
    return ((dv + dh) / (h * w)).mean()
