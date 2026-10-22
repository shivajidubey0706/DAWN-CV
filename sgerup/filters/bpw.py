"""Bezier pixel-wise (BPW) tone curve, Eq. (1) of the paper / Eqs. 11-14 of ERUP-YOLO.

For each channel c a cubic Bezier curve with end points (0,0) and (1,1) and two
interior control points P_{c,1}, P_{c,2} defines a monotone map f_c: [0,1] -> [0,1].
The curve is sampled at L+1 values of the Bezier parameter t and evaluated as a
piecewise-linear function of the input intensity.

Documented approximation (the exact mapping is not given in the paper):
each control point is parameterised by (r, theta) in [-1, 1]^2 and placed at
    x_j = b_j + r     * (b_j if r < 0 else 1 - b_j)
    y_j = b_j + theta * (b_j if theta < 0 else 1 - b_j),     b_j = j / 3,
so all-zero parameters give the identity curve (control points on the diagonal at
1/3 and 2/3) and the extremes reach the unit-square corners. With both control
points at (0, 1) and L = 8 the steepest segment has slope ~169, which matches the
paper's re-implementation check (Sec. III-B; see scripts/check_equations.py).
Because all control points stay inside [0,1]^2, x(t) and y(t) are non-decreasing,
so each f_c is monotone (proof in docs/METHOD_NOTES.md).
"""
from __future__ import annotations

import torch

_EPS = 1e-6


def control_points(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map BPW parameters (B, 12) in [-1, 1] to control points.

    Layout of the 12 values: channel-major, then control point j in {1, 2},
    then (r, theta). Returns xs, ys of shape (B, 3, 2).
    """
    if params.ndim != 2 or params.shape[1] != 12:
        raise ValueError(f"BPW params must have shape (B, 12), got {tuple(params.shape)}")
    p = params.clamp(-1, 1).view(-1, 3, 2, 2)
    base = torch.tensor([1.0 / 3.0, 2.0 / 3.0], device=params.device, dtype=params.dtype).view(1, 1, 2)
    r, th = p[..., 0], p[..., 1]
    xs = base + torch.where(r < 0, r * base, r * (1 - base))
    ys = base + torch.where(th < 0, th * base, th * (1 - base))
    return xs, ys


def bezier_knots(params: torch.Tensor, n_intervals: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the per-channel Bezier curves at t = k / L, k = 0..L.

    Returns knot abscissae and ordinates, each (B, 3, L+1), with x non-decreasing.
    """
    if n_intervals < 1:
        raise ValueError("n_intervals must be >= 1")
    xs, ys = control_points(params)
    t = torch.linspace(0, 1, n_intervals + 1, device=params.device, dtype=params.dtype)
    b1 = 3 * (1 - t) ** 2 * t
    b2 = 3 * (1 - t) * t**2
    b3 = t**3
    kx = xs[..., 0:1] * b1 + xs[..., 1:2] * b2 + b3
    ky = ys[..., 0:1] * b1 + ys[..., 1:2] * b2 + b3
    # Guard against round-off breaking monotonicity of the abscissae.
    kx = torch.cummax(kx, dim=-1).values
    return kx, ky


def bpw_filter(img: torch.Tensor, params: torch.Tensor, n_intervals: int = 8) -> torch.Tensor:
    """Apply the BPW curve: B_theta(I)_c(x) = f_c(I_c(x); theta). Differentiable in img and params."""
    if img.ndim != 4 or img.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W) RGB image, got {tuple(img.shape)}")
    b, c, h, w = img.shape
    if params.shape[0] != b:
        raise ValueError("batch size of params and image differ")
    kx, ky = bezier_knots(params.to(img.dtype), n_intervals)  # (B, 3, L+1)
    x = img.clamp(0, 1).reshape(b * c, h * w)
    kx = kx.reshape(b * c, -1)
    ky = ky.reshape(b * c, -1)
    # Segment index in [0, L-1]: search among interior knots only.
    idx = torch.searchsorted(kx[:, 1:-1].contiguous(), x.detach().contiguous(), right=True)
    x0 = kx.gather(1, idx)
    x1 = kx.gather(1, idx + 1)
    y0 = ky.gather(1, idx)
    y1 = ky.gather(1, idx + 1)
    frac = ((x - x0) / (x1 - x0).clamp_min(_EPS)).clamp(0, 1)
    out = y0 + frac * (y1 - y0)
    return out.clamp(0, 1).view(b, c, h, w)


def max_segment_slope(params: torch.Tensor, n_intervals: int = 8) -> torch.Tensor:
    """Largest slope of the piecewise-linear curve per image (diagnostic used in Sec. III-B)."""
    kx, ky = bezier_knots(params, n_intervals)
    dx = (kx[..., 1:] - kx[..., :-1]).clamp_min(_EPS)
    dy = ky[..., 1:] - ky[..., :-1]
    return (dy / dx).flatten(1).max(dim=1).values
