"""Photometric augmentation: ERUP-style random BPW curves and SHA (Eq. 12).

All functions take a torch.Generator so draws are reproducible per seed.

Documented approximations:
* ERUP-YOLO's augmentation distribution [8, Sec. 4.4] is not fully specified.
  Here a "brightening" curve draws r ~ U(-a, 0), theta ~ U(0, a) per control point
  and channel (control points move up-left, curve above the diagonal); a
  "darkening" curve uses the mirrored ranges. a = strength in (0, 1].
* SHA mask M = clip(vertical ramp (bright at the top) + low-frequency random field).
  The ramp is the paper's "horizon heuristic"; the field is a 4x4 Gaussian grid
  bilinearly upsampled. Optional scattering branch: I t + A (1 - t),
  t = exp(-kappa d~), pseudo-depth d~ increasing towards the top.
* Validation degradations: synthetic fog follows IA-YOLO's atmospheric model
  (centre-distance pseudo-depth, A = 0.5, beta = 0.01 i + 0.05, i in 0..9);
  synthetic darkening is gamma I^gamma with gamma ~ U(1.5, 5).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .filters import bpw_filter


def random_bpw_params(n: int, direction: str, strength: float, gen: torch.Generator,
                      device: torch.device | str = "cpu") -> torch.Tensor:
    """(n, 12) params for brightening ('up') or darkening ('down') curves."""
    u = torch.rand(n, 3, 2, 2, generator=gen) * strength
    if direction == "up":
        r, th = -u[..., 0], u[..., 1]
    elif direction == "down":
        r, th = u[..., 0], -u[..., 1]
    else:
        raise ValueError("direction must be 'up' or 'down'")
    return torch.stack([r, th], dim=-1).reshape(n, 12).to(device)


def random_bpw(img: torch.Tensor, gen: torch.Generator, strength: float = 1.0) -> torch.Tensor:
    """One random global BPW curve per image, brightening or darkening with equal probability."""
    b = img.shape[0]
    up = random_bpw_params(b, "up", strength, gen, img.device)
    down = random_bpw_params(b, "down", strength, gen, img.device)
    pick = (torch.rand(b, generator=gen) < 0.5).to(img.device).view(b, 1)
    return bpw_filter(img, torch.where(pick, up, down))


def sha_mask(b: int, h: int, w: int, gen: torch.Generator, device="cpu") -> torch.Tensor:
    ramp = torch.linspace(1, 0, h).view(1, 1, h, 1).expand(b, 1, h, w)
    amp = torch.rand(b, 1, 1, 1, generator=gen) * 0.6 + 0.4
    field = torch.randn(b, 1, 4, 4, generator=gen) * 0.35
    field = F.interpolate(field, size=(h, w), mode="bicubic", align_corners=True)
    return (amp * ramp + field).clamp(0, 1).to(device)


def scatter(img: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Scattering-model brightening I t + A (1 - t) with top-increasing pseudo-depth."""
    b, _, h, w = img.shape
    depth = torch.linspace(1, 0, h, device=img.device).view(1, 1, h, 1).expand(b, 1, h, w)
    kappa = (torch.rand(b, 1, 1, 1, generator=gen) * 2.5 + 0.5).to(img.device)
    a = (torch.rand(b, 1, 1, 1, generator=gen) * 0.25 + 0.7).to(img.device)
    t = torch.exp(-kappa * depth)
    return img * t + a * (1 - t)


def sha(img: torch.Tensor, gen: torch.Generator, strength: float = 1.0, scatter_prob: float = 0.0) -> torch.Tensor:
    """Eq. (12): I_aug = M * B_a(I) + (1 - M) * B_a'(I), a brightening, a' darkening."""
    b, _, h, w = img.shape
    up = bpw_filter(img, random_bpw_params(b, "up", strength, gen, img.device))
    if scatter_prob > 0:
        use = (torch.rand(b, generator=gen) < scatter_prob).to(img.device).view(b, 1, 1, 1)
        up = torch.where(use, scatter(img, gen), up)
    down = bpw_filter(img, random_bpw_params(b, "down", strength, gen, img.device))
    m = sha_mask(b, h, w, gen, img.device)
    return (m * up + (1 - m) * down).clamp(0, 1)


def photometric_augment(img: torch.Tensor, gen: torch.Generator, p_bpw: float, p_sha: float,
                        strength: float = 1.0, scatter_prob: float = 0.0) -> torch.Tensor:
    """Per image: SHA with prob p_sha, else global random BPW with prob p_bpw, else unchanged."""
    out = img.clone()
    for i in range(img.shape[0]):
        u = torch.rand(1, generator=gen).item()
        x = img[i : i + 1]
        if u < p_sha:
            out[i : i + 1] = sha(x, gen, strength, scatter_prob)
        elif u < p_sha + (1 - p_sha) * p_bpw:
            out[i : i + 1] = random_bpw(x, gen, strength)
    return out


def synthetic_fog(img: torch.Tensor, level: int, a: float = 0.5) -> torch.Tensor:
    """IA-YOLO style fog: d = -0.04 * dist_to_centre + sqrt(max(H, W)), t = exp(-beta d)."""
    _, _, h, w = img.shape
    beta = 0.01 * level + 0.05
    size = math.sqrt(max(h, w))
    ys = torch.arange(h, device=img.device).view(h, 1).float()
    xs = torch.arange(w, device=img.device).view(1, w).float()
    d = -0.04 * torch.sqrt((ys - h / 2) ** 2 + (xs - w / 2) ** 2) + size
    t = torch.exp(-beta * d).view(1, 1, h, w)
    return (img * t + a * (1 - t)).clamp(0, 1)


def synthetic_dark(img: torch.Tensor, gamma: float) -> torch.Tensor:
    return img.clamp(0, 1) ** gamma
