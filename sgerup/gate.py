"""Spatial gate, Eqs. (6)-(7), and the gate variants used as baselines.

    g = sigmoid(W_F F + W_C C + b) in (0,1)^{h x w}      (1x1 conv, 1024+3+1 = 1028 params)
    G = U(g)                                             bilinear upsampling to H x W
    I~ = G * B(I) + (1 - G) * I

Modes:
    "one"     G == 1  -> ERUP-YOLO (B2)
    "zero"    G == 0  -> KBL-only (B3)
    "learned" learned spatial gate on an h x w grid (proposed)
    "global"  learned gate pooled to 1 x 1 (B4, the E2-i 1x1 case)
    "fixed"   hand-designed G = sigmoid(-(mu - tau_b) / s), no learning (B5)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cues import pool_to_grid

GATE_MODES = ("one", "zero", "learned", "global", "fixed")


class GateHead(nn.Module):
    def __init__(self, feat_channels: int = 1024, n_cues: int = 3, b0: float = 2.0,
                 use_features: bool = True, use_cues: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(feat_channels + n_cues, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, b0)  # (P4): G ~= sigmoid(b0) at start
        self.feat_channels = feat_channels
        self.use_features = use_features  # E2-f: W_F = 0
        self.use_cues = use_cues          # E2-f: W_C = 0

    def forward(self, feature: torch.Tensor, cues_grid: torch.Tensor) -> torch.Tensor:
        if not self.use_features:
            feature = torch.zeros_like(feature)
        if not self.use_cues:
            cues_grid = torch.zeros_like(cues_grid)
        return torch.sigmoid(self.conv(torch.cat([feature, cues_grid], dim=1)))


def upsample_gate(g: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """G = U(g): bilinear upsampling, stays in (0, 1) because it is a convex interpolation."""
    if g.shape[-2:] == tuple(size):
        return g
    return F.interpolate(g, size=size, mode="bilinear", align_corners=False)


def fixed_gate(mu: torch.Tensor, grid: tuple[int, int], tau_b: float, s: float) -> torch.Tensor:
    """B5: G = sigmoid(-(mu - tau_b) / s) evaluated on the gate grid."""
    return torch.sigmoid(-(pool_to_grid(mu, grid) - tau_b) / s)


def blend(original: torch.Tensor, processed: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Eq. (7). gate (B,1,H,W) is broadcast over channels."""
    return gate * processed + (1 - gate) * original
