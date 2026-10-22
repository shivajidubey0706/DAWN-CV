"""The SG-ERUP image-processing front end, Eqs. (6)-(8) and Fig. 2.

    theta_B, theta_K, F = E(I)
    C   = P_{h x w}([mu; delta; rho])(I)
    g   = sigmoid(W_F F + W_C C + b)          (gate head, learned modes only)
    G   = U(g)
    I~  = G * B_{theta_B}(I) + (1 - G) * I
    I^  = K_{theta_K}(I~)

With gate_mode="one" this is ERUP-YOLO; "zero" is KBL-only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import FrontEndConfig
from .cues import cue_maps, pool_to_grid
from .encoder import ERUPEncoder
from .filters import bpw_filter, kbl_filter
from .gate import GATE_MODES, GateHead, blend, fixed_gate, upsample_gate


class SGERUPFrontEnd(nn.Module):
    def __init__(self, cfg: FrontEndConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or FrontEndConfig()
        if cfg.gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {GATE_MODES}")
        self.encoder = ERUPEncoder(input_size=cfg.encoder_input) if cfg.enabled else None
        self.gate_head = (
            GateHead(self.encoder.out_channels, 3, cfg.b0, cfg.use_gate_features, cfg.use_gate_cues)
            if cfg.enabled and cfg.gate_mode in ("learned", "global") else None
        )

    @property
    def learnable(self) -> bool:
        return any(p.requires_grad for p in self.parameters())

    def _grid(self, feat: torch.Tensor) -> tuple[int, int]:
        if self.cfg.gate_mode == "global":
            return (1, 1)
        if self.cfg.gate_grid:
            return (self.cfg.gate_grid, self.cfg.gate_grid)
        return tuple(feat.shape[-2:])

    def forward(self, img: torch.Tensor, theta_b: torch.Tensor | None = None,
                theta_k: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Process a batch. Optional theta_b / theta_k override the encoder's predictions
        (used for manual, clearly-labelled illustrative curves in the web app).

        Returns a dict with: out (I^), bpw (B(I)), blended (I~), gate (G, B1HW), gate_lowres (g),
        cues (full-res [mu; delta; rho]), theta_b, theta_k.
        """
        cfg = self.cfg
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W) RGB image, got {tuple(img.shape)}")
        h, w = img.shape[-2:]
        if not cfg.enabled:
            ones = img.new_ones(img.shape[0], 1, h, w)
            return {"out": img, "bpw": img, "blended": img, "gate": ones, "gate_lowres": ones[..., :1, :1],
                    "cues": cue_maps(img, cfg.cue_radius, cfg.tau_hi)}
        enc = self.encoder(img)
        tb = enc["theta_b"] if theta_b is None else theta_b
        tk = enc["theta_k"] if theta_k is None else theta_k
        cues = cue_maps(img, cfg.cue_radius, cfg.tau_hi)
        grid = self._grid(enc["feature"])

        bpw = bpw_filter(img, tb, cfg.n_intervals) if cfg.use_bpw else img
        if cfg.gate_mode == "one" or not cfg.use_bpw:
            g = img.new_ones(img.shape[0], 1, *grid) if cfg.use_bpw else img.new_zeros(img.shape[0], 1, *grid)
        elif cfg.gate_mode == "zero":
            g = img.new_zeros(img.shape[0], 1, *grid)
        elif cfg.gate_mode == "fixed":
            g = fixed_gate(cues[:, :1], grid, cfg.tau_b, cfg.s)
        else:
            feat = pool_to_grid(enc["feature"], grid)
            g = self.gate_head(feat, pool_to_grid(cues, grid))
        gate = upsample_gate(g, (h, w))
        blended = blend(img, bpw, gate) if cfg.use_bpw else img
        if cfg.use_kbl:
            local = kbl_filter(blended, tk, cfg.kbl_kernel_scale)
            out = blend(blended, local, gate) if cfg.gate_kbl else local
        else:
            out = blended
        return {"out": out, "bpw": bpw, "blended": blended, "gate": gate, "gate_lowres": g, "cues": cues,
                "theta_b": tb, "theta_k": tk}


def count_parameters(module: nn.Module | None) -> int:
    return 0 if module is None else sum(p.numel() for p in module.parameters())
