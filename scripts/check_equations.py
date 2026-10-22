"""Re-derive the paper's analytic checks (Sec. III-B, III-C, IV-C) from this implementation.

* BPW steepest slope with both control points at (0, 1), L = 8   (paper: ~169)
* BPW slope for random parameters in the allowed range              (paper: ~50)
* ERUP encoder parameter count                                      (paper: 6.78 M)
* Encoder MACs at 256 and 448 input                                 (paper: 4.95 / 15.14 GMAC)
* Gate head parameters                                              (paper: 1028)
"""
import _common  # noqa: F401
import torch

from sgerup.cost import count_flops
from sgerup.encoder import ERUPEncoder
from sgerup.filters.bpw import max_segment_slope
from sgerup.gate import GateHead
from sgerup.utils import write_json
from sgerup.config import ROOT


def main():
    corner = torch.tensor([[-1.0, 1.0] * 6])
    s_corner = max_segment_slope(corner, 8).item()
    g = torch.Generator().manual_seed(0)
    rand = torch.rand(10000, 12, generator=g) * 2 - 1
    slopes = max_segment_slope(rand, 8)
    enc = ERUPEncoder()
    n_enc = sum(p.numel() for p in enc.parameters())
    macs = {}
    for s in (256, 448):
        enc.input_size = s
        f = count_flops(enc, torch.rand(1, 3, s, s))
        macs[s] = None if f is None else f / 2 / 1e9
    n_gate = sum(p.numel() for p in GateHead().parameters())
    res = {
        "bpw_slope_both_cps_at_0_1_L8": s_corner, "paper_value": "about 169",
        "bpw_slope_random_params": {"median": slopes.median().item(), "mean": slopes.mean().item(),
                                    "p90": slopes.quantile(0.9).item(), "paper_value": "about 50"},
        "encoder_params": n_enc, "encoder_params_paper": "6.78 M",
        "encoder_gmac": macs, "encoder_gmac_paper": {"256": 4.95, "448": 15.14},
        "gate_params": n_gate, "gate_params_paper": 1028,
        "note": "MACs from torch FlopCounterMode / 2 (conv + linear only).",
    }
    for k, v in res.items():
        print(f"{k}: {v}")
    write_json(ROOT / "results" / "checks" / "equations.json", res)


if __name__ == "__main__":
    main()
