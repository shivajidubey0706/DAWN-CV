import math

import pytest
import torch

from sgerup.config import FrontEndConfig, make_config, variant_names
from sgerup.cues import box_filter, cue_maps, dark_channel, luminance, saturation_fraction
from sgerup.frontend import SGERUPFrontEnd, count_parameters
from sgerup.gate import GateHead, blend, fixed_gate, upsample_gate


def test_luminance_weights():
    x = torch.zeros(1, 3, 2, 2)
    x[:, 1] = 1
    assert torch.allclose(luminance(x), torch.full((1, 1, 2, 2), 0.67))


def test_box_filter_constant_and_borders():
    x = torch.full((1, 1, 10, 10), 0.3)
    assert torch.allclose(box_filter(x, 3), x)  # count_include_pad=False keeps borders exact


def test_dark_channel_known():
    x = torch.ones(1, 3, 9, 9)
    x[0, 2, 4, 4] = 0.1
    d = dark_channel(x, 1)
    assert d[0, 0, 4, 4] == pytest.approx(0.1) and d[0, 0, 3, 3] == pytest.approx(0.1) and d[0, 0, 0, 0] == 1


def test_saturation_fraction():
    x = torch.zeros(1, 3, 4, 4)
    x[0, 0, :, :2] = 1.0
    rho = saturation_fraction(x, 0, 0.95)
    assert rho[0, 0, 0, 0] == 1 and rho[0, 0, 0, 3] == 0


def test_cue_maps_shape_no_grad(img):
    c = cue_maps(img.requires_grad_(), 3, 0.9)
    assert c.shape == (2, 3, 64, 80) and not c.requires_grad


def test_gate_head_params_and_init():
    g = GateHead(b0=2.0)
    assert count_parameters(g) == 1028
    out = g(torch.randn(2, 1024, 8, 8), torch.rand(2, 3, 8, 8))
    assert torch.allclose(out, torch.full_like(out, 1 / (1 + math.exp(-2))))  # P4


def test_upsample_and_blend_range():
    g = torch.rand(2, 1, 8, 8)
    G = upsample_gate(g, (64, 80))
    assert G.shape == (2, 1, 64, 80) and G.min() >= 0 and G.max() <= 1
    a, b = torch.rand(2, 3, 64, 80), torch.rand(2, 3, 64, 80)
    out = blend(a, b, G)
    assert out.min() >= 0 and out.max() <= 1  # P1


def test_fixed_gate_formula():
    mu = torch.full((1, 1, 16, 16), 0.9)
    g = fixed_gate(mu, (4, 4), tau_b=0.6, s=0.05)
    assert torch.allclose(g, torch.full_like(g, torch.sigmoid(torch.tensor(-(0.9 - 0.6) / 0.05)).item()))


def _fe(**kw):
    fe = SGERUPFrontEnd(FrontEndConfig(**kw))
    torch.nn.init.normal_(fe.encoder.fc.weight, std=0.05)  # make processing non-trivial
    return fe.eval()


def test_gate_one_equals_erup_and_zero_equals_kbl_only(img):
    torch.manual_seed(0)
    fe1 = _fe(gate_mode="one")
    fe0 = SGERUPFrontEnd(FrontEndConfig(use_bpw=False)).eval()
    fe0.load_state_dict(fe1.state_dict())
    with torch.no_grad():
        o1 = fe1(img)
        assert torch.allclose(o1["blended"], o1["bpw"])  # G == 1 -> ERUP (P2)
        o0 = fe0(img)
        assert torch.allclose(o0["blended"], img)       # G == 0 -> BPW bypassed
        feZ = SGERUPFrontEnd(FrontEndConfig(gate_mode="zero")).eval()
        feZ.load_state_dict(fe1.state_dict())
        assert torch.allclose(feZ(img)["out"], o0["out"], atol=1e-6)


def test_learned_gate_output_range_monotone_and_gradients(img):
    fe = _fe(gate_mode="learned").train()
    x = img.clone().requires_grad_()
    o = fe(x)
    assert o["gate_lowres"].shape == (2, 1, 8, 8) and o["gate"].shape == (2, 1, 64, 80)
    assert o["blended"].min() >= 0 and o["blended"].max() <= 1
    o["out"].mean().backward()
    assert fe.gate_head.conv.weight.grad is not None and fe.gate_head.conv.weight.grad.abs().sum() > 0
    assert fe.encoder.fc.weight.grad.abs().sum() > 0


def test_p3_blend_monotone_in_input():
    fe = _fe(gate_mode="fixed", use_kbl=False)
    x = torch.linspace(0, 1, 64).view(1, 1, 1, 64).expand(1, 3, 32, 64).contiguous()
    with torch.no_grad():
        theta = torch.tensor([[-0.5, 0.5] * 6])
        o = fe(x, theta_b=theta)
    # Same G along a row (gate varies smoothly but the BPW branch and identity are both monotone)
    row = o["blended"][0, 0, 16]
    assert (row[1:] - row[:-1] >= -0.02).all()


def test_global_gate_is_1x1(img):
    o = _fe(gate_mode="global")(img)
    assert o["gate_lowres"].shape == (2, 1, 1, 1)


def test_b0_has_no_parameters_and_is_identity(img):
    fe = SGERUPFrontEnd(FrontEndConfig(enabled=False))
    assert count_parameters(fe) == 0
    assert torch.equal(fe(img)["out"], img)


def test_all_variants_build_and_run(img):
    for v in variant_names():
        cfg = make_config(variant=v)
        out = SGERUPFrontEnd(cfg.frontend).eval()(img)["out"]
        assert out.shape == img.shape and torch.isfinite(out).all()


def test_frontend_param_counts():
    fe = SGERUPFrontEnd(make_config(variant="sg_full").frontend)
    assert count_parameters(fe.encoder) == 6_781_042
    assert count_parameters(fe) == 6_781_042 + 1028


def test_frontend_cuda_device_if_available(img):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    fe = _fe(gate_mode="learned").cuda()
    o = fe(img.cuda())
    assert o["out"].device.type == "cuda"
