import pytest
import torch

from sgerup.filters.bpw import bezier_knots, bpw_filter, control_points, max_segment_slope
from sgerup.filters.kbl import kbl_filter


def test_bpw_zero_params_is_identity(img):
    assert torch.allclose(bpw_filter(img, torch.zeros(2, 12)), img, atol=1e-6)


def test_bpw_output_range_and_shape(img):
    p = torch.rand(2, 12) * 2 - 1
    out = bpw_filter(img, p)
    assert out.shape == img.shape
    assert out.min() >= 0 and out.max() <= 1


def test_bpw_monotone_per_channel():
    x = torch.linspace(0, 1, 500).view(1, 1, 1, 500).expand(4, 3, 1, 500).contiguous()
    p = torch.rand(4, 12, generator=torch.Generator().manual_seed(1)) * 2 - 1
    y = bpw_filter(x, p)
    assert (y[..., 1:] - y[..., :-1] >= -1e-6).all()


def test_bpw_control_points_in_unit_square():
    xs, ys = control_points(torch.rand(50, 12) * 2 - 1)
    for t in (xs, ys):
        assert t.min() >= 0 and t.max() <= 1


def test_bpw_slope_matches_paper_check():
    # Paper Sec. III-B: both control points at (0, 1), L = 8 -> steepest slope ~169
    assert max_segment_slope(torch.tensor([[-1.0, 1.0] * 6]), 8).item() == pytest.approx(169.0, rel=1e-3)


def test_bpw_knots_monotone_and_endpoints():
    kx, ky = bezier_knots(torch.rand(3, 12) * 2 - 1, 8)
    assert torch.allclose(kx[..., 0], torch.zeros(3, 3)) and torch.allclose(kx[..., -1], torch.ones(3, 3))
    assert (kx[..., 1:] >= kx[..., :-1]).all()


def test_bpw_gradients_flow(img):
    p = (torch.rand(2, 12) * 1.6 - 0.8).requires_grad_()
    x = img.clone().requires_grad_()
    bpw_filter(x, p).sum().backward()
    assert p.grad is not None and p.grad.abs().sum() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_bpw_extreme_inputs_stable():
    x = torch.tensor([0.0, 1e-9, 0.5, 1 - 1e-9, 1.0]).view(1, 1, 1, 5).expand(1, 3, 1, 5).contiguous()
    for p in (torch.full((1, 12), -1.0), torch.full((1, 12), 1.0), torch.tensor([[-1.0, 1.0] * 6])):
        y = bpw_filter(x, p)
        assert torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1


def test_bpw_shape_errors():
    with pytest.raises(ValueError):
        bpw_filter(torch.rand(1, 1, 4, 4), torch.zeros(1, 12))
    with pytest.raises(ValueError):
        bpw_filter(torch.rand(1, 3, 4, 4), torch.zeros(1, 11))


def test_kbl_zero_kernels_identity(img):
    assert torch.allclose(kbl_filter(img, torch.zeros(2, 2, 3, 9, 9)), img)


def test_kbl_range_and_gradients(img):
    k = (torch.rand(2, 2, 3, 9, 9) * 2 - 1).requires_grad_()
    out = kbl_filter(img, k * 0.05)
    assert out.min() >= 0 and out.max() <= 1
    out.sum().backward()
    assert k.grad is not None and k.grad.abs().sum() > 0


def test_kbl_k2_delta_adds_input():
    x = torch.full((1, 3, 16, 16), 0.2)
    k = torch.zeros(1, 2, 3, 9, 9)
    k[:, 1, :, 4, 4] = 1.0  # k2 = delta -> output = J + J
    assert torch.allclose(kbl_filter(x, k), torch.full_like(x, 0.4))


def test_kbl_small_image_and_bad_kernels():
    out = kbl_filter(torch.rand(1, 3, 3, 3), torch.zeros(1, 2, 3, 9, 9))  # smaller than padding -> replicate
    assert out.shape == (1, 3, 3, 3)
    with pytest.raises(ValueError):
        kbl_filter(torch.rand(1, 3, 8, 8), torch.zeros(1, 2, 3, 8, 8))
