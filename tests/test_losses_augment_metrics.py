import numpy as np
import pytest
import torch

from sgerup.augment import photometric_augment, random_bpw, sha, sha_mask, synthetic_dark, synthetic_fog
from sgerup.data.class_maps import IGNORE
from sgerup.losses import gate_tv_loss, saturation_loss
from sgerup.metrics.detection import ap_coco, ap_voc, box_iou, evaluate_detections
from sgerup.metrics.mechanism import bright_region_brightening, mechanism_metrics, newly_saturated_ratio
from sgerup.stats import holm, mean_ci, paired_comparison


# ------------------------------------------------------------------ losses
def test_lsat_zero_for_identity(img):
    assert saturation_loss(img, img).item() == pytest.approx(0.0, abs=1e-9)


def test_lsat_ignores_brightening_dark_and_darkening_bright():
    dark = torch.full((1, 3, 32, 32), 0.1)
    assert saturation_loss(dark, dark + 0.3).item() < 1e-4  # beta ~ 0 in dark regions
    bright = torch.full((1, 3, 32, 32), 0.85)
    assert saturation_loss(bright, bright - 0.3).item() == pytest.approx(0.0, abs=1e-9)


def test_lsat_penalises_brightening_bright_and_new_saturation():
    bright = torch.full((1, 3, 32, 32), 0.85)
    l1 = saturation_loss(bright, (bright + 0.1).clamp(0, 1)).item()
    assert l1 > 0
    l2 = saturation_loss(bright, torch.ones_like(bright)).item()  # pushes past tau_hi -> second term
    assert l2 > l1


def test_lsat_gradient_wrt_processed_only(img):
    p = (img * 1.2).clamp(0, 1).requires_grad_()
    x = img.clone().requires_grad_()
    saturation_loss(x, p).backward()
    assert p.grad is not None and p.grad.abs().sum() > 0
    assert x.grad is None or x.grad.abs().sum() == 0  # beta and indicator depend only on the input (detached)


def test_tv_loss_values():
    assert gate_tv_loss(torch.full((2, 1, 8, 8), 0.7)).item() == 0
    assert gate_tv_loss(torch.rand(1, 1, 1, 1)).item() == 0
    g = torch.zeros(1, 1, 2, 2)
    g[0, 0, 0, 0] = 1.0
    assert gate_tv_loss(g).item() == pytest.approx(0.5)  # two unit differences / (2*2)


# ------------------------------------------------------------------ augmentation
def test_augmentations_range_and_reproducible(img):
    a = photometric_augment(img, torch.Generator().manual_seed(3), 0.5, 0.5)
    b = photometric_augment(img, torch.Generator().manual_seed(3), 0.5, 0.5)
    assert torch.equal(a, b) and a.min() >= 0 and a.max() <= 1
    for f in (lambda: sha(img, torch.Generator().manual_seed(1), scatter_prob=1.0), lambda: random_bpw(img, torch.Generator().manual_seed(1)),
              lambda: synthetic_fog(img, 9), lambda: synthetic_dark(img, 3.0)):
        out = f()
        assert out.shape == img.shape and out.min() >= 0 and out.max() <= 1


def test_sha_mask_and_heterogeneity():
    m = sha_mask(4, 64, 64, torch.Generator().manual_seed(0))
    assert m.min() >= 0 and m.max() <= 1
    assert (m[:, :, :8].mean() > m[:, :, -8:].mean())  # brighter weight towards the top (horizon heuristic)


def test_fog_brightens_dark_gamma_darkens():
    x = torch.full((1, 3, 32, 32), 0.3)
    assert synthetic_fog(x, 9).mean() > x.mean() and synthetic_dark(x, 2.5).mean() < x.mean()


# ------------------------------------------------------------------ mechanism metrics
def test_nsr_brb():
    x = torch.full((1, 3, 10, 10), 0.9)
    y = x.clone()
    y[..., :5] = 1.0
    assert newly_saturated_ratio(x, y, 0.95).item() == pytest.approx(0.5)
    assert newly_saturated_ratio(x, x, 0.95).item() == 0
    assert bright_region_brightening(x, y).item() == pytest.approx(0.05, abs=1e-3)
    assert mechanism_metrics(x, x, x)["brb"] == 0


# ------------------------------------------------------------------ detection metrics
def test_iou():
    a = np.array([[0, 0, 10, 10]], float)
    assert box_iou(a, a)[0, 0] == pytest.approx(1)
    assert box_iou(a, np.array([[5, 0, 15, 10]], float))[0, 0] == pytest.approx(1 / 3)


def test_ap_perfect_and_empty():
    assert ap_voc(np.array([True, True]), np.array([0.9, 0.8]), 2) == pytest.approx(1)
    assert ap_coco(np.array([True, True]), np.array([0.9, 0.8]), 2) == pytest.approx(1)
    assert ap_voc(np.zeros(0, bool), np.zeros(0), 3) == 0
    assert np.isnan(ap_voc(np.zeros(0, bool), np.zeros(0), 0))


def test_ap_half():
    # one TP at top, one FP, 2 GTs -> recall 0.5 at precision 1
    assert ap_voc(np.array([True, False]), np.array([0.9, 0.5]), 2) == pytest.approx(0.5)


def _gt(boxes, labels, lum=None):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "labels": labels, "box_lum": None if lum is None else np.array(lum)}


def _pr(boxes, scores, labels):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "scores": np.array(scores, float), "labels": labels}


def test_evaluate_perfect_and_ignore_regions():
    gts = [_gt([[0, 0, 50, 50], [100, 100, 140, 160], [200, 200, 260, 260]], ["car", "person", IGNORE], [0.2, 0.5, 0.9])]
    preds = [_pr([[0, 0, 50, 50], [100, 100, 140, 160], [200, 200, 260, 260]], [0.9, 0.8, 0.95], ["car", "person", "car"])]
    m = evaluate_detections(gts, preds, ["car", "person", "bus"])
    assert m["map50"] == pytest.approx(1.0)  # det on the ignore region is neither TP nor FP; bus has no GT -> excluded
    assert m["precision"] == pytest.approx(1.0) and m["recall"] == pytest.approx(1.0)
    assert m["per_class"]["bus"]["ap50"] is None
    assert m["map50_95"] == pytest.approx(1.0)


def test_evaluate_duplicates_and_misses():
    gts = [_gt([[0, 0, 50, 50], [60, 60, 100, 100]], ["car", "car"], [0.1, 0.9])]
    preds = [_pr([[0, 0, 50, 50], [1, 1, 50, 50]], [0.9, 0.8], ["car", "car"])]
    m = evaluate_detections(gts, preds, ["car"])
    assert m["tp"] == 1 and m["fp"] == 1 and m["recall"] == pytest.approx(0.5)
    br = m["brightness_recall"]
    assert br is None or br["dark"]["n_gt"] + br["bright"]["n_gt"] + br["medium"]["n_gt"] == 2


def test_brightness_tertiles_and_sizes():
    boxes = [[i * 20, 0, i * 20 + 10, 10] for i in range(6)] + [[0, 100, 200, 300]]
    lums = [0.1, 0.15, 0.5, 0.55, 0.9, 0.95, 0.6]  # tertile cut-offs: 0.5 and 0.6
    gts = [_gt(boxes, ["car"] * 7, lums)]
    preds = [_pr(boxes[:2], [0.9, 0.9], ["car", "car"])]  # only the two darkest small boxes found
    m = evaluate_detections(gts, preds, ["car"])
    assert m["brightness_recall"]["dark"]["n_gt"] == 3
    assert m["brightness_recall"]["dark"]["recall"] == pytest.approx(2 / 3)
    assert m["brightness_recall"]["bright"]["recall"] == 0
    assert m["sizes"]["small"]["n_gt"] == 6 and m["sizes"]["large"]["n_gt"] == 1
    assert m["sizes"]["large"]["ap"] == 0


def test_bootstrap_ci():
    gts = [_gt([[0, 0, 50, 50]], ["car"]) for _ in range(20)]
    preds = [_pr([[0, 0, 50, 50]] if i % 2 else [], [0.9] if i % 2 else [], ["car"] if i % 2 else []) for i in range(20)]
    m = evaluate_detections(gts, preds, ["car"], bootstrap=50)
    lo, hi = m["map50_bootstrap"]["ci95"]
    assert lo <= m["map50"] <= hi


def test_misaligned_inputs_raise():
    with pytest.raises(ValueError):
        evaluate_detections([_gt([], [])], [], ["car"])


# ------------------------------------------------------------------ statistics
def test_stats():
    s = mean_ci([1.0, 2.0, 3.0])
    assert s["mean"] == 2 and s["ci"][0] < 2 < s["ci"][1]
    assert mean_ci([0.5])["ci"] is None and mean_ci([])["mean"] is None
    c = paired_comparison({0: 0.5, 1: 0.6}, {0: 0.4, 1: 0.4})
    assert c["n_pairs"] == 2 and c["diff"]["mean"] == pytest.approx(0.15) and "inconclusive" in c["note"]
    h = holm({"a": 0.01, "b": 0.04, "c": None})
    assert h["a"]["p_holm"] == pytest.approx(0.02) and h["b"]["p_holm"] == pytest.approx(0.04) and h["c"]["p"] is None
