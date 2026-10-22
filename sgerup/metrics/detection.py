"""Detection metrics (own implementation; conventions stated in every result file).

* mAP@0.5: PASCAL VOC all-point interpolation (VOC2010+), mean over classes with >= 1 GT.
* mAP@[.5:.95]: COCO-style 101-point interpolation averaged over IoU 0.50:0.05:0.95.
  Box area (not mask area) is used for size ranges. Not pycocotools.
* Size AP: COCO area ranges at original resolution (small < 32^2 <= medium < 96^2 <= large).
* Precision / recall at a fixed score threshold, IoU 0.5, pooled over classes.
* Brightness-stratified recall: GT boxes split into tertiles of mean input luminance
  inside the box (thresholds computed over the evaluated set), recall per tertile.

Matching: detections sorted by score; each takes the unmatched non-ignore GT of the same
class with the highest IoU >= thr (TP); otherwise, if it overlaps an ignore GT with
IoU >= thr it is ignored; otherwise FP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.class_maps import IGNORE

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)
SIZE_RANGES = {"small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, 1e12)}


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-9)


def _area(b: np.ndarray) -> np.ndarray:
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) if len(b) else np.zeros(0)


@dataclass
class ImageRecord:
    """Per image and class, the match status of each detection for one IoU threshold / area range."""
    scores: dict          # cls -> (M,) scores of non-ignored detections
    tp: dict              # cls -> (M,) bool
    n_gt: dict            # cls -> int (non-ignored GT)
    gt_matched: dict      # cls -> (N,) bool for every GT of that class (ignored GT -> False)
    gt_counted: dict      # cls -> (N,) bool, which GT count towards recall


def match_image(gt_boxes, gt_labels, det_boxes, det_scores, det_labels, classes, thr, area_rng=None) -> ImageRecord:
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    det_boxes = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
    det_scores = np.asarray(det_scores, dtype=np.float64).reshape(-1)
    gt_labels, det_labels = list(gt_labels), list(det_labels)
    gl = np.array(gt_labels, dtype=object)
    dl = np.array(det_labels, dtype=object)
    gt_area, det_area = _area(gt_boxes), _area(det_boxes)
    rec = ImageRecord({}, {}, {}, {}, {})
    ign_all = gl == IGNORE
    for c in classes:
        gi = np.where(gl == c)[0]
        di = np.where(dl == c)[0]
        di = di[np.argsort(-det_scores[di], kind="stable")]
        g_ignore = np.zeros(len(gi), bool)
        if area_rng is not None:
            g_ignore = (gt_area[gi] < area_rng[0]) | (gt_area[gi] >= area_rng[1])
        ign_idx = np.where(ign_all)[0]
        iou = box_iou(det_boxes[di], gt_boxes[gi])
        iou_ign = box_iou(det_boxes[di], gt_boxes[ign_idx])
        matched = np.zeros(len(gi), bool)
        tp, keep = [], []
        for k in range(len(di)):
            cand = np.where(~matched & ~g_ignore & (iou[k] >= thr))[0] if len(gi) else np.array([], int)
            if len(cand):
                j = cand[np.argmax(iou[k, cand])]
                matched[j] = True
                tp.append(True)
                keep.append(k)
                continue
            # matched to an ignored GT of this class (area range) or an ignore region -> ignored
            hits_ign_gt = len(gi) and np.any(g_ignore & (iou[k] >= thr))
            hits_region = len(ign_idx) and np.any(iou_ign[k] >= thr)
            if hits_ign_gt or hits_region:
                continue
            if area_rng is not None and not (area_rng[0] <= det_area[di[k]] < area_rng[1]):
                continue  # unmatched detection outside the area range: ignored (COCO rule)
            tp.append(False)
            keep.append(k)
        rec.scores[c] = det_scores[di][keep] if keep else np.zeros(0)
        rec.tp[c] = np.array(tp, bool)
        rec.n_gt[c] = int((~g_ignore).sum())
        rec.gt_matched[c] = matched
        rec.gt_counted[c] = ~g_ignore
    return rec


def ap_voc(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """All-point interpolated AP (VOC2010+). NaN when there is no GT."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tpc = np.cumsum(tp[order])
    fpc = np.cumsum(~tp[order])
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def ap_coco(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP (COCO). NaN when there is no GT."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    tpc = np.cumsum(tp[order])
    fpc = np.cumsum(~tp[order])
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    q = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)
    return float(q.mean())


def _pool(records: list[ImageRecord], c: str, idx=None):
    rs = records if idx is None else [records[i] for i in idx]
    scores = np.concatenate([r.scores[c] for r in rs]) if rs else np.zeros(0)
    tp = np.concatenate([r.tp[c] for r in rs]) if rs else np.zeros(0, bool)
    return tp.astype(bool), scores, sum(r.n_gt[c] for r in rs)


def _nanmean(xs) -> float | None:
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    return float(np.mean(xs)) if xs else None


def map50_from_records(records: list[ImageRecord], classes: list[str], idx=None) -> float | None:
    return _nanmean([ap_voc(*_pool(records, c, idx)) for c in classes])


def evaluate_detections(gts: list[dict], preds: list[dict], classes: list[str], conf_pr: float = 0.25,
                        bootstrap: int = 0, seed: int = 0) -> dict:
    """gts[i]: {boxes, labels, box_lum (optional)}; preds[i]: {boxes, scores, labels}. Lists aligned by image."""
    if len(gts) != len(preds):
        raise ValueError("gts and preds must be aligned per image")

    def records(thr, area=None):
        return [match_image(g["boxes"], g["labels"], p["boxes"], p["scores"], p["labels"], classes, thr, area)
                for g, p in zip(gts, preds)]

    rec50 = records(0.5)
    per_class = {}
    for c in classes:
        tp, sc, n = _pool(rec50, c)
        per_class[c] = {"ap50": None if n == 0 else ap_voc(tp, sc, n), "n_gt": n}
    map50 = _nanmean([v["ap50"] for v in per_class.values()])

    # COCO-style over IoU thresholds
    coco_by_thr = {}
    per_class_coco = {c: [] for c in classes}
    for thr in IOU_THRS:
        rs = rec50 if thr == 0.5 else records(float(thr))
        aps = {c: ap_coco(*_pool(rs, c)) for c in classes}
        for c in classes:
            per_class_coco[c].append(aps[c])
        coco_by_thr[float(thr)] = _nanmean(aps.values())
    map_coco = _nanmean(coco_by_thr.values())
    for c in classes:
        per_class[c]["ap_coco"] = _nanmean(per_class_coco[c])
        per_class[c]["ap50_coco101"] = per_class_coco[c][0] if per_class[c]["n_gt"] else None

    # size-stratified
    sizes = {}
    for name, rng in SIZE_RANGES.items():
        aps50, apsall, n_gt = [], [], 0
        for thr in IOU_THRS:
            rs = records(float(thr), rng)
            vals = [ap_coco(*_pool(rs, c)) for c in classes]
            apsall.append(_nanmean(vals))
            if thr == 0.5:
                aps50 = vals
                n_gt = sum(r.n_gt[c] for r in rs for c in classes)
        sizes[name] = {"ap": _nanmean(apsall), "ap50": _nanmean(aps50), "n_gt": n_gt}

    # precision / recall at fixed threshold, IoU 0.5
    tp_n = fp_n = 0
    n_gt_total = sum(per_class[c]["n_gt"] for c in classes)
    gt_hit_lists = []
    for g, r in zip(gts, records_at_conf(gts, preds, classes, conf_pr)):
        tp_n += sum(int(r.tp[c].sum()) for c in classes)
        fp_n += sum(int((~r.tp[c]).sum()) for c in classes)
        gt_hit_lists.append(r)
    precision = tp_n / (tp_n + fp_n) if tp_n + fp_n else None
    recall = tp_n / n_gt_total if n_gt_total else None

    strat = brightness_stratified_recall(gts, gt_hit_lists, classes)

    out = {
        "n_images": len(gts), "classes": classes, "n_gt": n_gt_total,
        "map50": map50, "map50_95": map_coco, "map_by_iou": coco_by_thr,
        "precision": precision, "recall": recall, "conf_pr": conf_pr, "tp": tp_n, "fp": fp_n,
        "per_class": per_class, "sizes": sizes, "brightness_recall": strat,
        "conventions": {
            "map50": "VOC all-point interpolation, IoU 0.5",
            "map50_95": "COCO-style 101-point, IoU 0.50:0.05:0.95, box areas, own implementation",
            "sizes": "COCO area thresholds 32^2, 96^2 at original resolution",
            "precision_recall": f"score >= {conf_pr}, IoU 0.5, pooled over classes",
        },
    }
    if bootstrap:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(bootstrap):
            idx = rng.integers(0, len(gts), len(gts))
            v = map50_from_records(rec50, classes, idx)
            if v is not None:
                vals.append(v)
        if vals:
            out["map50_bootstrap"] = {"n": len(vals), "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                                      "std": float(np.std(vals))}
    return out


def records_at_conf(gts, preds, classes, conf):
    out = []
    for g, p in zip(gts, preds):
        s = np.asarray(p["scores"]).reshape(-1)
        keep = s >= conf
        out.append(match_image(g["boxes"], g["labels"], np.asarray(p["boxes"]).reshape(-1, 4)[keep], s[keep],
                               [lab for lab, k in zip(p["labels"], keep) if k], classes, 0.5))
    return out


def brightness_stratified_recall(gts, recs, classes) -> dict | None:
    lums, hits = [], []
    for g, r in zip(gts, recs):
        if "box_lum" not in g or g["box_lum"] is None:
            return None
        gl = np.array(g["labels"], dtype=object)
        bl = np.asarray(g["box_lum"])
        for c in classes:
            gi = np.where(gl == c)[0]
            counted = r.gt_counted[c]
            lums.extend(bl[gi][counted].tolist())
            hits.extend(r.gt_matched[c][counted].tolist())
    if len(lums) < 3:
        return None
    lums, hits = np.array(lums), np.array(hits, bool)
    t1, t2 = np.percentile(lums, [100 / 3, 200 / 3])
    bins = {"dark": lums <= t1, "medium": (lums > t1) & (lums <= t2), "bright": lums > t2}
    return {
        "thresholds": [float(t1), float(t2)],
        **{k: {"recall": float(hits[m].mean()) if m.any() else None, "n_gt": int(m.sum())} for k, m in bins.items()},
    }
