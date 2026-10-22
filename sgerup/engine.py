"""Training and evaluation loops (used by scripts/ and the API's job runner)."""
from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augment import photometric_augment, sha, synthetic_dark, synthetic_fog
from .config import ROOT, RunConfig, load_variants
from .cues import luminance
from .data.class_maps import CLASS_SETS, COCO_IDS, IGNORE
from .data.datasets import check_disjoint, load_samples, registry, split_train_val
from .data.torch_dataset import DetectionDataset, collate
from .detector import YOLODetector
from .frontend import SGERUPFrontEnd, count_parameters
from .imaging import letterbox, load_image, to_tensor, unletterbox_boxes
from .losses import gate_tv_loss, saturation_loss
from .metrics.detection import evaluate_detections
from .metrics.mechanism import bright_region_brightening, newly_saturated_ratio
from .utils import environment_record, read_json, seed_everything, write_json

PROTOCOL_SPLIT_SEED = 0  # fixed across runs and arms so every arm sees the same split


def run_dir(cfg: RunConfig) -> Path:
    p = Path(cfg.output_dir)
    p = p if p.is_absolute() else ROOT / p
    return p / cfg.name / cfg.variant / f"seed{cfg.train.seed}"


# --------------------------------------------------------------------------- data
def training_samples(cfg: RunConfig) -> tuple[list, list, dict]:
    """VOC 2007+2012 trainval restricted to images with protocol objects; fixed 10% validation split."""
    samples = [s for s in load_samples(cfg.train_dataset, split="trainval")
               if any(lab != IGNORE for lab in s.labels)]
    if not samples:
        raise RuntimeError(f"No training samples found for dataset '{cfg.train_dataset}'. "
                           "Run scripts/download_voc.py or set the dataset path.")
    train, val = split_train_val(samples, cfg.train.val_fraction, PROTOCOL_SPLIT_SEED)
    rng = np.random.default_rng(PROTOCOL_SPLIT_SEED)
    if cfg.train.max_train_images and len(train) > cfg.train.max_train_images:
        train = [train[i] for i in sorted(rng.choice(len(train), cfg.train.max_train_images, replace=False))]
    if cfg.train.max_val_images and len(val) > cfg.train.max_val_images:
        val = [val[i] for i in sorted(rng.choice(len(val), cfg.train.max_val_images, replace=False))]
    # Leakage control: training/validation must not contain any test image.
    named = {"train": train, "val": val}
    reg = registry()
    for name in ("dawn",) + (("voc_test",) if cfg.train_dataset == "voc" else ()):
        test = load_samples("voc", split="test") if name == "voc_test" else (
            load_samples(name) if Path(reg[name]["root"]).exists() or (ROOT / reg[name]["root"]).exists() else [])
        if test:
            named[name] = test
    leak = check_disjoint(named)
    if not leak["disjoint"]:
        raise RuntimeError(f"Split leakage detected: {leak['overlaps']}")
    return train, val, leak


def degrade_val(x: torch.Tensor, idx: int) -> tuple[torch.Tensor, str]:
    """Fixed per-index validation degradation: clean / synthetic fog / gamma dark / SHA."""
    g = torch.Generator().manual_seed(10_000 + idx)
    kind = idx % 4
    if kind == 1:
        return synthetic_fog(x, idx % 10), "fog"
    if kind == 2:
        return synthetic_dark(x, 1.5 + 3.5 * torch.rand(1, generator=g).item()), "dark"
    if kind == 3:
        return sha(x, g), "sha"
    return x, "clean"


# --------------------------------------------------------------------------- training
class StatusWriter:
    def __init__(self, path: Path):
        self.path = path
        self.state: dict = {}

    def update(self, **kw):
        self.state.update(kw, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
        write_json(self.path, self.state)


def train(cfg: RunConfig, resume: bool = False, device: str | None = None, log=print) -> Path:
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    status = StatusWriter(out / "status.json")
    status.update(state="starting", variant=cfg.variant, seed=cfg.train.seed, name=cfg.name)
    try:
        return _train(cfg, out, status, resume, device, log)
    except Exception as e:
        (out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        status.update(state="failed", error=f"{e.__class__.__name__}: {e}")
        raise


def _train(cfg: RunConfig, out: Path, status: StatusWriter, resume: bool, device, log) -> Path:
    tc = cfg.train
    seed_everything(tc.seed, tc.deterministic)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    write_json(out / "config.json", cfg.to_dict())
    write_json(out / "environment.json", {**environment_record(), "deterministic": tc.deterministic})

    det = YOLODetector(cfg.detector_weights, dev)
    fe = SGERUPFrontEnd(cfg.frontend).to(dev)
    params = [p for p in fe.parameters() if p.requires_grad]
    if not tc.freeze_detector:
        params += list(det.model.parameters())
    train_s, val_s, leak = training_samples(cfg)
    write_json(out / "split.json", {"train_ids": [s.id for s in train_s], "val_ids": [s.id for s in val_s],
                                    "leakage_check": leak, "split_seed": PROTOCOL_SPLIT_SEED})
    log(f"[{cfg.variant} seed{tc.seed}] train={len(train_s)} val={len(val_s)} device={dev} "
        f"trainable={sum(p.numel() for p in params)}")

    ckpt_meta = {"config": cfg.to_dict(), "run_name": cfg.name}
    if not params:
        # B0 with a frozen detector has nothing to train: record that explicitly.
        torch.save({"frontend": fe.state_dict(), **ckpt_meta, "epoch": 0, "trained": False}, out / "best.pt")
        status.update(state="done", note="no trainable parameters (frozen detector, no front end)", progress=1.0)
        write_json(out / "history.json", {"epochs": [], "note": "nothing to train"})
        return out

    ds = DetectionDataset(train_s, tc.imgsz, cfg.aug.hflip, tc.seed)
    g = torch.Generator().manual_seed(tc.seed)
    loader = DataLoader(ds, batch_size=tc.batch_size, shuffle=True, collate_fn=collate, num_workers=tc.num_workers,
                        generator=g, drop_last=True)
    opt = torch.optim.SGD(params, lr=tc.lr0, momentum=tc.momentum, weight_decay=tc.weight_decay, nesterov=False)
    total_steps = max(1, tc.epochs * len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (tc.lr_min + 0.5 * (tc.lr0 - tc.lr_min) * (1 + math.cos(math.pi * min(s, total_steps) / total_steps))) / tc.lr0)
    use_amp = tc.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    aug_gen = torch.Generator().manual_seed(1_000 + tc.seed)

    history, start_epoch, best = {"epochs": []}, 0, -1.0
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        fe.load_state_dict(ck["frontend"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        aug_gen.set_state(ck["aug_gen"])
        history, start_epoch, best = ck["history"], ck["epoch"] + 1, ck.get("best", -1.0)
        log(f"resumed from epoch {ck['epoch']}")

    lc = cfg.loss
    fc = cfg.frontend
    t_start = time.time()
    step_log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    for epoch in range(start_epoch, tc.epochs):
        fe.train()
        det.train_mode(not tc.freeze_detector)
        ds.epoch = epoch
        sums = {"loss": 0.0, "l_det": 0.0, "l_sat": 0.0, "l_tv": 0.0}
        n = 0
        for it, batch in enumerate(loader):
            img = batch["img"].to(dev, non_blocking=True)
            img = photometric_augment(img, aug_gen, cfg.aug.p_bpw, cfg.aug.p_sha, cfg.aug.bpw_strength,
                                      cfg.aug.sha_scatter_prob)
            fo = fe(img)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                preds = det.forward(fo["out"])
            batch_d = {"img": fo["out"], "batch_idx": batch["batch_idx"].to(dev), "cls": batch["cls"].to(dev),
                       "bboxes": batch["bboxes"].to(dev)}
            l_det, _ = det.loss(preds, batch_d)
            l_sat = torch.zeros((), device=dev)
            if lc.lambda_s > 0:
                target = fo["blended"] if lc.sat_on == "blended" else fo["out"]
                l_sat = saturation_loss(img, target, fc.cue_radius, fc.tau_b, fc.s, fc.tau_hi, lc.eta)
            l_tv = gate_tv_loss(fo["gate_lowres"]) if lc.lambda_tv > 0 else torch.zeros((), device=dev)
            loss = l_det + lc.lambda_s * l_sat + lc.lambda_tv * l_tv
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {it}: {loss.item()}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in (("loss", loss), ("l_det", l_det), ("l_sat", l_sat), ("l_tv", l_tv)):
                sums[k] += float(v.detach())
            n += 1
            done = (epoch * len(loader) + it + 1) / total_steps
            if it % tc.log_every == 0:
                rec = {"epoch": epoch, "step": it, "lr": sched.get_last_lr()[0], **{k: float(v.detach()) for k, v in
                       (("loss", loss), ("l_det", l_det), ("l_sat", l_sat), ("l_tv", l_tv))},
                       "gate_mean": float(fo["gate_lowres"].mean())}
                step_log.write(json.dumps(rec) + "\n")
                step_log.flush()
                elapsed = time.time() - t_start
                status.update(state="training", epoch=epoch, epochs=tc.epochs, step=it, steps=len(loader),
                              progress=round(done, 4), loss=rec["loss"],
                              eta_s=round(elapsed / max(done - start_epoch / tc.epochs, 1e-6) * (1 - done)))
        train_stats = {k: v / max(n, 1) for k, v in sums.items()}
        val = validate(cfg, fe, det, val_s, dev)
        rec = {"epoch": epoch, "train": train_stats, "val": val, "lr": sched.get_last_lr()[0],
               "time_s": round(time.time() - t_start, 1)}
        history["epochs"].append(rec)
        log(f"epoch {epoch}: loss={train_stats['loss']:.4f} l_det={train_stats['l_det']:.4f} "
            f"l_sat={train_stats['l_sat']:.5f} val_map50={val['map50']} val_loss={val['loss']:.4f}")
        is_best = (val["map50"] or 0) > best
        best = max(best, val["map50"] or 0)
        state = {"frontend": fe.state_dict(), **ckpt_meta, "epoch": epoch, "trained": True, "val": val}
        if is_best:
            torch.save(state, out / "best.pt")  # selected on the validation split only
        torch.save({**state, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "aug_gen": aug_gen.get_state(), "history": history, "best": best},
                   out / "last.pt")
        write_json(out / "history.json", history)
    step_log.close()
    plot_history(history, out / "loss_curve.png")
    status.update(state="done", progress=1.0, best_val_map50=best, train_seconds=round(time.time() - t_start, 1))
    return out


@torch.no_grad()
def validate(cfg: RunConfig, fe, det, val_s, dev) -> dict:
    """Loss and mAP@0.5 on the fixed, synthetically degraded validation split (never test data)."""
    fe.eval()
    classes = cfg.protocol_classes
    ids = [COCO_IDS[c] for c in classes]
    inv = {COCO_IDS[c]: c for c in classes}
    gts, preds, losses = [], [], []
    ds = DetectionDataset(val_s, cfg.train.imgsz, 0.0, 0)
    for i in range(len(ds)):
        x, t, _ = ds[i]
        x = x.unsqueeze(0).to(dev)
        x, _ = degrade_val(x, i)
        fo = fe(x)
        det.train_mode(False)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            p_train = det.forward(fo["out"])
        b = {"img": fo["out"], "batch_idx": torch.zeros(len(t), device=dev), "cls": t[:, :1].to(dev),
             "bboxes": t[:, 1:].to(dev)}
        losses.append(det.loss(p_train, b)[0].item())
        d = det.predict(fo["out"], conf=cfg.conf_eval, iou=cfg.iou_nms, classes=ids)[0].cpu().numpy()
        S = cfg.train.imgsz
        tb = t[:, 1:].numpy() * S
        gts.append({"boxes": np.stack([tb[:, 0] - tb[:, 2] / 2, tb[:, 1] - tb[:, 3] / 2, tb[:, 0] + tb[:, 2] / 2,
                                       tb[:, 1] + tb[:, 3] / 2], 1) if len(tb) else np.zeros((0, 4)),
                    "labels": [inv[int(c)] for c in t[:, 0].tolist()]})
        preds.append({"boxes": d[:, :4], "scores": d[:, 4], "labels": [inv[int(c)] for c in d[:, 5]]})
    m = evaluate_detections(gts, preds, classes, cfg.conf_pr)
    return {"loss": float(np.mean(losses)) if losses else float("nan"), "map50": m["map50"],
            "map50_95": m["map50_95"], "n_images": len(val_s)}


def plot_history(history: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    eps = history.get("epochs", [])
    if not eps:
        return
    x = [e["epoch"] for e in eps]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(x, [e["train"]["loss"] for e in eps], "o-", label="train total")
    ax[0].plot(x, [e["train"]["l_det"] for e in eps], "o-", label="train L_det")
    ax[0].plot(x, [e["val"]["loss"] for e in eps], "s--", label="val L_det")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(x, [e["val"]["map50"] for e in eps], "o-", color="C3")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val mAP@0.5"); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- evaluation
def box_luminance(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    y = luminance(torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255).unsqueeze(0))[0, 0].numpy()
    out = []
    for x1, y1, x2, y2 in boxes.astype(int):
        patch = y[max(y1, 0) : max(y2, y1 + 1), max(x1, 0) : max(x2, x1 + 1)]
        out.append(float(patch.mean()) if patch.size else float("nan"))
    return np.array(out)


@torch.no_grad()
def evaluate(cfg: RunConfig, checkpoint: str | Path | None, dataset: str, split: str | None = None,
             limit: int | None = None, device: str | None = None, bootstrap: int = 0, batch: int = 8,
             det: YOLODetector | None = None, log=print) -> dict:
    if ":" in dataset:  # e.g. "voc:test"
        dataset, split = dataset.split(":", 1)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    det = det or YOLODetector(cfg.detector_weights, dev)
    ck_info = {"checkpoint": None, "trained": False}
    if checkpoint:
        ck = torch.load(checkpoint, map_location=dev, weights_only=False)
        from .config import FrontEndConfig
        fe = SGERUPFrontEnd(FrontEndConfig(**ck["config"]["frontend"])).to(dev)
        fe.load_state_dict(ck["frontend"])
        ck_info = {"checkpoint": str(Path(checkpoint).relative_to(ROOT) if Path(checkpoint).is_relative_to(ROOT) else checkpoint),
                   "trained": bool(ck.get("trained")), "epoch": ck.get("epoch"), "val": ck.get("val")}
    else:
        fe = SGERUPFrontEnd(cfg.frontend).to(dev)
    fe.eval()
    spec = registry()[dataset]
    classes = CLASS_SETS[spec.get("class_set", "protocol5")]
    ids = [COCO_IDS[c] for c in classes]
    inv = {COCO_IDS[c]: c for c in classes}
    samples = load_samples(dataset, split=split, limit=limit)
    if not samples:
        raise RuntimeError(f"No samples for dataset '{dataset}' — is it downloaded? (root {spec['root']})")
    S = cfg.train.imgsz
    gts, preds, mech, weathers = [], [], [], []
    t0 = time.time()
    for b0 in range(0, len(samples), batch):
        chunk = samples[b0 : b0 + batch]
        imgs, metas = [], []
        for s in chunk:
            im = load_image(s.image)
            h, w = im.shape[:2]
            boxes = s.boxes.copy()
            if s.width and s.height and (s.width != w or s.height != h):
                boxes[:, [0, 2]] *= w / s.width
                boxes[:, [1, 3]] *= h / s.height
            lb, r, pad = letterbox(im, S)
            imgs.append(to_tensor(lb))
            metas.append((s, im, boxes, r, pad, w, h))
        x = torch.cat(imgs).to(dev)
        fo = fe(x)
        dets = det.predict(fo["out"], conf=cfg.conf_eval, iou=cfg.iou_nms, classes=ids)
        nsr = newly_saturated_ratio(x, fo["blended"], cfg.frontend.tau_hi)
        brb = bright_region_brightening(x, fo["blended"], cfg.frontend.cue_radius, cfg.frontend.tau_b, cfg.frontend.s)
        nsr_f = newly_saturated_ratio(x, fo["out"], cfg.frontend.tau_hi)
        brb_f = bright_region_brightening(x, fo["out"], cfg.frontend.cue_radius, cfg.frontend.tau_b, cfg.frontend.s)
        for k, ((s, im, boxes, r, pad, w, h), d) in enumerate(zip(metas, dets)):
            d = d.cpu().numpy()
            pb = unletterbox_boxes(d[:, :4], r, pad, w, h) if len(d) else np.zeros((0, 4))
            gts.append({"boxes": boxes, "labels": s.labels, "box_lum": box_luminance(im, boxes)})
            preds.append({"boxes": pb, "scores": d[:, 4], "labels": [inv[int(c)] for c in d[:, 5]]})
            mech.append({"nsr": float(nsr[k]), "brb": float(brb[k]), "nsr_final": float(nsr_f[k]),
                         "brb_final": float(brb_f[k]), "gate_mean": float(fo["gate_lowres"][k].mean())})
            weathers.append(s.weather or "all")
        if (b0 // batch) % 20 == 0:
            log(f"  [{cfg.variant}] {dataset}: {min(b0 + batch, len(samples))}/{len(samples)} images")
    elapsed = time.time() - t0

    def summarize(idx):
        m = evaluate_detections([gts[i] for i in idx], [preds[i] for i in idx], classes, cfg.conf_pr,
                                bootstrap=bootstrap, seed=0)
        m["mechanism"] = {k: float(np.mean([mech[i][k] for i in idx])) for k in mech[0]}
        m["mechanism"]["definition"] = "NSR/BRB on I~ (Eq. 13-14); *_final on I^. Mean over images, at detector input resolution."
        return m

    result = {
        "status": "evaluated", "dataset": dataset, "dataset_title": spec.get("title"), "split": split,
        "variant": cfg.variant, "variant_label": load_variants()[cfg.variant].get("label", cfg.variant),
        "seed": cfg.train.seed, "run_name": cfg.name, **ck_info,
        "settings": {"imgsz": S, "conf_eval": cfg.conf_eval, "conf_pr": cfg.conf_pr, "iou_nms": cfg.iou_nms,
                     "classes": classes, "limit": limit, "detector": str(cfg.detector_weights),
                     "frontend": asdict(cfg.frontend) if not checkpoint else ck["config"]["frontend"]},
        "overall": summarize(list(range(len(gts)))),
        "by_weather": {w: summarize([i for i, x in enumerate(weathers) if x == w]) for w in sorted(set(weathers))}
        if len(set(weathers)) > 1 else {},
        "frontend_params": count_parameters(fe), "detector_params": det.n_params,
        "eval_seconds": round(elapsed, 1), "environment": environment_record(),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return result


def load_history(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None
