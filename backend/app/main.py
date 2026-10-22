"""SG-ERUP AI backend (FastAPI). Run: uvicorn backend.app.main:app --port 8000"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import platform
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .settings import ROOT, settings
from .state import dataset_samples, invalidate_dataset, jobs, model_state, results

import numpy as np  # noqa: E402  (after sys.path setup in state)
import torch  # noqa: E402

from sgerup import __version__  # noqa: E402
from sgerup.config import load_variants, make_config  # noqa: E402
from sgerup.data.class_maps import IGNORE  # noqa: E402
from sgerup.data.datasets import registry, resolve_root, scan_dataset, set_dataset_root  # noqa: E402
from sgerup.draw import draw_detections  # noqa: E402
from sgerup.imaging import ImageError, colormap, encode_jpeg, encode_png, load_image, validate_upload  # noqa: E402
from sgerup.pipeline import CheckpointError, manual_bpw_params  # noqa: E402
from sgerup.utils import environment_record, read_json, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sgerup.api")

@asynccontextmanager
async def lifespan(_app):
    if settings.preload_detector:
        threading.Thread(target=model_state.load, daemon=True).start()
    yield


app = FastAPI(title="SG-ERUP AI API", version=__version__, lifespan=lifespan,
              description="Adverse-weather detection research prototype (SG-ERUP). Results are research outputs.")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])

_kb = None
_kb_lock = threading.Lock()


def kb():
    global _kb
    with _kb_lock:
        if _kb is None:
            from .knowledge import KnowledgeBase
            _kb = KnowledgeBase()
        return _kb


def _err(status: int, msg: str):
    raise HTTPException(status_code=status, detail=msg)


# ---------------------------------------------------------------- health / system
@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


@app.get("/api/ready")
def ready():
    p = model_state.pipeline
    return {"ready": p is not None, "loading": model_state.loading, "error": model_state.error,
            "device": str(p.device) if p else None}


@app.get("/api/system")
def system():
    env = environment_record()
    # Querying free memory creates a CUDA context (~300 MB); skip it when the server runs on CPU.
    if torch.cuda.is_available() and settings.device != "cpu":
        free, total = torch.cuda.mem_get_info()
        env["gpu_memory_free_gb"] = round(free / 2**30, 2)
        env["gpu_memory_total_gb"] = round(total / 2**30, 2)
    env["machine"] = platform.machine()
    env["torch_threads"] = torch.get_num_threads()
    return env


# ---------------------------------------------------------------- model / variants
def _checkpoints() -> list[dict]:
    out = []
    for best in sorted((ROOT / "runs").glob("*/*/seed*/best.pt")):
        rd = best.parent
        st = read_json(rd / "status.json") if (rd / "status.json").exists() else {}
        out.append({"path": str(best.relative_to(ROOT)).replace("\\", "/"), "run": rd.parts[-3], "variant": rd.parts[-2],
                    "seed": int(rd.name.replace("seed", "")), "state": st.get("state"),
                    "best_val_map50": st.get("best_val_map50"), "size_mb": round(best.stat().st_size / 2**20, 2)})
    return out


def _safe_checkpoint(path: str | None) -> str | None:
    if not path:
        return None
    allowed = {c["path"] for c in _checkpoints()}
    if path not in allowed:
        _err(400, "Unknown checkpoint. Choose one listed by /api/model/status.")
    return path


@app.get("/api/model/status")
def model_status():
    from sgerup.detector import resolve_weights
    w = resolve_weights(settings.detector_weights)
    p = model_state.pipeline
    det = None
    if p:
        d = p.detector
        det = {"weights": str(w.relative_to(ROOT)) if w.is_relative_to(ROOT) else str(w), "params": d.n_params,
               "n_classes": len(d.names), "device": str(d.device), "imgsz": p.imgsz,
               "architecture": "Ultralytics YOLOv3u (Darknet-53 backbone, anchor-free head), COCO-pretrained"}
    return {"detector_loaded": p is not None, "detector_loading": model_state.loading, "detector_error": model_state.error,
            "weights_path": str(w), "weights_exist": w.exists(), "weights_mb": round(w.stat().st_size / 2**20, 1) if w.exists() else None,
            "detector": det, "checkpoints": _checkpoints(),
            "status_note": ("No trained SG-ERUP checkpoint yet: front ends run identity-initialised (untrained)."
                            if not any(c["state"] == "done" for c in _checkpoints()) else
                            "Trained front-end checkpoints come from the reduced short_real protocol (frozen detector, "
                            "3 epochs, 2 seeds); they are not a fully trained SG-ERUP model.")}


@app.get("/api/variants")
def variants():
    out = []
    cks = _checkpoints()
    for key, spec in load_variants().items():
        cfg = make_config(variant=key)
        out.append({"key": key, "label": spec.get("label", key), "paper_id": spec.get("paper_id"),
                    "description": spec.get("description"), "frontend": cfg.frontend.__dict__, "loss": cfg.loss.__dict__,
                    "aug": cfg.aug.__dict__, "checkpoints": [c for c in cks if c["variant"] == key and c["state"] == "done"]})
    return out


# ---------------------------------------------------------------- inference
async def _read_upload(file: UploadFile) -> np.ndarray:
    data = await file.read(int(settings.max_upload_mb * 2**20) + 1)
    try:
        validate_upload(data, int(settings.max_upload_mb * 2**20))
        return load_image(data)
    except ImageError as e:
        _err(400, str(e))


def _b64png(img: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(img)).decode()


def _run_pipeline(**kw):
    try:
        pipe = model_state.get()
    except RuntimeError as e:
        _err(503, f"Detector unavailable: {e}")
    with model_state.lock:
        try:
            return pipe.run(**kw)
        except CheckpointError as e:
            _err(400, str(e))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _err(507, "GPU out of memory (a training job may be using the GPU). Try again or set SGERUP_DEVICE=cpu.")


def _check_variant(variant: str):
    if variant not in load_variants():
        _err(400, f"Unknown variant '{variant}'.")


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), variant: str = Form("b0_no_filter"), checkpoint: str | None = Form(None),
                 conf: float = Form(0.25), iou: float = Form(0.6), class_filter: str = Form("all")):
    _check_variant(variant)
    if not (0.01 <= conf <= 0.99) or not (0.1 <= iou <= 0.95):
        _err(400, "conf must be in [0.01, 0.99] and iou in [0.1, 0.95].")
    if class_filter not in ("all", "protocol5"):
        _err(400, "class_filter must be 'all' or 'protocol5'.")
    ck = _safe_checkpoint(checkpoint)
    img = await _read_upload(file)
    r = await asyncio.to_thread(_run_pipeline, img=img, variant=variant, checkpoint=ck, conf=conf, iou=iou,
                                class_filter=class_filter)
    annotated = draw_detections(r["processed"], r["detections"])
    payload = {"detections": r["detections"], "timing_ms": r["timing_ms"], "frontend": r["frontend"],
               "mechanism": r["mechanism"], "device": r["device"], "imgsz": r["imgsz"],
               "image": {"width": img.shape[1], "height": img.shape[0], "filename": file.filename},
               "settings": {"variant": variant, "checkpoint": ck, "conf": conf, "iou": iou, "class_filter": class_filter}}
    rid = results.put({"original.jpg": (encode_jpeg(img), "image/jpeg"),
                       "processed.png": (encode_png(r["processed"]), "image/png"),
                       "annotated.jpg": (encode_jpeg(annotated), "image/jpeg"),
                       "detections.json": (json.dumps(payload, indent=2).encode(), "application/json")}, payload)
    payload["id"] = rid
    payload["files"] = {k: f"/api/results/{rid}/{k}" for k in ("original.jpg", "processed.png", "annotated.jpg", "detections.json")}
    return payload


@app.post("/api/process")
async def process(file: UploadFile = File(...), variant: str = Form("sg_full"), checkpoint: str | None = Form(None),
                  bpw_source: str = Form("model"), manual_strength: float = Form(0.6), gate_override: str = Form("none"),
                  conf: float = Form(0.25)):
    """SG-ERUP processing lab: every intermediate of Eqs. 3-8 plus NSR/BRB, from an actual forward pass."""
    _check_variant(variant)
    if bpw_source not in ("model", "manual"):
        _err(400, "bpw_source must be 'model' or 'manual'.")
    if gate_override not in ("none", "one", "zero", "fixed", "learned", "global"):
        _err(400, "invalid gate_override")
    ck = _safe_checkpoint(checkpoint)
    cfg = make_config(variant=variant)
    if not cfg.frontend.enabled:
        _err(400, "B0 has no processing stage; choose a variant with a front end.")
    img = await _read_upload(file)
    theta = manual_bpw_params(manual_strength) if bpw_source == "manual" else None
    r = await asyncio.to_thread(_run_pipeline, img=img, variant=variant, checkpoint=ck, conf=conf, iou=0.6,
                                class_filter="all", intermediates=True, theta_b=theta,
                                gate_override=None if gate_override == "none" else gate_override)
    im = r["intermediates"]
    gate = im["gate"]
    maps = {
        "original": _b64png(img), "bpw": _b64png(im["bpw"]), "blended": _b64png(im["blended"]),
        "final": _b64png(r["processed"]), "gate": _b64png(colormap(gate, 0, 1)),
        "luminance": _b64png(colormap(im["luminance"], 0, 1)), "dark_channel": _b64png(colormap(im["dark_channel"], 0, 1)),
        "saturation": _b64png(colormap(im["saturation"], 0, 1)),
        "annotated": _b64png(draw_detections(r["processed"], r["detections"])),
    }
    fe = r["frontend"]
    trained = fe.get("trained")
    provenance = ("Manual BPW curve (illustrative, not a model output)" if bpw_source == "manual" else
                  "Trained checkpoint" if trained else "Untrained front end (identity-initialised encoder)")
    gate_mode = gate_override if gate_override != "none" else cfg.frontend.gate_mode
    gate_note = {"fixed": "Fixed analytic gate B5: G = sigmoid(-(mu - tau_b)/s) — computed, no learning.",
                 "one": "G == 1 (ERUP-YOLO).", "zero": "G == 0 (KBL-only).",
                 "learned": "Learned spatial gate" + ("" if trained else " — untrained, so G = sigmoid(b0) everywhere."),
                 "global": "Learned per-image gate" + ("" if trained else " — untrained.")}.get(gate_mode, "")
    return {"maps": maps, "mechanism": r["mechanism"], "detections": r["detections"], "timing_ms": r["timing_ms"],
            "gate_stats": {"mean": float(gate.mean()), "min": float(gate.min()), "max": float(gate.max()),
                           "lowres": np.round(im["gate_lowres"], 4).tolist()},
            "theta_b": r.get("theta_b"), "provenance": provenance, "gate_note": gate_note, "gate_mode": gate_mode,
            "frontend": fe, "settings": {"variant": variant, "checkpoint": ck, "bpw_source": bpw_source,
                                         "manual_strength": manual_strength, "gate_override": gate_override,
                                         "tau_hi": cfg.frontend.tau_hi, "tau_b": cfg.frontend.tau_b, "s": cfg.frontend.s,
                                         "cue_radius": cfg.frontend.cue_radius}}


@app.get("/api/results/{rid}/{name}")
def result_file(rid: str, name: str):
    item = results.get(rid, name)
    if item is None:
        _err(404, "Result not found or expired (results are kept in memory for the most recent requests).")
    data, mime = item
    return Response(data, media_type=mime, headers={"Content-Disposition": f'attachment; filename="sgerup_{rid}_{name}"'})


# ---------------------------------------------------------------- datasets
def _report_path(name: str, split: str | None = None) -> Path:
    return ROOT / "results" / "datasets" / (f"{name}_{split}.json" if split else f"{name}.json")


@app.get("/api/datasets")
def datasets():
    out = []
    for name, spec in registry().items():
        root = resolve_root(spec)
        splits = ["trainval", "test"] if spec["format"] == "voc" else [None]
        reports = {}
        for sp in splits:
            p = _report_path(name, sp)
            reports[sp or "all"] = read_json(p) if p.exists() else None
        out.append({"name": name, "title": spec.get("title"), "format": spec["format"], "root": str(root),
                    "available": root.exists(), "role": spec.get("role"), "url": spec.get("url"),
                    "notes": spec.get("notes"), "class_set": spec.get("class_set"), "reports": reports})
    leak = ROOT / "results" / "datasets" / "summary.json"
    return {"datasets": out, "leakage_check": read_json(leak).get("_leakage_check") if leak.exists() else None}


def _check_dataset(name: str) -> dict:
    reg = registry()
    if name not in reg:
        _err(404, f"Unknown dataset '{name}'.")
    return reg[name]


@app.post("/api/datasets/{name}/validate")
async def validate_dataset(name: str, verify_images: bool = False, split: str | None = None):
    spec = _check_dataset(name)
    if spec["format"] == "voc" and split is None:
        split = "trainval"
    rep = await asyncio.to_thread(scan_dataset, name, verify_images, split)
    write_json(_report_path(name, split), rep)
    invalidate_dataset(name)
    return rep


class PathBody(BaseModel):
    root: str = Field(..., min_length=1, max_length=500)


@app.put("/api/datasets/{name}/path")
def set_path(name: str, body: PathBody):
    _check_dataset(name)
    p = Path(body.root).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_dir():
        _err(400, f"Directory does not exist: {p}")
    set_dataset_root(name, str(p))
    invalidate_dataset(name)
    return {"name": name, "root": str(p), "message": "Path saved to configs/datasets.local.yaml. Run validation to scan it."}


@app.get("/api/datasets/{name}/samples")
def samples(name: str, weather: str | None = None, split: str | None = None, offset: int = 0,
            limit: int = Query(24, le=100), cls: str | None = None):
    spec = _check_dataset(name)
    if not resolve_root(spec).exists():
        _err(404, f"Dataset '{name}' is not available at {resolve_root(spec)}.")
    if spec["format"] == "voc" and split is None:
        split = "test"
    ss = dataset_samples(name, split)
    if weather:
        ss = [s for s in ss if s.weather == weather]
    if cls:
        ss = [s for s in ss if cls in s.labels]
    page = ss[offset : offset + limit]
    return {"total": len(ss), "offset": offset, "split": split, "items": [
        {"id": s.id, "weather": s.weather, "width": s.width, "height": s.height,
         "n_objects": sum(lab != IGNORE for lab in s.labels), "n_ignore": sum(lab == IGNORE for lab in s.labels),
         "labels": sorted({lab for lab in s.labels if lab != IGNORE})} for s in page]}


def _find_sample(name: str, sid: str, split: str | None):
    spec = _check_dataset(name)
    if spec["format"] == "voc" and split is None:
        split = "test"
    for s in dataset_samples(name, split):
        if s.id == sid:
            return s
    _err(404, "Sample not found.")


@app.get("/api/datasets/{name}/annotation")
def annotation(name: str, id: str, split: str | None = None):
    return _find_sample(name, id, split).to_json() | {"image": None}


@app.get("/api/datasets/{name}/image")
def sample_image(name: str, id: str, split: str | None = None, annotated: bool = True, max_size: int = Query(900, le=2000)):
    s = _find_sample(name, id, split)
    from PIL import Image, ImageDraw
    im = Image.open(s.image).convert("RGB")
    if annotated:
        d = ImageDraw.Draw(im)
        lw = max(2, round(max(im.size) / 350))
        for box, lab in zip(s.boxes.tolist(), s.labels):
            color = (148, 163, 184) if lab == IGNORE else (34, 211, 238)
            d.rectangle(box, outline=color, width=lw)
            d.text((box[0] + 3, box[1] + 2), "ignore" if lab == IGNORE else lab, fill=color)
    im.thumbnail((max_size, max_size))
    return Response(encode_jpeg(np.asarray(im), 85), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"})


# ---------------------------------------------------------------- experiments / results
EXP_ROOT = ROOT / "results" / "experiments"


def _exp_dir(name: str) -> Path:
    d = (EXP_ROOT / name).resolve()
    if not d.is_relative_to(EXP_ROOT.resolve()) or not d.is_dir():
        _err(404, f"Experiment '{name}' not found.")
    return d


@app.get("/api/experiments")
def experiments():
    out = []
    for d in sorted(EXP_ROOT.glob("*")) if EXP_ROOT.exists() else []:
        if not d.is_dir():
            continue
        s = d / "summary.json"
        summ = read_json(s) if s.exists() else None
        out.append({"name": d.name, "n_result_files": len([p for p in d.glob("*.json") if p.name != "summary.json"]),
                    "has_summary": s.exists(), "variants": summ.get("variants_evaluated") if summ else None,
                    "seeds": summ.get("seeds") if summ else None, "datasets": summ.get("datasets") if summ else None,
                    "generated_at": summ.get("generated_at") if summ else None})
    return out


@app.get("/api/experiments/{name}")
def experiment(name: str, refresh: bool = False):
    d = _exp_dir(name)
    s = d / "summary.json"
    if refresh or not s.exists():
        from sgerup.report import summarize_experiment
        summ = summarize_experiment(d)
        summ["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json(s, summ)
    summ = read_json(s)
    files = []
    for p in sorted(d.glob("*.json")):
        if p.name == "summary.json":
            continue
        r = read_json(p)
        files.append({"file": p.name, "variant": r.get("variant"), "seed": r.get("seed"), "dataset": r.get("dataset"),
                      "trained": r.get("trained"), "checkpoint": r.get("checkpoint"), "settings": r.get("settings"),
                      "n_images": r.get("overall", {}).get("n_images"), "eval_seconds": r.get("eval_seconds"),
                      "per_class": r.get("overall", {}).get("per_class"), "sizes": r.get("overall", {}).get("sizes"),
                      "bootstrap": r.get("overall", {}).get("map50_bootstrap"),
                      "evaluated_at": r.get("evaluated_at"), "environment": {k: r.get("environment", {}).get(k) for k in ("gpu", "torch", "ultralytics")}})
    plots = [p.name for p in d.glob("*.png")]
    return {"summary": summ, "files": files, "plots": plots}


@app.get("/api/experiments/{name}/export")
def export(name: str, format: str = "csv"):
    d = _exp_dir(name)
    if format == "csv":
        p = d / "summary.csv"
        if not p.exists():
            experiment(name, refresh=True)
        return FileResponse(p, media_type="text/csv", filename=f"{name}_summary.csv")
    if format == "json":
        return FileResponse(d / "summary.json", media_type="application/json", filename=f"{name}_summary.json")
    if format == "md":
        return FileResponse(d / "summary.md", media_type="text/markdown", filename=f"{name}_summary.md")
    _err(400, "format must be csv, json or md")


@app.get("/api/experiments/{name}/plot/{file}")
def plot(name: str, file: str):
    d = _exp_dir(name)
    p = (d / file).resolve()
    if p.parent != d or p.suffix != ".png" or not p.exists():
        _err(404, "plot not found")
    return FileResponse(p, media_type="image/png")


@app.get("/api/cost")
def cost():
    p = ROOT / "results" / "cost" / "cost.json"
    return read_json(p) if p.exists() else {"status": "not_measured",
                                            "message": "Run: python scripts/benchmark_cost.py"}


@app.get("/api/checks")
def checks():
    p = ROOT / "results" / "checks" / "equations.json"
    return read_json(p) if p.exists() else {"status": "not_run", "message": "Run: python scripts/check_equations.py"}


# ---------------------------------------------------------------- training runs / jobs
@app.get("/api/training/runs")
def training_runs():
    out = []
    for st in sorted((ROOT / "runs").glob("*/*/seed*/status.json")):
        rd = st.parent
        s = read_json(st)
        h = read_json(rd / "history.json") if (rd / "history.json").exists() else {}
        cfg = read_json(rd / "config.json") if (rd / "config.json").exists() else {}
        out.append({"run": rd.parts[-3], "variant": rd.parts[-2], "seed": rd.name, "status": s,
                    "epochs_done": len(h.get("epochs", [])), "has_error": (rd / "error.log").exists(),
                    "train": cfg.get("train"), "loss": cfg.get("loss"), "aug": cfg.get("aug")})
    return out


def _run_dir(run: str, variant: str, seed: str) -> Path:
    base = (ROOT / "runs").resolve()
    d = (base / run / variant / seed).resolve()
    if not d.is_relative_to(base) or not d.is_dir():
        _err(404, "run not found")
    return d


@app.get("/api/training/runs/{run}/{variant}/{seed}")
def training_run(run: str, variant: str, seed: str):
    d = _run_dir(run, variant, seed)
    def j(n):
        return read_json(d / n) if (d / n).exists() else None
    steps = []
    if (d / "train_log.jsonl").exists():
        steps = [json.loads(line) for line in (d / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"status": j("status.json"), "history": j("history.json"), "config": j("config.json"),
            "environment": j("environment.json"), "steps": steps[-2000:],
            "error": (d / "error.log").read_text(encoding="utf-8")[-4000:] if (d / "error.log").exists() else None,
            "has_loss_curve": (d / "loss_curve.png").exists(),
            "split": {k: (len(v) if isinstance(v, list) else v) for k, v in (j("split.json") or {}).items()}}


@app.get("/api/training/runs/{run}/{variant}/{seed}/loss_curve.png")
def loss_curve(run: str, variant: str, seed: str):
    p = _run_dir(run, variant, seed) / "loss_curve.png"
    if not p.exists():
        _err(404, "no loss curve yet")
    return FileResponse(p, media_type="image/png")


class JobBody(BaseModel):
    config: str = "smoke"
    variant: str = "sg_full"
    seed: int = Field(0, ge=0, le=99)


@app.post("/api/training/jobs")
def start_job(body: JobBody):
    if not settings.allow_training_jobs:
        _err(403, "Training jobs are disabled (SGERUP_ALLOW_TRAINING_JOBS=false).")
    try:
        return jobs.start(body.config, body.variant, body.seed)
    except ValueError as e:
        _err(400, str(e))
    except RuntimeError as e:
        _err(409, str(e))


@app.get("/api/training/jobs")
def list_jobs():
    return {"enabled": settings.allow_training_jobs, "configs": list(jobs.ALLOWED_CONFIGS),
            "jobs": [jobs.describe(j) for j in jobs.jobs.values()]}


# ---------------------------------------------------------------- paper / docs
@app.get("/api/paper")
def paper():
    meta = ROOT / "knowledge" / "paper_meta.json"
    refs = ROOT / "knowledge" / "references.json"
    return {"meta": read_json(meta) if meta.exists() else None, "references": read_json(refs) if refs.exists() else []}


@app.get("/api/docs/{name}")
def docs(name: str):
    allowed = {"method": ROOT / "docs" / "METHOD_NOTES.md", "datasets": ROOT / "docs" / "DATASETS.md",
               "models": ROOT / "docs" / "MODELS.md", "architecture": ROOT / "docs" / "ARCHITECTURE.md",
               "testing": ROOT / "docs" / "TESTING_REPORT.md", "readme": ROOT / "README.md"}
    p = allowed.get(name)
    if not p or not p.exists():
        _err(404, "document not found")
    return {"name": name, "markdown": p.read_text(encoding="utf-8")}


# ---------------------------------------------------------------- Ask AI
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=8000)


class AskBody(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=40)


@app.get("/api/ask/status")
def ask_status():
    ok, reason = settings.llm_configured()
    return {"mode": "llm" if ok else "fallback", "model": settings.llm_model if ok else None, "reason": reason,
            "n_chunks": len(kb().chunks)}


@app.post("/api/ask/reload")
def ask_reload():
    kb().reload()
    return {"n_chunks": len(kb().chunks)}


@app.post("/api/ask")
async def ask(body: AskBody):
    from .ask import answer_stream
    if body.messages[-1].role != "user":
        _err(400, "The last message must be from the user.")
    msgs = [m.model_dump() for m in body.messages]
    return StreamingResponse(answer_stream(kb(), msgs), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.exception_handler(Exception)
async def unhandled(request, exc):  # noqa: ARG001
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc.__class__.__name__}"})
