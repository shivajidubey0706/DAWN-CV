# Architecture and directory guide

```
                ┌──────────────────────── React + Vite frontend (frontend/) ──────────────────────────┐
                │ Overview · Detection Studio · Processing Lab · Analytics · Datasets · Experiments   │
                │ Docs · Ask AI (floating + page) · canvas weather particles · scroll reveals         │
                └──────────────┬──────────────────────────────────────────────────────────────────────┘
                               │ /api (Vite proxy in dev; same origin, no secrets in the browser)
┌──────────────────────────────▼──────────────── FastAPI backend (backend/app) ─────────────────────────┐
│ main.py   health/ready/system · model status · /detect · /process · results download                 │
│           datasets (scan, validate, path, samples, images) · experiments (summary, export, plots)     │
│           training runs + whitelisted smoke jobs (subprocess) · /paper · /docs · /ask (SSE)           │
│ state.py  lazy detector pipeline (+ GPU lock) · in-memory result store · dataset cache · job runner   │
│ ask.py    Claude (anthropic SDK, server-side key) with retrieved context ─or─ local fallback mode      │
│ knowledge.py  BM25 over knowledge/paper.md, faq.md, docs/*.md, README + digest of measured results    │
└──────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                               │ imports
┌──────────────────────────────▼──────────────── sgerup/ (research package) ──────────────────────────┐
│ filters/bpw.py, filters/kbl.py   Eq. 1, Eq. 2                                                         │
│ cues.py      μ, δ, ρ (Eqs. 3-5)        encoder.py   ERUP encoder (6.78 M)                             │
│ gate.py      gate head (1028), upsample, blend, fixed gate (Eqs. 6-7)                                 │
│ frontend.py  SG-ERUP front end, all variants (Eq. 8)                                                  │
│ losses.py    L_sat, L_tv (Eqs. 9-10)   augment.py   random BPW, SHA (Eq. 12), synthetic fog/dark     │
│ detector.py  YOLOv3u wrapper: loss for training, NMS for inference                                   │
│ pipeline.py  letterbox → front end → detector → boxes in original pixels (+ intermediates)          │
│ engine.py    training loop (Eq. 11), validation, evaluation       report.py  tables/plots/stats      │
│ metrics/     detection (VOC AP, COCO AP, sizes, P/R, brightness tertiles, bootstrap), NSR/BRB       │
│ data/        registry, parsers (DAWN, VOC, RTTS, ExDark, BDD, ACDC, Cityscapes, YOLO), splits       │
│ stats.py     t-CI, Wilcoxon, Holm      cost.py   FLOPs, latency, memory      utils.py   seeds, env    │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Directory

| Path | Purpose |
|---|---|
| `sgerup/` | Method implementation (importable package, no web dependencies) |
| `scripts/` | CLIs: `setup_env.ps1/.sh`, `setup_weights.py`, `download_voc.py`, `validate_datasets.py`, `train.py`, `evaluate.py`, `infer.py`, `run_ablation.py`, `benchmark_cost.py`, `check_equations.py` |
| `configs/` | `base.yaml`, `variants.yaml`, `datasets.yaml`, `smoke.yaml`, `short_real.yaml`, `paper_full.yaml` |
| `backend/app/` | FastAPI server |
| `frontend/` | React 19 + TypeScript + Vite, framer-motion, recharts, KaTeX |
| `knowledge/` | Paper text, references, structured paper metadata, FAQ (Ask AI corpus) |
| `docs/` | Method notes, dataset/model guides, this file, testing report |
| `tests/` | pytest unit + API tests; `tests/e2e/browser_check.py` (Playwright) |
| `data/` | Datasets (git-ignored); DAWN and VOC are present locally |
| `weights/` | Detector weights (git-ignored) |
| `runs/` | Training runs and checkpoints |
| `results/` | `experiments/<exp>/*.json` (per variant, seed, dataset), `summary.{json,csv,md}`, plots; `cost/`, `checks/`, `datasets/` |
| `logs/` | Server logs, e2e report, screenshots, ablation log |

## Data flow for one detection request

1. **Upload validation:** size ≤ 15 MB; the file must decode; JPEG, PNG and WebP only.
2. **Letterbox:** EXIF transpose, then letterbox to 448 with grey (114) padding, as a float tensor in [0, 1].
3. **Front end:** either trained (from `runs/…/best.pt`) or identity-initialised. It returns Î plus the intermediates.
4. **Detection:** YOLOv3u forward pass, then NMS (classes optionally limited to protocol5).
5. **Post-processing:** boxes are mapped back to original pixels. NSR and BRB are computed. Annotated, processed and JSON outputs go into an in-memory LRU store and are downloadable from `/api/results/{id}/…`.
