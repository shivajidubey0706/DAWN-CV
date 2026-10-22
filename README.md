# SG-ERUP AI | Adverse Weather Intelligence

A research prototype of **SG-ERUP**, the spatially gated, saturation-aware extension of ERUP-YOLO proposed in
*"Spatially Gated, Saturation-Aware Image-Adaptive Processing for Object Detection in Adverse Weather: A Proposed
Extension of ERUP-YOLO and a Falsifiable Evaluation Protocol"* (preprint, research proposal).

> **Status: SG-ERUP is a hypothesis.** The paper does not evaluate the method. This repository implements it and
> runs a *reduced* study on DAWN (see [Measured results](#measured-results)). That study does **not** validate
> SG-ERUP, and every limitation is listed in [docs/METHOD_NOTES.md](docs/METHOD_NOTES.md).

What is included:
- **Method:** a modular PyTorch implementation of every component: BPW, KBL, the ERUP encoder, cue maps, the spatial gate, the blend, L_sat, L_tv, SHA, NSR and BRB. It also reproduces the paper's analytic checks (slope 169, 6.78 M encoder parameters, 4.95/15.14 GMAC, 1,028 gate parameters).
- **Detection:** a real YOLOv3 detector (Ultralytics YOLOv3u) used for both training and inference.
- **Data:** dataset loaders with validation and leakage checks for DAWN, VOC, RTTS, ExDark, BDD100K, ACDC, Foggy Cityscapes and YOLO-format folders.
- **Training and evaluation:** training, evaluation, ablation, statistics and cost scripts, with configs for smoke tests, the short study and the full paper protocol.
- **Backend:** a FastAPI server with real inference, a processing lab, the dataset explorer, experiment analytics, a training-job runner and Ask AI (Claude, or a labelled local fallback).
- **Frontend:** a React dashboard with weather-particle animation, scroll reveals and responsive layouts.

---

## 1. Install

Requirements: Python 3.10+ (tested on 3.14.8), Node 18+ (tested on 24.19), and optionally an NVIDIA GPU (tested on an RTX 2050 with 4 GB).

**Windows PowerShell**
```powershell
cd C:\path\to\DAWN
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1        # add -Cuda cpu for CPU-only PyTorch
```

**Linux / macOS**
```bash
cd /path/to/DAWN
bash scripts/setup_env.sh            # or: bash scripts/setup_env.sh cpu
```

The setup script does the following:
1. Creates `.venv`.
2. Installs PyTorch (CUDA 12.8 wheels by default) and `requirements.txt`.
3. Installs Playwright Chromium (for the browser tests).
4. Downloads `weights/yolov3u.pt`.
5. Runs `npm install` in `frontend/`.
6. Copies `.env.example` to `.env`.

Manual equivalent:
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/setup_weights.py
cd frontend && npm install && cd ..
```

## 2. Configure environment variables

Copy `.env.example` to `.env`. All variables are optional:
- `ANTHROPIC_API_KEY` enables LLM answers in Ask AI. The key is read only by the backend and never sent to the browser. Do not paste keys into chat or commit them.
- `SGERUP_DEVICE=cpu` keeps inference off a small GPU while training runs.
- Upload limit, CORS origins and the model id are also set here.

## 3. Add datasets

DAWN goes in `data/DAWN/{Fog,Rain,Sand,Snow}`: extract the four DAWN zips there. VOC comes from `python scripts/download_voc.py --sets 2007trainval 2007test 2012trainval`. For the other datasets, see [docs/DATASETS.md](docs/DATASETS.md). Then validate:
```bash
python scripts/validate_datasets.py            # counts, class maps, invalid boxes, leakage check
```

## 4. Model weights

`python scripts/setup_weights.py` downloads `weights/yolov3u.pt`. Trained front-end checkpoints appear under `runs/`. Details are in [docs/MODELS.md](docs/MODELS.md).

## 5. Start the app

**Terminal 1: backend** (http://127.0.0.1:8000, docs at /docs)
```powershell
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000      # Windows
```
```bash
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000            # Linux/macOS
```

**Terminal 2: frontend** (http://localhost:5173)
```bash
cd frontend
npm run dev
```

Production build: `npm run build && npm run preview`. This serves `dist/` on port 4173 with the same `/api` proxy.

## 6. Use it

- **Detection Studio:** drop an image (JPG, PNG, WebP), or click a DAWN *Fog / Rain / Snow / Sand* sample. Pick a variant and front-end weights, set the confidence threshold and click **Run Detection**. Drag the slider to compare the original with the processed image. Download the annotated image, the processed image or the JSON.
- **Processing Lab:** shows every stage: I, B(I), G, Ĩ, Î, μ, δ, ρ, and the detections. It also shows NSR and BRB. Choose a manual (illustrative) BPW curve or the encoder's prediction, and the B5 analytic gate or the variant's own gate.
- **Research Analytics:** measured tables and charts from `results/experiments/*`. Variants without results show "Not evaluated". Export as CSV, JSON or Markdown.
- **Dataset Explorer:** availability, validation, class and weather distributions, a gallery with ground-truth boxes, and path configuration.
- **Model & Experiments:** detector and front-end info, checkpoints, live training runs with loss curves, hardware, and a smoke-training launcher.
- **Ask AI:** the floating button or the `/ask` page. It shows a mode label: *LLM answer* (with `ANTHROPIC_API_KEY`) or *Local fallback mode*.

## 7. Command-line workflows

```bash
python scripts/check_equations.py                                   # analytic checks vs the paper
python scripts/infer.py data/DAWN/Fog/foggy-001.jpg                 # single image -> outputs/infer
python scripts/infer.py data/DAWN/Sand --variant b2_erup --checkpoint runs/short_real/b2_erup/seed0/best.pt
python scripts/train.py --config configs/smoke.yaml --variant sg_full
python scripts/evaluate.py --run runs/short_real/sg_full/seed0 --dataset dawn --bootstrap 200
python scripts/evaluate.py --variant b0_no_filter --dataset voc:test --limit 500
python scripts/run_ablation.py --config configs/smoke.yaml --limit 40          # quick end-to-end check
python scripts/run_ablation.py --config configs/short_real.yaml                # the reported reduced study
python scripts/run_ablation.py --config configs/short_real.yaml --summarize-only
python scripts/benchmark_cost.py                                    # E7: params, FLOPs, latency, memory
```

Exported results are written to:
- `results/experiments/<exp>/summary.{csv,json,md}`
- `map50_<dataset>.png`
- one JSON file per (variant, seed, dataset)

## 8. Tests

```bash
python -m pytest -q                                   # unit + API tests (79)
cd frontend && npm run lint && npm run build          # frontend static checks
python tests/e2e/browser_check.py                     # needs backend + frontend running; Playwright
```

The results of the last run are in [docs/TESTING_REPORT.md](docs/TESTING_REPORT.md).

## Measured results

See [docs/TESTING_REPORT.md](docs/TESTING_REPORT.md) §5 and `results/experiments/short_real/summary.md`. The protocol in brief:
- frozen COCO YOLOv3u detector;
- front ends trained on 1,200 VOC images for 3 epochs, with 2 paired seeds;
- evaluated on all 1,027 DAWN images.

This is far from the paper's protocol (8 seeds, 80 epochs, joint training). With 2 seeds no comparison can be statistically significant.

## Documentation

- [docs/METHOD_NOTES.md](docs/METHOD_NOTES.md): equations to code, assumptions, deviations
- [docs/DATASETS.md](docs/DATASETS.md) · [docs/MODELS.md](docs/MODELS.md) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/TESTING_REPORT.md](docs/TESTING_REPORT.md)
