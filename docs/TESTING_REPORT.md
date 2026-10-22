# Testing report

Environment:
- Windows 11, Python 3.14.8, torch 2.11.0+cu128, ultralytics 8.4.183
- NVIDIA RTX 2050 (4 GB), Node 24.19
- Date: 2026-10-22

Only tests that actually ran are marked PASS.

## 1. Code-level tests

| Test | Command | Status | Observed |
|---|---|---|---|
| Unit + API tests (BPW, KBL, cues, gate, blend, losses, gradients, ranges, augmentation, metrics, stats, dataset parsers, config, upload validation, API errors, Ask AI fallback mode, Ask AI configured mode with a mocked SDK, path traversal) | `python -m pytest -q` | **PASS** | 79 passed |
| Analytic checks vs paper | `python scripts/check_equations.py` | **PASS (4 of 5)** | Slope 169.0, encoder 6,781,042 params, 4.946 / 15.145 GMAC, gate 1,028 params all match. The random-parameter slope (≈50 in the paper) is **not** reproduced: median 3.9. |
| Frontend type-check + lint | `npx tsc -b`, `npm run lint` | **PASS** | 0 errors, 8 warnings (fast-refresh export hints, dev only) |
| Frontend production build | `npm run build` | **PASS** | built in about 5 s |
| Dataset validation + leakage | `python scripts/validate_datasets.py` | **PASS** | DAWN 1,027 images / 7,199 protocol boxes; VOC trainval 16,551; VOC test 4,952; train, test and DAWN disjoint. RTTS, ExDark, BDD100K, ACDC and Foggy Cityscapes are reported as *missing*. |

## 2. Integration tests (backend on CPU + Vite dev server, real HTTP)

| Test | Status | Observed |
|---|---|---|
| Frontend reaches the backend through the /api proxy | **PASS** | `/api/health` returned ok through port 5173 |
| Real detection on an uploaded DAWN image | **PASS** | 7 cars on `sand_storm-151.jpg`, boxes inside the image, scores ≥ 0.25 |
| Processing pipeline (all intermediate maps) | **PASS** | 9 maps plus NSR/BRB returned |
| Downloads (annotated, processed, JSON) | **PASS** | 200 responses; JSON downloaded in the browser |
| Dataset Explorer uses scanned files | **PASS** | sand filter → 323 images; annotation modal shows the real XML boxes |
| Analytics reads saved result files | **PASS** | 18 charts rendered from `results/experiments` |
| Ask AI, fallback mode | **PASS** | streamed, labelled "Local fallback mode" |
| Ask AI, configured mode with a real key | **NOT RUN** | No API key is configured. The configured path is covered only by the mocked-SDK test. |

## 3. Browser tests (Playwright, Chromium): `python tests/e2e/browser_check.py`

**59/59 PASS** (`logs/e2e_report.json`, screenshots in `logs/screenshots/`):
- All 8 pages at 1440 px and 390 px have no console errors, no 5xx responses and no horizontal overflow.
- Scroll reveals replay when scrolling down and up.
- Studio: detection, download, and the unsupported-file error.
- Lab: processing.
- Datasets: the image modal.
- Ask AI: streaming.
- Reduced-motion mode leaves content visible.

Bugs found and fixed:
- mobile overflow (`min-width: 0` on grid children);
- the hamburger menu appearing on desktop (CSS precedence);
- a B0 checkpoint being pre-selected;
- `/api/system` creating a CUDA context on a CPU-configured server.

## 4. Final smoke test: `python scripts/smoke_test.py`

**9/9 PASS** (`logs/smoke_test.json`): image → SG-ERUP front end → YOLOv3u. Recall against DAWN ground truth is 1.0 on the test image. The gate stays in [0, 1].

## 5. Reduced training study (`configs/short_real.yaml`): INCOMPLETE

The protocol:
- frozen COCO YOLOv3u detector;
- front ends trained on 1,200 VOC images, 3 epochs, lr 10⁻³;
- evaluated on all 1,027 DAWN images, 5 classes, 448 input.

The run was **stopped by the system (low memory)** after 5 of 18 runs:
- **Completed:** seed 0 of B0, B2, B3, B4 and B5.
- **Not evaluated:** sg_gate, sg_gate_sat, sg_gate_sha, sg_full (the proposed method), and all of seed 1.

mAP@0.5 (%) for seed 0. These are single runs, so no statistical claim is possible.

| Variant | all | fog | rain | sand | snow |
|---|---|---|---|---|---|
| B0 detector only (untrained) | 60.34 | 63.45 | 66.04 | 58.62 | 82.65 |
| B2 ERUP (BPW+KBL) | 57.01 | 59.75 | 61.05 | 56.07 | 79.32 |
| B3 KBL-only | 55.78 | 60.56 | 64.38 | 53.14 | 79.11 |
| B4 global gate | 58.38 | 60.64 | 65.67 | 57.97 | 81.13 |
| B5 fixed gate | 58.43 | 61.69 | 69.42 | 56.74 | 80.17 |
| SG-ERUP variants | Not evaluated | | | | |

Interpretation (exploratory, one seed):
- Every trained front end is below the untrained COCO detector on DAWN. The front ends were trained on VOC images while the detector stayed frozen, so this setup does not reproduce the paper's B0-vs-ERUP comparison.
- NSR is ≈ 0 for every variant. The trained curves barely brighten, so the mechanism that SG-ERUP targets does not occur in this short regime.

## 6. Not done / blockers

- **Proposed-method results and seed 1:** resume with `python scripts/run_ablation.py --config configs/short_real.yaml`. Finished runs are skipped.
- **Cost benchmark (E7):** not run, because the GPU was busy with training. Command: `python scripts/benchmark_cost.py`.
- **Full paper protocol:** 8 seeds, 80 epochs, joint detector training, plus RTTS, ExDark, BDD100K and ACDC. This needs a larger GPU and the manual dataset downloads.
- **Real-LLM Ask AI test:** needs `ANTHROPIC_API_KEY` in `.env`.
