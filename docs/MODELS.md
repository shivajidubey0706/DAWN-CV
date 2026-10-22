# Model and checkpoint setup guide

## Detector weights

- **File:** `weights/yolov3u.pt`. It is about 198 MB with 103,754,144 parameters, COCO-pretrained, and has 80 classes.
- **Fetch it:** `python scripts/setup_weights.py`. This downloads from the Ultralytics GitHub release. Add `--tiny` to also get `yolov3-tinyu.pt`.
- **What it is:** Ultralytics YOLOv3u. It uses the YOLOv3 Darknet-53 backbone and neck, but with an anchor-free head and the v8 loss. It is **not** the original anchor-based YOLOv3 used by ERUP-YOLO; see `docs/METHOD_NOTES.md` §3.
- **Using different weights:**
  - Set `SGERUP_DETECTOR_WEIGHTS` for the API.
  - Set `detector_weights:` in a config for training and evaluation.
  - The loader rejects these cases with a clear message: a missing file, a truncated file (< 1 MB), a non-detection model, or an unreadable file.
- **Device:** set `SGERUP_DEVICE=auto|cpu|cuda`. Inference falls back to CPU automatically when CUDA is unavailable.

## SG-ERUP front-end checkpoints

`scripts/train.py` writes everything for a run to `runs/<run_name>/<variant>/seed<k>/`:

| File | Content |
|---|---|
| `best.pt` | Front-end state dict and full config. Selected by validation mAP@0.5 on the synthetic validation split, never on test data. |
| `last.pt` | Same as `best.pt` plus optimizer, scheduler, AMP scaler, augmentation RNG and history, for `--resume`. |
| `config.json`, `environment.json` | Exact configuration; library versions, GPU, driver and deterministic flag. |
| `split.json` | Train and val image ids, plus the leakage-check result. |
| `history.json`, `train_log.jsonl`, `loss_curve.png` | Per-epoch and per-step losses, validation metrics, gate mean. |
| `status.json`, `error.log` | Live progress for the UI; traceback on failure. |

- Checkpoints are loaded with `torch.load(weights_only=True)` and validated for the `frontend` and `config` keys (`sgerup/pipeline.py`).
- The web app lists only checkpoints found under `runs/`. It never loads an arbitrary path from a request.
- **Labelling rule used throughout the UI:**
  - Without a trained checkpoint, a front end is *untrained (identity-initialised)*. Detection then equals "the baseline detector with the SG-ERUP processing prototype", not a trained SG-ERUP model.
  - `short_real` checkpoints are trained with a **frozen** detector for 3 epochs. They are not a fully trained SG-ERUP model.

## Training commands

```bash
# smoke test (minutes)
python scripts/train.py --config configs/smoke.yaml --variant sg_full
# reduced study used for the reported results (about 8 min per run on an RTX 2050)
python scripts/run_ablation.py --config configs/short_real.yaml
# resume an interrupted run
python scripts/train.py --config configs/short_real.yaml --variant sg_full --seed 1 --resume
# paper protocol (8 seeds, 80 epochs, joint detector training: needs a much larger GPU)
python scripts/run_ablation.py --config configs/paper_full.yaml
# override any field
python scripts/train.py --config configs/smoke.yaml --variant sg_gate_sat loss.lambda_s=5 frontend.b0=0 train.epochs=4
```
