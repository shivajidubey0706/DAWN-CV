# Dataset setup guide

Nothing here downloads data without being asked, and no count is assumed. Every number shown in the app
comes from `scripts/validate_datasets.py` (or the **Validate** button in the Dataset Explorer) scanning
files on disk. Reports are saved to `results/datasets/<name>.json`.

| Dataset | Role (paper) | Format expected under its root | Local status | How to obtain |
|---|---|---|---|---|
| DAWN | test only | `{Fog,Rain,Sand,Snow}/*.jpg` + `<Subset>_PASCAL_VOC/*.xml` | **present**: 1,027 images, 7,199 protocol boxes | [Mendeley Data](https://data.mendeley.com/datasets/766ygrbt8y/3), extract the four zips into `data/DAWN/` |
| PASCAL VOC 2007 + 2012 | train / val; VOC07 test | `VOCdevkit/VOC2007`, `VOCdevkit/VOC2012` | **present**: 16,551 trainval + 4,952 test | `python scripts/download_voc.py --sets 2007trainval 2007test 2012trainval` (public mirrors) |
| RTTS | test only | `JPEGImages/` + `Annotations/` (VOC XML) | not downloaded | RESIDE page (Google Drive / Baidu links); manual |
| ExDark | test only | `ExDark/<Class>/<img>` + `ExDark_Annno/<Class>/<img>.txt` | not downloaded | github.com/cs-chan/Exclusively-Dark-Image-Dataset; manual |
| BDD100K | out-of-protocol test | `images/100k/val/` + `labels/det_20/det_val.json` | not downloaded | bdd-data.berkeley.edu (registration) |
| ACDC | out-of-protocol test | `rgb_anon/<cond>/val/` + `gt_detection/*val*.json` (COCO format) | not downloaded | acdc.vision.ee.ethz.ch (registration) |
| Foggy Cityscapes | depth-aware analysis | `leftImg8bit_foggy/val/<city>/*_foggy_beta_0.02.png` + `gtFine/val/<city>/*_polygons.json` | not downloaded | cityscapes-dataset.com (registration) |

## Steps

1. Put the dataset anywhere and point the project at it in one of two ways:
   - **Web UI:** Dataset Explorer → choose the dataset → *Dataset root path* → **Save path**. This writes `configs/datasets.local.yaml`.
   - **Edit** `configs/datasets.local.yaml` yourself:
     ```yaml
     datasets:
       rtts: {root: D:/data/RTTS}
     ```
2. Validate the dataset:
   - All datasets: `python scripts/validate_datasets.py` (add `--verify-images` to decode-check every file).
   - From the UI: press **Validate**.
3. The report lists:
   - valid images and missing annotations;
   - invalid boxes (x2 ≤ x1, outside the image);
   - corrupt images (with `--verify-images`);
   - raw label counts, protocol class counts, and per-weather counts.

## Class mappings (fixed in advance; `sgerup/data/class_maps.py`)

- **protocol5** (fog and multi-weather): person, bicycle, car, bus, motorbike. The source labels map as follows:
  - motorcycle and motor → motorbike
  - pedestrian and people → person
  - **truck, train, rider, van → ignore regions**. These are neither true nor false positives, like VOC "difficult" objects.
  - VOC `difficult` objects are also ignore regions.
- **lowlight10** (ExDark): bicycle, boat, bottle, bus, car, cat, chair, dog, motorbike, person. Cup and table become ignore regions. This choice of ten classes is an assumption.
- DAWN check (the paper asks for it): this release labels **bicycle (26)** and **motorcycle (81)** separately. It also contains **truck (646)** and **train (1)**, which are treated as ignore regions.

## Splits and leakage controls

- **Training and validation:**
  - Training uses VOC 2007 + 2012 trainval images that contain at least one protocol object.
  - A fixed 10% validation split (seed 0, the same for every arm) is held out.
  - Validation images get synthetic fog, gamma darkening or SHA, fixed per image index.
- **Test only:** DAWN (no official split, so all 1,027 images), VOC 2007 test, RTTS, ExDark, BDD100K, ACDC.
- **Leakage check:** every training run asserts that train, val, VOC test and DAWN are disjoint by resolved path. It aborts otherwise and writes the check to `runs/<run>/<variant>/seed*/split.json`. `scripts/validate_datasets.py` runs the same check (with optional content digests through `check_disjoint(by_content=True)`).
- **Smoke subsets:** use `train.max_train_images` / `train.max_val_images` (fixed seeded subsets, identical across arms) and `--limit N` for evaluation.

## Not implemented

- Generating the VOC-Foggy and VOC-Dark test sets (IA-YOLO's protocol). `synthetic_fog` and `synthetic_dark` exist in `sgerup/augment.py`.
- Severity sweeps.
- Distance-stratified evaluation, which needs Cityscapes disparity.
