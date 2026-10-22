# Experiment summary (generated from result files)

Values in % (mean ± s.d. over seeds). 'Not evaluated' = no result file.

| Variant | Dataset | Subset | Seeds | Images | mAP@0.5 | mAP@[.5:.95] | P | R | Recall bright | NSR | BRB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 · YOLOv3 + BPW aug, no filter | dawn | all | 1 | 40 | 73.24 | 49.13 | 92.86 | 66.20 | 63.89 | 0.00 | 0.00 |
| B2 · ERUP (BPW + KBL) | dawn | all | 1 | 40 | 73.19 | 46.89 | 92.16 | 65.28 | 61.11 | 0.00 | 0.00 |
| E2-e · Full SG-ERUP (gate + L_sat + SHA) | dawn | all | 1 | 40 | 69.12 | 44.44 | 92.86 | 60.19 | 52.78 | 0.00 | 0.00 |

## Paired differences vs reference (mAP@0.5 points)

| Variant | Dataset | Subset | Pairs | Δ mean | 95% CI | Wilcoxon p |
|---|---|---|---|---|---|---|
| b0_no_filter | dawn | all | 1 | +0.06 | n/a | n/a |
| sg_full | dawn | all | 1 | -4.06 | n/a | n/a |
