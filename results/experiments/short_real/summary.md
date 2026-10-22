# Experiment summary (generated from result files)

Values in % (mean ± s.d. over seeds). 'Not evaluated' = no result file.

| Variant | Dataset | Subset | Seeds | Images | mAP@0.5 | mAP@[.5:.95] | P | R | Recall bright | NSR | BRB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 · YOLOv3 + BPW aug, no filter | dawn | all | 1 | 1027 | 60.34 | 38.92 | 90.45 | 62.52 | 52.54 | 0.00 | 0.00 |
| B0 · YOLOv3 + BPW aug, no filter | dawn | fog | 1 | 300 | 63.45 | 40.28 | 91.72 | 61.66 | 51.25 | 0.00 | 0.00 |
| B0 · YOLOv3 + BPW aug, no filter | dawn | rain | 1 | 200 | 66.04 | 45.45 | 88.95 | 66.71 | 56.10 | 0.00 | 0.00 |
| B0 · YOLOv3 + BPW aug, no filter | dawn | sand | 1 | 323 | 58.62 | 39.49 | 93.14 | 62.63 | 48.82 | 0.00 | 0.00 |
| B0 · YOLOv3 + BPW aug, no filter | dawn | snow | 1 | 204 | 82.65 | 49.63 | 87.75 | 60.21 | 51.38 | 0.00 | 0.00 |
| B2 · ERUP (BPW + KBL) | dawn | all | 1 | 1027 | 57.01 | 36.69 | 92.92 | 58.31 | 47.50 | 0.00 | 0.00 |
| B2 · ERUP (BPW + KBL) | dawn | fog | 1 | 300 | 59.75 | 37.65 | 92.64 | 57.43 | 47.25 | 0.00 | 0.00 |
| B2 · ERUP (BPW + KBL) | dawn | rain | 1 | 200 | 61.05 | 40.70 | 92.88 | 61.50 | 50.75 | 0.00 | 0.00 |
| B2 · ERUP (BPW + KBL) | dawn | sand | 1 | 323 | 56.07 | 37.48 | 94.00 | 59.88 | 45.88 | 0.00 | 0.00 |
| B2 · ERUP (BPW + KBL) | dawn | snow | 1 | 204 | 79.32 | 49.07 | 92.02 | 55.22 | 43.88 | 0.00 | 0.00 |
| B3 · ERUP KBL-only | dawn | all | 1 | 1027 | 55.78 | 36.43 | 92.82 | 56.24 | 42.71 | 0.00 | 0.00 |
| B3 · ERUP KBL-only | dawn | fog | 1 | 300 | 60.56 | 39.13 | 93.04 | 57.26 | 43.41 | 0.00 | 0.00 |
| B3 · ERUP KBL-only | dawn | rain | 1 | 200 | 64.38 | 42.95 | 92.43 | 60.14 | 46.04 | 0.00 | 0.00 |
| B3 · ERUP KBL-only | dawn | sand | 1 | 323 | 53.14 | 36.06 | 93.59 | 55.86 | 39.26 | 0.00 | 0.00 |
| B3 · ERUP KBL-only | dawn | snow | 1 | 204 | 79.11 | 48.12 | 92.11 | 52.93 | 40.21 | 0.00 | 0.00 |
| B4 · Global per-image gate | dawn | all | 1 | 1027 | 58.38 | 37.06 | 91.63 | 59.29 | 49.25 | 0.00 | 0.00 |
| B4 · Global per-image gate | dawn | fog | 1 | 300 | 60.64 | 38.05 | 91.97 | 58.65 | 48.91 | 0.00 | 0.00 |
| B4 · Global per-image gate | dawn | rain | 1 | 200 | 65.67 | 43.16 | 90.12 | 63.21 | 52.89 | 0.00 | 0.00 |
| B4 · Global per-image gate | dawn | sand | 1 | 323 | 57.97 | 38.22 | 94.01 | 60.81 | 47.79 | 0.00 | 0.00 |
| B4 · Global per-image gate | dawn | snow | 1 | 204 | 81.13 | 48.08 | 89.93 | 55.48 | 45.87 | 0.00 | 0.00 |
| B5 · Fixed hand-designed gate | dawn | all | 1 | 1027 | 58.43 | 36.82 | 91.86 | 58.17 | 47.12 | 0.00 | 0.00 |
| B5 · Fixed hand-designed gate | dawn | fog | 1 | 300 | 61.69 | 38.16 | 91.86 | 58.43 | 48.41 | 0.00 | 0.00 |
| B5 · Fixed hand-designed gate | dawn | rain | 1 | 200 | 69.42 | 46.88 | 90.09 | 62.36 | 51.61 | 0.00 | 0.00 |
| B5 · Fixed hand-designed gate | dawn | sand | 1 | 323 | 56.74 | 37.41 | 93.81 | 59.49 | 45.15 | 0.00 | 0.00 |
| B5 · Fixed hand-designed gate | dawn | snow | 1 | 204 | 80.17 | 47.38 | 91.16 | 53.59 | 42.66 | 0.00 | 0.00 |

## Paired differences vs reference (mAP@0.5 points)

| Variant | Dataset | Subset | Pairs | Δ mean | 95% CI | Wilcoxon p |
|---|---|---|---|---|---|---|
| b0_no_filter | dawn | all | 1 | +3.33 | n/a | n/a |
| b0_no_filter | dawn | fog | 1 | +3.70 | n/a | n/a |
| b0_no_filter | dawn | rain | 1 | +4.99 | n/a | n/a |
| b0_no_filter | dawn | sand | 1 | +2.55 | n/a | n/a |
| b0_no_filter | dawn | snow | 1 | +3.33 | n/a | n/a |
| b3_kbl_only | dawn | all | 1 | -1.23 | n/a | n/a |
| b3_kbl_only | dawn | fog | 1 | +0.82 | n/a | n/a |
| b3_kbl_only | dawn | rain | 1 | +3.33 | n/a | n/a |
| b3_kbl_only | dawn | sand | 1 | -2.93 | n/a | n/a |
| b3_kbl_only | dawn | snow | 1 | -0.21 | n/a | n/a |
| b4_global_gate | dawn | all | 1 | +1.37 | n/a | n/a |
| b4_global_gate | dawn | fog | 1 | +0.90 | n/a | n/a |
| b4_global_gate | dawn | rain | 1 | +4.62 | n/a | n/a |
| b4_global_gate | dawn | sand | 1 | +1.90 | n/a | n/a |
| b4_global_gate | dawn | snow | 1 | +1.81 | n/a | n/a |
| b5_fixed_gate | dawn | all | 1 | +1.42 | n/a | n/a |
| b5_fixed_gate | dawn | fog | 1 | +1.95 | n/a | n/a |
| b5_fixed_gate | dawn | rain | 1 | +8.37 | n/a | n/a |
| b5_fixed_gate | dawn | sand | 1 | +0.67 | n/a | n/a |
| b5_fixed_gate | dawn | snow | 1 | +0.85 | n/a | n/a |
