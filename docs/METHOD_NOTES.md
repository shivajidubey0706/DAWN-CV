# SG-ERUP method notes: equations, implementation mapping, assumptions and deviations

**Status.** SG-ERUP is a *research hypothesis*. The paper states the method had not been implemented
or evaluated. This repository implements it as faithfully as the paper permits. It does **not** show
that SG-ERUP improves detection; only measured result files in `results/experiments/` may be
cited, and they come from a reduced protocol (see "Deviations").

Tensor convention everywhere: `float32`, `N×C×H×W`, **RGB**, values in **[0, 1]**, any device.

## 1. Equations and where they live

| Paper | Equation | Code |
|---|---|---|
| Luminance (Sec. III-E) | Y(x) = wᵀI(x), w = (0.27, 0.67, 0.06) | `sgerup/cues.py::luminance` |
| Eq. 1 BPW | B_θB(I)_c(x) = f_c(I_c(x); θ_B), θ_B ∈ [−1,1]¹² | `sgerup/filters/bpw.py::bpw_filter` |
| Eq. 2 KBL | K_θK(J)_c = clip(J_c ⊙ (k¹_c ∗ J_c) + k²_c ∗ J_c + J_c, 0, 1) | `sgerup/filters/kbl.py::kbl_filter` |
| Encoder | 5 × (3×3 conv, 64→1024 ch, stride-2 pool) → GAP → FC(498) → sigmoid → [−1,1] | `sgerup/encoder.py::ERUPEncoder` |
| Eq. 3 | μ(x) = (A_r ∗ Y)(x) | `cues.py::local_luminance` |
| Eq. 4 | δ(x) = min_{y∈Ω_r(x)} min_c I_c(y) (dark channel) | `cues.py::dark_channel` |
| Eq. 5 | ρ(x) = (A_r ∗ 1[max_c I_c > τ_hi])(x) | `cues.py::saturation_fraction` |
| Cue pooling | C = P_{h×w}([μ; δ; ρ]) ∈ R^{3×h×w} | `cues.py::pool_to_grid` |
| Eq. 6 gate | g = σ(W_F F + W_C C + b) ∈ (0,1)^{h×w} (1028 params) | `sgerup/gate.py::GateHead` |
| Upsampling | G = U(g), bilinear | `gate.py::upsample_gate` |
| Eq. 7 blend | Ĩ = G ⊙ B_θB(I) + (1 − G) ⊙ I | `gate.py::blend`, `frontend.py` |
| Eq. 8 | Î = K_θK(Ĩ), prediction = D(Î) | `frontend.py`, `detector.py` |
| Eq. 9 L_sat | 1/HW Σ_x [β(x) ReLU(Ỹ(x) − Y(x))² + η ReLU(Ỹ(x) − τ_hi) 1[Y(x) ≤ τ_hi]], β = σ((μ − τ_b)/s) | `sgerup/losses.py::saturation_loss` |
| Eq. 10 L_tv | 1/(hw) Σ_{i,j} \|g_{i+1,j} − g_{i,j}\| + \|g_{i,j+1} − g_{i,j}\| | `losses.py::gate_tv_loss` |
| Eq. 11 | L = L_det + λ_s L_sat + λ_tv L_tv | `sgerup/engine.py::_train` |
| Eq. 12 SHA | I_aug = M ⊙ B_θa(I) + (1 − M) ⊙ B_θa′(I) | `sgerup/augment.py::sha` |
| Eq. 13 NSR | 1/HW Σ_x 1[Ỹ(x) > τ_hi ∧ Y(x) ≤ τ_hi] | `sgerup/metrics/mechanism.py` |
| Eq. 14 BRB | Σ_x β(x) ReLU(Ỹ(x) − Y(x)) / Σ_x β(x) | `metrics/mechanism.py` |
| B5 fixed gate | G = σ(−(μ − τ_b)/s) | `gate.py::fixed_gate` |

Properties checked by unit tests (`tests/`):
- **P1:** Ĩ stays in [0, 1].
- **P2:** G ≡ 1 gives ERUP exactly, and G ≡ 0 gives KBL-only.
- **P3:** Ĩ is monotone in I for a fixed G.
- **P4:** at initialisation, G ≈ σ(b₀).
- BPW with all-zero parameters is exactly the identity.
- Gradients reach θ_B, the gate head, and the encoder.

Verified analytic reproductions (`python scripts/check_equations.py`, output in `results/checks/equations.json`):

| Quantity | Paper | This code |
|---|---|---|
| BPW slope, both control points at (0, 1), L = 8 | ≈ 169 | **169.0** |
| Encoder parameters | 6.78 M | **6,781,042** |
| Encoder MACs at 256 input | 4.95 G | **4.946 G** |
| Encoder MACs at 448 input | 15.14 G | **15.145 G** |
| Gate head parameters | 1,028 | **1,028** |
| BPW slope, random parameters | "about 50" | **median 3.9, p90 8.6** (not reproduced; see A1) |

## 2. Assumptions (details the paper or ERUP-YOLO leave unspecified)

| # | Item | Choice here | Where |
|---|---|---|---|
| A1 | BPW (r, θ) → control point mapping | x_j = b_j + r·(b_j if r<0 else 1−b_j), y_j likewise with θ, b_j = j/3. Zero gives identity. Reproduces the 169 slope but **not** the paper's "≈50 for random parameters", so the original mapping is probably different. | `filters/bpw.py` |
| A2 | BPW interval count L | 8 (the paper's own re-implementation value) | `configs/base.yaml` |
| A3 | KBL padding, kernel constraint | reflect padding; kernels in [−1, 1], `kbl_kernel_scale` = 1 | `filters/kbl.py` |
| A4 | Encoder activation, pooling, input | LeakyReLU(0.1), max pool, 256×256 input | `encoder.py` |
| A5 | Encoder FC initialisation | N(0, 10⁻³) weights, zero bias, so processing starts near the identity | `encoder.py` |
| A6 | Gate head initialisation | W = 0, b = b₀ = 2 (paper: b₀ ∈ {0, 2}) | `gate.py` |
| A7 | Cue radius r, τ_hi, τ_b, s, η | 7, 0.95, 0.6, 0.05, 1.0 | `configs/base.yaml` |
| A8 | λ_s, λ_tv | 10, 0.1 (provisional, not tuned on the validation split) | `configs/variants.yaml` |
| A9 | ERUP random-curve augmentation | p = 0.5. Brightening: r ~ U(−1, 0), θ ~ U(0, 1); darkening is the mirror. | `augment.py` |
| A10 | SHA mask | vertical ramp (amplitude U(0.4, 1)) plus 4×4 bicubic Gaussian field (σ = 0.35), clipped; p_sha = 0.5; optional scattering branch off by default | `augment.py` |
| A11 | Validation degradations | per image (fixed by index): clean, IA-YOLO fog, gamma U(1.5, 5), or SHA | `engine.py::degrade_val` |
| A12 | Gate grid | encoder feature map (8×8 for a 256 input); configurable | `frontend.py` |
| A13 | Cues on letterboxed input | the cues see the grey letterbox padding | `pipeline.py` |
| A14 | ExDark "ten classes" | ExDark's 12 classes minus cup and table | `data/class_maps.py` |
| A15 | Non-protocol classes (truck, train, rider, van) | ignore regions: neither TP nor FP | `data/class_maps.py` |

## 3. Deviations from the paper's protocol (honest list)

1. **Detector.** The paper uses YOLOv3 with its original anchor-based head and loss. Here the detector is Ultralytics **YOLOv3u**: the YOLOv3 Darknet-53 backbone and neck, an anchor-free head, the v8 loss (TAL, BCE, CIoU, DFL), and COCO pretraining. L_det is this loss.
2. **Frozen detector in the reported runs.** The RTX 2050 (4 GB) cannot train a 104 M-parameter detector jointly. `train.freeze_detector: true` trains only the front end (encoder + gate). Joint training is supported (`freeze_detector: false`) but has not been run. As a consequence, B0 in the reported runs is the pretrained detector *without* BPW-augmentation training.
3. **Training budget.** `configs/short_real.yaml` uses 1,200 VOC trainval images with protocol objects, 3 epochs, lr 10⁻³ cosine to 10⁻⁶, batch 4, and **2 seeds**. The paper calls for full VOC, 80 epochs, lr 10⁻⁴, batch 6 and **8 seeds**. With 2 seeds no Wilcoxon test can reach significance, so every comparison is inconclusive by construction.
4. **Hyperparameters were not tuned.** A7 and A8 were not searched on the validation split.
5. **Datasets.** Only DAWN and VOC 2007/2012 were available locally. RTTS, ExDark, BDD100K, ACDC and Foggy Cityscapes loaders are implemented and unit-tested on synthetic fixtures but were **not evaluated**. VOC-Foggy and VOC-Dark test sets are not generated.
6. **Metrics.** The evaluator is our own implementation (VOC all-point AP; COCO-style 101-point AP with box areas), not pycocotools. TIDE, D-ECE and severity sweeps (E6, E10) are not implemented.
7. **Statistics.** t-intervals, the paired Wilcoxon test, Holm correction and an image-level bootstrap (200 resamples in the short study; the paper specifies 1,000) are implemented in `sgerup/stats.py` and `metrics/detection.py`.

## 4. Variants (configs/variants.yaml)

| Key | Paper ID | Front end |
|---|---|---|
| `b0_no_filter` | B0 | none (detector only) |
| `b2_erup` | B2 | BPW + KBL, G ≡ 1 |
| `b3_kbl_only` | B3 | KBL only (G ≡ 0) |
| `bpw_only` | [8] Fig. 7 | BPW only |
| `b4_global_gate` | B4 | learned 1×1 gate + L_sat + SHA |
| `b5_fixed_gate` | B5 | fixed σ(−(μ − τ_b)/s) gate |
| `sg_gate` | E2-a | learned spatial gate (+ L_tv) |
| `sg_lsat_global` | E2-b | L_sat on ERUP, no gate |
| `sg_gate_sha` | E2-c | gate + SHA, λ_s = 0 |
| `sg_gate_sat` | E2-d | gate + L_sat, no SHA |
| `sg_full` | E2-e | gate + L_sat + SHA (proposed) |

Further ablations from the paper are config switches:
- E2-f: `frontend.use_gate_cues` / `frontend.use_gate_features`
- E2-g: `frontend.gate_kbl`
- E2-h: `loss.sat_on: final`
- E2-i: `frontend.gate_grid`
- E2-k: `frontend.b0`

E2-j (guided-filter upsampling) is not implemented.
