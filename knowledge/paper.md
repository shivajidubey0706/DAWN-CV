# Paper full text (extracted from the supplied PDF with pdftotext)

Source: Spatially Gated, Saturation-Aware Image-Adaptive Processing for Object Detection in Adverse Weather (preprint, research proposal, not peer reviewed). Equation symbols may be garbled by PDF extraction; see knowledge/method.md for clean equations.

PREPRINT – RESEARCH PROPOSAL, NOT PEER REVIEWED

1

Spatially Gated, Saturation-Aware Image-Adaptive Processing for Object Detection in Adverse
Weather: A Proposed Extension of ERUP-YOLO and a Falsifiable Evaluation Protocol
Author Name(s)

Abstract—Image-adaptive object detectors place a small, differentiable image-processing stage in front of a detector and learn its parameters end-to-end from the detection loss. ERUP-YOLO, the most recent method of this family built on YOLOv3, unifies classical pixel-wise filters into one Bézier-curve pixel-wise (BPW) filter and classical sharpening and defogging into one kernelbased local (KBL) filter. Its authors report that the globally applied BPW curve can brighten regions that are already bright because of atmospheric scattering, and that on the sand subset of DAWN ERUP-YOLO scores 27.82 mAP@0.5 against 31.18 for a YOLOv3 baseline trained with the same augmentation. A rereading of the published numbers further shows that five of the seven per-condition gains over that matched baseline are below 1.6 points, comparable to or smaller than the run-to-run spread the same paper reports for eight repeated trainings. We propose, but have not yet evaluated, a minimal extension that keeps the BPW and KBL filters and adds three components: (i) a lowresolution spatial gate, with about one thousand parameters, that blends the BPW output with the unprocessed image according to learned features and explicit luminance and haze cues; (ii) a saturation-aware penalty on brightening already-bright regions; and (iii) a spatially heterogeneous photometric augmentation, without which a spatial gate receives no training signal. ERUPYOLO is the special case in which the gate is identically one. We formulate four falsifiable hypotheses, a paired multi-seed design with pre-specified non-inferiority margins, cross-dataset and held-out-condition tests, brightness- and size-stratified error analysis, and a cost protocol covering parameters, FLOPs, latency, and memory. No experimental result of the proposed method is reported; the contribution of this manuscript is the audit, the method specification, and the protocol needed to test it.
Index Terms—Adverse weather, image-adaptive object detection, differentiable image processing, illumination-aware gating, oversaturation, domain generalization, YOLO, reproducibility.
I. INTRODUCTION
O BJECT detectors trained on clear, well-exposed images lose accuracy under fog, haze, rain, snow, sandstorms, and low light because the test distribution departs from the training distribution [1], [2]. Three families of remedies exist: restoring the image before detection, adapting or generalizing the detector’s features across domains [3]–[5], and imageadaptive detection, in which a small network predicts the parameters of differentiable image filters that are applied
Manuscript status: research proposal and study protocol (October 2026). The method proposed here has not yet been implemented or evaluated; all tables reporting the proposed method are templates without values.
Correspondence: [author e-mail].

before the detector and trained with the detection loss [6]– [8].
ERUP-YOLO [8] is a recent and compact member of the third family. It replaces the hand-picked chains of whitebalance, gamma, contrast, tone, sharpening, and defogging filters used by IA-YOLO [6] and GDIP-YOLO [7] with two filters: a per-channel monotone Bézier tone curve (BPW) and a locally linear convolutional filter (KBL). The same BPW curve is reused, with random parameters, as a single dataaugmentation policy for both fog and low light. Trained only on PASCAL VOC [9] with this augmentation, ERUP-YOLO reports the best mAP@0.5 among the compared YOLOv3based methods on VOC-Foggy, RTTS [10], VOC-Dark, and ExDark [11].
The same paper documents a failure mode. On the sand subset of DAWN [12], ERUP-YOLO reaches 27.82 mAP@0.5 against 31.18 for YOLOv3 trained with the same BPW augmentation, and the authors attribute this, and a smaller drop of BPW+KBL relative to KBL alone on foggy test sets, to the BPW curve raising the brightness of distant regions that scattering has already brightened [8, Secs. 5.4– 5.5, Supp. Sec. 7–8]. The cause is structural: the BPW curve is a global mapping, one curve per color channel for the whole image, so it cannot brighten a dark foreground without also brightening a hazy background of similar or higher intensity.
Spatially varying enhancement is not new in image enhancement: Zero-DCE predicts per-pixel curve parameters [13], DeepLPF predicts local parametric filters [14], HDRNet predicts a bilateral grid of local affine transforms [15], and exposure-correction networks treat over- and under-exposure jointly [16], [17]. Detection-oriented low-light methods modulate features spatially [18]–[20]. What we did not find, in a search of the literature up to October 2026 (Section II), is a study that adds spatial control to the compact, parameterpredicting filter family of IA-YOLO, GDIP-YOLO, and ERUP-YOLO and tests, under a matched-augmentation, multiseed protocol, whether doing so removes the documented sand and fog regression without losing the low-light gain. We therefore present the idea as a targeted, incremental hypothesis rather than a broad novelty claim.
Research questions.
RQ1 Does restricting the BPW curve spatially, using a learned gate conditioned on luminance and haze cues, recover the


PREPRINT – RESEARCH PROPOSAL, NOT PEER REVIEWED

2

sand and fog accuracy lost by ERUP-YOLO relative to a matched YOLOv3 baseline? RQ2 Does it do so without a meaningful loss on low-light and clear-weather data? RQ3 Is the effect caused by the intended mechanism, namely less brightening of already-bright regions, rather than by extra capacity or a changed training recipe? RQ4 Are any differences larger than seed-to-seed variation, and do they persist on datasets and conditions not used for any design decision? Contributions. Every contribution below is a specification or an analysis; none is an experimental result.
1) An audit of ERUP-YOLO that separates limitations the authors acknowledge from concerns we infer, including a re-analysis of the published tables suggesting that the change of augmentation policy, rather than the filters, accounts for most of the difference between the two YOLOv3 baselines the paper reports (Section III).
2) A minimal extension, SG-ERUP (saturation-gated ERUP), consisting of a spatial gate with ≈1.0 k parameters, a saturation-aware penalty, and a spatially heterogeneous augmentation, with ERUP-YOLO as an exact special case (Section IV).
3) Four falsifiable hypotheses with pre-specified endpoints and margins, mechanism metrics that do not require depth annotations, and an analytic cost estimate of the ERUP parameter encoder that motivates measured efficiency reporting (Sections III and V).
4) An experimental matrix with mandatory and optional experiments, leakage controls, and paired statistical analysis, released with this manuscript as experiments.md.
II. RELATED WORK
A. Image-adaptive object detection
IA-YOLO [6] predicts, from a down-sampled copy of the input, the parameters of six differentiable filters (defog, white balance, gamma, contrast, tone, sharpen) that are applied in a fixed order before YOLOv3, and is trained on a mixture of clear and synthetically degraded VOC images. GDIPYOLO [7] applies the filters in parallel and combines them with learned gating weights; the weights are per image, not per pixel. ERUP-YOLO [8] reduces the filter bank to BPW and KBL and replaces condition-specific augmentation with random BPW curves. Recent journal work keeps the same design with global parameters, either adding feature-level attention [21] or compacting the predictor [22]. In all of these methods the image-level filter parameters are shared by every pixel of the image; the only spatially varying parts are fixed-size convolution kernels (KBL, sharpening) or the darkchannel transmission inside the defog filter of IA-YOLO.
B. Spatially adaptive and exposure-aware enhancement
Outside detection, spatially varying photometric mappings are well established. Zero-DCE [13] estimates per-pixel quadratic curve parameters and includes an exposure-control

loss that pulls local mean intensity towards a target level; this is the closest existing analogue of our saturation penalty, but it is defined for visual quality, not for detection, and it pulls dark and bright regions towards the same level. DeepLPF [14] predicts elliptical, graduated, and polynomial local filters; HDRNet [15] predicts a low-resolution bilateral grid of affine colour transforms and slices it with a guidance map, the same low-resolution-then-upsample pattern we adopt for the gate. Image-adaptive 3D look-up tables [23], [24] are global but highly expressive. Multi-scale exposure correction [16] and IAT [17] explicitly handle over-exposure. Physical haze models [25] and the dark-channel prior [26] relate local brightness and low saturation to scattering, which motivates the cues we feed to the gate.
C. Detection under low light and adverse weather
Low-light detectors increasingly operate on features rather than pixels: MAET learns degradation-aware representations [27], FeatEnHancer enhances hierarchical features [18], YOLA learns illumination-invariant features [19], PE-YOLO uses a pyramid enhancement network [28], and IADNet gates backbone features with illumination-guided per-pixel weights [20]. TogetherNet jointly trains restoration and detection for adverse weather [29]. These methods are strong comparison points for low light but are usually evaluated with different detectors and on a single condition, which makes direct comparison with the YOLOv3-based image-adaptive line difficult.
D. Generalization, robustness, and evaluation practice
Single-domain generalization benchmarks train on clear daytime driving data and test on unseen dusk, night, rain, and fog conditions [4], [5]; corruption benchmarks quantify degradation as a function of severity [1], [2]. Driving datasets with adverse-weather tags (BDD100K [30], ACDC [31]) and with depth (Foggy Cityscapes, whose fog is synthesized from depth [32]; Seeing Through Fog, recorded with lidar [33]) support cross-dataset and distance-aware evaluation. Error decomposition [34], calibration [35], [36], uncertainty modelling [37], and variance-aware benchmarking [38] provide the analysis tools we use.
Table I summarizes the methods closest to the proposed work.
III. AUDIT OF ERUP-YOLO, RESEARCH GAPS, AND PROBLEM FORMULATION
All numbers in this section are quoted from [8]; differences are our arithmetic.
A. What the paper establishes
ERUP-YOLO trains YOLOv3 with an image-processing front end for 80 epochs (SGD, cosine schedule from 10−4 to 10−6, weight decay 5 × 10−4, input 448 × 448, batch size 6) on VOC 2007+2012 trainval with BPW augmentation, without real adverse-weather training data [8, Sec. 5.1]. A fivelayer convolutional encoder predicts 12 BPW parameters (two control points per channel) and 2 × 9 × 9 × 3 = 486 KBL kernel weights. Evaluation uses mAP@0.5 only.


PREPRINT – RESEARCH PROPOSAL, NOT PEER REVIEWED

3

TABLE I COMPARISON OF CLOSELY RELATED METHODS. ENTRIES SUMMARIZE THE CITED PAPERS; “GLOBAL” MEANS ONE PARAMETER SET PER IMAGE.

Reference IA-YOLO [6]

Method
Six differentiable filters, global parameters, fixed order, YOLOv3

GDIP-YOLO [7] ERUP-YOLO [8]

Parallel filters with learned per-image gates BPW tone curve + KBL local linear filter; BPW augmentation

Zero-DCE [13]
HDRNet [15]
DeepLPF [14]
Exposure corr. [16], [17] FeatEnHancer [18], YOLA [19], IADNet [20] Single-DG detection [4], [5]

Per-pixel curve maps, no-reference losses incl. exposure control Low-res bilateral grid of local affine transforms Learned local parametric filters Networks for over- and under-exposure Feature-level, often spatial, enhancement or gating Train on clear daytime, test on unseen weather

Data (as reported)
VOC (+synthetic fog / dark), RTTS, ExDark
VOC-Foggy, RTTS, ExDark VOC, VOC-Foggy/Dark, RTTS, ExDark, DAWN Low-light enhancement data
Photo retouching
Photo enhancement datasets Exposure datasets
ExDark, DARK FACE, others
Diverse-weather driving benchmark

Strength
First end-to-end image-adaptive YOLO
Removes manual ordering Compact, customization-free; single augmentation
Spatially varying, lightweight
Real-time spatially varying edits Interpretable local filters Handle both directions Strong low-light detection
Clean unseen-condition protocol

Limitation
Hand-set filter set, order, ranges; condition-specific augmentation Gates are global; more filter compute Global BPW oversaturates bright regions (sand, fog); single-run tables Optimized for appearance, not detection Needs paired targets; heavier than a gate Appearance objective
Not detection-driven
Low-light only; different detectors
Different detectors; no image-adaptive stage

Relevance to this work Direct baseline; same protocol family
Gating precedent at image level; baseline Primary reference; extended here
Prior art for per-pixel curves and exposure penalties
Prior art for low-res-thenupsample control Prior art for local filter parameterization Motivates saturation handling
Alternative to pixel-level gating; comparison where compatible
Protocol for RQ4

B. Re-analysis of the published numbers
Matched-augmentation gains are small except for snow. Table 3 of [8] trains YOLOv3, GDIP-YOLO, and ERUPYOLO with the same BPW augmentation on five classes. Fig. 1 plots the ERUP-YOLO minus YOLOv3 differences: +1.51 (VOC), +1.59 (ExDark), +0.80 (RTTS), +0.65 (DAWN rain), +0.29 (DAWN fog), +11.00 (DAWN snow), and −3.36 (DAWN sand). Each entry is a single number with no reported dispersion.
Run-to-run spread is of the same order. The ablation of [8, Fig. 7] repeats training eight times per configuration. Read visually from the box plots (our reading, approximately ±0.2 points), the interquartile ranges of the BPW+KBL configuration span roughly 0.5–1.5 mAP points. Three of the seven matched differences (0.29–0.80) are smaller than these ranges and two (1.51, 1.59) are of the same order, so none of these five can be distinguished from seed variation without repeated, paired runs.
Tabulated values appear to come from the upper part of the run distribution. For the BPW+KBL configuration, the values in Tables 1–2 of [8] (e.g., 77.89 on VOC test, 74.09 on VOC-Foggy, 49.81 on RTTS) lie at or above the upper quartile, and in several cases near the maximum, of the corresponding eight-run box plots. The paper does not state which run is tabulated. If the best run was reported, comparisons with single-run baselines are biased in favour of ERUP-YOLO.
The augmentation change is larger than any filter effect. The same YOLOv3, five classes, and test sets appear in Table 1 (fog augmentation, 64.13 VOC / 30.80 RTTS) and Table 3 (BPW augmentation, 76.38 / 49.01). The differences, +12.25 and +18.21 points, exceed every gain attributed to the filters. The paper does not state whether the Table 1 baselines were

11.00 10

∆ mAP@0.5

5 1.51
0

1.59

0.80

0.65

0.29

−5

−3.36

VOC ExDark RTTS Rain Fog Snow

Fig. 1. ERUP-YOLO minus YOLOv3 mAP@0.5, both trained with the same BPW augmentation, computed from Table 3 of [8] (single runs; Rain, Fog, Snow, Sand are DAWN subsets). The eight-run interquartile ranges reported in Fig. 7 of [8] are roughly 0.5–1.5 points.

retrained or transcribed from earlier work, so part of this difference may also reflect other training details. Either way, a fair evaluation of any filter change must hold the augmentation fixed.
The ablation has no “no filter” arm. Fig. 7 of [8] compares BPW+KBL, KBL, and BPW, but not YOLOv3 with the same augmentation and no filter, nor ERUP filters with the earlier augmentation. KBL alone matches or exceeds BPW+KBL on VOC-Foggy and RTTS.
The BPW curve can be very steep. Re-implementing Eqs. 11–14 of [8] with L = 8 intervals (the value of L is not given), the slope of a single segment reaches about 169 when both control points sit at (0, 1), and about 50 for random parameters in the allowed range (script tools/check_equations.py). Such slopes amplify noise in dark regions; no slope bound or regularizer is described. This is a property of the parameterization, not a measured failure.


PREPRINT – RESEARCH PROPOSAL, NOT PEER REVIEWED

4

C. Cost of the image-processing front end
The paper reports no parameter count, FLOPs, or latency. From its description of the encoder (five 3×3 convolutions with 64–1024 channels, each followed by stride-2 pooling, then a fully connected layer to 498 outputs), and assuming same-padding with biases, we compute 6.78 M parameters. The multiply–accumulate count (MAC) depends on the encoder’s input resolution, which the paper does not state: 4.95 GMAC at 256×256 and 15.14 GMAC at 448×448 (script tools/encoder_cost.py). For scale, the Darknet project page lists 65.86 B FLOPs for YOLOv3 at 416 × 416 [39]; under the common convention of two FLOPs per MAC (which should be confirmed for that figure), a full-resolution encoder would add a substantial fraction of the detector’s compute. These are estimates from a textual description, not measurements; they motivate measuring cost (Gap C) rather than establish it.
D. Research gaps
Table II lists the six candidate gaps examined, the evidence for each, and our assessment of whether it is established. We use established when the original paper documents it, supported when it follows from the paper’s own numbers or omissions, and tentative when it rests on our reasoning and still needs a measurement.
We choose Gap A as the primary direction because it is the only one with a documented accuracy regression and a mechanism stated by the original authors, it can be addressed with a small change to the method, and its test requires the controls of Gap F anyway. Gaps B–F enter as evaluation requirements rather than as additional modules.
E. Problem formulation
Let I ∈ [0, 1]3×H×W be an RGB image with ground-truth boxes Y, and let Y (x) = w⊤I(x) be its luminance with w = (0.27, 0.67, 0.06)⊤, the weights used in [8]. An imageadaptive detector is a composition Dω ◦ Pθ(I) of a processing operator P , whose parameters θ(I) are predicted from the image, and a detector Dω. In ERUP-YOLO, P = KθK ◦ BθB and θ is constant over pixels. We seek P such that (a) in regions that are dark the processing can brighten as in ERUPYOLO, (b) in regions that are bright, for example because of scattering, brightening is suppressed, and (c) detection loss, not image appearance, decides the trade-off.
Assumptions. (A1) The documented sand and fog regression is caused, at least in part, by brightening of alreadybright regions, as [8] states; this is checked by experiment S1 before any method comparison. (A2) Local luminance, the dark channel [26], and the fraction of near-saturated pixels are informative proxies for “already bright” and “scatteringdominated”; they are not physical measurements of transmission. (A3) Training remains restricted to clear VOC images plus synthetic augmentation, as in [8], so that the comparison isolates the method.
Hypotheses. Let ∆d be the mean over paired seeds of mAP@0.5(SG-ERUP) − mAP@0.5(ERUP) on dataset d.

H1 (benefit) ∆DAWN-sand > 0, and ∆d ≥ 0 for DAWN-fog and RTTS.
H2 (non-inferiority) For d ∈ {VOC test, VOC-Dark, ExDark}, the lower bound of the one-sided 95% confidence interval of ∆d is above −δ, with δ = 0.5 mAP points fixed in advance.
H3 (mechanism) On DAWN-sand, DAWN-fog, and RTTS, SG-ERUP lowers the newly-saturated-pixel ratio (Eq. (13)) and raises recall for ground-truth boxes in the brightest tertile, relative to ERUP-YOLO.
H4 (spatial vs. global) A spatial gate outperforms a single per-image gate of equal training recipe on DAWN-sand and DAWN-fog.
Each hypothesis is rejected if its stated inequality fails after Holm correction across the confirmatory family (Section V-E).
IV. PROPOSED METHODOLOGY
Fig. 2 shows the proposed pipeline. Blue components are retained from ERUP-YOLO; orange components are proposed here and have not been implemented.

A. Retained ERUP components

BPW filter. For each channel c, ERUP-YOLO defines a

monotone mapping fc : [0, 1] → [0, 1] from a cubic Bézier

curve with end points (0, 0) and (1, 1) and two interior control points parameterized by (rc,j, ϑc,j) ∈ [−1, 1]2, j ∈ {1, 2},

such that all-zero parameters give the identity; the curve is

evaluated as a piecewise-linear function over L parameter

intervals [8, Eqs. 11–14]. We write

(︁

c )︁

BθB (I)c(x) = fc Ic(x); θB ,

θB ∈ [−1, 1]12. (1)

KBL

filter.

For

an

input

J

and

per-channel

kernels

kc
1

,

kc
2

∈

[−1, 1]9×9,

(︁

c

c

)︁

KθK (J )c = clip Jc ⊙ (k1 ∗ Jc) + k2 ∗ Jc + Jc, 0, 1 , (2)

where ∗ is 2-D convolution and ⊙ is element-wise multiplica-
tion. The clipping is our addition; the original formulation [8,
Eq. 15] does not state how out-of-range values are handled.
Encoder. Eϕ is the five-layer encoder of [8]. We denote its last convolutional feature map, before global average pooling, by F ∈ R1024×h×w and its pooled output, mapped by a fully connected layer and a sigmoid and then rescaled, by θ = (θB, θK ). ERUP-YOLO computes Iˆ = KθK (BθB (I)).

B. Cue maps

From the unprocessed input we compute three singlechannel maps that summarize local brightness and haze, with a box filter Ar of radius r:

µ(x) = (Ar ∗ Y )(x),

(3)

δ(x) = min min Ic(y),

(4)

y∈Ωr(x) c

(︁

)︁

ρ(x) = Ar ∗ 1[max Ic > τhi] (x).

(5)

c

Here Ωr(x) is the (2r+1)2 window around x, δ is the dark channel [26], which is high in hazy or uniformly bright regions, and ρ is the local fraction of near-saturated pixels


PREPRINT – RESEARCH PROPOSAL, NOT PEER REVIEWED

5

TABLE II CANDIDATE RESEARCH GAPS. “ACK.” = ACKNOWLEDGED BY THE AUTHORS OF [8]; “INF.” = INFERRED BY US.

Gap A. Oversaturation of bright regions (fog, sand)
B. Generalization to unseen conditions
C. Accuracy vs. cost
D. Small, distant, low-contrast objects
E. Robustness and failure detection F. Ablation and reproducibility

Evidence in [8]
Ack.: Sec. 5.4, 5.5, Supp. 7–8; sand 27.82 vs 31.18; KBL ≥ BPW+KBL on VOC-Foggy/RTTS
Inf.: training uses BPW augmentation that can mimic fog and low light (Fig. 5); test conditions are partly “covered” photometrically; no held-out-condition protocol Inf.: no params, FLOPs, latency, memory reported; analytic encoder estimate 6.78 M params Inf.: Supp. Fig. 9–10 show distant misses; no size-stratified metrics; no depth annotations on RTTS/DAWN Inf.: no severity sweep, calibration, or uncertainty analysis Inf.: single-run tables; no no-filter arm; augmentation not ablated; encoder input size, BPW interval count L, KBL constraint, NMS/AP settings unspecified; no code located

Status Established (phenomenon); mechanism tentative Supported
Supported
Tentative
Tentative Supported

What would close it Spatially controlled BPW; mechanism metrics (NSR, bright-tertile recall); paired multi-seed test (H1–H3)
Coverage-aware split (photometric vs. particulate conditions), cross-dataset tests on BDD100K/ACDC, single-DG benchmark [4]
Measured cost table (E7) with a fixed protocol
Size-stratified AP; distance-stratified AP only on Foggy Cityscapes / Seeing Through Fog where depth exists Severity sweeps; ECE/D-ECE [36]; TIDE [34] Full ablation, 8 paired seeds, released configurations

identity path, (1 − G) ⊙ I

λs Lsat(I, I˜)

Input I

BPW filter G⊙ ⊕ BθB (I)
θB

KBL filter

YOLOv3

I˜

KθK (I˜)

Iˆ detector Dω

θK

boxes

Encoder Eϕ

FC + sigmoid

(ERUP, 5 conv)

θB , θK

G

last feature map F

Cue maps C
luminance, dark channel, bright fraction

Gate head hψ
1×1 conv + sigmoid
g ∈ (0, 1)h×w

Upsample G = U(g)

Ldet λtv Ltv(g)

ERUP-YOLO component (prior work) proposed in this manuscript (not yet evaluated)

Fig. 2. Proposed SG-ERUP pipeline. The ERUP-YOLO encoder still predicts global BPW and KBL parameters. A gate head reads the encoder’s last feature map and three cue maps and outputs a low-resolution gate g, upsampled to G. The BPW output is blended with the unprocessed image by G before the KBL filter. With G ≡ 1 the pipeline is ERUP-YOLO. Dashed boxes are training losses.

with threshold τhi. The cues are average-pooled to the gate grid: C = Ph×w([µ; δ; ρ]) ∈ R3×h×w. The cues contain no learnable parameters.
C. Spatial gate The gate head is a 1×1 convolution followed by a sigmoid: g = σ(︁WF F + WC C + b)︁ ∈ (0, 1)h×w, (6)

with WF ∈ R1×1024, WC ∈ R1×3, and b ∈ R, i.e. 1028 parameters. The gate is upsampled to full resolution by bilinear interpolation, G = U (g) ∈ (0, 1)H×W , and applied to the
BPW output only:

I˜ = G ⊙ BθB (I) + (1 − G) ⊙ I,

(7)

Iˆ = KθK (I˜), prediction = Dω(Iˆ),

(8)

with G broadcast over channels.


PREPRINT – RESEARCH PROPOSAL, NOT PEER REVIEWED

6

Properties. (P1) Because I˜ is a convex combination of two images in [0, 1], it stays in [0, 1]. (P2) G ≡ 1 gives exactly ERUP-YOLO, and G ≡ 0 gives the KBL-only variant that [8] found strong on foggy data; the gate interpolates between the two configurations the original ablation compared. (P3) At a fixed pixel, I˜c(x) is a non-decreasing function of Ic(x) for a fixed value of G(x) ∈ [0, 1], because both blended terms are, so the per-pixel ordering of intensities is preserved; across pixels, different G values can alter relative intensities, which is intended, but sharp gate edges can create halos, hence the low-resolution gate and the smoothness term below. (P4) The bias is initialized to b0 so that G ≈ σ(b0) at the start; we treat b0 ∈ {0, 2} as a hyperparameter because an initialization very close to G ≡ 1 may leave the gate near ERUP-YOLO if detection gradients are weak.
Why only BPW is gated. The original authors attribute the regression to BPW’s global brightening, and their ablation shows KBL alone performing well on foggy data [8, Sec. 5.5]. Gating KBL as well is an ablation (E2-g), not the default, to keep the change minimal.

D. Saturation-aware penalty

(︁Let Y˜ = w⊤I˜)a︁ nd define a soft bright-region weight β(x) = σ (µ(x) − τb)/s . The penalty is

Lsat = 1 β(x) ReLU Y ∑︂ [︂ (︁ ˜ (x) − Y (x))︁2

HW x

(︁

)︁ [︁

]︁]︂

+ η ReLU Y˜ (x) − τhi 1 Y (x) ≤ τhi .

(9)

The first term penalizes brightening in regions that are already
bright and does not penalize brightening of dark regions or
darkening of bright ones. The second term penalizes pushing
pixels that were not near saturation above τhi. The indicator depends only on the input, so the loss is differentiable with
respect to θB and g. Unlike the exposure-control loss of ZeroDCE [13], which pulls every region towards one target level,
Eq. (9) is one-sided and conditional on the input brightness. The penalty is computed on I˜ (before KBL); computing it on Iˆ is ablation E2-h.

E. Gate smoothness

1 ∑︂ (︁

)︁

Ltv = hw

|gi+1,j − gi,j | + |gi,j+1 − gi,j | . (10)

i,j

G. Spatially heterogeneous augmentation (SHA)

ERUP-YOLO’s augmentation applies one random BPW curve to the whole image [8, Sec. 4.4]. Training images then contain no brightness heterogeneity created by the degradation, so a spatial gate has nothing spatial to learn from. With probability psha we instead apply

Iaug = M ⊙ Bθa (I) + (1 − M ) ⊙ Bθa′ (I), (12)

where

θ

a

,

θ′
a

are

drawn

as

in

ERUP-YOLO’s

augmentation

(one brightening, one darkening curve) and M ∈ [0, 1]H×W

is a smooth random mask, the sum of a vertical ramp (bright

towards the top, a horizon heuristic) and a low-frequency

random field, clipped to [0, 1]. Optionally the brightening

branch is replaced by the scattering model I t + A(1 − t) with t = exp(−κ d˜) [25], using a pseudo-depth d˜ that increases

towards the top of the image. This pseudo-depth is a heuristic,

not a depth estimate; it biases the augmentation towards road-

scene geometry, which we list as a threat to validity. SHA uses

no test-domain data.

H. Inference
At test time the pipeline is Eq. (8) with cue computation, the encoder forward pass, the gate head, one bilinear upsampling, one blend, BPW and KBL, and the detector. The added cost over ERUP-YOLO is the three cue maps (box filters and a min-filter), a 1 × 1 convolution on an h × w grid, and one H × W upsampling and blend; any claim about the size of this overhead will rest on the measurements of experiment E7.

I. Differences from prior work
Compared with ERUP-YOLO, SG-ERUP adds spatial control of the BPW curve, an input-conditioned one-sided brightness penalty, and spatially heterogeneous augmentation, while keeping the filters, the encoder, and the training recipe. Compared with GDIP-YOLO, whose gates weight filters per image, the gate here varies over the image. Compared with Zero-DCE and HDRNet, the spatial control is driven by the detection loss and acts on a pre-existing global curve with three cue channels, rather than predicting a full per-pixel curve or affine grid. Compared with feature-level gating [19], [20], SG-ERUP acts on pixels and keeps the processed image inspectable.
V. EXPERIMENTAL DESIGN
The methodology is summarized in Fig. 3; the full matrix is in Table III and in experiments.md.

F. Training objective
L = Ldet(︁Dω(Iˆ), Y)︁ + λs Lsat + λtv Ltv, (11)
where Ldet is the YOLOv3 loss used by [8]. All parameters (ϕ, ψ, ω) are trained jointly, with ψ = (WF , WC , b). λs = 0 tests whether detection loss alone teaches the gate (ablation E2-c). The penalty can trade dark-region enhancement for bright-region protection only through θB; the gate exists so that this trade-off is not forced.

A. Datasets and splits
Training: VOC 2007 trainval + VOC 2012 trainval [9], clear images only, with BPW augmentation (and SHA where stated), matching [8]. Validation for all tuning: a fixed 10% split of the training images with synthetic fog, synthetic gamma darkening, and SHA applied with fixed seeds; no image from any test set is used for tuning, early stopping, or checkpoint selection. Test (in-protocol): VOC 2007 test; VOC-Foggy and VOC-Dark test sets generated from it as in [6]; RTTS [10]; ExDark [11]; DAWN [12] fog, rain, snow, and sand subsets.


PREPRINT – RESEARCH PROPOSAL, NOT PEER REVIEWED

7

S0 Replicate YOLOv3 + ERUP-YOLO 8 seeds, Tables 1–3 protocol

no: report discrepancy, use own baseline

within 95% CI of reported?

yes

S1 Diagnose

saturation statistics,

TIDE errors on sand/fog

failure linked to brightening?
yes

no: revise hypothesis

S2 Components
gate, Lsat, SHA one at a time

S3 Confirmatory test H1–H4, paired seeds,
Holm correction

S4 Generalization cross-dataset, held-out
conditions, severity sweeps

S5 Cost & report params, FLOPs, latency,
memory; negative results

Fig. 3. Research workflow with decision gates. A failed replication (S0) or a failure to link the regression to brightening (S1) changes the plan before any method comparison is run.

Test (out-of-protocol, RQ4): BDD100K validation images tagged foggy, rainy, snowy, or night [30]; ACDC [31] with boxes derived from its instance-level annotations; the diverseweather single-domain-generalization benchmark [4]; Foggy Cityscapes [32] for distance-stratified analysis. Class sets follow [8]: five classes (person, bicycle, car, bus, motorbike) for fog and multi-weather, ten for low light; class-name mappings for each dataset are fixed in advance and released. The exact number of images per DAWN subset, and whether DAWN separates bicycle and motorcycle labels, must be checked against the release used, because DAWN’s description groups motorcycles and bicycles in its statistics [12].
Leakage controls. (i) VOC-Foggy and VOC-Dark are derived from VOC 2007 test images, which are excluded from training and validation. (ii) DAWN, RTTS, ExDark, BDD100K, and ACDC are used only for final testing. (iii) Hyperparameters (λs, λtv, τb, s, τhi, η, r, b0, psha) are fixed on the validation split and frozen before any test run; we report the full search grid. (iv) The SHA pseudo-depth heuristic is designed without looking at test images; we document its design date relative to the first test evaluation.
B. Baselines and variants
B0: YOLOv3 with BPW augmentation, no filter. B1: YOLOv3 with the earlier condition-specific augmentation. B2: ERUP-YOLO reproduced (BPW+KBL). B3: ERUP KBL-only. B4: ERUP with a single learned per-image gate. B5: ERUP with a fixed, hand-designed gate G = σ(−(µ − τb)/s) (no learning). Proposed variants add, one at a time and then jointly, the learned spatial gate, Lsat, and SHA. IA-YOLO [6] and GDIP-YOLO [7] are retrained under the same protocol if their code runs with our detector implementation; otherwise published numbers are shown separately and marked as not comparable. Feature-level low-light methods [18], [19] are included only on ExDark and only if they can be run on the same YOLOv3 detector.

becomes available), 80 epochs, cosine learning rate 10−4 → 10−6, weight decay 5 × 10−4, input 448 × 448, batch size 6; YOLOv3 initialization from the same pretrained weights for every arm; encoder input 256 × 256 (assumption, by analogy with [6]; the value used by [8] is unknown); BPW intervals L fixed and reported. The software stack, GPU model, driver, and library versions are recorded per run. Every configuration file and random seed is released.

D. Metrics

Accuracy: mAP@0.5 (VOC all-point interpolation) for con-

tinuity with [8], plus COCO-style mAP@[0.5:0.95] [40],

precision and recall at a fixed confidence threshold, and

per-class AP. Size: AP for small, medium, and large boxes using the COCO area thresholds (322, 962 pixels at original

resolution [40]) and, because resolutions differ across datasets,

also relative-area tertiles. Brightness stratification: ground-

truth boxes are split into tertiles of mean input luminance

inside the box, and recall is reported per tertile. Distance: only

on Foggy Cityscapes, whose fog is rendered from depth [32],

using the median depth inside each box; we make no distance

claims for RTTS or DAWN, which lack depth. Errors: TIDE

decomposition [34]. Calibration: detection expected calibra-

tion error [36]. Mechanism:

NSR = 1 1 Y ∑︂ [︁ ˜ (x) > τhi ∧ Y (x) ≤ τhi]︁, (13)

HW ∑︁ x (︁ )︁

x β(x) ReLU Y˜ (x) − Y (x)

BRB =

∑︁

,

(14)

x β(x)

the newly-saturated ratio and the mean bright-region brighten-

ing. Cost: parameters; FLOPs at 448 × 448 with the counting

tool and convention stated; latency at batch size 1 on a

stated GPU (50 warm-up and 500 timed iterations with device

synchronization, median and 90th percentile, FP32 and FP16),

FPS as the inverse of the median; peak GPU memory during

inference and training.

C. Training configuration
We follow [8] exactly where it is specified and fix the remainder before experiments: SGD with momentum 0.9 (to be confirmed against a public reference implementation if one

E. Statistical analysis
Each configuration is trained with n = 8 seeds, matching the number used in [8, Fig. 7]; seed k fixes data order, augmentation draws, and initialization of new parameters for every


PREPRINT – RESEARCH PROPOSAL, NOT PEER REVIEWED

8

arm, which pairs runs across arms (pairing is approximate when nondeterministic GPU kernels are used, and we report whether deterministic mode was enabled). For each dataset we report mean, standard deviation, and a 95% t-interval over seeds, and for each comparison the paired difference with its 95% interval and a two-sided Wilcoxon signed-rank test. With n = 8 the smallest attainable two-sided Wilcoxon pvalue is 2/28 ≈ 0.0078, so the test can reach significance after Holm correction across up to six comparisons. Test-set sampling uncertainty is estimated by an image-level bootstrap (1,000 resamples) of each model and reported alongside seed variance; small subsets such as DAWN-sand will have wide intervals, and we report them rather than pooling. The confirmatory family is H1 (three datasets) and H2 (three datasets); everything else is exploratory and labelled as such. Effect sizes are reported in mAP points, not only p-values [38].
F. Ablations
E2-a gate only; E2-b Lsat only (global BPW); E2-c gate + SHA, λs = 0; E2-d gate + Lsat, no SHA; E2-e full model; E2-f cue maps removed (WC = 0) or features removed (WF = 0); E2-g gate on KBL as well; E2-h Lsat on Iˆ; E2-i gate resolution h×w ∈ {1×1, 4×4, 8×8, 16×16} (the 1×1 case is B4); E2-j bilinear vs. guided-filter upsampling [41]; E2-k b0 ∈ {0, 2}.
VI. RESULTS AND DISCUSSION
No experiments of the proposed method have been run. This section contains result templates only. Cells marked “– ” are intentionally empty and must be filled from completed runs; no value in them is estimated or anticipated.
Planned analyses. (i) If E1 does not reproduce the published ERUP-YOLO numbers within the published eightrun range, all comparisons use our reproduction, and the discrepancy is reported. (ii) If S1 shows that missed detections on DAWN-sand are not concentrated in bright regions, or that NSR under ERUP-YOLO is not higher than under B0, assumption A1 fails, the gate is not the right remedy, and we report this as a negative finding before running E2–E5. (iii) H1–H2 are decided by Table IV; H3 by Table V; H4 by B4 vs. SG-ERUP. (iv) Outcomes that would count against the method include: gains only on DAWN-sand with losses on ExDark beyond δ; gains that vanish when B4 (global gate) is given the same SHA augmentation, which would show that SHA, not spatial control, drives the effect; and gains on inprotocol data that do not appear on BDD100K or ACDC. All such outcomes will be reported.
VII. LIMITATIONS AND THREATS TO VALIDITY
Status. The method is unimplemented; its benefit is a hypothesis. The cost figures in Section III-C are analytic estimates based on a textual description with stated assumptions.
Internal validity. SHA changes the training distribution and could improve accuracy for reasons unrelated to gating; arm B4+SHA and ablation E2-c control for this. Extra parameters could help independent of mechanism; the gate adds about 1.0 k parameters to a front end of 6.78 M (estimated), so a capacity explanation is unlikely but is tested by the fixed-gate

arm B5 and the mechanism metrics. Hyperparameters of the penalty are new degrees of freedom; tuning them only on the validation split limits optimistic bias.
Construct validity. The cue maps are proxies. The dark channel is high for any bright, low-saturation region, including white vehicles and overcast sky, not only haze [26]; the gate may suppress useful enhancement there. Brightness tertiles are not distance; we avoid distance claims except where depth exists.
External validity. All training data are VOC images with synthetic degradations; the DAWN sand subset is small, collected from the web, and has no official split, so test-set uncertainty will be large. Results on YOLOv3 may not transfer to modern detectors (E12). The vertical-ramp heuristic in SHA reflects road scenes and may not suit aerial or indoor imagery.
Statistical conclusion validity. Eight seeds give limited power for effects below the seed spread; we report intervals and treat non-significant results as inconclusive rather than as evidence of no effect.
Reproducibility of the reference method. We did not locate a public implementation of ERUP-YOLO when this manuscript was written. Several details (encoder input size, BPW interval count, KBL constraint mechanism, NMS and AP settings, which run is tabulated) are unspecified, so our reproduction may differ from the original.
VIII. CONCLUSION
ERUP-YOLO simplifies image-adaptive detection to two differentiable filters, but its global brightness curve, by the authors’ own account, can oversaturate scattering-bright regions, and most of its published gains over a matchedaugmentation baseline are comparable to the seed spread the same paper reports. We have specified a minimal extension, a spatial gate with about one thousand parameters, a one-sided saturation penalty, and a spatially heterogeneous augmentation, that contains ERUP-YOLO as a special case, together with hypotheses, controls, and statistics that can confirm or refute it. Whether the extension works is unknown until the experiments in Table III are run; we commit to reporting the outcome either way.
IX. FUTURE RESEARCH DIRECTIONS
In priority order:
1) Replication and diagnosis (E1, S1). Reimplement ERUP-YOLO, train 8 seeds, and measure NSR, BRB, and brightness-stratified recall on DAWN-sand and RTTS. This decides whether the proposed direction is pursued.
2) Confirmatory test (E2, E3, E8). Run the ablation and test H1–H4 with the pre-specified analysis.
3) Generalization (E4, E11). Freeze models and evaluate on BDD100K, ACDC, and the single-domaingeneralization benchmark [4]; separate conditions that BPW augmentation can imitate from those it cannot (rain streaks, snow particles).
4) Cost (E7). Measure the front end on a desktop GPU and an embedded device; if the encoder dominates, test a


PREPRINT – RESEARCH PROPOSAL, NOT PEER REVIEWED

9

TABLE III EXPERIMENTAL MATRIX. MANDATORY EXPERIMENTS (M) MUST BE COMPLETE BEFORE JOURNAL SUBMISSION; OPTIONAL (O) EXPERIMENTS
STRENGTHEN BUT DO NOT GATE SUBMISSION.

ID Pri. Objective
E1 M Replicate YOLOv3 + ERUP

Controlled variables Datasets Recipe of [8], 8 seeds All in-protocol

Baselines
Published values

Metrics mAP@0.5

E2 M Component ablation Everything except

All in-protocol

B0–B5

mAP@0.5,

listed component

mAP@[.5:.95]

E3 M Fog/sand

Same models as E2 DAWN sand, fog;

B2, B3, B4 mAP, NSR, BRB,

oversaturation

RTTS

bright-tertile recall,

TIDE

E4 M Cross-dataset

Frozen models, no BDD100K weather B0, B2

mAP@0.5,

generalization

retuning

subsets, ACDC,

mAP@[.5:.95]

single-DG benchmark

E5 M Size and per-class

Frozen models

All test sets

B0, B2

APS /APM /APL,

analysis

per-class AP

E6 O Severity robustness

Synthetic fog density, VOC test (synthetic) B0, B2

mAP vs. severity

gamma, noise levels

curves

E7 M Cost

Same hardware, input –

B0, B2

Params, FLOPs,

size

latency, FPS,

memory

E8 M Statistical reliability 8 paired seeds

All

All

CIs, Wilcoxon,

Holm

E9 O Distance-stratified

Frozen models

Foggy Cityscapes

B0, B2

AP per depth bin

analysis

E10 O Calibration and failure Frozen models

All test sets

B0, B2

D-ECE, reliability

detection

diagrams

E11 O Held-out photometric BPW augmentation ExDark, RTTS

B2

mAP

family

restricted to

brightening or

darkening only

E12 O Other detectors

Same front end on a Subset

Detector mAP, latency

modern detector

alone

Interpretation criterion Reproduced mean within the published Fig. 7 range; otherwise report gap and continue with own baseline Contribution = paired ∆ with 95% CI H1, H3, H4
Sign and CI of ∆ on data never used for decisions
Whether gains concentrate in a size or class Area under curve; crossover points
Overhead reported; no efficiency claim without measurement
Confirmatory family only
Only where depth exists
Descriptive
Separates covered vs. uncovered conditions
Whether findings transfer beyond YOLOv3

TABLE IV TEMPLATE: MAP@0.5, MEAN ± S.D. OVER 8 SEEDS. TO BE FILLED BY
E1–E3.

Method

VOC V-Dark ExDark RTTS D-fog D-sand D-snow

B0 YOLOv3+BPW aug. –

–

–

–

–

–

–

B2 ERUP (repro.)

–

–

–

–

–

–

–

B3 ERUP KBL-only

–

–

–

–

–

–

–

B4 global gate

–

–

–

–

–

–

–

SG-ERUP (full)

–

–

–

–

–

–

–

∆ vs. B2 [CI]

–

–

–

–

–

–

–

TABLE V TEMPLATE: MECHANISM METRICS ON DAWN-SAND (E3). TO BE FILLED.

Method

NSR BRB Recall, bright tertile Recall, dark tertile

B2 ERUP –

–

–

–

SG-ERUP –

–

–

–

5) Depth-aware evaluation (E9). Use Foggy Cityscapes or Seeing Through Fog [33] to test whether the gate helps distant objects specifically.
6) Confidence-aware processing (E10). Examine whether gate statistics or BPW parameters predict detection failures, and whether calibration [35], [36] degrades under processing.
7) Transfer to modern detectors (E12). Apply the front end to a current one-stage detector to test whether the findings hold beyond YOLOv3.
ACKNOWLEDGMENT
[Funding and acknowledgments.] The ERUP-YOLO method, its equations restated in Section IV, and all numbers quoted in Section III are from [8]; no figure from that paper is reproduced.

TABLE VI TEMPLATE: COST AT 448 × 448, BATCH 1 (E7). HARDWARE: [GPU]. TO
BE FILLED FROM MEASUREMENTS.

Method

Params GFLOPs Latency (ms) FPS Mem. (MB)

B0 YOLOv3

–

–

B2 ERUP

–

–

SG-ERUP

–

–

–

–

–

–

–

–

–

–

–

reduced encoder whose capacity matches the 498 global parameters it predicts.

