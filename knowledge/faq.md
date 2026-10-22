# FAQ: SG-ERUP concepts (project documentation for Ask SG-ERUP AI)

## What is SG-ERUP?
SG-ERUP (saturation-gated ERUP) is a proposed extension of ERUP-YOLO. ERUP-YOLO keeps two differentiable filters in front of YOLOv3: a global Bézier pixel-wise (BPW) tone curve and a kernel-based local (KBL) filter, and an encoder predicts their parameters. SG-ERUP adds three things:
- a low-resolution spatial gate G that blends the BPW output with the original image (I~ = G*BPW(I) + (1-G)*I);
- a saturation-aware penalty L_sat;
- spatially heterogeneous augmentation (SHA).

With G = 1 everywhere it reduces to ERUP-YOLO, and with G = 0 it reduces to KBL-only. It is a hypothesis: the paper does not evaluate it.

## What is the BPW filter?
BPW is a per-channel monotone cubic Bézier tone curve with end points (0,0) and (1,1) and two interior control points. Its 12 parameters lie in [-1, 1], and all zeros give the identity. The curve is evaluated as a piecewise-linear function over L intervals. Because it is global, it brightens bright hazy regions as well as dark foreground regions.

## What is the KBL filter?
KBL is a locally linear filter: K(J)_c = clip(J_c * (k1_c conv J_c) + k2_c conv J_c + J_c, 0, 1), with per-image 9x9 kernels k1 and k2 per channel (486 values). It generalises sharpening and defogging-style local operations.

## What are the cue maps?
There are three parameter-free maps computed from the input:
- local luminance mu (box filter of Y = 0.27R + 0.67G + 0.06B);
- the dark channel delta (min over a window of the per-pixel channel minimum), which is high in hazy or uniformly bright regions;
- rho, the local fraction of near-saturated pixels (max channel above tau_hi).

They are average-pooled to the gate grid and fed to the gate head along with the encoder's last feature map.

## What is the spatial gate?
The gate is a 1x1 convolution plus sigmoid over [encoder feature map F (1024 channels); cues C (3 channels)], which is 1028 parameters. It produces g on an h x w grid (8x8 here), which is bilinearly upsampled to G. The bias is initialised to b0 (2 by default, so G ≈ 0.88 at the start). L_tv keeps g smooth to avoid halos.

## What does the saturation-aware loss do?
L_sat has two terms:
- beta(x) * ReLU(Y~ - Y)^2 penalises brightening where the input is already bright, with beta = sigmoid((mu - tau_b)/s);
- eta * ReLU(Y~ - tau_hi) * 1[Y <= tau_hi] penalises pushing pixels that were not saturated above tau_hi.

It is one-sided: brightening dark regions and darkening bright ones are never penalised.

## What is SHA and why is it needed?
ERUP's augmentation applies one random BPW curve to the whole image, so training images contain no spatial brightness heterogeneity and a spatial gate would get no training signal. SHA mixes a brightening curve and a darkening curve with a smooth mask M (a vertical ramp, bright at the top, plus a low-frequency random field).

## What are NSR and BRB?
Both are mechanism metrics, computed without depth annotations:
- NSR (newly saturated ratio) is the fraction of pixels whose luminance crosses tau_hi because of processing.
- BRB (bright-region brightening) is the beta-weighted mean brightening.

In the paper both are computed on I~ (before KBL). This project also reports them on the final image I^ (nsr_final, brb_final). Lower values mean less brightening of already-bright regions. They are not accuracy metrics.

## What is brightness-stratified recall?
Ground-truth boxes are split into tertiles (dark, medium, bright) by the mean input luminance inside the box. Recall at a fixed score threshold (0.25) and IoU 0.5 is then reported per tertile. H3 predicts that SG-ERUP raises bright-tertile recall on DAWN-sand, DAWN-fog and RTTS.

## How are mAP, precision and recall computed here?
- mAP@0.5 uses PASCAL VOC all-point interpolation per class, averaged over classes with ground truth.
- mAP@[.5:.95] is a COCO-style 101-point AP averaged over IoU 0.5:0.05:0.95.
- Precision and recall are pooled over classes at score >= 0.25 and IoU 0.5.

Detections that overlap ignore regions (truck, train and other non-protocol classes) are not counted as false positives. The evaluator is this project's own implementation, not pycocotools.

## Which detector is used?
Ultralytics YOLOv3u, COCO-pretrained (103.8 M parameters): a YOLOv3 Darknet-53 backbone with an anchor-free head and the v8 loss. This differs from the paper's original YOLOv3. In the reported short runs the detector is frozen and only the front end is trained, because a 4 GB GPU cannot train it jointly.

## Why is the DAWN sand subset important?
ERUP-YOLO reports 27.82 mAP@0.5 on DAWN-sand against 31.18 for a YOLOv3 trained with the same BPW augmentation, a 3.36-point regression. Its authors attribute this to BPW brightening scattering-bright regions. That is the failure SG-ERUP targets (gap A, hypothesis H1).

## How do I start the app?
1. Activate the venv.
2. Start the backend: `python -m uvicorn backend.app.main:app --port 8000`.
3. In `frontend/`, start the frontend: `npm run dev`.
4. Open http://localhost:5173.

See README.md for the full commands.

## How do I enable the LLM for Ask AI?
Set ANTHROPIC_API_KEY in a `.env` file at the project root (copy .env.example) and restart the backend. The key stays on the server. Without it, Ask AI runs in local fallback mode, which is a ranked excerpt search over the paper, docs and result files.

## Troubleshooting
- "Detector weights not found": run `python scripts/setup_weights.py`.
- "Dataset root not found": download the dataset and set its path in the Dataset Explorer or in configs/datasets.local.yaml.
- GPU out of memory: stop the training job or set SGERUP_DEVICE=cpu.
- Slow first request: the detector loads in the background at startup; /api/ready reports when it is ready.
