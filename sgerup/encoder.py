"""ERUP-YOLO parameter encoder E_phi (paper Sec. III-C and IV-A).

Five 3x3 convolutions with 64-1024 channels, each followed by stride-2 pooling,
global average pooling, then one fully connected layer to 12 BPW + 486 KBL = 498
outputs, mapped by a sigmoid and rescaled to [-1, 1]. With same-padding and biases
this has 6,781,042 parameters, matching the paper's analytic 6.78 M.

Documented assumptions (not specified by the paper or ERUP-YOLO):
* activation LeakyReLU(0.1) (as in IA-YOLO's CNN-PP), max pooling;
* encoder input 256x256 (the paper's own assumption, Sec. V-C);
* FC initialised near zero so the initial processing is close to identity
  (sigmoid(0) rescaled = 0 -> identity BPW curve and zero KBL kernels). This lets a
  pretrained detector start from unmodified images.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BPW = 12
KBL_SIZE = 9
N_KBL = 2 * KBL_SIZE * KBL_SIZE * 3  # 486


class ERUPEncoder(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512, 1024), input_size: int = 256, init_std: float = 1e-3):
        super().__init__()
        layers, cin = [], 3
        for cout in channels:
            layers += [nn.Conv2d(cin, cout, 3, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2)]
            cin = cout
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, N_BPW + N_KBL)
        nn.init.normal_(self.fc.weight, std=init_std)
        nn.init.zeros_(self.fc.bias)
        self.input_size = input_size
        self.out_channels = cin

    def forward(self, img: torch.Tensor) -> dict[str, torch.Tensor]:
        x = F.interpolate(img, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        feat = self.features(x)  # F in R^{1024 x h x w}
        theta = torch.sigmoid(self.fc(feat.mean(dim=(2, 3)))) * 2 - 1
        b = img.shape[0]
        return {
            "feature": feat,
            "theta_b": theta[:, :N_BPW],
            "theta_k": theta[:, N_BPW:].reshape(b, 2, 3, KBL_SIZE, KBL_SIZE),
        }
