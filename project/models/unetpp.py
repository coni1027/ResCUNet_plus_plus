"""
U-Net++ (Zhou et al., 2018) -- nested, densely skip-connected
encoder-decoder. Plain double-conv encoder/decoder blocks (no residual
connections, no CBAM) -- a "vanilla" baseline that isolates what the
ResNet encoder and CBAM specifically add in the proposed model, since the
nested-skip-pathway topology itself is otherwise identical. Filter widths
follow the [32, 64, 128, 256, 512] scheme commonly used in the original
paper and public reference implementations.

Supports the same deep_supervision flag as the proposed model: with it
on, returns all four decoder-column outputs (X0,1..X0,4) for a hybrid
loss averaged over intermediate outputs (Methodology 3.7.3); with it off,
returns only the final X0,4 output.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from config import DatasetConfig, DEVICE
from models.common import DoubleConv, upsample_to


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 1, deep_supervision: bool = False) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        f0, f1, f2, f3, f4 = 32, 64, 128, 256, 512
        self.pool = nn.MaxPool2d(2)

        self.conv0_0 = DoubleConv(in_channels, f0)
        self.conv1_0 = DoubleConv(f0, f1)
        self.conv2_0 = DoubleConv(f1, f2)
        self.conv3_0 = DoubleConv(f2, f3)
        self.conv4_0 = DoubleConv(f3, f4)

        self.conv0_1 = DoubleConv(f0 + f1, f0)
        self.conv1_1 = DoubleConv(f1 + f2, f1)
        self.conv2_1 = DoubleConv(f2 + f3, f2)
        self.conv3_1 = DoubleConv(f3 + f4, f3)

        self.conv0_2 = DoubleConv(f0 * 2 + f1, f0)
        self.conv1_2 = DoubleConv(f1 * 2 + f2, f1)
        self.conv2_2 = DoubleConv(f2 * 2 + f3, f2)

        self.conv0_3 = DoubleConv(f0 * 3 + f1, f0)
        self.conv1_3 = DoubleConv(f1 * 3 + f2, f1)

        self.conv0_4 = DoubleConv(f0 * 4 + f1, f0)

        self.out1 = nn.Conv2d(f0, 1, kernel_size=1)
        self.out2 = nn.Conv2d(f0, 1, kernel_size=1)
        self.out3 = nn.Conv2d(f0, 1, kernel_size=1)
        self.out4 = nn.Conv2d(f0, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        input_size = x.shape[-2:]

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, upsample_to(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, upsample_to(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, upsample_to(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, upsample_to(x4_0, x3_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, upsample_to(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, upsample_to(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, upsample_to(x3_1, x2_0)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, upsample_to(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, upsample_to(x2_2, x1_0)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, upsample_to(x1_3, x0_0)], dim=1))

        outputs = [self.out1(x0_1), self.out2(x0_2), self.out3(x0_3), self.out4(x0_4)]
        outputs = [
            F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
            for out in outputs
        ]

        return outputs if self.deep_supervision else outputs[-1]


def build_model(config: DatasetConfig) -> UNetPlusPlus:
    return UNetPlusPlus(
        in_channels=config.in_channels,
        deep_supervision=config.deep_supervision,
    ).to(DEVICE)
