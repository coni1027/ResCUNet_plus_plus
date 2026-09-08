"""
ResUNet (Zhang et al., 2018, "Road Extraction by Deep Residual U-Net")
-- a U-Net-shaped encoder-decoder where every double-conv block is
replaced with a pre-activation residual block (see models/common.py's
ResidualBlock), and downsampling happens via strided convolution inside
each residual block rather than a separate pooling layer.

The original paper used a shallower, road-extraction-specific network;
this adapts the same core idea ("residual units instead of plain conv
blocks in a U-Net topology") to a standard 5-level depth matching this
project's other baselines and input resolutions (512x512 / 256x256), so
it is a faithful-in-spirit reimplementation rather than a literal port of
the original paper's exact layer count.
"""

from __future__ import annotations

import torch
from torch import nn

from config import DatasetConfig, DEVICE
from models.common import ResidualBlock, upsample_to


class ResUNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.enc1 = ResidualBlock(in_channels, c1)
        self.enc2 = ResidualBlock(c1, c2, stride=2)
        self.enc3 = ResidualBlock(c2, c3, stride=2)
        self.enc4 = ResidualBlock(c3, c4, stride=2)
        self.bridge = ResidualBlock(c4, c5, stride=2)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(c1 * 2, c1)

        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bridge(e4)

        d4 = self.dec4(torch.cat([upsample_to(self.up4(b), e4), e4], dim=1))
        d3 = self.dec3(torch.cat([upsample_to(self.up3(d4), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([upsample_to(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([upsample_to(self.up1(d2), e1), e1], dim=1))

        out = self.out_conv(d1)
        return upsample_to(out, x)


def build_model(config: DatasetConfig) -> ResUNet:
    return ResUNet(in_channels=config.in_channels).to(DEVICE)
