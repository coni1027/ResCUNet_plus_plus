"""
U-Net (Ronneberger et al., 2015) -- the standard baseline for medical
image segmentation. Symmetric encoder-decoder with plain double-conv
blocks, max-pool downsampling, and skip connections concatenated at each
matching resolution. No residual connections, no attention, no nested
skip pathways, single output (no deep supervision). Classic 64-1024
channel-doubling scheme.
"""

from __future__ import annotations

import torch
from torch import nn

from config import DatasetConfig, DEVICE
from models.common import DoubleConv, upsample_to


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.bottleneck = DoubleConv(c4, c5)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)

        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([upsample_to(self.up4(b), e4), e4], dim=1))
        d3 = self.dec3(torch.cat([upsample_to(self.up3(d4), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([upsample_to(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([upsample_to(self.up1(d2), e1), e1], dim=1))

        out = self.out_conv(d1)
        return upsample_to(out, x)


def build_model(config: DatasetConfig) -> UNet:
    return UNet(in_channels=config.in_channels).to(DEVICE)
