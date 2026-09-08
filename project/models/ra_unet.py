"""
RA-UNet (Jin et al., 2020, "RA-UNet: A hybrid deep attention-aware
network to extract liver and tumor in CT scans") -- a U-Net-shaped
encoder-decoder built from residual blocks throughout (except the first
and last layer, per the original paper), with Attention Residual Modules
(models.common.AttentionResidualModule -- Wang et al., 2017's trunk/
soft-mask branch attention) inserted at selected depths so
attention-aware features are learned adaptively as the network goes
deeper.

The original paper is a 3D architecture for CT volumes; this is a 2D
adaptation for single-slice mammograms/MRI, since this project only
trains on 2D slices. Attention Residual Modules are placed at the two
middle encoder/decoder stages (where feature maps are small enough that
the modules' internal pooling is cheap, but the network is still deep
enough to benefit); the shallowest (highest-resolution) and deepest
(bottleneck) stages use plain residual blocks, matching the paper's
"except the first and last layer" description. This placement is a
reasonable, documented choice for the 2D adaptation -- cross-reference
the original paper's Fig. 2(d) if exact stage-by-stage placement matters
for your comparison.
"""

from __future__ import annotations

import torch
from torch import nn

from config import DatasetConfig, DEVICE
from models.common import AttentionResidualModule, ResidualBlock, upsample_to


class RAUNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        f0 = base_channels          # 32
        f1 = base_channels * 2       # 64
        f2 = base_channels * 4       # 128
        f3 = base_channels * 8       # 256
        f4 = base_channels * 16      # 512

        # "First layer": plain conv, not a residual block.
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels, f0, kernel_size=3, padding=1),
            nn.BatchNorm2d(f0),
            nn.ReLU(inplace=True),
        )

        self.enc1 = ResidualBlock(f0, f1, stride=2)                 # H/2,  f1  (plain)
        self.enc2_down = ResidualBlock(f1, f2, stride=2)             # H/4,  f2
        self.enc2_attn = AttentionResidualModule(f2)                  # H/4,  f2  (attention)
        self.enc3_down = ResidualBlock(f2, f3, stride=2)               # H/8,  f3
        self.enc3_attn = AttentionResidualModule(f3)                    # H/8,  f3  (attention)
        self.bridge = ResidualBlock(f3, f4, stride=2)                     # H/16, f4  (plain bottleneck)

        self.up4 = nn.ConvTranspose2d(f4, f3, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(f3 * 2, f3)
        self.dec3_attn = AttentionResidualModule(f3)                        # H/8,  f3  (attention)

        self.up3 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(f2 * 2, f2)
        self.dec2_attn = AttentionResidualModule(f2)                          # H/4,  f2  (attention)

        self.up2 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(f1 * 2, f1)                                   # H/2,  f1  (plain)

        self.up1 = nn.ConvTranspose2d(f1, f0, kernel_size=2, stride=2)
        self.dec0 = ResidualBlock(f0 * 2, f0)                                     # H,    f0  (plain)

        # "Last layer": plain 1x1 conv, not a residual block.
        self.out_conv = nn.Conv2d(f0, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.stem_conv(x)                                       # H,    f0
        e1 = self.enc1(s)                                             # H/2,  f1
        e2 = self.enc2_attn(self.enc2_down(e1))                         # H/4,  f2
        e3 = self.enc3_attn(self.enc3_down(e2))                           # H/8,  f3
        b = self.bridge(e3)                                                 # H/16, f4

        d3 = upsample_to(self.up4(b), e3)
        d3 = self.dec3_attn(self.dec3(torch.cat([d3, e3], dim=1)))              # H/8,  f3

        d2 = upsample_to(self.up3(d3), e2)
        d2 = self.dec2_attn(self.dec2(torch.cat([d2, e2], dim=1)))                 # H/4,  f2

        d1 = upsample_to(self.up2(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))                                    # H/2,  f1

        d0 = upsample_to(self.up1(d1), s)
        d0 = self.dec0(torch.cat([d0, s], dim=1))                                        # H,    f0

        out = self.out_conv(d0)
        return upsample_to(out, x)


def build_model(config: DatasetConfig) -> RAUNet:
    return RAUNet(in_channels=config.in_channels).to(DEVICE)
