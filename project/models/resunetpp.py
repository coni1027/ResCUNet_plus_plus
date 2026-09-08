"""
ResUNet++ (Jha et al., 2019, "ResUNet++: An Advanced Architecture for
Medical Image Segmentation") -- extends ResUNet with:
  - a stem block (conv-bn-relu-conv with a parallel 1x1-conv shortcut),
  - a Squeeze-and-Excitation block after each encoder residual block,
  - an ASPP block as the encoder-decoder bridge (replacing a plain
    residual block, to enlarge the receptive field without downsampling
    further),
  - an attention gate before each decoder residual block, gating the
    encoder skip connection using the (upsampled) decoder feature from
    below, and
  - a second ASPP block right before the final 1x1 output convolution.

Filter widths follow the [32, 64, 128, 256, 512] scheme reported in the
original paper (stem + 3 encoder blocks + ASPP bridge), with the decoder
mirroring back down to 32 before the output ASPP. This is a faithful
reimplementation of the architecture's components and data flow as
described in the paper; exact channel counts/kernel choices are a
reasonable reading of the published description rather than a verified
line-for-line port of the authors' original code.
"""

from __future__ import annotations

import torch
from torch import nn

from config import DatasetConfig, DEVICE
from models.common import ASPP, AttentionGate, ResidualBlock, SqueezeExcite, upsample_to


class ResUNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        f0, f1, f2, f3, f4 = 32, 64, 128, 256, 512  # stem, enc1, enc2, enc3, bridge

        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels, f0, kernel_size=3, padding=1),
            nn.BatchNorm2d(f0),
            nn.ReLU(inplace=True),
            nn.Conv2d(f0, f0, kernel_size=3, padding=1),
        )
        self.stem_shortcut = nn.Conv2d(in_channels, f0, kernel_size=1)

        self.se1 = SqueezeExcite(f0)
        self.enc1 = ResidualBlock(f0, f1, stride=2)
        self.se2 = SqueezeExcite(f1)
        self.enc2 = ResidualBlock(f1, f2, stride=2)
        self.se3 = SqueezeExcite(f2)
        self.enc3 = ResidualBlock(f2, f3, stride=2)

        self.bridge = ASPP(f3, f4)

        self.attn3 = AttentionGate(gate_channels=f4, skip_channels=f3, inter_channels=f3)
        self.dec3 = ResidualBlock(f4 + f3, f3)

        self.attn2 = AttentionGate(gate_channels=f3, skip_channels=f2, inter_channels=f2)
        self.dec2 = ResidualBlock(f3 + f2, f2)

        self.attn1 = AttentionGate(gate_channels=f2, skip_channels=f1, inter_channels=f1)
        self.dec1 = ResidualBlock(f2 + f1, f1)

        self.attn0 = AttentionGate(gate_channels=f1, skip_channels=f0, inter_channels=f0)
        self.output_aspp = ASPP(f1 + f0, f0)
        self.out_conv = nn.Conv2d(f0, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.stem_conv(x) + self.stem_shortcut(x)   # H,    f0
        e1 = self.enc1(self.se1(s))                       # H/2,  f1
        e2 = self.enc2(self.se2(e1))                       # H/4,  f2
        e3 = self.enc3(self.se3(e2))                        # H/8,  f3

        b = self.bridge(e3)                                  # H/8,  f4

        a3 = self.attn3(gate=b, skip=e3)
        d = self.dec3(torch.cat([b, a3], dim=1))               # H/8,  f3
        d = upsample_to(d, e2)                                   # H/4

        a2 = self.attn2(gate=d, skip=e2)
        d = self.dec2(torch.cat([d, a2], dim=1))                  # H/4,  f2
        d = upsample_to(d, e1)                                      # H/2

        a1 = self.attn1(gate=d, skip=e1)
        d = self.dec1(torch.cat([d, a1], dim=1))                     # H/2,  f1
        d = upsample_to(d, s)                                          # H

        a0 = self.attn0(gate=d, skip=s)
        d = self.output_aspp(torch.cat([d, a0], dim=1))                  # H,    f0
        out = self.out_conv(d)
        return upsample_to(out, x)


def build_model(config: DatasetConfig) -> ResUNetPlusPlus:
    return ResUNetPlusPlus(in_channels=config.in_channels).to(DEVICE)
