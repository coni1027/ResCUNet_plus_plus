"""
Building blocks shared by more than one model in this package. Keeping
these in one place means every model that uses e.g. CBAM or a residual
block gets the exact same implementation, instead of six slightly
different copy-pasted versions.

Contents:
  - DoubleConv, upsample_to      : plain U-Net-style building blocks
  - ChannelAttention, SpatialAttention, CBAM
        (Woo et al., 2018)       : used by resunetpp_cbam.py and cbam_unet.py
  - ResidualBlock                : pre-activation residual unit, used by
                                    resunet.py, resunetpp.py, ra_unet.py
  - SqueezeExcite                : (Hu et al., 2018) used by resunetpp.py
  - ASPP                         : Atrous Spatial Pyramid Pooling
                                    (Chen et al., 2017) used by resunetpp.py
  - AttentionGate                : additive attention gate (Oktay et al.,
                                    2018 style), used by resunetpp.py's
                                    decoder skip connections
  - AttentionResidualModule      : trunk/soft-mask attention residual unit
                                    (Wang et al., 2017), used by ra_unet.py
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def upsample_to(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        x,
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# -----------------------------------------------------------------------------
# CBAM (Woo et al., 2018)
# -----------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(attn)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018): channel
    attention followed by spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


# -----------------------------------------------------------------------------
# Residual block (He et al., 2016 pre-activation style) -- used by
# resunet.py, resunetpp.py, and ra_unet.py
# -----------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Pre-activation residual unit: BN-ReLU-Conv, twice, with an identity
    (or 1x1-conv projection, when channels/stride change) skip connection.
    `stride` on the first conv is how these models downsample instead of
    a separate pooling layer, following the residual-unit U-Net design
    used by Zhang et al. (2018)'s ResUNet and its derivatives.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        needs_projection = (stride != 1) or (in_channels != out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if needs_projection
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + shortcut


# -----------------------------------------------------------------------------
# Squeeze-and-Excitation (Hu et al., 2018) -- used by resunetpp.py
# -----------------------------------------------------------------------------

class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


# -----------------------------------------------------------------------------
# ASPP -- Atrous Spatial Pyramid Pooling (Chen et al., 2017) -- used by
# resunetpp.py as the encoder-decoder bridge and again before the final
# output convolution
# -----------------------------------------------------------------------------

class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates: tuple[int, ...] = (1, 6, 12, 18)) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for rate in rates:
            if rate == 1:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        global_feat = self.global_branch(x)
        global_feat = upsample_to(global_feat, x)
        outputs.append(global_feat)
        return self.project(torch.cat(outputs, dim=1))


# -----------------------------------------------------------------------------
# Additive attention gate (Oktay et al., 2018 style) -- used by
# resunetpp.py's decoder skip connections
# -----------------------------------------------------------------------------

class AttentionGate(nn.Module):
    """
    Uses a coarser 'gating' signal from the decoder to weight a finer-
    resolution skip connection from the encoder before they are combined.
    Returns the *attended skip feature* (same shape as `skip`), to be
    concatenated with the (separately upsampled) decoder feature.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_conv = nn.Conv2d(gate_channels, inter_channels, 1)
        self.skip_conv = nn.Conv2d(skip_channels, inter_channels, 1)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = upsample_to(self.gate_conv(gate), skip)
        s = self.skip_conv(skip)
        attn = self.psi(self.relu(g + s))
        return skip * attn


# -----------------------------------------------------------------------------
# Attention Residual Module (Wang et al., 2017's trunk/soft-mask branch
# attention, as used inside RA-UNet -- Jin et al., 2020) -- used by
# ra_unet.py
# -----------------------------------------------------------------------------

class AttentionResidualModule(nn.Module):
    """
    2D adaptation of the 'Attention Residual Learning' module: a trunk
    branch (feature processing) runs in parallel with a soft mask branch
    (downsample -> process -> upsample -> squash to [0, 1]), combined as

        H(x) = (1 + M(x)) * T(x)

    so that when the mask is near zero, features pass through unchanged
    (the "+1" is what makes this safe to stack deeply). The original
    RA-UNet is a 3D network for volumetric CT; this module is a 2D,
    single-scale (one pool/upsample step) simplification for this
    project's 2D slice inputs.
    """

    def __init__(self, channels: int, trunk_blocks: int = 2) -> None:
        super().__init__()
        self.trunk = nn.Sequential(*[ResidualBlock(channels, channels) for _ in range(trunk_blocks)])
        self.mask_down = nn.MaxPool2d(2)
        self.mask_block = ResidualBlock(channels, channels)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk_out = self.trunk(x)
        mask = self.mask_block(self.mask_down(x))
        mask = upsample_to(mask, trunk_out)
        mask = self.mask_conv(mask)
        return (1.0 + mask) * trunk_out
