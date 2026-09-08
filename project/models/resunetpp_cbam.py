"""
Proposed model: ResNet34 encoder + U-Net++ nested decoder + CBAM on the
skip connections (Methodology sections 3.7.1-3.7.4), hybrid BCE-Dice loss
(3.7.5, applied outside this file -- see training/losses.py).

Every one of those five methodology components is now an isolated,
independently toggleable flag on this same model class, so
experiments/run_ablation.py can remove exactly one at a time and hold
everything else identical:

- use_cbam=False swaps every CBAM module for nn.Identity(). Doesn't
  touch channel counts or scales anywhere -- a pure no-op removal.
- deep_supervision=False changes which outputs feed the loss, not the
  forward pass itself (all four nested outputs are still computed).
- use_resnet=False swaps ResNet34Encoder for PlainEncoder, a
  structural twin built from PlainBasicBlock (same channel counts,
  same spatial scales, same per-stage depth) instead of residual
  BasicBlocks -- isolates whether the residual connections in the
  backbone matter, without also changing width or depth the way
  comparing against a genuinely different architecture (e.g.
  models.unetpp.UNetPlusPlus) would.
- nested_decoder=False swaps the dense U-Net++ nested grid for a
  single-path plain decoder (one skip connection per level, no
  cross-column reuse) with matched per-level channel capacity --
  isolates whether the nesting itself matters. Requires
  deep_supervision=False: a single-path decoder has one output, so
  there are no intermediate nested nodes left to supervise.

BCE-Dice composition isn't a flag here since it's a loss-side choice,
not an architecture one -- see experiments/run_ablation.py's
bce_weight/dice_weight per-variant overrides (bce_only / dice_only).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34

from config import DatasetConfig, DEVICE
from models.common import CBAM, DoubleConv, upsample_to


class ResNet34Encoder(nn.Module):
    """ResNet34 feature extractor returning five spatial scales."""

    def __init__(self, in_channels: int = 1, pretrained: bool = False) -> None:
        super().__init__()

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)

        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

            if pretrained:
                with torch.no_grad():
                    if in_channels == 1:
                        # Preserve pretrained information by averaging RGB filters.
                        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                    else:
                        # General initialization for multi-channel MRI input.
                        mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                        new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
                        new_conv.weight.mul_(3.0 / in_channels)

            backbone.conv1 = new_conv

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Channels returned by forward():
        # f0=64, f1=64, f2=128, f3=256, f4=512
        self.out_channels = (64, 64, 128, 256, 512)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f0 = self.relu(self.bn1(self.conv1(x)))       # H/2
        f1 = self.layer1(self.maxpool(f0))            # H/4
        f2 = self.layer2(f1)                          # H/8
        f3 = self.layer3(f2)                          # H/16
        f4 = self.layer4(f3)                          # H/32
        return f0, f1, f2, f3, f4


class PlainBasicBlock(nn.Module):
    """
    Structural twin of torchvision's resnet.BasicBlock (conv3x3 - bn -
    relu - conv3x3 - bn, stride on the first conv, ReLU after both
    stages) but WITHOUT the residual shortcut addition. Every other
    detail -- kernel sizes, stride placement, channel progression,
    number of blocks per stage -- stays identical to what
    ResNet34Encoder actually uses, so PlainEncoder isolates exactly one
    thing: whether the residual connection itself contributes anything.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out)  # no shortcut add -- the only structural difference from BasicBlock


def _make_plain_stage(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
    layers = [PlainBasicBlock(in_channels, out_channels, stride=stride)]
    layers += [PlainBasicBlock(out_channels, out_channels, stride=1) for _ in range(num_blocks - 1)]
    return nn.Sequential(*layers)


class PlainEncoder(nn.Module):
    """
    Structural twin of ResNet34Encoder: identical stem, identical
    per-stage depth ([3, 4, 6, 3] blocks -- ResNet-34's actual
    configuration), identical channel counts and spatial scales at
    every tap point -- (64, 64, 128, 256, 512) at H/2..H/32 -- but every
    stage is built from PlainBasicBlock instead of a residual
    BasicBlock. This is what makes "does the ResNet backbone matter" a
    clean, isolated toggle (ResNetUNetPlusPlus's use_resnet=False)
    instead of a confounded comparison against a differently-shaped
    architecture like models.unetpp.UNetPlusPlus (which also removes
    CBAM and uses different channel widths at the same time).
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = _make_plain_stage(64, 64, num_blocks=3, stride=1)
        self.stage2 = _make_plain_stage(64, 128, num_blocks=4, stride=2)
        self.stage3 = _make_plain_stage(128, 256, num_blocks=6, stride=2)
        self.stage4 = _make_plain_stage(256, 512, num_blocks=3, stride=2)
        self.out_channels = (64, 64, 128, 256, 512)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f0 = self.stem(x)                     # H/2,  64ch  -- identical op to ResNet's conv1+bn1+relu
        f1 = self.stage1(self.maxpool(f0))     # H/4,  64ch
        f2 = self.stage2(f1)                   # H/8,  128ch
        f3 = self.stage3(f2)                   # H/16, 256ch
        f4 = self.stage4(f3)                   # H/32, 512ch
        return f0, f1, f2, f3, f4


class ResNetUNetPlusPlus(nn.Module):
    """
    U-Net++ with a ResNet34 encoder and CBAM on the shortest skip
    connections -- see the module docstring for how each of the four
    architectural flags below maps to a methodology section and isolates
    exactly one component.

    The nested dense skip connections are x(i,j), where j > 0 represents
    increasingly refined decoder nodes; node x(i,j) is fused from every
    earlier same-row node x(i,0..j-1) plus an upsampled feature from the
    row below. CBAM is applied only to the *shortest* hop into each node
    -- i.e. its immediate same-row predecessor, x(i,j-1) -- right before
    that concatenation. With deep supervision enabled, the model returns
    four full-resolution logits maps during training/evaluation.

    When nested_decoder=False, forward() takes a completely different
    path (_forward_plain): a single top-to-bottom decoder with one skip
    connection per level (CBAM applied to each, same placement idea as
    the nested design's first column), matched to the same per-level
    channel budget (d0..d3) as the nested decoder, and exactly one
    output -- see _forward_plain.
    """

    def __init__(
        self,
        in_channels: int = 1,
        pretrained_encoder: bool = False,
        deep_supervision: bool = True,
        use_cbam: bool = True,
        use_resnet: bool = True,
        nested_decoder: bool = True,
    ) -> None:
        super().__init__()
        if not nested_decoder and deep_supervision:
            raise ValueError(
                "deep_supervision=True requires nested_decoder=True: a plain "
                "(non-nested) decoder has only one output, so there are no "
                "intermediate nested nodes left to supervise. Pass "
                "deep_supervision=False when nested_decoder=False."
            )

        self.deep_supervision = deep_supervision
        self.use_cbam = use_cbam
        self.nested_decoder = nested_decoder
        self.encoder = (
            ResNet34Encoder(in_channels, pretrained_encoder) if use_resnet
            else PlainEncoder(in_channels)
        )

        # Decoder node widths per row. Keeping these close to the encoder widths
        # makes the architecture easy to inspect and modify later (e.g., CBAM).
        # Used by BOTH decoder paths below, so the nested-vs-plain comparison
        # has a matched per-level channel budget, not just a matched backbone.
        c0, c1, c2, c3, c4 = self.encoder.out_channels
        d0, d1, d2, d3 = 64, 64, 128, 256

        def _attn(channels: int) -> nn.Module:
            return CBAM(channels) if use_cbam else nn.Identity()

        if nested_decoder:
            # First nested column
            self.conv0_1 = DoubleConv(c0 + c1, d0)
            self.conv1_1 = DoubleConv(c1 + c2, d1)
            self.conv2_1 = DoubleConv(c2 + c3, d2)
            self.conv3_1 = DoubleConv(c3 + c4, d3)

            # Second nested column
            self.conv0_2 = DoubleConv(c0 + d0 + d1, d0)
            self.conv1_2 = DoubleConv(c1 + d1 + d2, d1)
            self.conv2_2 = DoubleConv(c2 + d2 + d3, d2)

            # Third nested column
            self.conv0_3 = DoubleConv(c0 + d0 + d0 + d1, d0)
            self.conv1_3 = DoubleConv(c1 + d1 + d1 + d2, d1)

            # Fourth/final nested column
            self.conv0_4 = DoubleConv(c0 + d0 + d0 + d0 + d1, d0)

            # One CBAM per shortest (adjacent, same-row) skip connection --
            # e.g. cbam0_1 refines x0_0 specifically for its use as the
            # immediate predecessor feeding conv0_1 (the X0,0->X0,1 edge);
            # cbam0_2 refines x0_1 specifically for the X0,1->X0,2 edge; and
            # so on. Longer-distance same-row reuses (e.g. x0_0 appearing
            # again inside X0,3's or X0,4's concatenation) and the vertical
            # upsample input are left unrefined -- see _forward_nested().
            # use_cbam=False swaps every CBAM for a no-op nn.Identity(), so
            # forward() doesn't need an if/else at each call site.
            self.cbam0_1 = _attn(d0)
            self.cbam1_1 = _attn(d1)
            self.cbam2_1 = _attn(d2)
            self.cbam3_1 = _attn(d3)
            self.cbam0_2 = _attn(d0)
            self.cbam1_2 = _attn(d1)
            self.cbam2_2 = _attn(d2)
            self.cbam0_3 = _attn(d0)
            self.cbam1_3 = _attn(d1)
            self.cbam0_4 = _attn(d0)

            # Binary lesion segmentation -> one output channel.
            self.out1 = nn.Conv2d(d0, 1, kernel_size=1)
            self.out2 = nn.Conv2d(d0, 1, kernel_size=1)
            self.out3 = nn.Conv2d(d0, 1, kernel_size=1)
            self.out4 = nn.Conv2d(d0, 1, kernel_size=1)
        else:
            # Single top-to-bottom path: each level fuses CBAM(encoder skip)
            # with the upsampled output of the PREVIOUS DECODER stage (not a
            # raw encoder feature the way the nested design's first column
            # does) -- this is what makes it a plain decoder rather than a
            # one-column slice of the nested one. Same d0..d3 channel budget
            # as the nested decoder, so this is a fair "same capacity,
            # different topology" comparison. See _forward_plain().
            self.plain_conv3 = DoubleConv(c3 + c4, d3)
            self.plain_conv2 = DoubleConv(c2 + d3, d2)
            self.plain_conv1 = DoubleConv(c1 + d2, d1)
            self.plain_conv0 = DoubleConv(c0 + d1, d0)

            self.plain_cbam3 = _attn(c3)
            self.plain_cbam2 = _attn(c2)
            self.plain_cbam1 = _attn(c1)
            self.plain_cbam0 = _attn(c0)

            self.plain_out = nn.Conv2d(d0, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        input_size = x.shape[-2:]
        x0_0, x1_0, x2_0, x3_0, x4_0 = self.encoder(x)
        if self.nested_decoder:
            return self._forward_nested(x0_0, x1_0, x2_0, x3_0, x4_0, input_size)
        return self._forward_plain(x0_0, x1_0, x2_0, x3_0, x4_0, input_size)

    def _forward_nested(
        self,
        x0_0: torch.Tensor, x1_0: torch.Tensor, x2_0: torch.Tensor,
        x3_0: torch.Tensor, x4_0: torch.Tensor,
        input_size: torch.Size,
    ) -> torch.Tensor | list[torch.Tensor]:
        # CBAM sits only on the shortest (adjacent, same-row) skip
        # connection feeding each node -- X0,0->X0,1, then X0,1->X0,2,
        # and so on for every row. x0_0's other, longer-distance reuse
        # inside X0,3's/X0,4's concatenation below stays raw.
        x0_1 = self.conv0_1(torch.cat([self.cbam0_1(x0_0), upsample_to(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([self.cbam1_1(x1_0), upsample_to(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([self.cbam2_1(x2_0), upsample_to(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([self.cbam3_1(x3_0), upsample_to(x4_0, x3_0)], dim=1))

        x0_2 = self.conv0_2(
            torch.cat([x0_0, self.cbam0_2(x0_1), upsample_to(x1_1, x0_0)], dim=1)
        )
        x1_2 = self.conv1_2(
            torch.cat([x1_0, self.cbam1_2(x1_1), upsample_to(x2_1, x1_0)], dim=1)
        )
        x2_2 = self.conv2_2(
            torch.cat([x2_0, self.cbam2_2(x2_1), upsample_to(x3_1, x2_0)], dim=1)
        )

        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, self.cbam0_3(x0_2), upsample_to(x1_2, x0_0)], dim=1)
        )
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, self.cbam1_3(x1_2), upsample_to(x2_2, x1_0)], dim=1)
        )

        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, self.cbam0_4(x0_3), upsample_to(x1_3, x0_0)], dim=1)
        )

        outputs = [self.out1(x0_1), self.out2(x0_2), self.out3(x0_3), self.out4(x0_4)]
        outputs = [
            F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
            for out in outputs
        ]

        return outputs if self.deep_supervision else outputs[-1]

    def _forward_plain(
        self,
        x0_0: torch.Tensor, x1_0: torch.Tensor, x2_0: torch.Tensor,
        x3_0: torch.Tensor, x4_0: torch.Tensor,
        input_size: torch.Size,
    ) -> torch.Tensor:
        p3 = self.plain_conv3(torch.cat([self.plain_cbam3(x3_0), upsample_to(x4_0, x3_0)], dim=1))
        p2 = self.plain_conv2(torch.cat([self.plain_cbam2(x2_0), upsample_to(p3, x2_0)], dim=1))
        p1 = self.plain_conv1(torch.cat([self.plain_cbam1(x1_0), upsample_to(p2, x1_0)], dim=1))
        p0 = self.plain_conv0(torch.cat([self.plain_cbam0(x0_0), upsample_to(p1, x0_0)], dim=1))

        out = self.plain_out(p0)
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


def build_model(config: DatasetConfig) -> ResNetUNetPlusPlus:
    """Shared model constructor so tuning and final training stay in sync.

    Always the full model (ResNet backbone, nested decoder) -- use_resnet
    and nested_decoder are ablation-only knobs, constructed directly by
    experiments/run_ablation.py, the same way use_cbam already was.
    """
    return ResNetUNetPlusPlus(
        in_channels=config.in_channels,
        pretrained_encoder=config.pretrained_encoder,
        deep_supervision=config.deep_supervision,
    ).to(DEVICE)
