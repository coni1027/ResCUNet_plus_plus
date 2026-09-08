"""
Hybrid BCE + Dice loss, with deep-supervision averaging (Methodology
3.7.5). Every model in this project returns either a single logits
tensor or a list of them (deep supervision) -- DeepSupervisionLoss and
final_logits() both branch on which it is, so this file works unchanged
for every architecture in models/, not just the proposed one.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.flatten(1)
        targets = targets.flatten(1)

        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )


class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss: nn.Module) -> None:
        super().__init__()
        self.base_loss = base_loss

    def forward(
        self,
        outputs: torch.Tensor | Sequence[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return self.base_loss(outputs, targets)

        losses = [self.base_loss(output, targets) for output in outputs]
        return torch.stack(losses).mean()


def final_logits(outputs: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    return outputs if isinstance(outputs, torch.Tensor) else outputs[-1]
