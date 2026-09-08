"""
Binary segmentation metrics and the shared evaluation loop. `evaluate()`
takes a generic `model: nn.Module`, so it works unchanged for every
architecture in models/ (each model's forward() returns a tensor or a
list of tensors; final_logits() from training.losses handles both).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.losses import final_logits


@torch.no_grad()
def batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-7,
) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
    targets_bool = targets >= 0.5

    dims = (1, 2, 3)
    tp = (predictions & targets_bool).sum(dim=dims).float()
    fp = (predictions & ~targets_bool).sum(dim=dims).float()
    fn = (~predictions & targets_bool).sum(dim=dims).float()
    tn = (~predictions & ~targets_bool).sum(dim=dims).float()

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "accuracy": accuracy.mean().item(),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    totals = {
        "loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
    }

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, masks)
        metrics = batch_metrics(final_logits(outputs), masks)

        totals["loss"] += loss.item()
        for key in metrics:
            totals[key] += metrics[key]

    count = max(1, len(loader))
    return {key: value / count for key, value in totals.items()}
