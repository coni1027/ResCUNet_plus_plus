"""
Model registry: maps a short model name to its build_model(config)
factory and a human-readable label. training/trainer.py,
tuning/bayesian.py, and experiments/*.py all go through this registry
instead of hardcoding any one architecture, so adding a model here is
enough to make it available everywhere (training, tuning, SOTA
comparison).

To add another baseline later: write models/your_model.py with a
build_model(config) -> nn.Module function (same signature as the others),
then add one line to MODEL_REGISTRY and MODEL_LABELS below.
"""

from __future__ import annotations

from typing import Callable

from torch import nn

from config import DatasetConfig
from models.resunetpp_cbam import build_model as _build_resunetpp_cbam
from models.unet import build_model as _build_unet
from models.unetpp import build_model as _build_unetpp
from models.resunet import build_model as _build_resunet
from models.resunetpp import build_model as _build_resunetpp
from models.ra_unet import build_model as _build_ra_unet
from models.cbam_unet import build_model as _build_cbam_unet

MODEL_REGISTRY: dict[str, Callable[[DatasetConfig], nn.Module]] = {
    "resunetpp_cbam": _build_resunetpp_cbam,  # proposed model
    "unet": _build_unet,
    "unetpp": _build_unetpp,
    "resunet": _build_resunet,
    "resunetpp": _build_resunetpp,
    "ra_unet": _build_ra_unet,
    "cbam_unet": _build_cbam_unet,
}

# Human-readable labels for printouts and comparison tables.
MODEL_LABELS: dict[str, str] = {
    "resunetpp_cbam": "ResNet34 U-Net++ + CBAM (proposed)",
    "unet": "U-Net",
    "unetpp": "U-Net++",
    "resunet": "ResUNet",
    "resunetpp": "ResUNet++",
    "ra_unet": "RA-UNet",
    "cbam_unet": "CBAM-UNet",
}
