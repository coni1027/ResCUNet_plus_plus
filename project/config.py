"""
Shared configuration: paths, reproducibility, and per-dataset settings for
the breast lesion segmentation project (mammograms via CBIS-DDSM, breast
DCE-MRI slices via MAMA-MIA).

Every other module in this project imports from here. This module itself
imports nothing project-local, to keep the dependency graph a clean tree
(config -> datasets -> cross_validation -> models -> training -> tuning
-> experiments) with no cycles.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Paths and reproducibility
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
SPLITS_DIR = BASE_DIR / "splits"    # persisted CV fold assignments (see cross_validation/folds.py)
                                    # cross-validation split should be saved instead of randomly recreated every time

MAMMOGRAM_ROOT = DATA_DIR / "mammograms"
MRI_ROOT = DATA_DIR / "breast_mri"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DataLoader worker processes. Loading + preprocessing (CLAHE, mean filter,
# resize, augmentation) runs on the CPU; with workers it overlaps GPU compute
# instead of stalling it. 0 (load in the main process) on Windows, where
# workers are spawned rather than forked and often slower or fragile; up to 8
# elsewhere (e.g. Colab). Override with the NUM_WORKERS environment variable.
NUM_WORKERS = int(os.environ.get(
    "NUM_WORKERS", 0 if sys.platform == "win32" else min(8, os.cpu_count() or 1)
))
SEED = 42

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".dcm"
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Per-dataset configuration
# -----------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    name: str
    root: Path
    input_size: tuple[int, int]  # (height, width)
    in_channels: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float = 1e-5
    pretrained_encoder: bool = False
    deep_supervision: bool = True

    # Preprocessing
    clahe: bool = False
    mean_filter: bool = False
    percentile_clip: tuple[float, float] = (1.0, 99.0)

    # Training augmentation
    horizontal_flip_probability: float = 0.5
    rotation_degrees: float = 15.0
    translation_fraction: float = 0.05

    # Hybrid BCE + Dice loss weights (defaults; can be overridden per-call
    # by tuning.bayesian.tune_bce_dice_weight() / training.trainer.train_model())
    bce_weight: float = 0.5
    dice_weight: float = 0.5

    # Optional grouping hook for the k-fold CV robustness check that runs
    # AFTER loss-weight tuning (see cross_validation/folds.py and
    # experiments/run_kfold_cv.py -- NOT tuning.bayesian, which uses a
    # single fixed split and never groups). Given an image Path, return a
    # string group ID -- typically a patient ID -- so every slice from the
    # same patient/case lands in the same CV fold. Leave as None to fall
    # back to per-sample folds; run_kfold_cv() prints a warning when that
    # happens for breast_mri, since methodology section 3.2 requires
    # MRI splits at the patient level. Left unset here deliberately (see
    # experiments/run_all.py, which sets this for MRI_CONFIG before running
    # the pipeline) so importing this module never has an opinion on
    # whether the heuristic in cross_validation/folds.py has been verified
    # against your actual filenames.
    patient_id_fn: Callable[[Path], str] | None = None


# Mammograms retain more spatial detail, so this default uses 512x512.
# Reduce to (256, 256) if GPU memory is limited.
# NOTE: CBIS-DDSM patients can contribute multiple images (views/lesions),
# e.g. "P_00001_LEFT_CC" and "P_00001_LEFT_MLO" are the same patient.
# experiments/run_all.py sets patient_id_fn = default_patient_id_from_
# filename for this config too, so CV folds are patient-safe here as
# well as for MRI -- verified against the actual "<patient>_<side>_
# <view>" export this project uses (see prepare_mammograms_presplit.py).
MAMMOGRAM_CONFIG = DatasetConfig(
    name="mammogram",
    root=MAMMOGRAM_ROOT,
    input_size=(1024, 1024),
    in_channels=1,
    batch_size=4,
    epochs=50,
    learning_rate=1e-4,
    pretrained_encoder=False,
    deep_supervision=True,
    clahe=True,
    mean_filter=True,
    horizontal_flip_probability=0.5,
    rotation_degrees=10.0,
    translation_fraction=0.03,
)

# MRI slices are trained in their own experiment. These defaults are intentionally
# independent from the mammogram settings.
# Data: MAMA-MIA (DUKE, ISPY1, ISPY2, NACT), written into MRI_ROOT by
# preparation/prepare_mama_mia.py. The collections have different numbers
# of DCE phases (e.g. DUKE _0000.._0004, NACT _0000.._0002), so every case
# is reduced to the three phases all of them have -- pre-contrast (_0000),
# 1st post-contrast (_0001), 2nd post-contrast (_0002) -- which become the
# 3 input channels, in that order. input_size is also the size
# prepare_mama_mia.py resizes slices to (native in-plane size is 256-512).
# For patient-safe CV folds set patient_id_fn, e.g.:
#   from cross_validation.folds import default_patient_id_from_filename
#   MRI_CONFIG.patient_id_fn = default_patient_id_from_filename
# and verify the grouping with cross_validation.folds.preview_cv_groups()
# before trusting the result. experiments/run_all.py does this already.
MRI_CONFIG = DatasetConfig(
    name="breast_mri",
    root=MRI_ROOT,
    input_size=(256, 256),
    in_channels=3,
    batch_size=8,
    epochs=50,
    learning_rate=1e-4,
    pretrained_encoder=False,
    deep_supervision=True,
    clahe=False,
    mean_filter=False,
    horizontal_flip_probability=0.5,
    rotation_degrees=20.0,
    translation_fraction=0.08,
)
