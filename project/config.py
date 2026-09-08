"""
Shared configuration: paths, reproducibility, and per-dataset settings for
the breast lesion segmentation project (mammograms via CBIS-DDSM, breast
MRI slices via RIDER).

Every other module in this project imports from here. This module itself
imports nothing project-local, to keep the dependency graph a clean tree
(config -> datasets -> cross_validation -> models -> training -> tuning
-> experiments) with no cycles.
"""

from __future__ import annotations

import random
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
SPLITS_DIR = BASE_DIR / "splits"  # persisted CV fold assignments (see cross_validation/folds.py)

MAMMOGRAM_ROOT = DATA_DIR / "mammograms"
MRI_ROOT = DATA_DIR / "breast_mri"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # safest default on Windows
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
    median_filter: bool = False
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
    # RIDER splits at the patient level. Left unset here deliberately (see
    # experiments/run_all.py, which sets this for MRI_CONFIG before running
    # the pipeline) so importing this module never has an opinion on
    # whether the heuristic in cross_validation/folds.py has been verified
    # against your actual filenames.
    patient_id_fn: Callable[[Path], str] | None = None


# Mammograms retain more spatial detail, so this default uses 512x512.
# Reduce to (256, 256) if GPU memory is limited.
# NOTE: CBIS-DDSM patients can contribute multiple images (views/lesions).
# The methodology's patient-level split requirement (section 3.2) is
# specific to RIDER, but the same leakage risk can apply here during
# Bayesian loss-weight tuning -- set patient_id_fn below if you want CV
# folds to be patient-safe for mammograms too.
MAMMOGRAM_CONFIG = DatasetConfig(
    name="mammogram",
    root=MAMMOGRAM_ROOT,
    input_size=(512, 512),
    in_channels=1,
    batch_size=4,
    epochs=30,
    learning_rate=1e-4,
    pretrained_encoder=False,
    deep_supervision=True,
    clahe=True,
    median_filter=True,
    horizontal_flip_probability=0.5,
    rotation_degrees=10.0,
    translation_fraction=0.03,
)

# MRI slices are trained in their own experiment. These defaults are intentionally
# independent from the mammogram settings.
# NOTE: RIDER provides only 5 patients. For Bayesian loss-weight tuning to
# build patient-safe CV folds (rather than per-slice folds), set
# patient_id_fn, e.g.:
#   from cross_validation.folds import default_patient_id_from_filename
#   MRI_CONFIG.patient_id_fn = default_patient_id_from_filename
# and verify the grouping with cross_validation.folds.preview_cv_groups()
# before trusting the result. experiments/run_all.py does this already.
MRI_CONFIG = DatasetConfig(
    name="breast_mri",
    root=MRI_ROOT,
    input_size=(288, 288),
    in_channels=4,
    batch_size=8,
    epochs=30,
    learning_rate=1e-4,
    pretrained_encoder=False,
    deep_supervision=True,
    clahe=False,
    median_filter=False,
    horizontal_flip_probability=0.5,
    rotation_degrees=20.0,
    translation_fraction=0.08,
)
