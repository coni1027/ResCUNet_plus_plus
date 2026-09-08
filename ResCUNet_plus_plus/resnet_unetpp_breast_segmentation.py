"""
ResNet34 + U-Net++ for breast lesion segmentation.

This script is adapted from the structure of the user's Lab 3 segmentation code,
but changes the task to binary breast-lesion segmentation and trains mammograms
and 2D breast MRI slices as separate experiments.

Expected directory structure
----------------------------
project/
├── resnet_unetpp_breast_segmentation.py
├── data/
│   ├── mammograms/
│   │   ├── train/images/
│   │   ├── train/masks/
│   │   ├── val/images/
│   │   ├── val/masks/
│   │   ├── test/images/
│   │   └── test/masks/
│   └── breast_mri/
│       ├── train/images/
│       ├── train/masks/
│       ├── val/images/
│       ├── val/masks/
│       ├── test/images/
│       └── test/masks/
└── checkpoints/

Supported image formats: PNG/JPG/TIFF/BMP, .npy, and DICOM (.dcm).
Masks are treated as binary: background=0, lesion=1 (or any value > 0).

Important: create patient-level train/val/test splits BEFORE running this script,
especially for MRI slices, so slices from the same patient do not leak across splits.

Architecture notes
------------------
- The ResNet34 encoder is trained from scratch (pretrained_encoder=False in both
  configs) rather than starting from ImageNet weights.
- CBAM (channel attention, then spatial attention) is applied to each encoder
  feature map at the point it branches into a U-Net++ skip connection -- i.e.
  right before it is fused (concatenated) with the upsampled decoder feature
  below it. See the CBAM/ResNetUNetPlusPlus docstrings for detail.

Optional: rendering a model architecture diagram
--------------------------------------------------
Run with --diagram (instead of training) to save a schematic PNG of the
model architecture -- a hand-drawn topology (not a traced forward pass),
styled after the WBC-Net figure: the full nested U-Net++ grid with dense
same-row skip connections drawn as arced dashed lines, a diagonal ResNet
encoder backbone, vertical upsampling, and a badge marking where CBAM is
applied on each skip connection. Since it's a static topology drawing
rather than a traced model, it's identical for every dataset -- one file,
`results/architecture_diagram.png`, regardless of --dataset.
Install with: pip install matplotlib
See save_architecture_diagram() for layout/color options.

Optional: Bayesian optimization for the hybrid loss weighting
---------------------------------------------------------------
Run with --tune-loss-weights to search for the balance between the BCE and
Dice terms of the hybrid loss (bce_weight vs. dice_weight, expressed as a
single alpha in [0, 1] with dice_weight = 1 - alpha) using Optuna's Bayesian
optimizer (TPE sampler, the same family of method as scikit-optimize/GPyOpt).
Install with: pip install optuna

Each trial is now scored with K-fold cross-validation (--cv-folds, default
5) over the pooled train+val samples, instead of a single fixed split: for
every fold, a fresh model trains for --tuning-epochs epochs on that fold's
train side and is evaluated on its held-out side, and the trial's objective
is the MEAN validation Dice across folds. The `test` split is never touched
by tuning. Run with --show-cv-groups to preview the fold assignment for
free before committing to a full tuning run.

Please read these caveats before trusting the result of that search:

1. Each trial still retrains FRESH models for a reduced number of epochs
   per fold (--tuning-epochs, default 5) instead of the full schedule, as a
   compute-saving proxy. The alpha that looks best after a few short epochs
   is not guaranteed to still be best after the full 30-epoch run -- treat
   the tuned value as a good starting point, not a certified optimum, and
   re-validate with a full training run.
2. Folds are only patient-safe -- i.e. no patient's slices land on both
   sides of a fold -- if you set DatasetConfig.patient_id_fn to a function
   that correctly maps an image Path to a patient/case ID for YOUR exported
   filenames (default_patient_id_from_filename is a best-effort starting
   point, not a validated parser for your files). Leave it unset and
   tune_bce_dice_weight() will build folds per SLICE instead, and will
   print a loud warning for breast_mri, since section 3.2 of the
   methodology requires RIDER splits at the patient level. Verify with
   --show-cv-groups before trusting grouped folds.
3. Even with correct patient grouping, RIDER has only 5 patients total
   (60/20/20 patient-level split per section 3.2, so at most 4 patients are
   ever pooled into train+val here -- 1 is permanently held out for test).
   That caps RIDER's CV at 4-fold (leave-one-patient-out): a real
   improvement over a single patient's validation score, but still a
   small-N estimate -- treat any MRI-side "optimal" alpha as a rough one.
4. Cross-validation multiplies compute roughly (n_trials x cv_folds x
   tuning_epochs) instead of (n_trials x tuning_epochs) -- several times
   the runtime of the old single-split search, especially for the 512x512
   mammogram config. The MedianPruner still cuts unpromising trials short,
   but now between folds rather than between epochs (a trial can't be
   pruned mid-fold, only after completing at least two full folds). Lower
   --n-trials, --tuning-epochs, or --cv-folds if the cost is prohibitive.
5. The same random seed re-initializes the model at the start of every
   fold of every trial, so the search isolates the effect of alpha (and
   each fold's data) rather than random-init noise. The result has still
   only been checked for one seed; consider re-running with 2-3 different
   seeds before trusting a close call.
6. This tunes only the BCE/Dice balance. It reuses whatever batch size,
   learning rate, augmentation, and input resolution are already hard-coded
   in each DatasetConfig -- those are not part of the search space here.
7. Tuning's CV folds and the final training run's fixed val split both
   ultimately come from the same train+val pool, and the final run's
   checkpoint selection still reads that same fixed val split. That's
   standard practice, but it means validation data is "used twice" (once,
   across fold combinations, to pick alpha; once, whole, to pick the best
   epoch) -- for a paper-grade generalization estimate, only the held-out
   test metrics printed at the end should be reported.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet34_Weights, resnet34

try:
    import pydicom
except ImportError:  # DICOM is optional unless .dcm files are used.
    pydicom = None

try:
    import optuna
except ImportError:  # Optuna is optional unless --tune-loss-weights is used.
    optuna = None


# -----------------------------------------------------------------------------
# Paths and reproducibility
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

MAMMOGRAM_ROOT = DATA_DIR / "mammograms"
MRI_ROOT = DATA_DIR / "breast_mri"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Separate configuration for each modality
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

    # Hybrid BCE + Dice loss weights (defaults; can be overridden by
    # tune_bce_dice_weight() / train_one_dataset()'s bce_weight/dice_weight args)
    bce_weight: float = 0.5
    dice_weight: float = 0.5

    # Optional grouping hook for cross-validation during --tune-loss-weights
    # (see tune_bce_dice_weight / make_group_folds). Given an image Path,
    # return a string group ID -- typically a patient ID -- so every slice
    # from the same patient/case lands in the same CV fold. Leave as None
    # to fall back to per-sample folds; tune_bce_dice_weight() prints a
    # warning when that happens for breast_mri, since methodology section
    # 3.2 requires RIDER splits at the patient level.
    patient_id_fn: Callable[[Path], str] | None = None


# Mammograms retain more spatial detail, so this default uses 512x512.
# Reduce to (256, 256) if GPU memory is limited.
# NOTE: CBIS-DDSM patients can contribute multiple images (views/lesions).
# The methodology's patient-level split requirement (section 3.2) is
# specific to RIDER, but the same leakage risk can apply here during
# --tune-loss-weights -- set patient_id_fn below if you want CV folds to
# be patient-safe for mammograms too.
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
# NOTE: RIDER provides only 5 patients (see module docstring, points 2-3).
# For --tune-loss-weights to build patient-safe CV folds (rather than
# per-slice folds), set patient_id_fn below, e.g.:
#   MRI_CONFIG.patient_id_fn = default_patient_id_from_filename
# and verify the grouping with --show-cv-groups before trusting the result.
MRI_CONFIG = DatasetConfig(
    name="breast_mri",
    root=MRI_ROOT,
    input_size=(256, 256),
    in_channels=1,
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


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def iter_image_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def canonical_stem(path: Path) -> str:
    """Make common mask suffixes compatible with image names."""
    stem = path.stem.lower()
    for suffix in ("_mask", "-mask", "_seg", "-seg", "_label", "-label"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def paired_samples(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    image_paths = iter_image_paths(images_dir)
    mask_paths = iter_image_paths(masks_dir)

    image_map = {canonical_stem(path): path for path in image_paths}
    mask_map = {canonical_stem(path): path for path in mask_paths}

    common = sorted(set(image_map) & set(mask_map))
    missing_masks = sorted(set(image_map) - set(mask_map))
    missing_images = sorted(set(mask_map) - set(image_map))

    if missing_masks or missing_images:
        message = []
        if missing_masks:
            message.append(f"missing masks for: {', '.join(missing_masks[:5])}")
        if missing_images:
            message.append(f"missing images for: {', '.join(missing_images[:5])}")
        raise ValueError("Image/mask mismatch: " + "; ".join(message))

    return [(image_map[key], mask_map[key]) for key in common]


def load_dicom(path: Path) -> np.ndarray:
    if pydicom is None:
        raise ImportError(
            "DICOM file encountered but pydicom is not installed. "
            "Install it with: pip install pydicom"
        )

    ds = pydicom.dcmread(str(path))
    array = ds.pixel_array.astype(np.float32)

    # Mammograms may use MONOCHROME1, where lower stored values appear brighter.
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        array = array.max() - array

    return array


def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.load(path).astype(np.float32)

    if suffix == ".dcm":
        return load_dicom(path)

    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError(f"Could not read: {path}")

    if array.ndim == 3:
        # Convert ordinary RGB/BGR images to grayscale for a 1-channel encoder.
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        else:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    return array.astype(np.float32)


def normalize_percentile(
    image: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    """Robustly normalize a medical image to [0, 1]."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    low = np.percentile(finite, low_percentile)
    high = np.percentile(finite, high_percentile)

    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return image.astype(np.float32)


def preprocess_image(image: np.ndarray, config: DatasetConfig) -> np.ndarray:
    """
    Returns HxW or CxHxW floating-point data in [0, 1].

    For multi-channel .npy MRI input, each channel is normalized independently.

    NOTE: when clahe/median_filter is enabled, the [0,1] float image is
    quantized to uint8 before CLAHE/median filtering (OpenCV requires 8-bit
    input for these ops), then converted back to float. For 16-bit CBIS-DDSM
    mammograms this discards most of the original bit depth in exchange for
    CLAHE contrast enhancement -- a reasonable and common trade-off, but
    worth stating explicitly if precision loss is ever questioned.
    """
    # Convert HWC NumPy arrays to CHW when they clearly contain channels.
    if image.ndim == 3 and image.shape[-1] <= 8 and image.shape[0] > 8:
        image = np.moveaxis(image, -1, 0)

    if image.ndim == 3:
        channels = []
        for channel in image:
            channel = normalize_percentile(
                channel,
                config.percentile_clip[0],
                config.percentile_clip[1],
            )
            channels.append(channel)
        image = np.stack(channels, axis=0)
    elif image.ndim == 2:
        image = normalize_percentile(
            image,
            config.percentile_clip[0],
            config.percentile_clip[1],
        )
    else:
        raise ValueError(f"Expected a 2D image or 3D channel array, got {image.shape}")

    # CLAHE and median filtering are applied channel-wise.
    if config.median_filter or config.clahe:
        is_chw = image.ndim == 3
        channels = image if is_chw else image[None, ...]
        processed = []

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if config.clahe else None

        for channel in channels:
            channel_u8 = np.clip(channel * 255.0, 0, 255).astype(np.uint8)

            if config.median_filter:
                channel_u8 = cv2.medianBlur(channel_u8, 3)
            if clahe is not None:
                channel_u8 = clahe.apply(channel_u8)

            processed.append(channel_u8.astype(np.float32) / 255.0)

        image = np.stack(processed, axis=0)
        if not is_chw:
            image = image[0]

    return image.astype(np.float32)


def load_binary_mask(path: Path) -> np.ndarray:
    mask = load_array(path)

    if mask.ndim == 3:
        # A mask should be 2D. If a singleton/channel dimension exists, collapse it.
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = np.max(mask, axis=0) if mask.shape[0] <= 8 else np.max(mask, axis=-1)

    return (mask > 0).astype(np.float32)


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size

    if image.ndim == 2:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    # CHW
    channels = [
        cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
        for channel in image
    ]
    return np.stack(channels, axis=0)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def apply_joint_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    config: DatasetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same random spatial transform to image and mask."""
    if random.random() < config.horizontal_flip_probability:
        if image.ndim == 2:
            image = np.ascontiguousarray(np.fliplr(image))
        else:
            image = np.ascontiguousarray(image[:, :, ::-1])
        mask = np.ascontiguousarray(np.fliplr(mask))

    height, width = mask.shape
    angle = random.uniform(-config.rotation_degrees, config.rotation_degrees)
    max_dx = config.translation_fraction * width
    max_dy = config.translation_fraction * height
    tx = random.uniform(-max_dx, max_dx)
    ty = random.uniform(-max_dy, max_dy)

    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    def warp(channel: np.ndarray, interpolation: int) -> np.ndarray:
        return cv2.warpAffine(
            channel,
            matrix,
            (width, height),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    if image.ndim == 2:
        image = warp(image, cv2.INTER_LINEAR)
    else:
        image = np.stack([warp(channel, cv2.INTER_LINEAR) for channel in image], axis=0)

    mask = warp(mask, cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.float32)

    return image, mask


class BreastSegmentationDataset(Dataset):
    def __init__(
        self,
        config: DatasetConfig,
        split: str | None = None,
        augment: bool = False,
        samples: list[tuple[Path, Path]] | None = None,
    ) -> None:
        """
        Two ways to build this dataset:
          - split-based (original behaviour): pass `split` ("train"/"val"/
            "test") and samples are read from config.root/<split>/{images,masks}.
          - explicit `samples` list: used by the cross-validation code in
            tune_bce_dice_weight(), which pools train+val samples and then
            slices them into folds. `samples` bypasses the on-disk split
            lookup entirely, so the same pooled list can back both an
            augmented view (train side of a fold) and a plain view (val
            side) via torch.utils.data.Subset.
        """
        super().__init__()
        self.config = config
        self.split = split
        self.augment = augment

        if samples is not None:
            self.images_dir = None
            self.masks_dir = None
            self.samples = samples
        else:
            if split is None:
                raise ValueError("BreastSegmentationDataset needs either `split` or `samples`.")

            split_dir = config.root / split
            self.images_dir = split_dir / "images"
            self.masks_dir = split_dir / "masks"

            if not self.images_dir.exists() or not self.masks_dir.exists():
                raise FileNotFoundError(
                    f"Missing {config.name} split: {split_dir}\n"
                    "Expected <split>/images and <split>/masks directories."
                )

            self.samples = paired_samples(self.images_dir, self.masks_dir)

        if not self.samples:
            where = f"split={split!r}" if samples is None else "the provided sample list"
            raise ValueError(f"No paired samples found for {config.name} ({where})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]

        image = preprocess_image(load_array(image_path), self.config)
        mask = load_binary_mask(mask_path)

        image = resize_image(image, self.config.input_size)
        mask = resize_mask(mask, self.config.input_size)

        if self.augment:
            image, mask = apply_joint_augmentation(image, mask, self.config)

        if image.ndim == 2:
            image = image[None, ...]

        if image.shape[0] != self.config.in_channels:
            raise ValueError(
                f"{image_path.name} has {image.shape[0]} channel(s), but "
                f"{self.config.name} config expects {self.config.in_channels}."
            )

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask[None, ...])).float()

        return image_tensor, mask_tensor


def create_loaders(config: DatasetConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = BreastSegmentationDataset(config, "train", augment=True)
    val_dataset = BreastSegmentationDataset(config, "val", augment=False)
    test_dataset = BreastSegmentationDataset(config, "test", augment=False)

    common = {
        "batch_size": config.batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)

    print(
        f"{config.name}: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


def pooled_trainval_samples(config: DatasetConfig) -> list[tuple[Path, Path]]:
    """
    Combine the `train` and `val` split samples into one pool for
    cross-validated loss-weight tuning (see tune_bce_dice_weight). The
    `test` split is deliberately excluded here and stays held out for the
    final, once-only evaluation in train_one_dataset -- CV folds are only
    ever carved out of data that was already earmarked for
    training/tuning under the methodology's train/val/test split.
    """
    samples: list[tuple[Path, Path]] = []
    for split in ("train", "val"):
        split_dir = config.root / split
        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"

        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(
                f"Missing {config.name} split: {split_dir}\n"
                "Expected <split>/images and <split>/masks directories."
            )

        samples.extend(paired_samples(images_dir, masks_dir))

    if not samples:
        raise ValueError(f"No paired train+val samples found for {config.name}.")

    return samples


# -----------------------------------------------------------------------------
# ResNet34 encoder
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# U-Net++ decoder
# -----------------------------------------------------------------------------

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


def upsample_to(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        x,
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


# -----------------------------------------------------------------------------
# CBAM (Woo et al., 2018) -- applied on the U-Net++ skip connections
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
    """Convolutional Block Attention Module (Woo et al., 2018).

    Applied to each encoder feature map (x0_0 .. x4_0) at the point it
    branches into a U-Net++ skip connection -- i.e. immediately before that
    feature is fused (concatenated) with the upsampled decoder feature
    below it. Channel attention runs first, then spatial attention.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


class ResNetUNetPlusPlus(nn.Module):
    """
    U-Net++ with a ResNet34 encoder and CBAM on the shortest skip
    connections.

    The nested dense skip connections are x(i,j), where j > 0 represents
    increasingly refined decoder nodes; node x(i,j) is fused from every
    earlier same-row node x(i,0..j-1) plus an upsampled feature from the
    row below. CBAM is applied only to the *shortest* hop into each node
    -- i.e. its immediate same-row predecessor, x(i,j-1) -- right before
    that concatenation (X0,0->X0,1, then X0,1->X0,2, X0,2->X0,3, and
    X0,3->X0,4, and equivalently for every other row). The other, older
    same-row contributions and the upsampled input are left unrefined.
    With deep supervision enabled, the model returns four full-resolution
    logits maps during training/evaluation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        pretrained_encoder: bool = False,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = ResNet34Encoder(in_channels, pretrained_encoder)

        # Decoder node widths per row. Keeping these close to the encoder widths
        # makes the architecture easy to inspect and modify later (e.g., CBAM).
        c0, c1, c2, c3, c4 = self.encoder.out_channels
        d0, d1, d2, d3 = 64, 64, 128, 256

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
        # upsample input are left unrefined -- see forward() and the
        # class docstring. Channel counts match because d0..d3 are set
        # equal to c0..c3 above.
        self.cbam0_1 = CBAM(d0)
        self.cbam1_1 = CBAM(d1)
        self.cbam2_1 = CBAM(d2)
        self.cbam3_1 = CBAM(d3)
        self.cbam0_2 = CBAM(d0)
        self.cbam1_2 = CBAM(d1)
        self.cbam2_2 = CBAM(d2)
        self.cbam0_3 = CBAM(d0)
        self.cbam1_3 = CBAM(d1)
        self.cbam0_4 = CBAM(d0)

        # Binary lesion segmentation -> one output channel.
        self.out1 = nn.Conv2d(d0, 1, kernel_size=1)
        self.out2 = nn.Conv2d(d0, 1, kernel_size=1)
        self.out3 = nn.Conv2d(d0, 1, kernel_size=1)
        self.out4 = nn.Conv2d(d0, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        input_size = x.shape[-2:]
        x0_0, x1_0, x2_0, x3_0, x4_0 = self.encoder(x)

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


def build_model(config: DatasetConfig) -> ResNetUNetPlusPlus:
    """Shared model constructor so tuning and final training stay in sync."""
    return ResNetUNetPlusPlus(
        in_channels=config.in_channels,
        pretrained_encoder=config.pretrained_encoder,
        deep_supervision=config.deep_supervision,
    ).to(DEVICE)


# -----------------------------------------------------------------------------
# Hybrid BCE + Dice loss
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Binary segmentation metrics
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Cross-validation utilities for Bayesian loss-weight tuning
# -----------------------------------------------------------------------------

_PATIENT_ID_PATTERN = re.compile(
    r"(?P<patient>P[_-]?\d{3,}|RIDER[\s_-]?[A-Za-z0-9]+[_-]?\d+|\d{3,})",
    re.IGNORECASE,
)


def default_patient_id_from_filename(image_path: Path) -> str:
    """
    Best-effort patient/case ID guess from a filename -- a starting point
    for grouped (patient-safe) CV folds, NOT a validated parser for CBIS-
    DDSM's or your exported RIDER files' actual naming convention.

    Intended to handle filenames along the lines of:
      "Mass-Training_P_00016_LEFT_CC_1.png" -> "P_00016"
      "RIDER-1023_slice014.png"             -> "RIDER-1023"
      "1023_042.png"                        -> "1023"

    VERIFY this against your real filenames (e.g. via --show-cv-groups)
    before trusting grouped folds. If it can't find a match it falls back
    to the full filename stem, which is equivalent to no grouping at all
    for that one file -- silently defeating patient-level separation for
    it, so a mismatch here won't necessarily raise an error.
    """
    match = _PATIENT_ID_PATTERN.search(image_path.stem)
    return match.group("patient") if match else image_path.stem


def make_group_folds(
    n_samples: int,
    n_folds: int,
    seed: int,
    groups: Sequence[str] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Partition `n_samples` sample indices into up to `n_folds` (train_idx,
    val_idx) pairs for cross-validation -- a hand-rolled equivalent of
    sklearn.model_selection.GroupKFold, written here to avoid adding
    scikit-learn as a dependency just for this.

    If `groups` is given (e.g. one patient ID per sample), every sample
    sharing the same group value is guaranteed to land in the same fold --
    this is what keeps a patient's slices from being split across the
    train/val sides of a fold. Fold sizes are balanced by total SAMPLE
    count (via a greedy fill-the-smallest-fold assignment), not just group
    count, since group sizes (slices per patient) can differ a lot. If
    `groups` is None, each sample is its own group (ordinary per-sample
    K-fold) -- fine for i.i.d. data, but not patient-safe for RIDER.

    Uses its own seeded numpy Generator, independent of the global
    torch/numpy/random seed set by set_seed(), so fold assignment is fixed
    once per tuning run regardless of how often set_seed() is called later
    for model re-initialization.

    Returns fewer than `n_folds` folds if fewer than `n_folds` unique
    groups exist (e.g. RIDER's ~4 non-test patients) -- this is how
    leave-one-patient-out CV naturally falls out of a generic "K-fold"
    request once grouping is enabled.
    """
    if groups is not None and len(groups) != n_samples:
        raise ValueError(
            f"groups has length {len(groups)}, expected {n_samples} (one per sample)."
        )

    rng = np.random.default_rng(seed)
    group_ids = np.arange(n_samples) if groups is None else np.asarray(groups)

    unique_groups = np.unique(group_ids)
    rng.shuffle(unique_groups)

    effective_folds = max(1, min(n_folds, len(unique_groups)))

    fold_sample_indices: list[list[int]] = [[] for _ in range(effective_folds)]
    fold_sizes = [0] * effective_folds

    for group in unique_groups:
        member_indices = np.where(group_ids == group)[0]
        target_fold = int(np.argmin(fold_sizes))
        fold_sample_indices[target_fold].extend(member_indices.tolist())
        fold_sizes[target_fold] += len(member_indices)

    all_indices = np.arange(n_samples)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in fold_sample_indices:
        val_idx = np.array(sorted(fold))
        train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=True)
        folds.append((train_idx, val_idx))

    return folds


# -----------------------------------------------------------------------------
# Bayesian optimization for the BCE/Dice loss balance
# -----------------------------------------------------------------------------

def tune_bce_dice_weight(
    config: DatasetConfig,
    n_trials: int = 15,
    tuning_epochs: int = 5,
    n_folds: int = 5,
    timeout: int | None = None,
) -> tuple[float, float]:
    """
    Bayesian-optimize the hybrid loss balance: bce_weight = alpha,
    dice_weight = 1 - alpha, alpha in [0, 1].

    Uses Optuna's TPE sampler (a sequential model-based / Bayesian optimizer)
    to pick the next alpha to try based on all previous trials' results,
    rather than a blind grid or random search.

    Each trial now cross-validates over `n_folds` folds carved out of the
    pooled train+val samples (see pooled_trainval_samples), instead of a
    single fixed train/val split: for every fold, a fresh model trains for
    `tuning_epochs` epochs on that fold's train side and is scored on its
    val side, and the trial's objective is the MEAN validation Dice across
    folds. Folds are grouped by config.patient_id_fn when set, so that (for
    example) all of one RIDER patient's slices stay together on one side
    of every fold -- see make_group_folds().

    See the module docstring for full caveats -- in particular, this
    search is still a cheap proxy (short training runs) and, unless
    config.patient_id_fn is set, folds are NOT guaranteed to respect
    patient boundaries.
    """
    if optuna is None:
        raise ImportError(
            "optuna is required for Bayesian optimization of the loss weights. "
            "Install it with: pip install optuna"
        )

    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 for cross-validation; got {n_folds}.")

    samples = pooled_trainval_samples(config)
    n_samples = len(samples)

    groups: list[str] | None = None
    if config.patient_id_fn is not None:
        groups = [config.patient_id_fn(image_path) for image_path, _ in samples]
        print(
            f"[{config.name}] Grouping {n_samples} pooled train+val samples "
            f"into {len(set(groups))} group(s) via {config.patient_id_fn.__name__} "
            "for cross-validation."
        )
    elif config.name == "breast_mri":
        print(
            "WARNING: MRI_CONFIG.patient_id_fn is not set, so breast_mri CV "
            "folds will be built per SLICE, not per patient. Section 3.2 of "
            "the methodology requires RIDER splits at the patient level -- "
            "without grouping, slices from the same patient can end up on "
            "both sides of a fold, which will make validation Dice look "
            "better than it would on a truly unseen patient. Set "
            "MRI_CONFIG.patient_id_fn to a function mapping an image Path "
            "to a patient ID (see default_patient_id_from_filename for a "
            "starting point) and re-check with --show-cv-groups before "
            "trusting these results."
        )

    folds = make_group_folds(n_samples, n_folds, seed=SEED, groups=groups)
    effective_folds = len(folds)
    if effective_folds < n_folds:
        print(
            f"[{config.name}] Requested {n_folds} folds but only "
            f"{effective_folds} unique group(s) are available in the pooled "
            f"train+val data; using {effective_folds}-fold cross-validation."
        )

    if config.name == "breast_mri":
        print(
            "NOTE: RIDER has only 5 patients total (60/20/20 patient-level "
            "split per section 3.2 -> at most 4 patients ever land in the "
            "train+val pool used here, 1 is permanently held out for test). "
            "Even with correct patient grouping this caps CV at 4-fold "
            "(leave-one-patient-out) -- a real improvement over a single "
            "patient's validation score, but still a small-N estimate."
        )

    augmented_view = BreastSegmentationDataset(config, augment=True, samples=samples)
    plain_view = BreastSegmentationDataset(config, augment=False, samples=samples)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
    }

    def objective(trial: "optuna.Trial") -> float:
        alpha = trial.suggest_float("bce_weight", 0.0, 1.0)
        fold_dices: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # Reset the seed at the start of every fold (not just once per
            # trial) so model init is controlled the same way across
            # folds, isolating the effect of alpha and each fold's data.
            set_seed(SEED)
            model = build_model(config)

            base_loss = BCEDiceLoss(bce_weight=alpha, dice_weight=1.0 - alpha)
            criterion = DeepSupervisionLoss(base_loss)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            fold_train_loader = DataLoader(
                Subset(augmented_view, train_idx), shuffle=True, **loader_kwargs
            )
            fold_val_loader = DataLoader(
                Subset(plain_view, val_idx), shuffle=False, **loader_kwargs
            )

            fold_val_dice = 0.0
            for _epoch in range(1, tuning_epochs + 1):
                model.train()
                for images, masks in fold_train_loader:
                    images = images.to(DEVICE, non_blocking=True)
                    masks = masks.to(DEVICE, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()

                val_metrics = evaluate(model, fold_val_loader, criterion, DEVICE)
                fold_val_dice = val_metrics["dice"]

            fold_dices.append(fold_val_dice)

            # Report/prune between folds rather than between epochs -- a
            # trial can now only be pruned after completing at least
            # `n_warmup_steps + 1` full folds (see MedianPruner below).
            running_mean_dice = float(np.mean(fold_dices))
            trial.report(running_mean_dice, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_dices))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    total_epoch_equivalents = n_trials * effective_folds * tuning_epochs
    print(
        f"[{config.name}] Starting Bayesian optimization: {n_trials} trials x "
        f"{effective_folds} folds x {tuning_epochs} epochs = up to "
        f"{total_epoch_equivalents} epoch-equivalents before pruning "
        "(MedianPruner will cut many unpromising trials short)..."
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_alpha = study.best_params["bce_weight"]
    print(
        f"[{config.name}] Best alpha (bce_weight) found: {best_alpha:.4f} "
        f"(dice_weight={1.0 - best_alpha:.4f}) -> "
        f"mean_cv_val_dice={study.best_value:.4f} over {effective_folds} folds, "
        f"{len(study.trials)} trials"
    )
    return best_alpha, 1.0 - best_alpha


def preview_cv_groups(config: DatasetConfig, n_folds: int) -> None:
    """
    Print how pooled train+val samples would be grouped and split into CV
    folds, WITHOUT training anything -- a cheap sanity check for
    config.patient_id_fn (or the lack of one) before committing GPU time
    to --tune-loss-weights. Triggered by --show-cv-groups.
    """
    samples = pooled_trainval_samples(config)
    n_samples = len(samples)

    groups: list[str] | None = None
    if config.patient_id_fn is not None:
        groups = [config.patient_id_fn(image_path) for image_path, _ in samples]

    folds = make_group_folds(n_samples, n_folds, seed=SEED, groups=groups)

    print(f"\n[{config.name}] Pooled train+val samples: {n_samples}")
    if groups is not None:
        unique = sorted(set(groups))
        print(f"[{config.name}] {len(unique)} group(s) via {config.patient_id_fn.__name__}: {unique}")
    else:
        print(
            f"[{config.name}] No patient_id_fn set -- grouping is per-sample "
            "(ordinary K-fold, NOT patient-safe)."
        )

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if groups is not None:
            val_groups = sorted({groups[i] for i in val_idx})
            print(
                f"  Fold {fold_idx}: train={len(train_idx)} samples, "
                f"val={len(val_idx)} samples, val_groups={val_groups}"
            )
        else:
            print(
                f"  Fold {fold_idx}: train={len(train_idx)} samples, "
                f"val={len(val_idx)} samples"
            )


# -----------------------------------------------------------------------------
# Training and checkpoints
# -----------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_dice: float,
    config: DatasetConfig,
    bce_weight: float,
    dice_weight: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": best_val_dice,
            "dataset_name": config.name,
            "input_size": config.input_size,
            "in_channels": config.in_channels,
            "encoder": "resnet34",
            "architecture": "unet++",
            "deep_supervision": config.deep_supervision,
            "bce_weight": bce_weight,
            "dice_weight": dice_weight,
        },
        path,
    )


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train_one_dataset(
    config: DatasetConfig,
    bce_weight: float | None = None,
    dice_weight: float | None = None,
) -> dict[str, float]:
    """
    Train a completely separate ResNet-U-Net++ model for one modality.

    bce_weight / dice_weight override the DatasetConfig defaults when given --
    this is how the Bayesian-optimized weights from tune_bce_dice_weight()
    get plugged into the full training run.
    """
    set_seed(SEED)

    resolved_bce_weight = config.bce_weight if bce_weight is None else bce_weight
    resolved_dice_weight = config.dice_weight if dice_weight is None else dice_weight

    train_loader, val_loader, test_loader = create_loaders(config)

    model = build_model(config)

    base_loss = BCEDiceLoss(resolved_bce_weight, resolved_dice_weight)
    criterion = DeepSupervisionLoss(base_loss)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    checkpoint_path = CHECKPOINT_DIR / f"{config.name}_resnet34_unetpp_best.pth"
    history_path = RESULTS_DIR / f"{config.name}_training_history.csv"

    best_val_dice = -1.0
    history: list[dict[str, float]] = []

    print("=" * 80)
    print(f"Training: {config.name} | ResNet34 encoder + U-Net++")
    print(f"Device: {DEVICE}")
    print(f"Input size: {config.input_size} | batch size: {config.batch_size}")
    print(
        f"Loss weights -> bce_weight={resolved_bce_weight:.4f}, "
        f"dice_weight={resolved_dice_weight:.4f}"
    )
    print("=" * 80)

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_metrics["dice"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        write_history(history_path, history)

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"dice={val_metrics['dice']:.4f} | "
            f"iou={val_metrics['iou']:.4f} | "
            f"precision={val_metrics['precision']:.4f} | "
            f"recall={val_metrics['recall']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                best_val_dice,
                config,
                resolved_bce_weight,
                resolved_dice_weight,
            )
            print(f"  Saved new best checkpoint: {checkpoint_path}")

    # Evaluate the best model on the held-out test split.
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)

    print("\nBest validation Dice:", checkpoint["best_val_dice"])
    print(f"Held-out test metrics for {config.name}:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    return test_metrics


# -----------------------------------------------------------------------------
# Architecture diagram (optional; pip install matplotlib)
# -----------------------------------------------------------------------------

_DIAG_ENCODER_COLOR = "#E0785A"
_DIAG_ENCODER_SHADOW = "#B85B41"
_DIAG_DECODER_COLOR = "#8C7FC9"
_DIAG_DECODER_SHADOW = "#6C5FA3"
_DIAG_IO_COLOR = "#9A9A9A"
_DIAG_DOWN_COLOR = "#1A1A1A"
_DIAG_UP_COLOR = "#D6431F"
_DIAG_SKIP_COLOR = "#D9A62A"
_DIAG_CBAM_BADGE = "#F2B705"


def _diagram_node_xy(i: int, j: int, col_w: float, row_h: float) -> tuple[float, float]:
    """Column = i + j, row = i -- matches the U-Net++ nested grid layout."""
    return (i + j) * col_w, -i * row_h


def save_architecture_diagram(output_path: Path, depth: int = 5) -> None:
    """
    Render a schematic PNG of the model architecture (ResNet-34 encoder +
    U-Net++ nested decoder + CBAM on the skip connections), styled after
    the WBC-Net figure: nested dense skip connections drawn as arced
    dashed lines (arc height grows with span so longer arcs don't cross
    shorter ones), a diagonal encoder backbone (solid, downsampling), and
    vertical upsampling arrows one row at a time.

    This is a static schematic, not a traced forward pass -- it doesn't
    need a model instance, dataset, checkpoint, or GPU, and the diagram
    is identical for every DatasetConfig (input size/channels don't
    change the topology). `depth` is 5 for a standard ResNet backbone
    (stem + 4 stages); it would only change if you swapped in a
    differently-staged encoder.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to render the architecture diagram. "
            "Install it with: pip install matplotlib"
        ) from exc

    col_w, row_h = 1.9, 2.35
    node_w, node_h = 0.95, 0.62
    shadow_dx, shadow_dy = 0.10, 0.10

    def draw_block(ax, cx, cy, color, shadow, label=None):
        ax.add_patch(Rectangle(
            (cx - node_w / 2 + shadow_dx, cy - node_h / 2 - shadow_dy),
            node_w, node_h, facecolor=shadow, edgecolor="none", zorder=2,
        ))
        ax.add_patch(Rectangle(
            (cx - node_w / 2, cy - node_h / 2),
            node_w, node_h, facecolor=color, edgecolor="#2b2b2b", linewidth=0.8, zorder=3,
        ))
        if label:
            ax.text(cx, cy, label, ha="center", va="center", fontsize=7.2,
                    color="white", zorder=4, fontweight="bold")

    def draw_arrow(ax, p1, p2, color, style="-", lw=1.4, z=1):
        ax.add_patch(FancyArrowPatch(
            p1, p2, connectionstyle="arc3,rad=0", arrowstyle="-|>", mutation_scale=10,
            linewidth=lw, linestyle=style, color=color, zorder=z, shrinkA=6, shrinkB=6,
        ))

    def draw_skip_arc(ax, x1, x2, y, span):
        # span == 1 (adjacent, same-row) is the only hop CBAM is actually
        # applied to -- draw it bolder/darker gold so it reads as distinct
        # from the longer, unrefined dense-skip arcs.
        shortest = span == 1
        rad = -(0.22 + 0.11 * span)
        color = _DIAG_CBAM_BADGE if shortest else _DIAG_SKIP_COLOR
        lw = 1.9 if shortest else 1.0
        ax.add_patch(FancyArrowPatch(
            (x1 + node_w / 2, y + node_h / 2 - 0.02),
            (x2 - node_w / 2, y + node_h / 2 - 0.02),
            connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=8,
            linewidth=lw, linestyle=(0, (4, 3)), color=color,
            zorder=(2 if shortest else 1), shrinkA=2, shrinkB=2,
        ))

    fig, ax = plt.subplots(figsize=(15, 12))
    ax.set_aspect("equal")
    ax.axis("off")

    positions: dict[tuple[int, int], tuple[float, float]] = {}
    for i in range(depth):
        for j in range(depth - i):
            x, y = _diagram_node_xy(i, j, col_w, row_h)
            positions[(i, j)] = (x, y)
            if j == 0:
                draw_block(ax, x, y, _DIAG_ENCODER_COLOR, _DIAG_ENCODER_SHADOW,
                           label=f"X{i},0")
            else:
                draw_block(ax, x, y, _DIAG_DECODER_COLOR, _DIAG_DECODER_SHADOW,
                           label=f"X{i},{j}")

    # Encoder backbone (diagonal, solid black, downsampling)
    for i in range(depth - 1):
        x1, y1 = positions[(i, 0)]
        x2, y2 = positions[(i + 1, 0)]
        draw_arrow(ax, (x1, y1 - node_h / 2), (x2, y2 + node_h / 2), _DIAG_DOWN_COLOR, lw=1.8, z=2)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.32, my, f"ResNet\nstage {i + 1}", fontsize=6.3, color="#333333",
                ha="left", va="center", style="italic")

    # Upsampling (vertical, solid red), same column, one row up
    for i in range(1, depth):
        for j in range(depth - i):
            below = positions[(i, j)]
            above = positions[(i - 1, j + 1)]
            draw_arrow(ax, (below[0], below[1] + node_h / 2),
                       (above[0], above[1] - node_h / 2), _DIAG_UP_COLOR, lw=1.3, z=2)

    # Dense skip connections (dashed arcs, within each row). Draw the
    # longer (unrefined) spans first, then the shortest/CBAM ones last so
    # they sit on top where arcs overlap.
    for i in range(depth):
        cols = list(range(depth - i))
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                if b - a != 1:
                    x1, y1 = positions[(i, a)]
                    x2, y2 = positions[(i, b)]
                    draw_skip_arc(ax, x1, x2, y1, span=(b - a))
        for a in range(len(cols) - 1):
            x1, y1 = positions[(i, a)]
            x2, y2 = positions[(i, a + 1)]
            draw_skip_arc(ax, x1, x2, y1, span=1)

    # Input / Output framing
    x0, y0 = positions[(0, 0)]
    xN, yN = positions[(0, depth - 1)]
    in_x, in_y = x0 - col_w, y0
    out_x, out_y = xN + col_w, yN

    for cx, cy, txt in [(in_x, in_y, "Input\nImage"), (out_x, out_y, "Output\nImage")]:
        ax.add_patch(FancyBboxPatch((cx - 0.55, cy - 0.4), 1.1, 0.8,
                                     boxstyle="round,pad=0.02,rounding_size=0.06",
                                     facecolor=_DIAG_IO_COLOR, edgecolor="#333333",
                                     linewidth=0.8, zorder=3))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=8, color="white",
                fontweight="bold", zorder=4)

    draw_arrow(ax, (in_x + 0.55, in_y), (x0 - node_w / 2, y0), "#333333", lw=1.4, z=2)
    draw_arrow(ax, (xN + node_w / 2, yN), (out_x - 0.55, out_y), "#333333", lw=1.4, z=2)

    top_y = y0 + 2.6
    draw_arrow(ax, (in_x, in_y + 0.4), (in_x, top_y), "#333333", lw=1.0, z=2)
    draw_arrow(ax, (in_x, top_y), (out_x, top_y), "#333333", lw=1.0, z=2)
    draw_arrow(ax, (out_x, top_y), (out_x, out_y + 0.4), "#333333", lw=1.0, z=2)

    # Legend
    legend_x = in_x - 0.2
    legend_y = -((depth - 1) * row_h) - 1.3
    items = [
        (_DIAG_ENCODER_COLOR, "Encoder feature (ResNet-34 stage output)"),
        (_DIAG_DECODER_COLOR, "Nested decoder conv block (U-Net++)"),
    ]
    for k, (color, text) in enumerate(items):
        ly = legend_y - k * 0.5
        ax.add_patch(Rectangle((legend_x, ly - 0.15), 0.4, 0.3, facecolor=color,
                                edgecolor="#2b2b2b", linewidth=0.7))
        ax.text(legend_x + 0.55, ly, text, fontsize=8, va="center")

    line_specs = [
        (_DIAG_DOWN_COLOR, "-", "Downsampling (ResNet-34 encoder)"),
        (_DIAG_UP_COLOR, "-", "Upsampling"),
        (_DIAG_CBAM_BADGE, (0, (4, 3)), "Shortest skip connection (CBAM applied)"),
        (_DIAG_SKIP_COLOR, (0, (4, 3)), "Longer dense skip connection (no CBAM)"),
    ]
    for k, (color, style, text) in enumerate(line_specs):
        ly = legend_y - (2 + k) * 0.5
        ax.plot([legend_x, legend_x + 0.4], [ly, ly], color=color, linewidth=1.8, linestyle=style)
        ax.text(legend_x + 0.55, ly, text, fontsize=8, va="center")

    min_x, max_x = in_x - 0.8, out_x + 0.8
    min_y = legend_y - 6 * 0.5 - 0.3
    max_y = top_y + 0.5
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved architecture diagram to {output_path}")


# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train separate ResNet34 U-Net++ models for mammograms and breast MRI slices."
    )
    parser.add_argument(
        "--dataset",
        choices=("mammogram", "mri", "both"),
        default="both",
        help="Which modality to train. Default: both (sequentially).",
    )
    parser.add_argument(
        "--tune-loss-weights",
        action="store_true",
        help=(
            "Run Bayesian optimization (Optuna TPE) over the BCE/Dice loss "
            "balance before training each selected dataset, cross-validated "
            "over --cv-folds folds of the pooled train+val data (see "
            "--cv-folds). Requires 'pip install optuna'. See the module "
            "docstring for caveats, especially regarding patient grouping "
            "for the small RIDER MRI split."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=15,
        help="Number of Optuna trials when --tune-loss-weights is set (default: 15).",
    )
    parser.add_argument(
        "--tuning-epochs",
        type=int,
        default=5,
        help=(
            "Epochs trained per trial (per fold) during loss-weight tuning "
            "(default: 5). Kept short on purpose as a cheap proxy -- see "
            "module docstring."
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help=(
            "Number of cross-validation folds used when --tune-loss-weights "
            "is set (default: 5). Each trial's objective is the mean "
            "validation Dice across this many folds of the pooled train+val "
            "data, instead of one fixed split. Automatically reduced if "
            "fewer unique groups exist (e.g. RIDER's ~4 non-test patients "
            "when DatasetConfig.patient_id_fn is set)."
        ),
    )
    parser.add_argument(
        "--show-cv-groups",
        action="store_true",
        help=(
            "Print the pooled train+val sample counts and CV fold/group "
            "assignment for the selected dataset(s) and exit -- no training "
            "or tuning. Use this to sanity-check patient grouping (or the "
            "lack of it) before running --tune-loss-weights."
        ),
    )
    parser.add_argument(
        "--diagram",
        action="store_true",
        help=(
            "Save a schematic architecture diagram (PNG) instead of "
            "training -- a static topology drawing, not a traced forward "
            "pass, so it's the same for every dataset. Requires "
            "'pip install matplotlib'."
        ),
    )
    return parser.parse_args()


def run_dataset(config: DatasetConfig, args: argparse.Namespace) -> None:
    bce_weight = dice_weight = None
    if args.tune_loss_weights:
        bce_weight, dice_weight = tune_bce_dice_weight(
            config,
            n_trials=args.n_trials,
            tuning_epochs=args.tuning_epochs,
            n_folds=args.cv_folds,
        )
    train_one_dataset(config, bce_weight=bce_weight, dice_weight=dice_weight)


def main() -> None:
    args = parse_args()

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if args.diagram:
        save_architecture_diagram(RESULTS_DIR / "architecture_diagram.png")
        return

    if args.show_cv_groups:
        if args.dataset in ("mammogram", "both"):
            preview_cv_groups(MAMMOGRAM_CONFIG, args.cv_folds)
        if args.dataset in ("mri", "both"):
            preview_cv_groups(MRI_CONFIG, args.cv_folds)
        return

    if args.dataset in ("mammogram", "both"):
        run_dataset(MAMMOGRAM_CONFIG, args)

    if args.dataset in ("mri", "both"):
        run_dataset(MRI_CONFIG, args)


if __name__ == "__main__":
    main()