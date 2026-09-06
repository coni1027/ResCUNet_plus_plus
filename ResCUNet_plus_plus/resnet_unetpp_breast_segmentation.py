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

Optional: Bayesian optimization for the hybrid loss weighting
---------------------------------------------------------------
Run with --tune-loss-weights to search for the balance between the BCE and
Dice terms of the hybrid loss (bce_weight vs. dice_weight, expressed as a
single alpha in [0, 1] with dice_weight = 1 - alpha) using Optuna's Bayesian
optimizer (TPE sampler, the same family of method as scikit-optimize/GPyOpt).
Install with: pip install optuna

Please read these caveats before trusting the result of that search:

1. Each trial retrains a FRESH model for a reduced number of epochs
   (--tuning-epochs, default 5) instead of the full schedule, as a
   compute-saving proxy. The alpha that looks best after 5 epochs is not
   guaranteed to still be best after the full 30-epoch run -- treat the
   tuned value as a good starting point, not a certified optimum, and
   re-validate with a full training run.
2. RIDER breast MRI has only 5 patients total, split 60/20/20 at the
   patient level (3 train / 1 val / 1 test patients per section 3.2 of the
   methodology). That means the validation Dice used as the optimization
   objective for this modality comes from ONE patient's slices per trial.
   A single-patient validation score is extremely high-variance and can
   easily reward an alpha that happens to suit that one patient's lesion
   size/contrast rather than the modality in general. Treat any MRI-side
   "optimal" alpha as a rough estimate. If this matters for your thesis
   results, prefer leave-one-patient-out cross-validation (loop the search
   over which of the 5 patients is held out as "val") over a single fixed
   split -- this script does not implement that for you.
3. The same random seed re-initializes the model in every trial, so the
   search isolates the effect of alpha rather than random-init noise. That
   also means the result has only been checked for one seed; consider
   re-running with 2-3 different seeds before trusting a close call,
   especially on the small MRI split.
4. This tunes only the BCE/Dice balance. It reuses whatever batch size,
   learning rate, augmentation, and input resolution are already hard-coded
   in each DatasetConfig -- those are not part of the search space here.
5. Tuning and final-model checkpoint selection both read from the same
   validation split. That's standard practice, but it means validation
   performance is now "used twice" (once to pick alpha, once to pick the
   best epoch) -- for a paper-grade generalization estimate, only the
   held-out test metrics printed at the end should be reported.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
    pretrained_encoder: bool = True
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


# Mammograms retain more spatial detail, so this default uses 512x512.
# Reduce to (256, 256) if GPU memory is limited.
MAMMOGRAM_CONFIG = DatasetConfig(
    name="mammogram",
    root=MAMMOGRAM_ROOT,
    input_size=(512, 512),
    in_channels=1,
    batch_size=4,
    epochs=30,
    learning_rate=1e-4,
    pretrained_encoder=True,
    deep_supervision=True,
    clahe=True,
    median_filter=True,
    horizontal_flip_probability=0.5,
    rotation_degrees=10.0,
    translation_fraction=0.03,
)

# MRI slices are trained in their own experiment. These defaults are intentionally
# independent from the mammogram settings.
# NOTE: RIDER provides only 5 patients (see module docstring, point 2). Any
# hyperparameter search run against MRI_CONFIG's validation split should be
# read with that in mind.
MRI_CONFIG = DatasetConfig(
    name="breast_mri",
    root=MRI_ROOT,
    input_size=(256, 256),
    in_channels=1,
    batch_size=8,
    epochs=30,
    learning_rate=1e-4,
    pretrained_encoder=True,
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
        split: str,
        augment: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.augment = augment

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
            raise ValueError(f"No paired samples found in {split_dir}")

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


# -----------------------------------------------------------------------------
# ResNet34 encoder
# -----------------------------------------------------------------------------

class ResNet34Encoder(nn.Module):
    """ResNet34 feature extractor returning five spatial scales."""

    def __init__(self, in_channels: int = 1, pretrained: bool = True) -> None:
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


class ResNetUNetPlusPlus(nn.Module):
    """
    U-Net++ with a ResNet34 encoder.

    The nested dense skip connections are x(i,j), where j > 0 represents
    increasingly refined decoder nodes. With deep supervision enabled, the
    model returns four full-resolution logits maps during training/evaluation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        pretrained_encoder: bool = True,
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

        # Binary lesion segmentation -> one output channel.
        self.out1 = nn.Conv2d(d0, 1, kernel_size=1)
        self.out2 = nn.Conv2d(d0, 1, kernel_size=1)
        self.out3 = nn.Conv2d(d0, 1, kernel_size=1)
        self.out4 = nn.Conv2d(d0, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        input_size = x.shape[-2:]
        x0_0, x1_0, x2_0, x3_0, x4_0 = self.encoder(x)

        x0_1 = self.conv0_1(torch.cat([x0_0, upsample_to(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, upsample_to(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, upsample_to(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, upsample_to(x4_0, x3_0)], dim=1))

        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, upsample_to(x1_1, x0_0)], dim=1)
        )
        x1_2 = self.conv1_2(
            torch.cat([x1_0, x1_1, upsample_to(x2_1, x1_0)], dim=1)
        )
        x2_2 = self.conv2_2(
            torch.cat([x2_0, x2_1, upsample_to(x3_1, x2_0)], dim=1)
        )

        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, upsample_to(x1_2, x0_0)], dim=1)
        )
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, upsample_to(x2_2, x1_0)], dim=1)
        )

        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, upsample_to(x1_3, x0_0)], dim=1)
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
# Bayesian optimization for the BCE/Dice loss balance
# -----------------------------------------------------------------------------

def tune_bce_dice_weight(
    config: DatasetConfig,
    n_trials: int = 15,
    tuning_epochs: int = 5,
    timeout: int | None = None,
) -> tuple[float, float]:
    """
    Bayesian-optimize the hybrid loss balance: bce_weight = alpha,
    dice_weight = 1 - alpha, alpha in [0, 1].

    Uses Optuna's TPE sampler (a sequential model-based / Bayesian optimizer)
    to pick the next alpha to try based on all previous trials' results,
    rather than a blind grid or random search. Each trial trains a fresh
    model for `tuning_epochs` epochs and reports the resulting validation
    Dice as the objective to maximize.

    See the module docstring for important caveats -- in particular, this
    search is a cheap proxy (short training runs) and, for the RIDER MRI
    config, is evaluated against a single patient's slices.
    """
    if optuna is None:
        raise ImportError(
            "optuna is required for Bayesian optimization of the loss weights. "
            "Install it with: pip install optuna"
        )

    if config.name == "breast_mri":
        print(
            "WARNING: tuning loss weights against the RIDER MRI validation "
            "split. Only 5 patients exist in total (3 train / 1 val / 1 test "
            "at the patient level), so this objective is computed from a "
            "single patient's slices per trial. Treat the result as a rough "
            "starting point -- consider leave-one-patient-out CV if you need "
            "a trustworthy optimum for the thesis."
        )

    train_loader, val_loader, _ = create_loaders(config)

    def objective(trial: "optuna.Trial") -> float:
        alpha = trial.suggest_float("bce_weight", 0.0, 1.0)

        # Fix the seed per trial so alpha's effect isn't confounded with a
        # different random initialization (see docstring caveat 3).
        set_seed(SEED)
        model = build_model(config)

        base_loss = BCEDiceLoss(bce_weight=alpha, dice_weight=1.0 - alpha)
        criterion = DeepSupervisionLoss(base_loss)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        val_dice = 0.0
        for epoch in range(1, tuning_epochs + 1):
            model.train()
            for images, masks in train_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            val_metrics = evaluate(model, val_loader, criterion, DEVICE)
            val_dice = val_metrics["dice"]

            trial.report(val_dice, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_dice

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    print(
        f"[{config.name}] Starting Bayesian optimization: {n_trials} trials x "
        f"{tuning_epochs} epochs each..."
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_alpha = study.best_params["bce_weight"]
    print(
        f"[{config.name}] Best alpha (bce_weight) found: {best_alpha:.4f} "
        f"(dice_weight={1.0 - best_alpha:.4f}) -> val_dice={study.best_value:.4f} "
        f"over {len(study.trials)} trials"
    )
    return best_alpha, 1.0 - best_alpha


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
            "balance before training each selected dataset. Requires "
            "'pip install optuna'. See the module docstring for caveats, "
            "especially regarding the small RIDER MRI split."
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
            "Epochs trained per trial during loss-weight tuning (default: 5). "
            "Kept short on purpose as a cheap proxy -- see module docstring."
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
        )
    train_one_dataset(config, bce_weight=bce_weight, dice_weight=dice_weight)


def main() -> None:
    args = parse_args()

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if args.dataset in ("mammogram", "both"):
        run_dataset(MAMMOGRAM_CONFIG, args)

    if args.dataset in ("mri", "both"):
        run_dataset(MRI_CONFIG, args)


if __name__ == "__main__":
    main()