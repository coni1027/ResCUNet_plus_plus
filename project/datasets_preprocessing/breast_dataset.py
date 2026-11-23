"""
Data loading, preprocessing, augmentation, and the PyTorch Dataset for
breast lesion segmentation (mammograms + breast MRI slices).

Expected on-disk layout (per DatasetConfig.root, i.e. data/mammograms or
data/breast_mri):
    <root>/train/images/, <root>/train/masks/
    <root>/val/images/,   <root>/val/masks/
    <root>/test/images/,  <root>/test/masks/

Supported image formats: PNG/JPG/TIFF/BMP, .npy, and DICOM (.dcm).
Masks are treated as binary: background=0, lesion=1 (or any value > 0).

Important: create patient-level train/val/test splits BEFORE populating
these folders, especially for MRI slices, so slices from the same patient
do not leak across splits (see Methodology section 3.2).
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import pydicom
except ImportError:  # DICOM is optional unless .dcm files are used.
    pydicom = None

from config import DatasetConfig, DEVICE, IMAGE_EXTENSIONS, NUM_WORKERS
from pathlib import Path


# -----------------------------------------------------------------------------
# File discovery / pairing
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


# -----------------------------------------------------------------------------
# Image / mask loading
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Dataset / loaders
# -----------------------------------------------------------------------------

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
          - split-based (standard use): pass `split` ("train"/"val"/"test")
            and samples are read from config.root/<split>/{images,masks}.
          - explicit `samples` list: used by cross_validation/folds.py's
            pooled_trainval_samples(), which pools train+val samples for
            the k-fold CV robustness check (experiments/run_kfold_cv.py),
            run AFTER loss-weight tuning, not during it. `samples` bypasses
            the on-disk split lookup entirely, so the same pooled list can
            back both an augmented view (train side of a fold) and a plain
            view (val side) via torch.utils.data.Subset.
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
