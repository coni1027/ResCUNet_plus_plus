"""
Image preprocessing (Methodology 3.3): percentile normalization -> mean
filter -> CLAHE, then resizing. Masks are never filtered, only resized.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import DatasetConfig


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

    NOTE: when clahe/mean_filter is enabled, the [0,1] float image is
    quantized to uint8 before CLAHE/mean filtering (OpenCV's CLAHE needs
    8-bit or 16-bit input), then converted back to float. For 16-bit CBIS-DDSM
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

    # Mean filtering and CLAHE are applied channel-wise.
    if config.mean_filter or config.clahe:
        is_chw = image.ndim == 3
        channels = image if is_chw else image[None, ...]
        processed = []

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if config.clahe else None

        for channel in channels:
            channel_u8 = np.clip(channel * 255.0, 0, 255).astype(np.uint8)

            if config.mean_filter:
                channel_u8 = cv2.blur(channel_u8, (3, 3))
            if clahe is not None:
                channel_u8 = clahe.apply(channel_u8)

            processed.append(channel_u8.astype(np.float32) / 255.0)

        image = np.stack(processed, axis=0)
        if not is_chw:
            image = image[0]

    return image.astype(np.float32)


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
