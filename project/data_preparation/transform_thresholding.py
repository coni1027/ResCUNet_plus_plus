"""
PROPOSED (not yet in the pipeline): separate the breast from the background.

The breast is segmented by intensity thresholding + largest connected
component; everything else (background, scanner labels like "RCC") is set
to 0, and image + lesion mask are cropped to the breast's bounding box.
After preprocessing, the crop is resized with its aspect ratio preserved
(longest side -> target, rest padded with black) so narrow crops are not
stretched:

  load -> remove_background -> crop -> normalize -> mean filter -> CLAHE
       -> resize_with_padding -> .npy

Why not Otsu / edge detection: on 200 CBIS-DDSM images Otsu cut into the
lesion in 24 (it splits dense from fatty tissue, not breast from
background); edge detectors only give outlines that still need filling.
A low fixed threshold takes ~5 ms per image.

Threshold choice (200 random CBIS-DDSM images): background removed is
51-56% of the image for any threshold in 0.01-0.15 (labels/empty space go
via the largest-component step, not the threshold), while lesion loss
grows with the threshold. 0.02 is the highest value where no image lost
>=1% of its lesion (worst: 0.77%; at 0.03, 2 images lost >1%).

Known limit: thin bright scanner strips along the top/bottom edge (in
142/200 images) can join the breast region, so the crop box stays close to
the full frame for those images. The background is still blacked out.
Breaking the strips would need an edge band (~2.4%) that overlaps where
lesions sit (5% lie within 2.8% of an edge), so it is deliberately not done.

Demo (writes an original / preprocessed / thresholded / thresholded+crop figure):
    python data_preparation/transform_thresholding.py <image.dcm> <mask.dcm> [...more pairs]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.transforms import normalize_percentile

# Normalized intensity below which a pixel counts as background (see docstring for why 0.02).
BACKGROUND_THRESHOLD = 0.02
# The breast mask is computed on a 1/4-scale copy: same result, ~10x faster.
SCALE = 4


def remove_background(image: np.ndarray) -> tuple[np.ndarray, tuple[slice, slice]]:
    """
    Zero out everything except the breast (the largest bright region).
    Returns the same-shape image and the breast's bounding box as slices,
    so the caller crops image and lesion mask identically: image[box], mask[box].
    The box comes from the image only -- lesion masks don't exist at test time.
    """
    height, width = image.shape
    small = cv2.resize(image, (width // SCALE, height // SCALE), interpolation=cv2.INTER_AREA)
    foreground = (normalize_percentile(small, 1.0, 99.0) > BACKGROUND_THRESHOLD).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    if n_labels < 2:
        raise ValueError("No breast region found above the background threshold.")
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    breast = cv2.resize((labels == largest).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

    x, y, w, h = stats[largest, :4] * SCALE
    box = (slice(y, y + h), slice(x, x + w))
    return np.where(breast.astype(bool), image, 0).astype(np.float32), box


def resize_with_padding(array: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    """Scale so the longest side fits `size`, then pad with 0 (black) to exactly `size`, centred."""
    target_h, target_w = size
    height, width = array.shape
    scale = min(target_h / height, target_w / width)
    new_h, new_w = round(height * scale), round(width * scale)
    resized = cv2.resize(array, (new_w, new_h), interpolation=interpolation)
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    return cv2.copyMakeBorder(
        resized, top, target_h - new_h - top, left, target_w - new_w - left, cv2.BORDER_CONSTANT, value=0
    )


if __name__ == "__main__":
    import pydicom

    from config import MAMMOGRAM_CONFIG, RESULTS_DIR
    from data_preparation.transforms import preprocess_image, resize_image, resize_mask
    from datasets.breast_dataset import load_dicom

    size = (1024, 1024)
    panel_height = 420  # every panel keeps its true aspect ratio, so the 1024x1024 ones show as squares

    # Self-check: padding keeps the shape, fills with 0, and never stretches.
    tall = np.ones((400, 100), np.float32)
    padded = resize_with_padding(tall, (200, 200), cv2.INTER_NEAREST)
    assert padded.shape == (200, 200) and padded[:, :75].max() == 0 and padded[:, 75:125].min() == 1

    def panel(image: np.ndarray, mask: np.ndarray, title: str) -> np.ndarray:
        view = cv2.cvtColor((np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, contours, -1, (0, 0, 255), max(2, view.shape[0] // 200))
        width = round(view.shape[1] * panel_height / view.shape[0])
        view = cv2.resize(view, (width, panel_height), interpolation=cv2.INTER_AREA)
        view = cv2.copyMakeBorder(view, 24, 0, 0, 6, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        cv2.putText(view, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        return view

    paths = sys.argv[1:]
    if not paths or len(paths) % 2:
        sys.exit("usage: transform_thresholding.py <image.dcm> <mask.dcm> [<image.dcm> <mask.dcm> ...]")

    rows = []
    for image_path, mask_path in zip(paths[::2], paths[1::2]):
        image = load_dicom(Path(image_path))
        mask = (pydicom.dcmread(mask_path).pixel_array > 0).astype(np.float32)
        name = Path(image_path).stem

        plain = resize_image(preprocess_image(image, MAMMOGRAM_CONFIG), size)

        background_removed, box = remove_background(image)
        thresholded = resize_image(preprocess_image(background_removed, MAMMOGRAM_CONFIG), size)

        cropped_image, cropped_mask = background_removed[box], mask[box]
        cropped = resize_with_padding(preprocess_image(cropped_image, MAMMOGRAM_CONFIG), size, cv2.INTER_LINEAR)
        cropped_mask_out = resize_with_padding(cropped_mask, size, cv2.INTER_NEAREST)

        lesion_zeroed = ((background_removed == 0) & (image > 0) & (mask > 0)).sum() / mask.sum()
        assert lesion_zeroed < 0.01, f"{name}: background removal zeroed {lesion_zeroed:.1%} of the lesion"
        assert cropped_mask.sum() >= 0.99 * mask.sum(), f"{name}: crop cut off part of the lesion"

        crop_h, crop_w = cropped_image.shape
        rows.append(np.hstack([
            panel(normalize_percentile(image, 1.0, 99.0), mask, f"{name} original"),
            panel(plain, resize_mask(mask, size), "preprocessed -> 1024 (stretched)"),
            panel(thresholded, resize_mask(mask, size), "thresholded -> 1024 (stretched)"),
            panel(cropped, cropped_mask_out,
                  f"thresholded + crop ({crop_h / image.shape[0]:.0%}x{crop_w / image.shape[1]:.0%}) -> 1024 padded"),
        ]))

    widest = max(row.shape[1] for row in rows)
    rows = [cv2.copyMakeBorder(row, 0, 0, 0, widest - row.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40)) for row in rows]
    out = RESULTS_DIR / "threshold_comparison.png"
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"Saved {out}")
