"""
CBIS-DDSM mammograms: already-split DICOM export -> resized .npy pairs.

  image: DICOM -> remove background -> normalize -> mean filter -> CLAHE -> resize -> .npy
  mask:  DICOM -> binarize (>0) -> resize (nearest)                          -> .npy

Background removal (data_preparation/transform_thresholding.py) is on by
default; pass --no-background-removal to build the dataset without it.
Resizing stretches to config.input_size (no crop, no padding).

Filtering and CLAHE run at full resolution, before the resize, so they
see the original detail rather than a downsampled image.

Raw layout (split already decided on disk; nesting depth doesn't matter):
    <raw-root>/<split>/images/P_00001_LEFT_CC.dcm
    <raw-root>/<split>/masks/P_00001_LEFT_CC.dcm    # same filename as its image

Output (what datasets/breast_dataset.py reads):
    <out-root>/<split>/{images,masks}/P_00001_LEFT_CC.npy

Usage:
    python data_preparation/prepare_mammograms.py --dry-run
    python data_preparation/prepare_mammograms.py
    python data_preparation/prepare_mammograms.py --no-background-removal --out-root data/mammograms_no_bg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR, MAMMOGRAM_CONFIG
from cross_validation.folds import default_patient_id_from_filename
from data_preparation.common import SPLITS, build_plan, check_output_is_clean, write_manifest
from data_preparation.transform_thresholding import remove_background
from data_preparation.transforms import preprocess_image, resize_image, resize_mask
from datasets.breast_dataset import load_dicom


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess an already-split CBIS-DDSM export into <out-root>/<split>/{images,masks}/*.npy")
    parser.add_argument("--raw-root", type=Path, default=DATA_DIR / "mammograms_raw")
    parser.add_argument("--out-root", type=Path, default=MAMMOGRAM_CONFIG.root)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; write nothing.")
    parser.add_argument(
        "--background-removal", action=argparse.BooleanOptionalAction, default=True,
        help="Black out everything except the breast before preprocessing (default: on).",
    )
    args = parser.parse_args()

    if args.raw_root.resolve() == args.out_root.resolve():
        parser.error("--raw-root and --out-root are the same directory.")

    # Patient ID comes from the filename (P_00001_LEFT_CC -> P_00001), the same
    # parser CV uses to group folds, so the leakage check and CV always agree.
    plan = build_plan(args.raw_root, ".dcm", default_patient_id_from_filename)
    patients = {pid for entries in plan.values() for _, _, pid in entries}
    print(f"{len(patients)} patient(s), {sum(map(len, plan.values()))} image(s) under {args.raw_root}:")
    for split in SPLITS:
        print(f"  {split}: {len(plan[split])} image(s)")

    if args.dry_run:
        print("--dry-run: nothing written.")
        return
    check_output_is_clean(args.out_root)

    size = MAMMOGRAM_CONFIG.input_size
    manifest_rows: list[dict] = []
    for split in SPLITS:
        images_out = args.out_root / split / "images"
        masks_out = args.out_root / split / "masks"
        images_out.mkdir(parents=True, exist_ok=True)
        masks_out.mkdir(parents=True, exist_ok=True)

        nonempty = 0
        for image_path, mask_path, patient_id in plan[split]:
            image = load_dicom(image_path)  # MONOCHROME1-corrected
            # Masks skip load_dicom's MONOCHROME1 inversion: it would swap lesion and background.
            mask = (pydicom.dcmread(mask_path).pixel_array > 0).astype(np.float32)
            if image.shape != mask.shape:
                raise ValueError(f"{image_path.name}: image {image.shape} != mask {mask.shape}, not pixel-aligned")
            original_shape = image.shape

            lesion_zeroed = 0.0
            if args.background_removal:
                cleaned, _ = remove_background(image)
                lesion_zeroed = float(((cleaned == 0) & (image > 0) & (mask > 0)).sum() / max(mask.sum(), 1))
                image = cleaned

            image = resize_image(preprocess_image(image, MAMMOGRAM_CONFIG), size).astype(np.float32)
            mask = resize_mask(mask, size)
            has_lesion = bool(mask.any())
            nonempty += has_lesion

            filename = f"{image_path.stem}.npy"
            np.save(images_out / filename, image)
            np.save(masks_out / filename, mask)

            manifest_rows.append({
                "filename": filename,
                "patient_id": patient_id,
                "split": split,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "original_shape": f"{original_shape[0]}x{original_shape[1]}",
                "has_lesion": has_lesion,
                "background_removed": args.background_removal,
                "lesion_fraction_zeroed": round(lesion_zeroed, 5),
            })
        print(f"  {split}: wrote {len(plan[split])} image(s) ({nonempty} with a lesion)")

    write_manifest(manifest_rows, args.out_root / "extraction_manifest.csv")


if __name__ == "__main__":
    main()
