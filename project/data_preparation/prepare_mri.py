"""
RIDER breast MRI: raw DICOM series -> per-slice .npy pairs.

Each patient's t+0 exam holds an "early anatomical reference" series and
a "reference VOI mask" series (60 slices each). Only those two are used;
B0/B800/ADC and the t+15 exam (which has no mask) are skipped.

  image: DICOM slice -> normalize -> mean filter -> CLAHE -> resize -> .npy
  mask:  DICOM slice -> binarize (>0) -> resize (nearest)       -> .npy

Raw layout (TCIA download):
    <raw-root>/RIDER-<id>/<study uid>/<series uid>/*.dcm

Output (what datasets/breast_dataset.py reads):
    <out-root>/<split>/{images,masks}/RIDER-<id>_vol0_slice000.npy

Usage:
    python data_preparation/prepare_mri.py --dry-run
    python data_preparation/prepare_mri.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR, MRI_CONFIG
from cross_validation.folds import default_patient_id_from_filename
from data_preparation.common import SPLITS, check_output_is_clean, write_manifest
from data_preparation.transforms import preprocess_image, resize_image, resize_mask

IMAGE_SERIES = "early anatomical reference"
MASK_SERIES = "reference VOI mask"

# Patient-level split (Methodology 3.2), same assignment as the previous dataset
# so results stay comparable. Every patient must appear here exactly once.
PATIENT_SPLIT = {
    "RIDER-1627409910": "train",
    "RIDER-2217584661": "train",
    "RIDER-2382137773": "train",
    "RIDER-5423859857": "val",
    "RIDER-1125105682": "test",
}


def find_series(patient_dir: Path, description: str) -> list[Path]:
    """Slice files of the one series under patient_dir with this SeriesDescription, in slice order."""
    matches = []
    for series_dir in sorted(patient_dir.glob("*/*")):
        files = sorted(series_dir.glob("*.dcm"))
        if files and pydicom.dcmread(files[0], stop_before_pixels=True).SeriesDescription == description:
            matches.append(files)
    if len(matches) != 1:
        raise ValueError(f"{patient_dir.name}: expected 1 '{description}' series, found {len(matches)}")
    return sorted(matches[0], key=lambda f: int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber))


def load_pairs(patient_dir: Path) -> list[tuple[pydicom.Dataset, pydicom.Dataset]]:
    image_files = find_series(patient_dir, IMAGE_SERIES)
    mask_files = find_series(patient_dir, MASK_SERIES)
    if len(image_files) != len(mask_files):
        raise ValueError(f"{patient_dir.name}: {len(image_files)} image slices vs {len(mask_files)} mask slices")

    pairs = []
    for image_file, mask_file in zip(image_files, mask_files):
        image_ds, mask_ds = pydicom.dcmread(image_file), pydicom.dcmread(mask_file)
        # Same slice position, or the mask would label the wrong slice.
        if list(image_ds.ImagePositionPatient) != list(mask_ds.ImagePositionPatient):
            raise ValueError(f"{patient_dir.name}: slice {image_ds.InstanceNumber} image/mask positions differ")
        pairs.append((image_ds, mask_ds))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice RIDER DICOM series into <out-root>/<split>/{images,masks}/*.npy")
    parser.add_argument("--raw-root", type=Path, default=DATA_DIR / "MRI" / "rider_breast_mri")
    parser.add_argument("--out-root", type=Path, default=MRI_CONFIG.root)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; write nothing.")
    args = parser.parse_args()

    if args.raw_root.resolve() == args.out_root.resolve():
        parser.error("--raw-root and --out-root are the same directory.")

    patient_dirs = sorted(p for p in args.raw_root.iterdir() if p.is_dir())
    found = {p.name for p in patient_dirs}
    if found != set(PATIENT_SPLIT):
        raise ValueError(
            f"Patients on disk don't match PATIENT_SPLIT. Missing from split: {sorted(found - set(PATIENT_SPLIT))}; "
            f"in split but not on disk: {sorted(set(PATIENT_SPLIT) - found)}"
        )

    # Output filenames must group back to one ID per patient, or CV folds leak.
    groups = {default_patient_id_from_filename(Path(f"{name}_vol0_slice000.npy")) for name in found}
    if len(groups) != len(found):
        raise ValueError(f"Patient IDs don't parse into distinct CV groups: {sorted(groups)}")

    if not args.dry_run:
        check_output_is_clean(args.out_root)

    manifest_rows: list[dict] = []
    totals = {split: [0, 0] for split in SPLITS}
    for patient_dir in patient_dirs:
        split = PATIENT_SPLIT[patient_dir.name]
        images_out = args.out_root / split / "images"
        masks_out = args.out_root / split / "masks"
        if not args.dry_run:
            images_out.mkdir(parents=True, exist_ok=True)
            masks_out.mkdir(parents=True, exist_ok=True)

        pairs = load_pairs(patient_dir)
        nonempty = 0
        for slice_idx, (image_ds, mask_ds) in enumerate(pairs):
            mask = (mask_ds.pixel_array > 0).astype(np.float32)
            has_lesion = bool(mask.any())
            nonempty += has_lesion

            filename = f"{patient_dir.name}_vol0_slice{slice_idx:03d}.npy"
            if not args.dry_run:
                image = preprocess_image(image_ds.pixel_array.astype(np.float32), MRI_CONFIG)
                np.save(images_out / filename, resize_image(image, MRI_CONFIG.input_size).astype(np.float32))
                np.save(masks_out / filename, resize_mask(mask, MRI_CONFIG.input_size))

            manifest_rows.append({
                "filename": filename,
                "patient_id": patient_dir.name,
                "split": split,
                "source_image": str(image_ds.filename),
                "source_mask": str(mask_ds.filename),
                "slice_idx": slice_idx,
                "has_lesion": has_lesion,
            })

        totals[split][0] += len(pairs)
        totals[split][1] += nonempty
        print(f"  [{split:<5}] {patient_dir.name}: {len(pairs)} slices ({nonempty} with a lesion)")

    print(f"\n{'Would write' if args.dry_run else 'Wrote'} under {args.out_root}:")
    for split in SPLITS:
        n, n_lesion = totals[split]
        print(f"  {split}: {n} slices ({n_lesion} with a lesion)")

    if args.dry_run:
        print("--dry-run: nothing written.")
        return
    write_manifest(manifest_rows, args.out_root / "extraction_manifest.csv")


if __name__ == "__main__":
    main()
