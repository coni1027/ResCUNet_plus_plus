"""
One-time data preparation: extract 2D slices from RIDER's raw volume
.npy files -- images shaped (n_slices, n_channels, H, W), e.g. the
(60, 4, 288, 288) you found; masks shaped (n_slices, H, W) or
(n_slices, 1, H, W) -- into the per-slice images/masks layout
datasets/breast_dataset.py actually reads: data/breast_mri/<split>/
{images,masks}/<file>.npy. Run this before anything else touches
data/breast_mri/.

  MRI volume
      |
      v
  verify image and mask shapes match (same n_slices)
      |
      v
  extract each slice
      |
      v
  preserve patient ID (baked into every slice's filename)
      |
      v
  save N image-mask pairs (all of them -- see below)

Design, per your answers:
  - ALL slices are kept, including ones with an empty (all-zero) lesion
    mask. No positive-only filtering -- negative/background slices stay
    in the dataset on purpose, so the model sees a realistic class mix.
  - Each mask volume is sliced along the same (first) axis as its
    paired image volume, so slice i of the mask always corresponds to
    slice i of the image -- verified (shapes must match) before either
    is touched.
  - The train/val/test split is decided ONCE, per PATIENT, before any
    slicing happens -- every slice from every volume belonging to a
    patient goes to that patient's assigned split. This is what
    guarantees patient safety (Methodology 3.2): a volume is never
    partially split across train/val/test, and neither is a patient
    with more than one volume (e.g. separate pre-/post-therapy scans
    for the same person). The split is cached to
    splits/breast_mri_patient_split.json (see preprocessing.common) so
    re-running this script never silently reshuffles which patients
    land in test.
  - Patient identity is preserved in the output filename: every slice
    is named "RIDER-<patient>_vol<N>_slice<###>.npy". This is
    deliberately the exact pattern
    cross_validation.folds.default_patient_id_from_filename already
    parses -- so MRI_CONFIG.patient_id_fn needs no further adjustment
    once this script has run; just set it, as run_all.py already does.
  - A manifest CSV (data/breast_mri/extraction_manifest.csv) records,
    per slice: source volume path, patient ID, split, volume index,
    slice index, and whether the mask was non-empty -- full traceability
    back to the raw files, and a quick way to check class balance
    per split afterward.

ASSUMPTIONS ABOUT YOUR RAW LAYOUT -- READ BEFORE RUNNING.
This script cannot see your actual raw directory structure, so it
assumes:
  1. Raw image volumes live under --raw-images-dir, one .npy file per
     volume, with the patient ID recoverable from either the file's
     PARENT FOLDER name or its filename (see patient_id_from_path()).
  2. Each image volume has a paired mask volume at the SAME relative
     path under --raw-masks-dir (a mirrored directory tree).
  3. Image volumes are (n_slices, n_channels, H, W); mask volumes are
     (n_slices, H, W) or (n_slices, 1, H, W), with the SAME n_slices.
If any of these don't match your actual files, edit
patient_id_from_path() and find_volume_pairs() below -- they're
isolated at the top specifically so you can adapt them without
touching the splitting/slicing/writing logic further down.

ALWAYS run with --dry-run first. It prints the full patient/volume/
split plan and every volume's detected shape without writing a single
file or the manifest, so you can catch a wrong assumption before it
costs you a rerun.

Usage:
  python preprocessing/prepare_mri.py --dry-run
  python preprocessing/prepare_mri.py \
      --raw-images-dir /path/to/raw/images --raw-masks-dir /path/to/raw/masks
  python preprocessing/prepare_mri.py   # uses the defaults below once edited
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MRI_CONFIG, SEED, SPLITS_DIR
from preprocessing.common import check_output_is_clean, get_or_create_patient_split, write_manifest

# -----------------------------------------------------------------------------
# EDIT THESE to point at your actual raw layout (see module docstring).
# -----------------------------------------------------------------------------

RAW_IMAGES_DIR = Path("data/raw/breast_mri/images")
RAW_MASKS_DIR = Path("data/raw/breast_mri/masks")

_PATIENT_ID_PATTERN = re.compile(
    r"(?P<patient>RIDER[\s_-]?[A-Za-z0-9]+[_-]?\d+|P[_-]?\d{3,}|\d{3,})",
    re.IGNORECASE,
)


def patient_id_from_path(volume_path: Path) -> str:
    """
    Best-effort patient ID from a raw volume file's path -- checks the
    PARENT FOLDER name first (e.g. a "RIDER-1023/" directory), then
    falls back to the filename itself. Always returns the BARE ID with
    any RIDER prefix already stripped (e.g. "1023", not "RIDER-1023"),
    since extract_volume() below adds "RIDER-" itself when building the
    output filename -- stripping it here, once, is what stops that from
    ever doubling into "RIDER-RIDER-1023...". VERIFY this against your
    actual raw files: --dry-run prints exactly what it recovers for
    every volume before anything is written.
    """
    for candidate in (volume_path.parent.name, volume_path.stem):
        match = _PATIENT_ID_PATTERN.search(candidate)
        if match:
            raw_id = match.group("patient").upper().replace(" ", "-")
            return re.sub(r"^RIDER[\s_-]?", "", raw_id, flags=re.IGNORECASE)
    raise ValueError(
        f"Could not recover a patient ID from {volume_path} (checked parent "
        f"folder '{volume_path.parent.name}' and filename '{volume_path.stem}'). "
        "Edit patient_id_from_path() in this script to match your naming."
    )


def find_volume_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    """
    Finds every raw image volume under images_dir and pairs it with a
    mask volume at the same relative path under masks_dir. Raises if
    ANY pairing is missing, rather than silently skipping a volume --
    a silently-dropped volume is a patient quietly missing from the
    dataset, which is worse than a loud crash here.
    """
    image_paths = sorted(images_dir.rglob("*.npy"))
    if not image_paths:
        raise FileNotFoundError(f"No .npy volumes found under {images_dir}")

    pairs = []
    missing = []
    for image_path in image_paths:
        relative = image_path.relative_to(images_dir)
        mask_path = masks_dir / relative
        if mask_path.exists():
            pairs.append((image_path, mask_path))
        else:
            missing.append(str(relative))

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image volume(s) have no mask at the same relative "
            f"path under {masks_dir}. First few: {missing[:5]}"
        )
    return pairs


# -----------------------------------------------------------------------------
# Slicing + writing
# -----------------------------------------------------------------------------

def load_mask_volume(mask_path: Path, n_slices: int) -> np.ndarray:
    mask_volume = np.load(mask_path)
    if mask_volume.ndim == 4 and mask_volume.shape[1] == 1:
        mask_volume = mask_volume[:, 0]
    if mask_volume.ndim != 3 or mask_volume.shape[0] != n_slices:
        raise ValueError(
            f"{mask_path}: expected a mask volume of shape ({n_slices}, H, W) "
            f"(or with a singleton channel axis), got {mask_volume.shape}"
        )
    return mask_volume


def extract_volume(
    image_path: Path,
    mask_path: Path,
    patient_id: str,
    volume_idx: int,
    split: str,
    out_root: Path,
    dry_run: bool,
    manifest_rows: list[dict],
) -> tuple[int, int]:
    """
    Slice one image/mask volume pair and write every slice out (verifies
    shapes match, then extracts + preserves patient ID + saves -- see
    module docstring). Returns (n_slices, n_slices_with_nonempty_mask).
    """
    image_volume = np.load(image_path)
    if image_volume.ndim != 4:
        raise ValueError(
            f"{image_path}: expected a 4D (slices, channels, H, W) volume, "
            f"got shape {image_volume.shape}"
        )
    n_slices = image_volume.shape[0]
    mask_volume = load_mask_volume(mask_path, n_slices)  # raises if shapes don't match

    images_out = out_root / split / "images"
    masks_out = out_root / split / "masks"
    if not dry_run:
        images_out.mkdir(parents=True, exist_ok=True)
        masks_out.mkdir(parents=True, exist_ok=True)

    nonempty = 0
    for slice_idx in range(n_slices):
        image_slice = image_volume[slice_idx]   # (channels, H, W) -- CHW as-is
        mask_slice = mask_volume[slice_idx]      # (H, W)
        has_lesion = bool(mask_slice.any())
        nonempty += int(has_lesion)

        filename = f"RIDER-{patient_id}_vol{volume_idx}_slice{slice_idx:03d}.npy"
        if not dry_run:
            np.save(images_out / filename, image_slice.astype(np.float32))
            np.save(masks_out / filename, (mask_slice > 0).astype(np.float32))

        manifest_rows.append({
            "filename": filename,
            "patient_id": patient_id,
            "split": split,
            "source_image": str(image_path),
            "source_mask": str(mask_path),
            "volume_idx": volume_idx,
            "slice_idx": slice_idx,
            "has_lesion": has_lesion,
        })

    return n_slices, nonempty


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-slice RIDER images/masks from raw volumes into "
                    "data/breast_mri/<split>/{images,masks}/, split patient-first."
    )
    parser.add_argument("--raw-images-dir", type=Path, default=RAW_IMAGES_DIR)
    parser.add_argument("--raw-masks-dir", type=Path, default=RAW_MASKS_DIR)
    parser.add_argument("--out-root", type=Path, default=MRI_CONFIG.root)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the patient/volume/split plan and detected shapes; write nothing.",
    )
    args = parser.parse_args()

    if not args.dry_run:
        check_output_is_clean(args.out_root)

    pairs = find_volume_pairs(args.raw_images_dir, args.raw_masks_dir)
    print(f"Found {len(pairs)} raw volume(s) under {args.raw_images_dir}")

    volumes_by_patient: dict[str, list[tuple[Path, Path]]] = {}
    for image_path, mask_path in pairs:
        patient_id = patient_id_from_path(image_path)
        volumes_by_patient.setdefault(patient_id, []).append((image_path, mask_path))

    ratios = {"train": args.train_frac, "val": args.val_frac, "test": args.test_frac}
    split_groups = get_or_create_patient_split(
        list(volumes_by_patient), args.seed, ratios,
        path=SPLITS_DIR / "breast_mri_patient_split.json",
    )
    split_assignment = {
        patient_id: split_name
        for split_name, patient_ids in split_groups.items()
        for patient_id in patient_ids
    }

    print(f"\n{len(volumes_by_patient)} patient(s) detected -> split assignment (seed={args.seed}):")
    for patient_id, volumes in sorted(volumes_by_patient.items()):
        split = split_assignment[patient_id]
        sources = ", ".join(p[0].name for p in volumes)
        print(f"  {patient_id:<14} -> {split:<5}  ({len(volumes)} volume(s): {sources})")

    if args.dry_run:
        print("\n--dry-run: inspecting shapes only, writing nothing.")

    manifest_rows: list[dict] = []
    total_slices = 0
    total_nonempty = 0
    for patient_id, volumes in sorted(volumes_by_patient.items()):
        split = split_assignment[patient_id]
        for volume_idx, (image_path, mask_path) in enumerate(volumes):
            n_slices, nonempty = extract_volume(
                image_path, mask_path, patient_id, volume_idx, split,
                args.out_root, args.dry_run, manifest_rows,
            )
            total_slices += n_slices
            total_nonempty += nonempty
            print(
                f"  [{split}] {image_path.name}: {n_slices} slices "
                f"({nonempty} with a non-empty mask, {n_slices - nonempty} background)"
            )

    print(
        f"\n{'Would write' if args.dry_run else 'Wrote'} {total_slices} slice pairs total "
        f"({total_nonempty} with a lesion, {total_slices - total_nonempty} background-only) "
        f"under {args.out_root}"
    )

    if args.dry_run:
        print("Re-run without --dry-run once this plan looks right.")
        return

    write_manifest(manifest_rows, args.out_root / "extraction_manifest.csv")


if __name__ == "__main__":
    main()
