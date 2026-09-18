"""
Variant of preprocessing/prepare_mammograms.py for raw mammogram data
that is ALREADY split into train/val/test on disk, with images/ and
masks/ paired by identical filename (e.g. a Kaggle-style CBIS-DDSM
export: "<raw-root>/train/images/P_00001_LEFT_CC.dcm" and the same
filename under "masks/") -- not CBIS-DDSM's own per-case folder layout
("<case> Whole.dcm" / "<case> ROI.dcm" / "<case> Zoomed.dcm") that
prepare_mammograms.py expects.

prepare_mammograms.py computes its own case pairing (Whole+ROI by
shared case folder) AND its own split (official_split_for_case(), a
placeholder that currently puts everything in "train" -- see that
module's docstring). This script does neither: it reads the split
straight off the directory each file is already sitting in and pairs
images with masks by identical filename within that split, so an
existing assignment is preserved exactly and the placeholder-split
problem doesn't apply to data prepared this way.

Input layout:
    <raw-root>/train/images/<case>.dcm
    <raw-root>/train/masks/<case>.dcm     # same filename as its image
    <raw-root>/val/...  <raw-root>/test/...
Each split's images/ and masks/ directories are located with rglob, so
extra nesting from how a .zip export happened to unpack (e.g.
<raw-root>/train/train/train/images/...) does not need to be flattened
by hand first -- point --raw-root at the split-level parent either way.

Output layout (what datasets/breast_dataset.py reads):
    <out-root>/train/images/<case>.npy
    <out-root>/train/masks/<case>.npy

Because the split comes from the folder, no patient can cross splits --
verified against the actual export this was written for (892 patients,
zero appearing in more than one split, via the same P[_-]?\\d{3,}
pattern cross_validation.folds.default_patient_id_from_filename uses).
This script re-derives patient IDs from the ORIGINAL filenames (never
renamed here, so there's no drift risk between raw and output naming)
and refuses to write if any patient ever shows up under more than one
split.

Masks are read via prepare_mammograms.load_mask_dicom(), which
deliberately skips the MONOCHROME1 inversion datasets.breast_dataset
.load_dicom() applies to images -- confirmed unnecessary for this
export (masks sampled as MONOCHROME2), but kept for safety in case a
different export ever mixes conventions.

Usage:
  python preprocessing/prepare_mammograms_presplit.py --raw-root data/mammogram --dry-run
  python preprocessing/prepare_mammograms_presplit.py --raw-root data/mammogram
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MAMMOGRAM_CONFIG
from cross_validation.folds import default_patient_id_from_filename
from datasets.breast_dataset import load_dicom, resize_image, resize_mask
from preprocessing.common import check_output_is_clean, write_manifest
from preprocessing.prepare_mammograms import load_mask_dicom

SPLITS = ("train", "val", "test")


def _find_only(base: Path, name: str) -> Path:
    """Locate the one `name` directory under `base`, at any depth."""
    matches = [path for path in base.rglob(name) if path.is_dir()]
    if not matches:
        raise FileNotFoundError(f"No '{name}' directory found under {base}")
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple '{name}' directories found under {base}: {matches}. "
            "Narrow --raw-root down to a single split export."
        )
    return matches[0]


def find_pairs(raw_root: Path, split: str) -> list[tuple[Path, Path]]:
    split_root = raw_root / split
    if not split_root.exists():
        return []

    images_dir = _find_only(split_root, "images")
    masks_dir = _find_only(split_root, "masks")

    pairs = []
    missing = []
    for image_path in sorted(images_dir.glob("*.dcm")):
        mask_path = masks_dir / image_path.name
        if mask_path.exists():
            pairs.append((image_path, mask_path))
        else:
            missing.append(image_path.name)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) in {images_dir} have no same-named mask "
            f"in {masks_dir}. First few: {missing[:5]}"
        )
    return pairs


def build_plan(raw_root: Path) -> dict[str, list[tuple[Path, Path, str]]]:
    """
    Resolve every case to (image, mask, patient_id), with the split
    coming from the folder it's already sitting in. Raises if the same
    patient ID is found under more than one split -- that would be
    cross-split leakage, which this presplit layout is supposed to
    already avoid.
    """
    plan: dict[str, list[tuple[Path, Path, str]]] = {}
    patient_split: dict[str, str] = {}

    for split in SPLITS:
        entries = []
        for image_path, mask_path in find_pairs(raw_root, split):
            patient_id = default_patient_id_from_filename(image_path)

            if patient_split.get(patient_id, split) != split:
                raise ValueError(
                    f"Patient {patient_id!r} appears in both "
                    f"{patient_split[patient_id]!r} and {split!r}. That is "
                    "cross-split leakage -- fix the input layout first."
                )
            patient_split[patient_id] = split

            entries.append((image_path, mask_path, patient_id))
        plan[split] = entries

    if not any(plan.values()):
        raise FileNotFoundError(f"No image/mask pairs found under {raw_root}")
    return plan


def extract_pair(
    image_path: Path,
    mask_path: Path,
    split: str,
    out_root: Path,
    target_size: tuple[int, int],
    dry_run: bool,
    manifest_rows: list[dict],
) -> bool:
    """
    Verifies the image/mask pair's shapes match, resizes both to
    target_size (image: linear, mask: nearest), and writes the pair
    out as .npy under the same filename. Returns whether the mask is
    non-empty (informational only -- nothing is filtered by it,
    matching prepare_mammograms.py / prepare_mri.py).
    """
    image = load_dicom(image_path)      # float32, MONOCHROME1-corrected
    mask = load_mask_dicom(mask_path)   # float32, raw (no inversion)

    if image.shape != mask.shape:
        raise ValueError(
            f"{image_path.name}: image shape {image.shape} != mask shape "
            f"{mask.shape} -- they must be pixel-aligned to be paired correctly."
        )
    original_shape = image.shape

    image = resize_image(image, target_size).astype(np.float32)
    mask = (resize_mask(mask, target_size) > 0).astype(np.float32)
    has_lesion = bool(mask.any())

    images_out = out_root / split / "images"
    masks_out = out_root / split / "masks"
    if not dry_run:
        images_out.mkdir(parents=True, exist_ok=True)
        masks_out.mkdir(parents=True, exist_ok=True)

    filename = f"{image_path.stem}.npy"
    if not dry_run:
        np.save(images_out / filename, image)
        np.save(masks_out / filename, mask)

    manifest_rows.append({
        "filename": filename,
        "patient_id": default_patient_id_from_filename(image_path),
        "split": split,
        "source_image": str(image_path),
        "source_mask": str(mask_path),
        "original_shape": f"{original_shape[0]}x{original_shape[1]}",
        "has_lesion": has_lesion,
    })
    return has_lesion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resize an already-split CBIS-DDSM images/masks export into "
                    "<out-root>/<split>/{images,masks}/, preserving the "
                    "train/val/test assignment already on disk."
    )
    parser.add_argument(
        "--raw-root", type=Path, required=True,
        help="Contains train/val/test, each with an images/ and masks/ of matching .dcm filenames "
             "(at any nesting depth -- located automatically).",
    )
    parser.add_argument("--out-root", type=Path, default=MAMMOGRAM_CONFIG.root)
    parser.add_argument(
        "--target-size", type=int, nargs=2, default=list(MAMMOGRAM_CONFIG.input_size),
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and detected shapes; write nothing.",
    )
    args = parser.parse_args()
    target_size = tuple(args.target_size)

    if args.raw_root.resolve() == Path(args.out_root).resolve():
        parser.error(
            "--raw-root and --out-root are the same directory. Point --out-root "
            "elsewhere (the default, config.MAMMOGRAM_CONFIG.root, already is) so "
            "the generated .npy files don't land on top of your raw .dcm files."
        )

    if not args.dry_run:
        check_output_is_clean(args.out_root)

    plan = build_plan(args.raw_root)
    patients = {pid for entries in plan.values() for _, _, pid in entries}
    n_cases = sum(len(entries) for entries in plan.values())
    print(f"{len(patients)} patient(s), {n_cases} case(s) under {args.raw_root}:")
    for split in SPLITS:
        print(f"  {split}: {len(plan[split])} case(s)")

    if args.dry_run:
        print("\n--dry-run: inspecting shapes only, writing nothing.")

    manifest_rows: list[dict] = []
    totals = {split: [0, 0] for split in SPLITS}
    for split in SPLITS:
        for image_path, mask_path, _ in plan[split]:
            has_lesion = extract_pair(
                image_path, mask_path, split, args.out_root, target_size, args.dry_run, manifest_rows,
            )
            totals[split][0] += 1
            totals[split][1] += int(has_lesion)

    print(f"\n{'Would write' if args.dry_run else 'Wrote'} {n_cases} case(s) total, resized to {target_size}:")
    for split in SPLITS:
        n, n_lesion = totals[split]
        print(f"  {split}: {n} cases ({n_lesion} with a lesion, {n - n_lesion} empty)")

    if args.dry_run:
        print("Re-run without --dry-run once this plan looks right.")
        return

    write_manifest(manifest_rows, args.out_root / "extraction_manifest.csv")


if __name__ == "__main__":
    main()
