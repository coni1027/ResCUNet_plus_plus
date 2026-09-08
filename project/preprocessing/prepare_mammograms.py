"""
One-time data preparation: finds each CBIS-DDSM case's Whole+ROI DICOM
pair, resizes both to 512x512 (MAMMOGRAM_CONFIG.input_size), and writes
them into the per-case, per-split layout datasets/breast_dataset.py
expects (data/mammograms/<split>/{images,masks}/).

Each case in your raw export has four files -- e.g. for "P_00731 Right
MLO": "... Whole.dcm" (the full mammogram, e.g. (5680, 3728) uint16),
"... ROI.dcm" (the pixel-aligned lesion mask, same shape, uint8), "...
Zoomed.dcm" (a small cropped patch around the finding, e.g. (395, 395)
-- a different image entirely, not a crop of Whole you can resize back),
and a "....png" export (a low-res RGBA preview, not full-resolution
pixel data). Only Whole+ROI are used here -- Zoomed and the .png are
deliberately never touched, per your instruction.

Resizing and saving here (rather than at load time, which
datasets/breast_dataset.py's resize_image() already does dynamically
for any file format it reads) turns a ~5680x3728, 16-bit DICOM decode +
resize into a cheap .npy load, repeated every epoch of training instead
of once here -- same reasoning as prepare_mri.py's slice extraction.
Only the resize happens here; percentile normalization / CLAHE / median
filtering still happen at load time via preprocess_image(), exactly as
before -- applying them here too would double-process every sample.

  CBIS-DDSM case (Whole + ROI DICOM pair)
      |
      v
  verify Whole and ROI shapes match
      |
      v
  resize both to 512 x 512 (image: linear, mask: nearest)
      |
      v
  preserve patient ID (baked into every file's filename, matching
  CBIS-DDSM's own "<patient>_<side>_<view>" convention)
      |
      v
  save the image-mask pair

*** READ BEFORE RUNNING: the train/test split below is a PLACEHOLDER. ***
Methodology 3.2 says CBIS-DDSM's OFFICIAL train/test split is retained,
with an 80:20 patient-level train:val carve-out from the official
training cases. This script only has your raw file LISTING to go on,
not the official split metadata -- CBIS-DDSM normally distributes that
as mass_case_description_{train,test}_set.csv and
calc_case_description_{train,test}_set.csv alongside the images.
official_split_for_case() below returns "train" for every case until
you edit it to consult whichever of those CSVs you have -- meaning
right now there is NO test set at all, which does not match the
methodology. main() prints a loud warning about this every run;
--dry-run's summary makes it obvious (100% of cases in train/val, 0 in
test) before you'd otherwise notice too late.

ALWAYS run with --dry-run first. It prints the full case/patient/split
plan and every case's detected shape without writing a single file or
the manifest.

Usage:
  python preprocessing/prepare_mammograms.py --dry-run
  python preprocessing/prepare_mammograms.py --raw-root /path/to/raw/cbis-ddsm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pydicom
except ImportError:  # matches datasets/breast_dataset.py's optional import
    pydicom = None

from config import MAMMOGRAM_CONFIG, SEED, SPLITS_DIR
from datasets.breast_dataset import load_dicom, resize_image, resize_mask
from preprocessing.common import check_output_is_clean, get_or_create_patient_split, write_manifest

# -----------------------------------------------------------------------------
# EDIT THIS to point at your actual raw layout (see module docstring).
# -----------------------------------------------------------------------------

RAW_ROOT = Path("data/raw/mammograms")


def patient_id_from_case(case_id: str) -> str:
    """
    Patient ID is the case ID's first token, e.g. "P_00731" from
    "P_00731 Right MLO" -- matches CBIS-DDSM's own patient-ID convention
    and cross_validation.folds.default_patient_id_from_filename's
    P[_-]?\\d{3,} pattern directly (verified against this script's exact
    output filenames before writing this file), so no further
    adjustment is needed once these files are written.
    """
    return case_id.split()[0]


def find_case_pairs(raw_root: Path) -> list[tuple[str, Path, Path]]:
    """
    Finds every case's "Whole" mammogram DICOM under raw_root and pairs
    it with its "ROI" mask DICOM in the same folder. "Zoomed" DICOMs
    and any .png exports are deliberately never touched -- only
    Whole+ROI are used (see module docstring). Returns (case_id,
    whole_path, roi_path), sorted for determinism.
    """
    whole_paths = sorted(raw_root.rglob("*Whole.dcm"))
    if not whole_paths:
        raise FileNotFoundError(f"No '*Whole.dcm' files found under {raw_root}")

    pairs = []
    missing = []
    for whole_path in whole_paths:
        case_id = whole_path.stem[: -len("Whole")].strip()
        roi_path = whole_path.with_name(f"{case_id} ROI.dcm")
        if roi_path.exists():
            pairs.append((case_id, whole_path, roi_path))
        else:
            missing.append(case_id)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} case(s) have a 'Whole.dcm' but no matching "
            f"'ROI.dcm' in the same folder. First few: {missing[:5]}"
        )
    return pairs


def official_split_for_case(case_id: str, patient_id: str) -> str:
    """
    PLACEHOLDER -- see the "READ BEFORE RUNNING" section of the module
    docstring. Returns "train" or "test" per CBIS-DDSM's official split.
    Edit this to consult your mass/calc_case_description_{train,test}_
    set.csv file(s) -- e.g. load them once at module level into a
    patient_id -> "train"/"test" lookup dict and return from that here.
    """
    return "train"


# -----------------------------------------------------------------------------
# Loading + resizing + writing
# -----------------------------------------------------------------------------

def load_mask_dicom(path: Path) -> np.ndarray:
    """
    Reads a mask DICOM's raw pixel array, deliberately WITHOUT
    datasets.breast_dataset.load_dicom()'s MONOCHROME1 inversion --
    that inversion is a display-brightness convention for grayscale
    images. Applying it to a label/mask image would flip which pixels
    mean "lesion" versus "background" if the mask DICOM happens to
    carry the same PhotometricInterpretation tag as its paired Whole
    image.
    """
    if pydicom is None:
        raise ImportError(
            "pydicom is required to read ROI DICOM masks. Install it with: pip install pydicom"
        )
    ds = pydicom.dcmread(str(path))
    return ds.pixel_array.astype(np.float32)


def extract_case(
    case_id: str,
    whole_path: Path,
    roi_path: Path,
    split: str,
    out_root: Path,
    target_size: tuple[int, int],
    dry_run: bool,
    manifest_rows: list[dict],
) -> bool:
    """
    Verifies the Whole/ROI pair's shapes match, resizes both to
    target_size (image: linear, mask: nearest -- see
    datasets.breast_dataset.resize_image/resize_mask), and writes the
    pair out. Returns whether the mask is non-empty (informational
    only -- nothing is filtered by it, matching prepare_mri.py).
    """
    image = load_dicom(whole_path)      # float32, MONOCHROME1-corrected
    mask = load_mask_dicom(roi_path)    # float32, raw (no inversion)

    if image.shape != mask.shape:
        raise ValueError(
            f"{case_id}: Whole image shape {image.shape} != ROI mask shape "
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

    filename = f"{case_id.replace(' ', '_')}.npy"
    if not dry_run:
        np.save(images_out / filename, image)
        np.save(masks_out / filename, mask)

    manifest_rows.append({
        "filename": filename,
        "case_id": case_id,
        "patient_id": patient_id_from_case(case_id),
        "split": split,
        "source_whole": str(whole_path),
        "source_roi": str(roi_path),
        "original_shape": f"{original_shape[0]}x{original_shape[1]}",
        "has_lesion": has_lesion,
    })
    return has_lesion


# -----------------------------------------------------------------------------
# Split resolution: official train/test, then an 80:20 val carve-out
# -----------------------------------------------------------------------------

def resolve_splits(
    cases: list[tuple[str, Path, Path]],
    seed: int,
    val_fraction: float,
) -> dict[str, str]:
    """
    case_id -> "train"/"val"/"test", in two stages:
      1. official_split_for_case() decides "train" vs "test" (see its
         docstring -- almost certainly a placeholder right now).
      2. Patients in the "train" pool are further split into
         train/val at (1 - val_fraction):val_fraction, patient-level,
         seeded and cached -- Methodology 3.2's 80:20.
    """
    case_to_official = {
        case_id: official_split_for_case(case_id, patient_id_from_case(case_id))
        for case_id, _, _ in cases
    }

    train_patients = sorted({
        patient_id_from_case(case_id)
        for case_id, split in case_to_official.items()
        if split == "train"
    })
    val_groups = get_or_create_patient_split(
        train_patients, seed, {"train": 1 - val_fraction, "val": val_fraction},
        path=SPLITS_DIR / "mammogram_train_val_split.json",
    )
    patient_to_final = {p: "train" for p in val_groups["train"]}
    patient_to_final.update({p: "val" for p in val_groups["val"]})

    return {
        case_id: "test" if official == "test" else patient_to_final[patient_id_from_case(case_id)]
        for case_id, official in case_to_official.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resize CBIS-DDSM Whole+ROI DICOM pairs (excluding Zoomed) into "
                    "data/mammograms/<split>/{images,masks}/."
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=MAMMOGRAM_CONFIG.root)
    parser.add_argument(
        "--target-size", type=int, nargs=2, default=list(MAMMOGRAM_CONFIG.input_size),
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the case/patient/split plan and detected shapes; write nothing.",
    )
    args = parser.parse_args()
    target_size = tuple(args.target_size)

    print(
        "WARNING: official_split_for_case() is still a placeholder that returns "
        "'train' for every case -- there is currently NO test set. See this "
        "script's module docstring before trusting the split below.\n"
    )

    if not args.dry_run:
        check_output_is_clean(args.output_root)

    cases = find_case_pairs(args.raw_root)
    print(f"Found {len(cases)} case(s) under {args.raw_root} (Zoomed/*.png excluded)")

    split_assignment = resolve_splits(cases, args.seed, args.val_fraction)

    patients = sorted({patient_id_from_case(case_id) for case_id, _, _ in cases})
    print(f"\n{len(patients)} patient(s), {len(cases)} case(s) -> split assignment (seed={args.seed}):")
    for case_id, _, _ in sorted(cases):
        print(f"  {case_id:<24} (patient {patient_id_from_case(case_id):<9}) -> {split_assignment[case_id]}")

    if args.dry_run:
        print("\n--dry-run: inspecting shapes only, writing nothing.")

    manifest_rows: list[dict] = []
    totals = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}  # [n_cases, n_with_lesion]
    for case_id, whole_path, roi_path in cases:
        split = split_assignment[case_id]
        has_lesion = extract_case(
            case_id, whole_path, roi_path, split,
            args.output_root, target_size, args.dry_run, manifest_rows,
        )
        totals[split][0] += 1
        totals[split][1] += int(has_lesion)

    print(f"\n{'Would write' if args.dry_run else 'Wrote'} {len(cases)} case(s) total, resized to {target_size}:")
    for split_name in ("train", "val", "test"):
        n_cases, n_with_lesion = totals[split_name]
        print(f"  {split_name}: {n_cases} cases ({n_with_lesion} with a lesion, {n_cases - n_with_lesion} empty)")

    if args.dry_run:
        print("Re-run without --dry-run once this plan looks right.")
        return

    write_manifest(manifest_rows, args.output_root / "extraction_manifest.csv")


if __name__ == "__main__":
    main()
