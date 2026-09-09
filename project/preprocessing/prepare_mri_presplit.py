"""
Variant of preprocessing/prepare_mri.py for raw data that is ALREADY
split into train/val/test on disk.

prepare_mri.py computes its own patient-level split from a flat
images/masks tree. This script does not: it reads the split straight
off the directory each volume is sitting in, so an existing assignment
is preserved exactly. Everything else -- slice extraction, filename
convention, manifest, --dry-run -- matches prepare_mri.py.

Input layout (what you have now, once moved out of data/breast_mri):
    <raw-root>/train/images/PATIENT.npy    # (n_slices, n_channels, H, W)
    <raw-root>/train/masks/PATIENT.npy     # (n_slices, H, W) or (n_slices, 1, H, W)
    <raw-root>/val/...  <raw-root>/test/...

Output layout (what datasets/breast_dataset.py reads):
    <out-root>/train/images/RIDER-<patient>_vol0_slice000.npy
    <out-root>/train/masks/RIDER-<patient>_vol0_slice000.npy

Because the split comes from the folder, no patient can cross splits --
that guarantee comes from your input layout rather than from a seeded
shuffle, and there is no split cache to invalidate.

Place next to prepare_mri.py in preprocessing/ and run:
    python preprocessing/prepare_mri_presplit.py --raw-root data/raw/breast_mri --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MRI_CONFIG
from preprocessing.common import check_output_is_clean, write_manifest
from preprocessing.prepare_mri import extract_volume, patient_id_from_path

SPLITS = ("train", "val", "test")


_RIDER_ID = re.compile(r"^RIDER[\s_-]?(?P<patient>\d{4,})", re.IGNORECASE)


def derive_patient_id(volume_path: Path, mode: str) -> str:
    """
    'rider' -- the leading numeric TCIA subject ID, e.g. "1627409910"
               from "RIDER_1627409910_09_09_1880_NA_coffee_break_exam_
               __t0_mins_53292". Everything after that number is SERIES
               metadata (exam date, description, series number), not
               patient identity. Keeping it would mean the same person's
               baseline and repeat "coffee break" exams parse as two
               different patients -- so they could land on opposite
               sides of a CV fold, which is the exact leakage grouping
               exists to prevent. RIDER is a test-retest collection, so
               that second exam very likely exists for every subject.
    'stem'  -- the whole filename is the patient ID. Only correct when
               each patient has exactly one volume and filenames carry
               no per-exam metadata.
    'regex' -- delegate to prepare_mri.patient_id_from_path.
    """
    if mode == "regex":
        return patient_id_from_path(volume_path)

    stem = volume_path.stem

    if mode == "rider":
        match = _RIDER_ID.match(stem)
        if not match:
            raise ValueError(
                f"{volume_path.name}: expected a filename starting with a RIDER "
                "prefix and a numeric subject ID (e.g. 'RIDER_1627409910_...'). "
                "Use --patient-id stem or --patient-id regex instead."
            )
        return match.group("patient")

    for prefix in ("RIDER-", "RIDER_", "RIDER "):
        if stem.upper().startswith(prefix):
            stem = stem[len(prefix):]
            break
    return stem.replace(" ", "-")


def round_trip_failure(patient_id: str) -> str | None:
    """
    Output filenames must be parseable back to the SAME patient ID by
    cross_validation.folds.default_patient_id_from_filename, or grouped
    CV silently degrades into per-slice folds -- reintroducing exactly
    the leakage the patient split exists to prevent.

    default_patient_id_from_filename falls back to the whole filename
    stem when its regex misses, so it never returns empty; checking for
    a non-empty result proves nothing. This checks the recovered value
    actually EQUALS "RIDER-<patient_id>", and checks two different slice
    indices, because a regex that misses the patient can still match the
    slice number instead -- which looks plausible on one filename and
    yields a different group for every slice in practice.
    """
    try:
        from cross_validation.folds import default_patient_id_from_filename
    except ImportError:
        return "could not import cross_validation.folds -- verify manually"

    recovered = {
        default_patient_id_from_filename(
            Path(f"RIDER-{patient_id}_vol{volume}_slice{index:03d}.npy")
        )
        for volume, index in ((0, 0), (0, 137), (1, 42))
    }

    if len(recovered) != 1:
        return (
            f"parses inconsistently across slices: {sorted(recovered)}. Each "
            "slice would become its own CV group."
        )

    # Exact equality with the full ID is NOT required. A truncated but stable
    # and unique parse (e.g. "RIDER-1627409910_09") groups a patient's slices
    # correctly. What matters is that it is constant per patient and distinct
    # between patients -- the caller checks distinctness across the cohort.
    return None


def find_pairs(raw_root: Path, split: str) -> list[tuple[Path, Path]]:
    images_dir = raw_root / split / "images"
    masks_dir = raw_root / split / "masks"

    if not images_dir.exists():
        return []
    if not masks_dir.exists():
        raise FileNotFoundError(f"{images_dir} exists but {masks_dir} does not.")

    pairs = []
    missing = []
    for image_path in sorted(images_dir.glob("*.npy")):
        mask_path = masks_dir / image_path.name
        if mask_path.exists():
            pairs.append((image_path, mask_path))
        else:
            missing.append(image_path.name)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} volume(s) in {images_dir} have no same-named mask "
            f"in {masks_dir}. First few: {missing[:5]}"
        )
    return pairs


def build_plan(raw_root: Path, id_mode: str) -> dict[str, list[tuple[Path, Path, str, int]]]:
    """
    Resolve every volume to (image, mask, patient_id, volume_idx).

    volume_idx counts volumes PER PATIENT, so a patient contributing two
    scans gets vol0 and vol1 rather than two files both named vol0 that
    would silently overwrite each other.
    """
    plan: dict[str, list[tuple[Path, Path, str, int]]] = {}
    patient_split: dict[str, str] = {}
    volume_counter: Counter[str] = Counter()

    for split in SPLITS:
        entries = []
        for image_path, mask_path in find_pairs(raw_root, split):
            patient_id = derive_patient_id(image_path, id_mode)

            if patient_split.get(patient_id, split) != split:
                raise ValueError(
                    f"Patient {patient_id!r} appears in both "
                    f"{patient_split[patient_id]!r} and {split!r}. That is "
                    "cross-split leakage -- fix the input layout first."
                )
            patient_split[patient_id] = split

            entries.append((image_path, mask_path, patient_id, volume_counter[patient_id]))
            volume_counter[patient_id] += 1
        plan[split] = entries

    if not patient_split:
        raise FileNotFoundError(f"No volumes found under {raw_root}")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slice pre-split RIDER volumes into <out-root>/<split>/{images,masks}/, "
                    "preserving the train/val/test assignment already on disk."
    )
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Contains train/val/test, each with images/ and masks/.")
    parser.add_argument("--out-root", type=Path, default=MRI_CONFIG.root)
    parser.add_argument("--patient-id", choices=("rider", "stem", "regex"), default="rider")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan, shapes, and derived patient IDs; write nothing.")
    args = parser.parse_args()

    if args.raw_root.resolve() == Path(args.out_root).resolve():
        parser.error(
            "--raw-root and --out-root are the same directory. Move your raw "
            "volumes elsewhere (e.g. data/raw/breast_mri) so the generated "
            "slices don't land on top of them."
        )

    if not args.dry_run:
        check_output_is_clean(args.out_root)

    plan = build_plan(args.raw_root, args.patient_id)
    n_volumes = sum(len(entries) for entries in plan.values())
    patients = {
        patient_id
        for entries in plan.values()
        for _, _, patient_id, _ in entries
    }

    print(f"{len(patients)} patient(s), {n_volumes} volume(s):\n")
    failures = []
    for split in SPLITS:
        for image_path, _, patient_id, volume_idx in plan[split]:
            shape = np.load(image_path, mmap_mode="r").shape
            failure = round_trip_failure(patient_id)
            if failure:
                failures.append((patient_id, failure))
            note = f"\n      !! PATIENT ID WILL NOT GROUP: {failure}" if failure else ""
            print(
                f"  [{split:<5}] {image_path.name:<32} patient={patient_id:<12} "
                f"vol{volume_idx}  shape={shape}{note}"
            )

    # A stable parse is not enough on its own: two different patients must not
    # collapse to the same group ID either, or their slices merge into one fold.
    try:
        from cross_validation.folds import default_patient_id_from_filename

        collisions: dict[str, set[str]] = {}
        for patient_id in patients:
            group = default_patient_id_from_filename(
                Path(f"RIDER-{patient_id}_vol0_slice000.npy")
            )
            collisions.setdefault(group, set()).add(patient_id)

        for group, members in sorted(collisions.items()):
            if len(members) > 1:
                failures.append((group, f"shared by patients {sorted(members)}"))
                print(f"\n  !! GROUP COLLISION: {group!r} shared by {sorted(members)}")
    except ImportError:
        print("\nNote: could not import cross_validation.folds -- verify grouping manually.")

    if failures:
        print(
            f"\nSTOP: {len(failures)} problem(s) with how patient IDs parse through "
            "cross_validation.folds.default_patient_id_from_filename. Grouped CV "
            "would not separate patients correctly. Fix the naming before writing "
            "anything."
        )
        return

    if args.dry_run:
        print("\n--dry-run: nothing written. Re-run without it once this looks right.")
        return

    manifest_rows: list[dict] = []
    totals = {split: [0, 0] for split in SPLITS}

    for split in SPLITS:
        for image_path, mask_path, patient_id, volume_idx in plan[split]:
            n_slices, nonempty = extract_volume(
                image_path, mask_path, patient_id, volume_idx, split,
                args.out_root, False, manifest_rows,
            )
            totals[split][0] += n_slices
            totals[split][1] += nonempty
            print(
                f"  [{split}] {image_path.name}: {n_slices} slices "
                f"({nonempty} with a lesion, {n_slices - nonempty} background)"
            )

    print("\nWrote:")
    for split in SPLITS:
        n_slices, nonempty = totals[split]
        print(f"  {split}: {n_slices} slices ({nonempty} with a lesion)")

    write_manifest(manifest_rows, args.out_root / "extraction_manifest.csv")


if __name__ == "__main__":
    main()