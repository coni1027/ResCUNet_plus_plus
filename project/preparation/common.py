"""
Shared utilities for this package's one-time data-preparation scripts
(prepare_mri.py, prepare_mammograms.py): patient-level splitting, split
caching, a pre-flight safety check, and manifest writing.

Nothing here is imported by the actual training pipeline -- these exist
purely to turn raw downloaded data into the per-split, per-sample layout
datasets/breast_dataset.py expects, once, before training starts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def split_patients(
    patient_ids: list[str],
    seed: int,
    ratios: dict[str, float],
) -> dict[str, list[str]]:
    """
    Assigns every unique patient ID to exactly one named group in
    `ratios` -- e.g. {"train": 0.6, "val": 0.2, "test": 0.2} for RIDER's
    Methodology-3.2 split, or {"train": 0.8, "val": 0.2} for CBIS-DDSM's
    train-set carve-out -- seeded for reproducibility.

    Every group except the last gets round(n * its own ratio) patients;
    the LAST group (in dict insertion order) absorbs whatever remains,
    so rounding can never drop or duplicate a patient regardless of how
    many groups there are or how small n is.
    """
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("patient_ids contains duplicates -- deduplicate before splitting.")

    rng = np.random.default_rng(seed)
    shuffled = list(patient_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    names = list(ratios)
    assignment: dict[str, list[str]] = {}
    cursor = 0
    for name in names[:-1]:
        count = min(round(n * ratios[name]), n - cursor)
        assignment[name] = sorted(shuffled[cursor:cursor + count])
        cursor += count
    assignment[names[-1]] = sorted(shuffled[cursor:])
    return assignment


def get_or_create_patient_split(
    patient_ids: list[str],
    seed: int,
    ratios: dict[str, float],
    path: Path,
) -> dict[str, list[str]]:
    """
    Loads a previously written patient split if one exists at `path`, so
    re-running a preparation script never silently reshuffles which
    patients land in which group. Computes and saves one, seeded, if not.

    Raises if the cached patient list no longer matches the one passed
    in -- e.g. because a raw file was added or removed since the cache
    was written -- rather than silently reusing a split built from a
    different set of patients.
    """
    group_names = list(ratios)
    if path.exists():
        saved = json.loads(path.read_text())
        saved_ids = sorted(pid for ids in saved.values() for pid in ids)
        if saved_ids != sorted(patient_ids):
            raise ValueError(
                f"Cached split at {path} was built from a different patient "
                f"list ({saved_ids}) than what's on disk now ({sorted(patient_ids)}). "
                "Delete the cache to recompute -- only after confirming which "
                "patient list is actually correct."
            )
        print(f"Loaded cached patient split from {path}")
        return {name: saved[name] for name in group_names}

    split = split_patients(patient_ids, seed, ratios)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2))
    print(f"Computed and saved patient split to {path}")
    return split


def check_output_is_clean(out_root: Path, split_names: tuple[str, ...] = ("train", "val", "test")) -> None:
    """
    Refuses to proceed if any split's images/ directory already has
    files in it, so old and newly-generated data can never end up mixed
    together silently.
    """
    for split_name in split_names:
        images_dir = out_root / split_name / "images"
        if images_dir.exists() and any(images_dir.iterdir()):
            raise FileExistsError(
                f"{images_dir} already has files in it. Remove {out_root} "
                "(or point --output-root elsewhere) before re-running, so "
                "old and new data/splits can't end up mixed together."
            )


def write_manifest(rows: list[dict], path: Path) -> None:
    """
    Writes a per-sample manifest CSV -- full traceability back to the
    raw source files, and a quick way to check class balance per split
    afterward. No-ops on an empty row list.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved manifest: {path}")
