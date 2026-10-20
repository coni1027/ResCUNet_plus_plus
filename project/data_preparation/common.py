"""
Shared bookkeeping for prepare_mammograms.py / prepare_mri.py: pairing
images with masks, checking no patient crosses splits, refusing to
overwrite old output, and writing the traceability manifest.

Raw input layout (split already decided on disk):
    <raw-root>/<split>/images/<name><suffix>
    <raw-root>/<split>/masks/<name><suffix>   # same filename as its image

Only used while preparing data -- training never imports this.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

SPLITS = ("train", "val", "test")


def _find_only(base: Path, name: str) -> Path:
    """Locate the one `name` directory under `base`, at any depth (zip exports nest oddly)."""
    matches = [path for path in base.rglob(name) if path.is_dir()]
    if not matches:
        raise FileNotFoundError(f"No '{name}' directory found under {base}")
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple '{name}' directories found under {base}: {matches}. "
            "Narrow --raw-root down to a single split export."
        )
    return matches[0]


def find_pairs(raw_root: Path, split: str, suffix: str) -> list[tuple[Path, Path]]:
    """
    Pairs every image in <raw-root>/<split>/images with the same-named
    file in masks/. Raises if ANY image has no mask, rather than silently
    dropping it -- a dropped file is a patient quietly missing from the data.
    """
    split_root = raw_root / split
    if not split_root.exists():
        return []

    images_dir = _find_only(split_root, "images")
    masks_dir = _find_only(split_root, "masks")

    pairs = []
    missing = []
    for image_path in sorted(images_dir.glob(f"*{suffix}")):
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


def build_plan(
    raw_root: Path,
    suffix: str,
    patient_id_fn: Callable[[Path], str],
) -> dict[str, list[tuple[Path, Path, str]]]:
    """
    Resolve every pair to (image, mask, patient_id) per split, with the
    split coming from the folder it already sits in. Raises if the same
    patient shows up under more than one split -- cross-split leakage.
    """
    plan: dict[str, list[tuple[Path, Path, str]]] = {}
    patient_split: dict[str, str] = {}

    for split in SPLITS:
        entries = []
        for image_path, mask_path in find_pairs(raw_root, split, suffix):
            patient_id = patient_id_fn(image_path)

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


def check_output_is_clean(out_root: Path) -> None:
    """
    Refuses to proceed if any split's images/ directory already has
    files in it, so old and newly-generated data can never end up mixed
    together silently.
    """
    for split_name in SPLITS:
        images_dir = out_root / split_name / "images"
        if images_dir.exists() and any(images_dir.iterdir()):
            raise FileExistsError(
                f"{images_dir} already has files in it. Remove {out_root} "
                "(or point --out-root elsewhere) before re-running, so "
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
