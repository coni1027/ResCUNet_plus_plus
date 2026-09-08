"""
Cross-validation utilities for the k-fold robustness check
(experiments/run_kfold_cv.py, via training.trainer.run_kfold_training).

BCE/Dice weight tuning (tuning/bayesian.py) does NOT use this module --
it searches on a single fixed split, deliberately kept separate from
(and run before) cross-validation. See tuning/bayesian.py's module
docstring for why. This module's folds are for training.trainer.
run_kfold_training(): one fresh model per fold, using loss weights
tuning already chose, to check how much that choice (and the model
itself) varies across which patients/slices land in val.

Either way, this has nothing to do with, and does not change, the fixed
train/val/test split used for final model training and evaluation
(Methodology section 3.2) -- that split is untouched and lives in
datasets/breast_dataset.py's create_loaders().
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from config import DatasetConfig, SEED, SPLITS_DIR
from datasets.breast_dataset import BreastSegmentationDataset, paired_samples


# -----------------------------------------------------------------------------
# Pooling train+val for cross-validation
# -----------------------------------------------------------------------------

def pooled_trainval_samples(config: DatasetConfig) -> list[tuple[Path, Path]]:
    """
    Combine the `train` and `val` split samples into one pool for the
    k-fold CV robustness check (experiments/run_kfold_cv.py), run AFTER
    loss-weight tuning has already picked bce_weight/dice_weight on a
    single split -- this pool is not used during tuning itself. The
    `test` split is deliberately excluded here and stays held out for
    the final, once-only evaluation in training.trainer.train_model() --
    CV folds are only ever carved out of data that was already earmarked
    for training/tuning under the methodology's train/val/test split.
    """
    samples: list[tuple[Path, Path]] = []
    for split in ("train", "val"):
        split_dir = config.root / split
        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"

        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(
                f"Missing {config.name} split: {split_dir}\n"
                "Expected <split>/images and <split>/masks directories."
            )

        samples.extend(paired_samples(images_dir, masks_dir))

    if not samples:
        raise ValueError(f"No paired train+val samples found for {config.name}.")

    return samples


# -----------------------------------------------------------------------------
# Patient/case ID heuristic (opt-in via DatasetConfig.patient_id_fn)
# -----------------------------------------------------------------------------

_PATIENT_ID_PATTERN = re.compile(
    r"(?P<patient>P[_-]?\d{3,}|RIDER[\s_-]?[A-Za-z0-9]+[_-]?\d+|\d{3,})",
    re.IGNORECASE,
)


def default_patient_id_from_filename(image_path: Path) -> str:
    """
    Best-effort patient/case ID guess from a filename -- a starting point
    for grouped (patient-safe) CV folds, NOT a validated parser for CBIS-
    DDSM's or your exported RIDER files' actual naming convention.

    Intended to handle filenames along the lines of:
      "Mass-Training_P_00016_LEFT_CC_1.png" -> "P_00016"
      "RIDER-1023_slice014.png"             -> "RIDER-1023"
      "1023_042.png"                        -> "1023"

    VERIFY this against your real filenames (e.g. via preview_cv_groups())
    before trusting grouped folds. If it can't find a match it falls back
    to the full filename stem, which is equivalent to no grouping at all
    for that one file -- silently defeating patient-level separation for
    it, so a mismatch here won't necessarily raise an error.
    """
    match = _PATIENT_ID_PATTERN.search(image_path.stem)
    return match.group("patient") if match else image_path.stem


# -----------------------------------------------------------------------------
# Grouped K-fold
# -----------------------------------------------------------------------------

def make_group_folds(
    n_samples: int,
    n_folds: int,
    seed: int,
    groups: Sequence[str] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Partition `n_samples` sample indices into up to `n_folds` (train_idx,
    val_idx) pairs for cross-validation -- a hand-rolled equivalent of
    sklearn.model_selection.GroupKFold, written here to avoid adding
    scikit-learn as a dependency just for this.

    If `groups` is given (e.g. one patient ID per sample), every sample
    sharing the same group value is guaranteed to land in the same fold --
    this is what keeps a patient's slices from being split across the
    train/val sides of a fold. Fold sizes are balanced by total SAMPLE
    count (via a greedy fill-the-smallest-fold assignment), not just group
    count, since group sizes (slices per patient) can differ a lot. If
    `groups` is None, each sample is its own group (ordinary per-sample
    K-fold) -- fine for i.i.d. data, but not patient-safe for RIDER.

    Uses its own seeded numpy Generator, independent of the global
    torch/numpy/random seed set by config.set_seed(), so fold assignment
    is fixed once per tuning run regardless of how often set_seed() is
    called later for model re-initialization.

    Returns fewer than `n_folds` folds if fewer than `n_folds` unique
    groups exist (e.g. RIDER's ~4 non-test patients) -- this is how
    leave-one-patient-out CV naturally falls out of a generic "K-fold"
    request once grouping is enabled.
    """
    if groups is not None and len(groups) != n_samples:
        raise ValueError(
            f"groups has length {len(groups)}, expected {n_samples} (one per sample)."
        )

    rng = np.random.default_rng(seed)
    group_ids = np.arange(n_samples) if groups is None else np.asarray(groups)

    unique_groups = np.unique(group_ids)
    rng.shuffle(unique_groups)

    effective_folds = max(1, min(n_folds, len(unique_groups)))

    fold_sample_indices: list[list[int]] = [[] for _ in range(effective_folds)]
    fold_sizes = [0] * effective_folds

    for group in unique_groups:
        member_indices = np.where(group_ids == group)[0]
        target_fold = int(np.argmin(fold_sizes))
        fold_sample_indices[target_fold].extend(member_indices.tolist())
        fold_sizes[target_fold] += len(member_indices)

    all_indices = np.arange(n_samples)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in fold_sample_indices:
        val_idx = np.array(sorted(fold))
        train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=True)
        folds.append((train_idx, val_idx))

    return folds


# -----------------------------------------------------------------------------
# Persisting folds to splits/ (so tuning, previews, and reruns agree)
# -----------------------------------------------------------------------------

def save_folds(
    folds: list[tuple[np.ndarray, np.ndarray]],
    path: Path,
    n_samples: int,
    groups: Sequence[str] | None = None,
) -> None:
    """Serialize a fold assignment to JSON, tagged with the sample count
    it was built from (see load_folds/get_or_create_folds)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_samples": n_samples,
        "n_folds": len(folds),
        "folds": [
            {"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()}
            for train_idx, val_idx in folds
        ],
    }
    if groups is not None:
        payload["groups"] = list(groups)
    path.write_text(json.dumps(payload, indent=2))


def load_folds(path: Path) -> tuple[list[tuple[np.ndarray, np.ndarray]], int | None]:
    """
    Load a fold assignment previously written by save_folds().

    Returns (folds, n_samples_at_save_time) so the caller can check the
    cache still matches the current data before trusting it -- see
    get_or_create_folds().
    """
    payload = json.loads(path.read_text())
    folds = [
        (np.array(fold["train_idx"]), np.array(fold["val_idx"]))
        for fold in payload["folds"]
    ]
    return folds, payload.get("n_samples")


def get_or_create_folds(
    config: DatasetConfig,
    n_samples: int,
    n_folds: int,
    groups: Sequence[str] | None,
    splits_dir: Path = SPLITS_DIR,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Load a persisted fold split for (config.name, n_folds) from
    `splits_dir/{config.name}_{n_folds}fold.json` if one exists; otherwise
    compute it fresh via make_group_folds() and save it there.

    This is what lets experiments/run_kfold_cv.py and preview_cv_groups()
    (and any future script) agree on the *exact* same fold assignment
    across separate runs/processes, instead of trusting two independent
    seeded computations to match -- and it leaves a reviewable record of
    exactly which sample/patient landed in which fold.

    Note the file is keyed by n_folds, so requesting a different fold
    count produces (and caches) a different file rather than overwriting
    this one. The cache also records the sample count it was built from;
    if data/<name>/train or val has since gained or lost files, the stale
    cache is recomputed instead of silently reused with indices that no
    longer line up with pooled_trainval_samples()'s current output.
    """
    path = splits_dir / f"{config.name}_{n_folds}fold.json"
    if path.exists():
        folds, cached_n_samples = load_folds(path)
        if cached_n_samples == n_samples:
            print(f"[{config.name}] Loaded cached {len(folds)}-fold split from {path}")
            return folds
        print(
            f"[{config.name}] Cached split at {path} was built for "
            f"{cached_n_samples} samples, but {n_samples} are present now -- "
            "recomputing (data/ contents likely changed since the cache was written)."
        )

    folds = make_group_folds(n_samples, n_folds, seed=SEED, groups=groups)
    save_folds(folds, path, n_samples=n_samples, groups=groups)
    print(f"[{config.name}] Computed and saved {len(folds)}-fold split to {path}")
    return folds


# -----------------------------------------------------------------------------
# Free preview (no training) -- sanity-check grouping before spending GPU time
# -----------------------------------------------------------------------------

def preview_cv_groups(config: DatasetConfig, n_folds: int) -> None:
    """
    Print how pooled train+val samples would be grouped and split into CV
    folds, WITHOUT training anything -- a cheap sanity check for
    config.patient_id_fn (or the lack of one) before committing GPU time
    to experiments/run_kfold_cv.py. Uses/creates the same cached split
    that run will use, via get_or_create_folds().
    """
    samples = pooled_trainval_samples(config)
    n_samples = len(samples)

    groups: list[str] | None = None
    if config.patient_id_fn is not None:
        groups = [config.patient_id_fn(image_path) for image_path, _ in samples]

    folds = get_or_create_folds(config, n_samples, n_folds, groups)

    print(f"\n[{config.name}] Pooled train+val samples: {n_samples}")
    if groups is not None:
        unique = sorted(set(groups))
        print(f"[{config.name}] {len(unique)} group(s) via {config.patient_id_fn.__name__}: {unique}")
    else:
        print(
            f"[{config.name}] No patient_id_fn set -- grouping is per-sample "
            "(ordinary K-fold, NOT patient-safe)."
        )

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if groups is not None:
            val_groups = sorted({groups[i] for i in val_idx})
            print(
                f"  Fold {fold_idx}: train={len(train_idx)} samples, "
                f"val={len(val_idx)} samples, val_groups={val_groups}"
            )
        else:
            print(
                f"  Fold {fold_idx}: train={len(train_idx)} samples, "
                f"val={len(val_idx)} samples"
            )
