"""
Training loop, checkpointing, and history logging.

run_training_loop() is the shared, model-agnostic training loop: it takes
an already-constructed model plus explicit paths/labels, and is used by:
  - train_model()          -- the normal entry point, model looked up by
                               name in models.MODEL_REGISTRY, trained on
                               config's fixed train/val/test split.
  - experiments/run_ablation.py -- constructs CBAM/deep-supervision
                               variants of the proposed model directly,
                               bypassing the registry.
  - run_kfold_training() (this file) -- trains one fresh model per
                               cross-validation fold, for the robustness
                               check that comes AFTER loss-weight tuning
                               (see tuning/bayesian.py and
                               experiments/run_kfold_cv.py).
All three share the exact same loop, logging, and checkpoint format.

Two checkpoints are kept per run, side by side:
  - "..._best.pth" -- the epoch with the highest validation Dice so far.
                      This is what gets loaded for final test-set
                      evaluation and is the one you want for actually
                      using/reporting the model.
  - "..._last.pth" -- overwritten after EVERY epoch (model, optimizer,
                      and LR-scheduler state), purely so a halted run can
                      resume. run_training_loop() automatically resumes
                      from "..._last.pth" if it exists when called again
                      with the same checkpoint_path -- no separate
                      --resume flag needed, since every entry point
                      derives checkpoint_path from (config.name,
                      model_name/label[, fold_idx]) the same way every
                      time, so re-running the same command finds it.

checkpoint_path is derived only from (config.name, model_name/label[,
fold_idx]) -- NOT from bce_weight/dice_weight or ablation flags. So two
calls that reuse the same model_name/config but pass DIFFERENT
hyperparameters (e.g. re-running experiments/run_kfold_cv.py with a
different --bce-weight/--dice-weight, or re-tuning with a different
--tuning-method) would otherwise collide on the same "..._last.pth" and
silently resume across the mismatch -- continuing training under the
OLD run's settings, or, if that old run had already finished, skipping
training entirely and returning its stale metrics with no error.
resume_mismatches() is the guard against this: before resuming,
run_training_loop() checks the loaded checkpoint's bce_weight/
dice_weight (and any extra_checkpoint_fields, e.g. ablation flags)
against what THIS call was given, and starts fresh instead of resuming
if anything differs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from config import CHECKPOINT_DIR, DatasetConfig, DEVICE, NUM_WORKERS, RESULTS_DIR, SEED, set_seed
from cross_validation.folds import pooled_trainval_samples
from datasets.breast_dataset import BreastSegmentationDataset, create_loaders
from models import MODEL_LABELS, MODEL_REGISTRY
from training.evaluator import evaluate
from training.losses import BCEDiceLoss, DeepSupervisionLoss


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    best_val_dice: float,
    val_metrics: dict[str, float],
    config: DatasetConfig,
    label: str,
    bce_weight: float,
    dice_weight: float,
    extra_fields: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_dice": best_val_dice,
        "val_metrics": val_metrics,
        "dataset_name": config.name,
        "label": label,
        "input_size": config.input_size,
        "in_channels": config.in_channels,
        "bce_weight": bce_weight,
        "dice_weight": dice_weight,
    }
    if extra_fields:
        payload.update(extra_fields)
    torch.save(payload, path)


def last_checkpoint_path(checkpoint_path: Path) -> Path:
    """
    Derive the "resume" checkpoint's path from the "best" one, e.g.
    ".../mri_resunetpp_cbam_best.pth" -> ".../mri_resunetpp_cbam_last.pth".
    Every checkpoint_path in this codebase is built with a "_best.pth"
    suffix (see train_model(), run_kfold_training(), run_ablation.py), so
    the plain suffix swap below always applies; the fallback only
    matters if a caller ever passes a differently-named path.
    """
    if checkpoint_path.name.endswith("_best.pth"):
        return checkpoint_path.with_name(checkpoint_path.name[: -len("_best.pth")] + "_last.pth")
    return checkpoint_path.with_name(checkpoint_path.stem + "_last" + checkpoint_path.suffix)


def resume_mismatches(
    resume: dict,
    bce_weight: float,
    dice_weight: float,
    extra_checkpoint_fields: dict | None,
) -> list[str]:
    """
    Compare a loaded "_last.pth" payload against the hyperparameters THIS
    call was actually given, and return a human-readable mismatch per
    differing field (empty list means safe to resume).

    Why this check exists: checkpoint_path/last_checkpoint_path are
    derived only from (config.name, model_name/label[, fold_idx]) --
    never from bce_weight/dice_weight or ablation flags (see
    last_checkpoint_path()'s docstring). So two calls that reuse the same
    model_name/config but pass DIFFERENT loss weights -- e.g. re-running
    experiments/run_kfold_cv.py with a different --bce-weight/
    --dice-weight, or re-tuning with a different --tuning-method, or
    simply re-tuning at all -- collide on the same checkpoint path.
    Resuming across such a mismatch would silently continue training
    under the OLD run's loss composition instead of the one just
    requested, or -- if the old run had already finished -- skip training
    entirely and return the old run's stale metrics with no error. This
    check is what makes run_training_loop() refuse that and start fresh
    instead whenever the checkpoint doesn't actually match this call.
    """
    mismatches = []
    resume_bce = resume.get("bce_weight")
    resume_dice = resume.get("dice_weight")
    if resume_bce != bce_weight or resume_dice != dice_weight:
        mismatches.append(
            f"bce_weight/dice_weight: checkpoint has {resume_bce}/{resume_dice}, "
            f"this call requests {bce_weight:.4f}/{dice_weight:.4f}"
        )
    for key, value in (extra_checkpoint_fields or {}).items():
        if key in resume and resume[key] != value:
            mismatches.append(f"{key}: checkpoint has {resume[key]!r}, this call requests {value!r}")
    return mismatches


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def load_history(path: Path) -> list[dict[str, float]]:
    """Read back a history CSV written by write_history(), so a resumed
    run can keep appending to the same in-memory list instead of
    overwriting earlier epochs' rows with only the new ones."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = []
        for raw_row in reader:
            rows.append({
                key: int(value) if key == "epoch" else float(value)
                for key, value in raw_row.items()
            })
        return rows


def run_training_loop(
    model: nn.Module,
    label: str,
    config: DatasetConfig,
    checkpoint_path: Path,
    history_path: Path,
    bce_weight: float,
    dice_weight: float,
    epochs: int,
    extra_checkpoint_fields: dict | None = None,
    loaders: tuple[DataLoader, DataLoader, DataLoader | None] | None = None,
) -> dict[str, float]:
    """
    Train `model` for `epochs` epochs, checkpoint on best validation Dice,
    then report final metrics. Model-agnostic: works for any nn.Module
    whose forward() returns a single logits tensor or a list of them.

    loaders: (train_loader, val_loader, test_loader) to use instead of
    config's fixed on-disk split -- run_kfold_training() below passes a
    fold's own (train, val, None) here. When test_loader is None, there's
    no separate held-out set for this call (a CV fold only has a
    train/val split, not a third test partition), so the returned
    metrics are the *best epoch's validation* metrics rather than a true
    held-out test score. Omit `loaders` entirely (the default) to use
    config's fixed data/<name>/{train,val,test} split, exactly as before.

    Resume: if last_checkpoint_path(checkpoint_path) already exists (a
    previous call to this exact checkpoint_path was interrupted) AND its
    bce_weight/dice_weight/extra_checkpoint_fields match what THIS call
    was given (see resume_mismatches()), model/optimizer/scheduler state
    and the epoch counter are restored from it before the loop starts,
    and history_path's existing rows are reloaded so the CSV keeps
    growing instead of being overwritten from epoch 1. Nothing else needs
    to change to resume -- re-run the same command. If the checkpoint
    exists but doesn't match (different loss weights, different ablation
    flags, ...), it's ignored and training starts fresh instead, so a
    stale checkpoint from an unrelated run is never silently continued.
    """
    if loaders is None:
        train_loader, val_loader, test_loader = create_loaders(config)
    else:
        train_loader, val_loader, test_loader = loaders

    base_loss = BCEDiceLoss(bce_weight, dice_weight)
    criterion = DeepSupervisionLoss(base_loss)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    start_epoch = 1
    best_val_dice = -1.0
    best_val_metrics: dict[str, float] | None = None
    history: list[dict[str, float]] = []

    resume_path = last_checkpoint_path(checkpoint_path)
    if resume_path.exists():
        resume = torch.load(resume_path, map_location=DEVICE)
        mismatches = resume_mismatches(resume, bce_weight, dice_weight, extra_checkpoint_fields)
        if mismatches:
            print(
                f"[{label}] Found {resume_path}, but it doesn't match this call -- "
                "starting fresh instead of resuming (see resume_mismatches()'s "
                "docstring for why this matters):"
            )
            for mismatch in mismatches:
                print(f"    {mismatch}")
        else:
            model.load_state_dict(resume["model_state_dict"])
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            scheduler.load_state_dict(resume["scheduler_state_dict"])
            start_epoch = resume["epoch"] + 1
            best_val_dice = resume["best_val_dice"]
            history = load_history(history_path)
            print(
                f"[{label}] Resuming from {resume_path}: "
                f"starting at epoch {start_epoch}/{epochs} "
                f"(best_val_dice so far: {best_val_dice:.4f})"
            )

    print("=" * 80)
    print(f"Training: {config.name} | {label}")
    print(f"Device: {DEVICE}")
    print(f"Input size: {config.input_size} | batch size: {config.batch_size}")
    print(f"Loss weights -> bce_weight={bce_weight:.4f}, dice_weight={dice_weight:.4f}")
    print("=" * 80)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_metrics["dice"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        write_history(history_path, history)

        print(
            f"[{label}] Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"dice={val_metrics['dice']:.4f} | "
            f"iou={val_metrics['iou']:.4f} | "
            f"precision={val_metrics['precision']:.4f} | "
            f"recall={val_metrics['recall']:.4f}"
        )

        is_new_best = val_metrics["dice"] > best_val_dice
        if is_new_best:
            best_val_dice = val_metrics["dice"]
            best_val_metrics = val_metrics

        # Saved every epoch (overwriting the previous one) purely so a
        # halted run can resume -- see last_checkpoint_path()'s docstring.
        # best_val_dice/val_metrics here already reflect THIS epoch (the
        # update above runs first), so a resume never re-derives a stale
        # "best so far" that's one epoch behind.
        save_checkpoint(
            resume_path, model, optimizer, scheduler, epoch, best_val_dice,
            val_metrics, config, label, bce_weight, dice_weight,
            extra_fields=extra_checkpoint_fields,
        )

        if is_new_best:
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, epoch, best_val_dice,
                val_metrics, config, label, bce_weight, dice_weight,
                extra_fields=extra_checkpoint_fields,
            )
            print(f"  Saved new best checkpoint: {checkpoint_path}")

    if best_val_metrics is None and checkpoint_path.exists():
        # Either every epoch was already completed by a prior run (this
        # call's loop above ran zero iterations), or epochs <= 0 was
        # passed -- either way, recover the best metrics already on disk
        # instead of leaving best_val_metrics unset.
        existing_best = torch.load(checkpoint_path, map_location=DEVICE)
        best_val_dice = existing_best["best_val_dice"]
        best_val_metrics = existing_best["val_metrics"]

    if test_loader is not None:
        # Evaluate the best model on the held-out test split.
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, criterion, DEVICE)

        print(f"\nBest validation Dice ({label}):", checkpoint["best_val_dice"])
        print(f"Held-out test metrics for {config.name} / {label}:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")

        return test_metrics

    print(
        f"\nBest validation Dice ({label}): {best_val_dice:.4f} "
        "(no held-out test set for this call -- returning its full metrics)"
    )
    assert best_val_metrics is not None
    return best_val_metrics


def train_model(
    model_name: str,
    config: DatasetConfig,
    bce_weight: float | None = None,
    dice_weight: float | None = None,
    epochs: int | None = None,
) -> dict[str, float]:
    """
    Build `model_name` from the registry (see models/__init__.py) and
    train it on `config` via run_training_loop().

    bce_weight / dice_weight override the DatasetConfig defaults when
    given -- this is how Bayesian-optimized weights from
    tuning.bayesian.tune_bce_dice_weight() get plugged into the full
    training run.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name {model_name!r}. Options: {sorted(MODEL_REGISTRY)}")

    set_seed(SEED)

    resolved_bce_weight = config.bce_weight if bce_weight is None else bce_weight
    resolved_dice_weight = config.dice_weight if dice_weight is None else dice_weight
    resolved_epochs = config.epochs if epochs is None else epochs

    model = MODEL_REGISTRY[model_name](config)
    label = MODEL_LABELS.get(model_name, model_name)

    checkpoint_path = CHECKPOINT_DIR / f"{config.name}_{model_name}_best.pth"
    history_path = RESULTS_DIR / f"{config.name}_{model_name}_training_history.csv"

    return run_training_loop(
        model, label, config, checkpoint_path, history_path,
        resolved_bce_weight, resolved_dice_weight, resolved_epochs,
        extra_checkpoint_fields={"model_name": model_name},
    )


def run_kfold_training(
    model_name: str,
    config: DatasetConfig,
    folds: list[tuple[np.ndarray, np.ndarray]],
    bce_weight: float,
    dice_weight: float,
    epochs: int | None = None,
) -> list[dict[str, float]]:
    """
    Train one fresh model per fold in `folds` (as produced by
    cross_validation.folds.get_or_create_folds over
    cross_validation.folds.pooled_trainval_samples), using the SAME fixed
    bce_weight/dice_weight for every fold.

    This is the "do the folds" step your adviser wants run AFTER
    tuning.bayesian.tune_bce_dice_weight() -- bce_weight/dice_weight are
    already decided by the time this runs, and are not re-searched per
    fold. See experiments/run_kfold_cv.py for the CLI wrapper that wires
    this together with tuning and result aggregation.

    Returns one metrics dict per fold. Each fold's dict is that fold's
    best validation-epoch metrics (see run_training_loop's `loaders`
    docstring for why these aren't "test" metrics -- a fold only has a
    train/val split, not a third partition). Aggregate the returned list
    yourself (mean/std) for a robust final estimate; the global `test`
    split used by train_model() is never touched here.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name {model_name!r}. Options: {sorted(MODEL_REGISTRY)}")
    if len(folds) < 2:
        raise ValueError(f"Need at least 2 folds for cross-validation; got {len(folds)}.")

    label = MODEL_LABELS.get(model_name, model_name)
    resolved_epochs = config.epochs if epochs is None else epochs

    samples = pooled_trainval_samples(config)
    augmented_view = BreastSegmentationDataset(config, augment=True, samples=samples)
    plain_view = BreastSegmentationDataset(config, augment=False, samples=samples)
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
    }

    fold_metrics: list[dict[str, float]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        set_seed(SEED)  # Same init recipe every fold, isolating the data split's effect.
        model = MODEL_REGISTRY[model_name](config)

        fold_train_loader = DataLoader(Subset(augmented_view, train_idx), shuffle=True, **loader_kwargs)
        fold_val_loader = DataLoader(Subset(plain_view, val_idx), shuffle=False, **loader_kwargs)

        checkpoint_path = CHECKPOINT_DIR / f"{config.name}_{model_name}_fold{fold_idx}_best.pth"
        history_path = RESULTS_DIR / f"{config.name}_{model_name}_fold{fold_idx}_history.csv"

        print(f"\n{'#' * 80}\n# Fold {fold_idx + 1}/{len(folds)}\n{'#' * 80}")
        metrics = run_training_loop(
            model, f"{label} (fold {fold_idx})", config, checkpoint_path, history_path,
            bce_weight, dice_weight, resolved_epochs,
            extra_checkpoint_fields={"model_name": model_name, "fold_idx": fold_idx},
            loaders=(fold_train_loader, fold_val_loader, None),
        )
        fold_metrics.append(metrics)

    return fold_metrics
