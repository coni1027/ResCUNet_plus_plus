"""
The full "optimize each model -> freeze -> cross-validate to compare"
protocol, per your adviser's instruction:

  1. For each model, tune its bce_weight/dice_weight independently on a
     single split -- NOT k-fold CV; see tuning.bayesian's module
     docstring for why tuning and cross-validation are kept separate and
     sequential rather than nested. --tuning-method selects the search:
     "grid" (default, tuning.grid_search -- deterministic sweep over
     --alphas, per adviser feedback that Bayesian's convergence is
     unpredictable) or "bayesian" (tuning.bayesian -- Optuna TPE over
     --n-trials trials).
  2. Freeze that model's tuned weights.
  3. Run k-fold CV with those frozen weights (one fresh model per fold,
     training.trainer.run_kfold_training), reporting mean +/- std of
     Dice/IoU/precision/recall/accuracy across folds.
  4. If more than one model is involved, compare them on the k-fold
     results -- each model at ITS OWN best-effort loss balance, not a
     shared fixed value.

Two entry points:
  - run_kfold_cv()         -- steps 2-3 for ONE model with weights you
                               already have (tuned or otherwise).
  - run_kfold_comparison()  -- the full steps 1-4 for one or more models;
                               this is what the CLI below calls, and
                               what "optimize each model, freeze, then
                               cross-validate to compare" means in code.

This is a DIFFERENT protocol from experiments/run_sota_comparison.py,
which by default gives every model the SAME fixed 0.5/0.5 loss balance
and trains each once on the fixed split -- a cheaper, controlled
comparison that isolates architecture differences, but not the one your
adviser asked for here. Use run_sota_comparison.py for that faster,
controlled question; use this file for "each model does its best."

The global `test` split (Methodology 3.2) is never touched by tuning or
by this script -- it stays reserved for the one-time check in
training.trainer.train_model(). This script only concerns the pooled
train+val portion.

Usage:
  # optimize -> freeze -> 5-fold CV for EVERY registered model, then compare
  python experiments/run_kfold_cv.py --dataset mri

  # same, but only a subset
  python experiments/run_kfold_cv.py --dataset mri --models resunetpp_cbam unet resunet

  # just one model, tuned then folded (no cross-model comparison table)
  python experiments/run_kfold_cv.py --dataset mri --models resunetpp_cbam

  # one model, skip tuning -- use loss weights you already know
  python experiments/run_kfold_cv.py --dataset mri --models resunetpp_cbam --bce-weight 0.4 --dice-weight 0.6

  # quick smoke test of the whole tune -> fold -> compare pipeline
  python experiments/run_kfold_cv.py --dataset mri --models unet resunet --n-folds 2 --epochs 2 --n-trials 2 --tuning-epochs 1
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DatasetConfig, MAMMOGRAM_CONFIG, MRI_CONFIG, RESULTS_DIR
from cross_validation.folds import (
    default_patient_id_from_filename,
    get_or_create_folds,
    pooled_trainval_samples,
)
from models import MODEL_LABELS, MODEL_REGISTRY
from training.trainer import run_kfold_training
from tuning import DEFAULT_TUNING_METHOD, TUNING_METHODS, tune_loss_weights
from tuning.grid_search import DEFAULT_ALPHAS


def run_kfold_cv(
    model_name: str,
    config: DatasetConfig,
    bce_weight: float,
    dice_weight: float,
    n_folds: int = 5,
    epochs: int | None = None,
) -> list[dict]:
    """Step 2-3 of the protocol above, for ONE model with weights already frozen."""
    samples = pooled_trainval_samples(config)
    n_samples = len(samples)

    groups: list[str] | None = None
    if config.patient_id_fn is not None:
        groups = [config.patient_id_fn(image_path) for image_path, _ in samples]
    else:
        print(
            f"WARNING: {config.name}'s patient_id_fn is not set, so its CV "
            "folds will be built per SAMPLE, not per patient. Section 3.2 of "
            "the methodology requires patient-level splits -- set "
            f"{config.name}.patient_id_fn (see "
            "cross_validation.folds.default_patient_id_from_filename for a "
            "starting point) and check with "
            "cross_validation.folds.preview_cv_groups() before trusting this."
        )

    folds = get_or_create_folds(config, n_samples, n_folds, groups)
    label = MODEL_LABELS.get(model_name, model_name)

    print(
        f"\n[{config.name}/{model_name}] Running {len(folds)}-fold CV with "
        f"FIXED bce_weight={bce_weight:.4f}, dice_weight={dice_weight:.4f} "
        "(already chosen by tuning -- not re-searched per fold)."
    )

    fold_dicts = run_kfold_training(model_name, config, folds, bce_weight, dice_weight, epochs)
    metric_keys = list(fold_dicts[0].keys())

    rows = [{"fold": i, **metrics} for i, metrics in enumerate(fold_dicts)]
    rows.append({"fold": "mean", **{k: statistics.mean(m[k] for m in fold_dicts) for k in metric_keys}})
    rows.append({
        "fold": "std",
        **{k: (statistics.stdev(m[k] for m in fold_dicts) if len(fold_dicts) > 1 else 0.0) for k in metric_keys},
    })

    out_path = RESULTS_DIR / f"{config.name}_{model_name}_kfold_cv.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", *metric_keys])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(fold_dicts)}-fold CV table: {out_path}")

    print(f"\n{label} -- {len(fold_dicts)}-fold CV results ({config.name}):")
    for key in metric_keys:
        values = [m[key] for m in fold_dicts]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"  {key:<12} {mean:.4f} +/- {std:.4f}")

    return rows


def run_kfold_comparison(
    model_names: list[str],
    config: DatasetConfig,
    n_folds: int = 5,
    n_trials: int = 15,
    tuning_epochs: int = 5,
    epochs: int | None = None,
    fixed_weights: dict[str, tuple[float, float]] | None = None,
    tuning_method: str = DEFAULT_TUNING_METHOD,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> list[dict]:
    """
    Steps 1-4 of the protocol above: for every model in model_names,
    tune (unless it's in `fixed_weights`, which skips tuning for that
    model and uses the given pair directly), freeze, run k-fold CV
    (run_kfold_cv), then collect all models' mean/std results into one
    comparison table -- sorted by mean Dice, best first.

    Each per-model fold breakdown is still written by run_kfold_cv() as
    usual; this additionally writes ONE combined summary CSV comparing
    every requested model side by side.
    """
    fixed_weights = fixed_weights or {}
    summary_rows: list[dict] = []

    for model_name in model_names:
        label = MODEL_LABELS.get(model_name, model_name)
        print(f"\n{'=' * 80}\n=== {config.name} / {label}: optimize -> freeze -> cross-validate ===\n{'=' * 80}")

        if model_name in fixed_weights:
            bce_weight, dice_weight = fixed_weights[model_name]
            print(
                f"[{config.name}/{model_name}] Using given fixed weights (skipping tuning): "
                f"bce_weight={bce_weight:.4f}, dice_weight={dice_weight:.4f}"
            )
        else:
            bce_weight, dice_weight = tune_loss_weights(
                tuning_method, model_name, config,
                n_trials=n_trials, tuning_epochs=tuning_epochs, alphas=alphas,
            )

        fold_rows = run_kfold_cv(model_name, config, bce_weight, dice_weight, n_folds=n_folds, epochs=epochs)
        mean_row = next(row for row in fold_rows if row["fold"] == "mean")
        std_row = next(row for row in fold_rows if row["fold"] == "std")
        metric_keys = [key for key in mean_row if key != "fold"]

        summary_rows.append({
            "model": model_name,
            "label": label,
            "bce_weight": bce_weight,
            "dice_weight": dice_weight,
            **{f"{key}_mean": mean_row[key] for key in metric_keys},
            **{f"{key}_std": std_row[key] for key in metric_keys},
        })

    summary_rows.sort(key=lambda row: row["dice_mean"], reverse=True)

    out_path = RESULTS_DIR / f"{config.name}_cv_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved cross-model CV comparison: {out_path}")

    if len(summary_rows) > 1:
        print(f"\n{config.name} -- each model tuned independently, then {n_folds}-fold CV'd (best Dice first):")
        header = f"{'Model':<34} {'bce/dice':>10} {'Dice':>20} {'IoU':>20}"
        print(header)
        for row in summary_rows:
            weights = f"{row['bce_weight']:.2f}/{row['dice_weight']:.2f}"
            dice = f"{row['dice_mean']:.4f} +/- {row['dice_std']:.4f}"
            iou = f"{row['iou_mean']:.4f} +/- {row['iou_std']:.4f}"
            print(f"{row['label']:<34} {weights:>10} {dice:>20} {iou:>20}")

    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize each model, freeze its weights, then cross-validate to compare "
                    "(tune first, then folds -- not nested)."
    )
    parser.add_argument("--dataset", choices=("mammogram", "mri", "both"), default="mri")
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY), choices=sorted(MODEL_REGISTRY),
        help=f"Models to optimize then cross-validate. Defaults to ALL registered models "
             f"(the full comparison). Options: {sorted(MODEL_REGISTRY)}",
    )
    parser.add_argument(
        "--bce-weight", type=float, default=None,
        help="Skip tuning and use this fixed BCE weight (only valid with exactly one --models entry).",
    )
    parser.add_argument(
        "--dice-weight", type=float, default=None,
        help="Skip tuning and use this fixed Dice weight (only valid with exactly one --models entry).",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--tuning-method", choices=TUNING_METHODS, default=DEFAULT_TUNING_METHOD,
        help="'grid' (default): deterministic sweep over --alphas -- predictable, adviser-recommended "
             "smokescreen. 'bayesian': Optuna TPE search over --n-trials trials (unpredictable convergence). "
             "Ignored for any model given via --bce-weight/--dice-weight.",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS),
        help=f"Grid of bce_weight values to try (only used with --tuning-method grid). Default: {list(DEFAULT_ALPHAS)}",
    )
    parser.add_argument(
        "--n-trials", type=int, default=15,
        help="Optuna trials per model during tuning (--tuning-method bayesian only; "
             "ignored for any model given via --bce-weight/--dice-weight).",
    )
    parser.add_argument(
        "--tuning-epochs", type=int, default=5,
        help="Epochs per tuning trial/grid point (ignored for any model given via --bce-weight/--dice-weight).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override each DatasetConfig's epoch count for every fold's training run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.bce_weight is None) != (args.dice_weight is None):
        raise SystemExit("Pass both --bce-weight and --dice-weight together, or neither (to tune every model).")
    if args.bce_weight is not None and len(args.models) != 1:
        raise SystemExit(
            "--bce-weight/--dice-weight only make sense with exactly one --models entry "
            "(each model needs its own tuned weights) -- omit them to tune every model listed."
        )

    # Patient-safe by default for both datasets, matching run_all.py -- verified
    # against RIDER's exported slice names and CBIS-DDSM's "<patient>_<side>_
    # <view>" export (see prepare_mammograms_presplit.py and config.py's
    # comment on MAMMOGRAM_CONFIG). Only backs off if a caller already set a
    # different patient_id_fn before calling main() (e.g. a test or script).
    if MRI_CONFIG.patient_id_fn is None:
        MRI_CONFIG.patient_id_fn = default_patient_id_from_filename
    if MAMMOGRAM_CONFIG.patient_id_fn is None:
        MAMMOGRAM_CONFIG.patient_id_fn = default_patient_id_from_filename

    fixed_weights = {args.models[0]: (args.bce_weight, args.dice_weight)} if args.bce_weight is not None else None

    configs = {
        "mammogram": [MAMMOGRAM_CONFIG],
        "mri": [MRI_CONFIG],
        "both": [MAMMOGRAM_CONFIG, MRI_CONFIG],
    }[args.dataset]

    for config in configs:
        run_kfold_comparison(
            args.models, config,
            n_folds=args.n_folds, n_trials=args.n_trials, tuning_epochs=args.tuning_epochs,
            epochs=args.epochs, fixed_weights=fixed_weights,
            tuning_method=args.tuning_method, alphas=tuple(args.alphas),
        )


if __name__ == "__main__":
    main()
