"""
Train and evaluate every registered model (the proposed ResNet34 +
U-Net++ + CBAM model and all 6 SOTA baselines: U-Net, U-Net++, ResUNet,
ResUNet++, RA-UNet, CBAM-UNet) on the same dataset/split, and write a
comparison table of held-out test metrics.

By default every model uses its DatasetConfig's fixed bce_weight/
dice_weight (0.5/0.5) -- i.e. the comparison is controlled to isolate
architecture differences, not confounded by seven different loss
balances. Pass --tune-each to run Bayesian loss-weight tuning
(tuning.bayesian.tune_bce_dice_weight, single split -- not cross-
validated, see that module's docstring) separately for each model
before training it, if you want every baseline to get its own
best-effort loss balance too -- this adds roughly n_trials x
tuning_epochs per model, on top of the training time itself.

Usage:
  python experiments/run_sota_comparison.py --dataset mri
  python experiments/run_sota_comparison.py --dataset both --epochs 10        # quick smoke test
  python experiments/run_sota_comparison.py --dataset mri --models resunetpp_cbam unet resunet
  python experiments/run_sota_comparison.py --dataset mri --tune-each
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DatasetConfig, MAMMOGRAM_CONFIG, MRI_CONFIG, RESULTS_DIR
from models import MODEL_LABELS, MODEL_REGISTRY
from training.trainer import train_model
from tuning.bayesian import tune_bce_dice_weight


def run_comparison(
    config: DatasetConfig,
    model_names: list[str],
    epochs: int | None = None,
    tune_each: bool = False,
    n_trials: int = 15,
    tuning_epochs: int = 5,
) -> list[dict]:
    rows = []
    for model_name in model_names:
        label = MODEL_LABELS.get(model_name, model_name)
        print(f"\n{'#' * 80}\n# {config.name} / {label}\n{'#' * 80}")

        bce_weight = dice_weight = None
        if tune_each:
            bce_weight, dice_weight = tune_bce_dice_weight(
                model_name, config, n_trials=n_trials, tuning_epochs=tuning_epochs,
            )

        metrics = train_model(model_name, config, bce_weight=bce_weight, dice_weight=dice_weight, epochs=epochs)
        rows.append({"model": model_name, "label": label, **metrics})

    out_path = RESULTS_DIR / f"sota_comparison_{config.name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved comparison table: {out_path}")

    print(f"\n{'Model':<36} {'Dice':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'Accuracy':>9}")
    for row in rows:
        print(
            f"{row['label']:<36} {row['dice']:>8.4f} {row['iou']:>8.4f} "
            f"{row['precision']:>10.4f} {row['recall']:>8.4f} {row['accuracy']:>9.4f}"
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate all registered models for SOTA comparison.")
    parser.add_argument("--dataset", choices=("mammogram", "mri", "both"), default="both")
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY),
        help=f"Subset of models to run. Options: {sorted(MODEL_REGISTRY)}",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override each DatasetConfig's epoch count (useful for a quick smoke test).",
    )
    parser.add_argument(
        "--tune-each", action="store_true",
        help=(
            "Run Bayesian loss-weight tuning (single split) for EACH model before "
            "training it, instead of using each DatasetConfig's fixed "
            "bce_weight/dice_weight for every model (the default, which "
            "keeps the comparison controlled to architecture differences)."
        ),
    )
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--tuning-epochs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = set(args.models) - set(MODEL_REGISTRY)
    if unknown:
        raise SystemExit(f"Unknown model name(s): {sorted(unknown)}. Options: {sorted(MODEL_REGISTRY)}")

    kwargs = dict(
        epochs=args.epochs, tune_each=args.tune_each,
        n_trials=args.n_trials, tuning_epochs=args.tuning_epochs,
    )
    if args.dataset in ("mammogram", "both"):
        run_comparison(MAMMOGRAM_CONFIG, args.models, **kwargs)
    if args.dataset in ("mri", "both"):
        run_comparison(MRI_CONFIG, args.models, **kwargs)


if __name__ == "__main__":
    main()
