"""
Train and evaluate every registered model (the proposed ResNet34 +
U-Net++ + CBAM model and all 6 SOTA baselines: U-Net, U-Net++, ResUNet,
ResUNet++, RA-UNet, CBAM-UNet) on the same dataset/split, and write a
comparison table of held-out test metrics.

By default every model uses its DatasetConfig's fixed bce_weight/
dice_weight (0.5/0.5) -- i.e. the comparison is controlled to isolate
architecture differences, not confounded by seven different loss
balances. Pass --tune-each to run loss-weight tuning (single split --
not cross-validated) separately for each model before training it, if
you want every baseline to get its own best-effort loss balance too.
--tuning-method picks the search: "grid" (default, tuning.grid_search --
deterministic sweep over --alphas) or "bayesian" (tuning.bayesian --
Optuna TPE over --n-trials trials). See tuning/__init__.py and each
module's docstring for the tradeoff. Either way this adds roughly
(len(alphas) or n_trials) x tuning_epochs per model, on top of the
training time itself.

The comparison CSV (results/sota_comparison_<dataset>.csv) is updated,
not overwritten: running this script again -- with a different --models
subset, or just re-running one model -- merges its new row(s) into
whatever's already in the file (matched by the "model" column), instead
of replacing the whole table with only the models from that one run.
Re-running the SAME model replaces its row with the fresh result; a
model from an earlier run that isn't in this run's --models is left
untouched. See update_comparison_csv().

Usage:
  python experiments/run_sota_comparison.py --dataset mri
  python experiments/run_sota_comparison.py --dataset both --epochs 10        # quick smoke test
  python experiments/run_sota_comparison.py --dataset mri --models resunetpp_cbam unet resunet
  python experiments/run_sota_comparison.py --dataset mri --tune-each
  python experiments/run_sota_comparison.py --dataset mri --tune-each --tuning-method bayesian
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DatasetConfig, MAMMOGRAM_CONFIG, MRI_CONFIG, RESULTS_DIR
from models import MODEL_LABELS, MODEL_REGISTRY
from training.trainer import train_model
from tuning import DEFAULT_TUNING_METHOD, TUNING_METHODS, tune_loss_weights
from tuning.grid_search import DEFAULT_ALPHAS


def _load_existing_rows(path: Path) -> dict[str, dict]:
    """
    Read a previously written sota_comparison_<dataset>.csv, keyed by its
    "model" column, so update_comparison_csv() can merge into it instead
    of overwriting it. Returns {} if the file doesn't exist yet.

    Every column except "model"/"label" is a float (see run_comparison()'s
    row construction), so those are converted back on the way in --
    otherwise a merged table mixing old (string, from CSV) and new (float,
    fresh from train_model()) values in the same column would break the
    ":.4f" formatting in the printed summary below.
    """
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows: dict[str, dict] = {}
        for raw_row in csv.DictReader(f):
            row = {
                key: value if key in ("model", "label") else float(value)
                for key, value in raw_row.items()
            }
            rows[row["model"]] = row
        return rows


def update_comparison_csv(path: Path, new_rows: list[dict]) -> list[dict]:
    """
    Merge new_rows into path's existing comparison table, matched by the
    "model" column: a model already in the file gets its row REPLACED
    with the fresh result, a new model gets APPENDED, and any model from
    an earlier run that isn't in new_rows is left untouched. This is what
    lets you run this script once per model (or per subset) over several
    sessions and end up with one combined table, instead of each run
    wiping out every other model's result. Returns the full merged row
    list that was actually written.
    """
    existing = _load_existing_rows(path)
    for row in new_rows:
        existing[row["model"]] = row
    merged_rows = list(existing.values())

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged_rows[0].keys()))
        writer.writeheader()
        writer.writerows(merged_rows)
    return merged_rows


def run_comparison(
    config: DatasetConfig,
    model_names: list[str],
    epochs: int | None = None,
    tune_each: bool = False,
    n_trials: int = 15,
    tuning_epochs: int = 5,
    tuning_method: str = DEFAULT_TUNING_METHOD,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> list[dict]:
    rows = []
    for model_name in model_names:
        label = MODEL_LABELS.get(model_name, model_name)
        print(f"\n{'#' * 80}\n# {config.name} / {label}\n{'#' * 80}")

        bce_weight = dice_weight = None
        if tune_each:
            bce_weight, dice_weight = tune_loss_weights(
                tuning_method, model_name, config,
                n_trials=n_trials, tuning_epochs=tuning_epochs, alphas=alphas,
            )

        metrics = train_model(model_name, config, bce_weight=bce_weight, dice_weight=dice_weight, epochs=epochs)
        rows.append({"model": model_name, "label": label, **metrics})

    out_path = RESULTS_DIR / f"sota_comparison_{config.name}.csv"
    merged_rows = update_comparison_csv(out_path, rows)
    print(f"\nSaved comparison table: {out_path} ({len(merged_rows)} model(s) total)")

    print(f"\n{'Model':<36} {'Dice':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'Accuracy':>9}")
    for row in merged_rows:
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
            "Run loss-weight tuning (single split) for EACH model before "
            "training it, instead of using each DatasetConfig's fixed "
            "bce_weight/dice_weight for every model (the default, which "
            "keeps the comparison controlled to architecture differences)."
        ),
    )
    parser.add_argument(
        "--tuning-method", choices=TUNING_METHODS, default=DEFAULT_TUNING_METHOD,
        help="'grid' (default): deterministic sweep over --alphas -- predictable, adviser-recommended "
             "smokescreen. 'bayesian': Optuna TPE search over --n-trials trials (unpredictable convergence). "
             "Only used with --tune-each.",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS),
        help=f"Grid of bce_weight values to try (only used with --tuning-method grid). Default: {list(DEFAULT_ALPHAS)}",
    )
    parser.add_argument("--n-trials", type=int, default=15, help="--tuning-method bayesian only.")
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
        tuning_method=args.tuning_method, alphas=tuple(args.alphas),
    )
    if args.dataset in ("mammogram", "both"):
        run_comparison(MAMMOGRAM_CONFIG, args.models, **kwargs)
    if args.dataset in ("mri", "both"):
        run_comparison(MRI_CONFIG, args.models, **kwargs)


if __name__ == "__main__":
    main()
