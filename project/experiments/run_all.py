"""
Top-level orchestrator: (optionally) Bayesian-tune the proposed model's
loss weights on a single split and train it on the fixed split, run the
CBAM/deep-supervision ablation, then run the full "optimize each model
-> freeze -> cross-validate to compare" protocol across every registered
model (the proposed model and all 6 baselines) -- in one command.

Tuning and cross-validation are two separate, sequential stages
throughout (tune first, then fold) rather than nested -- see
tuning/bayesian.py's module docstring and experiments/run_kfold_cv.py
for the full reasoning. The comparison stage below is
experiments.run_kfold_cv.run_kfold_comparison(): every model gets its
OWN independently tuned weights, then is k-fold CV'd with those frozen
weights, then all models are compared on the fold results. This is a
different (and more expensive) protocol than
experiments/run_sota_comparison.py's fixed-split, shared-weight
comparison -- that script remains available standalone if you want the
cheaper, controlled alternative, but this pipeline no longer calls it,
since the adviser-specified protocol supersedes it as the primary result.

If --tune-loss-weights is set, the proposed model's tuned weights from
the first stage are reused (not re-tuned) inside the comparison stage,
so it isn't searched twice; every OTHER model is still tuned fresh
inside the comparison stage regardless.

This is the only file that opts MRI_CONFIG into patient-grouped CV folds
by default (via default_patient_id_from_filename). Verify that heuristic
actually matches your RIDER filenames first, e.g. from the project root:

    from config import MRI_CONFIG
    from cross_validation.folds import default_patient_id_from_filename, preview_cv_groups
    MRI_CONFIG.patient_id_fn = default_patient_id_from_filename
    preview_cv_groups(MRI_CONFIG, n_folds=5)

and read the printed group list before trusting a full --tune-loss-weights run.

COMPUTE WARNING: the comparison stage tunes AND k-fold CVs every
requested model (7 by default) -- roughly
n_models x (n_trials x tuning_epochs + n_folds x epochs) epoch-
equivalents. This is the most expensive stage in the whole codebase by
a wide margin. Use --models to restrict it, --skip-comparison to omit it
for a cheap smoke test, and --epochs/--n-trials/--tuning-epochs/--n-folds
to shrink every stage at once.

Usage:
  python experiments/run_all.py --dataset mri
  python experiments/run_all.py --dataset both --tune-loss-weights
  python experiments/run_all.py --dataset mri --models resunetpp_cbam unet resunet   # cheaper comparison subset
  python experiments/run_all.py --dataset mri --skip-ablation --skip-comparison       # just train the proposed model
  python experiments/run_all.py --dataset mri --epochs 2 --n-trials 2 --tuning-epochs 1 --n-folds 2  # smoke test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DatasetConfig, MAMMOGRAM_CONFIG, MRI_CONFIG
from cross_validation.folds import default_patient_id_from_filename
from experiments.run_ablation import ABLATION_VARIANTS, run_ablation
from experiments.run_kfold_cv import run_kfold_comparison
from models import MODEL_REGISTRY
from training.trainer import train_model
from tuning.bayesian import tune_bce_dice_weight

# Enable patient-safe CV grouping for RIDER by default in the full
# pipeline (see module docstring above for how to verify this first).
# Mammogram grouping is left opt-in -- see config.py's comment on
# MAMMOGRAM_CONFIG -- since CBIS-DDSM's filename convention hasn't been
# checked against default_patient_id_from_filename for this project.
MRI_CONFIG.patient_id_fn = default_patient_id_from_filename


def run_pipeline(config: DatasetConfig, args: argparse.Namespace) -> None:
    bce_weight = config.bce_weight
    dice_weight = config.dice_weight
    if args.tune_loss_weights:
        bce_weight, dice_weight = tune_bce_dice_weight(
            "resunetpp_cbam", config, n_trials=args.n_trials, tuning_epochs=args.tuning_epochs,
        )

    if not args.skip_train:
        train_model("resunetpp_cbam", config, bce_weight=bce_weight, dice_weight=dice_weight, epochs=args.epochs)

    if not args.skip_ablation:
        run_ablation(config, ABLATION_VARIANTS, epochs=args.epochs)

    if not args.skip_comparison:
        # If the proposed model was already tuned above, reuse that result
        # here instead of tuning it a second time inside the comparison
        # loop. Every OTHER model in args.models still gets tuned fresh.
        fixed_weights = {"resunetpp_cbam": (bce_weight, dice_weight)} if args.tune_loss_weights else None
        run_kfold_comparison(
            args.models, config,
            n_folds=args.n_folds, n_trials=args.n_trials, tuning_epochs=args.tuning_epochs,
            epochs=args.epochs, fixed_weights=fixed_weights,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline: tune -> train -> ablation -> "
                    "(optimize each model -> freeze -> cross-validate to compare)."
    )
    parser.add_argument("--dataset", choices=("mammogram", "mri", "both"), default="both")
    parser.add_argument(
        "--tune-loss-weights", action="store_true",
        help="Tune the proposed model's weights (single split) before its own fixed-split training run. "
             "The comparison stage tunes every model regardless of this flag.",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY), choices=sorted(MODEL_REGISTRY),
        help=f"Models included in the comparison stage. Defaults to ALL. Options: {sorted(MODEL_REGISTRY)}",
    )
    parser.add_argument("--n-trials", type=int, default=15, help="Optuna trials per model during tuning.")
    parser.add_argument("--tuning-epochs", type=int, default=5, help="Epochs per tuning trial.")
    parser.add_argument("--n-folds", type=int, default=5, help="Folds for the comparison stage's cross-validation.")
    parser.add_argument("--epochs", type=int, default=None, help="Override every stage's epoch count.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training the proposed model on the fixed split.")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip the CBAM/deep-supervision ablation.")
    parser.add_argument(
        "--skip-comparison", action="store_true",
        help="Skip the optimize-each-model -> freeze -> cross-validate comparison stage (the expensive one).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset in ("mammogram", "both"):
        run_pipeline(MAMMOGRAM_CONFIG, args)
    if args.dataset in ("mri", "both"):
        run_pipeline(MRI_CONFIG, args)


if __name__ == "__main__":
    main()
