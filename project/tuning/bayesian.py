"""
Bayesian optimization for the BCE/Dice loss balance.

bce_weight = alpha, dice_weight = 1 - alpha, alpha in [0, 1]. Uses
Optuna's TPE sampler (a sequential model-based / Bayesian optimizer) to
pick the next alpha to try based on all previous trials' results, rather
than a blind grid or random search.

Each trial trains on config's FIXED train split for `tuning_epochs`
epochs and is scored on config's FIXED val split -- the same split
training.trainer.train_model() uses -- NOT K-fold cross-validation.
Optimization and cross-validation are deliberately two separate,
sequential steps (search first, then validate with folds), rather than
nesting CV inside every trial:

  1. This module finds the best alpha on the fixed split (this file).
  2. experiments/run_kfold_cv.py takes that fixed alpha and trains one
     model per cross-validation fold, reporting mean +/- std metrics --
     a robustness check, run once, AFTER search, instead of once per
     trial.

An earlier version of this file cross-validated every trial (folds
nested inside the search), which cost roughly
n_trials x n_folds x tuning_epochs epoch-equivalents. This sequential
version costs n_trials x tuning_epochs for the search, then n_folds x
epochs once for the robustness check in run_kfold_cv.py -- same total
number of full-length trainings, but the expensive search phase is no
longer multiplied by n_folds.

This module is model-agnostic: pass any model_name from
models.MODEL_REGISTRY (not just the proposed "resunetpp_cbam"). Most
SOTA baselines here return a single logits tensor rather than a list
(see their forward() methods); DeepSupervisionLoss and evaluate()
already handle both cases identically, so no special-casing is needed
here.

Please read these caveats before trusting the result of this search:

1. Each trial retrains a FRESH model for a reduced number of epochs
   (tuning_epochs, default 5) instead of the full schedule, as a
   compute-saving proxy. The alpha that looks best after a few short
   epochs is not guaranteed to still be best after the full training
   run -- treat the tuned value as a good starting point, not a
   certified optimum, and re-check it against
   experiments/run_kfold_cv.py's per-fold spread.
2. The search is scored on a SINGLE fixed val split -- the same one
   used for checkpoint selection during final training. That matches
   what train_model() actually optimizes against, but it's a
   single-split estimate: it doesn't by itself say how sensitive that
   choice of alpha is to which patients/slices happen to land in val.
   That's what run_kfold_cv.py's spread across folds is for, especially
   given RIDER's small patient count.
3. This tunes only the BCE/Dice balance. It reuses whatever batch size,
   learning rate, augmentation, and input resolution are already
   hard-coded in each DatasetConfig -- those are not part of the search
   space here.
4. The same random seed re-initializes the model at the start of every
   trial, so the search isolates the effect of alpha rather than
   random-init noise. The result has still only been checked for one
   seed; consider re-running with 2-3 different seeds before trusting a
   close call.
"""

from __future__ import annotations

from torch import optim

try:
    import optuna
except ImportError:  # Optuna is optional unless this module is used.
    optuna = None

from config import DatasetConfig, DEVICE, SEED, set_seed
from datasets.breast_dataset import create_loaders
from models import MODEL_REGISTRY
from training.evaluator import evaluate
from training.losses import BCEDiceLoss, DeepSupervisionLoss


def tune_bce_dice_weight(
    model_name: str,
    config: DatasetConfig,
    n_trials: int = 15,
    tuning_epochs: int = 5,
    timeout: int | None = None,
) -> tuple[float, float]:
    """See the module docstring for the full method description and caveats."""
    if optuna is None:
        raise ImportError(
            "optuna is required for Bayesian optimization of the loss weights. "
            "Install it with: pip install optuna"
        )
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name {model_name!r}. Options: {sorted(MODEL_REGISTRY)}")

    # Built once, outside the objective, so every trial reuses the same
    # file listing / preprocessing instead of re-scanning disk each time.
    train_loader, val_loader, _test_loader = create_loaders(config)

    def objective(trial: "optuna.Trial") -> float:
        alpha = trial.suggest_float("bce_weight", 0.0, 1.0)

        set_seed(SEED)
        model = MODEL_REGISTRY[model_name](config)

        base_loss = BCEDiceLoss(bce_weight=alpha, dice_weight=1.0 - alpha)
        criterion = DeepSupervisionLoss(base_loss)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        val_dice = 0.0
        for epoch in range(1, tuning_epochs + 1):
            model.train()
            for images, masks in train_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            val_metrics = evaluate(model, val_loader, criterion, DEVICE)
            val_dice = val_metrics["dice"]

            # Report/prune between epochs now that there's no fold loop.
            trial.report(val_dice, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_dice

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    total_epoch_equivalents = n_trials * tuning_epochs
    print(
        f"[{config.name}/{model_name}] Starting Bayesian optimization: "
        f"{n_trials} trials x {tuning_epochs} epochs = up to "
        f"{total_epoch_equivalents} epoch-equivalents before pruning "
        "(MedianPruner will cut many unpromising trials short)..."
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_alpha = study.best_params["bce_weight"]
    print(
        f"[{config.name}/{model_name}] Best alpha (bce_weight) found: "
        f"{best_alpha:.4f} (dice_weight={1.0 - best_alpha:.4f}) -> "
        f"val_dice={study.best_value:.4f} over {len(study.trials)} trials"
    )
    return best_alpha, 1.0 - best_alpha
