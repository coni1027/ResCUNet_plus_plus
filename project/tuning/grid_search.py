"""
Grid search for the BCE/Dice loss balance -- a deterministic alternative
to tuning.bayesian's Optuna (TPE) search, added per adviser feedback:
Bayesian optimization's convergence is unpredictable (you can't know in
advance how many trials it needs, or whether it's actually found the
best region rather than gotten unlucky with its sampler seed), whereas a
grid over a small, representative set of alphas is deterministic,
exhaustively covers exactly the values you asked about, and produces a
full Dice-vs-alpha curve you can inspect directly -- at the cost of not
adaptively concentrating trials in a promising region the way TPE does.
For a single scalar hyperparameter (bce_weight = alpha, dice_weight =
1 - alpha, see below) that adaptivity buys little anyway, so the
tradeoff mostly disappears here.

Same "smokescreen, then confirm" protocol as tuning.bayesian:
  1. This module scores every alpha in a small fixed grid, each for
     `tuning_epochs` epochs on config's FIXED train/val split -- the
     same split training.trainer.train_model() uses, NOT K-fold CV
     (see tuning.bayesian's module docstring for why tuning and
     cross-validation are kept separate and sequential).
  2. experiments/run_kfold_cv.py takes the winning alpha and trains one
     model per cross-validation fold on the FULL dataset, reporting
     mean +/- std metrics -- the robustness check that confirms the
     smokescreen result generalizes, run once, after the grid.

This module is model-agnostic (pass any model_name from
models.MODEL_REGISTRY) and mirrors tuning.bayesian.tune_bce_dice_weight's
signature and return value on purpose, so the two are interchangeable at
every call site -- see tuning/__init__.py's tune_loss_weights() dispatcher.

Please read these caveats before trusting the result of this search:

1. Like tuning.bayesian, each grid point retrains a FRESH model for a
   reduced number of epochs (tuning_epochs, default 5) instead of the
   full schedule, as a compute-saving proxy. The alpha that looks best
   after a few short epochs is not guaranteed to still be best after the
   full training run -- treat the winner as a good starting point, not a
   certified optimum, and re-check it against
   experiments/run_kfold_cv.py's per-fold spread.
2. Scored on a SINGLE fixed val split, same caveat as tuning.bayesian:
   this is a single-split estimate of how sensitive the choice of alpha
   is to which patients/slices happen to land in val. That's what
   run_kfold_cv.py's spread across folds is for.
3. This tunes only the BCE/Dice balance -- batch size, learning rate,
   augmentation, and input resolution stay whatever's already hard-coded
   in each DatasetConfig.
4. The same random seed re-initializes the model before every grid
   point, so the grid isolates the effect of alpha rather than
   random-init noise. Still only checked for one seed.
5. Total cost is len(alphas) x tuning_epochs epoch-equivalents -- there
   is no pruning here (unlike tuning.bayesian's MedianPruner), since
   every grid point is a value you explicitly asked about, not a
   sampler's guess to abandon early. Keep the grid small (a handful of
   points, not a fine sweep) if this needs to stay a "smokescreen."
"""

from __future__ import annotations

from torch import optim

from config import DatasetConfig, DEVICE, SEED, set_seed
from datasets.breast_dataset import create_loaders
from models import MODEL_REGISTRY
from training.evaluator import evaluate
from training.losses import BCEDiceLoss, DeepSupervisionLoss

# The adviser-specified grid: bce_weight (lambda) from 0.1 to 0.9 in
# steps of 0.1, with dice_weight = 1 - lambda implied. 0.0 and 1.0 are
# deliberately excluded (matching the original proposal) -- pass a
# different `alphas` sequence to tune_bce_dice_weight_grid() if you want
# to include the pure-BCE/pure-Dice endpoints (already covered
# separately by experiments/run_ablation.py's bce_only/dice_only
# variants for the proposed model specifically).
DEFAULT_ALPHAS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def tune_bce_dice_weight_grid(
    model_name: str,
    config: DatasetConfig,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    tuning_epochs: int = 5,
    return_history: bool = False,
) -> tuple[float, float] | tuple[float, float, list[tuple[float, float]]]:
    """
    See the module docstring for the full method description and caveats.

    return_history=True additionally returns the full (alpha, val_dice)
    list for every grid point searched, e.g. for plotting a Dice-vs-alpha
    curve -- otherwise only the winning pair survives past this call (see
    tuning/__init__.py's tune_loss_weights(), which needs the plain
    2-tuple to stay a drop-in match for tuning.bayesian's return value).
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name {model_name!r}. Options: {sorted(MODEL_REGISTRY)}")
    if not alphas:
        raise ValueError("alphas must be a non-empty sequence of BCE weights in [0, 1].")

    # Built once, so every grid point reuses the same file listing /
    # preprocessing instead of re-scanning disk each time.
    train_loader, val_loader, _test_loader = create_loaders(config)

    print(
        f"[{config.name}/{model_name}] Starting grid search over {len(alphas)} "
        f"alpha(s) x {tuning_epochs} epochs = {len(alphas) * tuning_epochs} "
        f"epoch-equivalents (no pruning -- every point runs to completion): {list(alphas)}"
    )

    results: list[tuple[float, float]] = []  # (alpha, val_dice)
    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Each alpha must be in [0, 1], got {alpha}.")

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
        for _epoch in range(1, tuning_epochs + 1):
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

        results.append((alpha, val_dice))
        print(f"  alpha={alpha:.2f} (bce={alpha:.2f}/dice={1.0 - alpha:.2f}) -> val_dice={val_dice:.4f}")

    best_alpha, best_dice = max(results, key=lambda pair: pair[1])

    print(f"\n[{config.name}/{model_name}] Grid search results (Dice vs. alpha):")
    for alpha, val_dice in results:
        marker = "  <- best" if alpha == best_alpha else ""
        print(f"  alpha={alpha:.2f}  val_dice={val_dice:.4f}{marker}")
    print(
        f"[{config.name}/{model_name}] Best alpha (bce_weight): {best_alpha:.4f} "
        f"(dice_weight={1.0 - best_alpha:.4f}) -> val_dice={best_dice:.4f} over {len(results)} grid point(s)"
    )
    if return_history:
        return best_alpha, 1.0 - best_alpha, results
    return best_alpha, 1.0 - best_alpha
