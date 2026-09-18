"""
Dispatcher between this package's two interchangeable BCE/Dice
loss-weight search strategies, so experiments/*.py can select one by
name (--tuning-method) instead of importing a specific module directly:

  - "bayesian" (tuning.bayesian.tune_bce_dice_weight): Optuna TPE
    search. Adaptive, but convergence isn't predictable in advance --
    see that module's docstring.
  - "grid" (tuning.grid_search.tune_bce_dice_weight_grid): deterministic
    sweep over a small, explicit alpha grid. The adviser-recommended
    default for the "smokescreen, then confirm on the full dataset"
    workflow -- see that module's docstring.

Both take the same (model_name, config, tuning_epochs) core arguments
and return the same (bce_weight, dice_weight) pair, so they're drop-in
replacements for each other at every call site.
"""

from __future__ import annotations

from config import DatasetConfig
from tuning.bayesian import tune_bce_dice_weight
from tuning.grid_search import DEFAULT_ALPHAS, tune_bce_dice_weight_grid

TUNING_METHODS: tuple[str, ...] = ("grid", "bayesian")
DEFAULT_TUNING_METHOD = "grid"  # per adviser feedback: predictable > adaptive for this 1-D search


def tune_loss_weights(
    method: str,
    model_name: str,
    config: DatasetConfig,
    n_trials: int = 15,
    tuning_epochs: int = 5,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> tuple[float, float]:
    """
    Run whichever loss-weight search `method` names and return its
    (bce_weight, dice_weight) result. `n_trials` is ignored by "grid"
    (its budget is len(alphas) instead); `alphas` is ignored by
    "bayesian" (it searches the continuous range, not a fixed grid).
    """
    if method == "grid":
        return tune_bce_dice_weight_grid(model_name, config, alphas=alphas, tuning_epochs=tuning_epochs)
    if method == "bayesian":
        return tune_bce_dice_weight(model_name, config, n_trials=n_trials, tuning_epochs=tuning_epochs)
    raise ValueError(f"Unknown tuning method {method!r}. Options: {TUNING_METHODS}")
