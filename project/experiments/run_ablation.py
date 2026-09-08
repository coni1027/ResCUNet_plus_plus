"""
Ablation study for the proposed model (ResNet34 + U-Net++ + CBAM +
hybrid BCE-Dice loss) -- one variant per methodology component
(3.7.1-3.7.5), each removed independently from the "full" model with
everything else held fixed, reusing the exact same training loop as the
main trainer (training.trainer.run_training_loop):

  full                          baseline: everything on
  no_cbam                       3.7.4 removed
  no_deep_supervision           3.7.3 removed
  no_cbam_no_deep_supervision   both of the above, together (interaction check)
  no_resnet_backbone            3.7.1 removed -- ResNet34Encoder -> PlainEncoder
                                 (matched channels/scales, no residual connections)
  no_nesting                    3.7.2 removed -- nested decoder -> single-path
                                 plain decoder (deep supervision forced off: no
                                 intermediate nested nodes left to supervise)
  bce_only                      3.7.5 narrowed -- hybrid loss -> pure BCE
  dice_only                     3.7.5 narrowed -- hybrid loss -> pure Dice

This is a leave-one-out design (plus the CBAM/deep-supervision 2x2,
kept since it's nearly free), not a full factorial across all four
architectural flags -- that would be up to 12 valid combinations (2
backbone x 2 nesting x 2 CBAM x {1 or 2} deep-supervision, since
deep-supervision only has 2 settings when nesting is on) before even
counting loss composition, which starts multiplying real GPU time for
diminishing return once you're past "does this component matter, other
things equal." Pass --variants to run any subset, including combinations
outside the eight above (e.g. --variants full no_resnet_backbone
no_nesting for just the two new architectural ones), if you want to
extend it later.

For comparisons against the 6 other published architectures (ResUNet++,
RA-UNet, U-Net, U-Net++, ResUNet, CBAM-UNet), see
experiments/run_kfold_cv.py instead -- and note that "unetpp" and
"cbam_unet" in that comparison are NOT clean substitutes for
no_resnet_backbone / no_nesting above: each of those baselines differs
from the proposed model on more than one axis at once (e.g. unetpp has
no ResNet backbone AND no CBAM simultaneously), so a delta against them
mixes multiple effects together rather than isolating one.

Usage:
  python experiments/run_ablation.py --dataset mri
  python experiments/run_ablation.py --dataset both --epochs 10   # quick smoke test
  python experiments/run_ablation.py --dataset mri --variants full no_resnet_backbone no_nesting
  python experiments/run_ablation.py --dataset mri --variants full bce_only dice_only
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CHECKPOINT_DIR, DEVICE, MAMMOGRAM_CONFIG, MRI_CONFIG, RESULTS_DIR, SEED, set_seed, DatasetConfig
from models.resunetpp_cbam import ResNetUNetPlusPlus
from training.trainer import run_training_loop

# Every entry sets the four architecture flags explicitly (nothing is
# implicit/defaulted) so the printed table and CSV are self-documenting.
# bce_weight/dice_weight are omitted except for the two loss variants,
# which is how run_ablation() knows to override config's defaults only
# for those two.
ABLATION_VARIANTS: dict[str, dict] = {
    "full": {
        "use_resnet": True, "nested_decoder": True,
        "use_cbam": True, "deep_supervision": True,
    },
    "no_cbam": {
        "use_resnet": True, "nested_decoder": True,
        "use_cbam": False, "deep_supervision": True,
    },
    "no_deep_supervision": {
        "use_resnet": True, "nested_decoder": True,
        "use_cbam": True, "deep_supervision": False,
    },
    "no_cbam_no_deep_supervision": {
        "use_resnet": True, "nested_decoder": True,
        "use_cbam": False, "deep_supervision": False,
    },
    "no_resnet_backbone": {
        "use_resnet": False, "nested_decoder": True,
        "use_cbam": True, "deep_supervision": True,
    },
    "no_nesting": {
        "use_resnet": True, "nested_decoder": False,
        "use_cbam": True, "deep_supervision": False,  # forced: see model docstring
    },
    "bce_only": {
        "use_resnet": True, "nested_decoder": True,
        "use_cbam": True, "deep_supervision": True,
        "bce_weight": 1.0, "dice_weight": 0.0,
    },
    "dice_only": {
        "use_resnet": True, "nested_decoder": True,
        "use_cbam": True, "deep_supervision": True,
        "bce_weight": 0.0, "dice_weight": 1.0,
    },
}


def run_ablation(
    config: DatasetConfig,
    variants: dict[str, dict] = ABLATION_VARIANTS,
    epochs: int | None = None,
) -> list[dict]:
    resolved_epochs = config.epochs if epochs is None else epochs
    rows = []

    for variant_name, flags in variants.items():
        set_seed(SEED)
        model = ResNetUNetPlusPlus(
            in_channels=config.in_channels,
            pretrained_encoder=config.pretrained_encoder,
            deep_supervision=flags["deep_supervision"],
            use_cbam=flags["use_cbam"],
            use_resnet=flags.get("use_resnet", True),
            nested_decoder=flags.get("nested_decoder", True),
        ).to(DEVICE)

        # Only bce_only/dice_only set these; every other variant falls
        # back to config's default (hybrid) weights, tuned or not.
        bce_weight = flags.get("bce_weight", config.bce_weight)
        dice_weight = flags.get("dice_weight", config.dice_weight)

        label = f"ablation_{variant_name}"
        checkpoint_path = CHECKPOINT_DIR / f"{config.name}_{label}_best.pth"
        history_path = RESULTS_DIR / f"{config.name}_{label}_history.csv"

        metrics = run_training_loop(
            model, label, config, checkpoint_path, history_path,
            bce_weight, dice_weight, resolved_epochs,
            extra_checkpoint_fields={"ablation_variant": variant_name, **flags},
        )
        rows.append({
            "variant": variant_name,
            "use_resnet": flags.get("use_resnet", True),
            "nested_decoder": flags.get("nested_decoder", True),
            "use_cbam": flags["use_cbam"],
            "deep_supervision": flags["deep_supervision"],
            "bce_weight": bce_weight,
            "dice_weight": dice_weight,
            **metrics,
        })

    out_path = RESULTS_DIR / f"ablation_{config.name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved ablation table: {out_path}")

    print(
        f"\n{'Variant':<30} {'Backbone':<8} {'Nested':<7} {'CBAM':<6} "
        f"{'DeepSup':<8} {'BCE/Dice':<10} {'Dice':>8} {'IoU':>8}"
    )
    for row in rows:
        backbone = "resnet" if row["use_resnet"] else "plain"
        nested = "yes" if row["nested_decoder"] else "no"
        weights = f"{row['bce_weight']:.2f}/{row['dice_weight']:.2f}"
        print(
            f"{row['variant']:<30} {backbone:<8} {nested:<7} {str(row['use_cbam']):<6} "
            f"{str(row['deep_supervision']):<8} {weights:<10} {row['dice']:>8.4f} {row['iou']:>8.4f}"
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study: remove one methodology component at a time from the full proposed model."
    )
    parser.add_argument("--dataset", choices=("mammogram", "mri", "both"), default="both")
    parser.add_argument(
        "--variants", nargs="+", default=list(ABLATION_VARIANTS),
        help=f"Subset of variants to run. Options: {list(ABLATION_VARIANTS)}",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override each DatasetConfig's epoch count (useful for a quick smoke test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = set(args.variants) - set(ABLATION_VARIANTS)
    if unknown:
        raise SystemExit(f"Unknown variant(s): {sorted(unknown)}. Options: {list(ABLATION_VARIANTS)}")
    selected = {name: ABLATION_VARIANTS[name] for name in args.variants}

    if args.dataset in ("mammogram", "both"):
        run_ablation(MAMMOGRAM_CONFIG, selected, epochs=args.epochs)
    if args.dataset in ("mri", "both"):
        run_ablation(MRI_CONFIG, selected, epochs=args.epochs)


if __name__ == "__main__":
    main()
