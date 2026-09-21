"""
Render actual predicted tumor segmentation masks as images, using a
trained model's best checkpoint. training/evaluator.py computes the same
thresholded prediction (sigmoid(logits) >= 0.5) but only reduces it to
scalar Dice/IoU/etc. -- nothing in the pipeline ever saves that mask as a
picture. This script does exactly that, on a handful of held-out test
samples, so the segmentation can actually be looked at.

For each sampled test image this saves ONE composite PNG with four
panels side by side:
  input image | ground-truth mask (green) | predicted mask (red) |
  agreement map (green=GT only, red=prediction only, yellow=overlap)

Uses only cv2/numpy (already project dependencies) -- no matplotlib.

Usage:
  python experiments/visualize_predictions.py --dataset mri
  python experiments/visualize_predictions.py --dataset mammogram --model resunetpp_cbam --n-samples 12
  python experiments/visualize_predictions.py --dataset mri --checkpoint checkpoints/breast_mri_resunetpp_cbam_best.pth
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CHECKPOINT_DIR, DatasetConfig, DEVICE, MAMMOGRAM_CONFIG, MRI_CONFIG, RESULTS_DIR
from datasets.breast_dataset import BreastSegmentationDataset
from models import MODEL_LABELS, MODEL_REGISTRY
from training.evaluator import batch_metrics
from training.losses import final_logits

PANEL_LABELS = ("input", "ground truth", "prediction", "agreement")


def _to_display_gray(image: np.ndarray) -> np.ndarray:
    """image is CxHxW float32 in [0, 1] (see preprocess_image()). Collapse
    multi-channel input (e.g. 4-channel MRI) to one grayscale panel by
    averaging channels -- there's no single "the" channel to prefer for
    generic display."""
    gray = image.mean(axis=0) if image.ndim == 3 else image
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


def _overlay_mask(base_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    overlay = base_bgr.copy()
    overlay[mask] = color_bgr
    return cv2.addWeighted(overlay, alpha, base_bgr, 1 - alpha, 0)


def _agreement_map(base_bgr: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    overlay = base_bgr.copy()
    overlay[gt & ~pred] = (0, 200, 0)      # missed lesion -- green
    overlay[pred & ~gt] = (0, 0, 220)      # false positive -- red
    overlay[gt & pred] = (0, 220, 220)     # correct overlap -- yellow
    return cv2.addWeighted(overlay, 0.55, base_bgr, 0.45, 0)


def _label_panel(panel: np.ndarray, text: str) -> np.ndarray:
    panel = panel.copy()
    cv2.putText(panel, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def render_composite(image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, dice: float) -> np.ndarray:
    base = _to_display_gray(image)
    gt_bool = gt_mask >= 0.5
    pred_bool = pred_mask >= 0.5

    panels = [
        base,
        _overlay_mask(base, gt_bool, (0, 200, 0)),
        _overlay_mask(base, pred_bool, (0, 0, 220)),
        _agreement_map(base, gt_bool, pred_bool),
    ]
    panels = [_label_panel(p, label) for p, label in zip(panels, PANEL_LABELS)]

    composite = cv2.hconcat(panels)
    footer = np.full((28, composite.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(footer, f"dice={dice:.4f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.vconcat([composite, footer])


def visualize_predictions(
    model_name: str,
    config: DatasetConfig,
    checkpoint_path: Path | None = None,
    split: str = "test",
    n_samples: int = 8,
    seed: int = 42,
    output_dir: Path | None = None,
) -> Path:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name {model_name!r}. Options: {sorted(MODEL_REGISTRY)}")

    checkpoint_path = checkpoint_path or CHECKPOINT_DIR / f"{config.name}_{model_name}_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path} -- train {model_name} on {config.name} first."
        )

    model = MODEL_REGISTRY[model_name](config).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = BreastSegmentationDataset(config, split, augment=False)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    indices = sorted(indices[:n_samples])

    label = MODEL_LABELS.get(model_name, model_name)
    output_dir = output_dir or RESULTS_DIR / "segmentation_previews" / f"{config.name}_{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{config.name}/{model_name}] Loaded {checkpoint_path} (best val dice={checkpoint['best_val_dice']:.4f})")
    print(f"Rendering {len(indices)} {split} sample(s) to {output_dir}")

    with torch.no_grad():
        for index in indices:
            image_tensor, mask_tensor = dataset[index]
            image_path = dataset.samples[index][0]

            logits = final_logits(model(image_tensor.unsqueeze(0).to(DEVICE)))
            metrics = batch_metrics(logits, mask_tensor.unsqueeze(0).to(DEVICE))
            pred_mask = (torch.sigmoid(logits) >= 0.5).squeeze().cpu().numpy()

            composite = render_composite(
                image_tensor.numpy(), mask_tensor.squeeze(0).numpy(), pred_mask, metrics["dice"],
            )
            out_path = output_dir / f"{image_path.stem}_pred.png"
            cv2.imwrite(str(out_path), composite)
            print(f"  {image_path.name}: dice={metrics['dice']:.4f} -> {out_path.name}")

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save predicted tumor segmentation masks (input | ground truth | prediction | "
                    "agreement) as PNGs, using a trained model's best checkpoint."
    )
    parser.add_argument("--dataset", choices=("mammogram", "mri", "both"), default="both")
    parser.add_argument("--model", default="resunetpp_cbam", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--n-samples", type=int, default=8, help="How many samples to render.")
    parser.add_argument("--seed", type=int, default=42, help="Sample selection seed.")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Override the checkpoint path (defaults to checkpoints/<dataset>_<model>_best.pth).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override the output directory (defaults to results/segmentation_previews/<dataset>_<model>/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = {
        "mammogram": [MAMMOGRAM_CONFIG],
        "mri": [MRI_CONFIG],
        "both": [MAMMOGRAM_CONFIG, MRI_CONFIG],
    }[args.dataset]

    for config in configs:
        visualize_predictions(
            args.model, config,
            checkpoint_path=args.checkpoint, split=args.split,
            n_samples=args.n_samples, seed=args.seed, output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
