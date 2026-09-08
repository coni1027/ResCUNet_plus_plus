# Breast lesion segmentation: ResNet34 + U-Net++ + CBAM

Binary breast lesion segmentation across two modalities (CBIS-DDSM
mammograms, RIDER breast MRI slices), comparing the proposed model
against 6 published baselines.

## Project layout

```
project/
├── config.py                    # DatasetConfig, MAMMOGRAM_CONFIG, MRI_CONFIG, paths, seed
├── preprocessing/
│   ├── common.py                # shared: patient splitting, split caching, manifest writing
│   ├── prepare_mri.py            # ONE-TIME: raw RIDER volumes -> per-slice train/val/test layout
│   └── prepare_mammograms.py     # ONE-TIME: raw CBIS-DDSM Whole+ROI pairs -> resized train/val/test layout
├── data/
│   ├── mammograms/{train,val,test}/{images,masks}/
│   └── breast_mri/{train,val,test}/{images,masks}/
├── splits/                      # persisted CV fold assignments (auto-generated)
├── checkpoints/                 # best-model .pth files (auto-generated)
├── results/                     # training history CSVs, comparison tables (auto-generated)
├── datasets/
│   └── breast_dataset.py        # loading, preprocessing, augmentation, Dataset/DataLoader
├── cross_validation/
│   └── folds.py                 # grouped K-fold CV (used by run_kfold_cv.py) + splits/ persistence
├── models/
│   ├── common.py                # shared building blocks (CBAM, ResidualBlock, ASPP, ...)
│   ├── resunetpp_cbam.py        # proposed model (ResNet34 + U-Net++ + CBAM)
│   ├── unet.py                  # U-Net (Ronneberger et al., 2015)
│   ├── unetpp.py                # U-Net++ (Zhou et al., 2018), plain encoder, no CBAM
│   ├── resunet.py               # ResUNet (Zhang et al., 2018)
│   ├── resunetpp.py             # ResUNet++ (Jha et al., 2019)
│   ├── ra_unet.py               # RA-UNet (Jin et al., 2020), 2D-adapted
│   ├── cbam_unet.py             # plain U-Net + CBAM on skip connections
│   ├── diagram.py                # matplotlib architecture diagram (proposed model only)
│   └── __init__.py               # MODEL_REGISTRY / MODEL_LABELS
├── training/
│   ├── losses.py                 # BCEDiceLoss, DeepSupervisionLoss
│   ├── evaluator.py              # metrics (Dice/IoU/precision/recall/accuracy), evaluate()
│   └── trainer.py                 # shared training loop + registry-based train_model()
├── tuning/
│   └── bayesian.py                # Bayesian (Optuna TPE) loss-weight tuning, single split
└── experiments/
    ├── run_ablation.py            # leave-one-out ablation: all 5 methodology components
    ├── run_sota_comparison.py     # fixed-split, shared-weight comparison (cheaper, controlled, optional)
    ├── run_kfold_cv.py            # optimize each model -> freeze -> cross-validate to compare (the main one)
    └── run_all.py                 # tune -> train -> ablation -> optimize-freeze-CV-compare, one command
```

**Tuning and cross-validation are two separate, sequential stages, not
nested, and there are two different comparison protocols available.**
`tuning/bayesian.py` searches for the best BCE/Dice weight on a single
fixed split. `experiments/run_kfold_cv.py` then either (a) trains one
model per cross-validation fold using an already-chosen, frozen weight
for a single model, or (b) via `run_kfold_comparison()` / its CLI's
default `--models` (all of them), runs the FULL "optimize each model →
freeze → cross-validate to compare" protocol: every model is tuned
independently, frozen, k-fold CV'd, and then compared on fold results —
each at its own best effort, not a shared value. This is the primary,
adviser-specified comparison and what `run_all.py` now runs by default.
`experiments/run_sota_comparison.py` is a different, cheaper protocol
still available standalone: every model gets the SAME fixed 0.5/0.5
weight and is trained once on the fixed split — useful for a fast,
architecture-only-controlled comparison, but not a substitute for the
optimize→freeze→CV-compare protocol above.

A few small additions beyond the original sketch, called out here rather
than silently: `checkpoints/` and `splits/` (needed at runtime, not shown
in the original diagram), `models/common.py` and `models/diagram.py`
(shared building blocks / the diagram generator, split out of
`resunetpp_cbam.py` to keep that file to just the model), `preprocessing/`
(one-time data preparation for both datasets, not model code -- see
"Data preparation" below), and `__init__.py` in every package so `experiments/*.py` can
`sys.path.insert` the project root and import cleanly without installing
this as a package. The SOTA model files are named by architecture
(`unet.py`, `resunet.py`, ...) rather than `sota1.py`...`sota6.py`,
since with six different published architectures a descriptive name is
one less thing to look up — rename them if you'd rather match the
original sketch exactly. Persisted CV splits are named
`<dataset_name>_<n_folds>fold.json` (e.g. `breast_mri_5fold.json`,
using the internal dataset name `breast_mri` rather than `mri`) for
consistency with how checkpoints and results are already named
elsewhere in this codebase.

## Data preparation

Both datasets need a one-time pass through `preprocessing/` before
anything under `data/` is ready for `datasets/breast_dataset.py` to
read. Both scripts share `preprocessing/common.py` for patient
splitting, split caching, and manifest writing.

**RIDER MRI** ships as whole-patient volumes (image volumes shaped
`(n_slices, 4, H, W)`, e.g. the `(60, 4, 288, 288)` you found), not the
per-slice files the training pipeline reads:

```bash
python preprocessing/prepare_mri.py --dry-run   # prints the full plan, writes nothing
python preprocessing/prepare_mri.py             # writes data/breast_mri/{train,val,test}/{images,masks}/
```

`prepare_mri.py`'s module docstring lists the exact raw-layout
assumptions it makes (image/mask volumes mirrored under separate
directory trees, patient ID recoverable from the parent folder or
filename) — if your raw files don't match, `patient_id_from_path()`
and `find_volume_pairs()` are the only two functions that need editing.
The split happens once, per patient, before any slice is written, so no
slice from one patient can ever leak into a different split from
another slice of the same patient — and every slice is kept, including
ones with an empty mask, per your answer on avoiding a positive-only
bias.

**CBIS-DDSM mammograms** ship as four DICOM/PNG files per case (e.g.
`P_00731 Right MLO {Whole,ROI,Zoomed}.dcm` plus a `.png` export).
Only `Whole` (the full mammogram) and `ROI` (the pixel-aligned lesion
mask) are used — `Zoomed` (a small cropped patch, a different image
entirely) and the `.png` (a low-res preview export) are deliberately
never touched:

```bash
python preprocessing/prepare_mammograms.py --dry-run
python preprocessing/prepare_mammograms.py --raw-root /path/to/raw/cbis-ddsm
```

This one resizes both image and mask to 512×512 (linear/nearest,
matching `MAMMOGRAM_CONFIG.input_size`) and writes `.npy` pairs — only
the resize happens at this stage; percentile normalization, CLAHE, and
median filtering still happen dynamically at load time via
`preprocess_image()`, same as before, so nothing gets double-processed.

**Read this before trusting `prepare_mammograms.py`'s split**:
Methodology 3.2 says CBIS-DDSM's *official* train/test split is
retained, with an 80:20 patient-level train:val carve-out from the
official training cases. This script has no way to see that official
split on its own — it only has your raw file listing to go on, not the
`mass_case_description_{train,test}_set.csv` / `calc_case_description_
{train,test}_set.csv` metadata CBIS-DDSM normally ships alongside the
images. `official_split_for_case()` is a **placeholder** that returns
`"train"` for every case until you edit it to consult whichever of
those CSVs you have — right now, that means zero cases land in test.
The script prints a loud warning about this every run, and `--dry-run`
makes it obvious in the split summary before you'd otherwise notice too
late.

## Setup

```bash
pip install torch torchvision optuna matplotlib pydicom
```

`pydicom` is only needed if your images are `.dcm`; `optuna` only for
`--tune-loss-weights`; `matplotlib` only for the architecture diagram.

## Running

All commands below are run from the `project/` root.

**Train the proposed model on one modality:**
```bash
python -m training.trainer  # not a CLI entry point by itself -- use experiments/run_all.py, below
```
There's no bare CLI for `training/trainer.py` on its own; use
`experiments/run_all.py --skip-ablation --skip-comparison` for "just
train my model," or call
`training.trainer.train_model("resunetpp_cbam", MRI_CONFIG)` directly
from a script/notebook.

**Preview cross-validation fold/patient assignment (free, no training):**
```bash
python -c "
from config import MRI_CONFIG
from cross_validation.folds import default_patient_id_from_filename, preview_cv_groups
MRI_CONFIG.patient_id_fn = default_patient_id_from_filename
preview_cv_groups(MRI_CONFIG, n_folds=5)
"
```
Do this before your first `--tune-loss-weights` run to confirm
`default_patient_id_from_filename` actually parses your RIDER filenames
correctly (it's a best-effort heuristic — see its docstring).

**Full pipeline (tune → train → ablation → optimize-each-model → freeze → cross-validate to compare) for one modality:**
```bash
python experiments/run_all.py --dataset mri --tune-loss-weights
```
This tunes and trains the proposed model on the fixed split, runs the
ablation, then — the expensive part — tunes, freezes, and 5-fold
cross-validates EVERY registered model (7 by default) and writes one
comparison table. See "Compute expectations" below before running this
on both modalities with default settings.

**Just the tuning step (single split, no folds), for one model:**
```bash
python -c "
from config import MRI_CONFIG
from tuning.bayesian import tune_bce_dice_weight
print(tune_bce_dice_weight('resunetpp_cbam', MRI_CONFIG))
"
```

**The full optimize → freeze → cross-validate comparison, for every model (or a subset):**
```bash
python experiments/run_kfold_cv.py --dataset mri                                          # every registered model
python experiments/run_kfold_cv.py --dataset mri --models resunetpp_cbam unet resunet     # cheaper subset
python experiments/run_kfold_cv.py --dataset mri --n-folds 4                              # RIDER caps at 4 non-test patients anyway
```
Each model is tuned independently (single split), frozen, then k-fold
CV'd with those frozen weights — so the comparison reflects every
architecture at its own best effort. Writes one CSV per model
(`results/<dataset>_<model>_kfold_cv.csv`) plus one combined,
best-first comparison table (`results/<dataset>_cv_comparison.csv`).

**The same protocol for just ONE model (no cross-model comparison table, just that model's tuned + folded result):**
```bash
python experiments/run_kfold_cv.py --dataset mri --models resunetpp_cbam
python experiments/run_kfold_cv.py --dataset mri --models resunetpp_cbam --bce-weight 0.4 --dice-weight 0.6  # skip tuning, use fixed weights
```

**Ablation study — remove one methodology component at a time from the full model:**
```bash
python experiments/run_ablation.py --dataset mri                                        # all 8 variants
python experiments/run_ablation.py --dataset mri --variants full no_resnet_backbone no_nesting  # just the two architectural ones
python experiments/run_ablation.py --dataset mri --variants full bce_only dice_only      # just the loss-composition ones
```
Eight variants: `full` (baseline), `no_cbam`, `no_deep_supervision`,
`no_cbam_no_deep_supervision` (the CBAM×deep-supervision interaction
check), `no_resnet_backbone`, `no_nesting`, `bce_only`, `dice_only` —
one for each of the five methodology components (3.7.1–3.7.5), each an
isolated toggle on the same model class rather than a swap to a
differently-shaped architecture. See `models/resunetpp_cbam.py`'s
module docstring for exactly how `use_resnet`/`nested_decoder` hold
channel counts and scales fixed while removing residual connections or
nesting specifically.

**The cheaper, fixed-weight, single-split comparison (a different, optional question — see the note in the project layout above):**
```bash
python experiments/run_sota_comparison.py --dataset mri
python experiments/run_sota_comparison.py --dataset mri --models resunetpp_cbam unet resunet  # subset
python experiments/run_sota_comparison.py --dataset mri --tune-each  # tune (single split) per model, still no folds
```

**Quick smoke test of everything (few epochs, cheap tuning/folds, both modalities):**
```bash
python experiments/run_all.py --dataset both --epochs 2 --n-trials 2 --tuning-epochs 1 --n-folds 2
```

**Architecture diagram (proposed model only, no GPU/data needed):**
```bash
python -c "from pathlib import Path; from models.diagram import save_architecture_diagram; save_architecture_diagram(Path('results/architecture_diagram.png'))"
```

## Compute expectations

`run_all.py`'s comparison stage is now the dominant cost in this
codebase, by a wide margin. For `n_models` models (7 by default), it
costs roughly:

```
n_models x (n_trials x tuning_epochs + n_folds x config.epochs)
```

epoch-equivalents. With every default (7 models, 15 trials, 5 tuning
epochs, 5 folds), that's `7 x (75 + 5 x config.epochs)` — e.g. at
`config.epochs = 50`, roughly `7 x 325 = 2,275` epoch-equivalents,
before the pipeline's train/ablation stages are even counted. The
ablation stage itself doubled in cost this round too: it's now 8
variants instead of 4 (`8 x config.epochs` per dataset), since it
covers all five methodology components instead of two. Compare all of
that to `run_sota_comparison.py`'s default (no `--tune-each`): just
`n_models x config.epochs` — one training run per model, no tuning, no
folds. That gap is the price of the adviser-specified protocol; it is
not a bug or an inefficiency to fix.

Ways to control the cost, without abandoning the protocol:
- `--models` on `run_all.py` / `run_kfold_cv.py` to compare a subset
  instead of all 7 (e.g. just the proposed model plus its closest
  baselines) while iterating, expanding to the full set for your final
  numbers.
- `--n-folds`, `--n-trials`, `--tuning-epochs`, `--epochs` to shrink
  every stage at once for a smoke test (see the command above).
- `--skip-comparison` / `--skip-ablation` / `--skip-train` on
  `run_all.py` to run only what you need right now.
- Run the comparison stage as its own job
  (`experiments/run_kfold_cv.py`) separately from training the official
  proposed-model result, rather than always paying for both together.

`experiments/run_sota_comparison.py` remains available at its original,
much cheaper cost if you want a fast, controlled, architecture-only
comparison in addition to (not instead of) the optimize→freeze→CV-compare
results.

## Caveats carried over from the tuning/CV work

- Tuning (`tuning/bayesian.py`) is a cheap, single-split proxy (short
  `tuning_epochs`, one seed, one val split, alpha-only search space) —
  see its module docstring for the full list. The k-fold CV step is
  what checks whether that choice of alpha is actually robust across
  different patients/slices; treat a tuned alpha as a good starting
  point until the fold spread confirms it, not before.
- Two different comparison protocols exist and answer different
  questions -- don't mix their numbers up in your write-up.
  `run_kfold_cv.py` (`run_kfold_comparison()`) tunes each model
  independently, freezes, and k-fold CVs it: "how does each
  architecture perform at its own best effort?" `run_sota_comparison.py`
  gives every model the same fixed 0.5/0.5 weight and trains once on the
  fixed split: "controlling for loss balance, how do architectures
  differ?" `run_all.py` now runs the former by default (the
  adviser-specified protocol); the latter stays available standalone
  when the cheaper, controlled question is what you want instead.
- RIDER's k-fold CV is capped at 4-fold (leave-one-patient-out) even
  with correct grouping, since only 4 of its 5 patients are ever pooled
  into train+val — `experiments/run_kfold_cv.py` prints a note about
  this at runtime, and `--n-folds 4` makes it explicit up front.
- `run_sota_comparison.py` uses a fixed 0.5/0.5 loss balance for every
  model by default (`--tune-each` to tune -- still single-split, no
  folds -- per model instead), specifically so architecture is the only
  thing that differs between rows in that comparison table.
- `run_ablation.py`'s eight variants are a leave-one-out design (each
  removes one component from `full`, holding the rest fixed), not the
  full factorial across all four architectural flags — that would be up
  to 12 valid combinations before even counting loss composition. Pass
  `--variants` with any subset (including combinations not listed above)
  to extend it if you want more than leave-one-out for your defense.
- `unetpp` and `cbam_unet` (from `run_kfold_cv.py`'s comparison, not
  from `run_ablation.py`) are NOT substitutes for `no_resnet_backbone` /
  `no_nesting` above — each differs from the proposed model on more than
  one axis at once (`unetpp` has no ResNet backbone AND no CBAM
  simultaneously; `cbam_unet` has no ResNet AND no nesting), so a delta
  against either one mixes multiple effects together rather than
  isolating a single component the way the ablation study's flags do.
- `ResUNet++` and `RA-UNet` are reimplementations of the components and
  data flow described in their papers (SE blocks + ASPP bridge +
  attention decoder; residual blocks + attention residual modules,
  2D-adapted from the original 3D design), not verified line-for-line
  ports of the authors' original code — cross-reference the papers if
  exact fidelity matters for your defense.
