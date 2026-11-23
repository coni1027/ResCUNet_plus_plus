# Breast Lesion Segmentation: ResNet34 + U-Net++ + CBAM vs. SOTA Baselines

This project trains and compares **7 segmentation architectures** on two breast-lesion datasets:

1. **Mammograms** (CBIS-DDSM)
2. **2D breast DCE-MRI slices** (MAMA-MIA: DUKE, ISPY1, ISPY2, NACT)

Each dataset is trained and evaluated **independently** (separate splits, checkpoints, results) -- the two are never combined.

The task is **binary semantic segmentation**:

- `0` = background
- `1` = breast lesion

All code referenced below lives under [project/](project/). Run every command from inside `project/`.

---

## Models

Registered in [models/\_\_init\_\_.py](project/models/__init__.py) as `MODEL_REGISTRY`, so any script that takes `--models` accepts these names:

| name             | label                              | source                                                    |
|------------------|-------------------------------------|------------------------------------------------------------|
| `resunetpp_cbam` | ResNet34 U-Net++ + CBAM (**proposed**) | [models/resunetpp_cbam.py](project/models/resunetpp_cbam.py) |
| `unet`           | U-Net                               | [models/unet.py](project/models/unet.py)                   |
| `unetpp`         | U-Net++                             | [models/unetpp.py](project/models/unetpp.py)                |
| `resunet`        | ResUNet                             | [models/resunet.py](project/models/resunet.py)              |
| `resunetpp`      | ResUNet++                           | [models/resunetpp.py](project/models/resunetpp.py)          |
| `ra_unet`        | RA-UNet                             | [models/ra_unet.py](project/models/ra_unet.py)               |
| `cbam_unet`      | CBAM-UNet                           | [models/cbam_unet.py](project/models/cbam_unet.py)           |

The **proposed model** (`resunetpp_cbam`) is a ResNet34 encoder + a nested U-Net++ decoder + CBAM on the decoder skip connections, trained with a hybrid BCE+Dice loss and deep supervision. Every one of those five components (ResNet backbone, nested decoder, CBAM, deep supervision, loss composition) is an independently toggleable flag on the same model class, which is what [experiments/run_ablation.py](project/experiments/run_ablation.py) uses to remove one at a time.

The other six are published SOTA baselines, each isolating a different subset of those components (e.g. `unetpp` has the nested decoder but no ResNet backbone and no CBAM; `cbam_unet` has CBAM on a plain U-Net; `resunetpp` and `ra_unet` are separate published architectures with their own encoder/attention designs). See each model file's module docstring for exactly what it does and does not share with the proposed model.

---

## Installation

Python 3.10+ is recommended.

```bash
pip install torch torchvision numpy opencv-python pydicom optuna
```

- `pydicom` is only needed if you're reading `.dcm` files (CBIS-DDSM mammograms).
- `optuna` is only needed for Bayesian loss-weight tuning (`tuning/bayesian.py` and anything that calls it -- `--tune-each`, `--tune-shared`, `run_kfold_cv.py`, `run_all.py`). Everything else works without it.

A CUDA-compatible GPU is strongly recommended. The code auto-detects it:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## Project Structure

```text
project/
├── config.py                    # DatasetConfig, MAMMOGRAM_CONFIG, MRI_CONFIG, seeding
│
├── data/
│   ├── mammograms/{train,val,test}/{images,masks}/
│   └── breast_mri/{train,val,test}/{images,masks}/
│
├── preprocessing/                # one-time raw-data -> data/ conversion (see below)
├── datasets/                     # PyTorch Dataset, loading, augmentation
├── models/                       # the 7 architectures + shared building blocks (common.py)
├── training/                     # loss, evaluator, training loop / checkpointing
├── tuning/                       # Bayesian (Optuna) BCE/Dice weight search
├── cross_validation/             # patient-grouped k-fold splitting
├── experiments/                  # the CLI entry points you actually run (see below)
│
├── checkpoints/                  # created automatically
├── results/                      # created automatically
└── splits/                       # cached CV fold / patient-split JSON, created automatically
```

There is no single top-level training script -- `experiments/*.py` are the entry points; each is documented below.

---

## Important: Split by Patient

Splits must be done at the **patient level**, not the individual image/slice level, *before* files land in `data/<dataset>/{train,val,test}/`. This matters most for MRI, where one patient contributes many slices -- letting slices from the same patient land in different splits causes data leakage and inflated metrics.

The `preprocessing/` scripts below handle this for you (MAMA-MIA's patient ID is its case folder name, e.g. `DUKE_001`; CBIS-DDSM's comes from its case filename convention). If you're populating `data/` by hand instead, keep to the same rule:

```text
Patient 001 -> train only
Patient 002 -> train only
Patient 003 -> validation only
Patient 004 -> test only
```

---

## Data Preparation

### Breast MRI (MAMA-MIA)

[MAMA-MIA](https://www.synapse.org/Synapse:syn60868042) bundles four TCIA collections -- DUKE, ISPY1, ISPY2 and NACT -- as NIfTI files: one 3D volume per DCE phase (`images/<CASE>/<CASE>_0000.nii.gz` = pre-contrast, `_0001` = 1st post-contrast, ...) plus one expert tumour mask per case (`segmentations/expert/`). The collections have different numbers of phases (e.g. DUKE `_0000`-`_0004`, NACT `_0000`-`_0002`), but the network needs a fixed channel count, so every case is reduced to the three phases all of them have: **`_0000`, `_0001`, `_0002` -> channels 0, 1, 2** (`MRI_CONFIG.in_channels = 3`).

[preparation/prepare_mama_mia.py](project/preparation/prepare_mama_mia.py) does the conversion, once:

- **Split**: test = the official MAMA-MIA test split (`train_test_splits.csv`); val = 20% of the official train patients, drawn separately within each collection (`--val-frac`). Every slice of a patient goes to that patient's split.
- **Slices**: each volume is sliced in the plane it was acquired in -- axial for DUKE and ISPY2, sagittal for all of ISPY1 and NACT (235 cases) -- so no slice is a stretched reformat; `--axial-only` drops the sagittal cases instead. Every tumour slice is kept plus an equal number of tumour-free slices from the same case (`--background-ratio`, `-1` = keep all).
- **Output**: `data/breast_mri/<split>/{images,masks}/<CASE>_slice<###>.npy` -- images `(3, 256, 256)` float16, masks `(256, 256)` uint8 -- plus `extraction_manifest.csv` and `patient_split.json`.

It is meant to run on Google Colab against the copy on Drive. Always dry-run first -- it checks every case (phases present, shapes match, listed in the split file, filename parses back to the patient) and prints the plan and the output size without writing anything:

```bash
python preparation/prepare_mama_mia.py --mama-mia-root /content/drive/MyDrive/thesis/data/MAMA-MIA --out-root /content/mama_mia_2d --dry-run
python preparation/prepare_mama_mia.py --mama-mia-root /content/drive/MyDrive/thesis/data/MAMA-MIA --out-root /content/mama_mia_2d
```

Before spending GPU time, verify the patient grouping used for cross-validation (`NACT_68_slice042.npy` -> `NACT_68`):

```bash
python -c "from config import MRI_CONFIG; from cross_validation.folds import default_patient_id_from_filename, preview_cv_groups; MRI_CONFIG.patient_id_fn = default_patient_id_from_filename; preview_cv_groups(MRI_CONFIG, 5)"
```

The earlier RIDER experiments (code, results, CV split) are archived in [project/archive/rider/](project/archive/rider/README.md).

### Mammograms (CBIS-DDSM)

[preprocessing/prepare_mammograms.py](project/preprocessing/prepare_mammograms.py) finds each case's Whole+ROI DICOM pair, resizes both to 512x512, and writes them into `data/mammograms/<split>/{images,masks}/`.

```bash
python preprocessing/prepare_mammograms.py --dry-run
python preprocessing/prepare_mammograms.py --raw-root /path/to/raw/cbis-ddsm
```

**Read the script's module docstring before trusting its output**: the train/test split it currently produces is a placeholder (everything lands in train/val until `official_split_for_case()` is edited to consult CBIS-DDSM's official `mass/calc_case_description_{train,test}_set.csv` files) -- it prints a warning every run until that's fixed.

---

## Loss Function

Hybrid BCE + Dice ([training/losses.py](project/training/losses.py)), averaged over every deep-supervision output when a model returns more than one:

```text
Loss = bce_weight * BCE + dice_weight * Dice
```

Default per `DatasetConfig`: `bce_weight=0.5`, `dice_weight=0.5`. These can be overridden per training call, and can be found automatically with Bayesian search (`tuning/bayesian.py`, see the "Loss-weight tuning" section under Experiments below).

---

## Dataset Configuration

Both configs live in [config.py](project/config.py):

```python
MAMMOGRAM_CONFIG = DatasetConfig(
    name="mammogram", input_size=(512, 512), in_channels=1,
    batch_size=4, epochs=30, learning_rate=1e-4,
    pretrained_encoder=False, deep_supervision=True,
    clahe=True, median_filter=True,
    horizontal_flip_probability=0.5, rotation_degrees=10.0, translation_fraction=0.03,
)

MRI_CONFIG = DatasetConfig(
    name="breast_mri", input_size=(256, 256), in_channels=3,
    batch_size=8, epochs=30, learning_rate=1e-4,
    pretrained_encoder=False, deep_supervision=True,
    clahe=False, median_filter=False,
    horizontal_flip_probability=0.5, rotation_degrees=20.0, translation_fraction=0.08,
)
```

`pretrained_encoder=False` for both -- the ResNet34 backbone in `resunetpp_cbam`/`resunet`/`resunetpp` trains from scratch, it does not use ImageNet weights by default. `in_channels=3` for MRI is the three DCE phases `_0000/_0001/_0002` stacked per slice by `prepare_mama_mia.py`; the encoder's first convolution is adapted automatically to whatever `in_channels` you set.

---

## Data Loading, Naming, and Augmentation

- Supported formats: `.png .jpg .jpeg .bmp .tif .tiff .npy .dcm` (DICOM requires `pydicom`; `MONOCHROME1` mammograms are handled by inverting stored intensity).
- Each image needs a same-stem mask; recognized mask suffixes: `_mask -mask _seg -seg _label -label`.
- Masks are binarized as `mask > 0`.
- Training-only spatial augmentation (same transform applied to image and mask): horizontal flip, rotation, translation, per the probabilities/ranges in each `DatasetConfig`. Validation and test are never augmented.

See [datasets/breast_dataset.py](project/datasets/breast_dataset.py) for the full pipeline (percentile clipping, CLAHE, median filtering, resize).

---

## Experiments (how to run everything)

All scripts live in `experiments/` and take `--dataset {mammogram,mri,both}`. Run them from `project/`.

### 1. Controlled SOTA comparison -- [run_sota_comparison.py](project/experiments/run_sota_comparison.py)

Trains every model once on the fixed train/val/test split and writes one comparison table (`results/sota_comparison_<dataset>.csv`).

```bash
# default: every model uses the SAME fixed 0.5/0.5 bce/dice weight
# (controlled comparison, isolates architecture differences only)
python experiments/run_sota_comparison.py --dataset mri

# quick smoke test
python experiments/run_sota_comparison.py --dataset both --epochs 10

# just a subset of models
python experiments/run_sota_comparison.py --dataset mri --models resunetpp_cbam unet resunet

# each model gets its OWN Bayesian-tuned weight (no longer controlled for loss balance)
python experiments/run_sota_comparison.py --dataset mri --tune-each

# tune ONCE (against resunetpp_cbam) and share that one weight across every
# model -- controlled AND tuned, instead of the hard-coded 0.5/0.5 default
python experiments/run_sota_comparison.py --dataset mri --tune-shared
```

`--tune-each` and `--tune-shared` are mutually exclusive.

### 2. Optimize-then-cross-validate comparison -- [run_kfold_cv.py](project/experiments/run_kfold_cv.py)

The adviser-specified protocol: for each model, tune its bce/dice weight independently (single split, Bayesian search), freeze it, then run patient-grouped k-fold CV with those frozen weights, then compare every model's fold mean/std. This is a different, more expensive protocol from `run_sota_comparison.py` above -- every model gets its *own* best-effort loss balance rather than a shared one.

```bash
# tune -> freeze -> 5-fold CV for every registered model, then compare
python experiments/run_kfold_cv.py --dataset mri

# only the proposed model
python experiments/run_kfold_cv.py --dataset mri --models resunetpp_cbam --n-folds 3

# skip tuning, use weights you already know
python experiments/run_kfold_cv.py --dataset mri --models resunetpp_cbam --bce-weight 0.4 --dice-weight 0.6

# smoke test
python experiments/run_kfold_cv.py --dataset mri --models unet resunet --n-folds 2 --epochs 2 --n-trials 2 --tuning-epochs 1
```

**!! Remove any stale files under `checkpoints/` and `results/` before a real run !!**

### 3. Ablation study -- [run_ablation.py](project/experiments/run_ablation.py)

Removes exactly one component of the proposed model at a time (ResNet backbone, nested decoder, CBAM, deep supervision, loss composition), everything else held fixed, reusing the same training loop.

```bash
python experiments/run_ablation.py --dataset mri
python experiments/run_ablation.py --dataset both --epochs 10   # smoke test
python experiments/run_ablation.py --dataset mri --variants full no_resnet_backbone no_nesting
```

### 4. Full pipeline -- [run_all.py](project/experiments/run_all.py)

Runs (optionally) tune -> train the proposed model -> ablation -> the full `run_kfold_cv.py` comparison, in one command. This is the most expensive script in the repo (it tunes AND k-fold CVs every requested model). Use `--models`, `--skip-ablation`, `--skip-comparison`, and `--epochs/--n-trials/--tuning-epochs/--n-folds` to shrink it.

```bash
python experiments/run_all.py --dataset mri
python experiments/run_all.py --dataset both --tune-loss-weights
python experiments/run_all.py --dataset mri --epochs 2 --n-trials 2 --tuning-epochs 1 --n-folds 2  # smoke test
```

### Loss-weight tuning on its own -- [tuning/bayesian.py](project/tuning/bayesian.py)

Called internally by the scripts above (`--tune-each`, `--tune-shared`, `--tune-loss-weights`), but callable directly too:

```python
from config import MRI_CONFIG
from tuning.bayesian import tune_bce_dice_weight

bce_weight, dice_weight = tune_bce_dice_weight("resunetpp_cbam", MRI_CONFIG, n_trials=15, tuning_epochs=5)
```

Read the module docstring before trusting the result -- it's a short-epoch, single-split, single-seed proxy search, meant as a starting point to check against `run_kfold_cv.py`'s per-fold spread, not a certified optimum.

---

## Optimizer, Scheduler, Model Selection

- Optimizer: `AdamW`, default `lr=1e-4`, `weight_decay=1e-5`.
- `ReduceLROnPlateau` (factor 0.5, patience 3) monitors validation Dice.
- The checkpoint is updated whenever `val_dice` improves on the current best -- selection is never based on the test set. The test split is evaluated exactly once, after training, using the restored best-validation checkpoint.

---

## Evaluation Metrics

Computed by [training/evaluator.py](project/training/evaluator.py) at a 0.5 probability threshold: **Dice, IoU, Precision, Recall, pixel Accuracy, BCE+Dice loss**. For lesion segmentation, Dice/IoU/precision/recall are generally more informative than accuracy, since background pixels dominate.

---

## Output Files

- **Checkpoints**: `checkpoints/<dataset>_<model_name>_best.pth` (k-fold CV additionally writes `..._fold<N>_best.pth`). Each contains epoch, model/optimizer state, best val Dice, dataset name, label, input size, channel count, and the bce/dice weights used.
- **Training history**: `results/<dataset>_<model_name>_training_history.csv` -- one row per epoch (`epoch, train_loss, val_loss, val_dice, val_iou, val_precision, val_recall, val_accuracy, learning_rate`).
- **Comparison tables**: `results/sota_comparison_<dataset>.csv` (run_sota_comparison.py), `results/<dataset>_<model_name>_kfold_cv.csv` and `results/<dataset>_cv_comparison.csv` (run_kfold_cv.py).
- **Cached splits**: `splits/<dataset>_<n>fold.json` (CV fold assignment), recomputed automatically if the underlying data changes. The MRI patient split is recorded in `data/breast_mri/patient_split.json` by `prepare_mama_mia.py`.

---

## Reproducibility

`SEED = 42` in `config.py`, applied to Python `random`, NumPy, and PyTorch (CPU + CUDA) via `set_seed()`, called before every model is (re-)initialized.

---

## Windows Notes

`NUM_WORKERS = 0` is the default (safest for DataLoader multiprocessing on Windows). Raise it later if you've confirmed multiprocessing works correctly in your environment and want the throughput.
