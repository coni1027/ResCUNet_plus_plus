# Basically the pipeline here is

## Load raw (DICOM / volume) -> Normalize -> Mean Filter -> CLAHE -> Resize (highly variable) -> save as .npy file

- transforms.py -> normalize, mean filter (cv2.blur 3x3), clahe (2.0, 8x8), resize
- common.py -> pair image and mask, patient split check, make manifest CSV to trace .npy back to raw file, check output
- prepare_mammograms.py ->  DICOM -> transform -> .npy
- prepare_mri.py -> volume -> slices -> transforms -> .npy


## Additional step to accomodate comments
Mammograms
- Load DICOM -> remove background -> normalize -> mean filter -> CLAHE -> resize -> save as .npy
MRI
- same without background removal

removed background to make values consistent, and separate it from the breast regions

blacked out background instead of cropping as cropping then resizing to 1024x1024 or any square ratio would stretch it and distort anatomy

### 1. Prepare RIDER MRI
Raw data expected at `data/MRI/rider_breast_mri/RIDER-<id>/...` (TCIA download).
```bash
python data_preparation/prepare_mri.py --dry-run   # check patients/splits, writes nothing
python data_preparation/prepare_mri.py             # writes data/breast_mri/{train,val,test}/{images,masks}
```
Patient split is set in `PATIENT_SPLIT` at the top of `prepare_mri.py`.

### 2. Prepare CBIS-DDSM mammograms
Raw data must already be split: `<raw-root>/{train,val,test}/{images,masks}/P_xxxxx_SIDE_VIEW.dcm`.
```bash
python data_preparation/prepare_mammograms.py --raw-root "D:/THESIS/CBIS DDSM dataset" --dry-run
python data_preparation/prepare_mammograms.py --raw-root "D:/THESIS/CBIS DDSM dataset"
```
Background removal is ON by default. To build the dataset without it (for comparison):
```bash
python data_preparation/prepare_mammograms.py --raw-root "D:/THESIS/CBIS DDSM dataset" --no-background-removal
--out-root data/mammograms_no_bg
```
Full run: ~20 min, ~13 GB at 1024x1024.

### Re-running
The scripts refuse to overwrite existing output. Delete the output folder first
(`data/breast_mri/` or `data/mammograms/`). If the number or names of files changed,
also delete the cached CV folds in `splits/*fold.json`.

### Comparison figure (original / preprocessed / background removed)
```bash
python data_preparation/transform_thresholding.py <image.dcm> <mask.dcm> [<image.dcm> <mask.dcm> ...]
```
Writes `results/threshold_comparison.png`.

### Settings
All preprocessing settings live in `config.py` (`MRI_CONFIG`, `MAMMOGRAM_CONFIG`):
`input_size`, `mean_filter`, `clahe`, `percentile_clip`. Background threshold (0.02) is
`BACKGROUND_THRESHOLD` in `transform_thresholding.py`.

### After preparing data: training
```bash
python experiments/run_all.py --dataset mri
python experiments/run_all.py --dataset mammogram --tune-loss-weights
```