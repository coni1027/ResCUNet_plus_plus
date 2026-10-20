# Basically the pipeline here is

## Load raw (DICOM / volume) -> Normalize -> Mean Filter -> CLAHE -> Resize (highly variable) -> save as .npy file

- transform.py -> normalize, mean filter (cv2.blur 3x3), clahe (2.0, 8x8), resize
- common.py -> pair image and mask, patient split check, make manifest CSV to trace .npy back to raw file, check output
- prepare_mammograms.py ->  DICOM -> transform -> .npy
- prepare_mri.py -> volume -> slices -> transforms -> .npy