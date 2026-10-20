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