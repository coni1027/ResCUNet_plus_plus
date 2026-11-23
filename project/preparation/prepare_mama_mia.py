"""
One-time data preparation for MAMA-MIA: turn the NIfTI release (one
3D volume per DCE phase per case, plus one expert tumour mask per case)
into the per-slice layout datasets_preprocessing/breast_dataset.py reads:

    <out-root>/{train,val,test}/images/<CASE>_slice<###>.npy   (3, H, W) float16
    <out-root>/{train,val,test}/masks/<CASE>_slice<###>.npy    (H, W)    uint8

Input (the Synapse release, as downloaded to Google Drive):

    <mama-mia-root>/
    ├── images/<CASE>/<CASE>_0000.nii.gz ... <CASE>_000N.nii.gz
    ├── segmentations/expert/<case>.nii.gz
    └── train_test_splits.csv

  MAMA-MIA case
      |
      v
  load phases _0000, _0001, _0002 (+ the expert mask), verify same grid
      |
      v
  reorient to RAS, stack phases as channels -> (3, X, Y, Z)
      |
      v
  slice along the acquisition axis (axial for DUKE/ISPY2, sagittal for
  ISPY1/NACT): every slice with tumour + a seeded sample of the rest
      |
      v
  resize to MRI_CONFIG.input_size, save image/mask pairs, patient ID in filename

Design:
  - CHANNELS. The four collections have different phase counts (DUKE
    _0000.._0004, NACT _0000.._0002, ISPY1/ISPY2 vary), but the network
    needs a fixed channel count (MRI_CONFIG.in_channels). Every case is
    reduced to the three phases all collections have: pre-contrast,
    1st post-contrast, 2nd post-contrast -- channel 0, 1, 2 in that
    order. A case missing any of them stops the run (see --dry-run).
    Phase index is only roughly comparable across collections: _0002 was
    not acquired at the same time after injection everywhere.
  - SPLITS. test = the official MAMA-MIA test split (train_test_splits.csv),
    untouched, so results stay comparable with other MAMA-MIA work. val is
    carved out of the official train split BY PATIENT, separately within
    each collection (so DUKE/ISPY1/ISPY2/NACT are all represented in val
    in their natural proportions), seeded. Every slice of a case goes to
    that case's split, so no patient crosses train/val/test.
  - ORIENTATION / PLANE. MAMA-MIA does no resampling and keeps each scan
    in its acquisition plane: DUKE and ISPY2 are axial (stored LAS), ALL
    of ISPY1 and NACT are sagittal (stored PSL/PSR) -- 235 of 1,506 cases
    (MAMA-MIA paper, Table I). Each volume is reoriented to RAS and cut
    along its ACQUISITION axis, so every slice keeps full in-plane
    resolution: axial slices have anterior at the top, sagittal slices
    superior at the top. (Re-cutting a sagittal 256x256x60 volume axially
    would give 256x60 slices stretched to 256x256.) The training set is
    therefore mixed-plane; the manifest's slice_plane column records which
    plane each slice is. --axial-only drops the sagittal cases instead,
    i.e. all of ISPY1 and NACT.
  - SLICES. All slices containing tumour are kept, plus
    --background-ratio x that many tumour-free slices from the same case
    (seeded per case, so --dry-run and the real run pick identical
    slices). --background-ratio -1 keeps every slice. Keeping everything
    is expensive -- --dry-run prints the output size for the current
    setting AND for keeping all slices, so you can choose.
  - INTENSITIES. Each case's three phases are divided by ONE shared
    constant (their joint maximum) before saving as float16. That keeps
    the relative enhancement between phases intact on disk. Note the
    loader (preprocess_image) still percentile-normalizes each channel
    of each slice independently at training time.
  - PATIENT ID. "<CASE>_slice<###>.npy" -- e.g. "DUKE_001_slice042.npy".
    cross_validation.folds.default_patient_id_from_filename parses that
    back to "DUKE_001"; this script verifies that round trip for every
    case before writing anything, because a mismatch would silently
    turn patient-grouped CV folds into per-slice folds.
  - RESUME. Each finished case leaves a marker in <out-root>/_done/.
    --resume skips finished cases, so a crashed run can continue. On
    Colab that only helps across disconnects if --out-root is on Drive.

ALWAYS run with --dry-run first. It reads every case's headers and mask,
checks phases/shapes/orientation/split membership, and prints the full
plan and estimated output size without writing anything.

Usage (Colab, after mounting Drive and cloning the repo; run from project/):
  python preparation/prepare_mama_mia.py \\
      --mama-mia-root /content/drive/MyDrive/thesis/data/MAMA-MIA \\
      --out-root /content/mama_mia_2d --dry-run
  python preparation/prepare_mama_mia.py \\
      --mama-mia-root /content/drive/MyDrive/thesis/data/MAMA-MIA \\
      --out-root /content/mama_mia_2d
  # quick end-to-end smoke test on a handful of cases:
  python preparation/prepare_mama_mia.py ... --limit-per-collection 3

Requires nibabel (preinstalled on Colab; otherwise `pip install nibabel`).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MRI_CONFIG, SEED
from cross_validation.folds import default_patient_id_from_filename
from datasets_preprocessing.breast_dataset import resize_image, resize_mask
from preparation.common import check_output_is_clean, split_patients, write_manifest

try:
    import nibabel as nib
except ImportError:  # checked in main() so --help still works without it
    nib = None

PHASES = (0, 1, 2)          # _0000 pre-contrast, _0001 1st post, _0002 2nd post
COLLECTIONS = ("DUKE", "ISPY1", "ISPY2", "NACT")
SPLITS = ("train", "val", "test")
DONE_DIR = "_done"


@dataclass
class Case:
    case_id: str                                  # "DUKE_001" (upper case, as in images/<CASE>/)
    collection: str                               # "DUKE"
    available_phases: list[int]
    phase_paths: list[Path] = field(default_factory=list)   # PHASES only, in order
    mask_path: Path | None = None
    split: str | None = None
    # filled by inspect_case()
    shape: tuple[int, ...] | None = None
    axcodes: tuple[str, ...] | None = None
    slice_axis: int = 2                           # RAS axis the volume is cut along (= its acquisition axis)
    n_slices: int = 0
    selected: list[int] = field(default_factory=list)
    lesion_slices: set[int] = field(default_factory=set)


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------

def _nifti_stem(name: str) -> str | None:
    lower = name.lower()
    for suffix in (".nii.gz", ".nii"):
        if lower.endswith(suffix):
            return lower[: -len(suffix)]
    return None


def discover_cases(root: Path, seg_subdir: str, problems: list[tuple[str, str]]) -> list[Case]:
    """
    One Case per folder under images/. File lookups are case-insensitive
    (the release mixes "NACT_68_0000.nii.gz" folders/files with lower-case
    mask names in some copies), so naming case never drops a case.
    """
    images_root = root / "images"
    seg_dir = root / "segmentations" / seg_subdir
    if not images_root.is_dir():
        raise FileNotFoundError(f"No images/ folder under {root}")
    if not seg_dir.is_dir():
        found = sorted(p.name for p in (root / "segmentations").iterdir()) if (root / "segmentations").is_dir() else []
        raise FileNotFoundError(
            f"No {seg_dir}. segmentations/ contains: {found}. Pass --seg-subdir to pick one."
        )

    masks = {}
    for path in seg_dir.iterdir():
        stem = _nifti_stem(path.name)
        if stem is not None:
            masks[stem.upper()] = path

    cases: list[Case] = []
    for case_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        case_id = case_dir.name.upper()
        prefix = case_id + "_"
        phase_files: dict[int, Path] = {}
        for path in case_dir.iterdir():
            stem = _nifti_stem(path.name)
            if stem is None:
                continue
            stem = stem.upper()
            if stem.startswith(prefix) and stem[len(prefix):].isdigit():
                phase_files[int(stem[len(prefix):])] = path

        case = Case(case_id=case_id, collection=case_id.split("_")[0], available_phases=sorted(phase_files))
        missing = [f"_{k:04d}" for k in PHASES if k not in phase_files]
        if missing:
            problems.append((case_id, f"missing phase(s) {missing} (has {[f'_{k:04d}' for k in case.available_phases]})"))
        else:
            case.phase_paths = [phase_files[k] for k in PHASES]
        case.mask_path = masks.get(case_id)
        if case.mask_path is None:
            problems.append((case_id, f"no mask named {case_id.lower()}.nii.gz in {seg_dir}"))
        cases.append(case)

    if not cases:
        raise FileNotFoundError(f"No case folders found under {images_root}")
    return cases


# -----------------------------------------------------------------------------
# Splits
# -----------------------------------------------------------------------------

def read_official_split(csv_path: Path) -> dict[str, str]:
    """
    case_id -> "train" / "test" from MAMA-MIA's train_test_splits.csv.
    Accepts either layout, since this script can't see your copy:
      - wide: one column per split, each listing case IDs
              (e.g. "train_split", "test_split"; columns can differ in length)
      - long: one ID column + one split column
              (e.g. "patient_id", "split")
    Anything else stops with the header printed, so it can be adapted.
    """
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)

    lower = {c.strip().lower(): c for c in columns}
    id_col = next((lower[k] for k in ("patient_id", "case_id", "patient", "case", "id") if k in lower), None)
    split_col = next((lower[k] for k in ("split", "set", "subset", "fold") if k in lower), None)

    def normalize_split(value: str) -> str | None:
        value = value.strip().lower()
        for name in SPLITS:
            if value.startswith(name):
                return name
        return "val" if value.startswith("valid") else None

    assignment: dict[str, str] = {}

    def assign(case_id: str, split: str) -> None:
        case_id = case_id.strip().upper()
        if not case_id:
            return
        if assignment.get(case_id, split) != split:
            raise ValueError(f"{csv_path.name}: {case_id} is listed in both {assignment[case_id]!r} and {split!r}.")
        assignment[case_id] = split

    if id_col and split_col:
        for row in rows:
            split = normalize_split(row[split_col] or "")
            if split is None:
                raise ValueError(f"{csv_path.name}: unrecognized split value {row[split_col]!r} for {row[id_col]!r}.")
            assign(row[id_col] or "", split)
    else:
        split_columns = {c: normalize_split(c) for c in columns}
        split_columns = {c: s for c, s in split_columns.items() if s is not None}
        if not {"train", "test"} <= set(split_columns.values()):
            raise ValueError(
                f"Can't interpret {csv_path} -- columns are {columns}. Expected either one column per "
                "split (train..., test...) or an ID column plus a split column. Adapt read_official_split()."
            )
        for row in rows:
            for column, split in split_columns.items():
                assign(row[column] or "", split)

    if not assignment:
        raise ValueError(f"{csv_path} lists no cases.")
    return assignment


def assign_splits(
    cases: list[Case],
    official: dict[str, str],
    val_frac: float,
    seed: int,
    problems: list[tuple[str, str]],
) -> None:
    """
    test = official test. val = val_frac of the official train cases,
    drawn per collection (stratified), by patient. If the official file
    already has a val split, it is used as-is and nothing is carved out.
    """
    official_has_val = "val" in official.values()
    train_by_collection: dict[str, list[str]] = defaultdict(list)

    for case in cases:
        split = official.get(case.case_id)
        if split is None:
            problems.append((case.case_id, "not listed in train_test_splits.csv"))
            continue
        case.split = split
        if split == "train" and not official_has_val:
            train_by_collection[case.collection].append(case.case_id)

    if official_has_val or val_frac <= 0:
        return

    val_ids: set[str] = set()
    for collection, ids in sorted(train_by_collection.items()):
        groups = split_patients(sorted(ids), seed, {"train": 1.0 - val_frac, "val": val_frac})
        val_ids.update(groups["val"])

    for case in cases:
        if case.case_id in val_ids:
            case.split = "val"


# -----------------------------------------------------------------------------
# Per-case inspection / slicing
# -----------------------------------------------------------------------------

def _canonical(path: Path):
    return nib.as_closest_canonical(nib.load(str(path)))


def _case_rng(case_id: str, seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(case_id.encode())])


def select_slices(lesion_per_slice: np.ndarray, background_ratio: float, rng: np.random.Generator) -> list[int]:
    lesion = np.flatnonzero(lesion_per_slice)
    if background_ratio < 0:
        return list(range(len(lesion_per_slice)))
    background = np.flatnonzero(~lesion_per_slice)
    n_background = min(len(background), int(round(background_ratio * len(lesion))))
    chosen = rng.choice(background, size=n_background, replace=False) if n_background else np.array([], dtype=int)
    return sorted(int(z) for z in np.concatenate([lesion, chosen]))


def inspect_case(
    case: Case, background_ratio: float, seed: int, problems: list[tuple[str, str]], warnings: list[str],
) -> None:
    """Header checks for every phase + mask; loads only the (small) mask."""
    phase_imgs = [nib.load(str(p)) for p in case.phase_paths]
    mask_img = nib.load(str(case.mask_path))

    shapes = {tuple(img.shape) for img in phase_imgs} | {tuple(mask_img.shape)}
    if len(shapes) != 1:
        problems.append((case.case_id, f"phases/mask have different shapes {sorted(shapes)}"))
        return
    shape = shapes.pop()
    if len(shape) != 3:
        problems.append((case.case_id, f"expected 3D volumes, got shape {shape}"))
        return
    if not all(np.allclose(img.affine, mask_img.affine, atol=1e-3) for img in phase_imgs):
        warnings.append(f"{case.case_id}: affines differ slightly between phases/mask (same voxel grid, continuing)")

    case.shape = shape
    case.axcodes = tuple(nib.aff2axcodes(mask_img.affine))

    case.slice_axis = acquisition_axis(case.axcodes)
    if case.slice_axis is None:
        problems.append((case.case_id, f"can't tell the acquisition plane from orientation {''.join(case.axcodes)}"))
        return

    mask = np.asanyarray(nib.as_closest_canonical(mask_img).dataobj) > 0      # (X, Y, Z), RAS
    in_plane = tuple(a for a in range(3) if a != case.slice_axis)
    lesion_per_slice = mask.any(axis=in_plane)
    case.n_slices = int(mask.shape[case.slice_axis])
    case.lesion_slices = set(int(z) for z in np.flatnonzero(lesion_per_slice))
    if not case.lesion_slices:
        warnings.append(f"{case.case_id}: expert mask is empty")
    case.selected = select_slices(lesion_per_slice, background_ratio, _case_rng(case.case_id, seed))


# A volume's acquisition plane is its STORED third axis: MAMA-MIA keeps axial
# scans as LAS (3rd axis S) and sagittal ones as PS? (3rd axis L or R).
# After reorienting to RAS, that axis is 0 (L/R), 1 (A/P) or 2 (S/I).
_RAS_AXIS = {"L": 0, "R": 0, "A": 1, "P": 1, "S": 2, "I": 2}
PLANE_NAMES = {0: "sagittal", 1: "coronal", 2: "axial"}


def acquisition_axis(axcodes: tuple[str, ...]) -> int | None:
    return _RAS_AXIS.get(axcodes[2])


def is_axial(case: Case) -> bool:
    return case.slice_axis == 2


def _to_display(plane: np.ndarray) -> np.ndarray:
    """
    (..., a, b) RAS plane (the two in-plane axes, in RAS order) ->
    (..., rows=b from high to low, cols=a). Axial (X, Y): anterior at the
    top. Sagittal (Y, Z) and coronal (X, Z): superior at the top.
    """
    return np.ascontiguousarray(np.swapaxes(plane, -1, -2)[..., ::-1, :])


def write_case(case: Case, out_root: Path, size: tuple[int, int], dtype: np.dtype) -> list[dict]:
    phases = [_canonical(p).get_fdata(dtype=np.float32) for p in case.phase_paths]
    volume = np.stack(phases, axis=0)                                   # (3, X, Y, Z)
    del phases
    mask = np.asanyarray(_canonical(case.mask_path).dataobj) > 0        # (X, Y, Z)

    scale = float(volume.max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{case.case_id}: image volume is empty or non-finite (max={scale})")
    volume /= scale

    images_out = out_root / case.split / "images"
    masks_out = out_root / case.split / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    rows = []
    ax = case.slice_axis
    for z in case.selected:
        image = resize_image(_to_display(np.take(volume, z, axis=ax + 1)), size)                  # (3, H, W)
        mask_slice = resize_mask(_to_display(np.take(mask, z, axis=ax)).astype(np.uint8), size)   # (H, W)
        filename = f"{case.case_id}_slice{z:03d}.npy"
        np.save(images_out / filename, image.astype(dtype))
        np.save(masks_out / filename, (mask_slice > 0).astype(np.uint8))
        rows.append({
            "filename": filename,
            "patient_id": case.case_id,
            "collection": case.collection,
            "split": case.split,
            "slice_idx": z,
            "slice_plane": PLANE_NAMES[ax],
            "has_lesion": z in case.lesion_slices,
            "lesion_pixels": int((mask_slice > 0).sum()),
            "source_images": ";".join(str(p) for p in case.phase_paths),
            "source_mask": str(case.mask_path),
            "original_shape": "x".join(map(str, case.shape)),
            "original_orientation": "".join(case.axcodes),
        })
    return rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.1f} GB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slice MAMA-MIA NIfTI volumes (phases _0000/_0001/_0002 as channels) into "
                    "<out-root>/{train,val,test}/{images,masks}/, split by patient."
    )
    parser.add_argument("--mama-mia-root", type=Path, required=True,
                        help="Folder holding images/, segmentations/, train_test_splits.csv.")
    parser.add_argument("--out-root", type=Path, default=MRI_CONFIG.root,
                        help=f"Where the 2D slices go (default {MRI_CONFIG.root}).")
    parser.add_argument("--seg-subdir", default="expert", help="Subfolder of segmentations/ to use.")
    parser.add_argument("--splits-csv", type=Path, default=None,
                        help="Default: <mama-mia-root>/train_test_splits.csv")
    parser.add_argument("--collections", nargs="+", default=list(COLLECTIONS), type=str.upper,
                        help=f"Subset of {COLLECTIONS} to include.")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Fraction of each collection's official-train patients moved to val.")
    parser.add_argument("--background-ratio", type=float, default=1.0,
                        help="Tumour-free slices kept per tumour slice, per case. -1 = keep every slice.")
    parser.add_argument("--axial-only", action="store_true",
                        help="Skip non-axial acquisitions (all of ISPY1 and NACT are sagittal) instead of "
                             "slicing them in their own plane.")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--limit-per-collection", type=int, default=None,
                        help="Only the first N cases of each collection -- smoke tests only.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-problem-cases", action="store_true",
                        help="Proceed without cases that failed a check, instead of stopping.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue into a non-empty --out-root, skipping cases already finished.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check everything and print the plan + size estimate; write nothing.")
    args = parser.parse_args()

    if nib is None:
        raise SystemExit("nibabel is not installed: pip install nibabel")
    unknown = set(args.collections) - set(COLLECTIONS)
    if unknown:
        parser.error(f"Unknown collection(s) {sorted(unknown)}; choose from {COLLECTIONS}.")
    if not 0 <= args.val_frac < 1:
        parser.error("--val-frac must be in [0, 1).")
    if args.mama_mia_root.resolve() == args.out_root.resolve():
        parser.error("--out-root must not be the MAMA-MIA folder itself.")

    size = tuple(MRI_CONFIG.input_size)
    if MRI_CONFIG.in_channels != len(PHASES):
        raise SystemExit(
            f"config.MRI_CONFIG.in_channels is {MRI_CONFIG.in_channels} but this script writes "
            f"{len(PHASES)} channels (phases {PHASES}). Make them agree first."
        )
    dtype = np.dtype(args.dtype)

    problems: list[tuple[str, str]] = []   # (case_id, message)
    warnings: list[str] = []

    # 1. Discover + filter
    cases = discover_cases(args.mama_mia_root, args.seg_subdir, problems)
    phase_hist = Counter((c.collection, len(c.available_phases)) for c in cases)
    cases = [c for c in cases if c.collection in args.collections]
    problems = [p for p in problems if p[0].split("_")[0] in args.collections]
    if args.limit_per_collection:
        kept, seen = [], Counter()
        for case in cases:
            if seen[case.collection] < args.limit_per_collection:
                kept.append(case)
                seen[case.collection] += 1
        cases = kept
        kept_ids = {c.case_id for c in cases}
        problems = [p for p in problems if p[0] in kept_ids]

    print(f"Found {len(cases)} case(s) under {args.mama_mia_root / 'images'}")
    print("Phases available per case (collection: n_phases -> n_cases):")
    for collection in COLLECTIONS:
        counts = {n: k for (col, n), k in sorted(phase_hist.items()) if col == collection}
        if counts:
            print(f"  {collection:<6} " + ", ".join(f"{n} phases: {k}" for n, k in counts.items()))

    # 2. Splits
    official = read_official_split(args.splits_csv or args.mama_mia_root / "train_test_splits.csv")
    assign_splits(cases, official, args.val_frac, args.seed, problems)
    on_disk = {c.case_id for c in cases}
    not_downloaded = Counter(
        cid.split("_")[0] for cid in official
        if cid not in on_disk and cid.split("_")[0] in args.collections
    )
    if not_downloaded and not args.limit_per_collection:
        warnings.append(
            "listed in train_test_splits.csv but not found under images/ (skipped): "
            + ", ".join(f"{col} {n}" for col, n in sorted(not_downloaded.items()))
        )

    # 3. Filenames must parse back to the patient (grouped CV depends on it)
    for case in cases:
        parsed = default_patient_id_from_filename(Path(f"{case.case_id}_slice000.npy"))
        parsed_other = default_patient_id_from_filename(Path(f"{case.case_id}_slice137.npy"))
        if parsed != case.case_id or parsed_other != case.case_id:
            problems.append((
                case.case_id,
                "cross_validation.folds.default_patient_id_from_filename parses its slices as "
                f"{parsed!r}/{parsed_other!r} -- CV folds would not group by patient.",
            ))

    # 4. Headers + masks
    ready = [c for c in cases if c.split and c.phase_paths and c.mask_path]
    print(f"\nInspecting {len(ready)} case(s) (headers + masks)...")
    for i, case in enumerate(ready, 1):
        try:
            inspect_case(case, args.background_ratio, args.seed, problems, warnings)
        except Exception as exc:  # unreadable/corrupt file: report, keep checking the rest
            problems.append((case.case_id, f"could not read ({exc})"))
        if i % 100 == 0:
            print(f"  {i}/{len(ready)}")

    problem_ids = {case_id for case_id, _ in problems}
    inspected = [c for c in ready if c.shape is not None and c.case_id not in problem_ids]
    skipped_planes = [c for c in inspected if not is_axial(c) and args.axial_only]
    plan = [c for c in inspected if c not in skipped_planes]

    # 5. Report
    per_slice_bytes = len(PHASES) * size[0] * size[1] * dtype.itemsize + size[0] * size[1] + 256
    print(f"\nPlan (slices resized to {size[0]}x{size[1]}, {args.dtype}, "
          f"background ratio {args.background_ratio}):")
    print(f"  {'split':<6}{'collection':<12}{'cases':>7}{'slices':>9}{'tumour':>9}{'all slices':>12}")
    totals = Counter()
    for split in SPLITS:
        for collection in COLLECTIONS:
            group = [c for c in plan if c.split == split and c.collection == collection]
            if not group:
                continue
            n_sel = sum(len(c.selected) for c in group)
            n_les = sum(len(c.lesion_slices) for c in group)
            n_all = sum(c.n_slices for c in group)
            totals.update(selected=n_sel, lesion=n_les, all=n_all)
            print(f"  {split:<6}{collection:<12}{len(group):>7}{n_sel:>9}{n_les:>9}{n_all:>12}")
    print(f"  {'total':<18}{len(plan):>7}{totals['selected']:>9}{totals['lesion']:>9}{totals['all']:>12}")
    print(f"\nEstimated output size: {_gb(totals['selected'] * per_slice_bytes)} "
          f"(keeping every slice instead: {_gb(totals['all'] * per_slice_bytes)})")
    planes = Counter((c.collection, PLANE_NAMES[c.slice_axis]) for c in plan)
    print("Slice plane per collection: " + ", ".join(
        f"{col} {plane} {n}" for (col, plane), n in sorted(planes.items())))
    if skipped_planes:
        print(f"Skipping {len(skipped_planes)} non-axial case(s) (--axial-only): "
              + ", ".join(f"{c.case_id}[{''.join(c.axcodes)}]" for c in skipped_planes))

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings[:30]:
            print(f"  - {w}")
        if len(warnings) > 30:
            print(f"  ... and {len(warnings) - 30} more")

    if problems:
        print(f"\n{len(problems)} problem(s) in {len(problem_ids)} case(s), all EXCLUDED from the plan above:")
        for case_id, message in problems[:50]:
            print(f"  - {case_id}: {message}")
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
        if not args.skip_problem_cases:
            print("STOP: nothing written. Fix these, narrow --collections, or pass "
                  "--skip-problem-cases to proceed without those cases.")
            sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: nothing written. Re-run without --dry-run once this plan looks right.")
        return

    # 6. Write
    done_dir = args.out_root / DONE_DIR
    if args.resume:
        done_dir.mkdir(parents=True, exist_ok=True)
    else:
        check_output_is_clean(args.out_root)
        done_dir.mkdir(parents=True, exist_ok=True)

    (args.out_root / "patient_split.json").write_text(json.dumps({
        split: sorted(c.case_id for c in plan if c.split == split) for split in SPLITS
    } | {
        "skipped_non_axial": sorted(c.case_id for c in skipped_planes),
        "sagittal_or_coronal": sorted(c.case_id for c in plan if not is_axial(c)),
        "settings": {
            "phases": [f"_{k:04d}" for k in PHASES], "size": list(size), "dtype": args.dtype,
            "background_ratio": args.background_ratio, "val_frac": args.val_frac, "seed": args.seed,
            "collections": args.collections, "limit_per_collection": args.limit_per_collection,
            "axial_only": args.axial_only,
        },
    }, indent=2))

    for i, case in enumerate(plan, 1):
        marker = done_dir / f"{case.case_id}.json"
        if args.resume and marker.exists():
            continue
        rows = write_case(case, args.out_root, size, dtype)
        marker.write_text(json.dumps(rows))
        n_les = sum(r["has_lesion"] for r in rows)
        print(f"  [{i}/{len(plan)}] {case.split:<5} {case.case_id}: {len(rows)} slices ({n_les} tumour)")

    manifest_rows = []
    for case in plan:
        manifest_rows.extend(json.loads((done_dir / f"{case.case_id}.json").read_text()))
    write_manifest(manifest_rows, args.out_root / "extraction_manifest.csv")

    print("\nWrote:")
    for split in SPLITS:
        rows = [r for r in manifest_rows if r["split"] == split]
        print(f"  {split}: {len(rows)} slices ({sum(r['has_lesion'] for r in rows)} with tumour) "
              f"from {len({r['patient_id'] for r in rows})} patients")


if __name__ == "__main__":
    main()
