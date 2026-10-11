"""DICOM reading and the study-level Dataset.

This is the only module that touches pixel data, which is what lets every
other part of the pipeline be developed against the small metadata CSVs.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

DEFAULT_SLOT_KEYS = ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression")

# Protocol slot identity for the model's per-series embedding. A series is keyed
# by its (plane, fluid) metadata, NOT by its position in the picked list: that
# order is arbitrary (select_series sorts by group size), so a position key
# would hand a different series the same slot from study to study. Index 0 is
# reserved for padding (an absent series). The dataset emits these ids and the
# model looks them up, so both sides share this count.
PLANE_ORDER = ("Sagittal", "Coronal", "Axial")
FLUID_VALUES = (0, 1)
N_SLOTS = 1 + len(PLANE_ORDER) * len(FLUID_VALUES)


def slot_id(plane, fluid):
    """Map a series' protocol metadata to its embedding index (0 = unknown)."""
    try:
        plane_idx = PLANE_ORDER.index(plane)
    except ValueError:
        return 0
    try:
        fluid_bit = 1 if int(fluid) == 1 else 0
    except (TypeError, ValueError):
        fluid_bit = 0
    return 1 + plane_idx * len(FLUID_VALUES) + fluid_bit


def _slice_location(ds):
    """Position of a slice along the acquisition normal.

    Projecting onto the cross product of ImageOrientationPatient rather than
    just taking ImagePositionPatient[2] keeps oblique acquisitions in the right
    order, which a plain z-sort silently scrambles.
    """
    pos = getattr(ds, "ImagePositionPatient", None)
    if pos is None or len(pos) != 3:
        return float(getattr(ds, "InstanceNumber", 0))
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is not None and len(iop) == 6:
        row = np.asarray(iop[:3], dtype=np.float64)
        col = np.asarray(iop[3:], dtype=np.float64)
        normal = np.cross(row, col)
    else:
        normal = np.array([0.0, 0.0, 1.0])
    return float(np.dot(np.asarray(pos, dtype=np.float64), normal))


def read_series(series_dir, img_size, n_keep=None):
    """Decode one series to a normalised float32 (n_keep, H, W) volume.

    Two passes: the first reads headers only to work out slice order, the
    second decodes just the slices that survive sampling. Decoding all ~30
    slices to then throw most away dominates runtime otherwise.
    """
    series_dir = Path(series_dir)
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        return None

    located = []
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue
        located.append((_slice_location(ds), path))
    if not located:
        return None
    located.sort(key=lambda item: item[0])

    if n_keep is not None and n_keep < len(located):
        picks = np.linspace(0, len(located) - 1, n_keep).round().astype(int)
        located = [located[i] for i in picks]

    slices = []
    photometric = ""
    for _, path in located:
        try:
            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        if not photometric:
            photometric = getattr(ds, "PhotometricInterpretation", "")
        slope = getattr(ds, "RescaleSlope", None)
        intercept = getattr(ds, "RescaleIntercept", None)
        if slope is not None:
            arr = arr * float(slope)
        if intercept is not None:
            arr = arr + float(intercept)
        slices.append(arr)
    if not slices:
        return None

    vol = np.stack(slices)
    if photometric == "MONOCHROME1":
        vol = vol.max() - vol

    lo, hi = (float(v) for v in np.percentile(vol, [0.5, 99.5]))
    if hi - lo < 1e-6:
        lo, hi = float(vol.min()), float(vol.max()) + 1e-6
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)

    return np.stack(
        [cv2.resize(s, (img_size, img_size), interpolation=cv2.INTER_AREA) for s in vol]
    ).astype(np.float32)


def select_series(series_df, max_series, slot_keys=DEFAULT_SLOT_KEYS):
    """Pick up to max_series series covering distinct protocol slots.

    Slots are (plane, fluid-sensitivity, fat-sat) combinations, so each kept
    series contributes a different view rather than four near-duplicates. The
    group sizes stand in for slice count, which avoids opening any file.
    """
    if series_df.empty:
        return series_df

    keys = [k for k in slot_keys if k in series_df.columns]
    if not keys:
        return series_df.head(max_series)

    groups = list(series_df.groupby(keys, dropna=False))
    groups.sort(key=lambda kv: len(kv[1]), reverse=True)
    return pd.concat([g for _, g in groups[:max_series]], ignore_index=True)


class KneeStudyDataset(Dataset):
    """One item == one study, as (S, K, H, W) grayscale slices plus an (S, K) mask.

    S is the series slot count, K the slices kept per series. A study holds
    3-14 series, so S is padded up to `max_series`, and a series with fewer than
    K slices leaves trailing black slices in K. The mask is 1 on real slices and
    0 on that padding. The series dimension is kept *separate* from the slices
    rather than flattened into one (S*K,) run: the model needs the grouping to
    tell which slices belong to the same series, and a flat concatenation threw
    that structure away before the model could see it.
    """

    def __init__(
        self,
        data_dir,
        series_df,
        study_ids,
        labels=None,
        series_root="train_series",
        img_size=224,
        slices_per_series=6,
        max_series=4,
        slot_keys=DEFAULT_SLOT_KEYS,
        cache_dir=None,
    ):
        self.data_dir = Path(data_dir)
        self.study_ids = list(study_ids)
        self.labels = labels
        self.series_root = series_root
        self.img_size = img_size
        self.slices_per_series = slices_per_series
        self.max_series = max_series
        self.slot_keys = tuple(slot_keys)
        self._by_study = {
            uid: grp for uid, grp in series_df.groupby("StudyInstanceUID", sort=False)
        }

        # Decoding a study costs ~1.5 s against ~0.1 s of GPU, so without this
        # the loader starves the GPU and every epoch re-reads the same pixels.
        # The tag folds in every parameter that changes the decoded volume, so
        # an edit to img_size or the slice counts starts a fresh cache instead
        # of silently serving arrays of the old shape.
        self.cache_dir = None
        if cache_dir is not None:
            # "series-v2" pins the cached layout: an older cache held a flat
            # (S*K, H, W) array without slot ids, which would load mostly right
            # into the new (S, K, H, W) reader and silently mis-group slices.
            spec = "|".join((
                "series-v2", str(series_root), str(img_size), str(max_series),
                str(slices_per_series), ",".join(self.slot_keys),
            ))
            self.cache_dir = Path(cache_dir) / hashlib.md5(spec.encode()).hexdigest()[:10]

    def __len__(self):
        return len(self.study_ids)

    def _load_study(self, study_uid):
        """Return (volume, mask, slots): (S, K, H, W), (S, K), (S,).

        volume and mask are padded to the fixed (max_series, slices_per_series)
        shape so batches stack; the mask is 1 on real slices and 0 on the
        padding. slots holds the per-series embedding index from `slot_id`
        (0 for a padding slot).
        """
        shape = (self.max_series, self.slices_per_series, self.img_size, self.img_size)
        vol = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape[:2], dtype=np.float32)
        slots = np.zeros(self.max_series, dtype=np.int64)

        cache_file = None
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{study_uid}.npz"
            if cache_file.exists():
                try:
                    # Bundled as .npz (uint8 volume + uint8 mask + slot ids):
                    # one byte per pixel instead of four keeps the whole set
                    # under Kaggle's ~20 GB working dir, and the mask rides
                    # along because it cannot be re-derived from pixel content —
                    # a real slice may legitimately be black.
                    with np.load(cache_file) as cached:
                        return (cached["vol"].astype(np.float32) / 255.0,
                                cached["mask"].astype(np.float32),
                                cached["slot"].astype(np.int64))
                except Exception as exc:
                    print(f"[cache] {cache_file} unreadable ({exc}); re-decoding")

        rows = self._by_study.get(study_uid)
        if rows is not None:
            picked = select_series(rows, self.max_series, self.slot_keys)
            for slot, (_, row) in enumerate(picked.iterrows()):
                if slot >= self.max_series:
                    break
                slots[slot] = slot_id(row.get("Anatomical_Plane"),
                                      row.get("Fluid_Sensitive"))
                series = read_series(
                    self.data_dir / self.series_root / study_uid /
                    row["SeriesInstanceUID"],
                    self.img_size,
                    n_keep=self.slices_per_series,
                )
                if series is None:
                    continue
                k = min(series.shape[0], self.slices_per_series)
                vol[slot, :k] = series[:k]
                mask[slot, :k] = 1.0

        if cache_file is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Write beside the target and rename, so a worker killed mid-write
            # leaves a .tmp rather than a half-written file that a later run
            # would happily load as garbage.
            tmp = cache_file.with_name(cache_file.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.savez(fh,
                         vol=np.rint(vol * 255.0).astype(np.uint8),
                         mask=mask.astype(np.uint8),
                         slot=slots.astype(np.uint8))
            os.replace(tmp, cache_file)
        return vol, mask, slots

    def __getitem__(self, index):
        study_uid = self.study_ids[index]
        vol, mask, slots = self._load_study(study_uid)
        image = torch.from_numpy(np.ascontiguousarray(vol))
        mask = torch.from_numpy(mask)
        slots = torch.from_numpy(slots)
        if self.labels is None:
            return image, mask, slots, study_uid
        return image, torch.from_numpy(self.labels[index]), mask, slots, study_uid


def collate_studies(batch):
    # Sample layouts: train (image, label, mask, slots, uid); test
    # (image, mask, slots, uid). Indexing from the end keeps both working.
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[-3] for b in batch])
    slots = torch.stack([b[-2] for b in batch])
    study_ids = [b[-1] for b in batch]
    if len(batch[0]) == 4:
        return images, masks, slots, study_ids
    return images, torch.stack([b[1] for b in batch]), masks, slots, study_ids
