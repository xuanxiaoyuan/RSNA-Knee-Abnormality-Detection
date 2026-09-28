"""DICOM reading and the study-level Dataset.

This is the only module that touches pixel data, which is what lets every
other part of the pipeline be developed against the small metadata CSVs.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

DEFAULT_SLOT_KEYS = ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression")


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
    """One item == one study, flattened to (T, H, W) grayscale slices.

    Also emits a (T,) mask that is 1 on real slices and 0 on the black padding.
    The model needs it: studies hold 3-14 series and get padded up to a fixed
    slot count, so without the mask a short study's findings are averaged
    against black filler.
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
    ):
        self.data_dir = Path(data_dir)
        self.study_ids = list(study_ids)
        self.labels = labels
        self.series_root = series_root
        self.img_size = img_size
        self.slices_per_series = slices_per_series
        self.max_series = max_series
        self.slot_keys = tuple(slot_keys)
        self.slices_per_study = max_series * slices_per_series
        self._by_study = {
            uid: grp for uid, grp in series_df.groupby("StudyInstanceUID", sort=False)
        }

    def __len__(self):
        return len(self.study_ids)

    def _load_study(self, study_uid):
        rows = self._by_study.get(study_uid)
        if rows is None:
            return np.zeros((0, self.img_size, self.img_size), dtype=np.float32)

        picked = select_series(rows, self.max_series, self.slot_keys)
        chunks = []
        for series_uid in picked["SeriesInstanceUID"]:
            vol = read_series(
                self.data_dir / self.series_root / study_uid / series_uid,
                self.img_size,
                n_keep=self.slices_per_series,
            )
            if vol is not None:
                chunks.append(vol)
        if not chunks:
            return np.zeros((0, self.img_size, self.img_size), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def __getitem__(self, index):
        study_uid = self.study_ids[index]
        vol = self._load_study(study_uid)

        # Pad or truncate to a fixed T so batches stack, and pad rather than
        # error so a study with missing series still contributes.
        target = self.slices_per_study
        n_real = min(vol.shape[0], target)
        if vol.shape[0] < target:
            pad = np.zeros((target - vol.shape[0], self.img_size, self.img_size), np.float32)
            vol = np.concatenate([vol, pad], axis=0)
        else:
            vol = vol[:target]

        mask = np.zeros(target, dtype=np.float32)
        mask[:n_real] = 1.0
        image = torch.from_numpy(np.ascontiguousarray(vol))
        mask = torch.from_numpy(mask)
        if self.labels is None:
            return image, mask, study_uid
        return image, torch.from_numpy(self.labels[index]), mask, study_uid


def collate_studies(batch):
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[-2] for b in batch])
    study_ids = [b[-1] for b in batch]
    if len(batch[0]) == 3:
        return images, masks, study_ids
    return images, torch.stack([b[1] for b in batch]), masks, study_ids
