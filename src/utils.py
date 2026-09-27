"""Config loading, data/label discovery, and the first-run diagnostics.

Deliberately imports nothing from torch or pydicom, so this module (and
anything that only needs metadata) runs on a laptop holding just the CSVs.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Columns that appear alongside the targets but are not targets themselves.
NON_LABEL_COLUMNS = {"StudyInstanceUID", "SeriesInstanceUID", "Report"}

METADATA_FILES = (
    "train.csv",
    "train_series.csv",
    "test.csv",
    "test_series.csv",
    "sample_submission.csv",
)


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config at {path} did not parse to a mapping")
    return cfg


def _find_data_root(root, marker="sample_submission.csv", max_depth=3):
    """Depth-capped search for the directory that holds `marker`.

    The cap is load-bearing. An unbounded walk would descend into the DICOM
    trees — tens of thousands of series directories — before giving up.
    """
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = 0 if rel == Path(".") else len(rel.parts)
        if marker in filenames:
            return Path(dirpath)
        if depth + 1 >= max_depth:
            dirnames[:] = []
    return None


def resolve_data_dir(cfg, search_roots=("/kaggle/input",), max_depth=3):
    """Find the directory that actually holds the competition data.

    Trusts cfg['data_dir'] first, then scans search_roots. The scan is not a
    nicety: Kaggle nests inputs by type, so competition data lands at
    /kaggle/input/competitions/<slug> and datasets at
    /kaggle/input/datasets/<owner>/<slug>. That layout has already moved once
    and the config's path silently went stale with it.
    """
    configured = cfg.get("data_dir")
    if configured and (Path(configured) / "sample_submission.csv").exists():
        return Path(configured)

    searched = []
    for root in search_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        found = _find_data_root(root, max_depth=max_depth)
        if found:
            return found
        searched.append(root)
    raise FileNotFoundError(
        "no sample_submission.csv found under: " + ", ".join(str(s) for s in searched)
    )


def resolve_metadata_dir(cfg):
    """Where the small CSVs live for local, pixel-free development."""
    data_dir = cfg.get("data_dir")
    if data_dir and (Path(data_dir) / "sample_submission.csv").exists():
        return Path(data_dir)
    return Path(cfg.get("local_dir", "data"))


def discover_labels(data_dir):
    """Read the target names from sample_submission.csv instead of hardcoding.

    The label strings include spaces and an apostrophe ("Medial Meniscus",
    "Baker's"), which is exactly the kind of thing that silently breaks a
    hardcoded list.
    """
    sub = pd.read_csv(Path(data_dir) / "sample_submission.csv", nrows=1)
    labels = [c for c in sub.columns if c not in NON_LABEL_COLUMNS]
    if not labels:
        raise ValueError("sample_submission.csv exposed no label columns")
    return labels


def read_metadata(data_dir, name):
    """Load one metadata CSV, or None when it is absent."""
    path = Path(data_dir) / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _rule(title):
    print("=" * 72)
    print(title)
    print("=" * 72)


def describe(data_dir, cfg, n_report_samples=2):
    """Print what the data actually looks like.

    This is the substitute for local testing that the competition's data size
    makes impractical: the first Kaggle run reports the real shape of things
    instead of the pipeline assuming it.
    """
    data_dir = Path(data_dir)
    _rule(f"data_dir = {data_dir}")

    for name in METADATA_FILES:
        path = data_dir / name
        size = f"{path.stat().st_size / 1e6:.1f} MB" if path.exists() else "-"
        print(f"  {name:<24} {'present' if path.exists() else 'MISSING':<8} {size}")
    for sub, root in (
        ("train_series", cfg.get("train_series_root", "train_series")),
        ("test_series", cfg.get("test_series_root", "test_series")),
    ):
        p = data_dir / root
        if p.is_dir():
            print(f"  {root + '/':<24} {len(list(p.iterdir()))} study dirs")
        else:
            print(f"  {root + '/':<24} MISSING")

    labels = discover_labels(data_dir)
    print()
    _rule(f"{len(labels)} target labels (from sample_submission.csv)")
    for lab in labels:
        print(f"  - {lab!r}")

    train = read_metadata(data_dir, "train.csv")
    if train is None:
        print("\n!! train.csv missing, cannot report label coverage")
        return labels

    print()
    _rule(f"train.csv: {len(train)} rows")
    print("  columns: " + ", ".join(repr(c) for c in train.columns))

    present = [c for c in labels if c in train.columns]
    absent = [c for c in labels if c not in train.columns]
    print()
    print("  LABEL COVERAGE — this decides whether this is plain supervised")
    print("  learning or a weak-supervision problem:")
    for col in present:
        n = int(train[col].notna().sum())
        pct = 100.0 * n / max(len(train), 1)
        print(f"    {col:<20} {n:>6} / {len(train)} labelled  ({pct:5.1f}%)")
    if absent:
        print(f"    !! in sample_submission but not in train.csv: {absent}")

    fully = train[present].notna().all(axis=1).sum() if present else 0
    print(f"\n    studies with EVERY label present: {fully} / {len(train)}")

    if "Report" in train.columns:
        reports = train["Report"].dropna()
        print(f"\n  Report column: {len(reports)} non-null")
        for text in reports.head(n_report_samples):
            flat = " ".join(str(text).split())
            print(f"    | {flat[:240]}")

    series = read_metadata(data_dir, "train_series.csv")
    if series is not None:
        print()
        _rule(f"train_series.csv: {len(series)} rows")
        print("  columns: " + ", ".join(repr(c) for c in series.columns))
        for col in cfg.get("series_slot_keys", []):
            if col in series.columns:
                counts = series[col].value_counts(dropna=False).to_dict()
                print(f"    {col}: {counts}")
        if "StudyInstanceUID" in series.columns:
            per_study = series.groupby("StudyInstanceUID").size()
            print(f"    series per study: min={per_study.min()} "
                  f"median={per_study.median():.0f} max={per_study.max()}")

    sub = read_metadata(data_dir, "sample_submission.csv")
    if sub is not None:
        print()
        _rule(f"sample_submission.csv: {len(sub)} rows (3 while interactive)")
        print("  head:")
        print(sub.head(3).to_string(index=False))

    print()
    return labels
