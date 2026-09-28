"""Train on the labelled studies and write submission.csv.

Run it from the repo root:

    python src/train.py --config configs/base.yaml

The first run should be in debug mode. It prints the real data layout, walks a
handful of studies through the full pipeline, and writes a submission — which
is how we find out what this dataset actually looks like without shipping it
to a laptop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import DEFAULT_SLOT_KEYS, KneeStudyDataset, collate_studies  # noqa: E402
from model import build_model  # noqa: E402
from utils import (  # noqa: E402
    describe,
    discover_labels,
    load_config,
    read_metadata,
    resolve_data_dir,
    seed_everything,
)

FINGERPRINT_TAGS = (
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "SoftwareVersions",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--debug", dest="debug", action="store_true", default=None)
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    return parser.parse_args()


def print_version_marker():
    """Show which code revision is running.

    The dataset that feeds this code is refreshed by hand, so a stale run is a
    real and silent failure mode. Printing a marker makes it visible.
    """
    marker = Path(__file__).resolve().parent.parent / "VERSION"
    if marker.exists():
        print(f"[version] {marker.read_text(encoding='utf-8').strip()}")
    else:
        print("[version] no VERSION file found")


def build_groups(data_dir, study_ids, series_df, cache_path, series_root):
    """Map each study to a scanner fingerprint.

    Grouped CV is not optional here: a random split lets the model recognise
    the scanner rather than the pathology, which inflates the score by a lot.
    The fingerprints are cached because the header pass costs a file read per
    study.
    """
    cache_path = Path(cache_path)
    known = {}
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        known = dict(zip(cached["StudyInstanceUID"], cached["group"]))

    todo = [uid for uid in study_ids if uid not in known]
    if todo:
        print(f"[groups] fingerprinting {len(todo)} studies...")
        rows = []
        for uid in tqdm(todo, disable=len(todo) < 20):
            rows.append({"StudyInstanceUID": uid, "group": fingerprint_study(
                data_dir, uid, series_df, series_root)})
        fresh = pd.DataFrame(rows)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            fresh = pd.concat([cached, fresh], ignore_index=True)
        fresh.to_csv(cache_path, index=False)
        known.update(dict(zip(fresh["StudyInstanceUID"], fresh["group"])))

    return [known.get(uid, "unknown") for uid in study_ids]


def fingerprint_study(data_dir, study_uid, series_df, series_root):
    import pydicom

    rows = series_df[series_df["StudyInstanceUID"] == study_uid]
    if rows.empty:
        return "unknown"
    series_uid = rows.iloc[0]["SeriesInstanceUID"]
    files = sorted((Path(data_dir) / series_root / study_uid / series_uid).glob("*.dcm"))
    if not files:
        return "unknown"
    try:
        ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
    except Exception:
        return "unknown"
    return "|".join(str(getattr(ds, tag, "?")) for tag in FINGERPRINT_TAGS)


def _autocast(device_type, enabled):
    try:
        return torch.amp.autocast(device_type, enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=enabled)


def _grad_scaler(device_type, enabled):
    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def masked_bce(logits, targets):
    """BCE that ignores targets we do not have.

    train.csv leaves most label cells empty; treating those as 0 would teach
    the model that nearly everything is absent.
    """
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    mask = ~torch.isnan(targets)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def macro_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score

    scores = []
    for column in range(y_true.shape[1]):
        truth = y_true[:, column]
        keep = ~np.isnan(truth)
        if keep.sum() < 2 or len(np.unique(truth[keep])) < 2:
            continue
        scores.append(roc_auc_score(truth[keep], y_score[keep, column]))
    if not scores:
        return float("nan"), 0
    return float(np.mean(scores)), len(scores)


def run_epoch(model, loader, optimizer, scaler, device, use_amp, max_steps=None):
    model.train()
    total, seen = 0.0, 0
    for step, (images, targets, _) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device.type, use_amp):
            loss = masked_bce(model(images), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach())
        seen += 1
    return total / max(seen, 1)


def evaluate(model, loader, device, use_amp):
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for images, target, _ in loader:
            images = images.to(device, non_blocking=True)
            with _autocast(device.type, use_amp):
                out = model(images)
            logits.append(out.float().cpu().numpy())
            targets.append(target.numpy())
    if not logits:
        return float("nan"), 0
    return macro_auc(np.concatenate(targets), np.concatenate(logits))


def predict(model, loader, device, use_amp):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for images, study_ids in loader:
            images = images.to(device, non_blocking=True)
            with _autocast(device.type, use_amp):
                out = model(images)
            probs = torch.sigmoid(out.float()).cpu().numpy()
            for study_uid, row in zip(study_ids, probs):
                predictions[study_uid] = row
    return predictions


def write_submission(data_dir, work_dir, labels, predictions):
    """Emit submission.csv in exactly sample_submission.csv's column order.

    Row ids and ordering come from sample_submission.csv because at scoring
    time that file is swapped for the real test set.
    """
    template = read_metadata(data_dir, "sample_submission.csv")
    if template is None:
        raise FileNotFoundError("sample_submission.csv is required to build a submission")

    n_missing = 0
    rows = []
    for study_uid in template["StudyInstanceUID"]:
        if study_uid in predictions:
            rows.append(predictions[study_uid])
        else:
            n_missing += 1
            rows.append(np.full(len(labels), 0.5))
    if n_missing:
        print(f"[submission] {n_missing}/{len(template)} rows had no series; filled 0.5")

    out = pd.DataFrame(rows, columns=labels)
    out.insert(0, "StudyInstanceUID", template["StudyInstanceUID"].values)
    out = out[template.columns]

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(work_dir / "submission.csv", index=False)
    print(f"[submission] wrote {work_dir / 'submission.csv'} shape={out.shape}")
    return out


def load_training_labels(train_df, labels, cfg):
    """Assemble the label table the model trains on.

    Only 58 of 4407 studies carry gold labels — far too few to train on. That
    is the whole point of extract_labels.py, which derives labels for the rest
    from their reports. Gold wins wherever both exist: those are the only
    labels a radiologist set from the images rather than from the text.
    """
    frames = []
    derived_path = Path(str(cfg.get("derived_labels") or "labels_derived.csv"))
    if not derived_path.is_absolute():
        derived_path = Path(cfg.get("work_dir", "work")) / derived_path
    if derived_path.exists():
        derived = pd.read_csv(derived_path)
        keep = [c for c in labels if c in derived.columns]
        if keep:
            sub = derived[["StudyInstanceUID"] + keep].copy()
            # -1 is a finding the extractor could not read. Turn it into NaN so
            # masked_bce ignores the cell instead of training on it as a 0.
            sub[keep] = sub[keep].replace(-1, np.nan)
            frames.append(sub)
            print(f"[labels] {len(sub)} studies from {derived_path}")
    else:
        print(f"[labels] {derived_path} not found — training on gold labels only")

    present = [c for c in labels if c in train_df.columns]
    if present:
        gold = train_df.loc[
            train_df[present].notna().any(axis=1), ["StudyInstanceUID"] + present
        ].copy()
        frames.append(gold)
        print(f"[labels] {len(gold)} gold studies override derived on overlap")

    if not frames:
        return pd.DataFrame(columns=["StudyInstanceUID"] + labels)
    merged = pd.concat(frames, ignore_index=True)
    # Gold is appended last, so keep="last" is what makes it win the overlap.
    merged = merged.drop_duplicates(subset="StudyInstanceUID", keep="last")
    merged = merged.reindex(columns=["StudyInstanceUID"] + labels)
    return merged[merged[labels].notna().any(axis=1)].reset_index(drop=True)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.debug is not None:
        cfg["debug"] = args.debug
    debug = bool(cfg.get("debug", False))

    print_version_marker()
    seed_everything(int(cfg.get("seed", 42)))

    data_dir = resolve_data_dir(cfg)
    print(f"[data] using {data_dir}")
    if debug:
        print("[data] debug mode: full data description follows\n")
        describe(data_dir, cfg)
        print()

    labels = discover_labels(data_dir)
    n_labels = len(labels)

    train_df = read_metadata(data_dir, "train.csv")
    train_series = read_metadata(data_dir, "train_series.csv")
    if train_df is None or train_series is None:
        raise FileNotFoundError("train.csv and train_series.csv are both required")

    labelled = load_training_labels(train_df, labels, cfg)
    print(f"[labels] training on {len(labelled)}/{len(train_df)} studies")
    if len(labelled) < 50:
        print(
            "[labels] WARNING: that is far too few to train on, which means "
            "labels_derived.csv is missing — run extract_labels.py first. "
            "Training below is a pipeline smoke test, not a real model."
        )

    if debug:
        labelled = labelled.head(int(cfg.get("debug_n_studies", 24)))
        print(f"[debug] capping labelled studies at {len(labelled)}")

    train_series_df = train_series[
        train_series["StudyInstanceUID"].isin(labelled["StudyInstanceUID"])
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    print(f"[device] {device} amp={use_amp}")

    label_matrix = labelled.reindex(columns=labels).to_numpy(dtype=np.float32)

    dataset_kwargs = dict(
        data_dir=data_dir,
        series_root=cfg.get("train_series_root", "train_series"),
        img_size=int(cfg.get("img_size", 224)),
        slices_per_series=int(cfg.get("slices_per_series", 6)),
        max_series=int(cfg.get("max_series", 4)),
        slot_keys=tuple(cfg.get("series_slot_keys", DEFAULT_SLOT_KEYS)),
    )

    model = build_model(cfg, n_labels, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scaler = _grad_scaler(device.type, use_amp)
    loader_kwargs = dict(
        batch_size=int(cfg.get("batch_size", 4)),
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=collate_studies,
        pin_memory=device.type == "cuda",
    )

    if len(labelled) > 0:
        work_dir = Path(cfg.get("work_dir", "work"))
        groups = build_groups(
            data_dir,
            labelled["StudyInstanceUID"].tolist(),
            train_series,
            work_dir / "study_groups.csv",
            cfg.get("train_series_root", "train_series"),
        )
        from sklearn.model_selection import GroupKFold

        n_folds = int(cfg.get("n_folds", 5))
        n_groups = len(set(groups))
        if n_groups >= n_folds:
            splits = list(GroupKFold(n_splits=n_folds).split(labelled, groups=groups))
            train_idx, val_idx = splits[int(cfg.get("fold", 0)) % n_folds]
        else:
            print(f"[cv] only {n_groups} scanner groups; training on everything")
            train_idx = np.arange(len(labelled))
            val_idx = np.arange(len(labelled))
        print(f"[cv] train={len(train_idx)} val={len(val_idx)} groups={n_groups}")

        train_ds = KneeStudyDataset(
            series_df=train_series_df[train_series_df["StudyInstanceUID"].isin(
                labelled.iloc[train_idx]["StudyInstanceUID"])],
            study_ids=labelled.iloc[train_idx]["StudyInstanceUID"].tolist(),
            labels=label_matrix[train_idx],
            **dataset_kwargs,
        )
        val_ds = KneeStudyDataset(
            series_df=train_series_df[train_series_df["StudyInstanceUID"].isin(
                labelled.iloc[val_idx]["StudyInstanceUID"])],
            study_ids=labelled.iloc[val_idx]["StudyInstanceUID"].tolist(),
            labels=label_matrix[val_idx],
            **dataset_kwargs,
        )
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

        epochs = 1 if debug else int(cfg.get("epochs", 4))
        max_steps = int(cfg.get("debug_max_steps", 2)) if debug else None
        for epoch in range(epochs):
            loss = run_epoch(
                model, train_loader, optimizer, scaler, device, use_amp, max_steps
            )
            auc, n_scored = evaluate(model, val_loader, device, use_amp)
            print(f"[epoch {epoch}] loss={loss:.4f} val_macro_auc={auc:.4f} "
                  f"({n_scored} labels scored)")

        torch.save(model.state_dict(), work_dir / "model.pt")
        print(f"[checkpoint] {work_dir / 'model.pt'}")
    else:
        print("[train] no labelled studies, skipping training and emitting 0.5 baseline")

    test_series = read_metadata(data_dir, "test_series.csv")
    template = read_metadata(data_dir, "sample_submission.csv")
    if test_series is not None and template is not None:
        test_ids = template["StudyInstanceUID"].tolist()
        if debug:
            test_ids = test_ids[: int(cfg.get("debug_n_studies", 24))]
        # dataset_kwargs already carries series_root (pointed at train_series);
        # override it in a copy rather than passing it twice.
        test_kwargs = dict(dataset_kwargs)
        test_kwargs["series_root"] = cfg.get("test_series_root", "test_series")
        test_ds = KneeStudyDataset(
            series_df=test_series,
            study_ids=test_ids,
            labels=None,
            **test_kwargs,
        )
        test_loader = DataLoader(
            test_ds,
            shuffle=False,
            batch_size=loader_kwargs["batch_size"],
            num_workers=loader_kwargs["num_workers"],
            collate_fn=collate_studies,
            pin_memory=loader_kwargs["pin_memory"],
        )
        predictions = predict(model, test_loader, device, use_amp)
    else:
        print("[test] no test_series.csv, emitting 0.5 baseline")
        predictions = {}

    write_submission(data_dir, cfg.get("work_dir", "work"), labels, predictions)
    print("[done]")


if __name__ == "__main__":
    main()
