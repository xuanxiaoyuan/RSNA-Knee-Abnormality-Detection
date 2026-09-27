"""Score the extracted labels against the 58 gold studies.

This is what the gold labels are for. Nothing else in the project can tell us
whether the report-derived labels are any good, and the image model inherits
whatever error rate shows up here.

    python src/eval_extractor.py --derived work/labels_derived.csv

Reference point: report-derived labels are reported to agree with the
image-derived gold labels only around 82% of the time. Treat a number near
that as the expected ceiling, not as a bug.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import discover_labels, load_config, read_metadata, resolve_data_dir  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/extract.yaml")
    parser.add_argument("--derived", default=None,
                        help="labels CSV from extract_labels.py")
    parser.add_argument("--worst", type=int, default=5,
                        help="how many worst-disagreeing studies to print")
    return parser.parse_args()


def contingency(gold, pred):
    tp = int(((gold == 1) & (pred == 1)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    tn = int(((gold == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall and not np.isnan(precision + recall) else float("nan"))
    return tp, fp, fn, tn, precision, recall, f1


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_dir = resolve_data_dir(cfg, search_roots=("/kaggle/input", "."))
    labels = discover_labels(data_dir)

    train = read_metadata(data_dir, "train.csv")
    gold = train[train[labels].notna().all(axis=1)][["StudyInstanceUID"] + labels]

    derived_path = Path(args.derived) if args.derived else (
        Path(cfg.get("work_dir", "work")) / cfg.get("output", "labels_derived.csv"))
    derived = pd.read_csv(derived_path)
    print(f"[eval] gold={len(gold)} studies, derived={len(derived)} studies from {derived_path}")

    merged = gold.merge(derived, on="StudyInstanceUID", suffixes=("_gold", "_pred"))
    if merged.empty:
        raise SystemExit(
            "[eval] no overlap between the gold studies and the derived labels. "
            "Run extract_labels.py on the full train set, not a disjoint subset."
        )
    print(f"[eval] scoring on {len(merged)} studies with both\n")

    gold_mat = merged[[f"{c}_gold" for c in labels]].to_numpy(dtype=float)
    pred_mat = merged[[f"{c}_pred" for c in labels]].to_numpy(dtype=float)

    # -1 marks a parse failure. Excluded per-cell so one bad generation does not
    # get scored as a confident negative.
    failed_cells = int((pred_mat < 0).sum())
    valid = pred_mat >= 0
    if failed_cells:
        print(f"[eval] {failed_cells} cells were parse failures and are excluded\n")

    header = f"{'finding':<20}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}{'prec':>7}{'rec':>7}{'F1':>7}"
    print(header)
    print("-" * len(header))
    f1s, accs = [], []
    rows = []
    for j, name in enumerate(labels):
        keep = valid[:, j]
        g, p = gold_mat[keep, j], pred_mat[keep, j]
        if not len(g):
            print(f"{name:<20}  (no valid predictions)")
            continue
        tp, fp, fn, tn, prec, rec, f1 = contingency(g, p)
        acc = (tp + tn) / len(g)
        f1s.append(f1)
        accs.append(acc)
        rows.append((name, acc, f1, tp, fp, fn))
        print(f"{name:<20}{tp:>4}{fp:>4}{fn:>4}{tn:>4}{prec:>7.3f}{rec:>7.3f}{f1:>7.3f}")

    macro_f1 = float(np.nanmean(f1s)) if f1s else float("nan")
    print("-" * len(header))
    print(f"{'macro':<20}{'':>16}{'':>7}{'':>7}{macro_f1:>7.3f}")

    overall = float(np.nanmean(accs))
    print(f"\n  macro F1        : {macro_f1:.4f}")
    print(f"  macro accuracy  : {overall:.4f}")
    print(f"  exact bit match : {float((gold_mat[valid] == pred_mat[valid]).mean()):.4f}")
    print("  (report-vs-image agreement is reportedly around 0.82, so a value "
          "near that is the ceiling, not a bug)")

    mismatches = (gold_mat != pred_mat) & valid
    per_study = mismatches.sum(axis=1)
    order = np.argsort(-per_study)
    if args.worst and per_study[order[0]] > 0:
        print(f"\n  Worst {args.worst} studies (most disagreeing cells):")
        for idx in order[: args.worst]:
            uid = merged["StudyInstanceUID"].iloc[idx]
            wrong = [labels[j] for j in range(len(labels)) if mismatches[idx, j]]
            print(f"    {per_study[idx]:>2} wrong  {uid}")
            print(f"              {', '.join(wrong)}")

    weak = sorted(rows, key=lambda r: r[2])[:3]
    print("\n  Weakest findings by F1 (where prompt work should go next):")
    for name, acc, f1, tp, fp, fn in weak:
        print(f"    {name:<20} F1={f1:.3f}  (missed {fn}, over-called {fp})")


if __name__ == "__main__":
    main()
