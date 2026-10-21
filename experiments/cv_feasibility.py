"""Run a small, reproducible CVA + Otsu feasibility check on selected OSCD pairs.

Example:
    python experiments/cv_feasibility.py --root data/raw --pairs beihai rennes
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from earthchange.cv.cva import cva_otsu
from earthchange.cv.oscd_loader import load_oscd_pair


def binary_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    """Calculate pixel-level precision, recall, F1, and IoU for two masks."""
    predicted = np.asarray(prediction, dtype=bool)
    actual = np.asarray(reference, dtype=bool)
    if predicted.shape != actual.shape:
        raise ValueError(f"Mask shapes must match; got {predicted.shape} and {actual.shape}")
    if predicted.ndim != 2:
        raise ValueError("Masks must be two-dimensional")

    true_positive = int(np.count_nonzero(predicted & actual))
    false_positive = int(np.count_nonzero(predicted & ~actual))
    false_negative = int(np.count_nonzero(~predicted & actual))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    union = true_positive + false_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0.0
    iou = true_positive / union if union else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def run(root: Path, pair_ids: list[str], output_path: Path) -> list[dict[str, Any]]:
    """Evaluate selected OSCD pairs and save one CSV row per pair."""
    rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        pair = load_oscd_pair(root, pair_id)
        result = cva_otsu(pair.image_a, pair.image_b)
        metrics = binary_metrics(result.change_mask, pair.change_mask)
        rows.append(
            {
                "pair_id": pair.pair_id,
                "date_a": pair.dates[0] if pair.dates else "",
                "date_b": pair.dates[1] if pair.dates else "",
                "width": pair.image_a_metadata.width,
                "height": pair.image_a_metadata.height,
                "bands": pair.image_a_metadata.band_count,
                "threshold": result.threshold,
                "predicted_change_pixels": int(np.count_nonzero(result.change_mask)),
                "reference_change_pixels": int(np.count_nonzero(pair.change_mask)),
                **metrics,
            }
        )

    if not rows:
        raise ValueError("Select at least one OSCD pair")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="OSCD extracted dataset root")
    parser.add_argument("--pairs", nargs="+", required=True, help="OSCD pair folder IDs to evaluate")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/cv_feasibility.csv"),
        help="CSV output path (default: experiments/outputs/cv_feasibility.csv)",
    )
    args = parser.parse_args()

    rows = run(args.root, args.pairs, args.output)
    print(f"Saved {len(rows)} pair result(s) to {args.output}")
    for row in rows:
        print(
            f"{row['pair_id']}: F1={row['f1']:.4f}, IoU={row['iou']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
