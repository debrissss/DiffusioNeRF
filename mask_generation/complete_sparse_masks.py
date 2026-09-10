#!/usr/bin/env python3
"""Complete a sparse mask set while preserving trusted reference masks exactly."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot decode {path}")
    return mask > 127


def largest_filled_component(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    output = np.zeros(mask.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(output, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return output


def metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    tp = int(np.count_nonzero(reference & prediction))
    fp = int(np.count_nonzero(~reference & prediction))
    fn = int(np.count_nonzero(reference & ~prediction))
    return {
        "iou": tp / (tp + fp + fn),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "dice": 2 * tp / (2 * tp + fp + fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--sam-dir", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=49)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_by_id = {int(path.stem): path for path in args.reference_dir.glob("*.png")}
    source_rows = []
    for frame in range(args.count):
        output = args.output_dir / f"{frame:06d}.png"
        if frame in reference_by_id:
            shutil.copy2(reference_by_id[frame], output)
            source = "trusted_reference"
        else:
            sam = read_mask(args.sam_dir / f"{frame:03d}.png")
            geometry = read_mask(args.geometry_dir / f"{frame:03d}.png")
            result = largest_filled_component(sam | geometry)
            if not cv2.imwrite(str(output), result):
                raise IOError(output)
            source = "sam2_union_colmap9"
        source_rows.append({"frame": frame, "source": source, "file": output.name})

    metric_rows = []
    for frame, reference_path in sorted(reference_by_id.items()):
        row = {"frame": frame}
        row.update(metrics(read_mask(reference_path), read_mask(args.output_dir / f"{frame:06d}.png")))
        metric_rows.append(row)
    macro = {
        key: float(np.mean([row[key] for row in metric_rows]))
        for key in ("iou", "precision", "recall", "dice")
    }
    report = {
        "protocol": "sparse-mask completion; trusted masks retained, missing views generated",
        "total_masks": args.count,
        "trusted_masks": len(reference_by_id),
        "generated_masks": args.count - len(reference_by_id),
        "macro_metrics_on_trusted_ids": macro,
        "per_frame": metric_rows,
        "sources": source_rows,
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    with (args.output_dir / "evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame", "iou", "precision", "recall", "dice"])
        writer.writeheader()
        writer.writerows(metric_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
