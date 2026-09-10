#!/usr/bin/env python3
"""Compare generated DTU masks with reference masks and make a preview."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_binary(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode mask: {path}")
    return image > 127


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    union = tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_pixels": int(pred.sum()),
        "target_pixels": int(target.sum()),
        "iou": tp / union if union else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-frames", default="0,10,20,30,40,48")
    parser.add_argument("--frames", default="", help="Comma-separated frame indices; empty means all")
    parser.add_argument("--prediction-label", default="generated mask")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = sorted(p for p in args.generated.glob("*.png") if not p.name.startswith("._"))
    references = sorted(p for p in args.reference.glob("*.png") if not p.name.startswith("._"))
    images = sorted(p for p in args.image_dir.glob("*.png") if not p.name.startswith("._"))
    if not (len(references) == len(images)):
        raise ValueError(
            f"Count mismatch: reference={len(references)}, images={len(images)}"
        )

    generated_by_name = {p.name: p for p in generated}
    selected_frames = (
        [int(value) for value in args.frames.split(",") if value.strip()]
        if args.frames
        else list(range(len(images)))
    )

    rows = []
    pairs = {}
    for index in selected_frames:
        pred_path = generated_by_name.get(f"{index:03d}.png")
        if pred_path is None:
            raise ValueError(f"Missing generated mask for frame {index}: {index:03d}.png")
        target_path = references[index]
        pred = load_binary(pred_path)
        target = load_binary(target_path)
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch at frame {index}: {pred.shape} vs {target.shape}")
        row = {"frame": index, **metrics(pred, target)}
        rows.append(row)
        pairs[index] = (pred, target)

    metric_names = ["iou", "precision", "recall", "dice"]
    summary = {
        "frames": len(rows),
        "generated": str(args.generated.resolve()),
        "reference": str(args.reference.resolve()),
        "macro_mean": {name: float(np.mean([r[name] for r in rows])) for name in metric_names},
        "macro_min": {name: float(np.min([r[name] for r in rows])) for name in metric_names},
        "macro_max": {name: float(np.max([r[name] for r in rows])) for name in metric_names},
        "frames_detail": rows,
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    preview_frames = [int(value) for value in args.preview_frames.split(",")]
    fig, axes = plt.subplots(len(preview_frames), 4, figsize=(14, 3 * len(preview_frames)), dpi=130)
    for row_index, frame in enumerate(preview_frames):
        pred, target = pairs[frame]
        rgb = cv2.cvtColor(cv2.imread(str(images[frame])), cv2.COLOR_BGR2RGB)
        overlay = np.zeros((*pred.shape, 3), dtype=np.uint8)
        overlay[np.logical_and(pred, target)] = (30, 190, 70)   # true positive: green
        overlay[np.logical_and(~pred, target)] = (230, 50, 50)  # false negative: red
        overlay[np.logical_and(pred, ~target)] = (40, 110, 240) # false positive: blue
        panels = [rgb, target, pred, overlay]
        titles = [
            f"RGB frame {frame}",
            "IDR manual mask",
            args.prediction_label,
            f"TP green / FN red / FP blue\nIoU={next(r for r in rows if r['frame'] == frame)['iou']:.3f}",
        ]
        for col, (panel, title) in enumerate(zip(panels, titles)):
            axes[row_index, col].imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            axes[row_index, col].set_title(title)
            axes[row_index, col].axis("off")
    fig.tight_layout()
    fig.savefig(args.output_dir / "comparison_preview.png", bbox_inches="tight")
    print(json.dumps(summary["macro_mean"], indent=2))


if __name__ == "__main__":
    main()
