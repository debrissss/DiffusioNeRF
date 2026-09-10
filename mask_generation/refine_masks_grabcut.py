#!/usr/bin/env python3
"""Refine conservative geometry masks with RGB-guided GrabCut."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def image_files(path: Path) -> list[Path]:
    return sorted(
        p for p in path.glob("*.png") if p.is_file() and not p.name.startswith("._")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--geometry-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--working-width", type=int, default=400)
    parser.add_argument("--band", type=int, default=25)
    parser.add_argument("--foreground-erode", type=int, default=5)
    parser.add_argument("--closing", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()

    for name in ("band", "foreground_erode", "closing"):
        value = getattr(args, name)
        if value < 1 or value % 2 == 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive odd integer")

    images = image_files(args.image_dir)
    masks = image_files(args.geometry_mask_dir)
    if not images or len(images) != len(masks):
        raise ValueError(f"Count mismatch: images={len(images)}, masks={len(masks)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.band, args.band))
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (args.foreground_erode, args.foreground_erode)
    )
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.closing, args.closing))

    frame_stats = []
    for index, (image_path, mask_path) in enumerate(zip(images, masks)):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        geometry = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or geometry is None:
            raise ValueError(f"Cannot decode frame {index}")
        height, width = image.shape[:2]
        working_height = round(height * args.working_width / width)
        small_image = cv2.resize(
            image, (args.working_width, working_height), interpolation=cv2.INTER_AREA
        )
        small_geometry = cv2.resize(
            geometry, (args.working_width, working_height), interpolation=cv2.INTER_NEAREST
        ) > 127

        sure_foreground = cv2.erode(
            small_geometry.astype(np.uint8), erode_kernel
        ) > 0
        uncertain_extent = cv2.dilate(
            small_geometry.astype(np.uint8), band_kernel
        ) > 0

        initialization = np.full(small_geometry.shape, cv2.GC_BGD, dtype=np.uint8)
        initialization[uncertain_extent] = cv2.GC_PR_BGD
        initialization[small_geometry] = cv2.GC_PR_FGD
        initialization[sure_foreground] = cv2.GC_FGD

        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            small_image,
            initialization,
            None,
            background_model,
            foreground_model,
            args.iterations,
            cv2.GC_INIT_WITH_MASK,
        )
        refined = np.logical_or(
            initialization == cv2.GC_FGD, initialization == cv2.GC_PR_FGD
        ).astype(np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, close_kernel)
        contours, _ = cv2.findContours(
            refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled = np.zeros_like(refined)
        if contours:
            cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        refined_full = cv2.resize(filled, (width, height), interpolation=cv2.INTER_NEAREST)

        output_path = args.output_dir / f"{index:03d}.png"
        if not cv2.imwrite(str(output_path), refined_full):
            raise IOError(f"Failed to write {output_path}")
        frame_stats.append(
            {
                "frame": index,
                "geometry_pixels": int((geometry > 127).sum()),
                "refined_pixels": int((refined_full > 127).sum()),
            }
        )
        print(f"[{index + 1:02d}/{len(images):02d}] {output_path.name}", flush=True)

    metadata = {
        "image_dir": str(args.image_dir.resolve()),
        "geometry_mask_dir": str(args.geometry_mask_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "frames": len(images),
        "working_width": args.working_width,
        "band": args.band,
        "foreground_erode": args.foreground_erode,
        "closing": args.closing,
        "iterations": args.iterations,
        "frame_stats": frame_stats,
    }
    (args.output_dir / "refinement_stats.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
