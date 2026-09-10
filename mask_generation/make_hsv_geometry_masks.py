#!/usr/bin/env python3
"""Create coarse single-subject geometry masks from HSV saturation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--saturation-threshold", type=int, default=86)
    args = parser.parse_args()

    images = sorted(
        p for p in args.image_dir.glob("*.png")
        if p.is_file() and not p.name.startswith("._")
    )
    if not images:
        raise ValueError(f"No PNG images in {args.image_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(images):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {image_path}")
        saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1]
        raw = (saturation >= args.saturation_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        mask = np.zeros(raw.shape, dtype=np.uint8)
        if contours:
            cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        output = args.output_dir / f"{index:03d}.png"
        if not cv2.imwrite(str(output), mask):
            raise IOError(output)
        print(f"[{index + 1:02d}/{len(images):02d}] {output.name}", flush=True)


if __name__ == "__main__":
    main()
