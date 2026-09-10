#!/usr/bin/env python3
"""Project a filtered DTU reference point cloud into binary object masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from filter_dtu_subject import read_binary_vertex_ply


def rasterize_mask(
    xyz: np.ndarray,
    projection: np.ndarray,
    height: int,
    width: int,
    dilation: int,
    closing: int,
) -> tuple[np.ndarray, dict]:
    camera = xyz @ projection[:, :3].T + projection[:, 3]
    positive_depth = camera[:, 2] > 0
    camera = camera[positive_depth]

    u = np.rint(camera[:, 0] / camera[:, 2]).astype(np.int32)
    v = np.rint(camera[:, 1] / camera[:, 2]).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    raw = np.zeros((height, width), dtype=np.uint8)
    raw[v[inside], u[inside]] = 255

    if dilation > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        raw = cv2.dilate(raw, kernel)
    if closing > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing, closing))
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(raw)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)

    stats = {
        "positive_depth_points": int(positive_depth.sum()),
        "projected_in_frame_points": int(inside.sum()),
        "raw_occupied_pixels": int((raw > 0).sum()),
        "final_foreground_pixels": int((mask > 0).sum()),
    }
    return mask, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dilation", type=int, default=9)
    parser.add_argument("--closing", type=int, default=31)
    args = parser.parse_args()

    for name, value in (("dilation", args.dilation), ("closing", args.closing)):
        if value < 1 or value % 2 == 0:
            raise ValueError(f"--{name} must be a positive odd integer")

    vertices, _ = read_binary_vertex_ply(args.points)
    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float64, copy=False
    )
    cameras = np.load(args.cameras)
    image_paths = sorted(
        p for p in args.image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and not p.name.startswith("._")
    )
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")

    sample = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise ValueError(f"Cannot decode image: {image_paths[0]}")
    height, width = sample.shape[:2]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_stats = []
    for index, image_path in enumerate(image_paths):
        key = f"world_mat_{index}"
        if key not in cameras:
            raise KeyError(f"Missing {key} in {args.cameras}")
        projection = np.asarray(cameras[key], dtype=np.float64)[:3, :4]
        mask, stats = rasterize_mask(
            xyz, projection, height, width, args.dilation, args.closing
        )
        output_path = args.output_dir / f"{index:03d}.png"
        if not cv2.imwrite(str(output_path), mask):
            raise IOError(f"Failed to write {output_path}")
        stats.update({"frame": index, "image": image_path.name, "mask": output_path.name})
        frame_stats.append(stats)
        print(f"[{index + 1:02d}/{len(image_paths):02d}] {output_path.name}", flush=True)

    summary = {
        "points": str(args.points.resolve()),
        "cameras": str(args.cameras.resolve()),
        "image_dir": str(args.image_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "image_size": [width, height],
        "frames": len(frame_stats),
        "dilation": args.dilation,
        "closing": args.closing,
        "postprocess": "largest external contour filled",
        "frame_stats": frame_stats,
    }
    (args.output_dir / "generation_stats.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
