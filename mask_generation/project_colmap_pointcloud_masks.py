#!/usr/bin/env python3
"""Project a COLMAP-aligned binary PLY point cloud into per-view silhouette masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ]
)


def read_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        count = None
        while True:
            line = stream.readline().strip()
            if line.startswith(b"element vertex"):
                count = int(line.split()[-1])
            if line == b"end_header":
                break
        if count is None:
            raise ValueError(f"No vertex count in {path}")
        points = np.fromfile(stream, dtype=PLY_DTYPE, count=count)
    return np.column_stack((points["x"], points["y"], points["z"]))


def quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_images(path: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    poses = []
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        fields = line.split()
        if len(fields) >= 10:
            qvec = np.asarray(fields[1:5], dtype=float)
            tvec = np.asarray(fields[5:8], dtype=float)
            poses.append((fields[9], quaternion_to_rotation(qvec), tvec))
            index += 2  # COLMAP's following line contains 2D observations.
        else:
            index += 1
    return sorted(poses, key=lambda item: item[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-cloud", type=Path, required=True)
    parser.add_argument("--images-txt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-width", type=int, default=1588)
    parser.add_argument("--source-height", type=int, default=1191)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--fx", type=float, default=2887.92)
    parser.add_argument("--fy", type=float, default=2888.73)
    parser.add_argument("--cx", type=float, default=794.0)
    parser.add_argument("--cy", type=float, default=595.5)
    parser.add_argument("--splat", type=int, default=9)
    parser.add_argument("--close", type=int, default=31)
    args = parser.parse_args()

    xyz = read_ply_xyz(args.point_cloud).astype(np.float64)
    poses = read_images(args.images_txt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sx, sy = args.width / args.source_width, args.height / args.source_height
    splat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.splat, args.splat))
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.close, args.close))

    for frame, (name, rotation, translation) in enumerate(poses):
        camera = xyz @ rotation.T + translation
        valid = camera[:, 2] > 0
        camera = camera[valid]
        u = (args.fx * camera[:, 0] / camera[:, 2] + args.cx) * sx
        v = (args.fy * camera[:, 1] / camera[:, 2] + args.cy) * sy
        valid = (u >= 0) & (u < args.width) & (v >= 0) & (v < args.height)
        u = np.clip(np.rint(u[valid]).astype(int), 0, args.width - 1)
        v = np.clip(np.rint(v[valid]).astype(int), 0, args.height - 1)
        raw = np.zeros((args.height, args.width), dtype=np.uint8)
        raw[v, u] = 255
        raw = cv2.dilate(raw, splat)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close)
        contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(raw)
        if contours:
            cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        output = args.output_dir / f"{frame:03d}.png"
        if not cv2.imwrite(str(output), mask):
            raise IOError(output)
        print(f"[{frame + 1:02d}/{len(poses):02d}] {name} -> {output.name}", flush=True)


if __name__ == "__main__":
    main()
