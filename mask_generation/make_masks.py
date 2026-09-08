#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DTU_ROOT = Path("/root/autodl-tmp/dtu_mvs/extracted")
CALIB_DIR = DTU_ROOT / "SampleSet/MVS Data/Calibration/cal18"


def load_vertices(path):
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if line.startswith(b"element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == b"end_header":
                break

        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        vertices = np.fromfile(f, dtype=dtype, count=vertex_count)

    return vertices["x"], vertices["y"], vertices["z"]


def make_mask(x, y, z, projection):
    depth = (
        projection[2, 0] * x
        + projection[2, 1] * y
        + projection[2, 2] * z
        + projection[2, 3]
    )
    visible = depth > 0
    x, y, z, depth = x[visible], y[visible], z[visible], depth[visible]

    u = (
        projection[0, 0] * x
        + projection[0, 1] * y
        + projection[0, 2] * z
        + projection[0, 3]
    ) / depth
    v = (
        projection[1, 0] * x
        + projection[1, 1] * y
        + projection[1, 2] * z
        + projection[1, 3]
    ) / depth

    u = np.rint(u / 2).astype(np.int32)
    v = np.rint(v / 2).astype(np.int32)
    inside = (u >= 0) & (u < 800) & (v >= 0) & (v < 600)

    mask = np.zeros((600, 800), dtype=np.uint8)
    mask[v[inside], u[inside]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)

    mask = cv2.resize(filled, (400, 300), interpolation=cv2.INTER_AREA)
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", type=int)
    args = parser.parse_args()

    ply_path = DTU_ROOT / f"Points/stl/stl{args.scan:03d}_total.ply"
    output_dir = PROJECT_ROOT / f"data/DTU_standard/scan{args.scan}/masks_diy"
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, z = load_vertices(ply_path)
    success = 0
    failed = 0

    for frame_id in range(49):
        try:
            projection = np.loadtxt(
                CALIB_DIR / f"pos_{frame_id + 1:03d}.txt", dtype=np.float32
            )
            mask = make_mask(x, y, z, projection)
            written = cv2.imwrite(str(output_dir / f"{frame_id:06d}.png"), mask)
            success += int(written)
            failed += int(not written)
        except Exception:
            failed += 1

    print(f"成功: {success}, 失败: {failed}")


if __name__ == "__main__":
    main()
