#!/usr/bin/env python3
"""Convert the official CoR-GS DTU package into this project's JSON format."""

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


STANDARD_SCANS = [8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114]
IMAGE_RE = re.compile(r"rect_(\d+)_")


def closest_point_2_lines(oa, da, ob, db):
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -1 + 1e-10:
        raise ValueError("Cannot robustly align opposite up vectors")
    s = np.linalg.norm(v)
    kmat = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
        dtype=np.float64,
    )
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def normalized_poses(poses_bounds):
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    poses = np.concatenate(
        [poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]],
        axis=-1,
    )
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))
    poses = np.concatenate([poses, last_row], axis=1)

    poses[:, 0:3, 1] *= -1
    poses[:, 0:3, 2] *= -1
    poses = poses[:, [1, 0, 2, 3], :]
    poses[:, 2, :] *= -1

    up = poses[:, 0:3, 1].sum(axis=0)
    up /= np.linalg.norm(up)
    rotation = np.pad(rotmat(up, np.array([0, 0, 1])), [0, 1])
    rotation[-1, -1] = 1
    poses = rotation @ poses

    total_weight = 0.0
    center = np.zeros(3, dtype=np.float64)
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            point, weight = closest_point_2_lines(
                poses[i, :3, 3], poses[i, :3, 2],
                poses[j, :3, 3], poses[j, :3, 2],
            )
            if weight > 0.01:
                center += point * weight
                total_weight += weight
    if total_weight == 0:
        raise ValueError("Failed to estimate a camera look-at center")
    poses[:, :3, 3] -= center / total_weight
    average_radius = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()
    poses[:, :3, 3] *= 4.0 / average_radius
    return poses


def parse_camera(camera_path):
    rows = [
        line.strip() for line in camera_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one camera in {camera_path}, got {len(rows)}")
    fields = rows[0].split()
    model = fields[1]
    width, height = int(fields[2]), int(fields[3])
    params = [float(value) for value in fields[4:]]
    if model == "PINHOLE":
        fx, fy, cx, cy = params
    elif model == "SIMPLE_PINHOLE":
        fx, cx, cy = params
        fy = fx
    else:
        raise ValueError(f"Unsupported camera model {model} in {camera_path}")
    return width, height, fx, fy, cx, cy


def image_id(path):
    match = IMAGE_RE.search(path.name)
    if not match:
        raise ValueError(f"Unexpected DTU image name: {path.name}")
    return int(match.group(1)) - 1


def convert_scan(source_root, output_root, scan_id, width, height):
    source = source_root / f"scan{scan_id}"
    output = output_root / f"scan{scan_id}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    images = sorted(source.joinpath("images").glob("*.png"), key=image_id)
    ids = [image_id(path) for path in images]
    if ids != list(range(49)):
        raise ValueError(f"scan{scan_id} image IDs are not exactly 0..48: {ids}")

    poses_bounds = np.load(source / "poses_bounds.npy")
    if poses_bounds.shape != (49, 17):
        raise ValueError(f"scan{scan_id} poses_bounds has shape {poses_bounds.shape}")
    poses = normalized_poses(poses_bounds)

    source_width, source_height, fx, fy, cx, cy = parse_camera(
        source / "sparse" / "0" / "cameras.txt"
    )
    image_dir = output / "images"
    image_dir.mkdir(parents=True)
    frames = []
    for frame_id, source_image in enumerate(images):
        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to decode {source_image}")
        if (image.shape[1], image.shape[0]) != (source_width, source_height):
            raise ValueError(
                f"{source_image} is {image.shape[1]}x{image.shape[0]}, "
                f"camera expects {source_width}x{source_height}"
            )
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        destination = image_dir / f"{frame_id:06d}.png"
        if not cv2.imwrite(str(destination), resized):
            raise IOError(f"Failed to write {destination}")
        frames.append(
            {
                "file_path": f"images/{frame_id:06d}.png",
                "transform_matrix": poses[frame_id].tolist(),
            }
        )

    shutil.copy2(source / "poses_bounds.npy", output / "poses_bounds.npy")
    transforms = {
        "w": width,
        "h": height,
        "fl_x": fx * width / source_width,
        "fl_y": fy * height / source_height,
        "cx": cx * width / source_width,
        "cy": cy * height / source_height,
        "aabb_scale": 2,
        "frames": frames,
    }
    (output / "transforms.json").write_text(
        json.dumps(transforms, indent=2) + "\n",
        encoding="utf-8",
    )


def validate(output_root, width, height):
    for scan_id in STANDARD_SCANS:
        root = output_root / f"scan{scan_id}"
        transforms = json.loads((root / "transforms.json").read_text())
        if len(transforms["frames"]) != 49:
            raise ValueError(f"scan{scan_id} does not contain 49 frames")
        if (transforms["w"], transforms["h"]) != (width, height):
            raise ValueError(f"scan{scan_id} has unexpected dimensions")
        for frame_id, frame in enumerate(transforms["frames"]):
            path = root / frame["file_path"]
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.shape[:2] != (height, width):
                raise ValueError(f"Invalid frame {frame_id} in scan{scan_id}: {path}")
            matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid pose {frame_id} in scan{scan_id}")
            if not math.isclose(np.linalg.det(matrix[:3, :3]), 1.0, abs_tol=1e-5):
                raise ValueError(f"Non-rotation pose {frame_id} in scan{scan_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=300)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    for scan_id in STANDARD_SCANS:
        print(f"Converting scan{scan_id} ...", flush=True)
        convert_scan(args.source_root, args.output_root, scan_id, args.width, args.height)
    validate(args.output_root, args.width, args.height)
    print(f"Validated {len(STANDARD_SCANS)} scans x 49 frames in {args.output_root}")


if __name__ == "__main__":
    main()
