#!/usr/bin/env python3
"""Validate the frozen legacy scan114 camera, images, split, and runtime rays."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "DTU" / "scan114"
DEFAULT_SPLIT = PROJECT_ROOT / "splits" / "dtu_3v_legacy" / "scan114.json"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "splits" / "dtu_3v_legacy" / "scan114_camera_manifest.json"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_tree_sha256(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def array_sha256(array, dtype):
    canonical = np.ascontiguousarray(array, dtype=dtype)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def runtime_poses(source_poses, scale, offset):
    converted = []
    scale = np.float32(scale)
    offset = np.asarray(offset, dtype=np.float32)
    for pose in source_poses:
        converted.append(np.array([
            [
                pose[1, 0], -pose[1, 1], -pose[1, 2],
                pose[1, 3] * scale + offset[0],
            ],
            [
                pose[2, 0], -pose[2, 1], -pose[2, 2],
                pose[2, 3] * scale + offset[1],
            ],
            [
                pose[0, 0], -pose[0, 1], -pose[0, 2],
                pose[0, 3] * scale + offset[2],
            ],
            [0, 0, 0, 1],
        ], dtype="<f4"))
    return np.stack(converted)


def sampled_rays(poses, intrinsics, pixels):
    fx, fy, cx, cy = [np.float32(value) for value in intrinsics]
    rays = []
    for pose in poses:
        for x, y in pixels:
            direction = np.array([
                (np.float32(x) + 0.5 - cx) / fx,
                (np.float32(y) + 0.5 - cy) / fy,
                1.0,
            ], dtype=np.float32)
            direction /= np.linalg.norm(direction)
            ray_direction = direction @ pose[:3, :3].T
            rays.append(np.concatenate([
                pose[:3, 3], ray_direction,
            ]).astype("<f4"))
    return np.stack(rays)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_close(actual, expected, label, tolerance=1e-12):
    require(
        math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance),
        f"{label} mismatch: expected {expected!r}, got {actual!r}",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fail closed unless legacy scan114 inputs and runtime camera settings match."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--scale", type=float, default=0.33)
    parser.add_argument("--offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--bound", type=float, default=2.0)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    split_path = args.split.resolve()
    manifest_path = args.manifest.resolve()
    require(dataset.is_dir(), f"Legacy dataset does not exist: {dataset}")
    require(split_path.is_file(), f"Legacy split does not exist: {split_path}")
    require(manifest_path.is_file(), f"Camera manifest does not exist: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    transforms_path = dataset / "transforms.json"
    require(transforms_path.is_file(), f"Missing transforms.json: {transforms_path}")
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))

    for relative, expected_hash in manifest["files"].items():
        path = dataset / relative
        require(path.is_file(), f"Missing frozen legacy file: {path}")
        actual_hash = sha256_file(path)
        require(
            actual_hash == expected_hash,
            f"SHA256 mismatch for {path}: expected {expected_hash}, got {actual_hash}",
        )

    camera_spec = manifest["camera"]
    image_spec = manifest["images"]
    runtime_spec = manifest["runtime"]
    frame_count = camera_spec["frame_count"]
    frames = transforms.get("frames", [])
    require(len(frames) == frame_count, f"Expected {frame_count} frames, got {len(frames)}")
    require_close(transforms["w"], camera_spec["width"], "camera width")
    require_close(transforms["h"], camera_spec["height"], "camera height")
    for key in ("fl_x", "fl_y", "cx", "cy"):
        require_close(transforms[key], camera_spec[key], key)

    expected_paths = [f"4/images/{frame_id:06d}.png" for frame_id in range(frame_count)]
    actual_paths = [frame.get("file_path") for frame in frames]
    require(
        actual_paths == expected_paths,
        "transforms.json frame paths are not the frozen ordered 0..48 legacy image list",
    )

    image_dir = dataset / image_spec["directory"]
    images = sorted(image_dir.glob("*.png"))
    require(len(images) == image_spec["count"], f"Expected 49 images, got {len(images)}")
    require(
        [path.name for path in images] == [f"{i:06d}.png" for i in range(frame_count)],
        "Legacy image filenames are not the exact ordered 000000.png..000048.png set",
    )
    for path in images:
        with Image.open(path) as image:
            require(
                image.size == (image_spec["width"], image_spec["height"]),
                f"Unexpected image dimensions for {path}: {image.size}",
            )
            require(image.mode == "RGB", f"Unexpected image mode for {path}: {image.mode}")
    actual_image_tree_hash = image_tree_sha256(images)
    require(
        actual_image_tree_hash == image_spec["tree_sha256"],
        "Legacy image tree SHA256 mismatch: "
        f"expected {image_spec['tree_sha256']}, got {actual_image_tree_hash}",
    )

    source_poses = np.asarray(
        [frame["transform_matrix"] for frame in frames], dtype="<f8"
    )
    require(source_poses.shape == (frame_count, 4, 4), f"Bad pose shape: {source_poses.shape}")
    require(np.isfinite(source_poses).all(), "Source poses contain NaN or Inf")
    require(
        np.allclose(source_poses[:, 3, :], np.array([0, 0, 0, 1]), atol=1e-12),
        "Source pose bottom rows are not homogeneous [0,0,0,1]",
    )
    determinants = np.linalg.det(source_poses[:, :3, :3])
    require(np.allclose(determinants, 1.0, atol=1e-8), "Source poses are not rotations")
    source_pose_hash = array_sha256(source_poses, "<f8")
    require(
        source_pose_hash == camera_spec["source_pose_f64le_sha256"],
        "Source pose SHA256 mismatch: "
        f"expected {camera_spec['source_pose_f64le_sha256']}, got {source_pose_hash}",
    )
    mean_radius = float(np.linalg.norm(source_poses[:, :3, 3], axis=1).mean())
    require_close(
        mean_radius, camera_spec["mean_source_camera_radius"],
        "mean source camera radius", tolerance=1e-10,
    )

    require(args.downscale == runtime_spec["downscale"], "Runtime downscale mismatch")
    require_close(args.scale, runtime_spec["scale"], "runtime scale")
    require_close(args.bound, runtime_spec["bound"], "runtime bound")
    require(
        np.array_equal(
            np.asarray(args.offset, dtype=np.float64),
            np.asarray(runtime_spec["offset"], dtype=np.float64),
        ),
        f"Runtime offset mismatch: expected {runtime_spec['offset']}, got {args.offset}",
    )

    converted_poses = runtime_poses(source_poses, args.scale, args.offset)
    converted_pose_hash = array_sha256(converted_poses, "<f4")
    require(
        converted_pose_hash == runtime_spec["runtime_pose_f32le_sha256"],
        "Runtime pose SHA256 mismatch: "
        f"expected {runtime_spec['runtime_pose_f32le_sha256']}, got {converted_pose_hash}",
    )
    rays = sampled_rays(
        converted_poses,
        [transforms["fl_x"], transforms["fl_y"], transforms["cx"], transforms["cy"]],
        runtime_spec["ray_sample_pixels"],
    )
    ray_hash = array_sha256(rays, "<f4")
    require(
        ray_hash == runtime_spec["ray_sample_f32le_sha256"],
        "Runtime ray SHA256 mismatch: "
        f"expected {runtime_spec['ray_sample_f32le_sha256']}, got {ray_hash}",
    )

    require(split.get("scene") == manifest["scene"], "Split scene mismatch")
    require(split.get("frame_count") == frame_count, "Split frame_count mismatch")
    expected_split = manifest["split"]
    for key in ("train_ids", "val_ids", "test_ids"):
        require(split.get(key) == expected_split[key], f"Split {key} differs from manifest")
    train_ids = set(split["train_ids"])
    val_ids = set(split["val_ids"])
    test_ids = set(split["test_ids"])
    require(not train_ids & val_ids, "train_ids and val_ids overlap")
    require(not train_ids & test_ids, "train_ids and test_ids overlap")
    require(not val_ids & test_ids, "val_ids and test_ids overlap")
    require(
        train_ids | val_ids | test_ids == set(range(frame_count)),
        "Legacy split does not cover each frame ID 0..48 exactly once",
    )

    print("[OK] Frozen legacy scan114 validation passed")
    print(f"  dataset: {dataset}")
    print(f"  split: train={split['train_ids']} val={split['val_ids']} test_count={len(test_ids)}")
    print(
        "  intrinsics: "
        f"fx={transforms['fl_x']:.15f} fy={transforms['fl_y']:.15f} "
        f"cx={transforms['cx']:.1f} cy={transforms['cy']:.1f}"
    )
    print(
        f"  runtime: downscale={args.downscale} scale={args.scale} "
        f"offset={args.offset} bound={args.bound}"
    )
    print(f"  transforms_sha256: {manifest['files']['transforms.json']}")
    print(f"  image_tree_sha256: {actual_image_tree_hash}")
    print(f"  runtime_pose_sha256: {converted_pose_hash}")
    print(f"  runtime_ray_sha256: {ray_hash}")


if __name__ == "__main__":
    main()
