#!/usr/bin/env python3
"""Filter a DTU structured-light point cloud with ObsMask and Plane data.

The implementation follows DTU's PointCompareMain.m coordinate convention:

    voxel = round((point - BB[0]) / Res)        # Python, zero based
    above = P.T @ [x, y, z, 1] > 0

The output keeps every vertex property from the input binary PLY (coordinates,
normals, and RGB for the official DTU STL point clouds).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def read_binary_vertex_ply(path: Path):
    header_lines: list[str] = []
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"PLY header is incomplete: {path}")
            line = raw.decode("ascii").rstrip("\r\n")
            header_lines.append(line)
            if line == "end_header":
                offset = handle.tell()
                break

    if "format binary_little_endian 1.0" not in header_lines:
        raise ValueError("Only binary_little_endian PLY files are supported")

    vertex_count = None
    properties: list[tuple[str, str]] = []
    in_vertices = False
    for line in header_lines:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertices = True
        elif parts[:1] == ["element"]:
            in_vertices = False
        elif in_vertices and parts[:1] == ["property"]:
            if len(parts) != 3 or parts[1] not in PLY_TYPES:
                raise ValueError(f"Unsupported vertex property: {line}")
            properties.append((parts[2], "<" + PLY_TYPES[parts[1]]))

    if vertex_count is None or not properties:
        raise ValueError(f"No vertex element found in {path}")

    dtype = np.dtype(properties)
    vertices = np.memmap(
        path, mode="r", dtype=dtype, offset=offset, shape=(vertex_count,)
    )
    return vertices, header_lines


def output_header(header_lines: list[str], vertex_count: int) -> bytes:
    rewritten = []
    for line in header_lines:
        if line.startswith("element vertex "):
            line = f"element vertex {vertex_count}"
        rewritten.append(line)
    return ("\n".join(rewritten) + "\n").encode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--obs-mask", type=Path, required=True)
    parser.add_argument("--plane", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()

    vertices, header = read_binary_vertex_ply(args.points)
    required = {"x", "y", "z"}
    if not required.issubset(vertices.dtype.names or ()):
        raise ValueError(f"PLY is missing coordinate fields: {required}")

    obs_data = loadmat(args.obs_mask)
    plane_data = loadmat(args.plane)
    obs_mask = np.asarray(obs_data["ObsMask"], dtype=bool)
    bb = np.asarray(obs_data["BB"], dtype=np.float64)
    resolution = float(np.asarray(obs_data["Res"]).squeeze())
    plane = np.asarray(plane_data["P"], dtype=np.float64).reshape(4)

    if resolution <= 0:
        raise ValueError(f"Invalid voxel resolution: {resolution}")

    keep = np.zeros(len(vertices), dtype=bool)
    in_bounds_count = 0
    observable_count = 0
    above_plane_count = 0

    for start in range(0, len(vertices), args.chunk_size):
        stop = min(start + args.chunk_size, len(vertices))
        part = vertices[start:stop]
        xyz = np.column_stack((part["x"], part["y"], part["z"])).astype(
            np.float64, copy=False
        )

        # DTU uses round((Q - BB_min) / Res + 1) in one-based MATLAB.
        # Values here are non-negative after subtracting BB_min, so floor(x+.5)
        # reproduces MATLAB's rounding while directly yielding zero-based indices.
        voxels = np.floor((xyz - bb[0]) / resolution + 0.5).astype(np.int64)
        in_bounds = np.all(voxels >= 0, axis=1)
        in_bounds &= voxels[:, 0] < obs_mask.shape[0]
        in_bounds &= voxels[:, 1] < obs_mask.shape[1]
        in_bounds &= voxels[:, 2] < obs_mask.shape[2]

        observable = np.zeros(stop - start, dtype=bool)
        valid_voxels = voxels[in_bounds]
        observable[in_bounds] = obs_mask[
            valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]
        ]
        above_plane = xyz @ plane[:3] + plane[3] > 0

        keep[start:stop] = observable & above_plane
        in_bounds_count += int(in_bounds.sum())
        observable_count += int(observable.sum())
        above_plane_count += int(above_plane.sum())

    kept_count = int(keep.sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(output_header(header, kept_count))
        for start in range(0, len(vertices), args.chunk_size):
            stop = min(start + args.chunk_size, len(vertices))
            selected = np.asarray(vertices[start:stop][keep[start:stop]])
            handle.write(selected.tobytes(order="C"))

    stats = {
        "input": str(args.points.resolve()),
        "obs_mask": str(args.obs_mask.resolve()),
        "plane": str(args.plane.resolve()),
        "output": str(args.output.resolve()),
        "input_points": int(len(vertices)),
        "inside_obs_grid": in_bounds_count,
        "inside_observability_mask": observable_count,
        "above_plane": above_plane_count,
        "kept_intersection": kept_count,
        "removed": int(len(vertices) - kept_count),
        "kept_ratio": kept_count / len(vertices),
        "obs_mask_shape": list(obs_mask.shape),
        "voxel_resolution_mm": resolution,
        "bounding_box": bb.tolist(),
        "plane_coefficients": plane.tolist(),
    }
    stats_path = args.stats or args.output.with_suffix(".json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
