#!/usr/bin/env python3
"""Attach RegNeRF object masks to a prepared standard DTU dataset."""

import argparse
from pathlib import Path

import cv2


STANDARD_SCANS = [8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114]
TEST_IDS = [
    1, 2, 9, 10, 11, 12, 14, 15, 23, 24, 26, 27, 29,
    30, 31, 32, 33, 34, 35, 41, 42, 43, 45, 46, 47,
]


def find_mask(mask_root, scan_id, frame_id):
    scan_root = mask_root / f"scan{scan_id}"
    candidates = [
        scan_root / f"{frame_id:03d}.png",
        scan_root / "mask" / f"{frame_id:03d}.png",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one mask for scan{scan_id} frame {frame_id}: {candidates}"
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_root", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=300)
    args = parser.parse_args()

    for scan_id in STANDARD_SCANS:
        scan_root = args.dataset_root / f"scan{scan_id}"
        if not (scan_root / "transforms.json").is_file():
            raise FileNotFoundError(f"Missing prepared scan: {scan_root}")
        output = scan_root / "masks"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite masks: {output}")
        output.mkdir()
        for frame_id in TEST_IDS:
            source = find_mask(args.mask_root, scan_id, frame_id)
            mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to decode {source}")
            mask = cv2.resize(
                mask, (args.width, args.height), interpolation=cv2.INTER_NEAREST
            )
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            destination = output / f"{frame_id:06d}.png"
            if not cv2.imwrite(str(destination), mask):
                raise IOError(f"Failed to write {destination}")
        print(f"Attached {len(TEST_IDS)} masks to scan{scan_id}")


if __name__ == "__main__":
    main()
