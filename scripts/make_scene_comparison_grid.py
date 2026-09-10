#!/usr/bin/env python3

import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", default="ema")
    parser.add_argument("--frames", type=int, nargs=5, required=True)
    args = parser.parse_args()

    frame_dir = args.archive / args.variant / "frames"
    rows = []
    for suffix in ("gt", "rgb"):
        rows.append([
            Image.open(frame_dir / f"frame_{frame_id:04d}_{suffix}.png").convert("RGB")
            for frame_id in args.frames
        ])

    width, height = rows[0][0].size
    canvas = Image.new("RGB", (width * 5, height * 2))
    for row_index, images in enumerate(rows):
        for column_index, image in enumerate(images):
            canvas.paste(image, (column_index * width, row_index * height))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == "__main__":
    main()
