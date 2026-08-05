#!/usr/bin/env python3
"""Generate canonical RegNeRF/FreeNeRF DTU sparse-view split manifests."""

import json
from pathlib import Path


SCANS = [8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114]
TRAIN_IDS = {
    3: [25, 22, 28],
    6: [25, 22, 28, 40, 44, 48],
    9: [25, 22, 28, 40, 44, 48, 0, 8, 13],
}
TEST_IDS = [
    1, 2, 9, 10, 11, 12, 14, 15, 23, 24, 26, 27, 29,
    30, 31, 32, 33, 34, 35, 41, 42, 43, 45, 46, 47,
]


README = """# Standard sparse-view DTU {view_count}-view splits

These manifests follow the DTU protocol documented by FreeNeRF and inherited
from RegNeRF/PixelNeRF. IDs are zero-based indices in each scene's 49-frame
`transforms.json`.

- Training IDs: `{train_ids}`
- Test IDs: `{test_ids}`
- Validation uses only test frame 1 because this benchmark does not define a
  separate validation set. It is monitoring-only and keeps periodic validation
  inexpensive; experiments use a fixed training-step budget and never select
  checkpoints by validation performance.
- Remaining frames are excluded from benchmark evaluation because the standard
  protocol excludes views with unsuitable exposure.
"""


def main():
    project_root = Path(__file__).resolve().parents[1]
    for view_count, train_ids in TRAIN_IDS.items():
        output = project_root / "splits" / f"dtu_{view_count}v"
        output.mkdir(parents=True, exist_ok=True)
        (output / "README.md").write_text(
            README.format(
                view_count=view_count,
                train_ids=", ".join(map(str, train_ids)),
                test_ids=", ".join(map(str, TEST_IDS)),
            ),
            encoding="utf-8",
        )
        for scan_id in SCANS:
            manifest = {
                "dataset": "DTU",
                "scene": f"scan{scan_id}",
                "protocol": f"dtu_regnerf_standard_{view_count}v_v1",
                "index_basis": "zero-based transforms.json frame order",
                "frame_count": 49,
                "train_ids": train_ids,
                "val_ids": [1],
                "test_ids": TEST_IDS,
            }
            (output / f"scan{scan_id}.json").write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"Generated {len(SCANS)} manifests in {output}")


if __name__ == "__main__":
    main()
