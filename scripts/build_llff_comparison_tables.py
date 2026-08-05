#!/usr/bin/env python3
"""Build LLFF 6-view/9-view comparison CSVs using the 3-view schema."""

from __future__ import annotations

import csv
import json
from pathlib import Path


SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
FIELDS = [
    "method",
    "journal",
    "year",
    "dataset",
    "scene",
    "views",
    "ckpt",
    "psnr↑",
    "ssim↑",
    "lpips↓",
    "scene_count",
    "source",
]

PAPER_VALUES = {
    6: {
        "MixNeRF": (23.76, 0.791, 0.115, "MixNeRF CVPR 2023 Table 2"),
        "DiffusioNeRF": (23.79, 0.747, 0.237, "DiffusioNeRF CVPR 2023 Table 1; LPIPS-VGG"),
        "CoR-GS": (24.49, 0.837, 0.115, "CoR-GS ECCV 2024 Table 1"),
        "SE-GS": (24.78, 0.839, 0.110, "SE-GS ICCV 2025 Table 1"),
        "ReVoRF": (22.21, 0.720, 0.269, "ReVoRF CVPR 2024 Table 4"),
        "ICO-GS": (25.37, 0.856, 0.109, "ICO-GS CVPR 2026 Table 1"),
    },
    9: {
        "MixNeRF": (25.20, 0.833, 0.087, "MixNeRF CVPR 2023 Table 2"),
        "DiffusioNeRF": (25.02, 0.785, 0.212, "DiffusioNeRF CVPR 2023 Table 1; LPIPS-VGG"),
        "CoR-GS": (26.06, 0.874, 0.089, "CoR-GS ECCV 2024 Table 1"),
        "SE-GS": (26.36, 0.878, 0.084, "SE-GS ICCV 2025 Table 1"),
        "ReVoRF": (23.04, 0.753, 0.225, "ReVoRF CVPR 2024 Table 4"),
        "ICO-GS": (26.45, 0.881, 0.096, "ICO-GS CVPR 2026 Table 1"),
    },
}


def row(**values):
    return {field: values.get(field, "") for field in FIELDS}


def checkpoint_kind(checkpoint_path: Path) -> str:
    if checkpoint_path.name == "ngp.pth":
        return "best"
    if "baselines" in checkpoint_path.parts:
        return "baseline"
    if "milestones" in checkpoint_path.parts:
        return "milestone"
    if checkpoint_path.name.startswith("ngp_ep"):
        return "final"
    return "checkpoint"


def checkpoint_label(checkpoint_path: Path, step: int, variant: str) -> str:
    kind = checkpoint_kind(checkpoint_path)
    if kind == "best":
        return f"best_step_{step:06d}/{variant}"
    if kind == "baseline":
        prefix = f"ngp_step_{step:06d}_"
        suffix = checkpoint_path.stem
        if suffix.startswith(prefix):
            suffix = suffix[len(prefix) :]
        return f"step_{step:06d}/{suffix}/{variant}"
    return f"step_{step:06d}/{variant}"


def load_project_records(repo: Path, views: int) -> list[dict]:
    records_by_key = {}
    for scene in SCENES:
        archive_root = repo / "test_LLFF" / f"test_{scene}" / f"few_shot{views}" / (
            "test_DiffusioNeRF_NeurTV_Ray_30k_seed0/evaluation/all_checkpoints"
        )
        if not archive_root.is_dir():
            raise RuntimeError(f"Missing evaluation directory: {archive_root}")

        for archive in sorted(path for path in archive_root.iterdir() if path.is_dir()):
            if not (archive / "COMPLETE").is_file():
                raise RuntimeError(f"Incomplete evaluation archive: {archive}")
            manifest = json.loads((archive / "manifest.json").read_text())
            checkpoint = manifest["checkpoint"]
            checkpoint_path = Path(checkpoint["path"])
            step = int(checkpoint["global_step"])
            frame_ids = manifest["dataset"]["frame_ids"]

            for variant in manifest["evaluation"]["variants"]:
                metrics = json.loads((archive / variant / "metrics.json").read_text())
                if metrics["frame_ids"] != frame_ids:
                    raise RuntimeError(f"Frame split mismatch: {archive}/{variant}")
                mean = metrics["mean"]
                record = {
                    "scene": scene,
                    "variant": variant,
                    "step": step,
                    "frame_count": metrics["frame_count"],
                    "psnr": float(mean["psnr"]),
                    "ssim": float(mean["ssim"]),
                    "lpips": float(mean["lpips_alex"]),
                    "checkpoint_sha256": checkpoint["sha256"],
                    "checkpoint_path": checkpoint_path,
                    "kind": checkpoint_kind(checkpoint_path),
                    "archive": str(archive.relative_to(repo)),
                }
                key = (scene, checkpoint["sha256"], variant)
                previous = records_by_key.get(key)
                if previous is not None:
                    comparable = ("step", "frame_count", "psnr", "ssim", "lpips")
                    if any(previous[field] != record[field] for field in comparable):
                        raise RuntimeError(f"Conflicting duplicate archives: {previous['archive']} and {record['archive']}")
                    # Prefer the canonical source-derived name from the reusable evaluator.
                    if "__" in archive.name and "__" not in Path(previous["archive"]).name:
                        records_by_key[key] = record
                else:
                    records_by_key[key] = record

    return list(records_by_key.values())


def main():
    repo = Path(__file__).resolve().parents[1]
    source_csv = repo / "papers_2025_2026_LLFF_DTU/LLFF_3view_comparison_summary.csv"
    source_rows = list(csv.DictReader(source_csv.open()))
    source_by_method = {item["method"]: item for item in source_rows}

    for views in (6, 9):
        output = repo / "papers_2025_2026_LLFF_DTU" / f"LLFF_{views}view_comparison_summary.csv"
        rows = []

        for method, (psnr, ssim, lpips, source) in PAPER_VALUES[views].items():
            original = source_by_method.get(method, {})
            if method == "ReVoRF":
                original = {"journal": "CVPR", "year": 2024}
            rows.append(
                row(
                    method=method,
                    journal=original["journal"],
                    year=original["year"],
                    dataset="LLFF",
                    scene="average",
                    views=views,
                    ckpt="",
                    **{
                        "psnr↑": f"{psnr:.2f}",
                        "ssim↑": f"{ssim:.3f}",
                        "lpips↓": f"{lpips:.3f}",
                    },
                    scene_count="NA",
                    source=source,
                )
            )

        method = "Ours (DiffusioNeRF + NeurTV + Ray)"
        project_records = load_project_records(repo, views)
        common_records = [item for item in project_records if item["kind"] in {"milestone", "final"}]
        for step in sorted({item["step"] for item in common_records}):
            for variant in ("raw", "ema"):
                selected = [
                    item for item in common_records if item["step"] == step and item["variant"] == variant
                ]
                if len(selected) != len(SCENES) or {item["scene"] for item in selected} != set(SCENES):
                    continue
                macro = {
                    metric: sum(item[metric] for item in selected) / len(selected)
                    for metric in ("psnr", "ssim", "lpips")
                }
                rows.append(
                    row(
                        method=method,
                        journal="project evaluation",
                        year=2026,
                        dataset="LLFF",
                        scene="average",
                        views=views,
                        ckpt=f"step_{step:06d}/{variant}",
                        **{
                            "psnr↑": f"{macro['psnr']:.2f}",
                            "ssim↑": f"{macro['ssim']:.3f}",
                            "lpips↓": f"{macro['lpips']:.3f}",
                        },
                        scene_count=8,
                        source=f"test_LLFF macro mean across 8 scene-level archives at step {step}",
                    )
                )

        variant_order = {"raw": 0, "ema": 1}
        kind_order = {"best": 0, "milestone": 1, "baseline": 2, "final": 3, "checkpoint": 4}
        project_records.sort(
            key=lambda item: (
                SCENES.index(item["scene"]),
                item["step"],
                kind_order[item["kind"]],
                variant_order[item["variant"]],
            )
        )
        for item in project_records:
            rows.append(
                row(
                    method=method,
                    journal="project evaluation",
                    year=2026,
                    dataset="LLFF",
                    scene=item["scene"],
                    views=views,
                    ckpt=checkpoint_label(item["checkpoint_path"], item["step"], item["variant"]),
                    **{
                        "psnr↑": f"{item['psnr']:.2f}",
                        "ssim↑": f"{item['ssim']:.3f}",
                        "lpips↓": f"{item['lpips']:.3f}",
                    },
                    scene_count=1,
                    source=item["archive"],
                )
            )

        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(output)


if __name__ == "__main__":
    main()
