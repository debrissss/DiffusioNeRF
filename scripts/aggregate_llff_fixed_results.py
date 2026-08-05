#!/usr/bin/env python3
"""Aggregate fixed-step LLFF Raw/EMA evaluation archives."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
VARIANTS = ["raw", "ema"]
METRICS = ["psnr", "lpips_alex", "ssim"]


def macro_mean(rows: list[dict], variant: str) -> dict[str, float]:
    selected = [row for row in rows if row["variant"] == variant]
    return {metric: sum(row[metric] for row in selected) / len(selected) for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", type=int, choices=(6, 9), required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    aggregate_dir = repo / "test_LLFF" / "aggregates" / (
        f"llff_{args.views}v_neurtv_ray_seed0_step_{args.step:06d}"
    )
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    incomplete: list[str] = []
    for scene in SCENES:
        workspace = repo / "test_LLFF" / f"test_{scene}" / f"few_shot{args.views}" / (
            "test_DiffusioNeRF_NeurTV_Ray_30k_seed0"
        )
        archive = workspace / "evaluation" / f"step_{args.step:06d}"
        manifest_path = archive / "manifest.json"
        complete_path = archive / "COMPLETE"

        try:
            manifest = json.loads(manifest_path.read_text())
            if not complete_path.is_file():
                raise FileNotFoundError("COMPLETE")
            if manifest["checkpoint"]["global_step"] != args.step:
                raise ValueError("checkpoint step mismatch")
            checkpoint_sha256 = manifest["checkpoint"]["sha256"]
            frame_ids = manifest["dataset"]["frame_ids"]

            for variant in VARIANTS:
                metrics_path = archive / variant / "metrics.json"
                metrics = json.loads(metrics_path.read_text())
                if metrics["frame_ids"] != frame_ids:
                    raise ValueError(f"{variant} frame split mismatch")
                mean = metrics["mean"]
                rows.append(
                    {
                        "scene": scene,
                        "variant": variant,
                        "global_step": args.step,
                        "frame_count": metrics["frame_count"],
                        "psnr": mean["psnr"],
                        "lpips_alex": mean["lpips_alex"],
                        "ssim": mean["ssim"],
                        "checkpoint_sha256": checkpoint_sha256,
                        "archive": str(archive.relative_to(repo)),
                    }
                )
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
            incomplete.append(f"{scene}: {exc}")

    if incomplete:
        raise SystemExit("Incomplete or inconsistent scenes:\n" + "\n".join(incomplete))

    macro = {variant: macro_mean(rows, variant) for variant in VARIANTS}
    best_by_metric = {}
    for metric in METRICS:
        direction = "min" if metric == "lpips_alex" else "max"
        variant = min(VARIANTS, key=lambda value: macro[value][metric]) if direction == "min" else max(
            VARIANTS, key=lambda value: macro[value][metric]
        )
        best_by_metric[metric] = {
            "all_values": {value: macro[value][metric] for value in VARIANTS},
            "direction": direction,
            "value": macro[variant][metric],
            "variant": variant,
        }

    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenes": SCENES,
        "scene_count": len(SCENES),
        "global_step": args.step,
        "variants": VARIANTS,
        "incomplete_scenes": [],
        "aggregation": "Unweighted macro mean across scene-level metrics. No per-scene model-variant cherry-picking is applied.",
        "macro_mean": macro,
        "best_by_metric": best_by_metric,
        "paper_reporting": {
            "preferred": "Report Raw and EMA as separate rows.",
            "hybrid_if_required": "For each metric, use the globally winning variant shown in best_by_metric and disclose the source variant.",
        },
    }
    (aggregate_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (aggregate_dir / "summary_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene",
                "variant",
                "global_step",
                "frame_count",
                "psnr",
                "lpips_alex",
                "ssim",
                "checkpoint_sha256",
                "archive",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# LLFF {args.views}-view NeurTV + Ray evaluation",
        "",
        f"- Fixed checkpoint: {args.step} steps",
        f"- Complete scenes: {len(SCENES)}",
        "- Average: unweighted macro mean across scenes",
        "",
        "| Scene | Variant | PSNR ↑ | LPIPS-Alex ↓ | SSIM ↑ |",
        "|---|---|---:|---:|---:|",
    ]
    for scene in SCENES:
        for variant in VARIANTS:
            row = next(item for item in rows if item["scene"] == scene and item["variant"] == variant)
            lines.append(
                f"| {scene} | {variant} | {row['psnr']:.6f} | {row['lpips_alex']:.6f} | {row['ssim']:.6f} |"
            )
    lines.extend(
        [
            f"| **Macro mean** | **raw** | {macro['raw']['psnr']:.6f} | {macro['raw']['lpips_alex']:.6f} | {macro['raw']['ssim']:.6f} |",
            f"| **Macro mean** | **ema** | {macro['ema']['psnr']:.6f} | {macro['ema']['lpips_alex']:.6f} | {macro['ema']['ssim']:.6f} |",
            "",
            "## Global best by metric",
            "",
            "| Metric | Source variant | Value |",
            "|---|---|---:|",
        ]
    )
    for metric in METRICS:
        best = best_by_metric[metric]
        lines.append(f"| {metric} | {best['variant']} | {best['value']:.6f} |")
    (aggregate_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (aggregate_dir / "COMPLETE").write_text(
        json.dumps(
            {
                "status": "complete",
                "scene_count": len(SCENES),
                "global_step": args.step,
                "scenes": SCENES,
            },
            indent=2,
        )
        + "\n"
    )
    print(aggregate_dir)


if __name__ == "__main__":
    main()
