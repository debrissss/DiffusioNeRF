#!/usr/bin/env python3
"""Aggregate the common milestone evaluations for all eight LLFF scenes."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
VARIANTS = ["raw", "ema"]
DEFAULT_STEPS = [6000, 9000, 12000, 18000]


def load_result(repo, scene, step, variant):
    base = repo / "test_LLFF" / f"test_{scene}" / "few_shot3" / "test_DiffusioNeRF_NeurTV_Ray_30k_seed0" / "evaluation" / "milestone_sweep" / f"step_{step:06d}"
    complete = base / "COMPLETE"
    metrics_path = base / variant / "metrics.json"
    manifest_path = base / "manifest.json"
    if not complete.is_file() or not metrics_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{scene} step {step} {variant}: missing archive files")
    complete_data = json.loads(complete.read_text())
    manifest = json.loads(manifest_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    if complete_data.get("status") != "complete":
        raise ValueError(f"{scene} step {step}: archive is not complete")
    if metrics.get("variant") != variant:
        raise ValueError(f"{scene} step {step}: variant label mismatch")
    if metrics["frame_ids"] != manifest["dataset"]["frame_ids"]:
        raise ValueError(f"{scene} step {step}: frame IDs disagree with manifest")
    return base, manifest, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output_dir or repo / "test_LLFF" / "aggregates" / "llff_3v_neurtv_ray_common_milestones"
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for step in args.steps:
        for scene in SCENES:
            for variant in VARIANTS:
                archive, manifest, metrics = load_result(repo, scene, step, variant)
                mean = metrics["mean"]
                rows.append({
                    "step": step,
                    "scene": scene,
                    "variant": variant,
                    "frame_count": metrics["frame_count"],
                    "psnr": mean["psnr"],
                    "lpips_alex": mean["lpips_alex"],
                    "ssim": mean["ssim"],
                    "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                    "archive": str(archive.relative_to(repo)),
                })

    fieldnames = list(rows[0].keys())
    with (output / "summary_long.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenes": SCENES,
        "steps": args.steps,
        "variants": VARIANTS,
        "aggregation": "Unweighted macro mean across the eight scene-level means.",
        "macro_mean": {},
    }
    for step in args.steps:
        summary["macro_mean"][str(step)] = {}
        for variant in VARIANTS:
            subset = [r for r in rows if r["step"] == step and r["variant"] == variant]
            summary["macro_mean"][str(step)][variant] = {
                metric: sum(r[metric] for r in subset) / len(subset)
                for metric in ("psnr", "lpips_alex", "ssim")
            }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# LLFF 3-view NeurTV + Ray common milestone evaluation",
        "",
        "All eight scenes use the same milestone checkpoint step. Values are unweighted macro means across scene-level means.",
        "",
        "| Step | Variant | PSNR ↑ | LPIPS-Alex ↓ | SSIM ↑ |",
        "|---:|---|---:|---:|---:|",
    ]
    for step in args.steps:
        for variant in VARIANTS:
            mean = summary["macro_mean"][str(step)][variant]
            lines.append(f"| {step} | {variant} | {mean['psnr']:.6f} | {mean['lpips_alex']:.6f} | {mean['ssim']:.6f} |")
    lines += ["", "Detailed per-scene rows are in `summary_long.csv`.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "COMPLETE").write_text(json.dumps({"status": "complete", "scene_count": 8, "steps": args.steps}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
