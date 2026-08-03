#!/usr/bin/env python3
"""Combine local LLFF aggregates, PDF tables, and literature baselines."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "all_method_metrics.csv"
FIELDS = [
    "year", "venue", "method", "dataset", "scene", "views", "checkpoint_step",
    "variant", "psnr↑", "ssim↑", "lpips↓",
]
WORKSPACE_PREFIX = "test_DiffusioNeRF_NeurTV_Ray_"


def row(**kwargs):
    return {k: str(v) for k, v in kwargs.items() if v is not None}


def workspace_candidates(scene):
    scene_root = ROOT / "test_LLFF" / f"test_{scene}" / "few_shot3"
    preferred = scene_root / "test_DiffusioNeRF_NeurTV_Ray_30k_seed0"
    candidates = [preferred]
    for candidate in sorted(scene_root.glob(f"{WORKSPACE_PREFIX}*")):
        if candidate == preferred or "_backup_" in candidate.name:
            continue
        if candidate.is_dir():
            candidates.append(candidate)
    return candidates


def add_aggregate(rows, path):
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            checkpoint_step = item.get("step") or item.get("global_step")
            rows.append(row(
                source_type="local_aggregate", source_reference=str(path.relative_to(ROOT)),
                year="2026", venue="project evaluation", method="DiffusioNeRF + NeurTV + Ray",
                method_original="DiffusioNeRF_NeurTV_Ray", family="NeRF + regularization",
                dataset="LLFF", scene=item["scene"], scope="scene_mean", views="3",
                checkpoint_step=checkpoint_step, variant=item["variant"], psnr=item["psnr"],
                ssim=item["ssim"], lpips=item["lpips_alex"],
                protocol_note="Standard 8-scene LLFF split; every 8th image test; same saved split IDs",
                result_type="local evaluation",
            ))

    summary_path = path.with_name("summary.json")
    summary = json.loads(summary_path.read_text())
    if "steps" in summary:
        macro_items = summary["macro_mean"].items()
    else:
        macro_items = [(str(summary["global_step"]), summary["macro_mean"])]
    for step, variants in macro_items:
        for variant, metrics in variants.items():
            rows.append(row(
                source_type="local_aggregate", source_reference=str(summary_path.relative_to(ROOT)),
                year="2026", venue="project evaluation", method="DiffusioNeRF + NeurTV + Ray",
                method_original="DiffusioNeRF_NeurTV_Ray", family="NeRF + regularization",
                dataset="LLFF", scene="average", scope="8_scene_macro_mean",
                views="3", checkpoint_step=step, variant=variant, psnr=metrics["psnr"],
                ssim=metrics["ssim"], lpips=metrics["lpips_alex"],
                protocol_note="Unweighted macro mean of the eight scene-level means",
                result_type="local evaluation",
            ))


def add_project(rows):
    add_aggregate(
        rows,
        ROOT / "test_LLFF/aggregates/llff_3v_neurtv_ray_common_milestones/summary_long.csv",
    )
    add_aggregate(
        rows,
        ROOT / "test_LLFF/aggregates/llff_3v_neurtv_ray_seed0_step_020001/summary_long.csv",
    )

    # Also retain the active-workspace final archives. Fern/Flower/Fortress/
    # use alternate workspaces when the preferred workspace has no archive at
    # a requested step, so common step_020001 remains an eight-scene result.
    final_rows = []
    seen_final_archives = set()
    for scene in ("fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"):
        for workspace in workspace_candidates(scene):
            evaluation = workspace / "evaluation"
            for archive in sorted(evaluation.glob("step_*")):
                if not archive.is_dir() or not archive.name[5:].isdigit():
                    continue
                step = int(archive.name[5:])
                archive_key = (scene, step)
                if archive_key in seen_final_archives:
                    continue
                manifest_path = archive / "manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text())
                for variant in ("raw", "ema"):
                    metrics_path = archive / variant / "metrics.json"
                    if not metrics_path.is_file():
                        continue
                    metrics = json.loads(metrics_path.read_text())
                    mean = metrics["mean"]
                    item = row(
                        source_type="local_final", source_reference=str(archive.relative_to(ROOT)),
                        year="2026", venue="project evaluation", method="DiffusioNeRF + NeurTV + Ray",
                        method_original="DiffusioNeRF_NeurTV_Ray", family="NeRF + regularization",
                        dataset="LLFF", scene=scene, scope="available_scene_final", views="3",
                        checkpoint_step=step, variant=variant, psnr=mean["psnr"], ssim=mean["ssim"],
                        lpips=mean["lpips_alex"], protocol_note="Active workspace final archive; step is not common to all eight scenes",
                        result_type="local evaluation",
                    )
                    rows.append(item)
                    final_rows.append(item)
                seen_final_archives.add(archive_key)
    for step in sorted({int(x["checkpoint_step"]) for x in final_rows}):
        for variant in ("raw", "ema"):
            subset = [x for x in final_rows if int(x["checkpoint_step"]) == step and x["variant"] == variant]
            if not subset:
                continue
            rows.append(row(
                source_type="local_final", source_reference="active final evaluation archives",
                year="2026", venue="project evaluation", method="DiffusioNeRF + NeurTV + Ray",
                method_original="DiffusioNeRF_NeurTV_Ray", family="NeRF + regularization",
                dataset="LLFF", scene="average", scope="available_scene_macro_mean",
                views="3", checkpoint_step=step, variant=variant,
                psnr=sum(float(x["psnr"]) for x in subset) / len(subset),
                ssim=sum(float(x["ssim"]) for x in subset) / len(subset),
                lpips=sum(float(x["lpips"]) for x in subset) / len(subset),
                protocol_note=f"Macro mean over {len(subset)} scenes available at this final step; not an eight-scene common step",
                result_type="local evaluation",
            ))


def add_pdf(rows):
    # Tables 4.1 and 4.2 of the supplied PDF. Each tuple is method, PSNR3,
    # PSNR9, SSIM3, SSIM9, LPIPS3, LPIPS9, Average3, Average9.
    pdf = "稀疏视点下神经辐射场重建的优化方法_提纯版.pdf"
    tables = {
        ("DTU", "scan63"): [
            ("MixNeRF", 15.23, 25.13, .781, .926, .320, .162, .164, .051),
            ("FlipNeRF", 18.34, 25.29, .872, .877, .242, .236, .108, .062),
            ("FD-NeRF", 20.92, 25.15, .855, .907, .293, .213, .096, .061),
            ("DiffusioNeRF", 16.80, 25.46, .808, .950, .300, .154, .140, .046),
            ("Ours (Full)", 18.75, 25.57, .889, .924, .240, .131, .102, .046),
        ],
        ("DTU", "scan41"): [
            ("MixNeRF", 14.95, 25.33, .565, .858, .567, .228, .229, .063),
            ("FlipNeRF", 17.43, 25.70, .720, .933, .359, .163, .151, .048),
            ("FD-NeRF", 19.66, 23.97, .689, .816, .442, .362, .138, .090),
            ("DiffusioNeRF", 18.10, 25.41, .524, .917, .454, .163, .169, .051),
            ("Ours (Full)", 18.91, 26.07, .705, .918, .353, .109, .135, .043),
        ],
        ("LLFF", "fern"): [
            ("MixNeRF", 19.01, 23.90, .664, .733, .368, .341, .116, .072),
            ("FlipNeRF", 19.35, 24.40, .678, .769, .358, .292, .110, .063),
            ("DiffusioNeRF", 19.63, 24.56, .753, .738, .242, .281, .087, .064),
            ("Ours (Full)", 20.52, 25.03, .813, .830, .181, .232, .067, .050),
        ],
        ("LLFF", "room"): [
            ("MixNeRF", 21.67, 24.78, .806, .887, .317, .239, .075, .045),
            ("FlipNeRF", 21.99, 24.52, .811, .891, .306, .202, .072, .043),
            ("DiffusioNeRF", 21.71, 24.99, .793, .886, .306, .243, .075, .044),
            ("Ours (Full)", 22.33, 25.74, .819, .871, .219, .186, .061, .040),
        ],
    }
    for (dataset, scene), methods in tables.items():
        for method, p3, p9, s3, s9, l3, l9, a3, a9 in methods:
            for views, psnr, ssim, lpips, average in [(3, p3, s3, l3, a3), (9, p9, s9, l9, a9)]:
                rows.append(row(
                    source_type="paper_pdf", source_reference=pdf + " (Table 4.1/4.2)",
                    year="2026", venue="supplied thesis/PDF", method=method,
                    method_original=method, family="NeRF", dataset=dataset, scene=scene,
                    scope="scene", views=views, checkpoint_step="", variant="",
                    psnr=psnr, ssim=ssim, lpips=lpips, average=average,
                    protocol_note="PDF-reported 3/9-view comparison", result_type="reported in PDF",
                ))

    # Table 4.3 ablations are retained as separate records, not merged with
    # the main comparison rows above.
    ablations = [
        ("LLFF", "fern", "DiffusioNeRF", 3, 19.63, .753, .242, .109),
        ("LLFF", "fern", "Ours (Full)", 3, 20.52, .813, .181, .089),
        ("LLFF", "fern", "Ours w/o NeuraTV", 3, 20.18, .783, .187, .094),
        ("DTU", "scan63", "DiffusioNeRF", 3, 16.80, .808, .300, .140),
        ("DTU", "scan63", "Ours (Full)", 3, 18.75, .889, .240, .102),
        ("DTU", "scan63", "Ours w/o NeuraTV", 3, 18.62, .882, .245, .105),
        ("DTU", "scan82", "DiffusioNeRF", 3, 18.39, .661, .432, .154),
        ("DTU", "scan82", "Ours (Full)", 3, 18.84, .656, .431, .149),
        ("DTU", "scan82", "Ours w/o NeuraTV", 3, 18.66, .651, .442, .153),
        ("DTU", "scan41", "DiffusioNeRF", 3, 18.10, .624, .454, .163),
        ("DTU", "scan41", "Ours (Full)", 3, 18.91, .705, .353, .135),
        ("DTU", "scan41", "Ours w/o NeuraTV", 3, 18.87, .675, .373, .140),
    ]
    for dataset, scene, method, views, psnr, ssim, lpips, average in ablations:
        rows.append(row(source_type="paper_pdf", source_reference=pdf + " (Table 4.3)", year="2026", venue="supplied thesis/PDF", method=method, method_original=method, family="NeRF ablation", dataset=dataset, scene=scene, scope="ablation", views=views, psnr=psnr, ssim=ssim, lpips=lpips, average=average, protocol_note="PDF-reported 3-view ablation", result_type="reported in PDF"))


def add_literature(rows):
    path = ROOT / "reports/sparse_view_experiment_plan_2026-07-28/comparison_matrix.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            rows.append(row(
                source_type="literature", source_reference=item["source_url"], year=item["year"], venue=item["venue"], method=item["method"], method_original=item["method"], family=item["family"], dataset=item["dataset"], scene="", scope="published_aggregate", views=item["views"], psnr=item["psnr"], ssim=item["ssim"], lpips=item["lpips"], average=item["avge"], protocol_note=item["protocol_note"], result_type=item["status"],
            ))
    extra = [
        ("2023", "SparseNeRF", "NeRF", "LLFF", 3, 19.86, .620, .330, "3-view LLFF table", "https://openreview.net/"),
        ("2023", "SPARF", "NeRF", "LLFF", 3, 20.20, .630, .330, "3-view LLFF table", "https://openreview.net/"),
        ("2024", "MVPGS", "3DGS", "LLFF", 3, 20.65, .880, .100, "3-view LLFF table", "UNIFIED_LLFF_3VIEW_COMPARISON.md"),
        ("2024", "SCGaussian", "3DGS", "LLFF", 3, 20.77, .710, .220, "3-view LLFF table", "UNIFIED_LLFF_3VIEW_COMPARISON.md"),
        ("2025", "NexusGS", "3DGS", "LLFF", 3, 20.80, .770, .240, "3-view LLFF table", "UNIFIED_LLFF_3VIEW_COMPARISON.md"),
        ("2026", "Path Matters", "3DGS", "LLFF", 3, 20.93, .780, .230, "ICLR 2026 submission, Table 3", "https://openreview.net/pdf/1ad8bd881c26849b79ebf5025a130a786ccd32ec.pdf"),
        ("2025", "GoLF-NRT", "Generalizable NVS", "LLFF", 3, 24.20, .821, .148, "3-view LLFF table", "https://klmav.cuc.edu.cn/2025/0529/c2286a256259/page.htm"),
    ]
    for year, method, family, dataset, views, psnr, ssim, lpips, note, source in extra:
        rows.append(row(source_type="literature", source_reference=source, year=year, venue="reported literature", method=method, method_original=method, family=family, dataset=dataset, scope="published_aggregate", views=views, psnr=psnr, ssim=ssim, lpips=lpips, protocol_note=note, result_type="published"))


def main():
    rows = []
    add_project(rows)
    add_pdf(rows)
    add_literature(rows)
    rows = [item for item in rows if item.get("dataset") == "LLFF"]
    rows = [item for item in rows if item.get("source_type") == "local_aggregate" or item.get("source_type") == "local_final" or item.get("views") == "3"]
    scene_order = {name: index for index, name in enumerate(("fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"))}
    rows.sort(key=lambda r: (
        r.get("dataset", ""),
        scene_order.get(r.get("scene", ""), 999),
        r.get("scene", ""),
        r.get("method", ""),
        r.get("checkpoint_step", ""),
        r.get("variant", ""),
        r.get("source_type", ""),
    ))
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        formatted = []
        for item in rows:
            source_field = {"psnr↑": "psnr", "ssim↑": "ssim", "lpips↓": "lpips"}
            output = {field: item.get(source_field.get(field, field), "") for field in FIELDS}
            if output["scene"] == "":
                output["scene"] = "average"
            for field, digits in (("psnr↑", 2), ("ssim↑", 3), ("lpips↓", 3)):
                if output[field] != "":
                    output[field] = f"{float(output[field]):.{digits}f}"
            formatted.append(output)
        writer.writerows(formatted)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
