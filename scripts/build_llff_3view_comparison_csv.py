"""Build the complete LLFF 3-view comparison CSV from paper and local metrics."""
import csv
import json
import re
from pathlib import Path


OUT = Path("papers_2025_2026_LLFF_DTU/LLFF_3view_comparison_summary.csv")
SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
WORKSPACE_PREFIX = "test_DiffusioNeRF_NeurTV_Ray_"
METHOD = "Ours (DiffusioNeRF + NeurTV + Ray)"
FIELDS = [
    "method", "journal", "year", "dataset", "scene", "views", "ckpt",
    "psnr↑", "ssim↑", "lpips↓", "scene_count", "source",
]


def fmt(d):
    return f'{d["psnr"]:.2f}', f'{d["ssim"]:.3f}', f'{d["lpips_alex"]:.3f}'


def workspace_candidates(scene):
    scene_root = Path("test_LLFF") / f"test_{scene}" / "few_shot3"
    preferred = scene_root / "test_DiffusioNeRF_NeurTV_Ray_30k_seed0"
    candidates = [preferred]
    for candidate in sorted(scene_root.glob(f"{WORKSPACE_PREFIX}*")):
        if candidate == preferred or "_backup_" in candidate.name:
            continue
        if candidate.is_dir():
            candidates.append(candidate)
    return candidates


def main():
    rows = [
        {"method": "FrugalNeRF", "journal": "CVPR", "year": 2025, "dataset": "LLFF", "scene": "average", "views": 3, "ckpt": "", "psnr↑": "19.66", "ssim↑": "0.610", "lpips↓": "0.300", "scene_count": "NA", "source": "FrugalNeRF CVPR 2025 Table 1"},
        {"method": "SE-GS", "journal": "ICCV", "year": 2025, "dataset": "LLFF", "scene": "average", "views": 3, "ckpt": "", "psnr↑": "20.79", "ssim↑": "0.724", "lpips↓": "0.183", "scene_count": "NA", "source": "SE-GS ICCV 2025 Table 1"},
        {"method": "DivCon-NeRF", "journal": "IJCAI", "year": 2026, "dataset": "LLFF", "scene": "average", "views": 3, "ckpt": "", "psnr↑": "20.46", "ssim↑": "0.662", "lpips↓": "0.221", "scene_count": "NA", "source": "DivCon-NeRF IJCAI 2026/arXiv v3 Table 2"},
        {"method": "MixNeRF", "journal": "CVPR", "year": 2023, "dataset": "LLFF", "scene": "average", "views": 3, "ckpt": "", "psnr↑": "19.27", "ssim↑": "0.629", "lpips↓": "0.236", "scene_count": "NA", "source": "MixNeRF CVPR 2023 Table 2"},
        {"method": "FlipNeRF", "journal": "ICCV", "year": 2023, "dataset": "LLFF", "scene": "average", "views": 3, "ckpt": "", "psnr↑": "19.34", "ssim↑": "0.631", "lpips↓": "0.235", "scene_count": "NA", "source": "FlipNeRF ICCV 2023 Table 7"},
        {"method": "DiffusioNeRF", "journal": "CVPR", "year": 2023, "dataset": "LLFF", "scene": "average", "views": 3, "ckpt": "", "psnr↑": "19.79", "ssim↑": "0.568", "lpips↓": "0.338", "scene_count": "NA", "source": "DiffusioNeRF CVPR 2023 Table 1; LPIPS-VGG"},
        {"method": "Binocular3DGS + Gaussian Dropout", "journal": "NeurIPS", "year": 2025, "dataset": "LLFF", "scene": "average", "views": 3, "ckpt": "", "psnr↑": "22.12", "ssim↑": "0.777", "lpips↓": "0.154", "scene_count": "NA", "source": "Binocular3DGS + Gaussian Dropout NeurIPS 2025"},
        {"method": "CoR-GS", "journal": "ECCV", "year": 2024, "dataset": "LLFF", "scene": "average", "views": 3, "ckpt": "", "psnr↑": "20.45", "ssim↑": "0.712", "lpips↓": "0.196", "scene_count": "NA", "source": "CoR-GS ECCV 2024"},
        {"method": "DNGaussian", "journal": "CVPR", "year": 2024, "dataset": "LLFF", "scene": "average", "views": 3, "ckpt": "", "psnr↑": "19.12", "ssim↑": "0.591", "lpips↓": "0.294", "scene_count": "NA", "source": "DNGaussian CVPR 2024"},
        {"method": "EdgeNeRF", "journal": "arXiv", "year": 2026, "dataset": "LLFF", "scene": "average", "views": 3, "ckpt": "", "psnr↑": "19.42", "ssim↑": "0.699", "lpips↓": "0.317", "scene_count": "NA", "source": "EdgeNeRF arXiv 2026"},
        {"method": "ICO-GS", "journal": "CVPR", "year": 2026, "dataset": "LLFF", "scene": "average", "views": 3, "ckpt": "", "psnr↑": "22.20", "ssim↑": "0.778", "lpips↓": "0.157", "scene_count": "NA", "source": "ICO-GS CVPR 2026"},
        {"method": "IPSM-Gaussian", "journal": "NeurIPS", "year": 2024, "dataset": "LLFF", "scene": "average", "views": 3, "ckpt": "", "psnr↑": "20.44", "ssim↑": "0.702", "lpips↓": "0.207", "scene_count": "NA", "source": "IPSM-Gaussian NeurIPS 2024"},
        {"method": "SYN3R", "journal": "NeurIPS", "year": 2025, "dataset": "LLFF", "scene": "average", "views": 3, "ckpt": "", "psnr↑": "20.61", "ssim↑": "0.705", "lpips↓": "0.201", "scene_count": "NA", "source": "SYN3R NeurIPS 2025"},
    ]
    local = []
    seen_archives = set()
    for scene in SCENES:
        for workspace in workspace_candidates(scene):
            for raw_path in workspace.glob("evaluation/**/step_*/raw/metrics.json"):
                step_match = re.search(r"step_(\d+)", str(raw_path))
                if step_match is None:
                    continue
                step = step_match.group(1)
                archive_key = (scene, step)
                if archive_key in seen_archives:
                    continue
                ema_path = raw_path.parent.parent / "ema" / "metrics.json"
                if not ema_path.is_file():
                    continue
                seen_archives.add(archive_key)
                for variant in ("raw", "ema"):
                    path = raw_path if variant == "raw" else ema_path
                    d = json.loads(path.read_text())["mean"]
                    local.append((scene, step, variant, d))

    for step in sorted({x[1] for x in local}, key=int):
        for variant in ("raw", "ema"):
            values = [x[3] for x in local if x[1] == step and x[2] == variant]
            if not values:
                continue
            mean = {k: sum(x[k] for x in values) / len(values) for k in ("psnr", "ssim", "lpips_alex")}
            psnr, ssim, lpips = fmt(mean)
            rows.append({"method": METHOD, "journal": "project evaluation", "year": 2026, "dataset": "LLFF", "scene": "average", "views": 3, "ckpt": f"step_{step}/{variant}", "psnr↑": psnr, "ssim↑": ssim, "lpips↓": lpips, "scene_count": len(values), "source": "test_LLFF macro mean across available scenes at checkpoint"})

    for scene, step, variant, d in local:
        psnr, ssim, lpips = fmt(d)
        rows.append({"method": METHOD, "journal": "project evaluation", "year": 2026, "dataset": "LLFF", "scene": scene, "views": 3, "ckpt": f"step_{step}/{variant}", "psnr↑": psnr, "ssim↑": ssim, "lpips↓": lpips, "scene_count": 1, "source": "test_LLFF milestone archive scene mean"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
