#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_assets"
LOCAL = OUT / "local"
LOCAL.mkdir(parents=True, exist_ok=True)

FONT = ImageFont.load_default()


def label_image(path, label, size=(420, 300)):
    im = Image.open(path).convert("RGB")
    im.thumbnail(size)
    canvas = Image.new("RGB", (size[0], size[1] + 34), "white")
    x = (size[0] - im.width) // 2
    y = (size[1] - im.height) // 2
    canvas.paste(im, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, size[1] + 9), label, fill="black", font=FONT)
    return canvas


def make_grid(items, output, cols=2, size=(420, 300)):
    cells = [label_image(path, label, size) for path, label in items]
    rows = (len(cells) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * size[0], rows * (size[1] + 34)), "white")
    for i, cell in enumerate(cells):
        canvas.paste(cell, ((i % cols) * size[0], (i // cols) * (size[1] + 34)))
    canvas.save(output, quality=95)


def frame(scene, kind, step=12000, variant="ema"):
    return ROOT / "test_LLFF" / f"test_{scene}" / "few_shot3" / "test_DiffusioNeRF_NeurTV_Ray_30k_seed0" / "evaluation" / "milestone_sweep" / f"step_{step:06d}" / variant / "frames" / f"frame_0000_{kind}.png"


make_grid(
    [(frame("fern", "gt"), "Fern / ground truth"), (frame("fern", "rgb"), "Fern / EMA @ 12k"),
     (frame("flower", "gt"), "Flower / ground truth"), (frame("flower", "rgb"), "Flower / EMA @ 12k"),
     (frame("orchids", "gt", 18000), "Orchids / ground truth"), (frame("orchids", "rgb", 18000), "Orchids / EMA @ 18k"),
     (frame("room", "gt"), "Room / ground truth"), (frame("room", "rgb"), "Room / EMA @ 12k")],
    LOCAL / "scene_grid.png",
    cols=2,
)

make_grid(
    [(frame("orchids", "gt", 18000), "Orchids / ground truth"),
     (frame("orchids", "rgb", 18000), "Orchids / EMA @ 18k"),
     (frame("orchids", "depth_preview", 18000), "Orchids / predicted depth")],
    LOCAL / "orchids_detail.png",
    cols=3,
    size=(360, 270),
)

methods = ["Ours\nEMA@12k", "SparseNeRF", "SPARF", "MVPGS", "SCGaussian", "NexusGS", "Path\nMatters", "GoLF-NRT"]
psnr = [19.796847, 19.86, 20.20, 20.65, 20.77, 20.80, 20.93, 24.20]
ssim = [0.677369, 0.620, 0.630, 0.880, 0.710, 0.770, 0.780, 0.821]
lpips = [0.196361, 0.330, 0.330, 0.100, 0.220, 0.240, 0.230, 0.148]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
for ax, values, title, ylabel in zip(axes, [psnr, ssim, lpips], ["PSNR (higher is better)", "SSIM (higher is better)", "LPIPS (lower is better)"], ["dB", "score", "score"]):
    colors = ["#d95f02" if i == 0 else "#4c78a8" for i in range(len(methods))]
    ax.bar(range(len(methods)), values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
fig.suptitle("LLFF 3-view reported comparison (protocol caveat applies)")
fig.tight_layout()
fig.savefig(OUT / "metric_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)
