#!/usr/bin/env python3
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "LLFF_3view_统一对比报告.docx"
AGG = ROOT / "test_LLFF/aggregates/llff_3v_neurtv_ray_common_milestones"
ASSETS = ROOT / "report_assets"


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell(cell, text, bold=False, color=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(str(text))
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for i, header in enumerate(headers):
        set_cell(table.rows[0].cells[i], header, bold=True, color=(255, 255, 255))
        shade(table.rows[0].cells[i], "1F4E78")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell(cells[i], value)
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    doc.add_paragraph()
    return table


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(90, 90, 90)


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item)


def main():
    summary = json.loads((AGG / "summary.json").read_text())
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.65)
    sec.bottom_margin = Inches(0.65)
    sec.left_margin = Inches(0.7)
    sec.right_margin = Inches(0.7)
    styles = doc.styles
    styles["Normal"].font.name = "Microsoft YaHei"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    styles["Normal"].font.size = Pt(10.5)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("LLFF 三视图稀疏重建方法统一对比报告")
    run.bold = True
    run.font.size = Pt(20)
    run.font.color.rgb = RGBColor(31, 78, 121)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run("DiffusioNeRF + NeurTV + Ray 与 2024–2026 代表方法").italic = True
    doc.add_paragraph()

    doc.add_heading("技术摘要", level=1)
    p = doc.add_paragraph()
    p.add_run("主要结论：").bold = True
    p.add_run("在当前统一为 LLFF 3-view、PSNR/SSIM/LPIPS 的比较表中，本项目 EMA@12000 的 PSNR=19.796847、SSIM=0.677369、LPIPS=0.196361。它在 LPIPS 上具有一定竞争力，但 PSNR 和 SSIM 低于近期基于多视图先验、3D Gaussian 或泛化模型的方法。")
    add_bullets(doc, [
        "本项目 8 个场景共同 checkpoint 为 6000、9000、12000、18000 steps；报告主表采用固定 EMA@12000，避免逐指标挑选 checkpoint。",
        "公开论文数值用于统一格式和趋势比较，并不等同于在本项目代码、split、分辨率和硬件上的重新复现。",
        "定性图选择 LLFF 中公开材料和本项目都覆盖的 Orchids/Flower 类场景；论文图与本地渲染图按来源分开展示，不宣称像素级对齐。",
    ])

    doc.add_heading("1. 统一指标对比", level=1)
    doc.add_paragraph("所有方法统一使用 PSNR（越高越好）、SSIM（越高越好）和 LPIPS（越低越好）三列。公开方法数值来自其论文或项目页面的 LLFF 3-view 报告；本项目数值来自本地 8 场景标准 holdout 评估。")
    methods = [
        ("SparseNeRF", "2023 CVPR", "19.86", "0.620", "0.330", "公开报告"),
        ("SPARF", "2023 CVPR", "20.20", "0.630", "0.330", "公开报告"),
        ("MVPGS", "2024 ECCV", "20.65", "0.880", "0.100", "公开报告"),
        ("SCGaussian", "2024 NeurIPS", "20.77", "0.710", "0.220", "公开报告"),
        ("NexusGS", "2025 CVPR", "20.80", "0.770", "0.240", "公开报告"),
        ("Path Matters", "2026 ICLR", "20.93", "0.780", "0.230", "公开报告"),
        ("GoLF-NRT", "2025 CVPR", "24.20", "0.821", "0.148", "公开报告"),
        ("本项目 EMA@12000", "本项目", "19.796847", "0.677369", "0.196361", "本地复测"),
    ]
    add_table(doc, ["方法", "年份/会议", "PSNR↑", "SSIM↑", "LPIPS↓", "数据来源"], methods, [1.55, 0.95, 0.85, 0.85, 0.9, 0.9])
    doc.add_picture(str(ASSETS / "metric_comparison.png"), width=Inches(6.7))
    add_caption(doc, "图 1  LLFF 3-view 指标对比。公开结果与本地结果的协议差异见第 4 节。")
    doc.add_paragraph("从统一表看，本项目 EMA@12000 的 PSNR 比 GoLF-NRT 低 4.403 dB，比 Path Matters 低 1.133 dB；SSIM 比 GoLF-NRT 低 0.144；LPIPS 则优于 SparseNeRF、SPARF、SCGaussian 和 NexusGS，但落后于 MVPGS 和 GoLF-NRT。")

    doc.add_heading("2. 本项目不同 checkpoint 的指标变化", level=1)
    rows = []
    for step in summary["steps"]:
        m = summary["macro_mean"][str(step)]["ema"]
        rows.append((step, f"{m['psnr']:.6f}", f"{m['ssim']:.6f}", f"{m['lpips_alex']:.6f}"))
    add_table(doc, ["全局 step", "EMA PSNR↑", "EMA SSIM↑", "EMA LPIPS↓"], rows, [1.2, 1.2, 1.2, 1.2])
    doc.add_paragraph("训练步数增加后，LPIPS 持续改善，但 PSNR 下降；因此报告中必须固定 checkpoint 选择规则。EMA@6000 的 PSNR 最高，EMA@12000 的 SSIM 最高，EMA@18000 的 LPIPS 最低。")

    doc.add_heading("3. 图片结果对比", level=1)
    doc.add_paragraph("公开方法中，GoLF-NRT 的定性结果使用了 LLFF 中常见的 Orchids 场景，展示 Target View、Source Views、局部几何、全局上下文和最终 GoLF-NRT 输出。本报告同时展示本项目 Orchids 场景的 ground truth、EMA 渲染和预测深度。两组图用于观察纹理、边界和结构保持能力；由于公开图与本地测试帧不保证为同一 frame ID，不做像素级方法排名。")
    public_img = ASSETS / "golf/a9cfa8ee-b5e3-49b1-9e5f-9919bd9f2dba.jpg"
    doc.add_picture(str(public_img), width=Inches(6.7))
    add_caption(doc, "图 2  GoLF-NRT 公开 Orchids 场景定性图：Target/Source、Only Local Geometry、Only Global Context 与 GoLF-NRT。来源：GoLF-NRT 项目介绍页。")
    doc.add_picture(str(ASSETS / "local/orchids_detail.png"), width=Inches(6.7))
    add_caption(doc, "图 3  本项目 Orchids 场景：ground truth、DiffusioNeRF+NeurTV+Ray EMA@18000 和预测深度。")
    doc.add_picture(str(ASSETS / "local/scene_grid.png"), width=Inches(5.9))
    add_caption(doc, "图 4  本项目跨场景定性结果：Fern、Flower、Orchids、Room 的 ground truth 与 EMA 渲染。")
    doc.add_paragraph("本地结果总体能够恢复场景颜色和主体布局，但在叶片、花瓣边缘、细密纹理和遮挡区域仍可观察到平滑化、颜色混合和局部伪影。这与当前 SSIM 偏低、LPIPS 尚有竞争力但 PSNR 不高的指标组合一致。")

    doc.add_heading("4. 实验协议、数据来源与可比性", level=1)
    add_bullets(doc, [
        "本项目：8 个 LLFF 场景独立训练；测试图像按原始轨迹每隔 8 张取一张；3 个训练视图从非测试池中固定均匀选择；评估同时包含 Raw 和 EMA。",
        "本报告主结果：8 场景宏平均，即先得到每个场景的帧平均，再对 8 个场景等权平均。",
        "公开方法：采用论文/项目页面公布的 LLFF 3-view 数值；不同论文可能在分辨率、输入视图、相机 pose、LPIPS backbone、测试帧和平均方式上存在差异。",
        "图片：GoLF-NRT 公开图属于论文/项目页面的定性结果；本项目图片来自本地 milestone_sweep 评估归档。",
    ])
    doc.add_paragraph("因此，本报告可以支持方法定位和论文讨论，但不能替代在完全相同代码、split、预处理和硬件下重新运行所有公开方法的严格复现实验。")

    doc.add_heading("5. 后续实验建议", level=1)
    add_bullets(doc, [
        "论文主表固定使用 EMA@12000 或预先注册的最佳 checkpoint 规则，不要对 PSNR、SSIM、LPIPS 分别挑选不同 step 后合并为一个方法分数。",
        "若需要声称超过近期方法，应优先复现 MVPGS、SCGaussian、NexusGS 或 GoLF-NRT 中至少一种，并强制使用当前项目的 split 文件和测试帧。",
        "定性图建议统一使用 Orchids、Fern、Flower 三个场景，并固定相同测试 frame ID；当前公开图只能作为文献参考图，不能作为严格横向拼图。",
    ])
    doc.add_heading("参考来源", level=1)
    doc.add_paragraph("GoLF-NRT（CVPR 2025）实验结果：https://klmav.cuc.edu.cn/2025/0529/c2286a256259/page.htm")
    doc.add_paragraph("Path Matters（ICLR 2026）Table 3：https://openreview.net/pdf/1ad8bd881c26849b79ebf5025a130a786ccd32ec.pdf")
    doc.add_paragraph("本项目统一指标汇总：test_LLFF/aggregates/llff_3v_neurtv_ray_common_milestones/summary.json")
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
