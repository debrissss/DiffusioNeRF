# 交接文档：补全 DTU 10 个 scan 的全量精确掩码（写于 2026-09-07）

> 交接对象：下一个接手的 agent。读完本文档应可直接开工，无需追问。
> 现状与已完成成果的完整背景见 **`MILESTONE_M1_ALPHA_PIPELINE.md`**（必读，
> 含管线原理、M1 配方、全部正负消融记录、评测口径）。

## 0. 一句话现状

DiffusioNeRF 少视图管线已修复并达标（scan63：9v 26.27 / 3v 20.07，masked
PSNR，见 M1 文档），但依赖**精确的全 49 帧物体掩码**（alpha 通道训练 +
masked 评测）；15 个标准 scan 中 **10 个缺训练帧掩码**（只有 25 个测试帧的
官方掩码）。官方几何与标定数据已下载并验证到位（见 §2），剩余工作是把它们
变成全量掩码并接入训练。

## 1. 为什么用"官方点云 + 相机投影"做掩码

- RegNeRF 的 full-mask 包已失效（Drive 权限拒绝），社区无镜像（已调研：
  DVR 公开包只有 3 个评测 scan、SparseNeRF/DNGaussian 都指向同一失效文件、
  SparseNeuS 的 SharePoint 404、CoR-GS 原始包无掩码）。
- 官方 DTU MVS 2014 数据自带两样东西，组合即可复现原始精度的掩码：
  `Points/stl/stlXXX_total.ply`（每 scan 的结构光扫描网格，官方评测用的
  就是它）+ `Calibration/cal18/pos_XXX.txt`（每帧投影矩阵 P）。
  把 PLY 顶点按 P 投影到像平面并填充，即得任意帧、任意分辨率的轮廓掩码。
  **掩码质量回验**（§3 第 4 步）保证与官方 25 帧掩码同水准后才采用。

## 2. 已就位的资源（已验证）

| 资源 | 路径 | 状态 |
|---|---|---|
| Points.zip（6.96GB，126 entries） | `/root/autodl-tmp/dtu_mvs/Points.zip` | ✅ 完整（条目可枚举）。`Points/stl/stlNNN_total.ply`，NNN=scan 号（stl001~stl122）。**需要的全部在列**：008/021/030/031/034/038/041/045/055/063/082/103/110/114 |
| SampleSet.zip（6.9GB，2115 entries） | `/root/autodl-tmp/dtu_mvs/SampleSet.zip` | ✅ 完整。所需文件：`SampleSet/MVS Data/Calibration/cal18/pos_001.txt` ~ `pos_064.txt`（共 64 个，已确认存在）。**整包含大量无关大文件（scan1/scan6 全数据、mp4），切勿整包解压**，用 `zipfile` 按名提取 |
| 待补全的 10 个 scan | `data/DTU_standard/scan{8,21,30,31,34,38,41,45,82,103}/` | transforms.json（49 帧位姿+内参）、images/（400×300）、masks/（25 测试帧官方掩码 400×300）均就绪 |
| 已有全量掩码的 scan | scan24/40/55/63（49 帧，IDR 系）；scan110/114 有 64 帧版待验证帧号 | 见 §3 第 6 步 |

PLY 数量 124 > 64 的原因：Points 里含多个版本的 stl 文件（total/其他后缀），
取 `stlNNN_total.ply` 即可。另：仓库根有个 14.9MB 的
`SampleSet.zip` 残缺文件（`/root/autodl-tmp/DiffusioNeRF-main/SampleSet.zip`，
上传中断产物），可删除。

## 3. 任务步骤（按序执行）

### 第 1 步：流式提取所需文件
```python
import zipfile
zs = zipfile.ZipFile('/root/autodl-tmp/dtu_mvs/SampleSet.zip')
for n in zs.namelist():
    if 'Calibration/cal18/' in n:           # 64 个 pos_XXX.txt，很小
        zs.extract(n, '/root/autodl-tmp/dtu_mvs/extracted')
zp = zipfile.ZipFile('/root/autodl-tmp/dtu_mvs/Points.zip')
# 逐个提取 10 个 scan 的 stlNNN_total.ply（每个几百 MB，zipfile 流式读取）
```
注意：`SampleSet.zip` 打开时可能报"98304 extra bytes"警告，zipfile 能容忍。

### 第 2 步：解析投影矩阵与 PLY
- **`pos_XXX.txt` 格式已实锤**：ASCII，3 行 × 4 列的投影矩阵 P
  （以 pos_001.txt 为例首行 `2607.43 -3.84 1498.18 -533936.66`）。
  投影：`u = (P[0]·X)/(P[2]·X)`，`v = (P[1]·X)/(P[2]·X)`，要求 `P[2]·X>0`。
  输出像素坐标基于 1600×1200 原始分辨率（缩到 400×300 时除以 4）。
  坐标系为 DTU 毫米制，与 stlXXX_total.ply 的顶点单位一致，可直接投影。
- **PLY 格式已实锤**：`format binary_little_endian`，每 scan 约 4M 顶点，
  属性 `float x,y,z + float nx,ny,nz + uchar r,g,b`（30 字节/顶点，
  stl041_total.ply 共 108MB / 4,032,117 顶点）。**无需 plyfile**：读头部
  `end_header` 偏移后用 `np.frombuffer` 按
  `dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('nx','<f4'),('ny','<f4'),('nz','<f4'),('r','u1'),('g','u1'),('b','u1')]`
  解析即可。顶点数百万级，全量读入内存约 120MB，无压力。

### 第 3 步：投影生成掩码（核心）
1. 帧号映射假设：`pos_XXX.txt` 的 XXX 对应官方图像编号（1-based），
   我们的 `transforms.json` frame_id 是零基 → **frame_id = XXX - 1**。
   ⚠️ 该映射必须先用第 4 步回验（若帧号系统性错位，IoU 会接近 0）。
2. 对每个顶点 X：`u = (P[0]·X)/(P[2]·X)`, `v = (P[1]·X)/(P[2]·X)`，
   只保留 `P[2]·X > 0` 且落在 1600×1200 原始分辨率内的点（pos 矩阵输出
   像素坐标基于 1600×1200，最终缩放到 400×300 时除 4，或直接在
   400×300 上生成：u/4, v/4）。
3. 像素坐标可能存在 0/1-based 半像素偏移——回验时若 IoU 差但形状对，
   尝试整体平移 ±1px。
4. 掩码栅格化：把投影点写入 400×300 布尔画布 → `cv2.morphologyEx`
   CLOSE（核 15~21）→ `cv2.floodFill`/孔洞填充 → 二值掩码。
   PLY 点密度足够时该法稳定；若边缘锯齿影响大，可先在 800×600 生成
   再 INTER_AREA 缩到 400×300。

### 第 4 步：质量回验（合格标准，不达标不要往下走）
对每个 scan 的 25 个有官方掩码的帧：IoU(生成, 官方)。
- **通过线：均值 ≥ 0.90 且无整块缺失**（可视抽查 2~3 帧叠加图最直观）。
- 已知风险与对策：
  - DTU 评测 STL 可能含**桌面/背板杂点**（不属物体）。对策：先在 25 个
    官方帧上统计每个顶点"投影落入官方掩码内"的比例，比例 <0.5 的顶点
    一次性剔除（只用官方数据，合法），再投影全部 49 帧。
  - 帧号映射错位（见第 3 步第 1 条）。
  - 投影 w 分量符号/手性错误（典型症状：掩码整体镜像或颠倒）。
- 顺带任务：用同样 IoU 手法验证 **scan110/114 的 64 帧包**（`idrmasks`
  包 `mask/000-063`）与现有 25 帧掩码的帧号是否对齐（对齐则这两个 scan
  也立即获得全量掩码）。

### 第 5 步：写入仓库并重制 alpha 数据集
1. 全量掩码写入 `data/DTU_standard/scan{X}/masks/{fid:06d}.png`
   （400×300，白=物体；现有 25 帧文件直接被官方来源的精确版覆盖即可）。
2. 重建 10 个 scan 的 `masks_train_approx/`：现在可用**精确掩码**（建议
   轻膨胀 9px 保护物体边缘），替换现有的"邻近帧+81px 膨胀"近似版。
3. 对 10 个 scan 运行 `python scripts/make_alpha_dtu.py data/DTU_standard/scanX`
   生成 `scanX_alpha/`（注意：该脚本目前只写 3v/9v split；**6v 需要补**：
   照 `splits/dtu_6v/scanX.json` 复制改 scene 名）。

### 第 6 步：训练验证（先 scan41 打样，再推广）
- 3v：12k 步，配方 = M1 文档 §4 的 2a 命令，把路径换成 scan41_alpha，
  **必须带 `--eval_bg_color gray`**（泄漏匹配，M1 §1.3，漏了会少 ~1-2 dB）。
- 9v：15k 步（14994），同配方换 9v split；可选 `--milestone_steps 11997 14004`
  + `scripts/soup_checkpoints.py`（9v 上验证过 +0.4）。
- 退火长度按视图数：3v=5000（12k 步）/ 9v=7000（15k 步）。
- 对照锚点：scan41 从未有过可用数字（白底时代 11→8 发散；黑底时代
  15.4 全图）；本次预期 masked PSNR 至少 19+（9v 应 23+）。显著低于此
  说明掩码或数据有问题，回查第 4 步。
- 通过后把其余 9 个 scan 全部跑完（或按用户指令），结果写回
  `MILESTONE_M1_ALPHA_PIPELINE.md`（消融表加行、遗留事项第 2 条销项）。

## 4. 环境与坑（前一个 agent 踩过的）

1. **代理白名单**：环境变量挂了 http_proxy。可直连的：DTU 官方服务器
   （`curl --noproxy '*'`）、S3、Google Drive（gdown）、raw.githubusercontent。
   被挡的：dl.fbaipublicfiles、hf-mirror、roboimagedata（走代理时）、github
   archive tarball。pip 默认走阿里云镜像且缺包（segment-anything 装不上），
   用 `-i https://pypi.org/simple` 重试。
2. **pkill 误杀**：`pkill -f "xxx"` 的模式若出现在自己命令行里会自杀。
   用 `ps -eo pid,cmd | grep [x]xx` 取 PID 再 kill。
3. **wget 旗标**：直连用 `wget --no-proxy`（curl 才是 `--noproxy '*'`）。
4. **wget -c 续传会损坏 zip**（偏移错位），大文件中断后请整删重下。
5. 本机 PIL/cv2 读图正常；PLY 解析优先 `plyfile`，装不上就手写
   （头部 ASCII 声明 format/vertex 属性）。
6. 训练/评测的所有坑（AMP 重试、epoch 对齐、eval_bg_color）见
   `MILESTONE_M1_ALPHA_PIPELINE.md` 与 `events` 记录；启动训练前先读 §4。

## 5. 验收标准

1. 10 个 scan 的 `masks/` 全部为 49 帧，且 25 帧官方回验 IoU ≥ 0.90；
2. 10 个 scan 的 `scanX_alpha/` 数据集就绪（3v/9v/6v split 齐）；
3. scan41 的 3v 与 9v 训练+评测完成，masked PSNR 达到合理区间
   （3v ≥ 19、9v ≥ 23 量级），结果与归档路径写回 M1 文档；
4. `MILESTONE_M1_ALPHA_PIPELINE.md` 的遗留事项第 2 条销项。

## 6. 关键路径速查

```
数据(已验证)      /root/autodl-tmp/dtu_mvs/{Points.zip, SampleSet.zip}
标准数据集        data/DTU_standard/scan{8..114}/  （10 个待补掩码 scan 在此）
alpha 数据范例    data/DTU_standard/scan63_alpha/  （M1 最优结果的数据形态）
掩码现况          data/DTU_standard/scan*/masks/    （25 帧官方 + 少量近似）
工具              scripts/make_alpha_dtu.py  scripts/soup_checkpoints.py
                  evaluate_dtu_masked_results.py
                  /tmp/hull_masks.py  （早期 visual-hull 尝试，投影/约定可参考，
                                        但其 IoU 仅 0.67，勿直接复用结论）
训练入口          main_nerf.py（+ run_DiffusioNeRF_DTU_NeurTV_Ray.sh）
里程碑文档        MILESTONE_M1_ALPHA_PIPELINE.md
9v 冠军归档       test_DTU/test_scan63_alpha/few_shot9/test_S2_18k_s9k/
                  evaluation/step_017991_soup_gray/   (26.27)
3v 冠军归档       test_DTU/test_scan63_alpha/few_shot3/test_S5k_e95/
                  evaluation/step_012000_graybg/      (20.07)
```
