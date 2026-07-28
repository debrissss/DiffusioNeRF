# DiffusioNeRF 稀疏视点实验补全计划书

> 调研范围：2024–2026 年，同时在 LLFF 和 DTU 上报告结果的 NeRF / 3DGS 稀疏视点方法
> 实验范围：只补跑本项目方法，不复现外部对比方法
> 日期：2026-07-28

## 1. 结论摘要

本计划遵循四条原则：

1. 外部对比方法只采用原论文明确报告的结果，不安排本地复现。
2. 本项目需要同时报告两套互不混用的配置：
   - **A：论文叙述忠实配置**，用于论文正文中的 `Ours`。
   - **B：项目历史兼容配置**，用于解释当前代码、脚本和已有 `test/` 结果。
3. 所有新实验使用标准 LLFF/DTU 划分、完整场景集合和统一评测脚本。
4. 先完成最重要的 3-view 主实验，再做消融、9-view 和 6-view。

最终筛出 7 组可以直接引用到论文中的新方法：

- 2024：DNGaussian、CoR-GS、IPSM-Gaussian。
- 2025：SYN3R、Binocular3DGS + Gaussian Dropout。
- 2026：EdgeNeRF、ICO-GS。

其中 EdgeNeRF 是 arXiv 预印本，论文表格中必须标注；其余为正式会议论文。

计划中的主实验共 345 次训练；加入去重后的关键消融 81 次，共计 **426 次唯一训练运行**。该数量不包含失败重跑以及只执行渲染/评测的任务。

## 2. 可以直接放入论文的对比方法

### 2.1 最高优先级

#### ICO-GS，CVPR 2026，3DGS

这是目前最重要的新基线：

- 同时覆盖 LLFF 和 DTU。
- 同时覆盖 3、6、9-view。
- 报告三个随机种子的平均结果。
- 是当前候选中最新且整体性能最强的方法之一。

作者报告结果：

| 数据集 | 视图数 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| LLFF | 3 | 22.20 | 0.778 | 0.157 |
| LLFF | 6 | 25.37 | 0.856 | 0.109 |
| LLFF | 9 | 26.45 | 0.881 | 0.096 |
| DTU | 3 | 21.77 | 0.888 | 0.092 |
| DTU | 6 | 25.09 | 0.928 | 0.064 |
| DTU | 9 | 27.19 | 0.953 | 0.045 |

来源：[ICO-GS，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Xiong_Intrinsic_Geometry-Appearance_Consistency_Optimization_for_Sparse-View_Gaussian_Splatting_CVPR_2026_paper.pdf)

#### SYN3R，NeurIPS 2025，3DGS + 视频扩散

SYN3R 使用预训练视频扩散模型补全中间视图，与本项目“利用生成先验约束未观测区域”的动机最接近。

| 数据集 | 视图数 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| LLFF | 3 | 20.61 | 0.705 | 0.201 |
| DTU | 3 | 20.51 | 0.840 | 0.137 |

来源：[SYN3R，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4254e856d01a5e7b7ea050477c3ef9b9-Abstract-Conference.html)

#### IPSM-Gaussian，NeurIPS 2024，3DGS + 图像扩散

IPSM-Gaussian 直接研究如何在稀疏视点下使用扩散先验，是与本项目最接近的扩散类比较方法。

| 数据集 | 视图数 | PSNR | SSIM | LPIPS | AVGE |
|---|---:|---:|---:|---:|---:|
| LLFF | 3 | 20.44 | 0.702 | 0.207 | 0.101 |
| LLFF | 6 | 23.94 | 0.818 | 0.135 | 0.061 |
| LLFF | 9 | 25.13 | 0.855 | 0.111 | 0.051 |
| DTU | 3 | 19.99 | 0.856 | 0.121 | 0.077 |

其 3-view 结果是三次独立运行的均值；LLFF 6/9-view 使用与 3-view 相同的超参数。

来源：[IPSM-Gaussian，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/362d683723e7d6c12a093961ec2e5051-Paper-Conference.pdf)

#### EdgeNeRF，2026，NeRF，预印本

EdgeNeRF 是本轮筛选中唯一同时满足下列条件的 2026 NeRF 类方法：

- 使用 NeRF 表示而非 3DGS。
- 同时报告 LLFF 和 DTU。
- 使用 3-view 稀疏输入。
- 原文给出数据集聚合指标。

| 数据集 | 视图数 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| LLFF | 3 | 19.42 | 0.699 | 0.317 |
| DTU | 3 | 19.42 | 0.828 | 0.205 |

它尚未正式同行评审，因此建议放在 NeRF 分组末尾，并写作 `EdgeNeRF (arXiv'26)`。

来源：[EdgeNeRF](https://arxiv.org/abs/2601.01431)

### 2.2 第二优先级

#### CoR-GS，ECCV 2024

CoR-GS 不依赖扩散模型或外部生成先验，并完整报告 LLFF/DTU 的 3、6、9-view，适合作为“纯几何共正则”基线。

| 数据集 | 视图数 | PSNR | SSIM | LPIPS | AVGE |
|---|---:|---:|---:|---:|---:|
| LLFF | 3 | 20.45 | 0.712 | 0.196 | 0.101 |
| LLFF | 6 | 24.49 | 0.837 | 0.115 | 0.060 |
| LLFF | 9 | 26.06 | 0.874 | 0.089 | 0.046 |
| DTU | 3 | 19.21 | 0.853 | 0.119 | 0.087 |
| DTU | 6 | 24.51 | 0.917 | 0.068 | 0.046 |
| DTU | 9 | 27.18 | 0.947 | 0.045 | 0.029 |

来源：[CoR-GS](https://arxiv.org/abs/2405.12110)

#### DNGaussian，CVPR 2024

DNGaussian 是深度先验类 3DGS 的代表方法，也是后续稀疏 3DGS 工作经常采用的比较基线。

| 数据集 | 视图数 | PSNR | SSIM | LPIPS | AVGE |
|---|---:|---:|---:|---:|---:|
| LLFF | 3 | 19.12 | 0.591 | 0.294 | 0.132 |
| DTU | 3 | 18.91 | 0.790 | 0.176 | 0.102 |

来源：[DNGaussian，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_DNGaussian_Optimizing_Sparse-View_3D_Gaussian_Radiance_Fields_with_Global-Local_Depth_CVPR_2024_paper.pdf)

#### Binocular3DGS + Gaussian Dropout，NeurIPS 2025

论文提出 Gaussian Dropout 和 opacity noise 两类即插即用正则，而不是一个完全独立的重建框架。

为了避免挑选单项最优，统一采用作者重点推荐的：

```text
Binocular3DGS + Gaussian Dropout, p = 0.2
```

| 数据集 | 视图数 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| LLFF | 3 | 22.12 | 0.777 | 0.154 |
| DTU | 3 | 21.03 | 0.875 | 0.108 |

不能在 LLFF 使用 opacity noise、在 DTU 使用 dropout/both，以获得逐数据集更好的单项指标。

来源：[Quantifying and Alleviating Co-Adaptation，NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/a831bd2b1bf8935641a5c890a158c519-Abstract-Conference.html)

### 2.3 暂不进入主表的方法

| 方法 | 不进入主表的原因 |
|---|---|
| DWTNeRF，2025 | 报告 LLFF 和 NeRF-Synthetic，但未同时报告 DTU |
| TrackNeRF，2024 | 主要研究稀疏且带噪声位姿，任务假设与本项目不一致 |
| ReconFusion | 所需数据集/视图设置覆盖不完整 |
| PViGS，2026 | 确认使用 LLFF/DTU 3/6/9-view，但当前正式卷期尚晚于本次调研时间，且没有获得可逐项核对的完整作者表格 |
| TWINGS，2026 | 重点是初始化机制；如果后续加入，应单独标注初始化条件，不能直接当作同类正则化方法 |

完整数值矩阵见 [comparison_matrix.csv](./comparison_matrix.csv)。

## 3. 引用作者报告值时的公平性边界

外部方法结果可以直接用于论文，但表题必须表明它们是作者报告值，例如：

> Published results under the standard sparse-view protocols. Baseline values are transcribed from the original papers; only our method is executed in this project.

建议论文表格额外增加三列：

| 列 | 作用 |
|---|---|
| Representation | NeRF / 3DGS |
| External prior / initialization | depth、image diffusion、video diffusion、SfM、dense point cloud 等 |
| Source | 原论文引用 |

需要明确：

- 3DGS 和 NeRF 的表示、背景行为及渲染效率不同。
- DNGaussian 原版、IPSM、ICO-GS、SYN3R 使用的初始化或外部先验并不相同。
- DTU 必须使用 object mask，否则指标不能和这些论文结果并列。
- 作者论文可能使用略有差异的 SSIM 实现，因此所有外部结果必须标记为 `reported`，本项目结果标记为 `ours-run`。
- 只能在同数据集、同视图数、同 mask 规则内计算粗体和下划线排名。

## 4. 当前项目与已有结果审计

### 4.1 现有结果不能直接进入新主表

[test/our/定量指标.txt](../../test/our/定量指标.txt) 目前主要整理了：

- LLFF：fern、room 的 3-view 和 9-view。
- DTU：scan41、scan63 的 3-view 和 9-view。

这些不是：

- LLFF 全部 8 场景的平均结果。
- DTU 标准 15 场景的平均结果。
- 三个随机种子的 mean ± std。

`test/` 中没有同时保存：

- 完整训练命令。
- resolved config。
- split 文件。
- seed。
- git commit。
- 训练日志。
- checkpoint provenance。

因此无法严格证明已有数字对应哪个代码版本和实验配置。旧结果只能用于：

- 定性展示。
- 回归检查。
- 帮助定义项目历史兼容配置 B。

不能直接与新方法的数据集聚合值并列。

### 4.2 当前启动脚本的显式配置

[run_DiffusioNeRF_LLFF.sh](../../run_DiffusioNeRF_LLFF.sh) 当前指定：

```text
fp16
few_shot = 3
iters = 15000
diff_reg_start_iter = 1000
dist_lambda = 2e-5
fg_lambda = 1e-4
patch_size = 4
depth_reg_lambda = 0.1
use_nll_color
use_nll_sigma
fre_nll_color
smoothing_lambda = 1e-5
smooth_sampling_method = near_pixel
```

当前脚本只循环 `fern/3-view`，没有跑全部 8 个场景，也没有传入 `--neurtv`。

### 4.3 当前代码中的隐式配置

[nerf/utils.py](../../nerf/utils.py) 和 [main_nerf.py](../../main_nerf.py) 中还存在脚本没有明确写出的行为：

- 虚拟射线默认开启。
- 虚拟射线从 global step 1000 开始。
- `virtual_ray_k=10`。
- `virtual_ray_jsd_th=0.02`。
- 虚拟射线 depth loss 的实际回退权重为 `0.1`，不是代码注释中的 `1e-7`。
- NeuraTV 默认 start 为 3000，但只有传入 `--neurtv` 才启用。
- 默认学习率为 `1e-2`，最终只衰减到 `1e-3`。
- 非 CUDA ray 模式使用 64 个基础采样和 128 个上采样。

### 4.4 当前划分不符合标准协议

[nerf/provider.py](../../nerf/provider.py) 当前 colmap 划分为：

```python
train = frames[5:]
val = frames[:1]
test = frames[1:5]
```

随后 few-shot 又在 `frames[5:]` 中使用 `linspace` 选图。

这会造成两个问题：

1. 测试集不是 LLFF 的 `i mod 8 = 0`。
2. 标准 holdout 帧可能进入训练池。

即使某些已有结果导出的测试图数量恰好等于标准测试数量，也不能证明训练 ID 和测试 ID 符合标准协议。

### 4.5 数据路径修复状态

预检发现以下 `transforms.json` 的 `file_path` 带有与实际目录不一致的前缀。现已统一修复：

| 场景 | 修复的 frame 数 | 修复内容 |
|---|---:|---|
| flower | 34 | `r/images/...` → `images/...` |
| fortress | 42 | `s/images/...` → `images/...` |
| horns | 62 | `s/images/...` → `images/...` |
| leaves | 26 | `s/images/...` → `images/...` |

修复后已对 LLFF 8 个场景进行全量验证：305/305 张图像路径存在、可由 OpenCV 解码，且不存在重复路径。orchids 保留 `s/images/...`，因为该场景的实际图片确实存放在 `orchids/s/images/`，不是错误路径。

## 5. 两套主实验配置

两套配置必须分别保存、分别聚合、分别报告。不能在场景或数据集级别挑选更高的结果。

### 5.1 配置 A：论文叙述忠实配置

定位：论文正文中的正式 `Ours`。

| 配置项 | 固定值 |
|---|---|
| 硬件 | 单卡 48 GB vGPU |
| CUDA | 12.8 |
| Python | 3.8 |
| 优化器 | Adam |
| 初始 LR | 2e-2 |
| 最终 LR | 2e-5 |
| LR 调度 | 按 optimizer step 指数衰减 |
| 总训练量 | 35,000 optimizer steps |
| rays/step | 4096 |
| samples/ray | 128 |
| 论文中的 batch size | 32，需在实现映射中单独解释 |
| 虚拟射线开始 | step 3500 |
| 虚拟射线 K | 10 |
| JSD 阈值 | 0.02 |
| 虚拟射线 depth λ | 1e-7 |
| NeuraTV 开始 | step 5000 |
| NeuraTV λ | 1e-7 |
| NeuraTV samples | 4096 |
| 视图设置 | 3-view、9-view |
| seeds | 0、1、2 |

实现时需要解决的歧义：

- `4096 rays/step` 和论文中的 `batch size=32` 不能互相覆盖。
- 建议将 `128 samples/ray` 明确实现为 `num_steps=128, upsample_steps=0`。
- 如果使用 FP16，需要在论文实验设置中补充说明。

### 5.2 配置 B：项目历史兼容配置

定位：解释当前实现和历史结果，放在附录或实现分析中。

| 配置项 | 固定值 |
|---|---|
| 当前硬件 | RTX 4080 SUPER vGPU，32760 MiB |
| Python | 3.8.10 |
| PyTorch/CUDA | 2.0.0+cu118 |
| 精度 | FP16 |
| 优化器 | Adam |
| betas | (0.9, 0.99) |
| eps | 1e-15 |
| 初始 LR | 1e-2 |
| 最终 LR | 1e-3 |
| LR 调度 | 按 optimizer step 指数衰减 |
| 总训练量 | 30,000 optimizer steps |
| rays/step | 4096 |
| 基础采样 | 64 |
| 上采样 | 128 |
| patch size | 4 |
| DiffusioNeRF start | step 1000 |
| dist λ | 2e-5 |
| foreground λ | 1e-4 |
| depth λ | 0.1 |
| smoothing λ | 1e-5 |
| smoothing | near_pixel |
| NLL | color + sigma |
| frequency mask | color |
| 虚拟射线开始 | step 1000 |
| 虚拟射线 K | 10 |
| JSD 阈值 | 0.02 |
| 虚拟射线 depth λ | 0.1 |
| NeuraTV 开始 | step 3000 |
| NeuraTV λ | 1e-7 |
| NeuraTV samples | 4096 |
| 视图设置 | 3-view、6-view、9-view |
| seeds | 0、1、2 |

B 的 Full 模型必须显式传入 `--neurtv`。为了连接旧结果，消融中另设：

1. DiffusioNeRF baseline。
2. baseline + Virtual Ray。
3. baseline + NeuraTV。
4. Full：Virtual Ray + NeuraTV。

### 5.3 几何缩放

LLFF 沿用当前脚本中的 scene-specific scale/bound：

| 场景 | scale | bound |
|---|---:|---:|
| trex | 0.33 | 4.5 |
| horns | 0.20 | 3.5 |
| fortress | 0.30 | 3.5 |
| fern | 0.33 | 3.0 |
| room | 0.30 | 3.0 |
| flower | 0.01 | 3.0 |
| orchids | 0.10 | 2.5 |
| leaves | 0.005 | 3.0 |

DTU 当前没有对应 runner。默认固定：

```text
scale = 0.33
bound = 2.0
downscale = 4
```

在 smoke test 中检查：

- 所有相机中心是否位于有效范围。
- near/far 是否有效。
- 有效射线采样比例。
- 是否发生大量空采样。

不能为了提高单个场景结果，事后单独调整 DTU scale/bound。

## 6. 标准数据协议

### 6.1 LLFF

固定使用全部 8 个场景：

```text
fern, flower, fortress, horns,
leaves, orchids, room, trex
```

测试集：

```text
test_ids = {i | i mod 8 = 0}
```

训练集生成步骤：

1. 从全部图像中移除测试图。
2. 按原相机轨迹保持顺序。
3. 在剩余 pool 上使用确定性等间距位置选择 3、6、9 张。
4. 使用 half-up rounding。
5. 将最终 ID 写入 JSON，运行时不再计算。

拟冻结的 ID：

| 场景 | test | 3-view | 6-view | 9-view |
|---|---|---|---|---|
| fern | 0,8,16 | 1,10,19 | 1,4,7,12,15,19 | 1,3,5,7,10,12,14,17,19 |
| flower | 0,8,16,24,32 | 1,17,33 | 1,7,13,20,26,33 | 1,5,9,13,17,21,25,29,33 |
| fortress | 0,8,16,24,32,40 | 1,21,41 | 1,9,17,25,33,41 | 1,5,11,15,21,26,30,36,41 |
| horns | 0,8,16,24,32,40,48,56 | 1,31,61 | 1,13,25,37,49,61 | 1,9,15,23,31,38,46,53,61 |
| leaves | 0,8,16,24 | 1,13,25 | 1,5,10,15,20,25 | 1,4,6,10,13,15,19,21,25 |
| orchids | 0,8,16,24 | 1,12,23 | 1,5,10,14,19,23 | 1,4,6,10,12,15,18,21,23 |
| room | 0,8,16,24,32,40 | 1,20,39 | 1,9,17,23,31,39 | 1,5,11,15,20,25,30,35,39 |
| trex | 0,8,16,24,32,40,48 | 1,28,54 | 1,11,22,33,44,54 | 1,7,14,21,28,34,41,47,54 |

所有 LLFF 图像统一 8× 下采样。

### 6.2 DTU

标准 15 场景：

```text
8, 21, 30, 31, 34,
38, 40, 41, 45, 55,
63, 82, 103, 110, 114
```

训练 ID：

```text
3-view = [25, 22, 28]
6-view = [25, 22, 28, 40, 44, 48]
9-view = [25, 22, 28, 40, 44, 48, 0, 8, 13]
```

测试 ID：

```text
[1, 2, 9, 10, 11, 12, 14, 15,
 23, 24, 26, 27, 29, 30, 31, 32,
 33, 34, 35, 41, 42, 43, 45, 46, 47]
```

评测要求：

- 图像 4× 下采样。
- 使用 object mask。
- 所有方法使用相同测试 ID。
- 15 个场景等权聚合。

## 7. 指标与聚合

### 7.1 指标

- PSNR。
- SSIM。
- LPIPS-Alex。
- AVGE。

DTU 在应用 object mask 后再计算指标。

### 7.2 聚合顺序

1. 对每张测试图计算指标。
2. 对同一场景的测试图取平均。
3. 对数据集中的场景等权平均。
4. 每个 seed 得到一个数据集结果。
5. 对 seeds 0、1、2 报告 mean ± std。

不能直接把所有测试图放在一起平均，否则测试图较多的场景权重更高。

AVGE 定义：

```text
MSE = 10^(-PSNR / 10)
AVGE = geomean(sqrt(1 - SSIM), LPIPS, MSE)
```

需要冻结 AVGE 是在数据集聚合指标上计算，还是先逐场景计算再平均；为对齐 IPSM、DNGaussian、CoR-GS，默认采用它们的官方评测顺序并通过小样本验证。

## 8. 实验优先级

### P0：协议和可复现性修复

任何大规模训练前必须完成。

任务：

1. 新增标准 split manifest 生成器。
2. 数据加载器改为读取明确的 train/test ID。
3. 删除 `frames[5:]`、`frames[1:5]` 对正式 benchmark 的影响。
4. 验证已完成的 flower、fortress、horns、leaves 路径修复。
5. 给虚拟射线增加以下显式 CLI：
   - enable/disable。
   - start step。
   - K。
   - JSD threshold。
   - depth lambda。
6. 显式配置：
   - NeuraTV。
   - LR 终点。
   - samples/ray。
   - optimizer steps。
7. 建立统一 evaluator。
8. 每次运行自动保存配置与 provenance。

P0 不计入正式训练运行数。

### P1：3-view 全场景主实验

这是最高优先级的正式实验，因为全部 7 组新方法都报告了 LLFF/DTU 3-view。

矩阵：

```text
A + B
× (LLFF 8 scenes + DTU 15 scenes)
× seeds {0,1,2}
= 2 × 23 × 3
= 138 runs
```

执行方式：

1. 先跑 A/B 的 seed 0，共 46 次。
2. 检查划分、指标、NaN、配置和结果趋势。
3. 通过后再补 seed 1、2。
4. 生成 3-view 主表。

### P2：关键消融

配置 A：

```text
23 scenes × 4 variants × seed 0 = 92
```

其中 Full 的 23 次已包含在 P1，新增：

```text
23 × 3 = 69 runs
```

配置 B 只做历史桥接场景：

```text
fern, room, scan41, scan63
4 scenes × 4 variants × seed 0 = 16
```

扣除 P1 中已经完成的 4 个 Full：

```text
新增 12 runs
```

P2 总增量：

```text
69 + 12 = 81 runs
```

四个变体：

1. DiffusioNeRF baseline。
2. baseline + Virtual Ray。
3. baseline + NeuraTV。
4. Full。

### P3：9-view 主实验

```text
A + B
× 23 scenes
× 3 seeds
= 138 runs
```

用途：

- 补全论文原有 9-view 叙述。
- 对比 CoR-GS 和 ICO-GS。
- LLFF 上进一步对比 IPSM-Gaussian。

### P4：6-view 扩展

配置 A 的论文叙述没有定义 6-view，因此默认只跑 B：

```text
B × 23 scenes × 3 seeds = 69 runs
```

若后续决定正文也需要 A-6，必须把它标为新增协议，不能悄悄归入原论文配置。

### P5：定性图和效率

复用已完成 checkpoint，不增加正式训练数。

推荐场景：

- LLFF：fern、flower、leaves、room。
- DTU：scan30、scan41、scan63、scan103。

输出：

- RGB。
- error map。
- depth。
- 固定测试 ID 的局部 crop。
- 单场景训练时间。
- 峰值显存。
- checkpoint 大小。
- 渲染 FPS。

引用论文的效率结果如果使用不同 GPU，只能单独列出硬件，不能直接宣称倍数提升。

## 9. 运行数汇总

| 优先级 | 工作包 | 唯一训练数 |
|---:|---|---:|
| P0 | 协议、数据、CLI、评测修复 | 0 |
| P1 | A/B 3-view，完整场景，3 seeds | 138 |
| P2 | 去重后的消融增量 | 81 |
| P3 | A/B 9-view，完整场景，3 seeds | 138 |
| P4 | B 6-view，完整场景，3 seeds | 69 |
| P5 | 定性图和效率评测 | 0 |
|  | **合计** | **426** |

资源不足时的停止顺序：

1. 先删除 P4。
2. 再删除 B 的 9-view。
3. 再缩减 B 的桥接消融。

P0、P1 和 A 的关键消融不应删除。

## 10. 验收门槛

### 10.1 数据

- LLFF 8 个场景全部可读。
- DTU 15 个场景全部可读。
- LLFF 测试图数量分别为：

```text
fern=3, flower=5, fortress=6, horns=8,
leaves=4, orchids=4, room=6, trex=7
```

- DTU 每场景测试图数量为 25。
- `train ∩ test = ∅`。
- split 文件保存 SHA256。
- 禁止静默跳过无法读取的图片。

### 10.2 训练

- A 精确完成 35,000 optimizer steps。
- B 精确完成 30,000 optimizer steps。
- 不以 epoch 文件名代替 optimizer step。
- 记录初始、模块启用时刻、最终学习率。
- A 的 Virtual Ray/NeuraTV 分别在 3500/5000 开始。
- B 的 Virtual Ray/NeuraTV 分别在 1000/3000 开始。
- 无 NaN/Inf。

### 10.3 评测

- 从保存的预测图重新计算指标，结果与汇总一致。
- DTU mask 的使用有日志记录。
- per-view、per-scene、per-seed 和 final aggregate 能相互重算。
- 同一论文表使用同一 split version 和 evaluator version。

### 10.4 结果决策

- 不根据 seed 0 或单场景结果更换主配置。
- 只允许在发现可证明的代码或协议错误后整组重跑。
- Full 的收益应在两个数据集、多数 seeds 上方向一致。
- 如果只改善某个指标，论文结论必须收缩到该指标。

## 11. 运行目录与 provenance

推荐目录：

```text
runs/
  A-paper/
    LLFF/{3v,9v}/{scene}/seed_{0,1,2}/
    DTU/{3v,9v}/{scene}/seed_{0,1,2}/
  B-project/
    LLFF/{3v,6v,9v}/{scene}/seed_{0,1,2}/
    DTU/{3v,6v,9v}/{scene}/seed_{0,1,2}/
```

每个运行目录必须包含：

```text
resolved_config.yaml
split.json
split.sha256
environment.json
git_commit.txt
train.log
checkpoint.pth
per_view_metrics.csv
scene_summary.json
renders/
```

聚合输出：

```text
results/{config}/{dataset}/{views}/per_scene.csv
results/{config}/{dataset}/{views}/per_seed.csv
results/{config}/{dataset}/{views}/summary_mean_std.json

paper_tables/table_main_3v.csv
paper_tables/table_main_9v.csv
paper_tables/table_supp_6v.csv
paper_tables/table_ablation.csv
```

## 12. 论文中的最终表格

### 主表 1：3-view LLFF/DTU

按 NeRF 和 3DGS 分组，包含：

- EdgeNeRF。
- DNGaussian。
- CoR-GS。
- IPSM-Gaussian。
- SYN3R。
- Binocular3DGS + Gaussian Dropout。
- ICO-GS。
- Ours-A。

Ours-B 放附录。

### 主表 2：9-view LLFF/DTU

包含：

- CoR-GS。
- IPSM-Gaussian；未报告的 DTU-9 写 `—`。
- ICO-GS。
- Ours-A。

### 补充表：6-view

包含：

- CoR-GS。
- IPSM-Gaussian；只在 LLFF 有结果。
- ICO-GS。
- Ours-B。

### 消融表

```text
DiffusioNeRF baseline
+ Virtual Ray
+ NeuraTV
Full
```

A 的全数据集消融放正文；B 的四场景桥接消融放附录。

## 13. 推荐论文表注

英文：

> Published baseline values are transcribed from the original papers under their reported standard sparse-view protocols; only our method is executed in this project. DTU metrics are mask-aware. EdgeNeRF is an arXiv preprint.

中文：

> 对比方法数值来自其原论文在标准稀疏视点协议下报告的结果，本项目仅重新运行本文方法。DTU 指标在目标掩码内计算。EdgeNeRF 为 arXiv 预印本。

## 14. 当前风险

1. 历史结果缺少日志，无法严格追溯配置。
2. 论文与代码在 LR、训练步数、Virtual Ray 和 NeuraTV 参数上明显不一致。
3. `ep10000` 等文件名不能直接解释为 10,000 optimizer steps。
4. 当前没有 DTU runner。
5. LLFF 四个场景的路径问题已经修复，但正式运行器仍需在启动前执行全量路径预检。
6. NeRF 与 3DGS 的初始化和外部先验不完全公平。
7. EdgeNeRF 尚未正式同行评审。
8. 2026 年方法仍可能更新，论文定稿前需要再核对一次正式版本。

## 15. 完成条件

只有同时满足以下条件，才认为实验补全完成：

- P0–P4 中保留的运行全部完成。
- 每个表格单元都能追溯到 scene、seed、config 和 split。
- 论文主表只使用配置 A。
- 配置 B 始终单独标注。
- 外部方法数值逐项核对原论文。
- 定性图和定量表使用相同 split。
- DTU 全部使用 object mask。
- 三个随机种子的 mean ± std 可以从原始结果重算。
