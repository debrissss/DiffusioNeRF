# DiffusioNeRF 项目梳理

> 基于当前工作目录的静态检查整理，更新时间：2026-07-27。

## 1. 项目定位

本项目是论文 **DiffusioNeRF: A Combination of Regularization Techniques for Few-Shot Neural Radiance Field View Synthesis**（3DV 2024）的研究代码，目标是在少视图（few-shot）条件下训练 NeRF，并通过多种正则化方法改善新视角合成质量。

代码建立在 `torch-ngp` 风格的 Instant-NGP/NeRF 实现之上，主要使用：

- 多分辨率 Hash Grid 编码空间位置；
- 球谐编码观察方向；
- PyTorch MLP、Fully Fused MLP 或 tiny-cuda-nn 作为网络后端；
- PyTorch 或自定义 CUDA Ray Marching 进行体渲染；
- 深度平滑、失真、前景、频率、熵、KL/JSD、NeurTV 和对抗扰动等正则化手段；
- PSNR、SSIM、LPIPS 作为重建质量指标。

当前目录并非干净的论文原始实现：其中已有中文注释、headless 运行适配、虚拟射线增强、NeurTV、NLL 网络及多批实验产物，属于持续实验开发版本。

## 2. 总体运行链路

```text
run_DiffusioNeRF_LLFF.sh
        │
        ▼
main_nerf.py
  ├─ 解析训练/测试及各类正则化参数
  ├─ 创建 NeRFDataset
  ├─ 创建 NeRFNetwork
  └─ 创建 Trainer
        │
        ├─ 采样图像射线
        ├─ NeRFRenderer 采样空间点并体渲染
        ├─ NeRFNetwork 预测密度 σ 和颜色 RGB
        ├─ 计算重建损失及可选正则项
        ├─ 反向传播、EMA、学习率调度、保存 checkpoint
        └─ 验证/测试，输出图像、指标和 mesh
```

训练结束后，`main_nerf.py` 会继续尝试执行测试、评估和网格导出。设置 `NO_GUI=1` 可跳过 DearPyGui，在无桌面的服务器环境运行。

## 3. 核心目录与文件

| 路径 | 作用 |
| --- | --- |
| `main_nerf.py` | 主入口；定义 CLI 参数，组装数据集、模型、优化器和训练器，控制训练/测试流程 |
| `run_DiffusioNeRF_LLFF.sh` | LLFF 实验脚本；当前固定运行 `fern` 的 3-view 配置 |
| `nerf/provider.py` | 数据读取、相机坐标转换、训练/验证/测试划分、射线数据加载 |
| `nerf/network.py` | 默认 NeRF 网络；Hash Grid 密度分支 + 球谐方向编码颜色分支，支持 NLL 和频率掩码 |
| `nerf/network_ff.py` | Fully Fused MLP 后端 |
| `nerf/network_tcnn.py` | tiny-cuda-nn 后端 |
| `nerf/renderer.py` | PyTorch/CUDA 体渲染、分层采样、密度网格、near/far 退火及中间渲染量 |
| `nerf/utils.py` | `Trainer`、射线生成、训练/验证/测试、指标、checkpoint、mesh 导出及大部分正则损失编排 |
| `nerf/nll.py` | 混合分布及颜色/深度 NLL 相关实现 |
| `nerf/info/` | 近邻相机/像素采样、熵损失和 KL 平滑损失 |
| `nerf/adv/` | PGD、随机扰动、相机扰动和 AWP 对抗权重扰动 |
| `encoding.py` | 编码器工厂，连接频率、Hash Grid、球谐和 TCNN 编码 |
| `loss.py` | MAPE、Huber 和高效 Distortion Loss |
| `activation.py` | 截断指数激活 `trunc_exp` |
| `table.py` | 将实验指标写入表格 |
| `gridencoder/` | Hash Grid CUDA 扩展 |
| `freqencoder/` | 频率编码 CUDA 扩展 |
| `shencoder/` | 球谐编码 CUDA 扩展 |
| `raymarching/` | CUDA Ray Marching 扩展 |
| `ffmlp/` | Fully Fused MLP 扩展及其依赖 |
| `testing/` | CUDA 编码器、FFMLP 和 Ray Marching 的测试脚本 |
| `data/` | LLFF、NeRF Synthetic、DTU 数据及部分压缩包/预处理结果，约 4 GB |
| `test/` | 汇总指标及其他实验结果，约 1.1 GB |
| `test_LLFF/` | LLFF 多轮实验的日志、checkpoint、渲染结果和 mesh，约 7.2 GB |

`main_nerf-Copy1.py`、`nerf/renderer-原.py`、`nerf/utils-原.py` 和 `.ipynb_checkpoints/` 是副本或历史文件，不在当前主执行链路中。

## 4. 数据处理

### 4.1 支持格式

`NeRFDataset` 根据 JSON 文件自动识别两类数据：

- 存在 `transforms.json`：按 COLMAP/LLFF 风格处理；
- 存在 `transforms_train.json`：按 Blender/NeRF Synthetic 风格处理。

图像路径、相机外参和相机内参均从 `transforms*.json` 读取。相机矩阵会经过 `nerf_matrix_to_ngp()` 进行坐标轴变换，并应用 `--scale` 和 `--offset`。

### 4.2 当前 COLMAP/LLFF 划分

当前 `provider.py` 对所有 `transforms.json` 数据采用硬编码切分：

- 训练：`frames[5:]`
- 验证：`frames[:1]`
- 测试：`frames[1:5]`

`--few_shot` 会在后续数据加载逻辑中进一步控制训练视图数。由于基础切分写死在代码里，使用 DTU 或自定义 COLMAP 数据前应确认该规则是否符合预期。

### 4.3 当前已有数据

- LLFF：`fern`、`flower`、`fortress`、`horns`、`leaves`、`orchids`、`room`、`trex`
- NeRF Synthetic：当前可见 `lego`
- DTU：`scan1`、`scan24`、`scan40`、`scan41`、`scan55`、`scan82`、`scan100`、`scan114`

## 5. 模型与渲染

默认 `NeRFNetwork` 由两条主分支构成：

1. **密度分支**
   - 输入三维坐标；
   - 使用多层 Hash Grid 编码；
   - 经小型 MLP 输出密度 `sigma` 和几何特征；
   - 密度使用 `trunc_exp` 激活。

2. **颜色分支**
   - 输入观察方向和密度分支输出的几何特征；
   - 方向使用球谐编码；
   - 经 MLP 和 Sigmoid 输出 RGB。

可通过 `--ff` 或 `--tcnn` 切换高性能后端。`--cuda_ray` 使用占用密度网格和自定义 CUDA Ray Marching；否则使用分层采样与 PDF 重采样的 PyTorch 路径。

## 6. 损失与正则化

基础重建损失为逐像素 MSE。当前代码还可组合以下机制：

| 机制 | 主要参数 | 作用 |
| --- | --- | --- |
| Patch 深度平滑 | `--patch_size`、`--depth_reg_lambda`、`--rgb_weighting` | 约束相邻像素预测深度平滑 |
| Sample Space Annealing | `--anneal_nearfar` | 训练初期收缩采样区间，再逐渐恢复 near/far |
| 频率正则 | `--fre_nll_sigma`、`--fre_nll_color` | 随训练进度逐步开放编码特征 |
| NLL 网络 | `--use_nll_sigma`、`--use_nll_color` | 使用定制 `nllMLP` 替代普通 MLP |
| Diffusion 几何正则 | `--diff_reg` | 在指定训练阶段启用几何相关损失 |
| Distortion Loss | `--loss_dist`、`--dist_lambda` | 约束沿射线的权重分布 |
| Foreground Loss | `--loss_fg`、`--fg_lambda` | 促进射线累积权重接近 1 |
| Ray Entropy | `--entropy` | 降低射线采样分布熵 |
| Information Gain/KL | `--smoothing` | 约束邻近射线分布一致性 |
| NeurTV | `--neurtv`、`--neurtv_lambda`、`--neurtv_fd_epsilon` | 通过中心有限差分近似密度场空间梯度并施加 TV 约束 |
| 对抗训练 | `--adv ...` | 支持随机/PGD 输入扰动与 AWP 权重扰动 |
| 虚拟射线增强 | `--virtual_ray`、`--virtual_ray_k`、`--virtual_ray_jsd_th` | 构造穿过估计表面点的虚拟射线，用 JSD 筛选并约束深度 |

当前 LLFF 3-view NeurTV + 虚拟射线脚本还启用了 Diffusion 几何正则、
Distortion、Foreground、深度 Patch 平滑、NLL、颜色频率正则和近像素
KL 平滑。

## 7. 环境与运行方式

### 7.1 推荐环境

`environment.yml` 指向 CUDA Toolkit 11.3，并依赖 PyTorch、OpenCV、LPIPS、torch-ema、PyMCubes、DearPyGui、tiny-cuda-nn 等。CUDA 扩展首次导入时通常会通过 PyTorch JIT/Ninja 编译，因此需要：

- NVIDIA GPU 与兼容驱动；
- CUDA 工具链；
- C/C++ 编译器；
- Ninja；
- 与 CUDA 版本匹配的 PyTorch。

建议优先使用 Conda 文件创建环境：

```bash
conda env create -f environment.yml
conda activate torch-ngp
```

### 7.2 运行当前 LLFF 实验

```bash
NO_GUI=1 bash run_DiffusioNeRF_LLFF.sh
```

脚本当前固定参数为：

- 场景：`fern`（`i_dataset=3`）
- 少视图：3 views（`i_shot=3`）
- GPU：`CUDA_VISIBLE_DEVICES=0`
- 输出：`test_LLFF/test_fern/few_shot3/test_DiffusioNeRF`

如需换场景或视图数，需要修改脚本中的循环索引，而不是只修改数组。

### 7.3 LLFF 3-view NeurTV + 虚拟射线正式实验

```bash
./run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh
```

该脚本使用 `splits/llff_3v/` 中固定的标准切分、30,000 个优化步和
seed 0；默认以 `latest` 模式自动恢复。也可以只运行指定场景：

```bash
./run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh fern room
```

首次运行且只训练到 step 3000（学习率仍按 30,000 step 衰减）：

```bash
CKPT_MODE=scratch STOP_AT_STEP=3000 \
  ./run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh fern
```

从该 checkpoint 继续到完整 30,000 step：

```bash
CKPT_MODE=latest ./run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh fern
```

完整 checkpoint 原子写入模型、优化器、学习率调度器、AMP scaler、
EMA、global step、实验配置及 Python/NumPy/PyTorch/CUDA RNG 状态。
恢复时若训练配置发生变化，程序会拒绝继续，防止混跑。
checkpoint 轮换只允许删除当前 workspace 的文件，使用外部 checkpoint
做评估或诊断不会影响来源文件。FP16 梯度溢出时，默认最多降低四次
GradScaler scale 并重放同一随机批次；`--detect_anomaly` 仅用于显式调试，
正式训练默认关闭。

### 7.4 直接训练

```bash
NO_GUI=1 python main_nerf.py data/nerf_llff_data/fern \
  --workspace workspace/fern \
  --fp16 \
  --few_shot 3 \
  --scale 0.33 \
  --bound 3.0 \
  --iters 15000
```

`--ckpt` 默认为 `latest`，同一 workspace 内存在 checkpoint 时会自动续训；希望从头训练应显式传入：

```bash
--ckpt scratch
```

### 7.5 测试已有模型

```bash
NO_GUI=1 python main_nerf.py data/nerf_llff_data/fern \
  --test \
  --workspace <训练时的 workspace> \
  --ckpt latest \
  --scale 0.33 \
  --bound 3.0
```

测试模式会计算 PSNR、SSIM、LPIPS，并尝试输出测试结果和 mesh。训练和测试必须保持数据缩放、场景边界及网络结构参数一致。

## 8. 输出结构

一个典型 workspace 包含：

```text
workspace/
├── checkpoints/     # .pth 模型检查点
├── log_ngp.txt      # 训练与评估日志
├── run/             # TensorBoard 日志
├── validation/      # 验证渲染
├── results/         # 测试渲染
└── meshes/          # Marching Cubes 导出的网格
```

指标由 `PSNRMeter`、`SSIMMeter` 和 `LPIPSMeter` 统计。`--write_table` 可配合 `--implementation_name`、`--dataset_name` 写表，但当前主脚本是否在所有评估路径调用写表逻辑，仍需以实际日志为准。

## 9. 当前已知问题与风险

以下问题来自静态检查，建议在继续大规模实验前优先处理：

1. **依赖文件存在格式错误**
   - `requirements.txt` 第一行是 `#Aug.neurtvtorch-ema`，实际不会安装 `torch-ema`；
   - `requirements.txt` 与 `environment.yml` 的依赖集合并不完全一致，建议以实际可运行环境生成锁定版本；
   - README 提到的 `run_DiffusioNeRF_NS.sh`、`run_DTU.sh` 和 `scripts/llff2nerf.py` 当前目录不存在。

2. **虚拟射线增强计算成本较高**
   - 已提供 `--virtual_ray` / `--no_virtual_ray`、`--virtual_ray_start_iter`、`--virtual_ray_k`、`--virtual_ray_jsd_th` 和 `--virtual_ray_depth_lambda`；
   - 为兼容历史运行，虚拟射线默认仍开启并从第 1000 步开始；
   - 该流程会额外执行多次渲染，显著增加显存和计算量；消融实验应显式传入开关与全部参数。

3. **NeurTV 会增加密度场查询开销**
   - 类内调用已修复为 `self.compute_neurtv_loss(...)`；
   - `--neurtv_num_samples` 已加入主参数解析器，默认值为 4096；
   - 当前实现用三个坐标轴的中心有限差分近似密度梯度，避免 Hash Grid CUDA 编码器不支持的二阶梯度；
   - `--neurtv_fd_epsilon` 是相对于场景 `bound` 的差分步长比例，正式脚本显式设为 `0.01`；
   - 每次 NeurTV 计算需要额外执行六次密度查询，开启后仍应监控耗时和显存。

4. **参数解析已改为严格模式**
   - `--smoothing_step_size` 的默认值已修正为整数；
   - 主脚本使用 `parse_args()`，未知或拼错的参数会立即报错。

5. **数据集切分和图像处理仍含旧兼容分支**
   - `--split_file` 可按 `transforms.json` 帧序号显式指定 train/val/test，并校验数量、范围、重复、交集和缺失图像；
   - 未提供 `--split_file` 时，COLMAP 数据仍沿用旧的硬编码切分；
   - 灰度图分支固定遍历 `1536 × 2048`，对其他分辨率可能越界或处理错误；
   - LLFF 3-view 正式实验应使用 `splits/llff_3v/` 下的固定清单。

6. **历史副本和产物占用较大**
   - 当前目录约有 12 GB 以上数据和实验结果；
   - 多个 `-原`、`Copy1`、Notebook checkpoint 与大量变体 workspace 容易混淆当前有效实现；
   - 应通过 Git 提交和独立 workspace 名称追踪有效实现与实验来源。

7. **自动化测试覆盖仍偏底层**
   - `testing/` 主要验证 CUDA 扩展；
   - 暂未看到对数据加载、完整训练 step、正则化组合和端到端推理的自动化测试；
   - LLFF fern 3-view 的 NeurTV + 虚拟射线配置已经过真实 GPU 训练验证：按 30,000-step 学习率计划运行并在 global step 3000 停止，随后成功完成验证、标准测试渲染和网格导出；
   - 训练曾从 epoch 999 / global step 2997 的完整 checkpoint 恢复，优化器、调度器、AMP scaler 与 RNG 状态均成功加载。

## 10. 建议的后续整理顺序

1. 修复 `requirements.txt`，统一 Conda/Pip 依赖和已验证的 CUDA/PyTorch 版本。
2. 将虚拟射线的兼容默认值改为关闭，并让所有正式实验脚本继续显式声明开关。
3. 将 DTU/Blender 的切分策略也从 `provider.py` 硬编码中抽离为配置；LLFF 已支持显式 JSON split。
4. 将主脚本拆分为参数配置、训练、评估三个模块，避免 `utils.py` 继续膨胀。
5. 删除或迁移历史副本、`.ipynb_checkpoints` 和重复实验目录，并恢复 Git 版本管理。
6. 将已人工验证的数据加载、训练 step、checkpoint 恢复和测试渲染流程固化为自动化 smoke test。
7. 为每组实验保存完整命令、代码版本、随机种子和指标汇总，提升结果可复现性。

## 11. 快速接手索引

- 想改训练参数：先看 `main_nerf.py`
- 想改实验组合：看 `run_DiffusioNeRF_LLFF.sh`
- 想改数据划分或相机：看 `nerf/provider.py`
- 想改网络：看 `nerf/network.py`
- 想改采样或渲染：看 `nerf/renderer.py`
- 想改损失和训练逻辑：看 `nerf/utils.py` 的 `Trainer.train_step()`
- 想查对抗扰动：看 `nerf/adv/`
- 想查熵/KL 平滑：看 `nerf/info/`
- 想查 CUDA 算子：看 `gridencoder/`、`raymarching/`、`freqencoder/`、`shencoder/`、`ffmlp/`
