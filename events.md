# Debug 事件记录

## 2026-07-28：LLFF 3-view 续训时 NLL 反向传播出现 NaN

### 1. 实验上下文

- 数据集：LLFF `fern`
- 划分：3-view，训练帧 `[1, 10, 19]`，测试帧 `[0, 8, 16]`
- 方法：DiffusioNeRF + NeurTV + virtual-ray augmentation
- 精度：PyTorch AMP FP16
- 学习率计划：`--iters 30000`
- 故障前完整 checkpoint：`epoch=5308`、`global_step=15924`
- 故障 epoch：5309。每个 epoch 有 3 个 batch，因此第三个 batch 对应
  `global_step=15927`
- 故障时学习率：约 `0.002945`

### 2. 报错信息

核心异常如下：

```text
==> Start Training Epoch 5309, lr=0.002946 ...
loss=0.0002 (0.0002), lr=0.002945: 67% 2/3

UserWarning: Error detected in MulBackward0.

nerf/network.py, line 336, in forward
    weight = self.normalization(
        weight,
        torch.nn.functional.softplus(self.nll_bound_per_layer[i])
    )

nerf/network.py, line 319, in normalization
    return w * scale[:, None]

RuntimeError: Function 'MulBackward0' returned nan values in its 1th output.
```

主要调用链：

```text
main_nerf.py
  -> Trainer.train
  -> Trainer.train_one_epoch
  -> Trainer.train_step
  -> NeRFRenderer.run
  -> NeRFNetwork.density
  -> nllMLP.forward
  -> nllMLP.normalization
  -> GradScaler.scale(loss).backward()
```

### 3. 排查结论

故障不是模型权重或 checkpoint 已经出现 NaN，而是一次可恢复的 FP16
梯度溢出被全局 autograd anomaly detection 转换成了致命异常。

排查证据：

1. `ngp_ep5308.pth` 中模型参数和 Adam 状态全部为有限值。
2. NLL 网络的 `nll_bound_per_layer` 全部有限；权重行绝对值和的最小值约
   为 `5.14`，不存在零行或非有限权重。
3. checkpoint 中 AMP GradScaler 的 scale 已增长到 `8,388,608`。
4. 从 `epoch=5308 / global_step=15924` 恢复 RNG、模型、优化器、调度器和
   scaler 后，可以在 `global_step=15927` 确定性复现同一错误。
5. 禁止导入模块时全局启用 anomaly detection 后，同一批次不再导致进程
   退出；GradScaler 检测到溢出并将 scale 从 `8,388,608` 回退到
   `4,194,304`，模型仍保持有限。

根因链如下：

```text
GradScaler 长期未检测到溢出
  -> scale 增长到 8,388,608
  -> step 15927 的 NLL 反向梯度在 FP16 缩放后溢出
  -> nerf/adv/pgd.py 在导入时全局开启 anomaly detection
  -> 即使当前 adv=[]，该全局设置仍然生效
  -> GradScaler 尚未来得及回退 scale，MulBackward0 先抛出 RuntimeError
```

`nllMLP.normalization()` 是 anomaly mode 定位到的反向节点，但当前
checkpoint 中其输入均为有限值，因此它不是模型已经损坏的证据。

### 4. 本次 NaN Debug 的代码修改

#### `main_nerf.py`

- 新增 `--amp_max_retries`，默认最多重试 4 次 AMP 溢出批次。
- 新增 `--detect_anomaly`；anomaly detection 默认关闭，只在明确调试时
  开启。
- 校验 `--amp_max_retries` 必须为非负整数。
- 参数解析完成后，通过
  `torch.autograd.set_detect_anomaly(opt.detect_anomaly)` 统一设置调试
  模式，不再依赖被导入模块的全局副作用。

#### `nerf/adv/pgd.py`

- 删除模块导入阶段的
  `torch.autograd.set_detect_anomaly(True)`。
- PGD 模块仍可正常使用，但不会在 `adv=[]` 时改变整个训练进程的
  autograd 行为。

#### `nerf/adv/camera.py`

- 删除模块导入阶段的
  `torch.autograd.set_detect_anomaly(True)`，避免相机扰动工具产生相同的
  全局副作用。

#### `nerf/network.py`

- 在 `nllMLP.normalization()` 中将权重行和限制为至少
  `torch.finfo(w.dtype).eps`。
- 该修改用于防御未来可能出现的全零权重行；当前 checkpoint 的最小行和
  约为 `5.14`，所以该保护不会改变当前模型的正常前向结果。

#### `nerf/utils.py`

- 新增 `_optimize_with_amp_retry()`：
  - 检查前向 loss 是否为有限值；
  - 记录 backward 前后的 GradScaler scale；
  - 当 scale 降低时，判定该次 `optimizer.step()` 因梯度溢出被跳过；
  - 恢复该批次开始前的 Python、NumPy、PyTorch 和 CUDA RNG；
  - 清空无效梯度，以更低 scale 重放同一随机批次；
  - 只有优化器成功更新后才返回训练循环，因此学习率调度器不会为空更新
    前进一步；
  - 超过 `--amp_max_retries` 后抛出包含 global step 和最终 scale 的明确
    异常。
- GUI 和 headless 两条训练路径统一使用上述 AMP 重放逻辑。
- 抽取 `_capture_rng_state()` 和 `_restore_rng_state()`，同时服务于 AMP
  批次重放和 checkpoint 恢复。
- checkpoint 原子写入模型、优化器、调度器、scaler、EMA、实验配置、
  global step 和全部 RNG 状态。
- 恢复时严格检查实验配置，并在恢复优化器、调度器或 scaler 失败时立即
  报错，避免静默使用不完整状态继续训练。

#### `run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh`

- 正式 LLFF 3-view 启动命令显式传入 `--amp_max_retries 4`。
- 不传 `--detect_anomaly`，保证正式 FP16 训练使用 GradScaler 的正常
  溢出处理机制。

#### `PROJECT_OVERVIEW.md`

- 补充 FP16 梯度溢出重放、显式 anomaly 调试开关和 checkpoint 安全恢复
  说明。

### 5. 调试过程中发现的 checkpoint 轮换问题

使用外部 checkpoint 在临时 workspace 做隔离复现时，又发现了一个独立的
resume 安全问题：

1. checkpoint 的 `stats["checkpoints"]` 保存了来源 workspace 的绝对路径；
2. 新 Trainer 直接继承该列表；
3. 临时 workspace 保存新 checkpoint 时，轮换逻辑可能删除列表中的来源
   checkpoint。

`nerf/utils.py` 对此增加了以下保护：

- `_is_managed_checkpoint_path()` 只允许删除当前 Trainer 的
  `checkpoints/` 目录中的文件；
- 加载 checkpoint 后过滤外部轮换历史；
- 如果加载的是当前 workspace 内的 checkpoint，则把该文件重新纳入当前
  workspace 的轮换列表；
- 删除前再次检查真实父目录，拒绝删除任何外部 checkpoint。

隔离测试触发旧轮换缺陷后，原 `ngp_ep5308.pth` 被旧逻辑移除。已经将隔离
诊断产生的健康后继 checkpoint 按字节一致的副本恢复到正式目录：

```text
test_LLFF/test_fern/few_shot3/
  test_DiffusioNeRF_NeurTV_Ray_30k_seed0/
  checkpoints/ngp_ep5309.pth
```

恢复状态：

```text
epoch                 = 5309
global_step           = 15927
lr_scheduler.last_epoch = 15927
GradScaler scale      = 4,194,304
model tensors         = finite
optimizer tensors     = finite
training horizon      = 30,000 steps
```

该 checkpoint 按 PyTorch 标准 GradScaler 行为跳过了发生溢出的 step 15927
参数更新；从后续 step 开始，新实现会降低 scale 并重放批次，不再把空更新
作为成功的训练更新。

### 6. 验证

完成了以下检查：

- 从故障前状态确定性复现 `global_step=15927` 的原错误。
- 在 anomaly detection 关闭时验证 GradScaler 能正确回退 scale。
- 在真实 NeRF checkpoint 上将 scaler 人为提高到 `268,435,456`：
  - 检测到梯度溢出；
  - scale 自动回退到 `134,217,728`；
  - 使用同一 RNG 状态重放批次；
  - epoch 正常完成；
  - scheduler 与 global step 对齐；
  - 本 epoch 的 3 次 Adam 更新全部成功；
  - 模型和优化器状态全部为有限值。
- 验证外部 checkpoint 在临时 workspace 保存和轮换后仍然存在且大小
  不变。
- `python -m py_compile` 通过。
- `bash -n run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh` 通过。
- `git diff --check` 通过。

### 7. 同一提交中包含的相关实验基础修复

本次提交还收口了此前尚未提交、且该实验运行依赖的修改：

#### `main_nerf.py`

- 增加独立的 `--stop_at_step`，将实际暂停点与 30,000-step 学习率计划
  分离。
- 增加 NeurTV、virtual-ray 和显式 split 的完整 CLI 参数。
- 改用严格 `parse_args()`，未知参数直接报错。
- 校验目标 global step、checkpoint global step 和 epoch 必须按
  `steps_per_epoch` 对齐。

#### `nerf/provider.py`

- 支持由 `--split_file` 指定 train/val/test 帧 ID。
- 校验场景、帧数、ID 类型、范围、重复、训练/测试交集和缺失/损坏图像。
- 显式 split 启用时，校验 `--few_shot` 与训练帧数量一致。
- 在数据集中保留真实 `frame_ids`，避免二次采样后丢失来源索引。

#### `nerf/renderer.py`

- 删除 renderer 内重复且未完成的 NeurTV 二阶梯度路径。
- NeurTV 统一由 Trainer 计算，避免重复调用、未定义变量和断开的梯度图。

#### `nerf/utils.py`

- NeurTV 改为三个坐标轴的中心有限差分，只需要 Hash Grid 编码器支持的
  一阶参数梯度。
- virtual-ray 的启动步、射线数量、JSD 阈值和深度权重全部读取显式参数。
- checkpoint 使用临时文件加 `os.replace()` 原子写入。
- `latest` checkpoint 按修改时间选择，不依赖文件名字符串排序。

#### `splits/llff_3v/`

- 新增 LLFF 八个标准场景的固定 3-view manifest。
- 测试帧遵循 `frame_id % 8 == 0`。
- 训练帧从非测试轨迹中近似等间隔选取 3 张。
- `README.md` 说明索引基准、验证用途和 loader 校验规则。

#### `run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh`

- 新增八场景统一启动脚本。
- 固定 30,000-step 学习率计划、seed 0、标准 split、NeurTV、virtual-ray
  和已有 DiffusioNeRF 正则化配置。
- 支持 `CKPT_MODE=scratch|latest` 和可选 `STOP_AT_STEP`。
- scratch 模式检测到已有 checkpoint 时拒绝覆盖。

### 8. 安全续训命令

```bash
unset STOP_AT_STEP
CKPT_MODE=latest ./run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh fern
```

该命令会从正式目录中最新的完整 checkpoint 恢复，并继续使用
30,000-step 学习率计划。

---

## 2026-07-28：修复 LLFF Raw/EMA 评测语义与正式结果归档

### 1. 事件背景

`fern` 的 30,000-step 训练结束后，对最终评测路径进行审计时发现，已有
日志指标、渲染图片和 checkpoint 之间没有完整且一致的来源标识：

1. `Trainer.evaluate_one_epoch()` 在存在 EMA 时会临时将 EMA 参数复制到
   模型，因此最终测试日志中的 PSNR、LPIPS 和 SSIM 来自 EMA。
2. `Trainer.test()` 不切换 EMA，因此旧 `results/` 目录中的 RGB 和 depth
   来自原始 Raw 参数。
3. 最终日志指标和 `results/` 图片虽然使用同一 checkpoint 和测试帧，但
   实际对应不同参数版本。
4. `results/` 同时包含 `ngp_ep1000_*` 和 `ngp_ep10000_*`，即 3,000-step
   与 30,000-step 结果混在同一目录。
5. 最终测试调用 `evaluate()` 时仍将图片写入 `validation/`，验证集与测试
   集结果没有目录级隔离。
6. 最终指标只存在于 `log_ngp.txt`，没有结构化文件记录权重版本、测试帧、
   checkpoint SHA256、global step、split、代码版本和逐图指标。
7. 原 `--test` 路径构造 Trainer 时没有 EMA 对象，无法恢复完整 checkpoint
   中的 EMA；同时可能错误尝试恢复只属于训练的 optimizer 和 scheduler。
8. 完整 checkpoint 在最终测试前保存，因此 checkpoint 自身不包含最终
   测试指标。

另外，当前固定 split 中 `val_ids=[0]`，而 `test_ids=[0, 8, 16]`。验证帧
0 与测试集重叠。验证不参与反向传播，所以固定 30,000-step 模型权重不受
影响；但是按验证结果保存的 `checkpoints/ngp.pth` 存在测试集选择泄漏，
不能作为正式论文 checkpoint。

本次修复规定正式评测只使用固定训练步数的完整 checkpoint：

```text
checkpoints/ngp_ep10000.pth
epoch       = 10000
global_step = 30000
```

Raw 和 EMA 必须在同一 checkpoint、同一显式测试 split 和相同渲染配置下
分别评测并完整归档。

### 2. 本次代码修改

#### `nerf/utils.py`

- 为 Trainer 增加三种明确的 checkpoint 加载模式：
  - `resume`：恢复模型、EMA、optimizer、scheduler、GradScaler、RNG 和
    checkpoint 轮换状态，用于继续训练；
  - `evaluation`：只恢复 Raw 模型、EMA、epoch、global step、stats 和
    checkpoint 配置，不恢复或修改训练状态；
  - `model`：仅加载模型参数，保留旧 model-only 用途。
- 记录 `last_checkpoint_path`、checkpoint 原始配置和 EMA 是否确实来自
  checkpoint。
- 增加异常安全的 Raw/EMA 参数上下文：
  - Raw 直接评测；
  - EMA 评测前保存 Raw 参数并复制 EMA shadow 参数；
  - 无论评测成功或异常都恢复 Raw；
  - 每个 variant 结束后对全部模型参数和 buffer 重新计算 SHA256，确认
    Raw 模型没有被 EMA 评测或渲染过程修改。
- 新增正式双模型评测与归档入口 `evaluate_variants_archive()`：
  - 强制使用带 GT 的单进程评测；
  - 校验 checkpoint 路径、global step、split 文件和有序 frame IDs；
  - 使用 `model.eval()`、`torch.no_grad()`、`perturb=False`、白色背景和
    当前实验的 FP16 autocast；
  - Raw 与 EMA 使用相同 loader 顺序和相同 GT；
  - 分别计算逐图和场景平均 PSNR、LPIPS-Alex、SSIM；
  - 指标从浮点预测直接计算，不从 8-bit PNG 反算；
  - 保存预测 RGB、GT、float32 depth 和 depth preview；
  - 文件名使用 transforms.json 原始 frame ID。
- 正式归档通过 staging 目录构建，全部完成后使用 `os.replace()` 原子发布。
- `COMPLETE` 最后生成，并包含其他 30 个归档文件的 SHA256 清单。
- 已存在的归档只有同时满足以下条件才允许幂等复用：
  - checkpoint SHA256 相同；
  - global step、Raw/EMA variants 相同；
  - split SHA256 和有序 frame IDs 相同；
  - 评测代码 SHA256 相同；
  - 所有归档文件存在且 SHA256 校验通过。
- manifest 记录：
  - checkpoint 路径、大小、SHA256、epoch、global step；
  - Raw/EMA 模型状态 SHA256；
  - EMA decay 和更新次数；
  - dataset、split、transforms SHA256；
  - 完整训练配置和评测配置；
  - Python、PyTorch、CUDA、cuDNN、GPU；
  - Git commit、dirty 状态和评测代码 SHA256。
- 正式评测输出目录必须是当前 workspace 的子目录，避免用户参数将归档
  写入或覆盖 workspace 根目录及外部路径。

#### `nerf/provider.py`

- dataloader batch 新增 `frame_id`。
- `frame_id` 是 transforms.json 顺序下的原始 ID，而不是 split 内部的
  局部序号。
- 正式评测会校验实际 loader 顺序与 split 文件中的 test IDs 完全一致。

#### `main_nerf.py`

- 新增正式评测参数：
  - `--eval_variants raw ema`
  - `--eval_split test|val`
  - `--eval_expected_step`
  - `--eval_output_dir`
  - `--eval_overwrite`
- `--test` 改为 evaluation-only checkpoint 加载和正式归档，不再恢复
  optimizer、scheduler、GradScaler 或 RNG。
- 正常训练结束后从刚保存的完整 checkpoint 重新以 evaluation 模式加载，
  保证指标对应磁盘中的确切 checkpoint 字节，而不是可变的内存状态。
- 正式评测失败会直接返回非零状态，不再将失败降级为 Warning 后继续退出
  为成功。
- 不再把最终正式测试图片继续写入旧 `results/` 或 `validation/`。

#### `run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh`

- 每个场景显式传入 Raw/EMA、test split 和目标 global step。
- 新增 `EVAL_ONLY=1`，用于只读评测已有 checkpoint。
- 新增 `EVAL_OVERWRITE=1`，重新评测时保留旧归档并原子替换权威目录。
- `EVAL_ONLY=1` 禁止与 `CKPT_MODE=scratch` 组合。
- 命令成功后必须存在对应 step 的 `COMPLETE`，否则 runner 返回失败。
- 已完成训练在 `global_step=30000` 恢复时不会多训练一步；如果正式归档
  完整且哈希匹配，则验证后跳过重复渲染。

#### `scripts/aggregate_llff_evaluations.py`

- 新增 LLFF 八场景正式汇总脚本。
- 默认要求
  `{fern, flower, fortress, horns, leaves, orchids, room, trex}` 全部存在
  完整 Raw/EMA 归档。
- 任一场景缺失时默认返回状态码 2，拒绝生成伪完整的论文汇总。
- `--allow-partial` 只用于诊断当前完成进度。
- 汇总前验证 `COMPLETE`、artifact SHA256、checkpoint step、Raw/EMA 标签
  和 frame IDs。
- 输出：
  - `summary_long.csv`
  - `summary.json`
  - `summary.md`
  - `COMPLETE`
- 论文平均采用八场景无权宏平均，不使用按测试图数量加权的平均。
- Raw/EMA 始终保存为两套完整结果；按指标择优时只允许在八场景宏平均后
  为每个指标选择一个全局 variant，禁止先逐场景挑最好结果再平均。

#### `EVALUATION_ARCHIVE.md`

- 新增 Markdown 使用说明。
- 记录训练/恢复、evaluation-only、overwrite、部分汇总和正式八场景汇总
  命令。
- 说明归档目录、完整性校验和论文指标选择规则。

### 3. 正式归档结构

Fern 权威结果位于：

```text
test_LLFF/test_fern/few_shot3/
  test_DiffusioNeRF_NeurTV_Ray_30k_seed0/
  evaluation/step_030000/
```

目录结构：

```text
step_030000/
├── COMPLETE
├── manifest.json
├── comparison.json
├── eval.log
├── split_snapshot.json
├── raw/
│   ├── metrics.json
│   └── frames/
│       ├── frame_0000_rgb.png
│       ├── frame_0000_gt.png
│       ├── frame_0000_depth.npy
│       ├── frame_0000_depth_preview.png
│       └── ...
└── ema/
    ├── metrics.json
    └── frames/
```

Fern 测试帧严格为：

```text
[0, 8, 16]
```

Raw 和 EMA 每套包含 3 张 RGB、3 张 GT、3 个 float32 depth 和 3 张 depth
preview。

### 4. Fern 30,000-step 正式结果

| 模型参数 | PSNR ↑ | LPIPS-Alex ↓ | SSIM ↑ |
|---|---:|---:|---:|
| Raw | **22.241452** | **0.182661** | **0.715599** |
| EMA | 22.239312 | 0.183012 | 0.715567 |

精确差值 `Raw - EMA`：

```text
PSNR        = +0.002140204
LPIPS-Alex  = -0.000351379
SSIM        = +0.000031610
```

Fern 三项指标均由 Raw 略优，但差异非常小。按论文常见显示精度取整后几乎
相同，因此归档保留两套完整结果和来源标签，不隐藏 EMA 结果。

EMA 元数据：

```text
decay       = 0.95
num_updates = 10000
```

此实现每个 epoch 更新一次 EMA，因此 10,000 epochs 对应 10,000 次 EMA
更新。

### 5. Checkpoint 完整性

正式 checkpoint：

```text
test_LLFF/test_fern/few_shot3/
  test_DiffusioNeRF_NeurTV_Ray_30k_seed0/
  checkpoints/ngp_ep10000.pth
```

检查结果：

```text
epoch        = 10000
global_step  = 30000
size         = 260661737 bytes
SHA256       = 1fe9f7e984909bef02487a87417260fb7c2e7e5bf4a62157cdd0d7180b0841f4
mtime        = 2026-07-28 17:59:00.848082632 +0800
```

修改和全部评测验证结束后，checkpoint 的大小、SHA256 和 mtime 均未变化，
说明正式评测没有重写或修改已有训练结果。

### 6. 可靠性验证

完成并通过以下验证：

1. `python -m py_compile`：
   - `main_nerf.py`
   - `nerf/provider.py`
   - `nerf/utils.py`
   - `scripts/aggregate_llff_evaluations.py`
2. `bash -n run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh`。
3. `git diff --check`。
4. 使用 Fern 真实 30,000-step checkpoint 完成 Raw/EMA 端到端评测。
5. Raw 和 EMA 指标精确复现此前独立诊断值。
6. loader 和 manifest 均确认测试帧顺序为 `[0, 8, 16]`。
7. 30 个归档 artifact 的 SHA256 全部通过。
8. Raw 与 EMA 中对应 GT PNG 的 SHA256 完全相同。
9. 所有 depth：
   - shape 为 `(378, 504)`；
   - dtype 为 `float32`；
   - 全部数值有限。
10. Raw/EMA 评测结束后模型状态 SHA256 恢复为评测前 Raw SHA256。
11. 对同一完整归档再次运行：
    - evaluation-only 正常加载；
    - 完整性校验通过；
    - 跳过 Raw/EMA 重复渲染；
    - 返回状态码 0。
12. 使用错误目标 `global_step=3000` 评测 30,000-step checkpoint：
    - 抛出明确 `ValueError`；
    - 返回状态码 1；
    - 不产生错误 step 的正式归档。
13. 八场景尚不完整时运行正式汇总：
    - 明确列出其余 7 个缺失场景；
    - 返回状态码 2；
    - 不将 Fern 单场景结果伪装为八场景结果。
14. 运行隔离的 scratch 3-step 全链路烟雾测试：
    - 从随机初始化训练 1 epoch；
    - 完成 3 次 optimizer 更新；
    - 保存 `epoch=1 / global_step=3` 完整 checkpoint；
    - 以 evaluation-only 模式重新加载；
    - 完成 Raw/EMA 双评测；
    - 生成 30 个带哈希的正式 artifact；
    - 全部断言通过。
15. 评测结束后 GPU 无残留训练或测试进程。

scratch 烟雾测试使用独立系统临时目录；验证成功后已移入系统回收站，不影响
正式 workspace，且仍可恢复。

### 7. 当前汇总状态

目前只有 `fern` 完成 NeurTV + virtual-ray、3-view、seed 0、30,000-step
正式双模型归档。

诊断性部分汇总位于：

```text
test_LLFF/aggregates/
  llff_3v_neurtv_ray_seed0_step_030000/
```

该汇总明确标记其余 7 个场景为 incomplete。只有八场景全部完成后，才能
不带 `--allow-partial` 生成正式论文汇总。

多次 `EVAL_OVERWRITE=1` 可靠性验证保留了若干
`step_030000.replaced.<timestamp>` 历史归档。权威结果始终只有：

```text
evaluation/step_030000/
```

### 8. 后续命令

只读复核 Fern 正式归档：

```bash
EVAL_ONLY=1 CKPT_MODE=latest \
  ./run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh fern
```

重新生成 Fern 正式归档，并保留旧归档：

```bash
EVAL_ONLY=1 EVAL_OVERWRITE=1 CKPT_MODE=latest \
  ./run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh fern
```

从 checkpoint 恢复或启动其余七个场景：

```bash
CKPT_MODE=latest \
  ./run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh \
  room orchids trex flower fortress horns leaves
```

八场景全部完成后生成正式汇总：

```bash
./scripts/aggregate_llff_evaluations.py
```

### 9. Git 与 provenance 状态

本次修改在写入本事件记录时尚未提交 Git。Fern 当前 manifest 记录：

```text
base commit = 91cb7c0d4501d2ea4d70719cb3ca4bbf72b58b51
git dirty   = true
```

manifest 同时记录了 `main_nerf.py`、`nerf/provider.py` 和 `nerf/utils.py`
的精确 SHA256，因此当前未提交状态下的评测代码仍可审计。正式启动剩余
七场景前，应先提交本次代码和文档修改，再使用 `EVAL_OVERWRITE=1` 重新
生成一次 Fern 归档，使 manifest 记录干净的正式 commit。
