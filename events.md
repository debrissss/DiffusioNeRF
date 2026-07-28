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
