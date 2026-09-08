# DTU 3-view 失败根因与修复记录（2026-09-05）

## 根因

`nerf/renderer.py` 的 `run()`/`run_cuda()` 中，`bg_color=None` 时默认按
**白色背景**合成（`bg_color = 1`，torch-ngp 历史遗留），训练与评估均如此：

```python
image[:num_rays] = image[:num_rays] + (1 - weights_sum[:num_rays]) * bg_color
```

- LLFF/NeRF-Synthetic 的每条射线都有真实内容（或白色背景 GT），alpha→1 无害；
- DTU rect 图背景为黑色。训练时黑背景射线要拟合"黑"，在白色合成下优化器的
  唯一解是 `rgb→0 且 alpha→1`，即**在全场堆积密度**——这正是少视图 DTU
  训练全场浓雾、指标 10~11 dB 且越训越差的直接原因。

修复：新增 `--black_bg`（renderer 的 `default_bg=0`，`eval_step`/train 渲染
同步使用黑底）。`run_DiffusioNeRF_DTU_NeurTV_Ray.sh` 已更新为验证配方。

## 修复效果（scan114，DNGaussian masked 协议）

| 配置 | PSNR | SSIM | LPIPS(vgg) |
|---|---:|---:|---:|
| 修复前：全套正则 30k（正式基线） | 10.30 | 0.646 | 0.290 |
| 修复后：3v + 退火 + 黑底 @10k | **13.68** | **0.711** | **0.251** |
| 修复后：24 视图 + 黑底 @10k | **19.36** | **0.896** | **0.120** |
| 旧目标 `scan114-10k可`（22 帧子集） | 19.95 | 0.816 | 0.222 |
| 旧目标 `scan114-30k`（22 帧子集） | 22.06 | 0.863 | 0.180 |

- 3-view 标准协议（相邻训练视图 + 360° 测试）：+3.4 dB，SSIM/LPIPS 同步改善；
  10k 步左右为最佳点，30k 略有过训练（13.2）。
- 24 视图：SSIM/LPIPS **超过**旧目标，PSNR 持平；视觉对比见
  `/tmp/final_cmp_frame2.png`、`/tmp/final_cmp_frame12.png`。
- 关于旧目标：`test/our/DTU/*-10k可` 每场景 43 张渲染、全 360° 质量均匀，
  属于**多视图训练**产物，不是 3-view 少样本结果（`9v/` 目录可佐证）。

## 探针结论（对 scan114 的系统消融）

| 手段 | 效果 |
|---|---|
| `--black_bg`（核心修复） | 24v：14.7→24.5（全图）；3v：+3.4 dB |
| `--anneal_nearfar`（3000 步退火） | 白底下 +2.2 dB；黑底下保留 |
| 外观嵌入 `--appearance_embedding`（新增） | 无增益（DTU 曝光差异 ~1.8×，但不是主导因素） |
| `--lit_ray_sampling`（新增，亮度采样） | 无增益 |
| 熵损失 / NeurTV / min_near=0.4 / distortion λ↑ | 无增益或负面 |
| LR 快速衰减（10k horizon） | ~+1 dB（白底下），黑底下不需要 |

## 其它修复

- `train_step` patch reshape：熵/平滑射线拼接后深度/权重未按 num_rays 切片，
  会导致 reshape 崩溃或维度错误（已修）。
- `get_rays` 的 `ray_inds` 分支不返回 `inds`（已修）。
- `N_entropy=100` 与 `patch_size=4` 不整除导致熵射线采样崩溃（已修）。

## 遗留问题

1. 3-view 标准协议 13.7 vs 公开 SOTA ~19.5（DivCon-NeRF/SE-GS，依赖深度
   先验或专门架构）。本环境无网络，Depth-Anything 等单目深度先验无法安装；
   这是继续提升 3-view 的主要路径。
2. `data/DTU_standard/scan1`、`scan24` 不存在（scan1 只在 `data/DTU` 有
   legacy 版本）；两者也缺评测掩码，masked 指标暂不可算。
3. scan1（白墙背景、复杂几何）3-view 结果 11.0，需单独调优。
4. 正式脚本 `VALID_SCAN_NUMBERS` 不含 scan1，本次用 main_nerf.py 直跑。
