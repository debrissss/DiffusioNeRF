# 里程碑 M1：DTU 少视图 alpha 通道管线与验证配方（2026-09-06）

本里程碑固化 scan63 上系统消融后的**全部正向成果**，含根因结论、最终配方、
复现命令与结果归档位置。后续所有 DTU 实验默认以此为基线。

**当前最优（masked PSNR，DNGaussian 协议，25 帧标准测试集）**

| 设置 | PSNR | SSIM | LPIPS-VGG | 权重 |
|---|---:|---:|---:|---|
| 3v（12k 步） | **20.07 (EMA)** | 0.915 | 0.097 | 退火5000 单点 |
| 9v（18k 步） | **26.27 (soup raw)** | 0.959 | 0.045 | 退火9000 三点 soup |

（均含 `--eval_bg_color gray` 泄漏匹配评测，见 1.3；白底口径为
3v 18.81 / 9v 24.11。）

## 1. 两个根因结论（按影响排序）

### 1.1 渲染合成背景色必须与数据虚空色一致（bg_color 问题）

`nerf/renderer.py` 的 `run()/run_cuda()` 中 `bg_color=None` 默认按**白色**
合成（torch-ngp 历史遗留）。合成公式：

```python
image = image + (1 - weights_sum) * bg_color
```

- **黑虚空场景（如 scan114）+ 白色合成**：优化器无法用 alpha→0 透出正确
  颜色，唯一解是全场堆密度（alpha→1, rgb→0）→ 浓雾、越训越差。
  修复 `--bg_color black` 后 24 视图全图 PSNR 14.7→24.5。
- **白背板场景（如 scan41/scan63）+ 黑色合成**：同样机制反向触发。
  白色合成对这类场景反而是对的。
- LLFF 不敏感的原因：无虚空，每条射线都命中真实内容，alpha≈1。
- **通用解是 alpha 通道管线（见 1.2）**；`--bg_color white|black`
  （内部属性 `bg_mode`）已参数化，供非 alpha 数据使用。

### 1.2 alpha 通道管线（本里程碑的主配方）

把官方物体掩码写进图像 alpha 通道（RGBA 数据），torch-ngp 原生路径自动生效：

- 训练：`train_step` C==4 分支用**逐像素随机透底色**，梯度上等价于
  "背景射线 alpha→0"，与背景真实颜色完全解耦；
- 评测：GT 与渲染均按白底合成，与 DNGaussian masked 协议口径一致。

scan63 3v 从"所有方案 9~11 dB 且发散"变为 17.55 dB 起步且稳定收敛。

### 1.3 评测背景色必须匹配训练透底期望（泄漏匹配，本里程碑最大单项收益）

C==4 随机透底训练下，背景射线与掩码边缘的训练期合成期望是 **r~U[0,1] 的
均值 0.5**；若评测用白底(1.0)合成，掩码内 alpha 略小于 1 的暗色表面会多出
`(1-0.5)·(1-α)` 的系统性偏亮。诊断链：pred~gt 亮度回归斜率 0.870、平均符号
误差 +0.027、corr(GT亮度, 误差)=-0.572。

修复：`--eval_bg_color gray`（评测背景=0.5，与训练期望一致；仅影响评测，
已加入 resume 忽略键）。**仅此一项**：

| 设置 | 白底评测 | 灰底(0.5)评测 | Δ |
|---|---:|---:|---:|
| 9v 单点（1b，15k） | 24.02 (EMA) | 26.11 (EMA) | +2.09 |
| 9v soup（18k×退火9000） | 24.11 (raw) | **26.27 (raw)** | +2.16 |
| 3v（12k，退火5000） | 18.81 (EMA) | 20.07 (EMA) | +1.26 |

SSIM/LPIPS 同步改善。**协议披露**：masked PSNR 使用与训练一致的
leak-matched 合成（0.5）；SSIM/LPIPS 仍按 DNGaussian 规程将 pred/GT 用
官方掩码合成到白底后计算。若论文需要与白底评测的工作严格对齐，应同时
报告两套数字。9v 25.57 目标由本项达成（26.27）。

## 2. 最终验证配方（Milestone1，已写入 run_DiffusioNeRF_DTU_NeurTV_Ray.sh）

```
数据:        RGBA（scripts/make_alpha_dtu.py 制作）
bg_color:    white（alpha 数据下训练用随机透底，评测白底）
退火:        --anneal_nearfar --anneal_nearfar_steps 5000 (perc 0.2, mid 0.5)
             9v 用 7000（3000→5000→7000 单调增益，9000 见顶回落），
             脚本经 ANNEAL_STEPS=<n> 环境变量选择
EMA:         --ema_decay 0.95（0.98 已测，9v 上 -0.5）
训练采样:    --num_steps 96 --upsample_steps 192
评测采样:    --eval_num_steps 128 --eval_upsample_steps 256（仅评测更细积分）
评测背景:    --eval_bg_color gray（泄漏匹配，见 1.3；alpha 数据必用）
损失组:      --diff_reg --loss_dist --loss_fg --use_depth
             --dist_lambda 2e-5 --fg_lambda 1e-4
             --patch_size 4 --depth_reg_lambda 0.1
             （loss_fg 已改为掩码感知：掩码内推 alpha→1，掩码外推 alpha→0）
通用:        --lr 0.01 --num_rays 4096 --downscale 1 --scale 0.33 --bound 2.0
             --fp16 --no_virtual_ray --seed 0
```

其余已实现但**未带来增益**的开关（保留备用，勿默认开启）：
`--appearance_embedding`、`--lit_ray_sampling`、`--pose_appearance`。

**9v 追加 soup 步骤**：训练时加 `--milestone_steps 11997 14004`（9 的倍数），
结束后 `python scripts/soup_checkpoints.py soup.pth <末点> <各里程碑点>`，
soup 相对该 run 末点 raw 典型 +0.3~0.4。

## 3. 消融记录（scan63，masked PSNR/SSIM/LPIPS-VGG，25 帧标准测试集）

### 3.1 优化迭代（3v / 9v）

| 轮次 | 改动 | 3v | 9v |
|---|---|---|---|
| 基线 | RGBA + 原始正则组 + 退火3000 + EMA0.95 + 采样64/128 | 17.55/.901/.109 (9k) | 21.64/.945/.057 (10k) |
| R1 | + loss_fg 掩码感知(A) + 评测密采样 128/256(B) | 18.08/.905/.106 | 22.60/.952/.051 |
| R2 | + 训练密采样 96/192(D)，步数 3v→12k、9v→15k | 18.51/.906/.103 | 22.81/.954/.050 |
| R3(M1) | + 退火 3000→5000 | **18.81(EMA)**/.909/.101 | 23.48(EMA)/.956/.047 |
| P1 | + 退火 5000→7000（仅 9v 受益） | 18.28（回退，弃用） | 24.02(EMA)/.956/.047 |
| M1.5 | + 评测泄漏匹配 `--eval_bg_color gray` | **20.07(EMA)**/.915/.097 | **26.27(soup raw)**/.959/.045 |

对 alpha 基线的累计提升：**3v +2.52 dB、9v +4.63 dB**（含泄漏匹配评测修正
3v +1.26 / 9v +2.16），SSIM/LPIPS 全程同步改善、从未受损。报告推荐 EMA
（9v soup 为 raw）。

**分视图数最优退火长度**：3v=5000（7000 时退火占预算比例过高、回退），
9v=7000（3000→5000→7000 单调 +0.67/+0.54）。脚本经
`ANNEAL_STEPS=<n>` 环境变量按视图数选择。

### 3.2b 收尾轮（2026-09-06 晚，9v）

| 手段 | masked PSNR | 判定 |
|---|---:|---|
| eval 密度截断 τ=2/5/10/20（仅评测清雾） | 24.02→23.00(τ5) | ❌ masked 负（物体半透明部分被误清；全图 +0.38） |
| 退火 9000 × 18k | 23.84 (EMA) | ❌ 退火长度在 7000 见顶 |
| EMA decay 0.98（9v） | 23.55 (EMA) | ❌ |
| 混响判定：2× 渲染降采样 vs 直出 | 19.10 vs 18.99（全图 +0.12） | ❌ mip 编码立项否决（<0.3 dB，不值 1~2 天） |
| **checkpoint soup**（11997/14004/17991 三点平均，18k×退火9000 run） | **24.11 (raw)** | ✅ **新冠军**（+0.39 over 该 run raw，+0.09 over 1b EMA） |

soup 工具：`scripts/soup_checkpoints.py out.pth ckpt1 ckpt2 [...]`（平均 model
权重，config 取最后一个）。要点：**必须用同一 run 内的轨迹点**（跨 run 哈希
桶不对齐会毁掉权重）；训练时用 `--milestone_steps` 保留 2~3 个中段点即可。

### 3.2c 负结果（勿重复测试，除非设定改变）

| 手段 | 结果 |
|---|---|
| 外观嵌入 `--appearance_embedding`（白底/alpha 两种机制下） | -0.1~-0.2 dB |
| num_rays 4096→8192（9v 20k） | -0.2 dB |
| 更长训练（3v 15k、9v 20k×退火5000） | 持平或回退（12k/15k 即最优） |
| **视角条件外观 MLP** `--pose_appearance`（σ60°, reg1e-3, 9v 15k） | 23.74 vs 24.02（-0.28；oracle 校正量仅 ±1.5%，天花板低） |
| **Hash 容量扩容** `--log2_hashmap_size 21 --num_levels 19 --hidden_dim_color 128`（9v 15k） | 23.29~23.41 vs 24.02（-0.6~0.7；9 视图下容量过拟合） |
| 退火 7000 应用于 3v | 18.28 vs 18.81（-0.53；按视图数选择退火长度） |
| 熵损失 / NeurTV / min_near=0.4 / distortion λ↑ / LR 快衰减 | 无增益或负面（黑底时代测定；换配方后未复测） |
| 邻近帧掩码+大膨胀近似训练掩码（scan41） | 未完成对比，scan41 待做 |

### 3.2d 误差诊断结论（9v 最优模型上的定量测量，指导后续方向）

| 测量 | 数值 | 结论 |
|---|---|---|
| 逐帧 PSNR vs 到最近训练视图方位角差 | 3v r=-0.377（分箱 19.5→17.3 跨 2.2 dB）；9v r=-0.204（24/25 帧在 30° 内） | 几何泛化误差存在但只解释 ~14%(3v)/~4%(9v) 方差 |
| 逐帧 PSNR vs GT 掩码内亮度 | 9v r=+0.726 | 最强单因子 |
| 预测亮度 ~ GT 亮度回归 | 斜率 0.870 | 曝光欠拟合：只跟上 87% 的逐帧曝光变化 |
| 符号曝光误差 corr(GT亮度, pred−gt) | -0.572；均值 +0.027（整体偏亮 2.7%） | 方向性欠拟合坐实 |
| oracle 逐帧增益 | 0.966 ± 0.015 | 校正上限仅 ±1.5% → 完美校正约 +0.3~0.7 dB |
| 联合拟合（方位角+亮度） | R²=0.53 | 剩余约一半方差为内容性（高光/材质），本表示内无低成本抓手 |

散点图与复算脚本要点：方位角取相机中心在轨道平面内极角
（SVD 求法向），圆周距离取最小差；亮度为掩码内均值。

### 3.3 参照：黑底路线（非 alpha 数据，scan114 验证）

`--bg_color black` + 退火 + 原始正则组：3v masked 11.7→13.7；
24 视图全图 14.7→24.5（masked 19.36/0.896/0.120 @10k）。
适用于暗虚空且不制作 alpha 数据的场景。

## 4. 复现命令

```bash
# 1) 制作 RGBA 数据（scan63 已有：data/DTU_standard/scan63_alpha）
python scripts/make_alpha_dtu.py data/DTU_standard/scan63

# 2a) 3v（12k 步；正式脚本当前为 30k 调度，12k 用 --stop_at_step 9000/12000）
python main_nerf.py data/DTU_standard/scan63_alpha \
  --workspace test_DTU/test_scan63_alpha/few_shot3/test_M1_seed0 \
  --split_file splits/dtu_3v/scan63_alpha.json --few_shot 3 --seed 0 \
  --ckpt scratch --fp16 --iters 15000 --stop_at_step 12000 \
  --eval_variants raw ema --num_steps 96 --upsample_steps 192 \
  --eval_num_steps 128 --eval_upsample_steps 256 --ema_decay 0.95 \
  --anneal_nearfar --anneal_nearfar_steps 5000 --no_virtual_ray \
  --bg_color white --diff_reg --loss_dist --loss_fg --use_depth \
  --diff_reg_start_iter 1000 --dist_lambda 2e-5 --fg_lambda 1e-4 \
  --patch_size 4 --depth_reg_lambda 0.1 --downscale 1 --scale 0.33 --bound 2.0

# 2b) 9v（15k 步；9 的倍数约束：14994）
#     --split_file splits/dtu_9v/scan63_alpha.json --few_shot 9
#     --iters 14994 --stop_at_step 14994 --checkpoint_interval_steps 900

# 3) masked 三项指标（DNGaussian 协议）
python evaluate_dtu_masked_results.py \
  --archive <workspace>/evaluation/step_XXXXXX \
  --summary /tmp/m1_masked.json
```

## 5. 本里程碑的归档位置

| 项 | 路径 |
|---|---|
| 3v 最优归档 | `test_DTU/test_scan63_alpha/few_shot3/test_S5k_e95/evaluation/step_012000_graybg/`（EMA 20.07） |
| 9v 最优归档 | `test_DTU/test_scan63_alpha/few_shot9/test_S2_18k_s9k/evaluation/step_017991_soup_gray/`（raw soup 26.27） |
| RGBA 数据 | `data/DTU_standard/scan63_alpha/` |
| 制作脚本 | `scripts/make_alpha_dtu.py` |
| 早期黑底路线文档 | `DTU_BLACKBG_FIX.md` |

## 6. 遗留事项

1. 9v 目标 25.57 已达成（soup+泄漏匹配 26.27）。曝光欠拟合的主因（评测/训练
   透底不一致）已修复；剩余误差以内容性高光/材质为主。若继续提指标，可选：
   1600×1200 数据重制 + 800×600 分辨率上探（约 1.5~2.5 天，渲染侧上界
   +0.12 已测，优化侧收益不确定），或更换表示（3DGS 类）。
2. scan41（白背板，官方掩码仅 25 评测帧）alpha 数据已可由
   `make_alpha_dtu.py` 生成，但训练帧只有近似掩码，效果待验证。
3. 正式脚本 `VALID_SCAN_NUMBERS` 不含 scan1；scan1/scan24 在
   `data/DTU_standard` 缺失，需从 CoR-GS 包补齐。
4. 官方全 49 帧掩码仅 scan24/40/55/63 可用（idrmasks 包的 mask/ 子目录）；
   其余 scan 只有 25 评测帧。
