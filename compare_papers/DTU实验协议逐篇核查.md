# DTU 实验协议逐篇核查与本地数据适配

核查日期：2026-09-08
核查范围：本目录现有 12 个 PDF；没有增加候选论文。

## 1. 先给结论

1. 这 12 篇论文的 DTU **新视角合成（NVS）主表**基本都指向 PixelNeRF/RegNeRF 形成的 15-scene、49-frame 稀疏视图协议，而不是 `data/DTU/` 中 IDR/UNISURF 的 15-scene 表面重建协议。两套 15 场景只重叠 `scan40, scan55, scan63, scan110, scan114`。
2. 标准 NVS 场景是 `8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114`。固定训练 ID（零基）是 3-view `[25,22,28]`，6-view 再加入 `[40,44,48]`，9-view 再加入 `[0,8,13]`；固定 25 个测试 ID 是 `[1,2,9,10,11,12,14,15,23,24,26,27,29,30,31,32,33,34,35,41,42,43,45,46,47]`。
3. `data/DTU/` 不能直接复现任何一篇论文的完整 DTU NVS 跨场景均值：缺少标准场景 `8,21,30,31,34,38,41,45,82,103`。它可以提供 5 个重叠场景的原图、mask、alpha 和相机；对 64 帧的 `scan110/114` 只取零基 `0..48` 即可。
4. `data/DTU_standard/` 已包含完整标准 15 场景、每场景 49 帧、固定 split 和 25 个测试 mask，是本项目复现这组 NVS 表格更合适的数据源。不同作者代码还需要 JSON、DVR/IDR NPZ 或 COLMAP 模型之间的格式转换。
5. 不能把所有表格视作完全同协议。最明显的例外是 ColNeRF：官方评测路径对 3/6/9-view 实际分别评 31/28/25 张（排除 15 张坏曝光和当前输入视图），不使用物体 mask，并在全图计算指标。多数 GS 方法固定评 25 张，并以白色合成 mask；MixNeRF 的 PSNR 只算前景、SSIM/LPIPS 算白底合成图。ReconFusion、CAT3D、FewViewGS、ICO-GS 没有公开足以复核全部指标细节的代码。

## 2. 证据标记和编号约定

- **A**：候选论文正文或官方补充材料明确给出。
- **B**：候选方法官方仓库的配置、loader、预处理或评测脚本明确给出。
- **C**：由候选论文引用的公开协议实现、代码执行路径、文件数量、索引关系或本地逐像素/相机核对推导；推断依据会写明。
- **未公开**：候选论文、官方补充材料和官方仓库均不能确定。复现建议会与作者事实分开写。

除非小节另有说明，本文的帧 ID 都是按文件名排序后的**零基索引**。DTU 原始文件 `rect_001_*` 对应零基 0，`rect_049_*` 对应零基 48。

## 3. 两类本地 DTU 数据和 49/64 帧关系

### 3.1 `data/DTU/` 实测清单

有效文件统计已排除 `._*`、`.DS_Store` 和目录项。所有 RGB、RGBA、mask 均为 1600×1200；`image/` 是 RGB，`image_alpha/` 是 RGBA。

| 场景 | RGB | RGBA | mask | `world_mat_i` | 帧池 |
|---|---:|---:|---:|---:|---|
| scan24, 37, 40, 55, 63, 65, 69 | 各 49 | 各 49 | 各 49 | 各 49 | 0–48 |
| scan83, 97, 105, 106, 110, 114, 118, 122 | 各 64 | 各 64 | 各 64 | 各 64 | 0–63 |

这是 IDR/UNISURF 常用的 15 场景集合 `[24,37,40,55,63,65,69,83,97,105,106,110,114,118,122]`，主要用于表面/几何评价。DiffusioNeRF 正文第 6 页同时提到 PixelNeRF 的 NVS 15 scans 和 UNISURF 等工作的几何 15 scans，不能把二者混为一组。

### 3.2 `data/DTU_standard/` 实测清单

标准集合 `[8,21,30,31,34,38,40,41,45,55,63,82,103,110,114]` 每场景均有 49 张 400×300 RGB、49 个 pose；每个场景的 25 个固定测试 ID 都有 mask。scan40/55/63 的 `masks/` 还保留了全部 49 张，其他 12 个标准场景各有 25 张；多出的训练/排除帧 mask 不得改变固定测试集合。目录中另有实验性 `scan24` 及 alpha 派生目录，但它们不在标准 15-scene manifest 中，计算论文均值时必须排除。

### 3.3 固定 split 的自动检查

- 3-view：`[25,22,28]`；6-view：`[25,22,28,40,44,48]`；9-view：`[25,22,28,40,44,48,0,8,13]`。
- 测试：`[1,2,9,10,11,12,14,15,23,24,26,27,29,30,31,32,33,34,35,41,42,43,45,46,47]`，共 25 张。
- 未训练且未测试：`[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]`，共 15 张；公开实现称其为 DTU bad views/不合适曝光。
- 检查结果：全部 ID 在 `0..48`；各列表内部无重复；`train ∩ test = ∅`；`3-view ⊂ 6-view ⊂ 9-view`；9 个训练 + 25 个测试 + 15 个排除恰好覆盖 49 帧。
- 该 benchmark 没有独立 per-scene validation。项目现有 split 把测试帧 1 另作训练过程监控，不能用于选模型，也不能从最终 25-view 测试均值中删掉。
- 本地 manifest 审计还发现：`splits/dtu_3v/` 额外存在 `scan1.json`、`scan63_alpha.json`、`scan63_alpha_approx.json`，`splits/dtu_9v/` 额外存在后两个 alpha manifest；`splits/dtu_6v/` 恰好是 15 个标准文件。这些额外文件不是论文标准场景，批量聚合必须用白名单 `[8,21,30,31,34,38,40,41,45,55,63,82,103,110,114]`，不能直接 glob 全目录。

### 3.4 49 帧是否为 64 帧的子集

结论是：**对本地数据以及这些论文使用的零基 0–48 协议，49 帧就是 64 帧采集的前 49 个 camera ID；64 帧场景的额外视图为 49–63。**证据如下。

- DTU 官方页面说明每个场景按 49 或 64 个位置采集，原始分辨率 1600×1200。【官方资料，访问 2026-09-08】
- MixNeRF 官方 loader 遍历原始 `rect_001...`，前 49 个位置读取 `r5000` 命名，后续位置改用 `r7000`；其标准 sparse split 明确只在 `np.arange(49)` 上构造。【B：`internal/datasets.py:1263-1304,1323-1326`】
- 对重叠场景 scan40/55/63/110/114，将 400×300 图像和 1600×1200 图像缩到 32×24 后做全候选归一化相关匹配，五个场景的 49 张全部得到唯一最佳匹配 `j→j`；对 scan110、scan114（本地为 64 帧）也完整成立。对角相关系数中位数约 0.999，差异来自缩放、压缩与背景预处理，不是不同相机。
- 对上述五场景分别分解相机投影矩阵，并用一个相似变换对齐到 `DTU_standard/transforms.json`；49 个相机中心最近邻全部为 `j→j`，最大对齐残差约 0.012（`DTU_standard` 的平均相机半径被归一到 4）。
- `data/DTU/` 任意 49 帧场景与任意 64 帧场景的 `world_mat_0..48` 在齐次尺度归一后逐元素完全相等；因此这些是共享的前 49 个标定相机，而不是仅凭文件名猜测。

上述结论只保证 camera ID 的对应关系。它不意味着 `data/DTU/` 的场景集合等于标准 NVS 15 场景，也不意味着不同论文的背景合成和指标实现相同。

### 3.5 `cameras.npz` 的解析

- `world_mat_i`：IDR 格式把 3×4 投影矩阵 `P = K[R|t]` 补成 4×4；它包含内参和 world-to-camera 外参，**不是纯外参**。【IDR 官方数据约定；本地矩阵实测】
- `scale_mat_i`：把归一化坐标映射回原始 DTU 世界坐标的平移+统一尺度矩阵。IDR 风格网络通常在规范化物体坐标中训练。
- 从规范化坐标恢复相机时，可令 `P_norm = (world_mat_i @ scale_mat_i)[:3,:4]`，用 OpenCV `decomposeProjectionMatrix` 得到 `K,R,C`，再构造 `c2w[:3,:3]=R.T`、`c2w[:3,3]=C[:3]/C[3]`。PixelNeRF/ColNeRF loader 采用等价的做法：先分解 `world_mat_i`，再用 `scale_mat_i` 的平移和对角尺度修正相机中心。【C/B：PixelNeRF `src/data/DVRDataset.py:157-181`；ColNeRF 对应文件同段】
- `world_mat_inv_i`、`scale_mat_inv_i` 是对应 4×4 矩阵的预存逆矩阵。`world_mat_inv_i` 因为连同 K 一起求逆，不能直接当 c2w；应分解投影矩阵。`scale_mat_inv_i` 用于从原始世界坐标映射到规范化坐标。
- 本地还有 `camera_mat_i` 及其逆矩阵；这是预存内参形式，是否使用取决于 loader。本文 12 篇的主流 raw-DTU/COLMAP 路径不要求直接读取它。

## 4. 逐篇核查

## 4.1 MixNeRF

**论文与范围。** MixNeRF: Modeling a Ray With Mixture Density for Novel View Synthesis From Sparse Inputs，CVPR 2023。本地 PDF 第 5、7 页；官方补充第 1–2 页。【A】

**场景与帧池。** 不是 DTU 全 124 场景；使用标准 NVS 15 场景 `8,21,30,31,34,38,40,41,45,55,63,82,103,110,114`。补充材料称采用指定 15 scenes，官方代码的 DTU near/far 字典逐项列出这 15 个 scan。【A/B：`render.py:57-72`】原始 loader 能读 49/65（代码注释如此，DTU 官方实际描述为 49/64），但 sparse split 只取前 49 帧。【B】

**训练、测试、验证和未使用帧。** 3/6/9-view 训练 ID 分别是 `[25,22,28]`、`[25,22,28,40,44,48]`、`[25,22,28,40,44,48,0,8,13]`；测试固定为 25 个 ID `[1,2,9,10,11,12,14,15,23,24,26,27,29,30,31,32,33,34,35,41,42,43,45,46,47]`；未使用 `[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]`；没有独立验证集。【B：`internal/datasets.py:1323-1326,1373-1376`】

**图像、背景和 mask。** 从官方 Rectified 目录读取 1600×1200 图和 `Calibration/cal18/pos_*.txt`，factor=4，得到约 400×300；无裁剪。配置采用黑背景训练。物体 mask 只为 25 个测试视图加载并按最近邻缩放；PSNR 仅在 `mask == 1` 的前景像素上算，SSIM 与 LPIPS 在预测和 GT 都合成为白背景后算。【B：`configs/dtu3.gin:2-14`，`internal/datasets.py:1268-1304,1379-1399`，`eval.py:203-217`】

**指标实现与聚合。** SSIM 是 Gaussian window 11、sigma 1.5 的 skimage 实现；LPIPS 是 AlexNet；每张测试图先算指标，再对 25 张取算术均值。跨场景表值按 15 个场景汇总，但仓库未提供一键跨场景聚合脚本。【B：`eval.py:74-105,203-217,240`；跨场景聚合细节未公开】

**相机和初始化。** 使用 DTU 官方投影矩阵，经 OpenCV 分解成 K 与 c2w；下采样时同步缩放 K。NeRF/mip-NeRF 参数随机初始化，不依赖 SfM 点云或外部深度。【B：`internal/datasets.py:1287-1305`】

**本地兼容性。** `data/DTU/` 只能复现 scan40/55/63/110/114 五个场景，不能复现 15-scene 均值；scan110/114 取 0–48。若直接跑作者代码，还需恢复其期望的 `Rectified/scan*/rect_*_3_r5000.png`、`Calibration/cal18/pos_*.txt` 和 `submission_data/idrmasks` 布局；更实际的方案是给 MixNeRF 写一个 `transforms.json`/NPZ adapter，使用 `data/DTU_standard/` 的完整 15 场景。【C】

## 4.2 DiffusioNeRF

**论文与范围。** DiffusioNeRF: Regularizing Neural Radiance Fields With Denoising Diffusion Models，CVPR 2023。本地 PDF 第 5–8 页。【A】

**场景与帧池。** NVS 主表使用 PixelNeRF test set 的 15 scans；标准集合可恢复为 `8,21,30,31,34,38,40,41,45,55,63,82,103,110,114`，每场景 49 帧。【A：第 6 页；C：PixelNeRF/RegNeRF 协议】同一页另述几何评价的 15 scans，指向 IDR/UNISURF 集合 `24,37,40,55,63,65,69,83,97,105,106,110,114,118,122`。这是另一项实验，不是 Table 1 的 DTU NVS 场景集。【A/C】Table 1 caption 写“averaged over all 8 scenes”，但正文明确 DTU 为 15 scans；该“8”与 LLFF 场景数一致，应视为排版冲突，不能据此把 DTU 改成 8 场景。【A/C】

**训练、测试、验证和未使用帧。** 论文只写 following PixelNeRF/相关协议，官方仓库没有 DTU loader 或 split。最强可恢复配置是 3/6/9-view `[25,22,28]`、`[25,22,28,40,44,48]`、`[25,22,28,40,44,48,0,8,13]`，25 个测试 ID `[1,2,9,10,11,12,14,15,23,24,26,27,29,30,31,32,33,34,35,41,42,43,45,46,47]`，15 个坏曝光排除 ID `[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]`，无独立验证集。【C】候选作者没有公开逐 ID 配置，因此这些 ID 不能标成 A/B。

**图像、背景和 mask。** DTU NVS 图像分辨率、factor、裁剪、训练背景、评价 mask 和阈值均未在候选论文或仓库公开。根据其比较对象与 PixelNeRF/RegNeRF 协议，400×300、物体 mask 评价是合理复现选择，但只能标 C，不能声称是作者设置。【未公开/C】

**指标实现与聚合。** 论文说明 PSNR、SSIM、LPIPS 逐测试视图计算并先报告每场景平均；最终跨场景聚合次序没有进一步说明。【A：第 6 页】官方仓库 `nerf/metrics.py:31-36` 实际 `lpips()` 调用 AlexNet，README 还明确说明 CVPR Table 1 误用了 AlexNet，后续 arXiv 才改 VGG。因此本目录 CVPR PDF 的数值应按 **LPIPS-Alex** 理解。SSIM 调用 skimage；DTU 是否 masked 未公开。【B】

**相机和初始化。** 论文的 NeRF 是 torch-ngp/Instant-NGP hash-grid，场景参数随机初始化；使用已知相机，不需要点云。DTU 相机导入和坐标转换没有发布；扩散先验在 Hypersim RGBD patches 上预训练，是外部生成先验而非 DTU 相机初始化。【A/B】

**本地兼容性。** `data/DTU/` 恰好适合该论文的 15-scene **几何**实验，但只覆盖 NVS 主表的 5/15 场景，不能复现主表均值。复现 NVS 应使用 `data/DTU_standard/`，并自行实现 Blender/NGP transforms loader、固定 split 和 mask evaluator；作者仓库无法直接消费当前两种 DTU 目录。【C】

## 4.3 CoR-GS

**论文与范围。** CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization，ECCV 2024。本地 PDF 第 9、12 页。【A】

**场景与帧池。** 使用 15 个标准 NVS 场景 `8,21,30,31,34,38,40,41,45,55,63,82,103,110,114`，每场景从 49 帧池划分。【B：`scripts/run_dtu.py:4`，`scene/dataset_readers.py:417-423`】

**训练、测试、验证和未使用帧。** 3/6/9-view 是完整顺序 `[25,22,28,40,44,48,0,8,13]` 的前 3/6/9 项；测试固定 25 个 ID `[1,2,9,10,11,12,14,15,23,24,26,27,29,30,31,32,33,34,35,41,42,43,45,46,47]`；未使用 15 个 ID `[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]`；无独立验证集。【B：`scene/dataset_readers.py:417-423`】

**图像、背景和 mask。** 原始 Rectified light-3/r5000 图进入作者提供的 COLMAP 数据布局，训练和评价均下采样 4 倍到约 400×300，不裁剪。训练命令启用 random background；物体 mask 在训练结束后复制进输出，仅用于评价。mask resize 后以 `==1` 取前景；预测和 GT 都合成为白底。【A：第 9 页；B：`scripts/run_dtu.sh:6-22`，`metrics_dtu.py:30-47`】

**指标实现与聚合。** PSNR 只算 mask 前景像素；主 SSIM 用仓库的 differentiable SSIM 在整张白底合成图上算（脚本还输出一个 skimage `SSIM_sk`，论文表对应主 `SSIM`）；LPIPS 用 VGG，在整张白底合成图上算。每图求值后在场景内算术均值；官方仓库未提供明确的 15-scene 聚合脚本，论文跨场景平均的具体脚本未公开。【B：`metrics_dtu.py:93-121`】

**相机和初始化。** loader 读取预制 COLMAP `cameras/images` 并按 image name 排序；已知标定 pose 不重新估计。初始化使用稀疏训练视图经 COLMAP PatchMatch Stereo + stereo fusion 的 `N_views/dense/fused.ply`。【A：第 9 页；B：`tools/colmap_dtu.py:142-201`，`scene/dataset_readers.py:372-423`】

**本地兼容性。** `data/DTU/` 只有 5 个目标场景，不能出完整均值；其 NPZ 也不是作者 loader 要求的 COLMAP 模型。可将 `data/DTU_standard/transforms.json` 转成 COLMAP `cameras.txt/images.txt`，保持文件名排序与零基 ID，再为每个 3/6/9 split 运行训练视图 stereo fusion；25 个 `masks/` 可直接适配指标脚本命名。【C】

## 4.4 ColNeRF

**论文与范围。** ColNeRF: Collaboration for Generalizable Sparse Input Neural Radiance Field，AAAI 2024。本地 PDF 第 4、6 页。【A】

**场景与帧池。** 这是 generalizable NeRF：按 PixelNeRF 协议在 DTU 训练 split 预训练，在 val split 评价。依赖协议有 88 个训练场景和 15 个评价场景；候选仓库未随代码发布 `new_train.lst/new_val.lst` 内容。评价 15 场景可恢复为 `8,21,30,31,34,38,40,41,45,55,63,82,103,110,114`，每场景 49 帧。【A：第 6 页；B：README 84；C：PixelNeRF 数据包/协议】88 个预训练 scan 的完整编号在候选官方材料中未公开。【未公开】

**训练、测试、验证和未使用帧。** 输入 ID 在作者 README 明列：3-view `[22,25,28]`，6-view `[22,25,28,40,44,48]`，9-view `[22,25,28,40,44,48,0,8,13]`；集合与标准协议相同，仅前三项顺序不同。【B：README 145-147】官方 render 路径默认排除当前 source views，metric 路径再排除 bad IDs `[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]`。因此实际测试数是 3-view 31 张、6-view 28 张、9-view 25 张，而不是三个档都固定 25 张；9-view 测试 ID 才恰好等于标准 25-ID 列表。per-scene 无额外 validation；跨场景的 DTU `val` split 被直接当 test。【B：`eval/eval.py:239-245`，`eval/calc_metrics.py:141-152,218-249`；README 84】未参与训练/测试的是 15 个 bad IDs；在 3/6-view 下，尚未加入训练的后续标准输入 ID 会被当测试图，而不是永久排除。【B/C】

**图像、背景和 mask。** 使用 PixelNeRF 的 `rs_dtu_4` DVR/IDR 格式，作者 eval 命令带 `--scale 0.25`，目标约 400×300；无裁剪证据。loader 会读取 mask 和 bbox，但 DTU 路径既不把 mask 合成到 RGB，也不在最终 metric 中使用 mask；PSNR/SSIM/LPIPS 都在原始全 RGB 图上计算。【B：README 75-79；`src/data/DVRDataset.py:112-181,210-269`；`eval/calc_metrics.py:186-249`】

**指标实现与聚合。** PSNR/SSIM 使用 skimage、data range 1；LPIPS 使用 VGG。先对每个场景所有实际渲染测试图取均值，再对评价场景等权平均。【B：`eval/calc_metrics.py:186-249,293-316`】

**相机和初始化。** 读取 DVR `cameras.npz`：分解 `world_mat_i` 得 K、R、camera center，用 `scale_mat_i` 修正归一化平移，并左右乘 DTU 坐标翻转矩阵 `diag(1,-1,-1,1)`。【B：`src/data/DVRDataset.py:157-181,204-208`】模型先在 88 DTU 场景预训练，再按论文设置对目标场景微调；没有 SfM 点云初始化。【A/B】

**本地兼容性。** `data/DTU/` 缺 10/15 评价场景，并且没有 88-scene 预训练集，不能直接复现论文结果。它的 NPZ 对 5 个重叠场景可被 loader 适配，但还要组织 `rs_dtu_4/DTU/new_val.lst`。完整复现需要 PixelNeRF DTU 数据包及训练 split；若只复现已有 checkpoint 的评价，可将 `data/DTU_standard/` 转成 DVR `image/mask/cameras.npz`，并严格保留作者“31/28/25 张”的评测路径。【C】

## 4.5 FewViewGS

**论文与范围。** FewViewGS: Gaussian Splatting with Few View Matching and Multi-stage Training，NeurIPS 2024。本地 PDF 第 7–8、13 页。【A】没有找到作者公开代码仓库，因此所有代码级细节均不能标 B。

**场景与帧池。** 正文明确选 15 scenes，但没有列 scan ID。结合其直接对齐 RegNeRF/FreeNeRF 的 DTU 表和固定 3/6/9 设置，最可复核的场景集合是 `8,21,30,31,34,38,40,41,45,55,63,82,103,110,114`，帧池 49。【A：第 7 页；C：协议与表格证据】作者未独立公开完整 scene list。【未公开】

**训练、测试、验证和未使用帧。** 论文只声明 3/6/9-view；没有逐 ID、测试 ID 或 validation 说明。为了复现表格，应冻结为训练 `[25,22,28]` / `[25,22,28,40,44,48]` / `[25,22,28,40,44,48,0,8,13]`，测试 25 ID `[1,2,9,10,11,12,14,15,23,24,26,27,29,30,31,32,33,34,35,41,42,43,45,46,47]`，排除 `[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]`，不设 validation；这些是协议推断而非作者明列。【未公开/C】

**图像、背景和 mask。** 正文明示 DTU 使用 1/4 resolution，评价时移除背景；从原始 1600×1200 推得 400×300。裁剪、训练背景、mask 来源、阈值、插值方式、PSNR 是否只在前景而 SSIM/LPIPS 用合成图，均未公开。【A：第 7 页；未公开】

**指标实现与聚合。** 正文只列 PSNR、SSIM、LPIPS；LPIPS backbone、SSIM 实现、逐图/逐场景聚合顺序均未公开。VGG16 是方法的语义特征提取器，不能据此断言 LPIPS 也用 VGG。【A：第 7 页；未公开】

**相机和初始化。** 已知相机 pose 的来源/坐标转换未公开。论文同时报告随机初始化和 SfM 初始化：主 Ours 行为 SfM 初始化，另有 `Ours (Rand. Init.)`；6/9-view 附表也分别给 random 和 SFM 版本。方法还用预训练 VGG16 提取语义特征，但没有外部深度监督。【A：第 7–8、13 页】

**本地兼容性。** `data/DTU/` 只能跑推断标准集合中的 5 场景，不能复现完整均值。由于官方实现未公开，即使使用 `data/DTU_standard/`，也必须自行确定 SfM 点云生成、背景 mask 和 evaluator；建议同时报告 random/SfM 初始化，且把 LPIPS backbone 和 mask 算法写入自己的实验协议。【C】

## 4.6 Binocular3DGS

**论文与范围。** Binocular-Guided 3D Gaussian Splatting with View Consistency for Sparse View Synthesis，NeurIPS 2024。本地 PDF 第 7–8、17–20 页。【A】

**场景与帧池。** 使用标准 15 场景 `8,21,30,31,34,38,40,41,45,55,63,82,103,110,114`，每场景 49 帧。【A：附录第 20 页；B：`script/run_dtu.py:6`】

**训练、测试、验证和未使用帧。** 训练顺序 `[25,22,28,40,44,48,0,8,13]`，各档取前 3/6/9；固定测试 25 ID `[1,2,9,10,11,12,14,15,23,24,26,27,29,30,31,32,33,34,35,41,42,43,45,46,47]`；其余 `[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]` 不参与；无独立验证集。【A：附录第 20 页；B：`scene/dataset_readers.py:165-170`】

**图像、背景和 mask。** 图像下采样 4 倍，300×400（H×W），无裁剪。主结果训练使用原始背景；附录 B.2 另做“训练时使用 mask”的消融，不能与主表混用。评价从 IDR masks 读取 25 张 mask，resize 后预测与 GT 合成为白底。【A：第 7、17、20 页；B：`metrics.py:68-105`】

**指标实现与聚合。** PSNR 使用 mask 参数只统计前景，SSIM 用 3DGS 自带实现算整张白底合成图，LPIPS-VGG 算整张白底合成图；先按 25 张图取场景均值。仓库没有跨 15 场景聚合代码，论文最终平均步骤未公开。【B：`metrics.py:68-117`】

**相机和初始化。** loader 读取预制 COLMAP K/pose，按文件名排序。初始化不是“无先验”：官方代码先用 PDCNet+ 稠密匹配并三角化为 dense point cloud；论文“不需要 external data as supervision”是指没有把外部深度当训练监督，不等于初始化不用预训练网络。训练 30k iterations。【A：第 8、19–20 页；B：`script/run_dtu.py:22-51`】

**本地兼容性。** `data/DTU/` 只有 5/15 场景且是 NPZ 格式，不能直接运行或复现均值。完整适配需要从 `data/DTU_standard/` 生成作者所需 COLMAP 模型、运行 PDCNet+ triangulation，并把 25 个测试 mask 放到其 `idrmasks` 路径；scan110/114 限定 0–48。【C】

## 4.7 ReconFusion

**论文与范围。** ReconFusion: 3D Reconstruction with Diffusion Priors，CVPR 2024。本地 PDF 第 5、7 页；CVF 官方补充第 3 页。【A】官方未发布训练/评测代码。

**场景与帧池。** 主文明确 LLFF/DTU 遵循 RegNeRF evaluation protocol，补充材料称使用 standard train/test splits，但都未列 DTU scan ID。可恢复为标准 15 场景 `8,21,30,31,34,38,40,41,45,55,63,82,103,110,114` 和 49 帧池。【A；C：RegNeRF 协议】

**训练、测试、验证和未使用帧。** 候选材料未逐项列 ID。最强协议恢复为训练 `[25,22,28]` / `[25,22,28,40,44,48]` / `[25,22,28,40,44,48,0,8,13]`，测试固定 25 ID `[1,2,9,10,11,12,14,15,23,24,26,27,29,30,31,32,33,34,35,41,42,43,45,46,47]`，排除 `[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]`，无 validation。【C；候选逐 ID 未公开】

**图像、背景和 mask。** diffusion conditioner 使用 512×512 输入；这不等于最终 DTU 指标分辨率。DTU evaluation 的 resize/crop 和 mask 细节没有公开代码。按 RegNeRF 协议应使用 1/4 图像（400×300）和 object mask，但只能标 C。主文“whole unmasked image”针对其重新训练的 SparseFusion baseline，不足以证明 ReconFusion 的 DTU evaluator 是全图指标。【A/C/未公开】

**指标实现与聚合。** 主表报告 PSNR/SSIM/LPIPS，但 SSIM 库、LPIPS backbone、mask 阈值、逐图/逐场景聚合顺序均未公开。【未公开】表中 RegNeRF 数值是作者重新评估版本，并非原 RegNeRF 表格逐字转录，进一步说明不可用其他论文 evaluator 代码冒充 ReconFusion evaluator。【C】

**相机和初始化。** 已知相机按补充材料把 focus point 移到原点，再缩放 camera positions 到 `[-1,1]^3` 以匹配 near=0.5。重建 backbone 是修改后的 Zip-NeRF，训练 1000 iterations，不使用 SfM 点云；外部先验是预训练并微调的 latent diffusion + PixelNeRF conditioner，扩散训练数据包含 Objaverse、CO3D、MVImgNet、RealEstate10K。【A：主文第 5 页、补充第 3 页】

**本地兼容性。** `data/DTU/` 缺 10 个标准场景，不能复现完整结果。即使使用 `data/DTU_standard/`，还要重现 focus-point/scale 变换、生成视图轨迹、内部 diffusion checkpoint 和未公开 evaluator；由于核心模型/代码未公开，只能进行近似复现，不能声称与论文完全一致。【C】

## 4.8 CAT3D

**论文与范围。** CAT3D: Create Anything in 3D with Multi-View Diffusion Models，NeurIPS 2024。本地 PDF 第 7–8、17–19 页。【A】项目页公开结果但没有可执行训练/评测代码。

**场景与帧池。** 论文明确 3/6/9-view 使用 ReconFusion 的 train/eval splits。恢复结果为标准 15 场景 `8,21,30,31,34,38,40,41,45,55,63,82,103,110,114`、每场景 49 帧。【A：第 7 页；C：ReconFusion→RegNeRF 协议链】候选论文没有自行列出 scan IDs。【未公开】

**训练、测试、验证和未使用帧。** 恢复后的训练 ID 是 `[25,22,28]` / `[25,22,28,40,44,48]` / `[25,22,28,40,44,48,0,8,13]`，测试固定 25 ID `[1,2,9,10,11,12,14,15,23,24,26,27,29,30,31,32,33,34,35,41,42,43,45,46,47]`，排除 `[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]`，没有 DTU per-scene validation。【C；候选逐 ID 未公开】

**图像、背景和 mask。** diffusion 网络处理 512×512；非方图先走 square padding，同时再跑 square crop，最后把 crop 中央与 padded 边缘合成。DTU 最终 benchmark render 的确切尺寸、crop 边界、object mask、背景颜色和阈值未公开。因为论文声称 train/eval split 复用 ReconFusion，并不自动证明 evaluator 的每个像素处理也被复用。【A：第 18–19 页；未公开】

**指标实现与聚合。** 主表报告 PSNR/SSIM/LPIPS；评价库、LPIPS backbone、mask 区域和平均顺序未公开。训练重建时额外使用 LPIPS loss（权重 0.25）也不能用来判断最终 LPIPS evaluator backbone。【A：第 7、19 页；未公开】

**相机和初始化。** DTU 先根据输入相机拟合 forward-facing circle path，缩放并沿 z 轴偏移，共生成 480 个视图；扩散模型以 3 个条件视图、每组 5 个目标视图生成图像。随后从 Zip-NeRF 参数初始化训练 1000 iterations，不依赖 SfM 点云。外部先验是多视图 latent diffusion，训练于 Objaverse、CO3D、RealEstate10K、MVImgNet。【A：第 7、17–19 页】

**本地兼容性。** `data/DTU/` 只能提供 5/15 标准场景；无法直接复现完整均值。`data/DTU_standard/` 能提供输入/测试相机与 RGB，但仍缺未公开的 CAT3D 权重、完整生成实现和 evaluator；只能近似适配，不能标成论文级复现。【C】

## 4.9 SE-GS

**论文与范围。** Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis，ICCV 2025。本地 PDF 第 5–6 页。【A】

**场景与帧池。** 使用标准 15 场景 `8,21,30,31,34,38,40,41,45,55,63,82,103,110,114`，每场景 49 帧；官方 run script 对 3/6/9 三档分别逐项列出全部场景。【B：`scripts/run_dtu.sh:5-54`】

**训练、测试、验证和未使用帧。** 训练顺序 `[25,22,28,40,44,48,0,8,13]` 的前 3/6/9 项；固定测试 25 ID `[1,2,9,10,11,12,14,15,23,24,26,27,29,30,31,32,33,34,35,41,42,43,45,46,47]`；排除 `[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]`；无独立 validation。【B：`scene/dataset_readers.py:496-502`】

**图像、背景和 mask。** 官方实现 factor=4，约 400×300，不裁剪。论文明确评价时 mask 背景；代码只在测试指标阶段加载 object masks、合成白底，没有用这些 object masks 训练。【A：第 5–6 页；B：`metrics_dtu.py:30-47` 和运行脚本顺序】

**指标实现与聚合。** PSNR 仅前景；SSIM 是 3DGS 自带整图白底版本；LPIPS-VGG 是整图白底版本；每场景先平均 25 张。仓库同时算 `SSIM_sk` 但主表使用 `SSIM`。15 场景最终聚合脚本未公开。【B：`metrics_dtu.py:93-125`】

**相机和初始化。** 读取预制 COLMAP 相机。原则上各比较方法使用相同 COLMAP 点云；3-view 的 scan8/40/110 和 6-view 的 scan21 因 COLMAP 失败，官方脚本显式切换随机点云，其他场景使用 fused stereo point cloud。SE-GS 方法本身不引入外部深度监督，训练 10k iterations。【A：第 5 页；B：`scripts/run_dtu.sh:5-54`，`scene/dataset_readers.py:451-502`】

**本地兼容性。** `data/DTU/` 不可直接复现 15-scene 均值，也缺 COLMAP/fused PLY 布局。使用 `data/DTU_standard/` 时需生成 COLMAP 模型和各 split 的 fused stereo PLY；对明确失败的场景必须复刻 random-init 分支，否则初始化协议已改变。【C】

## 4.10 ICO-GS

**论文与范围。** Intrinsic Geometry-Appearance Consistency Optimization for Sparse-View Gaussian Splatting，CVPR 2026。本地 PDF 第 6–8 页。【A】论文给出的项目 URL 在核查日返回 404，未找到作者公开仓库，因此没有 B 级代码证据。

**场景与帧池。** 正文称按 Binocular3DGS/CoR-GS 等设置使用 DTU 3/6/9-view，故可恢复为 15 场景 `8,21,30,31,34,38,40,41,45,55,63,82,103,110,114` 和 49 帧池。【A/C】作者未在论文中重新列出 scan ID。【未公开】

**训练、测试、验证和未使用帧。** 复现时最有依据的配置是训练 `[25,22,28]` / `[25,22,28,40,44,48]` / `[25,22,28,40,44,48,0,8,13]`，测试固定 25 ID `[1,2,9,10,11,12,14,15,23,24,26,27,29,30,31,32,33,34,35,41,42,43,45,46,47]`，排除 `[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]`，无独立 validation。【C】候选材料没有逐 ID 明示。【未公开】

**图像、背景和 mask。** 正文明示 DTU 下采样 4 倍，约 400×300；未说明裁剪。方法内部的 reliable-depth/validity mask 是训练正则 mask，不是 DTU ground-truth object mask。主表是否用 object mask、背景合成颜色、阈值和插值未公开；与 Binocular/CoR 数值并列不足以升为 A/B。【A：第 6 页；未公开】

**指标实现与聚合。** 论文报告三次不同种子结果的平均，但没有公开 PSNR/SSIM/LPIPS 的实现、LPIPS backbone、mask 区域，以及“先视图、再场景、再种子”的确切聚合次序。【A：第 6 页；未公开】

**相机和初始化。** 构建于 BinocularGS，LLFF/DTU 使用 dense point cloud initialization，训练 30k；论文没有说明 dense 点由哪个 checkpoint/命令生成，但框架来源指向 PDCNet+ 路径。方法另使用冻结的预训练特征提取器做多视图一致性，不把外部深度图作为监督。【A/C】

**本地兼容性。** `data/DTU/` 只有 5/15 场景且没有作者要求的 dense/COLMAP 初始化，不能直接复现。`data/DTU_standard/` 可作为 RGB/pose/test-mask 基础，但因作者代码和 evaluator 未公开，只能按 Binocular3DGS 公开路径近似实现，并应明确标注“协议复刻”而非官方复现。【C】

## 4.11 TWINGS

**论文与范围。** TWINGS: Thin Plate Splines Warp-aligned Initialization for Sparse-View Gaussian Splatting，CVPR 2026。本地 PDF 第 5、7 页。【A】

**场景与帧池。** 官方脚本列出标准 15 场景 `8,21,30,31,34,38,40,41,45,55,63,82,103,110,114`，每场景 49 帧。【B：`scripts/dtu.sh:3-5` 及 loader】

**训练、测试、验证和未使用帧。** 训练顺序 `[25,22,28,40,44,48,0,8,13]` 的前 3/6/9；测试固定 25 ID `[1,2,9,10,11,12,14,15,23,24,26,27,29,30,31,32,33,34,35,41,42,43,45,46,47]`；排除 `[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]`；无 validation。【B：`scene/dataset_readers.py:384-390`】

**图像、背景和 mask。** DTU factor=4，约 400×300，无裁剪。object masks 只在评测阶段读取；预测和 GT 合成白底，训练仍使用原始图像背景。【A：第 5 页；B：`metrics_dtu.py:30-47`】

**指标实现与聚合。** PSNR 仅前景；SSIM 使用 3DGS 自带实现算白底整图；LPIPS-VGG 算白底整图。每图先算、场景内平均；`metrics_means.py` 再收集每个 scan 的 JSON 并用 `np.mean` 对场景等权平均。脚本还保存 `SSIM_sk`，主表采用 `SSIM`。【B：`metrics_dtu.py:89-120`，`metrics_means.py:24-75`】

**相机和初始化。** 相机来自预制 COLMAP 模型。初始点先由训练视图获得深度/匹配并反投影，再以多视图可靠重建点作为控制点做 3D thin-plate-spline warp 和邻域采样；这是外部 matching/depth 初始化先验。论文说明对若干 3-view COLMAP 失败的 baseline 使用 random init，但 TWINGS 自身使用 TWINGS-Init。【A：第 5、7 页；B：loader 与 `TWINGS-Init/`】

**本地兼容性。** `data/DTU/` 只有 5/15 场景且格式不符。完整复现要把 `data/DTU_standard/` 转成 COLMAP 格式，运行 TWINGS-Init 的匹配、反投影、TPS 和采样，再按固定 25 masks 评价；仅将 IDR NPZ 相机直接喂给代码不可行。【C】

## 4.12 StereoGS

**论文与范围。** StereoGS: Sparse-View 3D Gaussian Splatting via Stereo Priors，ECCV 2026 accepted，当前文件为 arXiv v2。本地 PDF 第 8–10、27–28 页。【A】

**场景与帧池。** 官方 run 配置列出标准 15 场景 `8,21,30,31,34,38,40,41,45,55,63,82,103,110,114`，每场景 49 帧。【B：`run_scripts/DTU/run_dtu.py:17-24`】

**训练、测试、验证和未使用帧。** 训练顺序 `[25,22,28,40,44,48,0,8,13]` 的前 3/6/9；测试固定 25 ID `[1,2,9,10,11,12,14,15,23,24,26,27,29,30,31,32,33,34,35,41,42,43,45,46,47]`；排除 `[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]`；无 validation。【B：`scene/dataset_readers.py:434-440`】

**图像、背景和 mask。** factor=4，约 400×300，无裁剪。训练时的 stereo validity mask 由左右一致性、异常 disparity 和 DTU 黑背景强度阈值组成；它不是 benchmark object mask。最终指标另从 IDR masks 读取 25 张 object mask，预测和 GT 合成为白底。【A：第 27–28 页；B：`metrics.py:70-108`】背景强度阈值符号在论文给出，但本地 PDF 这两页未给具体数值；官方代码配置中应以运行参数为准，若未设置则记录为未公开。

**指标实现与聚合。** PSNR 通过 mask 参数只算前景；SSIM 用 3DGS 自带实现算白底整图；LPIPS-VGG 算白底整图。先平均 25 测试图，再由官方 run script 收集 15 个 `results.json` 做等权算术平均。【B：`metrics.py:91-125`，`run_scripts/DTU/run_dtu.py:297-350`】

**相机和初始化。** 已知相机来自 COLMAP。MVSAnywhere 为每个训练视图预测深度，经跨视图几何过滤后融合为 dense point cloud；训练到 20k 时启用 FoundationStereo 正则，总计 30k。主结果还分不带 dropout 的 Ours 与固定 0.3 dropout 的 Ours*，比较时不能混行。【A：第 8、27 页；B：`run_scripts/DTU/run_dtu.py:177-205`】

**本地兼容性。** `data/DTU/` 缺 10/15 场景且没有 COLMAP/MVSAnywhere 初始化产物，不能直接复现。应从 `data/DTU_standard/` 构造 COLMAP K/pose，安装并固定 MVSAnywhere 与 FoundationStereo checkpoint，生成 dense init，再用其 25 masks evaluator；scan110/114 只取前 49 帧。【C】

## 5. 官方仓库快照与关键证据

仓库均克隆到 `/tmp/dtu_protocol_audit.2Yw4oD/`，没有写入项目。核查日期均为 2026-09-08。

| 方法 | 官方仓库 | commit SHA | 关键文件 |
|---|---|---|---|
| MixNeRF | https://github.com/shawn615/MixNeRF | `1523089781465a848881a52cb12fc7d475c5ef78` | `configs/dtu*.gin`; `internal/datasets.py:1254-1400`; `eval.py:74-240`; `render.py:57-72` |
| DiffusioNeRF | https://github.com/nianticlabs/diffusionerf | `4cc3c4b06269b93f641aa5e1445cc3ba01926f02` | `nerf/metrics.py:31-36`; README 的 LPIPS 勘误；仓库无 DTU loader |
| CoR-GS | https://github.com/jiaw-z/CoR-GS | `1db13a4da6ef6dd61da674d87d76cc342572badf` | `scene/dataset_readers.py:372-423`; `metrics_dtu.py:30-125`; `tools/colmap_dtu.py:131-201`; `scripts/run_dtu.py:4` |
| ColNeRF | https://github.com/eezkni/ColNeRF | `9327c97ebad9ced5649802aa33a76a93fe39bd38` | `README.md:55-84,138-170`; `src/data/DVRDataset.py:112-269`; `eval/calc_metrics.py:141-316` |
| Binocular3DGS | https://github.com/hanl2010/Binocular3DGS | `2b3b7a4ff757af946233516b28b6eb37b8666fb2` | `scene/dataset_readers.py:137-175`; `metrics.py:68-117`; `script/run_dtu.py:6-51` |
| SE-GS | https://github.com/sailor-z/SE-GS | `7bf0287d6d4340b1ff3401565a59e293d24e8e5e` | `scripts/run_dtu.sh:5-54`; `scene/dataset_readers.py:451-502`; `metrics_dtu.py:30-125` |
| TWINGS | https://github.com/sandokim/TWINGS-official | `1585e8933516a6abadf6383346e887746e1128e4` | `scripts/dtu.sh`; `scene/dataset_readers.py:335-390`; `metrics_dtu.py:30-120`; `metrics_means.py:24-75` |
| StereoGS | https://github.com/StringerYwh00/StereoGS | `2238f559ce6da2c8f9b0e98110eedc7e53ea952b` | `scene/dataset_readers.py:404-440`; `metrics.py:70-125`; `run_scripts/DTU/run_dtu.py:17-24,177-230,297-350` |
| PixelNeRF（协议依赖，仅用于解释 ColNeRF） | https://github.com/sxyu/pixel-nerf | `91a044bdd62aebe0ed3a5685ca37cb8a9dc8e8ee` | `README.md:91-104,198-214`; `src/data/DVRDataset.py:80-208`; `eval/calc_metrics.py:141-316` |

FewViewGS、ReconFusion、CAT3D 未找到作者公开的可执行训练/评测仓库；ICO-GS 论文项目链接在核查日不可用。这里的“未找到”只描述核查日官方公开状态，不证明未来不会发布。

## 6. 最终简表

表中“标准固定”均在本行完整指：训练 `[25,22,28]` / `[25,22,28,40,44,48]` / `[25,22,28,40,44,48,0,8,13]`，测试 `[1,2,9,10,11,12,14,15,23,24,26,27,29,30,31,32,33,34,35,41,42,43,45,46,47]`。场景“标准15”完整指 `[8,21,30,31,34,38,40,41,45,55,63,82,103,110,114]`；这些完整定义写在表内说明，避免依赖其他论文小节。

| 论文 | 场景/帧池 | 训练与测试 | mask/分辨率 | 指标实现 | 初始化 | 最高证据 | `data/DTU/` 可复现性 |
|---|---|---|---|---|---|---|---|
| MixNeRF | 标准15；49 | 标准固定；无 val；15 bad 排除 | 物体 mask；400×300 | 前景 PSNR；白底 SSIM；Alex LPIPS；图均值 | 随机 NeRF | A+B | 仅 5/15，不可复现总均值 |
| DiffusioNeRF | NVS 标准15/49；几何另用 IDR15 | 标准固定为 C；候选未列 ID | 推断 400×300+mask，细节未公开 | CVPR 为 Alex LPIPS；区域未公开 | 随机 hash-grid；Hypersim DDM prior | A+B+C | NVS 仅 5/15；几何 15/15 |
| CoR-GS | 标准15；49 | 标准固定；无 val；15 bad 排除 | 评价 mask；400×300 | 前景 PSNR；白底 SSIM/VGG LPIPS | fused stereo PLY | A+B | 仅 5/15，且需 COLMAP/PLY |
| ColNeRF | 88 train scenes + 标准15 eval；49 | 3/6/9 实评 31/28/25；val split 当 test | 不用 mask；约 400×300 | 全图 skimage PSNR/SSIM；VGG LPIPS；图均值→场景均值 | DTU 预训练+微调 | A+B+C | 仅 5/15，且缺 88-scene 预训练集 |
| FewViewGS | 15 scenes；推断标准15/49 | 标准固定为 C；逐 ID 未公开 | 评价去背景；1/4 | backbone/区域/聚合未公开 | 主行 SfM；另报 random；VGG16 特征 | A+C | 仅 5/15；官方代码未公开 |
| Binocular3DGS | 标准15；49 | 标准固定；无 val；15 bad 排除 | 主训练不 mask；评价 mask；400×300 | 前景 PSNR；白底 SSIM/VGG LPIPS | PDCNet+ dense init | A+B | 仅 5/15，且需 COLMAP/PDCNet+ |
| ReconFusion | 标准15；49（恢复） | 标准固定为 C；逐 ID 未公开 | conditioner 512；评价 resize/mask 未公开 | backbone/区域/聚合未公开 | Zip-NeRF；diffusion+PixelNeRF prior | A+C | 仅 5/15；核心实现未公开 |
| CAT3D | 标准15；49（恢复） | 标准固定为 C；逐 ID 未公开 | diffusion 512 square；评价尺寸/mask 未公开 | backbone/区域/聚合未公开 | 480 生成视图→Zip-NeRF | A+C | 仅 5/15；核心实现未公开 |
| SE-GS | 标准15；49 | 标准固定；无 val；15 bad 排除 | 评价 mask；400×300 | 前景 PSNR；白底 SSIM/VGG LPIPS | fused stereo；指定失败场景 random | A+B | 仅 5/15，且需 COLMAP/PLY |
| ICO-GS | 标准15；49（恢复） | 标准固定为 C；逐 ID 未公开 | 1/4；评价 object mask 未公开 | 三种子均值；其余未公开 | dense init；预训练特征 | A+C | 仅 5/15；官方代码未公开 |
| TWINGS | 标准15；49 | 标准固定；无 val；15 bad 排除 | 评价 mask；400×300 | 前景 PSNR；白底 SSIM/VGG LPIPS；图→场景均值 | matching+TPS dense init | A+B | 仅 5/15，且需 COLMAP/TWINGS-Init |
| StereoGS | 标准15；49 | 标准固定；无 val；15 bad 排除 | 训练 validity mask；评价 object mask；400×300 | 前景 PSNR；白底 SSIM/VGG LPIPS；图→场景均值 | MVSAnywhere init + FoundationStereo | A+B | 仅 5/15，且需 COLMAP/两个先验模型 |

## 7. 建议冻结的本项目复现实验协议

这部分是**复现建议，不是作者设置补写**：使用 `data/DTU_standard/` 的标准 15 场景与固定 3/6/9 train IDs；所有档固定评 25 个 test IDs；不设用于选 checkpoint 的 validation；训练图保留原始背景；评价 PSNR 只算 `mask==1` 前景，SSIM 和 LPIPS-VGG 在白底合成整图计算；先对每个场景 25 张取均值，再对 15 场景等权平均。这样可与 CoR-GS、SE-GS、TWINGS、StereoGS、Binocular3DGS 的公开实现最接近。ColNeRF 应另列为“作者原协议”，保留其全图和 31/28/25 测试图设置，不能混入上述同协议排名。

## 8. 官方网页证据

- DTU 官方 MVS 数据页：https://roboimagedata.compute.dtu.dk/?page_id=36
- IDR 相机数据约定：https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md
- ReconFusion CVF 页面与官方 supplement：https://openaccess.thecvf.com/content/CVPR2024/html/Wu_ReconFusion_3D_Reconstruction_with_Diffusion_Priors_CVPR_2024_paper.html
- CAT3D 官方项目页：https://cat3d.github.io/
- 其余官方仓库 URL 和固定 commit 已列于第 5 节。
