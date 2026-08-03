# 数据盘运行环境迁移

数据盘根目录的 `.runtime/miniconda3/` 保存完整的 Python 3.8.10、PyTorch 2.0.0+cu118 及已安装依赖，`.runtime/cache/` 保存当前 Torch 和 CUDA 扩展缓存。入口同时设置 `TORCH_EXTENSIONS_DIR`，避免 JIT 扩展回落到 `/root/.cache`。`scripts/runtime_env.sh` 会按项目自身位置选择该运行时，不依赖系统 Conda 或 `/root/miniconda3`。

复制 `/root/autodl-tmp/` 到目标机时应保持项目相对结构。目标机仍需兼容的 Linux x86_64、NVIDIA 驱动和 CUDA 驱动接口；内核、驱动和 GPU 固件不属于用户态文件迁移范围。

```bash
cd /root/autodl-tmp/DiffusioNeRF-main
source scripts/runtime_env.sh
"$PYTHON_BIN" scripts/runtime_check.py
CUDA_VISIBLE_DEVICES=0 ./run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh fern
```
