#!/usr/bin/env python
"""Offline validation for a copied DiffusioNeRF runtime."""
import importlib
import json
import os
import platform
import sys

import torch


def main():
    cuda_available = bool(torch.cuda.is_available())
    report = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "project_root": os.environ.get("PROJECT_ROOT"),
    }
    modules = ["numpy", "cv2", "imageio", "skimage", "lpips"]
    for name in modules:
        importlib.import_module(name)
    if cuda_available:
        importlib.import_module("raymarching")
        report["raymarching"] = "ok"
    else:
        report["raymarching"] = "not_checked_host_cuda_unavailable"
    print(json.dumps(report, indent=2, sort_keys=True))
    if not cuda_available:
        raise SystemExit("CUDA is unavailable; check the target NVIDIA driver/runtime")


if __name__ == "__main__":
    main()
