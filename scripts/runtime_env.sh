#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ENV_DIR="$(cd -- "${PROJECT_ROOT}/../.runtime/miniconda3" && pwd)"
if [[ ! -x "${RUNTIME_ENV_DIR}/bin/python" ]]; then
    echo "[runtime] missing data-disk Python: ${RUNTIME_ENV_DIR}/bin/python" >&2
    return 1 2>/dev/null || exit 1
fi
export PROJECT_ROOT RUNTIME_ENV_DIR
export PYTHON_BIN="${RUNTIME_ENV_DIR}/bin/python"
export PYTHONNOUSERSITE=1
export PATH="${RUNTIME_ENV_DIR}/bin:${PATH}"
export XDG_CACHE_HOME="$(cd -- "${PROJECT_ROOT}/../.runtime/cache" && pwd)"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
fi
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
