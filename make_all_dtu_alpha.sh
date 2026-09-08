#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCENES=(
    scan37
    scan40
    scan55
    scan63
    scan65
    scan69
    scan83
    scan97
    scan105
    scan106
    scan110
    scan114
    scan118
    scan122
)

for SCENE in "${SCENES[@]}"; do
    "$PYTHON_BIN" scripts/make_alpha_dtu.py "data/DTU/$SCENE" \
        --image-dir image \
        --mask-dir mask \
        --output-dir image_alpha
done
