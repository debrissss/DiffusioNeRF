#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLFF_VIEWS=9 exec "$REPO_DIR/eval_DiffusioNeRF_LLFF_6v_all_ckpts.sh" "$@"
