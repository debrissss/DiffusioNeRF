#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/runtime_env.sh"

if [[ $# -ne 1 ]]; then
  echo "Usage: GPU_ID=0 $0 <scan_id>" >&2
  exit 2
fi

SCAN_ID="$1"
case "$SCAN_ID" in
  40|55|63|110) ;;
  *)
    echo "Supported scan IDs: 40, 55, 63, 110" >&2
    exit 2
    ;;
esac

GPU_ID="${GPU_ID:-0}"
DATASET="$REPO_DIR/data/DTU_standard_original_alpha_400x300/scan${SCAN_ID}"
SPLIT_FILE="$REPO_DIR/splits/dtu_9v/scan${SCAN_ID}.json"
WORKSPACE="$REPO_DIR/test_DTU/test_scan${SCAN_ID}_alpha_original/few_shot9/test_M1_repro_14994_seed0"

CUDA_VISIBLE_DEVICES="$GPU_ID" NO_GUI=1 "$PYTHON_BIN" main_nerf.py "$DATASET" \
  --workspace "$WORKSPACE" \
  --split_file "$SPLIT_FILE" \
  --few_shot 9 \
  --seed 0 \
  --ckpt scratch \
  --fp16 \
  --iters 14994 \
  --stop_at_step 14994 \
  --checkpoint_interval_steps 900 \
  --milestone_steps 11997 14004 \
  --eval_variants raw ema \
  --eval_split test \
  --eval_expected_step 14994 \
  --amp_max_retries 4 \
  --lr 0.01 \
  --num_rays 4096 \
  --num_steps 96 \
  --upsample_steps 192 \
  --eval_num_steps 128 \
  --eval_upsample_steps 256 \
  --ema_decay 0.95 \
  --anneal_nearfar \
  --anneal_nearfar_steps 7000 \
  --anneal_nearfar_perc 0.2 \
  --anneal_mid_perc 0.5 \
  --no_virtual_ray \
  --bg_color white \
  --eval_bg_color gray \
  --diff_reg \
  --loss_dist \
  --loss_fg \
  --use_depth \
  --diff_reg_start_iter 1000 \
  --dist_lambda 2e-5 \
  --fg_lambda 1e-4 \
  --patch_size 4 \
  --depth_reg_lambda 0.1 \
  --downscale 1 \
  --scale 0.33 \
  --bound 2.0 \
  --dataset_name "scan${SCAN_ID} 9-views Alpha reproduction" \
  --implementation_name "DiffusioNeRF-Milestone1"
