#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

GPU_ID="${GPU_ID:-0}"
CKPT_MODE="${CKPT_MODE:-latest}"
STOP_AT_STEP="${STOP_AT_STEP:-}"

if [[ "$CKPT_MODE" != "latest" && "$CKPT_MODE" != "scratch" ]]; then
    echo "CKPT_MODE must be either 'latest' or 'scratch'." >&2
    exit 2
fi

STOP_ARGS=()
if [[ -n "$STOP_AT_STEP" ]]; then
    STOP_ARGS=(--stop_at_step "$STOP_AT_STEP")
fi

# Default order follows the staged execution plan: validate representative
# scenes first, then complete the remaining four scenes. Pass scene names as
# positional arguments to run a subset, for example: ./run_...sh fern room
if (( $# > 0 )); then
    SCENES=("$@")
else
    SCENES=(fern room orchids trex flower fortress horns leaves)
fi

declare -A SCALE=(
    [fern]=0.33
    [flower]=0.01
    [fortress]=0.3
    [horns]=0.2
    [leaves]=0.005
    [orchids]=0.1
    [room]=0.3
    [trex]=0.33
)

declare -A BOUND=(
    [fern]=3.0
    [flower]=3.0
    [fortress]=3.5
    [horns]=3.5
    [leaves]=3.0
    [orchids]=2.5
    [room]=3.0
    [trex]=4.5
)

for SCENE in "${SCENES[@]}"; do
    if [[ -z "${SCALE[$SCENE]+x}" ]]; then
        echo "Unknown LLFF scene: $SCENE" >&2
        exit 2
    fi

    DATASET="$REPO_DIR/data/nerf_llff_data/$SCENE"
    SPLIT_FILE="$REPO_DIR/splits/llff_3v/$SCENE.json"
    WORKSPACE="$REPO_DIR/test_LLFF/test_$SCENE/few_shot3/test_DiffusioNeRF_NeurTV_Ray_30k_seed0"

    if [[ "$CKPT_MODE" == "scratch" ]] && compgen -G "$WORKSPACE/checkpoints/*.pth" > /dev/null; then
        echo "Refusing scratch mode because checkpoints already exist: $WORKSPACE" >&2
        echo "Use CKPT_MODE=latest to resume, or choose a new workspace." >&2
        exit 2
    fi

    echo "Starting LLFF 3-view NeurTV + virtual-ray experiment: $SCENE"
    echo "Checkpoint mode: $CKPT_MODE; stop_at_step: ${STOP_AT_STEP:-full 30000}"

    CUDA_VISIBLE_DEVICES="$GPU_ID" NO_GUI=1 python main_nerf.py \
        "$DATASET" \
        --workspace "$WORKSPACE" \
        --split_file "$SPLIT_FILE" \
        --few_shot 3 \
        --seed 0 \
        --ckpt "$CKPT_MODE" \
        --fp16 \
        --iters 30000 \
        "${STOP_ARGS[@]}" \
        --amp_max_retries 4 \
        --lr 0.01 \
        --num_rays 4096 \
        --num_steps 64 \
        --upsample_steps 128 \
        --downscale 8 \
        --scale "${SCALE[$SCENE]}" \
        --bound "${BOUND[$SCENE]}" \
        --dataset_name "$SCENE 3-views" \
        --implementation_name "DiffusioNeRF+NeurTV+Ray" \
        --diff_reg \
        --loss_dist \
        --loss_fg \
        --use_depth \
        --diff_reg_start_iter 1000 \
        --dist_lambda 2e-5 \
        --fg_lambda 1e-4 \
        --patch_size 4 \
        --depth_reg_lambda 0.1 \
        --use_nll_color \
        --use_nll_sigma \
        --fre_nll_color \
        --total_iter_end_rate 0.9 \
        --smoothing \
        --smoothing_lambda 1e-5 \
        --smooth_sampling_method near_pixel \
        --virtual_ray \
        --virtual_ray_start_iter 1000 \
        --virtual_ray_k 10 \
        --virtual_ray_jsd_th 0.02 \
        --virtual_ray_depth_lambda 0.1 \
        --neurtv \
        --neurtv_start_iter 3000 \
        --neurtv_lambda 1e-7 \
        --neurtv_num_samples 4096 \
        --neurtv_fd_epsilon 0.01
done
