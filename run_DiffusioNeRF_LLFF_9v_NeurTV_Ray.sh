#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/runtime_env.sh"

GPU_ID="${GPU_ID:-0}"
CKPT_MODE="${CKPT_MODE:-latest}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-297}"
PROFILE_TRAINING="${PROFILE_TRAINING:-0}"
PROFILE_REPORT_STEPS="${PROFILE_REPORT_STEPS:-300}"
STOP_AT_STEP="${STOP_AT_STEP:-}"
MILESTONE_STEPS_SPEC="${MILESTONE_STEPS-6003 9000 11997 18000 24003 27000}"
EVAL_OVERWRITE="${EVAL_OVERWRITE:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

if [[ "$CKPT_MODE" != "latest" && "$CKPT_MODE" != "scratch" ]]; then
    echo "CKPT_MODE must be either 'latest' or 'scratch'." >&2
    exit 2
fi
if [[ ! "$CHECKPOINT_INTERVAL_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "CHECKPOINT_INTERVAL_STEPS must be a positive integer." >&2
    exit 2
fi
if [[ "$PROFILE_TRAINING" != "0" && "$PROFILE_TRAINING" != "1" ]]; then
    echo "PROFILE_TRAINING must be either 0 or 1." >&2
    exit 2
fi
if [[ ! "$PROFILE_REPORT_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "PROFILE_REPORT_STEPS must be a positive integer." >&2
    exit 2
fi
if [[ "$EVAL_OVERWRITE" != "0" && "$EVAL_OVERWRITE" != "1" ]]; then
    echo "EVAL_OVERWRITE must be either 0 or 1." >&2
    exit 2
fi
if [[ "$EVAL_ONLY" != "0" && "$EVAL_ONLY" != "1" ]]; then
    echo "EVAL_ONLY must be either 0 or 1." >&2
    exit 2
fi
if [[ "$EVAL_ONLY" == "1" && "$CKPT_MODE" == "scratch" ]]; then
    echo "EVAL_ONLY=1 requires CKPT_MODE=latest." >&2
    exit 2
fi

STOP_ARGS=()
if [[ -n "$STOP_AT_STEP" ]]; then
    STOP_ARGS=(--stop_at_step "$STOP_AT_STEP")
fi
MILESTONE_ARGS=()
MILESTONE_STEP_VALUES=()
if [[ -n "$MILESTONE_STEPS_SPEC" ]]; then
    read -r -a MILESTONE_STEP_VALUES <<< "$MILESTONE_STEPS_SPEC"
    for MILESTONE_STEP in "${MILESTONE_STEP_VALUES[@]}"; do
        if [[ ! "$MILESTONE_STEP" =~ ^[1-9][0-9]*$ ]]; then
            echo "MILESTONE_STEPS must contain positive integer steps." >&2
            exit 2
        fi
    done
    MILESTONE_ARGS=(--milestone_steps "${MILESTONE_STEP_VALUES[@]}")
fi
EVAL_OVERWRITE_ARGS=()
if [[ "$EVAL_OVERWRITE" == "1" ]]; then
    EVAL_OVERWRITE_ARGS=(--eval_overwrite)
fi
MODE_ARGS=()
if [[ "$EVAL_ONLY" == "1" ]]; then
    MODE_ARGS=(--test)
fi
PROFILE_ARGS=()
if [[ "$PROFILE_TRAINING" == "1" ]]; then
    PROFILE_ARGS=(--profile_training --profile_report_steps "$PROFILE_REPORT_STEPS")
fi
TARGET_STEP="${STOP_AT_STEP:-29997}"
printf -v TARGET_STEP_PADDED '%06d' "$TARGET_STEP"

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
    SPLIT_FILE="$REPO_DIR/splits/llff_9v/$SCENE.json"
    WORKSPACE="$REPO_DIR/test_LLFF/test_$SCENE/few_shot9/test_DiffusioNeRF_NeurTV_Ray_30k_seed0"

    if [[ ! -f "$DATASET/transforms.json" ]]; then
        echo "LLFF transforms.json is missing: $DATASET/transforms.json" >&2
        exit 2
    fi
    if [[ ! -f "$SPLIT_FILE" ]]; then
        echo "LLFF standard 9-view split is missing: $SPLIT_FILE" >&2
        exit 2
    fi
    if [[ "$CKPT_MODE" == "scratch" ]] \
        && [[ -n "$(find "$WORKSPACE/checkpoints" -type f -name '*.pth' -print -quit 2>/dev/null)" ]]; then
        echo "Refusing scratch mode because checkpoints already exist: $WORKSPACE" >&2
        echo "Use CKPT_MODE=latest to resume, or choose a new workspace." >&2
        exit 2
    fi

    echo "Starting standard LLFF 9-view NeurTV + virtual-ray experiment: $SCENE"
    echo "Checkpoint mode: $CKPT_MODE; stop_at_step: ${STOP_AT_STEP:-full 29997}"
    echo "Protected milestone steps: ${MILESTONE_STEPS_SPEC:-disabled}"

    CUDA_VISIBLE_DEVICES="$GPU_ID" NO_GUI=1 "$PYTHON_BIN" main_nerf.py \
        "$DATASET" \
        --workspace "$WORKSPACE" \
        --split_file "$SPLIT_FILE" \
        --few_shot 9 \
        --seed 0 \
        --ckpt "$CKPT_MODE" \
        --checkpoint_interval_steps "$CHECKPOINT_INTERVAL_STEPS" \
        "${PROFILE_ARGS[@]}" \
        "${MODE_ARGS[@]}" \
        --fp16 \
        --iters 29997 \
        "${STOP_ARGS[@]}" \
        "${MILESTONE_ARGS[@]}" \
        --eval_variants raw ema \
        --eval_split test \
        --eval_expected_step "$TARGET_STEP" \
        "${EVAL_OVERWRITE_ARGS[@]}" \
        --amp_max_retries 4 \
        --lr 0.01 \
        --num_rays 4096 \
        --num_steps 64 \
        --upsample_steps 128 \
        --downscale 8 \
        --scale "${SCALE[$SCENE]}" \
        --bound "${BOUND[$SCENE]}" \
        --dataset_name "$SCENE 9-views" \
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

    COMPLETE_FILE="$WORKSPACE/evaluation/step_${TARGET_STEP_PADDED}/COMPLETE"
    if [[ ! -f "$COMPLETE_FILE" ]]; then
        echo "Formal evaluation archive is incomplete: $COMPLETE_FILE" >&2
        exit 1
    fi
    echo "Completed and archived: $SCENE -> $COMPLETE_FILE"
done
