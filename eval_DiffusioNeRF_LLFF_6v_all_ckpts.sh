#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/runtime_env.sh"

GPU_ID="${GPU_ID:-0}"
LLFF_VIEWS="${LLFF_VIEWS:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

case "$LLFF_VIEWS" in
    6)
        TOTAL_ITERS=30000
        CHECKPOINT_INTERVAL_STEPS=300
        MILESTONE_STEPS=(6000 9000 12000 18000 24000 27000)
        ;;
    9)
        TOTAL_ITERS=29997
        CHECKPOINT_INTERVAL_STEPS=297
        MILESTONE_STEPS=(6003 9000 11997 18000 24003 27000)
        ;;
    *)
        echo "LLFF_VIEWS must be either 6 or 9." >&2
        exit 2
        ;;
esac

if (( $# > 0 )); then
    SCENES=("$@")
else
    SCENES=(fern room orchids trex flower fortress horns leaves)
fi

declare -A SCALE=(
    [fern]=0.33
    [room]=0.3
    [orchids]=0.1
    [trex]=0.33
    [flower]=0.01
    [fortress]=0.3
    [horns]=0.2
    [leaves]=0.005
)

declare -A BOUND=(
    [fern]=3.0
    [room]=3.0
    [orchids]=2.5
    [trex]=4.5
    [flower]=3.0
    [fortress]=3.5
    [horns]=3.5
    [leaves]=3.0
)

for SCENE in "${SCENES[@]}"; do
    if [[ -z "${SCALE[$SCENE]+x}" ]]; then
        echo "Unknown LLFF scene: $SCENE" >&2
        exit 2
    fi

    DATASET="$REPO_DIR/data/nerf_llff_data/$SCENE"
    SPLIT_FILE="$REPO_DIR/splits/llff_${LLFF_VIEWS}v/$SCENE.json"
    WORKSPACE="$REPO_DIR/test_LLFF/test_$SCENE/few_shot${LLFF_VIEWS}/test_DiffusioNeRF_NeurTV_Ray_30k_seed0"
    CHECKPOINT_DIR="$WORKSPACE/checkpoints"

    if [[ ! -f "$DATASET/transforms.json" ]]; then
        echo "LLFF transforms.json is missing: $DATASET/transforms.json" >&2
        exit 2
    fi
    if [[ ! -f "$SPLIT_FILE" ]]; then
        echo "LLFF standard $LLFF_VIEWS-view split is missing: $SPLIT_FILE" >&2
        exit 2
    fi
    if [[ ! -d "$CHECKPOINT_DIR" ]]; then
        echo "Checkpoint directory is missing: $CHECKPOINT_DIR" >&2
        exit 2
    fi
    if [[ -z "$(find "$CHECKPOINT_DIR" -type f -name '*.pth' -print -quit)" ]]; then
        echo "No checkpoint files found: $CHECKPOINT_DIR" >&2
        exit 2
    fi

    while IFS= read -r -d '' CKPT; do
        read -r STEP HAS_EMA < <(
            "$PYTHON_BIN" - "$CKPT" <<'PY'
import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
step = checkpoint.get("global_step")
if not isinstance(step, int) or step <= 0:
    raise RuntimeError(f"Invalid checkpoint step: {step}")
print(step, int("ema" in checkpoint))
PY
        )

        printf -v STEP_PADDED '%06d' "$STEP"
        RELATIVE_CKPT="${CKPT#"$CHECKPOINT_DIR/"}"
        OUTPUT_NAME="${RELATIVE_CKPT%.pth}"
        OUTPUT_NAME="${OUTPUT_NAME//\//__}"
        OUTPUT_DIR="$WORKSPACE/evaluation/all_checkpoints/${OUTPUT_NAME}__step_${STEP_PADDED}"

        VARIANTS=(raw)
        if [[ "$HAS_EMA" == "1" ]]; then
            VARIANTS+=(ema)
        fi

        echo
        echo "============================================================"
        echo "Scene:      $SCENE"
        echo "Checkpoint: $CKPT"
        echo "Step:       $STEP"
        echo "Variants:   ${VARIANTS[*]}"
        echo "Output:     $OUTPUT_DIR"
        echo "============================================================"

        CUDA_VISIBLE_DEVICES="$GPU_ID" NO_GUI=1 "$PYTHON_BIN" main_nerf.py \
            "$DATASET" \
            --workspace "$WORKSPACE" \
            --split_file "$SPLIT_FILE" \
            --few_shot "$LLFF_VIEWS" \
            --seed 0 \
            --ckpt "$CKPT" \
            --test \
            --fp16 \
            --iters "$TOTAL_ITERS" \
            --checkpoint_interval_steps "$CHECKPOINT_INTERVAL_STEPS" \
            --milestone_steps "${MILESTONE_STEPS[@]}" \
            --eval_variants "${VARIANTS[@]}" \
            --eval_split test \
            --eval_expected_step "$STEP" \
            --eval_output_dir "$OUTPUT_DIR" \
            --amp_max_retries 4 \
            --lr 0.01 \
            --num_rays 4096 \
            --num_steps 64 \
            --upsample_steps 128 \
            --downscale 8 \
            --scale "${SCALE[$SCENE]}" \
            --bound "${BOUND[$SCENE]}" \
            --dataset_name "$SCENE $LLFF_VIEWS-views" \
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

        if [[ ! -f "$OUTPUT_DIR/COMPLETE" ]]; then
            echo "Formal evaluation archive is incomplete: $OUTPUT_DIR" >&2
            exit 1
        fi
        echo "Completed: $SCENE step $STEP -> $OUTPUT_DIR"
    done < <(find "$CHECKPOINT_DIR" -type f -name '*.pth' -print0 | sort -z)
done

echo
echo "All requested LLFF $LLFF_VIEWS-view checkpoints have been evaluated."
