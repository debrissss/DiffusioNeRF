#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/runtime_env.sh"

if (( $# < 1 )); then
    echo "Usage: $0 <3|6|9> [scan8|8 ...]" >&2
    echo "Example: GPU_ID=0 CKPT_MODE=scratch $0 3 scan8" >&2
    exit 2
fi

VIEW_COUNT="$1"
shift

case "$VIEW_COUNT" in
    3)
        TOTAL_ITERS=30000
        DEFAULT_CHECKPOINT_INTERVAL=300
        DEFAULT_MILESTONES="6000 9000 12000 18000 24000 27000"
        ;;
    6)
        TOTAL_ITERS=30000
        DEFAULT_CHECKPOINT_INTERVAL=300
        DEFAULT_MILESTONES="6000 9000 12000 18000 24000 27000"
        ;;
    9)
        # The trainer requires epoch-aligned targets; 30000 is not divisible by 9.
        TOTAL_ITERS=29997
        DEFAULT_CHECKPOINT_INTERVAL=297
        DEFAULT_MILESTONES="6003 9000 11997 18000 24003 27000"
        ;;
    *)
        echo "View count must be one of: 3, 6, 9." >&2
        exit 2
        ;;
esac

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
CKPT_MODE="${CKPT_MODE:-latest}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-$DEFAULT_CHECKPOINT_INTERVAL}"
PROFILE_TRAINING="${PROFILE_TRAINING:-0}"
PROFILE_REPORT_STEPS="${PROFILE_REPORT_STEPS:-300}"
STOP_AT_STEP="${STOP_AT_STEP:-}"
MILESTONE_STEPS_SPEC="${MILESTONE_STEPS-$DEFAULT_MILESTONES}"
EVAL_OVERWRITE="${EVAL_OVERWRITE:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
SPLIT_FILE_OVERRIDE="${SPLIT_FILE_OVERRIDE:-}"
WORKSPACE_OVERRIDE="${WORKSPACE_OVERRIDE:-}"
BOUND="${BOUND:-2.0}"
ANNEAL_STEPS="${ANNEAL_STEPS:-5000}"
MASK_COMPOSITE="${MASK_COMPOSITE:-0}"
BG_COLOR="${BG_COLOR:-black}"

if [[ "$CKPT_MODE" != "latest" && "$CKPT_MODE" != "scratch" ]]; then
    echo "CKPT_MODE must be either 'latest' or 'scratch'." >&2
    exit 2
fi
for ITEM in \
    "SEED:$SEED" \
    "CHECKPOINT_INTERVAL_STEPS:$CHECKPOINT_INTERVAL_STEPS" \
    "PROFILE_REPORT_STEPS:$PROFILE_REPORT_STEPS"; do
    NAME="${ITEM%%:*}"
    VALUE="${ITEM#*:}"
    if [[ ! "$VALUE" =~ ^[1-9][0-9]*$ && ! ( "$NAME" == "SEED" && "$VALUE" == "0" ) ]]; then
        echo "$NAME must be a non-negative integer (positive for step intervals)." >&2
        exit 2
    fi
done
for ITEM in \
    "PROFILE_TRAINING:$PROFILE_TRAINING" \
    "EVAL_OVERWRITE:$EVAL_OVERWRITE" \
    "EVAL_ONLY:$EVAL_ONLY" \
    "DRY_RUN:$DRY_RUN"; do
    NAME="${ITEM%%:*}"
    VALUE="${ITEM#*:}"
    if [[ "$VALUE" != "0" && "$VALUE" != "1" ]]; then
        echo "$NAME must be either 0 or 1." >&2
        exit 2
    fi
done
if [[ "$EVAL_ONLY" == "1" && "$CKPT_MODE" == "scratch" ]]; then
    echo "EVAL_ONLY=1 requires CKPT_MODE=latest." >&2
    exit 2
fi

TARGET_STEP="${STOP_AT_STEP:-$TOTAL_ITERS}"
if [[ ! "$TARGET_STEP" =~ ^[1-9][0-9]*$ ]]; then
    echo "STOP_AT_STEP must be a positive integer." >&2
    exit 2
fi
if (( TARGET_STEP > TOTAL_ITERS )); then
    echo "STOP_AT_STEP=$TARGET_STEP exceeds the schedule horizon $TOTAL_ITERS." >&2
    exit 2
fi
if (( TARGET_STEP % VIEW_COUNT != 0 )); then
    echo "Target step $TARGET_STEP must be divisible by $VIEW_COUNT steps/epoch." >&2
    exit 2
fi
if (( CHECKPOINT_INTERVAL_STEPS % VIEW_COUNT != 0 )); then
    echo "CHECKPOINT_INTERVAL_STEPS=$CHECKPOINT_INTERVAL_STEPS must be divisible by $VIEW_COUNT." >&2
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
    for STEP in "${MILESTONE_STEP_VALUES[@]}"; do
        if [[ ! "$STEP" =~ ^[1-9][0-9]*$ ]]; then
            echo "MILESTONE_STEPS must contain positive integers." >&2
            exit 2
        fi
        if (( STEP > TOTAL_ITERS || STEP % VIEW_COUNT != 0 )); then
            echo "Milestone $STEP must not exceed $TOTAL_ITERS and must be divisible by $VIEW_COUNT." >&2
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
MASK_COMPOSITE_ARGS=()
if [[ "$MASK_COMPOSITE" == "1" ]]; then
    MASK_COMPOSITE_ARGS=(--train_mask_composite)
fi
PROFILE_ARGS=()
if [[ "$PROFILE_TRAINING" == "1" ]]; then
    PROFILE_ARGS=(--profile_training --profile_report_steps "$PROFILE_REPORT_STEPS")
fi

VALID_SCAN_NUMBERS=" 8 21 30 31 34 38 40 41 45 55 63 82 103 110 114 "
if (( $# > 0 )); then
    ALL_SCENES_REQUESTED=0
    RAW_SCENES=("$@")
else
    ALL_SCENES_REQUESTED=1
    RAW_SCENES=(8 21 30 31 34 38 40 41 45 55 63 82 103 110 114)
fi

SCENES=()
for RAW_SCENE in "${RAW_SCENES[@]}"; do
    if [[ "$RAW_SCENE" =~ ^scan([0-9]+)$ ]]; then
        SCAN_NUMBER="${BASH_REMATCH[1]}"
    elif [[ "$RAW_SCENE" =~ ^[0-9]+$ ]]; then
        SCAN_NUMBER="$RAW_SCENE"
    else
        echo "Invalid DTU scene: $RAW_SCENE" >&2
        exit 2
    fi
    if [[ "$VALID_SCAN_NUMBERS" != *" $SCAN_NUMBER "* ]]; then
        echo "Scene scan$SCAN_NUMBER is not in the standard 15-scene DTU benchmark." >&2
        exit 2
    fi
    SCENES+=("scan$SCAN_NUMBER")
done

if (( ${#SCENES[@]} != 1 )) && [[ -n "$SPLIT_FILE_OVERRIDE" || -n "$WORKSPACE_OVERRIDE" ]]; then
    echo "SPLIT_FILE_OVERRIDE and WORKSPACE_OVERRIDE require exactly one scene." >&2
    exit 2
fi

printf -v TARGET_STEP_PADDED '%06d' "$TARGET_STEP"
for SCENE in "${SCENES[@]}"; do
    DATASET="$REPO_DIR/data/DTU_standard/$SCENE"
    SPLIT_FILE="${SPLIT_FILE_OVERRIDE:-$REPO_DIR/splits/dtu_${VIEW_COUNT}v/$SCENE.json}"
    WORKSPACE="${WORKSPACE_OVERRIDE:-$REPO_DIR/test_DTU/test_$SCENE/few_shot${VIEW_COUNT}/test_DiffusioNeRF_NeurTV_Ray_30k_seed${SEED}}"
    COMPLETE_FILE="$WORKSPACE/evaluation/step_${TARGET_STEP_PADDED}/COMPLETE"

    # In all-scenes resume mode, completed scenes are immutable inputs to the
    # batch run. An incomplete or absent workspace is passed to main_nerf.py
    # with --ckpt latest, which resumes ngp.pth when present and otherwise
    # starts from scratch.
    if [[ "$ALL_SCENES_REQUESTED" == "1" && "$CKPT_MODE" == "latest" \
        && "$EVAL_ONLY" == "0" && -f "$COMPLETE_FILE" ]]; then
        echo "Skipping completed DTU scene: $SCENE"
        echo "Completed archive: $COMPLETE_FILE"
        continue
    fi

    if [[ ! -f "$DATASET/transforms.json" ]]; then
        echo "DTU transforms.json is missing: $DATASET/transforms.json" >&2
        exit 2
    fi
    if [[ ! -f "$SPLIT_FILE" ]]; then
        echo "DTU standard ${VIEW_COUNT}-view split is missing: $SPLIT_FILE" >&2
        exit 2
    fi
    if [[ "$CKPT_MODE" == "scratch" ]] \
        && [[ -n "$(find "$WORKSPACE/checkpoints" -type f -name '*.pth' -print -quit 2>/dev/null)" ]]; then
        echo "Refusing scratch mode because checkpoints already exist: $WORKSPACE" >&2
        echo "Use CKPT_MODE=latest to resume, or move to a new workspace." >&2
        exit 2
    fi

    CMD=(
        "$PYTHON_BIN" main_nerf.py "$DATASET"
        --workspace "$WORKSPACE"
        --split_file "$SPLIT_FILE"
        --few_shot "$VIEW_COUNT"
        --seed "$SEED"
        --ckpt "$CKPT_MODE"
        --checkpoint_interval_steps "$CHECKPOINT_INTERVAL_STEPS"
        "${PROFILE_ARGS[@]}"
        "${MODE_ARGS[@]}"
        --fp16
        --iters "$TOTAL_ITERS"
        "${STOP_ARGS[@]}"
        "${MILESTONE_ARGS[@]}"
        --eval_variants raw ema
        --eval_split test
        --eval_expected_step "$TARGET_STEP"
        "${EVAL_OVERWRITE_ARGS[@]}"
        --amp_max_retries 4
        --lr 0.01
        --num_rays 4096
        --num_steps 96
        --upsample_steps 192
        --downscale 1
        --scale 0.33
        --bound "$BOUND"
        --dataset_name "$SCENE ${VIEW_COUNT}-views"
        --implementation_name "DiffusioNeRF-Milestone1"
        --diff_reg
        --loss_dist
        --loss_fg
        --use_depth
        --diff_reg_start_iter 1000
        --dist_lambda 2e-5
        --fg_lambda 1e-4
        --patch_size 4
        --depth_reg_lambda 0.1
        --anneal_nearfar
        --anneal_nearfar_steps "$ANNEAL_STEPS"
        --ema_decay 0.95
        --eval_num_steps 128
        --eval_upsample_steps 256
        --anneal_nearfar_perc 0.2
        --anneal_mid_perc 0.5
        --no_virtual_ray
        --bg_color "$BG_COLOR"
        "${MASK_COMPOSITE_ARGS[@]}"
    )

    echo "Starting standard DTU ${VIEW_COUNT}-view NeurTV + virtual-ray experiment: $SCENE"
    echo "Checkpoint mode: $CKPT_MODE; target step: $TARGET_STEP; seed: $SEED"
    echo "Dataset: $DATASET"
    echo "Split: $SPLIT_FILE"
    echo "Workspace: $WORKSPACE"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q NO_GUI=1 ' "$GPU_ID"
        printf '%q ' "${CMD[@]}"
        printf '\n'
        continue
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" NO_GUI=1 "${CMD[@]}"

    if [[ ! -f "$COMPLETE_FILE" ]]; then
        echo "Formal evaluation archive is incomplete: $COMPLETE_FILE" >&2
        exit 1
    fi
    echo "Completed and archived: $SCENE -> $COMPLETE_FILE"
done
