#!/usr/bin/env bash
set -euo pipefail

# Requires the Isaac Sim task scene to already be running.

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

TRAIN_RUN="outputs/train/recentered_safe_x130_210_y095_from_yposhard015k_20000"
POSES_FILE="${POSES_FILE:-outputs/eval/recentered_safe_setD_in_h020_d8_poses.json}"
STATS="data/lerobot/recentered_safe_x130_210_y095_train/meta/stats.json"
EPISODES="${EPISODES:-50}"

if [[ ! -f "${POSES_FILE}" ]]; then
  echo "missing poses file: ${POSES_FILE}" >&2
  echo "run plan_grasp_poses.py for the held-out recentered-safe positions first" >&2
  exit 1
fi

COMMON=(
  --object red_cube
  --park-non-targets
  --task "pick the red cube"
  --fps 10
  --max-steps 450
  --dataset-stats "$STATS"
  --gripper-snap
  --gripper-snap-threshold 0.04
  --max-action-step 0.05
  --max-joint-velocity 0.5
  --n-action-steps 50
  --noise-mode fixed-random
  --noise-seed 0
)

RESULTS=()
for STEP in 005000 010000 015000 020000; do
  CKPT="${TRAIN_RUN}/checkpoints/${STEP}/pretrained_model"
  OUT="outputs/eval/recentered_safe_setD_yposhardadapt_${STEP}_snap_thr004_x${EPISODES}.json"
  TRACE_DIR="outputs/eval/recentered_safe_setD_yposhardadapt_${STEP}_snap_thr004_x${EPISODES}_traces"
  if [[ ! -d "${CKPT}" ]]; then
    echo "missing checkpoint: ${CKPT}" >&2
    exit 1
  fi
  .venv/bin/python scripts/rollout_policy.py \
    --checkpoint "$CKPT" \
    --poses-file "$POSES_FILE" \
    --episodes "$EPISODES" \
    --out "$OUT" \
    --trace-dir "$TRACE_DIR" \
    "${COMMON[@]}"
  RESULTS+=("$OUT")
done

.venv/bin/python scripts/summarize_pick_eval.py \
  "${RESULTS[@]}" \
  --out-json outputs/eval/recentered_safe_setD_yposhardadapt_summary.json \
  --failure-csv outputs/eval/recentered_safe_setD_yposhardadapt_failures.csv
