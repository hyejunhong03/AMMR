#!/usr/bin/env bash
set -euo pipefail

# Requires the Isaac Sim task scene to already be running.

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

CKPT="outputs/train/recentered_safe_x130_210_y095_from_yposhard015k_20000/checkpoints/040000/pretrained_model"
POSES_FILE="${POSES_FILE:-outputs/eval/recentered_safe_setE_in_h020_d8_poses.json}"
STATS="data/lerobot/recentered_safe_x130_210_y095_train/meta/stats.json"
TRAIN_RAW="data/smolvla_raw_recentered_safe_x130_210_y095"
EPISODES="${EPISODES:-60}"

if [[ ! -d "${CKPT}" ]]; then
  echo "missing checkpoint: ${CKPT}" >&2
  exit 1
fi

if [[ ! -f "${POSES_FILE}" ]]; then
  echo "missing poses file: ${POSES_FILE}" >&2
  exit 1
fi

OUT="outputs/eval/recentered_safe_setE_yposhardadapt_040000_snap_thr004_x${EPISODES}.json"
TRACE_DIR="outputs/eval/recentered_safe_setE_yposhardadapt_040000_snap_thr004_x${EPISODES}_traces"

.venv/bin/python scripts/rollout_policy.py \
  --checkpoint "$CKPT" \
  --poses-file "$POSES_FILE" \
  --episodes "$EPISODES" \
  --out "$OUT" \
  --trace-dir "$TRACE_DIR" \
  --object red_cube \
  --park-non-targets \
  --task "pick the red cube" \
  --fps 10 \
  --max-steps 450 \
  --dataset-stats "$STATS" \
  --gripper-snap \
  --gripper-snap-threshold 0.04 \
  --max-action-step 0.05 \
  --max-joint-velocity 0.5 \
  --n-action-steps 50 \
  --noise-mode fixed-random \
  --noise-seed 0

.venv/bin/python scripts/summarize_pick_eval.py \
  "$OUT" \
  --train-raw-dir "$TRAIN_RAW" \
  --out-json outputs/eval/recentered_safe_setE_yposhardadapt_040000_summary_rawtrain_support.json \
  --failure-csv outputs/eval/recentered_safe_setE_yposhardadapt_040000_failures_rawtrain_support.csv
