#!/usr/bin/env bash
set -euo pipefail

# Requires the Isaac Sim task scene to already be running.

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

CKPT_HARD15="outputs/train/phase2_red_cube_pick_only_lift060_h020_d8_sample60_ypos_hard_from_base_20000/checkpoints/015000/pretrained_model"
STATS="data/lerobot/phase2_red_cube_pick_only_lift060_h020_d8_sample60_ypos_hard_train/meta/stats.json"

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

.venv/bin/python scripts/rollout_policy.py \
  --checkpoint "$CKPT_HARD15" \
  --poses-file outputs/eval/phase2_setC_in_general_h020_d8_poses.json \
  --episodes 47 \
  --out outputs/eval/setC_in_general_yposhard015k_snap_thr004_x47.json \
  --trace-dir outputs/eval/setC_in_general_yposhard015k_snap_thr004_x47_traces \
  "${COMMON[@]}"

.venv/bin/python scripts/rollout_policy.py \
  --checkpoint "$CKPT_HARD15" \
  --poses-file outputs/eval/phase2_setC_in_ypos_high_spaced_h020_d8_poses.json \
  --episodes 22 \
  --out outputs/eval/setC_in_ypos_high_spaced_yposhard015k_snap_thr004_x22.json \
  --trace-dir outputs/eval/setC_in_ypos_high_spaced_yposhard015k_snap_thr004_x22_traces \
  "${COMMON[@]}"

.venv/bin/python scripts/summarize_pick_eval.py \
  outputs/eval/setC_in_general_yposhard015k_snap_thr004_x47.json \
  outputs/eval/setC_in_ypos_high_spaced_yposhard015k_snap_thr004_x22.json \
  --out-json outputs/eval/setC_yposhard015k_summary.json \
  --failure-csv outputs/eval/setC_yposhard015k_failures.csv
