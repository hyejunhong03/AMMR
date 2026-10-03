#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/autolab/AMMR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

BASE_CKPT="${PROJECT_ROOT}/outputs/train/phase2_red_cube_pick_only_lift060_h020_d8_sample60_ypos_hard_from_base_20000/checkpoints/015000/pretrained_model"
DATASET="${PROJECT_ROOT}/data/lerobot/recentered_safe_x130_210_y095_train"
OUT="${PROJECT_ROOT}/outputs/train/recentered_safe_x130_210_y095_from_yposhard015k_20000"

"${PROJECT_ROOT}/.venv/bin/lerobot-train" \
  --config_path "${BASE_CKPT}/train_config.json" \
  --policy.pretrained_path "${BASE_CKPT}" \
  --dataset.root "${DATASET}" \
  --dataset.repo_id "ammr/recentered_safe_x130_210_y095_train" \
  --output_dir "${OUT}" \
  --steps 20000 \
  --save_freq 5000 \
  --eval_freq 20000 \
  --log_freq 100 \
  --wandb.enable false
