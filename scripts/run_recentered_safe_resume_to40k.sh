#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/autolab/AMMR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN_DIR="${PROJECT_ROOT}/outputs/train/recentered_safe_x130_210_y095_from_yposhard015k_20000"
LAST_CKPT="${RUN_DIR}/checkpoints/020000/pretrained_model"

"${PROJECT_ROOT}/.venv/bin/lerobot-train" \
  --config_path="${LAST_CKPT}/train_config.json" \
  --resume true \
  --output_dir "${RUN_DIR}" \
  --steps 40000 \
  --save_freq 5000 \
  --eval_freq 40000 \
  --log_freq 100 \
  --wandb.enable false
