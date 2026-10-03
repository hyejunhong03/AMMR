#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/autolab/AMMR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"${PROJECT_ROOT}/.venv/bin/lerobot-train" \
  --config_path "${PROJECT_ROOT}/outputs/train/phase1_red_cube_strict_train_from_base_3000/checkpoints/003000/pretrained_model/train_config.json" \
  --dataset.root "${PROJECT_ROOT}/data/lerobot/phase1_red_cube_strict_train_state_std005" \
  --dataset.repo_id "ammr/phase1_red_cube_strict_train_state_std005" \
  --output_dir "${PROJECT_ROOT}/outputs/train/phase1_red_cube_state_std005_from_base_20000" \
  --steps 20000 \
  --save_freq 5000 \
  --eval_freq 20000 \
  --log_freq 100 \
  --wandb.enable false
