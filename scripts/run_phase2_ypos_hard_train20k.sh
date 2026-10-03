#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/autolab/AMMR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"${PROJECT_ROOT}/.venv/bin/lerobot-train" \
  --config_path "${PROJECT_ROOT}/outputs/train/phase2_red_cube_pick_only_lift060_h020_d8_sample60_ypos_from_base_20000/checkpoints/020000/pretrained_model/train_config.json" \
  --dataset.root "${PROJECT_ROOT}/data/lerobot/phase2_red_cube_pick_only_lift060_h020_d8_sample60_ypos_hard_train" \
  --dataset.repo_id "ammr/phase2_red_cube_pick_only_lift060_h020_d8_sample60_ypos_hard_train" \
  --output_dir "${PROJECT_ROOT}/outputs/train/phase2_red_cube_pick_only_lift060_h020_d8_sample60_ypos_hard_from_base_20000" \
  --steps 20000 \
  --save_freq 5000 \
  --eval_freq 20000 \
  --log_freq 100 \
  --wandb.enable false
