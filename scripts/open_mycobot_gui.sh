#!/usr/bin/env bash
set -euo pipefail

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only

USD_PATH="/home/autolab/AMMR/isaac_usd/mycobot_280_m5_adaptive_gripper_reimport/mycobot_280_m5_adaptive_gripper/mycobot_280_m5_adaptive_gripper.usda"
OPEN_SCRIPT="/home/autolab/AMMR/scripts/open_mycobot_stage.py"

echo "Opening Isaac Sim with:"
echo "  ${USD_PATH}"
echo
echo "If Isaac Sim opens an empty stage, use File > Open and select the same USD path."

exec /home/autolab/isaacsim/isaac-sim.sh --no-ros-env --exec "${OPEN_SCRIPT}"
