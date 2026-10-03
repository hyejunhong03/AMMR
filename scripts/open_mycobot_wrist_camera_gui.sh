#!/usr/bin/env bash
set -euo pipefail

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only

exec /home/autolab/isaacsim/python.sh /home/autolab/AMMR/scripts/control_mycobot_sliders_gui.py --view-wrist-camera "$@"
