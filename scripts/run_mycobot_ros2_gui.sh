#!/usr/bin/env bash
set -euo pipefail

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only

export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH="/home/autolab/isaacsim/exts/isaacsim.ros2.core/humble/rclpy:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/autolab/isaacsim/exts/isaacsim.ros2.core/humble/lib:${LD_LIBRARY_PATH:-}"

exec /home/autolab/isaacsim/python.sh /home/autolab/AMMR/scripts/control_mycobot_sliders_gui.py --ros2 --view-wrist-camera "$@"
