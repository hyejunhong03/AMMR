#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
set -u

exec python3 /home/autolab/AMMR/scripts/scripted_pick_demo.py "$@"
