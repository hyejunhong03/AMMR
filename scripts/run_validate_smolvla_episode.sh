#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
set -u

exec python3 /home/autolab/AMMR/scripts/validate_smolvla_episode.py "$@"
