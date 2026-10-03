#!/usr/bin/env python3
"""Evaluate pick success from a recorded AMMR raw episode."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ammr_mycobot_interface import OBJECT_NAMES, OBJECT_STATE_NAMES, object_state_z_index


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate object lift from an AMMR SmolVLA raw episode.")
    parser.add_argument("episode_dir", help="Path to an episode directory.")
    parser.add_argument("--object", choices=OBJECT_NAMES, default="red_cube", help="Target object.")
    parser.add_argument(
        "--min-lift-m",
        type=float,
        default=0.02,
        help="Minimum z increase over initial object height required for success.",
    )
    parser.add_argument(
        "--min-hold-sec",
        type=float,
        default=0.0,
        help=(
            "Minimum continuous time the object must remain above initial_z + min_lift_m. "
            "Set 0 to keep the legacy max-height-only criterion."
        ),
    )
    parser.add_argument(
        "--max-hold-gap-sec",
        type=float,
        default=None,
        help=(
            "Break a hold segment when consecutive recorded timestamps are farther apart "
            "than this. Defaults to 2.5 / rate_hz from metadata."
        ),
    )
    return parser.parse_args()


def _metadata_rate_hz(episode_dir):
    metadata_path = episode_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            return float(metadata.get("rate_hz", 10.0))
        except Exception:
            pass
    return 10.0


def _load_step_times(episode_dir, num_steps):
    timestamp_path = episode_dir / "timestamps.npy"
    if timestamp_path.exists():
        values = np.load(timestamp_path)
        if values.ndim == 1 and values.shape[0] == num_steps and np.all(np.isfinite(values)):
            return values.astype(np.float64)

    rate_hz = _metadata_rate_hz(episode_dir)
    return np.arange(num_steps, dtype=np.float64) / max(rate_hz, 1e-6)


def _max_continuous_hold(times, above, max_gap_sec):
    if len(times) == 0 or len(above) == 0:
        return 0.0
    max_hold = 0.0
    start_index = None
    last_index = None
    for index, is_above in enumerate(above):
        if (
            start_index is not None
            and last_index is not None
            and float(times[index] - times[last_index]) > max_gap_sec
        ):
            max_hold = max(max_hold, float(times[last_index] - times[start_index]))
            start_index = None
            last_index = None
        if is_above:
            if start_index is None:
                start_index = index
            last_index = index
        elif start_index is not None:
            max_hold = max(max_hold, float(times[last_index] - times[start_index]))
            start_index = None
            last_index = None
    if start_index is not None:
        max_hold = max(max_hold, float(times[last_index] - times[start_index]))
    return max_hold


def main():
    args = parse_args()
    episode_dir = Path(args.episode_dir)
    if not episode_dir.is_dir():
        print(f"Episode directory does not exist: {episode_dir}", file=sys.stderr)
        return 2

    object_state_path = episode_dir / "object_state.npy"
    if not object_state_path.exists():
        print(f"Cannot evaluate pick success; missing {object_state_path}", file=sys.stderr)
        print("Record a new episode after restarting Isaac Sim with the updated ROS2 bridge.", file=sys.stderr)
        return 2

    object_state = np.load(object_state_path)
    if object_state.ndim != 2 or object_state.shape[1] != len(OBJECT_STATE_NAMES):
        print(f"Invalid object_state.npy shape: {object_state.shape}", file=sys.stderr)
        return 1
    if object_state.shape[0] == 0:
        print("object_state.npy has no steps", file=sys.stderr)
        return 1

    z = object_state[:, object_state_z_index(args.object)]
    valid = np.isfinite(z)
    if not np.any(valid):
        print(f"No finite z samples for {args.object}", file=sys.stderr)
        return 1

    valid_indices = np.flatnonzero(valid)
    first_index = int(valid_indices[0])
    max_index = int(valid_indices[np.argmax(z[valid])])
    initial_z = float(z[first_index])
    max_z = float(z[max_index])
    final_z = float(z[valid_indices[-1]])
    lift_delta = max_z - initial_z
    lift_threshold = initial_z + args.min_lift_m
    times = _load_step_times(episode_dir, object_state.shape[0])
    rate_hz = _metadata_rate_hz(episode_dir)
    max_hold_gap_sec = (
        float(args.max_hold_gap_sec)
        if args.max_hold_gap_sec is not None
        else 2.5 / max(rate_hz, 1e-6)
    )
    above = np.isfinite(z) & (z >= lift_threshold)
    hold_sec = _max_continuous_hold(times, above, max_hold_gap_sec)
    height_success = lift_delta >= args.min_lift_m
    hold_success = args.min_hold_sec <= 0.0 or hold_sec >= args.min_hold_sec
    success = height_success and hold_success

    print(f"episode_dir: {episode_dir}")
    print(f"object: {args.object}")
    print(f"initial_z: {initial_z:.4f} m")
    print(f"max_z: {max_z:.4f} m at step {max_index}")
    print(f"final_z: {final_z:.4f} m")
    print(f"lift_delta: {lift_delta:.4f} m")
    print(f"min_lift_m: {args.min_lift_m:.4f} m")
    print(f"lift_hold_sec: {hold_sec:.4f} s")
    print(f"min_hold_sec: {args.min_hold_sec:.4f} s")
    print(f"max_hold_gap_sec: {max_hold_gap_sec:.4f} s")
    print(f"height_success: {str(height_success).lower()}")
    print(f"hold_success: {str(hold_success).lower()}")
    print(f"pick_success: {str(success).lower()}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
