#!/usr/bin/env python3
"""Replay one recorded raw episode's action.npy against the Isaac ROS2 scene.

This isolates the execution stack from the learned policy. If replaying the exact
demonstration actions fails, the problem is reset/timing/joint order/simulator setup,
not SmolVLA generalization.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ammr_mycobot_interface import OBJECT_NAMES, object_state_slice  # noqa: E402
from rollout_policy import RolloutClient, object_xyz, velocity_diagnostics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, help="Raw episode_* directory.")
    parser.add_argument("--object", choices=OBJECT_NAMES, default="red_cube")
    parser.add_argument("--fps", type=float, default=10.0, help="Fallback fixed replay rate.")
    parser.add_argument(
        "--timing",
        choices=("recorded", "fixed"),
        default="recorded",
        help="Use recorded timestamp deltas or a fixed 1/fps period.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0 means all recorded actions.")
    parser.add_argument(
        "--max-action-step",
        type=float,
        default=0.0,
        help=(
            "Optional arm-joint limiter in rad/control-step. Default 0 replays exactly. "
            "Raw and published commands are both saved either way."
        ),
    )
    parser.add_argument("--settle-sec", type=float, default=6.0)
    parser.add_argument("--min-lift-m", type=float, default=0.02)
    parser.add_argument("--park-non-targets", action="store_true")
    parser.add_argument("--out", default=None, help="Write summary JSON.")
    parser.add_argument("--trace", default=None, help="Write trace NPZ.")
    return parser.parse_args()


def load_object_position(episode_dir, object_name):
    object_state_path = episode_dir / "object_state.npy"
    if object_state_path.exists():
        object_state = np.load(object_state_path)
        if object_state.ndim == 2 and object_state.shape[0] > 0:
            pos = object_state[0, object_state_slice(object_name)]
            if np.all(np.isfinite(pos)):
                return np.asarray(pos, dtype=np.float64)

    metadata = json.loads((episode_dir / "metadata.json").read_text())
    collection = metadata.get("collection") or {}
    for key in ("object_position", "sampled_object_pose", "cube_position"):
        value = collection.get(key) or metadata.get(key)
        if value and len(value) >= 3:
            return np.asarray(value[:3], dtype=np.float64)
    raise ValueError(f"Could not infer {object_name} position from {episode_dir}")


def load_periods(episode_dir, steps, fps, timing):
    default_period = 1.0 / max(float(fps), 1e-6)
    if timing == "fixed":
        return np.full((steps,), default_period, dtype=np.float64)

    timestamp_path = episode_dir / "timestamps.npy"
    if not timestamp_path.exists():
        return np.full((steps,), default_period, dtype=np.float64)
    timestamps = np.load(timestamp_path).astype(np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        return np.full((steps,), default_period, dtype=np.float64)

    deltas = np.diff(timestamps[:steps], append=timestamps[min(steps - 1, len(timestamps) - 1)] + default_period)
    # Keep replay timing close to the dataset while avoiding one bad stamp stalling
    # the script or compressing several commands into one physics frame.
    return np.clip(deltas, 0.02, 0.25)


def main():
    args = parse_args()
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    episode_dir = Path(args.episode_dir).resolve()
    actions = np.load(episode_dir / "action.npy").astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected action.npy shape [T, 7], got {actions.shape}")
    if args.max_steps > 0:
        actions = actions[: args.max_steps]
    periods = load_periods(episode_dir, len(actions), args.fps, args.timing)
    object_position = load_object_position(episode_dir, args.object)

    client = RolloutClient(drain=True)
    states, published, raw_actions, objects, stamps = [], [], [], [], []
    action_limited = []
    raw_arm_steps, published_arm_steps = [], []
    try:
        client.place_and_reset(
            object_position,
            object_name=args.object,
            settle=args.settle_sec,
            park_non_targets=args.park_non_targets,
        )
        if not client.wait_for_inputs():
            raise RuntimeError("No image/state feedback after reset")
        client.clear_grasp_events()
        start_objects = client.objects.copy() if client.objects is not None else None
        start_target = object_xyz(start_objects, args.object)

        previous_action = None
        for index, raw_action in enumerate(actions):
            client.spin(0.0)
            command = raw_action.copy()
            limited = False
            if args.max_action_step > 0.0:
                reference = previous_action[:6] if previous_action is not None else client.state[:6]
                raw_delta = command[:6] - reference
                clipped = np.clip(raw_delta, -args.max_action_step, args.max_action_step)
                limited = bool(np.any(np.abs(clipped - raw_delta) > 1e-7))
                command[:6] = reference + clipped
            else:
                reference = previous_action[:6] if previous_action is not None else raw_action[:6]
                raw_delta = raw_action[:6] - reference

            if client.state is not None:
                states.append(client.state.copy())
            else:
                states.append(np.full((7,), np.nan, dtype=np.float32))
            if client.objects is not None:
                objects.append(client.objects.copy())
            else:
                objects.append(np.full((len(OBJECT_NAMES) * 3,), np.nan, dtype=np.float32))
            raw_actions.append(raw_action.copy())
            published.append(command.copy())
            action_limited.append(limited)
            raw_arm_steps.append(float(np.max(np.abs(raw_delta))))
            if previous_action is None:
                published_arm_steps.append(0.0)
            else:
                published_arm_steps.append(float(np.max(np.abs(command[:6] - previous_action[:6]))))
            stamps.append(time.monotonic())

            client.publish(command)
            previous_action = command.copy()
            time.sleep(float(periods[min(index, len(periods) - 1)]))
    finally:
        client.shutdown()

    states = np.asarray(states, dtype=np.float32)
    published = np.asarray(published, dtype=np.float32)
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    objects = np.asarray(objects, dtype=np.float32)
    stamps = np.asarray(stamps, dtype=np.float64)
    target_positions = objects[:, object_state_slice(args.object)] if len(objects) else np.empty((0, 3))
    start_z = float(start_target[2]) if start_target is not None else float(object_position[2])
    finite_z = target_positions[:, 2][np.isfinite(target_positions[:, 2])] if len(target_positions) else []
    lift_m = float(np.max(finite_z) - start_z) if len(finite_z) else None

    result = {
        "episode_dir": str(episode_dir),
        "object_name": args.object,
        "object_position": [float(v) for v in object_position],
        "steps": int(len(raw_actions)),
        "timing": args.timing,
        "fps": float(args.fps),
        "max_action_step_limit": float(args.max_action_step),
        "limited_steps": int(np.sum(action_limited)),
        "raw_arm_step_max": float(np.max(raw_arm_steps)) if raw_arm_steps else None,
        "published_arm_step_max": float(np.max(published_arm_steps)) if published_arm_steps else None,
        "lift_m": round(lift_m, 4) if lift_m is not None else None,
        "success": bool(lift_m is not None and lift_m >= args.min_lift_m),
    }
    result.update(
        velocity_diagnostics(
            states,
            published,
            raw_actions,
            stamps,
            client.grasp_events,
            max_joint_velocity=0.5,
            max_action_step=args.max_action_step,
        )
    )

    print(json.dumps(result, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
    if args.trace:
        trace_path = Path(args.trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            trace_path,
            states=states,
            actions=published,
            raw_actions=raw_actions,
            action_limited=np.asarray(action_limited, dtype=np.bool_),
            objects=objects,
            stamps=stamps,
            grasp_events=np.asarray(client.grasp_events, dtype=np.float64),
        )
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
