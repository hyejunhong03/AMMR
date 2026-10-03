#!/usr/bin/env python3
"""Test gripper close from a saved rollout trace close pose.

This is a contact-isolation test. It moves the arm to the joint state observed at a
saved trace step while keeping the gripper open, then closes only the gripper and
holds. Use it to tell whether a failed policy rollout's close pose is itself an
unstable contact configuration.
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
from rollout_policy import (  # noqa: E402
    RolloutClient,
    load_dataset_stats,
    object_xyz,
    start_state_diagnostics,
    tcp_diagnostics,
    velocity_diagnostics,
    z_scores,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="Source rollout_*.npz trace.")
    parser.add_argument("--object", choices=OBJECT_NAMES, default="red_cube")
    parser.add_argument(
        "--source-step",
        default="close",
        help="Trace step to use for arm pose, or 'close' for first action with gripper < 0.",
    )
    parser.add_argument(
        "--object-position",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override object reset position. Defaults to trace objects[0].",
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-action-step", type=float, default=0.05)
    parser.add_argument("--settle-sec", type=float, default=6.0)
    parser.add_argument("--move-hold-sec", type=float, default=1.0)
    parser.add_argument("--pre-close-hold-sec", type=float, default=1.0)
    parser.add_argument("--close-hold-sec", type=float, default=3.0)
    parser.add_argument("--gripper-open", type=float, default=0.08)
    parser.add_argument("--gripper-close", type=float, default=-0.245)
    parser.add_argument(
        "--replay-tail",
        action="store_true",
        help="After moving to the source pose, replay trace actions from that step instead of close-only.",
    )
    parser.add_argument(
        "--tail-action-key",
        choices=("actions", "raw_actions", "policy_raw_actions"),
        default="actions",
        help="Trace action array to replay when --replay-tail is set.",
    )
    parser.add_argument(
        "--tail-start-step",
        default=None,
        help="Tail replay start step. Defaults to --source-step after resolution.",
    )
    parser.add_argument(
        "--tail-max-steps",
        type=int,
        default=0,
        help="Maximum number of tail actions to replay. 0 means replay to the end of the trace.",
    )
    parser.add_argument(
        "--tail-latch-gripper",
        action="store_true",
        help="During --replay-tail, keep the gripper closed after the first close command.",
    )
    parser.add_argument(
        "--tail-latch-threshold",
        type=float,
        default=0.0,
        help="First gripper command below this value starts the tail latch.",
    )
    parser.add_argument("--park-non-targets", action="store_true")
    parser.add_argument("--min-lift-m", type=float, default=0.02)
    parser.add_argument("--max-joint-velocity", type=float, default=0.5)
    parser.add_argument("--dataset-stats", default=None)
    parser.add_argument("--state-std-floor", type=float, default=0.0)
    parser.add_argument("--start-state-z-threshold", type=float, default=3.0)
    parser.add_argument("--reset-retries", type=int, default=2)
    parser.add_argument("--out", default=None)
    parser.add_argument("--trace-dir", default=None)
    return parser.parse_args()


def resolve_source_step(trace, value):
    actions = np.asarray(trace["actions"], dtype=np.float32)
    if value == "close":
        indices = np.where(actions[:, 6] < 0.0)[0]
        if len(indices) == 0:
            raise ValueError("could not find close step: no action gripper < 0")
        return int(indices[0])
    step = int(value)
    if step < 0 or step >= len(actions):
        raise ValueError(f"source step {step} outside trace length {len(actions)}")
    return step


def trace_object_position(trace, object_name, override):
    if override is not None:
        return np.asarray(override, dtype=np.float64)
    objects = trace.get("objects")
    if objects is None or len(objects) == 0:
        raise ValueError("--object-position is required because trace has no objects array")
    position = object_xyz(np.asarray(objects[0], dtype=np.float64), object_name)
    if position is None or not np.all(np.isfinite(position)):
        raise ValueError("--object-position is required because trace objects[0] is invalid")
    return np.asarray(position, dtype=np.float64)


def append_sample(client, action, states, actions, objects, stamps):
    states.append(client.state.copy() if client.state is not None else np.full((7,), np.nan, dtype=np.float32))
    objects.append(
        client.objects.copy()
        if client.objects is not None
        else np.full((len(OBJECT_NAMES) * 3,), np.nan, dtype=np.float32)
    )
    actions.append(action.copy())
    stamps.append(time.monotonic())


def publish_for_period(client, action, seconds, fps, states, actions, objects, stamps):
    steps = max(1, int(round(float(seconds) * float(fps))))
    period = 1.0 / max(float(fps), 1e-6)
    next_deadline = time.monotonic()
    for _ in range(steps):
        client.spin(0.0)
        append_sample(client, action, states, actions, objects, stamps)
        client.publish(action)
        now = time.monotonic()
        next_deadline += period
        if next_deadline < now:
            next_deadline = now + period
        sleep_s = next_deadline - now
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def publish_action_sequence(client, sequence, fps, states, actions, objects, stamps):
    period = 1.0 / max(float(fps), 1e-6)
    next_deadline = time.monotonic()
    for command in sequence:
        command = np.asarray(command, dtype=np.float32)
        client.spin(0.0)
        append_sample(client, command, states, actions, objects, stamps)
        client.publish(command)
        now = time.monotonic()
        next_deadline += period
        if next_deadline < now:
            next_deadline = now + period
        sleep_s = next_deadline - now
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def latch_gripper_after_first_close(sequence, threshold, close_value):
    sequence = np.asarray(sequence, dtype=np.float32).copy()
    indices = np.where(sequence[:, 6] < float(threshold))[0]
    if len(indices):
        sequence[int(indices[0]) :, 6] = float(close_value)
    return sequence, (int(indices[0]) if len(indices) else None)


def move_to_pose(client, target, args, states, actions, objects, stamps):
    client.spin(0.0)
    if client.state is None:
        raise RuntimeError("no robot state before move")
    current = client.state.copy()
    current[6] = float(args.gripper_open)
    target = np.asarray(target, dtype=np.float32).copy()
    target[6] = float(args.gripper_open)
    max_delta = float(np.max(np.abs(target[:6] - current[:6])))
    steps = max(1, int(np.ceil(max_delta / max(float(args.max_action_step), 1e-6))))
    period = 1.0 / max(float(args.fps), 1e-6)
    next_deadline = time.monotonic()
    for i in range(1, steps + 1):
        alpha = i / steps
        action = current + alpha * (target - current)
        action[6] = float(args.gripper_open)
        client.spin(0.0)
        append_sample(client, action, states, actions, objects, stamps)
        client.publish(action)
        now = time.monotonic()
        next_deadline += period
        if next_deadline < now:
            next_deadline = now + period
        sleep_s = next_deadline - now
        if sleep_s > 0.0:
            time.sleep(sleep_s)
    if args.move_hold_sec > 0:
        publish_for_period(client, target, args.move_hold_sec, args.fps, states, actions, objects, stamps)


def run_one(client, arm_pose, object_position, args, dataset_stats, tail_actions=None):
    reset_diagnostics = []
    feedback_ok = False
    start_diag = {"checked": False, "ok": True}
    for attempt in range(max(1, args.reset_retries)):
        client.place_and_reset(
            object_position,
            object_name=args.object,
            settle=args.settle_sec,
            park_non_targets=args.park_non_targets,
        )
        feedback_ok = client.wait_for_inputs()
        if not feedback_ok:
            reset_diagnostics.append({"attempt": attempt + 1, "error": "no image/state feedback"})
            continue
        start_diag = start_state_diagnostics(client.state, dataset_stats, args.start_state_z_threshold)
        reset_diagnostics.append({"attempt": attempt + 1, **start_diag})
        if start_diag["ok"]:
            break
    if not feedback_ok or not start_diag["ok"]:
        return {
            "object_name": args.object,
            "object_position": object_position.tolist(),
            "invalid_trial": True,
            "error": "no image/state feedback" if not feedback_ok else "reset_state_out_of_distribution",
            "reset_attempts": len(reset_diagnostics),
            "reset_diagnostics": reset_diagnostics,
            "success": False,
        }, None

    client.clear_grasp_events()
    start_objects = client.objects.copy() if client.objects is not None else None
    start_target = object_xyz(start_objects, args.object)
    states, actions, objects, stamps = [], [], [], []

    open_target = np.asarray([*arm_pose[:6], float(args.gripper_open)], dtype=np.float32)
    close_target = np.asarray([*arm_pose[:6], float(args.gripper_close)], dtype=np.float32)
    move_to_pose(client, open_target, args, states, actions, objects, stamps)
    if args.pre_close_hold_sec > 0:
        publish_for_period(client, open_target, args.pre_close_hold_sec, args.fps, states, actions, objects, stamps)
    if tail_actions is not None:
        publish_action_sequence(client, tail_actions, args.fps, states, actions, objects, stamps)
    else:
        publish_for_period(client, close_target, args.close_hold_sec, args.fps, states, actions, objects, stamps)

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    objects = np.asarray(objects, dtype=np.float32)
    stamps = np.asarray(stamps, dtype=np.float64)
    tcp_distances, tcp_positions = tcp_diagnostics(states, objects, args.object)
    target_positions = objects[:, object_state_slice(args.object)] if len(objects) else np.empty((0, 3))
    start_z = float(start_target[2]) if start_target is not None else float(object_position[2])
    finite_z = target_positions[:, 2][np.isfinite(target_positions[:, 2])] if len(target_positions) else []
    lift_m = float(np.max(finite_z) - start_z) if len(finite_z) else None
    xy_m = None
    if len(target_positions):
        xy = np.linalg.norm(target_positions[:, :2] - target_positions[0, :2], axis=1)
        xy_m = float(np.nanmax(xy))

    trace = {
        "states": states,
        "actions": actions,
        "raw_actions": actions.copy(),
        "objects": objects,
        "stamps": stamps,
        "grasp_events": np.asarray(client.grasp_events, dtype=np.float64),
        "tcp_positions": tcp_positions,
        "tcp_target_distances": tcp_distances,
    }
    if dataset_stats is not None and len(states):
        trace.update(
            {
                "state_z": z_scores(states, dataset_stats["observation.state"]),
                "action_z": z_scores(actions, dataset_stats["action"]),
                "raw_action_z": z_scores(actions, dataset_stats["action"]),
                "target_state_error": actions - states,
                "raw_target_state_error": actions - states,
            }
        )
    result = {
        "object_name": args.object,
        "object_position": object_position.tolist(),
        "source_pose": [float(v) for v in arm_pose[:6]],
        "gripper_open": float(args.gripper_open),
        "gripper_close": float(args.gripper_close),
        "mode": "tail_replay_from_source_pose" if tail_actions is not None else "close_only",
        "steps": int(len(states)),
        "reset_attempts": len(reset_diagnostics),
        "reset_diagnostics": reset_diagnostics,
        "start_state_z_max": start_diag if start_diag.get("checked") else None,
        "lift_m": round(lift_m, 4) if lift_m is not None else None,
        "xy_displacement_m": round(xy_m, 4) if xy_m is not None else None,
        "success": bool(lift_m is not None and lift_m >= args.min_lift_m),
    }
    if len(tcp_distances) and np.any(np.isfinite(tcp_distances)):
        result["tcp_target_distance_min_m"] = float(np.nanmin(tcp_distances))
        result["tcp_target_distance_final_m"] = float(tcp_distances[-1])
    if len(states) > 1:
        result.update(
            velocity_diagnostics(
                states,
                actions,
                actions,
                stamps,
                client.grasp_events,
                args.max_joint_velocity,
                args.max_action_step,
            )
        )
    if dataset_stats is not None and len(states):
        state_z = trace["state_z"]
        dim = np.unravel_index(np.nanargmax(np.abs(state_z)), state_z.shape)
        result["state_z_max"] = {
            "abs": float(abs(state_z[dim])),
            "value": float(state_z[dim]),
            "index": [int(dim[0]), int(dim[1])],
            "dim_1based": int(dim[1]) + 1,
        }
    return result, trace


def main():
    args = parse_args()
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    trace_path = Path(args.trace).resolve()
    source = np.load(trace_path, allow_pickle=True)
    step = resolve_source_step(source, args.source_step)
    states = np.asarray(source["states"], dtype=np.float32)
    arm_pose = states[step, :6]
    tail_actions = None
    tail_start_step = None
    if args.replay_tail:
        tail_start_step = resolve_source_step(source, args.tail_start_step) if args.tail_start_step else step
        source_actions = np.asarray(source[args.tail_action_key], dtype=np.float32)
        tail_actions = source_actions[tail_start_step:]
        if args.tail_max_steps > 0:
            tail_actions = tail_actions[: args.tail_max_steps]
        if tail_actions.ndim != 2 or tail_actions.shape[1] != 7 or len(tail_actions) == 0:
            raise ValueError(f"invalid tail action slice with shape {tail_actions.shape}")
        latch_index = None
        if args.tail_latch_gripper:
            tail_actions, latch_index = latch_gripper_after_first_close(
                tail_actions,
                args.tail_latch_threshold,
                args.gripper_close,
            )
    object_position = trace_object_position(source, args.object, args.object_position)
    dataset_stats = (
        load_dataset_stats(args.dataset_stats, state_std_floor=args.state_std_floor)
        if args.dataset_stats
        else None
    )
    print(
        f"using {trace_path.name} step {step}: arm={np.round(arm_pose, 4).tolist()} "
        f"object={np.round(object_position, 6).tolist()}"
    )
    if args.replay_tail:
        print(
            f"replaying {len(tail_actions)} {args.tail_action_key} actions "
            f"from source step {tail_start_step}"
        )
        if args.tail_latch_gripper:
            print(
                f"tail gripper latch enabled: local index {latch_index}, "
                f"threshold {args.tail_latch_threshold}, close {args.gripper_close}"
            )

    client = RolloutClient(drain=True)
    results = []
    traces = []
    try:
        for index in range(1, args.repeat + 1):
            print(f"=== close-pose trial {index}/{args.repeat} ===")
            result, trial_trace = run_one(client, arm_pose, object_position, args, dataset_stats, tail_actions)
            result["trial"] = index
            result["source_trace"] = str(trace_path)
            result["source_step"] = step
            if args.replay_tail:
                result["tail_action_key"] = args.tail_action_key
                result["tail_start_step"] = tail_start_step
                result["tail_steps"] = int(len(tail_actions))
                result["tail_latch_gripper"] = bool(args.tail_latch_gripper)
                result["tail_latch_index"] = latch_index
            results.append(result)
            traces.append(trial_trace)
            print(json.dumps(result, ensure_ascii=False))
    finally:
        client.shutdown()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2) + "\n")
    if args.trace_dir:
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        for index, trial_trace in enumerate(traces, 1):
            if trial_trace is not None:
                np.savez(trace_dir / f"close_pose_{index:03d}.npz", **trial_trace)
    failures = sum(not item.get("success", False) for item in results)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
