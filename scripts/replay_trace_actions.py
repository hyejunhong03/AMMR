#!/usr/bin/env python3
"""Replay saved rollout trace actions against the Isaac ROS2 scene.

This isolates simulator/contact determinism from policy inference. Use it with a
failed rollout trace, for example rollout_005.npz, to answer whether the same
published action sequence reproduces the same joint/state spike after reset.
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
    event_diagnostics,
    load_dataset_stats,
    object_xyz,
    start_motion_diagnostics,
    start_object_diagnostics,
    start_state_diagnostics,
    tcp_diagnostics,
    velocity_diagnostics,
    z_scores,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="rollout_*.npz trace to replay.")
    parser.add_argument("--object", choices=OBJECT_NAMES, default="red_cube")
    parser.add_argument(
        "--action-key",
        choices=("actions", "raw_actions", "policy_raw_actions"),
        default="actions",
        help="Trace array to publish. Use actions for exact published-action replay.",
    )
    parser.add_argument(
        "--object-position",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override object reset position. Defaults to trace objects[0].",
    )
    parser.add_argument("--repeat", type=int, default=3, help="Number of replay trials.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means full trace.")
    parser.add_argument("--settle-sec", type=float, default=6.0)
    parser.add_argument("--park-non-targets", action="store_true")
    parser.add_argument("--min-lift-m", type=float, default=0.02)
    parser.add_argument("--max-joint-velocity", type=float, default=0.5)
    parser.add_argument("--dataset-stats", default=None)
    parser.add_argument("--state-std-floor", type=float, default=0.0)
    parser.add_argument("--start-state-z-threshold", type=float, default=3.0)
    parser.add_argument("--start-object-position-threshold", type=float, default=0.005)
    parser.add_argument("--start-settle-check-sec", type=float, default=0.5)
    parser.add_argument("--start-settle-sample-sec", type=float, default=0.05)
    parser.add_argument("--start-settle-velocity-threshold", type=float, default=0.05)
    parser.add_argument("--reset-retries", type=int, default=2)
    parser.add_argument("--state-jump-event-threshold", type=float, default=0.2)
    parser.add_argument("--smooth-command-threshold", type=float, default=0.02)
    parser.add_argument("--stuck-duration-sec", type=float, default=0.5)
    parser.add_argument("--stuck-state-motion-threshold", type=float, default=0.01)
    parser.add_argument("--stuck-tracking-error-threshold", type=float, default=0.15)
    parser.add_argument("--stuck-command-motion-threshold", type=float, default=0.03)
    parser.add_argument("--stuck-error-growth-threshold", type=float, default=0.03)
    parser.add_argument("--out", default=None, help="Write JSON results.")
    parser.add_argument("--trace-dir", default=None, help="Write replay trace NPZ files.")
    return parser.parse_args()


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


def run_one(client, actions, object_position, args, dataset_stats):
    reset_diagnostics = []
    feedback_ok = False
    start_diag = {"checked": False, "ok": True}
    start_object_diag = {"checked": False, "ok": True}
    start_motion_diag = {"checked": False, "ok": True}
    start_ok = True
    for attempt in range(int(args.reset_retries) + 1):
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
        start_diag = start_state_diagnostics(
            client.state,
            dataset_stats,
            args.start_state_z_threshold,
        )
        start_object_diag = start_object_diagnostics(
            client.objects,
            args.object,
            object_position,
            args.start_object_position_threshold,
        )
        settle_samples, settle_stamps = client.sample_state_motion(
            args.start_settle_check_sec,
            args.start_settle_sample_sec,
        )
        start_motion_diag = start_motion_diagnostics(
            settle_samples,
            settle_stamps,
            args.start_settle_velocity_threshold,
        )
        start_ok = bool(start_diag["ok"] and start_object_diag["ok"])
        reset_diagnostics.append(
            {
                "attempt": attempt + 1,
                **start_diag,
                "start_object": start_object_diag,
                "start_motion": start_motion_diag,
                "start_ok": start_ok,
            }
        )
        if start_ok:
            break

    if not feedback_ok or not start_ok:
        if not feedback_ok:
            error = "no image/state feedback"
        elif not start_diag["ok"]:
            error = "reset_state_out_of_distribution"
        elif not start_object_diag["ok"]:
            error = "reset_object_position_error"
        else:
            error = "reset_start_invalid"
        return {
            "object_name": args.object,
            "object_position": object_position.tolist(),
            "invalid_replay": True,
            "error": error,
            "reset_attempts": len(reset_diagnostics),
            "reset_diagnostics": reset_diagnostics,
            "start_state_z_max": start_diag if start_diag.get("checked") else None,
            "start_object_position": start_object_diag if start_object_diag.get("checked") else None,
            "start_motion": start_motion_diag if start_motion_diag.get("checked") else None,
            "success": False,
        }, None

    client.clear_grasp_events()
    start_objects = client.objects.copy() if client.objects is not None else None
    start_target = object_xyz(start_objects, args.object)
    period = 1.0 / max(args.fps, 1e-6)
    next_deadline = time.monotonic()

    states = []
    objects = []
    stamps = []
    published = []
    for command in actions:
        client.spin(0.0)
        states.append(client.state.copy() if client.state is not None else np.full((7,), np.nan, dtype=np.float32))
        objects.append(
            client.objects.copy()
            if client.objects is not None
            else np.full((len(OBJECT_NAMES) * 3,), np.nan, dtype=np.float32)
        )
        published.append(command.copy())
        stamps.append(time.monotonic())
        client.publish(command)

        now = time.monotonic()
        next_deadline += period
        if next_deadline < now:
            next_deadline = now + period
        sleep_s = next_deadline - now
        if sleep_s > 0.0:
            time.sleep(sleep_s)

    states = np.asarray(states, dtype=np.float32)
    objects = np.asarray(objects, dtype=np.float32)
    published = np.asarray(published, dtype=np.float32)
    stamps = np.asarray(stamps, dtype=np.float64)
    tcp_distances, tcp_positions = tcp_diagnostics(states, objects, args.object)
    raw_actions = published.copy()
    trace = {
        "states": states,
        "actions": published,
        "raw_actions": raw_actions,
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
                "action_z": z_scores(published, dataset_stats["action"]),
                "raw_action_z": z_scores(raw_actions, dataset_stats["action"]),
                "target_state_error": published - states,
                "raw_target_state_error": raw_actions - states,
            }
        )

    target_positions = objects[:, object_state_slice(args.object)] if len(objects) else np.empty((0, 3))
    start_z = float(start_target[2]) if start_target is not None else float(object_position[2])
    finite_z = target_positions[:, 2][np.isfinite(target_positions[:, 2])] if len(target_positions) else []
    lift_m = float(np.max(finite_z) - start_z) if len(finite_z) else None

    result = {
        "object_name": args.object,
        "object_position": object_position.tolist(),
        "action_key": args.action_key,
        "steps": int(len(published)),
        "fps": float(args.fps),
        "reset_attempts": len(reset_diagnostics),
        "reset_diagnostics": reset_diagnostics,
        "start_state_z_max": start_diag if start_diag.get("checked") else None,
        "start_object_position": start_object_diag if start_object_diag.get("checked") else None,
        "start_motion": start_motion_diag if start_motion_diag.get("checked") else None,
        "lift_m": round(lift_m, 4) if lift_m is not None else None,
        "success": bool(lift_m is not None and lift_m >= args.min_lift_m),
    }
    if len(tcp_distances) and np.any(np.isfinite(tcp_distances)):
        result["tcp_target_distance_min_m"] = float(np.nanmin(tcp_distances))
        result["tcp_target_distance_final_m"] = float(tcp_distances[-1])
    if len(states) > 1:
        result.update(
            velocity_diagnostics(
                states,
                published,
                raw_actions,
                stamps,
                client.grasp_events,
                args.max_joint_velocity,
                max_action_step=0.0,
            )
        )
        result.update(
            event_diagnostics(
                states,
                published,
                raw_actions,
                stamps,
                state_jump_threshold=args.state_jump_event_threshold,
                smooth_command_threshold=args.smooth_command_threshold,
                stuck_duration_sec=args.stuck_duration_sec,
                stuck_state_motion_threshold=args.stuck_state_motion_threshold,
                stuck_tracking_error_threshold=args.stuck_tracking_error_threshold,
                stuck_command_motion_threshold=args.stuck_command_motion_threshold,
                stuck_error_growth_threshold=args.stuck_error_growth_threshold,
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
    actions = np.asarray(source[args.action_key], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"{args.action_key} must have shape [T, 7], got {actions.shape}")
    if args.max_steps > 0:
        actions = actions[: args.max_steps]
    object_position = trace_object_position(source, args.object, args.object_position)
    dataset_stats = (
        load_dataset_stats(args.dataset_stats, state_std_floor=args.state_std_floor)
        if args.dataset_stats
        else None
    )

    client = RolloutClient(drain=True)
    results = []
    traces = []
    try:
        for index in range(1, args.repeat + 1):
            print(f"=== replay {index}/{args.repeat} {trace_path.name} {args.object} at {object_position.tolist()} ===")
            result, replay_trace = run_one(client, actions, object_position, args, dataset_stats)
            result["trial"] = index
            result["source_trace"] = str(trace_path)
            results.append(result)
            traces.append(replay_trace)
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
        for index, replay_trace in enumerate(traces, 1):
            if replay_trace is not None:
                np.savez(trace_dir / f"replay_{index:03d}.npz", **replay_trace)
    failures = sum(not item.get("success", False) for item in results)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
