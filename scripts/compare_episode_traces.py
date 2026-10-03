#!/usr/bin/env python3
"""Compare recorded demos/replays/rollouts with the same TCP and phase metrics.

This is an offline diagnostic: it does not need Isaac Sim or ROS2.  It answers
whether a failed rollout already diverged during approach, or only around close
and lift, by computing the same TCP-to-object distance and phase events for each
source trace.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ammr_ik import tcp_from_joints  # noqa: E402
from ammr_mycobot_interface import GRIPPER_OPEN_RAD, OBJECT_NAMES, object_state_slice  # noqa: E402


def parse_named_path(value):
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected NAME:PATH")
    name, path = value.split(":", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Source name must not be empty")
    return name, Path(path).expanduser().resolve()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        action="append",
        type=parse_named_path,
        default=[],
        help="Recorded raw episode as NAME:/path/to/episode_*. May be repeated.",
    )
    parser.add_argument(
        "--trace",
        action="append",
        type=parse_named_path,
        default=[],
        help="Replay/rollout trace NPZ as NAME:/path/to/trace.npz. May be repeated.",
    )
    parser.add_argument("--object", choices=OBJECT_NAMES, default="red_cube")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--gripper-open", type=float, default=GRIPPER_OPEN_RAD)
    parser.add_argument("--gripper-close", type=float, default=-0.245)
    parser.add_argument("--gripper-decrease-eps", type=float, default=0.002)
    parser.add_argument("--gripper-closed-threshold", type=float, default=-0.20)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--lift-threshold", type=float, default=0.02)
    parser.add_argument("--out", required=True, help="Summary JSON path.")
    parser.add_argument("--csv", default=None, help="Optional per-step CSV path.")
    return parser.parse_args()


def _load_stamps(path, length, fps):
    for name in ("timestamps.npy", "state_timestamp.npy", "action_timestamp.npy"):
        stamp_path = path / name
        if stamp_path.exists():
            stamps = np.asarray(np.load(stamp_path), dtype=np.float64)
            if stamps.ndim == 1 and len(stamps) >= length:
                return stamps[:length]
    return np.arange(length, dtype=np.float64) / max(float(fps), 1e-6)


def load_episode(name, path, object_name, fps):
    states = np.asarray(np.load(path / "robot_state.npy"), dtype=np.float64)
    actions = np.asarray(np.load(path / "action.npy"), dtype=np.float64)
    objects = np.asarray(np.load(path / "object_state.npy"), dtype=np.float64)
    length = min(len(states), len(actions), len(objects))
    stamps = _load_stamps(path, length, fps)
    return {
        "name": name,
        "kind": "episode",
        "path": str(path),
        "states": states[:length],
        "actions": actions[:length],
        "objects": objects[:length],
        "stamps": stamps[:length],
        "extra": {},
    }


def load_trace(name, path):
    data = np.load(path, allow_pickle=False)
    states = np.asarray(data["states"], dtype=np.float64)
    actions = np.asarray(data["actions"], dtype=np.float64)
    objects = np.asarray(data["objects"], dtype=np.float64)
    length = min(len(states), len(actions), len(objects))
    stamps = np.asarray(data["stamps"], dtype=np.float64)[:length] if "stamps" in data else np.arange(length)
    extra = {}
    for key in (
        "state_z",
        "action_z",
        "raw_action_z",
        "policy_raw_action_z",
        "target_state_error",
        "raw_target_state_error",
        "policy_target_state_error",
        "policy_raw_actions",
        "raw_actions",
        "action_limited",
        "action_clamped",
        "gripper_snapped",
    ):
        if key in data:
            extra[key] = np.asarray(data[key])[:length]
    return {
        "name": name,
        "kind": "trace",
        "path": str(path),
        "states": states[:length],
        "actions": actions[:length],
        "objects": objects[:length],
        "stamps": stamps[:length],
        "extra": extra,
    }


def first_index(mask):
    mask = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def step_time(stamps, step):
    if step is None or len(stamps) == 0:
        return None
    return float(stamps[step] - stamps[0])


def max_velocity(states, stamps):
    if len(states) < 2:
        return {"value": 0.0, "step": None, "joint_1based": None}
    dt = np.diff(stamps)
    if len(dt) != len(states) - 1 or not np.isfinite(dt).all():
        dt = np.full(len(states) - 1, np.nan)
    good_dt = dt > 1e-6
    if not good_dt.any():
        dt = np.full(len(states) - 1, 1.0)
        good_dt = np.ones(len(states) - 1, dtype=bool)
    velocity = np.full((len(states) - 1, 6), np.nan, dtype=np.float64)
    velocity[good_dt] = np.abs(np.diff(states[:, :6], axis=0)[good_dt] / dt[good_dt, None])
    if not np.isfinite(velocity).any():
        return {"value": float("nan"), "step": None, "joint_1based": None}
    flat = int(np.nanargmax(velocity))
    step, joint = np.unravel_index(flat, velocity.shape)
    return {
        "value": float(velocity[step, joint]),
        "step": int(step + 1),
        "joint_1based": int(joint + 1),
    }


def first_abs_gt(array, threshold):
    if array is None or len(array) == 0:
        return None
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        mask = np.abs(values) > threshold
        idx = first_index(mask)
        return None if idx is None else {"step": idx, "dim_1based": None, "value": float(values[idx])}
    mask = np.abs(values) > threshold
    hits = np.argwhere(mask)
    if len(hits) == 0:
        return None
    step, dim = hits[0]
    return {
        "step": int(step),
        "dim_1based": int(dim + 1),
        "value": float(values[step, dim]),
    }


def max_abs(array):
    if array is None or len(array) == 0:
        return None
    values = np.asarray(array, dtype=np.float64)
    if not np.isfinite(values).any():
        return None
    flat = int(np.nanargmax(np.abs(values)))
    if values.ndim == 1:
        return {"step": flat, "dim_1based": None, "value": float(values[flat])}
    step, dim = np.unravel_index(flat, values.shape)
    return {
        "step": int(step),
        "dim_1based": int(dim + 1),
        "value": float(values[step, dim]),
    }


def summarize(source, args):
    states = source["states"]
    actions = source["actions"]
    objects = source["objects"]
    stamps = source["stamps"]
    extra = source["extra"]
    target = objects[:, object_state_slice(args.object)]
    valid_target = np.isfinite(target).all(axis=1)
    tcp = np.asarray([tcp_from_joints(row[:6]) for row in states], dtype=np.float64)
    distance = np.full(len(states), np.nan, dtype=np.float64)
    distance[valid_target] = np.linalg.norm(tcp[valid_target] - target[valid_target], axis=1)

    if valid_target.any():
        valid_indices = np.flatnonzero(valid_target)
        min_distance_local = int(np.nanargmin(distance[valid_target]))
        min_distance_step = int(valid_indices[min_distance_local])
        first_target = target[valid_indices[0]]
        lift = target[:, 2] - first_target[2]
    else:
        min_distance_step = None
        lift = np.full(len(states), np.nan, dtype=np.float64)

    gripper_action = actions[:, 6]
    gripper_state = states[:, 6]
    midpoint = (float(args.gripper_open) + float(args.gripper_close)) / 2.0
    first_decrease = first_index(gripper_action < float(args.gripper_open) - float(args.gripper_decrease_eps))
    first_mid = first_index(gripper_action < midpoint)
    first_closed = first_index(gripper_action < float(args.gripper_closed_threshold))
    first_lift = first_index(lift >= float(args.lift_threshold))

    state_z_gt = first_abs_gt(extra.get("state_z"), args.z_threshold)
    action_z_gt = first_abs_gt(extra.get("action_z"), args.z_threshold)
    raw_action_z_gt = first_abs_gt(extra.get("raw_action_z"), args.z_threshold)
    policy_action_z_gt = first_abs_gt(extra.get("policy_raw_action_z"), args.z_threshold)

    summary = {
        "name": source["name"],
        "kind": source["kind"],
        "path": source["path"],
        "frames": int(len(states)),
        "duration_s": float(stamps[-1] - stamps[0]) if len(stamps) else 0.0,
        "object": args.object,
        "target_start_xyz": target[0].tolist() if len(target) else None,
        "target_final_xyz": target[-1].tolist() if len(target) else None,
        "target_lift_max_m": float(np.nanmax(lift)) if np.isfinite(lift).any() else None,
        "target_lift_final_m": float(lift[-1]) if len(lift) and np.isfinite(lift[-1]) else None,
        "tcp_target_distance_min_m": (
            float(distance[min_distance_step]) if min_distance_step is not None else None
        ),
        "tcp_target_distance_min_step": min_distance_step,
        "tcp_target_distance_min_time_s": step_time(stamps, min_distance_step),
        "tcp_target_distance_at_first_gripper_decrease_m": (
            float(distance[first_decrease]) if first_decrease is not None and np.isfinite(distance[first_decrease]) else None
        ),
        "tcp_target_distance_at_mid_close_m": (
            float(distance[first_mid]) if first_mid is not None and np.isfinite(distance[first_mid]) else None
        ),
        "tcp_target_distance_at_closed_threshold_m": (
            float(distance[first_closed]) if first_closed is not None and np.isfinite(distance[first_closed]) else None
        ),
        "phase_steps": {
            "first_gripper_decrease": first_decrease,
            "first_gripper_below_midpoint": first_mid,
            "first_gripper_below_closed_threshold": first_closed,
            "first_lift_threshold": first_lift,
            "state_z_abs_gt_threshold": state_z_gt,
            "action_z_abs_gt_threshold": action_z_gt,
            "raw_action_z_abs_gt_threshold": raw_action_z_gt,
            "policy_raw_action_z_abs_gt_threshold": policy_action_z_gt,
        },
        "phase_times_s": {
            "first_gripper_decrease": step_time(stamps, first_decrease),
            "first_gripper_below_midpoint": step_time(stamps, first_mid),
            "first_gripper_below_closed_threshold": step_time(stamps, first_closed),
            "first_lift_threshold": step_time(stamps, first_lift),
        },
        "gripper_action_min": float(np.nanmin(gripper_action)),
        "gripper_action_max": float(np.nanmax(gripper_action)),
        "gripper_state_min": float(np.nanmin(gripper_state)),
        "gripper_state_max": float(np.nanmax(gripper_state)),
        "max_joint_velocity": max_velocity(states, stamps),
        "target_state_error_abs_max": max_abs(extra.get("target_state_error")),
        "raw_target_state_error_abs_max": max_abs(extra.get("raw_target_state_error")),
        "policy_target_state_error_abs_max": max_abs(extra.get("policy_target_state_error")),
        "state_z_abs_max": max_abs(extra.get("state_z")),
        "action_z_abs_max": max_abs(extra.get("action_z")),
        "raw_action_z_abs_max": max_abs(extra.get("raw_action_z")),
        "policy_raw_action_z_abs_max": max_abs(extra.get("policy_raw_action_z")),
    }
    series = {
        "tcp": tcp,
        "target": target,
        "distance": distance,
        "lift": lift,
        "gripper_action": gripper_action,
        "gripper_state": gripper_state,
    }
    return summary, series


def comparison(reference, other):
    n = min(len(reference["series"]["distance"]), len(other["series"]["distance"]))
    if n == 0:
        return {"frames_compared": 0}
    ref_distance = reference["series"]["distance"][:n]
    other_distance = other["series"]["distance"][:n]
    distance_delta = other_distance - ref_distance
    ref_state = reference["source"]["states"][:n, :6]
    other_state = other["source"]["states"][:n, :6]
    ref_action = reference["source"]["actions"][:n, :6]
    other_action = other["source"]["actions"][:n, :6]
    state_linf = np.max(np.abs(other_state - ref_state), axis=1)
    action_linf = np.max(np.abs(other_action - ref_action), axis=1)
    return {
        "frames_compared": int(n),
        "tcp_distance_delta_p50_m": float(np.nanpercentile(distance_delta, 50)),
        "tcp_distance_delta_p95_m": float(np.nanpercentile(distance_delta, 95)),
        "tcp_distance_delta_max_m": float(np.nanmax(distance_delta)),
        "arm_state_linf_p50_rad": float(np.nanpercentile(state_linf, 50)),
        "arm_state_linf_p95_rad": float(np.nanpercentile(state_linf, 95)),
        "arm_state_linf_max_rad": float(np.nanmax(state_linf)),
        "arm_action_linf_p50_rad": float(np.nanpercentile(action_linf, 50)),
        "arm_action_linf_p95_rad": float(np.nanpercentile(action_linf, 95)),
        "arm_action_linf_max_rad": float(np.nanmax(action_linf)),
        "gripper_action_abs_p95_rad": float(
            np.nanpercentile(np.abs(other["source"]["actions"][:n, 6] - reference["source"]["actions"][:n, 6]), 95)
        ),
    }


def write_csv(path, records):
    fieldnames = [
        "source",
        "kind",
        "step",
        "time_s",
        "target_x",
        "target_y",
        "target_z",
        "tcp_x",
        "tcp_y",
        "tcp_z",
        "tcp_target_distance_m",
        "target_lift_m",
        "gripper_action",
        "gripper_state",
        "state_z_abs_max",
        "action_z_abs_max",
        "target_state_error_abs_max",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            source = record["source"]
            series = record["series"]
            stamps = source["stamps"]
            extra = source["extra"]
            state_z = extra.get("state_z")
            action_z = extra.get("action_z")
            target_error = extra.get("target_state_error")
            for step in range(len(source["states"])):
                row = {
                    "source": source["name"],
                    "kind": source["kind"],
                    "step": step,
                    "time_s": float(stamps[step] - stamps[0]) if len(stamps) else 0.0,
                    "target_x": float(series["target"][step, 0]),
                    "target_y": float(series["target"][step, 1]),
                    "target_z": float(series["target"][step, 2]),
                    "tcp_x": float(series["tcp"][step, 0]),
                    "tcp_y": float(series["tcp"][step, 1]),
                    "tcp_z": float(series["tcp"][step, 2]),
                    "tcp_target_distance_m": float(series["distance"][step]),
                    "target_lift_m": float(series["lift"][step]),
                    "gripper_action": float(series["gripper_action"][step]),
                    "gripper_state": float(series["gripper_state"][step]),
                    "state_z_abs_max": "",
                    "action_z_abs_max": "",
                    "target_state_error_abs_max": "",
                }
                if state_z is not None:
                    row["state_z_abs_max"] = float(np.nanmax(np.abs(state_z[step])))
                if action_z is not None:
                    row["action_z_abs_max"] = float(np.nanmax(np.abs(action_z[step])))
                if target_error is not None:
                    row["target_state_error_abs_max"] = float(np.nanmax(np.abs(target_error[step])))
                writer.writerow(row)


def main():
    args = parse_args()
    if not args.episode and not args.trace:
        raise SystemExit("Provide at least one --episode or --trace")

    sources = []
    for name, path in args.episode:
        sources.append(load_episode(name, path, args.object, args.fps))
    for name, path in args.trace:
        sources.append(load_trace(name, path))

    records = []
    for source in sources:
        summary, series = summarize(source, args)
        records.append({"source": source, "summary": summary, "series": series})

    reference = records[0]
    comparisons = {
        record["source"]["name"]: comparison(reference, record)
        for record in records[1:]
    }
    report = {
        "reference": reference["source"]["name"],
        "object": args.object,
        "gripper_open": args.gripper_open,
        "gripper_close": args.gripper_close,
        "gripper_midpoint": (args.gripper_open + args.gripper_close) / 2.0,
        "summaries": [record["summary"] for record in records],
        "comparisons_to_reference": comparisons,
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(csv_path, records)
    print(json.dumps(report["summaries"], indent=2))
    print(f"Wrote {out_path}")
    if args.csv:
        print(f"Wrote {Path(args.csv).expanduser().resolve()}")


if __name__ == "__main__":
    main()
