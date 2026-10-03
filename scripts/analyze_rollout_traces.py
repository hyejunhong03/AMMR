#!/usr/bin/env python3
"""Classify SmolVLA rollout failures from rollout_policy.py trace files."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ammr_ik import tcp_from_joints  # noqa: E402
from ammr_mycobot_interface import (  # noqa: E402
    GRASP_EVENT_FIELD_NAMES,
    GRIPPER_GRASP_ENGAGE_RAD,
    OBJECT_INDEX_BY_NAME,
    OBJECT_NAMES,
    object_state_slice,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", help="Directory containing rollout_*.npz traces.")
    parser.add_argument("--results", default=None, help="Optional rollout_policy.py results JSON.")
    parser.add_argument("--object", choices=OBJECT_NAMES, default="red_cube", help="Target object.")
    parser.add_argument("--attach-distance", type=float, default=0.03, help="Reach threshold in metres.")
    parser.add_argument("--engage-threshold", type=float, default=GRIPPER_GRASP_ENGAGE_RAD)
    parser.add_argument("--wrist-error", type=float, default=0.10, help="j5/j6 state-action error threshold.")
    parser.add_argument("--wrist-velocity", type=float, default=0.5, help="j5/j6 velocity threshold in rad/s.")
    parser.add_argument("--action-jump", type=float, default=0.075, help="Command step threshold in rad.")
    parser.add_argument("--late-lift-low", type=float, default=0.015)
    parser.add_argument("--late-lift-high", type=float, default=0.02)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Classify lift successes as failures unless the grasp-event geometry is also valid.",
    )
    parser.add_argument(
        "--strict-attach-distance",
        type=float,
        default=0.02,
        help="Maximum attach-time grasp-point/object distance for a valid success.",
    )
    parser.add_argument(
        "--strict-lateral-offset",
        type=float,
        default=0.015,
        help="Maximum attach-time object offset in the gripper local x/z plane.",
    )
    parser.add_argument(
        "--strict-tool-axis-offset",
        type=float,
        default=0.02,
        help="Maximum attach-time object offset along the gripper local tool axis.",
    )
    parser.add_argument(
        "--event-pose-tolerance",
        type=float,
        default=0.03,
        help="Discard attach events whose recorded object position is this far from the rollout start pose.",
    )
    parser.add_argument("--out", default=None, help="Write classification JSON here.")
    return parser.parse_args()


def load_results(path):
    if not path:
        return {}
    payload = json.loads(Path(path).read_text())
    rows = payload.get("results", payload if isinstance(payload, list) else [])
    return {index + 1: row for index, row in enumerate(rows)}


def max_velocity(values, stamps):
    if len(values) < 2:
        return 0.0
    dt = np.diff(stamps)
    dt[dt <= 0.0] = 1e-3
    return float((np.abs(np.diff(values, axis=0)) / dt[:, None]).max())


def object_slice(objects, object_name):
    return objects[:, object_state_slice(object_name)]


def lift_for(objects_xyz):
    valid = np.isfinite(objects_xyz).all(axis=1)
    if not valid.any():
        return float("nan")
    values = objects_xyz[valid, 2]
    return float(values.max() - values[0])


def grasp_event_rows(data):
    if "grasp_events" not in data.files:
        return np.empty((0, len(GRASP_EVENT_FIELD_NAMES) + 1), dtype=np.float64)
    events = np.asarray(data["grasp_events"], dtype=np.float64)
    if events.ndim == 1:
        events = events.reshape(1, -1) if events.size else np.empty((0, 0), dtype=np.float64)
    return events


def first_attach_event(events, target_object, expected_object_position=None, pose_tolerance=0.03):
    if events.size == 0:
        return None
    field_index = {name: index + 1 for index, name in enumerate(GRASP_EVENT_FIELD_NAMES)}
    required_width = len(GRASP_EVENT_FIELD_NAMES) + 1
    if events.shape[1] < required_width:
        return None
    target_index = float(OBJECT_INDEX_BY_NAME[target_object])
    attach = events[
        (np.isclose(events[:, field_index["event_type"]], 1.0))
        & (np.isclose(events[:, field_index["object_index"]], target_index))
    ]
    if len(attach) == 0:
        return None
    if expected_object_position is not None:
        expected = np.asarray(expected_object_position, dtype=np.float64)
        object_columns = [
            field_index["object_x"],
            field_index["object_y"],
            field_index["object_z"],
        ]
        object_positions = attach[:, object_columns]
        pose_error = np.linalg.norm(object_positions - expected[None, :], axis=1)
        attach = attach[pose_error <= pose_tolerance]
        if len(attach) == 0:
            return None
    row = attach[0]
    return {
        "time": float(row[0]),
        "distance_m": float(row[field_index["distance_m"]]),
        "offset_m": float(row[field_index["offset_m"]]),
        "lateral_offset_m": float(row[field_index["lateral_offset_m"]]),
        "tool_axis_offset_m": float(row[field_index["tool_axis_offset_m"]]),
        "gripper_state": float(row[field_index["gripper_state"]]),
        "desired_gripper_target": float(row[field_index["desired_gripper_target"]]),
        "local_xyz": [
            float(row[field_index["local_x"]]),
            float(row[field_index["local_y"]]),
            float(row[field_index["local_z"]]),
        ],
        "object_xyz": [
            float(row[field_index["object_x"]]),
            float(row[field_index["object_y"]]),
            float(row[field_index["object_z"]]),
        ],
        "grasp_point_xyz": [
            float(row[field_index["grasp_point_x"]]),
            float(row[field_index["grasp_point_y"]]),
            float(row[field_index["grasp_point_z"]]),
        ],
    }


def invalid_success_reasons(
    success,
    attach_event,
    wrong_lift,
    wrist_velocity,
    arm_jump,
    args,
):
    reasons = []
    if not success:
        reasons.append("not_lift_success")
        return reasons
    if np.isfinite(wrong_lift) and wrong_lift >= args.late_lift_high:
        reasons.append("wrong_object_lift")
    if attach_event is None:
        reasons.append("missing_grasp_event")
        return reasons
    if attach_event["distance_m"] > args.strict_attach_distance:
        reasons.append("attach_distance")
    if attach_event["lateral_offset_m"] > args.strict_lateral_offset:
        reasons.append("lateral_offset")
    if attach_event["tool_axis_offset_m"] > args.strict_tool_axis_offset:
        reasons.append("tool_axis_offset")
    if wrist_velocity > args.wrist_velocity:
        reasons.append("wrist_velocity")
    if arm_jump > args.action_jump:
        reasons.append("action_jump")
    return reasons


def geometry_reasons(success, attach_event, wrong_lift, args):
    reasons = []
    if not success:
        reasons.append("not_lift_success")
        return reasons
    if np.isfinite(wrong_lift) and wrong_lift >= args.late_lift_high:
        reasons.append("wrong_object_lift")
    if attach_event is None:
        reasons.append("missing_grasp_event")
        return reasons
    if attach_event["distance_m"] > args.strict_attach_distance:
        reasons.append("attach_distance")
    if attach_event["lateral_offset_m"] > args.strict_lateral_offset:
        reasons.append("lateral_offset")
    if attach_event["tool_axis_offset_m"] > args.strict_tool_axis_offset:
        reasons.append("tool_axis_offset")
    return reasons


def classify(path, result, args):
    data = np.load(path)
    states = np.asarray(data["states"], dtype=np.float64)
    actions = np.asarray(data["actions"], dtype=np.float64)
    objects = np.asarray(data["objects"], dtype=np.float64)
    stamps = np.asarray(data["stamps"], dtype=np.float64)
    events = grasp_event_rows(data)

    if len(states) == 0 or len(actions) == 0 or len(objects) == 0:
        return {"trace": path.name, "failure_class": "unknown", "reason": "empty trace"}

    target_object = result.get("target_object") or result.get("object_name") or args.object
    target_xyz = object_slice(objects, target_object)
    valid_objects = np.isfinite(target_xyz).all(axis=1)
    if valid_objects.any():
        tcp = np.asarray([tcp_from_joints(state[:6]) for state in states], dtype=np.float64)
        distances = np.linalg.norm(tcp[valid_objects] - target_xyz[valid_objects], axis=1)
        min_distance = float(distances.min())
        lift = lift_for(target_xyz)
    else:
        min_distance = float("nan")
        lift = float("nan")
    other_lifts = {
        name: lift_for(object_slice(objects, name))
        for name in OBJECT_INDEX_BY_NAME
        if name != target_object
    }
    wrong_object = max(other_lifts, key=other_lifts.get) if other_lifts else None
    wrong_lift = other_lifts.get(wrong_object, float("nan")) if wrong_object else float("nan")

    gripper_min = float(states[:, 6].min())
    arm_jump = float(np.abs(np.diff(actions[:, :6], axis=0)).max()) if len(actions) > 1 else 0.0
    wrist_error = float(np.abs(states[:, 4:6] - actions[:, 4:6]).max())
    wrist_velocity = max_velocity(states[:, 4:6], stamps)
    expected_object_position = result.get("object_position") or result.get("cube_position")
    attach_event = first_attach_event(
        events,
        target_object,
        expected_object_position=expected_object_position,
        pose_tolerance=args.event_pose_tolerance,
    )

    success = bool(result.get("success", False))
    strict_geometry_reasons = geometry_reasons(success, attach_event, wrong_lift, args)
    geometry_valid_success = bool(success and not strict_geometry_reasons)
    stable_geometry_success = bool(geometry_valid_success and wrist_velocity <= args.wrist_velocity)
    strict_reasons = invalid_success_reasons(
        success, attach_event, wrong_lift, wrist_velocity, arm_jump, args
    )
    valid_success = bool(success and not strict_reasons)
    if args.strict and valid_success:
        failure_class = "valid_success"
    elif args.strict and success:
        failure_class = f"invalid_success_{strict_reasons[0]}"
    elif success:
        failure_class = "success"
    elif np.isfinite(wrong_lift) and wrong_lift >= args.late_lift_high:
        failure_class = "wrong_object_pick"
    elif np.isfinite(min_distance) and min_distance > args.attach_distance:
        failure_class = "no_reach"
    elif gripper_min > args.engage_threshold:
        failure_class = "gripper_not_closed"
    elif wrist_error > args.wrist_error or wrist_velocity > args.wrist_velocity:
        failure_class = "wrist_pop"
    elif arm_jump > args.action_jump:
        failure_class = "action_jump"
    elif np.isfinite(lift) and args.late_lift_low <= lift < args.late_lift_high:
        failure_class = "late_lift"
    else:
        failure_class = "unknown"

    return {
        "trace": path.name,
        "failure_class": failure_class,
        "success": success,
        "target_object": target_object,
        "wrong_object": wrong_object if np.isfinite(wrong_lift) else None,
        "wrong_object_lift_m": round(wrong_lift, 4) if np.isfinite(wrong_lift) else None,
        "lift_m": result.get("lift_m", round(lift, 4) if np.isfinite(lift) else None),
        "min_tcp_target_distance_m": round(min_distance, 5) if np.isfinite(min_distance) else None,
        "gripper_min": round(gripper_min, 5),
        "max_arm_action_jump": round(arm_jump, 5),
        "max_wrist_state_action_error": round(wrist_error, 5),
        "max_wrist_velocity": round(wrist_velocity, 5),
        "attach_event": (
            {
                key: (
                    round(value, 5)
                    if isinstance(value, float) and np.isfinite(value)
                    else value
                )
                for key, value in attach_event.items()
            }
            if attach_event is not None
            else None
        ),
        "geometry_valid_success": geometry_valid_success,
        "stable_geometry_success": stable_geometry_success,
        "valid_success": valid_success,
        "invalid_geometry_reasons": strict_geometry_reasons if success else [],
        "invalid_success_reasons": strict_reasons if success else [],
    }


def main():
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    paths = sorted(trace_dir.glob("rollout_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no rollout_*.npz traces in {trace_dir}")

    results = load_results(args.results)
    rows = []
    for path in paths:
        try:
            index = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            index = len(rows) + 1
        rows.append(classify(path, results.get(index, {}), args))

    counts = Counter(row["failure_class"] for row in rows)
    payload = {
        "counts": dict(sorted(counts.items())),
        "summary": {
            "lift_success": sum(1 for row in rows if row.get("success")),
            "geometry_valid_success": sum(1 for row in rows if row.get("geometry_valid_success")),
            "stable_geometry_success": sum(1 for row in rows if row.get("stable_geometry_success")),
            "valid_success": sum(1 for row in rows if row.get("valid_success")),
        },
        "rollouts": rows,
    }

    print(json.dumps(payload, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
