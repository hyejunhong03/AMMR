#!/usr/bin/env python3
"""Validate scripted dynamic-contact picks over a fixed set of positions.

This is not a data collector. It keeps script exit status separate from data-quality
status and classifies each placement before it can become a training candidate:

- planning_failed: no usable approach plan in the pose cache.
- open_approach_contact: the open gripper touched/moved the object before close.
- full_preclose_contact: the full pick touched/moved the object before close.
- diagnostics_invalid: required diagnostics were missing or invalid.
- full_pick_success: clean open approach and lift success.
- full_pick_failed: clean open approach but the full pick did not pass.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from ammr_mycobot_interface import GRIPPER_OPEN_RAD, OBJECT_INDEX_BY_NAME, OBJECT_NAMES

SCRIPTS_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-file", required=True, help="JSON list of [x, y, z] positions.")
    parser.add_argument("--poses-file", required=True, help="Pose cache to use for runnable positions.")
    parser.add_argument("--diagnostics-csv", required=True, help="Isaac physics diagnostics CSV written by the sim.")
    parser.add_argument("--out", required=True, help="Where to write the validation JSON summary.")
    parser.add_argument("--log-dir", required=True, help="Directory for per-position demo logs.")
    parser.add_argument("--object", choices=OBJECT_NAMES, default="red_cube")
    parser.add_argument("--close", type=float, default=-0.245)
    parser.add_argument("--arm-speed", type=float, default=0.18)
    parser.add_argument("--lift-arm-speed", type=float, default=0.04)
    parser.add_argument("--arm-tolerance", type=float, default=0.03)
    parser.add_argument("--gripper-tolerance", type=float, default=0.12)
    parser.add_argument("--object-position-guard-tolerance", type=float, default=0.003)
    parser.add_argument("--open-contact-max-rows", type=int, default=0)
    parser.add_argument("--open-contact-max-force-n", type=float, default=0.05)
    parser.add_argument("--open-contact-max-xy-mm", type=float, default=0.5)
    parser.add_argument("--open-contact-max-z-mm", type=float, default=0.5)
    parser.add_argument(
        "--open-contact-penetration-tolerance-m",
        type=float,
        default=0.0001,
        help="Negative separation beyond this during open/pre-close approach counts as contact, not proximity.",
    )
    parser.add_argument("--min-lift-mm", type=float, default=20.0)
    parser.add_argument(
        "--min-hold-sec",
        type=float,
        default=0.2,
        help="Continuous sim-time hold above the lift threshold required for a training candidate.",
    )
    parser.add_argument(
        "--max-hold-gap-sec",
        type=float,
        default=0.25,
        help="Break a lift-hold segment if consecutive diagnostics samples are farther apart than this.",
    )
    parser.add_argument("--reset-settle-sec", type=float, default=3.0)
    parser.add_argument("--between-run-sec", type=float, default=1.0)
    parser.add_argument("--gripper-open", type=float, default=GRIPPER_OPEN_RAD)
    parser.add_argument(
        "--close-start-drop-rad",
        type=float,
        default=0.002,
        help="Full-run pre-close ends when gripper target first drops this far below the open target.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Validate only the first N positions.")
    parser.add_argument(
        "--position-indices",
        default=None,
        help="Comma-separated 1-based source position indices to validate, preserving the original index.",
    )
    return parser.parse_args()


def ros_env():
    env = dict(os.environ)
    env.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    return env


def run(cmd, *, timeout=None, capture=True):
    return subprocess.run(
        cmd,
        cwd=SCRIPTS_DIR,
        env=ros_env(),
        timeout=timeout,
        capture_output=capture,
        text=True,
        check=False,
    )


def load_positions(path):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be a JSON list of positions")
    positions = []
    for value in payload:
        if len(value) != 3:
            raise ValueError(f"invalid position: {value}")
        positions.append([float(v) for v in value])
    return positions


def load_pose_entries(path):
    payload = json.loads(Path(path).read_text())
    by_key = {}
    for entry in payload.get("entries", []):
        position = entry.get("object_position") or entry.get("sampled_object_pose") or entry.get("cube_position")
        if position is None:
            continue
        by_key[position_key(position)] = entry
    return payload, by_key


def position_key(position):
    return tuple(round(float(v), 9) for v in position)


def parse_position_indices(value, count):
    if not value:
        return None
    indices = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"invalid descending range in --position-indices: {item}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(item))
    bad = [index for index in indices if index < 1 or index > count]
    if bad:
        raise ValueError(f"--position-indices out of range 1..{count}: {bad}")
    seen = set()
    unique = []
    for index in indices:
        if index not in seen:
            seen.add(index)
            unique.append(index)
    return unique


def csv_line_count(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def place_and_reset(position, object_name, settle_sec):
    object_index = OBJECT_INDEX_BY_NAME[object_name]
    data = ", ".join(str(float(v)) for v in [object_index, *position])
    run(
        [
            "ros2",
            "topic",
            "pub",
            "--once",
            "/ammr/set_object_pose",
            "std_msgs/Float64MultiArray",
            f"{{data: [{data}]}}",
        ],
        timeout=15.0,
    )
    time.sleep(1.0)
    run(["ros2", "topic", "pub", "--once", "/ammr/reset_task_scene", "std_msgs/Empty", "{}"], timeout=15.0)
    time.sleep(settle_sec)


def demo_command(args, position, *, stop_after_approach):
    cmd = [
        sys.executable,
        "scripted_pick_demo.py",
        "--object",
        args.object,
        "--poses-file",
        str(Path(args.poses_file).resolve()),
        "--no-check-cube",
        "--expected-object-position",
        *[str(float(v)) for v in position],
        "--object-position-guard-tolerance",
        str(args.object_position_guard_tolerance),
        "--object-position-guard-samples",
        "2",
        "--close",
        str(args.close),
        "--arm-tolerance",
        str(args.arm_tolerance),
        "--gripper-tolerance",
        str(args.gripper_tolerance),
        "--arm-speed",
        str(args.arm_speed),
        "--lift-arm-speed",
        str(args.lift_arm_speed),
    ]
    if stop_after_approach:
        cmd += ["--stop-after-approach", "--final-hold", "1.0"]
    else:
        cmd += ["--lift-hold", "3.0", "--place-after-lift", "--place-hold", "0.5", "--final-hold", "0.5"]
    return cmd


def run_demo(args, index, position, *, stop_after_approach, log_dir):
    phase = "open" if stop_after_approach else "full"
    log_path = log_dir / f"position_{index:02d}_{phase}.log"
    result = run(demo_command(args, position, stop_after_approach=stop_after_approach), timeout=160.0)
    text = (result.stdout or "") + (result.stderr or "")
    log_path.write_text(text, encoding="utf-8")
    return result.returncode, log_path


def f(row, key, default=0.0):
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return default if math.isnan(value) else value


def required_float(row, key):
    if key not in row:
        return None
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def read_diagnostic_rows(path, start_line):
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    # start_line is wc -l including the header; DictReader rows are data-only.
    start_index = max(0, start_line - 1)
    return rows[start_index:]


def object_prefix(object_name):
    return f"object.{object_name}"


def rows_for_position(rows, position, object_name):
    if not rows:
        return rows
    prefix = object_prefix(object_name)
    start = None
    for idx, row in enumerate(rows):
        if (
            abs(f(row, f"{prefix}.x") - position[0]) < 1e-5
            and abs(f(row, f"{prefix}.y") - position[1]) < 1e-5
            and abs(f(row, f"{prefix}.z") - position[2]) < 1e-5
        ):
            start = idx
            break
    return rows[start:] if start is not None else rows


def rows_before_close_start(rows, gripper_open, close_start_drop_rad):
    if not rows:
        return rows
    reference_candidates = [
        required_float(row, "dof.gripper_controller.target")
        for row in rows[: min(len(rows), 20)]
    ]
    reference_candidates = [value for value in reference_candidates if value is not None]
    reference_open = max(reference_candidates + [float(gripper_open)])
    close_start_threshold = reference_open - abs(float(close_start_drop_rad))
    for idx, row in enumerate(rows):
        target = required_float(row, "dof.gripper_controller.target")
        if target is not None and target < close_start_threshold:
            return rows[:idx]
    return rows


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
            and (
                float(times[index]) < float(times[last_index])
                or float(times[index] - times[last_index]) > max_gap_sec
            )
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


def lift_hold_seconds(rows, object_name, min_lift_mm, max_hold_gap_sec):
    if not rows:
        return 0.0
    prefix = object_prefix(object_name)
    initial_z = f(rows[0], f"{prefix}.z")
    threshold_z = initial_z + float(min_lift_mm) / 1000.0
    times = [f(row, "sim_time_s") for row in rows]
    above = [f(row, f"{prefix}.z") >= threshold_z for row in rows]
    return _max_continuous_hold(times, above, max_hold_gap_sec)


def summarize_rows(rows, object_name, *, min_lift_mm=None, max_hold_gap_sec=None):
    if not rows:
        return {"rows": 0, "diagnostics_valid": False, "diagnostics_errors": ["no_rows"]}
    prefix = object_prefix(object_name)
    diagnostics_errors = diagnostics_errors_for_rows(rows, object_name)
    x0 = f(rows[0], f"{prefix}.x")
    y0 = f(rows[0], f"{prefix}.y")
    z0 = f(rows[0], f"{prefix}.z")
    open_rows = [row for row in rows if f(row, "dof.gripper_controller.target") > 0.079]
    close_rows = [row for row in rows if f(row, "dof.gripper_controller.target") < 0.0]
    arm_vel_keys = [key for key in rows[0] if key.startswith("dof.") and key.endswith(".vel") and "gripper" not in key]
    mimic_keys = [key for key in rows[0] if key.startswith("mimic_error.")]
    summary = {
        "rows": len(rows),
        "diagnostics_valid": not diagnostics_errors,
        "diagnostics_errors": diagnostics_errors[:8],
        "duration_sim_s": f(rows[-1], "sim_time_s") - f(rows[0], "sim_time_s"),
        "lift_mm": (max(f(row, f"{prefix}.z") for row in rows) - z0) * 1000.0,
        "end_z_delta_mm": (f(rows[-1], f"{prefix}.z") - z0) * 1000.0,
        "max_z_disp_mm": max(abs(f(row, f"{prefix}.z") - z0) for row in rows) * 1000.0,
        "max_xy_disp_mm": max(
            ((f(row, f"{prefix}.x") - x0) ** 2 + (f(row, f"{prefix}.y") - y0) ** 2) ** 0.5
            for row in rows
        )
        * 1000.0,
        "end_xy_disp_mm": (
            ((f(rows[-1], f"{prefix}.x") - x0) ** 2 + (f(rows[-1], f"{prefix}.y") - y0) ** 2)
            ** 0.5
        )
        * 1000.0,
        "open_left_contact_rows": sum(
            1 for row in open_rows if f(row, f"{prefix}.contact.left_finger.count") > 0
        ),
        "open_right_contact_rows": sum(
            1 for row in open_rows if f(row, f"{prefix}.contact.right_finger.count") > 0
        ),
        "open_contact_rows": sum(
            1
            for row in open_rows
            if (
                f(row, f"{prefix}.contact.left_finger.count") > 0
                or f(row, f"{prefix}.contact.right_finger.count") > 0
            )
        ),
        "close_left_contact_rows": sum(
            1 for row in close_rows if f(row, f"{prefix}.contact.left_finger.count") > 0
        ),
        "close_right_contact_rows": sum(
            1 for row in close_rows if f(row, f"{prefix}.contact.right_finger.count") > 0
        ),
        "left_max_force_n": max(f(row, f"{prefix}.contact.left_finger.force_mag_sum") for row in rows),
        "right_max_force_n": max(f(row, f"{prefix}.contact.right_finger.force_mag_sum") for row in rows),
        "min_contact_separation_m": min_contact_separation(rows, object_name),
        "max_arm_velocity_rad_s": max((abs(f(row, key)) for row in rows for key in arm_vel_keys), default=0.0),
        "max_gripper_mimic_error_rad": max((abs(f(row, key)) for row in rows for key in mimic_keys), default=0.0),
    }
    if min_lift_mm is not None and max_hold_gap_sec is not None:
        summary["lift_hold_sec"] = lift_hold_seconds(rows, object_name, min_lift_mm, max_hold_gap_sec)
    return summary


def diagnostics_errors_for_rows(rows, object_name):
    prefix = object_prefix(object_name)
    errors = []
    always_required = [
        "sim_time_s",
        "dof.gripper_controller.target",
        f"{prefix}.x",
        f"{prefix}.y",
        f"{prefix}.z",
    ]
    contact_labels = ("left_finger", "right_finger")
    for idx, row in enumerate(rows):
        for key in always_required:
            if required_float(row, key) is None:
                errors.append(f"row{idx}:{key}")
        for label in contact_labels:
            count_key = f"{prefix}.contact.{label}.count"
            force_key = f"{prefix}.contact.{label}.force_mag_sum"
            count = required_float(row, count_key)
            force = required_float(row, force_key)
            if count is None:
                errors.append(f"row{idx}:{count_key}")
            if force is None:
                errors.append(f"row{idx}:{force_key}")
            if count is not None and count > 0:
                for key in (
                    f"{prefix}.contact.{label}.separation_min",
                    f"{prefix}.contact.{label}.separation_max",
                ):
                    if required_float(row, key) is None:
                        errors.append(f"row{idx}:{key}")
        if len(errors) >= 8:
            return errors
    return errors


def min_contact_separation(rows, object_name):
    prefix = object_prefix(object_name)
    values = []
    for row in rows:
        for label in ("left_finger", "right_finger"):
            count = required_float(row, f"{prefix}.contact.{label}.count")
            if count is None or count <= 0:
                continue
            separation = required_float(row, f"{prefix}.contact.{label}.separation_min")
            if separation is not None:
                values.append(separation)
    return min(values) if values else None


def approach_interaction(summary, args):
    if not summary.get("diagnostics_valid", False):
        return "diagnostics_invalid"
    contact_rows = int(summary.get("open_contact_rows", 0))
    max_force = max(float(summary.get("left_max_force_n", 0.0)), float(summary.get("right_max_force_n", 0.0)))
    force_contact = max_force > args.open_contact_max_force_n
    motion_contact = (
        float(summary.get("max_xy_disp_mm", 0.0)) > args.open_contact_max_xy_mm
        or float(summary.get("max_z_disp_mm", 0.0)) > args.open_contact_max_z_mm
    )
    min_separation = summary.get("min_contact_separation_m")
    penetration_contact = (
        min_separation is not None
        and float(min_separation) < -abs(float(args.open_contact_penetration_tolerance_m))
    )
    if force_contact or motion_contact or penetration_contact:
        return "contact"
    if contact_rows > args.open_contact_max_rows:
        return "proximity_only"
    return "clean"


def round_metrics(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: round_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_metrics(item) for item in value]
    return value


def main():
    args = parse_args()
    positions = load_positions(args.position_file)
    selected_indices = parse_position_indices(args.position_indices, len(positions))
    indexed_positions = list(enumerate(positions, start=1))
    if selected_indices is not None:
        wanted = set(selected_indices)
        indexed_positions = [(index, position) for index, position in indexed_positions if index in wanted]
    elif args.limit is not None:
        indexed_positions = indexed_positions[: args.limit]
    _, pose_by_position = load_pose_entries(args.poses_file)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, position in indexed_positions:
        entry = pose_by_position.get(position_key(position))
        base_result = {
            "index": index,
            "object": args.object,
            "position": position,
            "status": None,
            "training_candidate": False,
        }
        if entry is None or not entry.get("solved"):
            base_result.update(
                {
                    "status": "planning_failed",
                    "reason": None if entry is None else entry.get("reason"),
                    "reject_reason": None if entry is None else entry.get("reject_reason"),
                }
            )
            results.append(base_result)
            print(f"{index:02d}: planning_failed")
            continue

        place_and_reset(position, args.object, args.reset_settle_sec)
        start_line = csv_line_count(args.diagnostics_csv)
        open_exit, open_log = run_demo(args, index, position, stop_after_approach=True, log_dir=log_dir)
        time.sleep(args.between_run_sec)
        open_rows = rows_for_position(read_diagnostic_rows(args.diagnostics_csv, start_line), position, args.object)
        open_summary = summarize_rows(open_rows, args.object)
        base_result.update(
            {
                "open_demo_exit": open_exit,
                "open_log": str(open_log),
                "open_summary": round_metrics(open_summary),
            }
        )
        open_interaction = approach_interaction(open_summary, args)
        base_result["open_interaction"] = open_interaction
        if open_interaction == "diagnostics_invalid":
            base_result["status"] = "diagnostics_invalid"
            results.append(base_result)
            print(f"{index:02d}: diagnostics_invalid")
            continue
        if open_interaction == "contact":
            base_result["status"] = "open_approach_contact"
            results.append(base_result)
            print(f"{index:02d}: open_approach_contact")
            continue
        if open_exit != 0:
            base_result["status"] = "open_approach_failed"
            results.append(base_result)
            print(f"{index:02d}: open_approach_failed")
            continue

        place_and_reset(position, args.object, args.reset_settle_sec)
        start_line = csv_line_count(args.diagnostics_csv)
        full_exit, full_log = run_demo(args, index, position, stop_after_approach=False, log_dir=log_dir)
        time.sleep(args.between_run_sec)
        full_rows = rows_for_position(read_diagnostic_rows(args.diagnostics_csv, start_line), position, args.object)
        full_preclose_rows = rows_before_close_start(full_rows, args.gripper_open, args.close_start_drop_rad)
        full_preclose_summary = summarize_rows(full_preclose_rows, args.object)
        full_summary = summarize_rows(
            full_rows,
            args.object,
            min_lift_mm=args.min_lift_mm,
            max_hold_gap_sec=args.max_hold_gap_sec,
        )
        lift_ok = float(full_summary.get("lift_mm", 0.0)) >= args.min_lift_mm
        hold_ok = args.min_hold_sec <= 0.0 or float(full_summary.get("lift_hold_sec", 0.0)) >= args.min_hold_sec
        preclose_has_rows = int(full_preclose_summary.get("rows", 0)) > 0
        full_diagnostics_valid = bool(full_summary.get("diagnostics_valid", False))
        preclose_interaction = approach_interaction(full_preclose_summary, args)
        preclose_ok = preclose_has_rows and preclose_interaction in ("clean", "proximity_only")
        success = full_exit == 0 and full_diagnostics_valid and lift_ok and hold_ok and preclose_ok
        failure_reasons = []
        if full_exit != 0:
            failure_reasons.append("demo_exit")
        if not full_diagnostics_valid or preclose_interaction == "diagnostics_invalid":
            failure_reasons.append("diagnostics_invalid")
        if not lift_ok:
            failure_reasons.append("lift")
        if not hold_ok:
            failure_reasons.append("lift_hold")
        if not preclose_has_rows:
            failure_reasons.append("preclose_missing")
        elif not preclose_ok:
            failure_reasons.append("preclose_contact")
        status = "full_pick_success" if success else "full_pick_failed"
        if not full_diagnostics_valid or preclose_interaction == "diagnostics_invalid":
            status = "diagnostics_invalid"
        elif preclose_has_rows and not preclose_ok:
            status = "full_preclose_contact"
        base_result.update(
            {
                "full_demo_exit": full_exit,
                "full_log": str(full_log),
                "full_preclose_summary": round_metrics(full_preclose_summary),
                "full_summary": round_metrics(full_summary),
                "full_diagnostics_valid": bool(full_diagnostics_valid),
                "lift_ok": bool(lift_ok),
                "hold_ok": bool(hold_ok),
                "preclose_has_rows": bool(preclose_has_rows),
                "preclose_interaction": preclose_interaction,
                "preclose_ok": bool(preclose_ok),
                "failure_reasons": failure_reasons,
                "status": status,
                "training_candidate": bool(success),
            }
        )
        results.append(base_result)
        print(
            f"{index:02d}: {base_result['status']} "
            f"lift={full_summary.get('lift_mm', 0.0):.1f}mm "
            f"hold={full_summary.get('lift_hold_sec', 0.0):.2f}s"
        )

    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    payload = {
        "positions_total": len(indexed_positions),
        "source_positions_total": len(positions),
        "source_position_indices": [index for index, _ in indexed_positions],
        "object": args.object,
        "poses_file": str(Path(args.poses_file).resolve()),
        "position_file": str(Path(args.position_file).resolve()),
        "diagnostics_csv": str(Path(args.diagnostics_csv).resolve()),
        "settings": {
            "close": args.close,
            "arm_speed": args.arm_speed,
            "lift_arm_speed": args.lift_arm_speed,
            "arm_tolerance": args.arm_tolerance,
            "open_contact_max_rows": args.open_contact_max_rows,
            "open_contact_max_force_n": args.open_contact_max_force_n,
            "open_contact_max_xy_mm": args.open_contact_max_xy_mm,
            "open_contact_max_z_mm": args.open_contact_max_z_mm,
            "open_contact_penetration_tolerance_m": args.open_contact_penetration_tolerance_m,
            "min_lift_mm": args.min_lift_mm,
            "min_hold_sec": args.min_hold_sec,
            "max_hold_gap_sec": args.max_hold_gap_sec,
            "gripper_open": args.gripper_open,
            "close_start_drop_rad": args.close_start_drop_rad,
        },
        "counts": counts,
        "training_candidates": sum(1 for result in results if result.get("training_candidate")),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "training_candidates": payload["training_candidates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
