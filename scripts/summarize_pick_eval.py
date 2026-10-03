#!/usr/bin/env python3
"""Summarize pick rollout JSON with train-range and failure diagnostics."""

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_TRAIN_POSES = "outputs/eval/phase2_red_cube_pick_only_lift060_h020_d8_sample60_poses.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="rollout_policy.py result JSON files.")
    parser.add_argument(
        "--train-poses",
        default=DEFAULT_TRAIN_POSES,
        help="Pose cache used for training. Only solved entries are used as the train support.",
    )
    parser.add_argument(
        "--train-raw-dir",
        action="append",
        default=[],
        help=(
            "Raw SmolVLA episode directory whose metadata collection.object_position values "
            "define the train support. If this is set without an explicit --train-poses, "
            "the default old pose cache is not included."
        ),
    )
    parser.add_argument("--object", default="red_cube")
    parser.add_argument("--out-json", default=None, help="Write machine-readable summary.")
    parser.add_argument("--failure-csv", default=None, help="Write one row per non-clean rollout.")
    parser.add_argument(
        "--tcp-fail-threshold",
        type=float,
        default=0.01,
        help="TCP minimum distance above this many metres is labelled as an approach miss.",
    )
    return parser.parse_args()


def load_results(path):
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        return payload
    for key in ("results", "episodes", "rollouts"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"{path}: could not find results list")


def entry_position(entry):
    for key in ("object_position", "cube_position", "position", "xyz", "sampled_object_pose"):
        if key in entry and entry[key] is not None:
            return entry[key]
    return None


def load_train_positions(path, object_name):
    payload = json.loads(Path(path).read_text())
    entries = payload.get("entries", payload if isinstance(payload, list) else [])
    positions = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("object_name", object_name) != object_name:
            continue
        if "solved" in entry and not entry.get("solved"):
            continue
        if "accepted_by_ik" in entry and not entry.get("accepted_by_ik"):
            continue
        pos = entry_position(entry)
        if pos is not None:
            positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
    if not positions:
        raise ValueError(f"{path}: no solved train positions for {object_name}")
    return positions


def load_raw_positions(path, object_name):
    root = Path(path)
    positions = []
    for metadata_path in sorted(root.glob("episode_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        collection = metadata.get("collection", {})
        if collection.get("object_name", object_name) != object_name:
            continue
        pos = collection.get("object_position")
        if pos is not None:
            positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
    if not positions:
        raise ValueError(f"{path}: no raw metadata positions for {object_name}")
    return positions


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return None
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return [centre - half, centre + half]


def clean_success(row):
    return bool(row.get("success")) and bool(row.get("velocity_ok", True)) and not bool(row.get("invalid_rollout"))


def position_for(row):
    pos = row.get("object_position") or row.get("cube_position")
    if pos is None:
        return None
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def support_metrics(pos, train_positions):
    xs = [p[0] for p in train_positions]
    ys = [p[1] for p in train_positions]
    x, y = float(pos[0]), float(pos[1])
    dx_out = max(min(xs) - x, 0.0, x - max(xs))
    dy_out = max(min(ys) - y, 0.0, y - max(ys))
    outside = math.hypot(dx_out, dy_out)
    nearest = min(math.dist((x, y), (p[0], p[1])) for p in train_positions)
    return {
        "in_train_xy_box": outside == 0.0,
        "outside_train_xy_box_m": outside,
        "nearest_train_xy_m": nearest,
    }


def state_z_dim(row):
    value = row.get("state_z_max")
    if not isinstance(value, dict):
        return None
    if value.get("dim_1based") is not None:
        return value.get("dim_1based")
    index = value.get("index")
    if isinstance(index, list) and len(index) >= 2:
        return int(index[1]) + 1
    return None


def classify_failure(row, tcp_fail_threshold):
    if clean_success(row):
        return "clean_success"
    if row.get("invalid_rollout"):
        return row.get("error", "invalid_rollout")
    if not row.get("success"):
        tcp_min = row.get("tcp_target_distance_min_m")
        if tcp_min is not None and float(tcp_min) > tcp_fail_threshold:
            return "approach_miss"
        return "lift_failure"
    if row.get("velocity_ok") is False:
        if row.get("stuck_event"):
            return "success_with_stuck"
        if row.get("state_jump_event"):
            return "success_with_state_jump"
        return row.get("velocity_violation_type", "success_with_velocity_violation")
    return "non_clean"


def summarize_file(path, train_positions, args):
    rows = load_results(path)
    enriched = []
    for index, row in enumerate(rows):
        pos = position_for(row)
        support = support_metrics(pos, train_positions) if pos is not None else {}
        item = {
            "file": str(path),
            "episode_index_0based": index,
            "episode_index_1based": index + 1,
            "success": bool(row.get("success")),
            "velocity_ok": bool(row.get("velocity_ok", True)),
            "clean_success": clean_success(row),
            "failure_class": classify_failure(row, args.tcp_fail_threshold),
            "object_position": pos,
            "x": pos[0] if pos else None,
            "y": pos[1] if pos else None,
            "z": pos[2] if pos else None,
            "tcp_target_distance_min_m": row.get("tcp_target_distance_min_m"),
            "tcp_target_distance_final_m": row.get("tcp_target_distance_final_m"),
            "lift_m": row.get("lift_m"),
            "velocity_violation_type": row.get("velocity_violation_type"),
            "velocity_peak_joint": row.get("velocity_peak_joint"),
            "velocity_peak_state_step": row.get("velocity_peak_state_step"),
            "velocity_peak_action_step": row.get("velocity_peak_action_step"),
            "state_z_dim_1based": state_z_dim(row),
            "state_jump_event": bool(row.get("state_jump_event", False)),
            "stuck_event": bool(row.get("stuck_event", False)),
            "reset_attempts": row.get("reset_attempts"),
            "start_object_error_m": (
                row.get("start_object_position", {}).get("error_m")
                if isinstance(row.get("start_object_position"), dict)
                else None
            ),
            "start_motion_max_joint_velocity": (
                row.get("start_motion", {}).get("max_joint_velocity")
                if isinstance(row.get("start_motion"), dict)
                else None
            ),
            **support,
        }
        enriched.append(item)

    def counts(subset):
        n = len(subset)
        raw = sum(item["success"] for item in subset)
        clean = sum(item["clean_success"] for item in subset)
        violations = sum(not item["velocity_ok"] for item in subset)
        return {
            "n": n,
            "raw_success": raw,
            "clean_success": clean,
            "velocity_violations": violations,
            "clean_rate": clean / n if n else None,
            "clean_wilson95": wilson_interval(clean, n),
        }

    in_range = [item for item in enriched if item.get("in_train_xy_box") is True]
    out_range = [item for item in enriched if item.get("in_train_xy_box") is False]
    summary = {
        "file": str(path),
        "overall": counts(enriched),
        "in_train_xy_box": counts(in_range),
        "outside_train_xy_box": counts(out_range),
        "failure_classes": {},
    }
    for item in enriched:
        if item["clean_success"]:
            continue
        key = item["failure_class"]
        summary["failure_classes"][key] = summary["failure_classes"].get(key, 0) + 1
    return summary, enriched


def rate_text(counts):
    n = counts["n"]
    clean = counts["clean_success"]
    raw = counts["raw_success"]
    vio = counts["velocity_violations"]
    if n == 0:
        return "n=0"
    lo, hi = counts["clean_wilson95"]
    return f"raw={raw}/{n} clean={clean}/{n} violations={vio}/{n} clean95=({lo:.3f},{hi:.3f})"


def main():
    args = parse_args()
    train_positions = []
    train_sources = []
    if not (args.train_raw_dir and args.train_poses == DEFAULT_TRAIN_POSES):
        train_positions.extend(load_train_positions(args.train_poses, args.object))
        train_sources.append(args.train_poses)
    for raw_dir in args.train_raw_dir:
        train_positions.extend(load_raw_positions(raw_dir, args.object))
        train_sources.append(raw_dir)
    xs = [p[0] for p in train_positions]
    ys = [p[1] for p in train_positions]
    print(
        f"train support: n={len(train_positions)} "
        f"x=[{min(xs):.5f},{max(xs):.5f}] y=[{min(ys):.5f},{max(ys):.5f}]"
    )

    all_summaries = []
    all_failures = []
    for name in args.results:
        summary, enriched = summarize_file(Path(name), train_positions, args)
        all_summaries.append(summary)
        all_failures.extend([item for item in enriched if not item["clean_success"]])
        print()
        print(Path(name).name)
        print("  overall:   " + rate_text(summary["overall"]))
        print("  in range:  " + rate_text(summary["in_train_xy_box"]))
        print("  out range: " + rate_text(summary["outside_train_xy_box"]))
        if summary["failure_classes"]:
            classes = ", ".join(f"{k}={v}" for k, v in sorted(summary["failure_classes"].items()))
            print(f"  failures:  {classes}")
        failures = [item for item in enriched if not item["clean_success"]]
        for item in failures:
            print(
                "    "
                f"#{item['episode_index_0based']:03d} {item['failure_class']} "
                f"xy=({item['x']:.4f},{item['y']:.4f}) "
                f"in_box={item.get('in_train_xy_box')} "
                f"outside={item.get('outside_train_xy_box_m', float('nan')):.4f}m "
                f"nearest={item.get('nearest_train_xy_m', float('nan')):.4f}m "
                f"tcp_min={item.get('tcp_target_distance_min_m')} "
                f"vio={item.get('velocity_violation_type')}"
            )

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps({"summaries": all_summaries}, indent=2) + "\n")

    if args.failure_csv:
        Path(args.failure_csv).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "file",
            "episode_index_0based",
            "failure_class",
            "success",
            "velocity_ok",
            "x",
            "y",
            "z",
            "in_train_xy_box",
            "outside_train_xy_box_m",
            "nearest_train_xy_m",
            "tcp_target_distance_min_m",
            "tcp_target_distance_final_m",
            "lift_m",
            "velocity_violation_type",
            "velocity_peak_joint",
            "velocity_peak_state_step",
            "velocity_peak_action_step",
            "state_z_dim_1based",
            "state_jump_event",
            "stuck_event",
            "reset_attempts",
            "start_object_error_m",
            "start_motion_max_joint_velocity",
        ]
        with Path(args.failure_csv).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for item in all_failures:
                writer.writerow({key: item.get(key) for key in fieldnames})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
