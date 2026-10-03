#!/usr/bin/env python3
"""Generate held-out object positions relative to the solved training pose support."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-poses",
        default="outputs/eval/phase2_red_cube_pick_only_lift060_h020_d8_sample60_poses.json",
        help="Training pose cache. Solved entries define the support box and nearest-neighbour filter.",
    )
    parser.add_argument(
        "--exclude-poses",
        action="append",
        default=[],
        help=(
            "Additional pose cache whose solved entries are excluded by the nearest-neighbour "
            "filter. These positions do not change the sampling support box."
        ),
    )
    parser.add_argument(
        "--exclude-raw-dir",
        action="append",
        default=[],
        help=(
            "Raw SmolVLA episode directory whose metadata collection.object_position values "
            "are excluded by the nearest-neighbour filter."
        ),
    )
    parser.add_argument("--object", default="red_cube")
    parser.add_argument("--mode", choices=("in", "out-y", "mixed"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--min-nearest-train-distance",
        type=float,
        default=0.004,
        help="Reject candidates closer than this many metres to any solved training pose.",
    )
    parser.add_argument(
        "--min-nearest-output-distance",
        type=float,
        default=0.0,
        help=(
            "Reject candidates closer than this many metres to an already accepted output "
            "candidate. Default 0 preserves the previous behaviour."
        ),
    )
    parser.add_argument(
        "--inner-margin",
        type=float,
        default=0.002,
        help="Margin kept inside the solved train box for --mode in.",
    )
    parser.add_argument(
        "--out-offset-low",
        type=float,
        default=0.005,
        help="Minimum y-distance outside the solved train box for --mode out-y.",
    )
    parser.add_argument(
        "--out-offset-high",
        type=float,
        default=0.015,
        help="Maximum y-distance outside the solved train box for --mode out-y.",
    )
    parser.add_argument("--z", type=float, default=None, help="Override object z. Defaults to train median z.")
    parser.add_argument("--x-min", type=float, default=None, help="Override lower x bound for sampling.")
    parser.add_argument("--x-max", type=float, default=None, help="Override upper x bound for sampling.")
    parser.add_argument("--y-min", type=float, default=None, help="Override lower y bound for in-range sampling.")
    parser.add_argument("--y-max", type=float, default=None, help="Override upper y bound for in-range sampling.")
    parser.add_argument("--max-attempts", type=int, default=100000)
    return parser.parse_args()


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
        pos = entry.get("object_position") or entry.get("cube_position") or entry.get("sampled_object_pose")
        if pos is not None:
            positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
    if not positions:
        raise ValueError(f"{path}: no solved train positions for {object_name}")
    return np.asarray(positions, dtype=np.float64)


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
    return np.asarray(positions, dtype=np.float64)


def nearest_xy(candidate, train_positions):
    xy = np.asarray(candidate[:2], dtype=np.float64)
    train_xy = train_positions[:, :2]
    return float(np.min(np.linalg.norm(train_xy - xy[None, :], axis=1)))


def nearest_xy_or_inf(candidate, positions):
    if not positions:
        return float("inf")
    xy = np.asarray(candidate[:2], dtype=np.float64)
    other_xy = np.asarray([p[:2] for p in positions], dtype=np.float64)
    return float(np.min(np.linalg.norm(other_xy - xy[None, :], axis=1)))


def rounded_key(position):
    return tuple(round(float(v), 4) for v in position[:2])


def sample_in(rng, bounds, z):
    x_min, x_max, y_min, y_max = bounds
    return np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max), z], dtype=np.float64)


def sample_out_y(rng, bounds, z, low, high):
    x_min, x_max, y_min, y_max = bounds
    side = -1.0 if rng.random() < 0.5 else 1.0
    offset = rng.uniform(low, high)
    y = y_min - offset if side < 0 else y_max + offset
    return np.array([rng.uniform(x_min, x_max), y, z], dtype=np.float64)


def main():
    args = parse_args()
    train = load_train_positions(args.train_poses, args.object)
    exclude_sets = [train]
    extra_excludes = []
    for path in args.exclude_poses:
        positions = load_train_positions(path, args.object)
        exclude_sets.append(positions)
        extra_excludes.append({"type": "pose_cache", "path": path, "count": int(len(positions))})
    for path in args.exclude_raw_dir:
        positions = load_raw_positions(path, args.object)
        exclude_sets.append(positions)
        extra_excludes.append({"type": "raw_dir", "path": path, "count": int(len(positions))})
    nearest_filter_positions = np.concatenate(exclude_sets, axis=0)
    z = float(args.z) if args.z is not None else float(np.median(train[:, 2]))
    x_min, x_max = float(train[:, 0].min()), float(train[:, 0].max())
    y_min, y_max = float(train[:, 1].min()), float(train[:, 1].max())
    in_x_min = x_min + args.inner_margin if args.x_min is None else float(args.x_min)
    in_x_max = x_max - args.inner_margin if args.x_max is None else float(args.x_max)
    in_y_min = y_min + args.inner_margin if args.y_min is None else float(args.y_min)
    in_y_max = y_max - args.inner_margin if args.y_max is None else float(args.y_max)
    in_bounds = (in_x_min, in_x_max, in_y_min, in_y_max)
    out_x_min = x_min if args.x_min is None else float(args.x_min)
    out_x_max = x_max if args.x_max is None else float(args.x_max)
    out_bounds = (out_x_min, out_x_max, y_min, y_max)
    if in_bounds[0] >= in_bounds[1] or in_bounds[2] >= in_bounds[3]:
        raise ValueError("--inner-margin is too large for the train support box")

    rng = np.random.default_rng(args.seed)
    positions = []
    seen = set()
    attempts = 0
    while len(positions) < args.count and attempts < args.max_attempts:
        attempts += 1
        if args.mode == "in":
            candidate = sample_in(rng, in_bounds, z)
        elif args.mode == "out-y":
            candidate = sample_out_y(rng, out_bounds, z, args.out_offset_low, args.out_offset_high)
        else:
            if len(positions) < math.ceil(args.count * 2 / 3):
                candidate = sample_in(rng, in_bounds, z)
            else:
                candidate = sample_out_y(rng, out_bounds, z, args.out_offset_low, args.out_offset_high)

        key = rounded_key(candidate)
        if key in seen:
            continue
        if nearest_xy(candidate, nearest_filter_positions) < args.min_nearest_train_distance:
            continue
        if nearest_xy_or_inf(candidate, positions) < args.min_nearest_output_distance:
            continue
        seen.add(key)
        positions.append([round(float(candidate[0]), 4), round(float(candidate[1]), 4), round(z, 4)])

    if len(positions) < args.count:
        raise RuntimeError(f"only generated {len(positions)}/{args.count} positions after {attempts} attempts")

    payload = {
        "object_name": args.object,
        "mode": args.mode,
        "seed": args.seed,
        "train_support": {
            "count": int(len(train)),
            "x_range": [x_min, x_max],
            "y_range": [y_min, y_max],
            "z": z,
        },
        "nearest_filter": {
            "count": int(len(nearest_filter_positions)),
            "min_nearest_train_distance": args.min_nearest_train_distance,
            "min_nearest_output_distance": args.min_nearest_output_distance,
            "extra_excludes": extra_excludes,
        },
        "positions": positions,
        "entries": [{"object_name": args.object, "object_position": pos, "cube_position": pos} for pos in positions],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(positions)} positions to {out}")
    print(f"train x=[{x_min:.5f},{x_max:.5f}] y=[{y_min:.5f},{y_max:.5f}]")
    print(
        "nearest filter positions="
        f"{len(nearest_filter_positions)} min_distance={args.min_nearest_train_distance:.4f}"
    )
    if args.min_nearest_output_distance > 0:
        print(f"output min_distance={args.min_nearest_output_distance:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
