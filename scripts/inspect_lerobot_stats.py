#!/usr/bin/env python3
"""Inspect LeRobot per-dimension normalization statistics for OOD risk."""

import argparse
import json
from pathlib import Path

import numpy as np

CONTROL_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--std-warning", type=float, default=0.02)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def feature_table(stats, key, std_warning):
    feature = stats[key]
    rows = []
    for idx, name in enumerate(CONTROL_JOINTS):
        row = {
            "dim": idx,
            "name": name,
            "min": float(feature["min"][idx]),
            "max": float(feature["max"][idx]),
            "mean": float(feature["mean"][idx]),
            "std": float(feature["std"][idx]),
            "range": float(feature["max"][idx] - feature["min"][idx]),
            "q01": float(feature["q01"][idx]),
            "q99": float(feature["q99"][idx]),
            "small_std_warning": bool(feature["std"][idx] < std_warning),
        }
        row["one_mm_scale_z_for_0.005_rad"] = float(0.005 / max(row["std"], 1e-8))
        rows.append(row)
    return rows


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    stats_path = dataset_root / "meta" / "stats.json"
    info_path = dataset_root / "meta" / "info.json"
    stats = json.loads(stats_path.read_text())
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    result = {
        "dataset_root": str(dataset_root),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "std_warning_threshold": float(args.std_warning),
        "observation_state": feature_table(stats, "observation.state", args.std_warning),
        "action": feature_table(stats, "action", args.std_warning),
    }
    warnings = []
    for feature_name in ("observation_state", "action"):
        for row in result[feature_name]:
            if row["small_std_warning"]:
                warnings.append(f"{feature_name}.{row['name']} std={row['std']:.6f}")
    result["warnings"] = warnings

    print("dataset:", dataset_root)
    print("fps:", result["fps"], "episodes:", result["total_episodes"], "frames:", result["total_frames"])
    for feature_name in ("observation_state", "action"):
        print()
        print(feature_name)
        for row in result[feature_name]:
            flag = "  <-- small std" if row["small_std_warning"] else ""
            print(
                f"  {row['name']:<8} std={row['std']:.6f} "
                f"range={row['range']:.6f} min={row['min']:.4f} max={row['max']:.4f}{flag}"
            )
    if warnings:
        print()
        print("WARNINGS")
        for warning in warnings:
            print("  -", warning)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
