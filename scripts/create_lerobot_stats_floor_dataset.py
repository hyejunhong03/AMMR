#!/usr/bin/env python3
"""Clone a LeRobot dataset and apply a per-feature std floor in meta/stats.json.

This is for controlled normalization ablations. It leaves the original dataset intact
and changes only the statistics that the training preprocessor will save into a new
checkpoint. Frames, actions, images, and metadata are otherwise copied unchanged.
"""

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Source LeRobot dataset root.")
    parser.add_argument("--target", required=True, help="Target dataset root to create.")
    parser.add_argument("--state-std-floor", type=float, default=0.05)
    parser.add_argument("--action-std-floor", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def apply_floor(values, floor):
    if floor <= 0.0:
        return values, []
    patched = []
    changed = []
    for index, value in enumerate(values):
        new_value = max(float(value), float(floor))
        patched.append(new_value)
        if new_value != float(value):
            changed.append({"dim": index, "old": float(value), "new": new_value})
    return patched, changed


def main():
    args = parse_args()
    source = Path(args.source)
    target = Path(args.target)
    if not (source / "meta" / "stats.json").exists():
        raise FileNotFoundError(source / "meta" / "stats.json")
    if target.exists():
        if not args.overwrite:
            raise FileExistsError(f"{target} already exists; pass --overwrite to replace it")
        shutil.rmtree(target)

    shutil.copytree(source, target)
    stats_path = target / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())

    state_std, state_changed = apply_floor(
        stats["observation.state"]["std"], args.state_std_floor
    )
    action_std, action_changed = apply_floor(stats["action"]["std"], args.action_std_floor)
    stats["observation.state"]["std"] = state_std
    stats["action"]["std"] = action_std
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    patch_record = {
        "source": str(source),
        "target": str(target),
        "state_std_floor": float(args.state_std_floor),
        "action_std_floor": float(args.action_std_floor),
        "state_changed": state_changed,
        "action_changed": action_changed,
    }
    (target / "meta" / "ammr_stats_floor_patch.json").write_text(
        json.dumps(patch_record, indent=2) + "\n"
    )

    print(
        json.dumps(patch_record, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
