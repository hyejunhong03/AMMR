#!/usr/bin/env python3
"""Backfill recorded episodes to the current dataset schema.

Only metadata.json is touched. The .npy arrays and the PNGs are never rewritten, so a
bad run costs nothing but a re-migrate.

The one substantive change is moving object_state_* out of "observation" and into a
top-level "privileged" block. Simulator ground truth does not exist on the real robot,
and leaving it under "observation" invites a converter to sweep it into the policy
input. See docs/smolvla_dataset_schema.md.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from ammr_mycobot_interface import (
    COMMAND_GRIPPER_CLOSE_RAD,
    CONTROL_JOINTS,
    GRIPPER_OPEN_RAD,
    IMAGE_SIZE,
    OBJECT_STATE_NAMES,
    QUAT_ORDER,
    SCHEMA_VERSION,
    SIM_GRIPPER_LIMITS,
)


DEFAULT_ROOT = "/home/autolab/AMMR/data/smolvla_raw"
PRIVILEGED_NOTE = "Simulator ground truth. Not available on the real robot."


def migrate(metadata, episode_dir):
    """Return (new_metadata, list of change descriptions). Pure — does not write."""
    changes = []
    updated = dict(metadata)
    observation = dict(updated.get("observation") or {})

    # Move the privileged keys out of observation.
    privileged = dict(updated.get("privileged") or {})
    moved = False
    for key in ("object_state_topic", "object_state_dim", "object_state_names"):
        if key in observation:
            privileged.setdefault(key, observation.pop(key))
            moved = True
    if moved:
        changes.append("moved object_state_* from observation to privileged")

    # Episodes 1-9 predate object_state entirely; record that rather than inventing values.
    if not privileged:
        has_object_state = (Path(episode_dir) / "object_state.npy").exists()
        privileged = {
            "object_state_topic": None,
            "object_state_dim": len(OBJECT_STATE_NAMES) if has_object_state else 0,
            "object_state_names": list(OBJECT_STATE_NAMES) if has_object_state else [],
        }
        changes.append(
            "added privileged block"
            + ("" if has_object_state else " (no object_state.npy in this episode)")
        )

    for key, value in (
        ("quat_order", QUAT_ORDER),
        ("policy_input", False),
        ("note", PRIVILEGED_NOTE),
    ):
        if privileged.get(key) != value:
            privileged[key] = value
            changes.append(f"set privileged.{key}")
    updated["privileged"] = privileged

    # observation gains the frozen image/state contract.
    if "state_unit" not in observation:
        observation["state_unit"] = "radian"
        changes.append("set observation.state_unit")
    if "image_size" not in observation:
        # Historical episodes were rendered at 640x480; record what they actually are
        # rather than the current target, so a converter can resize correctly.
        observation["image_size"] = list(legacy_image_size(episode_dir))
        changes.append(f"set observation.image_size={observation['image_size']}")
    updated["observation"] = observation

    # task mirrors instruction. Keep both: task_name stays the short profile label.
    instruction = updated.get("instruction")
    if instruction and updated.get("task") != instruction:
        updated["task"] = instruction
        changes.append("set task = instruction")

    action = dict(updated.get("action") or {})
    for key, value in (
        ("gripper_open_rad", GRIPPER_OPEN_RAD),
        ("gripper_close_rad", COMMAND_GRIPPER_CLOSE_RAD),
        ("gripper_limits_rad", list(SIM_GRIPPER_LIMITS)),
    ):
        if action.get(key) != value:
            action[key] = value
            changes.append(f"set action.{key}")
    if action:
        updated["action"] = action

    if updated.get("schema_version") != SCHEMA_VERSION:
        updated["schema_version"] = SCHEMA_VERSION
        changes.append(f"schema_version -> {SCHEMA_VERSION}")

    return updated, changes


def legacy_image_size(episode_dir):
    """Actual pixel size of this episode's first image, falling back to the current target."""
    images = sorted((Path(episode_dir) / "images").glob("*.png"))
    if not images:
        return IMAGE_SIZE
    try:
        from PIL import Image

        with Image.open(images[0]) as handle:
            return handle.size
    except Exception:
        return IMAGE_SIZE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Directory holding episode_* folders.")
    parser.add_argument("--apply", action="store_true", help="Write changes. Default is a dry run.")
    parser.add_argument("--no-backup", action="store_true", help="Skip the metadata backup copy.")
    args = parser.parse_args()

    root = Path(args.root)
    episodes = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if not episodes:
        print(f"No episodes under {root}", file=sys.stderr)
        return 2

    backup_root = None
    if args.apply and not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = root.parent / f"{root.name}_metadata_backup_{stamp}"
        backup_root.mkdir(parents=True)

    changed = 0
    for episode in episodes:
        path = episode / "metadata.json"
        if not path.exists():
            print(f"{episode.name}: no metadata.json, skipped")
            continue

        metadata = json.loads(path.read_text())
        updated, changes = migrate(metadata, episode)
        if not changes:
            continue

        changed += 1
        print(f"{episode.name}: {', '.join(changes)}")
        if not args.apply:
            continue

        if backup_root is not None:
            target = backup_root / episode.name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / "metadata.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(updated, handle, indent=2)
            handle.write("\n")

    print()
    print(f"{changed}/{len(episodes)} episodes {'migrated' if args.apply else 'would change'}")
    if backup_root is not None:
        print(f"metadata backed up to {backup_root}")
    if not args.apply:
        print("dry run: re-run with --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
