#!/usr/bin/env python3
"""Convert raw AMMR episodes into a LeRobot dataset for SmolVLA fine-tuning.

Run with the project venv, which is where lerobot lives:

    /home/autolab/AMMR/.venv/bin/python convert_to_lerobot.py --dry-run

The privileged block in metadata.json is deliberately not carried across. It holds
simulator ground truth (object poses) that has no counterpart on the real robot, so a
policy that learned to use it would not transfer. It stays in the raw episodes for
success scoring and demo generation.

--dry-run validates every episode against the schema and reports what would be written
without importing lerobot, so the data can be checked before the dependency is in place.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_RAW_DIR = "/home/autolab/AMMR/data/smolvla_raw"
DEFAULT_REPO_ID = "ammr/pick_red_cube"
EXPECTED_SCHEMA_VERSION = 2
# SmolVLA's image input. Episodes recorded before the camera was retuned are larger and
# get resized on the way in.
TARGET_IMAGE_SIZE = (256, 256)
STATE_DIM = 7
ACTION_DIM = 7
CAMERA_KEY = "observation.images.wrist"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        action="append",
        default=None,
        help="Directory of episode_* folders. May be passed more than once to merge sources.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo id.")
    parser.add_argument("--root", default=None, help="Where to write the dataset (default: lerobot cache).")
    parser.add_argument("--fps", type=int, default=10, help="Dataset frame rate.")
    parser.add_argument(
        "--only-success",
        action="store_true",
        help="Skip episodes whose metadata success flag is not true.",
    )
    parser.add_argument(
        "--require-image-size",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=list(TARGET_IMAGE_SIZE),
        help=(
            "Only take episodes recorded at exactly this size. Mixing aspect ratios "
            "would hand the policy two different geometries of the same scene: the "
            "older 640x480 frames squash horizontally when forced to 256x256."
        ),
    )
    parser.add_argument(
        "--any-image-size",
        action="store_true",
        help="Accept every image size and resize. Only sound if the aspect ratios match.",
    )
    parser.add_argument(
        "--allow-mixed-tasks",
        action="store_true",
        help="Permit more than one task string in one dataset.",
    )
    parser.add_argument("--episodes", nargs="*", help="Explicit episode ids; default is every episode.")
    parser.add_argument(
        "--episodes-file",
        default=None,
        help="Text file with one episode id per line. Lines starting with # are ignored.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing.")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing dataset at the target root.")
    args = parser.parse_args()
    if args.raw_dir is None:
        args.raw_dir = [DEFAULT_RAW_DIR]
    if args.episodes_file:
        entries = []
        for line in Path(args.episodes_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line)
        args.episodes = (args.episodes or []) + entries
    return args


def load_episode(episode_dir):
    """Read one raw episode and check it against the frozen schema.

    Returns (payload, errors). Callers decide whether to skip or abort, so one bad
    episode does not silently poison a conversion.
    """
    episode_dir = Path(episode_dir)
    errors = []

    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.exists():
        return None, [f"{episode_dir.name}: missing metadata.json"]
    metadata = json.loads(metadata_path.read_text())

    if metadata.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        errors.append(
            f"{episode_dir.name}: schema_version {metadata.get('schema_version')!r} "
            f"!= {EXPECTED_SCHEMA_VERSION}; run migrate_episode_metadata.py"
        )
    leaked = sorted(k for k in (metadata.get("observation") or {}) if "object_state" in k)
    if leaked:
        errors.append(f"{episode_dir.name}: privileged keys under observation: {leaked}")

    task = metadata.get("task") or metadata.get("instruction")
    if not task:
        errors.append(f"{episode_dir.name}: no task/instruction string")

    try:
        state = np.load(episode_dir / "robot_state.npy")
        action = np.load(episode_dir / "action.npy")
    except FileNotFoundError as exc:
        return None, [f"{episode_dir.name}: {exc}"]

    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        errors.append(f"{episode_dir.name}: robot_state shape {state.shape} != [T, {STATE_DIM}]")
    if action.ndim != 2 or action.shape[1] != ACTION_DIM:
        errors.append(f"{episode_dir.name}: action shape {action.shape} != [T, {ACTION_DIM}]")

    images = sorted((episode_dir / "images").glob("*.png"))
    if len(images) != state.shape[0] or len(images) != action.shape[0]:
        errors.append(
            f"{episode_dir.name}: length mismatch images={len(images)} "
            f"state={state.shape[0]} action={action.shape[0]}"
        )

    payload = {
        "episode_id": metadata.get("episode_id", episode_dir.name),
        "dir": episode_dir,
        "task": task,
        "success": metadata.get("success"),
        "image_size": tuple(metadata.get("observation", {}).get("image_size") or ()),
        "num_steps": len(images),
        "state": state,
        "action": action,
        "images": images,
    }
    return payload, errors


def load_frame(path, target_size):
    """RGB uint8 HWC at the target size."""
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
        if image.size != tuple(target_size):
            image = image.resize(tuple(target_size), Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def select_episodes(args):
    raw_dirs = [Path(raw_dir) for raw_dir in args.raw_dir]
    if args.episodes and len(raw_dirs) != 1:
        raise ValueError("--episodes is only supported with one --raw-dir")

    candidates = []
    for raw_dir in raw_dirs:
        if args.episodes:
            candidates.extend(raw_dir / name for name in args.episodes)
        else:
            candidates.extend(
                sorted(p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("episode_"))
            )

    selected, skipped, errors = [], [], []
    for episode_dir in candidates:
        payload, episode_errors = load_episode(episode_dir)
        if payload is None:
            errors.extend(episode_errors)
            continue
        if episode_errors:
            errors.extend(episode_errors)
            skipped.append((payload["episode_id"], "schema errors"))
            continue
        if args.only_success and not payload["success"]:
            skipped.append((payload["episode_id"], f"success={payload['success']}"))
            continue
        if not args.any_image_size and payload["image_size"]:
            if tuple(payload["image_size"]) != tuple(args.require_image_size):
                size = "x".join(str(v) for v in payload["image_size"])
                skipped.append((payload["episode_id"], f"image_size {size} != required"))
                continue
        selected.append(payload)
    return selected, skipped, errors


def report(selected, skipped, errors):
    print(f"{'episode':<16}{'steps':>7}{'success':>9}{'image_size':>14}  task")
    for payload in selected:
        size = "x".join(str(v) for v in payload["image_size"]) if payload["image_size"] else "?"
        print(
            f"{payload['episode_id']:<16}{payload['num_steps']:>7}"
            f"{str(payload['success']):>9}{size:>14}  {payload['task']}"
        )
    if skipped:
        print()
        print("skipped:")
        for episode_id, reason in skipped:
            print(f"  {episode_id}: {reason}")
    if errors:
        print()
        print("errors:")
        for error in errors:
            print(f"  {error}")

    total_steps = sum(p["num_steps"] for p in selected)
    tasks = sorted({p["task"] for p in selected})
    print()
    print(f"{len(selected)} episodes, {total_steps} frames, tasks={tasks}")
    return total_steps


def build_dataset(args, selected):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        CAMERA_KEY: {
            "dtype": "image",
            "shape": (TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0], 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    }

    root = Path(args.root) if args.root else None
    if args.overwrite and root is not None and root.exists():
        import shutil

        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=root,
        features=features,
        use_videos=False,
    )

    for payload in selected:
        for index in range(payload["num_steps"]):
            # lerobot 0.4.x takes the task inside the frame dict, not as a keyword.
            dataset.add_frame(
                {
                    CAMERA_KEY: load_frame(payload["images"][index], TARGET_IMAGE_SIZE),
                    "observation.state": payload["state"][index].astype(np.float32),
                    "action": payload["action"][index].astype(np.float32),
                    "task": payload["task"],
                }
            )
        dataset.save_episode()
        print(f"  wrote {payload['episode_id']} ({payload['num_steps']} frames)")

    return dataset


def main():
    args = parse_args()
    selected, skipped, errors = select_episodes(args)
    total_steps = report(selected, skipped, errors)

    if not selected:
        print("nothing to convert", file=sys.stderr)
        return 1

    tasks = sorted({p["task"] for p in selected})
    if len(tasks) > 1 and not args.allow_mixed_tasks:
        print(file=sys.stderr)
        print(f"refusing to convert: {len(tasks)} different task strings: {tasks}", file=sys.stderr)
        print(
            "The schema fixes one string per task. Normalise the raw episodes, or pass "
            "--allow-mixed-tasks if the variation is intentional.",
            file=sys.stderr,
        )
        return 1
    if args.dry_run:
        print()
        print("dry run: no dataset written. Re-run without --dry-run to convert.")
        return 0

    print()
    print(f"writing dataset {args.repo_id} ({total_steps} frames)")
    dataset = build_dataset(args, selected)
    print()
    print(f"done: {dataset.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
