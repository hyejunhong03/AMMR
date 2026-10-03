#!/usr/bin/env python3
"""Split a LeRobot dataset into train and validation halves by episode.

    /home/autolab/AMMR/.venv/bin/python split_lerobot_dataset.py \
        --root /home/autolab/AMMR/data/lerobot/pick_red_cube

Splitting by episode rather than by frame matters here: consecutive frames inside one
episode are nearly identical, so a frame-level split leaks almost every validation frame
into training and the validation loss stops meaning anything.

Each half is written as its own dataset so the training entry point can take it by
repo-id without needing split support of its own.
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ammr/pick_red_cube", help="Source dataset repo id.")
    parser.add_argument("--root", required=True, help="Source dataset root.")
    parser.add_argument("--train-root", default=None, help="Defaults to <root>_train.")
    parser.add_argument("--val-root", default=None, help="Defaults to <root>_val.")
    parser.add_argument("--val-episodes", type=int, default=15, help="Episodes held out.")
    parser.add_argument("--seed", type=int, default=0, help="Split seed.")
    parser.add_argument("--fps", type=int, default=10, help="Frame rate for the outputs.")
    return parser.parse_args()


def copy_episodes(source, episode_indices, repo_id, root, fps, label):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if root.exists():
        shutil.rmtree(root)

    target = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        features=source.meta.features,
        use_videos=False,
    )

    # Frame ranges per episode, so frames are appended in their original order.
    # lerobot 0.4.x keeps them on meta.episodes as dataset_from_index/dataset_to_index.
    episode_rows = {int(row["episode_index"]): row for row in source.meta.episodes}
    written = 0
    for episode_index in episode_indices:
        row = episode_rows[episode_index]
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        for frame_index in range(start, end):
            sample = source[frame_index]
            frame = {
                "observation.images.wrist": (
                    sample["observation.images.wrist"].permute(1, 2, 0).numpy() * 255
                ).astype(np.uint8),
                "observation.state": sample["observation.state"].numpy(),
                "action": sample["action"].numpy(),
                "task": sample["task"],
            }
            target.add_frame(frame)
        target.save_episode()
        written += end - start
    print(f"{label}: {len(episode_indices)} episodes, {written} frames -> {root}")
    return target


def main():
    args = parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = LeRobotDataset(args.repo_id, root=args.root)
    episodes = list(range(source.num_episodes))
    random.Random(args.seed).shuffle(episodes)

    val = sorted(episodes[: args.val_episodes])
    train = sorted(episodes[args.val_episodes :])
    print(f"source: {source.num_episodes} episodes, {source.num_frames} frames")
    print(f"val episode indices : {val}")

    train_root = args.train_root or f"{args.root}_train"
    val_root = args.val_root or f"{args.root}_val"
    copy_episodes(source, train, f"{args.repo_id}_train", train_root, args.fps, "train")
    copy_episodes(source, val, f"{args.repo_id}_val", val_root, args.fps, "val")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
