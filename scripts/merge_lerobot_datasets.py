#!/usr/bin/env python3
"""Merge LeRobot datasets by copying whole episodes.

This keeps task strings intact, so it can build a multi-task dataset such as
red-cube plus blue-cylinder picking. Episodes are selected at episode granularity to
avoid frame-level leakage and to make balancing by object straightforward.
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np


def parse_source(value):
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("sources must be repo_id=root")
    repo_id, root = parts
    if not repo_id or not root:
        raise argparse.ArgumentTypeError("sources must be repo_id=root")
    return repo_id, root


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        required=True,
        help="Source dataset as repo_id=/absolute/root. May be passed more than once.",
    )
    parser.add_argument("--repo-id", required=True, help="Merged dataset repo id.")
    parser.add_argument("--root", required=True, help="Merged dataset root.")
    parser.add_argument("--fps", type=int, default=10, help="Output frame rate.")
    parser.add_argument(
        "--max-episodes-per-source",
        type=int,
        default=None,
        help="Subsample this many episodes from each source before merging.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Episode subsampling/shuffle seed.")
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="Shuffle selected source episodes together instead of writing source-by-source.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing output root.")
    return parser.parse_args()


def frame_from_sample(sample):
    return {
        "observation.images.wrist": (
            sample["observation.images.wrist"].permute(1, 2, 0).numpy() * 255
        ).astype(np.uint8),
        "observation.state": sample["observation.state"].numpy().astype(np.float32),
        "action": sample["action"].numpy().astype(np.float32),
        "task": sample["task"],
    }


def selected_episode_indices(source, max_episodes, rng):
    indices = list(range(source.num_episodes))
    if max_episodes is not None:
        if max_episodes > source.num_episodes:
            raise ValueError(
                f"requested {max_episodes} episodes from {source.root}, "
                f"but it has only {source.num_episodes}"
            )
        rng.shuffle(indices)
        indices = sorted(indices[:max_episodes])
    return indices


def load_sources(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rng = random.Random(args.seed)
    loaded = []
    reference_features = None
    for repo_id, root in args.source:
        dataset = LeRobotDataset(repo_id, root=root)
        if reference_features is None:
            reference_features = dataset.meta.features
        elif dataset.meta.features != reference_features:
            raise ValueError(f"feature mismatch for {root}")
        indices = selected_episode_indices(dataset, args.max_episodes_per_source, rng)
        loaded.append((repo_id, dataset, indices))
        print(f"source {repo_id}: {len(indices)}/{dataset.num_episodes} episodes, {dataset.num_frames} frames")
    return loaded, reference_features


def episode_frame_range(dataset, episode_index):
    rows = {int(row["episode_index"]): row for row in dataset.meta.episodes}
    row = rows[int(episode_index)]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])


def main():
    args = parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sources, features = load_sources(args)
    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{root} exists; pass --overwrite")
        shutil.rmtree(root)

    target = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=root,
        features=features,
        use_videos=False,
    )

    work = []
    for repo_id, dataset, episode_indices in sources:
        for episode_index in episode_indices:
            work.append((repo_id, dataset, episode_index))

    if args.interleave:
        random.Random(args.seed).shuffle(work)

    frames_written = 0
    task_counts = {}
    for repo_id, dataset, episode_index in work:
        start, end = episode_frame_range(dataset, episode_index)
        task = None
        for frame_index in range(start, end):
            sample = dataset[frame_index]
            task = sample["task"]
            target.add_frame(frame_from_sample(sample))
        target.save_episode()
        frames_written += end - start
        task_counts[task] = task_counts.get(task, 0) + 1
        print(f"wrote {repo_id} episode_index={episode_index} ({end - start} frames, task={task})")

    print()
    print(f"done: {root}")
    print(f"episodes: {len(work)}, frames: {frames_written}, task_episode_counts: {task_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
