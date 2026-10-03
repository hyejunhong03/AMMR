#!/usr/bin/env python3
"""Check whether a SmolVLA checkpoint is deterministic on fixed observations.

The rollout can only be diagnosed after this passes. If the same image/state/task
produces different action chunks while the robot is not moving, more episodes will not
fix rollout instability; the inference path itself has to be made deterministic first.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ammr_mycobot_interface import CONTROL_JOINTS  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trained policy directory.")
    parser.add_argument("--dataset-root", required=True, help="LeRobot dataset root.")
    parser.add_argument("--repo-id", default="ammr/pick_red_cube_val", help="Dataset repo id.")
    parser.add_argument("--task", default="pick the red cube", help="Language instruction.")
    parser.add_argument("--samples", type=int, default=5, help="Dataset observations to test.")
    parser.add_argument("--repeats", type=int, default=50, help="Repeated inferences per observation.")
    parser.add_argument("--seed", type=int, default=0, help="Python/numpy/torch seed.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--warn-std", type=float, default=1e-5, help="Warn threshold for max std.")
    parser.add_argument("--fail-std", type=float, default=1e-3, help="Fail threshold for max std.")
    parser.add_argument(
        "--noise-mode",
        choices=("random", "fixed-random", "zero"),
        default="random",
        help=(
            "Noise passed to SmolVLA flow matching inference. random matches normal "
            "rollout; fixed-random tests whether stochastic initial noise explains "
            "repeat variation; zero is a deterministic ablation."
        ),
    )
    return parser.parse_args()


def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    seed_everything(args.seed)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        args.checkpoint,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": device}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        args.checkpoint,
        config_filename="policy_postprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
    )
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)

    if dataset.num_frames <= 0:
        raise ValueError(f"{args.dataset_root} has no frames")

    sample_count = min(args.samples, dataset.num_frames)
    indices = np.linspace(0, dataset.num_frames - 1, sample_count).astype(int)
    all_max_std = []

    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset   : {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    print(f"device    : {device}")
    print(f"samples   : {sample_count}, repeats: {args.repeats}")
    print()

    for sample_number, index in enumerate(indices, start=1):
        sample = dataset[int(index)]
        batch = {
            "observation.images.wrist": sample["observation.images.wrist"].to(device).unsqueeze(0),
            "observation.state": sample["observation.state"].to(device).unsqueeze(0),
            "task": [args.task],
        }
        processed = preprocessor(batch)
        fixed_noise = None
        if args.noise_mode != "random":
            shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
            if args.noise_mode == "zero":
                fixed_noise = torch.zeros(shape, device=device)
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(args.seed + int(index))
                fixed_noise = torch.randn(shape, generator=generator, device=device)
        chunks = []
        for _ in range(args.repeats):
            policy.reset()
            with torch.inference_mode():
                noise = fixed_noise.clone() if fixed_noise is not None else None
                chunk = policy.predict_action_chunk(processed, noise=noise)
                chunk = postprocessor({"action": chunk})
            if isinstance(chunk, dict):
                chunk = chunk["action"]
            chunks.append(np.asarray(chunk.squeeze(0).float().cpu()))

        stacked = np.stack(chunks, axis=0)
        std = stacked.std(axis=0)
        max_std = float(std.max())
        all_max_std.append(max_std)
        flat_index = int(std.argmax())
        chunk_step, channel = np.unravel_index(flat_index, std.shape)
        print(
            f"sample {sample_number}/{sample_count} frame={int(index)} "
            f"max_std={max_std:.8g} at step={chunk_step}, "
            f"channel={channel} ({CONTROL_JOINTS[channel]})"
        )

    overall = float(max(all_max_std)) if all_max_std else 0.0
    print()
    print(f"overall max std: {overall:.8g}")
    if overall >= args.fail_std:
        print("DETERMINISM CHECK FAILED")
        return 1
    if overall >= args.warn_std:
        print("DETERMINISM CHECK WARN")
        return 0
    print("DETERMINISM CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
