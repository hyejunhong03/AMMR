#!/usr/bin/env python3
"""Compute offline SmolVLA loss on a LeRobot dataset split.

This intentionally loads the checkpoint's saved preprocessor so normalization
statistics come from the training split that produced the checkpoint. The
evaluation dataset is used only to provide observations/actions with the same
delta timestamp layout as training.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Policy directory with config/model/processors.")
    parser.add_argument("--dataset-root", required=True, help="LeRobot dataset root to evaluate.")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo id.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means the full split.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Register SmolVLA config choices before parsing train_config.json.
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    checkpoint = Path(args.checkpoint)
    train_config = checkpoint / "train_config.json"
    cfg = TrainPipelineConfig.from_pretrained(train_config)
    cfg.dataset.repo_id = args.repo_id
    cfg.dataset.root = Path(args.dataset_root)
    cfg.dataset.episodes = None
    cfg.dataset.streaming = False

    dataset = make_dataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": device}},
    )

    total_weighted_loss = 0.0
    total_samples = 0
    batches = 0
    loss_items = []
    with torch.inference_mode():
        for batch in dataloader:
            processed = preprocessor(batch)
            loss, output = policy.forward(processed)
            batch_size = int(processed["action"].shape[0])
            loss_value = float(loss.item())
            total_weighted_loss += loss_value * batch_size
            total_samples += batch_size
            batches += 1
            loss_items.append(loss_value)
            if args.max_batches and batches >= args.max_batches:
                break

    mean_loss = total_weighted_loss / max(total_samples, 1)
    result = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(Path(args.dataset_root)),
        "repo_id": args.repo_id,
        "episodes": int(dataset.num_episodes),
        "frames": int(dataset.num_frames),
        "batch_size": int(args.batch_size),
        "batches": int(batches),
        "samples": int(total_samples),
        "mean_loss": mean_loss,
        "batch_loss_min": min(loss_items) if loss_items else None,
        "batch_loss_max": max(loss_items) if loss_items else None,
        "device": device,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
