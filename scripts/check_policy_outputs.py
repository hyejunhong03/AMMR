#!/usr/bin/env python3
"""Inspect a trained policy's outputs without touching the simulator.

    /home/autolab/AMMR/.venv/bin/python check_policy_outputs.py \
        --checkpoint outputs/train/d3_small/checkpoints/002000/pretrained_model \
        --dataset-root data/lerobot/pick_red_cube_val

Runs the checks that do not need a running scene: the checkpoint loads, inference
produces a chunk of the right shape, the output is denormalised back into radians, the
gripper lands near its two commanded values, and every joint target sits inside its URDF
limit. Catching a problem here costs seconds; catching it in a rollout costs a simulator
start plus the rollout itself.

Real observations from the validation split are used rather than random tensors, because
a policy fed noise can produce plausible-looking numbers while being wrong on the actual
input distribution.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ammr_mycobot_interface import (  # noqa: E402
    ARM_JOINTS,
    COMMAND_GRIPPER_CLOSE_RAD,
    CONTROL_JOINTS,
    GRIPPER_OPEN_RAD,
    SIM_GRIPPER_LIMITS,
)

ARM_JOINT_LIMITS = {
    "joint2_to_joint1": (-2.9321, 2.9321),
    "joint3_to_joint2": (-2.4434, 2.4434),
    "joint4_to_joint3": (-2.6179, 2.6179),
    "joint5_to_joint4": (-2.6179, 2.6179),
    "joint6_to_joint5": (-2.7052, 2.7925),
    "joint6output_to_joint6": (-3.14159, 3.14159),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trained policy directory.")
    parser.add_argument("--dataset-root", required=True, help="Dataset to draw observations from.")
    parser.add_argument("--repo-id", default="ammr/pick_red_cube_val", help="Dataset repo id.")
    parser.add_argument("--samples", type=int, default=8, help="Observations to run.")
    parser.add_argument("--task", default="pick the red cube", help="Language instruction.")
    parser.add_argument(
        "--image-layout",
        choices=("ammr_wrist", "camera2_wrist"),
        default="ammr_wrist",
        help=(
            "How to map dataset observation.images.wrist into policy image keys. "
            "camera2_wrist uses black dummy camera1/camera3 and wrist as camera2."
        ),
    )
    parser.add_argument("--device", default="cuda", help="Torch device.")
    return parser.parse_args()


def image_batch_from_wrist(wrist_image, image_layout):
    if image_layout == "ammr_wrist":
        return {"observation.images.wrist": wrist_image}
    if image_layout == "camera2_wrist":
        dummy = wrist_image.new_zeros(wrist_image.shape)
        return {
            "observation.images.camera1": dummy,
            "observation.images.camera2": wrist_image,
            "observation.images.camera3": dummy,
        }
    raise ValueError(f"unknown image layout: {image_layout}")


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    failures = []

    # 1. checkpoint loads
    print("=== 1. checkpoint load ===")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    device = args.device if torch.cuda.is_available() else "cpu"
    policy.to(device)
    policy.eval()
    config = policy.config
    # Normalisation and language tokenisation live in processor pipelines saved next to
    # the weights, not inside the policy. Bypassing them is what makes a policy appear
    # to emit nonsense: the input never gets normalised and the output never gets
    # converted back into radians.
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        args.checkpoint, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch,
        overrides={"device_processor": {"device": device}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        args.checkpoint, config_filename="policy_postprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
    )
    print(f"  loaded on {device}")
    print(f"  preprocessor steps : {[type(s).__name__ for s in preprocessor.steps]}")
    print(f"  postprocessor steps: {[type(s).__name__ for s in postprocessor.steps]}")
    print(f"  chunk_size={config.chunk_size} n_action_steps={config.n_action_steps}")
    print(f"  input_features={sorted(config.input_features)}")
    print(f"  output_features={sorted(config.output_features)}")

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    print(f"  dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    # Dataset action statistics, used below to judge denormalisation.
    stats = dataset.meta.stats["action"]
    ds_min = np.asarray(stats["min"], dtype=np.float64)
    ds_max = np.asarray(stats["max"], dtype=np.float64)
    print(f"  dataset action min: {np.round(ds_min, 4).tolist()}")
    print(f"  dataset action max: {np.round(ds_max, 4).tolist()}")

    indices = np.linspace(0, dataset.num_frames - 1, args.samples).astype(int)

    # 2 & 3. inference runs, chunk shape is (batch, chunk, 7)
    print()
    print("=== 2/3. inference and chunk shape ===")
    chunks = []
    for index in indices:
        sample = dataset[int(index)]
        wrist_image = sample["observation.images.wrist"].to(device).unsqueeze(0)
        batch = image_batch_from_wrist(wrist_image, args.image_layout)
        batch.update(
            {
                "observation.state": sample["observation.state"].to(device).unsqueeze(0),
                "task": [args.task],
            }
        )
        with torch.inference_mode():
            processed = preprocessor(batch)
            chunk = policy.predict_action_chunk(processed)
            # Denormalise back to radians, the same path a rollout takes. The pipeline
            # works on transition dicts, so the raw tensor has to be wrapped.
            chunk = postprocessor({"action": chunk})
        if isinstance(chunk, dict):
            chunk = chunk["action"]
        chunks.append(np.asarray(chunk.squeeze(0).float().cpu()))

    chunk_shape = chunks[0].shape
    print(f"  predict_action_chunk -> {tuple(chunks[0].shape)} per sample")
    expected = (config.chunk_size, len(CONTROL_JOINTS))
    if chunk_shape != expected:
        failures.append(f"chunk shape {chunk_shape} != expected {expected}")
        print(f"  FAIL: expected {expected}")
    else:
        print(f"  ok: chunk x action_dim = {expected}")

    actions = np.concatenate(chunks, axis=0)

    # 4. denormalisation -- values must be radians, not normalised units
    print()
    print("=== 4. unnormalization ===")
    print(f"  policy action min: {np.round(actions.min(axis=0), 4).tolist()}")
    print(f"  policy action max: {np.round(actions.max(axis=0), 4).tolist()}")
    span = ds_max - ds_min
    slack = np.maximum(0.5 * span, 0.15)
    below = actions.min(axis=0) < ds_min - slack
    above = actions.max(axis=0) > ds_max + slack
    if below.any() or above.any():
        offenders = sorted(set(np.where(below)[0]).union(np.where(above)[0]))
        names = [CONTROL_JOINTS[i] for i in offenders]
        failures.append(f"action range far outside dataset range for {names}")
        print(f"  FAIL: outside dataset range for {names}")
    else:
        print("  ok: within the dataset action range plus slack, so output is in radians")

    # 6. gripper sits near its two commanded values
    print()
    print("=== 6. gripper values ===")
    gripper = actions[:, 6]
    near_open = np.abs(gripper - GRIPPER_OPEN_RAD) < 0.05
    near_close = np.abs(gripper - COMMAND_GRIPPER_CLOSE_RAD) < 0.05
    covered = float((near_open | near_close).mean())
    print(f"  range {gripper.min():.4f} .. {gripper.max():.4f} "
          f"(commanded open {GRIPPER_OPEN_RAD}, close {COMMAND_GRIPPER_CLOSE_RAD})")
    print(f"  near open {near_open.mean() * 100:.0f}%, near close {near_close.mean() * 100:.0f}%, "
          f"either {covered * 100:.0f}%")
    low, high = SIM_GRIPPER_LIMITS
    if gripper.min() < low - 0.05 or gripper.max() > high + 0.05:
        failures.append(f"gripper outside {SIM_GRIPPER_LIMITS}")
        print(f"  FAIL: outside {SIM_GRIPPER_LIMITS}")
    else:
        print(f"  ok: inside {SIM_GRIPPER_LIMITS}")

    # 7. arm targets inside URDF limits
    print()
    print("=== 7. joint limits ===")
    violations = []
    for index, name in enumerate(ARM_JOINTS):
        lower, upper = ARM_JOINT_LIMITS[name]
        column = actions[:, index]
        if column.min() < lower or column.max() > upper:
            violations.append(f"{name}: {column.min():.4f}..{column.max():.4f} outside [{lower}, {upper}]")
        print(f"  {name:<24}{column.min():>9.4f} ..{column.max():>9.4f}   [{lower}, {upper}]")
    if violations:
        failures.extend(violations)
        for violation in violations:
            print(f"  FAIL: {violation}")
    else:
        print("  ok: every arm target inside its URDF limit")

    # Chunk continuity: a chunk that jumps wildly step to step would drive the arm hard.
    print()
    print("=== chunk continuity (informational) ===")
    steps = np.abs(np.diff(chunks[0][:, :6], axis=0))
    print(f"  max per-step arm change within one chunk: {steps.max():.4f} rad")
    print(f"  at 10 Hz that is {steps.max() * 10:.3f} rad/s against a 0.5 drive limit")

    print()
    if failures:
        print("POLICY OUTPUT CHECK FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("POLICY OUTPUT CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
