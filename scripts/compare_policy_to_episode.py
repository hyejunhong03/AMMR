#!/usr/bin/env python3
"""Compare a SmolVLA checkpoint's predicted action chunks to one raw episode.

This is an imitation-gap diagnostic. It answers whether a trained checkpoint, given
the exact wrist image and 7-D state recorded in a successful demonstration, predicts
the future action chunk that was actually executed.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ammr_mycobot_interface import CONTROL_JOINTS, GRIPPER_OPEN_RAD  # noqa: E402
from rollout_policy import image_batch_from_wrist  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Policy checkpoint directory.")
    parser.add_argument("--episode-dir", required=True, help="Raw episode_* directory.")
    parser.add_argument("--task", default=None, help="Defaults to metadata task/instruction.")
    parser.add_argument("--indices", nargs="*", type=int, default=None, help="Frame indices to inspect.")
    parser.add_argument("--num-samples", type=int, default=12, help="Used when --indices is omitted.")
    parser.add_argument(
        "--noise-mode",
        choices=("zero", "fixed-random", "random"),
        default="fixed-random",
        help="Noise for flow matching inference.",
    )
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--image-layout", choices=("ammr_wrist", "camera2_wrist"), default="ammr_wrist")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="Write JSON summary.")
    return parser.parse_args()


def load_image_tensor(path, device):
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).to(device).permute(2, 0, 1).float().div(255.0)


def infer_phase_breaks(action):
    gripper = action[:, 6]
    close_candidates = np.where(gripper < GRIPPER_OPEN_RAD - 0.002)[0]
    close_start = int(close_candidates[0]) if len(close_candidates) else len(action)

    min_gripper = float(np.nanmin(gripper)) if len(gripper) else GRIPPER_OPEN_RAD
    close_done_candidates = np.where(gripper <= min_gripper + 0.01)[0]
    close_done_candidates = close_done_candidates[close_done_candidates >= close_start]
    close_done = int(close_done_candidates[0]) if len(close_done_candidates) else close_start

    arm_delta = np.linalg.norm(np.diff(action[:, :6], axis=0), axis=1) if len(action) > 1 else np.asarray([])
    lift_candidates = np.where((np.arange(len(arm_delta)) > close_done) & (arm_delta > 0.002))[0]
    lift_start = int(lift_candidates[0] + 1) if len(lift_candidates) else close_done
    return {
        "close_start": close_start,
        "close_done": close_done,
        "lift_start": lift_start,
    }


def phase_for_index(index, breaks):
    if index < breaks["close_start"]:
        return "preclose"
    if index < breaks["lift_start"]:
        return "close"
    return "lift_or_place"


def default_indices(length, breaks, count):
    anchors = [
        0,
        max(0, breaks["close_start"] - 20),
        max(0, breaks["close_start"] - 5),
        breaks["close_start"],
        breaks["close_done"],
        breaks["lift_start"],
        min(length - 1, breaks["lift_start"] + 10),
        min(length - 1, breaks["lift_start"] + 30),
        length - 1,
    ]
    anchors = [int(i) for i in anchors if 0 <= int(i) < length]
    evenly = np.linspace(0, length - 1, max(count, 1)).astype(int).tolist()
    ordered = []
    for index in anchors + evenly:
        if index not in ordered:
            ordered.append(index)
    return ordered[: max(count, len(anchors))]


def make_noise(policy, mode, seed, sample_index, device):
    if mode == "random":
        return None
    shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
    if mode == "zero":
        return torch.zeros(shape, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + int(sample_index) * 1009)
    return torch.randn(shape, generator=generator, device=device)


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    episode_dir = Path(args.episode_dir).resolve()
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    task = args.task or metadata.get("task") or metadata.get("instruction") or "pick the red cube"
    state = np.load(episode_dir / "robot_state.npy").astype(np.float32)
    action = np.load(episode_dir / "action.npy").astype(np.float32)
    image_files = sorted((episode_dir / "images").glob("*.png"))
    length = min(len(image_files), len(state), len(action))
    state = state[:length]
    action = action[:length]
    image_files = image_files[:length]
    if length == 0:
        raise ValueError(f"{episode_dir} has no frames")

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

    breaks = infer_phase_breaks(action)
    indices = args.indices if args.indices is not None else default_indices(length, breaks, args.num_samples)
    rows = []
    for sample_number, index in enumerate(indices):
        if index < 0 or index >= length:
            continue
        policy.reset()
        image = load_image_tensor(image_files[index], device)
        batch = image_batch_from_wrist(image, args.image_layout)
        batch.update(
            {
                "observation.state": torch.from_numpy(state[index]).to(device).unsqueeze(0),
                "task": [task],
            }
        )
        noise = make_noise(policy, args.noise_mode, args.noise_seed, index, device)
        with torch.inference_mode():
            predicted = policy.predict_action_chunk(preprocessor(batch), noise=noise)
            predicted = postprocessor({"action": predicted})
        if isinstance(predicted, dict):
            predicted = predicted["action"]
        predicted = predicted.squeeze(0).float().cpu().numpy()

        horizon = min(predicted.shape[0], length - index)
        demo = action[index : index + horizon]
        pred = predicted[:horizon, : demo.shape[1]]
        error = pred - demo
        first_error = error[0]
        row = {
            "index": int(index),
            "phase": phase_for_index(index, breaks),
            "horizon": int(horizon),
            "demo_first": np.round(demo[0], 5).tolist(),
            "policy_first": np.round(pred[0], 5).tolist(),
            "first_abs_error": np.round(np.abs(first_error), 5).tolist(),
            "first_arm_max_abs_error": float(np.max(np.abs(first_error[:6]))),
            "first_gripper_abs_error": float(abs(first_error[6])),
            "chunk_mae": np.round(np.mean(np.abs(error), axis=0), 5).tolist(),
            "chunk_max_abs_error": np.round(np.max(np.abs(error), axis=0), 5).tolist(),
            "chunk_arm_mae": float(np.mean(np.abs(error[:, :6]))),
            "chunk_gripper_mae": float(np.mean(np.abs(error[:, 6]))),
            "policy_min": np.round(np.min(pred, axis=0), 5).tolist(),
            "policy_max": np.round(np.max(pred, axis=0), 5).tolist(),
            "demo_min": np.round(np.min(demo, axis=0), 5).tolist(),
            "demo_max": np.round(np.max(demo, axis=0), 5).tolist(),
        }
        rows.append(row)

    phase_summary = {}
    for phase in sorted({row["phase"] for row in rows}):
        phase_rows = [row for row in rows if row["phase"] == phase]
        phase_summary[phase] = {
            "samples": len(phase_rows),
            "first_arm_max_abs_error_mean": float(np.mean([r["first_arm_max_abs_error"] for r in phase_rows])),
            "first_gripper_abs_error_mean": float(np.mean([r["first_gripper_abs_error"] for r in phase_rows])),
            "chunk_arm_mae_mean": float(np.mean([r["chunk_arm_mae"] for r in phase_rows])),
            "chunk_gripper_mae_mean": float(np.mean([r["chunk_gripper_mae"] for r in phase_rows])),
        }

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episode_dir": str(episode_dir),
        "task": task,
        "device": device,
        "noise_mode": args.noise_mode,
        "noise_seed": args.noise_seed,
        "image_layout": args.image_layout,
        "length": int(length),
        "control_joints": list(CONTROL_JOINTS),
        "phase_breaks": breaks,
        "phase_summary": phase_summary,
        "rows": rows,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
