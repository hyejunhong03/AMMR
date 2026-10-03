#!/usr/bin/env python3
"""Check whether a policy's action changes when the wrist image is destroyed.

If normal, black, and wrong-frame images produce nearly the same action while the
state is fixed, the policy is mostly using proprioception/time-like trajectory cues
instead of vision.  This does not prove zero visual usage, but it is a strong warning
for position generalization.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ammr_mycobot_interface import CONTROL_JOINTS  # noqa: E402
from compare_policy_to_episode import (  # noqa: E402
    default_indices,
    infer_phase_breaks,
    load_image_tensor,
    make_noise,
    phase_for_index,
)
from rollout_policy import image_batch_from_wrist, load_policy  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--image-layout", choices=("ammr_wrist", "camera2_wrist"), default="ammr_wrist")
    parser.add_argument("--noise-mode", choices=("zero", "fixed-random", "random"), default="fixed-random")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def metadata_task(episode_dir):
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    return metadata.get("task") or metadata.get("instruction") or "pick the red cube"


def predict_first(policy, preprocessor, postprocessor, image, state, task, args, index, device):
    policy.reset()
    batch = image_batch_from_wrist(image, args.image_layout)
    batch.update(
        {
            "observation.state": torch.from_numpy(state).to(device).unsqueeze(0),
            "task": [task],
        }
    )
    noise = make_noise(policy, args.noise_mode, args.noise_seed, index, device)
    with torch.inference_mode():
        action = policy.predict_action_chunk(preprocessor(batch), noise=noise)
        action = postprocessor({"action": action})
    if isinstance(action, dict):
        action = action["action"]
    return action.squeeze(0).float().cpu().numpy()


def action_delta(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if b.ndim == 1:
        b = b[None, :]
    horizon = min(len(a), len(b))
    a = a[:horizon, :7]
    b = b[:horizon, :7]
    diff = a[0] - b[0]
    return {
        "first_action_diff": np.round(diff, 6).tolist(),
        "first_arm_max_abs": float(np.max(np.abs(diff[:6]))),
        "first_gripper_abs": float(abs(diff[6])),
        "horizon": int(horizon),
        "chunk_arm_mae": float(np.mean(np.abs(a[:, :6] - b[:, :6]))),
        "chunk_gripper_mae": float(np.mean(np.abs(a[:, 6] - b[:, 6]))),
    }


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    episode_dir = Path(args.episode_dir).resolve()
    task = args.task or metadata_task(episode_dir)
    state = np.load(episode_dir / "robot_state.npy").astype(np.float32)
    action = np.load(episode_dir / "action.npy").astype(np.float32)
    image_files = sorted((episode_dir / "images").glob("*.png"))
    length = min(len(image_files), len(state), len(action))
    if length <= 0:
        raise ValueError(f"{episode_dir} has no frames")
    state = state[:length]
    action = action[:length]
    image_files = image_files[:length]

    breaks = infer_phase_breaks(action)
    indices = args.indices if args.indices is not None else default_indices(length, breaks, args.num_samples)
    indices = [int(i) for i in indices if 0 <= int(i) < length]
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    policy, preprocessor, postprocessor = load_policy(args.checkpoint, device, n_action_steps=None)

    rows = []
    for index in indices:
        normal = load_image_tensor(image_files[index], device)
        black = torch.zeros_like(normal)
        wrong_index = (index + max(length // 2, 1)) % length
        wrong = load_image_tensor(image_files[wrong_index], device)
        normal_action = predict_first(policy, preprocessor, postprocessor, normal, state[index], task, args, index, device)
        black_action = predict_first(policy, preprocessor, postprocessor, black, state[index], task, args, index, device)
        wrong_action = predict_first(policy, preprocessor, postprocessor, wrong, state[index], task, args, index, device)
        rows.append(
            {
                "index": int(index),
                "phase": phase_for_index(index, breaks),
                "wrong_image_index": int(wrong_index),
                "normal_first": np.round(normal_action[0, :7], 6).tolist(),
                "demo_first": np.round(action[index, :7], 6).tolist(),
                "normal_vs_demo": action_delta(normal_action, action[index : index + normal_action.shape[0]]),
                "black_vs_normal": action_delta(black_action, normal_action),
                "wrong_vs_normal": action_delta(wrong_action, normal_action),
            }
        )

    def summarize(key):
        arm = np.asarray([row[key]["first_arm_max_abs"] for row in rows], dtype=np.float64)
        grip = np.asarray([row[key]["first_gripper_abs"] for row in rows], dtype=np.float64)
        chunk_arm = np.asarray([row[key]["chunk_arm_mae"] for row in rows], dtype=np.float64)
        return {
            "first_arm_max_abs_mean": float(arm.mean()),
            "first_arm_max_abs_max": float(arm.max()),
            "first_gripper_abs_mean": float(grip.mean()),
            "first_gripper_abs_max": float(grip.max()),
            "chunk_arm_mae_mean": float(chunk_arm.mean()),
            "chunk_arm_mae_max": float(chunk_arm.max()),
        }

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episode_dir": str(episode_dir),
        "task": task,
        "device": device,
        "noise_mode": args.noise_mode,
        "noise_seed": int(args.noise_seed),
        "image_layout": args.image_layout,
        "length": int(length),
        "indices": indices,
        "control_joints": list(CONTROL_JOINTS),
        "summary": {
            "normal_vs_demo": summarize("normal_vs_demo"),
            "black_vs_normal": summarize("black_vs_normal"),
            "wrong_vs_normal": summarize("wrong_vs_normal"),
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": result["summary"], "indices": indices}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
