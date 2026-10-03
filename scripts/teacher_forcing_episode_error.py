#!/usr/bin/env python3
"""Run a checkpoint on recorded observations across an episode.

Unlike a few hand-picked samples, this produces a per-frame teacher-forcing error
curve.  It is used to check whether failures begin in a specific phase such as
descent, gripper close, or lift even when observations are still on-demo.
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
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--image-layout", choices=("ammr_wrist", "camera2_wrist"), default="ammr_wrist")
    parser.add_argument("--noise-mode", choices=("zero", "fixed-random", "random"), default="fixed-random")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-npz", default=None)
    return parser.parse_args()


def metadata_task(episode_dir):
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    return metadata.get("task") or metadata.get("instruction") or "pick the red cube"


def make_indices(length, breaks, args):
    if args.indices is not None:
        return [i for i in args.indices if 0 <= i < length]
    indices = list(range(0, length, max(args.stride, 1)))
    if args.max_frames > 0 and len(indices) > args.max_frames:
        # Keep important phase boundaries while thinning the rest deterministically.
        anchors = default_indices(length, breaks, min(args.max_frames, 12))
        remaining = [i for i in indices if i not in anchors]
        need = max(args.max_frames - len(anchors), 0)
        if need > 0 and remaining:
            pick = np.linspace(0, len(remaining) - 1, need).astype(int)
            anchors += [remaining[int(i)] for i in pick]
        indices = sorted(set(anchors))[: args.max_frames]
    return indices


def summarize_phase(rows, phase):
    phase_rows = [row for row in rows if row["phase"] == phase]
    if not phase_rows:
        return None
    first_arm = np.asarray([row["first_arm_max_abs_error"] for row in phase_rows], dtype=np.float64)
    first_grip = np.asarray([row["first_gripper_abs_error"] for row in phase_rows], dtype=np.float64)
    chunk_arm = np.asarray([row["chunk_arm_mae"] for row in phase_rows], dtype=np.float64)
    chunk_grip = np.asarray([row["chunk_gripper_mae"] for row in phase_rows], dtype=np.float64)
    return {
        "frames": int(len(phase_rows)),
        "first_arm_max_abs_error_mean": float(first_arm.mean()),
        "first_arm_max_abs_error_p95": float(np.quantile(first_arm, 0.95)),
        "first_arm_max_abs_error_max": float(first_arm.max()),
        "first_gripper_abs_error_mean": float(first_grip.mean()),
        "first_gripper_abs_error_p95": float(np.quantile(first_grip, 0.95)),
        "first_gripper_abs_error_max": float(first_grip.max()),
        "chunk_arm_mae_mean": float(chunk_arm.mean()),
        "chunk_arm_mae_p95": float(np.quantile(chunk_arm, 0.95)),
        "chunk_gripper_mae_mean": float(chunk_grip.mean()),
        "chunk_gripper_mae_p95": float(np.quantile(chunk_grip, 0.95)),
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

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    policy, preprocessor, postprocessor = load_policy(args.checkpoint, device, n_action_steps=None)
    breaks = infer_phase_breaks(action)
    indices = make_indices(length, breaks, args)

    rows = []
    first_errors = []
    for index in indices:
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
        pred = predicted.squeeze(0).float().cpu().numpy()
        horizon = min(pred.shape[0], length - index)
        demo = action[index : index + horizon]
        pred = pred[:horizon, : demo.shape[1]]
        error = pred - demo
        first_error = error[0]
        first_errors.append(first_error)
        rows.append(
            {
                "index": int(index),
                "phase": phase_for_index(index, breaks),
                "horizon": int(horizon),
                "first_abs_error": np.round(np.abs(first_error), 6).tolist(),
                "first_arm_max_abs_error": float(np.max(np.abs(first_error[:6]))),
                "first_gripper_abs_error": float(abs(first_error[6])),
                "chunk_mae": np.round(np.mean(np.abs(error), axis=0), 6).tolist(),
                "chunk_max_abs_error": np.round(np.max(np.abs(error), axis=0), 6).tolist(),
                "chunk_arm_mae": float(np.mean(np.abs(error[:, :6]))),
                "chunk_gripper_mae": float(np.mean(np.abs(error[:, 6]))),
            }
        )

    first_errors = np.asarray(first_errors, dtype=np.float32)
    per_dim_abs = np.abs(first_errors)
    phases = sorted(set(row["phase"] for row in rows))
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episode_dir": str(episode_dir),
        "task": task,
        "device": device,
        "noise_mode": args.noise_mode,
        "noise_seed": int(args.noise_seed),
        "image_layout": args.image_layout,
        "length": int(length),
        "evaluated_frames": int(len(rows)),
        "stride": int(args.stride),
        "control_joints": list(CONTROL_JOINTS),
        "phase_breaks": breaks,
        "overall": {
            "first_abs_error_mean": np.mean(per_dim_abs, axis=0).round(6).tolist(),
            "first_abs_error_p95": np.quantile(per_dim_abs, 0.95, axis=0).round(6).tolist(),
            "first_abs_error_max": np.max(per_dim_abs, axis=0).round(6).tolist(),
            "first_arm_max_abs_error_mean": float(np.mean([r["first_arm_max_abs_error"] for r in rows])),
            "first_arm_max_abs_error_p95": float(np.quantile([r["first_arm_max_abs_error"] for r in rows], 0.95)),
            "first_arm_max_abs_error_max": float(np.max([r["first_arm_max_abs_error"] for r in rows])),
            "first_gripper_abs_error_mean": float(np.mean([r["first_gripper_abs_error"] for r in rows])),
            "first_gripper_abs_error_p95": float(np.quantile([r["first_gripper_abs_error"] for r in rows], 0.95)),
            "first_gripper_abs_error_max": float(np.max([r["first_gripper_abs_error"] for r in rows])),
        },
        "phase_summary": {phase: summarize_phase(rows, phase) for phase in phases},
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.out_npz:
        out_npz = Path(args.out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_npz,
            indices=np.asarray([row["index"] for row in rows], dtype=np.int32),
            first_errors=first_errors,
            phase=np.asarray([row["phase"] for row in rows]),
        )
    print(json.dumps({k: result[k] for k in ("episode_dir", "evaluated_frames", "overall", "phase_summary")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
