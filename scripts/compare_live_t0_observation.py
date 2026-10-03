#!/usr/bin/env python3
"""Compare a live reset observation against frame 0 of a recorded episode.

This is the cheapest train/deploy mismatch diagnostic.  If the same object
placement produces a materially different wrist image or robot state before the
first action, rollout failures are not evidence of closed-loop drift yet.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ammr_mycobot_interface import CONTROL_JOINTS, OBJECT_NAMES, object_state_slice  # noqa: E402
from compare_policy_to_episode import load_image_tensor, make_noise  # noqa: E402
from replay_episode_actions import load_object_position  # noqa: E402
from rollout_policy import RolloutClient, image_batch_from_wrist, load_policy, object_xyz  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, help="Raw episode_* directory.")
    parser.add_argument("--object", choices=OBJECT_NAMES, default="red_cube")
    parser.add_argument("--checkpoint", default=None, help="Optional policy checkpoint for t=0 action comparison.")
    parser.add_argument("--task", default=None, help="Defaults to metadata task/instruction.")
    parser.add_argument("--settle-sec", type=float, default=6.0)
    parser.add_argument("--park-non-targets", action="store_true")
    parser.add_argument("--image-layout", choices=("ammr_wrist", "camera2_wrist"), default="ammr_wrist")
    parser.add_argument("--noise-mode", choices=("zero", "fixed-random", "random"), default="fixed-random")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True, help="Directory for JSON and image artifacts.")
    return parser.parse_args()


def metadata_task(episode_dir):
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    return metadata.get("task") or metadata.get("instruction") or "pick the red cube"


def first_image_path(episode_dir):
    images = sorted((episode_dir / "images").glob("*.png"))
    if not images:
        raise ValueError(f"{episode_dir} has no images")
    return images[0]


def image_stats(live, demo):
    live = np.asarray(live, dtype=np.float32)
    demo = np.asarray(demo, dtype=np.float32)
    if live.shape != demo.shape:
        return {
            "same_shape": False,
            "live_shape": list(live.shape),
            "demo_shape": list(demo.shape),
        }
    diff = live - demo
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff * diff)
    return {
        "same_shape": True,
        "shape": list(live.shape),
        "mae_0_255": float(mae),
        "mae_0_1": float(mae / 255.0),
        "max_abs_0_255": float(np.max(np.abs(diff))),
        "rmse_0_255": float(np.sqrt(mse)),
        "per_channel_mae_0_255": np.mean(np.abs(diff), axis=(0, 1)).round(4).tolist(),
    }


def infer_policy_first(checkpoint, image_uint8, state, task, args, sample_index=0):
    import torch

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    policy, preprocessor, postprocessor = load_policy(checkpoint, device, n_action_steps=None)
    policy.reset()
    image = torch.from_numpy(np.ascontiguousarray(image_uint8).copy()).to(device).permute(2, 0, 1).float().div(255.0)
    batch = image_batch_from_wrist(image, args.image_layout)
    batch.update(
        {
            "observation.state": torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
            "task": [task],
        }
    )
    noise = make_noise(policy, args.noise_mode, args.noise_seed, sample_index, device)
    with torch.inference_mode():
        action = policy.predict_action_chunk(preprocessor(batch), noise=noise)
        action = postprocessor({"action": action})
    if isinstance(action, dict):
        action = action["action"]
    return action.squeeze(0).float().cpu().numpy(), device


def main():
    args = parse_args()
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    episode_dir = Path(args.episode_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_state = np.load(episode_dir / "robot_state.npy").astype(np.float32)[0]
    demo_action = np.load(episode_dir / "action.npy").astype(np.float32)[0]
    demo_objects = np.load(episode_dir / "object_state.npy").astype(np.float32)[0]
    object_position = load_object_position(episode_dir, args.object)
    task = args.task or metadata_task(episode_dir)
    demo_image = np.asarray(Image.open(first_image_path(episode_dir)).convert("RGB"), dtype=np.uint8)

    client = RolloutClient(drain=True)
    try:
        client.place_and_reset(
            object_position,
            object_name=args.object,
            settle=args.settle_sec,
            park_non_targets=args.park_non_targets,
        )
        if not client.wait_for_inputs():
            raise RuntimeError("No live image/state feedback after reset")
        client.spin(0.2)
        live_state = np.asarray(client.state, dtype=np.float32).copy()
        live_image = np.asarray(client.image, dtype=np.uint8).copy()
        live_objects = np.asarray(client.objects, dtype=np.float32).copy() if client.objects is not None else None
    finally:
        client.shutdown()

    Image.fromarray(demo_image).save(out_dir / "demo_t0.png")
    Image.fromarray(live_image).save(out_dir / "live_t0.png")
    if demo_image.shape == live_image.shape:
        diff = np.clip(np.abs(live_image.astype(np.int16) - demo_image.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        Image.fromarray(diff).save(out_dir / "absdiff_x4.png")

    result = {
        "episode_dir": str(episode_dir),
        "object": args.object,
        "object_position_requested": object_position.round(8).tolist(),
        "task": task,
        "settle_sec": float(args.settle_sec),
        "image_layout": args.image_layout,
        "image": image_stats(live_image, demo_image),
        "state": {
            "names": list(CONTROL_JOINTS),
            "demo_t0": np.round(demo_state, 6).tolist(),
            "live_t0": np.round(live_state, 6).tolist(),
            "diff": np.round(live_state - demo_state, 6).tolist(),
            "max_abs_diff": float(np.max(np.abs(live_state - demo_state))),
        },
        "object_state": {
            "demo_target": np.round(demo_objects[object_state_slice(args.object)], 8).tolist(),
            "live_target": (
                np.round(object_xyz(live_objects, args.object), 8).tolist() if live_objects is not None else None
            ),
            "target_diff": (
                np.round(object_xyz(live_objects, args.object) - demo_objects[object_state_slice(args.object)], 8).tolist()
                if live_objects is not None
                else None
            ),
        },
        "demo_first_action": np.round(demo_action, 6).tolist(),
    }

    if args.checkpoint:
        demo_chunk, device = infer_policy_first(args.checkpoint, demo_image, demo_state, task, args, sample_index=0)
        live_chunk, _ = infer_policy_first(args.checkpoint, live_image, live_state, task, args, sample_index=0)
        result["policy"] = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "device": device,
            "noise_mode": args.noise_mode,
            "noise_seed": int(args.noise_seed),
            "demo_obs_first_action": np.round(demo_chunk[0], 6).tolist(),
            "live_obs_first_action": np.round(live_chunk[0], 6).tolist(),
            "demo_obs_vs_demo_action_diff": np.round(demo_chunk[0] - demo_action, 6).tolist(),
            "live_obs_vs_demo_action_diff": np.round(live_chunk[0] - demo_action, 6).tolist(),
            "live_vs_demo_policy_first_diff": np.round(live_chunk[0] - demo_chunk[0], 6).tolist(),
            "live_vs_demo_policy_first_arm_max_abs": float(np.max(np.abs(live_chunk[0, :6] - demo_chunk[0, :6]))),
            "live_vs_demo_policy_first_gripper_abs": float(abs(live_chunk[0, 6] - demo_chunk[0, 6])),
        }

    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
