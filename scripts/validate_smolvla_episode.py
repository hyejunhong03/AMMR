#!/usr/bin/env python3
"""Validate one raw AMMR SmolVLA episode directory."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PilImage

from ammr_mycobot_interface import (
    COMMAND_GRIPPER_CLOSE_RAD,
    CONTROL_JOINTS,
    OBJECT_STATE_NAMES,
    SCHEMA_VERSION,
)

FLOAT_TOLERANCE = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a raw SmolVLA episode.")
    parser.add_argument("episode_dir", help="Path to an episode directory.")
    return parser.parse_args()


def fail(errors, message):
    errors.append(message)


def check_schema(errors, metadata, episode_dir):
    """Enforce the frozen dataset contract in docs/smolvla_dataset_schema.md.

    The load-bearing check is that no object_state key survives under "observation".
    That data is simulator ground truth with no real-robot counterpart, so a converter
    that walks "observation" must not be able to pull it into the policy input.
    """
    observation = metadata.get("observation") or {}
    leaked = sorted(key for key in observation if "object_state" in key)
    if leaked:
        errors.append(
            f"privileged keys found under observation: {leaked}. "
            "Run migrate_episode_metadata.py --apply."
        )

    privileged = metadata.get("privileged")
    if privileged is None:
        errors.append("missing top-level 'privileged' block")
    elif privileged.get("policy_input") is not False:
        errors.append("privileged.policy_input must be false")

    if metadata.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version {metadata.get('schema_version')!r} != {SCHEMA_VERSION}"
        )

    instruction = metadata.get("instruction")
    if instruction and metadata.get("task") != instruction:
        errors.append("task must carry the same string as instruction")

    if observation.get("state_dim") != len(CONTROL_JOINTS):
        errors.append(f"observation.state_dim must be {len(CONTROL_JOINTS)}")

    action = metadata.get("action") or {}
    if action.get("type") != "absolute_joint_target":
        errors.append("action.type must be 'absolute_joint_target'")
    if action.get("unit") != "radian":
        errors.append("action.unit must be 'radian'")
    collection = metadata.get("collection") or {}
    expected_gripper_close = collection.get("executed_close_rad", COMMAND_GRIPPER_CLOSE_RAD)
    actual_gripper_close = action.get("gripper_close_rad")
    if (
        actual_gripper_close is None
        or abs(float(actual_gripper_close) - float(expected_gripper_close)) > FLOAT_TOLERANCE
    ):
        errors.append(
            f"action.gripper_close_rad must match executed close {expected_gripper_close}, "
            f"got {actual_gripper_close!r}"
        )

    # Timestamps are optional on pre-migration episodes but must line up when present.
    steps = metadata.get("num_steps")
    for name in ("image_timestamp", "state_timestamp", "action_timestamp"):
        path = Path(episode_dir) / f"{name}.npy"
        if not path.exists():
            continue
        values = np.load(path)
        if values.ndim != 1 or (steps is not None and values.shape[0] != steps):
            errors.append(f"{name}.npy shape {values.shape} does not match num_steps {steps}")


def main():
    args = parse_args()
    episode_dir = Path(args.episode_dir)
    errors = []

    if not episode_dir.is_dir():
        print(f"Not an episode directory: {episode_dir}", file=sys.stderr)
        return 2

    metadata_path = episode_dir / "metadata.json"
    state_path = episode_dir / "robot_state.npy"
    action_path = episode_dir / "action.npy"
    object_state_path = episode_dir / "object_state.npy"
    timestamps_path = episode_dir / "timestamps.npy"
    image_dir = episode_dir / "images"

    for path in [metadata_path, state_path, action_path, timestamps_path]:
        if not path.exists():
            fail(errors, f"missing file: {path.name}")
    if not image_dir.is_dir():
        fail(errors, "missing directory: images")

    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            fail(errors, f"cannot read metadata.json: {exc}")

    state = np.empty((0, len(CONTROL_JOINTS)), dtype=np.float32)
    action = np.empty((0, len(CONTROL_JOINTS)), dtype=np.float32)
    object_state = None
    timestamps = np.empty((0,), dtype=np.float64)
    try:
        if state_path.exists():
            state = np.load(state_path)
        if action_path.exists():
            action = np.load(action_path)
        if object_state_path.exists():
            object_state = np.load(object_state_path)
        if timestamps_path.exists():
            timestamps = np.load(timestamps_path)
    except Exception as exc:
        fail(errors, f"cannot load npy files: {exc}")

    images = sorted(image_dir.glob("*.png")) if image_dir.is_dir() else []
    image_count = len(images)

    if not metadata.get("instruction"):
        fail(errors, "metadata instruction is missing or empty")
    if metadata.get("action", {}).get("type") != "absolute_joint_target":
        fail(errors, "metadata action.type must be absolute_joint_target")

    if state.ndim != 2 or state.shape[1] != len(CONTROL_JOINTS):
        fail(errors, f"robot_state.npy shape must be [T, {len(CONTROL_JOINTS)}], got {state.shape}")
    if action.ndim != 2 or action.shape[1] != len(CONTROL_JOINTS):
        fail(errors, f"action.npy shape must be [T, {len(CONTROL_JOINTS)}], got {action.shape}")
    if timestamps.ndim != 1:
        fail(errors, f"timestamps.npy shape must be [T], got {timestamps.shape}")
    if object_state is not None:
        if object_state.ndim != 2 or object_state.shape[1] != len(OBJECT_STATE_NAMES):
            fail(
                errors,
                f"object_state.npy shape must be [T, {len(OBJECT_STATE_NAMES)}], got {object_state.shape}",
            )

    lengths = {
        "images": image_count,
        "robot_state": int(state.shape[0]) if state.ndim >= 1 else -1,
        "action": int(action.shape[0]) if action.ndim >= 1 else -1,
        "timestamps": int(timestamps.shape[0]) if timestamps.ndim >= 1 else -1,
    }
    if object_state is not None:
        lengths["object_state"] = int(object_state.shape[0]) if object_state.ndim >= 1 else -1
    if len(set(lengths.values())) != 1:
        fail(errors, f"length mismatch: {lengths}")
    if image_count == 0:
        fail(errors, "episode has no images")
    if metadata.get("num_steps") != image_count:
        fail(errors, f"metadata num_steps {metadata.get('num_steps')} != image count {image_count}")

    if images:
        expected_names = [f"{index:06d}.png" for index in range(image_count)]
        actual_names = [path.name for path in images]
        if actual_names != expected_names:
            fail(errors, "image filenames are not contiguous 000000.png..")
        try:
            with PilImage.open(images[0]) as image:
                image.verify()
        except Exception as exc:
            fail(errors, f"first image is not a valid PNG: {exc}")

    check_schema(errors, metadata, episode_dir)

    if errors:
        print("Episode validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Episode validation passed")
    print(f"episode_dir: {episode_dir}")
    print(f"schema_version: {metadata.get('schema_version')}")
    print(f"instruction: {metadata.get('instruction')}")
    print(f"task: {metadata.get('task')}")
    print(f"task_name: {metadata.get('task_name')}")
    print(f"num_steps: {image_count}")
    print(f"robot_state_shape: {tuple(state.shape)}")
    print(f"action_shape: {tuple(action.shape)}")
    if object_state is not None:
        print(f"object_state_shape: {tuple(object_state.shape)}")
    print(f"timestamp_shape: {tuple(timestamps.shape)}")
    print(f"first_image: {images[0].name if images else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
