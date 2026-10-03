#!/usr/bin/env python3
"""Record raw AMMR episodes for SmolVLA fine-tuning."""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
from PIL import Image as PilImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from ammr_mycobot_interface import (
    COMMAND_GRIPPER_CLOSE_RAD,
    CONTROL_JOINTS,
    GRIPPER_OPEN_RAD,
    IMAGE_SIZE,
    OBJECT_STATE_NAMES,
    QUAT_ORDER,
    ROS2_IMAGE_TOPIC,
    ROS2_JOINT_TARGET_TOPIC,
    ROS2_OBJECT_STATES_TOPIC,
    ROS2_ROBOT_STATE_TOPIC,
    SCHEMA_VERSION,
    SIM_GRIPPER_LIMITS,
    coerce_control_targets,
)


DEFAULT_OUTPUT_DIR = "/home/autolab/AMMR/data/smolvla_raw"


def parse_success(value):
    normalized = value.lower()
    if normalized == "unset":
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("success must be true, false, or unset")


def parse_args():
    parser = argparse.ArgumentParser(description="Record one raw SmolVLA episode from AMMR ROS2 topics.")
    parser.add_argument("--instruction", required=True, help="Language instruction for this episode.")
    parser.add_argument("--task-name", required=True, help="Short task name, e.g. pick_red_cube.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Root directory for raw episodes.")
    parser.add_argument("--episode-id", default="auto", help="Episode id or 'auto'.")
    parser.add_argument("--rate-hz", type=float, default=10.0, help="Maximum recording rate.")
    parser.add_argument("--duration-sec", type=float, default=20.0, help="Recording duration. Use 0 for Ctrl-C only.")
    parser.add_argument("--success", type=parse_success, default=None, help="true, false, or unset.")
    parser.add_argument("--image-topic", default=ROS2_IMAGE_TOPIC, help="RGB image topic.")
    parser.add_argument("--state-topic", default=ROS2_ROBOT_STATE_TOPIC, help="Robot state topic.")
    parser.add_argument("--action-topic", default=ROS2_JOINT_TARGET_TOPIC, help="Joint target action topic.")
    parser.add_argument("--object-state-topic", default=ROS2_OBJECT_STATES_TOPIC, help="Optional object position topic.")
    parser.add_argument("--no-object-state", action="store_true", help="Do not record object_state.npy.")
    parser.add_argument(
        "--gripper-close-rad",
        type=float,
        default=COMMAND_GRIPPER_CLOSE_RAD,
        help="Executed gripper close target to record in metadata.",
    )
    parser.add_argument("--node-name", default="smolvla_episode_recorder", help="ROS2 node name.")
    parser.add_argument("--allow-empty", action="store_true", help="Exit successfully even if no steps were recorded.")
    return parser.parse_args()


def next_episode_id(output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    existing = []
    for child in output_path.iterdir():
        if child.is_dir() and child.name.startswith("episode_"):
            suffix = child.name.removeprefix("episode_")
            if suffix.isdigit():
                existing.append(int(suffix))
    return f"episode_{(max(existing) + 1) if existing else 1:06d}"


def image_stamp_seconds(msg):
    stamp = msg.header.stamp
    value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return value if value > 0.0 else time.time()


def image_to_numpy(msg):
    encoding = msg.encoding.lower()
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    channels = channels_by_encoding[encoding]
    row_width = int(msg.width) * channels
    if msg.step < row_width:
        raise ValueError(f"Invalid image step {msg.step} for width={msg.width}, encoding={msg.encoding}")

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    rows = raw.reshape(int(msg.height), int(msg.step))
    pixels = rows[:, :row_width]
    if channels == 1:
        return pixels.reshape(int(msg.height), int(msg.width))

    image = pixels.reshape(int(msg.height), int(msg.width), channels)
    if encoding == "bgr8":
        image = image[:, :, ::-1]
    elif encoding == "rgba8":
        image = image[:, :, :3]
    elif encoding == "bgra8":
        image = image[:, :, [2, 1, 0]]
    return np.ascontiguousarray(image)


class SmolVlaEpisodeRecorder(Node):
    def __init__(self, args, episode_dir):
        super().__init__(args.node_name)
        self._args = args
        self._episode_dir = Path(episode_dir)
        self._image_dir = self._episode_dir / "images"
        self._image_dir.mkdir(parents=True, exist_ok=False)

        self._latest_image = None
        self._latest_image_key = None
        self._latest_state = None
        self._latest_action = None
        self._latest_object_state = None
        self._last_recorded_image_key = None
        self._last_wait_log = 0.0
        self._step_id = 0
        self._timestamps = []
        self._states = []
        self._actions = []
        self._object_states = []
        self._image_files = []
        # Arrival time of the most recent message on each topic. A single episode-wide
        # timestamp cannot show how far the state lags the action, which is exactly the
        # check that catches one channel being rate-limited where it should not be.
        self._latest_image_time = float("nan")
        self._latest_state_time = float("nan")
        self._latest_action_time = float("nan")
        self._image_timestamps = []
        self._state_timestamps = []
        self._action_timestamps = []
        self._started_at = datetime.now(timezone.utc)

        self._image_sub = self.create_subscription(Image, args.image_topic, self._on_image, 10)
        self._state_sub = self.create_subscription(Float64MultiArray, args.state_topic, self._on_state, 10)
        self._action_sub = self.create_subscription(Float64MultiArray, args.action_topic, self._on_action, 10)
        self._object_state_sub = None
        if not args.no_object_state:
            self._object_state_sub = self.create_subscription(
                Float64MultiArray,
                args.object_state_topic,
                self._on_object_state,
                10,
            )
        self._record_timer = self.create_timer(1.0 / max(args.rate_hz, 1e-6), self._record_step)
        self._write_metadata(final=False)

        self.get_logger().info(f"recording episode: {self._episode_dir}")
        self.get_logger().info(f"instruction: {args.instruction}")
        self.get_logger().info(f"waiting for image/state/action topics before saving steps")

    @property
    def num_steps(self):
        return self._step_id

    def _on_image(self, msg):
        self._latest_image = msg
        self._latest_image_key = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec), len(msg.data))
        self._latest_image_time = time.time()

    def _on_state(self, msg):
        try:
            self._latest_state = np.asarray(coerce_control_targets(msg.data), dtype=np.float32)
        except ValueError as exc:
            self.get_logger().warn(f"ignoring invalid state: {exc}")
            return
        self._latest_state_time = time.time()

    def _on_action(self, msg):
        try:
            self._latest_action = np.asarray(coerce_control_targets(msg.data), dtype=np.float32)
        except ValueError as exc:
            self.get_logger().warn(f"ignoring invalid action: {exc}")
            return
        self._latest_action_time = time.time()

    def _on_object_state(self, msg):
        if len(msg.data) < len(OBJECT_STATE_NAMES):
            self.get_logger().warn(
                f"ignoring invalid object state: expected {len(OBJECT_STATE_NAMES)} values, got {len(msg.data)}"
            )
            return
        self._latest_object_state = np.asarray(msg.data[: len(OBJECT_STATE_NAMES)], dtype=np.float32)

    def _missing_inputs(self):
        missing = []
        if self._latest_image is None:
            missing.append("image")
        if self._latest_state is None:
            missing.append("robot_state")
        if self._latest_action is None:
            missing.append("action")
        return missing

    def _record_step(self):
        missing = self._missing_inputs()
        if missing:
            now = time.monotonic()
            if now - self._last_wait_log > 2.0:
                self.get_logger().warn(f"not recording yet; missing: {', '.join(missing)}")
                self._last_wait_log = now
            return
        if self._latest_image_key == self._last_recorded_image_key:
            return

        image_msg = self._latest_image
        state = self._latest_state.copy()
        action = self._latest_action.copy()
        if self._args.no_object_state:
            object_state = None
        elif self._latest_object_state is None:
            object_state = np.full((len(OBJECT_STATE_NAMES),), np.nan, dtype=np.float32)
        else:
            object_state = self._latest_object_state.copy()
        timestamp = image_stamp_seconds(image_msg)
        filename = f"{self._step_id:06d}.png"
        path = self._image_dir / filename

        try:
            array = image_to_numpy(image_msg)
            PilImage.fromarray(array).save(path)
        except Exception as exc:
            self.get_logger().error(f"failed to save image step {self._step_id}: {exc}")
            return

        self._timestamps.append(timestamp)
        self._image_timestamps.append(self._latest_image_time)
        self._state_timestamps.append(self._latest_state_time)
        self._action_timestamps.append(self._latest_action_time)
        self._states.append(state)
        self._actions.append(action)
        if object_state is not None:
            self._object_states.append(object_state)
        self._image_files.append(str(Path("images") / filename))
        self._last_recorded_image_key = self._latest_image_key
        self._step_id += 1

        if self._step_id == 1 or self._step_id % 25 == 0:
            self.get_logger().info(f"recorded {self._step_id} steps")

    def finalize(self):
        np.save(self._episode_dir / "timestamps.npy", np.asarray(self._timestamps, dtype=np.float64))
        np.save(self._episode_dir / "image_timestamp.npy", np.asarray(self._image_timestamps, dtype=np.float64))
        np.save(self._episode_dir / "state_timestamp.npy", np.asarray(self._state_timestamps, dtype=np.float64))
        np.save(self._episode_dir / "action_timestamp.npy", np.asarray(self._action_timestamps, dtype=np.float64))
        state_array = np.asarray(self._states, dtype=np.float32).reshape((-1, len(CONTROL_JOINTS)))
        action_array = np.asarray(self._actions, dtype=np.float32).reshape((-1, len(CONTROL_JOINTS)))
        np.save(self._episode_dir / "robot_state.npy", state_array)
        np.save(self._episode_dir / "action.npy", action_array)
        if not self._args.no_object_state:
            object_state_array = np.asarray(self._object_states, dtype=np.float32).reshape((-1, len(OBJECT_STATE_NAMES)))
            np.save(self._episode_dir / "object_state.npy", object_state_array)
        self._write_metadata(final=True)
        self.get_logger().info(f"finalized episode with {self._step_id} steps: {self._episode_dir}")

    def _write_metadata(self, final):
        metadata = {
            "episode_id": self._episode_dir.name,
            "task_name": self._args.task_name,
            "instruction": self._args.instruction,
            "created_at": self._started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat() if final else None,
            "task": self._args.instruction,
            "observation": {
                "image_topic": self._args.image_topic,
                "image_size": list(IMAGE_SIZE),
                "state_topic": self._args.state_topic,
                "state_dim": len(CONTROL_JOINTS),
                "state_names": list(CONTROL_JOINTS),
                "state_unit": "radian",
            },
            # Simulator ground truth, kept out of "observation" on purpose: a converter that
            # walks observation must not be able to sweep it into the policy input. The real
            # robot has no such signal, so a policy trained on it would not transfer.
            # Legitimate uses are success labelling and scripted demo generation.
            "privileged": {
                "object_state_topic": None if self._args.no_object_state else self._args.object_state_topic,
                "object_state_dim": 0 if self._args.no_object_state else len(OBJECT_STATE_NAMES),
                "object_state_names": [] if self._args.no_object_state else list(OBJECT_STATE_NAMES),
                "quat_order": QUAT_ORDER,
                "policy_input": False,
                "note": "Simulator ground truth. Not available on the real robot.",
            },
            "action": {
                "topic": self._args.action_topic,
                "type": "absolute_joint_target",
                "dim": len(CONTROL_JOINTS),
                "unit": "radian",
                "names": list(CONTROL_JOINTS),
                "gripper_open_rad": GRIPPER_OPEN_RAD,
                "gripper_close_rad": self._args.gripper_close_rad,
                "gripper_limits_rad": list(SIM_GRIPPER_LIMITS),
            },
            "schema_version": SCHEMA_VERSION,
            "success": self._args.success,
            "num_steps": self._step_id,
            "image_files": list(self._image_files),
            "rate_hz": self._args.rate_hz,
            "duration_sec": self._args.duration_sec,
        }
        with open(self._episode_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")


def main():
    args = parse_args()
    episode_id = next_episode_id(args.output_dir) if args.episode_id == "auto" else args.episode_id
    episode_dir = Path(args.output_dir) / episode_id
    if episode_dir.exists():
        print(f"Episode directory already exists: {episode_dir}", file=sys.stderr)
        return 2
    episode_dir.mkdir(parents=True)

    rclpy.init()
    node = SmolVlaEpisodeRecorder(args, episode_dir)
    stop_requested = False
    exit_code = 0

    def _request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, _request_stop)
    start_time = time.monotonic()
    try:
        while rclpy.ok() and not stop_requested:
            if args.duration_sec > 0.0 and time.monotonic() - start_time >= args.duration_sec:
                break
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        node.finalize()
        if node.num_steps == 0 and not args.allow_empty:
            node.get_logger().error("recorded 0 steps; check image/state/action topics")
            exit_code = 1
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
