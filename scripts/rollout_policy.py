#!/usr/bin/env python3
"""Run a trained SmolVLA policy closed-loop against the Isaac scene over ROS2.

    /home/autolab/AMMR/.venv/bin/python rollout_policy.py \
        --checkpoint outputs/train/d3_small/checkpoints/last/pretrained_model \
        --poses-file data/grasp_poses_sampled.json --episodes 5

Runs on the venv interpreter, which has both lerobot and -- via the sourced ROS2
environment -- rclpy. The policy sees exactly what the dataset recorded: the wrist image
and the 7-D joint state, with no privileged object pose, so a rollout here fails the same
way the real robot would.

Actions are executed as chunks. SmolVLA predicts n_action_steps at a time and the runner
publishes them in order at the dataset frame rate, which is how the policy was trained to
be consumed; issuing only the first action of each chunk would change the effective
control rate and is a common reason a policy that trained fine rolls out badly.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from ammr_mycobot_interface import OBJECT_NAMES, object_state_slice


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trained policy directory.")
    parser.add_argument("--poses-file", default=None, help="Pose cache supplying object placements.")
    parser.add_argument("--episodes", type=int, default=5, help="Rollouts to run.")
    parser.add_argument("--start-index", type=int, default=0, help="First pose-cache entry to use.")
    parser.add_argument("--object", choices=OBJECT_NAMES, default="red_cube", help="Target object.")
    parser.add_argument(
        "--park-non-targets",
        action="store_true",
        help=(
            "Move non-target task objects out of the workspace before each rollout. "
            "Use this for single-object evaluation so grasp assist cannot attach a "
            "distractor object first."
        ),
    )
    parser.add_argument("--task", default=None, help="Language instruction.")
    parser.add_argument("--fps", type=float, default=10.0, help="Action publish rate.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help=(
            "Control steps per rollout. Generous on purpose: a run cut off mid-lift is "
            "scored as a failure, and one rollout reached 19.8 mm of a 20 mm threshold "
            "with the cube still rising when the cap hit."
        ),
    )
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        default=True,
        help="End the rollout once the lift threshold is met, rather than running the cap.",
    )
    parser.add_argument("--min-lift-m", type=float, default=0.02, help="Lift counted as success.")
    parser.add_argument("--max-joint-velocity", type=float, default=0.5, help="Spike threshold.")
    parser.add_argument("--out", default=None, help="Write results JSON here.")
    parser.add_argument(
        "--trace-dir",
        default=None,
        help=(
            "Save per-step state, action, object pose and timestamps for each rollout. "
            "Needed to locate when a velocity spike happens rather than only that one did."
        ),
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help=(
            "Override how many actions of each predicted chunk are executed before the "
            "policy sees a new observation. The default of 50 runs open loop for five "
            "seconds at 10 Hz, long enough for error to accumulate; a shorter value "
            "trades inference cost for reactivity."
        ),
    )
    parser.add_argument(
        "--max-action-step",
        type=float,
        default=0.05,
        help=(
            "Cap on how far each published arm joint target may move from the previous "
            "published target, in radians per control step. The default 0.05 rad at "
            "10 Hz matches the 0.5 rad/s arm drive limit. Use 0 to disable and inspect "
            "the raw policy output."
        ),
    )
    parser.add_argument(
        "--dataset-stats",
        default=None,
        help=(
            "Optional LeRobot meta/stats.json. When supplied, traces include per-joint "
            "state/action z-scores and action clamp can use the training distribution."
        ),
    )
    parser.add_argument(
        "--reset-retries",
        type=int,
        default=2,
        help=(
            "Retry scene reset this many times when the initial state is outside the "
            "training distribution. This prevents stale or failed resets from being "
            "counted as policy failures."
        ),
    )
    parser.add_argument(
        "--start-state-z-threshold",
        type=float,
        default=3.0,
        help=(
            "Reject a rollout before inference when the reset state exceeds this "
            "absolute z-score in observation.state. Requires --dataset-stats."
        ),
    )
    parser.add_argument(
        "--start-object-position-threshold",
        type=float,
        default=0.005,
        help="Reject a rollout before inference when the target object reset error exceeds this many metres.",
    )
    parser.add_argument(
        "--start-settle-check-sec",
        type=float,
        default=0.5,
        help="Seconds of post-reset state samples used only to record whether the arm has settled.",
    )
    parser.add_argument(
        "--start-settle-sample-sec",
        type=float,
        default=0.05,
        help="Sampling period for --start-settle-check-sec.",
    )
    parser.add_argument(
        "--start-settle-velocity-threshold",
        type=float,
        default=0.05,
        help="Diagnostic threshold for post-reset arm motion, in rad/s.",
    )
    parser.add_argument(
        "--state-std-floor",
        type=float,
        default=0.0,
        help=(
            "Minimum std used only for rollout z-score diagnostics. This does not change "
            "the policy input unless a checkpoint was trained with the same floor."
        ),
    )
    parser.add_argument(
        "--action-clamp",
        choices=("none", "q01q99", "minmax"),
        default="none",
        help=(
            "Clamp denormalized policy actions to the training action distribution before "
            "the arm step limiter. q01q99 is the cheap deploy-safety test recommended "
            "for OOD action spikes."
        ),
    )
    parser.add_argument(
        "--gripper-snap",
        action="store_true",
        help=(
            "Snap the gripper channel to binary open/close after policy inference. This "
            "is an action postprocessor, not a modular object detector."
        ),
    )
    parser.add_argument("--gripper-open", type=float, default=0.08, help="Open gripper target for --gripper-snap.")
    parser.add_argument("--gripper-close", type=float, default=-0.245, help="Close gripper target for --gripper-snap.")
    parser.add_argument(
        "--gripper-snap-threshold",
        type=float,
        default=None,
        help="Threshold below which --gripper-snap publishes --gripper-close. Default is open/close midpoint.",
    )
    parser.add_argument(
        "--gripper-latch-after-close",
        action="store_true",
        help=(
            "After the gripper first snaps/closes, keep publishing --gripper-close for "
            "the rest of the rollout. This prevents early release during pick "
            "evaluation while leaving arm actions fully policy-driven."
        ),
    )
    parser.add_argument(
        "--gripper-latch-threshold",
        type=float,
        default=None,
        help=(
            "Threshold below which the gripper latch activates. Defaults to "
            "--gripper-snap-threshold when set, otherwise the open/close midpoint."
        ),
    )
    parser.add_argument(
        "--queued-observations",
        action="store_true",
        help=(
            "Read observations the way this script did before the queues were drained: "
            "depth-10 subscriptions and one callback per control step, so the policy "
            "acts on whatever surfaces from a permanently backlogged queue. Kept to A/B "
            "against the drained path; draining is what this script should do, but the "
            "stale path scored higher before the attach test was fixed, so the "
            "comparison is worth being able to repeat."
        ),
    )
    parser.add_argument(
        "--noise-mode",
        choices=("random", "fixed-random", "zero"),
        default="random",
        help=(
            "Noise passed to SmolVLA flow matching inference. random is the library "
            "default; fixed-random makes rollout deterministic for debugging; zero is "
            "a deterministic ablation."
        ),
    )
    parser.add_argument("--noise-seed", type=int, default=0, help="Seed for fixed-random noise.")
    parser.add_argument(
        "--state-jump-event-threshold",
        type=float,
        default=0.2,
        help="Event-level state jump threshold in radians for any arm joint.",
    )
    parser.add_argument(
        "--smooth-command-threshold",
        type=float,
        default=0.02,
        help="Treat an action step below this threshold as smooth for event-level spike classification.",
    )
    parser.add_argument(
        "--stuck-duration-sec",
        type=float,
        default=0.5,
        help="Minimum duration of low state motion and high tracking error counted as a stuck event.",
    )
    parser.add_argument(
        "--stuck-state-motion-threshold",
        type=float,
        default=0.01,
        help="Maximum arm state motion over --stuck-duration-sec counted as stuck.",
    )
    parser.add_argument(
        "--stuck-tracking-error-threshold",
        type=float,
        default=0.15,
        help="Minimum arm target-state error over --stuck-duration-sec counted as stuck.",
    )
    parser.add_argument(
        "--stuck-command-motion-threshold",
        type=float,
        default=0.03,
        help="Minimum command motion over --stuck-duration-sec required for a stuck event.",
    )
    parser.add_argument(
        "--stuck-error-growth-threshold",
        type=float,
        default=0.03,
        help="Minimum tracking-error growth over --stuck-duration-sec required for a stuck event.",
    )
    parser.add_argument(
        "--image-layout",
        choices=("ammr_wrist", "camera2_wrist"),
        default="ammr_wrist",
        help=(
            "How to map the AMMR wrist frame into policy image keys. "
            "camera2_wrist uses SmolVLA's convention: camera1/camera3 are black "
            "dummy images and camera2 is the wrist camera."
        ),
    )
    parser.add_argument("--device", default="cuda", help="Torch device.")
    return parser.parse_args()


# Callbacks drained per control step. Three subscriptions publishing at up to 30 Hz
# against a 10 Hz loop leave at most a handful pending; this is a generous cap, not a
# tuned value.
MAX_SPINS_PER_STEP = 16


class RolloutClient:
    """Minimal ROS2 client: publish joint targets, read state / image / object pose."""

    def __init__(self, drain=True):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_msgs.msg import Empty, Float64MultiArray

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ammr_mycobot_interface import (
            OBJECT_NAMES,
            OBJECT_INDEX_BY_NAME,
            ROS2_GRASP_EVENTS_TOPIC,
            ROS2_IMAGE_TOPIC,
            ROS2_JOINT_TARGET_TOPIC,
            ROS2_OBJECT_STATES_TOPIC,
            ROS2_RESET_TASK_SCENE_TOPIC,
            ROS2_ROBOT_STATE_TOPIC,
            ROS2_SET_OBJECT_POSE_TOPIC,
            TASK_OBJECT_TABLE_Z,
            coerce_control_targets,
        )

        if not rclpy.ok():
            rclpy.init(args=None)
        self._rclpy = rclpy
        self._Float64MultiArray = Float64MultiArray
        self._Empty = Empty
        self._coerce = coerce_control_targets
        self._object_index_by_name = OBJECT_INDEX_BY_NAME
        self._object_names = OBJECT_NAMES
        self._object_state_len = len(self._object_names) * 3
        self._non_target_park_positions = {
            name: (0.55, 0.25 if index % 2 == 0 else -0.25, TASK_OBJECT_TABLE_Z[name])
            for index, name in enumerate(self._object_names)
        }

        class _Node(Node):
            pass

        self.node = _Node("ammr_policy_rollout")
        self.state = None
        self.image = None
        self.objects = None
        self.grasp_events = []

        # Depth 1, not the default 10. Every callback here overwrites a single latest
        # value, so a queued message is only ever a staler version of what the next one
        # says. With depth 10 the rollout read from the back of a permanently full queue
        # and acted on observations up to a second old.
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        if drain:
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        else:
            qos = 10
        self._drain = bool(drain)
        self.node.create_subscription(Float64MultiArray, ROS2_ROBOT_STATE_TOPIC, self._on_state, qos)
        self.node.create_subscription(Image, ROS2_IMAGE_TOPIC, self._on_image, qos)
        self.node.create_subscription(Float64MultiArray, ROS2_OBJECT_STATES_TOPIC, self._on_objects, qos)
        # Events are sparse and meaningful. Unlike state/image samples, dropping an old
        # attach event would erase exactly the evidence this trace is meant to keep.
        self.node.create_subscription(Float64MultiArray, ROS2_GRASP_EVENTS_TOPIC, self._on_grasp_event, 10)
        self.target_pub = self.node.create_publisher(Float64MultiArray, ROS2_JOINT_TARGET_TOPIC, 10)
        self.reset_pub = self.node.create_publisher(Empty, ROS2_RESET_TASK_SCENE_TOPIC, 10)
        self.place_pub = self.node.create_publisher(Float64MultiArray, ROS2_SET_OBJECT_POSE_TOPIC, 10)

    def _on_state(self, msg):
        try:
            self.state = np.asarray(self._coerce(msg.data), dtype=np.float32)
        except ValueError:
            pass

    def _on_image(self, msg):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        rows = raw.reshape(int(msg.height), int(msg.step))
        pixels = rows[:, : int(msg.width) * 3].reshape(int(msg.height), int(msg.width), 3)
        self.image = np.ascontiguousarray(pixels)

    def _on_objects(self, msg):
        if len(msg.data) >= 3:
            self.objects = np.asarray(msg.data[: self._object_state_len], dtype=np.float64)

    def _on_grasp_event(self, msg):
        if msg.data:
            self.grasp_events.append([time.monotonic(), *[float(value) for value in msg.data]])

    def spin(self, seconds=0.0):
        """Drain every pending callback, then keep draining until `seconds` elapse.

        spin_once dispatches at most one callback. Calling it once per control step,
        against three subscriptions, consumed about one message per topic every third
        step, so the policy acted on whatever surfaced from a depth-10 queue that never
        emptied rather than on the newest reading. The sim-side bridge had the same
        defect and was fixed the same way.

        Draining did not by itself improve the success rate -- the failures had another
        cause entirely, in the attach test -- but acting on the freshest observation is
        the behaviour this script is supposed to have, so it stays.
        """
        deadline = time.monotonic() + seconds
        spins = MAX_SPINS_PER_STEP if self._drain else 1
        while True:
            for _ in range(spins):
                self._rclpy.spin_once(self.node, timeout_sec=0.0)
            if time.monotonic() >= deadline:
                return

    def wait_for_inputs(self, timeout=15.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.state is not None and self.image is not None:
                return True
        return False

    def sample_state_motion(self, seconds=0.5, sample_period=0.05):
        samples = []
        stamps = []
        seconds = max(0.0, float(seconds))
        sample_period = max(1e-3, float(sample_period))
        if seconds <= 0.0:
            return np.empty((0, 7), dtype=np.float64), np.asarray([], dtype=np.float64)

        deadline = time.monotonic() + seconds
        next_sample = 0.0
        while time.monotonic() < deadline:
            self.spin(0.0)
            now = time.monotonic()
            if self.state is not None and now >= next_sample:
                samples.append(np.asarray(self.state, dtype=np.float64).copy())
                stamps.append(now)
                next_sample = now + sample_period
            sleep_s = min(0.01, max(0.0, min(deadline, next_sample) - time.monotonic()))
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        self.spin(0.0)
        now = time.monotonic()
        if self.state is not None and (not stamps or now > stamps[-1]):
            samples.append(np.asarray(self.state, dtype=np.float64).copy())
            stamps.append(now)
        return np.asarray(samples, dtype=np.float64), np.asarray(stamps, dtype=np.float64)

    def publish(self, action):
        msg = self._Float64MultiArray()
        msg.data = [float(v) for v in action]
        self.target_pub.publish(msg)
        self.spin(0.0)

    def _place_object(self, position, object_name, settle=2.0):
        msg = self._Float64MultiArray()
        msg.data = [float(self._object_index_by_name[object_name]), *[float(v) for v in position]]
        self.place_pub.publish(msg)
        self.spin(settle)

    def place_and_reset(self, position, object_name="red_cube", settle=6.0, park_non_targets=False):
        self.grasp_events = []
        # Force wait_for_inputs() to observe fresh post-reset samples rather than
        # accepting the final state/image from the previous rollout. A failed reset
        # should be visible as missing or OOD feedback, not silently fed to policy.
        self.state = None
        self.image = None
        self.objects = None
        if park_non_targets:
            for other_name in self._object_names:
                if other_name != object_name:
                    self._place_object(self._non_target_park_positions[other_name], other_name)
        self._place_object(position, object_name)
        self.reset_pub.publish(self._Empty())
        self.spin(settle)

    def clear_grasp_events(self):
        self.grasp_events = []

    def shutdown(self):
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


def load_policy(checkpoint, device, n_action_steps=None):
    """Policy plus its saved processor pipelines.

    Normalisation and language tokenisation live in the pipelines, not the policy, so
    loading the weights alone feeds unnormalised observations in and returns normalised
    actions out -- which looks like a trained policy behaving randomly.
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    if n_action_steps is not None:
        # select_action slices each chunk by config.n_action_steps when filling its
        # queue, so this is all that is needed to change the execution horizon.
        policy.config.n_action_steps = int(n_action_steps)
    policy.to(device)
    policy.eval()
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch,
        overrides={"device_processor": {"device": device}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="policy_postprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, preprocessor, postprocessor


def load_dataset_stats(path, state_std_floor=0.0):
    """Return numpy stats arrays used for diagnostics and optional action clamping."""
    if not path:
        return None
    payload = json.loads(Path(path).read_text())
    stats = {}
    for feature in ("observation.state", "action"):
        feature_stats = payload[feature]
        stats[feature] = {
            name: np.asarray(feature_stats[name], dtype=np.float64)
            for name in ("min", "max", "mean", "std", "q01", "q99")
        }
    if state_std_floor > 0.0:
        stats["observation.state"]["std_for_z"] = np.maximum(
            stats["observation.state"]["std"], float(state_std_floor)
        )
    else:
        stats["observation.state"]["std_for_z"] = stats["observation.state"]["std"].copy()
    stats["action"]["std_for_z"] = stats["action"]["std"].copy()
    return stats


def z_scores(values, feature_stats):
    values = np.asarray(values, dtype=np.float64)
    mean = feature_stats["mean"]
    std = np.maximum(feature_stats["std_for_z"], 1e-9)
    return (values - mean) / std


def start_state_diagnostics(state, dataset_stats, threshold):
    """Return whether the reset state is inside the training distribution."""
    if dataset_stats is None or state is None:
        return {"checked": False, "ok": True}
    z = z_scores(np.asarray(state, dtype=np.float64)[None, :], dataset_stats["observation.state"])[0]
    dim = int(np.nanargmax(np.abs(z)))
    value = float(z[dim])
    return {
        "checked": True,
        "ok": bool(abs(value) <= float(threshold)),
        "max_abs_z": float(abs(value)),
        "max_z": value,
        "dim": dim,
        "dim_1based": dim + 1,
        "state": np.asarray(state, dtype=np.float64).tolist(),
    }


def start_object_diagnostics(objects, object_name, expected_position, threshold):
    """Return whether the target object is still at the requested reset pose."""
    actual = object_xyz(objects, object_name)
    if actual is None or expected_position is None or not np.all(np.isfinite(actual)):
        return {"checked": False, "ok": True}
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected_position, dtype=np.float64)
    error = float(np.linalg.norm(actual - expected))
    return {
        "checked": True,
        "ok": bool(error <= float(threshold)),
        "error_m": error,
        "threshold_m": float(threshold),
        "actual": actual.tolist(),
        "expected": expected.tolist(),
    }


def start_motion_diagnostics(samples, stamps, threshold=0.05):
    """Estimate residual post-reset arm motion from short state sampling."""
    samples = np.asarray(samples, dtype=np.float64)
    stamps = np.asarray(stamps, dtype=np.float64)
    if len(samples) < 2 or len(stamps) < 2:
        return {"checked": False, "ok": True}
    dt = np.diff(stamps)
    dt[dt <= 0.0] = 1e-3
    state_delta = np.diff(samples[:, :6], axis=0)
    velocity = np.abs(state_delta) / dt[:, None]
    peak_flat = int(np.nanargmax(velocity))
    step, joint = np.unravel_index(peak_flat, velocity.shape)
    max_velocity = float(velocity[step, joint])
    return {
        "checked": True,
        "ok": bool(max_velocity <= float(threshold)),
        "threshold_rad_s": float(threshold),
        "sample_count": int(len(samples)),
        "duration_s": float(stamps[-1] - stamps[0]),
        "max_joint_velocity": max_velocity,
        "max_state_step": float(np.max(np.abs(state_delta))),
        "peak_joint": int(joint + 1),
        "peak_step_before": int(step),
        "peak_step_after": int(step + 1),
    }


def clamp_action_to_stats(action, stats, mode):
    if stats is None or mode == "none":
        return action, False
    feature = stats["action"]
    if mode == "q01q99":
        lower = feature["q01"]
        upper = feature["q99"]
    elif mode == "minmax":
        lower = feature["min"]
        upper = feature["max"]
    else:
        raise ValueError(f"unknown action clamp mode {mode!r}")
    clipped = np.clip(action, lower, upper)
    return clipped, bool(np.any(np.abs(clipped - action) > 1e-9))


def snap_gripper(action, args):
    if not args.gripper_snap:
        return action, False
    snapped = action.copy()
    threshold = args.gripper_snap_threshold
    if threshold is None:
        threshold = 0.5 * (float(args.gripper_open) + float(args.gripper_close))
    target = float(args.gripper_close) if float(action[6]) < float(threshold) else float(args.gripper_open)
    changed = abs(float(snapped[6]) - target) > 1e-9
    snapped[6] = target
    return snapped, changed


def tcp_diagnostics(states, objects, object_name):
    """TCP-to-target distance from numpy FK. Missing values remain NaN."""
    if len(states) == 0:
        return np.asarray([], dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    try:
        from ammr_ik import tcp_from_joints
    except Exception:
        return (
            np.full((len(states),), np.nan, dtype=np.float64),
            np.full((len(states), 3), np.nan, dtype=np.float64),
        )
    tcp_positions = []
    distances = []
    object_array = np.asarray(objects, dtype=np.float64)
    for idx, state in enumerate(states):
        try:
            tcp = np.asarray(tcp_from_joints(state[:6]), dtype=np.float64)
        except Exception:
            tcp = np.full((3,), np.nan, dtype=np.float64)
        tcp_positions.append(tcp)
        target = object_xyz(object_array[idx], object_name) if idx < len(object_array) else None
        if target is None or not np.all(np.isfinite(tcp)) or not np.all(np.isfinite(target)):
            distances.append(np.nan)
        else:
            distances.append(float(np.linalg.norm(tcp - target)))
    return np.asarray(distances, dtype=np.float64), np.asarray(tcp_positions, dtype=np.float64)


def default_task(object_name):
    tasks = {
        "red_cube": "pick the red cube",
        "blue_cylinder": "pick the blue cylinder",
    }
    return tasks.get(object_name, f"pick the {object_name.replace('_', ' ')}")


def object_xyz(objects, object_name):
    if objects is None:
        return None
    return objects[object_state_slice(object_name)]


def image_batch_from_wrist(image, image_layout):
    if image_layout == "ammr_wrist":
        return {"observation.images.wrist": image.unsqueeze(0)}
    if image_layout == "camera2_wrist":
        dummy = image.new_zeros(image.shape)
        return {
            "observation.images.camera1": dummy.unsqueeze(0),
            "observation.images.camera2": image.unsqueeze(0),
            "observation.images.camera3": dummy.unsqueeze(0),
        }
    raise ValueError(f"unknown image layout: {image_layout}")


def velocity_diagnostics(states, actions, raw_actions, stamps, grasp_events, max_joint_velocity, max_action_step):
    """Classify the largest observed state velocity.

    A rollout can violate the velocity threshold for different reasons. The important
    split is whether the policy asked for a large command step, or whether the state
    jumped even though both the raw and published actions were smooth. The latter is a
    simulator/constraint artifact and should not be counted as a policy action jump.
    """
    states = np.asarray(states)
    actions = np.asarray(actions)
    raw_actions = np.asarray(raw_actions)
    stamps = np.asarray(stamps)
    if len(states) <= 1 or len(actions) <= 1 or len(stamps) <= 1:
        return {
            "velocity_ok": True,
            "velocity_violation_type": "none",
        }

    changed = [0] + [i for i in range(1, len(states)) if not np.allclose(states[i], states[i - 1])]
    result = {
        "state_change_steps": len(changed),
        "state_change_hz": round(float(len(changed) / max(stamps[-1] - stamps[0], 1e-6)), 2),
    }
    if len(changed) <= 1:
        result.update(
            {
                "max_joint_velocity": 0.0,
                "velocity_ok": True,
                "velocity_violation_type": "none",
            }
        )
        return result

    unique_states = states[changed]
    unique_stamps = stamps[changed]
    dt = np.diff(unique_stamps)
    dt[dt <= 0] = 1e-3
    velocity = np.abs(np.diff(unique_states[:, :6], axis=0)) / dt[:, None]
    peak_flat = int(np.argmax(velocity))
    peak_changed_index, peak_joint = np.unravel_index(peak_flat, velocity.shape)
    before = int(changed[peak_changed_index])
    after = int(changed[peak_changed_index + 1])

    state_delta = states[after, :6] - states[before, :6]
    action_delta = actions[after, :6] - actions[before, :6]
    raw_action_delta = raw_actions[after, :6] - raw_actions[before, :6]
    peak_velocity = float(velocity[peak_changed_index, peak_joint])
    peak_state_step = float(np.max(np.abs(state_delta)))
    peak_action_step = float(np.max(np.abs(action_delta)))
    peak_raw_action_step = float(np.max(np.abs(raw_action_delta)))
    tracking_error_before = float(np.max(np.abs(states[before, :6] - actions[before, :6])))
    tracking_error_after = float(np.max(np.abs(states[after, :6] - actions[after, :6])))

    nearest_event_delta = None
    nearest_attach_delta = None
    grasp_events = np.asarray(grasp_events, dtype=np.float64)
    if grasp_events.ndim == 1 and grasp_events.size:
        grasp_events = grasp_events.reshape(1, -1)
    if grasp_events.ndim == 2 and grasp_events.size and grasp_events.shape[1] >= 3:
        peak_time = float(stamps[before])
        event_deltas = grasp_events[:, 0] - peak_time
        nearest_event_delta = float(event_deltas[int(np.argmin(np.abs(event_deltas)))])
        attach_events = grasp_events[np.isclose(grasp_events[:, 2], 1.0)]
        if len(attach_events):
            attach_deltas = attach_events[:, 0] - peak_time
            nearest_attach_delta = float(attach_deltas[int(np.argmin(np.abs(attach_deltas)))])

    velocity_ok = bool(peak_velocity < max_joint_velocity)
    violation_type = "none"
    if not velocity_ok:
        command_is_smooth = peak_action_step <= 0.02 and peak_raw_action_step <= 0.02
        state_jump_is_large = peak_state_step >= 0.20
        if command_is_smooth and state_jump_is_large:
            violation_type = "sim_state_spike"
        elif max_action_step > 0.0 and peak_raw_action_step > max_action_step * 1.5:
            violation_type = "policy_action_jump"
        elif max_action_step > 0.0 and peak_action_step >= max_action_step * 0.95:
            violation_type = "tracking_after_limited_command"
        else:
            violation_type = "tracking_state_spike"

    result.update(
        {
            "max_joint_velocity": peak_velocity,
            "velocity_ok": velocity_ok,
            "velocity_violation_type": violation_type,
            "velocity_peak_joint": int(peak_joint + 1),
            "velocity_peak_step_before": before,
            "velocity_peak_step_after": after,
            "velocity_peak_time_rel_s": float(stamps[before] - stamps[0]),
            "velocity_peak_dt_s": float(dt[peak_changed_index]),
            "velocity_peak_state_step": peak_state_step,
            "velocity_peak_action_step": peak_action_step,
            "velocity_peak_raw_action_step": peak_raw_action_step,
            "velocity_peak_tracking_error_before": tracking_error_before,
            "velocity_peak_tracking_error_after": tracking_error_after,
            "velocity_peak_nearest_event_delta_s": nearest_event_delta,
            "velocity_peak_nearest_attach_delta_s": nearest_attach_delta,
        }
    )
    return result


def event_diagnostics(
    states,
    actions,
    raw_actions,
    stamps,
    state_jump_threshold=0.2,
    smooth_command_threshold=0.02,
    stuck_duration_sec=0.5,
    stuck_state_motion_threshold=0.01,
    stuck_tracking_error_threshold=0.15,
    stuck_command_motion_threshold=0.03,
    stuck_error_growth_threshold=0.03,
):
    """Classify multi-step state jump and stuck/contact events.

    velocity_diagnostics intentionally reports the single largest velocity sample.
    This event-level pass records whether any arm joint jumped under smooth commands,
    and whether the arm was effectively stuck while target-state error stayed large.
    """
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    raw_actions = np.asarray(raw_actions, dtype=np.float64)
    stamps = np.asarray(stamps, dtype=np.float64)
    result = {
        "state_jump_event": False,
        "state_jump_event_count": 0,
        "stuck_event": False,
        "stuck_event_count": 0,
    }
    if len(states) < 2 or len(actions) < 2 or len(stamps) < 2:
        return result

    state_delta = np.diff(states[:, :6], axis=0)
    action_delta = np.diff(actions[:, :6], axis=0)
    raw_action_delta = np.diff(raw_actions[:, :6], axis=0)
    state_step = np.max(np.abs(state_delta), axis=1)
    action_step = np.max(np.abs(action_delta), axis=1)
    raw_action_step = np.max(np.abs(raw_action_delta), axis=1)
    jump_mask = (
        (state_step >= float(state_jump_threshold))
        & (action_step <= float(smooth_command_threshold))
        & (raw_action_step <= float(smooth_command_threshold))
    )
    jump_indices = np.flatnonzero(jump_mask)
    result["state_jump_event_count"] = int(len(jump_indices))
    if len(jump_indices):
        idx = int(jump_indices[0])
        peak_joint = int(np.argmax(np.abs(state_delta[idx])) + 1)
        result.update(
            {
                "state_jump_event": True,
                "state_jump_first_step_before": idx,
                "state_jump_first_step_after": idx + 1,
                "state_jump_first_time_rel_s": float(stamps[idx] - stamps[0]),
                "state_jump_first_joint": peak_joint,
                "state_jump_first_state_step": float(state_step[idx]),
                "state_jump_first_action_step": float(action_step[idx]),
                "state_jump_first_raw_action_step": float(raw_action_step[idx]),
                "state_jump_max_state_step": float(np.max(state_step[jump_indices])),
            }
        )

    duration = float(stuck_duration_sec)
    if duration <= 0.0:
        return result
    tracking_error = np.max(np.abs(actions[:, :6] - states[:, :6]), axis=1)
    stuck_events = []
    j = 0
    for i in range(len(states)):
        j = max(j, i + 1)
        while j < len(states) and stamps[j] - stamps[i] < duration:
            j += 1
        if j >= len(states):
            break
        state_motion = float(np.max(np.abs(states[j, :6] - states[i, :6])))
        command_motion = float(np.max(np.abs(actions[j, :6] - actions[i, :6])))
        max_error = float(np.max(tracking_error[i : j + 1]))
        error_growth = float(tracking_error[j] - tracking_error[i])
        if (
            state_motion <= float(stuck_state_motion_threshold)
            and command_motion >= float(stuck_command_motion_threshold)
            and max_error >= float(stuck_tracking_error_threshold)
            and error_growth >= float(stuck_error_growth_threshold)
        ):
            stuck_events.append((i, j, state_motion, command_motion, max_error, error_growth))
            while i + 1 < len(states) and stamps[i + 1] <= stamps[j]:
                i += 1
    result["stuck_event_count"] = int(len(stuck_events))
    if stuck_events:
        i, j, state_motion, command_motion, max_error, error_growth = stuck_events[0]
        result.update(
            {
                "stuck_event": True,
                "stuck_first_step_start": int(i),
                "stuck_first_step_end": int(j),
                "stuck_first_time_rel_s": float(stamps[i] - stamps[0]),
                "stuck_first_duration_s": float(stamps[j] - stamps[i]),
                "stuck_first_state_motion": state_motion,
                "stuck_first_command_motion": command_motion,
                "stuck_first_tracking_error": max_error,
                "stuck_first_tracking_error_growth": error_growth,
            }
        )
    return result


def _first_z_exceedance(z_values, threshold=3.0):
    z_values = np.asarray(z_values, dtype=np.float64)
    if z_values.size == 0:
        return None
    hits = np.argwhere(np.abs(z_values) > float(threshold))
    if hits.size == 0:
        return None
    step, dim = [int(v) for v in hits[0]]
    return {
        "step": step,
        "dim": dim,
        "dim_1based": dim + 1,
        "z": float(z_values[step, dim]),
    }


def _max_abs_with_index(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    flat = int(np.nanargmax(np.abs(values)))
    index = np.unravel_index(flat, values.shape)
    return {
        "value": float(values[index]),
        "abs": float(abs(values[index])),
        "index": [int(v) for v in index],
    }


def run_episode(client, policy, preprocessor, postprocessor, args, object_position, object_name, device, dataset_stats=None):
    import torch

    reset_diagnostics = []
    start_diagnostics = {"checked": False, "ok": True}
    start_object_diag = {"checked": False, "ok": True}
    start_motion_diag = {"checked": False, "ok": True}
    feedback_ok = False
    start_ok = True
    for attempt in range(int(args.reset_retries) + 1):
        client.place_and_reset(object_position, object_name=object_name, park_non_targets=args.park_non_targets)
        feedback_ok = client.wait_for_inputs()
        if not feedback_ok:
            reset_diagnostics.append({"attempt": attempt + 1, "error": "no image/state feedback"})
            continue
        start_diagnostics = start_state_diagnostics(
            client.state,
            dataset_stats,
            args.start_state_z_threshold,
        )
        start_object_diag = start_object_diagnostics(
            client.objects,
            object_name,
            object_position,
            args.start_object_position_threshold,
        )
        settle_samples, settle_stamps = client.sample_state_motion(
            args.start_settle_check_sec,
            args.start_settle_sample_sec,
        )
        start_motion_diag = start_motion_diagnostics(
            settle_samples,
            settle_stamps,
            args.start_settle_velocity_threshold,
        )
        start_ok = bool(start_diagnostics["ok"] and start_object_diag["ok"])
        reset_diagnostics.append(
            {
                "attempt": attempt + 1,
                **start_diagnostics,
                "start_object": start_object_diag,
                "start_motion": start_motion_diag,
                "start_ok": start_ok,
            }
        )
        if start_ok:
            break
    if not feedback_ok:
        return {
            "object_name": object_name,
            "target_object": object_name,
            "object_position": list(map(float, object_position)),
            "cube_position": list(map(float, object_position)),
            "error": "no image/state feedback",
            "noise_mode": args.noise_mode,
            "noise_seed": int(args.noise_seed),
            "reset_attempts": len(reset_diagnostics),
            "reset_diagnostics": reset_diagnostics,
            "invalid_rollout": True,
            "success": False,
        }
    if not start_ok:
        if not start_diagnostics["ok"]:
            error = "reset_state_out_of_distribution"
        elif not start_object_diag["ok"]:
            error = "reset_object_position_error"
        else:
            error = "reset_start_invalid"
        return {
            "object_name": object_name,
            "target_object": object_name,
            "object_position": list(map(float, object_position)),
            "cube_position": list(map(float, object_position)),
            "error": error,
            "noise_mode": args.noise_mode,
            "noise_seed": int(args.noise_seed),
            "reset_attempts": len(reset_diagnostics),
            "reset_diagnostics": reset_diagnostics,
            "start_state_z_max": start_diagnostics,
            "start_object_position": start_object_diag,
            "start_motion": start_motion_diag,
            "start_object_position_threshold_m": float(args.start_object_position_threshold),
            "start_settle_check_sec": float(args.start_settle_check_sec),
            "start_settle_velocity_threshold": float(args.start_settle_velocity_threshold),
            "invalid_rollout": True,
            "success": False,
        }
    # Placement/reset can deliver sparse detach/attach events from the previous run
    # after the buffer was initially cleared. Keep only events generated by this
    # policy rollout.
    client.clear_grasp_events()

    policy.reset()
    start_objects = client.objects.copy() if client.objects is not None else None
    start_target = object_xyz(start_objects, object_name)
    period = 1.0 / max(args.fps, 1e-6)

    states, actions, raw_actions, policy_raw_actions, action_limited = [], [], [], [], []
    action_clamped, gripper_snapped, gripper_latched = [], [], []
    raw_arm_steps, published_arm_steps = [], []
    object_z, stamps, objects = [], [], []
    fixed_noise = None
    if args.noise_mode != "random":
        shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
        if args.noise_mode == "zero":
            fixed_noise = torch.zeros(shape, device=device)
        else:
            generator = torch.Generator(device=device)
            seed_offset = int(round(float(object_position[0]) * 10000.0))
            seed_offset += int(round(float(object_position[1]) * 10000.0)) * 1009
            generator.manual_seed(args.noise_seed + seed_offset)
            fixed_noise = torch.randn(shape, generator=generator, device=device)
    task = args.task or default_task(object_name)
    next_deadline = time.monotonic()
    latch_active = False
    latch_threshold = args.gripper_latch_threshold
    if latch_threshold is None:
        latch_threshold = args.gripper_snap_threshold
    if latch_threshold is None:
        latch_threshold = 0.5 * (float(args.gripper_open) + float(args.gripper_close))
    for _ in range(args.max_steps):
        client.spin(0.0)
        if client.state is None or client.image is None:
            break

        image = torch.from_numpy(client.image).to(device).permute(2, 0, 1).float().div(255.0)
        batch = image_batch_from_wrist(image, args.image_layout)
        batch.update(
            {
                "observation.state": torch.from_numpy(client.state).to(device).unsqueeze(0),
                "task": [task],
            }
        )
        with torch.inference_mode():
            noise = fixed_noise.clone() if fixed_noise is not None else None
            action = policy.select_action(preprocessor(batch), noise=noise)
            action = postprocessor({"action": action})
        if isinstance(action, dict):
            action = action["action"]
        action = action.squeeze(0).float().cpu().numpy()
        policy_raw_action = action.copy()

        action, clamped = clamp_action_to_stats(action, dataset_stats, args.action_clamp)
        action, snapped = snap_gripper(action, args)
        latched = False
        if args.gripper_latch_after_close:
            if latch_active or float(action[6]) < float(latch_threshold):
                latch_active = True
                latched = abs(float(action[6]) - float(args.gripper_close)) > 1e-9
                action[6] = float(args.gripper_close)
        raw_action = action.copy()

        reference_arm = actions[-1][:6] if len(actions) else client.state[:6]
        raw_delta = raw_action[:6] - reference_arm
        limited = False
        if args.max_action_step > 0.0:
            clipped_delta = np.clip(raw_delta, -args.max_action_step, args.max_action_step)
            limited = bool(np.any(np.abs(clipped_delta - raw_delta) > 1e-7))
            action[:6] = reference_arm + clipped_delta
        published_delta = action[:6] - reference_arm

        states.append(client.state.copy())
        policy_raw_actions.append(policy_raw_action.copy())
        raw_actions.append(raw_action.copy())
        actions.append(action.copy())
        action_limited.append(limited)
        action_clamped.append(clamped)
        gripper_snapped.append(snapped)
        gripper_latched.append(latched)
        raw_arm_steps.append(float(np.max(np.abs(raw_delta))))
        published_arm_steps.append(float(np.max(np.abs(published_delta))))
        stamps.append(time.monotonic())
        if client.objects is not None:
            target = object_xyz(client.objects, object_name)
            object_z.append(float(target[2]) if target is not None else float("nan"))
            objects.append(client.objects.copy())
        else:
            objects.append(np.full(6, np.nan))

        client.publish(action)
        now = time.monotonic()
        next_deadline += period
        if next_deadline < now:
            # Inference can overrun a control period, especially at chunk boundaries.
            # Keep the next publish one period after the latest publish instead of
            # catching up with a near-zero dt command that appears as a velocity spike.
            next_deadline = now + period
        sleep_s = next_deadline - now
        if sleep_s > 0.0:
            time.sleep(sleep_s)

        # Stop as soon as the objective is met so a generous cap costs nothing.
        if (
            args.stop_on_success
            and start_target is not None
            and object_z
            and max(object_z) - start_target[2] >= args.min_lift_m
        ):
            break

    states = np.asarray(states)
    actions = np.asarray(actions)
    raw_actions_array = np.asarray(raw_actions)
    policy_raw_actions_array = np.asarray(policy_raw_actions)
    objects_array = np.asarray(objects)
    tcp_target_distances, tcp_positions = tcp_diagnostics(states, objects_array, object_name)
    trace = {
        "states": states,
        "actions": actions,
        "raw_actions": raw_actions_array,
        "policy_raw_actions": policy_raw_actions_array,
        "action_limited": np.asarray(action_limited, dtype=np.bool_),
        "action_clamped": np.asarray(action_clamped, dtype=np.bool_),
        "gripper_snapped": np.asarray(gripper_snapped, dtype=np.bool_),
        "gripper_latched": np.asarray(gripper_latched, dtype=np.bool_),
        "raw_arm_steps": np.asarray(raw_arm_steps),
        "published_arm_steps": np.asarray(published_arm_steps),
        "objects": objects_array,
        "stamps": np.asarray(stamps),
        "grasp_events": np.asarray(client.grasp_events, dtype=np.float64),
        "tcp_positions": tcp_positions,
        "tcp_target_distances": tcp_target_distances,
    }
    if dataset_stats is not None and len(states):
        state_z = z_scores(states, dataset_stats["observation.state"])
        action_z = z_scores(actions, dataset_stats["action"])
        raw_action_z = z_scores(raw_actions_array, dataset_stats["action"])
        policy_raw_action_z = z_scores(policy_raw_actions_array, dataset_stats["action"])
        trace.update(
            {
                "state_z": state_z,
                "action_z": action_z,
                "raw_action_z": raw_action_z,
                "policy_raw_action_z": policy_raw_action_z,
                "target_state_error": actions - states,
                "raw_target_state_error": raw_actions_array - states,
                "policy_target_state_error": policy_raw_actions_array - states,
            }
        )
    result = {
        "object_name": object_name,
        "target_object": object_name,
        "object_position": list(map(float, object_position)),
        "cube_position": list(map(float, object_position)),
        "task": task,
        "noise_mode": args.noise_mode,
        "noise_seed": int(args.noise_seed),
        "reset_attempts": len(reset_diagnostics),
        "reset_diagnostics": reset_diagnostics,
        "start_state_z_max": start_diagnostics if start_diagnostics.get("checked") else None,
        "start_object_position": start_object_diag if start_object_diag.get("checked") else None,
        "start_motion": start_motion_diag if start_motion_diag.get("checked") else None,
        "start_object_position_threshold_m": float(args.start_object_position_threshold),
        "start_settle_check_sec": float(args.start_settle_check_sec),
        "start_settle_velocity_threshold": float(args.start_settle_velocity_threshold),
        "state_jump_event_threshold": float(args.state_jump_event_threshold),
        "smooth_command_threshold": float(args.smooth_command_threshold),
        "stuck_duration_sec": float(args.stuck_duration_sec),
        "stuck_state_motion_threshold": float(args.stuck_state_motion_threshold),
        "stuck_tracking_error_threshold": float(args.stuck_tracking_error_threshold),
        "stuck_command_motion_threshold": float(args.stuck_command_motion_threshold),
        "stuck_error_growth_threshold": float(args.stuck_error_growth_threshold),
        "steps": int(len(states)),
        "action_min": np.round(actions.min(axis=0), 4).tolist() if len(actions) else None,
        "action_max": np.round(actions.max(axis=0), 4).tolist() if len(actions) else None,
        "gripper_range": (
            [float(actions[:, 6].min()), float(actions[:, 6].max())] if len(actions) else None
        ),
        "max_action_step_limit": float(args.max_action_step),
        "action_clamp": args.action_clamp,
        "action_clamped_steps": int(np.sum(action_clamped)),
        "gripper_snap": bool(args.gripper_snap),
        "gripper_snapped_steps": int(np.sum(gripper_snapped)),
        "gripper_latch_after_close": bool(args.gripper_latch_after_close),
        "gripper_latch_threshold": float(latch_threshold),
        "gripper_latched_steps": int(np.sum(gripper_latched)),
        "limited_steps": int(np.sum(action_limited)),
        "raw_arm_step_max": float(np.max(raw_arm_steps)) if raw_arm_steps else None,
        "published_arm_step_max": float(np.max(published_arm_steps)) if published_arm_steps else None,
    }
    if len(policy_raw_actions_array):
        result["policy_raw_action_min"] = np.round(policy_raw_actions_array.min(axis=0), 4).tolist()
        result["policy_raw_action_max"] = np.round(policy_raw_actions_array.max(axis=0), 4).tolist()
    if len(tcp_target_distances) and np.any(np.isfinite(tcp_target_distances)):
        result["tcp_target_distance_min_m"] = float(np.nanmin(tcp_target_distances))
        result["tcp_target_distance_max_m"] = float(np.nanmax(tcp_target_distances))
        result["tcp_target_distance_final_m"] = float(tcp_target_distances[-1])
    if dataset_stats is not None and len(states):
        result["state_z_max"] = _max_abs_with_index(trace["state_z"])
        result["state_z_first_abs_gt3"] = _first_z_exceedance(trace["state_z"], 3.0)
        result["policy_action_z_max"] = _max_abs_with_index(trace["policy_raw_action_z"])
        result["policy_action_z_first_abs_gt3"] = _first_z_exceedance(trace["policy_raw_action_z"], 3.0)
        result["target_state_error_max"] = _max_abs_with_index(trace["target_state_error"][:, :6])
        result["policy_target_state_error_max"] = _max_abs_with_index(trace["policy_target_state_error"][:, :6])
    if len(states) > 1:
        result.update(
            velocity_diagnostics(
                states,
                actions,
                raw_actions_array,
                stamps,
                client.grasp_events,
                args.max_joint_velocity,
                args.max_action_step,
            )
        )
        result.update(
            event_diagnostics(
                states,
                actions,
                raw_actions_array,
                stamps,
                state_jump_threshold=args.state_jump_event_threshold,
                smooth_command_threshold=args.smooth_command_threshold,
                stuck_duration_sec=args.stuck_duration_sec,
                stuck_state_motion_threshold=args.stuck_state_motion_threshold,
                stuck_tracking_error_threshold=args.stuck_tracking_error_threshold,
                stuck_command_motion_threshold=args.stuck_command_motion_threshold,
                stuck_error_growth_threshold=args.stuck_error_growth_threshold,
            )
        )
        stamp_array = np.asarray(stamps)
        loop_dt = np.diff(stamp_array)
        loop_dt[loop_dt <= 0] = 1e-3
        result["effective_hz"] = round(float(1.0 / loop_dt.mean()), 2)
    result["_trace"] = trace
    finite_object_z = [z for z in object_z if np.isfinite(z)]
    if finite_object_z and start_target is not None:
        lift = float(max(finite_object_z) - start_target[2])
        result["lift_m"] = round(lift, 4)
        result["success"] = bool(lift >= args.min_lift_m)
    else:
        result["success"] = False
        result["lift_m"] = None
    return result


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    policy, preprocessor, postprocessor = load_policy(args.checkpoint, device, args.n_action_steps)
    print(f"policy loaded on {device}: chunk_size={policy.config.chunk_size}, "
          f"n_action_steps={policy.config.n_action_steps}")
    if args.action_clamp != "none" and not args.dataset_stats:
        raise ValueError("--action-clamp requires --dataset-stats")
    dataset_stats = load_dataset_stats(args.dataset_stats, args.state_std_floor)
    if dataset_stats is not None:
        print(
            "dataset stats loaded: "
            f"{args.dataset_stats}, state_std_floor={args.state_std_floor}, "
            f"action_clamp={args.action_clamp}"
        )

    if args.poses_file:
        payload = json.loads(Path(args.poses_file).read_text())
        entries = [
            e for e in payload.get("entries", [])
            if e.get("solved") and e.get("object_name", "red_cube") == args.object
        ]
        object_positions = [
            e.get("object_position", e["cube_position"])
            for e in entries[args.start_index : args.start_index + args.episodes]
        ]
        if len(object_positions) < args.episodes:
            print(
                "warning: requested "
                f"{args.episodes} episodes from {args.poses_file}, but only "
                f"{len(object_positions)} solved entries are available after "
                f"start-index {args.start_index}."
            )
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ammr_mycobot_interface import demo_pose_cube_position

        object_positions = [list(demo_pose_cube_position("p1"))] * args.episodes

    client = RolloutClient(drain=not args.queued_observations)
    results = []
    try:
        for index, object_position in enumerate(object_positions, start=1):
            print(
                f"=== rollout {index}/{len(object_positions)} {args.object} "
                f"at {np.round(object_position, 4).tolist()} ==="
            )
            result = run_episode(
                client,
                policy,
                preprocessor,
                postprocessor,
                args,
                object_position,
                args.object,
                device,
                dataset_stats=dataset_stats,
            )
            trace = result.pop("_trace", None)
            if trace is not None and args.trace_dir:
                trace_dir = Path(args.trace_dir)
                trace_dir.mkdir(parents=True, exist_ok=True)
                np.savez(trace_dir / f"rollout_{index:03d}.npz", **trace)
            results.append(result)
            print(f"  {json.dumps(result)}")
    finally:
        client.shutdown()

    successes = [r for r in results if r.get("success")]
    spikes = [r for r in results if r.get("velocity_ok") is False]
    print()
    print(f"success {len(successes)}/{len(results)}, velocity violations {len(spikes)}")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"results": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
