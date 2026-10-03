#!/usr/bin/env python3
"""Scripted pick-only trajectory publisher for AMMR SmolVLA data collection."""

import argparse
import json
import sys
import time

from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from ammr_mycobot_interface import (
    CONTROL_JOINTS,
    DEFAULT_DEMO_POSE,
    OBJECT_NAMES,
    OBJECT_STATE_NAMES,
    GRIPPER_GRASP_ATTACH_RAD,
    GRIPPER_GRASP_CLOSE_RAD,
    GRIPPER_GRASP_ENGAGE_RAD,
    GRIPPER_OPEN_RAD,
    ROS2_JOINT_TARGET_TOPIC,
    ROS2_OBJECT_STATES_TOPIC,
    ROS2_ROBOT_STATE_TOPIC,
    SURVEY_POSE,
    coerce_control_targets,
    demo_pose_cube_position,
    object_state_slice,
)


# Initial red cube grasp attempt measured from Isaac Sim.
RED_CUBE_PRE_GRASP = [
    -0.0005506679881364107,
    -0.8866308331489563,
    -0.014749297872185707,
    -0.5623526573181152,
    0.000752868945710361,
    0.0005103197763673961,
    GRIPPER_OPEN_RAD,
]
RED_CUBE_LIFT = [
    0.0007871668785810471,
    -0.7505346536636353,
    -0.015758760273456573,
    -0.5643607378005981,
    -0.013691341504454613,
    -0.011589787900447845,
    -0.25,
]
# Measured in Isaac Sim against the p1 cube placement in DEMO_POSE_CUBE_POSITIONS.
#
# This replaces two earlier profiles. The original "p1" put the gripper at
# z=0.2047 while a cube on the table sits at z=0.115, so the vertical gap alone
# was 0.0897 m and grasp assist could never attach within its 0.08 m radius -- no
# cube placement could fix that. Its "p1_deep" replacement reached the cube but
# was not holdable: with the arm and gripper commands both constant, joint5 and
# joint6 sat still for about half a second after the gripper finished closing and
# then jumped roughly 0.29 rad to a different equilibrium, so the lift gate below
# always refused to lift. These values were verified to hold with zero joint
# movement for 45 s at full close, and to complete a pick that lifts the cube.
RED_CUBE_P1_PRE_GRASP = [
    0.05926,
    -0.88532,
    -0.01444,
    -0.5618,
    0.00116,
    0.10097,
    GRIPPER_OPEN_RAD,
]
RED_CUBE_P1_LIFT = [
    0.05926,
    -0.74457,
    -0.01444,
    -0.5618,
    0.00116,
    0.10097,
    -0.25,
]
RED_CUBE_DEMO_POSES = {
    "p0": (RED_CUBE_PRE_GRASP, RED_CUBE_LIFT),
    "p1": (RED_CUBE_P1_PRE_GRASP, RED_CUBE_P1_LIFT),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Publish a scripted AMMR pick-only trajectory.")
    parser.add_argument("--object", choices=OBJECT_NAMES, default="red_cube", help="Scripted target object.")
    parser.add_argument(
        "--demo-pose",
        choices=sorted(RED_CUBE_DEMO_POSES),
        default=DEFAULT_DEMO_POSE,
        help=(
            "Scripted red cube pose profile. Each profile is paired with the cube "
            "placement it was measured against; launch the Isaac scene with the same "
            "--demo-pose so the two agree."
        ),
    )
    parser.add_argument(
        "--poses-file",
        default=None,
        help=(
            "JSON from plan_grasp_poses.py. Uses the IK-solved joints for the entry "
            "matching the scene cube instead of the hardcoded profile, which is what "
            "removes the per-placement slider session."
        ),
    )
    parser.add_argument(
        "--poses-match-tolerance",
        type=float,
        default=0.005,
        help="How close a pose-cache entry must be to the scene cube to be used.",
    )
    parser.add_argument(
        "--poses-entry-index",
        type=int,
        default=None,
        help=(
            "Use this zero-based solved entry from --poses-file instead of matching the "
            "current object_state. Intended only for approach-path diagnostics where the "
            "object has been moved away."
        ),
    )
    parser.add_argument(
        "--expected-object-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help=(
            "Wait until /ammr/object_states reports the target object at this world "
            "position before selecting an IK pose from --poses-file. This prevents "
            "stale object_state samples from selecting the previous trial's pose."
        ),
    )
    parser.add_argument(
        "--object-position-guard-tolerance",
        type=float,
        default=0.002,
        help="Allowed object-state error in metres for --expected-object-position.",
    )
    parser.add_argument(
        "--object-position-guard-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for /ammr/object_states to match --expected-object-position.",
    )
    parser.add_argument(
        "--object-position-guard-samples",
        type=int,
        default=2,
        help=(
            "Consecutive fresh /ammr/object_states messages required before proceeding. "
            "Freshness is tracked by callback sequence because Float64MultiArray has no header stamp."
        ),
    )
    parser.add_argument(
        "--cube-tolerance",
        type=float,
        default=0.02,
        help="Warn if the scene cube is further than this from the profile's placement.",
    )
    parser.add_argument(
        "--no-check-cube",
        dest="check_cube",
        action="store_false",
        help="Skip the scene cube placement check.",
    )
    parser.set_defaults(check_cube=True)
    parser.add_argument("--close", type=float, default=GRIPPER_GRASP_CLOSE_RAD, help="Gripper close command in radians.")
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=20.0,
        help=(
            "Rate at which interpolated targets are published. Keep it above the episode "
            "recording rate so every recorded step sees a fresh action."
        ),
    )
    parser.add_argument(
        "--arm-speed",
        type=float,
        default=0.25,
        help=(
            "Commanded arm joint speed in rad/s, used to derive segment durations. The "
            "drive lags a ramp by damping/stiffness * speed = 0.2 * speed, so keeping "
            "this under arm_tolerance/0.2 lets the state track the command while it "
            "moves instead of settling for seconds afterwards."
        ),
    )
    parser.add_argument(
        "--gripper-speed",
        type=float,
        default=0.12,
        help=(
            "Commanded gripper ramp rate in rad/s. This is the ramp that used to live in "
            "the Isaac bridge; issuing it here keeps the published action equal to the "
            "applied command. The default matches how fast the gripper drive actually "
            "closes (~2.3 s over the 0.26 rad range), so the command does not outrun the "
            "hardware and sit constant while the fingers catch up."
        ),
    )
    parser.add_argument(
        "--no-survey",
        dest="survey",
        action="store_false",
        help="Skip the survey stop and drive straight at the pre-grasp pose.",
    )
    parser.set_defaults(survey=True)
    parser.add_argument(
        "--survey-hold",
        type=float,
        default=0.6,
        help=(
            "Seconds to sit at the survey pose. Long enough to record a few frames of "
            "the table with the object in view; this pause is functional, not slack."
        ),
    )
    parser.add_argument("--pre-grasp-hold", type=float, default=0.2, help="Seconds to hold the open pre-grasp pose.")
    parser.add_argument(
        "--close-hold",
        type=float,
        default=0.5,
        help="Seconds to keep publishing the close target after the gripper ramp finishes.",
    )
    parser.add_argument("--lift-hold", type=float, default=0.3, help="Seconds to hold the lifted target.")
    parser.add_argument("--final-hold", type=float, default=0.2, help="Seconds to hold the final target.")
    parser.add_argument(
        "--lift-arm-speed",
        type=float,
        default=None,
        help=(
            "Arm joint speed used only for lift/lower segments. Defaults to --arm-speed. "
            "Use a smaller value for dynamic-contact validation so lift does not jerk the object."
        ),
    )
    parser.add_argument(
        "--place-after-lift",
        action="store_true",
        help=(
            "After lift and lift-hold, lower back to the pre-grasp pose and open the gripper. "
            "This is for dynamic-contact lift validation; the default pick-only behavior is unchanged."
        ),
    )
    parser.add_argument(
        "--place-hold",
        type=float,
        default=0.5,
        help="Seconds to hold the lowered closed pose before opening when --place-after-lift is set.",
    )
    parser.add_argument("--wait-timeout", type=float, default=2.0, help="Seconds to wait for a command subscriber.")
    parser.add_argument("--state-timeout", type=float, default=6.0, help="Seconds to wait for robot_state feedback.")
    parser.add_argument("--arm-tolerance", type=float, default=0.06, help="Arm target tolerance in radians.")
    parser.add_argument("--gripper-tolerance", type=float, default=0.12, help="Gripper target tolerance in radians.")
    parser.add_argument(
        "--lift-gripper-tolerance",
        type=float,
        default=abs(GRIPPER_GRASP_CLOSE_RAD - GRIPPER_GRASP_ENGAGE_RAD),
        help="Allowed gripper error before lift. Default waits for the grasp-assist engage threshold.",
    )
    parser.add_argument(
        "--lift-gripper-wait-timeout",
        type=float,
        default=60.0,
        help="Maximum seconds to wait for the gripper to close enough before lift.",
    )
    parser.add_argument(
        "--post-gripper-ready-hold",
        type=float,
        default=0.3,
        help="Seconds to keep the close target after the gripper reaches the lift threshold.",
    )
    parser.add_argument(
        "--post-gripper-ready-arm-settle-timeout",
        type=float,
        default=12.0,
        help="Seconds to wait for the arm to settle back to the close target after grasp assist engages.",
    )
    parser.add_argument(
        "--no-require-arm-settle-before-lift",
        dest="require_arm_settle_before_lift",
        action="store_false",
        help="Lift even if the arm does not settle after grasp assist engages.",
    )
    parser.add_argument(
        "--no-require-approach-settle",
        dest="require_approach_settle",
        action="store_false",
        help="Continue to close even if the open pre-grasp approach did not settle.",
    )
    parser.add_argument(
        "--no-require-close-before-lift",
        dest="require_close_before_lift",
        action="store_false",
        help="Lift even if the gripper did not close to the requested threshold.",
    )
    parser.add_argument("--no-wait-state", action="store_true", help="Do not wait for /ammr/robot_state feedback.")
    parser.add_argument("--return-open", action="store_true", help="Open the gripper after hold.")
    parser.add_argument(
        "--stop-after-close",
        action="store_true",
        help=(
            "Stop after approach + close_on_cube and do not execute lift. Useful for "
            "dynamic-contact diagnostics of finger/cube contact before tuning lift."
        ),
    )
    parser.add_argument(
        "--stop-after-approach",
        action="store_true",
        help=(
            "Stop after the open pre-grasp approach and do not close the gripper. "
            "Useful for diagnosing approach-path contact before grasp tuning."
        ),
    )
    parser.add_argument(
        "--settle-approach-waypoints",
        action="store_true",
        help=(
            "Require state settle at every IK approach waypoint. The default treats "
            "intermediate descent waypoints as one continuous path and only settles at "
            "the first pre-approach and final approach targets."
        ),
    )
    parser.add_argument(
        "--descent-monitor-arm-error",
        type=float,
        default=0.25,
        help=(
            "Abort intermediate continuous approach descent when the arm deviates this "
            "far from the currently commanded interpolated target. This is intentionally "
            "looser than --arm-tolerance because it monitors moving targets, not settled "
            "targets. The first pre-approach and final approach still use the ordinary "
            "settled-state check instead."
        ),
    )
    parser.add_argument(
        "--descent-monitor-object-move",
        type=float,
        default=0.002,
        help=(
            "Abort approach descent when the target object moves this far from its "
            "position at descent start. This is the online proxy for open-gripper "
            "contact; detailed contact forces still come from Isaac diagnostics CSV."
        ),
    )
    parser.add_argument(
        "--gripper-sweep-values",
        type=float,
        nargs="+",
        default=None,
        metavar="RAD",
        help=(
            "Diagnostic mode: after reaching the open pre-grasp pose, keep the arm fixed "
            "and command these gripper targets in order, then exit before lift. Use this "
            "to find the close target that produces bilateral contact force."
        ),
    )
    parser.add_argument(
        "--gripper-sweep-hold",
        type=float,
        default=0.8,
        help="Seconds to hold each target in --gripper-sweep-values after its ramp finishes.",
    )
    parser.set_defaults(require_close_before_lift=True)
    parser.set_defaults(require_arm_settle_before_lift=True)
    parser.set_defaults(require_approach_settle=True)
    return parser.parse_args()


class ScriptedPickClient(Node):
    def __init__(self):
        super().__init__("ammr_scripted_pick_demo")
        self._last_state = None
        self._last_object_states = None
        self._object_state_seq = 0
        self._segment_abort_reason = None
        self._target_pub = self.create_publisher(Float64MultiArray, ROS2_JOINT_TARGET_TOPIC, 10)
        self._state_sub = self.create_subscription(Float64MultiArray, ROS2_ROBOT_STATE_TOPIC, self._on_state, 10)
        self._object_sub = self.create_subscription(
            Float64MultiArray, ROS2_OBJECT_STATES_TOPIC, self._on_object_state, 10
        )

    def _on_state(self, msg):
        try:
            self._last_state = np.asarray(coerce_control_targets(msg.data), dtype=np.float64)
        except ValueError:
            return

    def _on_object_state(self, msg):
        if len(msg.data) < len(OBJECT_STATE_NAMES):
            return
        self._last_object_states = np.asarray(msg.data[: len(OBJECT_STATE_NAMES)], dtype=np.float64)
        self._object_state_seq += 1

    def object_position(self, object_name, timeout=3.0):
        """Latest object position from /ammr/object_states, or None."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline and self._last_object_states is None:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self._last_object_states is None:
            return None
        return self._last_object_states[object_state_slice(object_name)].tolist()

    def wait_for_object_position(self, object_name, expected, tolerance, timeout, samples=2):
        """Wait for fresh object_state feedback near an expected world position."""
        expected_array = np.asarray(expected, dtype=np.float64)
        deadline = time.monotonic() + timeout
        required_samples = max(1, int(samples))
        matched_samples = 0
        best = None
        last_counted_seq = None

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._last_object_states is None:
                continue
            sample_seq = self._object_state_seq
            position = self._last_object_states[object_state_slice(object_name)]
            error = float(np.linalg.norm(position - expected_array))
            if best is None or error < best[0]:
                best = (error, position.copy())
            if error <= tolerance:
                if sample_seq != last_counted_seq:
                    matched_samples += 1
                    last_counted_seq = sample_seq
                    if matched_samples >= required_samples:
                        self.get_logger().info(
                            f"{object_name} object_state guard matched: "
                            f"position={np.round(position, 4).tolist()}, "
                            f"expected={np.round(expected_array, 4).tolist()}, error={error:.4f} m, "
                            f"fresh_samples={matched_samples}"
                        )
                        return position.tolist()
            else:
                matched_samples = 0
                last_counted_seq = None

        if best is None:
            raise TimeoutError(
                f"no {ROS2_OBJECT_STATES_TOPIC} feedback while waiting for {object_name} "
                f"at {np.round(expected_array, 4).tolist()}"
            )
        best_error, best_position = best
        raise TimeoutError(
            f"{object_name} did not reach expected object_state within {timeout:.1f}s: "
            f"expected={np.round(expected_array, 4).tolist()}, "
            f"best={np.round(best_position, 4).tolist()}, error={best_error:.4f} m, "
            f"tolerance={tolerance:.4f} m"
        )

    def check_object_placement(self, object_name, demo_pose, tolerance, timeout=3.0):
        """Warn when the scene object is not where this profile expects it.

        Each profile only reaches its own cube placement, and a mismatch fails
        quietly: the arm runs the whole trajectory, grasp assist never attaches
        because the cube is outside its radius, and the lift is skipped.
        """
        if object_name != "red_cube":
            self.get_logger().warn(
                f"no built-in placement check exists for {object_name}; use --poses-file"
            )
            return None
        expected = np.asarray(demo_pose_cube_position(demo_pose), dtype=np.float64)
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline and self._last_object_states is None:
            rclpy.spin_once(self, timeout_sec=0.05)

        position = self.object_position(object_name, timeout=0.0)
        if position is None:
            self.get_logger().warn(
                f"no {ROS2_OBJECT_STATES_TOPIC} feedback; cannot check the {object_name} placement"
            )
            return None

        position = np.asarray(position, dtype=np.float64)
        error = float(np.linalg.norm(position - expected))
        if error <= tolerance:
            self.get_logger().info(
                f"{object_name} placement matches {demo_pose}: "
                f"{np.round(position, 4).tolist()}, error={error:.4f} m"
            )
            return True

        self.get_logger().warn(
            f"{object_name} placement does not match {demo_pose}: "
            f"scene={np.round(position, 4).tolist()}, "
            f"expected={expected.tolist()}, error={error:.4f} m. "
            f"Relaunch the Isaac scene with --demo-pose {demo_pose}, or send "
            f"/ammr/reset_task_scene if it is already running with that profile."
        )
        return False

    def wait_for_subscriber(self, timeout):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self._target_pub.get_subscription_count() > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return self._target_pub.get_subscription_count() > 0

    def hold_target(self, target, hold_time, publish_hz):
        msg = Float64MultiArray()
        msg.data = [float(value) for value in target]
        period = 1.0 / max(publish_hz, 1e-6)
        deadline = time.monotonic() + max(0.0, hold_time)
        while rclpy.ok() and time.monotonic() < deadline:
            self._target_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

    def run_segment(self, start, end, duration, publish_hz, monitor=None):
        """Publish a continuously interpolated target from start to end.

        This is what makes the recorded action stream usable for training. Publishing a
        waypoint and holding it leaves the motion only in robot_state, so action[t]
        repeats the same value for most of the episode and carries no signal about what
        the arm is about to do. Interpolating means every recorded step sees an action
        that reflects the next increment of motion.
        """
        start_array = np.asarray(start, dtype=np.float64)
        end_array = np.asarray(end, dtype=np.float64)
        period = 1.0 / max(publish_hz, 1e-6)
        duration = max(0.0, float(duration))
        msg = Float64MultiArray()
        self._segment_abort_reason = None

        begin = time.monotonic()
        while rclpy.ok():
            elapsed = time.monotonic() - begin
            ratio = 1.0 if duration <= 0.0 else min(1.0, elapsed / duration)
            pose = start_array + (end_array - start_array) * ratio
            msg.data = [float(value) for value in pose]
            self._target_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            if monitor is not None:
                arm_limit = monitor.get("max_arm_error")
                if arm_limit is not None and self._last_state is not None:
                    arm_error = float(np.max(np.abs(self._last_state[:6] - pose[:6])))
                    if arm_error > float(arm_limit):
                        self._segment_abort_reason = (
                            f"{monitor['label']} arm tracking error {arm_error:.4f} rad "
                            f"> {float(arm_limit):.4f} rad"
                        )
                        return [float(value) for value in pose]
                object_name = monitor.get("object_name")
                object_start = monitor.get("object_start")
                object_limit = monitor.get("max_object_move")
                if (
                    object_name is not None
                    and object_start is not None
                    and object_limit is not None
                    and self._last_object_states is not None
                ):
                    current_object = self._last_object_states[object_state_slice(object_name)]
                    object_move = float(np.linalg.norm(current_object - object_start))
                    if object_move > float(object_limit):
                        self._segment_abort_reason = (
                            f"{monitor['label']} moved {object_name} by "
                            f"{object_move * 1000:.2f} mm > {float(object_limit) * 1000:.2f} mm"
                        )
                        return [float(value) for value in pose]
            if ratio >= 1.0:
                break
            time.sleep(period)
        return [float(value) for value in end_array]

    def current_state(self, timeout=3.0):
        """Latest /ammr/robot_state, or None. Used as the start of the approach."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline and self._last_state is None:
            rclpy.spin_once(self, timeout_sec=0.05)
        return None if self._last_state is None else self._last_state.tolist()

    def wait_for_state(self, target, timeout, arm_tolerance, gripper_tolerance):
        target_array = np.asarray(target, dtype=np.float64)
        deadline = time.monotonic() + timeout
        best_error = None
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._last_state is None:
                continue
            errors = np.abs(self._last_state - target_array)
            arm_error = float(np.max(errors[:6]))
            gripper_error = float(errors[6])
            total_error = max(arm_error, gripper_error)
            if best_error is None or total_error < best_error[0]:
                best_error = (total_error, arm_error, gripper_error)
            if arm_error <= arm_tolerance and gripper_error <= gripper_tolerance:
                return True, (total_error, arm_error, gripper_error)
        return False, best_error

    def wait_for_gripper_close_before_lift(self, close_target, tolerance, timeout, hold_target, publish_hz):
        threshold = float(close_target) + abs(float(tolerance))
        deadline = time.monotonic() + timeout
        period = 1.0 / max(publish_hz, 1e-6)
        next_publish = 0.0
        best_state = None

        msg = Float64MultiArray()
        msg.data = [float(value) for value in hold_target]

        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_publish:
                self._target_pub.publish(msg)
                next_publish = now + period

            rclpy.spin_once(self, timeout_sec=0.05)
            if self._last_state is None:
                continue

            gripper_state = float(self._last_state[6])
            if best_state is None or abs(gripper_state - close_target) < abs(best_state - close_target):
                best_state = gripper_state
            if gripper_state <= threshold:
                return True, gripper_state, threshold, best_state

        return False, None, threshold, best_state


def with_gripper(pose, gripper):
    updated = list(pose)
    updated[6] = float(gripper)
    return updated


def make_pose_plan(pre_grasp, lift, approach_waypoints=None):
    return {
        "pre_grasp": list(pre_grasp),
        "lift": list(lift),
        "approach_waypoints": [list(pose) for pose in (approach_waypoints or [])],
    }


def load_planned_poses(path, object_position, object_name, tolerance, close_rad, entry_index=None):
    """Pick the pose-cache entry for this object. Returns a pose plan dict.

    Matching on the object position rather than an index means a stale cache cannot be
    silently applied to a scene it was not solved for -- the failure mode that made the
    removed p1 profile look like a working pick that never grasped anything.
    """
    payload = json.loads(Path(path).read_text())
    entries = [
        entry for entry in payload.get("entries", [])
        if entry.get("solved") and entry.get("object_name", object_name) == object_name
    ]
    if not entries:
        raise ValueError(f"{path} has no solved entries for {object_name}")

    if entry_index is not None:
        if entry_index < 0 or entry_index >= len(entries):
            raise ValueError(
                f"--poses-entry-index {entry_index} is outside solved entry range 0..{len(entries) - 1}"
            )
        entry = entries[entry_index]
    elif object_position is None:
        if len(entries) > 1:
            raise ValueError(
                f"{path} holds {len(entries)} entries but the scene {object_name} position is "
                "unknown, so the right one cannot be chosen"
            )
        entry = entries[0]
    else:
        position = np.asarray(object_position, dtype=np.float64)
        distances = [
            float(np.linalg.norm(np.asarray(e.get("object_position", e["cube_position"])) - position))
            for e in entries
        ]
        best = int(np.argmin(distances))
        if distances[best] > tolerance:
            raise ValueError(
                f"{path} has no entry within {tolerance} m of the scene {object_name} "
                f"{np.round(position, 4).tolist()}; nearest is {distances[best]:.4f} m away. "
                "Re-run plan_grasp_poses.py for this placement."
            )
        entry = entries[best]

    pre_grasp = [*entry["pre_grasp_joints"], GRIPPER_OPEN_RAD]
    lift = [*entry["lift_joints"], close_rad]
    approach_waypoints = [
        [*joints, GRIPPER_OPEN_RAD]
        for joints in entry.get("approach_waypoint_joints", [])
    ]
    if approach_waypoints:
        final_error = float(np.max(np.abs(np.asarray(approach_waypoints[-1][:6]) - np.asarray(pre_grasp[:6]))))
        if final_error > 1e-6:
            approach_waypoints.append(pre_grasp)
    return make_pose_plan(pre_grasp, lift, approach_waypoints)


def segment_duration(start, end, arm_speed, gripper_speed, minimum=0.3):
    """Seconds to traverse start -> end at the configured joint speeds.

    Deriving the duration from the actual joint deltas keeps the commanded velocity
    roughly constant across segments, so the drives track it instead of saturating on
    a large step.
    """
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    arm_delta = float(np.max(np.abs(end_array[:6] - start_array[:6])))
    gripper_delta = float(abs(end_array[6] - start_array[6]))
    arm_time = arm_delta / max(arm_speed, 1e-6)
    gripper_time = gripper_delta / max(gripper_speed, 1e-6)
    return max(minimum, arm_time, gripper_time)


def build_segments(args, start_pose, poses=None):
    """Segments of (label, end_pose, duration, hold_after) traversed by interpolation.

    Three things matter about this ordering.

    The survey segment comes first and is identical for every placement, so the robot
    looks at the table before committing to a direction. Without it the approach decides
    joint1 from frames where the cube is not yet visible, which makes the task
    unlearnable from the wrist camera no matter how much data is collected.

    The approach then runs survey -> pre-grasp with the cube in view the whole way, so
    the frames that carry the steering decision also carry the information it needs.

    Both start from the robot's measured state rather than an assumed one, which is what
    puts the motion into the action stream instead of leaving it only in robot_state.
    """
    if poses is None:
        pre_grasp_pose, lift_pose = RED_CUBE_DEMO_POSES[args.demo_pose]
        pose_plan = make_pose_plan(pre_grasp_pose, lift_pose)
    elif isinstance(poses, dict):
        pose_plan = poses
        pre_grasp_pose = pose_plan["pre_grasp"]
        lift_pose = pose_plan["lift"]
    else:
        # Backward compatibility for callers/tests that still pass (pre_grasp, lift).
        pre_grasp_pose, lift_pose = poses
        pose_plan = make_pose_plan(pre_grasp_pose, lift_pose)

    pre_grasp_open = with_gripper(pre_grasp_pose, GRIPPER_OPEN_RAD)
    pre_grasp_close = with_gripper(pre_grasp_pose, args.close)
    lift_close = with_gripper(lift_pose, args.close)
    approach_waypoints = [with_gripper(pose, GRIPPER_OPEN_RAD) for pose in pose_plan.get("approach_waypoints", [])]
    # SURVEY_POSE holds only the six arm joints; the demo works in 7-D control vectors.
    survey_open = [*SURVEY_POSE, GRIPPER_OPEN_RAD]

    def duration(start, end, arm_speed=None):
        return segment_duration(start, end, args.arm_speed if arm_speed is None else arm_speed, args.gripper_speed)

    lift_arm_speed = args.arm_speed if args.lift_arm_speed is None else args.lift_arm_speed

    segments = []
    if args.survey:
        segments.append(
            ("survey", survey_open, duration(start_pose, survey_open), args.survey_hold)
        )
        approach_from = survey_open
    else:
        approach_from = start_pose
    if approach_waypoints:
        for index, waypoint in enumerate(approach_waypoints):
            is_final = index == len(approach_waypoints) - 1
            if args.settle_approach_waypoints:
                label = "approach" if is_final else f"approach_waypoint_{index + 1}"
            elif is_final:
                label = "approach"
            elif index == 0:
                label = "approach_pre"
            else:
                label = f"approach_descent_{index}"
            segments.append(
                (label, waypoint, duration(approach_from, waypoint), args.pre_grasp_hold if is_final else 0.0)
            )
            approach_from = waypoint
    else:
        segments.append(
            ("approach", pre_grasp_open, duration(approach_from, pre_grasp_open), args.pre_grasp_hold)
        )

    segments += [
        ("close_on_cube", pre_grasp_close, duration(pre_grasp_open, pre_grasp_close), args.close_hold),
        ("lift", lift_close, duration(pre_grasp_close, lift_close, lift_arm_speed), args.lift_hold),
    ]
    if args.place_after_lift:
        segments.append(
            ("place_lower", pre_grasp_close, duration(lift_close, pre_grasp_close, lift_arm_speed), args.place_hold)
        )
        segments.append(
            ("release_open", pre_grasp_open, duration(pre_grasp_close, pre_grasp_open), args.final_hold)
        )
    elif args.return_open:
        lift_open = with_gripper(lift_close, GRIPPER_OPEN_RAD)
        segments.append(
            ("release_open", lift_open, duration(lift_close, lift_open), args.final_hold)
        )
    else:
        segments.append(("hold", lift_close, 0.0, args.final_hold))
    return segments


def should_wait_for_segment(label):
    """Intermediate Cartesian descent waypoints are monitored continuously, not settled."""
    return not label.startswith("approach_descent_")


def run_gripper_sweep(node, args, current_pose):
    arm_pose = list(current_pose[:6])
    current = list(current_pose)
    node.get_logger().info(
        "gripper sweep: arm fixed at "
        f"{np.round(arm_pose, 4).tolist()}, values={np.round(args.gripper_sweep_values, 4).tolist()}"
    )
    for value in args.gripper_sweep_values:
        target = [*arm_pose, float(value)]
        duration = segment_duration(current, target, args.arm_speed, args.gripper_speed)
        node.get_logger().info(f"gripper sweep target {value:.4f} rad: ramp {duration:.2f}s")
        current = node.run_segment(current, target, duration, args.publish_hz)
        if not args.no_wait_state:
            reached, error = node.wait_for_state(
                target,
                args.state_timeout,
                args.arm_tolerance,
                args.gripper_tolerance,
            )
            if reached:
                _, arm_error, gripper_error = error
                node.get_logger().info(
                    f"gripper sweep reached {value:.4f}: "
                    f"arm_error={arm_error:.4f}, gripper_error={gripper_error:.4f}"
                )
            elif error is None:
                node.get_logger().warn(f"no feedback while sweeping gripper to {value:.4f}")
            else:
                _, arm_error, gripper_error = error
                node.get_logger().warn(
                    f"gripper sweep not within tolerance for {value:.4f}: "
                    f"best_arm_error={arm_error:.4f}, best_gripper_error={gripper_error:.4f}"
                )
        if args.gripper_sweep_hold > 0.0:
            node.get_logger().info(f"gripper sweep hold {value:.4f} for {args.gripper_sweep_hold:.1f}s")
            node.hold_target(target, args.gripper_sweep_hold, args.publish_hz)
    node.get_logger().info("gripper sweep complete; lift skipped by request")
    return current


def main():
    args = parse_args()
    if args.object != "red_cube" and not args.poses_file:
        raise ValueError(f"--poses-file is required for {args.object}")

    rclpy.init()
    node = ScriptedPickClient()
    try:
        if not node.wait_for_subscriber(args.wait_timeout):
            node.get_logger().warn(f"no subscriber detected on {ROS2_JOINT_TARGET_TOPIC}")
        if args.check_cube:
            node.check_object_placement(args.object, args.demo_pose, args.cube_tolerance)

        guarded_object_position = None
        if args.expected_object_position is not None:
            guarded_object_position = node.wait_for_object_position(
                args.object,
                args.expected_object_position,
                args.object_position_guard_tolerance,
                args.object_position_guard_timeout,
                args.object_position_guard_samples,
            )

        start_pose = node.current_state()
        if start_pose is None:
            node.get_logger().warn(
                f"no {ROS2_ROBOT_STATE_TOPIC} feedback; starting the approach from the "
                "pre-grasp pose, so the approach motion will not appear in the action stream"
            )
            if args.object != "red_cube" and not args.poses_file:
                raise ValueError(f"--poses-file is required for {args.object}")
            start_pose = with_gripper(RED_CUBE_DEMO_POSES[args.demo_pose][0], GRIPPER_OPEN_RAD)
        else:
            node.get_logger().info(f"approach starts from measured state: {np.round(start_pose, 4).tolist()}")

        poses = None
        if args.poses_file:
            poses = load_planned_poses(
                args.poses_file,
                guarded_object_position if guarded_object_position is not None else node.object_position(args.object),
                args.object,
                args.poses_match_tolerance,
                args.close,
                args.poses_entry_index,
            )
            node.get_logger().info(
                f"using IK poses from {args.poses_file}: "
                f"pre_grasp={np.round(poses['pre_grasp'], 5).tolist()}, "
                f"approach_waypoints={len(poses.get('approach_waypoints', []))}"
            )

        segments = build_segments(args, start_pose, poses=poses)
        current = start_pose
        approach_object_start = None
        if guarded_object_position is not None:
            approach_object_start = np.asarray(guarded_object_position, dtype=np.float64)
        else:
            latest_object_position = node.object_position(args.object, timeout=0.0)
            if latest_object_position is not None:
                approach_object_start = np.asarray(latest_object_position, dtype=np.float64)

        for label, target, duration, hold_time in segments:
            node.get_logger().info(f"segment {label}: {duration:.2f}s -> {np.round(target, 4).tolist()}")
            monitor = None
            if label.startswith("approach"):
                monitor = {
                    "label": label,
                    "max_arm_error": (
                        args.descent_monitor_arm_error
                        if label.startswith("approach_descent_")
                        else None
                    ),
                    "object_name": args.object,
                    "object_start": approach_object_start,
                    "max_object_move": args.descent_monitor_object_move,
                }
            current = node.run_segment(current, target, duration, args.publish_hz, monitor=monitor)
            if node._segment_abort_reason is not None:
                node.get_logger().warn(f"segment aborted: {node._segment_abort_reason}")
                if label.startswith("approach"):
                    node.get_logger().warn("close skipped: approach descent monitor tripped")
                    return 1
            wait_for_this_segment = should_wait_for_segment(label)
            if not args.no_wait_state and wait_for_this_segment:
                reached, error = node.wait_for_state(
                    target,
                    args.state_timeout,
                    args.arm_tolerance,
                    args.gripper_tolerance,
                )
                if reached:
                    _, arm_error, gripper_error = error
                    node.get_logger().info(
                        f"state reached {label}: arm_error={arm_error:.4f}, gripper_error={gripper_error:.4f}"
                    )
                elif error is None:
                    node.get_logger().warn(f"no {ROS2_ROBOT_STATE_TOPIC} feedback for {label}")
                    if label.startswith("approach") and args.require_approach_settle:
                        node.get_logger().warn("close skipped: no approach feedback")
                        return 1
                else:
                    _, arm_error, gripper_error = error
                    node.get_logger().warn(
                        f"state not within tolerance for {label}: "
                        f"best_arm_error={arm_error:.4f}, best_gripper_error={gripper_error:.4f}"
                    )
                    if label.startswith("approach") and args.require_approach_settle:
                        node.get_logger().warn("close skipped: approach did not settle")
                        return 1
            elif not args.no_wait_state:
                node.get_logger().info(
                    f"continuous descent segment {label}: settle wait skipped; "
                    "tracking/object-motion monitor stayed within limits"
                )
            if hold_time > 0.0:
                node.get_logger().info(f"hold {label} for {hold_time:.1f}s")
                node.hold_target(target, hold_time, args.publish_hz)
            if label == "approach" and args.stop_after_approach:
                if args.final_hold > 0.0:
                    node.get_logger().info(
                        f"stop-after-approach: hold open target for {args.final_hold:.1f}s"
                    )
                    node.hold_target(target, args.final_hold, args.publish_hz)
                node.get_logger().info("stop-after-approach: close and lift skipped by request")
                return 0
            if label == "approach" and args.gripper_sweep_values:
                run_gripper_sweep(node, args, current)
                return 0
            if (
                label == "close_on_cube"
                and args.require_close_before_lift
                and not args.no_wait_state
            ):
                node.get_logger().info(
                    "waiting for gripper before lift: "
                    f"target={args.close:.4f}, "
                    f"threshold<={args.close + abs(args.lift_gripper_tolerance):.4f}, "
                    f"grasp_assist_command_threshold={GRIPPER_GRASP_ATTACH_RAD:.4f}, "
                    f"grasp_assist_engage_threshold={GRIPPER_GRASP_ENGAGE_RAD:.4f}, "
                    f"timeout={args.lift_gripper_wait_timeout:.1f}s"
                )
                gripper_ready, gripper_state, threshold, best_state = node.wait_for_gripper_close_before_lift(
                    args.close,
                    args.lift_gripper_tolerance,
                    args.lift_gripper_wait_timeout,
                    target,
                    args.publish_hz,
                )
                if gripper_ready:
                    node.get_logger().info(
                        f"gripper ready before lift: state={gripper_state:.4f}, threshold<={threshold:.4f}"
                    )
                    if args.post_gripper_ready_hold > 0.0:
                        node.get_logger().info(
                            f"hold close target for grasp assist: {args.post_gripper_ready_hold:.1f}s"
                        )
                        node.hold_target(target, args.post_gripper_ready_hold, args.publish_hz)
                    if args.post_gripper_ready_arm_settle_timeout > 0.0:
                        node.get_logger().info(
                            "waiting for arm to settle after grasp assist: "
                            f"timeout={args.post_gripper_ready_arm_settle_timeout:.1f}s"
                        )
                        settled, settle_error = node.wait_for_state(
                            target,
                            args.post_gripper_ready_arm_settle_timeout,
                            args.arm_tolerance,
                            args.gripper_tolerance,
                        )
                        if settled:
                            _, arm_error, gripper_error = settle_error
                            node.get_logger().info(
                                "arm settled after grasp assist: "
                                f"arm_error={arm_error:.4f}, gripper_error={gripper_error:.4f}"
                            )
                        elif settle_error is None:
                            node.get_logger().warn("no feedback while waiting for post-grasp arm settle")
                            if args.require_arm_settle_before_lift:
                                node.get_logger().warn("lift skipped: arm did not settle after grasp assist")
                                return 1
                        else:
                            _, arm_error, gripper_error = settle_error
                            node.get_logger().warn(
                                "arm did not settle before lift: "
                                f"best_arm_error={arm_error:.4f}, best_gripper_error={gripper_error:.4f}"
                            )
                            if args.require_arm_settle_before_lift:
                                node.get_logger().warn("lift skipped: arm did not settle after grasp assist")
                                return 1
                else:
                    best_text = "none" if best_state is None else f"{best_state:.4f}"
                    node.get_logger().warn(
                        "lift skipped: gripper did not close enough "
                        f"(target={args.close:.4f}, threshold<={threshold:.4f}, best_state={best_text})"
                    )
                    return 1
            if label == "close_on_cube" and args.stop_after_close:
                if args.final_hold > 0.0:
                    node.get_logger().info(
                        f"stop-after-close: hold close target for {args.final_hold:.1f}s"
                    )
                    node.hold_target(target, args.final_hold, args.publish_hz)
                node.get_logger().info("stop-after-close: lift skipped by request")
                return 0
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
