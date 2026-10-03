#!/usr/bin/env python3
"""Small ROS2 command publisher for the common AMMR myCobot arm interface."""

import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from ammr_mycobot_interface import (
    CONTROL_JOINTS,
    GRIPPER_GRASP_CLOSE_RAD,
    GRIPPER_OPEN_RAD,
    ROS2_JOINT_TARGET_TOPIC,
    ROS2_ROBOT_STATE_TOPIC,
    coerce_control_targets,
)


HOME_OPEN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN_RAD]
HOME_CLOSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_GRASP_CLOSE_RAD]
DEMO_SEQUENCE = [
    ("home_open", HOME_OPEN, 1.0),
    ("reach_left", [0.15, -0.10, 0.20, 0.0, 0.0, 0.0, GRIPPER_OPEN_RAD], 1.0),
    ("reach_right", [-0.15, 0.10, 0.10, 0.0, 0.0, 0.0, GRIPPER_OPEN_RAD], 1.0),
    ("gripper_close", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_GRASP_CLOSE_RAD], 1.2),
    ("home_open", HOME_OPEN, 1.0),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Publish AMMR arm target commands.")
    parser.add_argument(
        "--mode",
        choices=["home", "open", "close", "custom", "demo"],
        default="home",
        help="Command mode. Default sends one home/open target.",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=len(CONTROL_JOINTS),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "GRIPPER"),
        help="Custom target in radians: j1 j2 j3 j4 j5 j6 gripper.",
    )
    parser.add_argument("--publish-count", type=int, default=3, help="Number of repeated publishes per target.")
    parser.add_argument("--publish-hz", type=float, default=10.0, help="Repeated publish rate.")
    parser.add_argument("--wait-timeout", type=float, default=2.0, help="Seconds to wait for a target subscriber.")
    parser.add_argument("--state-timeout", type=float, default=5.0, help="Seconds to wait for robot_state after each target.")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Arm state-vs-target tolerance in radians.")
    parser.add_argument(
        "--gripper-tolerance",
        type=float,
        default=0.30,
        help="Gripper state-vs-target tolerance in radians. Isaac gripper linkage can settle away from the command target.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeat count for demo mode.")
    parser.add_argument("--no-wait-state", action="store_true", help="Do not wait for /ammr/robot_state feedback.")
    return parser.parse_args()


class AmmrArmCommandClient(Node):
    def __init__(self):
        super().__init__("ammr_arm_command_demo")
        self._last_state = None
        self._target_pub = self.create_publisher(Float64MultiArray, ROS2_JOINT_TARGET_TOPIC, 10)
        self._state_sub = self.create_subscription(
            Float64MultiArray,
            ROS2_ROBOT_STATE_TOPIC,
            self._on_robot_state,
            10,
        )

    def _on_robot_state(self, msg):
        try:
            self._last_state = np.asarray(coerce_control_targets(msg.data), dtype=np.float64)
        except ValueError:
            return

    def wait_for_target_subscriber(self, timeout):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self._target_pub.get_subscription_count() > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return self._target_pub.get_subscription_count() > 0

    def publish_target(self, target, publish_count, publish_hz):
        msg = Float64MultiArray()
        msg.data = [float(value) for value in target]
        period = 1.0 / max(publish_hz, 1e-6)
        for _ in range(max(1, publish_count)):
            self._target_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

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


def build_sequence(args):
    if args.mode == "home" or args.mode == "open":
        return [("home_open", HOME_OPEN, 0.0)]
    if args.mode == "close":
        return [("home_close", HOME_CLOSE, 0.0)]
    if args.mode == "custom":
        if args.target is None:
            raise ValueError("--mode custom requires --target with 7 values")
        return [("custom", coerce_control_targets(args.target), 0.0)]
    sequence = []
    for _ in range(max(1, args.repeat)):
        sequence.extend(DEMO_SEQUENCE)
    return sequence


def main():
    args = parse_args()
    try:
        sequence = build_sequence(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    rclpy.init()
    node = AmmrArmCommandClient()
    try:
        if not node.wait_for_target_subscriber(args.wait_timeout):
            node.get_logger().warn(f"no subscriber detected on {ROS2_JOINT_TARGET_TOPIC}")

        for label, target, hold_time in sequence:
            node.get_logger().info(f"publish {label}: {target}")
            node.publish_target(target, args.publish_count, args.publish_hz)
            if not args.no_wait_state:
                reached, error = node.wait_for_state(
                    target,
                    args.state_timeout,
                    args.tolerance,
                    args.gripper_tolerance,
                )
                if reached:
                    _, arm_error, gripper_error = error
                    node.get_logger().info(
                        f"state reached {label}: arm_error={arm_error:.4f}, gripper_error={gripper_error:.4f}"
                    )
                elif error is None:
                    node.get_logger().warn(f"no {ROS2_ROBOT_STATE_TOPIC} feedback for {label}")
                else:
                    _, arm_error, gripper_error = error
                    node.get_logger().warn(
                        f"state not within tolerance for {label}: "
                        f"best_arm_error={arm_error:.4f}, best_gripper_error={gripper_error:.4f}"
                    )
            if hold_time > 0.0:
                deadline = time.monotonic() + hold_time
                while rclpy.ok() and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
