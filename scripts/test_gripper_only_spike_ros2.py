#!/usr/bin/env python3
"""Open/close the gripper with no object and report arm joint spikes."""

import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from ammr_mycobot_interface import (
    ARM_JOINTS,
    CONTROL_JOINTS,
    GRIPPER_GRASP_CLOSE_RAD,
    GRIPPER_OPEN_RAD,
    ROS2_JOINT_STATES_TOPIC,
    ROS2_JOINT_TARGET_TOPIC,
    ROS2_ROBOT_STATE_TOPIC,
)


class GripperOnlySpikeTest(Node):
    def __init__(self):
        super().__init__("ammr_gripper_only_spike_test")
        self.robot_state = None
        self.samples = []
        self.pub = self.create_publisher(Float64MultiArray, ROS2_JOINT_TARGET_TOPIC, 10)
        self.create_subscription(Float64MultiArray, ROS2_ROBOT_STATE_TOPIC, self._on_robot_state, 10)
        self.create_subscription(JointState, ROS2_JOINT_STATES_TOPIC, self._on_joint_state, 10)

    def _on_robot_state(self, msg):
        if len(msg.data) >= len(CONTROL_JOINTS):
            self.robot_state = np.asarray(msg.data[: len(CONTROL_JOINTS)], dtype=np.float64)

    def _on_joint_state(self, msg):
        if len(msg.position) < len(CONTROL_JOINTS) or len(msg.velocity) < len(CONTROL_JOINTS):
            return
        self.samples.append(
            {
                "time": time.monotonic(),
                "position": np.asarray(msg.position[: len(CONTROL_JOINTS)], dtype=np.float64),
                "velocity": np.asarray(msg.velocity[: len(CONTROL_JOINTS)], dtype=np.float64),
            }
        )

    def wait_for_state(self, timeout):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.robot_state is not None:
                return self.robot_state.copy()
        raise TimeoutError("Timed out waiting for /ammr/robot_state")

    def publish_target(self, arm_target, gripper_target):
        msg = Float64MultiArray()
        msg.data = [*map(float, arm_target), float(gripper_target)]
        self.pub.publish(msg)

    def hold_target(self, arm_target, gripper_target, duration, hz):
        period = 1.0 / max(float(hz), 1e-6)
        deadline = time.monotonic() + float(duration)
        while rclpy.ok() and time.monotonic() < deadline:
            self.publish_target(arm_target, gripper_target)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--open", dest="open_rad", type=float, default=GRIPPER_OPEN_RAD)
    parser.add_argument("--close", dest="close_rad", type=float, default=GRIPPER_GRASP_CLOSE_RAD)
    parser.add_argument("--hold", type=float, default=3.0)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--publish-hz", type=float, default=20.0)
    parser.add_argument("--state-timeout", type=float, default=5.0)
    return parser.parse_args()


def summarize(samples, initial_arm):
    if not samples:
        raise RuntimeError("No joint state samples collected")
    positions = np.stack([sample["position"] for sample in samples])
    velocities = np.stack([sample["velocity"] for sample in samples])
    times = np.asarray([sample["time"] for sample in samples])

    arm_pos = positions[:, :6]
    arm_vel = velocities[:, :6]
    gripper = positions[:, 6]
    gripper_vel = velocities[:, 6]

    max_arm_velocity_per_joint = np.max(np.abs(arm_vel), axis=0)
    max_arm_velocity = float(np.max(max_arm_velocity_per_joint))
    peak_joint = int(np.argmax(max_arm_velocity_per_joint))
    max_arm_delta_per_joint = np.max(np.abs(arm_pos - initial_arm[None, :]), axis=0)
    max_arm_delta = float(np.max(max_arm_delta_per_joint))
    peak_delta_joint = int(np.argmax(max_arm_delta_per_joint))

    print("=== gripper-only spike test ===")
    print(f"samples: {len(samples)}, duration: {times[-1] - times[0]:.2f}s")
    print(f"gripper range: {float(np.min(gripper)):.5f} .. {float(np.max(gripper)):.5f} rad")
    print(f"max gripper velocity: {float(np.max(np.abs(gripper_vel))):.5f} rad/s")
    print(
        "max arm velocity: "
        f"{max_arm_velocity:.5f} rad/s on {ARM_JOINTS[peak_joint]}"
    )
    print(
        "max arm position drift from start: "
        f"{max_arm_delta:.5f} rad on {ARM_JOINTS[peak_delta_joint]}"
    )
    print("arm max velocity per joint:")
    for name, value in zip(ARM_JOINTS, max_arm_velocity_per_joint):
        print(f"  {name}: {float(value):.5f}")
    print("arm max position drift per joint:")
    for name, value in zip(ARM_JOINTS, max_arm_delta_per_joint):
        print(f"  {name}: {float(value):.5f}")


def main():
    args = parse_args()
    rclpy.init()
    node = GripperOnlySpikeTest()
    try:
        initial = node.wait_for_state(args.state_timeout)
        arm_target = initial[:6].copy()
        print(f"initial arm target: {[round(float(v), 5) for v in arm_target]}")
        node.samples.clear()
        node.hold_target(arm_target, args.open_rad, args.settle, args.publish_hz)
        for _ in range(max(1, args.cycles)):
            node.hold_target(arm_target, args.close_rad, args.hold, args.publish_hz)
            node.hold_target(arm_target, args.open_rad, args.hold, args.publish_hz)
        summarize(node.samples, arm_target)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
