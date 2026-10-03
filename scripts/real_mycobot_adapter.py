#!/usr/bin/env python3
"""ROS2 adapter that maps the AMMR arm interface to a real myCobot 280.

Default mode is dry-run. Pass --real to open the serial port and command hardware.
"""

import argparse
import fcntl
import os
import time
import traceback
from contextlib import contextmanager

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from ammr_mycobot_interface import (
    COMMAND_GRIPPER_CLOSE_RAD,
    COMMAND_GRIPPER_OPEN_RAD,
    CONTROL_JOINTS,
    GRIPPER_OPEN_RAD,
    ROS2_JOINT_STATES_TOPIC,
    ROS2_JOINT_TARGET_TOPIC,
    ROS2_ROBOT_STATE_TOPIC,
    arm_radians_to_degrees,
    coerce_control_targets,
    gripper_command_to_percent,
)


def parse_args():
    parser = argparse.ArgumentParser(description="AMMR real myCobot ROS2 adapter.")
    parser.add_argument("--real", action="store_true", help="Connect to hardware and send commands.")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="myCobot serial port.")
    parser.add_argument("--baud", type=int, default=115200, help="myCobot serial baud rate.")
    parser.add_argument("--speed", type=int, default=25, help="Arm command speed for send_angles().")
    parser.add_argument("--gripper-speed", type=int, default=80, help="Gripper command speed.")
    parser.add_argument("--state-hz", type=float, default=30.0, help="State publish rate.")
    parser.add_argument("--lock-file", default="/tmp/mycobot_lock", help="Serial lock file.")
    parser.add_argument(
        "--command-gripper-close",
        type=float,
        default=COMMAND_GRIPPER_CLOSE_RAD,
        help="Canonical AMMR gripper command that maps to real gripper value 0.",
    )
    parser.add_argument(
        "--command-gripper-open",
        type=float,
        default=COMMAND_GRIPPER_OPEN_RAD,
        help="Canonical AMMR gripper command that maps to real gripper value 100.",
    )
    parser.add_argument("--node-name", default="ammr_real_mycobot_adapter", help="ROS2 node name.")
    return parser.parse_args()


@contextmanager
def serial_lock(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class RealMyCobotAdapter(Node):
    def __init__(self, args):
        super().__init__(args.node_name)
        self._args = args
        self._mc = None
        self._last_state = [0.0] * len(CONTROL_JOINTS)
        self._last_state[-1] = GRIPPER_OPEN_RAD
        self._last_command_time = 0.0
        self._last_dry_run_log = 0.0
        self._last_read_error_log = 0.0

        self._joint_state_pub = self.create_publisher(JointState, ROS2_JOINT_STATES_TOPIC, 10)
        self._robot_state_pub = self.create_publisher(Float64MultiArray, ROS2_ROBOT_STATE_TOPIC, 10)
        self._target_sub = self.create_subscription(
            Float64MultiArray,
            ROS2_JOINT_TARGET_TOPIC,
            self._on_joint_targets,
            10,
        )
        self._timer = self.create_timer(1.0 / max(args.state_hz, 1e-6), self._publish_state)

        if args.real:
            self._connect_robot()
        else:
            self.get_logger().warn("dry-run mode: no serial connection, commands are mirrored into state only")

        self.get_logger().info(f"subscribe {ROS2_JOINT_TARGET_TOPIC} std_msgs/Float64MultiArray length=7")
        self.get_logger().info(f"publish   {ROS2_JOINT_STATES_TOPIC} sensor_msgs/JointState")
        self.get_logger().info(f"publish   {ROS2_ROBOT_STATE_TOPIC} std_msgs/Float64MultiArray")

    def _connect_robot(self):
        try:
            import pymycobot
            from packaging import version
            from pymycobot import MyCobot280
        except ImportError as exc:
            raise RuntimeError(
                "pymycobot is required for --real mode. Install it in the ROS2 Python environment."
            ) from exc

        min_version = "4.0.0"
        if version.parse(pymycobot.__version__) < version.parse(min_version):
            raise RuntimeError(
                f"pymycobot >= {min_version} is required, got {pymycobot.__version__}"
            )

        self.get_logger().info(f"connecting real myCobot: port={self._args.port}, baud={self._args.baud}")
        self._mc = MyCobot280(self._args.port, self._args.baud)
        time.sleep(0.05)
        with serial_lock(self._args.lock_file):
            self._mc.set_fresh_mode(1)
        time.sleep(0.05)

    def _on_joint_targets(self, msg):
        try:
            targets = coerce_control_targets(msg.data)
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return

        self._last_state = list(targets)
        self._last_command_time = time.monotonic()

        arm_degrees = arm_radians_to_degrees(targets[:6])
        gripper_value = gripper_command_to_percent(
            targets[6],
            close_rad=self._args.command_gripper_close,
            open_rad=self._args.command_gripper_open,
        )

        if self._mc is None:
            now = time.monotonic()
            if now - self._last_dry_run_log >= 1.0:
                self.get_logger().info(f"dry-run target arm_deg={arm_degrees}, gripper_value={gripper_value}")
                self._last_dry_run_log = now
            return

        try:
            with serial_lock(self._args.lock_file):
                try:
                    self._mc.send_angles(arm_degrees, self._args.speed, _async=True)
                except TypeError:
                    self._mc.send_angles(arm_degrees, self._args.speed)
                try:
                    self._mc.set_gripper_value(gripper_value, self._args.gripper_speed, gripper_type=1)
                except TypeError:
                    self._mc.set_gripper_value(gripper_value, self._args.gripper_speed)
        except Exception:
            self.get_logger().error(f"failed to command myCobot:\n{traceback.format_exc()}")

    def _read_real_arm_state(self):
        if self._mc is None:
            return None
        try:
            with serial_lock(self._args.lock_file):
                angles = self._mc.get_angles()
        except Exception:
            now = time.monotonic()
            if now - self._last_read_error_log >= 2.0:
                self.get_logger().warn(f"failed to read myCobot angles:\n{traceback.format_exc()}")
                self._last_read_error_log = now
            return None
        if not isinstance(angles, list) or len(angles) != 6:
            return None
        return [float(angle) * 3.141592653589793 / 180.0 for angle in angles]

    def _publish_state(self):
        real_arm_state = self._read_real_arm_state()
        if real_arm_state is not None:
            self._last_state[:6] = real_arm_state

        stamp = self.get_clock().now().to_msg()
        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.name = list(CONTROL_JOINTS)
        joint_state.position = list(self._last_state)
        joint_state.velocity = [0.0] * len(CONTROL_JOINTS)
        joint_state.effort = []
        self._joint_state_pub.publish(joint_state)

        robot_state = Float64MultiArray()
        robot_state.data = list(self._last_state)
        self._robot_state_pub.publish(robot_state)


def main():
    args = parse_args()
    rclpy.init()
    node = RealMyCobotAdapter(args)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
