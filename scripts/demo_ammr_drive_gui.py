import argparse
import csv
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description="GUI Isaac Sim drive demo for the combined AMMR robot.")
parser.add_argument(
    "--usd",
    default="/home/autolab/AMMR/isaac_usd/ammr_center_mount/ammr_center_mount.usda",
    help="Path to the combined AMMR USD.",
)
parser.add_argument("--duration", type=float, default=10.0, help="Seconds to run the active demo.")
parser.add_argument("--wheel-speed", type=float, default=0.6, help="Wheel velocity target in rad/s.")
parser.add_argument("--settle-time", type=float, default=3.0, help="Seconds to let the robot settle before driving.")
parser.add_argument("--ramp-time", type=float, default=5.0, help="Seconds to ramp up wheel speed.")
parser.add_argument("--physics-hz", type=float, default=240.0, help="Physics update rate in Hz.")
parser.add_argument("--render-hz", type=float, default=60.0, help="Viewport update rate in Hz.")
parser.add_argument("--arm-amplitude", type=float, default=0.06, help="Small arm joint target amplitude in radians.")
parser.add_argument("--gripper-amplitude", type=float, default=0.08, help="Adaptive gripper target amplitude in radians.")
parser.add_argument("--drive-stiffness", type=float, default=1000.0, help="Runtime arm position drive stiffness.")
parser.add_argument("--drive-damping", type=float, default=100.0, help="Runtime arm position drive damping.")
parser.add_argument("--max-effort", type=float, default=1000.0, help="Runtime arm drive max effort.")
parser.add_argument("--max-velocity", type=float, default=2.0, help="Runtime arm DOF max velocity in rad/s.")
parser.add_argument("--no-drive", action="store_true", help="Keep the mobile base stationary.")
parser.add_argument("--no-arm", action="store_true", help="Do not command the myCobot arm joints.")
parser.add_argument("--enable-gripper", action="store_true", help="Also command the imported adaptive gripper joint.")
parser.add_argument("--headless", action="store_true", help="Run without opening the Isaac Sim window.")
parser.add_argument(
    "--log-csv",
    default=None,
    help="CSV path for robot state logging. Defaults to /home/autolab/AMMR/logs/ammr_drive_<timestamp>.csv. Use 'none' to disable.",
)
parser.add_argument("--close-when-done", action="store_true", help="Close Isaac Sim after the demo instead of keeping it open.")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.viewports import set_camera_view
from pxr import UsdPhysics


ROBOT_PRIM = "/World/AMMR"
ARTICULATION_PRIM_HINT = "/World/AMMR/mechabot/Geometry/base_footprint/base_link"
LEFT_WHEEL_JOINT = "left_wheel_joint"
RIGHT_WHEEL_JOINT = "right_wheel_joint"
CASTER_ROLLER_JOINTS = [
    "front_right_caster_roller_joint",
    "front_left_caster_roller_joint",
    "rear_right_caster_roller_joint",
    "rear_left_caster_roller_joint",
]
ARM_JOINTS = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]
GRIPPER_CONTROLLER = "gripper_controller"
GRIPPER_JOINT_SIGNS = {
    "gripper_controller": 1.0,
    "gripper_base_to_gripper_left2": 1.0,
    "gripper_base_to_gripper_right3": -1.0,
    "gripper_base_to_gripper_right2": -1.0,
    "gripper_left3_to_gripper_left1": -1.0,
    "gripper_right3_to_gripper_right1": 1.0,
}
GROUND_Z = 0.0
ROBOT_START_Z = 0.020827


def _as_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _find_articulation_roots(stage):
    roots = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
    return roots


def _quat_wxyz_to_euler(qw, qx, qy, qz):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _resolve_log_path():
    if args.log_csv and args.log_csv.lower() == "none":
        return None
    if args.log_csv:
        return Path(args.log_csv).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"/home/autolab/AMMR/logs/ammr_drive_{timestamp}.csv")


def _step_frame(physics_steps_per_frame):
    SimulationManager.step(steps=physics_steps_per_frame)
    simulation_app.update()
    return physics_steps_per_frame / args.physics_hz


def _configure_arm_drives(robot, dof_names, arm_indices):
    if not arm_indices:
        return
    shape = (1, len(arm_indices))
    robot.set_dof_max_velocities(np.full(shape, args.max_velocity, dtype=np.float32), dof_indices=arm_indices)
    robot.set_dof_max_efforts(np.full(shape, args.max_effort, dtype=np.float32), dof_indices=arm_indices)
    robot.set_dof_gains(
        np.full(shape, args.drive_stiffness, dtype=np.float32),
        np.full(shape, args.drive_damping, dtype=np.float32),
        dof_indices=arm_indices,
    )
    configured_names = [dof_names[index] for index in arm_indices]
    print(
        "Configured arm drives: "
        f"{configured_names}, stiffness={args.drive_stiffness}, damping={args.drive_damping}, "
        f"max_effort={args.max_effort}, max_velocity={args.max_velocity} rad/s"
    )


def main():
    usd_path = os.path.abspath(args.usd)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(usd_path)

    stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)
    GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, GROUND_Z])
    light = DistantLight("/World/DistantLight")
    light.set_intensities(500)

    stage_utils.add_reference_to_stage(usd_path=usd_path, path=ROBOT_PRIM)
    app_utils.update_app()

    stage = stage_utils.get_current_stage()
    roots = _find_articulation_roots(stage)
    print(f"USD: {usd_path}")
    print(f"Articulation roots: {roots}")
    if ARTICULATION_PRIM_HINT in roots:
        articulation_prim = ARTICULATION_PRIM_HINT
    elif len(roots) == 1:
        articulation_prim = roots[0]
    else:
        raise RuntimeError(f"Expected one AMMR articulation root, got: {roots}")

    robot_xform = XformPrim(ROBOT_PRIM, reset_xform_op_properties=True)
    robot_xform.set_world_poses(positions=[0.0, 0.0, ROBOT_START_Z])
    robot = Articulation(articulation_prim)

    if not args.headless:
        set_camera_view(
            eye=[2.6, -2.4, 1.35],
            target=[0.35, 0.0, 0.28],
            camera_prim_path="/OmniverseKit_Persp",
        )

    SimulationManager.set_physics_dt(1.0 / args.physics_hz)
    app_utils.play()
    simulation_app.update()

    dof_names = list(robot.dof_names)
    print(f"DOF count: {len(dof_names)}")
    print(f"DOF names: {dof_names}")

    required = [LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT] + ARM_JOINTS
    missing = [name for name in required if name not in dof_names]
    if missing:
        raise RuntimeError(f"Missing expected AMMR joints: {missing}")

    left_index = dof_names.index(LEFT_WHEEL_JOINT)
    right_index = dof_names.index(RIGHT_WHEEL_JOINT)
    caster_indices = {name: dof_names.index(name) for name in CASTER_ROLLER_JOINTS if name in dof_names}
    arm_indices = [dof_names.index(name) for name in ARM_JOINTS if name in dof_names]
    gripper_indices = []
    if args.enable_gripper:
        gripper_indices = [dof_names.index(name) for name in GRIPPER_JOINT_SIGNS if name in dof_names]
        missing_gripper_joints = [name for name in GRIPPER_JOINT_SIGNS if name not in dof_names]
        if missing_gripper_joints:
            print(f"Missing optional gripper joints: {missing_gripper_joints}")
    command_indices = arm_indices + gripper_indices
    _configure_arm_drives(robot, dof_names, command_indices)

    physics_steps_per_frame = max(1, int(round(args.physics_hz / args.render_hz)))
    actual_render_hz = args.physics_hz / physics_steps_per_frame
    print(f"Physics: {args.physics_hz:.0f} Hz, viewport/app update: {actual_render_hz:.0f} Hz.")
    print(f"Settling for {args.settle_time:.1f}s, then running demo for {args.duration:.1f}s.")

    log_path = _resolve_log_path()
    log_file = None
    log_writer = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", newline="")
        fieldnames = [
            "sample",
            "time_s",
            "phase",
            "target_left_wheel_rad_s",
            "target_right_wheel_rad_s",
            "base_x_m",
            "base_y_m",
            "base_z_m",
            "base_roll_rad",
            "base_pitch_rad",
            "base_yaw_rad",
            "base_speed_xy_m_s",
            "base_wx_rad_s",
            "base_wy_rad_s",
            "base_wz_rad_s",
            "left_wheel_vel_rad_s",
            "right_wheel_vel_rad_s",
        ]
        for joint_name in CASTER_ROLLER_JOINTS:
            if joint_name in caster_indices:
                fieldnames.append(f"{joint_name}_vel_rad_s")
        for joint_name in ARM_JOINTS + list(GRIPPER_JOINT_SIGNS):
            if joint_name in dof_names:
                fieldnames.append(f"{joint_name}_pos_rad")
                fieldnames.append(f"{joint_name}_vel_rad_s")
        log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        log_writer.writeheader()
        print(f"Logging AMMR state to: {log_path}")

    sim_time_s = 0.0
    sample_index = 0

    def log_state(phase, target_left=0.0, target_right=0.0):
        nonlocal sample_index
        if log_writer is None:
            return
        positions, orientations = robot.get_world_poses()
        base_position = _as_numpy(positions)[0]
        base_orientation = _as_numpy(orientations)[0]
        dof_positions = _as_numpy(robot.get_dof_positions())[0]
        dof_velocities = _as_numpy(robot.get_dof_velocities())[0]
        try:
            linear_velocities, angular_velocities = robot.get_velocities()
            linear_velocity = _as_numpy(linear_velocities)[0]
            angular_velocity = _as_numpy(angular_velocities)[0]
        except Exception:
            linear_velocity = np.array([math.nan, math.nan, math.nan])
            angular_velocity = np.array([math.nan, math.nan, math.nan])

        qw, qx, qy, qz = [float(v) for v in base_orientation]
        roll, pitch, yaw = _quat_wxyz_to_euler(qw, qx, qy, qz)
        row = {
            "sample": sample_index,
            "time_s": f"{sim_time_s:.6f}",
            "phase": phase,
            "target_left_wheel_rad_s": f"{target_left:.6f}",
            "target_right_wheel_rad_s": f"{target_right:.6f}",
            "base_x_m": f"{float(base_position[0]):.9f}",
            "base_y_m": f"{float(base_position[1]):.9f}",
            "base_z_m": f"{float(base_position[2]):.9f}",
            "base_roll_rad": f"{roll:.9f}",
            "base_pitch_rad": f"{pitch:.9f}",
            "base_yaw_rad": f"{yaw:.9f}",
            "base_speed_xy_m_s": f"{math.hypot(float(linear_velocity[0]), float(linear_velocity[1])):.9f}",
            "base_wx_rad_s": f"{float(angular_velocity[0]):.9f}",
            "base_wy_rad_s": f"{float(angular_velocity[1]):.9f}",
            "base_wz_rad_s": f"{float(angular_velocity[2]):.9f}",
            "left_wheel_vel_rad_s": f"{float(dof_velocities[left_index]):.9f}",
            "right_wheel_vel_rad_s": f"{float(dof_velocities[right_index]):.9f}",
        }
        for joint_name, joint_index in caster_indices.items():
            row[f"{joint_name}_vel_rad_s"] = f"{float(dof_velocities[joint_index]):.9f}"
        for joint_name in ARM_JOINTS + list(GRIPPER_JOINT_SIGNS):
            if joint_name in dof_names:
                joint_index = dof_names.index(joint_name)
                row[f"{joint_name}_pos_rad"] = f"{float(dof_positions[joint_index]):.9f}"
                row[f"{joint_name}_vel_rad_s"] = f"{float(dof_velocities[joint_index]):.9f}"
        log_writer.writerow(row)
        sample_index += 1

    zero_wheel_targets = np.zeros((1, 2), dtype=np.float32)
    initial_positions = _as_numpy(robot.get_dof_positions())[0].copy()
    command_targets = (
        initial_positions[command_indices].reshape(1, len(command_indices)).astype(np.float32)
        if command_indices
        else None
    )
    if command_targets is not None and not args.no_arm:
        for local_index, dof_index in enumerate(command_indices):
            joint_name = dof_names[dof_index]
            if joint_name in GRIPPER_JOINT_SIGNS:
                sign = GRIPPER_JOINT_SIGNS[joint_name]
                amplitude = args.gripper_amplitude
            else:
                sign = -1.0 if local_index % 2 else 1.0
                amplitude = args.arm_amplitude
            command_targets[0, local_index] = initial_positions[dof_index] + sign * amplitude

    log_state("start")
    for _ in range(int(args.settle_time * actual_render_hz)):
        if not simulation_app.is_running():
            break
        robot.set_dof_velocity_targets(zero_wheel_targets, dof_indices=[left_index, right_index])
        sim_time_s += _step_frame(physics_steps_per_frame)
        log_state("settle")

    if command_targets is not None and not args.no_arm:
        robot.set_dof_position_targets(command_targets, dof_indices=command_indices)
        if args.enable_gripper:
            print("Commanded small myCobot arm and gripper pose during AMMR demo.")
        else:
            print("Commanded small myCobot arm pose during AMMR demo.")
    if args.no_drive:
        print("Mobile base drive disabled by --no-drive.")

    steps = int(args.duration * actual_render_hz)
    ramp_steps = max(1, int(args.ramp_time * actual_render_hz))
    for step in range(steps):
        if not simulation_app.is_running():
            break
        ramp = min(1.0, float(step + 1) / float(ramp_steps))
        wheel_speed = 0.0 if args.no_drive else args.wheel_speed * ramp
        wheel_targets = np.array([[wheel_speed, wheel_speed]], dtype=np.float32)
        robot.set_dof_velocity_targets(wheel_targets, dof_indices=[left_index, right_index])
        sim_time_s += _step_frame(physics_steps_per_frame)
        log_state("drive", target_left=wheel_speed, target_right=wheel_speed)

    robot.set_dof_velocity_targets(zero_wheel_targets, dof_indices=[left_index, right_index])
    log_state("stop_command")
    app_utils.stop()
    print("AMMR GUI demo finished.")
    if log_file is not None:
        log_file.close()
        print(f"Saved AMMR state CSV: {log_path}")

    if not args.close_when_done:
        print("Isaac Sim will stay open. Close the window when done.")
        while simulation_app.is_running():
            simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
