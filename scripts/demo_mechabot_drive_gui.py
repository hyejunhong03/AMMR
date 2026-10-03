import argparse
import csv
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description="GUI Isaac Sim drive demo for AMMR mechabot.")
parser.add_argument(
    "--usd",
    default="/home/autolab/AMMR/isaac_usd/mechabot_scaled_rolling_casters/mechabot_scaled_no_castor_collision/mechabot_scaled_no_castor_collision.usda",
    help="Path to the imported mechabot USD.",
)
parser.add_argument("--duration", type=float, default=10.0, help="Seconds to drive forward.")
parser.add_argument("--wheel-speed", type=float, default=1.6, help="Wheel velocity target in rad/s.")
parser.add_argument("--settle-time", type=float, default=2.0, help="Seconds to let the robot settle before driving.")
parser.add_argument("--ramp-time", type=float, default=3.0, help="Seconds to ramp up wheel speed.")
parser.add_argument("--physics-hz", type=float, default=240.0, help="Physics update rate in Hz.")
parser.add_argument("--render-hz", type=float, default=60.0, help="Viewport update rate in Hz.")
parser.add_argument(
    "--log-csv",
    default=None,
    help="CSV path for robot state logging. Defaults to /home/autolab/AMMR/logs/mechabot_drive_<timestamp>.csv. Use 'none' to disable.",
)
parser.add_argument("--keep-open", action="store_true", default=True, help="Keep Isaac Sim open after the demo.")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": False})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.viewports import set_camera_view
from pxr import UsdPhysics


ROBOT_PRIM = "/World/Mechabot"
ARTICULATION_PRIM_HINT = "/World/Mechabot/Geometry/base_footprint/base_link"
LEFT_WHEEL_JOINT = "left_wheel_joint"
RIGHT_WHEEL_JOINT = "right_wheel_joint"
CASTER_ROLLER_JOINTS = [
    "front_right_caster_roller_joint",
    "front_left_caster_roller_joint",
    "rear_right_caster_roller_joint",
    "rear_left_caster_roller_joint",
]
GROUND_Z = 0.0
WHEEL_RADIUS = 0.051258
WHEEL_JOINT_Z = 0.030431
ROBOT_START_Z = GROUND_Z + WHEEL_RADIUS - WHEEL_JOINT_Z


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
    return Path(f"/home/autolab/AMMR/logs/mechabot_drive_{timestamp}.csv")


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
    stage = stage_utils.get_current_stage()
    robot_prim = stage.GetPrimAtPath(ROBOT_PRIM)
    robot_prim.GetVariantSet("Physics").SetVariantSelection("physics")
    app_utils.update_app()

    roots = _find_articulation_roots(stage)
    print(f"USD: {usd_path}")
    print(f"Articulation roots: {roots}")
    if ARTICULATION_PRIM_HINT in roots:
        articulation_prim = ARTICULATION_PRIM_HINT
    elif len(roots) == 1:
        articulation_prim = roots[0]
    else:
        raise RuntimeError(f"Expected one articulation root, got: {roots}")

    robot_xform = XformPrim(ROBOT_PRIM, reset_xform_op_properties=True)
    robot_xform.set_world_poses(positions=[0.0, 0.0, ROBOT_START_Z])
    robot = Articulation(articulation_prim)

    set_camera_view(
        eye=[2.4, -2.2, 1.2],
        target=[0.4, 0.0, 0.1],
        camera_prim_path="/OmniverseKit_Persp",
    )

    SimulationManager.set_physics_dt(1.0 / args.physics_hz)
    app_utils.play()
    simulation_app.update()

    dof_names = list(robot.dof_names)
    left_index = dof_names.index(LEFT_WHEEL_JOINT)
    right_index = dof_names.index(RIGHT_WHEEL_JOINT)
    caster_indices = {name: dof_names.index(name) for name in CASTER_ROLLER_JOINTS if name in dof_names}
    print(f"DOF names: {dof_names}")
    physics_steps_per_frame = max(1, int(round(args.physics_hz / args.render_hz)))
    actual_render_hz = args.physics_hz / physics_steps_per_frame
    print(f"Physics: {args.physics_hz:.0f} Hz, viewport: {actual_render_hz:.0f} Hz.")
    print(f"Settling for {args.settle_time:.1f}s, then driving for {args.duration:.1f}s.")

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
            "base_qw",
            "base_qx",
            "base_qy",
            "base_qz",
            "base_roll_rad",
            "base_pitch_rad",
            "base_yaw_rad",
            "base_vx_m_s",
            "base_vy_m_s",
            "base_vz_m_s",
            "base_speed_xy_m_s",
            "base_wx_rad_s",
            "base_wy_rad_s",
            "base_wz_rad_s",
            "left_wheel_pos_rad",
            "right_wheel_pos_rad",
            "left_wheel_vel_rad_s",
            "right_wheel_vel_rad_s",
        ]
        for joint_name in CASTER_ROLLER_JOINTS:
            if joint_name in caster_indices:
                fieldnames.append(f"{joint_name}_pos_rad")
                fieldnames.append(f"{joint_name}_vel_rad_s")
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=fieldnames,
        )
        log_writer.writeheader()
        print(f"Logging robot state to: {log_path}")

    sample_index = 0
    sim_time_s = 0.0

    def log_state(phase, target_left=0.0, target_right=0.0):
        nonlocal sample_index, sim_time_s
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
            "base_qw": f"{qw:.9f}",
            "base_qx": f"{qx:.9f}",
            "base_qy": f"{qy:.9f}",
            "base_qz": f"{qz:.9f}",
            "base_roll_rad": f"{roll:.9f}",
            "base_pitch_rad": f"{pitch:.9f}",
            "base_yaw_rad": f"{yaw:.9f}",
            "base_vx_m_s": f"{float(linear_velocity[0]):.9f}",
            "base_vy_m_s": f"{float(linear_velocity[1]):.9f}",
            "base_vz_m_s": f"{float(linear_velocity[2]):.9f}",
            "base_speed_xy_m_s": f"{math.hypot(float(linear_velocity[0]), float(linear_velocity[1])):.9f}",
            "base_wx_rad_s": f"{float(angular_velocity[0]):.9f}",
            "base_wy_rad_s": f"{float(angular_velocity[1]):.9f}",
            "base_wz_rad_s": f"{float(angular_velocity[2]):.9f}",
            "left_wheel_pos_rad": f"{float(dof_positions[left_index]):.9f}",
            "right_wheel_pos_rad": f"{float(dof_positions[right_index]):.9f}",
            "left_wheel_vel_rad_s": f"{float(dof_velocities[left_index]):.9f}",
            "right_wheel_vel_rad_s": f"{float(dof_velocities[right_index]):.9f}",
        }
        for joint_name, joint_index in caster_indices.items():
            row[f"{joint_name}_pos_rad"] = f"{float(dof_positions[joint_index]):.9f}"
            row[f"{joint_name}_vel_rad_s"] = f"{float(dof_velocities[joint_index]):.9f}"
        log_writer.writerow(row)
        sample_index += 1

    zero_targets = np.zeros((1, len(dof_names)), dtype=np.float32)
    log_state("start")
    for _ in range(int(args.settle_time * actual_render_hz)):
        if not simulation_app.is_running():
            break
        robot.set_dof_velocity_targets(zero_targets)
        SimulationManager.step(steps=physics_steps_per_frame)
        sim_time_s += physics_steps_per_frame / args.physics_hz
        simulation_app.update()
        log_state("settle")

    steps = int(args.duration * actual_render_hz)
    ramp_steps = max(1, int(args.ramp_time * actual_render_hz))
    for step in range(steps):
        if not simulation_app.is_running():
            break
        ramp = min(1.0, float(step + 1) / float(ramp_steps))
        wheel_speed = args.wheel_speed * ramp
        velocity_targets = np.zeros((1, len(dof_names)), dtype=np.float32)
        velocity_targets[0, left_index] = wheel_speed
        velocity_targets[0, right_index] = wheel_speed
        robot.set_dof_velocity_targets(velocity_targets)
        SimulationManager.step(steps=physics_steps_per_frame)
        sim_time_s += physics_steps_per_frame / args.physics_hz
        simulation_app.update()
        log_state("drive", target_left=wheel_speed, target_right=wheel_speed)

    robot.set_dof_velocity_targets(zero_targets)
    log_state("stop_command")
    app_utils.stop()
    print("Demo finished. Close Isaac Sim when done.")
    if log_file is not None:
        log_file.close()
        print(f"Saved robot state CSV: {log_path}")

    if args.keep_open:
        while simulation_app.is_running():
            simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
