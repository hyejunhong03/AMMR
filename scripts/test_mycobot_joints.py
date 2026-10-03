import argparse
import math
import os

import numpy as np

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description="Headless Isaac Sim joint smoke test for myCobot.")
parser.add_argument(
    "--usd",
    default="/home/autolab/AMMR/isaac_usd/mycobot_280_m5_adaptive_gripper_reimport/mycobot_280_m5_adaptive_gripper/mycobot_280_m5_adaptive_gripper.usda",
    help="Path to the imported myCobot USD.",
)
parser.add_argument("--physics-hz", type=float, default=240.0, help="Physics update rate in Hz.")
parser.add_argument("--render-hz", type=float, default=60.0, help="App update rate in Hz.")
parser.add_argument("--settle-steps", type=int, default=60, help="Physics steps to hold the imported pose.")
parser.add_argument("--steps-per-target", type=int, default=240, help="Physics steps to hold the commanded target.")
parser.add_argument("--target-amplitude", type=float, default=0.15, help="Arm joint target amplitude in radians.")
parser.add_argument("--drive-stiffness", type=float, default=1000.0, help="Runtime position drive stiffness.")
parser.add_argument("--drive-damping", type=float, default=100.0, help="Runtime position drive damping.")
parser.add_argument("--max-effort", type=float, default=1000.0, help="Runtime drive max effort.")
parser.add_argument("--max-velocity", type=float, default=2.0, help="Runtime DOF max velocity in rad/s.")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics


ROBOT_PRIM = "/World/MyCobot"
ARM_JOINTS = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]
GRIPPER_JOINTS = [
    "gripper_controller",
    "gripper_base_to_gripper_left2",
    "gripper_left3_to_gripper_left1",
    "gripper_base_to_gripper_right3",
    "gripper_base_to_gripper_right2",
    "gripper_right3_to_gripper_right1",
]


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


def _step_app(physics_steps_per_frame, remaining_steps):
    while remaining_steps > 0:
        batch_steps = min(physics_steps_per_frame, remaining_steps)
        SimulationManager.step(steps=batch_steps)
        simulation_app.update()
        remaining_steps -= batch_steps


def _assert_finite(label, values):
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{label} contains non-finite values: {values.tolist()}")


def _configure_position_drives(robot, dof_names, joint_names):
    joint_indices = [dof_names.index(name) for name in joint_names if name in dof_names]
    if not joint_indices:
        return
    shape = (1, len(joint_indices))
    robot.set_dof_max_velocities(np.full(shape, args.max_velocity, dtype=np.float32), dof_indices=joint_indices)
    robot.set_dof_max_efforts(np.full(shape, args.max_effort, dtype=np.float32), dof_indices=joint_indices)
    robot.set_dof_gains(
        np.full(shape, args.drive_stiffness, dtype=np.float32),
        np.full(shape, args.drive_damping, dtype=np.float32),
        dof_indices=joint_indices,
    )
    print(
        "Configured runtime drives: "
        f"joints={len(joint_indices)}, stiffness={args.drive_stiffness}, damping={args.drive_damping}, "
        f"max_effort={args.max_effort}, max_velocity={args.max_velocity} rad/s"
    )


def main():
    usd_path = os.path.abspath(args.usd)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(usd_path)

    stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)
    stage_utils.add_reference_to_stage(usd_path=usd_path, path=ROBOT_PRIM)

    stage = stage_utils.get_current_stage()
    robot_prim = stage.GetPrimAtPath(ROBOT_PRIM)
    robot_prim.GetVariantSet("Physics").SetVariantSelection("physics")
    app_utils.update_app()

    roots = _find_articulation_roots(stage)
    print(f"USD: {usd_path}")
    print(f"Articulation roots: {roots}")
    if len(roots) != 1:
        raise RuntimeError(f"Expected one articulation root, got: {roots}")

    robot_xform = XformPrim(ROBOT_PRIM, reset_xform_op_properties=True)
    robot_xform.set_world_poses(positions=[0.0, 0.0, 0.0])
    robot = Articulation(roots[0])

    SimulationManager.set_physics_dt(1.0 / args.physics_hz)
    app_utils.play()
    simulation_app.update()

    dof_names = list(robot.dof_names)
    print(f"DOF names: {dof_names}")
    if not dof_names:
        raise RuntimeError("No articulation DOFs found.")

    missing_arm_joints = [name for name in ARM_JOINTS if name not in dof_names]
    if missing_arm_joints:
        raise RuntimeError(f"Missing expected myCobot arm joints: {missing_arm_joints}")

    commanded_joints = [name for name in ARM_JOINTS + ["gripper_controller"] if name in dof_names]
    _configure_position_drives(robot, dof_names, commanded_joints)

    physics_steps_per_frame = max(1, int(round(args.physics_hz / args.render_hz)))
    print(f"Physics: {args.physics_hz:.0f} Hz, app update: {args.physics_hz / physics_steps_per_frame:.0f} Hz.")

    initial_positions = _as_numpy(robot.get_dof_positions())[0].copy()
    _assert_finite("Initial DOF positions", initial_positions)
    hold_targets = initial_positions.reshape(1, len(dof_names)).astype(np.float32)
    robot.set_dof_position_targets(hold_targets)
    _step_app(physics_steps_per_frame, args.settle_steps)

    initial_positions = _as_numpy(robot.get_dof_positions())[0].copy()
    _assert_finite("Post-settle DOF positions", initial_positions)
    print(f"Post-settle DOF positions: {initial_positions.tolist()}")

    targets = initial_positions.reshape(1, len(dof_names)).astype(np.float32)
    for index, joint_name in enumerate(commanded_joints):
        joint_index = dof_names.index(joint_name)
        sign = -1.0 if index % 2 else 1.0
        amplitude = 0.03 if joint_name == "gripper_controller" else args.target_amplitude
        targets[0, joint_index] = initial_positions[joint_index] + sign * amplitude

    robot.set_dof_position_targets(targets)
    _step_app(physics_steps_per_frame, args.steps_per_target)

    current_positions = _as_numpy(robot.get_dof_positions())[0].copy()
    _assert_finite("Commanded DOF positions", current_positions)

    measured_deltas = {}
    for joint_name in commanded_joints:
        joint_index = dof_names.index(joint_name)
        measured_deltas[joint_name] = float(current_positions[joint_index] - initial_positions[joint_index])
        print(
            f"Moved {joint_name}: target={targets[0, joint_index]:.3f}, "
            f"position={current_positions[joint_index]:.6f}, delta={measured_deltas[joint_name]:.6f}"
        )

    final_positions = _as_numpy(robot.get_dof_positions())[0].copy()
    final_velocities = _as_numpy(robot.get_dof_velocities())[0].copy()
    _assert_finite("Final DOF positions", final_positions)
    _assert_finite("Final DOF velocities", final_velocities)
    print(f"Final DOF positions: {final_positions.tolist()}")
    print(f"Final DOF velocities: {final_velocities.tolist()}")

    too_small = {}
    for name, delta in measured_deltas.items():
        min_delta = 0.00001 if name == "gripper_controller" else 0.005
        if not math.isfinite(delta) or abs(delta) < min_delta:
            too_small[name] = delta
    if too_small:
        raise RuntimeError(f"Some commanded joints did not move enough: {too_small}")

    app_utils.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
