import argparse
import os

import numpy as np

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description="GUI Isaac Sim joint and gripper demo for myCobot.")
parser.add_argument(
    "--usd",
    default="/home/autolab/AMMR/isaac_usd/mycobot_280_m5_adaptive_gripper_reimport/mycobot_280_m5_adaptive_gripper/mycobot_280_m5_adaptive_gripper.usda",
    help="Path to the imported myCobot USD.",
)
parser.add_argument("--physics-hz", type=float, default=240.0, help="Physics update rate in Hz.")
parser.add_argument("--render-hz", type=float, default=60.0, help="Viewport update rate in Hz.")
parser.add_argument("--settle-time", type=float, default=2.0, help="Seconds to hold the imported pose.")
parser.add_argument("--seconds-per-move", type=float, default=1.6, help="Seconds to hold each joint target.")
parser.add_argument("--arm-amplitude", type=float, default=0.08, help="Arm joint target amplitude in radians.")
parser.add_argument("--gripper-open", type=float, default=0.04, help="Gripper open target in radians.")
parser.add_argument("--gripper-close", type=float, default=-0.06, help="Gripper close target in radians.")
parser.add_argument("--drive-stiffness", type=float, default=80.0, help="Runtime position drive stiffness.")
parser.add_argument("--drive-damping", type=float, default=12.0, help="Runtime position drive damping.")
parser.add_argument("--max-effort", type=float, default=20.0, help="Runtime drive max effort.")
parser.add_argument("--max-velocity", type=float, default=0.6, help="Runtime DOF max velocity in rad/s.")
parser.add_argument("--gripper-stiffness", type=float, default=20.0, help="Runtime gripper drive stiffness.")
parser.add_argument("--gripper-damping", type=float, default=3.0, help="Runtime gripper drive damping.")
parser.add_argument("--gripper-max-effort", type=float, default=2.0, help="Runtime gripper drive max effort.")
parser.add_argument("--gripper-max-velocity", type=float, default=0.5, help="Runtime gripper DOF max velocity in rad/s.")
parser.add_argument(
    "--drive-all-gripper-joints",
    action="store_true",
    help="Also drive the mimic gripper linkage joints. Default commands only gripper_controller.",
)
parser.add_argument("--disable-gravity", action="store_true", help="Disable gravity for debug-only joint checks.")
parser.add_argument("--no-fix-base", action="store_true", help="Do not add a world fixed joint to the myCobot base.")
parser.add_argument("--add-ground", action="store_true", help="Add a physical ground plane. Default leaves only the viewport grid.")
parser.add_argument("--headless", action="store_true", help="Run without opening the Isaac Sim window.")
parser.add_argument("--close-when-done", action="store_true", help="Close Isaac Sim after the demo.")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Sdf, UsdPhysics


ROBOT_PRIM = "/World/MyCobot"
ARM_JOINTS = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]
GRIPPER_JOINT_SIGNS = {
    "gripper_controller": 1.0,
    "gripper_base_to_gripper_left2": 1.0,
    "gripper_left3_to_gripper_left1": -1.0,
    "gripper_base_to_gripper_right3": -1.0,
    "gripper_base_to_gripper_right2": -1.0,
    "gripper_right3_to_gripper_right1": 1.0,
}
GRIPPER_CONTROLLER = "gripper_controller"


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


def _step_frames(physics_steps_per_frame, frames):
    for _ in range(frames):
        if not simulation_app.is_running():
            return False
        SimulationManager.step(steps=physics_steps_per_frame)
        simulation_app.update()
    return True


def _get_base_position(robot):
    positions, _ = robot.get_world_poses()
    return _as_numpy(positions)[0].copy()


def _configure_position_drives(
    robot,
    dof_names,
    joint_names,
    stiffness,
    damping,
    max_effort,
    max_velocity,
    label,
):
    joint_indices = [dof_names.index(name) for name in joint_names if name in dof_names]
    if not joint_indices:
        return []
    shape = (1, len(joint_indices))
    robot.set_dof_max_velocities(np.full(shape, max_velocity, dtype=np.float32), dof_indices=joint_indices)
    robot.set_dof_max_efforts(np.full(shape, max_effort, dtype=np.float32), dof_indices=joint_indices)
    robot.set_dof_gains(
        np.full(shape, stiffness, dtype=np.float32),
        np.full(shape, damping, dtype=np.float32),
        dof_indices=joint_indices,
    )
    print(
        f"Configured {label} position drives: "
        f"{[dof_names[index] for index in joint_indices]}, "
        f"stiffness={stiffness}, damping={damping}, max_effort={max_effort}, max_velocity={max_velocity}"
    )
    return joint_indices


def _disable_gravity_for_joint_check():
    physics_scenes = SimulationManager.get_physics_scenes()
    if not physics_scenes:
        print("No physics scene found for disabling gravity.")
        return
    for physics_scene in physics_scenes:
        physics_scene.set_enabled_gravity(False)
        physics_scene.set_gravity((0.0, 0.0, 0.0))
        print(f"Disabled gravity for stable arm-only joint check: {physics_scene.path}")


def _fix_base_to_world(stage, body_path):
    body_prim = stage.GetPrimAtPath(body_path)
    if not body_prim or not body_prim.IsValid():
        raise RuntimeError(f"Cannot fix missing base body to world: {body_path}")
    joint_path = f"{ROBOT_PRIM}/world_fixed_joint"
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
    print(f"Fixed myCobot base to world: {joint_path} -> {body_path}")


def _command_target(robot, dof_names, joint_names, target_positions):
    joint_indices = [dof_names.index(name) for name in joint_names]
    positions = np.asarray(target_positions, dtype=np.float32).reshape(1, len(joint_indices))
    robot.set_dof_position_targets(positions, dof_indices=joint_indices)


def _get_gripper_command_joints(dof_names):
    if args.drive_all_gripper_joints:
        return [name for name in GRIPPER_JOINT_SIGNS if name in dof_names]
    if GRIPPER_CONTROLLER in dof_names:
        return [GRIPPER_CONTROLLER]
    return []


def _get_gripper_targets(joint_names, value):
    if args.drive_all_gripper_joints:
        return [GRIPPER_JOINT_SIGNS[name] * value for name in joint_names]
    return [value for _ in joint_names]


def main():
    usd_path = os.path.abspath(args.usd)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(usd_path)

    stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)
    if args.add_ground:
        GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, -0.02])
    light = DistantLight("/World/DistantLight")
    light.set_intensities(600)
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
    if not args.no_fix_base:
        _fix_base_to_world(stage, roots[0])

    robot_xform = XformPrim(ROBOT_PRIM, reset_xform_op_properties=True)
    robot_xform.set_world_poses(positions=[0.0, 0.0, 0.0])
    robot = Articulation(roots[0])

    if not args.headless:
        set_camera_view(
            eye=[0.65, -0.85, 0.55],
            target=[0.0, 0.0, 0.16],
            camera_prim_path="/OmniverseKit_Persp",
        )

    SimulationManager.set_physics_dt(1.0 / args.physics_hz)
    if args.disable_gravity:
        _disable_gravity_for_joint_check()
    app_utils.play()
    simulation_app.update()
    if args.disable_gravity:
        _disable_gravity_for_joint_check()

    dof_names = list(robot.dof_names)
    print(f"DOF count: {len(dof_names)}")
    print(f"DOF names: {dof_names}")
    missing_arm_joints = [name for name in ARM_JOINTS if name not in dof_names]
    if missing_arm_joints:
        raise RuntimeError(f"Missing expected myCobot arm joints: {missing_arm_joints}")

    arm_drive_indices = _configure_position_drives(
        robot,
        dof_names,
        ARM_JOINTS,
        stiffness=args.drive_stiffness,
        damping=args.drive_damping,
        max_effort=args.max_effort,
        max_velocity=args.max_velocity,
        label="arm",
    )
    gripper_command_joints = _get_gripper_command_joints(dof_names)
    gripper_drive_indices = _configure_position_drives(
        robot,
        dof_names,
        gripper_command_joints,
        stiffness=args.gripper_stiffness,
        damping=args.gripper_damping,
        max_effort=args.gripper_max_effort,
        max_velocity=args.gripper_max_velocity,
        label="gripper",
    )
    if not arm_drive_indices or not gripper_drive_indices:
        raise RuntimeError("Failed to configure expected arm or gripper drives.")
    if args.drive_all_gripper_joints:
        print("Gripper command mode: driving controller plus mimic follower joints.")
    else:
        print("Gripper command mode: driving only gripper_controller; USD mimic joints follow it.")

    physics_steps_per_frame = max(1, int(round(args.physics_hz / args.render_hz)))
    actual_render_hz = args.physics_hz / physics_steps_per_frame
    settle_frames = int(args.settle_time * actual_render_hz)
    move_frames = max(1, int(args.seconds_per_move * actual_render_hz))
    print(f"Physics: {args.physics_hz:.0f} Hz, viewport/app update: {actual_render_hz:.0f} Hz.")

    home_positions = _as_numpy(robot.get_dof_positions())[0].copy()
    base_position_before = _get_base_position(robot)
    print(f"Base position before settle: {base_position_before.tolist()}")
    arm_home = [home_positions[dof_names.index(name)] for name in ARM_JOINTS]
    gripper_home = [home_positions[dof_names.index(name)] for name in gripper_command_joints]
    _command_target(robot, dof_names, ARM_JOINTS, arm_home)
    _command_target(robot, dof_names, gripper_command_joints, gripper_home)
    print(f"Settling for {args.settle_time:.1f}s.")
    _step_frames(physics_steps_per_frame, settle_frames)
    base_position_after_settle = _get_base_position(robot)
    base_drift = float(np.linalg.norm(base_position_after_settle - base_position_before))
    print(f"Base position after settle: {base_position_after_settle.tolist()}, drift={base_drift:.9f} m")

    for joint_number, joint_name in enumerate(ARM_JOINTS, start=1):
        joint_index = dof_names.index(joint_name)
        for direction, label in ((1.0, "positive"), (-1.0, "negative"), (0.0, "home")):
            targets = [home_positions[dof_names.index(name)] for name in ARM_JOINTS]
            if direction:
                targets[ARM_JOINTS.index(joint_name)] = home_positions[joint_index] + direction * args.arm_amplitude
            _command_target(robot, dof_names, ARM_JOINTS, targets)
            print(f"Moving arm joint {joint_number}: {joint_name} -> {label}")
            if not _step_frames(physics_steps_per_frame, move_frames):
                return

    for label, gripper_value in (
        ("open", args.gripper_open),
        ("close", args.gripper_close),
        ("open", args.gripper_open),
    ):
        targets = _get_gripper_targets(gripper_command_joints, gripper_value)
        _command_target(robot, dof_names, gripper_command_joints, targets)
        print(f"Moving gripper -> {label} ({gripper_value:.3f} rad)")
        if not _step_frames(physics_steps_per_frame, move_frames * 2):
            return

    base_position_final = _get_base_position(robot)
    base_total_drift = float(np.linalg.norm(base_position_final - base_position_before))
    print(f"Base position final: {base_position_final.tolist()}, total_drift={base_total_drift:.9f} m")
    app_utils.stop()
    print("myCobot joint and gripper GUI demo finished.")

    if not args.close_when_done:
        print("Isaac Sim will stay open. Close the window when done.")
        while simulation_app.is_running():
            simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
