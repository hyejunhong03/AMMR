import argparse
import math
import os

import numpy as np

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description="Headless Isaac Sim drive smoke test for AMMR mechabot.")
parser.add_argument(
    "--usd",
    default="/home/autolab/AMMR/isaac_usd/mechabot_scaled_rolling_casters/mechabot_scaled_no_castor_collision/mechabot_scaled_no_castor_collision.usda",
    help="Path to the imported mechabot USD.",
)
parser.add_argument("--steps", type=int, default=240, help="Number of physics steps to run.")
parser.add_argument("--wheel-speed", type=float, default=2.0, help="Wheel velocity target in rad/s.")
parser.add_argument("--physics-hz", type=float, default=240.0, help="Physics update rate in Hz.")
parser.add_argument("--render-hz", type=float, default=60.0, help="App update rate in Hz.")
parser.add_argument("--settle-steps", type=int, default=240, help="Zero-velocity physics steps before measuring.")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import GroundPlane
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics


ROBOT_PRIM = "/World/Mechabot"
ARTICULATION_PRIM_HINT = "/World/Mechabot/Geometry/base_footprint/base_link"
LEFT_WHEEL_JOINT = "left_wheel_joint"
RIGHT_WHEEL_JOINT = "right_wheel_joint"
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


def main():
    usd_path = os.path.abspath(args.usd)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(usd_path)

    stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)
    GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, GROUND_Z])
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

    SimulationManager.set_physics_dt(1.0 / args.physics_hz)
    app_utils.play()
    simulation_app.update()

    dof_names = list(robot.dof_names)
    print(f"DOF names: {dof_names}")
    left_index = dof_names.index(LEFT_WHEEL_JOINT)
    right_index = dof_names.index(RIGHT_WHEEL_JOINT)
    print(f"Wheel indices: left={left_index}, right={right_index}")
    physics_steps_per_frame = max(1, int(round(args.physics_hz / args.render_hz)))
    print(f"Physics: {args.physics_hz:.0f} Hz, app update: {args.physics_hz / physics_steps_per_frame:.0f} Hz.")

    initial_pose = _as_numpy(robot.get_world_poses()[0])[0].copy()
    initial_positions = _as_numpy(robot.get_dof_positions())[0].copy()
    print(f"Initial base position: {initial_pose.tolist()}")
    print(f"Initial DOF positions: {initial_positions.tolist()}")

    velocity_targets = np.zeros((1, len(dof_names)), dtype=np.float32)
    velocity_targets[0, left_index] = args.wheel_speed
    velocity_targets[0, right_index] = args.wheel_speed

    zero_targets = np.zeros((1, len(dof_names)), dtype=np.float32)
    settle_remaining_steps = args.settle_steps
    while settle_remaining_steps > 0:
        batch_steps = min(physics_steps_per_frame, settle_remaining_steps)
        robot.set_dof_velocity_targets(zero_targets)
        SimulationManager.step(steps=batch_steps)
        simulation_app.update()
        settle_remaining_steps -= batch_steps

    initial_pose = _as_numpy(robot.get_world_poses()[0])[0].copy()
    initial_positions = _as_numpy(robot.get_dof_positions())[0].copy()
    print(f"Post-settle base position: {initial_pose.tolist()}")
    print(f"Post-settle DOF positions: {initial_positions.tolist()}")

    robot.set_dof_velocity_targets(velocity_targets)

    z_samples = [float(initial_pose[2])]
    remaining_steps = args.steps
    while remaining_steps > 0:
        batch_steps = min(physics_steps_per_frame, remaining_steps)
        SimulationManager.step(steps=batch_steps)
        simulation_app.update()
        current_pose = _as_numpy(robot.get_world_poses()[0])[0].copy()
        z_samples.append(float(current_pose[2]))
        remaining_steps -= batch_steps

    final_pose = _as_numpy(robot.get_world_poses()[0])[0].copy()
    final_positions = _as_numpy(robot.get_dof_positions())[0].copy()
    final_velocities = _as_numpy(robot.get_dof_velocities())[0].copy()
    print(f"Final base position: {final_pose.tolist()}")
    print(f"Final DOF positions: {final_positions.tolist()}")
    print(f"Final DOF velocities: {final_velocities.tolist()}")

    wheel_position_delta = {
        LEFT_WHEEL_JOINT: float(final_positions[left_index] - initial_positions[left_index]),
        RIGHT_WHEEL_JOINT: float(final_positions[right_index] - initial_positions[right_index]),
    }
    base_delta = final_pose - initial_pose
    print(f"Wheel position delta rad: {wheel_position_delta}")
    print(f"Base position delta m: {base_delta.tolist()}")
    print(f"Base z min/max/range m: {min(z_samples):.6f} / {max(z_samples):.6f} / {max(z_samples) - min(z_samples):.6f}")

    if not all(math.isfinite(delta) for delta in wheel_position_delta.values()):
        raise RuntimeError("Wheel joint position delta contains a non-finite value.")
    if max(abs(delta) for delta in wheel_position_delta.values()) < 0.01:
        raise RuntimeError("Wheel joints did not move enough to count as a drive smoke-test pass.")

    app_utils.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
