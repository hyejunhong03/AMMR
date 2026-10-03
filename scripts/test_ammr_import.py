import argparse
import os

import numpy as np

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description="Headless Isaac Sim import smoke test for the combined AMMR USD.")
parser.add_argument(
    "--usd",
    default="/home/autolab/AMMR/isaac_usd/ammr_center_mount/ammr_center_mount.usda",
    help="Path to the combined AMMR USD.",
)
parser.add_argument("--physics-hz", type=float, default=240.0, help="Physics update rate in Hz.")
parser.add_argument("--steps", type=int, default=60, help="Physics steps to run after import.")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import GroundPlane
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics


ROBOT_PRIM = "/World/AMMR"
ARTICULATION_PRIM_HINT = "/World/AMMR/mechabot/Geometry/base_footprint/base_link"
GROUND_Z = 0.0
ROBOT_START_Z = 0.020827
EXPECTED_JOINTS = [
    "left_wheel_joint",
    "right_wheel_joint",
    "front_right_caster_roller_joint",
    "front_left_caster_roller_joint",
    "rear_right_caster_roller_joint",
    "rear_left_caster_roller_joint",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
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


def main():
    usd_path = os.path.abspath(args.usd)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(usd_path)

    stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)
    GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, GROUND_Z])
    stage_utils.add_reference_to_stage(usd_path=usd_path, path=ROBOT_PRIM)
    app_utils.update_app()

    stage = stage_utils.get_current_stage()
    roots = _find_articulation_roots(stage)
    print(f"USD: {usd_path}")
    print(f"Articulation roots: {roots}")
    if ARTICULATION_PRIM_HINT not in roots:
        raise RuntimeError(f"Expected mechabot articulation root {ARTICULATION_PRIM_HINT}, got: {roots}")
    if len(roots) != 1:
        raise RuntimeError(f"Expected one combined articulation root, got: {roots}")

    robot_xform = XformPrim(ROBOT_PRIM, reset_xform_op_properties=True)
    robot_xform.set_world_poses(positions=[0.0, 0.0, ROBOT_START_Z])
    robot = Articulation(ARTICULATION_PRIM_HINT)

    SimulationManager.set_physics_dt(1.0 / args.physics_hz)
    app_utils.play()
    simulation_app.update()
    SimulationManager.step(steps=args.steps)
    simulation_app.update()

    dof_names = list(robot.dof_names)
    print(f"DOF count: {len(dof_names)}")
    print(f"DOF names: {dof_names}")
    missing = [name for name in EXPECTED_JOINTS if name not in dof_names]
    if missing:
        raise RuntimeError(f"Missing expected AMMR joints: {missing}")

    positions = _as_numpy(robot.get_dof_positions())[0]
    velocities = _as_numpy(robot.get_dof_velocities())[0]
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise RuntimeError("AMMR DOF state contains non-finite values.")
    print("AMMR import smoke test passed.")

    app_utils.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
