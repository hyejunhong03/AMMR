import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

from ammr_mycobot_interface import (
    ARM_JOINTS,
    CONTROL_JOINTS,
    DEFAULT_DEMO_POSE,
    DEMO_POSE_CUBE_POSITIONS,
    GRIPPER_CONTROLLER,
    GRIPPER_GRASP_ATTACH_RAD,
    GRIPPER_GRASP_CLOSE_RAD,
    GRIPPER_GRASP_ENGAGE_RAD,
    GRIPPER_OPEN_RAD,
    GRASP_EVENT_FIELD_NAMES,
    IMAGE_SIZE,
    OBJECT_INDEX_BY_NAME,
    OBJECT_NAMES,
    OBJECT_PRIM_PATHS,
    OBJECT_STATE_NAMES,
    ROS2_CAMERA_INFO_TOPIC,
    ROS2_GRASP_EVENTS_TOPIC,
    ROS2_IMAGE_TOPIC,
    ROS2_JOINT_STATES_TOPIC,
    ROS2_JOINT_TARGET_TOPIC,
    ROS2_OBJECT_STATES_TOPIC,
    ROS2_RESET_TASK_SCENE_TOPIC,
    ROS2_SET_OBJECT_POSE_TOPIC,
    ROS2_ROBOT_STATE_TOPIC,
    SIM_GRIPPER_LIMITS,
    WRIST_CAMERA_FRAME_ID,
    demo_pose_cube_position,
    task_object_spec,
)
from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description="Interactive Isaac Sim slider control for myCobot.")
parser.add_argument(
    "--usd",
    default="/home/autolab/AMMR/isaac_usd/mycobot_280_m5_adaptive_gripper_reimport/mycobot_280_m5_adaptive_gripper/mycobot_280_m5_adaptive_gripper.usda",
    help="Path to the imported myCobot USD.",
)
parser.add_argument("--physics-hz", type=float, default=240.0, help="Physics update rate in Hz.")
parser.add_argument("--render-hz", type=float, default=60.0, help="Viewport update rate in Hz.")
parser.add_argument("--settle-time", type=float, default=1.0, help="Seconds to hold the imported pose before UI control.")
parser.add_argument("--drive-stiffness", type=float, default=120.0, help="Runtime arm position drive stiffness.")
parser.add_argument("--drive-damping", type=float, default=24.0, help="Runtime arm position drive damping.")
parser.add_argument("--max-effort", type=float, default=60.0, help="Runtime arm drive max effort.")
parser.add_argument("--max-velocity", type=float, default=0.5, help="Runtime arm DOF max velocity in rad/s.")
parser.add_argument("--gripper-stiffness", type=float, default=200.0, help="Runtime gripper drive stiffness.")
parser.add_argument("--gripper-damping", type=float, default=20.0, help="Runtime gripper drive damping.")
parser.add_argument("--gripper-max-effort", type=float, default=5.0, help="Runtime gripper drive max effort.")
parser.add_argument("--gripper-max-velocity", type=float, default=3.0, help="Runtime gripper DOF max velocity in rad/s.")
parser.add_argument(
    "--kinematic-gripper",
    action="store_true",
    help=(
        "Set gripper DOF positions directly instead of driving them through PhysX "
        "position drives. Use this for VLA data collection/evaluation where object "
        "pickup is handled by grasp assist and gripper drive reaction forces should "
        "not perturb the arm articulation."
    ),
)
parser.add_argument(
    "--simple-gripper",
    action="store_true",
    help=(
        "Use a VLA-friendly virtual parallel gripper: hide the imported adaptive "
        "gripper visual, disable its drives, publish a virtual scalar gripper state, "
        "and move two simple finger meshes under gripper_base."
    ),
)
parser.add_argument(
    "--drive-all-gripper-joints",
    action="store_true",
    help=(
        "Legacy mode: drive the controller plus every mimic follower joint. "
        "This can fight the imported mimic constraints in Isaac Sim and is kept "
        "only for comparing against old behavior."
    ),
)
parser.add_argument(
    "--controller-only-gripper-drive",
    dest="drive_all_gripper_joints",
    action="store_false",
    help=(
        "Default mode: drive only gripper_controller and leave mimic follower "
        "joints passive by disabling their runtime drives."
    ),
)
parser.add_argument("--gripper-open", type=float, default=GRIPPER_OPEN_RAD, help="Open button gripper target in radians.")
parser.add_argument("--gripper-close", type=float, default=GRIPPER_GRASP_CLOSE_RAD, help="Close button gripper target in radians.")
parser.add_argument(
    "--gripper-ramp-rate",
    type=float,
    default=0.5,
    help=(
        "Ramp rate in rad/s for the UI Open/Close buttons. ROS2 commands are applied "
        "verbatim; shape those at the publisher instead, so the recorded action equals "
        "the applied command."
    ),
)
parser.add_argument("--disable-gravity", action="store_true", help="Disable gravity for debug-only slider checks.")
parser.add_argument("--no-fix-base", action="store_true", help="Do not add a world fixed joint to the myCobot base.")
parser.add_argument("--add-ground", action="store_true", help="Add a physical ground plane. Default leaves only the viewport grid.")
parser.add_argument("--no-wrist-camera", action="store_true", help="Do not add the wrist RGB camera.")
parser.add_argument("--view-wrist-camera", action="store_true", help="Start the viewport from the wrist RGB camera.")
parser.add_argument("--no-task-scene", action="store_true", help="Do not add the table and target objects.")
parser.add_argument(
    "--demo-pose",
    choices=sorted(DEMO_POSE_CUBE_POSITIONS),
    default=DEFAULT_DEMO_POSE,
    help=(
        "Place the red cube where this scripted pose profile expects it. Run "
        "scripted_pick_demo.py with the same --demo-pose."
    ),
)
parser.add_argument(
    "--cube-position",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=None,
    help="Override the red cube position in world meters instead of using --demo-pose.",
)
parser.add_argument(
    "--no-grasp-assist",
    dest="grasp_assist",
    action="store_false",
    help="Disable kinematic target-object attachment during simulated grasps.",
)
parser.add_argument(
    "--dynamic-contact-grasp",
    action="store_true",
    help=(
        "Experimental physics-validation mode: task objects are dynamic and "
        "collision-enabled, and kinematic grasp assist is forced off. This keeps "
        "the stable data-collection mode unchanged while separating full contact "
        "grasp experiments."
    ),
)
parser.add_argument(
    "--grasp-attach-threshold",
    type=float,
    default=GRIPPER_GRASP_ATTACH_RAD,
    help="Close command target required for grasp assist attach.",
)
parser.add_argument(
    "--grasp-engage-threshold",
    type=float,
    default=GRIPPER_GRASP_ENGAGE_RAD,
    help="Observed gripper state required before grasp assist attach.",
)
parser.add_argument(
    "--grasp-release-threshold",
    type=float,
    default=0.02,
    help="Detach the target object when gripper state is at or above this value.",
)
parser.add_argument(
    "--grasp-attach-distance",
    type=float,
    default=0.03,
    help=(
        "Maximum distance in metres between the point where the fingers close and the "
        "target object. Was 0.08 while the distance was measured from the gripper_base "
        "origin 74 mm behind that point; with the reference point corrected, 0.08 would "
        "attach objects nowhere near the fingers. The scripted demo grasps at 0.6-2.4 mm "
        "and rollouts that reach the cube at 7-17 mm, so 30 mm -- one cube width -- "
        "separates a grasp from a miss."
    ),
)
parser.add_argument(
    "--grasp-max-lateral-offset",
    type=float,
    default=0.0,
    help=(
        "Optional strict grasp assist gate. If positive, refuse attach when the "
        "object is farther than this from the grasp point in the gripper local x/z "
        "plane. 0 keeps the legacy distance-only behaviour."
    ),
)
parser.add_argument(
    "--grasp-max-tool-axis-offset",
    type=float,
    default=0.0,
    help=(
        "Optional strict grasp assist gate. If positive, refuse attach when the "
        "object offset along the gripper local tool axis exceeds this value. 0 keeps "
        "the legacy distance-only behaviour."
    ),
)
parser.add_argument(
    "--grasp-still-time",
    type=float,
    default=0.2,
    help="Seconds the gripper state must remain nearly steady before grasp assist attach.",
)
parser.add_argument(
    "--grasp-still-tolerance",
    type=float,
    default=0.003,
    help="Gripper-state change tolerated while checking grasp assist stillness.",
)
parser.add_argument(
    "--hold-gripper-after-attach",
    action="store_true",
    help=(
        "After grasp assist attaches an object, clamp deeper close commands to "
        "--attached-gripper-hold. This avoids pressing the imported adaptive "
        "gripper farther into an over-constrained close pose after pickup is "
        "already handled kinematically."
    ),
)
parser.add_argument(
    "--attached-gripper-hold",
    type=float,
    default=GRIPPER_GRASP_ENGAGE_RAD,
    help="Minimum gripper target allowed while an object is attached.",
)
parser.add_argument(
    "--headless",
    action="store_true",
    help=(
        "Run without the Isaac viewport. Nobody watches the window during batch "
        "collection or policy rollout, and rendering it is the single largest cost in "
        "the app loop: 20 Hz with the viewport, 45 Hz without. The wrist camera still "
        "renders, and its frames are indistinguishable from the windowed ones (the "
        "headless-vs-windowed pixel difference matches the frame-to-frame difference "
        "within a single mode)."
    ),
)
parser.add_argument(
    "--loop-profile",
    type=float,
    default=0.0,
    help="Seconds between app-loop timing reports. 0 disables.",
)
parser.add_argument(
    "--physics-diagnostics",
    default=None,
    help=(
        "Write per-physics-step diagnostics to this CSV path. Intended for "
        "--dynamic-contact-grasp debugging; records joint state/targets, object "
        "pose/velocity, gripper mimic error, and contact force if available."
    ),
)
parser.add_argument(
    "--physics-diagnostics-stride",
    type=int,
    default=1,
    help="Record every N physics steps when --physics-diagnostics is set.",
)
parser.add_argument(
    "--physics-diagnostics-max-contact-count",
    type=int,
    default=64,
    help="Maximum contact points reserved for dynamic-object contact diagnostics.",
)
parser.add_argument("--ros2", action="store_true", help="Enable direct rclpy publishers/subscriber for ROS2 control.")
parser.add_argument("--ros2-state-hz", type=float, default=30.0, help="ROS2 joint/state publish rate.")
parser.add_argument("--ros2-image-hz", type=float, default=15.0, help="ROS2 wrist RGB image publish rate.")
parser.add_argument("--ros2-node-name", default="ammr_isaac_mycobot", help="ROS2 node name.")
parser.set_defaults(grasp_assist=True, drive_all_gripper_joints=False)
args, _ = parser.parse_known_args()
if args.dynamic_contact_grasp:
    args.grasp_assist = False
    args.hold_gripper_after_attach = False
    if args.kinematic_gripper:
        parser.error("--dynamic-contact-grasp cannot be combined with --kinematic-gripper")
    if args.simple_gripper:
        parser.error("--dynamic-contact-grasp cannot be combined with --simple-gripper")
    if args.drive_all_gripper_joints:
        parser.error("--dynamic-contact-grasp keeps mimic followers passive; do not use --drive-all-gripper-joints")
if args.physics_diagnostics_stride < 1:
    parser.error("--physics-diagnostics-stride must be >= 1")

simulation_app = SimulationApp({"headless": args.headless})

import omni.ui as ui
import omni.replicator.core as rep
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


STARTUP_ERROR_PATH = "/tmp/ammr_sim_startup_error.txt"
ROBOT_PRIM = "/World/MyCobot"
TASK_SCENE_ROOT = "/World/TaskScene"
TABLE_PRIM_PATH = f"{TASK_SCENE_ROOT}/table"
# Longest frame gap the gripper ramp will integrate over, in seconds. Caps the
# target jump if the app loop stalls.
MAX_GRIPPER_RAMP_DT = 0.2
# Callbacks dispatched per frame, enough to drain the command queues at the
# publish rates used for data collection.
MAX_SPINS_PER_FRAME = 16
GRIPPER_JOINT_SIGNS = {
    GRIPPER_CONTROLLER: 1.0,
    "gripper_base_to_gripper_left2": 1.0,
    "gripper_left3_to_gripper_left1": -1.0,
    "gripper_base_to_gripper_right3": -1.0,
    "gripper_base_to_gripper_right2": -1.0,
    "gripper_right3_to_gripper_right1": 1.0,
}
GRIPPER_JOINTS = list(GRIPPER_JOINT_SIGNS)
GRIPPER_FOLLOWER_JOINTS = [joint_name for joint_name in GRIPPER_JOINTS if joint_name != GRIPPER_CONTROLLER]
# Where the fingers close, in the gripper_base frame. Same value as
# ammr_ik.TCP_OFFSET_IN_GRIPPER_BASE, derived from the two verified grasp poses; kept
# here as a literal because this module must not import the IK helper.
GRASP_POINT_IN_ATTACH_FRAME = np.array([-0.0013, 0.0743, -0.0092])
SIMPLE_GRIPPER_ROOT_NAME = "simple_parallel_gripper"
SIMPLE_FINGER_LENGTH_Y = 0.060
SIMPLE_FINGER_THICKNESS_X = 0.008
SIMPLE_FINGER_THICKNESS_Z = 0.014
SIMPLE_FINGER_OPEN_HALF_GAP = 0.028
SIMPLE_FINGER_CLOSED_HALF_GAP = 0.016
WRIST_CAMERA_TRANSLATION = (0.0, -0.028, 0.04)
WRIST_CAMERA_RPY = (1.5708, -1.5708, 0.0)
WRIST_CAMERA_FOV = 1.2
# Render at the policy's input size: SmolVLA consumes 256x256, and the render product
# readback is the main cost in the app loop.
WRIST_CAMERA_RESOLUTION = IMAGE_SIZE
TABLE_POSITION = (0.174, 0.0, 0.08)
TABLE_SIZE = (0.158, 0.236, 0.04)
# --cube-position overrides; otherwise the placement comes from the pose profile.
CUBE_POSITION = tuple(
    float(value)
    for value in (
        args.cube_position
        if args.cube_position is not None
        else demo_pose_cube_position(args.demo_pose)
    )
)
# Live placement targets, seeded from the profile and updated by
# ROS2_SET_OBJECT_POSE_TOPIC. _reset_task_scene restores these rather than the startup
# constants, so a randomised placement survives the reset between episodes.
OBJECT_TARGET_POSITIONS = {}
OBJECT_TARGET_ORIENTATIONS = {}
CYLINDER_POSITION = (0.31, 0.06, task_object_spec("blue_cylinder")["table_z"])
OBJECT_STATE_PRIMS = dict(OBJECT_PRIM_PATHS)
OBJECT_RIGID_PRIMS = {}
OBJECT_CONTACT_FILTER_LABELS = {}
JOINT_LABELS = {
    "joint2_to_joint1": "joint1",
    "joint3_to_joint2": "joint2",
    "joint4_to_joint3": "joint3",
    "joint5_to_joint4": "joint4",
    "joint6_to_joint5": "joint5",
    "joint6output_to_joint6": "joint6",
    GRIPPER_CONTROLLER: "gripper",
}
JOINT_LIMITS = {
    "joint2_to_joint1": (-2.9321, 2.9321),
    "joint3_to_joint2": (-2.4434, 2.4434),
    "joint4_to_joint3": (-2.6179, 2.6179),
    "joint5_to_joint4": (-2.6179, 2.6179),
    "joint6_to_joint5": (-2.7052, 2.7925),
    "joint6output_to_joint6": (-3.14159, 3.14159),
    # Keep the UI inside a conservative dynamic range. The raw controller limit is
    # -0.74..0.15, but large close targets make the imported linkage unstable.
    GRIPPER_CONTROLLER: SIM_GRIPPER_LIMITS,
    "gripper_base_to_gripper_left2": (-0.8, 0.5),
    "gripper_left3_to_gripper_left1": (-0.5, 0.5),
    "gripper_base_to_gripper_right3": (-0.15, 0.7),
    "gripper_base_to_gripper_right2": (-0.5, 0.8),
    "gripper_right3_to_gripper_right1": (-0.5, 0.5),
}


def _uses_virtual_gripper():
    return bool(args.simple_gripper)


def _uses_dynamic_contact_objects():
    return bool(args.dynamic_contact_grasp)


def _as_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _quat_from_rpy(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def _quat_from_rotation_matrix(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def _reset_xform(prim, translate, orient):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(orient)


def _set_display_color(prim, color):
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _apply_collision(prim):
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)


def _apply_contact_report(prim, threshold=0.0):
    api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
    attr = api.GetThresholdAttr()
    if not attr:
        attr = api.CreateThresholdAttr()
    attr.Set(float(threshold))


def _apply_rigid_body(prim, mass):
    _apply_collision(prim)
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))


def _find_articulation_roots(stage):
    roots = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
    return roots


def _get_base_position(robot):
    positions, _ = robot.get_world_poses()
    return _as_numpy(positions)[0].copy()


def _step_frames(physics_steps_per_frame, frames):
    for _ in range(frames):
        if not simulation_app.is_running():
            return False
        SimulationManager.step(steps=physics_steps_per_frame)
        simulation_app.update()
    return True


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
        print(f"Disabled gravity for stable arm-only slider check: {physics_scene.path}")


def _fix_base_to_world(stage, body_path):
    body_prim = stage.GetPrimAtPath(body_path)
    if not body_prim or not body_prim.IsValid():
        raise RuntimeError(f"Cannot fix missing base body to world: {body_path}")
    joint_path = f"{ROBOT_PRIM}/world_fixed_joint"
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
    print(f"Fixed myCobot base to world: {joint_path} -> {body_path}")


def _get_joint6_flange_path(articulation_root):
    return (
        f"{articulation_root}/joint1/joint2/joint3/joint4/"
        "joint5/joint6/joint6_flange"
    )


def _get_gripper_base_path(articulation_root):
    return f"{_get_joint6_flange_path(articulation_root)}/gripper_base"


def _get_grasp_attach_frame_path(stage, articulation_root):
    gripper_base_path = _get_gripper_base_path(articulation_root)
    gripper_base = stage.GetPrimAtPath(gripper_base_path)
    if gripper_base and gripper_base.IsValid():
        return gripper_base_path
    return _get_joint6_flange_path(articulation_root)


def _hide_original_gripper_subtree(stage, gripper_base_path, keep_path):
    hidden = 0
    collision_disabled = 0
    link_deactivated = 0
    joint_deactivated = 0
    base_prefix = f"{gripper_base_path}/"
    keep_prefix = f"{keep_path}/"
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(base_prefix) or path == keep_path or path.startswith(keep_prefix):
            continue
        if prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            _set_collision_enabled(prim, False)
            collision_disabled += 1
    gripper_base = stage.GetPrimAtPath(gripper_base_path)
    if gripper_base and gripper_base.IsValid():
        for child in list(gripper_base.GetChildren()):
            path = str(child.GetPath())
            if path == keep_path or path.startswith(keep_prefix):
                continue
            child.SetActive(False)
            link_deactivated += 1
    physics_root = stage.GetPrimAtPath(f"{ROBOT_PRIM}/Physics")
    if physics_root and physics_root.IsValid():
        for child in list(physics_root.GetChildren()):
            name = child.GetName()
            if name not in GRIPPER_JOINTS:
                continue
            child.SetActive(False)
            joint_deactivated += 1
    print(
        "Original adaptive gripper subtree hidden for simple gripper mode: "
        f"{hidden} imageable prims, {collision_disabled} collisions disabled, "
        f"{link_deactivated} top-level link children deactivated, "
        f"{joint_deactivated} physics joints deactivated."
    )


class SimpleParallelGripperVisual:
    def __init__(self, stage, articulation_root):
        self._stage = stage
        self._gripper_base_path = _get_gripper_base_path(articulation_root)
        self.root_path = f"{self._gripper_base_path}/{SIMPLE_GRIPPER_ROOT_NAME}"
        self._identity = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        self._root = UsdGeom.Xform.Define(stage, self.root_path).GetPrim()
        _reset_xform(self._root, (0.0, 0.0, 0.0), self._identity)
        _hide_original_gripper_subtree(stage, self._gripper_base_path, self.root_path)

        self._left_root = UsdGeom.Xform.Define(stage, f"{self.root_path}/left_finger").GetPrim()
        self._right_root = UsdGeom.Xform.Define(stage, f"{self.root_path}/right_finger").GetPrim()
        self._add_box(
            f"{self.root_path}/palm",
            translate=(0.0, 0.038, -0.004),
            scale=(0.070, 0.012, 0.022),
            color=(0.82, 0.84, 0.86),
        )
        self._add_box(
            f"{self.root_path}/left_finger/finger_body",
            translate=(0.0, 0.0, 0.0),
            scale=(SIMPLE_FINGER_THICKNESS_X, SIMPLE_FINGER_LENGTH_Y, SIMPLE_FINGER_THICKNESS_Z),
            color=(0.92, 0.93, 0.94),
        )
        self._add_box(
            f"{self.root_path}/right_finger/finger_body",
            translate=(0.0, 0.0, 0.0),
            scale=(SIMPLE_FINGER_THICKNESS_X, SIMPLE_FINGER_LENGTH_Y, SIMPLE_FINGER_THICKNESS_Z),
            color=(0.92, 0.93, 0.94),
        )
        self.update(GRIPPER_OPEN_RAD)
        print(
            "Added simple parallel gripper visual: "
            f"root={self.root_path}, open_half_gap={SIMPLE_FINGER_OPEN_HALF_GAP:.3f} m, "
            f"closed_half_gap={SIMPLE_FINGER_CLOSED_HALF_GAP:.3f} m"
        )

    def _add_box(self, path, translate, scale, color):
        cube = UsdGeom.Cube.Define(self._stage, path)
        cube.CreateSizeAttr(1.0)
        _reset_xform(cube.GetPrim(), translate, self._identity)
        cube.AddScaleOp().Set(Gf.Vec3f(*scale))
        _set_display_color(cube.GetPrim(), color)
        return cube.GetPrim()

    def update(self, gripper_value):
        gripper_value = _clip_joint_value(GRIPPER_CONTROLLER, gripper_value)
        span = max(GRIPPER_OPEN_RAD - GRIPPER_GRASP_CLOSE_RAD, 1e-6)
        close_fraction = np.clip((GRIPPER_OPEN_RAD - gripper_value) / span, 0.0, 1.0)
        half_gap = SIMPLE_FINGER_OPEN_HALF_GAP + close_fraction * (
            SIMPLE_FINGER_CLOSED_HALF_GAP - SIMPLE_FINGER_OPEN_HALF_GAP
        )
        finger_offset = float(half_gap + SIMPLE_FINGER_THICKNESS_X * 0.5)
        center_y = float(GRASP_POINT_IN_ATTACH_FRAME[1] - 0.008)
        center_z = float(GRASP_POINT_IN_ATTACH_FRAME[2])
        _reset_xform(self._left_root, (-finger_offset, center_y, center_z), self._identity)
        _reset_xform(self._right_root, (finger_offset, center_y, center_z), self._identity)


def _add_wrist_rgb_camera(stage, articulation_root):
    flange_path = _get_joint6_flange_path(articulation_root)
    flange_prim = stage.GetPrimAtPath(flange_path)
    if not flange_prim or not flange_prim.IsValid():
        raise RuntimeError(f"Cannot add wrist camera because joint6_flange is missing: {flange_path}")

    hand_camera_link_path = f"{flange_path}/hand_camera_link"
    camera_path = f"{hand_camera_link_path}/hand_rgb_camera"
    hand_link = UsdGeom.Xform.Define(stage, hand_camera_link_path).GetPrim()
    _reset_xform(hand_link, WRIST_CAMERA_TRANSLATION, _quat_from_rpy(*WRIST_CAMERA_RPY))

    body = UsdGeom.Cube.Define(stage, f"{hand_camera_link_path}/camera_body")
    body.CreateSizeAttr(1.0)
    _reset_xform(body.GetPrim(), (-0.012, 0.0, 0.0), Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    body.AddScaleOp().Set(Gf.Vec3f(0.030, 0.034, 0.028))
    _set_display_color(body.GetPrim(), (0.05, 0.05, 0.05))

    lens = UsdGeom.Cylinder.Define(stage, f"{hand_camera_link_path}/camera_lens")
    lens.CreateRadiusAttr(0.009)
    lens.CreateHeightAttr(0.016)
    _reset_xform(lens.GetPrim(), (0.008, 0.0, 0.0), _quat_from_rpy(0.0, 1.5708, 0.0))
    _set_display_color(lens.GetPrim(), (0.02, 0.02, 0.02))

    # Gazebo camera sensor aims along hand_camera_link +X with +Z as up.
    # USD cameras render along local -Z with local +Y up.
    camera_to_link = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    camera = UsdGeom.Camera.Define(stage, camera_path)
    _reset_xform(camera.GetPrim(), (0.0, 0.0, 0.0), _quat_from_rotation_matrix(camera_to_link))
    horizontal_aperture = 20.0
    width, height = WRIST_CAMERA_RESOLUTION
    focal_length = horizontal_aperture / (2.0 * np.tan(WRIST_CAMERA_FOV * 0.5))
    camera.CreateProjectionAttr("perspective")
    camera.CreateHorizontalApertureAttr(float(horizontal_aperture))
    camera.CreateVerticalApertureAttr(float(horizontal_aperture * height / width))
    camera.CreateFocalLengthAttr(float(focal_length))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 3.0))
    print(
        "Added wrist RGB camera: "
        f"{camera_path}, parent={flange_path}, xyz={WRIST_CAMERA_TRANSLATION}, "
        f"rpy={WRIST_CAMERA_RPY}, fov={WRIST_CAMERA_FOV}, resolution={WRIST_CAMERA_RESOLUTION}"
    )
    return camera_path


def _set_viewport_camera(camera_path):
    viewport = get_active_viewport()
    if viewport is None:
        print(f"Could not switch viewport camera; no active viewport. Camera path: {camera_path}")
        return
    viewport.camera_path = Sdf.Path(camera_path)
    print(f"Viewport camera set to: {camera_path}")


def _add_task_scene(stage):
    scene_root = TASK_SCENE_ROOT
    UsdGeom.Xform.Define(stage, scene_root)

    table = UsdGeom.Cube.Define(stage, TABLE_PRIM_PATH)
    table.CreateSizeAttr(1.0)
    _reset_xform(table.GetPrim(), TABLE_POSITION, Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    table.AddScaleOp().Set(Gf.Vec3f(*TABLE_SIZE))
    _set_display_color(table.GetPrim(), (0.45, 0.48, 0.50))
    _apply_collision(table.GetPrim())

    initial_positions = {
        "red_cube": CUBE_POSITION,
        "blue_cylinder": CYLINDER_POSITION,
    }
    for object_name in OBJECT_NAMES:
        _add_task_object(stage, object_name, initial_positions[object_name])

    print(
        "Added task scene: "
        f"table pos={TABLE_POSITION}, size={TABLE_SIZE}; "
        f"objects={_format_object_positions()}"
    )
    return scene_root


def _add_task_object(stage, object_name, position):
    spec = task_object_spec(object_name)
    prim_path = spec["prim_path"]
    orientation = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    if spec["shape_class"] == "box":
        prim = UsdGeom.Cube.Define(stage, prim_path)
        prim.CreateSizeAttr(1.0)
    elif spec["shape_class"] == "cylinder":
        prim = UsdGeom.Cylinder.Define(stage, prim_path)
        prim.CreateRadiusAttr(float(spec["radius_m"]))
        prim.CreateHeightAttr(float(spec["height_m"]))
    else:
        raise ValueError(f"Unsupported task object shape: {spec['shape_class']}")

    _reset_xform(prim.GetPrim(), position, orientation)
    if spec["shape_class"] == "box":
        prim.AddScaleOp().Set(Gf.Vec3f(*spec["size_m"]))
    _set_display_color(prim.GetPrim(), spec["color_rgb"])
    _apply_rigid_body(prim.GetPrim(), mass=0.03)
    if _uses_dynamic_contact_objects():
        _set_rigid_body_kinematic(prim.GetPrim(), False)
        _set_collision_enabled(prim.GetPrim(), True)
        _apply_contact_report(prim.GetPrim(), threshold=0.0)
    else:
        # Stable data-collection mode: the grasp target is kinematic and
        # non-colliding for the whole session. The object never pushes on the
        # fingers and these flags are not toggled while physics is running.
        # Flipping kinematicEnabled/collisionEnabled at runtime forces PhysX to
        # recreate the actor, and that resync is what made the wrist joints pop.
        _set_rigid_body_kinematic(prim.GetPrim(), True)
        _set_collision_enabled(prim.GetPrim(), False)
    OBJECT_TARGET_POSITIONS[object_name] = tuple(float(value) for value in position)
    OBJECT_TARGET_ORIENTATIONS[object_name] = orientation
    return prim.GetPrim()


def _format_object_positions():
    return ", ".join(
        f"{name}={OBJECT_TARGET_POSITIONS.get(name)}"
        for name in OBJECT_NAMES
    )


def _set_default_viewport_lighting():
    import carb
    import omni.usd

    settings = carb.settings.get_settings()
    settings.set("/exts/omni.kit.viewport.menubar.lighting/defaultRig", "Default")
    settings.set("/persistent/exts/omni.kit.viewport.menubar.lighting/autoLightRig/enabled", True)
    settings.set("/persistent/exts/omni.kit.viewport.menubar.lighting/autoLightRig/searchFromDefaultPrim", False)
    settings.set("/exts/omni.kit.viewport.menubar.lighting/preserveActiveRig", False)
    try:
        from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode

        result = _set_lighting_mode("Default", usd_context=omni.usd.get_context())
        print(f"Viewport lighting mode set to Default: {result}")
    except Exception as exc:
        print(f"Could not set viewport lighting mode to Default: {exc}")


def _set_world_pose(path, position, orientation=None):
    orientation = orientation if orientation is not None else Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    xform = XformPrim(path)
    xform.set_world_poses(
        positions=np.asarray([[*position]], dtype=np.float32),
        orientations=np.asarray(
            [[orientation.GetReal(), *orientation.GetImaginary()]],
            dtype=np.float32,
        ),
    )


def _quat_to_wxyz_array(orientation):
    orientation = orientation if orientation is not None else Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    return np.asarray(
        [[orientation.GetReal(), *orientation.GetImaginary()]],
        dtype=np.float32,
    )


def _get_object_rigid_prim(object_name):
    rigid_prim = OBJECT_RIGID_PRIMS.get(object_name)
    if rigid_prim is not None:
        return rigid_prim
    prim_path = OBJECT_STATE_PRIMS.get(object_name)
    if prim_path is None:
        return None
    try:
        rigid_prim = RigidPrim(prim_path)
    except Exception as exc:
        print(f"Could not create RigidPrim handle for {object_name} ({prim_path}): {exc}")
        return None
    OBJECT_RIGID_PRIMS[object_name] = rigid_prim
    return rigid_prim


def _dynamic_contact_labels():
    return ["left_finger", "right_finger", "table"]


def _classify_contact_counterpart(path):
    path = str(path)
    if "gripper_left" in path:
        return "left_finger"
    if "gripper_right" in path:
        return "right_finger"
    if path == TABLE_PRIM_PATH or path.endswith("/table") or "/table/" in path:
        return "table"
    return None


def _initialize_task_object_rigid_prims(enable_contact_tracking=False):
    OBJECT_RIGID_PRIMS.clear()
    OBJECT_CONTACT_FILTER_LABELS.clear()
    if args.no_task_scene:
        return
    contact_labels = _dynamic_contact_labels() if enable_contact_tracking and _uses_dynamic_contact_objects() else []
    if enable_contact_tracking and not contact_labels:
        print(
            "Contact-force diagnostics requested, but raw contact classification is only "
            "enabled in --dynamic-contact-grasp mode. Contact columns will stay NaN."
        )
    for object_name in OBJECT_NAMES:
        prim_path = OBJECT_STATE_PRIMS[object_name]
        try:
            if contact_labels:
                rigid_prim = RigidPrim(
                    prim_path,
                    max_contact_count=max(1, int(args.physics_diagnostics_max_contact_count)),
                )
                rigid_prim.set_enabled_contact_tracking(True)
                OBJECT_CONTACT_FILTER_LABELS[object_name] = list(contact_labels)
                print(
                    f"Raw contact diagnostics for {object_name}: labels={contact_labels}; "
                    "counterpart actor paths will be classified each physics step."
                )
            else:
                rigid_prim = RigidPrim(prim_path)
            OBJECT_RIGID_PRIMS[object_name] = rigid_prim
        except Exception as exc:
            print(f"Could not initialize dynamic handle for {object_name} ({prim_path}): {exc}")


def _set_dynamic_object_pose_and_zero_velocity(object_name, position, orientation=None):
    prim_path = OBJECT_STATE_PRIMS[object_name]
    rigid_prim = _get_object_rigid_prim(object_name)
    if rigid_prim is None:
        _set_world_pose(prim_path, position, orientation)
        return False
    try:
        rigid_prim.set_world_poses(
            positions=np.asarray([[*position]], dtype=np.float32),
            orientations=_quat_to_wxyz_array(orientation),
        )
        zeros = np.zeros((1, 3), dtype=np.float32)
        rigid_prim.set_velocities(linear_velocities=zeros, angular_velocities=zeros)
        return True
    except Exception as exc:
        print(f"Rigid dynamic pose reset failed for {object_name}; falling back to Xform pose: {exc}")
        _set_world_pose(prim_path, position, orientation)
        return False


def _set_object_target_position(stage, object_name, position, orientation=None):
    """Move a task object and remember where it belongs for the next reset."""
    prim_path = OBJECT_STATE_PRIMS.get(object_name)
    if prim_path is None:
        print(f"Unknown task object: {object_name}")
        return False
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"Task object prim is missing: {prim_path}")
        return False
    OBJECT_TARGET_POSITIONS[object_name] = tuple(float(value) for value in position)
    if orientation is not None:
        OBJECT_TARGET_ORIENTATIONS[object_name] = orientation
    orientation = OBJECT_TARGET_ORIENTATIONS.get(object_name)
    if _uses_dynamic_contact_objects():
        _set_rigid_body_kinematic(prim, False)
        _set_collision_enabled(prim, True)
        if not _set_dynamic_object_pose_and_zero_velocity(
            object_name,
            OBJECT_TARGET_POSITIONS[object_name],
            orientation,
        ):
            return False
    else:
        _set_world_pose(prim_path, OBJECT_TARGET_POSITIONS[object_name], orientation)
    return True


def _reset_task_scene(stage, robot, dof_names, home_positions, gripper_command_joints, grasp_assist, ros2_bridge):
    reset_ok = True
    if grasp_assist is not None:
        grasp_assist.reset()

    arm_home = {
        joint_name: float(home_positions[dof_names.index(joint_name)])
        for joint_name in ARM_JOINTS
    }
    gripper_home = _get_gripper_joint_values(dof_names, GRIPPER_OPEN_RAD)
    # Reset is not a demonstration segment. Put the robot directly into a known safe
    # state before placing dynamic objects so the arm cannot sweep through a newly
    # sampled cube while returning home.
    _set_joint_positions(robot, dof_names, {**arm_home, **gripper_home})
    _command_targets(robot, dof_names, arm_home)
    _command_gripper_target(robot, dof_names, GRIPPER_OPEN_RAD)
    if ros2_bridge is not None:
        ros2_bridge.set_desired_gripper_target(GRIPPER_OPEN_RAD)

    for object_name in OBJECT_NAMES:
        prim = stage.GetPrimAtPath(OBJECT_STATE_PRIMS[object_name])
        if prim and prim.IsValid():
            if prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeVisible()
            orientation = OBJECT_TARGET_ORIENTATIONS.get(object_name)
            if _uses_dynamic_contact_objects():
                _set_rigid_body_kinematic(prim, False)
                _set_collision_enabled(prim, True)
                reset_ok = _set_dynamic_object_pose_and_zero_velocity(
                    object_name,
                    OBJECT_TARGET_POSITIONS[object_name],
                    orientation,
                ) and reset_ok
            else:
                _set_collision_enabled(prim, False)
                _set_rigid_body_kinematic(prim, True)
                _set_world_pose(
                    OBJECT_STATE_PRIMS[object_name],
                    OBJECT_TARGET_POSITIONS[object_name],
                    orientation,
                )
    print(
        f"Reset task scene: objects={_format_object_positions()}, gripper={GRIPPER_OPEN_RAD:.3f}, "
        f"dynamic_reset_ok={reset_ok}"
    )
    return reset_ok


def _get_prim_world_position(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return (float("nan"), float("nan"), float("nan"))
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = transform.ExtractTranslation()
    return (float(translation[0]), float(translation[1]), float(translation[2]))


def _get_prim_world_transform(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _get_prim_parent_world_transform(prim):
    parent = prim.GetParent()
    if parent and parent.IsValid():
        return UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return Gf.Matrix4d(1.0)


def _set_rigid_body_kinematic(prim, enabled):
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return
    rigid_body = UsdPhysics.RigidBodyAPI(prim)
    attr = rigid_body.GetKinematicEnabledAttr()
    if not attr:
        attr = rigid_body.CreateKinematicEnabledAttr()
    attr.Set(bool(enabled))


def _set_rigid_body_enabled(prim, enabled):
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return
    rigid_body = UsdPhysics.RigidBodyAPI(prim)
    attr = rigid_body.GetRigidBodyEnabledAttr()
    if not attr:
        attr = rigid_body.CreateRigidBodyEnabledAttr()
    attr.Set(bool(enabled))


def _set_collision_enabled(prim, enabled):
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        return
    collision = UsdPhysics.CollisionAPI(prim)
    attr = collision.GetCollisionEnabledAttr()
    if not attr:
        attr = collision.CreateCollisionEnabledAttr()
    attr.Set(bool(enabled))


def _command_single_target(robot, dof_index, value):
    target = np.asarray([[value]], dtype=np.float32)
    robot.set_dof_position_targets(target, dof_indices=[dof_index])


def _command_targets(robot, dof_names, joint_values):
    joint_indices = []
    targets = []
    for joint_name, value in joint_values.items():
        joint_indices.append(dof_names.index(joint_name))
        targets.append(float(value))
    if not joint_indices:
        return
    robot.set_dof_position_targets(np.asarray([targets], dtype=np.float32), dof_indices=joint_indices)


def _set_joint_positions(robot, dof_names, joint_values):
    joint_indices = []
    positions = []
    for joint_name, value in joint_values.items():
        joint_indices.append(dof_names.index(joint_name))
        positions.append(float(value))
    if not joint_indices:
        return
    robot.set_dof_positions(np.asarray([positions], dtype=np.float32), dof_indices=joint_indices)
    robot.set_dof_velocities(np.zeros((1, len(joint_indices)), dtype=np.float32), dof_indices=joint_indices)


def _clip_joint_value(joint_name, value):
    lower, upper = JOINT_LIMITS[joint_name]
    return min(max(float(value), lower), upper)


def _get_gripper_command_joints(dof_names):
    if _uses_virtual_gripper():
        return []
    if args.drive_all_gripper_joints:
        return [joint_name for joint_name in GRIPPER_JOINTS if joint_name in dof_names]
    if GRIPPER_CONTROLLER in dof_names:
        return [GRIPPER_CONTROLLER]
    return []


def _get_gripper_joint_values(dof_names, controller_value):
    controller_value = _clip_joint_value(GRIPPER_CONTROLLER, controller_value)
    joint_values = {}
    for joint_name in _get_gripper_command_joints(dof_names):
        sign = GRIPPER_JOINT_SIGNS[joint_name] if args.drive_all_gripper_joints else 1.0
        if joint_name in dof_names:
            joint_values[joint_name] = _clip_joint_value(joint_name, sign * controller_value)
    return joint_values


def _command_gripper_target(robot, dof_names, controller_value):
    if _uses_virtual_gripper():
        return
    joint_values = _get_gripper_joint_values(dof_names, controller_value)
    if args.kinematic_gripper:
        _set_joint_positions(robot, dof_names, joint_values)
    else:
        _command_targets(robot, dof_names, joint_values)


def _gripper_ramp_step(elapsed_seconds):
    """Gripper target increment for one update, in radians.

    The ramp rate is rad/s so the commanded gripper speed stays the same whether
    the app loop runs at 5 Hz or 60 Hz. Long stalls are clamped so a hitch cannot
    turn into a single large jump in the position target.
    """
    dt = min(max(float(elapsed_seconds), 0.0), MAX_GRIPPER_RAMP_DT)
    return max(0.0, float(args.gripper_ramp_rate) * dt)


def _get_control_state(robot, dof_names, gripper_override=None):
    positions = _as_numpy(robot.get_dof_positions())[0]
    values = [positions[dof_names.index(name)] for name in ARM_JOINTS]
    if gripper_override is None:
        values.append(positions[dof_names.index(GRIPPER_CONTROLLER)])
    else:
        values.append(float(gripper_override))
    return np.asarray(values, dtype=np.float64)


def _clip_control_targets(values):
    clipped = []
    for joint_name, value in zip(CONTROL_JOINTS, values):
        clipped.append(_clip_joint_value(joint_name, value))
    return np.asarray(clipped, dtype=np.float32)


def _command_control_targets(robot, dof_names, values):
    targets = _clip_control_targets(values)
    arm_values = {joint_name: targets[index] for index, joint_name in enumerate(ARM_JOINTS)}
    _command_targets(robot, dof_names, arm_values)
    _command_gripper_target(robot, dof_names, targets[-1])
    return targets


def _command_arm_targets(robot, dof_names, values):
    targets = _clip_control_targets(values)
    arm_values = {joint_name: targets[index] for index, joint_name in enumerate(ARM_JOINTS)}
    _command_targets(robot, dof_names, arm_values)
    return targets


def _create_rgb_annotator(camera_path):
    render_product = rep.create.render_product(camera_path, WRIST_CAMERA_RESOLUTION, name="ammr_hand_rgb")
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)
    return render_product, annotator


class LoopProfiler:
    """Per-stage timing for the app loop.

    State and images are published from inside the loop, so its rate caps both. This
    exists because the loop was assumed to be the reason a rollout saw its inputs
    change only 4.8 times a second; the timing showed the loop running at 20-45 Hz the
    whole time and sent the search somewhere more useful. Keep it for the next such
    question -- measuring which stage costs what is cheap, guessing is not.
    """

    STAGES = ("ui", "ros2", "physics", "grasp", "app")
    # Sub-stages of ros2, reported separately: the readback and the publish cost very
    # different amounts and only one of them is worth cutting.
    SUBSTAGES = ("spin", "state", "image")

    def __init__(self, period):
        self._period = float(period)
        self._totals = dict.fromkeys(self.STAGES + self.SUBSTAGES, 0.0)
        self._frames = 0
        self._window_start = time.monotonic()

    def report_due(self):
        return time.monotonic() - self._window_start >= self._period

    def add(self, stage, seconds):
        self._totals[stage] += seconds

    def frame(self):
        self._frames += 1

    def report(self):
        now = time.monotonic()
        elapsed = now - self._window_start
        if self._frames == 0 or elapsed <= 0.0:
            return
        rate = self._frames / elapsed
        parts = " ".join(
            f"{stage} {self._totals[stage] / self._frames * 1000.0:6.1f}ms"
            for stage in self.STAGES
        )
        accounted = sum(self._totals[stage] for stage in self.STAGES) / self._frames * 1000.0
        sub = " ".join(
            f"{stage} {self._totals[stage] / self._frames * 1000.0:5.1f}ms"
            for stage in self.SUBSTAGES
        )
        print(
            f"[loop] {rate:5.2f} Hz  frame {elapsed / self._frames * 1000.0:6.1f}ms  "
            f"({parts}  accounted {accounted:6.1f}ms)  [ros2: {sub}]",
            flush=True,
        )
        self._totals = dict.fromkeys(self.STAGES + self.SUBSTAGES, 0.0)
        self._frames = 0
        self._window_start = now


class PhysicsDiagnosticsLogger:
    """Per-physics-step CSV logger for dynamic contact debugging."""

    def __init__(self, path, robot, dof_names, physics_dt, object_names):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._robot = robot
        self._dof_names = list(dof_names)
        self._physics_dt = float(physics_dt)
        self._object_names = list(object_names)
        self._step = 0
        self._last_wall = time.monotonic()
        self._stride = max(1, int(args.physics_diagnostics_stride))
        self._warned_keys = set()
        self._file = self._path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames())
        self._writer.writeheader()
        print(
            "Physics diagnostics enabled: "
            f"path={self._path}, stride={self._stride}, physics_dt={self._physics_dt:.6f}s"
        )

    def _fieldnames(self):
        fields = [
            "step",
            "sim_time_s",
            "wall_dt_s",
            "physics_dt_s",
            "max_arm_velocity_rad_s",
            "max_control_target_error_rad",
            "max_gripper_mimic_error_rad",
        ]
        for name in self._dof_names:
            fields.extend([f"dof.{name}.pos", f"dof.{name}.vel", f"dof.{name}.target"])
        for follower in GRIPPER_FOLLOWER_JOINTS:
            if follower in self._dof_names:
                fields.append(f"mimic_error.{follower}")
        for object_name in self._object_names:
            fields.extend(
                [
                    f"object.{object_name}.x",
                    f"object.{object_name}.y",
                    f"object.{object_name}.z",
                    f"object.{object_name}.linvel_x",
                    f"object.{object_name}.linvel_y",
                    f"object.{object_name}.linvel_z",
                    f"object.{object_name}.angvel_x",
                    f"object.{object_name}.angvel_y",
                    f"object.{object_name}.angvel_z",
                    f"object.{object_name}.speed",
                    f"object.{object_name}.contact_force_norm",
                ]
            )
            for label in OBJECT_CONTACT_FILTER_LABELS.get(object_name, []):
                fields.extend(
                    [
                        f"object.{object_name}.contact.{label}.actor_paths",
                        f"object.{object_name}.contact.{label}.force_x",
                        f"object.{object_name}.contact.{label}.force_y",
                        f"object.{object_name}.contact.{label}.force_z",
                        f"object.{object_name}.contact.{label}.force_norm",
                        f"object.{object_name}.contact.{label}.force_mag_sum",
                        f"object.{object_name}.contact.{label}.force_mag_max",
                        f"object.{object_name}.contact.{label}.count",
                        f"object.{object_name}.contact.{label}.point_mean_x",
                        f"object.{object_name}.contact.{label}.point_mean_y",
                        f"object.{object_name}.contact.{label}.point_mean_z",
                        f"object.{object_name}.contact.{label}.point_min_z",
                        f"object.{object_name}.contact.{label}.point_max_z",
                        f"object.{object_name}.contact.{label}.separation_min",
                        f"object.{object_name}.contact.{label}.separation_max",
                    ]
                )
        return fields

    def close(self):
        self._file.flush()
        self._file.close()
        print(f"Physics diagnostics closed: {self._path}")

    def record(self):
        self._step += 1
        if self._step % self._stride != 0:
            return
        now = time.monotonic()
        row = {
            "step": self._step,
            "sim_time_s": self._step * self._physics_dt,
            "wall_dt_s": now - self._last_wall,
            "physics_dt_s": self._physics_dt,
        }
        self._last_wall = now
        positions = self._safe_array(lambda: self._robot.get_dof_positions())
        velocities = self._safe_array(lambda: self._robot.get_dof_velocities())
        targets = self._safe_array(lambda: self._robot.get_dof_position_targets())
        if positions is not None and positions.ndim >= 2:
            positions = positions[0]
        if velocities is not None and velocities.ndim >= 2:
            velocities = velocities[0]
        if targets is not None and targets.ndim >= 2:
            targets = targets[0]

        arm_velocities = []
        control_target_errors = []
        for name in self._dof_names:
            index = self._dof_names.index(name)
            position = self._value_at(positions, index)
            velocity = self._value_at(velocities, index)
            target = self._value_at(targets, index)
            row[f"dof.{name}.pos"] = position
            row[f"dof.{name}.vel"] = velocity
            row[f"dof.{name}.target"] = target
            if name in ARM_JOINTS and np.isfinite(velocity):
                arm_velocities.append(abs(velocity))
            if name in CONTROL_JOINTS and np.isfinite(position) and np.isfinite(target):
                control_target_errors.append(abs(target - position))
        row["max_arm_velocity_rad_s"] = max(arm_velocities) if arm_velocities else float("nan")
        row["max_control_target_error_rad"] = max(control_target_errors) if control_target_errors else float("nan")

        mimic_errors = []
        controller_value = self._value_at(positions, self._dof_names.index(GRIPPER_CONTROLLER)) if GRIPPER_CONTROLLER in self._dof_names else float("nan")
        for follower in GRIPPER_FOLLOWER_JOINTS:
            if follower not in self._dof_names:
                continue
            follower_value = self._value_at(positions, self._dof_names.index(follower))
            expected = GRIPPER_JOINT_SIGNS[follower] * controller_value
            error = follower_value - expected if np.isfinite(follower_value) and np.isfinite(expected) else float("nan")
            row[f"mimic_error.{follower}"] = error
            if np.isfinite(error):
                mimic_errors.append(abs(error))
        row["max_gripper_mimic_error_rad"] = max(mimic_errors) if mimic_errors else float("nan")

        for object_name in self._object_names:
            self._add_object_row(row, object_name)
        self._writer.writerow(row)

    def _add_object_row(self, row, object_name):
        position = np.full(3, np.nan, dtype=np.float64)
        orientation = None
        linear_velocity = np.full(3, np.nan, dtype=np.float64)
        angular_velocity = np.full(3, np.nan, dtype=np.float64)
        rigid_prim = OBJECT_RIGID_PRIMS.get(object_name)
        if rigid_prim is not None:
            try:
                positions, orientations = rigid_prim.get_world_poses()
                position = _as_numpy(positions)[0]
                orientation = _as_numpy(orientations)[0]
            except Exception as exc:
                self._warn_once(
                    f"{object_name}:pose",
                    f"Physics diagnostics could not read {object_name} pose from RigidPrim: {exc}",
                )
            try:
                linear, angular = rigid_prim.get_velocities()
                linear_velocity = _as_numpy(linear)[0]
                angular_velocity = _as_numpy(angular)[0]
            except Exception as exc:
                self._warn_once(
                    f"{object_name}:velocity",
                    f"Physics diagnostics could not read {object_name} velocities from RigidPrim: {exc}",
                )
        else:
            self._warn_once(
                f"{object_name}:no_rigid_prim",
                f"Physics diagnostics has no RigidPrim handle for {object_name}; pose falls back to USD xform.",
            )
            position = _get_prim_world_position(stage_utils.get_current_stage(), OBJECT_STATE_PRIMS[object_name])
        row[f"object.{object_name}.x"] = position[0]
        row[f"object.{object_name}.y"] = position[1]
        row[f"object.{object_name}.z"] = position[2]
        contact_force_norm = float("nan")
        if rigid_prim is not None:
            contact_force_norm = self._add_contact_rows(row, object_name, rigid_prim)
        row[f"object.{object_name}.linvel_x"] = float(linear_velocity[0])
        row[f"object.{object_name}.linvel_y"] = float(linear_velocity[1])
        row[f"object.{object_name}.linvel_z"] = float(linear_velocity[2])
        row[f"object.{object_name}.angvel_x"] = float(angular_velocity[0])
        row[f"object.{object_name}.angvel_y"] = float(angular_velocity[1])
        row[f"object.{object_name}.angvel_z"] = float(angular_velocity[2])
        row[f"object.{object_name}.speed"] = float(np.linalg.norm(linear_velocity)) if np.all(np.isfinite(linear_velocity)) else float("nan")
        row[f"object.{object_name}.contact_force_norm"] = contact_force_norm

    def _add_contact_rows(self, row, object_name, rigid_prim):
        labels = OBJECT_CONTACT_FILTER_LABELS.get(object_name, [])
        net_force_norm = float("nan")
        if labels:
            for label in labels:
                self._set_contact_pair_row(row, object_name, label, np.zeros(3), 0)
            try:
                forces, points, normals, separations, counts, starts, other_actor_ids = rigid_prim.get_raw_contact_data(
                    dt=self._physics_dt
                )
                raw_forces = _as_numpy(forces)
                points = _as_numpy(points).reshape((-1, 3))
                normals = _as_numpy(normals).reshape((-1, 3))
                separations = _as_numpy(separations).reshape((-1,))
                if raw_forces.ndim >= 2 and raw_forces.shape[-1] == 3:
                    force_vectors = raw_forces.reshape((-1, 3))
                    force_magnitudes = np.linalg.norm(force_vectors, axis=1)
                else:
                    force_magnitudes = raw_forces.reshape((-1,))
                    vector_count = min(len(force_magnitudes), len(normals))
                    force_vectors = np.zeros((len(force_magnitudes), 3), dtype=np.float64)
                    if vector_count:
                        force_vectors[:vector_count] = force_magnitudes[:vector_count, None] * normals[:vector_count]
                counts = _as_numpy(counts)
                starts = _as_numpy(starts)
                other_paths = rigid_prim.get_actor_paths_from_ids(other_actor_ids)
                pair_stats = {
                    label: {
                        "force": np.zeros(3, dtype=np.float64),
                        "force_mag_sum": 0.0,
                        "force_mag_max": 0.0,
                        "count": 0,
                        "points": [],
                        "separations": [],
                        "actor_paths": set(),
                    }
                    for label in labels
                }
                if counts.size and starts.size:
                    count = int(counts.reshape((-1,))[0])
                    start = int(starts.reshape((-1,))[0])
                    for raw_index in range(start, start + count):
                        if (
                            raw_index >= len(force_vectors)
                            or raw_index >= len(force_magnitudes)
                            or raw_index >= len(other_paths)
                        ):
                            continue
                        other_path = other_paths[raw_index]
                        label = _classify_contact_counterpart(other_path)
                        if label is None:
                            if other_path:
                                self._warn_once(
                                    f"{object_name}:unclassified_contact:{other_path}",
                                    f"Unclassified contact counterpart for {object_name}: {other_path}",
                                )
                            continue
                        stats = pair_stats[label]
                        force_vector = force_vectors[raw_index]
                        force_magnitude = abs(float(force_magnitudes[raw_index]))
                        stats["force"] += force_vector
                        stats["force_mag_sum"] += force_magnitude
                        stats["force_mag_max"] = max(stats["force_mag_max"], force_magnitude)
                        stats["count"] += 1
                        stats["actor_paths"].add(str(other_path))
                        if raw_index < len(points):
                            stats["points"].append(points[raw_index])
                        if raw_index < len(separations):
                            stats["separations"].append(float(separations[raw_index]))
                    net_force = np.zeros(3, dtype=np.float64)
                    for label in labels:
                        stats = pair_stats[label]
                        force = stats["force"]
                        net_force += force
                        self._set_contact_pair_row(
                            row,
                            object_name,
                            label,
                            force,
                            stats["count"],
                            actor_paths=stats["actor_paths"],
                            force_mag_sum=stats["force_mag_sum"],
                            force_mag_max=stats["force_mag_max"],
                            points=stats["points"],
                            separations=stats["separations"],
                        )
                    net_force_norm = float(np.linalg.norm(net_force))
            except Exception as exc:
                self._warn_once(
                    f"{object_name}:raw_contact",
                    f"Physics diagnostics could not read raw contacts for {object_name}: {exc}",
                )
        else:
            try:
                forces = _as_numpy(rigid_prim.get_net_contact_forces(dt=self._physics_dt))
                if forces.size:
                    net_force_norm = float(np.linalg.norm(forces[0]))
            except Exception as exc:
                self._warn_once(
                    f"{object_name}:net_contact",
                    f"Physics diagnostics could not read net contact force for {object_name}: {exc}",
                )
        return net_force_norm

    @staticmethod
    def _set_contact_pair_row(
        row,
        object_name,
        label,
        force,
        count,
        actor_paths=None,
        force_mag_sum=0.0,
        force_mag_max=0.0,
        points=None,
        separations=None,
    ):
        points = [] if points is None else list(points)
        separations = [] if separations is None else list(separations)
        actor_paths = [] if actor_paths is None else sorted(str(path) for path in actor_paths if path)
        if points:
            points_array = np.asarray(points, dtype=np.float64).reshape((-1, 3))
            point_mean = np.nanmean(points_array, axis=0)
            point_min_z = float(np.nanmin(points_array[:, 2]))
            point_max_z = float(np.nanmax(points_array[:, 2]))
        else:
            point_mean = np.full(3, np.nan, dtype=np.float64)
            point_min_z = float("nan")
            point_max_z = float("nan")
        if separations:
            separations_array = np.asarray(separations, dtype=np.float64).reshape((-1,))
            separation_min = float(np.nanmin(separations_array))
            separation_max = float(np.nanmax(separations_array))
        else:
            separation_min = float("nan")
            separation_max = float("nan")
        row[f"object.{object_name}.contact.{label}.actor_paths"] = ";".join(actor_paths)
        row[f"object.{object_name}.contact.{label}.force_x"] = float(force[0])
        row[f"object.{object_name}.contact.{label}.force_y"] = float(force[1])
        row[f"object.{object_name}.contact.{label}.force_z"] = float(force[2])
        row[f"object.{object_name}.contact.{label}.force_norm"] = (
            float(np.linalg.norm(force)) if np.all(np.isfinite(force)) else float("nan")
        )
        row[f"object.{object_name}.contact.{label}.force_mag_sum"] = float(force_mag_sum)
        row[f"object.{object_name}.contact.{label}.force_mag_max"] = float(force_mag_max)
        row[f"object.{object_name}.contact.{label}.count"] = count
        row[f"object.{object_name}.contact.{label}.point_mean_x"] = float(point_mean[0])
        row[f"object.{object_name}.contact.{label}.point_mean_y"] = float(point_mean[1])
        row[f"object.{object_name}.contact.{label}.point_mean_z"] = float(point_mean[2])
        row[f"object.{object_name}.contact.{label}.point_min_z"] = point_min_z
        row[f"object.{object_name}.contact.{label}.point_max_z"] = point_max_z
        row[f"object.{object_name}.contact.{label}.separation_min"] = separation_min
        row[f"object.{object_name}.contact.{label}.separation_max"] = separation_max

    def _warn_once(self, key, message):
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        print(message, flush=True)

    @staticmethod
    def _safe_array(getter):
        try:
            return _as_numpy(getter())
        except Exception:
            return None

    @staticmethod
    def _value_at(values, index):
        if values is None or index >= len(values):
            return float("nan")
        return float(values[index])


class MyCobotRos2Bridge:
    def __init__(self, robot, dof_names, camera_path=None, simple_gripper_visual=None, virtual_gripper=False):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from std_msgs.msg import Empty, Float64MultiArray

        if not rclpy.ok():
            rclpy.init(args=None)

        class BridgeNode(Node):
            pass

        self._rclpy = rclpy
        self._JointState = JointState
        self._Image = Image
        self._CameraInfo = CameraInfo
        self._Empty = Empty
        self._Float64MultiArray = Float64MultiArray
        self._robot = robot
        self._dof_names = dof_names
        self._camera_path = camera_path
        self._reset_callback = None
        self._object_placed_callback = None
        self._render_product = None
        self._rgb_annotator = None
        self._last_state_publish = 0.0
        self._last_image_publish = 0.0
        self._state_period = 1.0 / max(args.ros2_state_hz, 1e-6)
        self._image_period = 1.0 / max(args.ros2_image_hz, 1e-6)
        self._object_position_provider = None
        self._attached_getter = None
        self._attached_gripper_hold = None
        self._simple_gripper_visual = simple_gripper_visual
        self._virtual_gripper_enabled = bool(virtual_gripper or self._simple_gripper_visual is not None)
        self._virtual_gripper_state = GRIPPER_OPEN_RAD if self._virtual_gripper_enabled else None
        initial_gripper = (
            float(self._virtual_gripper_state)
            if self._virtual_gripper_enabled
            else float(_get_control_state(robot, dof_names)[-1])
        )
        self._desired_gripper_target = initial_gripper
        self._last_gripper_command = None if self._virtual_gripper_enabled else initial_gripper
        if self._simple_gripper_visual is not None:
            self._simple_gripper_visual.update(initial_gripper)
        self._node = BridgeNode(args.ros2_node_name)
        self._joint_state_pub = self._node.create_publisher(JointState, ROS2_JOINT_STATES_TOPIC, 10)
        self._robot_state_pub = self._node.create_publisher(Float64MultiArray, ROS2_ROBOT_STATE_TOPIC, 10)
        self._object_state_pub = self._node.create_publisher(Float64MultiArray, ROS2_OBJECT_STATES_TOPIC, 10)
        self._grasp_event_pub = self._node.create_publisher(Float64MultiArray, ROS2_GRASP_EVENTS_TOPIC, 10)
        self._image_pub = self._node.create_publisher(Image, ROS2_IMAGE_TOPIC, 10)
        self._camera_info_pub = self._node.create_publisher(CameraInfo, ROS2_CAMERA_INFO_TOPIC, 10)
        # Depth 1: a joint target is absolute state, not an event. Publishers run
        # faster than this app loop, so a deeper queue just makes the sim execute
        # an ever-growing backlog of stale targets. Keeping only the newest one
        # removes that lag entirely.
        self._target_sub = self._node.create_subscription(
            Float64MultiArray,
            ROS2_JOINT_TARGET_TOPIC,
            self._on_joint_targets,
            rclpy.qos.QoSProfile(
                depth=1,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            ),
        )
        self._reset_sub = self._node.create_subscription(
            Empty,
            ROS2_RESET_TASK_SCENE_TOPIC,
            self._on_reset_task_scene,
            10,
        )
        self._set_object_pose_sub = self._node.create_subscription(
            Float64MultiArray,
            ROS2_SET_OBJECT_POSE_TOPIC,
            self._on_set_object_pose,
            10,
        )
        if camera_path:
            self._render_product, self._rgb_annotator = _create_rgb_annotator(camera_path)
        print("ROS2 bridge enabled:")
        print(f"  subscribe {ROS2_JOINT_TARGET_TOPIC} std_msgs/Float64MultiArray length=7")
        print(f"  subscribe {ROS2_RESET_TASK_SCENE_TOPIC} std_msgs/Empty")
        print(f"  subscribe {ROS2_SET_OBJECT_POSE_TOPIC} std_msgs/Float64MultiArray [index, x, y, z]")
        print(f"  publish   {ROS2_JOINT_STATES_TOPIC} sensor_msgs/JointState")
        print(f"  publish   {ROS2_ROBOT_STATE_TOPIC} std_msgs/Float64MultiArray")
        print(f"  publish   {ROS2_OBJECT_STATES_TOPIC} std_msgs/Float64MultiArray {OBJECT_STATE_NAMES}")
        print(f"  publish   {ROS2_GRASP_EVENTS_TOPIC} std_msgs/Float64MultiArray {GRASP_EVENT_FIELD_NAMES}")
        if camera_path:
            print(f"  publish   {ROS2_IMAGE_TOPIC} sensor_msgs/Image")
            print(f"  publish   {ROS2_CAMERA_INFO_TOPIC} sensor_msgs/CameraInfo")

    def _on_joint_targets(self, msg):
        if len(msg.data) < len(CONTROL_JOINTS):
            self._node.get_logger().warn(
                f"Expected {len(CONTROL_JOINTS)} joint targets, got {len(msg.data)}"
            )
            return
        targets = _command_arm_targets(self._robot, self._dof_names, msg.data[: len(CONTROL_JOINTS)])
        self._desired_gripper_target = float(targets[-1])
        self._node.get_logger().debug(
            f"Applied arm targets and queued gripper target: {targets.tolist()}"
        )

    def _on_set_object_pose(self, msg):
        """[object_index, x, y, z, (qw, qx, qy, qz)] -- see ROS2_SET_OBJECT_POSE_TOPIC.

        Orientation is accepted for forward compatibility but not applied yet: the task
        objects are axis-aligned and the grasp family assumes that.
        """
        if len(msg.data) < 4:
            self._node.get_logger().warn(
                f"set_object_pose needs at least [index, x, y, z], got {len(msg.data)} values"
            )
            return
        index = int(round(float(msg.data[0])))
        names = list(OBJECT_NAMES)
        if not 0 <= index < len(names):
            self._node.get_logger().warn(f"set_object_pose index {index} out of range {names}")
            return
        name = names[index]
        position = [float(value) for value in msg.data[1:4]]
        orientation = None
        if len(msg.data) >= 8:
            qw, qx, qy, qz = [float(value) for value in msg.data[4:8]]
            orientation = Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz))
        stage = stage_utils.get_current_stage()
        if self._object_placed_callback is not None:
            self._object_placed_callback()
        if _set_object_target_position(stage, name, position, orientation):
            self._node.get_logger().info(f"set_object_pose {name} -> {position}")
        else:
            self._node.get_logger().error(f"set_object_pose failed for {name} -> {position}")

    def set_object_placed_callback(self, callback):
        self._object_placed_callback = callback

    def _on_reset_task_scene(self, _msg):
        if self._reset_callback is None:
            self._node.get_logger().warn("reset requested but no reset callback is configured")
            return
        ok = self._reset_callback()
        if ok is False:
            self._node.get_logger().error("reset task scene failed")
        else:
            self._node.get_logger().info("reset task scene")

    def _stamp(self):
        return self._node.get_clock().now().to_msg()

    def get_desired_gripper_target(self):
        return float(self._desired_gripper_target)

    def get_gripper_state(self):
        if self._virtual_gripper_enabled:
            return float(self._virtual_gripper_state)
        return float(_get_control_state(self._robot, self._dof_names)[-1])

    def set_object_position_provider(self, provider):
        self._object_position_provider = provider

    def set_attached_gripper_hold(self, attached_getter, hold_target):
        self._attached_getter = attached_getter
        self._attached_gripper_hold = float(hold_target)

    def set_reset_callback(self, callback):
        self._reset_callback = callback

    def set_desired_gripper_target(self, target):
        self._desired_gripper_target = float(target)
        if self._virtual_gripper_enabled:
            self._last_gripper_command = None
            self._update_gripper_command()
        else:
            self._last_gripper_command = float(target)

    def publish_grasp_event(self, values):
        msg = self._Float64MultiArray()
        msg.data = [float(value) for value in values]
        self._grasp_event_pub.publish(msg)

    def set_profiler(self, profiler):
        self._profiler = profiler

    def update(self):
        profiler = getattr(self, "_profiler", None)
        mark = time.monotonic()
        # spin_once dispatches at most one callback, so drain the queues instead
        # of falling one message further behind on every frame.
        for _ in range(MAX_SPINS_PER_FRAME):
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
        self._update_gripper_command()
        now = time.monotonic()
        if profiler is not None:
            profiler.add("spin", now - mark)
            mark = now
        if now - self._last_state_publish >= self._state_period:
            self._publish_joint_state()
            self._publish_object_state()
            self._last_state_publish = now
            if profiler is not None:
                mark2 = time.monotonic()
                profiler.add("state", mark2 - mark)
                mark = mark2
        if self._rgb_annotator is not None and now - self._last_image_publish >= self._image_period:
            self._publish_image()
            self._last_image_publish = now
            if profiler is not None:
                profiler.add("image", time.monotonic() - mark)

    def _update_gripper_command(self):
        """Apply the received gripper target verbatim.

        The bridge used to ramp toward the target here, which meant the value on
        /ammr/arm/joint_targets was not the value being applied. A recorded episode
        then showed the gripper action as a two-value step function while the arm
        channels were dense, so the action stream had inconsistent semantics across
        its own dimensions. Shaping now belongs to whoever publishes the command
        (scripted_pick_demo.py --gripper-speed), so action == applied command.

        This removes only the software ramp. The drive still has its own response --
        stiffness/damping/max-velocity give roughly a 2.3 s close -- and that is the
        physical gripper dynamic we want to keep modelling.
        """
        target = float(self._desired_gripper_target)
        if self._attached_getter is not None and self._attached_gripper_hold is not None:
            try:
                attached = bool(self._attached_getter())
            except Exception:
                attached = False
            if attached and target < self._attached_gripper_hold:
                target = self._attached_gripper_hold
        if self._last_gripper_command is not None and target == float(self._last_gripper_command):
            return
        if self._virtual_gripper_enabled:
            target = _clip_joint_value(GRIPPER_CONTROLLER, target)
            self._virtual_gripper_state = target
            if self._simple_gripper_visual is not None:
                self._simple_gripper_visual.update(target)
        else:
            _command_gripper_target(self._robot, self._dof_names, target)
        self._last_gripper_command = target

    def _publish_joint_state(self):
        gripper_override = self._virtual_gripper_state if self._virtual_gripper_enabled else None
        state = _get_control_state(self._robot, self._dof_names, gripper_override=gripper_override)
        velocities = _as_numpy(self._robot.get_dof_velocities())[0]
        msg = self._JointState()
        msg.header.stamp = self._stamp()
        msg.name = list(CONTROL_JOINTS)
        msg.position = state.tolist()
        msg.velocity = [float(velocities[self._dof_names.index(name)]) for name in ARM_JOINTS]
        msg.velocity.append(0.0 if self._virtual_gripper_enabled else float(velocities[self._dof_names.index(GRIPPER_CONTROLLER)]))
        msg.effort = []
        self._joint_state_pub.publish(msg)

        state_msg = self._Float64MultiArray()
        state_msg.data = state.tolist()
        self._robot_state_pub.publish(state_msg)

    def _publish_object_state(self):
        stage = stage_utils.get_current_stage()
        msg = self._Float64MultiArray()
        msg.data = [
            value
            for object_name in OBJECT_NAMES
            for value in self._get_object_position(stage, object_name)
        ]
        self._object_state_pub.publish(msg)

    def _get_object_position(self, stage, object_name):
        if self._object_position_provider is not None:
            position = self._object_position_provider(object_name)
            if position is not None:
                return position
        return _get_prim_world_position(stage, OBJECT_STATE_PRIMS[object_name])

    def _publish_image(self):
        rgb = self._rgb_annotator.get_data()
        if rgb is None:
            return
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return
        rgb = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
        height, width, _ = rgb.shape
        stamp = self._stamp()

        msg = self._Image()
        msg.header.stamp = stamp
        msg.header.frame_id = WRIST_CAMERA_FRAME_ID
        msg.height = int(height)
        msg.width = int(width)
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(width * 3)
        msg.data = rgb.tobytes()
        self._image_pub.publish(msg)

        info = self._CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = WRIST_CAMERA_FRAME_ID
        info.height = int(height)
        info.width = int(width)
        fx = width / (2.0 * np.tan(WRIST_CAMERA_FOV * 0.5))
        fy = fx
        cx = width * 0.5
        cy = height * 0.5
        info.k = [float(fx), 0.0, float(cx), 0.0, float(fy), float(cy), 0.0, 0.0, 1.0]
        info.p = [float(fx), 0.0, float(cx), 0.0, 0.0, float(fy), float(cy), 0.0, 0.0, 0.0, 1.0, 0.0]
        info.d = []
        info.distortion_model = "plumb_bob"
        self._camera_info_pub.publish(info)

    def shutdown(self):
        if self._rgb_annotator is not None:
            self._rgb_annotator.detach()
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


class KinematicGraspAssist:
    def __init__(
        self,
        stage,
        robot,
        dof_names,
        attach_frame_path,
        object_name="red_cube",
        attach_threshold=GRIPPER_GRASP_ATTACH_RAD,
        engage_threshold=GRIPPER_GRASP_ENGAGE_RAD,
        release_threshold=0.02,
        attach_distance=0.12,
        max_lateral_offset=0.0,
        max_tool_axis_offset=0.0,
        still_time=0.15,
        still_tolerance=0.003,
        gripper_state_getter=None,
        gripper_target_getter=None,
        event_callback=None,
    ):
        self._stage = stage
        self._robot = robot
        self._dof_names = dof_names
        self._attach_frame_path = attach_frame_path
        self._object_name = object_name
        self._object_path = OBJECT_STATE_PRIMS[object_name]
        self._attach_threshold = float(attach_threshold)
        self._engage_threshold = float(engage_threshold)
        self._release_threshold = float(release_threshold)
        self._attach_distance = float(attach_distance)
        self._max_lateral_offset = max(0.0, float(max_lateral_offset))
        self._max_tool_axis_offset = max(0.0, float(max_tool_axis_offset))
        self._still_time = max(0.0, float(still_time))
        self._still_tolerance = max(0.0, float(still_tolerance))
        self._gripper_state_getter = gripper_state_getter
        self._gripper_target_getter = gripper_target_getter
        self._event_callback = event_callback
        self._event_id = 0
        self._attached = False
        self._object_to_frame = None
        self._object_xform = None
        self._settled_gripper = None
        self._last_waiting_log = 0.0

        object_prim = self._stage.GetPrimAtPath(self._object_path)
        frame_prim = self._stage.GetPrimAtPath(self._attach_frame_path)
        if not object_prim or not object_prim.IsValid():
            raise RuntimeError(f"Grasp assist object prim is missing: {self._object_path}")
        if not frame_prim or not frame_prim.IsValid():
            raise RuntimeError(f"Grasp assist attach frame is missing: {self._attach_frame_path}")
        self._object_xform = XformPrim(self._object_path)
        print(
            "Grasp assist enabled: "
            f"object={self._object_name}, frame={self._attach_frame_path}, "
            f"command_threshold={self._attach_threshold:.3f}, "
            f"engage_threshold={self._engage_threshold:.3f}, "
            f"release_threshold={self._release_threshold:.3f}, "
            f"attach_distance={self._attach_distance:.3f} m, "
            f"max_lateral_offset={self._max_lateral_offset:.3f} m, "
            f"max_tool_axis_offset={self._max_tool_axis_offset:.3f} m, "
            f"still_time={self._still_time:.3f}s, "
            f"still_tolerance={self._still_tolerance:.4f} rad"
        )

    def update(self):
        gripper_state = self._get_gripper_state()
        if self._attached:
            if gripper_state >= self._release_threshold:
                self._detach(gripper_state)
                return
            self._follow()
            return

        desired_target = self._get_desired_gripper_target()
        if not self._closing_requested(gripper_state, desired_target):
            self._settled_gripper = None
            return

        geometry = self._target_geometry()
        distance = None if geometry is None else geometry["distance"]
        if distance is None or distance > self._attach_distance:
            self._log_waiting(gripper_state, desired_target, distance)
            self._settled_gripper = None
            return
        if not self._geometry_within_strict_limits(geometry):
            self._log_waiting(gripper_state, desired_target, distance)
            self._settled_gripper = None
            return

        if gripper_state > self._engage_threshold:
            self._settled_gripper = None
            return

        if not self._gripper_is_still(gripper_state):
            self._log_waiting(gripper_state, desired_target, distance)
            return

        self._attach(gripper_state, desired_target, geometry)

    def _get_gripper_state(self):
        if self._gripper_state_getter is not None:
            try:
                return float(self._gripper_state_getter())
            except Exception:
                pass
        return float(_get_control_state(self._robot, self._dof_names)[-1])

    def _get_desired_gripper_target(self):
        if self._gripper_target_getter is None:
            return None
        try:
            return float(self._gripper_target_getter())
        except Exception:
            return None

    def _closing_requested(self, gripper_state, desired_target):
        if desired_target is not None and desired_target <= self._attach_threshold:
            return True
        return gripper_state <= self._attach_threshold

    def _target_distance(self):
        geometry = self._target_geometry()
        return None if geometry is None else geometry["distance"]

    def _target_geometry(self):
        """Distance from the point between the fingers to the target.

        The attach frame is the gripper_base link, whose origin sits 74 mm short of
        where the fingers actually close (TCP_OFFSET_IN_GRIPPER_BASE). Measuring from
        the origin made the test both wrong and orientation-dependent: the scripted
        demo grasps with its tool point 0.6-2.4 mm from the cube yet reads 60-71 mm,
        just inside the 80 mm limit by luck. A policy that reaches the same cube to
        within 10 mm reads up to 83 mm and is refused -- the arm was on the target and
        the test was looking at the wrong point.
        """
        object_position = np.asarray(_get_prim_world_position(self._stage, self._object_path), dtype=np.float64)
        frame_world = _get_prim_world_transform(self._stage, self._attach_frame_path)
        if frame_world is None:
            return None
        rotation = np.asarray(frame_world.ExtractRotationMatrix(), dtype=np.float64).T
        translation = np.asarray(frame_world.ExtractTranslation(), dtype=np.float64)
        grasp_point = translation + rotation @ GRASP_POINT_IN_ATTACH_FRAME
        if not np.all(np.isfinite(object_position)) or not np.all(np.isfinite(grasp_point)):
            return None
        local = rotation.T @ (object_position - translation)
        offset = local - GRASP_POINT_IN_ATTACH_FRAME
        return {
            "distance": float(np.linalg.norm(object_position - grasp_point)),
            "local": local,
            "offset": offset,
            "object_position": object_position,
            "grasp_point": grasp_point,
        }

    def _geometry_within_strict_limits(self, geometry):
        if geometry is None:
            return False
        offset = geometry["offset"]
        if self._max_lateral_offset > 0.0:
            lateral_offset = float(np.linalg.norm(offset[[0, 2]]))
            if lateral_offset > self._max_lateral_offset:
                return False
        if self._max_tool_axis_offset > 0.0:
            if abs(float(offset[1])) > self._max_tool_axis_offset:
                return False
        return True

    def _gripper_is_still(self, gripper_state):
        now = time.monotonic()
        if self._settled_gripper is None:
            self._settled_gripper = (float(gripper_state), now)
            return self._still_time == 0.0
        previous_value, since = self._settled_gripper
        if abs(float(gripper_state) - previous_value) > self._still_tolerance:
            self._settled_gripper = (float(gripper_state), now)
            return False
        return now - since >= self._still_time

    def _attach(self, gripper_state, desired_target, geometry):
        frame_world = _get_prim_world_transform(self._stage, self._attach_frame_path)
        object_world = _get_prim_world_transform(self._stage, self._object_path)
        if frame_world is None or object_world is None:
            return
        object_pose_world = object_world
        object_pose_world.Orthonormalize()
        frame_pose_world = frame_world
        frame_pose_world.Orthonormalize()
        self._object_to_frame = object_pose_world * frame_pose_world.GetInverse()
        self._attached = True
        self._follow()
        distance = geometry["distance"] if geometry is not None else float("nan")
        self._publish_event(1.0, gripper_state, desired_target, geometry)
        print(
            "Grasp assist attached: "
            f"object={self._object_name}, gripper_state={gripper_state:.4f}, "
            f"target={desired_target if desired_target is not None else float('nan'):.4f}, "
            f"distance={distance:.4f} m"
        )

    def _detach(self, gripper_state):
        geometry = self._target_geometry()
        self._publish_event(0.0, gripper_state, self._get_desired_gripper_target(), geometry)
        # The target stays kinematic, so it simply holds the pose it had at
        # release. Nothing about the stage or the rigid body changes here.
        self._attached = False
        self._object_to_frame = None
        self._settled_gripper = None
        print(
            "Grasp assist detached: "
            f"object={self._object_name}, gripper_state={gripper_state:.4f}"
        )

    def reset(self):
        self._attached = False
        self._object_to_frame = None
        self._settled_gripper = None

    def is_attached(self):
        return self._attached

    def _follow(self):
        if self._object_to_frame is None or self._object_xform is None:
            return
        frame_world = _get_prim_world_transform(self._stage, self._attach_frame_path)
        if frame_world is None:
            return
        desired_world = self._object_to_frame * frame_world
        self._set_original_object_world_pose(desired_world)

    def get_object_world_position(self, object_name):
        if object_name != self._object_name or not self._attached:
            return None
        return _get_prim_world_position(self._stage, self._object_path)

    def _publish_event(self, event_type, gripper_state, desired_target, geometry):
        if self._event_callback is None:
            return
        self._event_id += 1
        if geometry is None:
            local = np.full(3, np.nan)
            offset = np.full(3, np.nan)
            object_position = np.full(3, np.nan)
            grasp_point = np.full(3, np.nan)
            distance = float("nan")
        else:
            local = geometry["local"]
            offset = geometry["offset"]
            object_position = geometry["object_position"]
            grasp_point = geometry["grasp_point"]
            distance = geometry["distance"]
        lateral_offset = float(np.linalg.norm(offset[[0, 2]])) if np.all(np.isfinite(offset[[0, 2]])) else float("nan")
        tool_axis_offset = float(abs(offset[1])) if np.isfinite(offset[1]) else float("nan")
        values = [
            float(self._event_id),
            float(event_type),
            float(OBJECT_INDEX_BY_NAME[self._object_name]),
            1.0 if event_type == 1.0 else 0.0,
            float(gripper_state),
            float(desired_target) if desired_target is not None else float("nan"),
            float(distance),
            *[float(value) for value in local],
            *[float(value) for value in GRASP_POINT_IN_ATTACH_FRAME],
            *[float(value) for value in object_position],
            *[float(value) for value in grasp_point],
            float(np.linalg.norm(offset)) if np.all(np.isfinite(offset)) else float("nan"),
            lateral_offset,
            tool_axis_offset,
        ]
        try:
            self._event_callback(values)
        except Exception as exc:
            print(f"Failed to publish grasp event for {self._object_name}: {exc}")

    def _set_original_object_world_pose(self, desired_world):
        pose = desired_world
        pose.Orthonormalize()
        translation = pose.ExtractTranslation()
        quaternion = pose.ExtractRotationQuat()
        position = np.asarray([[translation[0], translation[1], translation[2]]], dtype=np.float32)
        orientation = np.asarray(
            [[quaternion.GetReal(), *quaternion.GetImaginary()]],
            dtype=np.float32,
        )
        self._object_xform.set_world_poses(positions=position, orientations=orientation)

    def _log_waiting(self, gripper_state, desired_target, distance):
        now = time.monotonic()
        if now - self._last_waiting_log < 1.0:
            return
        self._last_waiting_log = now
        distance_text = "nan" if distance is None else f"{distance:.4f}"
        target_text = "nan" if desired_target is None else f"{desired_target:.4f}"
        print(
            "Grasp assist waiting: "
            f"gripper_state={gripper_state:.4f}, target={target_text}, "
            f"distance={distance_text} m"
        )


class MultiObjectKinematicGraspAssist:
    """Runs one kinematic grasp helper per object and allows only one attachment."""

    def __init__(self, object_names, *args, **kwargs):
        self._assists = [
            KinematicGraspAssist(*args, object_name=object_name, **kwargs)
            for object_name in object_names
        ]

    def update(self):
        attached = [assist for assist in self._assists if assist.is_attached()]
        if attached:
            attached[0].update()
            return
        for assist in self._assists:
            assist.update()
            if assist.is_attached():
                return

    def reset(self):
        for assist in self._assists:
            assist.reset()

    def is_attached(self):
        return any(assist.is_attached() for assist in self._assists)

    def get_object_world_position(self, object_name):
        for assist in self._assists:
            position = assist.get_object_world_position(object_name)
            if position is not None:
                return position
        return None


class MyCobotSliderUi:
    def __init__(
        self,
        robot,
        dof_names,
        home_positions,
        wrist_camera_path=None,
        simple_gripper_visual=None,
        virtual_gripper=False,
    ):
        self._robot = robot
        self._dof_names = dof_names
        self._home_positions = home_positions
        self._wrist_camera_path = wrist_camera_path
        self._simple_gripper_visual = simple_gripper_visual
        self._virtual_gripper_enabled = bool(virtual_gripper or simple_gripper_visual is not None)
        self._virtual_gripper_value = GRIPPER_OPEN_RAD if self._virtual_gripper_enabled else None
        self._models = {}
        self._callbacks_enabled = True
        self._gripper_ramp_target = None
        self._gripper_ramp_time = None
        self._window = ui.Window("myCobot Joint Slider Control", width=620, height=470)
        self._build()

    def _build(self):
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                if args.simple_gripper:
                    gripper_label = "simple virtual gripper"
                else:
                    gripper_label = "adaptive gripper"
                ui.Label(f"myCobot 280 M5 {gripper_label}", height=24)
                with ui.HStack(spacing=8, height=24):
                    ui.Button("Home", clicked_fn=self._set_home, width=90)
                    ui.Button("Zero Arm", clicked_fn=self._set_zero_arm, width=90)
                    ui.Button("Open", clicked_fn=lambda: self._ramp_gripper(args.gripper_open), width=80)
                    ui.Button("Close", clicked_fn=lambda: self._ramp_gripper(args.gripper_close), width=80)
                if self._wrist_camera_path:
                    with ui.HStack(spacing=8, height=24):
                        ui.Button("Wrist Cam", clicked_fn=self._view_wrist_camera, width=100)
                        ui.Button("External View", clicked_fn=self._view_external, width=110)

                ui.Spacer(height=4)
                for joint_name in ARM_JOINTS:
                    self._add_joint_slider(joint_name)
                if self._virtual_gripper_enabled or GRIPPER_CONTROLLER in self._dof_names:
                    self._add_joint_slider(GRIPPER_CONTROLLER)

    def _add_joint_slider(self, joint_name):
        if joint_name == GRIPPER_CONTROLLER and self._virtual_gripper_enabled:
            dof_index = None
            home_value = float(self._virtual_gripper_value)
        else:
            dof_index = self._dof_names.index(joint_name)
            home_value = float(self._home_positions[dof_index])
        lower, upper = JOINT_LIMITS[joint_name]
        step = 0.005 if joint_name == GRIPPER_CONTROLLER else 0.01
        label = f"{JOINT_LABELS[joint_name]}  {joint_name}"

        with ui.HStack(spacing=6, height=34):
            ui.Label(label, width=190, alignment=ui.Alignment.LEFT_CENTER)
            field_model = ui.FloatField(width=76, alignment=ui.Alignment.LEFT_CENTER).model
            field_model.set_value(home_value)
            ui.FloatSlider(
                min=lower,
                max=upper,
                step=step,
                width=ui.Fraction(1),
                alignment=ui.Alignment.LEFT_CENTER,
                model=field_model,
            )
            ui.Label("rad", width=28, alignment=ui.Alignment.LEFT_CENTER)

        def _on_value_changed(model, *, index=dof_index, name=joint_name):
            if not self._callbacks_enabled:
                return
            raw_value = model.get_value_as_float()
            lower_limit, upper_limit = JOINT_LIMITS[name]
            value = min(max(raw_value, lower_limit), upper_limit)
            if value != raw_value:
                model.set_value(value)
                return
            if name == GRIPPER_CONTROLLER:
                self._gripper_ramp_target = None
                self._apply_gripper_value(value)
                return
            _command_single_target(self._robot, index, value)

        field_model.add_value_changed_fn(_on_value_changed)
        self._models[joint_name] = field_model

    def _set_model_value(self, joint_name, value):
        if joint_name in self._models:
            self._models[joint_name].set_value(_clip_joint_value(joint_name, value))

    def _set_home(self):
        self._gripper_ramp_target = None
        self._callbacks_enabled = False
        values = {}
        for joint_name in ARM_JOINTS:
            value = float(self._home_positions[self._dof_names.index(joint_name)])
            values[joint_name] = value
            self._set_model_value(joint_name, value)
        for joint_name in _get_gripper_command_joints(self._dof_names):
            if joint_name not in self._dof_names:
                continue
            value = float(self._home_positions[self._dof_names.index(joint_name)])
            values[joint_name] = value
            self._set_model_value(joint_name, value)
        if self._virtual_gripper_enabled:
            self._set_model_value(GRIPPER_CONTROLLER, GRIPPER_OPEN_RAD)
        self._callbacks_enabled = True
        arm_values = {name: value for name, value in values.items() if name in ARM_JOINTS}
        gripper_values = {name: value for name, value in values.items() if name not in ARM_JOINTS}
        _command_targets(self._robot, self._dof_names, arm_values)
        if self._virtual_gripper_enabled:
            self._apply_gripper_value(GRIPPER_OPEN_RAD)
        elif args.kinematic_gripper:
            _set_joint_positions(self._robot, self._dof_names, gripper_values)
        elif gripper_values:
            _command_targets(self._robot, self._dof_names, gripper_values)

    def _set_zero_arm(self):
        self._callbacks_enabled = False
        values = {}
        for joint_name in ARM_JOINTS:
            values[joint_name] = 0.0
            self._set_model_value(joint_name, 0.0)
        self._callbacks_enabled = True
        _command_targets(self._robot, self._dof_names, values)

    def _set_gripper(self, value):
        self._gripper_ramp_target = None
        self._callbacks_enabled = False
        self._set_model_value(GRIPPER_CONTROLLER, value)
        self._callbacks_enabled = True
        self._apply_gripper_value(value)

    def _apply_gripper_value(self, value):
        value = _clip_joint_value(GRIPPER_CONTROLLER, value)
        if self._virtual_gripper_enabled:
            self._virtual_gripper_value = value
            if self._simple_gripper_visual is not None:
                self._simple_gripper_visual.update(value)
            return
        _command_gripper_target(self._robot, self._dof_names, value)

    def get_gripper_state(self):
        if self._virtual_gripper_enabled:
            return float(self._virtual_gripper_value)
        return float(_get_control_state(self._robot, self._dof_names)[-1])

    def _ramp_gripper(self, value):
        self._gripper_ramp_target = _clip_joint_value(GRIPPER_CONTROLLER, value)
        self._gripper_ramp_time = time.monotonic()

    def update(self):
        if self._gripper_ramp_target is None:
            return
        now = time.monotonic()
        previous = self._gripper_ramp_time
        self._gripper_ramp_time = now
        if previous is None:
            return
        step = _gripper_ramp_step(now - previous)
        current = self._models[GRIPPER_CONTROLLER].get_value_as_float()
        error = self._gripper_ramp_target - current
        if abs(error) <= step:
            next_value = self._gripper_ramp_target
            self._gripper_ramp_target = None
        else:
            next_value = current + step * np.sign(error)
        self._callbacks_enabled = False
        self._set_model_value(GRIPPER_CONTROLLER, next_value)
        self._callbacks_enabled = True
        self._apply_gripper_value(next_value)

    def _view_wrist_camera(self):
        _set_viewport_camera(self._wrist_camera_path)

    def _view_external(self):
        _set_viewport_camera("/OmniverseKit_Persp")
        set_camera_view(
            eye=[0.65, -0.85, 0.55],
            target=[0.0, 0.0, 0.16],
            camera_prim_path="/OmniverseKit_Persp",
        )


def _step_physics(physics_steps_per_frame, diagnostics_logger=None):
    if diagnostics_logger is None:
        SimulationManager.step(steps=physics_steps_per_frame)
        return
    for _ in range(physics_steps_per_frame):
        SimulationManager.step(steps=1)
        diagnostics_logger.record()


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
    physics_variant = robot_prim.GetVariantSet("Physics")
    if physics_variant and "physics" in physics_variant.GetVariantNames():
        physics_variant.SetVariantSelection("physics")
    app_utils.update_app()

    roots = _find_articulation_roots(stage)
    print(f"USD: {usd_path}")
    print(f"Articulation roots: {roots}")
    if len(roots) != 1:
        raise RuntimeError(f"Expected one articulation root, got: {roots}")
    if not args.no_fix_base:
        _fix_base_to_world(stage, roots[0])
    wrist_camera_path = None
    if not args.no_wrist_camera:
        wrist_camera_path = _add_wrist_rgb_camera(stage, roots[0])
    simple_gripper_visual = None
    if args.simple_gripper:
        simple_gripper_visual = SimpleParallelGripperVisual(stage, roots[0])
    if not args.no_task_scene:
        _add_task_scene(stage)
    _set_default_viewport_lighting()

    robot_xform = XformPrim(ROBOT_PRIM, reset_xform_op_properties=True)
    robot_xform.set_world_poses(positions=[0.0, 0.0, 0.0])
    robot = Articulation(roots[0])
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
    _initialize_task_object_rigid_prims(enable_contact_tracking=bool(args.physics_diagnostics))

    dof_names = list(robot.dof_names)
    print(f"DOF count: {len(dof_names)}")
    print(f"DOF names: {dof_names}")
    if _uses_dynamic_contact_objects():
        print(
            "Task object mode: dynamic contact grasp experiment "
            "(rigid bodies dynamic, collisions enabled, grasp assist forced off)."
        )
    else:
        print(
            "Task object mode: stable grasp-assist baseline "
            "(task objects kinematic, collisions disabled)."
        )
    virtual_gripper = _uses_virtual_gripper()
    required_joints = ARM_JOINTS if virtual_gripper else CONTROL_JOINTS
    missing_joints = [name for name in required_joints if name not in dof_names]
    if missing_joints:
        raise RuntimeError(f"Missing expected myCobot control joints: {missing_joints}")

    _configure_position_drives(
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
    if virtual_gripper:
        _configure_position_drives(
            robot,
            dof_names,
            [joint_name for joint_name in GRIPPER_JOINTS if joint_name in dof_names],
            stiffness=0.0,
            damping=0.0,
            max_effort=0.0,
            max_velocity=args.gripper_max_velocity,
            label="imported adaptive gripper disabled for virtual mode",
        )
    elif args.kinematic_gripper:
        _configure_position_drives(
            robot,
            dof_names,
            gripper_command_joints,
            stiffness=0.0,
            damping=0.0,
            max_effort=0.0,
            max_velocity=args.gripper_max_velocity,
            label="gripper disabled for kinematic mode",
        )
    else:
        if not args.drive_all_gripper_joints:
            _configure_position_drives(
                robot,
                dof_names,
                [joint_name for joint_name in GRIPPER_FOLLOWER_JOINTS if joint_name in dof_names],
                stiffness=0.0,
                damping=0.0,
                max_effort=0.0,
                max_velocity=args.gripper_max_velocity,
                label="passive mimic gripper followers",
            )
        _configure_position_drives(
            robot,
            dof_names,
            gripper_command_joints,
            stiffness=args.gripper_stiffness,
            damping=args.gripper_damping,
            max_effort=args.gripper_max_effort,
            max_velocity=args.gripper_max_velocity,
            label="gripper",
        )
    if args.simple_gripper:
        print("Gripper command mode: virtual scalar controls simple visual fingers.")
    elif args.drive_all_gripper_joints:
        print("Gripper command mode: one UI slider, controller plus mimic follower targets.")
    else:
        print("Gripper command mode: one UI slider, driving only gripper_controller; mimic followers are passive.")
    if args.simple_gripper:
        print("Gripper dynamics mode: simple virtual parallel gripper; imported gripper drives disabled.")
    if args.kinematic_gripper:
        print("Gripper dynamics mode: kinematic DOF positions; gripper drive reaction forces disabled.")

    physics_steps_per_frame = max(1, int(round(args.physics_hz / args.render_hz)))
    actual_render_hz = args.physics_hz / physics_steps_per_frame
    settle_frames = int(args.settle_time * actual_render_hz)
    print(f"Physics: {args.physics_hz:.0f} Hz, viewport/app update: {actual_render_hz:.0f} Hz.")

    home_positions = _as_numpy(robot.get_dof_positions())[0].copy()
    base_position_before = _get_base_position(robot)
    print(f"Base position before settle: {base_position_before.tolist()}")
    _command_targets(
        robot,
        dof_names,
        {joint_name: home_positions[dof_names.index(joint_name)] for joint_name in ARM_JOINTS},
    )
    if virtual_gripper:
        pass
    elif args.kinematic_gripper:
        _command_gripper_target(robot, dof_names, GRIPPER_OPEN_RAD)
    elif gripper_command_joints:
        _command_targets(
            robot,
            dof_names,
            {joint_name: home_positions[dof_names.index(joint_name)] for joint_name in gripper_command_joints},
        )
    print(f"Settling for {args.settle_time:.1f}s.")
    _step_frames(physics_steps_per_frame, settle_frames)
    base_position_after_settle = _get_base_position(robot)
    base_drift = float(np.linalg.norm(base_position_after_settle - base_position_before))
    print(f"Base position after settle: {base_position_after_settle.tolist()}, drift={base_drift:.9f} m")

    slider_ui = (
        None
        if args.headless
        else MyCobotSliderUi(
            robot,
            dof_names,
            home_positions,
            wrist_camera_path=wrist_camera_path,
            simple_gripper_visual=simple_gripper_visual,
            virtual_gripper=virtual_gripper,
        )
    )
    ros2_bridge = (
        MyCobotRos2Bridge(
            robot,
            dof_names,
            camera_path=wrist_camera_path,
            simple_gripper_visual=simple_gripper_visual,
            virtual_gripper=virtual_gripper,
        )
        if args.ros2
        else None
    )
    grasp_assist = None
    if args.grasp_assist and not args.no_task_scene:
        grasp_assist = MultiObjectKinematicGraspAssist(
            list(OBJECT_NAMES),
            stage,
            robot,
            dof_names,
            _get_grasp_attach_frame_path(stage, roots[0]),
            attach_threshold=args.grasp_attach_threshold,
            engage_threshold=args.grasp_engage_threshold,
            release_threshold=args.grasp_release_threshold,
            attach_distance=args.grasp_attach_distance,
            max_lateral_offset=args.grasp_max_lateral_offset,
            max_tool_axis_offset=args.grasp_max_tool_axis_offset,
            still_time=args.grasp_still_time,
            still_tolerance=args.grasp_still_tolerance,
            gripper_state_getter=(
                ros2_bridge.get_gripper_state
                if ros2_bridge is not None
                else slider_ui.get_gripper_state if slider_ui is not None and virtual_gripper else None
            ),
            gripper_target_getter=ros2_bridge.get_desired_gripper_target if ros2_bridge is not None else None,
            event_callback=ros2_bridge.publish_grasp_event if ros2_bridge is not None else None,
        )
        if ros2_bridge is not None:
            ros2_bridge.set_object_position_provider(grasp_assist.get_object_world_position)
            if args.hold_gripper_after_attach:
                ros2_bridge.set_attached_gripper_hold(grasp_assist.is_attached, args.attached_gripper_hold)
                print(
                    "Attached gripper hold enabled: "
                    f"deeper close targets clamp to {args.attached_gripper_hold:.4f} rad while attached."
                )
            # Detach before a placement lands: while attached the assist keeps writing
            # the object pose from the gripper frame, so a move would be overwritten on
            # the next frame and the scene would silently disagree with the request.
            ros2_bridge.set_object_placed_callback(grasp_assist.reset)
    if ros2_bridge is not None:
        ros2_bridge.set_reset_callback(
            lambda: _reset_task_scene(
                stage,
                robot,
                dof_names,
                home_positions,
                gripper_command_joints,
                grasp_assist,
                ros2_bridge,
            )
        )
    print("myCobot slider UI is ready. Move the sliders in the Isaac Sim window.")
    if wrist_camera_path:
        print(f"Wrist RGB camera path: {wrist_camera_path}")
        if args.view_wrist_camera:
            _set_viewport_camera(wrist_camera_path)

    profiler = LoopProfiler(args.loop_profile) if args.loop_profile > 0.0 else None
    if profiler is not None and ros2_bridge is not None:
        ros2_bridge.set_profiler(profiler)
    diagnostics_logger = (
        PhysicsDiagnosticsLogger(
            args.physics_diagnostics,
            robot,
            dof_names,
            1.0 / args.physics_hz,
            OBJECT_NAMES,
        )
        if args.physics_diagnostics
        else None
    )
    try:
        while simulation_app.is_running():
            mark = time.monotonic()
            if slider_ui is not None:
                slider_ui.update()
            if profiler is not None:
                now = time.monotonic()
                profiler.add("ui", now - mark)
                mark = now
            if ros2_bridge is not None:
                ros2_bridge.update()
            if profiler is not None:
                now = time.monotonic()
                profiler.add("ros2", now - mark)
                mark = now
            _step_physics(physics_steps_per_frame, diagnostics_logger)
            if profiler is not None:
                now = time.monotonic()
                profiler.add("physics", now - mark)
                mark = now
            # Run grasp assist exactly once per frame, after the physics step, so
            # the attached pose is written against settled joint transforms.
            if grasp_assist is not None:
                grasp_assist.update()
            if profiler is not None:
                now = time.monotonic()
                profiler.add("grasp", now - mark)
                mark = now
            simulation_app.update()
            if profiler is not None:
                profiler.add("app", time.monotonic() - mark)
                profiler.frame()
                if profiler.report_due():
                    profiler.report()
    finally:
        if diagnostics_logger is not None:
            diagnostics_logger.close()
        if ros2_bridge is not None:
            ros2_bridge.shutdown()


if __name__ == "__main__":
    # Isaac runs with fastShutdown, which tears the process down hard enough to lose an
    # in-flight traceback: a startup failure otherwise looks like a clean exit right
    # after "Settling", with nothing in the log to act on.
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        Path(STARTUP_ERROR_PATH).write_text(traceback.format_exc())
        print(f"startup failed; traceback written to {STARTUP_ERROR_PATH}", flush=True)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
