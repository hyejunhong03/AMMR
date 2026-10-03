"""Shared AMMR ROS2 interface constants for Isaac and real myCobot adapters."""

import math


ARM_JOINTS = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]
GRIPPER_CONTROLLER = "gripper_controller"
CONTROL_JOINTS = ARM_JOINTS + [GRIPPER_CONTROLLER]

ROS2_JOINT_TARGET_TOPIC = "/ammr/arm/joint_targets"
ROS2_JOINT_STATES_TOPIC = "/ammr/joint_states"
ROS2_ROBOT_STATE_TOPIC = "/ammr/robot_state"
ROS2_IMAGE_TOPIC = "/ammr/hand_camera/image_raw"
ROS2_CAMERA_INFO_TOPIC = "/ammr/hand_camera/camera_info"
ROS2_OBJECT_STATES_TOPIC = "/ammr/object_states"
ROS2_GRASP_EVENTS_TOPIC = "/ammr/grasp_events"
ROS2_RESET_TASK_SCENE_TOPIC = "/ammr/reset_task_scene"
# Move one scene object at runtime. Restarting Isaac per placement costs about a minute,
# which makes randomised collection impractical.
#   data = [object_index, x, y, z, qw, qx, qy, qz]   (quat_order = QUAT_ORDER)
# object_index follows OBJECT_NAMES. The orientation fields may be omitted, in which
# case the current orientation is kept.
ROS2_SET_OBJECT_POSE_TOPIC = "/ammr/set_object_pose"
OBJECT_INDEX_RED_CUBE = 0
OBJECT_INDEX_BLUE_CYLINDER = 1
OBJECT_NAMES = [
    "red_cube",
    "blue_cylinder",
]
OBJECT_INDEX_BY_NAME = {
    "red_cube": OBJECT_INDEX_RED_CUBE,
    "blue_cylinder": OBJECT_INDEX_BLUE_CYLINDER,
}

OBJECT_STATE_NAMES = [
    "red_cube_x",
    "red_cube_y",
    "red_cube_z",
    "blue_cylinder_x",
    "blue_cylinder_y",
    "blue_cylinder_z",
]

# Float64MultiArray schema for ROS2_GRASP_EVENTS_TOPIC. Each row is an event, not a
# sampled state. event_type: 1 = attach, 0 = detach.
GRASP_EVENT_FIELD_NAMES = [
    "event_id",
    "event_type",
    "object_index",
    "is_attached",
    "gripper_state",
    "desired_gripper_target",
    "distance_m",
    "local_x",
    "local_y",
    "local_z",
    "grasp_point_local_x",
    "grasp_point_local_y",
    "grasp_point_local_z",
    "object_x",
    "object_y",
    "object_z",
    "grasp_point_x",
    "grasp_point_y",
    "grasp_point_z",
    "offset_m",
    "lateral_offset_m",
    "tool_axis_offset_m",
]

WRIST_CAMERA_FRAME_ID = "hand_rgb_camera"

# Dataset contract. See docs/smolvla_dataset_schema.md; change that document first.
SCHEMA_VERSION = 2
# SmolVLA consumes 256x256, so render at that size rather than downscaling later.
IMAGE_SIZE = (256, 256)
# ROS, Isaac and SciPy each default to a different quaternion ordering, so state it
# explicitly wherever a rotation crosses a boundary.
QUAT_ORDER = "wxyz"

# Each scripted demo pose profile was measured against one red cube placement, so
# the two belong together: the Isaac scene builder and the trajectory publisher
# both read this table from --demo-pose instead of taking the position by hand.
# Launching the scene with the wrong cube leaves the target outside the grasp
# assist attach radius and the pick silently never completes.
DEMO_POSE_CUBE_POSITIONS = {
    "p0": (0.24, -0.063, 0.115),
    "p1": (0.2465, -0.0482, 0.115),
}
DEFAULT_DEMO_POSE = "p0"
# Cube centre height for an object resting on the task table: top face at 0.10 m
# plus half of the 0.03 m cube.
TASK_TABLE_CUBE_Z = 0.115
# Cylinder centre height for the 0.05 m tall cylinder resting on the same table.
TASK_TABLE_CYLINDER_Z = 0.125
TASK_OBJECT_TABLE_Z = {
    "red_cube": TASK_TABLE_CUBE_Z,
    "blue_cylinder": TASK_TABLE_CYLINDER_Z,
}

# Active simulator object catalog.
#
# Keep this as the single source of truth for the currently instantiated task objects.
# It is intentionally limited to the primitive objects that existing datasets already
# encode in object_state.npy. The broader unseen-object split lives in
# data/unseen_single_object_catalog_v0.json until we intentionally migrate the runtime
# object-state schema.
TASK_OBJECT_CATALOG_VERSION = "active_primitives_v0"
TASK_OBJECT_CATALOG = {
    "red_cube": {
        "object_id": "red_cube",
        "index": OBJECT_INDEX_RED_CUBE,
        "prim_path": "/World/TaskScene/target_cube",
        "shape_class": "box",
        "seen_split": "regression",
        "size_m": (0.03, 0.03, 0.03),
        "table_z": TASK_TABLE_CUBE_Z,
        "color_rgb": (0.90, 0.18, 0.12),
    },
    "blue_cylinder": {
        "object_id": "blue_cylinder",
        "index": OBJECT_INDEX_BLUE_CYLINDER,
        "prim_path": "/World/TaskScene/target_cylinder",
        "shape_class": "cylinder",
        "seen_split": "regression",
        "radius_m": 0.015,
        "height_m": 0.05,
        "table_z": TASK_TABLE_CYLINDER_Z,
        "color_rgb": (0.10, 0.35, 0.90),
    },
}
OBJECT_PRIM_PATHS = {
    name: spec["prim_path"]
    for name, spec in TASK_OBJECT_CATALOG.items()
}
OBJECT_SHAPE_CLASS_BY_NAME = {
    name: spec["shape_class"]
    for name, spec in TASK_OBJECT_CATALOG.items()
}

# Task table footprint in world metres, from the scene builder in
# control_mycobot_sliders_gui.py. The table top height remains 0.10 m, but the
# footprint is pulled toward the robot so all four table corners sit inside the
# myCobot 280 nominal 0.28 m working radius.
TASK_TABLE_X_RANGE = (0.095, 0.253)
TASK_TABLE_Y_RANGE = (-0.118, 0.118)
# Keep sampled objects clear of the table edge so a grasp cannot knock one off.
TASK_TABLE_MARGIN_M = 0.02

# Object-centre placement region used for scripted data collection on the
# recentered table.  A 10 mm IK grid over the full table passed 259/308 points;
# this rectangle stays inside the all-pass interior for p1 vertical approach
# with a 20 mm pre-approach and 60 mm lift.
TASK_PLACEMENT_X_RANGE = (0.130, 0.210)
TASK_PLACEMENT_Y_RANGE = (-0.095, 0.095)

# Grasp families. A family fixes the tool attitude, and every episode records which one
# produced it: mixing families in one dataset changes the action distribution, so they
# have to stay distinguishable after the fact.
GRASP_FAMILY_CANONICAL_P1 = "canonical_p1"

# Arm configuration the demo pauses at to look at the table before approaching.
#
# Without it the task is not learnable from the wrist camera. Measured on the first
# collection: the cube first exceeds 1% of the frame at step 32-33, by which point
# joint1 is already past 90% of its travel -- the target only becomes visible because
# the arm has turned toward it, so the frame that has to decide joint1 carries no
# information about where the cube is. A policy trained on that can only output the
# average motion, which is what it did: it used 21% of joint1's demonstrated range and
# lifted the cube in 1 of 10 rollouts.
#
# From this pose all 60 sampled placements land inside the frame with 15% margin, and
# the cube's image column tracks its y position at r = -0.998 (62..209 px of 256).
SURVEY_POSE = (0.27854, 0.12200, 0.42224, -1.45238, 0.02073, 0.00060)


def demo_pose_cube_position(demo_pose):
    try:
        return DEMO_POSE_CUBE_POSITIONS[demo_pose]
    except KeyError:
        raise ValueError(
            f"Unknown demo pose {demo_pose!r}; expected one of "
            f"{sorted(DEMO_POSE_CUBE_POSITIONS)}"
        ) from None


def task_object_spec(object_name):
    try:
        return TASK_OBJECT_CATALOG[object_name]
    except KeyError:
        raise ValueError(
            f"Unknown task object {object_name!r}; expected one of {OBJECT_NAMES}"
        ) from None


def object_state_slice(object_name):
    start = OBJECT_INDEX_BY_NAME[object_name] * 3
    return slice(start, start + 3)


def object_state_z_index(object_name):
    return object_state_slice(object_name).start + 2

GRIPPER_OPEN_RAD = 0.08
GRIPPER_STABLE_CLOSE_RAD = -0.35
GRIPPER_GRASP_CLOSE_RAD = -0.18
# Command level at which grasp assist accepts that a close was requested.
#
# This used to equal GRIPPER_GRASP_CLOSE_RAD, so the value the demo commands and the
# value the attach test requires were the same number and the test had no margin at
# all. A scripted demo publishing exactly -0.18 passes; a policy regressing the same
# channel lands anywhere in -0.175..-0.190 and crosses the bar on a coin flip. Across
# 40 measured rollouts every success had a gripper minimum at or below -0.18 and all
# 14 episodes that stayed above it failed. Matching the engage threshold gives the
# commanded test the same margin the observed-state test already had.
GRIPPER_GRASP_ATTACH_RAD = -0.16
# Simulated grasp assist starts once the close command is active and the fingers
# have moved far enough to indicate an object-sized grip. This intentionally does
# not wait for the commanded close target, because over-closing the adaptive
# linkage can excite the wrist joints in Isaac Sim.
GRIPPER_GRASP_ENGAGE_RAD = -0.16
SIM_GRIPPER_LIMITS = (-0.38, 0.10)

# Canonical AMMR command range for the adaptive gripper.
# The real adapter maps this range to Elephant Robotics' 0..100 gripper value.
COMMAND_GRIPPER_CLOSE_RAD = GRIPPER_GRASP_CLOSE_RAD
COMMAND_GRIPPER_OPEN_RAD = GRIPPER_OPEN_RAD


def clamp(value, lower, upper):
    return min(max(float(value), float(lower)), float(upper))


def arm_radians_to_degrees(joint_values):
    return [round(math.degrees(float(value)), 2) for value in joint_values]


def gripper_command_to_percent(
    command_rad,
    close_rad=COMMAND_GRIPPER_CLOSE_RAD,
    open_rad=COMMAND_GRIPPER_OPEN_RAD,
):
    if open_rad == close_rad:
        raise ValueError("open_rad and close_rad must be different")
    ratio = (float(command_rad) - float(close_rad)) / (float(open_rad) - float(close_rad))
    return int(round(clamp(ratio, 0.0, 1.0) * 100.0))


def gripper_percent_to_command(
    percent,
    close_rad=COMMAND_GRIPPER_CLOSE_RAD,
    open_rad=COMMAND_GRIPPER_OPEN_RAD,
):
    ratio = clamp(float(percent) / 100.0, 0.0, 1.0)
    return float(close_rad) + ratio * (float(open_rad) - float(close_rad))


def coerce_control_targets(values):
    if len(values) < len(CONTROL_JOINTS):
        raise ValueError(f"Expected at least {len(CONTROL_JOINTS)} targets, got {len(values)}")
    return [float(value) for value in values[: len(CONTROL_JOINTS)]]
