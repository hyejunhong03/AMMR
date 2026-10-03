#!/usr/bin/env python3
"""Forward and inverse kinematics for the AMMR myCobot arm.

Replaces the hand-measured joint constants in scripted_pick_demo.py. Measuring a pose
by hand takes one slider session per object placement, and a pose measured for one cube
position silently fails for another -- the removed p1 profile left a 0.0897 m gap to a
cube on the table, outside the 0.08 m grasp-assist radius, so the pick never completed.

Two layers, deliberately separate:

* forward kinematics in plain numpy, parsed straight from the URDF. No Isaac needed, so
  the TCP regression test runs anywhere and is the reference the IK is checked against.
* inverse kinematics via Lula, which only works inside a running Isaac Sim stage.

Run directly to execute the regression tests:

    python3 ammr_ik.py --self-test
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ammr_mycobot_interface import ARM_JOINTS, demo_pose_cube_position

URDF_PATH = "/home/autolab/AMMR/isaac_urdf/mycobot_280_m5_adaptive_gripper.urdf"
# Derived from URDF_PATH. The vendor URDF declares velocity="0" on every joint limit,
# which the Isaac importer ignores but Lula rejects outright:
#   [Lula] Joint 'joint2_to_joint1' specifies velocity limit as [0].
#          Velocity limit must be positive.
# Patching a separate file keeps the URDF that produced the verified USD untouched.
IK_URDF_PATH = "/home/autolab/AMMR/isaac_urdf/mycobot_280_m5_adaptive_gripper_ik.urdf"
# Kinematic bound for the solver only. Actual motion is limited by the drive settings
# (max velocity 0.5 rad/s) and by --arm-speed in the trajectory publisher.
IK_JOINT_VELOCITY_LIMIT = 3.0
ROBOT_DESCRIPTION_PATH = "/home/autolab/AMMR/isaac_urdf/mycobot_robot_description.yaml"

# Link chain from the URDF root to the frame IK targets. The gripper fingers hang off
# gripper_base and are not part of the arm chain.
END_EFFECTOR_FRAME = "gripper_base"
CHAIN_JOINTS = [
    "g_base_to_joint1",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
    "joint6output_to_gripper_base",
]

# Where the grasped object sits relative to gripper_base, in that link's own frame.
# The URDF has no TCP link, so this offset is the tool point. Derived from the two
# verified poses, which agree to within 0.6-2.8 mm:
#   p0 -> [-0.0016, 0.0739, -0.0106]
#   p1 -> [-0.0010, 0.0746, -0.0078]
TCP_OFFSET_IN_GRIPPER_BASE = np.array([-0.0013, 0.0743, -0.0092])

# Verified joint vectors, kept as the IK regression reference and warm-start seed.
VERIFIED_POSES = {
    "p0": np.array([-0.00055, -0.88663, -0.01475, -0.56235, 0.00075, 0.00051]),
    "p1": np.array([0.05926, -0.88532, -0.01444, -0.56180, 0.00116, 0.10097]),
}

# Horizontal radius from the base beyond which no IK solution exists for a grasp that
# holds the verified tool attitude. Measured by sweeping cube positions around p1:
# every target at radius <= 0.2472 m solved and every one at >= 0.2586 m failed, with
# no other variable correlating. p1 itself sits at 0.2512 m, so the verified pose is
# already close to this limit.
#
# It is a property of the fixed grasp attitude, not of the solver: the result did not
# move when the orientation tolerance was loosened fiftyfold or when several warm-start
# seeds were tried. Sampling for randomised collection has to stay inside it, or let the
# tool tilt for farther targets.
GRASP_REACH_LIMIT_M = 0.25


def _rpy_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle_to_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def ensure_ik_urdf(source_path=URDF_PATH, target_path=IK_URDF_PATH,
                   velocity_limit=IK_JOINT_VELOCITY_LIMIT):
    """Write the Lula-loadable URDF, regenerating it when the source changes.

    Only the zero velocity limits are rewritten; link geometry, joint origins and axes
    are copied through untouched, so the kinematics this solves are the kinematics the
    simulator uses.
    """
    source = Path(source_path)
    target = Path(target_path)
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target

    tree = ET.parse(source)
    root = tree.getroot()
    patched = 0
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        try:
            current = float(limit.get("velocity", "0"))
        except ValueError:
            current = 0.0
        if current <= 0.0:
            limit.set("velocity", str(velocity_limit))
            patched += 1

    target.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target, encoding="unicode", xml_declaration=True)
    print(f"wrote {target} ({patched} joint velocity limits set to {velocity_limit})")
    return target


def load_chain(urdf_path=URDF_PATH):
    """Origin, rotation, axis and type for each joint in the arm chain."""
    root = ET.parse(urdf_path).getroot()
    joints = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        joints[joint.get("name")] = {
            "type": joint.get("type"),
            "xyz": np.fromstring(origin.get("xyz", "0 0 0").strip(), sep=" ")
            if origin is not None
            else np.zeros(3),
            "rotation": _rpy_to_matrix(*np.fromstring(origin.get("rpy", "0 0 0").strip(), sep=" "))
            if origin is not None
            else np.eye(3),
            "axis": np.fromstring(axis.get("xyz").strip(), sep=" ")
            if axis is not None
            else np.array([0.0, 0.0, 1.0]),
        }
    missing = [name for name in CHAIN_JOINTS if name not in joints]
    if missing:
        raise RuntimeError(f"URDF is missing chain joints: {missing}")
    return joints


def forward_kinematics(joint_positions, chain=None):
    """World pose of gripper_base as a 4x4 matrix, for the six arm joint angles."""
    chain = chain if chain is not None else load_chain()
    transform = np.eye(4)
    for name in CHAIN_JOINTS:
        joint = chain[name]
        local = np.eye(4)
        local[:3, :3] = joint["rotation"]
        local[:3, 3] = joint["xyz"]
        if joint["type"] == "revolute":
            angle = float(joint_positions[ARM_JOINTS.index(name)])
            rotation = np.eye(4)
            rotation[:3, :3] = _axis_angle_to_matrix(joint["axis"], angle)
            local = local @ rotation
        transform = transform @ local
    return transform


def tcp_from_joints(joint_positions, chain=None):
    """World position of the tool point for a joint configuration."""
    pose = forward_kinematics(joint_positions, chain=chain)
    return pose[:3, 3] + pose[:3, :3] @ TCP_OFFSET_IN_GRIPPER_BASE


def gripper_base_target_for_tcp(tcp_position, orientation):
    """gripper_base position that puts the tool point at tcp_position.

    Inverting the tool offset, since IK targets a real link and the TCP is virtual.
    """
    return np.asarray(tcp_position, dtype=np.float64) - orientation @ TCP_OFFSET_IN_GRIPPER_BASE


# The fingers separate along gripper_base local X: the URDF puts gripper_left3 at
# x=-0.012 and gripper_right3 at x=+0.012 of that frame, with every gripper joint
# turning about local Z.
FINGER_AXIS_IN_GRIPPER_BASE = 0


def _yaw_matrix(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def finger_axis_face_angle(rotation):
    """Degrees between the finger axis and the nearest world axis. 0 = face-parallel.

    The target is a box resting axis-aligned on the table, so its faces are normal to
    world X and Y and this is the angle between the fingers and the face they should be
    closing onto. 45 degrees means closing on a corner.
    """
    axis = np.asarray(rotation, dtype=np.float64)[:, FINGER_AXIS_IN_GRIPPER_BASE]
    yaw = np.degrees(np.arctan2(axis[1], axis[0])) % 90.0
    return float(min(yaw, 90.0 - yaw))


def grasp_orientation(cube_position, reference_pose="p1", chain=None, yaw_mode="faces"):
    """Grasp rotation for a cube, from the verified wrist attitude.

    Keeping the verified attitude and varying only the base yaw is what keeps generated
    poses in the wrist branch that was measured to hold; a freely solved orientation can
    land in a configuration where the wrist jumps once the gripper closes.

    yaw_mode picks what that yaw is for:

    "face_cube" points the tool down the base-to-cube ray. It is what produced every
    recorded episode, and it grasps the target squarely only when the cube sits near the
    reference bearing: the cube stays axis-aligned on the table while the tool turns with
    the cube's bearing, so the fingers meet a face at the middle of the workspace and a
    corner at its edges. Measured across the 60 collected episodes the finger-to-face
    angle runs 0.3 to 33.0 degrees, correlating with how far off-centre the cube is
    (r = +0.65 against |cube y|), and 7 of 60 grasps close at more than 30 degrees. The
    cube is kinematic and non-colliding, so those still attach; on a real gripper the
    fingers would meet a 42 mm diagonal where they expect a 30 mm face.

    "faces" snaps that yaw to the nearest quarter turn, so the fingers stay parallel to a
    cube face wherever the cube is. The tool yaw is then fixed in world and the arm has
    to reach laterally with the wrist compensating, which is a harder ask of the IK --
    check the solved fraction before collecting against it.
    """
    chain = chain if chain is not None else load_chain()
    reference_joints = VERIFIED_POSES[reference_pose]
    reference_rotation = forward_kinematics(reference_joints, chain=chain)[:3, :3]
    reference_cube = np.asarray(demo_pose_cube_position(reference_pose), dtype=np.float64)

    reference_yaw = np.arctan2(reference_cube[1], reference_cube[0])
    target_yaw = np.arctan2(float(cube_position[1]), float(cube_position[0]))
    rotation = _yaw_matrix(target_yaw - reference_yaw) @ reference_rotation

    if yaw_mode == "face_cube":
        return rotation
    if yaw_mode != "faces":
        raise ValueError(f"Unknown yaw_mode {yaw_mode!r}; expected 'face_cube' or 'faces'")

    # Snap to the nearest quarter turn rather than to zero: the four choices grasp the
    # same box identically, and the nearest one is the smallest move away from the
    # verified wrist branch.
    axis = rotation[:, FINGER_AXIS_IN_GRIPPER_BASE]
    current = np.arctan2(axis[1], axis[0])
    quarter = np.pi / 2.0
    return _yaw_matrix(quarter * np.round(current / quarter) - current) @ rotation


def matrix_to_quaternion(matrix):
    """wxyz, matching QUAT_ORDER in the interface module."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m[2, 1] - m[1, 2]) / scale
        y = (m[0, 2] - m[2, 0]) / scale
        z = (m[1, 0] - m[0, 1]) / scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / scale
        x = 0.25 * scale
        y = (m[0, 1] + m[1, 0]) / scale
        z = (m[0, 2] + m[2, 0]) / scale
    elif m[1, 1] > m[2, 2]:
        scale = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / scale
        x = (m[0, 1] + m[1, 0]) / scale
        y = 0.25 * scale
        z = (m[1, 2] + m[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / scale
        x = (m[0, 2] + m[2, 0]) / scale
        y = (m[1, 2] + m[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.array([w, x, y, z])
    return quaternion / np.linalg.norm(quaternion)


class MyCobotIk:
    """Lula IK wrapper. Construct only inside a running Isaac Sim stage."""

    def __init__(
        self,
        urdf_path=None,
        robot_description_path=ROBOT_DESCRIPTION_PATH,
        robot_base_position=(0.0, 0.0, 0.0),
        robot_base_orientation=(1.0, 0.0, 0.0, 0.0),
    ):
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

        urdf_path = ensure_ik_urdf() if urdf_path is None else Path(urdf_path)
        self._solver = LulaKinematicsSolver(
            robot_description_path=str(robot_description_path),
            urdf_path=str(urdf_path),
        )
        # Targets are expressed in the robot base frame internally, so the base pose
        # has to be declared even when the arm sits at the world origin.
        self._solver.set_robot_base_pose(
            np.asarray(robot_base_position, dtype=np.float64),
            np.asarray(robot_base_orientation, dtype=np.float64),
        )
        self._chain = load_chain(urdf_path)

        solver_joints = list(self._solver.get_joint_names())
        if solver_joints != list(ARM_JOINTS):
            raise RuntimeError(
                "Lula cspace order does not match ARM_JOINTS.\n"
                f"  solver: {solver_joints}\n"
                f"  ARM_JOINTS: {list(ARM_JOINTS)}"
            )
        frames = list(self._solver.get_all_frame_names())
        if END_EFFECTOR_FRAME not in frames:
            raise RuntimeError(f"{END_EFFECTOR_FRAME} not in solver frames: {frames}")

    @property
    def frame_names(self):
        return list(self._solver.get_all_frame_names())

    def solve(self, target_position, target_orientation, warm_start=None,
              position_tolerance=0.001, orientation_tolerance=0.01):
        """Joint angles placing gripper_base at the target. Returns (joints, success)."""
        joints, success = self._solver.compute_inverse_kinematics(
            frame_name=END_EFFECTOR_FRAME,
            target_position=np.asarray(target_position, dtype=np.float64),
            # None means position-only; np.asarray(None) yields a 0-d array that the
            # solver then tries to index as a quaternion.
            target_orientation=(
                None if target_orientation is None
                else np.asarray(target_orientation, dtype=np.float64)
            ),
            warm_start=None if warm_start is None else np.asarray(warm_start, dtype=np.float64),
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )
        return np.asarray(joints, dtype=np.float64), bool(success)

    def solve_lift(self, grasp_joints, lift_height, max_delta=0.6):
        """Raise the grasped object by lifting the shoulder joint. Returns (joints, ok).

        Deliberately not a task-space IK solve. Two task-space attempts were rejected
        on evidence:

        * Holding the grasp attitude is unreachable. The raised target puts gripper_base
          0.3376 m from the base against 0.3145 m at the grasp, and the solver refused
          even when warm-started from the verified lift pose.
        * Leaving orientation free returns a wrist flipped 167.7 degrees (joint6 at
          3.01 rad), which would turn the carried cube upside down.

        The verified profile lifts in joint space instead, moving only joint3_to_joint2,
        and that raises the tool point 34.1 mm with 0.1 mm of horizontal drift. Solving
        for the shoulder delta keeps every other joint at its grasp value, so the pose
        cannot leave the wrist branch that was measured to hold.
        """
        grasp_joints = np.asarray(grasp_joints, dtype=np.float64)
        shoulder = ARM_JOINTS.index("joint3_to_joint2")
        base_z = float(tcp_from_joints(grasp_joints, chain=self._chain)[2])
        target_z = base_z + float(lift_height)

        def z_at(delta):
            joints = grasp_joints.copy()
            joints[shoulder] += delta
            return float(tcp_from_joints(joints, chain=self._chain)[2])

        # Height rises monotonically with the shoulder over this range, so bisect.
        low, high = 0.0, float(max_delta)
        if z_at(high) < target_z:
            return grasp_joints, False
        for _ in range(60):
            mid = 0.5 * (low + high)
            if z_at(mid) < target_z:
                low = mid
            else:
                high = mid
        joints = grasp_joints.copy()
        joints[shoulder] += 0.5 * (low + high)
        return joints, True

    def solve_grasp(self, cube_position, reference_pose="p1", warm_start=None,
                    position_tolerance=0.002, orientation_tolerance=0.05,
                    yaw_mode="faces"):
        """Joint angles that put the tool point on the cube. Returns (joints, success).

        Warm starts from the verified configuration so the solver returns the wrist
        branch that was measured to hold.

        Convergence is bounded by GRASP_REACH_LIMIT_M, not by the tolerances: loosening
        the orientation tolerance fiftyfold and retrying from several seeds both left
        the solvable set unchanged.

        position_tolerance sits at 2 mm: loose enough for TCP_OFFSET_IN_GRIPPER_BASE,
        which is one offset averaged over two measured poses and good to about 1.5 mm,
        but tight enough that the solver actually converges. At 5 mm the warm start
        already satisfies the tolerance at a reference position and is returned
        unchanged, which reads as a perfect reproduction while proving nothing.
        """
        cube_position = np.asarray(cube_position, dtype=np.float64)
        orientation = grasp_orientation(
            cube_position, reference_pose, chain=self._chain, yaw_mode=yaw_mode
        )
        target = gripper_base_target_for_tcp(cube_position, orientation)
        seed = VERIFIED_POSES[reference_pose] if warm_start is None else warm_start
        return self.solve(
            target,
            matrix_to_quaternion(orientation),
            warm_start=seed,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )


def run_self_test(position_tolerance_m=0.005):
    """FK and TCP regression, no Isaac required.

    Without this the grasp pose can drift by millimetres every time the offset or the
    chain is touched, and the failure shows up much later as a pick that misses.
    """
    chain = load_chain()
    failures = []

    print(f"{'pose':<6}{'cube (verified)':<28}{'TCP from FK':<28}{'error':>10}")
    for name, joints in VERIFIED_POSES.items():
        cube = np.asarray(demo_pose_cube_position(name), dtype=np.float64)
        tcp = tcp_from_joints(joints, chain=chain)
        error = float(np.linalg.norm(tcp - cube))
        status = "ok" if error <= position_tolerance_m else "FAIL"
        print(f"{name:<6}{str(np.round(cube, 4)):<28}{str(np.round(tcp, 4)):<28}{error * 1000:>7.1f}mm {status}")
        if error > position_tolerance_m:
            failures.append(f"{name}: TCP error {error * 1000:.1f} mm > {position_tolerance_m * 1000:.0f} mm")

    print()
    print("round trip: cube -> grasp orientation -> gripper_base target -> TCP")
    for name in VERIFIED_POSES:
        cube = np.asarray(demo_pose_cube_position(name), dtype=np.float64)
        orientation = grasp_orientation(cube, reference_pose=name, chain=chain)
        target = gripper_base_target_for_tcp(cube, orientation)
        recovered = target + orientation @ TCP_OFFSET_IN_GRIPPER_BASE
        error = float(np.linalg.norm(recovered - cube))
        status = "ok" if error <= 1e-9 else "FAIL"
        print(f"  {name}: error {error * 1000:.6f} mm {status}")
        if error > 1e-9:
            failures.append(f"{name}: round trip error {error * 1000:.6f} mm")

    print()
    print("grasp orientation reproduces the verified wrist attitude at its own cube:")
    for name, joints in VERIFIED_POSES.items():
        cube = np.asarray(demo_pose_cube_position(name), dtype=np.float64)
        expected = forward_kinematics(joints, chain=chain)[:3, :3]
        produced = grasp_orientation(cube, reference_pose=name, chain=chain)
        error = float(np.linalg.norm(produced - expected))
        status = "ok" if error <= 1e-9 else "FAIL"
        print(f"  {name}: frobenius error {error:.2e} {status}")
        if error > 1e-9:
            failures.append(f"{name}: grasp orientation error {error:.2e}")

    print()
    if failures:
        print("SELF TEST FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("SELF TEST PASSED")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run the FK/TCP regression tests.")
    parser.add_argument("--tcp-tolerance", type=float, default=0.005, help="TCP error budget in metres.")
    args = parser.parse_args()
    if args.self_test:
        return run_self_test(args.tcp_tolerance)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
