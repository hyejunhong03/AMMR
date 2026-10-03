#!/usr/bin/env python3
"""Solve grasp and lift joint vectors for cube positions and cache them to JSON.

Run under the Isaac interpreter, since Lula needs a live stage:

    /home/autolab/isaacsim/python.sh plan_grasp_poses.py --demo-pose p1 --out poses.json

Batching exists because Isaac takes about a minute to start. Calling IK once per episode
would dominate collection time, so every pose for a run is solved in one session and the
trajectory publisher, which runs on the ROS2 interpreter, just reads the file.

Positions outside the reach limit are reported as unsolved rather than silently dropped,
so a sampling range that is mostly unreachable is visible immediately.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp

from ammr_mycobot_interface import OBJECT_NAMES

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--demo-pose", default="p1", help="Reference profile for the grasp attitude.")
parser.add_argument(
    "--yaw-mode",
    choices=("face_cube", "faces"),
    default="faces",
    help=(
        "How the tool yaw is chosen. 'face_cube' aims down the base-to-cube ray, which "
        "closes the fingers on a cube corner the further the cube sits off-centre "
        "(0.3-33.0 degrees across the collected episodes). 'faces' snaps the yaw to a "
        "quarter turn so the fingers stay parallel to a cube face everywhere, at the "
        "cost of asking more of the IK."
    ),
)
parser.add_argument(
    "--max-finger-face-angle",
    type=float,
    default=5.0,
    help=(
        "Reject box grasps where the finger closing axis is more than this many "
        "degrees away from the nearest object face. Set to 45 to allow corner "
        "grasps for debugging. The default keeps generated cube demonstrations "
        "face-aligned."
    ),
)
parser.add_argument("--out", required=True, help="Where to write the pose cache JSON.")
parser.add_argument(
    "--object",
    choices=OBJECT_NAMES,
    default="red_cube",
    help="Task object to solve poses for.",
)
parser.add_argument(
    "--cube",
    action="append",
    nargs=3,
    type=float,
    metavar=("X", "Y", "Z"),
    help="Cube position to solve. Repeatable. Defaults to the profile's own placement.",
)
parser.add_argument(
    "--object-position",
    action="append",
    nargs=3,
    type=float,
    metavar=("X", "Y", "Z"),
    help="Object centre position to solve. Repeatable. Overrides --cube when present.",
)
parser.add_argument(
    "--object-position-file",
    default=None,
    help=(
        "JSON file containing positions to solve. Accepts either a list of [x,y,z] "
        "positions or a dict with an entries list containing object_position, "
        "sampled_object_pose, or cube_position."
    ),
)
parser.add_argument(
    "--lift-height",
    type=float,
    default=0.035,
    help=(
        "Metres to raise the cube. The default matches the verified profile, which "
        "lifts the tool point 34.1 mm with 0.1 mm of horizontal drift."
    ),
)
parser.add_argument(
    "--lift-orientation-tolerance",
    type=float,
    default=0.25,
    help=(
        "Wrist attitude slack for the lift, in radians. The grasp needs the measured "
        "attitude so the fingers line up with the cube, but once the cube is attached a "
        "tilt costs nothing -- the verified lift rotates the wrist by 8.06 degrees "
        "(0.141 rad), which a grasp-tight tolerance rejects outright."
    ),
)
parser.add_argument(
    "--pre-approach-height",
    type=float,
    default=0.03,
    help=(
        "Metres to place the first open-gripper approach waypoint above the grasp TCP. "
        "The demo then descends vertically in Cartesian space instead of interpolating "
        "one joint-space move straight into the cube."
    ),
)
parser.add_argument(
    "--descent-steps",
    type=int,
    default=4,
    help=(
        "Number of Cartesian descent intervals from the pre-approach TCP to the final "
        "grasp TCP. A value of 4 stores 5 open-gripper approach waypoints including "
        "the top and final grasp pose."
    ),
)
parser.add_argument(
    "--sample",
    type=int,
    default=0,
    help=(
        "Sample this many object positions across the table instead of taking --cube. "
        "Candidates are drawn wide and filtered by IK rather than by a hand-picked "
        "radius, so the accepted set is whatever the arm can actually reach."
    ),
)
parser.add_argument("--seed", type=int, default=0, help="RNG seed for --sample.")
parser.add_argument(
    "--max-attempts-per-sample",
    type=int,
    default=40,
    help="Candidate draws allowed per accepted position before giving up.",
)
parser.add_argument(
    "--max-tcp-error",
    type=float,
    default=0.005,
    help="Reject a solution whose FK tool point misses the object by more than this.",
)
parser.add_argument("--report", default=None, help="Outcome file; survives Isaac fastShutdown.")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402

from ammr_ik import (  # noqa: E402
    GRASP_REACH_LIMIT_M,
    MyCobotIk,
    finger_axis_face_angle,
    gripper_base_target_for_tcp,
    grasp_orientation,
    load_chain,
    matrix_to_quaternion,
    tcp_from_joints,
)
from ammr_mycobot_interface import (  # noqa: E402
    GRASP_FAMILY_CANONICAL_P1,
    OBJECT_INDEX_BY_NAME,
    OBJECT_SHAPE_CLASS_BY_NAME,
    TASK_OBJECT_CATALOG_VERSION,
    TASK_TABLE_CUBE_Z,
    TASK_OBJECT_TABLE_Z,
    TASK_PLACEMENT_X_RANGE,
    TASK_PLACEMENT_Y_RANGE,
    ARM_JOINTS,
    demo_pose_cube_position,
    task_object_spec,
)

# Half the 0.03 m cube, so a sampled centre keeps the whole body on the table.
CUBE_HALF_SIZE_M = 0.015
# URDF revolute limits for the arm joints, in ARM_JOINTS order.
ARM_JOINT_LIMITS = {
    "joint2_to_joint1": (-2.9321, 2.9321),
    "joint3_to_joint2": (-2.4434, 2.4434),
    "joint4_to_joint3": (-2.6179, 2.6179),
    "joint5_to_joint4": (-2.6179, 2.6179),
    "joint6_to_joint5": (-2.7052, 2.7925),
    "joint6output_to_joint6": (-3.14159, 3.14159),
}


def joint_limits_ok(joints):
    """True when every joint sits inside its URDF limit."""
    for name, value in zip(ARM_JOINTS, joints):
        lower, upper = ARM_JOINT_LIMITS[name]
        if not lower <= float(value) <= upper:
            return False, f"{name} at {float(value):.4f} outside [{lower}, {upper}]"
    return True, None


def sample_object_positions(count, seed, max_attempts, object_name):
    """Candidate object positions inside the IK-validated placement region.

    The table itself is larger than this region.  The placement bounds are the
    conservative interior of the recentered-table IK pass map, so random
    collection avoids edge cases that only barely satisfy vertical approach.
    """
    rng = np.random.default_rng(seed)
    x_low, x_high = TASK_PLACEMENT_X_RANGE
    y_low, y_high = TASK_PLACEMENT_Y_RANGE
    for _ in range(count * max_attempts):
        yield np.array(
            [
                rng.uniform(x_low, x_high),
                rng.uniform(y_low, y_high),
                TASK_OBJECT_TABLE_Z[object_name],
            ]
        )


def load_object_positions(path):
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a list or a dict with entries")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path} has no entries list")
    positions = []
    for entry in entries:
        if isinstance(entry, dict):
            position = (
                entry.get("object_position")
                or entry.get("sampled_object_pose")
                or entry.get("cube_position")
            )
        else:
            position = entry
        if position is None:
            continue
        if len(position) != 3:
            raise ValueError(f"invalid position in {path}: {position}")
        positions.append(position)
    if not positions:
        raise ValueError(f"{path} has no usable positions")
    return positions


def solve_tcp_waypoint(solver, tcp_position, orientation, warm_start,
                       position_tolerance=0.002, orientation_tolerance=0.05):
    target = gripper_base_target_for_tcp(tcp_position, orientation)
    return solver.solve(
        target,
        matrix_to_quaternion(orientation),
        warm_start=warm_start,
        position_tolerance=position_tolerance,
        orientation_tolerance=orientation_tolerance,
    )


def plan_vertical_approach(solver, chain, object_position, orientation, grasp_seed,
                           pre_approach_height, descent_steps, max_tcp_error):
    """Solve a top-down open approach path in Cartesian TCP space.

    A single joint-space interpolation from the survey pose to the grasp pose can sweep
    an open finger through the cube. Intermediate TCP waypoints keep the final descent
    vertical while staying on the wrist branch selected by the grasp seed.
    """
    if pre_approach_height <= 0.0 or descent_steps <= 0:
        return None, None

    object_position = np.asarray(object_position, dtype=np.float64)
    steps = max(1, int(descent_steps))
    tcp_positions = []
    waypoint_joints = []
    waypoint_errors = []
    warm_start = np.asarray(grasp_seed, dtype=np.float64)

    for index in range(steps + 1):
        remaining = float(pre_approach_height) * (1.0 - index / steps)
        tcp = object_position + np.array([0.0, 0.0, remaining])
        joints, ok = solve_tcp_waypoint(
            solver,
            tcp,
            orientation,
            warm_start=warm_start,
        )
        if not ok:
            return None, {
                "reject_reason": "ik_no_vertical_approach_solution",
                "reason": (
                    f"no vertical approach solution at waypoint {index}/{steps} "
                    f"(tcp={np.round(tcp, 4).tolist()})"
                ),
            }

        tcp_error = float(np.linalg.norm(tcp_from_joints(joints, chain=chain) - tcp))
        if tcp_error > max_tcp_error:
            return None, {
                "reject_reason": "vertical_approach_tcp_error_too_large",
                "reason": (
                    f"vertical approach waypoint {index}/{steps} tcp error "
                    f"{tcp_error * 1000:.2f} mm > {max_tcp_error * 1000:.0f} mm"
                ),
            }

        limits_ok, limit_detail = joint_limits_ok(joints)
        if not limits_ok:
            return None, {
                "reject_reason": "joint_limit_vertical_approach",
                "reason": f"vertical approach waypoint {index}/{steps} {limit_detail}",
            }

        tcp_positions.append(tcp)
        waypoint_joints.append(joints)
        waypoint_errors.append(tcp_error)
        warm_start = joints

    return {
        "joints": waypoint_joints,
        "tcp_positions": tcp_positions,
        "tcp_errors": waypoint_errors,
    }, None


def plan(solver, chain, object_position, object_name, demo_pose, lift_height, lift_orientation_tolerance,
         max_tcp_error=0.005, yaw_mode="faces", max_finger_face_angle=5.0,
         pre_approach_height=0.03, descent_steps=4):
    """Grasp joints at the cube and at the cube raised by lift_height.

    Every rejection carries a reason. A sampling range that is mostly unreachable then
    shows up as a distribution of reasons instead of a low yield with no explanation.
    """
    object_position = np.asarray(object_position, dtype=np.float64)
    radius = float(np.hypot(object_position[0], object_position[1]))
    object_index = OBJECT_INDEX_BY_NAME[object_name]
    object_spec = task_object_spec(object_name)

    orientation = grasp_orientation(object_position, demo_pose, chain=chain, yaw_mode=yaw_mode)
    finger_face_angle = finger_axis_face_angle(orientation)

    entry = {
        "catalog_version": TASK_OBJECT_CATALOG_VERSION,
        "object_id": object_spec["object_id"],
        "object_name": object_name,
        "target_object": object_name,
        "object_index": object_index,
        "shape_class": OBJECT_SHAPE_CLASS_BY_NAME[object_name],
        "seen_split": object_spec.get("seen_split", "unknown"),
        "object_size_m": object_spec.get("size_m"),
        "object_radius_m": object_spec.get("radius_m"),
        "object_height_m": object_spec.get("height_m"),
        "object_color_rgb": object_spec.get("color_rgb"),
        "object_position": object_position.tolist(),
        # Backward-compatible key consumed by older red-cube scripts.
        "cube_position": object_position.tolist(),
        "sampled_object_pose": object_position.tolist(),
        "distance_from_base_m": round(radius, 5),
        "radius_m": round(radius, 5),
        "reach_limit_m": GRASP_REACH_LIMIT_M,
        "grasp_family": GRASP_FAMILY_CANONICAL_P1,
        "yaw_mode": yaw_mode,
        # Angle between the fingers and the nearest cube face, recorded per entry so a
        # cache can be checked for corner grasps without re-deriving the kinematics.
        "finger_face_angle_deg": round(finger_face_angle, 2),
        "max_finger_face_angle_deg": float(max_finger_face_angle),
        "accepted_by_ik": False,
        "reject_reason": None,
    }
    if OBJECT_SHAPE_CLASS_BY_NAME[object_name] == "box" and finger_face_angle > max_finger_face_angle:
        entry["solved"] = False
        entry["reject_reason"] = "finger_face_angle_too_large"
        entry["reason"] = (
            f"finger-face angle {finger_face_angle:.2f} deg > "
            f"{max_finger_face_angle:.2f} deg"
        )
        return entry

    grasp_joints, grasp_ok = solver.solve_grasp(object_position, reference_pose=demo_pose, yaw_mode=yaw_mode)
    if not grasp_ok:
        entry["solved"] = False
        entry["reject_reason"] = (
            "ik_no_grasp_solution"
            if radius <= GRASP_REACH_LIMIT_M
            else "ik_no_grasp_solution_beyond_reach"
        )
        entry["reason"] = f"no grasp solution (radius {radius:.4f} m)"
        return entry

    grasp_tcp_error = float(np.linalg.norm(tcp_from_joints(grasp_joints, chain=chain) - object_position))
    if grasp_tcp_error > max_tcp_error:
        entry["solved"] = False
        entry["reject_reason"] = "tcp_error_too_large"
        entry["ik_tcp_error"] = round(grasp_tcp_error, 6)
        entry["reason"] = f"grasp tcp error {grasp_tcp_error * 1000:.2f} mm > {max_tcp_error * 1000:.0f} mm"
        return entry

    limits_ok, limit_detail = joint_limits_ok(grasp_joints)
    if not limits_ok:
        entry["solved"] = False
        entry["reject_reason"] = "joint_limit"
        entry["reason"] = f"grasp {limit_detail}"
        return entry

    vertical_approach, vertical_reject = plan_vertical_approach(
        solver,
        chain,
        object_position,
        orientation,
        grasp_joints,
        pre_approach_height,
        descent_steps,
        max_tcp_error,
    )
    if vertical_reject is not None:
        entry["solved"] = False
        entry.update(vertical_reject)
        return entry
    if vertical_approach is not None:
        grasp_joints = vertical_approach["joints"][-1]
        grasp_tcp_error = float(np.linalg.norm(tcp_from_joints(grasp_joints, chain=chain) - object_position))

    lifted = object_position + np.array([0.0, 0.0, float(lift_height)])
    lift_joints, lift_ok = solver.solve_lift(grasp_joints, lift_height)
    if not lift_ok:
        entry["solved"] = False
        entry["reject_reason"] = "no_lift_solution"
        entry["reason"] = f"grasp solved but no lift solution at +{lift_height} m"
        return entry

    limits_ok, limit_detail = joint_limits_ok(lift_joints)
    if not limits_ok:
        entry["solved"] = False
        entry["reject_reason"] = "joint_limit_lift"
        entry["reason"] = f"lift {limit_detail}"
        return entry

    entry.update(
        {
            "solved": True,
            "accepted_by_ik": True,
            "ik_tcp_error": round(grasp_tcp_error, 6),
            "pre_grasp_joints": [round(float(v), 6) for v in grasp_joints],
            "lift_joints": [round(float(v), 6) for v in lift_joints],
            "approach_type": "vertical_cartesian" if vertical_approach is not None else "direct_joint",
            "pre_approach_height_m": float(pre_approach_height),
            "descent_steps": int(descent_steps),
            "grasp_tcp_error_m": round(grasp_tcp_error, 6),
            "lift_tcp_error_m": round(float(np.linalg.norm(tcp_from_joints(lift_joints, chain=chain) - lifted)), 6),
            "lift_height_m": float(lift_height),
        }
    )
    if vertical_approach is not None:
        entry.update(
            {
                "approach_waypoint_joints": [
                    [round(float(v), 6) for v in joints]
                    for joints in vertical_approach["joints"]
                ],
                "approach_waypoint_tcp_positions": [
                    [round(float(v), 6) for v in tcp]
                    for tcp in vertical_approach["tcp_positions"]
                ],
                "approach_waypoint_tcp_errors_m": [
                    round(float(error), 6)
                    for error in vertical_approach["tcp_errors"]
                ],
            }
        )
    return entry


def main():
    stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)

    solver = MyCobotIk()
    chain = load_chain()

    def solve_one(object_position):
        return plan(
            solver, chain, object_position, args.object, args.demo_pose, args.lift_height,
            args.lift_orientation_tolerance, args.max_tcp_error, args.yaw_mode,
            args.max_finger_face_angle, args.pre_approach_height, args.descent_steps,
        )

    if args.sample:
        entries = []
        accepted = 0
        for object_position in sample_object_positions(
            args.sample, args.seed, args.max_attempts_per_sample, args.object
        ):
            entry = solve_one(object_position)
            entries.append(entry)
            if entry["solved"]:
                accepted += 1
                if accepted >= args.sample:
                    break
    else:
        object_positions = (
            load_object_positions(args.object_position_file)
            if args.object_position_file
            else args.object_position or args.cube
        )
        if object_positions is None:
            if args.object != "red_cube":
                raise ValueError("--object-position is required when --object is not red_cube")
            object_positions = [list(demo_pose_cube_position(args.demo_pose))]
        entries = [solve_one(object_position) for object_position in object_positions]

    solved = [e for e in entries if e["solved"]]
    print(f"{args.object:<14}{'position':<30}{'radius':>9}{'grasp err':>11}{'lift err':>10}  status")
    for entry in (entries if not args.sample else [e for e in entries if e["solved"]]):
        position = "[" + ", ".join(f"{v:.4f}" for v in entry["object_position"]) + "]"
        if entry["solved"]:
            print(
                f"{entry['object_name']:<14}{position:<30}{entry['radius_m']:>9.4f}"
                f"{entry['grasp_tcp_error_m'] * 1000:>9.2f}mm"
                f"{entry['lift_tcp_error_m'] * 1000:>8.2f}mm  ok"
            )
        else:
            print(
                f"{entry['object_name']:<14}{position:<30}{entry['radius_m']:>9.4f}"
                f"{'':>11}{'':>10}  {entry['reason']}"
            )

    rejected = [e for e in entries if not e["solved"]]
    if rejected:
        reasons = {}
        for entry in rejected:
            reasons[entry["reject_reason"]] = reasons.get(entry["reject_reason"], 0) + 1
        print()
        print("rejections:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<36}{count}")
        radii = [e["distance_from_base_m"] for e in solved]
        if radii:
            print(f"  accepted radius range: {min(radii):.4f} - {max(radii):.4f} m")

    payload = {
        "demo_pose": args.demo_pose,
        "catalog_version": TASK_OBJECT_CATALOG_VERSION,
        "object_name": args.object,
        "target_object": args.object,
        "object_index": OBJECT_INDEX_BY_NAME[args.object],
        "object_spec": task_object_spec(args.object),
        "grasp_family": GRASP_FAMILY_CANONICAL_P1,
        "lift_height_m": args.lift_height,
        "pre_approach_height_m": args.pre_approach_height,
        "descent_steps": args.descent_steps,
        "reach_limit_m": GRASP_REACH_LIMIT_M,
        "max_tcp_error_m": args.max_tcp_error,
        "yaw_mode": args.yaw_mode,
        "max_finger_face_angle_deg": args.max_finger_face_angle,
        "sample_count": args.sample,
        "seed": args.seed,
        "attempted": len(entries),
        "accepted": len(solved),
        "entries": entries,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print()
    print(f"solved {len(solved)}/{len(entries)} -> {out}")
    return 0 if solved else 1


if __name__ == "__main__":
    import traceback

    report_path = Path(args.report) if args.report else None
    code = 0
    try:
        code = main()
    except Exception:
        code = 1
        traceback.print_exc()
        if report_path is not None:
            report_path.write_text("PLAN ERROR\n" + traceback.format_exc())
    else:
        if report_path is not None:
            report_path.write_text(f"PLAN {'OK' if code == 0 else 'FAILED'}\n")
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
    raise SystemExit(code)
