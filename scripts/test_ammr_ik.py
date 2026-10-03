#!/usr/bin/env python3
"""Check that Lula IK reproduces the hand-measured p0/p1 poses.

Run under the Isaac interpreter -- LulaKinematicsSolver needs a live stage:

    /home/autolab/isaacsim/python.sh test_ammr_ik.py

This is the gate before IK replaces the hardcoded joint constants. Reproducing the two
verified configurations from their own cube positions is the evidence that a generated
pose for a new cube position will be trustworthy.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--joint-tolerance", type=float, default=0.02, help="Per-joint error budget in radians.")
parser.add_argument("--tcp-tolerance", type=float, default=0.005, help="TCP error budget in metres.")
parser.add_argument("--report", default=None, help="Write the pass/fail outcome here; survives Isaac fastShutdown.")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402

from ammr_ik import (  # noqa: E402
    END_EFFECTOR_FRAME,
    VERIFIED_POSES,
    MyCobotIk,
    forward_kinematics,
    grasp_orientation,
    gripper_base_target_for_tcp,
    load_chain,
    matrix_to_quaternion,
    tcp_from_joints,
)
from ammr_mycobot_interface import ARM_JOINTS, demo_pose_cube_position  # noqa: E402


def main():
    # LulaKinematicsSolver reads stage units on construction, so a stage must exist.
    stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)

    solver = MyCobotIk()
    print(f"solver cspace : {list(ARM_JOINTS)}")
    print(f"end effector  : {END_EFFECTOR_FRAME}")
    print()

    chain = load_chain()
    failures = []

    print("=== pose reproduction: verified cube position -> IK -> joints ===")
    for name, expected in VERIFIED_POSES.items():
        cube = np.asarray(demo_pose_cube_position(name), dtype=np.float64)
        joints, success = solver.solve_grasp(cube, reference_pose=name)
        if not success:
            failures.append(f"{name}: IK did not converge")
            print(f"{name}: IK FAILED to converge")
            continue

        joint_error = np.abs(joints - expected)
        tcp = tcp_from_joints(joints, chain=chain)
        tcp_error = float(np.linalg.norm(tcp - cube))

        joint_ok = float(joint_error.max()) <= args.joint_tolerance
        tcp_ok = tcp_error <= args.tcp_tolerance
        print(f"{name}:")
        print(f"  expected : {np.round(expected, 5).tolist()}")
        print(f"  ik       : {np.round(joints, 5).tolist()}")
        print(f"  per-joint error : {np.round(joint_error, 5).tolist()}")
        print(f"  max joint error : {joint_error.max():.5f} rad  {'ok' if joint_ok else 'FAIL'}")
        print(f"  tcp error       : {tcp_error * 1000:.2f} mm  {'ok' if tcp_ok else 'FAIL'}")
        if not joint_ok:
            failures.append(f"{name}: max joint error {joint_error.max():.5f} > {args.joint_tolerance}")
        if not tcp_ok:
            failures.append(f"{name}: tcp error {tcp_error * 1000:.2f} mm > {args.tcp_tolerance * 1000:.0f} mm")
        print()

    print("=== solver FK agrees with the numpy FK ===")
    for name, joints in VERIFIED_POSES.items():
        position, _ = solver._solver.compute_forward_kinematics(END_EFFECTOR_FRAME, joints)
        reference = forward_kinematics(joints, chain=chain)[:3, 3]
        error = float(np.linalg.norm(np.asarray(position) - reference))
        print(f"  {name}: lula {np.round(position, 4)} vs numpy {np.round(reference, 4)} -> {error * 1000:.3f} mm")
        if error > 1e-3:
            failures.append(f"{name}: FK mismatch {error * 1000:.3f} mm")

    # Workspace probe, informational: how far the fixed-attitude grasp can be moved
    # before the solver stops converging. Phase C randomisation needs this envelope.
    print()
    print("=== workspace probe: nearby cube positions vs orientation tolerance ===")
    base = np.asarray(demo_pose_cube_position("p1"), dtype=np.float64)
    offsets = [
        [0.02, 0.0, 0.0], [-0.02, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, -0.03, 0.0],
        [0.04, 0.0, 0.0], [0.0, 0.06, 0.0], [0.0, -0.06, 0.0],
    ]
    for label, single_seed in (("single seed ", True), ("multi seed  ", False)):
        solved = []
        for offset in offsets:
            cube = base + np.asarray(offset)
            if single_seed:
                orientation = grasp_orientation(cube, "p1", chain=chain)
                target = gripper_base_target_for_tcp(cube, orientation)
                joints, success = solver.solve(
                    target, matrix_to_quaternion(orientation),
                    warm_start=VERIFIED_POSES["p1"], orientation_tolerance=0.05,
                )
            else:
                joints, success = solver.solve_grasp(cube, reference_pose="p1")
            ok = success and float(np.linalg.norm(tcp_from_joints(joints, chain=chain) - cube)) <= args.tcp_tolerance
            solved.append("o" if ok else ".")
        print(f"  {label}: {' '.join(solved)}  ({solved.count('o')}/{len(offsets)})")
    print(f"  offsets: {offsets}")

    print()
    if failures:
        print("IK TEST FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("IK TEST PASSED")
    return 0


if __name__ == "__main__":
    # Isaac shuts down with fastShutdown, which exits the process hard enough to lose an
    # in-flight traceback and report success anyway. Capture the outcome to a file and
    # re-print it, so a failure here cannot look like a pass.
    import traceback

    report_path = Path(args.report) if args.report else None
    code = 0
    try:
        code = main()
    except Exception:
        code = 1
        traceback.print_exc()
        if report_path is not None:
            report_path.write_text("IK TEST ERROR\n" + traceback.format_exc())
    else:
        if report_path is not None:
            report_path.write_text(f"IK TEST {'PASSED' if code == 0 else 'FAILED'}\n")
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
    raise SystemExit(code)
