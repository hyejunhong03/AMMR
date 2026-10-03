#!/usr/bin/env python3
"""Search for an arm pose whose wrist camera sees the whole sampled workspace.

    python3 find_survey_pose.py

The recorded episodes show the target only after the arm has already turned toward it:
the cube first exceeds 1% of the frame at step 32-33, by which point joint1 is over 90%
of the way to its final value. The policy is therefore asked to choose joint1 from a
frame that carries no information about where the cube is, which no amount of extra data
can fix.

A survey pose breaks that circularity -- look first from one fixed configuration, then
decide. For that to work the pose has to see every placement the sampler can produce, so
this projects the sampled cube positions through the camera model and reports which poses
keep all of them inside the frame.

Geometry only, no Isaac needed.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ammr_ik import ARM_JOINTS, _axis_angle_to_matrix, _rpy_to_matrix, load_chain

# Camera mount, mirroring _add_wrist_rgb_camera in control_mycobot_sliders_gui.py.
FLANGE_CHAIN = [
    "g_base_to_joint1",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]
WRIST_CAMERA_TRANSLATION = np.array([0.0, -0.028, 0.04])
WRIST_CAMERA_RPY = (1.5708, -1.5708, 0.0)
# The Gazebo sensor aims along hand_camera_link +X with +Z up; a USD camera looks along
# local -Z with +Y up.
CAMERA_TO_LINK = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
CAMERA_FOV = 1.2
# Keep targets away from the very edge, where they are a few pixels and half occluded by
# the fingers.
EDGE_MARGIN = 0.15


def flange_pose(joint_positions, chain):
    transform = np.eye(4)
    for name in FLANGE_CHAIN:
        joint = chain[name]
        local = np.eye(4)
        local[:3, :3] = joint["rotation"]
        local[:3, 3] = joint["xyz"]
        if joint["type"] == "revolute":
            rotation = np.eye(4)
            rotation[:3, :3] = _axis_angle_to_matrix(
                joint["axis"], float(joint_positions[ARM_JOINTS.index(name)])
            )
            local = local @ rotation
        transform = transform @ local
    return transform


def camera_pose(joint_positions, chain):
    mount = np.eye(4)
    mount[:3, :3] = _rpy_to_matrix(*WRIST_CAMERA_RPY)
    mount[:3, 3] = WRIST_CAMERA_TRANSLATION
    orient = np.eye(4)
    orient[:3, :3] = CAMERA_TO_LINK
    return flange_pose(joint_positions, chain) @ mount @ orient


def project(camera, point):
    """Normalised image coordinates, or None when the point is behind the camera."""
    rotation, translation = camera[:3, :3], camera[:3, 3]
    local = rotation.T @ (np.asarray(point, dtype=np.float64) - translation)
    depth = -local[2]
    if depth <= 1e-6:
        return None
    half = np.tan(CAMERA_FOV * 0.5)
    return np.array([local[0] / depth / half, local[1] / depth / half])


def visible_fraction(joints, targets, chain, margin=EDGE_MARGIN):
    limit = 1.0 - margin
    coords, seen = [], 0
    camera = camera_pose(joints, chain)
    for target in targets:
        ndc = project(camera, target)
        if ndc is not None and abs(ndc[0]) <= limit and abs(ndc[1]) <= limit:
            seen += 1
            coords.append(ndc)
    return seen / len(targets), coords


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses-file", default="/home/autolab/AMMR/data/grasp_poses_sampled.json")
    parser.add_argument("--samples", type=int, default=20000, help="Random configurations to try.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=5, help="Best poses to report.")
    args = parser.parse_args()

    payload = json.loads(Path(args.poses_file).read_text())
    targets = [e["cube_position"] for e in payload["entries"] if e.get("solved")]
    targets = np.asarray(targets, dtype=np.float64)
    print(f"targets: {len(targets)} sampled cube positions")
    print(f"  x {targets[:, 0].min():.3f}..{targets[:, 0].max():.3f}  "
          f"y {targets[:, 1].min():.3f}..{targets[:, 1].max():.3f}")

    chain = load_chain()

    # Sample joint1..joint4 -- joint5 and joint6 barely move the camera's aim here, so
    # searching them only adds dimensions without adding coverage.
    rng = np.random.default_rng(args.seed)
    ranges = np.array([[-0.4, 0.4], [-1.6, 0.2], [-1.4, 1.4], [-1.6, 0.6]])
    results = []
    for _ in range(args.samples):
        joints = np.zeros(6)
        joints[:4] = rng.uniform(ranges[:, 0], ranges[:, 1])
        fraction, coords = visible_fraction(joints, targets, chain)
        if fraction > 0.0:
            spread = float(np.std([c[0] for c in coords])) if len(coords) > 1 else 0.0
            results.append((fraction, spread, joints.copy()))

    if not results:
        print("\nno configuration saw any target")
        return 1

    # Prefer full coverage, then the widest horizontal spread: a pose where placements
    # map to distinct image columns is one a policy can actually read joint1 from.
    results.sort(key=lambda r: (-r[0], -r[1]))
    print(f"\nconfigurations seeing every target: {sum(1 for r in results if r[0] >= 1.0)}")
    print()
    print(f"{'coverage':>9}{'x-spread':>10}  joints (j1..j6)")
    for fraction, spread, joints in results[: args.top]:
        print(f"{fraction * 100:>8.1f}%{spread:>10.3f}  {np.round(joints, 4).tolist()}")

    best = results[0]
    fraction, coords = visible_fraction(best[2], targets, chain)
    xs = np.array([c[0] for c in coords])
    print()
    print("best pose detail:")
    print(f"  joints   : {np.round(best[2], 5).tolist()}")
    print(f"  coverage : {fraction * 100:.1f}%")
    print(f"  image x  : {xs.min():+.3f} .. {xs.max():+.3f} (normalised, +-1 is the frame edge)")
    print(f"  in pixels: {(xs.min() + 1) * 128:.0f} .. {(xs.max() + 1) * 128:.0f} of 256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
