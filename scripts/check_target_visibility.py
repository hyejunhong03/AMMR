#!/usr/bin/env python3
"""Check that the target is visible before the demo commits to a direction.

    python3 check_target_visibility.py --root data/smolvla_survey

The first collection produced a dataset the wrist camera could not support. The cube
first exceeded 1% of the frame at step 32-33, by which point joint1 had already covered
90% of its travel: the target became visible only because the arm had turned toward it,
so the frames that had to choose joint1 carried no information about where the cube was.
A policy trained on that used 21% of joint1's demonstrated range and lifted the cube in
1 of 10 rollouts.

This measures the ordering directly. For every episode it finds the first frame where
the target is visible and the frame by which joint1 is 90% decided, and fails when the
decision comes first.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Red cube against a white table and a blue cylinder. Deliberately strict on the red
# channel dominating, so the blue distractor and the grey table cannot register.
RED_MIN = 110
RED_MARGIN = 40


def red_mask(image):
    array = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    return (red > RED_MIN) & (red - green > RED_MARGIN) & (red - blue > RED_MARGIN)


def approach_start_index(action, tolerance=1e-3):
    """First step after the arm settles at the survey pose.

    Found from the commanded trajectory rather than assumed: the survey hold is the run
    of constant action that follows the survey move, and the approach is whatever comes
    after it.
    """
    from ammr_mycobot_interface import SURVEY_POSE

    survey = np.asarray(SURVEY_POSE, dtype=np.float64)
    at_survey = np.where(np.abs(action[:, :6] - survey).max(axis=1) < tolerance)[0]
    if len(at_survey) == 0:
        return 0
    return int(at_survey[-1])


def episode_report(episode_dir, visible_percent, decided_fraction):
    episode_dir = Path(episode_dir)
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    steps = metadata["num_steps"]

    first_visible = None
    columns = []
    for index in range(steps):
        with Image.open(episode_dir / "images" / f"{index:06d}.png") as image:
            mask = red_mask(image)
        fraction = mask.mean() * 100.0
        if fraction > visible_percent:
            if first_visible is None:
                first_visible = index
            columns.append(float(np.argwhere(mask)[:, 1].mean()))

    # Only the approach steers toward the target. The survey move is identical for every
    # placement, so counting it as part of the decision mismeasures episodes whose grasp
    # happens to sit near the survey heading: one cube landed 0.004 rad from it, its
    # joint1 barely moved during the approach, and 90% of the episode's total joint1
    # travel was reached during the survey move -- flagging a sound episode as bad.
    action = np.load(episode_dir / "action.npy")
    joint1 = action[:, 0]
    approach_start = approach_start_index(action)
    segment = joint1[approach_start:]
    moving = np.where(np.abs(np.diff(segment)) > 1e-4)[0]
    decided = int(approach_start + moving[int(len(moving) * decided_fraction)]) if len(moving) else None

    cube = (metadata.get("collection") or {}).get("sampled_object_pose")
    return {
        "episode_id": metadata["episode_id"],
        "cube_y": None if cube is None else float(cube[1]),
        "steps": steps,
        "first_visible": first_visible,
        "joint1_decided": decided,
        # A decided of None means joint1 never moved during the approach, which happens
        # when the survey heading already points at the cube. That is a correct episode,
        # not a missed decision, so it only has to clear the visibility requirement.
        "ordering_ok": (
            first_visible is not None and (decided is None or first_visible < decided)
        ),
        "visible_frames": len(columns),
        "visible_ratio": len(columns) / steps if steps else 0.0,
        "column_mean": float(np.mean(columns)) if columns else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Directory of episode_* folders.")
    parser.add_argument("--visible-percent", type=float, default=1.0,
                        help="Frame share of red pixels that counts as visible.")
    parser.add_argument("--decided-fraction", type=float, default=0.9,
                        help="Share of joint1 motion that counts as decided.")
    parser.add_argument("--limit", type=int, default=0, help="Only check this many episodes.")
    args = parser.parse_args()

    episodes = sorted(p for p in Path(args.root).iterdir()
                      if p.is_dir() and p.name.startswith("episode_"))
    if args.limit:
        episodes = episodes[: args.limit]
    if not episodes:
        print(f"no episodes under {args.root}")
        return 2

    reports = [episode_report(e, args.visible_percent, args.decided_fraction) for e in episodes]
    failed = [r for r in reports if not r["ordering_ok"]]

    print(f"{'episode':<16}{'cube y':>9}{'first_vis':>11}{'j1_decided':>12}{'vis%':>8}  order")
    for report in reports:
        mark = "ok" if report["ordering_ok"] else "FAIL"
        print(
            f"{report['episode_id']:<16}"
            f"{-9 if report['cube_y'] is None else report['cube_y']:>9.4f}"
            f"{str(report['first_visible']):>11}{str(report['joint1_decided']):>12}"
            f"{report['visible_ratio'] * 100:>7.0f}%  {mark}"
        )

    visible = [r for r in reports if r["first_visible"] is not None]
    margins = [
        r["joint1_decided"] - r["first_visible"]
        for r in reports
        if r["ordering_ok"] and r["joint1_decided"] is not None
    ]
    no_steer = sum(1 for r in reports if r["ordering_ok"] and r["joint1_decided"] is None)
    print()
    print(f"episodes            : {len(reports)}")
    print(f"target ever visible : {len(visible)}/{len(reports)}")
    print(f"ordering ok         : {len(reports) - len(failed)}/{len(reports)}")
    if no_steer:
        print(f"  ({no_steer} needed no joint1 change: the survey heading already aimed at the cube)")
    if margins:
        print(f"margin (frames between first sight and joint1 decided): "
              f"min {min(margins)}, median {int(np.median(margins))}, max {max(margins)}")
    if visible:
        ratios = [r["visible_ratio"] * 100 for r in visible]
        print(f"visible frame share : mean {np.mean(ratios):.0f}%, min {min(ratios):.0f}%")

    # The signal a policy has to read: cube y must map to a distinct image column.
    pairs = [(r["cube_y"], r["column_mean"]) for r in visible
             if r["cube_y"] is not None and r["column_mean"] is not None]
    if len(pairs) > 2:
        ys = np.array([p[0] for p in pairs])
        xs = np.array([p[1] for p in pairs])
        print(f"cube y vs image column: r = {np.corrcoef(ys, xs)[0, 1]:+.4f} "
              f"over {xs.min():.0f}..{xs.max():.0f} px")

    if failed:
        print()
        print("FAILED: joint1 is decided before the target is visible in")
        for report in failed:
            print(f"  {report['episode_id']}: first_visible={report['first_visible']}, "
                  f"joint1_decided={report['joint1_decided']}")
        return 1
    print()
    print("VISIBILITY CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
