#!/usr/bin/env python3
"""Collect SmolVLA episodes back to back against a running Isaac scene.

Start the simulator once and leave it up; startup costs about a minute, so looping
here instead of relaunching per episode is what makes batch collection practical.

Recording is stopped as soon as the demo finishes. Letting the recorder run to a fixed
duration appended a long stretch of a motionless robot to every episode -- in one
measured case 27 s of the 37 s recorded, 73% of the steps, all carrying an action
identical to the previous one.

Success is judged from the data by evaluate_pick_episode.py rather than asserted on the
command line, and written back into metadata.json.
"""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from ammr_mycobot_interface import (
    OBJECT_INDEX_BY_NAME,
    OBJECT_NAMES,
    TASK_OBJECT_CATALOG_VERSION,
    TASK_OBJECT_TABLE_Z,
    task_object_spec,
)

SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = "/home/autolab/AMMR/data/smolvla_raw"
RESET_SETTLE_SEC = 6.0
PLACEMENT_SETTLE_SEC = 2.0
RECORDER_START_SEC = 3.0
RECORDER_STOP_TIMEOUT_SEC = 20.0
DEMO_LOG_TAIL_LINES = 80

LEGACY_COLLECTION_DEFAULTS = {
    "min_lift_m": 0.02,
    "min_hold_sec": 0.2,
    "close_hold": 0.5,
    "lift_hold": 0.3,
    "final_hold": 0.2,
}

RESEARCH_PICK_ONLY_DEFAULTS = {
    # Keep the task to "pick and hold" so failures are not mixed with release/place
    # behavior. The larger lift and hold margins make weak grasps visible early.
    "min_lift_m": 0.05,
    "min_hold_sec": 2.0,
    "close_hold": 0.8,
    "lift_hold": 2.0,
    "final_hold": 1.0,
}

POSE_ENTRY_PASSTHROUGH_KEYS = (
    "source_pilot30_index",
    "phase1_repeat_index",
    "phase1_dataset_role",
    "phase1_source_validator",
    "phase1_source_result_status",
    "phase1_training_candidate",
    "source_validation_method",
    "source_diagnostics_row_start",
    "source_diagnostics_row_count",
    "source_close_start_row",
    "source_preclose_interaction",
    "source_lift_hold_sec",
    "source_max_arm_velocity_rad_s",
    "source_max_gripper_mimic_error_rad",
)

NON_TARGET_PARK_POSITIONS = {
    name: (0.55, 0.25 if index % 2 == 0 else -0.25, TASK_OBJECT_TABLE_Z[name])
    for index, name in enumerate(OBJECT_NAMES)
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5, help="Number of episodes to collect.")
    parser.add_argument(
        "--poses-file",
        default=None,
        help=(
            "Pose cache from plan_grasp_poses.py. Each accepted entry becomes one "
            "episode: the object is moved to its position and the IK joints drive the "
            "pick, which is what removes the per-placement slider session."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Visit pose-cache entries in random order.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    parser.add_argument("--demo-pose", default="p1", help="Pose profile; the scene must use the same one.")
    parser.add_argument("--object", choices=OBJECT_NAMES, default="red_cube", help="Target object.")
    parser.add_argument(
        "--park-non-targets",
        action="store_true",
        help=(
            "Move every non-target task object out of the workspace before each "
            "episode. Use this for single-object datasets so grasp assist cannot "
            "attach a distractor object first."
        ),
    )
    parser.add_argument("--instruction", default=None, help="Language instruction.")
    parser.add_argument("--task-name", default=None, help="Defaults to pick_<object>_<demo-pose>.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Root directory for raw episodes.")
    parser.add_argument(
        "--collection-preset",
        choices=("legacy", "research_pick_only"),
        default="legacy",
        help=(
            "Collection/evaluation preset. 'legacy' keeps the previous loose lift/hold "
            "timing; 'research_pick_only' records pick-and-hold episodes with stricter "
            "lift and hold criteria for thesis experiments."
        ),
    )
    parser.add_argument("--min-lift-m", type=float, default=None, help="Lift required to count as a success.")
    parser.add_argument(
        "--min-hold-sec",
        type=float,
        default=None,
        help="Continuous time the object must remain above the lift threshold to count as success.",
    )
    parser.add_argument("--demo-timeout", type=float, default=120.0, help="Seconds before a demo run is abandoned.")
    parser.add_argument(
        "--close",
        type=float,
        default=-0.245,
        help="Gripper close target passed to scripted_pick_demo.py.",
    )
    parser.add_argument(
        "--arm-speed",
        type=float,
        default=0.18,
        help="Arm speed passed to scripted_pick_demo.py. Slower defaults are used for dynamic contact.",
    )
    parser.add_argument(
        "--lift-arm-speed",
        type=float,
        default=0.04,
        help="Lift/lower arm speed passed to scripted_pick_demo.py.",
    )
    parser.add_argument(
        "--arm-tolerance",
        type=float,
        default=0.03,
        help="Arm settling tolerance passed to scripted_pick_demo.py.",
    )
    parser.add_argument(
        "--state-timeout",
        type=float,
        default=14.0,
        help="Per-segment state wait timeout passed to scripted_pick_demo.py.",
    )
    parser.add_argument(
        "--post-gripper-ready-arm-settle-timeout",
        type=float,
        default=20.0,
        help="Post-close arm settling timeout passed to scripted_pick_demo.py.",
    )
    parser.add_argument(
        "--object-position-guard-tolerance",
        type=float,
        default=0.003,
        help="Object placement guard tolerance passed to scripted_pick_demo.py.",
    )
    parser.add_argument(
        "--close-hold",
        type=float,
        default=None,
        help="Seconds to hold after closing the gripper before lift.",
    )
    parser.add_argument(
        "--lift-hold",
        type=float,
        default=None,
        help="Seconds to hold the lifted pose inside scripted_pick_demo.py.",
    )
    parser.add_argument(
        "--final-hold",
        type=float,
        default=None,
        help="Final hold seconds before scripted_pick_demo.py exits.",
    )
    parser.add_argument(
        "--place-after-lift",
        action="store_true",
        help="Lower and release after lift. Useful for dynamic-contact validation datasets.",
    )
    parser.add_argument(
        "--no-park-target-before-reset",
        dest="park_target_before_reset",
        action="store_false",
        help=(
            "Do not move the target object to a safe park pose before resetting the arm. "
            "The default avoids placing a new object under an arm that is still returning home."
        ),
    )
    parser.add_argument(
        "--max-record-sec",
        type=float,
        default=180.0,
        help="Recorder safety cap. Normal termination is the demo finishing, not this.",
    )
    args = parser.parse_args()
    # Every child runs with cwd=scripts, so a relative path given here resolves against
    # scripts/ inside them and against the working directory out here. An output dir
    # sent that way makes the run die reading a metadata.json written somewhere else,
    # and a pose cache sent that way makes all sixty demos exit 1 on FileNotFoundError.
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    if args.poses_file:
        args.poses_file = str(Path(args.poses_file).expanduser().resolve())
    apply_collection_preset(args, parser)
    return args


def apply_collection_preset(args, parser=None):
    defaults = (
        RESEARCH_PICK_ONLY_DEFAULTS
        if args.collection_preset == "research_pick_only"
        else LEGACY_COLLECTION_DEFAULTS
    )
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.collection_preset == "research_pick_only" and args.place_after_lift:
        message = "research_pick_only preset must not be combined with --place-after-lift"
        if parser is not None:
            parser.error(message)
        raise ValueError(message)


def ros_env():
    """rclpy needs the ROS2 environment; inherit whatever the caller sourced."""
    import os

    env = dict(os.environ)
    env.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    return env


def run(cmd, timeout=None, capture=True):
    return subprocess.run(
        cmd,
        cwd=SCRIPTS_DIR,
        env=ros_env(),
        timeout=timeout,
        capture_output=capture,
        text=True,
        check=False,
    )


def default_instruction(object_name):
    instructions = {
        "red_cube": "pick the red cube",
        "blue_cylinder": "pick the blue cylinder",
    }
    return instructions.get(object_name, f"pick the {object_name.replace('_', ' ')}")


def place_object(position, object_name):
    """Move a task object before the reset, so the reset restores it there."""
    object_index = OBJECT_INDEX_BY_NAME[object_name]
    data = ", ".join(str(float(v)) for v in [object_index, *position])
    run(
        [
            "ros2", "topic", "pub", "--once", "/ammr/set_object_pose",
            "std_msgs/Float64MultiArray", f"{{data: [{data}]}}",
        ],
        timeout=15.0,
    )
    time.sleep(PLACEMENT_SETTLE_SEC)


def park_non_target_objects(target_name):
    for object_name in OBJECT_NAMES:
        if object_name == target_name:
            continue
        place_object(NON_TARGET_PARK_POSITIONS[object_name], object_name)


def load_pose_cache(path):
    payload = json.loads(Path(path).read_text())
    entries = [e for e in payload.get("entries", []) if e.get("solved")]
    if not entries:
        raise ValueError(f"{path} has no accepted entries")
    return payload, entries


def reset_scene():
    run(
        ["ros2", "topic", "pub", "--once", "/ammr/reset_task_scene", "std_msgs/Empty", "{}"],
        timeout=15.0,
    )
    time.sleep(RESET_SETTLE_SEC)


def stop_recorder(process):
    """SIGINT, then escalate. The recorder finalizes the episode on SIGINT."""
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=RECORDER_STOP_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()


def episode_from_log(text):
    import re

    found = re.findall(r"episode_\d{6}", text or "")
    return found[-1] if found else None


def score_episode(episode_dir, min_lift_m, min_hold_sec, object_name):
    """Pass/fail from the recorded object trajectory, not from the demo's own opinion."""
    result = run(
        [
            sys.executable,
            "evaluate_pick_episode.py",
            str(episode_dir),
            "--object",
            object_name,
            "--min-lift-m",
            str(min_lift_m),
            "--min-hold-sec",
            str(min_hold_sec),
        ],
        timeout=60.0,
    )
    lift = None
    for line in (result.stdout or "").splitlines():
        if line.startswith("lift_delta"):
            # Reported as "lift_delta: 0.0335 m" -- drop the unit before parsing.
            try:
                lift = float(line.split(":", 1)[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
    return result.returncode == 0, lift


def classify_demo_failure(text, demo_exit):
    if demo_exit == 0:
        return None
    if demo_exit == 124:
        return "demo_timeout"
    if not text:
        return "demo_failed_no_output"
    if "did not reach expected object_state" in text:
        return "object_position_guard_timeout"
    if "close skipped: approach did not settle" in text:
        return "approach_not_settled"
    if "close skipped: no approach feedback" in text:
        return "approach_no_feedback"
    if "arm did not settle before lift" in text or "lift skipped: arm did not settle" in text:
        return "post_grasp_arm_not_settled"
    if "gripper did not close enough" in text:
        return "gripper_not_closed"
    return "demo_failed"


def write_success(episode_dir, success, lift, demo_exit, entry=None, *, args=None, demo_log_path=None, demo_output=None):
    path = Path(episode_dir) / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["success"] = bool(success)
    failure_reason = classify_demo_failure(demo_output, demo_exit)
    collection = {
        "demo_exit_code": demo_exit,
        "lift_delta_m": lift,
        "failure_reason": failure_reason,
        "scored_by": "evaluate_pick_episode.py",
    }
    if args is not None:
        collection.update(
            {
                "executed_close_rad": args.close,
                "arm_speed": args.arm_speed,
                "lift_arm_speed": args.lift_arm_speed,
                "arm_tolerance": args.arm_tolerance,
                "state_timeout": args.state_timeout,
                "post_gripper_ready_arm_settle_timeout": args.post_gripper_ready_arm_settle_timeout,
                "object_position_guard_tolerance": args.object_position_guard_tolerance,
                "collection_preset": args.collection_preset,
                "close_hold": args.close_hold,
                "lift_hold": args.lift_hold,
                "final_hold": args.final_hold,
                "min_lift_m": args.min_lift_m,
                "min_hold_sec": args.min_hold_sec,
                "place_after_lift": bool(args.place_after_lift),
                "park_target_before_reset": bool(args.park_target_before_reset),
            }
        )
    if demo_log_path is not None:
        collection["demo_log_path"] = str(demo_log_path)
    if demo_output:
        collection["demo_log_tail"] = demo_output.strip().splitlines()[-DEMO_LOG_TAIL_LINES:]
    if entry is not None:
        object_spec = task_object_spec(entry.get("object_name", entry.get("target_object", "red_cube")))
        # Carried through so a later analysis can relate success to where the object
        # was and how well IK placed the tool, without re-deriving any of it.
        collection.update(
            {
                "catalog_version": entry.get("catalog_version", TASK_OBJECT_CATALOG_VERSION),
                "object_id": entry.get("object_id", object_spec["object_id"]),
                "sampled_object_pose": entry.get("sampled_object_pose", entry["cube_position"]),
                "target_object": entry.get("target_object", entry.get("object_name", "red_cube")),
                "object_name": entry.get("object_name", "red_cube"),
                "object_index": entry.get("object_index"),
                "shape_class": entry.get("shape_class", object_spec["shape_class"]),
                "seen_split": entry.get("seen_split", object_spec.get("seen_split", "unknown")),
                "object_size_m": entry.get("object_size_m", object_spec.get("size_m")),
                "object_radius_m": entry.get("object_radius_m", object_spec.get("radius_m")),
                "object_height_m": entry.get("object_height_m", object_spec.get("height_m")),
                "object_color_rgb": entry.get("object_color_rgb", object_spec.get("color_rgb")),
                "object_position": entry.get("object_position", entry.get("cube_position")),
                "accepted_by_ik": entry.get("accepted_by_ik", True),
                "ik_tcp_error": entry.get("ik_tcp_error"),
                "distance_from_base_m": entry.get("distance_from_base_m"),
                "grasp_family": entry.get("grasp_family"),
                "reject_reason": entry.get("reject_reason"),
            }
        )
        for key in POSE_ENTRY_PASSTHROUGH_KEYS:
            if key in entry:
                collection[key] = entry[key]
    metadata["collection"] = collection
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def collect_one(args, index, entry=None):
    task_name = args.task_name or f"pick_{args.object}_{args.demo_pose}"
    instruction = args.instruction or default_instruction(args.object)
    log_dir = Path(args.output_dir) / "_collection_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Reset should bring the arm and gripper home before the new target object is placed
    # in the workspace. If we place first and reset second, the arm can return through the
    # new object and move it before the demo even starts.
    if args.park_non_targets:
        park_non_target_objects(args.object)
    if entry is not None and args.park_target_before_reset:
        place_object(NON_TARGET_PARK_POSITIONS[args.object], args.object)
    reset_scene()
    if entry is not None:
        place_object(entry.get("object_position", entry["cube_position"]), args.object)

    recorder = subprocess.Popen(
        [
            sys.executable,
            "record_smolvla_episode.py",
            "--instruction", instruction,
            "--task-name", task_name,
            "--output-dir", args.output_dir,
            "--duration-sec", str(args.max_record_sec),
            "--gripper-close-rad", str(args.close),
        ],
        cwd=SCRIPTS_DIR,
        env=ros_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(RECORDER_START_SEC)

    try:
        demo_cmd = [sys.executable, "scripted_pick_demo.py", "--demo-pose", args.demo_pose]
        demo_cmd += ["--object", args.object]
        if args.poses_file:
            demo_cmd += ["--poses-file", args.poses_file, "--no-check-cube"]
            if entry is not None:
                object_position = entry.get("object_position", entry["cube_position"])
                demo_cmd += [
                    "--expected-object-position",
                    *[str(float(value)) for value in object_position],
                    "--object-position-guard-tolerance",
                    str(args.object_position_guard_tolerance),
                    "--object-position-guard-timeout",
                    "8.0",
                    "--object-position-guard-samples",
                    "2",
                ]
        demo_cmd += [
            "--close",
            str(args.close),
            "--arm-speed",
            str(args.arm_speed),
            "--lift-arm-speed",
            str(args.lift_arm_speed),
            "--arm-tolerance",
            str(args.arm_tolerance),
            "--state-timeout",
            str(args.state_timeout),
            "--post-gripper-ready-arm-settle-timeout",
            str(args.post_gripper_ready_arm_settle_timeout),
            "--close-hold",
            str(args.close_hold),
            "--lift-hold",
            str(args.lift_hold),
            "--final-hold",
            str(args.final_hold),
        ]
        if args.place_after_lift:
            demo_cmd += ["--place-after-lift"]
        demo = run(demo_cmd, timeout=args.demo_timeout)
        demo_exit = demo.returncode
        demo_output = (demo.stdout or "") + (demo.stderr or "")
        if demo_exit != 0:
            # Without this a failed batch reports nothing but the exit code, and the
            # reason the demo refused -- gripper never closed, arm never settled -- has
            # already been thrown away by the time anyone looks.
            for line in demo_output.strip().splitlines()[-6:]:
                print(f"    demo: {line}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        demo_exit = 124
        demo_output = f"demo timed out after {args.demo_timeout:.1f}s"

    stop_recorder(recorder)
    recorder_log = recorder.stdout.read() if recorder.stdout else ""

    episode_id = episode_from_log(recorder_log)
    if episode_id is None:
        print(f"  run {index}: recorder produced no episode", file=sys.stderr)
        return None

    episode_dir = Path(args.output_dir) / episode_id
    demo_log_path = log_dir / f"{episode_id}_demo.log"
    with open(demo_log_path, "w", encoding="utf-8") as handle:
        handle.write(demo_output or "")
        if demo_output and not demo_output.endswith("\n"):
            handle.write("\n")
        handle.write("\n--- recorder log ---\n")
        handle.write(recorder_log or "")

    success, lift = score_episode(episode_dir, args.min_lift_m, args.min_hold_sec, args.object)
    write_success(
        episode_dir,
        success,
        lift,
        demo_exit,
        entry,
        args=args,
        demo_log_path=demo_log_path,
        demo_output=demo_output,
    )

    lift_text = "n/a" if lift is None else f"{lift * 1000:.1f}mm"
    failure_reason = classify_demo_failure(demo_output, demo_exit)
    reason_text = "" if failure_reason is None else f"  reason={failure_reason}"
    print(f"  run {index}: {episode_id}  demo_exit={demo_exit}  lift={lift_text}  success={success}{reason_text}")
    return {
        "episode_id": episode_id,
        "demo_exit": demo_exit,
        "lift": lift,
        "success": success,
        "failure_reason": failure_reason,
    }


def main():
    args = parse_args()

    entries = [None] * args.count
    if args.poses_file:
        _, accepted = load_pose_cache(args.poses_file)
        accepted = [
            entry for entry in accepted
            if entry.get("object_name", "red_cube") == args.object
        ]
        if args.shuffle:
            import random

            random.Random(args.seed).shuffle(accepted)
        if len(accepted) < args.count:
            print(
                f"pose cache holds {len(accepted)} accepted entries but {args.count} "
                f"episodes were requested; collecting {len(accepted)}",
                file=sys.stderr,
            )
        entries = accepted[: args.count]

    results = []
    for index, entry in enumerate(entries, start=1):
        print(f"=== run {index}/{len(entries)} ===")
        outcome = collect_one(args, index, entry)
        if outcome is not None:
            if entry is not None:
                outcome["object"] = args.object
                outcome["object_position"] = entry.get("object_position", entry["cube_position"])
                outcome["cube"] = entry["cube_position"]
                outcome["radius"] = entry.get("distance_from_base_m")
                for key in POSE_ENTRY_PASSTHROUGH_KEYS:
                    if key in entry:
                        outcome[key] = entry[key]
            results.append(outcome)

    successes = [r for r in results if r["success"]]
    print()
    print(f"collected {len(results)}/{args.count}, success {len(successes)}/{len(results)}")
    for r in results:
        if not r["success"]:
            print(f"  failed: {r['episode_id']} (demo_exit={r['demo_exit']})")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
