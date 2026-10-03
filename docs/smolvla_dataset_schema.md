# AMMR SmolVLA Dataset Schema

Frozen contract for every episode recorded into `data/smolvla_raw/`. Everything downstream —
`record_smolvla_episode.py`, `validate_smolvla_episode.py`, the LeRobot converter, and the policy
runner — follows this document. Change it here first, then change the code.

## Action

```
action[t] = the absolute joint target to publish on the next control cycle
            NOT a delta, NOT a velocity
```

| Field | Value |
| --- | --- |
| `action` dim | 7 (6 arm joints + 1 gripper) |
| Unit | radian, every channel including the gripper |
| Arm channels 0-5 | Absolute joint angle targets, in `ARM_JOINTS` order |
| Gripper channel 6 | Commanded gripper target in radians |
| ROS2 topic | `/ammr/arm/joint_targets` (`std_msgs/Float64MultiArray`, length 7) |

Channel order is `ammr_mycobot_interface.CONTROL_JOINTS`:

```
joint2_to_joint1, joint3_to_joint2, joint4_to_joint3,
joint5_to_joint4, joint6_to_joint5, joint6output_to_joint6,
gripper_controller
```

### Gripper command values

| Name | Value | Constant |
| --- | --- | --- |
| Open | `0.08` | `GRIPPER_OPEN_RAD` |
| Kinematic/grasp-assist close | `-0.18` | `GRIPPER_GRASP_CLOSE_RAD` = `COMMAND_GRIPPER_CLOSE_RAD` |
| Dynamic-contact red cube close | `-0.245` | episode `metadata.collection.executed_close_rad` |
| Clamp range | `(-0.38, 0.10)` | `SIM_GRIPPER_LIMITS` |

For legacy kinematic/grasp-assist episodes, close is **-0.18**, not the
`GRIPPER_STABLE_CLOSE_RAD = -0.35` that also exists in the interface module. `-0.35` is never issued
as a command anywhere. Three reasons to keep `-0.18` for that profile:

- Legacy recorded episodes use `[-0.18, 0.08]` for the gripper action.
- `real_mycobot_adapter.py` maps `COMMAND_GRIPPER_CLOSE_RAD` (-0.18) to hardware gripper value 0,
  so changing it silently rescales the real robot's grasp.
- The interface module warns that the raw controller limit is -0.74..0.15 but "large close targets
  make the imported linkage unstable" — the same failure mode as the wrist instability seen during
  pose measurement.

For dynamic-contact physical grasp episodes, the close value is part of the collection profile and
must be recorded in `metadata.collection.executed_close_rad`. The Phase 1 red-cube contact dataset
uses `-0.245` because `-0.18` did not create bilateral contact force on the 30 mm cube.
`validate_smolvla_episode.py` checks `action.gripper_close_rad` against the episode's executed close
value when that metadata is present, and falls back to `COMMAND_GRIPPER_CLOSE_RAD` for legacy data.

## Observation

| Field | Value |
| --- | --- |
| `observation.state` dim | 7, the same space as `action` |
| `observation.state` unit | radian |
| Image | **256x256** RGB |
| ROS2 state topic | `/ammr/robot_state` (`std_msgs/Float64MultiArray`, length 7) |
| ROS2 image topic | `/ammr/hand_camera/image_raw` (`sensor_msgs/Image`) |

`action.npy` is the **commanded** action — what a policy must learn to output.
`robot_state.npy` is the **observed applied state** — what the simulator or robot actually reached.
They are intentionally not identical: the gripper and the arm both lag their commands, and that lag
is the physical response being modelled, not an error.

The real wrist camera must eventually use the same preprocessing (resolution, crop, normalisation)
as the simulated one, or a policy trained here will not transfer.

## Privileged data

Simulator ground truth that does **not** exist on the real robot. Recorded for evaluation and for
automatic demonstration generation; **never fed to a policy**.

```json
"privileged": {
    "object_state_topic": "/ammr/object_states",
    "object_state_dim": 6,
    "object_state_names": ["red_cube_x", ..., "blue_cylinder_z"],
    "quat_order": "wxyz",
    "policy_input": false,
    "note": "Simulator ground truth. Not available on the real robot."
}
```

This block lives at the top level of `metadata.json`, deliberately **outside** `observation`, so a
converter that walks `observation` cannot sweep it into the policy input by accident.

Legitimate uses: success labelling (`evaluate_pick_episode.py`), grasp pose computation for scripted
demo generation, and debugging. If a policy ever needs it, use a teacher-student or asymmetric
actor-critic setup rather than feeding it directly.

## Task string

`task` carries the same string as `instruction`. One canonical phrasing per task:

```
"pick the red cube"
"pick the blue cylinder"
```

Paraphrase augmentation can come later; the initial dataset keeps one string per task.

## Quaternion order

```
quat_order = "wxyz"
```

ROS, Isaac, and SciPy each default to a different ordering, so it is stated in the metadata and used
consistently in code and topics — including the `/ammr/set_object_pose` payload.

## Files per episode

| File | Shape | Notes |
| --- | --- | --- |
| `images/%06d.png` | 256x256x3 | Contiguous from `000000.png` |
| `robot_state.npy` | `[T, 7]` float32 | Observed applied state |
| `action.npy` | `[T, 7]` float32 | Commanded action |
| `object_state.npy` | `[T, 6]` float32 | Privileged; excluded from training |
| `image_timestamp.npy` | `[T]` float64 | Image arrival time |
| `state_timestamp.npy` | `[T]` float64 | State arrival time |
| `action_timestamp.npy` | `[T]` float64 | Action arrival time |
| `timestamps.npy` | `[T]` float64 | Kept for backward compatibility |
| `metadata.json` | — | Structure above |

For object-randomized collection, `metadata.json["collection"]` additionally records the target
object metadata used for evaluation and debugging:

```json
{
    "target_object": "red_cube",
    "object_name": "red_cube",
    "object_index": 0,
    "object_position": [0.22, -0.08, 0.115],
    "sampled_object_pose": [0.22, -0.08, 0.115],
    "ik_tcp_error": 0.0006,
    "grasp_family": "canonical_p1"
}
```

These fields are metadata only. The policy still receives only wrist image, language, and robot
state.

### Why three timestamps

A single timestamp cannot tell you how far the state lags the action. The three arrival times let
you measure the lag `k` in `action[t] ~= state[t+k]` per channel and confirm it is consistent across
all seven, which is the check that catches a channel being rate-limited somewhere it should not be.

## Training environment

`lerobot` lives in a project virtualenv, deliberately separate from the system Python. Isaac Sim and
the ROS2 tooling run on the system interpreter with `torch 2.10.0+cu128`; installing `lerobot`
there risks pip resolving a different torch under them.

```bash
cd /home/autolab/AMMR
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/pip install lerobot
```

Roughly 8 GB and 20 minutes. Versions in use:

| Package | Version |
| --- | --- |
| lerobot | 0.4.4 |
| torch / torchvision | 2.10.0 / 0.25.0 |
| datasets | 4.8.5 |
| numpy | 2.2.6 |
| pyarrow | 25.0.1 |

Which interpreter to use:

| Task | Interpreter |
| --- | --- |
| Recording, the scripted demo, batch collection, validation | system `python3` (needs ROS2) |
| `convert_to_lerobot.py`, training | `.venv/bin/python` |
| Isaac Sim scripts | `/home/autolab/isaacsim/python.sh` |

`convert_to_lerobot.py --dry-run` is the exception: it validates without importing `lerobot`, so it
runs on the system interpreter too.

## Conversion

```bash
cd /home/autolab/AMMR/scripts
../.venv/bin/python convert_to_lerobot.py \
    --only-success \
    --root /home/autolab/AMMR/data/lerobot/pick_red_cube
```

Three guards refuse rather than quietly produce a bad dataset:

- **Mixed task strings.** More than one `task` across the selected episodes aborts the run. Early
  episodes say "pick up the red cube" and later ones "pick the red cube"; converting both would
  teach one task under two phrasings. `--allow-mixed-tasks` overrides. Use that override only when
  the variation is intentional, for example the multi-primitive dataset containing exactly
  `"pick the red cube"` and `"pick the blue cylinder"`.
- **Image size.** Only episodes recorded at exactly 256x256 are taken. The older 640x480 frames are
  4:3, so forcing them square squashes them horizontally and the same scene reaches the policy with
  two different geometries. `--any-image-size` overrides, and is only sound when the aspect ratios
  already agree.
- **Schema version.** An episode that has not been migrated is skipped with a pointer to
  `migrate_episode_metadata.py`.

`privileged` never reaches the dataset by construction: the converter does not read
`object_state.npy` at all, and only mentions the key to check that none leaked into `observation`.
