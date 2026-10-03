# Pick-Only Research Data Collection

## Goal

The thesis target is stable natural-language pick behavior, not place behavior. For the
next research dataset, keep the task to:

1. approach the commanded object,
2. close the gripper,
3. lift the object,
4. hold the lifted object.

Do not release or place during this dataset. Release/place adds another failure mode and
makes it harder to tell whether the policy failed at grasping or at post-grasp handling.

## Preset

Use `collect_episodes.py --collection-preset research_pick_only`.

The preset expands the previous loose collection criteria:

| Field | Legacy | Research pick-only |
| --- | ---: | ---: |
| `min_lift_m` | 0.02 | 0.05 |
| `min_hold_sec` | 0.2 | 2.0 |
| `close_hold` | 0.5 | 0.8 |
| `lift_hold` | 0.3 | 2.0 |
| `final_hold` | 0.2 | 1.0 |

The stricter lift and hold margins are intentional. A policy that barely crosses 20 mm
for one frame should not count as a stable pick.

## Pose Cache

The pose cache must be regenerated with a higher lift target. A cache generated with the
old 35 mm lift is not suitable for a 50 mm success threshold.

Start with 60 mm lift target and 50 mm success threshold:

```bash
cd ~/AMMR
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

/home/autolab/isaacsim/python.sh scripts/plan_grasp_poses.py \
  --object red_cube \
  --demo-pose p1 \
  --yaw-mode faces \
  --pre-approach-height 0.01 \
  --descent-steps 4 \
  --lift-height 0.06 \
  --object-position-file outputs/eval/phase1_grasp_funnel_source30_poses_h010.json \
  --out outputs/eval/phase2_red_cube_pick_only_lift060_poses.json
```

If too many positions fail planning at 60 mm, keep the failed/rejected summary and make a
second cache at 50 mm. Do not silently lower the evaluation threshold without recording
the reason.

## Pilot Collection

Run a 5 episode pilot before collecting the full dataset:

```bash
cd ~/AMMR
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

.venv/bin/python scripts/collect_episodes.py \
  --poses-file outputs/eval/phase2_red_cube_pick_only_lift060_poses.json \
  --count 5 \
  --object red_cube \
  --park-non-targets \
  --instruction "pick the red cube" \
  --task-name pick_red_cube_pick_only_lift060 \
  --output-dir data/smolvla_raw_pick_only_lift060_pilot \
  --collection-preset research_pick_only
```

Accept the pilot only if:

- `demo_exit=0` for the successful episodes,
- `lift_delta_m >= 0.05`,
- the hold validator passes for at least 2 seconds,
- max arm velocity stays below the configured limit,
- no pre-close collision/interference is observed in diagnostics.

## Full Collection

After the pilot passes, collect 60 to 120 episodes first. Do not jump directly to object
diversity until this pick-only baseline is stable.

```bash
.venv/bin/python scripts/collect_episodes.py \
  --poses-file outputs/eval/phase2_red_cube_pick_only_lift060_poses.json \
  --count 60 \
  --object red_cube \
  --park-non-targets \
  --instruction "pick the red cube" \
  --task-name pick_red_cube_pick_only_lift060 \
  --output-dir data/smolvla_raw_pick_only_lift060 \
  --collection-preset research_pick_only
```

## Evaluation Notes

Report both:

- relaxed pick success, for comparison with older 20 mm experiments,
- strict pick success, using the research pick-only threshold.

For rollout, keep gripper snap enabled if the checkpoint was trained on binary gripper
commands. Do not use gripper latch as a default policy evaluation setting unless it is
explicitly part of the proposed method, because previous tests showed latch can damage
the closed-loop rollout distribution.
