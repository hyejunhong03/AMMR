# Zero-Shot Transfer Result, 2026-09-22

## Definition

This is not the old red/blue fine-tuned policy. The evaluated checkpoint is the
official `lerobot/smolvla_base` weight with AMMR-compatible config and processors,
without any AMMR gradient update.

Report name:

```text
zero-gradient AMMR-adapted SmolVLA base
```

## Result Summary

| Checkpoint | Image Mapping | Result File | Success | Velocity OK | Main Observation |
|---|---|---|---:|---:|---|
| `models/smolvla_base_ammr_zeroshot` | `observation.images.wrist` | `outputs/eval/zeroshot_smolvla_base_single_object_red10.json` | 0/10 | 0/10 | Wrong image-key convention caused unstable action output. Keep only as an ablation. |
| `models/smolvla_base_ammr_zeroshot_camera2_wrist` | `camera1=black`, `camera2=wrist`, `camera3=black` | `outputs/eval/zeroshot_smolvla_base_camera2_wrist_red10.json` | 0/10 | 10/10 | Action scale is stable, but the policy never commands a useful close. Use this as the official zero-shot baseline. |

## Measured Details

Old wrist-key adapter:

- `success`: 0/10
- `velocity_ok`: 0/10
- `max_joint_velocity`: 40.2 to 48.5 rad/s in the first inspected runs
- `raw_arm_step_max`: up to 621.75 rad
- `gripper_range`: 0.0053 to 0.0559 rad, never reaches close

Camera2 wrist adapter:

- `success`: 0/10
- `velocity_ok`: 10/10
- `max_joint_velocity`: about 0.32 rad/s
- `raw_arm_step_max`: up to 0.3503 rad before the rollout limiter
- `published_arm_step_max`: capped at 0.05 rad
- `gripper_range`: 0.0295 to 0.0343 rad, never reaches close
- `effective_hz`: about 9.18 Hz

## Interpretation

The first failure was largely an observation-key mismatch. SmolVLA's pretrained
processor expects the wrist view at `observation.images.camera2`; putting the AMMR
wrist image under a new `observation.images.wrist` key produced unusable actions.

After mapping wrist to `camera2`, the base model is safe enough to roll out but still
does not solve the AMMR task. The remaining failure is expected zero-shot transfer
failure from robot/action/gripper mismatch, not a training-data failure.

This gives the thesis a clean baseline:

1. Zero-shot transfer to AMMR: 0/10.
2. Fine-tuned seen-object performance: report separately.
3. Fine-tuned unseen-object performance: main research result.
