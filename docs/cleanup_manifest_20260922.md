# AMMR Cleanup Manifest 2026-09-22

This manifest is the cleanup gate for `/home/autolab/AMMR`. Do not delete data or
checkpoints just because they are listed here. Delete only after the item is moved
from `delete_candidate` to an explicit deletion batch.

## Current Disk Use

Measured on 2026-09-22:

| Path | Size | Notes |
| --- | ---: | --- |
| `outputs/train` | 30G | Fine-tuning checkpoints; biggest cleanup target |
| `data` | 8.9G | Raw episodes, LeRobot datasets, traces, backups |
| `.venv` | 8.2G | LeRobot/SmolVLA environment |
| `isaac_usd` | 105M | Isaac assets |
| `scripts` | 964K | Pipeline code |
| `docs` | 24K | Project documentation |

## Keep

These are required for the current unseen-object research direction or for safe
reproduction of the known-good pipeline.

- `scripts/`
- `docs/`
- `isaac_urdf/`
- `.venv/`
- `models/smolvla_base_pinned/`
- `models/smolvla_base_ammr_zeroshot/` once generated
- `isaac_usd/mycobot_280_m5_adaptive_gripper_reimport/`
- `outputs/eval/`
- `outputs/train/d6_survey_focus_extra_from_d3_cuda/checkpoints/003000/`
- `data/grasp_poses_red_cube_focus.json`
- `data/grasp_poses_blue_cylinder_smoke20.json`
- `data/grasp_poses_blue_cylinder_train100.json`
- `data/lerobot/pick_survey_focus_extra/`
- `data/lerobot/pick_red_cube/`
- `data/lerobot/pick_blue_cylinder/`

## Archive Candidates

These are not on the current main research path, but they may still be useful for
method history, ablations, or education material.

- `outputs/train/d3_small/`
- `outputs/train/d3_smoke/`
- `outputs/train/d3_survey/`
- `outputs/train/d4_faces/`
- `outputs/train/d5_survey_focus_from_d3_cuda/`
- `outputs/train/d7_red_blue_balanced_from_d6_cuda_3000/`
- `outputs/train/d7_red_blue_balanced_from_d6_cuda_smoke/`
- `outputs/train/d6_survey_focus_extra_from_d3_cuda/checkpoints/001000/`
- `outputs/train/d6_survey_focus_extra_from_d3_cuda/checkpoints/002000/`
- `data/smolvla_raw/`
- `data/smolvla_survey/`
- `data/smolvla_faces/`
- `data/smolvla_red_cube_focus/`
- `data/smolvla_raw_train_extra_focus_exclude_focus20/`
- `data/smolvla_blue_cylinder_train100_isolated/`
- `data/rollout_traces/`

## Delete Candidates

These are either known-bad, temporary, or reproducible from preserved sources. They
should still be deleted only in an explicit cleanup pass.

- `scripts/__pycache__/`
- `data/smolvla_blue_cylinder_train100/` — known wrong-object contamination
- `data/smolvla_raw_metadata_backup_20260917_165407/`
- `data/smolvla_raw_metadata_backup_20260917_205340/`
- `data/smolvla_blue_cylinder_smoke/`
- `data/smolvla_blue_cylinder_isolated_smoke5/`
- `data/debug_gripper_fix/`
- `data/validation_action_limiter_004/`
- `data/validation_action_limiter_005/`
- `data/validation_controller_only_rollout/`
- `data/validation_controller_only_scripted_20/`

## Cleanup Rule

Before deleting a candidate:

1. Confirm it is not referenced by the current zero-shot, d8 unseen-object, or d6
   regression workflow.
2. Record the exact deletion command in a cleanup note.
3. Run `du -h --max-depth=1 data outputs` before and after the cleanup.
4. Never delete `.venv`, `scripts`, `docs`, or the d6 `003000` checkpoint as part
   of storage cleanup.
