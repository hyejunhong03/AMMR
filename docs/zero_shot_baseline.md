# SmolVLA Zero-Shot Baseline for AMMR

## Purpose

Before fine-tuning `d8_unseen_single_object`, run a zero-gradient baseline to answer:

> How much of SmolVLA transfers to AMMR/myCobot without any AMMR gradient update?

This baseline is useful only if it uses the same runtime path as the fine-tuned
policy: wrist RGB image, 7-D robot state, natural language task, and 7-D absolute
joint target output.

## Important Definition

The official `lerobot/smolvla_base` checkpoint in the local Hugging Face cache is not
directly compatible with AMMR:

- Official base input: 3 cameras, 6-D state
- Official base output: 6-D action
- AMMR input: 1 wrist camera, 7-D state
- AMMR output: 7-D action, including gripper

Therefore the runnable AMMR baseline is:

> Official SmolVLA base weights + AMMR config/processor adapter, with no gradient
> fine-tuning.

This should be reported as a **zero-gradient AMMR-adapted SmolVLA base baseline**.
Do not call d6/d7 zero-shot; they were fine-tuned.

## Create the Baseline Checkpoint

```bash
cd /home/autolab/AMMR
python3 scripts/prepare_smolvla_zeroshot_checkpoint.py --overwrite
```

Default output:

```text
models/smolvla_base_ammr_zeroshot/
```

The official SmolVLA camera convention places the wrist view at
`observation.images.camera2`. For the AMMR wrist-only baseline, create the camera2
adapter variant and use that one for the reported zero-shot result:

```bash
cd /home/autolab/AMMR
python3 scripts/prepare_smolvla_zeroshot_checkpoint.py \
  --out models/smolvla_base_ammr_zeroshot_camera2_wrist \
  --camera-layout camera2_wrist \
  --overwrite
```

The generated checkpoint uses:

- `model.safetensors` from the local official `lerobot/smolvla_base` cache
- AMMR `config.json`
- AMMR `policy_preprocessor.json`
- AMMR `policy_postprocessor.json`
- AMMR normalizer/unnormalizer processor state

The model weights are symlinked by default to avoid duplicating ~1.2 GB.

## Static Check

Before opening Isaac Sim, run:

```bash
cd /home/autolab/AMMR
.venv/bin/python scripts/check_policy_outputs.py \
  --checkpoint models/smolvla_base_ammr_zeroshot_camera2_wrist \
  --dataset-root data/lerobot/pick_red_cube_val \
  --repo-id ammr/pick_red_cube_val \
  --task "pick the object" \
  --image-layout camera2_wrist \
  --device cuda
```

This must pass action shape, joint limit, and gripper range checks before rollout.

## Rollout Evaluation

Use the same path as fine-tuned policies.

```bash
cd /home/autolab/AMMR
scripts/run_mycobot_ros2_gui.sh \
  --headless \
  --ros2-state-hz 30 \
  --ros2-image-hz 15 \
  --hold-gripper-after-attach \
  --loop-profile 10
```

In another shell:

```bash
cd /home/autolab/AMMR
.venv/bin/python scripts/rollout_policy.py \
  --checkpoint models/smolvla_base_ammr_zeroshot_camera2_wrist \
  --poses-file data/grasp_poses_red_cube_focus.json \
  --episodes 10 \
  --object red_cube \
  --park-non-targets \
  --task "pick the object" \
  --image-layout camera2_wrist \
  --noise-mode fixed-random \
  --noise-seed 0 \
  --out outputs/eval/zeroshot_smolvla_base_camera2_wrist_red10.json
```

For unseen-object experiments, replace the pose file and object catalog once the
generic single-object scene is implemented.

## Reporting Rule

Report zero-shot results separately from fine-tuned results:

- `zero_gradient_base_success`
- `fine_tuned_seen_success`
- `fine_tuned_unseen_success`

Do not mix red/blue primitive fine-tuned checkpoints with zero-shot results.
