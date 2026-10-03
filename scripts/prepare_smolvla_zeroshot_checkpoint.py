#!/usr/bin/env python3
"""Create an AMMR-compatible zero-gradient SmolVLA baseline checkpoint.

The upstream ``lerobot/smolvla_base`` cache is not directly runnable on this
project: it declares three camera inputs and 6-D state/action, while AMMR uses a
single wrist camera and 7-D state/action including the gripper. For a fair
pre-finetuning baseline, keep the upstream base weights untouched but wrap them
with the AMMR config and processor files that define this project's interface and
normalisation.

This is not a trained red/blue checkpoint. It is base SmolVLA weights plus AMMR
I/O adapters, so it measures zero-gradient transfer through the same runtime path
that later fine-tuned checkpoints use.
"""

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = Path(
    "/home/autolab/.cache/huggingface/hub/models--lerobot--smolvla_base/"
    "snapshots/d9f33c94a60fb382c90dea2164c96845bd955e28"
)
DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "outputs/train/d6_survey_focus_extra_from_d3_cuda/checkpoints/003000/pretrained_model"
)
DEFAULT_OUT = PROJECT_ROOT / "models/smolvla_base_ammr_zeroshot"

REQUIRED_BASE_FILES = ["model.safetensors"]
REQUIRED_REFERENCE_FILES = [
    "config.json",
    "policy_preprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(DEFAULT_BASE), help="Upstream SmolVLA base checkpoint.")
    parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE),
        help="AMMR checkpoint providing config and processor files only.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output checkpoint directory.")
    parser.add_argument(
        "--camera-layout",
        choices=("ammr_wrist", "camera2_wrist"),
        default="ammr_wrist",
        help=(
            "ammr_wrist keeps the existing AMMR single-camera key. camera2_wrist "
            "uses SmolVLA's convention: camera1=dummy top, camera2=wrist, "
            "camera3=dummy additional view."
        ),
    )
    parser.add_argument(
        "--copy-weights",
        action="store_true",
        help="Copy model.safetensors instead of symlinking it.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def require_files(root, filenames, label):
    missing = [name for name in filenames if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{label} missing required files: {missing}")


def load_json(path):
    return json.loads(path.read_text())


def validate_ammr_config(config):
    input_features = config.get("input_features") or {}
    output_features = config.get("output_features") or {}
    image = input_features.get("observation.images.wrist")
    state = input_features.get("observation.state")
    action = output_features.get("action")
    errors = []
    if not image or image.get("shape") != [3, 256, 256]:
        errors.append("expected observation.images.wrist shape [3, 256, 256]")
    if not state or state.get("shape") != [7]:
        errors.append("expected observation.state shape [7]")
    if not action or action.get("shape") != [7]:
        errors.append("expected action shape [7]")
    if errors:
        raise ValueError("; ".join(errors))


def camera2_wrist_config(reference_config):
    """Use SmolVLA camera slots while keeping AMMR 7-D state/action."""
    config = dict(reference_config)
    config["input_features"] = {
        "observation.images.camera1": {
            "type": "VISUAL",
            "shape": [3, 256, 256],
        },
        "observation.images.camera2": {
            "type": "VISUAL",
            "shape": [3, 256, 256],
        },
        "observation.images.camera3": {
            "type": "VISUAL",
            "shape": [3, 256, 256],
        },
        "observation.state": {
            "type": "STATE",
            "shape": [7],
        },
    }
    config["output_features"] = {
        "action": {
            "type": "ACTION",
            "shape": [7],
        }
    }
    # The rollout supplies black dummy images for camera1 and camera3. Keeping
    # empty_cameras at 0 makes missing image keys a visible caller error.
    config["empty_cameras"] = 0
    return config


def patch_processor_features(payload, camera_layout):
    if camera_layout != "camera2_wrist":
        return payload

    patched = json.loads(json.dumps(payload))
    for step in patched.get("steps", []):
        config = step.get("config") or {}
        features = config.get("features")
        if not isinstance(features, dict):
            continue
        if "observation.images.wrist" not in features:
            continue
        visual_feature = features.pop("observation.images.wrist")
        features["observation.images.camera1"] = visual_feature
        features["observation.images.camera2"] = visual_feature
        features["observation.images.camera3"] = visual_feature
    return patched


def replace_or_create_dir(path, overwrite):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def link_or_copy(src, dst, copy_weights):
    if copy_weights:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def main():
    args = parse_args()
    base = Path(args.base).expanduser().resolve()
    reference = Path(args.reference).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    require_files(base, REQUIRED_BASE_FILES, "base checkpoint")
    require_files(reference, REQUIRED_REFERENCE_FILES, "reference checkpoint")
    reference_config = load_json(reference / "config.json")
    validate_ammr_config(reference_config)

    replace_or_create_dir(out, args.overwrite)
    link_or_copy(base / "model.safetensors", out / "model.safetensors", args.copy_weights)
    if args.camera_layout == "camera2_wrist":
        (out / "config.json").write_text(json.dumps(camera2_wrist_config(reference_config), indent=2) + "\n")
        for name in [
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        ]:
            payload = patch_processor_features(load_json(reference / name), args.camera_layout)
            (out / name).write_text(json.dumps(payload, indent=2) + "\n")
        for name in [
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        ]:
            shutil.copy2(reference / name, out / name)
    else:
        for name in REQUIRED_REFERENCE_FILES:
            shutil.copy2(reference / name, out / name)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "type": "smolvla_base_ammr_zeroshot",
        "description": (
            "Upstream SmolVLA base weights with AMMR wrist-camera/7-D "
            "state-action config and processor adapters. No gradient fine-tuning."
        ),
        "base_weights": str(base / "model.safetensors"),
        "ammr_interface_reference": str(reference),
        "weights_are_symlink": not args.copy_weights,
        "camera_layout": args.camera_layout,
        "policy_input": [
            (
                "observation.images.wrist"
                if args.camera_layout == "ammr_wrist"
                else "observation.images.camera1=dummy_black, observation.images.camera2=wrist, observation.images.camera3=dummy_black"
            ),
            "observation.state",
            "task",
        ],
        "policy_output": "action, 7-D absolute joint target",
        "caveat": (
            "The processor statistics come from the AMMR reference checkpoint so "
            "the upstream normalised actions can be mapped into this robot's "
            "radian command space. The policy weights are not fine-tuned."
        ),
    }
    (out / "ZERO_SHOT_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {out}")
    print("base weights:", metadata["base_weights"])
    print("interface reference:", metadata["ammr_interface_reference"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
