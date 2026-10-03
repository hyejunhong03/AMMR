import argparse
import os
import re
from pathlib import Path

from isaacsim import SimulationApp


DEFAULT_SOURCE_URDF = "/home/autolab/Downloads/mycobot_280_m5_adaptive_gripper.urdf"
DEFAULT_MESH_DIR = "/home/autolab/AMMR/mechabot_description/meshes/mycobot_280"
DEFAULT_PREPARED_DIR = "/home/autolab/AMMR/isaac_urdf"
DEFAULT_OUTPUT_ROOT = "/home/autolab/AMMR/isaac_usd/mycobot_280_m5_adaptive_gripper_reimport"
ROBOT_NAME = "mycobot_280_m5_adaptive_gripper"


parser = argparse.ArgumentParser(description="Import Elephant Robotics myCobot 280 M5 adaptive gripper URDF to Isaac USD.")
parser.add_argument("--source-urdf", default=DEFAULT_SOURCE_URDF, help="Elephant Robotics source URDF.")
parser.add_argument("--mesh-dir", default=DEFAULT_MESH_DIR, help="Directory containing myCobot and gripper DAE meshes.")
parser.add_argument("--prepared-dir", default=DEFAULT_PREPARED_DIR, help="Directory for the prepared URDF.")
parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Directory where Isaac USD output package is written.")
parser.add_argument("--robot-name", default=ROBOT_NAME, help="Robot name and prepared URDF stem.")
parser.add_argument("--no-asset-transformer", action="store_true", help="Write a simpler flat USD without Isaac asset packaging.")
parser.add_argument("--debug", action="store_true", help="Keep importer debug intermediates.")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig


def _resolve_mesh_paths(urdf_text, mesh_dir):
    replacements = {
        "package://mycobot_description/urdf/mycobot_280_m5/": f"{mesh_dir}/",
        "package://mycobot_description/urdf/adaptive_gripper/": f"{mesh_dir}/",
    }
    for old, new in replacements.items():
        urdf_text = urdf_text.replace(old, new)
    return urdf_text


def _set_robot_name(urdf_text, robot_name):
    pattern = re.compile(r'(<robot\b[^>]*\bname=")([^"]+)(")')
    return pattern.sub(lambda match: f"{match.group(1)}{robot_name}{match.group(3)}", urdf_text, count=1)


def _validate_mesh_paths(urdf_text):
    missing = []
    for mesh_path in re.findall(r'<mesh\s+filename="([^"]+)"', urdf_text):
        if mesh_path.startswith("package://"):
            missing.append(mesh_path)
        elif os.path.isabs(mesh_path) and not os.path.exists(mesh_path):
            missing.append(mesh_path)
    if missing:
        raise FileNotFoundError("Unresolved mesh paths:\n" + "\n".join(missing))


def _write_prepared_urdf(source_urdf, prepared_dir, robot_name, mesh_dir):
    source_path = Path(source_urdf)
    mesh_path = Path(mesh_dir)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not mesh_path.is_dir():
        raise FileNotFoundError(mesh_path)

    urdf_text = source_path.read_text()
    urdf_text = _resolve_mesh_paths(urdf_text, str(mesh_path))
    urdf_text = _set_robot_name(urdf_text, robot_name)
    _validate_mesh_paths(urdf_text)

    prepared_path = Path(prepared_dir) / f"{robot_name}.urdf"
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_path.write_text(urdf_text)
    return prepared_path


def main():
    app_utils.enable_extension("isaacsim.asset.importer.urdf")

    prepared_urdf = _write_prepared_urdf(
        source_urdf=args.source_urdf,
        prepared_dir=args.prepared_dir,
        robot_name=args.robot_name,
        mesh_dir=args.mesh_dir,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = URDFImporterConfig(
        urdf_path=str(prepared_urdf),
        usd_path=str(output_root),
        merge_fixed_joints=False,
        merge_mesh=False,
        debug_mode=args.debug,
        collision_from_visuals=False,
        allow_self_collision=False,
        fix_base=True,
        run_asset_transformer=not args.no_asset_transformer,
    )
    importer = URDFImporter(config)
    output_path = importer.import_urdf()

    print(f"Prepared URDF: {prepared_urdf}")
    print(f"Imported USD: {output_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
