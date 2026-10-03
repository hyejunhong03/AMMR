#!/usr/bin/env python3
"""Compute link inertials for the myCobot 280 M5 adaptive gripper URDF.

The Elephant Robotics URDF ships with no <inertial> blocks at all, so Isaac's
importer falls back to a default density applied to each convex hull. That gives
the gripper finger links a near-zero inertia tensor, which makes the articulation
numerically stiff: a small contact impulse turns into a large wrist velocity.

The official mycobot_description URDF is not a usable source either. Its tensors
are degenerate (ixx=izz=0, iyy=0.5), every centre of mass sits at 0 0 0.1
regardless of geometry, and the base link is 10 kg. Those are Gazebo placeholders.

So derive the tensors from the actual collision geometry: parse each COLLADA
mesh, integrate the polyhedron inertia at uniform density, then rescale to a
per-link target mass. Run with --write to emit the URDF.
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


COLLADA_NS = "{http://www.collada.org/2005/11/COLLADASchema}"

# Target masses in kg. The myCobot 280 is documented at 850 g with a 250 g
# payload; the adaptive gripper adds roughly 100 g. Distribute that along the
# chain so the links taper toward the wrist the way the real arm does.
LINK_MASSES = {
    "g_base": 0.250,
    "joint1": 0.140,
    "joint2": 0.140,
    "joint3": 0.110,
    "joint4": 0.090,
    "joint5": 0.080,
    "joint6": 0.050,
    "joint6_flange": 0.020,
    "gripper_base": 0.070,
    "gripper_left1": 0.006,
    "gripper_left2": 0.005,
    "gripper_left3": 0.005,
    "gripper_right1": 0.006,
    "gripper_right2": 0.005,
    "gripper_right3": 0.005,
}

# Lower bound on each principal moment, in kg*m^2. A finger link is genuinely
# tiny, but letting a principal moment collapse toward zero is what destabilises
# the solver, so hold them at a small physically harmless floor.
MIN_PRINCIPAL_INERTIA = 1.0e-7


def _text_floats(element):
    return np.fromstring(element.text.strip().replace("\n", " "), sep=" ")


def load_collada_mesh(path):
    """Return (vertices, triangles) in metres, in the mesh's own frame."""
    tree = ET.parse(path)
    root = tree.getroot()

    unit = root.find(f"{COLLADA_NS}asset/{COLLADA_NS}unit")
    scale = float(unit.get("meter", "1.0")) if unit is not None else 1.0

    vertices = []
    triangles = []
    for geometry in root.iter(f"{COLLADA_NS}geometry"):
        mesh = geometry.find(f"{COLLADA_NS}mesh")
        if mesh is None:
            continue

        sources = {}
        for source in mesh.findall(f"{COLLADA_NS}source"):
            array = source.find(f"{COLLADA_NS}float_array")
            if array is None or not array.text:
                continue
            accessor = source.find(f"{COLLADA_NS}technique_common/{COLLADA_NS}accessor")
            stride = int(accessor.get("stride", "3")) if accessor is not None else 3
            sources["#" + source.get("id")] = _text_floats(array).reshape((-1, stride))

        vertices_tag = mesh.find(f"{COLLADA_NS}vertices")
        position_source = None
        if vertices_tag is not None:
            for inp in vertices_tag.findall(f"{COLLADA_NS}input"):
                if inp.get("semantic") == "POSITION":
                    position_source = inp.get("source")
        if position_source is None or position_source not in sources:
            continue

        points = sources[position_source]
        base = len(vertices)
        vertices.extend(points.tolist())

        for prim in list(mesh.findall(f"{COLLADA_NS}triangles")) + list(
            mesh.findall(f"{COLLADA_NS}polylist")
        ):
            inputs = prim.findall(f"{COLLADA_NS}input")
            stride = max(int(i.get("offset", "0")) for i in inputs) + 1
            offset = 0
            for inp in inputs:
                if inp.get("semantic") == "VERTEX":
                    offset = int(inp.get("offset", "0"))
            p_tag = prim.find(f"{COLLADA_NS}p")
            if p_tag is None or not p_tag.text:
                continue
            indices = np.fromstring(p_tag.text.strip().replace("\n", " "), sep=" ", dtype=np.int64)
            indices = indices.reshape((-1, stride))[:, offset]

            if prim.tag.endswith("polylist"):
                vcount_tag = prim.find(f"{COLLADA_NS}vcount")
                counts = np.fromstring(
                    vcount_tag.text.strip().replace("\n", " "), sep=" ", dtype=np.int64
                )
                cursor = 0
                for count in counts:
                    face = indices[cursor : cursor + count]
                    cursor += count
                    # Fan-triangulate any convex polygon.
                    for k in range(1, count - 1):
                        triangles.append([base + face[0], base + face[k], base + face[k + 1]])
            else:
                for tri in indices.reshape((-1, 3)):
                    triangles.append([base + tri[0], base + tri[1], base + tri[2]])

    if not triangles:
        raise ValueError(f"No triangles found in {path}")
    return np.asarray(vertices, dtype=np.float64) * scale, np.asarray(triangles, dtype=np.int64)


def mesh_inertia(vertices, triangles):
    """Volume, centre of mass, and inertia about the COM at unit density.

    Each triangle forms a tetrahedron with the origin. Summing the covariance of
    those tetrahedra gives the covariance of the enclosed solid, which converts
    to an inertia tensor. Signed contributions make the outside cancel, so the
    result is correct for any closed mesh regardless of where the origin sits.
    """
    canonical = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 120.0

    volume = 0.0
    weighted_centroid = np.zeros(3)
    covariance = np.zeros((3, 3))

    for i0, i1, i2 in triangles:
        a, b, c = vertices[i0], vertices[i1], vertices[i2]
        matrix = np.column_stack((a, b, c))
        determinant = np.linalg.det(matrix)
        if determinant == 0.0:
            continue
        tet_volume = determinant / 6.0
        volume += tet_volume
        weighted_centroid += tet_volume * (a + b + c) / 4.0
        covariance += determinant * (matrix @ canonical @ matrix.T)

    if abs(volume) < 1e-15:
        raise ValueError("Degenerate mesh volume")

    # A mesh wound inwards yields a negative volume; flip it rather than fail.
    if volume < 0.0:
        volume = -volume
        weighted_centroid = -weighted_centroid
        covariance = -covariance

    com = weighted_centroid / volume
    # Shift the covariance to the centre of mass, then convert to inertia.
    covariance -= volume * np.outer(com, com)
    inertia = np.trace(covariance) * np.eye(3) - covariance
    return volume, com, inertia


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def parse_link_geometry(urdf_path):
    """Map link name -> (mesh path, translation, rotation) from its <collision>."""
    tree = ET.parse(urdf_path)
    geometry = {}
    for link in tree.getroot().findall("link"):
        name = link.get("name")
        collision = link.find("collision")
        if collision is None:
            continue
        mesh = collision.find("geometry/mesh")
        if mesh is None:
            continue
        origin = collision.find("origin")
        translation = np.zeros(3)
        rotation = np.eye(3)
        if origin is not None:
            if origin.get("xyz"):
                translation = np.fromstring(origin.get("xyz").strip(), sep=" ")
            if origin.get("rpy"):
                rotation = rpy_to_matrix(*np.fromstring(origin.get("rpy").strip(), sep=" "))
        geometry[name] = (mesh.get("filename"), translation, rotation)
    return geometry


def compute_link_inertials(urdf_path):
    results = {}
    for name, (mesh_path, translation, rotation) in parse_link_geometry(urdf_path).items():
        mass = LINK_MASSES.get(name)
        if mass is None:
            print(f"  skip {name}: no target mass configured")
            continue

        vertices, triangles = load_collada_mesh(mesh_path)
        volume, com, inertia_unit_density = mesh_inertia(vertices, triangles)

        # Rescale from unit density to the target mass, then express the result
        # in the link frame rather than the mesh frame.
        density = mass / volume
        inertia = rotation @ (inertia_unit_density * density) @ rotation.T
        com_link = rotation @ com + translation

        eigenvalues = np.linalg.eigvalsh(inertia)
        if eigenvalues.min() < MIN_PRINCIPAL_INERTIA:
            # Raise every principal moment by the same amount so the tensor stays
            # a valid one and the shape of the ellipsoid is preserved.
            inertia = inertia + np.eye(3) * (MIN_PRINCIPAL_INERTIA - eigenvalues.min())

        results[name] = {
            "mass": mass,
            "com": com_link,
            "inertia": inertia,
            "volume": volume,
            "density": density,
            "floored": bool(eigenvalues.min() < MIN_PRINCIPAL_INERTIA),
        }
    return results


def format_inertial(entry, indent="    "):
    com = entry["com"]
    inertia = entry["inertia"]
    return (
        f"{indent}<inertial>\n"
        f'{indent}  <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0"/>\n'
        f'{indent}  <mass value="{entry["mass"]:.6f}"/>\n'
        f"{indent}  <inertia\n"
        f'{indent}    ixx="{inertia[0, 0]:.9e}" ixy="{inertia[0, 1]:.9e}" ixz="{inertia[0, 2]:.9e}"\n'
        f'{indent}    iyy="{inertia[1, 1]:.9e}" iyz="{inertia[1, 2]:.9e}"\n'
        f'{indent}    izz="{inertia[2, 2]:.9e}"/>\n'
        f"{indent}</inertial>\n"
    )


def write_urdf_with_inertials(source_path, output_path, entries):
    text = Path(source_path).read_text()
    if "<inertial>" in text:
        raise RuntimeError(f"{source_path} already contains inertial blocks")

    def insert(match):
        name = match.group("name")
        entry = entries.get(name)
        if entry is None:
            return match.group(0)
        return match.group(0) + "\n" + format_inertial(entry)

    pattern = re.compile(r'<link\s+name="(?P<name>[^"]+)"\s*>')
    updated, count = pattern.subn(insert, text)
    if count == 0:
        raise RuntimeError("No <link name=...> elements matched")
    Path(output_path).write_text(updated)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        default="/home/autolab/AMMR/isaac_urdf/mycobot_280_m5_adaptive_gripper.urdf",
        help="URDF to read link geometry from (needs resolved mesh paths).",
    )
    parser.add_argument("--source", help="URDF to copy and add inertials to.")
    parser.add_argument("--output", help="Where to write the URDF with inertials.")
    parser.add_argument("--write", action="store_true", help="Write the output URDF.")
    args = parser.parse_args()

    print(f"Reading link geometry from {args.urdf}")
    entries = compute_link_inertials(args.urdf)

    print()
    print(f"{'link':<18}{'mass_kg':>9}{'volume_cm3':>12}{'density':>10}{'ixx':>12}{'iyy':>12}{'izz':>12}")
    total = 0.0
    for name, entry in entries.items():
        inertia = entry["inertia"]
        total += entry["mass"]
        flag = " *floored" if entry["floored"] else ""
        print(
            f"{name:<18}{entry['mass']:>9.4f}{entry['volume'] * 1e6:>12.2f}"
            f"{entry['density']:>10.0f}{inertia[0, 0]:>12.3e}{inertia[1, 1]:>12.3e}"
            f"{inertia[2, 2]:>12.3e}{flag}"
        )
    print(f"{'TOTAL':<18}{total:>9.4f} kg")

    if args.write:
        if not args.source or not args.output:
            raise SystemExit("--write needs --source and --output")
        count = write_urdf_with_inertials(args.source, args.output, entries)
        print(f"\nWrote {args.output} ({count} links processed)")


if __name__ == "__main__":
    main()
