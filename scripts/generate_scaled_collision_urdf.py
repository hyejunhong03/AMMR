from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


SRC = Path("/home/autolab/AMMR/mechabot/mechabot/mechabot_scaled.urdf")
DST = Path("/home/autolab/AMMR/mechabot/mechabot/mechabot_scaled_simplified_collision.urdf")
NO_CASTORS_DST = Path("/home/autolab/AMMR/mechabot/mechabot/mechabot_scaled_no_castor_collision.urdf")
CYLINDER_WHEELS_DST = Path("/home/autolab/AMMR/mechabot/mechabot/mechabot_scaled_cylinder_wheels.urdf")


BOX_COLLISIONS = {
    "base_link": {
        "origin": {"xyz": "0.000000 0.000000 0.035000", "rpy": "0.0 0.0 0.0"},
        "size": "0.353276 0.304032 0.070770",
    },
    "toplid_link": {
        "origin": {"xyz": "0.000000 0.000000 0.000000", "rpy": "0.0 0.0 0.0"},
        "size": "0.289044 0.266028 0.016985",
    },
}

BOX_SENSOR_COLLISIONS = {
    "lidar_link": {
        "origin": {"xyz": "-0.044962 0.000000 -0.186833", "rpy": "0.0 0.0 0.0"},
        "size": "0.055 0.055 0.045",
    },
    "camera_link": {
        "origin": {"xyz": "-0.189484 0.000000 -0.130217", "rpy": "0.0 0.0 0.0"},
        "size": "0.045 0.035 0.035",
    },
    "imu_link": {
        "origin": {"xyz": "0.000000 0.000000 -0.049539", "rpy": "0.0 0.0 0.0"},
        "size": "0.035 0.035 0.018",
    },
}


def indent(elem: ET.Element, level: int = 0) -> None:
    spacing = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = spacing + "  "
        for child in elem:
            indent(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = spacing
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = spacing


def make_box_collision(origin_attrs: dict[str, str], size: str) -> ET.Element:
    collision = ET.Element("collision")
    ET.SubElement(collision, "origin", origin_attrs)
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", {"size": size})
    return collision


def make_cylinder_collision(origin_attrs: dict[str, str], radius: str, length: str) -> ET.Element:
    collision = ET.Element("collision")
    ET.SubElement(collision, "origin", origin_attrs)
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "cylinder", {"radius": radius, "length": length})
    return collision


def replace_collision(link: ET.Element, collision: ET.Element) -> None:
    for existing in list(link.findall("collision")):
        link.remove(existing)
    link.append(collision)


def main() -> None:
    root = ET.parse(SRC).getroot()

    for link in root.findall("link"):
        name = link.attrib.get("name")
        if name in BOX_COLLISIONS:
            config = BOX_COLLISIONS[name]
            replace_collision(link, make_box_collision(config["origin"], config["size"]))
        elif name in BOX_SENSOR_COLLISIONS:
            config = BOX_SENSOR_COLLISIONS[name]
            replace_collision(link, make_box_collision(config["origin"], config["size"]))

    root.insert(0, ET.Comment("Same scale and joint geometry as mechabot_scaled.urdf; simplified collision geometry for Isaac Sim."))
    indent(root)
    ET.ElementTree(root).write(DST, encoding="utf-8", xml_declaration=True)

    no_castors_root = deepcopy(root)
    for link in no_castors_root.findall("link"):
        name = link.attrib.get("name", "")
        if "castor_link" in name:
            for collision in list(link.findall("collision")):
                link.remove(collision)
    no_castors_root.insert(
        0,
        ET.Comment(
            "Same as mechabot_scaled_simplified_collision.urdf, but fixed castor collisions are removed to reduce contact jitter in Isaac Sim."
        ),
    )
    indent(no_castors_root)
    ET.ElementTree(no_castors_root).write(NO_CASTORS_DST, encoding="utf-8", xml_declaration=True)

    cylinder_wheels_root = deepcopy(no_castors_root)
    for link in cylinder_wheels_root.findall("link"):
        name = link.attrib.get("name", "")
        if name in {"left_wheel_link", "right_wheel_link"}:
            replace_collision(
                link,
                make_cylinder_collision(
                    {"xyz": "0.000000 0.000000 0.000000", "rpy": "-1.5707963268 0.0 0.0"},
                    radius="0.051258",
                    length="0.058431",
                ),
            )
    cylinder_wheels_root.insert(
        0,
        ET.Comment(
            "Same scale as mechabot_scaled.urdf; fixed castor collisions removed and wheel collisions changed from spheres to Y-axis cylinders."
        ),
    )
    indent(cylinder_wheels_root)
    ET.ElementTree(cylinder_wheels_root).write(CYLINDER_WHEELS_DST, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
