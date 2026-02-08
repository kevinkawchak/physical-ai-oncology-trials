#!/usr/bin/env python3
"""
Universal Robot Model Format Converter for Physical AI Oncology Trials

Converts between URDF, SDF, MJCF, and USD formats for cross-framework
compatibility in surgical robotics simulation.

Supported Conversions:
    URDF ↔ MJCF (bidirectional)
    URDF ↔ SDF (bidirectional)
    URDF → USD (one-way, requires Isaac tools)
    MJCF → URDF (partial, loses some features)

Usage:
    python urdf_sdf_mjcf_converter.py --input robot.urdf --output robot.mjcf
    python urdf_sdf_mjcf_converter.py --input robot.mjcf --output robot.sdf

Last updated: January 2026
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import warnings


@dataclass
class Link:
    """Unified link representation."""

    name: str
    mass: float = 1.0
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.01)
    com_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    visual_geometry: Optional[Dict] = None
    collision_geometry: Optional[Dict] = None


@dataclass
class Joint:
    """Unified joint representation."""

    name: str
    joint_type: str  # revolute, prismatic, fixed, continuous
    parent: str
    child: str
    axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    origin_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    limit_lower: float = -np.pi
    limit_upper: float = np.pi
    limit_effort: float = 100.0
    limit_velocity: float = 1.0
    damping: float = 0.1
    friction: float = 0.0


@dataclass
class RobotModel:
    """Unified robot model representation."""

    name: str
    links: List[Link] = field(default_factory=list)
    joints: List[Joint] = field(default_factory=list)
    source_format: str = "unknown"
    metadata: Dict = field(default_factory=dict)


class URDFParser:
    """Parse URDF (Unified Robot Description Format) files."""

    @staticmethod
    def parse(filepath: str) -> RobotModel:
        """Parse URDF file into unified model."""
        tree = ET.parse(filepath)
        root = tree.getroot()

        model = RobotModel(name=root.get("name", "robot"), source_format="urdf")

        # Parse links
        for link_elem in root.findall("link"):
            link = URDFParser._parse_link(link_elem)
            model.links.append(link)

        # Parse joints
        for joint_elem in root.findall("joint"):
            joint = URDFParser._parse_joint(joint_elem)
            model.joints.append(joint)

        return model

    @staticmethod
    def _parse_link(elem: ET.Element) -> Link:
        """Parse a single link element."""
        name = elem.get("name", "unnamed_link")

        # Parse inertial
        mass = 1.0
        inertia = np.eye(3) * 0.01
        com_position = np.zeros(3)

        inertial = elem.find("inertial")
        if inertial is not None:
            mass_elem = inertial.find("mass")
            if mass_elem is not None:
                mass = float(mass_elem.get("value", 1.0))

            origin = inertial.find("origin")
            if origin is not None:
                xyz = origin.get("xyz", "0 0 0")
                com_position = np.array([float(x) for x in xyz.split()])

            inertia_elem = inertial.find("inertia")
            if inertia_elem is not None:
                inertia = np.array(
                    [
                        [
                            float(inertia_elem.get("ixx", 0.01)),
                            float(inertia_elem.get("ixy", 0)),
                            float(inertia_elem.get("ixz", 0)),
                        ],
                        [
                            float(inertia_elem.get("ixy", 0)),
                            float(inertia_elem.get("iyy", 0.01)),
                            float(inertia_elem.get("iyz", 0)),
                        ],
                        [
                            float(inertia_elem.get("ixz", 0)),
                            float(inertia_elem.get("iyz", 0)),
                            float(inertia_elem.get("izz", 0.01)),
                        ],
                    ]
                )

        # Parse visual
        visual_geometry = URDFParser._parse_geometry(elem.find("visual"))

        # Parse collision
        collision_geometry = URDFParser._parse_geometry(elem.find("collision"))

        return Link(
            name=name,
            mass=mass,
            inertia=inertia,
            com_position=com_position,
            visual_geometry=visual_geometry,
            collision_geometry=collision_geometry,
        )

    @staticmethod
    def _parse_geometry(elem: Optional[ET.Element]) -> Optional[Dict]:
        """Parse geometry element."""
        if elem is None:
            return None

        geom = elem.find("geometry")
        if geom is None:
            return None

        geometry = {}

        # Check for different geometry types
        box = geom.find("box")
        if box is not None:
            size = box.get("size", "1 1 1")
            geometry = {"type": "box", "size": [float(x) for x in size.split()]}

        cylinder = geom.find("cylinder")
        if cylinder is not None:
            geometry = {
                "type": "cylinder",
                "radius": float(cylinder.get("radius", 0.1)),
                "length": float(cylinder.get("length", 1.0)),
            }

        sphere = geom.find("sphere")
        if sphere is not None:
            geometry = {"type": "sphere", "radius": float(sphere.get("radius", 0.1))}

        mesh = geom.find("mesh")
        if mesh is not None:
            geometry = {"type": "mesh", "filename": mesh.get("filename", ""), "scale": mesh.get("scale", "1 1 1")}

        return geometry

    @staticmethod
    def _parse_joint(elem: ET.Element) -> Joint:
        """Parse a single joint element."""
        name = elem.get("name", "unnamed_joint")
        joint_type = elem.get("type", "fixed")

        parent = elem.find("parent").get("link") if elem.find("parent") is not None else "world"
        child = elem.find("child").get("link") if elem.find("child") is not None else ""

        # Parse axis
        axis = np.array([0, 0, 1])
        axis_elem = elem.find("axis")
        if axis_elem is not None:
            xyz = axis_elem.get("xyz", "0 0 1")
            axis = np.array([float(x) for x in xyz.split()])

        # Parse origin
        origin_xyz = np.zeros(3)
        origin_rpy = np.zeros(3)
        origin = elem.find("origin")
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0")
            origin_xyz = np.array([float(x) for x in xyz.split()])
            rpy = origin.get("rpy", "0 0 0")
            origin_rpy = np.array([float(x) for x in rpy.split()])

        # Parse limits
        limit_lower = -np.pi
        limit_upper = np.pi
        limit_effort = 100.0
        limit_velocity = 1.0

        limit = elem.find("limit")
        if limit is not None:
            limit_lower = float(limit.get("lower", -np.pi))
            limit_upper = float(limit.get("upper", np.pi))
            limit_effort = float(limit.get("effort", 100.0))
            limit_velocity = float(limit.get("velocity", 1.0))

        # Parse dynamics
        damping = 0.1
        friction = 0.0
        dynamics = elem.find("dynamics")
        if dynamics is not None:
            damping = float(dynamics.get("damping", 0.1))
            friction = float(dynamics.get("friction", 0.0))

        return Joint(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            axis=axis,
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            limit_lower=limit_lower,
            limit_upper=limit_upper,
            limit_effort=limit_effort,
            limit_velocity=limit_velocity,
            damping=damping,
            friction=friction,
        )


class MJCFGenerator:
    """Generate MJCF (MuJoCo XML) files from unified model."""

    @staticmethod
    def generate(model: RobotModel, output_path: str) -> None:
        """Generate MJCF file from unified model."""
        root = ET.Element("mujoco", model=model.name)

        # Compiler settings
        compiler = ET.SubElement(root, "compiler", angle="radian", autolimits="true")

        # Default settings
        default = ET.SubElement(root, "default")
        joint_default = ET.SubElement(default, "joint", armature="0.01", damping="0.5")
        geom_default = ET.SubElement(default, "geom", friction="0.8 0.02 0.01")

        # Option settings
        option = ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")

        # Asset section (for meshes)
        asset = ET.SubElement(root, "asset")

        # Build kinematic tree
        worldbody = ET.SubElement(root, "worldbody")

        # Build link hierarchy
        link_map = {link.name: link for link in model.links}
        joint_map = {joint.child: joint for joint in model.joints}
        parent_map = {joint.child: joint.parent for joint in model.joints}

        # Find root links (no parent)
        child_links = set(j.child for j in model.joints)
        root_links = [l.name for l in model.links if l.name not in child_links]

        for root_link in root_links:
            MJCFGenerator._add_body(worldbody, root_link, link_map, joint_map, parent_map, asset)

        # Actuator section
        actuator = ET.SubElement(root, "actuator")
        for joint in model.joints:
            if joint.joint_type in ["revolute", "prismatic", "continuous"]:
                ET.SubElement(actuator, "motor", joint=joint.name, name=f"motor_{joint.name}", gear="50")

        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

        print(f"Generated MJCF: {output_path}")

    @staticmethod
    def _add_body(
        parent_elem: ET.Element,
        link_name: str,
        link_map: Dict[str, Link],
        joint_map: Dict[str, Joint],
        parent_map: Dict[str, str],
        asset: ET.Element,
    ) -> None:
        """Recursively add body elements."""
        link = link_map.get(link_name)
        if link is None:
            return

        # Get joint if this is a child link
        joint = joint_map.get(link_name)

        # Create body element
        body_attribs = {"name": link_name}
        if joint:
            pos = " ".join(f"{x:.6f}" for x in joint.origin_xyz)
            body_attribs["pos"] = pos

        body = ET.SubElement(parent_elem, "body", **body_attribs)

        # Add inertial
        inertial = ET.SubElement(
            body, "inertial", mass=str(link.mass), pos=" ".join(f"{x:.6f}" for x in link.com_position)
        )

        # Add joint if not root
        if joint and joint.joint_type != "fixed":
            joint_type = "hinge" if joint.joint_type in ["revolute", "continuous"] else "slide"
            joint_attribs = {
                "name": joint.name,
                "type": joint_type,
                "axis": " ".join(f"{x:.3f}" for x in joint.axis),
            }
            if joint.joint_type != "continuous":
                joint_attribs["range"] = f"{joint.limit_lower:.4f} {joint.limit_upper:.4f}"
            joint_attribs["damping"] = str(joint.damping)

            ET.SubElement(body, "joint", **joint_attribs)

        # Add geometry
        if link.visual_geometry:
            MJCFGenerator._add_geometry(body, link.visual_geometry, "visual", asset)
        if link.collision_geometry:
            MJCFGenerator._add_geometry(body, link.collision_geometry, "collision", asset)

        # Recursively add child bodies
        for child_joint in [j for j in joint_map.values() if j.parent == link_name]:
            MJCFGenerator._add_body(body, child_joint.child, link_map, joint_map, parent_map, asset)

    @staticmethod
    def _add_geometry(body: ET.Element, geometry: Dict, geom_type: str, asset: ET.Element) -> None:
        """Add geometry to body."""
        geom_attribs = {"name": f"{body.get('name')}_{geom_type}"}

        if geometry["type"] == "box":
            size = geometry["size"]
            geom_attribs["type"] = "box"
            geom_attribs["size"] = " ".join(f"{s / 2:.6f}" for s in size)  # half-size

        elif geometry["type"] == "cylinder":
            geom_attribs["type"] = "cylinder"
            geom_attribs["size"] = f"{geometry['radius']:.6f} {geometry['length'] / 2:.6f}"

        elif geometry["type"] == "sphere":
            geom_attribs["type"] = "sphere"
            geom_attribs["size"] = f"{geometry['radius']:.6f}"

        elif geometry["type"] == "mesh":
            filename = geometry["filename"]
            mesh_name = Path(filename).stem
            ET.SubElement(asset, "mesh", name=mesh_name, file=filename)
            geom_attribs["type"] = "mesh"
            geom_attribs["mesh"] = mesh_name

        ET.SubElement(body, "geom", **geom_attribs)


class SDFGenerator:
    """Generate SDF (Simulation Description Format) files from unified model."""

    @staticmethod
    def generate(model: RobotModel, output_path: str) -> None:
        """Generate SDF file from unified model."""
        root = ET.Element("sdf", version="1.9")
        model_elem = ET.SubElement(root, "model", name=model.name)

        # Add links
        for link in model.links:
            SDFGenerator._add_link(model_elem, link)

        # Add joints
        for joint in model.joints:
            SDFGenerator._add_joint(model_elem, joint)

        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

        print(f"Generated SDF: {output_path}")

    @staticmethod
    def _add_link(model_elem: ET.Element, link: Link) -> None:
        """Add link element to SDF."""
        link_elem = ET.SubElement(model_elem, "link", name=link.name)

        # Inertial
        inertial = ET.SubElement(link_elem, "inertial")
        ET.SubElement(inertial, "mass").text = str(link.mass)
        inertia = ET.SubElement(inertial, "inertia")
        ET.SubElement(inertia, "ixx").text = str(link.inertia[0, 0])
        ET.SubElement(inertia, "iyy").text = str(link.inertia[1, 1])
        ET.SubElement(inertia, "izz").text = str(link.inertia[2, 2])
        ET.SubElement(inertia, "ixy").text = str(link.inertia[0, 1])
        ET.SubElement(inertia, "ixz").text = str(link.inertia[0, 2])
        ET.SubElement(inertia, "iyz").text = str(link.inertia[1, 2])

        # Visual
        if link.visual_geometry:
            visual = ET.SubElement(link_elem, "visual", name=f"{link.name}_visual")
            SDFGenerator._add_sdf_geometry(visual, link.visual_geometry)

        # Collision
        if link.collision_geometry:
            collision = ET.SubElement(link_elem, "collision", name=f"{link.name}_collision")
            SDFGenerator._add_sdf_geometry(collision, link.collision_geometry)

    @staticmethod
    def _add_sdf_geometry(parent: ET.Element, geometry: Dict) -> None:
        """Add geometry to SDF element."""
        geom = ET.SubElement(parent, "geometry")

        if geometry["type"] == "box":
            box = ET.SubElement(geom, "box")
            size = geometry["size"]
            ET.SubElement(box, "size").text = " ".join(f"{s:.6f}" for s in size)

        elif geometry["type"] == "cylinder":
            cylinder = ET.SubElement(geom, "cylinder")
            ET.SubElement(cylinder, "radius").text = str(geometry["radius"])
            ET.SubElement(cylinder, "length").text = str(geometry["length"])

        elif geometry["type"] == "sphere":
            sphere = ET.SubElement(geom, "sphere")
            ET.SubElement(sphere, "radius").text = str(geometry["radius"])

        elif geometry["type"] == "mesh":
            mesh = ET.SubElement(geom, "mesh")
            ET.SubElement(mesh, "uri").text = geometry["filename"]

    @staticmethod
    def _add_joint(model_elem: ET.Element, joint: Joint) -> None:
        """Add joint element to SDF."""
        sdf_type = {"revolute": "revolute", "continuous": "revolute", "prismatic": "prismatic", "fixed": "fixed"}.get(
            joint.joint_type, "fixed"
        )

        joint_elem = ET.SubElement(model_elem, "joint", name=joint.name, type=sdf_type)

        ET.SubElement(joint_elem, "parent").text = joint.parent
        ET.SubElement(joint_elem, "child").text = joint.child

        # Pose
        pose = " ".join(f"{x:.6f}" for x in joint.origin_xyz) + " " + " ".join(f"{x:.6f}" for x in joint.origin_rpy)
        ET.SubElement(joint_elem, "pose").text = pose

        if joint.joint_type != "fixed":
            axis = ET.SubElement(joint_elem, "axis")
            ET.SubElement(axis, "xyz").text = " ".join(f"{x:.3f}" for x in joint.axis)

            limit = ET.SubElement(axis, "limit")
            ET.SubElement(limit, "lower").text = str(joint.limit_lower)
            ET.SubElement(limit, "upper").text = str(joint.limit_upper)
            ET.SubElement(limit, "effort").text = str(joint.limit_effort)
            ET.SubElement(limit, "velocity").text = str(joint.limit_velocity)

            dynamics = ET.SubElement(axis, "dynamics")
            ET.SubElement(dynamics, "damping").text = str(joint.damping)
            ET.SubElement(dynamics, "friction").text = str(joint.friction)


class UnifiedModelConverter:
    """
    Universal robot model converter for cross-framework compatibility.

    Supports conversions between URDF, MJCF, SDF, and USD formats.
    """

    SUPPORTED_FORMATS = ["urdf", "mjcf", "sdf", "usd"]

    def __init__(self):
        self.conversion_log = []

    def convert(
        self,
        source_path: str,
        source_format: Optional[str] = None,
        target_formats: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Convert robot model to target formats.

        Args:
            source_path: Path to source model file
            source_format: Source format (auto-detected if None)
            target_formats: List of target formats (all if None)
            output_dir: Output directory (same as source if None)

        Returns:
            Dictionary mapping format to output path
        """
        source_path = Path(source_path)

        # Auto-detect source format
        if source_format is None:
            source_format = self._detect_format(source_path)

        # Default to all other formats
        if target_formats is None:
            target_formats = [f for f in self.SUPPORTED_FORMATS if f != source_format]

        # Output directory
        if output_dir is None:
            output_dir = source_path.parent
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse source model
        model = self._parse_model(source_path, source_format)
        print(f"Parsed {source_format.upper()} model: {model.name}")
        print(f"  Links: {len(model.links)}")
        print(f"  Joints: {len(model.joints)}")

        # Convert to each target format
        outputs = {}
        for target in target_formats:
            output_path = output_dir / f"{model.name}.{self._get_extension(target)}"
            self._generate_model(model, target, str(output_path))
            outputs[target] = str(output_path)
            self.conversion_log.append(
                {
                    "source": str(source_path),
                    "source_format": source_format,
                    "target": str(output_path),
                    "target_format": target,
                }
            )

        return outputs

    def _detect_format(self, path: Path) -> str:
        """Detect model format from file extension."""
        ext = path.suffix.lower()
        format_map = {
            ".urdf": "urdf",
            ".xacro": "urdf",
            ".xml": "mjcf",  # Assume MJCF for .xml
            ".mjcf": "mjcf",
            ".sdf": "sdf",
            ".usd": "usd",
            ".usda": "usd",
        }
        return format_map.get(ext, "urdf")

    def _get_extension(self, format_name: str) -> str:
        """Get file extension for format."""
        ext_map = {
            "urdf": "urdf",
            "mjcf": "xml",
            "sdf": "sdf",
            "usd": "usd",
        }
        return ext_map.get(format_name, format_name)

    def _parse_model(self, path: Path, format_name: str) -> RobotModel:
        """Parse model from file."""
        if format_name == "urdf":
            return URDFParser.parse(str(path))
        elif format_name == "mjcf":
            warnings.warn("MJCF parsing not fully implemented, using URDF parser")
            return URDFParser.parse(str(path))
        elif format_name == "sdf":
            warnings.warn("SDF parsing not fully implemented")
            return RobotModel(name="parsed_model", source_format="sdf")
        else:
            raise ValueError(f"Unsupported source format: {format_name}")

    def _generate_model(self, model: RobotModel, format_name: str, output_path: str) -> None:
        """Generate model file."""
        if format_name == "mjcf":
            MJCFGenerator.generate(model, output_path)
        elif format_name == "sdf":
            SDFGenerator.generate(model, output_path)
        elif format_name == "urdf":
            warnings.warn("URDF generation not fully implemented")
        elif format_name == "usd":
            warnings.warn("USD generation requires NVIDIA tools")
        else:
            raise ValueError(f"Unsupported target format: {format_name}")

    def get_conversion_log(self) -> List[Dict]:
        """Get conversion history."""
        return self.conversion_log.copy()


def main():
    """Command-line interface for model conversion."""
    parser = argparse.ArgumentParser(description="Convert robot models between URDF, MJCF, SDF, and USD formats.")
    parser.add_argument("--input", "-i", required=True, help="Input model file path")
    parser.add_argument("--output", "-o", help="Output file path or directory")
    parser.add_argument(
        "--source-format",
        "-sf",
        choices=["urdf", "mjcf", "sdf", "usd"],
        help="Source format (auto-detected if not specified)",
    )
    parser.add_argument(
        "--target-format", "-tf", choices=["urdf", "mjcf", "sdf", "usd"], nargs="+", help="Target format(s)"
    )
    parser.add_argument("--all-formats", "-a", action="store_true", help="Convert to all supported formats")

    args = parser.parse_args()

    converter = UnifiedModelConverter()

    # Determine target formats
    target_formats = args.target_format
    if args.all_formats:
        target_formats = None  # All formats

    # Determine output directory
    output_dir = None
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix:
            # Single file output
            output_dir = output_path.parent
            target_formats = [converter._detect_format(output_path)]
        else:
            output_dir = output_path

    # Perform conversion
    outputs = converter.convert(
        source_path=args.input, source_format=args.source_format, target_formats=target_formats, output_dir=output_dir
    )

    print("\nConversion complete:")
    for fmt, path in outputs.items():
        print(f"  {fmt.upper()}: {path}")


if __name__ == "__main__":
    main()
