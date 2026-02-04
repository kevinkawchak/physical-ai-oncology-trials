#!/usr/bin/env python3
"""
MuJoCo to Isaac Lab Conversion Pipeline

This module provides comprehensive conversion from MuJoCo MJCF models
to formats compatible with NVIDIA Isaac Lab (URDF, USD).

Framework Versions:
    - MuJoCo: 3.4.0 (https://github.com/google-deepmind/mujoco/releases)
    - Isaac Lab: 2.3.2+ (https://github.com/isaac-sim/IsaacLab/releases)
    - Isaac Sim: 4.5.0+ (native MJCF importer available)

References:
    - mjcf2urdf: https://github.com/iory/mjcf2urdf
    - MuJoCo MJCF Reference: https://mujoco.readthedocs.io/en/stable/XMLreference.html
    - Isaac Sim MJCF Importer: https://docs.isaacsim.omniverse.nvidia.com/latest/

Usage:
    python mujoco_to_isaac_pipeline.py --input robot.xml --output robot.urdf
    python mujoco_to_isaac_pipeline.py --input robot.xml --output robot.urdf --validate

Last updated: February 2026
"""

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import warnings
import json

# Optional imports
try:
    import mujoco

    MUJOCO_AVAILABLE = True
    MUJOCO_VERSION = mujoco.__version__
except ImportError:
    MUJOCO_AVAILABLE = False
    MUJOCO_VERSION = None
    warnings.warn("MuJoCo not installed. Some features will be limited.")


@dataclass
class MJCFBody:
    """Parsed MJCF body representation."""

    name: str
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    mass: float = 1.0
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.01)
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    joints: List[Dict] = field(default_factory=list)
    geoms: List[Dict] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None


@dataclass
class ConversionConfig:
    """Configuration for MuJoCo → Isaac conversion."""

    # Output format
    output_format: str = "urdf"  # "urdf" or "usd"

    # Physics parameter defaults (for URDF)
    default_effort: float = 100.0
    default_velocity: float = 1.0
    default_damping: float = 0.5
    default_friction: float = 0.1

    # Mesh handling
    convert_meshes: bool = True
    mesh_output_dir: Optional[str] = None

    # Validation
    validate_output: bool = True


class MJCFToURDFConverter:
    """
    Convert MuJoCo MJCF models to URDF format.

    MJCF → URDF conversion challenges:
    1. MJCF uses nested body hierarchy; URDF uses flat links + joints
    2. MJCF joints are defined within child bodies
    3. MJCF actuators need mapping to ROS control
    4. MJCF defaults propagate to children; URDF has no defaults

    References:
    - mjcf2urdf: https://github.com/iory/mjcf2urdf
    - MJCF Structure: https://mujoco.readthedocs.io/en/stable/modeling.html
    """

    JOINT_TYPE_MAP = {
        "hinge": "revolute",
        "slide": "prismatic",
        "ball": "floating",  # Approximation
        "free": "floating",
    }

    def __init__(self, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
        self.bodies: Dict[str, MJCFBody] = {}
        self.defaults: Dict[str, Dict] = {}
        self.meshes: Dict[str, str] = {}
        self.actuators: List[Dict] = []
        self.warnings: List[str] = []

    def convert(self, mjcf_path: str, output_path: str, robot_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert MJCF file to URDF format.

        Args:
            mjcf_path: Path to input MJCF file
            output_path: Path to output URDF file
            robot_name: Override robot name

        Returns:
            Conversion result dictionary
        """
        # Reset state
        self.bodies = {}
        self.defaults = {}
        self.meshes = {}
        self.actuators = []
        self.warnings = []

        # Parse MJCF
        mjcf_tree = ET.parse(mjcf_path)
        mjcf_root = mjcf_tree.getroot()

        robot_name = robot_name or mjcf_root.get("model", "robot")

        # Parse defaults (for applying to elements)
        self._parse_defaults(mjcf_root)

        # Parse assets (meshes)
        self._parse_assets(mjcf_root, mjcf_path)

        # Parse worldbody
        worldbody = mjcf_root.find("worldbody")
        if worldbody is not None:
            self._parse_worldbody(worldbody)

        # Parse actuators
        self._parse_actuators(mjcf_root)

        # Generate URDF
        urdf_root = self._generate_urdf(robot_name)

        # Write output
        tree = ET.ElementTree(urdf_root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

        # Validate if requested
        validation_result = None
        if self.config.validate_output:
            validation_result = self._validate_urdf(output_path)

        return {
            "success": True,
            "input": mjcf_path,
            "output": output_path,
            "robot_name": robot_name,
            "bodies_converted": len(self.bodies),
            "joints_converted": sum(len(b.joints) for b in self.bodies.values()),
            "actuators_found": len(self.actuators),
            "warnings": self.warnings.copy(),
            "validation": validation_result,
        }

    def _parse_defaults(self, root: ET.Element) -> None:
        """Parse default elements for property inheritance."""
        default_elem = root.find("default")
        if default_elem is None:
            return

        # Parse class defaults
        for class_elem in default_elem.findall("default"):
            class_name = class_elem.get("class", "default")
            self.defaults[class_name] = {}

            for child in class_elem:
                self.defaults[class_name][child.tag] = dict(child.attrib)

        # Parse global defaults (no class attribute)
        for child in default_elem:
            if child.tag != "default":
                if "global" not in self.defaults:
                    self.defaults["global"] = {}
                self.defaults["global"][child.tag] = dict(child.attrib)

    def _parse_assets(self, root: ET.Element, mjcf_path: str) -> None:
        """Parse asset elements (meshes, textures, materials)."""
        asset_elem = root.find("asset")
        if asset_elem is None:
            return

        mjcf_dir = Path(mjcf_path).parent

        for mesh in asset_elem.findall("mesh"):
            name = mesh.get("name")
            file_path = mesh.get("file", "")

            # Resolve file path
            if file_path:
                resolved = self._resolve_asset_path(file_path, mjcf_dir)
                self.meshes[name] = resolved

    def _resolve_asset_path(self, path: str, base_dir: Path) -> str:
        """Resolve asset path relative to MJCF file."""
        # Try relative to base_dir
        full_path = base_dir / path
        if full_path.exists():
            return str(full_path)

        # Try in meshes subdirectory
        mesh_path = base_dir / "meshes" / Path(path).name
        if mesh_path.exists():
            return str(mesh_path)

        # Return as-is
        return path

    def _parse_worldbody(self, worldbody: ET.Element, parent: str = None) -> None:
        """Recursively parse worldbody element."""
        for body_elem in worldbody.findall("body"):
            self._parse_body(body_elem, parent)

    def _parse_body(self, body_elem: ET.Element, parent: Optional[str]) -> None:
        """Parse a single body element."""
        name = body_elem.get("name", f"body_{len(self.bodies)}")

        # Position and orientation
        pos = self._parse_vec3(body_elem.get("pos", "0 0 0"))
        quat = self._parse_quat(body_elem.get("quat", "1 0 0 0"))

        # Create body
        body = MJCFBody(name=name, pos=pos, quat=quat, parent=parent)

        # Parse inertial
        inertial = body_elem.find("inertial")
        if inertial is not None:
            body.mass = float(inertial.get("mass", 1.0))
            body.com = self._parse_vec3(inertial.get("pos", "0 0 0"))

            diag = inertial.get("diaginertia")
            full = inertial.get("fullinertia")

            if diag:
                vals = [float(x) for x in diag.split()]
                body.inertia = np.diag(vals)
            elif full:
                vals = [float(x) for x in full.split()]
                body.inertia = np.array(
                    [[vals[0], vals[3], vals[4]], [vals[3], vals[1], vals[5]], [vals[4], vals[5], vals[2]]]
                )

        # Parse joints
        for joint_elem in body_elem.findall("joint"):
            joint = self._parse_joint(joint_elem)
            body.joints.append(joint)

        # Parse geoms
        for geom_elem in body_elem.findall("geom"):
            geom = self._parse_geom(geom_elem)
            body.geoms.append(geom)

        # Store body
        self.bodies[name] = body

        # Update parent's children list
        if parent and parent in self.bodies:
            self.bodies[parent].children.append(name)

        # Recursively parse child bodies
        for child_body in body_elem.findall("body"):
            self._parse_body(child_body, name)

    def _parse_joint(self, joint_elem: ET.Element) -> Dict:
        """Parse a joint element."""
        joint = {
            "name": joint_elem.get("name", f"joint_{len(self.bodies)}"),
            "type": joint_elem.get("type", "hinge"),
            "axis": self._parse_vec3(joint_elem.get("axis", "0 0 1")),
            "pos": self._parse_vec3(joint_elem.get("pos", "0 0 0")),
            "range": None,
            "damping": float(joint_elem.get("damping", self.config.default_damping)),
            "frictionloss": float(joint_elem.get("frictionloss", 0.0)),
            "armature": float(joint_elem.get("armature", 0.01)),
        }

        # Parse range
        range_str = joint_elem.get("range")
        if range_str:
            vals = [float(x) for x in range_str.split()]
            joint["range"] = (vals[0], vals[1])
        else:
            # Default range based on joint type
            if joint["type"] == "hinge":
                joint["range"] = (-np.pi, np.pi)
            elif joint["type"] == "slide":
                joint["range"] = (-1.0, 1.0)

        return joint

    def _parse_geom(self, geom_elem: ET.Element) -> Dict:
        """Parse a geometry element."""
        geom = {
            "name": geom_elem.get("name", ""),
            "type": geom_elem.get("type", "sphere"),
            "pos": self._parse_vec3(geom_elem.get("pos", "0 0 0")),
            "quat": self._parse_quat(geom_elem.get("quat", "1 0 0 0")),
            "size": None,
            "mesh": geom_elem.get("mesh"),
        }

        # Parse size based on type
        size_str = geom_elem.get("size", "0.1")
        size_vals = [float(x) for x in size_str.split()]

        geom_type = geom["type"]
        if geom_type == "sphere":
            geom["size"] = {"radius": size_vals[0]}
        elif geom_type == "box":
            # MuJoCo uses half-sizes
            geom["size"] = {"x": size_vals[0] * 2, "y": size_vals[1] * 2, "z": size_vals[2] * 2}
        elif geom_type == "cylinder":
            geom["size"] = {"radius": size_vals[0], "length": size_vals[1] * 2}
        elif geom_type == "capsule":
            geom["size"] = {"radius": size_vals[0], "length": size_vals[1] * 2}
        elif geom_type == "mesh":
            geom["size"] = None  # Mesh has no size params

        return geom

    def _parse_actuators(self, root: ET.Element) -> None:
        """Parse actuator elements."""
        actuator_elem = root.find("actuator")
        if actuator_elem is None:
            return

        for motor in actuator_elem.findall("motor"):
            self.actuators.append(
                {
                    "type": "motor",
                    "name": motor.get("name", ""),
                    "joint": motor.get("joint", ""),
                    "gear": float(motor.get("gear", 1.0)),
                    "forcerange": motor.get("forcerange"),
                }
            )

        for position in actuator_elem.findall("position"):
            self.actuators.append(
                {
                    "type": "position",
                    "name": position.get("name", ""),
                    "joint": position.get("joint", ""),
                    "kp": float(position.get("kp", 100.0)),
                }
            )

        for velocity in actuator_elem.findall("velocity"):
            self.actuators.append(
                {
                    "type": "velocity",
                    "name": velocity.get("name", ""),
                    "joint": velocity.get("joint", ""),
                    "kv": float(velocity.get("kv", 10.0)),
                }
            )

    def _parse_vec3(self, s: str) -> np.ndarray:
        """Parse a 3D vector from string."""
        return np.array([float(x) for x in s.split()])

    def _parse_quat(self, s: str) -> np.ndarray:
        """Parse a quaternion from string (wxyz format)."""
        return np.array([float(x) for x in s.split()])

    def _quat_to_rpy(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (wxyz) to roll-pitch-yaw."""
        w, x, y, z = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def _generate_urdf(self, robot_name: str) -> ET.Element:
        """Generate URDF XML from parsed MJCF data."""
        robot = ET.Element("robot", name=robot_name)

        # Find root bodies (those with no parent)
        root_bodies = [name for name, body in self.bodies.items() if body.parent is None]

        # Generate links and joints
        for name, body in self.bodies.items():
            # Generate link
            link = self._generate_link(body)
            robot.append(link)

        # Generate joints
        for name, body in self.bodies.items():
            if body.parent is not None:
                joint = self._generate_joint(body)
                if joint is not None:
                    robot.append(joint)

        return robot

    def _generate_link(self, body: MJCFBody) -> ET.Element:
        """Generate a URDF link element."""
        link = ET.Element("link", name=body.name)

        # Inertial
        inertial = ET.SubElement(link, "inertial")

        origin = ET.SubElement(inertial, "origin")
        origin.set("xyz", " ".join(f"{x:.6f}" for x in body.com))
        origin.set("rpy", "0 0 0")

        ET.SubElement(inertial, "mass", value=f"{body.mass:.6f}")

        inertia = ET.SubElement(inertial, "inertia")
        inertia.set("ixx", f"{body.inertia[0, 0]:.6g}")
        inertia.set("iyy", f"{body.inertia[1, 1]:.6g}")
        inertia.set("izz", f"{body.inertia[2, 2]:.6g}")
        inertia.set("ixy", f"{body.inertia[0, 1]:.6g}")
        inertia.set("ixz", f"{body.inertia[0, 2]:.6g}")
        inertia.set("iyz", f"{body.inertia[1, 2]:.6g}")

        # Visual and collision from geoms
        for i, geom in enumerate(body.geoms):
            # Visual
            visual = ET.SubElement(link, "visual", name=f"{body.name}_visual_{i}")
            self._add_geometry_to_urdf(visual, geom)

            # Collision
            collision = ET.SubElement(link, "collision", name=f"{body.name}_collision_{i}")
            self._add_geometry_to_urdf(collision, geom)

        return link

    def _add_geometry_to_urdf(self, parent: ET.Element, geom: Dict) -> None:
        """Add geometry element to URDF visual/collision."""
        # Origin
        origin = ET.SubElement(parent, "origin")
        origin.set("xyz", " ".join(f"{x:.6f}" for x in geom["pos"]))
        rpy = self._quat_to_rpy(geom["quat"])
        origin.set("rpy", " ".join(f"{x:.6f}" for x in rpy))

        # Geometry
        geometry = ET.SubElement(parent, "geometry")

        geom_type = geom["type"]
        if geom_type == "sphere":
            ET.SubElement(geometry, "sphere", radius=f"{geom['size']['radius']:.6f}")

        elif geom_type == "box":
            size = geom["size"]
            ET.SubElement(geometry, "box", size=f"{size['x']:.6f} {size['y']:.6f} {size['z']:.6f}")

        elif geom_type == "cylinder" or geom_type == "capsule":
            size = geom["size"]
            ET.SubElement(geometry, "cylinder", radius=f"{size['radius']:.6f}", length=f"{size['length']:.6f}")

        elif geom_type == "mesh" and geom.get("mesh"):
            mesh_name = geom["mesh"]
            mesh_file = self.meshes.get(mesh_name, f"{mesh_name}.stl")
            ET.SubElement(geometry, "mesh", filename=mesh_file)

    def _generate_joint(self, body: MJCFBody) -> Optional[ET.Element]:
        """Generate a URDF joint element."""
        if body.parent is None:
            return None

        # Use first joint if available, otherwise create fixed joint
        if body.joints:
            mjcf_joint = body.joints[0]
            urdf_type = self.JOINT_TYPE_MAP.get(mjcf_joint["type"], "fixed")

            joint = ET.Element("joint", name=mjcf_joint["name"], type=urdf_type)

            # Parent and child
            ET.SubElement(joint, "parent", link=body.parent)
            ET.SubElement(joint, "child", link=body.name)

            # Origin (body position relative to parent)
            origin = ET.SubElement(joint, "origin")
            origin.set("xyz", " ".join(f"{x:.6f}" for x in body.pos))
            rpy = self._quat_to_rpy(body.quat)
            origin.set("rpy", " ".join(f"{x:.6f}" for x in rpy))

            # Axis
            if urdf_type in ["revolute", "prismatic", "continuous"]:
                axis = ET.SubElement(joint, "axis")
                axis.set("xyz", " ".join(f"{x:.3f}" for x in mjcf_joint["axis"]))

            # Limits
            if urdf_type in ["revolute", "prismatic"] and mjcf_joint.get("range"):
                limit = ET.SubElement(joint, "limit")
                limit.set("lower", f"{mjcf_joint['range'][0]:.4f}")
                limit.set("upper", f"{mjcf_joint['range'][1]:.4f}")
                limit.set("effort", f"{self.config.default_effort:.1f}")
                limit.set("velocity", f"{self.config.default_velocity:.1f}")

            # Dynamics
            dynamics = ET.SubElement(joint, "dynamics")
            dynamics.set("damping", f"{mjcf_joint['damping']:.4f}")
            dynamics.set("friction", f"{mjcf_joint['frictionloss']:.4f}")

            return joint
        else:
            # Fixed joint (no MJCF joint defined)
            joint = ET.Element("joint", name=f"{body.parent}_to_{body.name}", type="fixed")

            ET.SubElement(joint, "parent", link=body.parent)
            ET.SubElement(joint, "child", link=body.name)

            origin = ET.SubElement(joint, "origin")
            origin.set("xyz", " ".join(f"{x:.6f}" for x in body.pos))
            rpy = self._quat_to_rpy(body.quat)
            origin.set("rpy", " ".join(f"{x:.6f}" for x in rpy))

            return joint

    def _validate_urdf(self, urdf_path: str) -> Dict[str, Any]:
        """Basic validation of generated URDF."""
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            links = root.findall("link")
            joints = root.findall("joint")

            return {
                "valid": True,
                "links": len(links),
                "joints": len(joints),
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
            }


class MuJoCoToIsaacConverter:
    """
    High-level converter for MuJoCo models to Isaac Lab.

    Supports multiple output formats:
    - URDF (for standard ROS/Isaac import)
    - USD (requires Isaac Sim tools)

    References:
    - Isaac Sim MJCF Importer: https://docs.isaacsim.omniverse.nvidia.com/latest/
    - mjcf2urdf: https://github.com/iory/mjcf2urdf
    """

    def __init__(self, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
        self.urdf_converter = MJCFToURDFConverter(config)

    def convert(self, mjcf_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        Convert MJCF to Isaac-compatible format.

        Args:
            mjcf_path: Input MJCF file
            output_path: Output file path

        Returns:
            Conversion result
        """
        output_format = Path(output_path).suffix.lower()

        if output_format in [".urdf", ".xacro"]:
            return self.urdf_converter.convert(mjcf_path, output_path, **kwargs)

        elif output_format in [".usd", ".usda", ".usdc"]:
            return self._convert_to_usd(mjcf_path, output_path, **kwargs)

        else:
            return {
                "success": False,
                "error": f"Unsupported output format: {output_format}",
            }

    def _convert_to_usd(self, mjcf_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        Convert MJCF to USD format.

        Note: Direct USD conversion requires Isaac Sim.
        For standalone conversion, use URDF as intermediate format.
        """
        try:
            # Check if Isaac Sim is available
            # This would require omniverse/isaac-sim to be installed
            return {
                "success": False,
                "error": "USD conversion requires Isaac Sim installation",
                "recommendation": (
                    "Use one of these alternatives:\n"
                    "1. Convert to URDF first, then import URDF to Isaac Sim\n"
                    "2. Use Isaac Sim's native MJCF importer (Isaac Sim 4.5.0+)\n"
                    "3. Use Newton's USD converter: https://github.com/newton-physics/newton"
                ),
            }
        except ImportError:
            return {
                "success": False,
                "error": "Isaac Sim not available for USD export",
            }

    def convert_with_isaac_sim(self, mjcf_path: str, output_usd_path: str) -> Dict[str, Any]:
        """
        Convert using Isaac Sim's native MJCF importer.

        This method requires Isaac Sim 4.5.0+ to be installed and running.

        Reference:
        - https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_mjcf.html
        """
        try:
            # Isaac Sim import (requires running Isaac Sim environment)
            from omni.isaac.mjcf import _mjcf

            mjcf_interface = _mjcf.acquire_mjcf_interface()

            # Import configuration
            import_config = _mjcf.ImportConfig()
            import_config.fix_base = False
            import_config.import_sites = True
            import_config.self_collision = False

            # Import MJCF
            success = mjcf_interface.import_asset(mjcf_path, output_usd_path, import_config)

            return {
                "success": success,
                "input": mjcf_path,
                "output": output_usd_path,
                "method": "isaac_sim_native",
            }

        except ImportError:
            return {
                "success": False,
                "error": "Isaac Sim not available. Run this within Isaac Sim Python environment.",
            }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Convert MuJoCo MJCF models to Isaac Lab compatible formats.")
    parser.add_argument("--input", "-i", required=True, help="Input MJCF file path")
    parser.add_argument("--output", "-o", required=True, help="Output file path (URDF or USD)")
    parser.add_argument("--name", "-n", help="Override robot name")
    parser.add_argument("--validate", action="store_true", help="Validate output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Configure
    config = ConversionConfig(
        validate_output=args.validate,
    )

    converter = MuJoCoToIsaacConverter(config)
    result = converter.convert(args.input, args.output, robot_name=args.name)

    # Print result
    if result.get("success"):
        print("Conversion successful!")
        print(f"  Output: {result['output']}")
        print(f"  Bodies: {result.get('bodies_converted', 'N/A')}")
        print(f"  Joints: {result.get('joints_converted', 'N/A')}")

        if result.get("validation"):
            val = result["validation"]
            if val["valid"]:
                print("  Validation: PASSED")
            else:
                print(f"  Validation: FAILED - {val.get('error', 'Unknown')}")

        if result.get("warnings"):
            print("  Warnings:")
            for w in result["warnings"]:
                print(f"    - {w}")
    else:
        print(f"Conversion failed: {result.get('error', 'Unknown error')}")
        if result.get("recommendation"):
            print(f"\nRecommendation:\n{result['recommendation']}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
