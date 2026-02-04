#!/usr/bin/env python3
"""
Isaac Lab to MuJoCo Conversion Pipeline

This module provides comprehensive conversion from NVIDIA Isaac Lab environments
and USD/URDF models to MuJoCo MJCF format, with full physics parameter mapping.

Framework Versions:
    - Isaac Lab: 2.3.2+ (https://github.com/isaac-sim/IsaacLab/releases)
    - Isaac Sim: 4.5.0+ (https://developer.nvidia.com/isaac-sim)
    - MuJoCo: 3.4.0 (https://github.com/google-deepmind/mujoco/releases)

References:
    - Isaac Sim MJCF Importer: https://docs.isaacsim.omniverse.nvidia.com/latest/
    - MuJoCo XML Reference: https://mujoco.readthedocs.io/en/stable/XMLreference.html
    - URDF2MJCF: https://github.com/eric-heiden/URDF2MJCF

Usage:
    python isaac_to_mujoco_pipeline.py --input robot.urdf --output robot.xml
    python isaac_to_mujoco_pipeline.py --input scene.usd --output scene.xml --validate

Last updated: February 2026
"""

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml
import warnings
import json

# Optional imports for full functionality
try:
    import mujoco

    MUJOCO_AVAILABLE = True
    MUJOCO_VERSION = mujoco.__version__
except ImportError:
    MUJOCO_AVAILABLE = False
    MUJOCO_VERSION = None
    warnings.warn("MuJoCo not installed. Validation features disabled.")


@dataclass
class PhysXContactParams:
    """Isaac PhysX contact parameters."""

    contact_offset: float = 0.001
    rest_offset: float = 0.0
    static_friction: float = 0.8
    dynamic_friction: float = 0.6
    restitution: float = 0.1
    bounce_threshold_velocity: float = 0.2


@dataclass
class MuJoCoContactParams:
    """MuJoCo contact parameters."""

    solref: Tuple[float, float] = (0.01, 1.0)
    solimp: Tuple[float, ...] = (0.9, 0.95, 0.001, 0.5, 2)
    friction: Tuple[float, float, float] = (0.8, 0.02, 0.01)
    condim: int = 3


@dataclass
class ConversionConfig:
    """Configuration for Isaac → MuJoCo conversion."""

    # Physics settings
    timestep: float = 0.002
    solver_iterations: int = 50
    solver_tolerance: float = 1e-6
    integrator: str = "implicitfast"

    # Contact settings
    default_friction: Tuple[float, float, float] = (0.8, 0.02, 0.01)
    default_solref: Tuple[float, float] = (0.01, 1.0)
    default_solimp: Tuple[float, ...] = (0.9, 0.95, 0.001, 0.5, 2)

    # Actuator settings
    default_gear: float = 50.0
    default_forcerange: Tuple[float, float] = (-100.0, 100.0)

    # Joint settings
    default_damping: float = 0.5
    default_armature: float = 0.01

    # Validation
    validate_output: bool = True
    strict_validation: bool = False


class PhysicsParameterConverter:
    """
    Convert physics parameters between Isaac PhysX and MuJoCo.

    Based on:
    - PhysX documentation: https://nvidia-omniverse.github.io/PhysX/
    - MuJoCo contact model: https://mujoco.readthedocs.io/en/stable/modeling.html#contact
    """

    @staticmethod
    def physx_to_mujoco_contact(params: PhysXContactParams) -> MuJoCoContactParams:
        """
        Convert PhysX contact parameters to MuJoCo format.

        The conversion is approximate as the physics models differ fundamentally:
        - PhysX uses penalty-based contact with explicit stiffness
        - MuJoCo uses constraint-based contact with impedance parameters
        """
        # Estimate stiffness from contact_offset
        # Smaller offset = stiffer contact
        estimated_stiffness = 1.0 / (params.contact_offset * 1000 + 1e-6)

        # Calculate timeconst (assuming unit mass for simplicity)
        # timeconst = 1 / sqrt(stiffness / mass)
        timeconst = 1.0 / np.sqrt(estimated_stiffness) if estimated_stiffness > 0 else 0.01
        timeconst = np.clip(timeconst, 0.001, 0.1)  # Reasonable range

        # Use critical damping (dampratio = 1.0)
        dampratio = 1.0

        # solimp parameters for impedance
        dmin = 0.9  # Minimum damping ratio
        dmax = 0.95  # Maximum damping ratio
        width = max(0.001, params.contact_offset)  # Transition width
        midpoint = 0.5
        power = 2

        # Friction mapping
        # MuJoCo friction: [sliding, torsional, rolling]
        friction = (
            params.static_friction,
            0.02,  # Torsional friction (typically small)
            0.01,  # Rolling friction (typically small)
        )

        return MuJoCoContactParams(
            solref=(timeconst, dampratio),
            solimp=(dmin, dmax, width, midpoint, power),
            friction=friction,
            condim=3,  # Standard 3D friction cone
        )

    @staticmethod
    def mujoco_to_physx_contact(params: MuJoCoContactParams) -> PhysXContactParams:
        """Convert MuJoCo contact parameters to PhysX format."""
        timeconst, dampratio = params.solref

        # Estimate stiffness from timeconst
        stiffness = 1.0 / (timeconst**2) if timeconst > 0 else 1e6

        # contact_offset from solimp width
        contact_offset = params.solimp[2] if len(params.solimp) > 2 else 0.001

        return PhysXContactParams(
            contact_offset=contact_offset,
            rest_offset=0.0,
            static_friction=params.friction[0],
            dynamic_friction=params.friction[0] * 0.8,
            restitution=0.1,
            bounce_threshold_velocity=0.2,
        )


class URDFToMJCFConverter:
    """
    Convert URDF robot models to MuJoCo MJCF format.

    This converter handles:
    - Kinematic structure (links, joints)
    - Inertial properties
    - Visual and collision geometry
    - Joint dynamics (damping, friction)
    - Actuator generation

    References:
    - URDF Specification: https://wiki.ros.org/urdf/XML
    - MJCF Reference: https://mujoco.readthedocs.io/en/stable/XMLreference.html
    """

    JOINT_TYPE_MAP = {
        "revolute": "hinge",
        "continuous": "hinge",
        "prismatic": "slide",
        "fixed": None,  # Fixed joints become direct parent-child
        "floating": "free",
        "planar": None,  # Requires decomposition
    }

    def __init__(self, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
        self.physics_converter = PhysicsParameterConverter()
        self.conversion_log: List[Dict] = []
        self.warnings: List[str] = []

    def convert(
        self,
        urdf_path: str,
        output_path: str,
        robot_name: Optional[str] = None,
        physx_params: Optional[PhysXContactParams] = None,
    ) -> Dict[str, Any]:
        """
        Convert URDF file to MJCF format.

        Args:
            urdf_path: Path to input URDF file
            output_path: Path to output MJCF file
            robot_name: Override robot name (uses URDF name if None)
            physx_params: PhysX contact parameters to convert

        Returns:
            Conversion result with statistics and any warnings
        """
        # Parse URDF
        urdf_tree = ET.parse(urdf_path)
        urdf_root = urdf_tree.getroot()

        robot_name = robot_name or urdf_root.get("name", "robot")

        # Build kinematic structure
        links, joints = self._parse_urdf(urdf_root)

        # Create MJCF document
        mjcf_root = self._create_mjcf_root(robot_name)

        # Add compiler and options
        self._add_compiler_options(mjcf_root)

        # Add defaults
        self._add_defaults(mjcf_root, physx_params)

        # Add assets (meshes)
        self._add_assets(mjcf_root, links, urdf_path)

        # Build worldbody hierarchy
        worldbody = ET.SubElement(mjcf_root, "worldbody")
        self._build_body_hierarchy(worldbody, links, joints)

        # Add actuators
        self._add_actuators(mjcf_root, joints)

        # Write output
        tree = ET.ElementTree(mjcf_root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

        # Validate if MuJoCo is available
        validation_result = None
        if self.config.validate_output and MUJOCO_AVAILABLE:
            validation_result = self._validate_mjcf(output_path)

        result = {
            "success": True,
            "input": urdf_path,
            "output": output_path,
            "robot_name": robot_name,
            "links_converted": len(links),
            "joints_converted": len([j for j in joints.values() if j["type"] != "fixed"]),
            "warnings": self.warnings.copy(),
            "validation": validation_result,
        }

        self.conversion_log.append(result)
        return result

    def _parse_urdf(self, root: ET.Element) -> Tuple[Dict, Dict]:
        """Parse URDF into internal representation."""
        links = {}
        joints = {}

        # Parse links
        for link_elem in root.findall("link"):
            name = link_elem.get("name")
            links[name] = self._parse_link(link_elem)

        # Parse joints
        for joint_elem in root.findall("joint"):
            name = joint_elem.get("name")
            joints[name] = self._parse_joint(joint_elem)

        return links, joints

    def _parse_link(self, elem: ET.Element) -> Dict:
        """Parse a single URDF link."""
        link = {
            "name": elem.get("name"),
            "mass": 1.0,
            "inertia": np.eye(3) * 0.01,
            "com": np.zeros(3),
            "visual": None,
            "collision": None,
        }

        # Inertial properties
        inertial = elem.find("inertial")
        if inertial is not None:
            mass_elem = inertial.find("mass")
            if mass_elem is not None:
                link["mass"] = float(mass_elem.get("value", 1.0))

            origin = inertial.find("origin")
            if origin is not None:
                xyz = origin.get("xyz", "0 0 0")
                link["com"] = np.array([float(x) for x in xyz.split()])

            inertia_elem = inertial.find("inertia")
            if inertia_elem is not None:
                link["inertia"] = np.array(
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

        # Visual geometry
        visual = elem.find("visual")
        if visual is not None:
            link["visual"] = self._parse_geometry(visual)

        # Collision geometry
        collision = elem.find("collision")
        if collision is not None:
            link["collision"] = self._parse_geometry(collision)

        return link

    def _parse_geometry(self, elem: ET.Element) -> Optional[Dict]:
        """Parse geometry element from URDF."""
        geom = elem.find("geometry")
        if geom is None:
            return None

        origin = elem.find("origin")
        pos = np.zeros(3)
        quat = np.array([1, 0, 0, 0])  # wxyz

        if origin is not None:
            xyz = origin.get("xyz", "0 0 0")
            pos = np.array([float(x) for x in xyz.split()])
            rpy = origin.get("rpy", "0 0 0")
            rpy_vals = np.array([float(x) for x in rpy.split()])
            quat = self._rpy_to_quat(rpy_vals)

        geometry = {"pos": pos, "quat": quat}

        # Check geometry type
        for geom_type in ["box", "cylinder", "sphere", "mesh"]:
            geom_elem = geom.find(geom_type)
            if geom_elem is not None:
                geometry["type"] = geom_type

                if geom_type == "box":
                    size = geom_elem.get("size", "1 1 1")
                    geometry["size"] = [float(x) / 2 for x in size.split()]  # Half-size for MuJoCo

                elif geom_type == "cylinder":
                    geometry["radius"] = float(geom_elem.get("radius", 0.1))
                    geometry["length"] = float(geom_elem.get("length", 1.0)) / 2  # Half-length

                elif geom_type == "sphere":
                    geometry["radius"] = float(geom_elem.get("radius", 0.1))

                elif geom_type == "mesh":
                    geometry["filename"] = geom_elem.get("filename", "")
                    scale = geom_elem.get("scale", "1 1 1")
                    geometry["scale"] = [float(x) for x in scale.split()]

                break

        return geometry

    def _parse_joint(self, elem: ET.Element) -> Dict:
        """Parse a single URDF joint."""
        joint = {
            "name": elem.get("name"),
            "type": elem.get("type", "fixed"),
            "parent": elem.find("parent").get("link") if elem.find("parent") is not None else "world",
            "child": elem.find("child").get("link") if elem.find("child") is not None else "",
            "origin_xyz": np.zeros(3),
            "origin_rpy": np.zeros(3),
            "axis": np.array([0, 0, 1]),
            "lower": -np.pi,
            "upper": np.pi,
            "effort": 100.0,
            "velocity": 1.0,
            "damping": self.config.default_damping,
            "friction": 0.0,
        }

        # Origin
        origin = elem.find("origin")
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0")
            joint["origin_xyz"] = np.array([float(x) for x in xyz.split()])
            rpy = origin.get("rpy", "0 0 0")
            joint["origin_rpy"] = np.array([float(x) for x in rpy.split()])

        # Axis
        axis = elem.find("axis")
        if axis is not None:
            xyz = axis.get("xyz", "0 0 1")
            joint["axis"] = np.array([float(x) for x in xyz.split()])

        # Limits
        limit = elem.find("limit")
        if limit is not None:
            joint["lower"] = float(limit.get("lower", -np.pi))
            joint["upper"] = float(limit.get("upper", np.pi))
            joint["effort"] = float(limit.get("effort", 100.0))
            joint["velocity"] = float(limit.get("velocity", 1.0))

        # Dynamics
        dynamics = elem.find("dynamics")
        if dynamics is not None:
            joint["damping"] = float(dynamics.get("damping", self.config.default_damping))
            joint["friction"] = float(dynamics.get("friction", 0.0))

        return joint

    def _rpy_to_quat(self, rpy: np.ndarray) -> np.ndarray:
        """Convert roll-pitch-yaw to quaternion (wxyz)."""
        r, p, y = rpy
        cy = np.cos(y * 0.5)
        sy = np.sin(y * 0.5)
        cp = np.cos(p * 0.5)
        sp = np.sin(p * 0.5)
        cr = np.cos(r * 0.5)
        sr = np.sin(r * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    def _create_mjcf_root(self, name: str) -> ET.Element:
        """Create MJCF root element."""
        return ET.Element("mujoco", model=name)

    def _add_compiler_options(self, root: ET.Element) -> None:
        """Add compiler and option elements."""
        # Compiler settings
        ET.SubElement(root, "compiler", angle="radian", autolimits="true", meshdir="meshes")

        # Physics options
        ET.SubElement(
            root,
            "option",
            timestep=str(self.config.timestep),
            iterations=str(self.config.solver_iterations),
            tolerance=str(self.config.solver_tolerance),
            integrator=self.config.integrator,
            gravity="0 0 -9.81",
        )

    def _add_defaults(self, root: ET.Element, physx_params: Optional[PhysXContactParams]) -> None:
        """Add default element with contact and joint parameters."""
        default = ET.SubElement(root, "default")

        # Convert PhysX params if provided
        if physx_params:
            mujoco_params = self.physics_converter.physx_to_mujoco_contact(physx_params)
            friction = mujoco_params.friction
            solref = mujoco_params.solref
            solimp = mujoco_params.solimp
        else:
            friction = self.config.default_friction
            solref = self.config.default_solref
            solimp = self.config.default_solimp

        # Joint defaults
        ET.SubElement(
            default, "joint", damping=str(self.config.default_damping), armature=str(self.config.default_armature)
        )

        # Geom defaults
        ET.SubElement(
            default,
            "geom",
            friction=" ".join(str(f) for f in friction),
            solref=" ".join(str(s) for s in solref),
            solimp=" ".join(str(s) for s in solimp),
        )

    def _add_assets(self, root: ET.Element, links: Dict, urdf_path: str) -> None:
        """Add asset element with meshes."""
        asset = ET.SubElement(root, "asset")

        # Add ground texture and material
        ET.SubElement(
            asset,
            "texture",
            name="grid",
            type="2d",
            builtin="checker",
            width="512",
            height="512",
            rgb1="0.1 0.2 0.3",
            rgb2="0.2 0.3 0.4",
        )
        ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="8 8", reflectance="0.2")

        # Collect unique meshes
        urdf_dir = Path(urdf_path).parent
        meshes = set()

        for link in links.values():
            for geom_key in ["visual", "collision"]:
                geom = link.get(geom_key)
                if geom and geom.get("type") == "mesh":
                    meshes.add(geom["filename"])

        # Add mesh assets
        for mesh_path in meshes:
            # Resolve mesh path
            mesh_name = Path(mesh_path).stem
            resolved_path = self._resolve_mesh_path(mesh_path, urdf_dir)

            if resolved_path:
                ET.SubElement(asset, "mesh", name=mesh_name, file=str(resolved_path))
            else:
                self.warnings.append(f"Mesh not found: {mesh_path}")

    def _resolve_mesh_path(self, mesh_path: str, base_dir: Path) -> Optional[str]:
        """Resolve mesh path from URDF package:// or file:// syntax."""
        if mesh_path.startswith("package://"):
            # Extract package-relative path
            path_part = mesh_path.replace("package://", "")
            # Try common locations
            for candidate in [base_dir / path_part, base_dir / "meshes" / Path(path_part).name]:
                if candidate.exists():
                    return str(candidate)
            return Path(path_part).name  # Return just filename

        elif mesh_path.startswith("file://"):
            return mesh_path.replace("file://", "")

        else:
            # Relative path
            candidate = base_dir / mesh_path
            if candidate.exists():
                return str(candidate)
            return mesh_path

    def _build_body_hierarchy(self, worldbody: ET.Element, links: Dict, joints: Dict) -> None:
        """Build MJCF body hierarchy from URDF links/joints."""
        # Build parent-child relationships
        child_to_joint = {j["child"]: j for j in joints.values()}
        parent_children = {}
        for joint in joints.values():
            parent = joint["parent"]
            child = joint["child"]
            if parent not in parent_children:
                parent_children[parent] = []
            parent_children[parent].append(child)

        # Find root links (those not appearing as children)
        all_children = set(j["child"] for j in joints.values())
        root_links = [name for name in links.keys() if name not in all_children]

        # Add ground plane
        ET.SubElement(worldbody, "geom", name="ground", type="plane", size="5 5 0.1", material="grid")

        # Build hierarchy starting from roots
        for root_name in root_links:
            self._add_body_recursive(worldbody, root_name, links, child_to_joint, parent_children)

    def _add_body_recursive(
        self, parent_elem: ET.Element, link_name: str, links: Dict, child_to_joint: Dict, parent_children: Dict
    ) -> None:
        """Recursively add body elements."""
        link = links.get(link_name)
        if link is None:
            return

        # Get joint connecting this link to parent (if any)
        joint_info = child_to_joint.get(link_name)

        # Body position (from joint origin if available)
        pos = "0 0 0"
        if joint_info:
            pos = " ".join(f"{x:.6f}" for x in joint_info["origin_xyz"])

        # Create body element
        body = ET.SubElement(parent_elem, "body", name=link_name, pos=pos)

        # Add inertial
        if link["mass"] > 0:
            com_pos = " ".join(f"{x:.6f}" for x in link["com"])
            inertia = link["inertia"]

            # Check if diagonal
            is_diagonal = abs(inertia[0, 1]) < 1e-10 and abs(inertia[0, 2]) < 1e-10 and abs(inertia[1, 2]) < 1e-10

            if is_diagonal:
                diag = f"{inertia[0, 0]:.6g} {inertia[1, 1]:.6g} {inertia[2, 2]:.6g}"
                ET.SubElement(body, "inertial", mass=str(link["mass"]), pos=com_pos, diaginertia=diag)
            else:
                full = (
                    f"{inertia[0, 0]:.6g} {inertia[1, 1]:.6g} {inertia[2, 2]:.6g} "
                    f"{inertia[0, 1]:.6g} {inertia[0, 2]:.6g} {inertia[1, 2]:.6g}"
                )
                ET.SubElement(body, "inertial", mass=str(link["mass"]), pos=com_pos, fullinertia=full)

        # Add joint if this is not a root body and joint is not fixed
        if joint_info and joint_info["type"] != "fixed":
            mjcf_type = self.JOINT_TYPE_MAP.get(joint_info["type"])
            if mjcf_type:
                axis = " ".join(f"{x:.3f}" for x in joint_info["axis"])
                joint_attribs = {
                    "name": joint_info["name"],
                    "type": mjcf_type,
                    "axis": axis,
                    "damping": str(joint_info["damping"]),
                }

                # Add limits for non-continuous joints
                if joint_info["type"] != "continuous":
                    joint_attribs["range"] = f"{joint_info['lower']:.4f} {joint_info['upper']:.4f}"

                if joint_info["friction"] > 0:
                    joint_attribs["frictionloss"] = str(joint_info["friction"])

                ET.SubElement(body, "joint", **joint_attribs)

        # Add geometry
        for geom_key, geom_class in [("collision", "collision"), ("visual", "visual")]:
            geom = link.get(geom_key)
            if geom:
                self._add_geom(body, f"{link_name}_{geom_class}", geom)

        # Recursively add children
        children = parent_children.get(link_name, [])
        for child_name in children:
            self._add_body_recursive(body, child_name, links, child_to_joint, parent_children)

    def _add_geom(self, body: ET.Element, name: str, geom: Dict) -> None:
        """Add geometry to body."""
        attribs = {"name": name}

        # Position and orientation
        if np.any(geom["pos"] != 0):
            attribs["pos"] = " ".join(f"{x:.6f}" for x in geom["pos"])

        geom_type = geom.get("type")
        if geom_type == "box":
            attribs["type"] = "box"
            attribs["size"] = " ".join(f"{x:.6f}" for x in geom["size"])

        elif geom_type == "cylinder":
            attribs["type"] = "cylinder"
            attribs["size"] = f"{geom['radius']:.6f} {geom['length']:.6f}"

        elif geom_type == "sphere":
            attribs["type"] = "sphere"
            attribs["size"] = f"{geom['radius']:.6f}"

        elif geom_type == "mesh":
            attribs["type"] = "mesh"
            mesh_name = Path(geom["filename"]).stem
            attribs["mesh"] = mesh_name

        ET.SubElement(body, "geom", **attribs)

    def _add_actuators(self, root: ET.Element, joints: Dict) -> None:
        """Add actuator element for movable joints."""
        actuator = ET.SubElement(root, "actuator")

        for joint in joints.values():
            if joint["type"] in ["revolute", "prismatic", "continuous"]:
                ET.SubElement(
                    actuator,
                    "motor",
                    name=f"motor_{joint['name']}",
                    joint=joint["name"],
                    gear=str(self.config.default_gear),
                    forcerange=f"{self.config.default_forcerange[0]} {self.config.default_forcerange[1]}",
                )

    def _validate_mjcf(self, mjcf_path: str) -> Dict[str, Any]:
        """Validate generated MJCF file."""
        try:
            model = mujoco.MjModel.from_xml_path(mjcf_path)
            data = mujoco.MjData(model)

            return {
                "valid": True,
                "nq": model.nq,
                "nv": model.nv,
                "nbody": model.nbody,
                "njnt": model.njnt,
                "ngeom": model.ngeom,
                "nu": model.nu,
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
            }

    def get_conversion_log(self) -> List[Dict]:
        """Get history of all conversions."""
        return self.conversion_log.copy()


class IsaacToMuJoCoConverter:
    """
    High-level converter for Isaac Lab environments to MuJoCo.

    This class handles:
    - USD scene conversion
    - URDF model conversion
    - Environment configuration translation
    """

    def __init__(self, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
        self.urdf_converter = URDFToMJCFConverter(config)

    def convert_urdf(self, urdf_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Convert URDF file to MJCF."""
        return self.urdf_converter.convert(urdf_path, output_path, **kwargs)

    def convert_usd(self, usd_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        Convert USD file to MJCF.

        Note: This requires intermediate conversion through URDF.
        For native USD support, use Isaac Sim's MJCF exporter.
        """
        # Check if we can use Isaac Sim
        try:
            # This would require Isaac Sim to be installed
            # For now, we warn that USD conversion needs additional tools
            self.warnings = [
                "USD → MJCF conversion requires either:",
                "  1. Isaac Sim installed for native export, or",
                "  2. Intermediate USD → URDF conversion via Newton tools",
                "See: https://github.com/newton-physics/newton for USD converter",
            ]

            return {
                "success": False,
                "error": "USD conversion not available without Isaac Sim",
                "recommendations": self.warnings,
            }

        except ImportError:
            return {
                "success": False,
                "error": "USD conversion requires Isaac Sim installation",
            }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Convert Isaac Lab URDF/USD models to MuJoCo MJCF format.")
    parser.add_argument("--input", "-i", required=True, help="Input file path (URDF or USD)")
    parser.add_argument("--output", "-o", required=True, help="Output MJCF file path")
    parser.add_argument("--name", "-n", help="Override robot name")
    parser.add_argument("--timestep", type=float, default=0.002, help="Simulation timestep (default: 0.002)")
    parser.add_argument("--validate", action="store_true", help="Validate output with MuJoCo")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Configure conversion
    config = ConversionConfig(
        timestep=args.timestep,
        validate_output=args.validate,
    )

    converter = IsaacToMuJoCoConverter(config)

    # Determine input format
    input_path = Path(args.input)
    if input_path.suffix.lower() in [".urdf", ".xacro"]:
        result = converter.convert_urdf(args.input, args.output, robot_name=args.name)
    elif input_path.suffix.lower() in [".usd", ".usda", ".usdc"]:
        result = converter.convert_usd(args.input, args.output, robot_name=args.name)
    else:
        print(f"Unsupported input format: {input_path.suffix}")
        return 1

    # Print result
    if result["success"]:
        print("Conversion successful!")
        print(f"  Output: {result['output']}")
        print(f"  Links: {result['links_converted']}")
        print(f"  Joints: {result['joints_converted']}")

        if result.get("validation"):
            val = result["validation"]
            if val["valid"]:
                print("  Validation: PASSED")
                print(f"    Bodies: {val['nbody']}, Joints: {val['njnt']}, Actuators: {val['nu']}")
            else:
                print(f"  Validation: FAILED - {val['error']}")

        if result.get("warnings"):
            print("  Warnings:")
            for w in result["warnings"]:
                print(f"    - {w}")
    else:
        print(f"Conversion failed: {result.get('error', 'Unknown error')}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
