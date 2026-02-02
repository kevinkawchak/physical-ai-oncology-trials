#!/usr/bin/env python3
"""
Robot Model Validator for Unified Repository

This tool validates robot models against the Q1 2026 standards for the
Unified Robot Model Repository, ensuring compatibility across formats
and frameworks.

Validation Levels:
    1. Format Validation - Ensures files load in target frameworks
    2. Kinematic Validation - Checks structural consistency
    3. Dynamic Validation - Verifies physics properties
    4. Cross-Framework Validation - Compares behavior across simulators

Framework Versions:
    - MuJoCo: 3.4.0 (https://github.com/google-deepmind/mujoco/releases)
    - Isaac Lab: 2.3.2+ (https://github.com/isaac-sim/IsaacLab/releases)

Usage:
    python model_validator.py --model path/to/model/ --level 3
    python model_validator.py --model path/to/model.urdf --output report.html

Last updated: February 2026
"""

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
import json
import warnings
from datetime import datetime

# Optional imports
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    passed: bool
    level: int  # 1-4
    message: str = ""
    details: Dict = field(default_factory=dict)


@dataclass
class ModelValidationReport:
    """Complete validation report for a robot model."""
    model_path: str
    model_name: str
    validation_level: int
    results: List[ValidationResult] = field(default_factory=list)
    overall_passed: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_result(self, result: ValidationResult) -> None:
        self.results.append(result)
        if not result.passed:
            self.overall_passed = False

    def to_dict(self) -> Dict:
        return {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "validation_level": self.validation_level,
            "overall_passed": self.overall_passed,
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "level": r.level,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
            "timestamp": self.timestamp,
        }


class FormatValidator:
    """Level 1: Validate model file formats."""

    def validate_urdf(self, path: str) -> ValidationResult:
        """Validate URDF file structure."""
        try:
            tree = ET.parse(path)
            root = tree.getroot()

            if root.tag != "robot":
                return ValidationResult(
                    check_name="urdf_structure",
                    passed=False,
                    level=1,
                    message="Root element must be 'robot'",
                )

            # Check for required elements
            links = root.findall("link")
            joints = root.findall("joint")

            if len(links) == 0:
                return ValidationResult(
                    check_name="urdf_structure",
                    passed=False,
                    level=1,
                    message="No links found in URDF",
                )

            return ValidationResult(
                check_name="urdf_structure",
                passed=True,
                level=1,
                message=f"Valid URDF with {len(links)} links and {len(joints)} joints",
                details={"links": len(links), "joints": len(joints)},
            )

        except ET.ParseError as e:
            return ValidationResult(
                check_name="urdf_structure",
                passed=False,
                level=1,
                message=f"XML parse error: {e}",
            )
        except FileNotFoundError:
            return ValidationResult(
                check_name="urdf_structure",
                passed=False,
                level=1,
                message=f"File not found: {path}",
            )

    def validate_mjcf(self, path: str) -> ValidationResult:
        """Validate MJCF file structure."""
        try:
            tree = ET.parse(path)
            root = tree.getroot()

            if root.tag != "mujoco":
                return ValidationResult(
                    check_name="mjcf_structure",
                    passed=False,
                    level=1,
                    message="Root element must be 'mujoco'",
                )

            # Check for worldbody
            worldbody = root.find("worldbody")
            if worldbody is None:
                return ValidationResult(
                    check_name="mjcf_structure",
                    passed=False,
                    level=1,
                    message="No worldbody element found",
                )

            # Count bodies
            bodies = worldbody.findall(".//body")

            return ValidationResult(
                check_name="mjcf_structure",
                passed=True,
                level=1,
                message=f"Valid MJCF with {len(bodies)} bodies",
                details={"bodies": len(bodies)},
            )

        except ET.ParseError as e:
            return ValidationResult(
                check_name="mjcf_structure",
                passed=False,
                level=1,
                message=f"XML parse error: {e}",
            )
        except FileNotFoundError:
            return ValidationResult(
                check_name="mjcf_structure",
                passed=False,
                level=1,
                message=f"File not found: {path}",
            )

    def validate_mujoco_load(self, path: str) -> ValidationResult:
        """Validate MJCF loads in MuJoCo."""
        if not MUJOCO_AVAILABLE:
            return ValidationResult(
                check_name="mujoco_load",
                passed=False,
                level=1,
                message="MuJoCo not installed",
            )

        try:
            model = mujoco.MjModel.from_xml_path(path)
            data = mujoco.MjData(model)

            return ValidationResult(
                check_name="mujoco_load",
                passed=True,
                level=1,
                message=f"Successfully loaded in MuJoCo {mujoco.__version__}",
                details={
                    "nq": model.nq,
                    "nv": model.nv,
                    "nbody": model.nbody,
                    "njnt": model.njnt,
                    "ngeom": model.ngeom,
                    "nu": model.nu,
                },
            )

        except Exception as e:
            return ValidationResult(
                check_name="mujoco_load",
                passed=False,
                level=1,
                message=f"MuJoCo load error: {e}",
            )

    def validate_sdf(self, path: str) -> ValidationResult:
        """Validate SDF file structure."""
        try:
            tree = ET.parse(path)
            root = tree.getroot()

            if root.tag != "sdf":
                return ValidationResult(
                    check_name="sdf_structure",
                    passed=False,
                    level=1,
                    message="Root element must be 'sdf'",
                )

            models = root.findall("model")
            if len(models) == 0:
                return ValidationResult(
                    check_name="sdf_structure",
                    passed=False,
                    level=1,
                    message="No model element found",
                )

            return ValidationResult(
                check_name="sdf_structure",
                passed=True,
                level=1,
                message=f"Valid SDF with {len(models)} model(s)",
                details={"models": len(models)},
            )

        except ET.ParseError as e:
            return ValidationResult(
                check_name="sdf_structure",
                passed=False,
                level=1,
                message=f"XML parse error: {e}",
            )
        except FileNotFoundError:
            return ValidationResult(
                check_name="sdf_structure",
                passed=False,
                level=1,
                message=f"File not found: {path}",
            )


class KinematicValidator:
    """Level 2: Validate kinematic properties."""

    def validate_mass_properties(self, urdf_path: str) -> ValidationResult:
        """Validate mass properties are physically reasonable."""
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            issues = []
            total_mass = 0.0

            for link in root.findall("link"):
                link_name = link.get("name")
                inertial = link.find("inertial")

                if inertial is not None:
                    mass_elem = inertial.find("mass")
                    if mass_elem is not None:
                        mass = float(mass_elem.get("value", 0))
                        total_mass += mass

                        if mass <= 0:
                            issues.append(f"Link '{link_name}' has non-positive mass: {mass}")
                        elif mass > 1000:
                            issues.append(f"Link '{link_name}' has unusually large mass: {mass} kg")

                    # Check inertia tensor
                    inertia = inertial.find("inertia")
                    if inertia is not None:
                        ixx = float(inertia.get("ixx", 0))
                        iyy = float(inertia.get("iyy", 0))
                        izz = float(inertia.get("izz", 0))

                        if ixx <= 0 or iyy <= 0 or izz <= 0:
                            issues.append(f"Link '{link_name}' has non-positive principal inertia")

            passed = len(issues) == 0

            return ValidationResult(
                check_name="mass_properties",
                passed=passed,
                level=2,
                message=f"Total mass: {total_mass:.2f} kg" if passed else "; ".join(issues),
                details={"total_mass": total_mass, "issues": issues},
            )

        except Exception as e:
            return ValidationResult(
                check_name="mass_properties",
                passed=False,
                level=2,
                message=f"Error checking mass properties: {e}",
            )

    def validate_joint_limits(self, urdf_path: str) -> ValidationResult:
        """Validate joint limits are reasonable."""
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            issues = []
            joint_count = 0

            for joint in root.findall("joint"):
                joint_name = joint.get("name")
                joint_type = joint.get("type")

                if joint_type in ["revolute", "prismatic"]:
                    joint_count += 1
                    limit = joint.find("limit")

                    if limit is None:
                        issues.append(f"Joint '{joint_name}' missing limit element")
                    else:
                        lower = float(limit.get("lower", 0))
                        upper = float(limit.get("upper", 0))
                        effort = float(limit.get("effort", 0))
                        velocity = float(limit.get("velocity", 0))

                        if lower >= upper:
                            issues.append(f"Joint '{joint_name}' has invalid limits: lower >= upper")

                        if effort <= 0:
                            issues.append(f"Joint '{joint_name}' has non-positive effort limit")

                        if velocity <= 0:
                            issues.append(f"Joint '{joint_name}' has non-positive velocity limit")

            passed = len(issues) == 0

            return ValidationResult(
                check_name="joint_limits",
                passed=passed,
                level=2,
                message=f"Validated {joint_count} joints" if passed else "; ".join(issues[:3]),
                details={"joint_count": joint_count, "issues": issues},
            )

        except Exception as e:
            return ValidationResult(
                check_name="joint_limits",
                passed=False,
                level=2,
                message=f"Error checking joint limits: {e}",
            )

    def validate_link_consistency(self, urdf_path: str) -> ValidationResult:
        """Validate link-joint connectivity."""
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            link_names = set(link.get("name") for link in root.findall("link"))
            issues = []

            for joint in root.findall("joint"):
                joint_name = joint.get("name")
                parent = joint.find("parent")
                child = joint.find("child")

                if parent is not None:
                    parent_link = parent.get("link")
                    if parent_link not in link_names:
                        issues.append(f"Joint '{joint_name}' references unknown parent '{parent_link}'")

                if child is not None:
                    child_link = child.get("link")
                    if child_link not in link_names:
                        issues.append(f"Joint '{joint_name}' references unknown child '{child_link}'")

            passed = len(issues) == 0

            return ValidationResult(
                check_name="link_consistency",
                passed=passed,
                level=2,
                message=f"All {len(link_names)} links properly connected" if passed else "; ".join(issues),
                details={"link_count": len(link_names), "issues": issues},
            )

        except Exception as e:
            return ValidationResult(
                check_name="link_consistency",
                passed=False,
                level=2,
                message=f"Error checking link consistency: {e}",
            )


class DynamicValidator:
    """Level 3: Validate dynamic properties."""

    def validate_physics_stability(self, mjcf_path: str, steps: int = 1000) -> ValidationResult:
        """Validate model doesn't explode during simulation."""
        if not MUJOCO_AVAILABLE or not NUMPY_AVAILABLE:
            return ValidationResult(
                check_name="physics_stability",
                passed=False,
                level=3,
                message="MuJoCo or NumPy not available",
            )

        try:
            model = mujoco.MjModel.from_xml_path(mjcf_path)
            data = mujoco.MjData(model)

            # Run simulation
            initial_qpos = data.qpos.copy()
            max_velocity = 0.0

            for i in range(steps):
                mujoco.mj_step(model, data)

                # Check for NaN/Inf
                if np.any(np.isnan(data.qpos)) or np.any(np.isinf(data.qpos)):
                    return ValidationResult(
                        check_name="physics_stability",
                        passed=False,
                        level=3,
                        message=f"Simulation became unstable at step {i}",
                        details={"failed_step": i},
                    )

                # Track max velocity
                max_velocity = max(max_velocity, np.max(np.abs(data.qvel)))

                # Check for unreasonable velocities
                if max_velocity > 1000:
                    return ValidationResult(
                        check_name="physics_stability",
                        passed=False,
                        level=3,
                        message=f"Unreasonable velocity at step {i}: {max_velocity:.1f}",
                        details={"failed_step": i, "max_velocity": max_velocity},
                    )

            return ValidationResult(
                check_name="physics_stability",
                passed=True,
                level=3,
                message=f"Stable simulation for {steps} steps",
                details={
                    "steps": steps,
                    "max_velocity": float(max_velocity),
                },
            )

        except Exception as e:
            return ValidationResult(
                check_name="physics_stability",
                passed=False,
                level=3,
                message=f"Error during stability test: {e}",
            )

    def validate_actuator_response(self, mjcf_path: str) -> ValidationResult:
        """Validate actuators respond to control inputs."""
        if not MUJOCO_AVAILABLE or not NUMPY_AVAILABLE:
            return ValidationResult(
                check_name="actuator_response",
                passed=False,
                level=3,
                message="MuJoCo or NumPy not available",
            )

        try:
            model = mujoco.MjModel.from_xml_path(mjcf_path)
            data = mujoco.MjData(model)

            if model.nu == 0:
                return ValidationResult(
                    check_name="actuator_response",
                    passed=True,
                    level=3,
                    message="No actuators defined (passive model)",
                )

            # Apply control and check response
            mujoco.mj_resetData(model, data)
            initial_qpos = data.qpos.copy()

            # Apply unit control to first actuator
            data.ctrl[0] = 1.0

            # Step simulation
            for _ in range(100):
                mujoco.mj_step(model, data)

            # Check if position changed
            position_change = np.linalg.norm(data.qpos - initial_qpos)

            if position_change < 1e-6:
                return ValidationResult(
                    check_name="actuator_response",
                    passed=False,
                    level=3,
                    message="Actuator did not produce movement",
                    details={"position_change": float(position_change)},
                )

            return ValidationResult(
                check_name="actuator_response",
                passed=True,
                level=3,
                message=f"Actuators respond to control ({model.nu} actuators)",
                details={
                    "nu": model.nu,
                    "position_change": float(position_change),
                },
            )

        except Exception as e:
            return ValidationResult(
                check_name="actuator_response",
                passed=False,
                level=3,
                message=f"Error during actuator test: {e}",
            )


class CrossFrameworkValidator:
    """Level 4: Validate consistency across frameworks."""

    def validate_format_consistency(
        self,
        urdf_path: str,
        mjcf_path: str
    ) -> ValidationResult:
        """Compare structure between URDF and MJCF versions."""
        try:
            # Parse URDF
            urdf_tree = ET.parse(urdf_path)
            urdf_root = urdf_tree.getroot()
            urdf_links = len(urdf_root.findall("link"))
            urdf_joints = len([j for j in urdf_root.findall("joint")
                              if j.get("type") != "fixed"])

            # Parse MJCF
            mjcf_tree = ET.parse(mjcf_path)
            mjcf_root = mjcf_tree.getroot()
            mjcf_bodies = len(mjcf_root.findall(".//body"))
            mjcf_joints = len(mjcf_root.findall(".//joint"))

            # Compare (accounting for worldbody)
            link_match = abs(urdf_links - mjcf_bodies) <= 1
            joint_match = urdf_joints == mjcf_joints

            passed = link_match and joint_match

            return ValidationResult(
                check_name="format_consistency",
                passed=passed,
                level=4,
                message="Structure consistent across formats" if passed else "Structure mismatch detected",
                details={
                    "urdf_links": urdf_links,
                    "urdf_joints": urdf_joints,
                    "mjcf_bodies": mjcf_bodies,
                    "mjcf_joints": mjcf_joints,
                },
            )

        except Exception as e:
            return ValidationResult(
                check_name="format_consistency",
                passed=False,
                level=4,
                message=f"Error comparing formats: {e}",
            )


class ModelValidator:
    """Complete model validation tool."""

    def __init__(self):
        self.format_validator = FormatValidator()
        self.kinematic_validator = KinematicValidator()
        self.dynamic_validator = DynamicValidator()
        self.cross_framework_validator = CrossFrameworkValidator()

    def validate_model(
        self,
        model_path: str,
        level: int = 4
    ) -> ModelValidationReport:
        """
        Validate a robot model at specified level.

        Args:
            model_path: Path to model file or directory
            level: Validation level (1-4)

        Returns:
            Comprehensive validation report
        """
        path = Path(model_path)

        # Determine model name
        model_name = path.stem if path.is_file() else path.name

        report = ModelValidationReport(
            model_path=str(path),
            model_name=model_name,
            validation_level=level,
        )

        # Find model files
        if path.is_dir():
            urdf_files = list(path.glob("*.urdf"))
            mjcf_files = list(path.glob("*.xml"))
            sdf_files = list(path.glob("*.sdf"))
        else:
            suffix = path.suffix.lower()
            if suffix == ".urdf":
                urdf_files = [path]
                mjcf_files = []
                sdf_files = []
            elif suffix == ".xml":
                urdf_files = []
                mjcf_files = [path]
                sdf_files = []
            elif suffix == ".sdf":
                urdf_files = []
                mjcf_files = []
                sdf_files = [path]
            else:
                urdf_files = []
                mjcf_files = []
                sdf_files = []

        # Level 1: Format Validation
        if level >= 1:
            for urdf in urdf_files:
                report.add_result(self.format_validator.validate_urdf(str(urdf)))

            for mjcf in mjcf_files:
                report.add_result(self.format_validator.validate_mjcf(str(mjcf)))
                report.add_result(self.format_validator.validate_mujoco_load(str(mjcf)))

            for sdf in sdf_files:
                report.add_result(self.format_validator.validate_sdf(str(sdf)))

        # Level 2: Kinematic Validation
        if level >= 2:
            for urdf in urdf_files:
                report.add_result(self.kinematic_validator.validate_mass_properties(str(urdf)))
                report.add_result(self.kinematic_validator.validate_joint_limits(str(urdf)))
                report.add_result(self.kinematic_validator.validate_link_consistency(str(urdf)))

        # Level 3: Dynamic Validation
        if level >= 3:
            for mjcf in mjcf_files:
                report.add_result(self.dynamic_validator.validate_physics_stability(str(mjcf)))
                report.add_result(self.dynamic_validator.validate_actuator_response(str(mjcf)))

        # Level 4: Cross-Framework Validation
        if level >= 4 and urdf_files and mjcf_files:
            report.add_result(
                self.cross_framework_validator.validate_format_consistency(
                    str(urdf_files[0]), str(mjcf_files[0])
                )
            )

        return report


def generate_html_report(report: ModelValidationReport, output_path: str) -> None:
    """Generate HTML validation report."""
    passed_count = sum(1 for r in report.results if r.passed)
    total_count = len(report.results)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Validation Report: {report.model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary {{ background-color: {'#d4edda' if report.overall_passed else '#f8d7da'};
                   padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .level {{ font-weight: bold; }}
        .level-1 {{ color: #17a2b8; }}
        .level-2 {{ color: #28a745; }}
        .level-3 {{ color: #ffc107; }}
        .level-4 {{ color: #dc3545; }}
    </style>
</head>
<body>
    <h1>Model Validation Report</h1>

    <div class="summary">
        <h2>Summary: <span class="{'passed' if report.overall_passed else 'failed'}">
            {'PASSED' if report.overall_passed else 'FAILED'}
        </span></h2>
        <p><strong>Model:</strong> {report.model_name}</p>
        <p><strong>Path:</strong> {report.model_path}</p>
        <p><strong>Validation Level:</strong> {report.validation_level}</p>
        <p><strong>Results:</strong> {passed_count}/{total_count} checks passed</p>
        <p><strong>Timestamp:</strong> {report.timestamp}</p>
    </div>

    <h2>Validation Results</h2>
    <table>
        <tr>
            <th>Level</th>
            <th>Check</th>
            <th>Status</th>
            <th>Message</th>
        </tr>
"""

    for result in report.results:
        status_class = "passed" if result.passed else "failed"
        status_text = "PASS" if result.passed else "FAIL"

        html += f"""
        <tr>
            <td class="level level-{result.level}">L{result.level}</td>
            <td>{result.check_name}</td>
            <td class="{status_class}">{status_text}</td>
            <td>{result.message}</td>
        </tr>
"""

    html += """
    </table>

    <h2>Validation Levels</h2>
    <ul>
        <li><span class="level level-1">Level 1</span>: Format Validation - File loads correctly</li>
        <li><span class="level level-2">Level 2</span>: Kinematic Validation - Structure is valid</li>
        <li><span class="level level-3">Level 3</span>: Dynamic Validation - Physics behavior is reasonable</li>
        <li><span class="level level-4">Level 4</span>: Cross-Framework - Consistent across formats</li>
    </ul>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Validate robot models against Q1 2026 standards."
    )
    parser.add_argument(
        "--model", "-m", required=True,
        help="Path to model file or directory"
    )
    parser.add_argument(
        "--level", "-l", type=int, default=4, choices=[1, 2, 3, 4],
        help="Validation level (1-4, default: 4)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output HTML report path"
    )
    parser.add_argument(
        "--json", "-j",
        help="Output JSON results path"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    validator = ModelValidator()
    report = validator.validate_model(args.model, args.level)

    # Print summary
    print("\n" + "=" * 60)
    print(f"MODEL VALIDATION REPORT: {report.model_name}")
    print("=" * 60)
    print(f"Path: {report.model_path}")
    print(f"Level: {report.validation_level}")
    print(f"Overall: {'PASSED' if report.overall_passed else 'FAILED'}")
    print("-" * 60)

    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [L{result.level}] {status}: {result.check_name}")
        if args.verbose or not result.passed:
            print(f"       {result.message}")

    print("=" * 60)

    # Save reports
    if args.output:
        generate_html_report(report, args.output)
        print(f"HTML report saved: {args.output}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"JSON results saved: {args.json}")

    return 0 if report.overall_passed else 1


if __name__ == "__main__":
    exit(main())
