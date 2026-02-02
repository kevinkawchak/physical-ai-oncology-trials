#!/usr/bin/env python3
"""
Physics Equivalence Test Suite for Isaac ↔ MuJoCo Conversion

This module provides comprehensive validation tests to ensure physics behavior
consistency between Isaac Lab and MuJoCo after model conversion.

Framework Versions:
    - MuJoCo: 3.4.0 (https://github.com/google-deepmind/mujoco/releases)
    - Isaac Lab: 2.3.2+ (https://github.com/isaac-sim/IsaacLab/releases)

Test Categories:
    1. Kinematic Validation - Forward kinematics, Jacobians
    2. Dynamic Validation - Free-fall, pendulum, trajectory tracking
    3. Contact Validation - Contact forces, friction behavior
    4. Round-Trip Validation - Format conversion fidelity

References:
    - MuJoCo Python Bindings: https://mujoco.readthedocs.io/en/stable/python.html
    - Physics Validation: https://mujoco.readthedocs.io/en/stable/computation.html

Usage:
    python physics_equivalence_tests.py --source robot.urdf --target robot.xml
    python physics_equivalence_tests.py --source robot.urdf --round-trip --report results.html

Last updated: February 2026
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import json
import warnings
from datetime import datetime

# Optional imports
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    warnings.warn("MuJoCo not installed. Some tests will be skipped.")


@dataclass
class ValidationTolerance:
    """Tolerance thresholds for validation tests."""
    position: float = 0.001        # rad or m
    velocity: float = 0.01         # rad/s or m/s
    acceleration: float = 0.1      # rad/s² or m/s²
    force: float = 0.1             # N or N·m
    trajectory: float = 0.001      # m
    jacobian: float = 1e-6         # dimensionless
    mass: float = 1e-6             # kg
    inertia: float = 1e-6          # kg·m²


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    error_value: float = 0.0
    tolerance: float = 0.0
    details: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ValidationReport:
    """Complete validation report."""
    source_model: str
    target_model: str
    direction: str  # "isaac_to_mujoco", "mujoco_to_isaac", "round_trip"
    tests: List[TestResult] = field(default_factory=list)
    overall_passed: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_test(self, result: TestResult) -> None:
        self.tests.append(result)
        if not result.passed:
            self.overall_passed = False

    def to_dict(self) -> Dict:
        return {
            "source_model": self.source_model,
            "target_model": self.target_model,
            "direction": self.direction,
            "overall_passed": self.overall_passed,
            "tests": [
                {
                    "name": t.name,
                    "passed": t.passed,
                    "error_value": t.error_value,
                    "tolerance": t.tolerance,
                    "details": t.details,
                }
                for t in self.tests
            ],
            "timestamp": self.timestamp,
        }


class KinematicValidator:
    """
    Validate kinematic properties between models.

    Tests:
    - Joint count and types
    - Link structure
    - Forward kinematics consistency
    - Jacobian consistency
    """

    def __init__(self, tolerance: ValidationTolerance):
        self.tolerance = tolerance

    def validate_structure(
        self,
        source_model,
        target_model
    ) -> TestResult:
        """Compare kinematic structure."""
        source_nq = source_model.nq if hasattr(source_model, 'nq') else 0
        target_nq = target_model.nq if hasattr(target_model, 'nq') else 0

        source_nv = source_model.nv if hasattr(source_model, 'nv') else 0
        target_nv = target_model.nv if hasattr(target_model, 'nv') else 0

        source_nbody = source_model.nbody if hasattr(source_model, 'nbody') else 0
        target_nbody = target_model.nbody if hasattr(target_model, 'nbody') else 0

        passed = (source_nq == target_nq and
                  source_nv == target_nv and
                  source_nbody == target_nbody)

        return TestResult(
            name="kinematic_structure",
            passed=passed,
            details={
                "source": {"nq": source_nq, "nv": source_nv, "nbody": source_nbody},
                "target": {"nq": target_nq, "nv": target_nv, "nbody": target_nbody},
            }
        )

    def validate_forward_kinematics(
        self,
        source_model,
        source_data,
        target_model,
        target_data,
        test_configs: Optional[np.ndarray] = None
    ) -> TestResult:
        """
        Compare forward kinematics at multiple configurations.

        Sets same joint positions in both models and compares
        end-effector (or body) positions.
        """
        if not MUJOCO_AVAILABLE:
            return TestResult(
                name="forward_kinematics",
                passed=False,
                details={"error": "MuJoCo not available"}
            )

        # Generate test configurations if not provided
        if test_configs is None:
            n_configs = 10
            nq = min(source_model.nq, target_model.nq)
            test_configs = np.random.uniform(-np.pi, np.pi, (n_configs, nq))

        max_error = 0.0
        errors = []

        for config in test_configs:
            # Set configuration in source
            source_data.qpos[:len(config)] = config
            mujoco.mj_forward(source_model, source_data)

            # Set configuration in target
            target_data.qpos[:len(config)] = config
            mujoco.mj_forward(target_model, target_data)

            # Compare body positions
            for i in range(min(source_model.nbody, target_model.nbody)):
                source_pos = source_data.xpos[i]
                target_pos = target_data.xpos[i]
                error = np.linalg.norm(source_pos - target_pos)
                errors.append(error)
                max_error = max(max_error, error)

        passed = max_error < self.tolerance.trajectory

        return TestResult(
            name="forward_kinematics",
            passed=passed,
            error_value=max_error,
            tolerance=self.tolerance.trajectory,
            details={
                "max_error": max_error,
                "mean_error": np.mean(errors),
                "n_configs_tested": len(test_configs),
            }
        )

    def validate_jacobian(
        self,
        source_model,
        source_data,
        target_model,
        target_data,
        body_id: int = -1
    ) -> TestResult:
        """
        Compare Jacobian matrices at current configuration.
        """
        if not MUJOCO_AVAILABLE:
            return TestResult(
                name="jacobian",
                passed=False,
                details={"error": "MuJoCo not available"}
            )

        # Use last body if not specified
        if body_id < 0:
            body_id = source_model.nbody - 1

        # Compute Jacobians
        source_jacp = np.zeros((3, source_model.nv))
        source_jacr = np.zeros((3, source_model.nv))
        mujoco.mj_jacBody(source_model, source_data, source_jacp, source_jacr, body_id)

        target_jacp = np.zeros((3, target_model.nv))
        target_jacr = np.zeros((3, target_model.nv))
        mujoco.mj_jacBody(target_model, target_data, target_jacp, target_jacr, body_id)

        # Compare
        nv = min(source_model.nv, target_model.nv)
        jacp_error = np.max(np.abs(source_jacp[:, :nv] - target_jacp[:, :nv]))
        jacr_error = np.max(np.abs(source_jacr[:, :nv] - target_jacr[:, :nv]))
        max_error = max(jacp_error, jacr_error)

        passed = max_error < self.tolerance.jacobian

        return TestResult(
            name="jacobian",
            passed=passed,
            error_value=max_error,
            tolerance=self.tolerance.jacobian,
            details={
                "position_jacobian_error": jacp_error,
                "rotation_jacobian_error": jacr_error,
                "body_id": body_id,
            }
        )


class DynamicValidator:
    """
    Validate dynamic behavior between models.

    Tests:
    - Free-fall test
    - Pendulum test
    - Trajectory tracking
    - Actuator response
    """

    def __init__(self, tolerance: ValidationTolerance):
        self.tolerance = tolerance

    def validate_free_fall(
        self,
        source_model,
        source_data,
        target_model,
        target_data,
        duration: float = 2.0
    ) -> TestResult:
        """
        Compare free-fall behavior.

        Drops both models from same initial height and compares
        final positions/velocities.
        """
        if not MUJOCO_AVAILABLE:
            return TestResult(
                name="free_fall",
                passed=False,
                details={"error": "MuJoCo not available"}
            )

        # Reset both models
        mujoco.mj_resetData(source_model, source_data)
        mujoco.mj_resetData(target_model, target_data)

        # Set initial height if floating base
        if source_model.nq > 7:  # Has floating base
            source_data.qpos[2] = 1.0  # z position
            target_data.qpos[2] = 1.0

        # Simulate
        n_steps = int(duration / source_model.opt.timestep)

        for _ in range(n_steps):
            mujoco.mj_step(source_model, source_data)
            mujoco.mj_step(target_model, target_data)

        # Compare final positions
        pos_error = np.linalg.norm(source_data.qpos - target_data.qpos)
        vel_error = np.linalg.norm(source_data.qvel - target_data.qvel)

        max_error = max(pos_error, vel_error)
        passed = pos_error < self.tolerance.position and vel_error < self.tolerance.velocity

        return TestResult(
            name="free_fall",
            passed=passed,
            error_value=max_error,
            tolerance=self.tolerance.position,
            details={
                "position_error": pos_error,
                "velocity_error": vel_error,
                "duration": duration,
                "n_steps": n_steps,
            }
        )

    def validate_pendulum(
        self,
        source_model,
        source_data,
        target_model,
        target_data,
        joint_id: int = 0,
        initial_angle: float = np.pi / 4,
        duration: float = 10.0
    ) -> TestResult:
        """
        Compare pendulum behavior for a specific joint.

        Initializes joint at given angle and lets it swing freely.
        """
        if not MUJOCO_AVAILABLE:
            return TestResult(
                name="pendulum",
                passed=False,
                details={"error": "MuJoCo not available"}
            )

        # Reset
        mujoco.mj_resetData(source_model, source_data)
        mujoco.mj_resetData(target_model, target_data)

        # Set initial angle
        if joint_id < source_model.nq and joint_id < target_model.nq:
            source_data.qpos[joint_id] = initial_angle
            target_data.qpos[joint_id] = initial_angle

        # Simulate and record positions
        n_steps = int(duration / source_model.opt.timestep)
        sample_interval = max(1, n_steps // 100)

        source_positions = []
        target_positions = []

        for i in range(n_steps):
            mujoco.mj_step(source_model, source_data)
            mujoco.mj_step(target_model, target_data)

            if i % sample_interval == 0:
                source_positions.append(source_data.qpos[joint_id])
                target_positions.append(target_data.qpos[joint_id])

        source_positions = np.array(source_positions)
        target_positions = np.array(target_positions)

        # Compute error
        position_error = np.abs(source_positions - target_positions)
        max_error = np.max(position_error)
        mean_error = np.mean(position_error)

        # Allow 1% deviation
        passed = max_error < (initial_angle * 0.01)

        return TestResult(
            name="pendulum",
            passed=passed,
            error_value=max_error,
            tolerance=initial_angle * 0.01,
            details={
                "max_error": max_error,
                "mean_error": mean_error,
                "joint_id": joint_id,
                "initial_angle": initial_angle,
                "duration": duration,
            }
        )

    def validate_trajectory_tracking(
        self,
        source_model,
        source_data,
        target_model,
        target_data,
        trajectory: Optional[np.ndarray] = None,
        duration: float = 5.0
    ) -> TestResult:
        """
        Compare trajectory tracking with same control inputs.
        """
        if not MUJOCO_AVAILABLE:
            return TestResult(
                name="trajectory_tracking",
                passed=False,
                details={"error": "MuJoCo not available"}
            )

        # Reset
        mujoco.mj_resetData(source_model, source_data)
        mujoco.mj_resetData(target_model, target_data)

        # Generate random control trajectory if not provided
        n_steps = int(duration / source_model.opt.timestep)
        nu = min(source_model.nu, target_model.nu)

        if trajectory is None:
            trajectory = np.random.uniform(-0.5, 0.5, (n_steps, nu))

        errors = []

        for i in range(min(n_steps, len(trajectory))):
            # Apply same control
            source_data.ctrl[:nu] = trajectory[i]
            target_data.ctrl[:nu] = trajectory[i]

            # Step
            mujoco.mj_step(source_model, source_data)
            mujoco.mj_step(target_model, target_data)

            # Record error
            nq = min(source_model.nq, target_model.nq)
            error = np.linalg.norm(source_data.qpos[:nq] - target_data.qpos[:nq])
            errors.append(error)

        max_error = np.max(errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))

        passed = rmse < self.tolerance.trajectory

        return TestResult(
            name="trajectory_tracking",
            passed=passed,
            error_value=rmse,
            tolerance=self.tolerance.trajectory,
            details={
                "max_error": max_error,
                "rmse": rmse,
                "duration": duration,
                "n_steps": len(errors),
            }
        )


class ContactValidator:
    """
    Validate contact behavior between models.
    """

    def __init__(self, tolerance: ValidationTolerance):
        self.tolerance = tolerance

    def validate_contact_force(
        self,
        source_model,
        source_data,
        target_model,
        target_data
    ) -> TestResult:
        """
        Compare contact forces when models are in contact with ground.
        """
        if not MUJOCO_AVAILABLE:
            return TestResult(
                name="contact_force",
                passed=False,
                details={"error": "MuJoCo not available"}
            )

        # Reset and step to establish contact
        mujoco.mj_resetData(source_model, source_data)
        mujoco.mj_resetData(target_model, target_data)

        # Step to reach steady state
        for _ in range(1000):
            mujoco.mj_step(source_model, source_data)
            mujoco.mj_step(target_model, target_data)

        # Compare contact forces
        source_forces = np.zeros(6)
        target_forces = np.zeros(6)

        for i in range(source_data.ncon):
            contact = source_data.contact[i]
            c_force = np.zeros(6)
            mujoco.mj_contactForce(source_model, source_data, i, c_force)
            source_forces += c_force

        for i in range(target_data.ncon):
            contact = target_data.contact[i]
            c_force = np.zeros(6)
            mujoco.mj_contactForce(target_model, target_data, i, c_force)
            target_forces += c_force

        force_error = np.linalg.norm(source_forces - target_forces)
        passed = force_error < self.tolerance.force

        return TestResult(
            name="contact_force",
            passed=passed,
            error_value=force_error,
            tolerance=self.tolerance.force,
            details={
                "source_force": source_forces.tolist(),
                "target_force": target_forces.tolist(),
                "source_ncon": source_data.ncon,
                "target_ncon": target_data.ncon,
            }
        )


class PhysicsEquivalenceValidator:
    """
    Complete physics equivalence validation suite.

    Coordinates all validation tests and generates comprehensive reports.
    """

    def __init__(self, tolerance: Optional[ValidationTolerance] = None):
        self.tolerance = tolerance or ValidationTolerance()
        self.kinematic = KinematicValidator(self.tolerance)
        self.dynamic = DynamicValidator(self.tolerance)
        self.contact = ContactValidator(self.tolerance)

    def validate(
        self,
        source_path: str,
        target_path: str,
        direction: str = "isaac_to_mujoco"
    ) -> ValidationReport:
        """
        Run complete validation suite.

        Args:
            source_path: Path to source model
            target_path: Path to target model
            direction: Conversion direction for reporting

        Returns:
            Comprehensive validation report
        """
        report = ValidationReport(
            source_model=source_path,
            target_model=target_path,
            direction=direction
        )

        if not MUJOCO_AVAILABLE:
            report.add_test(TestResult(
                name="mujoco_check",
                passed=False,
                details={"error": "MuJoCo not available"}
            ))
            return report

        try:
            # Load models
            source_model = mujoco.MjModel.from_xml_path(source_path)
            source_data = mujoco.MjData(source_model)

            target_model = mujoco.MjModel.from_xml_path(target_path)
            target_data = mujoco.MjData(target_model)

            # Kinematic tests
            report.add_test(self.kinematic.validate_structure(
                source_model, target_model
            ))

            report.add_test(self.kinematic.validate_forward_kinematics(
                source_model, source_data, target_model, target_data
            ))

            report.add_test(self.kinematic.validate_jacobian(
                source_model, source_data, target_model, target_data
            ))

            # Dynamic tests
            report.add_test(self.dynamic.validate_free_fall(
                source_model, source_data, target_model, target_data
            ))

            report.add_test(self.dynamic.validate_pendulum(
                source_model, source_data, target_model, target_data
            ))

            report.add_test(self.dynamic.validate_trajectory_tracking(
                source_model, source_data, target_model, target_data
            ))

            # Contact tests
            report.add_test(self.contact.validate_contact_force(
                source_model, source_data, target_model, target_data
            ))

        except Exception as e:
            report.add_test(TestResult(
                name="load_models",
                passed=False,
                details={"error": str(e)}
            ))

        return report

    def validate_round_trip(
        self,
        original_path: str,
        intermediate_path: str,
        recovered_path: str
    ) -> ValidationReport:
        """
        Validate round-trip conversion fidelity.

        Compares: original → intermediate → recovered
        """
        report = ValidationReport(
            source_model=original_path,
            target_model=recovered_path,
            direction="round_trip"
        )

        # First validate original vs intermediate
        intermediate_report = self.validate(
            original_path, intermediate_path, "forward"
        )
        for test in intermediate_report.tests:
            test.name = f"forward_{test.name}"
            report.add_test(test)

        # Then validate intermediate vs recovered
        recovery_report = self.validate(
            intermediate_path, recovered_path, "reverse"
        )
        for test in recovery_report.tests:
            test.name = f"reverse_{test.name}"
            report.add_test(test)

        # Finally validate original vs recovered (full round-trip)
        full_report = self.validate(
            original_path, recovered_path, "full_round_trip"
        )
        for test in full_report.tests:
            test.name = f"roundtrip_{test.name}"
            report.add_test(test)

        return report


def generate_html_report(report: ValidationReport, output_path: str) -> None:
    """Generate HTML report from validation results."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Physics Equivalence Validation Report</title>
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
    </style>
</head>
<body>
    <h1>Physics Equivalence Validation Report</h1>

    <div class="summary">
        <h2>Summary: <span class="{'passed' if report.overall_passed else 'failed'}">
            {'PASSED' if report.overall_passed else 'FAILED'}
        </span></h2>
        <p><strong>Source:</strong> {report.source_model}</p>
        <p><strong>Target:</strong> {report.target_model}</p>
        <p><strong>Direction:</strong> {report.direction}</p>
        <p><strong>Timestamp:</strong> {report.timestamp}</p>
    </div>

    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Status</th>
            <th>Error Value</th>
            <th>Tolerance</th>
            <th>Details</th>
        </tr>
"""

    for test in report.tests:
        status_class = "passed" if test.passed else "failed"
        status_text = "PASSED" if test.passed else "FAILED"
        details_str = json.dumps(test.details, indent=2) if test.details else "-"

        html += f"""
        <tr>
            <td>{test.name}</td>
            <td class="{status_class}">{status_text}</td>
            <td>{test.error_value:.6g}</td>
            <td>{test.tolerance:.6g}</td>
            <td><pre>{details_str}</pre></td>
        </tr>
"""

    html += """
    </table>

    <h2>References</h2>
    <ul>
        <li><a href="https://mujoco.readthedocs.io/en/stable/">MuJoCo Documentation</a></li>
        <li><a href="https://isaac-sim.github.io/IsaacLab/">Isaac Lab Documentation</a></li>
        <li><a href="https://github.com/google-deepmind/mujoco/releases">MuJoCo Releases</a></li>
    </ul>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Physics equivalence tests for Isaac ↔ MuJoCo conversion."
    )
    parser.add_argument(
        "--source", "-s", required=True,
        help="Source model path (MJCF or URDF)"
    )
    parser.add_argument(
        "--target", "-t",
        help="Target model path (for direct comparison)"
    )
    parser.add_argument(
        "--direction", "-d",
        choices=["isaac_to_mujoco", "mujoco_to_isaac"],
        default="isaac_to_mujoco",
        help="Conversion direction"
    )
    parser.add_argument(
        "--round-trip", action="store_true",
        help="Perform round-trip validation"
    )
    parser.add_argument(
        "--tolerance",
        choices=["strict", "standard", "relaxed"],
        default="standard",
        help="Tolerance level"
    )
    parser.add_argument(
        "--report", "-r",
        help="Output HTML report path"
    )
    parser.add_argument(
        "--json", "-j",
        help="Output JSON results path"
    )

    args = parser.parse_args()

    # Set tolerance
    if args.tolerance == "strict":
        tolerance = ValidationTolerance(
            position=0.0001,
            velocity=0.001,
            trajectory=0.0001,
        )
    elif args.tolerance == "relaxed":
        tolerance = ValidationTolerance(
            position=0.01,
            velocity=0.1,
            trajectory=0.01,
        )
    else:
        tolerance = ValidationTolerance()

    validator = PhysicsEquivalenceValidator(tolerance)

    # Run validation
    if args.round_trip:
        # For round-trip, we need intermediate and recovered paths
        print("Round-trip validation not yet implemented in CLI.")
        print("Use Python API for round-trip validation.")
        return 1
    else:
        if not args.target:
            print("Error: --target required for direct comparison")
            return 1

        report = validator.validate(args.source, args.target, args.direction)

    # Output results
    print("\n" + "=" * 60)
    print("PHYSICS EQUIVALENCE VALIDATION REPORT")
    print("=" * 60)
    print(f"Source: {report.source_model}")
    print(f"Target: {report.target_model}")
    print(f"Direction: {report.direction}")
    print(f"Overall: {'PASSED' if report.overall_passed else 'FAILED'}")
    print("-" * 60)

    for test in report.tests:
        status = "✓ PASS" if test.passed else "✗ FAIL"
        print(f"  {status} {test.name}: error={test.error_value:.6g} (tol={test.tolerance:.6g})")

    print("=" * 60)

    # Save reports
    if args.report:
        generate_html_report(report, args.report)
        print(f"HTML report saved: {args.report}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"JSON results saved: {args.json}")

    return 0 if report.overall_passed else 1


if __name__ == "__main__":
    exit(main())
