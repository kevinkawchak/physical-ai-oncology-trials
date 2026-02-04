#!/usr/bin/env python3
"""
Validation Benchmark Suite Runner

This tool executes the Q1 2026 validation benchmark suite for cross-framework
validation of physical AI simulations in oncology clinical trials.

Framework Versions:
    - MuJoCo: 3.4.0 (https://github.com/google-deepmind/mujoco/releases)
    - Isaac Lab: 2.3.2+ (https://github.com/isaac-sim/IsaacLab/releases)

Benchmark Categories:
    1. Physics Accuracy - Dynamics consistency across frameworks
    2. Performance - Simulation throughput and efficiency
    3. Sim-to-Real Gap - Transfer readiness assessment
    4. Cross-Framework - Framework equivalence validation

References:
    - robosuite: https://robosuite.ai/
    - ORBIT-Surgical: https://orbit-surgical.github.io/
    - Isaac Lab-Arena: https://developer.nvidia.com/blog/

Usage:
    python benchmark_runner.py --scenario peg_transfer --frameworks isaac,mujoco
    python benchmark_runner.py --suite full --output results/
    python benchmark_runner.py --list-scenarios

Last updated: February 2026
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

# Optional imports
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available")

try:
    import mujoco

    MUJOCO_AVAILABLE = True
    MUJOCO_VERSION = mujoco.__version__
except ImportError:
    MUJOCO_AVAILABLE = False
    MUJOCO_VERSION = None


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    scenario: str
    frameworks: List[str]
    num_episodes: int = 100
    max_steps: int = 1000
    seed: int = 42
    verbose: bool = False


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    category: str  # physics, performance, sim2real, cross_framework
    passed: bool
    value: float
    target: float
    unit: str
    details: Dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    scenario: str
    frameworks: List[str]
    results: List[BenchmarkResult] = field(default_factory=list)
    overall_passed: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0

    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)
        if not result.passed:
            self.overall_passed = False

    def to_dict(self) -> Dict:
        return {
            "scenario": self.scenario,
            "frameworks": self.frameworks,
            "overall_passed": self.overall_passed,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "results": [
                {
                    "name": r.name,
                    "category": r.category,
                    "passed": r.passed,
                    "value": r.value,
                    "target": r.target,
                    "unit": r.unit,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


# =============================================================================
# Benchmark Scenarios
# =============================================================================

SCENARIOS = {
    "peg_transfer": {
        "description": "Transfer peg between posts (FLS task)",
        "category": "manipulation",
        "difficulty": "easy",
        "robot": "any",
        "metrics": ["success_rate", "completion_time", "path_efficiency"],
    },
    "needle_insertion": {
        "description": "Insert needle into tissue phantom",
        "category": "surgical",
        "difficulty": "medium",
        "robot": "dvrk_psm",
        "metrics": ["insertion_accuracy", "force_profile", "tissue_damage"],
    },
    "tissue_manipulation": {
        "description": "Grasp and retract tissue",
        "category": "surgical",
        "difficulty": "medium",
        "robot": "dvrk_psm",
        "metrics": ["grasp_success", "retraction_distance", "tissue_stress"],
    },
    "suturing": {
        "description": "Place surgical suture through tissue",
        "category": "surgical",
        "difficulty": "hard",
        "robot": "dvrk_psm",
        "metrics": ["suture_quality", "tension_uniformity", "time_to_complete"],
    },
    "reach": {
        "description": "Reach target end-effector position",
        "category": "manipulation",
        "difficulty": "easy",
        "robot": "any",
        "metrics": ["position_error", "settling_time", "overshoot"],
    },
    "push": {
        "description": "Push object to goal position",
        "category": "manipulation",
        "difficulty": "medium",
        "robot": "any",
        "metrics": ["final_distance", "path_length", "contact_consistency"],
    },
    "free_fall": {
        "description": "Validate gravity dynamics",
        "category": "physics",
        "difficulty": "easy",
        "robot": "any",
        "metrics": ["position_error", "velocity_error", "energy_conservation"],
    },
    "pendulum": {
        "description": "Validate oscillatory dynamics",
        "category": "physics",
        "difficulty": "easy",
        "robot": "any",
        "metrics": ["period_error", "amplitude_decay", "frequency_match"],
    },
}


# =============================================================================
# Physics Benchmarks
# =============================================================================


class PhysicsBenchmark:
    """Physics accuracy benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run_free_fall(self, model_path: str) -> BenchmarkResult:
        """
        Free-fall benchmark: validate gravity-driven motion.

        Expected behavior:
        - Position: z(t) = z0 - 0.5 * g * t^2
        - Velocity: v(t) = -g * t
        """
        if not MUJOCO_AVAILABLE or not NUMPY_AVAILABLE:
            return BenchmarkResult(
                name="free_fall",
                category="physics",
                passed=False,
                value=0.0,
                target=0.001,
                unit="m",
                details={"error": "MuJoCo or NumPy not available"},
            )

        try:
            model = mujoco.MjModel.from_xml_path(model_path)
            data = mujoco.MjData(model)

            # Reset and record initial state
            mujoco.mj_resetData(model, data)

            # Set initial height if floating base
            initial_height = 1.0
            if model.nq >= 3:
                data.qpos[2] = initial_height

            mujoco.mj_forward(model, data)

            # Simulate free fall
            g = 9.81  # m/s^2
            dt = model.opt.timestep
            duration = 0.5  # seconds
            n_steps = int(duration / dt)

            errors = []

            for i in range(n_steps):
                mujoco.mj_step(model, data)

                t = (i + 1) * dt
                expected_z = initial_height - 0.5 * g * t * t
                actual_z = data.qpos[2] if model.nq >= 3 else 0

                error = abs(expected_z - actual_z)
                errors.append(error)

            max_error = max(errors)
            mean_error = np.mean(errors)

            passed = max_error < 0.01  # 1 cm tolerance

            return BenchmarkResult(
                name="free_fall",
                category="physics",
                passed=passed,
                value=max_error,
                target=0.01,
                unit="m",
                details={
                    "max_error": max_error,
                    "mean_error": mean_error,
                    "duration": duration,
                    "timestep": dt,
                },
            )

        except Exception as e:
            return BenchmarkResult(
                name="free_fall",
                category="physics",
                passed=False,
                value=0.0,
                target=0.01,
                unit="m",
                details={"error": str(e)},
            )

    def run_pendulum(self, model_path: str) -> BenchmarkResult:
        """
        Pendulum benchmark: validate oscillatory dynamics.

        Expected behavior:
        - Period: T = 2π * sqrt(L/g) for small angles
        """
        if not MUJOCO_AVAILABLE or not NUMPY_AVAILABLE:
            return BenchmarkResult(
                name="pendulum",
                category="physics",
                passed=False,
                value=0.0,
                target=0.01,
                unit="ratio",
                details={"error": "MuJoCo or NumPy not available"},
            )

        try:
            model = mujoco.MjModel.from_xml_path(model_path)
            data = mujoco.MjData(model)

            # Reset and set initial angle
            mujoco.mj_resetData(model, data)
            initial_angle = 0.1  # radians (small angle)
            data.qpos[0] = initial_angle

            # Simulate and find period
            dt = model.opt.timestep
            duration = 10.0
            n_steps = int(duration / dt)

            positions = []
            times = []

            for i in range(n_steps):
                mujoco.mj_step(model, data)
                positions.append(data.qpos[0])
                times.append((i + 1) * dt)

            positions = np.array(positions)
            times = np.array(times)

            # Find zero crossings to estimate period
            zero_crossings = np.where(np.diff(np.sign(positions)))[0]

            if len(zero_crossings) >= 4:
                # Period is time between every other zero crossing
                periods = []
                for i in range(0, len(zero_crossings) - 2, 2):
                    period = times[zero_crossings[i + 2]] - times[zero_crossings[i]]
                    periods.append(period)

                measured_period = np.mean(periods)

                # Expected period for simple pendulum (assuming unit length)
                expected_period = 2 * np.pi * np.sqrt(1.0 / 9.81)

                period_error = abs(measured_period - expected_period) / expected_period

                passed = period_error < 0.05  # 5% tolerance

                return BenchmarkResult(
                    name="pendulum",
                    category="physics",
                    passed=passed,
                    value=period_error,
                    target=0.05,
                    unit="ratio",
                    details={
                        "measured_period": measured_period,
                        "expected_period": expected_period,
                        "period_error": period_error,
                    },
                )
            else:
                return BenchmarkResult(
                    name="pendulum",
                    category="physics",
                    passed=False,
                    value=0.0,
                    target=0.05,
                    unit="ratio",
                    details={"error": "Not enough oscillations detected"},
                )

        except Exception as e:
            return BenchmarkResult(
                name="pendulum",
                category="physics",
                passed=False,
                value=0.0,
                target=0.05,
                unit="ratio",
                details={"error": str(e)},
            )


# =============================================================================
# Performance Benchmarks
# =============================================================================


class PerformanceBenchmark:
    """Performance and throughput benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run_throughput(self, model_path: str) -> BenchmarkResult:
        """Measure simulation steps per second."""
        if not MUJOCO_AVAILABLE:
            return BenchmarkResult(
                name="throughput",
                category="performance",
                passed=False,
                value=0.0,
                target=100000,
                unit="steps/sec",
                details={"error": "MuJoCo not available"},
            )

        try:
            model = mujoco.MjModel.from_xml_path(model_path)
            data = mujoco.MjData(model)

            # Warmup
            for _ in range(1000):
                mujoco.mj_step(model, data)

            # Benchmark
            n_steps = 10000
            start_time = time.perf_counter()

            for _ in range(n_steps):
                mujoco.mj_step(model, data)

            elapsed = time.perf_counter() - start_time
            steps_per_sec = n_steps / elapsed

            passed = steps_per_sec > 100000  # 100K steps/sec target

            return BenchmarkResult(
                name="throughput",
                category="performance",
                passed=passed,
                value=steps_per_sec,
                target=100000,
                unit="steps/sec",
                details={
                    "n_steps": n_steps,
                    "elapsed_seconds": elapsed,
                    "mujoco_version": MUJOCO_VERSION,
                },
            )

        except Exception as e:
            return BenchmarkResult(
                name="throughput",
                category="performance",
                passed=False,
                value=0.0,
                target=100000,
                unit="steps/sec",
                details={"error": str(e)},
            )


# =============================================================================
# Cross-Framework Benchmarks
# =============================================================================


class CrossFrameworkBenchmark:
    """Cross-framework consistency benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run_trajectory_match(self, model_path_a: str, model_path_b: str, n_steps: int = 1000) -> BenchmarkResult:
        """Compare trajectories between two framework instances."""
        if not MUJOCO_AVAILABLE or not NUMPY_AVAILABLE:
            return BenchmarkResult(
                name="trajectory_match",
                category="cross_framework",
                passed=False,
                value=0.0,
                target=0.001,
                unit="m",
                details={"error": "MuJoCo or NumPy not available"},
            )

        try:
            # Load both models
            model_a = mujoco.MjModel.from_xml_path(model_path_a)
            data_a = mujoco.MjData(model_a)

            model_b = mujoco.MjModel.from_xml_path(model_path_b)
            data_b = mujoco.MjData(model_b)

            # Reset both
            mujoco.mj_resetData(model_a, data_a)
            mujoco.mj_resetData(model_b, data_b)

            # Generate random control sequence
            np.random.seed(self.config.seed)
            nu = min(model_a.nu, model_b.nu)
            controls = np.random.uniform(-0.5, 0.5, (n_steps, nu))

            # Simulate and compare
            errors = []

            for i in range(n_steps):
                # Apply same control
                data_a.ctrl[:nu] = controls[i]
                data_b.ctrl[:nu] = controls[i]

                # Step
                mujoco.mj_step(model_a, data_a)
                mujoco.mj_step(model_b, data_b)

                # Compare positions
                nq = min(model_a.nq, model_b.nq)
                error = np.linalg.norm(data_a.qpos[:nq] - data_b.qpos[:nq])
                errors.append(error)

            max_error = max(errors)
            mean_error = np.mean(errors)

            passed = max_error < 0.001  # 1 mm tolerance

            return BenchmarkResult(
                name="trajectory_match",
                category="cross_framework",
                passed=passed,
                value=max_error,
                target=0.001,
                unit="m",
                details={
                    "max_error": max_error,
                    "mean_error": mean_error,
                    "n_steps": n_steps,
                },
            )

        except Exception as e:
            return BenchmarkResult(
                name="trajectory_match",
                category="cross_framework",
                passed=False,
                value=0.0,
                target=0.001,
                unit="m",
                details={"error": str(e)},
            )


# =============================================================================
# Benchmark Runner
# =============================================================================


class BenchmarkRunner:
    """Main benchmark execution engine."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig(
            scenario="peg_transfer",
            frameworks=["mujoco"],
        )

        self.physics_benchmark = PhysicsBenchmark(self.config)
        self.performance_benchmark = PerformanceBenchmark(self.config)
        self.cross_framework_benchmark = CrossFrameworkBenchmark(self.config)

    def run(
        self, scenario: str, model_path: Optional[str] = None, model_path_b: Optional[str] = None
    ) -> BenchmarkReport:
        """
        Run benchmark suite for a scenario.

        Args:
            scenario: Benchmark scenario name
            model_path: Path to robot model
            model_path_b: Path to second model (for cross-framework tests)

        Returns:
            Complete benchmark report
        """
        start_time = time.perf_counter()

        report = BenchmarkReport(
            scenario=scenario,
            frameworks=self.config.frameworks,
        )

        scenario_info = SCENARIOS.get(scenario)
        if scenario_info is None:
            report.add_result(
                BenchmarkResult(
                    name="scenario_check",
                    category="setup",
                    passed=False,
                    value=0.0,
                    target=1.0,
                    unit="bool",
                    details={"error": f"Unknown scenario: {scenario}"},
                )
            )
            return report

        # Run physics benchmarks
        if scenario in ["free_fall", "pendulum"] and model_path:
            if scenario == "free_fall":
                report.add_result(self.physics_benchmark.run_free_fall(model_path))
            elif scenario == "pendulum":
                report.add_result(self.physics_benchmark.run_pendulum(model_path))

        # Run performance benchmark
        if model_path:
            report.add_result(self.performance_benchmark.run_throughput(model_path))

        # Run cross-framework benchmark
        if model_path and model_path_b:
            report.add_result(self.cross_framework_benchmark.run_trajectory_match(model_path, model_path_b))

        report.duration_seconds = time.perf_counter() - start_time

        return report

    def run_suite(
        self, scenarios: Optional[List[str]] = None, model_path: Optional[str] = None
    ) -> List[BenchmarkReport]:
        """Run multiple scenarios."""
        if scenarios is None:
            scenarios = list(SCENARIOS.keys())

        reports = []
        for scenario in scenarios:
            report = self.run(scenario, model_path)
            reports.append(report)

        return reports


def generate_html_report(report: BenchmarkReport, output_path: str) -> None:
    """Generate HTML benchmark report."""
    passed_count = sum(1 for r in report.results if r.passed)
    total_count = len(report.results)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report: {report.scenario}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary {{ background-color: {"#d4edda" if report.overall_passed else "#f8d7da"};
                   padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .category {{ font-weight: bold; text-transform: uppercase; font-size: 0.8em; }}
    </style>
</head>
<body>
    <h1>Validation Benchmark Report</h1>

    <div class="summary">
        <h2>Summary: <span class="{"passed" if report.overall_passed else "failed"}">
            {"PASSED" if report.overall_passed else "FAILED"}
        </span></h2>
        <p><strong>Scenario:</strong> {report.scenario}</p>
        <p><strong>Frameworks:</strong> {", ".join(report.frameworks)}</p>
        <p><strong>Results:</strong> {passed_count}/{total_count} benchmarks passed</p>
        <p><strong>Duration:</strong> {report.duration_seconds:.2f} seconds</p>
        <p><strong>Timestamp:</strong> {report.timestamp}</p>
    </div>

    <h2>Benchmark Results</h2>
    <table>
        <tr>
            <th>Benchmark</th>
            <th>Category</th>
            <th>Status</th>
            <th>Value</th>
            <th>Target</th>
            <th>Unit</th>
        </tr>
"""

    for result in report.results:
        status_class = "passed" if result.passed else "failed"
        status_text = "PASS" if result.passed else "FAIL"

        html += f"""
        <tr>
            <td>{result.name}</td>
            <td class="category">{result.category}</td>
            <td class="{status_class}">{status_text}</td>
            <td>{result.value:.6g}</td>
            <td>{result.target:.6g}</td>
            <td>{result.unit}</td>
        </tr>
"""

    html += """
    </table>

    <h2>References</h2>
    <ul>
        <li><a href="https://robosuite.ai/">robosuite Benchmarks</a></li>
        <li><a href="https://orbit-surgical.github.io/">ORBIT-Surgical</a></li>
        <li><a href="https://mujoco.readthedocs.io/">MuJoCo Documentation</a></li>
    </ul>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Run Q1 2026 validation benchmark suite.")
    parser.add_argument("--scenario", "-s", help="Benchmark scenario to run")
    parser.add_argument("--model", "-m", help="Path to robot model (MJCF)")
    parser.add_argument("--model-b", help="Path to second model for cross-framework comparison")
    parser.add_argument("--frameworks", "-f", default="mujoco", help="Comma-separated list of frameworks")
    parser.add_argument("--suite", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios")
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument("--report", "-r", help="Generate HTML report at specified path")
    parser.add_argument("--json", "-j", help="Output JSON results at specified path")

    args = parser.parse_args()

    # List scenarios
    if args.list_scenarios:
        print("\nAvailable Benchmark Scenarios:")
        print("=" * 60)
        for name, info in SCENARIOS.items():
            print(f"\n  {name}")
            print(f"    Description: {info['description']}")
            print(f"    Category: {info['category']}")
            print(f"    Difficulty: {info['difficulty']}")
            print(f"    Robot: {info['robot']}")
        print()
        return 0

    # Parse frameworks
    frameworks = [f.strip() for f in args.frameworks.split(",")]

    # Configure
    config = BenchmarkConfig(
        scenario=args.scenario or "peg_transfer",
        frameworks=frameworks,
    )

    runner = BenchmarkRunner(config)

    # Run benchmarks
    if args.suite:
        reports = runner.run_suite(model_path=args.model)
        for report in reports:
            print(f"\n{report.scenario}: {'PASSED' if report.overall_passed else 'FAILED'}")
    else:
        if not args.scenario:
            print("Error: --scenario required (or use --suite)")
            return 1

        report = runner.run(args.scenario, model_path=args.model, model_path_b=args.model_b)

        # Print summary
        print("\n" + "=" * 60)
        print(f"BENCHMARK REPORT: {report.scenario}")
        print("=" * 60)
        print(f"Frameworks: {', '.join(report.frameworks)}")
        print(f"Overall: {'PASSED' if report.overall_passed else 'FAILED'}")
        print(f"Duration: {report.duration_seconds:.2f}s")
        print("-" * 60)

        for result in report.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{result.category}] {status}: {result.name}")
            print(f"         Value: {result.value:.6g} {result.unit} (target: {result.target})")

        print("=" * 60)

        # Save outputs
        if args.report:
            generate_html_report(report, args.report)
            print(f"HTML report: {args.report}")

        if args.json:
            with open(args.json, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"JSON results: {args.json}")

    return 0


if __name__ == "__main__":
    exit(main())
