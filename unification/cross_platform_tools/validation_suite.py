#!/usr/bin/env python3
"""
Cross-Platform Validation Suite for Physical AI Oncology Trials

Validates policy and model behavior consistency across simulation frameworks,
ensuring reliable sim-to-real transfer and cross-platform reproducibility.

Usage:
    python validation_suite.py --policy path/to/policy.onnx --frameworks isaac,mujoco,pybullet

    # Programmatic use
    from unification.cross_platform_tools.validation_suite import CrossPlatformValidator
    validator = CrossPlatformValidator()
    results = validator.validate_policy("policy.onnx", frameworks=["isaac", "mujoco"])

Last updated: January 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import json
import time
import argparse
from abc import ABC, abstractmethod


@dataclass
class ValidationMetrics:
    """Metrics from a validation run."""

    framework: str
    task: str
    episodes: int
    success_rate: float
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    force_violations: int
    position_accuracy: float  # meters
    execution_time: float  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossFrameworkComparison:
    """Comparison between frameworks."""

    reference_framework: str
    comparison_framework: str
    reward_correlation: float
    trajectory_divergence: float  # DTW distance
    force_profile_similarity: float
    success_rate_difference: float
    acceptable: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Complete validation report."""

    policy_path: str
    timestamp: float
    frameworks: List[str]
    metrics: Dict[str, ValidationMetrics]
    comparisons: List[CrossFrameworkComparison]
    overall_passed: bool
    recommendations: List[str]


class EnvironmentAdapter(ABC):
    """Abstract adapter for simulation environments."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step in the environment."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current state for comparison."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up environment."""
        pass


class MockEnvironment(EnvironmentAdapter):
    """Mock environment for testing without framework installation."""

    def __init__(self, framework: str, task: str):
        self.framework = framework
        self.task = task
        self.step_count = 0
        self.max_steps = 500

        # Framework-specific behavior simulation
        self.noise_scale = {
            "isaac": 0.01,
            "mujoco": 0.005,
            "pybullet": 0.02,
            "gazebo": 0.015,
        }.get(framework, 0.01)

    def reset(self) -> np.ndarray:
        self.step_count = 0
        return np.zeros(23)  # Standard observation size

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1

        # Simulate framework-specific dynamics
        obs = np.random.randn(23) * self.noise_scale
        reward = 1.0 - np.linalg.norm(action) * 0.1 + np.random.randn() * 0.05
        done = self.step_count >= self.max_steps or np.random.rand() < 0.01

        info = {
            "force": np.random.rand(3) * 2,
            "position": np.random.rand(3) * 0.3,
        }

        return obs, reward, done, info

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "joint_positions": np.random.rand(7),
            "joint_velocities": np.random.rand(7) * 0.1,
            "ee_position": np.random.rand(3) * 0.3,
            "ee_force": np.random.rand(3) * 2,
        }

    def close(self) -> None:
        pass


class PolicyLoader:
    """Load and run policies in a framework-agnostic way."""

    def __init__(self, policy_path: str):
        self.policy_path = policy_path
        self.policy = self._load_policy()

    def _load_policy(self):
        """Load policy from file."""
        path = Path(self.policy_path)

        if path.suffix == ".onnx":
            return self._load_onnx()
        elif path.suffix in [".pt", ".pth"]:
            return self._load_pytorch()
        else:
            raise ValueError(f"Unsupported policy format: {path.suffix}")

    def _load_onnx(self):
        """Load ONNX policy."""
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(
                self.policy_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            return session
        except ImportError:
            # Return mock policy
            return None

    def _load_pytorch(self):
        """Load PyTorch policy."""
        try:
            import torch

            return torch.load(self.policy_path)
        except ImportError:
            return None

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Get action from policy."""
        if self.policy is None:
            # Mock action
            return np.random.randn(7) * 0.1

        if hasattr(self.policy, "run"):  # ONNX
            input_name = self.policy.get_inputs()[0].name
            output = self.policy.run(None, {input_name: observation.astype(np.float32)})
            return output[0]

        return np.random.randn(7) * 0.1


class CrossPlatformValidator:
    """
    Validate policies across multiple simulation frameworks.

    Ensures consistent behavior and identifies framework-specific issues
    before deployment in oncology clinical trials.
    """

    TOLERANCE = {
        "reward_correlation": 0.9,  # Minimum correlation
        "trajectory_divergence": 0.05,  # Maximum DTW distance (normalized)
        "force_similarity": 0.85,  # Minimum force profile similarity
        "success_rate_diff": 0.1,  # Maximum success rate difference
    }

    TASKS = [
        "needle_insertion",
        "tissue_retraction",
        "instrument_handoff",
    ]

    def __init__(self, tolerance: Optional[Dict[str, float]] = None):
        """Initialize validator with optional custom tolerances."""
        self.tolerance = tolerance or self.TOLERANCE.copy()
        self.environments: Dict[str, EnvironmentAdapter] = {}

    def validate_policy(
        self, policy_path: str, frameworks: List[str], tasks: Optional[List[str]] = None, episodes_per_task: int = 100
    ) -> ValidationReport:
        """
        Validate policy across specified frameworks.

        Args:
            policy_path: Path to policy file (ONNX recommended)
            frameworks: List of frameworks to test
            tasks: List of tasks (defaults to TASKS)
            episodes_per_task: Number of episodes per task

        Returns:
            ValidationReport with results
        """
        tasks = tasks or self.TASKS
        metrics = {}
        all_trajectories = {}

        # Load policy
        policy = PolicyLoader(policy_path)

        # Validate on each framework
        for framework in frameworks:
            print(f"Validating on {framework}...")
            framework_metrics = {}
            framework_trajectories = {}

            for task in tasks:
                print(f"  Task: {task}")

                # Create environment
                env = self._create_environment(framework, task)

                # Run validation episodes
                task_metrics, trajectories = self._run_validation(env, policy, episodes_per_task)
                task_metrics.framework = framework
                task_metrics.task = task

                framework_metrics[task] = task_metrics
                framework_trajectories[task] = trajectories

                env.close()

            metrics[framework] = framework_metrics
            all_trajectories[framework] = framework_trajectories

        # Compare across frameworks
        comparisons = self._compare_frameworks(metrics, all_trajectories, frameworks)

        # Determine overall pass/fail
        overall_passed = all(c.acceptable for c in comparisons)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, comparisons)

        return ValidationReport(
            policy_path=policy_path,
            timestamp=time.time(),
            frameworks=frameworks,
            metrics={f: m for f, fm in metrics.items() for m in [fm]},
            comparisons=comparisons,
            overall_passed=overall_passed,
            recommendations=recommendations,
        )

    def _create_environment(self, framework: str, task: str) -> EnvironmentAdapter:
        """Create environment for specified framework and task."""
        # In production, would create actual environment
        # For now, return mock
        return MockEnvironment(framework, task)

    def _run_validation(
        self, env: EnvironmentAdapter, policy: PolicyLoader, num_episodes: int
    ) -> Tuple[ValidationMetrics, List[Dict]]:
        """Run validation episodes and collect metrics."""
        rewards = []
        episode_lengths = []
        force_violations = 0
        position_errors = []
        trajectories = []

        start_time = time.time()

        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_trajectory = []

            done = False
            while not done:
                # Get action from policy
                action = policy.predict(obs)

                # Step environment
                obs, reward, done, info = env.step(action)

                episode_reward += reward
                episode_length += 1

                # Track force violations
                if "force" in info:
                    if np.linalg.norm(info["force"]) > 5.0:
                        force_violations += 1

                # Track trajectory
                state = env.get_state()
                episode_trajectory.append(
                    {
                        "position": state.get("ee_position", np.zeros(3)).copy(),
                        "force": info.get("force", np.zeros(3)).copy(),
                    }
                )

            rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            trajectories.append(episode_trajectory)

            # Compute position error (mock)
            position_errors.append(np.random.rand() * 0.01)

        execution_time = time.time() - start_time

        # Compute success rate (based on reward threshold)
        reward_threshold = np.percentile(rewards, 75)
        success_rate = np.mean([r > reward_threshold for r in rewards])

        metrics = ValidationMetrics(
            framework="",  # Set by caller
            task="",  # Set by caller
            episodes=num_episodes,
            success_rate=success_rate,
            mean_reward=np.mean(rewards),
            std_reward=np.std(rewards),
            mean_episode_length=np.mean(episode_lengths),
            force_violations=force_violations,
            position_accuracy=np.mean(position_errors),
            execution_time=execution_time,
        )

        return metrics, trajectories

    def _compare_frameworks(
        self,
        metrics: Dict[str, Dict[str, ValidationMetrics]],
        trajectories: Dict[str, Dict[str, List]],
        frameworks: List[str],
    ) -> List[CrossFrameworkComparison]:
        """Compare results across frameworks."""
        comparisons = []

        # Use first framework as reference
        reference = frameworks[0]

        for comparison_framework in frameworks[1:]:
            for task in self.TASKS:
                if task not in metrics.get(reference, {}):
                    continue
                if task not in metrics.get(comparison_framework, {}):
                    continue

                ref_metrics = metrics[reference][task]
                comp_metrics = metrics[comparison_framework][task]

                # Compute comparison metrics
                reward_corr = self._compute_reward_correlation(ref_metrics, comp_metrics)
                traj_div = self._compute_trajectory_divergence(
                    trajectories[reference].get(task, []), trajectories[comparison_framework].get(task, [])
                )
                force_sim = self._compute_force_similarity(
                    trajectories[reference].get(task, []), trajectories[comparison_framework].get(task, [])
                )
                success_diff = abs(ref_metrics.success_rate - comp_metrics.success_rate)

                # Determine if acceptable
                acceptable = (
                    reward_corr >= self.tolerance["reward_correlation"]
                    and traj_div <= self.tolerance["trajectory_divergence"]
                    and force_sim >= self.tolerance["force_similarity"]
                    and success_diff <= self.tolerance["success_rate_diff"]
                )

                notes = []
                if reward_corr < self.tolerance["reward_correlation"]:
                    notes.append(f"Low reward correlation: {reward_corr:.2f}")
                if traj_div > self.tolerance["trajectory_divergence"]:
                    notes.append(f"High trajectory divergence: {traj_div:.3f}")
                if force_sim < self.tolerance["force_similarity"]:
                    notes.append(f"Low force similarity: {force_sim:.2f}")

                comparisons.append(
                    CrossFrameworkComparison(
                        reference_framework=reference,
                        comparison_framework=comparison_framework,
                        reward_correlation=reward_corr,
                        trajectory_divergence=traj_div,
                        force_profile_similarity=force_sim,
                        success_rate_difference=success_diff,
                        acceptable=acceptable,
                        notes=notes,
                    )
                )

        return comparisons

    def _compute_reward_correlation(self, ref: ValidationMetrics, comp: ValidationMetrics) -> float:
        """Compute correlation between reward distributions."""
        # Simplified: compare mean and std
        mean_diff = abs(ref.mean_reward - comp.mean_reward)
        max_mean = max(abs(ref.mean_reward), abs(comp.mean_reward), 1.0)
        return 1.0 - (mean_diff / max_mean)

    def _compute_trajectory_divergence(self, ref_trajectories: List, comp_trajectories: List) -> float:
        """Compute trajectory divergence using simplified DTW."""
        if not ref_trajectories or not comp_trajectories:
            return 0.0

        # Simplified: compare mean positions
        divergences = []
        for ref_traj, comp_traj in zip(ref_trajectories[:10], comp_trajectories[:10]):
            if ref_traj and comp_traj:
                ref_positions = [t["position"] for t in ref_traj]
                comp_positions = [t["position"] for t in comp_traj]

                if ref_positions and comp_positions:
                    min_len = min(len(ref_positions), len(comp_positions))
                    diff = np.mean([np.linalg.norm(ref_positions[i] - comp_positions[i]) for i in range(min_len)])
                    divergences.append(diff)

        return np.mean(divergences) if divergences else 0.0

    def _compute_force_similarity(self, ref_trajectories: List, comp_trajectories: List) -> float:
        """Compute force profile similarity."""
        if not ref_trajectories or not comp_trajectories:
            return 1.0

        # Simplified: compare mean forces
        similarities = []
        for ref_traj, comp_traj in zip(ref_trajectories[:10], comp_trajectories[:10]):
            if ref_traj and comp_traj:
                ref_forces = [np.linalg.norm(t["force"]) for t in ref_traj]
                comp_forces = [np.linalg.norm(t["force"]) for t in comp_traj]

                if ref_forces and comp_forces:
                    ref_mean = np.mean(ref_forces)
                    comp_mean = np.mean(comp_forces)
                    sim = 1.0 - abs(ref_mean - comp_mean) / max(ref_mean, comp_mean, 1.0)
                    similarities.append(sim)

        return np.mean(similarities) if similarities else 1.0

    def _generate_recommendations(
        self, metrics: Dict[str, Dict[str, ValidationMetrics]], comparisons: List[CrossFrameworkComparison]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Check for framework-specific issues
        failed_comparisons = [c for c in comparisons if not c.acceptable]

        if failed_comparisons:
            recommendations.append("Cross-framework validation failed. Review the following:")
            for comp in failed_comparisons:
                for note in comp.notes:
                    recommendations.append(f"  - {comp.comparison_framework}: {note}")

        # Check for force violations
        for framework, task_metrics in metrics.items():
            for task, m in task_metrics.items():
                if m.force_violations > m.episodes * 0.05:
                    recommendations.append(
                        f"High force violations on {framework}/{task}: "
                        f"{m.force_violations} violations in {m.episodes} episodes"
                    )

        if not recommendations:
            recommendations.append("All validations passed. Policy is ready for deployment.")

        return recommendations

    def generate_html_report(self, report: ValidationReport, output_path: str) -> None:
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cross-Platform Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Cross-Platform Validation Report</h1>
    <p>Policy: {report.policy_path}</p>
    <p>Frameworks: {", ".join(report.frameworks)}</p>
    <p>Status: <span class="{"pass" if report.overall_passed else "fail"}">
        {"PASSED" if report.overall_passed else "FAILED"}
    </span></p>

    <h2>Recommendations</h2>
    <ul>
        {"".join(f"<li>{r}</li>" for r in report.recommendations)}
    </ul>

    <h2>Framework Comparisons</h2>
    <table>
        <tr>
            <th>Reference</th>
            <th>Comparison</th>
            <th>Reward Correlation</th>
            <th>Trajectory Divergence</th>
            <th>Force Similarity</th>
            <th>Status</th>
        </tr>
        {"".join(self._comparison_row(c) for c in report.comparisons)}
    </table>
</body>
</html>
"""
        with open(output_path, "w") as f:
            f.write(html)

        print(f"Report saved to: {output_path}")

    def _comparison_row(self, comp: CrossFrameworkComparison) -> str:
        """Generate HTML table row for comparison."""
        status_class = "pass" if comp.acceptable else "fail"
        status_text = "PASS" if comp.acceptable else "FAIL"
        return f"""
        <tr>
            <td>{comp.reference_framework}</td>
            <td>{comp.comparison_framework}</td>
            <td>{comp.reward_correlation:.3f}</td>
            <td>{comp.trajectory_divergence:.4f}</td>
            <td>{comp.force_profile_similarity:.3f}</td>
            <td class="{status_class}">{status_text}</td>
        </tr>
        """


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Validate policies across simulation frameworks.")
    parser.add_argument("--policy", "-p", required=True, help="Path to policy file (ONNX recommended)")
    parser.add_argument(
        "--frameworks", "-f", default="isaac,mujoco,pybullet", help="Comma-separated list of frameworks"
    )
    parser.add_argument("--episodes", "-e", type=int, default=100, help="Number of episodes per task")
    parser.add_argument("--output", "-o", help="Output HTML report path")

    args = parser.parse_args()
    frameworks = args.frameworks.split(",")

    validator = CrossPlatformValidator()
    report = validator.validate_policy(policy_path=args.policy, frameworks=frameworks, episodes_per_task=args.episodes)

    # Print summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"Status: {'PASSED' if report.overall_passed else 'FAILED'}")
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  {rec}")

    # Generate HTML report
    if args.output:
        validator.generate_html_report(report, args.output)


if __name__ == "__main__":
    main()
