"""
=============================================================================
EXAMPLE 04: Autonomous Simulation Experiment Orchestrator
=============================================================================

Implements an agentic system that autonomously designs, configures, runs,
and analyzes simulation experiments across multiple physics frameworks
for robotic oncology procedure validation.

CLINICAL CONTEXT:
-----------------
Before deploying robotic surgical policies to physical hardware, extensive
simulation validation is required. This includes:
  - Policy testing across multiple physics engines (sim-to-sim transfer)
  - Systematic parameter sweeps (tissue stiffness, friction, anatomy variants)
  - Failure mode enumeration and robustness testing
  - Statistical analysis of success rates across conditions
  - Regression testing when policies or environments change

An autonomous orchestrator agent reduces the manual effort required for
these validation campaigns from days to hours.

DISTINCTION FROM EXISTING CODE:
-------------------------------
- tools/sim-job-runner/: Low-level simulation job launcher (single runs)
- examples/03_cross_framework_validation.py: One-time cross-framework test
- This example: Agent that designs experiment campaigns, iterates on
  results, and converges on validation conclusions autonomously

ARCHITECTURE:
-------------
  Research     -->  Experiment  -->  Simulation  -->  Analysis  -->  Report
  Question         Designer         Runner           Agent          Generator
                      |                |
                 Parameter         Framework
                 Space Def.        Selection
                                  (Isaac, MuJoCo,
                                   PyBullet, Gazebo)

FRAMEWORK REQUIREMENTS:
-----------------------
Required: (none - pure Python implementation)

Optional:
    - numpy >= 1.24.0 (statistical analysis)
    - scipy >= 1.11.0 (statistical tests)

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import hashlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# SECTION 1: EXPERIMENT DATA MODELS
# =============================================================================


class SimFramework(Enum):
    """Supported simulation frameworks."""

    ISAAC_LAB = "isaac_lab"
    MUJOCO = "mujoco"
    PYBULLET = "pybullet"
    GAZEBO = "gazebo"


class ExperimentStatus(Enum):
    """Status of a simulation experiment."""

    DESIGNED = "designed"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ANALYZING = "analyzing"


class SamplingStrategy(Enum):
    """Parameter space sampling strategies."""

    GRID = "grid"
    RANDOM = "random"
    LATIN_HYPERCUBE = "latin_hypercube"
    SOBOL = "sobol"


@dataclass
class ParameterRange:
    """Definition of a parameter to sweep."""

    name: str
    min_value: float
    max_value: float
    num_samples: int = 5
    distribution: str = "uniform"  # uniform, log_uniform, normal
    units: str = ""

    def sample_values(self, n: Optional[int] = None) -> list[float]:
        """Generate sample values for this parameter."""
        count = n or self.num_samples
        if self.distribution == "uniform":
            step = (self.max_value - self.min_value) / max(1, count - 1)
            return [self.min_value + i * step for i in range(count)]
        elif self.distribution == "log_uniform":
            log_min = math.log10(max(1e-10, self.min_value))
            log_max = math.log10(max(1e-10, self.max_value))
            step = (log_max - log_min) / max(1, count - 1)
            return [10 ** (log_min + i * step) for i in range(count)]
        else:
            # Normal distribution centered between min and max
            center = (self.min_value + self.max_value) / 2
            std = (self.max_value - self.min_value) / 4
            return [
                max(self.min_value, min(self.max_value, center + std * (2 * i / max(1, count - 1) - 1)))
                for i in range(count)
            ]


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""

    config_id: str
    framework: SimFramework
    policy_path: str = ""
    environment: str = ""
    parameters: dict = field(default_factory=dict)
    num_episodes: int = 100
    max_steps_per_episode: int = 1000
    random_seed: int = 42


@dataclass
class SimulationResult:
    """Result from a single simulation run."""

    config_id: str
    framework: SimFramework
    parameters: dict = field(default_factory=dict)
    success_rate: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_episode_length: int = 0
    collision_rate: float = 0.0
    completion_time_seconds: float = 0.0
    metrics: dict = field(default_factory=dict)
    error_message: str = ""
    status: ExperimentStatus = ExperimentStatus.COMPLETED


@dataclass
class ExperimentDesign:
    """Complete experiment design specification."""

    experiment_id: str
    research_question: str
    hypothesis: str
    independent_variables: list = field(default_factory=list)
    dependent_variables: list = field(default_factory=list)
    control_variables: dict = field(default_factory=dict)
    frameworks: list = field(default_factory=list)
    sampling_strategy: SamplingStrategy = SamplingStrategy.GRID
    total_configurations: int = 0
    statistical_test: str = "mann_whitney_u"
    significance_level: float = 0.05


@dataclass
class ExperimentReport:
    """Final report from a completed experiment campaign."""

    experiment_id: str
    research_question: str
    hypothesis: str
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    key_findings: list = field(default_factory=list)
    statistical_results: dict = field(default_factory=dict)
    recommendations: list = field(default_factory=list)
    parameter_sensitivity: dict = field(default_factory=dict)
    framework_comparison: dict = field(default_factory=dict)


# =============================================================================
# SECTION 2: EXPERIMENT DESIGNER
# =============================================================================


class ExperimentDesigner:
    """
    Designs simulation experiments from research questions.

    Translates high-level research questions about robotic oncology
    procedures into structured experiment designs with parameter
    spaces, sampling strategies, and statistical analysis plans.
    """

    # Standard parameter ranges for oncology robotics simulation
    STANDARD_PARAMETERS = {
        "tissue_stiffness_kpa": ParameterRange("tissue_stiffness_kpa", 1.0, 50.0, 5, "log_uniform", "kPa"),
        "tissue_friction": ParameterRange("tissue_friction", 0.1, 0.9, 5, "uniform", ""),
        "needle_diameter_mm": ParameterRange("needle_diameter_mm", 0.5, 3.0, 4, "uniform", "mm"),
        "insertion_speed_mm_s": ParameterRange("insertion_speed_mm_s", 0.5, 5.0, 5, "uniform", "mm/s"),
        "tumor_depth_mm": ParameterRange("tumor_depth_mm", 10.0, 50.0, 5, "uniform", "mm"),
        "tumor_size_mm": ParameterRange("tumor_size_mm", 5.0, 40.0, 4, "uniform", "mm"),
        "anatomy_variation": ParameterRange("anatomy_variation", 0.0, 1.0, 5, "uniform", ""),
        "robot_noise_level": ParameterRange("robot_noise_level", 0.0, 0.05, 3, "uniform", ""),
        "force_threshold_n": ParameterRange("force_threshold_n", 2.0, 10.0, 5, "uniform", "N"),
        "camera_noise_std": ParameterRange("camera_noise_std", 0.0, 0.1, 3, "uniform", ""),
    }

    # Predefined experiment templates
    EXPERIMENT_TEMPLATES = {
        "tissue_robustness": {
            "description": "Test policy robustness to tissue property variations",
            "parameters": ["tissue_stiffness_kpa", "tissue_friction"],
            "dependent_variables": ["success_rate", "collision_rate", "mean_reward"],
            "hypothesis": "Policy maintains >90% success rate across tissue property variations",
        },
        "sim_to_sim_transfer": {
            "description": "Validate policy consistency across simulation frameworks",
            "parameters": ["anatomy_variation"],
            "frameworks": ["isaac_lab", "mujoco", "pybullet"],
            "dependent_variables": ["success_rate", "mean_reward"],
            "hypothesis": "Success rate variance across frameworks is <5%",
        },
        "noise_sensitivity": {
            "description": "Evaluate policy sensitivity to sensor and actuator noise",
            "parameters": ["robot_noise_level", "camera_noise_std"],
            "dependent_variables": ["success_rate", "collision_rate"],
            "hypothesis": "Policy degrades gracefully with increasing noise levels",
        },
        "procedure_parameter_sweep": {
            "description": "Sweep procedure parameters to find optimal ranges",
            "parameters": ["insertion_speed_mm_s", "force_threshold_n"],
            "dependent_variables": ["success_rate", "completion_time_seconds"],
            "hypothesis": "Optimal insertion speed exists that maximizes success rate",
        },
    }

    def design_experiment(
        self,
        research_question: str,
        template: Optional[str] = None,
        custom_parameters: Optional[list[str]] = None,
        frameworks: Optional[list[str]] = None,
    ) -> ExperimentDesign:
        """
        Design an experiment from a research question.

        Args:
            research_question: The research question to investigate
            template: Optional template name from EXPERIMENT_TEMPLATES
            custom_parameters: Override parameter list
            frameworks: Simulation frameworks to use

        Returns:
            Complete experiment design
        """
        experiment_id = f"EXP-{hashlib.md5(research_question.encode()).hexdigest()[:8].upper()}"

        # Use template or infer design
        if template and template in self.EXPERIMENT_TEMPLATES:
            tmpl = self.EXPERIMENT_TEMPLATES[template]
            param_names = custom_parameters or tmpl["parameters"]
            dependent_vars = tmpl["dependent_variables"]
            hypothesis = tmpl["hypothesis"]
            framework_names = frameworks or tmpl.get("frameworks", ["mujoco"])
        else:
            param_names = custom_parameters or ["tissue_stiffness_kpa", "anatomy_variation"]
            dependent_vars = ["success_rate", "mean_reward", "collision_rate"]
            hypothesis = "Policy performance is robust to parameter variations"
            framework_names = frameworks or ["mujoco"]

        # Build parameter ranges
        independent_vars = []
        for name in param_names:
            param = self.STANDARD_PARAMETERS.get(name)
            if param:
                independent_vars.append(param)

        # Calculate total configurations
        total_configs = 1
        for param in independent_vars:
            total_configs *= param.num_samples
        total_configs *= len(framework_names)

        sim_frameworks = []
        for name in framework_names:
            try:
                sim_frameworks.append(SimFramework(name))
            except ValueError:
                logger.warning(f"Unknown framework: {name}")

        design = ExperimentDesign(
            experiment_id=experiment_id,
            research_question=research_question,
            hypothesis=hypothesis,
            independent_variables=independent_vars,
            dependent_variables=dependent_vars,
            frameworks=sim_frameworks,
            sampling_strategy=SamplingStrategy.GRID,
            total_configurations=total_configs,
        )

        logger.info(
            f"Designed experiment {experiment_id}: {total_configs} configurations "
            f"across {len(sim_frameworks)} framework(s)"
        )

        return design


# =============================================================================
# SECTION 3: SIMULATION RUNNER (MOCK)
# =============================================================================


class SimulationRunner:
    """
    Executes simulation configurations and returns results.

    In production, this would interface with actual simulation
    frameworks via subprocess calls, ROS 2 launch files, or
    framework-specific Python APIs.

    For this example, results are simulated with physics-plausible
    distributions to demonstrate the orchestrator logic.
    """

    def __init__(self):
        self._run_count = 0

    def run_simulation(self, config: SimulationConfig) -> SimulationResult:
        """
        Execute a single simulation configuration.

        Args:
            config: Simulation configuration

        Returns:
            Simulation result with metrics
        """
        self._run_count += 1
        start_time = time.time()

        # Simulate physics-plausible results based on parameters
        base_success = 0.92
        params = config.parameters

        # Tissue stiffness affects success (harder tissue = slightly lower success)
        stiffness = params.get("tissue_stiffness_kpa", 10.0)
        stiffness_penalty = max(0, (stiffness - 20.0) / 100.0)

        # Noise affects success
        noise = params.get("robot_noise_level", 0.0)
        noise_penalty = noise * 2.0

        # Anatomy variation affects success
        anatomy_var = params.get("anatomy_variation", 0.0)
        anatomy_penalty = anatomy_var * 0.15

        # Framework-specific bias (slight differences)
        framework_bias = {
            SimFramework.ISAAC_LAB: 0.0,
            SimFramework.MUJOCO: -0.02,
            SimFramework.PYBULLET: -0.04,
            SimFramework.GAZEBO: -0.03,
        }
        fw_bias = framework_bias.get(config.framework, 0.0)

        # Calculate success rate with some stochasticity
        success_rate = max(
            0.0,
            min(
                1.0,
                base_success - stiffness_penalty - noise_penalty - anatomy_penalty + fw_bias + random.gauss(0, 0.02),
            ),
        )

        # Collision rate inversely related to success
        collision_rate = max(0.0, min(0.5, (1.0 - success_rate) * 0.6 + random.gauss(0, 0.01)))

        # Reward correlated with success
        mean_reward = success_rate * 100.0 + random.gauss(0, 5.0)

        elapsed = time.time() - start_time

        return SimulationResult(
            config_id=config.config_id,
            framework=config.framework,
            parameters=config.parameters,
            success_rate=round(success_rate, 4),
            mean_reward=round(mean_reward, 2),
            std_reward=round(abs(random.gauss(8.0, 2.0)), 2),
            mean_episode_length=int(500 + success_rate * 400 + random.gauss(0, 50)),
            collision_rate=round(collision_rate, 4),
            completion_time_seconds=round(elapsed, 3),
            status=ExperimentStatus.COMPLETED,
        )


# =============================================================================
# SECTION 4: ANALYSIS ENGINE
# =============================================================================


class AnalysisEngine:
    """
    Analyzes simulation results to draw statistical conclusions.

    Performs:
    - Summary statistics across parameter configurations
    - Parameter sensitivity analysis (which parameters matter most)
    - Cross-framework consistency checks
    - Hypothesis testing against stated hypotheses
    """

    def analyze_results(
        self,
        design: ExperimentDesign,
        results: list[SimulationResult],
    ) -> ExperimentReport:
        """
        Analyze simulation results and generate report.

        Args:
            design: The experiment design
            results: List of simulation results

        Returns:
            Complete experiment report with findings
        """
        successful = [r for r in results if r.status == ExperimentStatus.COMPLETED]
        failed = [r for r in results if r.status == ExperimentStatus.FAILED]

        report = ExperimentReport(
            experiment_id=design.experiment_id,
            research_question=design.research_question,
            hypothesis=design.hypothesis,
            total_runs=len(results),
            successful_runs=len(successful),
            failed_runs=len(failed),
        )

        if not successful:
            report.key_findings = ["All simulation runs failed. Check configuration."]
            return report

        # Summary statistics
        success_rates = [r.success_rate for r in successful]
        mean_success = sum(success_rates) / len(success_rates)
        std_success = (sum((x - mean_success) ** 2 for x in success_rates) / len(success_rates)) ** 0.5

        report.statistical_results["overall"] = {
            "mean_success_rate": round(mean_success, 4),
            "std_success_rate": round(std_success, 4),
            "min_success_rate": round(min(success_rates), 4),
            "max_success_rate": round(max(success_rates), 4),
            "mean_collision_rate": round(sum(r.collision_rate for r in successful) / len(successful), 4),
        }

        # Parameter sensitivity analysis
        report.parameter_sensitivity = self._analyze_sensitivity(design, successful)

        # Framework comparison
        report.framework_comparison = self._compare_frameworks(successful)

        # Hypothesis evaluation
        hypothesis_result = self._evaluate_hypothesis(design, report)
        report.key_findings.append(hypothesis_result)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _analyze_sensitivity(
        self,
        design: ExperimentDesign,
        results: list[SimulationResult],
    ) -> dict:
        """Analyze which parameters most affect performance."""
        sensitivity = {}

        for param in design.independent_variables:
            param_name = param.name
            # Group results by parameter value
            groups: dict[float, list[float]] = {}
            for result in results:
                param_val = result.parameters.get(param_name)
                if param_val is not None:
                    rounded_val = round(param_val, 4)
                    if rounded_val not in groups:
                        groups[rounded_val] = []
                    groups[rounded_val].append(result.success_rate)

            if len(groups) < 2:
                continue

            # Calculate variance across groups
            group_means = {k: sum(v) / len(v) for k, v in groups.items()}
            overall_mean = sum(group_means.values()) / len(group_means)
            between_group_var = sum((m - overall_mean) ** 2 for m in group_means.values()) / len(group_means)

            sensitivity[param_name] = {
                "between_group_variance": round(between_group_var, 6),
                "group_means": {str(k): round(v, 4) for k, v in sorted(group_means.items())},
                "impact": "high" if between_group_var > 0.01 else "moderate" if between_group_var > 0.001 else "low",
            }

        return sensitivity

    def _compare_frameworks(self, results: list[SimulationResult]) -> dict:
        """Compare performance across simulation frameworks."""
        framework_results: dict[str, list[float]] = {}

        for result in results:
            fw_name = result.framework.value
            if fw_name not in framework_results:
                framework_results[fw_name] = []
            framework_results[fw_name].append(result.success_rate)

        comparison = {}
        for fw_name, rates in framework_results.items():
            mean_rate = sum(rates) / len(rates)
            std_rate = (sum((x - mean_rate) ** 2 for x in rates) / len(rates)) ** 0.5
            comparison[fw_name] = {
                "mean_success_rate": round(mean_rate, 4),
                "std_success_rate": round(std_rate, 4),
                "num_runs": len(rates),
            }

        # Cross-framework consistency
        if len(comparison) > 1:
            means = [v["mean_success_rate"] for v in comparison.values()]
            max_diff = max(means) - min(means)
            comparison["cross_framework_consistency"] = {
                "max_difference": round(max_diff, 4),
                "consistent": max_diff < 0.05,
            }

        return comparison

    def _evaluate_hypothesis(self, design: ExperimentDesign, report: ExperimentReport) -> str:
        """Evaluate stated hypothesis against results."""
        overall = report.statistical_results.get("overall", {})
        mean_success = overall.get("mean_success_rate", 0.0)
        std_success = overall.get("std_success_rate", 0.0)

        hypothesis = design.hypothesis.lower()

        if "90%" in hypothesis or "0.9" in hypothesis:
            if mean_success >= 0.90:
                return (
                    f"HYPOTHESIS SUPPORTED: Mean success rate ({mean_success:.1%}) "
                    f"meets the 90% threshold (std: {std_success:.1%})"
                )
            else:
                return (
                    f"HYPOTHESIS NOT SUPPORTED: Mean success rate ({mean_success:.1%}) "
                    f"below 90% threshold (std: {std_success:.1%})"
                )

        if "variance" in hypothesis or "consistent" in hypothesis:
            fw_comp = report.framework_comparison.get("cross_framework_consistency", {})
            if fw_comp.get("consistent", False):
                return (
                    f"HYPOTHESIS SUPPORTED: Cross-framework variance within acceptable range "
                    f"(max diff: {fw_comp.get('max_difference', 0):.1%})"
                )
            else:
                return (
                    f"HYPOTHESIS NOT SUPPORTED: Cross-framework variance exceeds threshold "
                    f"(max diff: {fw_comp.get('max_difference', 0):.1%})"
                )

        return f"INCONCLUSIVE: Mean success rate {mean_success:.1%} (std: {std_success:.1%})"

    def _generate_recommendations(self, report: ExperimentReport) -> list[str]:
        """Generate actionable recommendations from analysis."""
        recs = []

        overall = report.statistical_results.get("overall", {})
        mean_success = overall.get("mean_success_rate", 0.0)
        collision_rate = overall.get("mean_collision_rate", 0.0)

        if mean_success < 0.85:
            recs.append(
                "Overall success rate below 85%. Consider retraining the policy "
                "with domain randomization covering the tested parameter range."
            )

        if collision_rate > 0.05:
            recs.append(
                f"Collision rate ({collision_rate:.1%}) exceeds 5% threshold. "
                "Review safety constraints and workspace boundaries in training."
            )

        # Check for high-sensitivity parameters
        for param, sensitivity in report.parameter_sensitivity.items():
            if sensitivity.get("impact") == "high":
                recs.append(
                    f"Parameter '{param}' has high impact on performance. "
                    "Ensure training distribution covers the full range tested."
                )

        # Framework-specific recommendations
        fw_comp = report.framework_comparison
        consistency = fw_comp.get("cross_framework_consistency", {})
        if not consistency.get("consistent", True):
            recs.append(
                f"Cross-framework gap ({consistency.get('max_difference', 0):.1%}). "
                "Investigate physics parameter alignment between frameworks."
            )

        if not recs:
            recs.append(
                "Policy meets all validation criteria. Ready for next stage of sim-to-real transfer validation."
            )

        return recs


# =============================================================================
# SECTION 5: ORCHESTRATOR AGENT
# =============================================================================


class SimulationOrchestrator:
    """
    Autonomous agent that orchestrates simulation experiment campaigns.

    The orchestrator manages the complete lifecycle:
    1. Design experiments from research questions
    2. Generate simulation configurations
    3. Execute simulations across frameworks
    4. Analyze results and draw conclusions
    5. Optionally iterate with refined experiments

    AUTONOMY LEVELS:
    ----------------
    - SEMI: Requires approval before running experiments
    - FULL: Runs experiments autonomously, reports findings
    """

    def __init__(self, autonomy: str = "full"):
        self.autonomy = autonomy
        self.designer = ExperimentDesigner()
        self.runner = SimulationRunner()
        self.analyzer = AnalysisEngine()
        self._experiment_history: list[ExperimentReport] = []
        self._reasoning_log: list[dict] = []

    def _log_reasoning(self, phase: str, reasoning: str) -> None:
        """Log agent reasoning step."""
        self._reasoning_log.append(
            {
                "phase": phase,
                "reasoning": reasoning,
                "timestamp": time.time(),
            }
        )
        logger.info(f"[{phase.upper()}] {reasoning}")

    def run_experiment_campaign(
        self,
        research_question: str,
        template: Optional[str] = None,
        frameworks: Optional[list[str]] = None,
        max_iterations: int = 1,
    ) -> ExperimentReport:
        """
        Run a complete experiment campaign.

        Args:
            research_question: The question to investigate
            template: Optional experiment template
            frameworks: Frameworks to test on
            max_iterations: Max refinement iterations

        Returns:
            Final experiment report
        """
        self._log_reasoning("planning", f"Investigating: {research_question}")

        report = None
        for iteration in range(max_iterations):
            self._log_reasoning("design", f"Iteration {iteration + 1}/{max_iterations}: Designing experiment")

            # 1. Design experiment
            design = self.designer.design_experiment(
                research_question=research_question,
                template=template,
                frameworks=frameworks,
            )

            self._log_reasoning(
                "design",
                f"Experiment {design.experiment_id}: "
                f"{design.total_configurations} configurations, "
                f"{len(design.frameworks)} framework(s)",
            )

            # 2. Generate configurations
            configs = self._generate_configurations(design)
            self._log_reasoning("execution", f"Generated {len(configs)} simulation configurations")

            # 3. Execute simulations
            results = self._execute_simulations(configs)
            completed = sum(1 for r in results if r.status == ExperimentStatus.COMPLETED)
            self._log_reasoning("execution", f"Completed {completed}/{len(results)} simulations")

            # 4. Analyze results
            report = self.analyzer.analyze_results(design, results)
            self._log_reasoning(
                "analysis",
                f"Analysis complete. Key finding: {report.key_findings[0] if report.key_findings else 'None'}",
            )

            # 5. Check if additional iterations needed
            overall = report.statistical_results.get("overall", {})
            std_success = overall.get("std_success_rate", 0.0)

            if std_success < 0.05 or iteration == max_iterations - 1:
                self._log_reasoning("conclusion", "Results converged. Generating final report.")
                break
            else:
                self._log_reasoning(
                    "refinement", f"High variance (std={std_success:.3f}). Refining experiment with more samples."
                )

        if report:
            self._experiment_history.append(report)

        return report

    def _generate_configurations(self, design: ExperimentDesign) -> list[SimulationConfig]:
        """Generate simulation configurations from experiment design."""
        configs = []
        config_counter = 0

        # Generate parameter combinations
        param_lists = []
        param_names = []
        for param in design.independent_variables:
            param_names.append(param.name)
            param_lists.append(param.sample_values())

        # Grid sampling
        combinations = self._grid_product(param_lists)

        for framework in design.frameworks:
            for combo in combinations:
                config_counter += 1
                params = dict(zip(param_names, combo))

                config = SimulationConfig(
                    config_id=f"{design.experiment_id}-{config_counter:04d}",
                    framework=framework,
                    policy_path=f"policies/{design.experiment_id}/policy.onnx",
                    environment="OncologySurgery-v1",
                    parameters=params,
                    num_episodes=100,
                    random_seed=42 + config_counter,
                )
                configs.append(config)

        return configs

    def _grid_product(self, lists: list[list]) -> list[list]:
        """Generate Cartesian product of parameter lists."""
        if not lists:
            return [[]]
        result = [[]]
        for lst in lists:
            result = [existing + [val] for existing in result for val in lst]
        return result

    def _execute_simulations(self, configs: list[SimulationConfig]) -> list[SimulationResult]:
        """Execute all simulation configurations."""
        results = []
        for i, config in enumerate(configs):
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Running simulation {i + 1}/{len(configs)}")
            result = self.runner.run_simulation(config)
            results.append(result)
        return results

    def get_experiment_history(self) -> list[dict]:
        """Get history of all experiments run."""
        history = []
        for report in self._experiment_history:
            history.append(
                {
                    "experiment_id": report.experiment_id,
                    "research_question": report.research_question,
                    "total_runs": report.total_runs,
                    "mean_success_rate": report.statistical_results.get("overall", {}).get("mean_success_rate", 0),
                    "key_findings": report.key_findings,
                }
            )
        return history

    def get_reasoning_log(self) -> list[dict]:
        """Get complete reasoning log."""
        return self._reasoning_log


# =============================================================================
# SECTION 6: REPORT FORMATTER
# =============================================================================


def format_experiment_report(report: ExperimentReport) -> str:
    """Format experiment report as readable text."""
    lines = [
        "=" * 70,
        "SIMULATION EXPERIMENT REPORT",
        "=" * 70,
        f"Experiment ID: {report.experiment_id}",
        f"Research Question: {report.research_question}",
        f"Hypothesis: {report.hypothesis}",
        "",
        f"Total Runs: {report.total_runs} (Successful: {report.successful_runs}, Failed: {report.failed_runs})",
        "",
        "OVERALL STATISTICS:",
        "-" * 40,
    ]

    overall = report.statistical_results.get("overall", {})
    for key, value in overall.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")

    if report.parameter_sensitivity:
        lines.append("")
        lines.append("PARAMETER SENSITIVITY:")
        lines.append("-" * 40)
        for param, info in report.parameter_sensitivity.items():
            lines.append(f"  {param}: impact={info['impact']}, variance={info['between_group_variance']:.6f}")

    if report.framework_comparison:
        lines.append("")
        lines.append("FRAMEWORK COMPARISON:")
        lines.append("-" * 40)
        for fw, info in report.framework_comparison.items():
            if fw == "cross_framework_consistency":
                lines.append(
                    f"  Consistency: {'PASS' if info['consistent'] else 'FAIL'} "
                    f"(max diff: {info['max_difference']:.4f})"
                )
            else:
                lines.append(
                    f"  {fw}: success={info['mean_success_rate']:.4f} "
                    f"(std={info['std_success_rate']:.4f}, n={info['num_runs']})"
                )

    lines.append("")
    lines.append("KEY FINDINGS:")
    lines.append("-" * 40)
    for finding in report.key_findings:
        lines.append(f"  * {finding}")

    lines.append("")
    lines.append("RECOMMENDATIONS:")
    lines.append("-" * 40)
    for rec in report.recommendations:
        lines.append(f"  -> {rec}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# SECTION 7: DEMO
# =============================================================================


def run_orchestrator_demo():
    """
    Demonstrate the autonomous simulation orchestrator.

    Runs two experiment campaigns:
    1. Tissue robustness testing
    2. Cross-framework validation
    """
    logger.info("=" * 60)
    logger.info("AUTONOMOUS SIMULATION ORCHESTRATOR DEMO")
    logger.info("=" * 60)

    orchestrator = SimulationOrchestrator(autonomy="full")

    # Campaign 1: Tissue robustness
    print("\n" + "=" * 60)
    print("CAMPAIGN 1: Tissue Robustness Testing")
    print("=" * 60)

    report1 = orchestrator.run_experiment_campaign(
        research_question="Is the needle insertion policy robust to variations in tissue mechanical properties?",
        template="tissue_robustness",
        frameworks=["mujoco"],
    )
    print(format_experiment_report(report1))

    # Campaign 2: Cross-framework validation
    print("\n" + "=" * 60)
    print("CAMPAIGN 2: Cross-Framework Validation")
    print("=" * 60)

    report2 = orchestrator.run_experiment_campaign(
        research_question="Does the surgical policy perform consistently across simulation frameworks?",
        template="sim_to_sim_transfer",
        frameworks=["isaac_lab", "mujoco", "pybullet"],
    )
    print(format_experiment_report(report2))

    # Print reasoning log
    print("\n" + "=" * 60)
    print("ORCHESTRATOR REASONING LOG")
    print("=" * 60)
    for entry in orchestrator.get_reasoning_log():
        print(f"  [{entry['phase'].upper():12s}] {entry['reasoning']}")

    return {
        "campaigns_run": 2,
        "total_simulations": report1.total_runs + report2.total_runs,
        "reasoning_steps": len(orchestrator.get_reasoning_log()),
    }


if __name__ == "__main__":
    result = run_orchestrator_demo()
    print(f"\nDemo result: {result}")
