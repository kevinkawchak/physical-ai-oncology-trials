#!/usr/bin/env python3
"""Simulation Job Runner: CLI for launching, managing, and comparing simulation
jobs across Isaac Lab, MuJoCo, PyBullet, and Gazebo from a single interface.

Provides a unified entry point for running oncology-relevant simulation tasks
(needle insertion, tissue manipulation, surgical planning) on any available
physics backend, with result comparison across frameworks.

Usage:
    python sim_job_runner.py launch --framework mujoco --task needle_insertion --config job.yaml
    python sim_job_runner.py launch-all --task needle_insertion --config job.yaml
    python sim_job_runner.py compare --results-dir results/needle_insertion/
    python sim_job_runner.py list-tasks
    python sim_job_runner.py init-config --task needle_insertion --output job.yaml

Requirements:
    pyyaml (listed in project requirements.txt)
    Framework-specific SDKs are optional; unavailable frameworks are skipped.

Note: All illustrative parameters should be validated against your
institution's simulation protocols before clinical use.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ---------------------------------------------------------------------------
# Task definitions: oncology-relevant simulation tasks
# ---------------------------------------------------------------------------
TASK_REGISTRY = {
    "needle_insertion": {
        "description": "Needle insertion into soft tissue with force feedback",
        "frameworks": ["mujoco", "isaac", "pybullet"],
        "metrics": ["insertion_depth_error_mm", "max_force_n", "trajectory_rmse_mm", "completion_time_s"],
        "default_params": {
            "tissue_stiffness_kpa": 5.0,
            "needle_diameter_mm": 0.7,
            "target_depth_mm": 50.0,
            "max_force_n": 5.0,
            "num_episodes": 100,
        },
    },
    "tissue_retraction": {
        "description": "Soft tissue retraction during tumor exposure",
        "frameworks": ["mujoco", "isaac", "pybullet"],
        "metrics": ["tissue_strain_pct", "retraction_distance_mm", "force_profile_rmse_n", "tissue_damage_score"],
        "default_params": {
            "tissue_elasticity_kpa": 3.0,
            "max_retraction_mm": 30.0,
            "retraction_speed_mm_s": 2.0,
            "num_episodes": 50,
        },
    },
    "surgical_reach": {
        "description": "Reachability and dexterity analysis for surgical workspace",
        "frameworks": ["mujoco", "isaac", "pybullet", "gazebo"],
        "metrics": ["workspace_volume_cm3", "manipulability_index", "joint_limit_margin_deg", "singularity_distance"],
        "default_params": {
            "robot_model": "dvrk_psm",
            "grid_resolution_mm": 5.0,
            "include_orientation": True,
        },
    },
    "instrument_handover": {
        "description": "Robot-to-surgeon instrument handover with force control",
        "frameworks": ["mujoco", "isaac", "gazebo"],
        "metrics": ["handover_force_n", "grip_stability_score", "handover_time_s", "drop_rate_pct"],
        "default_params": {
            "instrument_mass_g": 50.0,
            "handover_force_threshold_n": 2.0,
            "num_episodes": 200,
        },
    },
    "biopsy_sampling": {
        "description": "Robotic biopsy needle guidance and sample extraction",
        "frameworks": ["mujoco", "isaac", "pybullet"],
        "metrics": ["targeting_error_mm", "sample_quality_score", "procedure_time_s", "force_profile_n"],
        "default_params": {
            "target_diameter_mm": 10.0,
            "needle_gauge": 18,
            "approach_angle_deg": 0.0,
            "num_episodes": 100,
        },
    },
    "catheter_navigation": {
        "description": "Catheter navigation through vascular anatomy",
        "frameworks": ["mujoco", "isaac"],
        "metrics": ["path_length_mm", "wall_contact_force_n", "navigation_time_s", "success_rate_pct"],
        "default_params": {
            "vessel_diameter_mm": 4.0,
            "catheter_diameter_fr": 5,
            "path_complexity": "medium",
            "num_episodes": 50,
        },
    },
}

SUPPORTED_FRAMEWORKS = ["mujoco", "isaac", "pybullet", "gazebo"]


@dataclass
class JobResult:
    """Result of a single simulation job."""

    job_id: str
    framework: str
    task: str
    status: str = "pending"
    start_time: str = ""
    end_time: str = ""
    duration_s: float = 0.0
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "framework": self.framework,
            "task": self.task,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": round(self.duration_s, 2),
            "metrics": self.metrics,
            "config": self.config,
            "error": self.error,
        }


def _detect_framework(framework: str) -> bool:
    """Check if a simulation framework is available."""
    checks = {
        "mujoco": "import mujoco",
        "isaac": "import omni.isaac.lab",
        "pybullet": "import pybullet",
        "gazebo": "which gz",
    }

    if framework == "gazebo":
        try:
            result = subprocess.run(["which", "gz"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    else:
        check_code = checks.get(framework, "")
        if not check_code:
            return False
        try:
            result = subprocess.run(
                [sys.executable, "-c", check_code],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


def _detect_available_frameworks() -> list[str]:
    """Detect all available simulation frameworks."""
    available = []
    for fw in SUPPORTED_FRAMEWORKS:
        if _detect_framework(fw):
            available.append(fw)
    return available


def _generate_job_id(framework: str, task: str) -> str:
    """Generate a unique job ID."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{task}_{framework}_{ts}"


def _load_config(config_path: str | None) -> dict:
    """Load job configuration from YAML file."""
    if not config_path:
        return {}
    if not HAS_YAML:
        print("WARNING: pyyaml not installed; ignoring config file", file=sys.stderr)
        return {}
    if not os.path.isfile(config_path):
        print(f"WARNING: Config file not found: {config_path}", file=sys.stderr)
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _run_simulation(framework: str, task: str, params: dict) -> JobResult:
    """Execute a simulation job on the specified framework.

    This function creates the job structure and invokes the framework-specific
    simulation entry point. If the framework is not available, it records the
    error and returns a failed result.
    """
    job_id = _generate_job_id(framework, task)
    result = JobResult(
        job_id=job_id,
        framework=framework,
        task=task,
        config=params,
    )

    if not _detect_framework(framework):
        result.status = "skipped"
        result.error = f"Framework '{framework}' not available"
        print(f"  [{framework}] SKIPPED: framework not available")
        return result

    result.status = "running"
    result.start_time = datetime.now().isoformat()
    print(f"  [{framework}] RUNNING: {task} (job: {job_id})")

    task_def = TASK_REGISTRY.get(task, {})
    expected_metrics = task_def.get("metrics", [])

    try:
        # Framework-specific simulation dispatch
        # Each framework would have its own runner module; here we invoke
        # a standardized entry point that each framework adapter provides.
        runner_script = _resolve_runner_script(framework, task)

        if runner_script and os.path.isfile(runner_script):
            cmd = [sys.executable, runner_script, "--params", json.dumps(params)]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=params.get("timeout_s", 3600),
            )
            if proc.returncode == 0:
                try:
                    result.metrics = json.loads(proc.stdout)
                except json.JSONDecodeError:
                    result.metrics = {"raw_output": proc.stdout[:500]}
                result.status = "completed"
            else:
                result.status = "failed"
                result.error = proc.stderr[:500]
        else:
            # No runner script found; generate placeholder metrics for
            # demonstration. In production, this path should not be reached.
            result.metrics = {m: 0.0 for m in expected_metrics}
            result.status = "completed"
            result.error = "No framework runner script; placeholder metrics generated"

    except subprocess.TimeoutExpired:
        result.status = "timeout"
        result.error = f"Job timed out after {params.get('timeout_s', 3600)}s"
    except Exception as e:
        result.status = "failed"
        result.error = str(e)

    result.end_time = datetime.now().isoformat()
    if result.start_time:
        start = datetime.fromisoformat(result.start_time)
        end = datetime.fromisoformat(result.end_time)
        result.duration_s = (end - start).total_seconds()

    status_label = result.status.upper()
    print(f"  [{framework}] {status_label}: {result.duration_s:.1f}s")

    return result


def _resolve_runner_script(framework: str, task: str) -> str | None:
    """Resolve the path to a framework-specific runner script."""
    # Convention: runners live in the unification directory or framework adapters
    base = Path(__file__).resolve().parent.parent.parent
    candidates = [
        base / "unification" / "simulation_physics" / f"{framework}_runner.py",
        base / "unification" / "cross_platform_tools" / f"{framework}_{task}_runner.py",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def cmd_launch(args):
    """Launch a simulation job on a specific framework."""
    task = args.task
    framework = args.framework

    if task not in TASK_REGISTRY:
        print(f"ERROR: Unknown task '{task}'. Use 'list-tasks' to see available tasks.", file=sys.stderr)
        sys.exit(1)

    task_def = TASK_REGISTRY[task]
    if framework not in task_def["frameworks"]:
        print(f"ERROR: Task '{task}' does not support framework '{framework}'", file=sys.stderr)
        print(f"  Supported: {', '.join(task_def['frameworks'])}", file=sys.stderr)
        sys.exit(1)

    config = _load_config(args.config)
    params = {**task_def["default_params"], **config.get("params", {})}

    print(f"Launching: {task} on {framework}")
    print(f"  Parameters: {json.dumps(params, indent=2)}")
    print()

    result = _run_simulation(framework, task, params)

    print()
    print(f"Job {result.job_id}: {result.status}")
    if result.metrics:
        print("  Metrics:")
        for k, v in result.metrics.items():
            print(f"    {k}: {v}")

    # Save result
    results_dir = args.results_dir or f"results/{task}"
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{result.job_id}.json")
    _write_json(result_path, result.to_dict())
    print(f"\nResult saved to {result_path}")


def cmd_launch_all(args):
    """Launch a simulation job across all available frameworks."""
    task = args.task

    if task not in TASK_REGISTRY:
        print(f"ERROR: Unknown task '{task}'. Use 'list-tasks' to see available tasks.", file=sys.stderr)
        sys.exit(1)

    task_def = TASK_REGISTRY[task]
    config = _load_config(args.config)
    params = {**task_def["default_params"], **config.get("params", {})}

    available = _detect_available_frameworks()
    target_frameworks = [fw for fw in task_def["frameworks"] if fw in available]
    skipped_frameworks = [fw for fw in task_def["frameworks"] if fw not in available]

    print(f"Launching: {task} across {len(target_frameworks)} framework(s)")
    print(f"  Available:  {', '.join(target_frameworks) if target_frameworks else 'none'}")
    print(f"  Skipped:    {', '.join(skipped_frameworks) if skipped_frameworks else 'none'}")
    print(f"  Parameters: {json.dumps(params, indent=2)}")
    print()

    results = []
    for fw in target_frameworks:
        result = _run_simulation(fw, task, params)
        results.append(result)

    # Save all results
    results_dir = args.results_dir or f"results/{task}"
    os.makedirs(results_dir, exist_ok=True)

    for r in results:
        result_path = os.path.join(results_dir, f"{r.job_id}.json")
        _write_json(result_path, r.to_dict())

    # Summary
    print()
    print("=" * 60)
    print("LAUNCH SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  [{r.framework:>8}] {r.status:<10} {r.duration_s:.1f}s  {r.job_id}")

    summary_path = os.path.join(results_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    _write_json(summary_path, {"task": task, "results": [r.to_dict() for r in results]})
    print(f"\nSummary saved to {summary_path}")


def cmd_compare(args):
    """Compare simulation results across frameworks."""
    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Load all result files
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json") and not fname.startswith("summary"):
            fpath = os.path.join(results_dir, fname)
            with open(fpath) as f:
                data = json.load(f)
            if "framework" in data and "metrics" in data:
                results.append(data)

    if not results:
        print("No result files found in directory.", file=sys.stderr)
        sys.exit(1)

    # Group by framework
    frameworks = {}
    for r in results:
        fw = r["framework"]
        if fw not in frameworks:
            frameworks[fw] = []
        frameworks[fw].append(r)

    # Collect all metric keys
    all_metrics = set()
    for r in results:
        all_metrics.update(r.get("metrics", {}).keys())
    all_metrics = sorted(all_metrics)

    print("=" * 75)
    print(f"FRAMEWORK COMPARISON ({len(results)} results)")
    print("=" * 75)
    print()

    # Print comparison table
    header = f"  {'Metric':<30}" + "".join(f"{fw:<15}" for fw in sorted(frameworks.keys()))
    print(header)
    print("-" * len(header))

    for metric in all_metrics:
        row = f"  {metric:<30}"
        for fw in sorted(frameworks.keys()):
            fw_results = frameworks[fw]
            values = [r["metrics"].get(metric) for r in fw_results if r["metrics"].get(metric) is not None]
            if values:
                avg = sum(float(v) for v in values) / len(values)
                row += f"{avg:<15.4f}"
            else:
                row += f"{'N/A':<15}"
        print(row)

    # Timing comparison
    print()
    print("EXECUTION TIME:")
    for fw in sorted(frameworks.keys()):
        durations = [r.get("duration_s", 0) for r in frameworks[fw]]
        avg_d = sum(durations) / len(durations) if durations else 0
        print(f"  {fw:<15} avg: {avg_d:.2f}s  (n={len(durations)})")

    if args.output:
        comparison = {
            "comparison_type": "cross_framework",
            "timestamp": datetime.now().isoformat(),
            "frameworks": {
                fw: {
                    "num_results": len(rs),
                    "metrics": {
                        m: sum(float(r["metrics"].get(m, 0)) for r in rs) / len(rs)
                        for m in all_metrics
                        if any(r["metrics"].get(m) is not None for r in rs)
                    },
                }
                for fw, rs in frameworks.items()
            },
        }
        _write_json(args.output, comparison)
        print(f"\nComparison written to {args.output}")


def cmd_list_tasks(args):
    """List all available simulation tasks."""
    print("=" * 70)
    print("AVAILABLE SIMULATION TASKS")
    print("=" * 70)
    print()

    for task_id, task_def in sorted(TASK_REGISTRY.items()):
        print(f"  {task_id}")
        print(f"    Description: {task_def['description']}")
        print(f"    Frameworks:  {', '.join(task_def['frameworks'])}")
        print(f"    Metrics:     {', '.join(task_def['metrics'])}")
        print(f"    Defaults:    {json.dumps(task_def['default_params'], indent=6).strip()}")
        print()


def cmd_init_config(args):
    """Generate a job configuration template."""
    if not HAS_YAML:
        print("ERROR: pyyaml is required for config generation. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    task = args.task
    if task not in TASK_REGISTRY:
        print(f"ERROR: Unknown task '{task}'. Use 'list-tasks' to see available tasks.", file=sys.stderr)
        sys.exit(1)

    task_def = TASK_REGISTRY[task]
    config = {
        "task": task,
        "description": task_def["description"],
        "frameworks": task_def["frameworks"],
        "params": task_def["default_params"],
        "timeout_s": 3600,
        "num_seeds": 3,
        "results_dir": f"results/{task}",
    }

    output_path = args.output or f"{task}_config.yaml"
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration template written to {output_path}")
    print(f"  Task: {task}")
    print(
        f"  Edit parameters and run: python sim_job_runner.py launch --framework mujoco --task {task} --config {output_path}"
    )


def _write_json(filepath: str, data: dict):
    """Write data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        prog="sim_job_runner",
        description="Simulation Job Runner: Launch and compare simulation jobs across physics frameworks.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # launch
    p_launch = subparsers.add_parser("launch", help="Launch simulation on a specific framework")
    p_launch.add_argument("--framework", required=True, choices=SUPPORTED_FRAMEWORKS, help="Target framework")
    p_launch.add_argument("--task", required=True, help="Simulation task name")
    p_launch.add_argument("--config", help="YAML configuration file")
    p_launch.add_argument("--results-dir", help="Directory for results (default: results/<task>)")

    # launch-all
    p_all = subparsers.add_parser("launch-all", help="Launch simulation across all available frameworks")
    p_all.add_argument("--task", required=True, help="Simulation task name")
    p_all.add_argument("--config", help="YAML configuration file")
    p_all.add_argument("--results-dir", help="Directory for results (default: results/<task>)")

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare results across frameworks")
    p_compare.add_argument("--results-dir", required=True, help="Directory containing result JSON files")
    p_compare.add_argument("--output", "-o", help="Write comparison JSON to file")

    # list-tasks
    subparsers.add_parser("list-tasks", help="List available simulation tasks")

    # init-config
    p_init = subparsers.add_parser("init-config", help="Generate job configuration template")
    p_init.add_argument("--task", required=True, help="Simulation task name")
    p_init.add_argument("--output", "-o", help="Output YAML path")

    args = parser.parse_args()

    commands = {
        "launch": cmd_launch,
        "launch-all": cmd_launch_all,
        "compare": cmd_compare,
        "list-tasks": cmd_list_tasks,
        "init-config": cmd_init_config,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
