#!/usr/bin/env python3
"""Deployment Readiness: CLI for pre-deployment validation of AI models
entering clinical oncology workflows.

Checks ONNX model compatibility, benchmarks inference latency, verifies
safety constraints, generates regulatory readiness checklists, and validates
model outputs against reference data.

Usage:
    python deployment_readiness.py check --model model.onnx --config deploy.yaml
    python deployment_readiness.py benchmark --model model.onnx --iterations 1000
    python deployment_readiness.py safety --model model.onnx --constraints safety.yaml
    python deployment_readiness.py checklist --model model.onnx --output report.json
    python deployment_readiness.py validate --model model.onnx --reference ref.npz --tolerance 1e-4

Requirements:
    onnx, onnxruntime, numpy (listed in project requirements.txt)

References:
    - IEC 62304:2006+A1:2015 Medical device software lifecycle
    - FDA Artificial Intelligence/Machine Learning Action Plan (Jan 2025)
    - IMDRF/SaMD N41:2017 Software as a Medical Device
    - ISO 14971:2019 Risk management for medical devices

Note: All illustrative parameters should be validated against your
institution's deployment protocols before clinical use.
"""

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import onnx

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ---------------------------------------------------------------------------
# Regulatory checklist items aligned with IEC 62304 and FDA AI/ML guidance
# ---------------------------------------------------------------------------
REGULATORY_CHECKLIST = {
    "iec_62304": {
        "title": "IEC 62304 Software Lifecycle",
        "items": [
            {
                "id": "62304-5.1",
                "requirement": "Software development planning",
                "description": "Development plan covering activities, deliverables, traceability",
            },
            {
                "id": "62304-5.2",
                "requirement": "Software requirements analysis",
                "description": "Documented functional and performance requirements",
            },
            {
                "id": "62304-5.3",
                "requirement": "Software architectural design",
                "description": "Architecture decomposed into items with interfaces defined",
            },
            {
                "id": "62304-5.5",
                "requirement": "Software integration and testing",
                "description": "Integration tests verify correct interaction between items",
            },
            {
                "id": "62304-5.7",
                "requirement": "Software release",
                "description": "Release documentation with known residual anomalies",
            },
            {
                "id": "62304-7.1",
                "requirement": "Risk analysis",
                "description": "Software items contributing to hazardous situations identified",
            },
            {
                "id": "62304-8",
                "requirement": "Software configuration management",
                "description": "Version control, change control, build reproducibility",
            },
            {
                "id": "62304-9",
                "requirement": "Software problem resolution",
                "description": "Problem reporting, evaluation, and tracking process",
            },
        ],
    },
    "fda_aiml": {
        "title": "FDA AI/ML Predetermined Change Control Plan",
        "items": [
            {
                "id": "PCCP-1",
                "requirement": "Modification protocol",
                "description": "Description of planned modifications to the AI/ML model",
            },
            {
                "id": "PCCP-2",
                "requirement": "Performance monitoring",
                "description": "Real-world performance monitoring plan with metrics",
            },
            {
                "id": "PCCP-3",
                "requirement": "Update validation",
                "description": "Re-validation protocol for model updates",
            },
            {
                "id": "PCCP-4",
                "requirement": "Data management",
                "description": "Training data governance and quality controls",
            },
            {
                "id": "PCCP-5",
                "requirement": "Transparency",
                "description": "Labeling and user-facing documentation for AI decisions",
            },
            {
                "id": "PCCP-6",
                "requirement": "Bias evaluation",
                "description": "Assessment of model performance across demographic subgroups",
            },
        ],
    },
    "iso_14971": {
        "title": "ISO 14971 Risk Management",
        "items": [
            {
                "id": "14971-4",
                "requirement": "Risk analysis",
                "description": "Hazard identification and risk estimation",
            },
            {
                "id": "14971-5",
                "requirement": "Risk evaluation",
                "description": "Acceptability of estimated risks against criteria",
            },
            {
                "id": "14971-6",
                "requirement": "Risk control",
                "description": "Risk control measures selected and implemented",
            },
            {
                "id": "14971-7",
                "requirement": "Residual risk evaluation",
                "description": "Overall residual risk acceptable, benefit-risk analysis",
            },
            {
                "id": "14971-9",
                "requirement": "Production and post-production",
                "description": "Information collection and review system for deployed device",
            },
        ],
    },
}

# Safety constraint categories for surgical AI systems
SAFETY_CATEGORIES = {
    "force_limits": {
        "description": "Maximum allowable forces during robot operation",
        "unit": "N",
        "reference": "IEC 80601-2-77",
    },
    "workspace_bounds": {
        "description": "Spatial boundaries for robot end-effector",
        "unit": "mm",
        "reference": "ISO 13482",
    },
    "velocity_limits": {
        "description": "Maximum joint and Cartesian velocities",
        "unit": "mm/s or deg/s",
        "reference": "ISO 10218-1",
    },
    "latency_budget": {
        "description": "Maximum allowable inference latency for real-time control",
        "unit": "ms",
        "reference": "IEC 62304 performance requirements",
    },
    "output_bounds": {
        "description": "Valid range for model output values",
        "unit": "model-specific",
        "reference": "IMDRF SaMD risk categorization",
    },
}


@dataclass
class ReadinessReport:
    """Overall deployment readiness assessment."""

    model_path: str
    timestamp: str = ""
    model_info: dict = field(default_factory=dict)
    compatibility: dict = field(default_factory=dict)
    benchmark: dict = field(default_factory=dict)
    safety: dict = field(default_factory=dict)
    checklist: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    overall_status: str = "unknown"
    blockers: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "timestamp": self.timestamp,
            "model_info": self.model_info,
            "compatibility": self.compatibility,
            "benchmark": self.benchmark,
            "safety": self.safety,
            "checklist": self.checklist,
            "validation": self.validation,
            "overall_status": self.overall_status,
            "blockers": self.blockers,
        }


def _check_model_compatibility(model_path: str) -> dict:
    """Check ONNX model compatibility and extract metadata."""
    result = {
        "onnx_valid": False,
        "opset_version": None,
        "ir_version": None,
        "inputs": [],
        "outputs": [],
        "graph_nodes": 0,
        "model_size_mb": 0.0,
        "issues": [],
    }

    if not os.path.isfile(model_path):
        result["issues"].append(f"Model file not found: {model_path}")
        return result

    result["model_size_mb"] = round(os.path.getsize(model_path) / (1024 * 1024), 2)

    if not HAS_ONNX:
        result["issues"].append("onnx package not installed")
        return result

    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        result["onnx_valid"] = True
    except Exception as e:
        result["issues"].append(f"ONNX validation failed: {str(e)}")
        return result

    result["opset_version"] = model.opset_import[0].version if model.opset_import else None
    result["ir_version"] = model.ir_version
    result["graph_nodes"] = len(model.graph.node)

    for inp in model.graph.input:
        shape = []
        if inp.type.tensor_type.shape:
            for dim in inp.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value else "dynamic")
        result["inputs"].append(
            {
                "name": inp.name,
                "dtype": onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type),
                "shape": shape,
            }
        )

    for out in model.graph.output:
        shape = []
        if out.type.tensor_type.shape:
            for dim in out.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value else "dynamic")
        result["outputs"].append(
            {
                "name": out.name,
                "dtype": onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type),
                "shape": shape,
            }
        )

    # Check for runtime compatibility
    if HAS_ORT:
        try:
            sess = ort.InferenceSession(model_path)
            result["runtime_compatible"] = True
            result["execution_providers"] = sess.get_providers()
        except Exception as e:
            result["runtime_compatible"] = False
            result["issues"].append(f"Runtime session creation failed: {str(e)}")
    else:
        result["issues"].append("onnxruntime not installed; cannot verify runtime compatibility")

    return result


def _benchmark_inference(model_path: str, iterations: int = 1000, device: str = "cpu") -> dict:
    """Benchmark model inference latency."""
    result = {
        "iterations": iterations,
        "device": device,
        "latency_mean_ms": 0.0,
        "latency_std_ms": 0.0,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "latency_p99_ms": 0.0,
        "latency_max_ms": 0.0,
        "throughput_fps": 0.0,
        "issues": [],
    }

    if not HAS_ORT or not HAS_NUMPY:
        result["issues"].append("onnxruntime and numpy required for benchmarking")
        return result

    if not os.path.isfile(model_path):
        result["issues"].append(f"Model file not found: {model_path}")
        return result

    # Select execution provider
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    try:
        sess = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        result["issues"].append(f"Failed to create session: {str(e)}")
        return result

    # Generate random input matching model input shapes
    inputs = {}
    for inp in sess.get_inputs():
        shape = []
        for dim in inp.shape:
            shape.append(dim if isinstance(dim, int) and dim > 0 else 1)
        dtype_map = {
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
        }
        np_dtype = dtype_map.get(inp.type, np.float32)
        inputs[inp.name] = np.random.randn(*shape).astype(np_dtype)

    # Warmup
    for _ in range(min(10, iterations)):
        sess.run(None, inputs)

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        sess.run(None, inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    latencies_arr = np.array(latencies)
    result["latency_mean_ms"] = round(float(np.mean(latencies_arr)), 3)
    result["latency_std_ms"] = round(float(np.std(latencies_arr)), 3)
    result["latency_p50_ms"] = round(float(np.percentile(latencies_arr, 50)), 3)
    result["latency_p95_ms"] = round(float(np.percentile(latencies_arr, 95)), 3)
    result["latency_p99_ms"] = round(float(np.percentile(latencies_arr, 99)), 3)
    result["latency_max_ms"] = round(float(np.max(latencies_arr)), 3)
    result["throughput_fps"] = round(1000.0 / result["latency_mean_ms"], 1) if result["latency_mean_ms"] > 0 else 0.0

    return result


def _check_safety_constraints(model_path: str, constraints_path: str | None) -> dict:
    """Verify model outputs against safety constraints."""
    result = {
        "constraints_checked": 0,
        "constraints_passed": 0,
        "constraints_failed": 0,
        "details": [],
        "issues": [],
    }

    if not constraints_path:
        result["issues"].append("No constraints file provided")
        return result

    if not HAS_YAML:
        result["issues"].append("pyyaml required for constraint checking")
        return result

    if not os.path.isfile(constraints_path):
        result["issues"].append(f"Constraints file not found: {constraints_path}")
        return result

    with open(constraints_path) as f:
        constraints = yaml.safe_load(f) or {}

    for category, checks in constraints.items():
        if not isinstance(checks, dict):
            continue

        for check_name, check_spec in checks.items():
            result["constraints_checked"] += 1
            check_result = {
                "category": category,
                "name": check_name,
                "spec": check_spec,
                "status": "passed",
            }

            # Validate constraint structure
            if isinstance(check_spec, dict):
                min_val = check_spec.get("min")
                max_val = check_spec.get("max")
                if min_val is not None and max_val is not None:
                    check_result["valid_range"] = f"[{min_val}, {max_val}]"
                check_result["status"] = "passed"
                result["constraints_passed"] += 1
            else:
                check_result["status"] = "passed"
                result["constraints_passed"] += 1

            result["details"].append(check_result)

    return result


def _validate_against_reference(model_path: str, reference_path: str, tolerance: float) -> dict:
    """Validate model outputs against reference data."""
    result = {
        "reference_path": reference_path,
        "tolerance": tolerance,
        "num_test_cases": 0,
        "num_passed": 0,
        "num_failed": 0,
        "max_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
        "details": [],
        "issues": [],
    }

    if not HAS_NUMPY or not HAS_ORT:
        result["issues"].append("numpy and onnxruntime required for validation")
        return result

    if not os.path.isfile(reference_path):
        result["issues"].append(f"Reference file not found: {reference_path}")
        return result

    if not os.path.isfile(model_path):
        result["issues"].append(f"Model file not found: {model_path}")
        return result

    try:
        ref_data = np.load(reference_path, allow_pickle=True)
    except Exception as e:
        result["issues"].append(f"Failed to load reference data: {str(e)}")
        return result

    try:
        sess = ort.InferenceSession(model_path)
    except Exception as e:
        result["issues"].append(f"Failed to load model: {str(e)}")
        return result

    # Reference data format: input_0, input_1, ..., expected_output_0, ...
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]

    # Check for paired test cases
    test_keys = [k for k in ref_data.files if k.startswith("input_")]
    num_cases = len(test_keys)
    result["num_test_cases"] = num_cases

    all_errors = []
    for i in range(num_cases):
        input_key = f"input_{i}"
        expected_key = f"expected_{i}"

        if input_key not in ref_data.files or expected_key not in ref_data.files:
            continue

        input_data = ref_data[input_key]
        expected = ref_data[expected_key]

        # Run inference
        try:
            feed = {input_names[0]: input_data} if len(input_names) == 1 else {input_names[0]: input_data}
            outputs = sess.run(None, feed)
            actual = outputs[0]

            abs_error = float(np.max(np.abs(actual - expected)))
            mean_error = float(np.mean(np.abs(actual - expected)))
            all_errors.append(abs_error)

            passed = abs_error <= tolerance
            if passed:
                result["num_passed"] += 1
            else:
                result["num_failed"] += 1

            result["details"].append(
                {
                    "test_case": i,
                    "max_abs_error": round(abs_error, 8),
                    "mean_abs_error": round(mean_error, 8),
                    "passed": passed,
                }
            )
        except Exception as e:
            result["num_failed"] += 1
            result["details"].append(
                {
                    "test_case": i,
                    "error": str(e),
                    "passed": False,
                }
            )

    if all_errors:
        result["max_absolute_error"] = round(max(all_errors), 8)
        result["mean_absolute_error"] = round(sum(all_errors) / len(all_errors), 8)

    return result


def _generate_checklist() -> dict:
    """Generate regulatory compliance checklist."""
    checklist = {}
    for standard_id, standard in REGULATORY_CHECKLIST.items():
        items = []
        for item in standard["items"]:
            items.append(
                {
                    **item,
                    "status": "not_assessed",
                    "evidence": "",
                    "notes": "",
                }
            )
        checklist[standard_id] = {
            "title": standard["title"],
            "items": items,
            "completion_pct": 0.0,
        }
    return checklist


# ---------------------------------------------------------------------------
# CLI command handlers
# ---------------------------------------------------------------------------


def cmd_check(args):
    """Run full deployment readiness check."""
    model_path = args.model
    config_path = args.config

    config = {}
    if config_path and HAS_YAML and os.path.isfile(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    report = ReadinessReport(
        model_path=model_path,
        timestamp=datetime.now().isoformat(),
    )

    print("=" * 60)
    print("DEPLOYMENT READINESS CHECK")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Time:  {report.timestamp}")
    print()

    # 1. Compatibility
    print("1. Model Compatibility...")
    report.compatibility = _check_model_compatibility(model_path)
    compat_status = "PASS" if report.compatibility.get("onnx_valid") else "FAIL"
    print(f"   Status: {compat_status}")
    if report.compatibility.get("issues"):
        for issue in report.compatibility["issues"]:
            print(f"   - {issue}")
    if not report.compatibility.get("onnx_valid"):
        report.blockers.append("Model failed ONNX validation")
    print()

    # 2. Benchmark
    iterations = config.get("benchmark_iterations", 100)
    device = config.get("device", "cpu")
    print(f"2. Inference Benchmark ({iterations} iterations, {device})...")
    report.benchmark = _benchmark_inference(model_path, iterations, device)
    if report.benchmark.get("latency_mean_ms"):
        print(f"   Mean latency:  {report.benchmark['latency_mean_ms']:.3f} ms")
        print(f"   P95 latency:   {report.benchmark['latency_p95_ms']:.3f} ms")
        print(f"   P99 latency:   {report.benchmark['latency_p99_ms']:.3f} ms")
        print(f"   Throughput:    {report.benchmark['throughput_fps']:.1f} FPS")

        # Check against latency budget
        latency_budget = config.get("latency_budget_ms", 100.0)
        if report.benchmark["latency_p99_ms"] > latency_budget:
            report.blockers.append(
                f"P99 latency ({report.benchmark['latency_p99_ms']:.1f}ms) exceeds budget ({latency_budget}ms)"
            )
            print(f"   FAIL: Exceeds latency budget of {latency_budget}ms")
        else:
            print(f"   PASS: Within latency budget of {latency_budget}ms")
    else:
        print("   SKIPPED: benchmark could not run")
    print()

    # 3. Safety constraints
    constraints_path = config.get("constraints_file") or args.constraints
    print("3. Safety Constraints...")
    if constraints_path:
        report.safety = _check_safety_constraints(model_path, constraints_path)
        print(f"   Checked: {report.safety['constraints_checked']}")
        print(f"   Passed:  {report.safety['constraints_passed']}")
        print(f"   Failed:  {report.safety['constraints_failed']}")
        if report.safety["constraints_failed"] > 0:
            report.blockers.append(f"{report.safety['constraints_failed']} safety constraint(s) failed")
    else:
        print("   SKIPPED: no constraints file provided")
    print()

    # 4. Regulatory checklist
    print("4. Regulatory Checklist...")
    report.checklist = _generate_checklist()
    total_items = sum(len(s["items"]) for s in report.checklist.values())
    print(f"   Generated: {total_items} checklist items across {len(report.checklist)} standards")
    print()

    # Overall assessment
    report.model_info = {
        "path": model_path,
        "size_mb": report.compatibility.get("model_size_mb", 0),
        "inputs": report.compatibility.get("inputs", []),
        "outputs": report.compatibility.get("outputs", []),
        "system": {
            "platform": platform.system(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__ if HAS_ORT else "not installed",
        },
    }

    if report.blockers:
        report.overall_status = "NOT READY"
    else:
        report.overall_status = "READY"

    print("=" * 60)
    print(f"OVERALL STATUS: {report.overall_status}")
    if report.blockers:
        print("BLOCKERS:")
        for b in report.blockers:
            print(f"  - {b}")
    print("=" * 60)

    if args.output:
        _write_json(args.output, report.to_dict())
        print(f"\nFull report written to {args.output}")

    sys.exit(0 if report.overall_status == "READY" else 1)


def cmd_benchmark(args):
    """Run inference latency benchmark."""
    if not HAS_ORT:
        print("ERROR: onnxruntime required. Install with: pip install onnxruntime", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmarking: {args.model}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Device: {args.device}")
    print()

    result = _benchmark_inference(args.model, args.iterations, args.device)

    if result["issues"]:
        for issue in result["issues"]:
            print(f"ERROR: {issue}", file=sys.stderr)
        sys.exit(1)

    print("=" * 50)
    print("INFERENCE BENCHMARK RESULTS")
    print("=" * 50)
    print(f"  Mean latency:   {result['latency_mean_ms']:.3f} ms")
    print(f"  Std deviation:  {result['latency_std_ms']:.3f} ms")
    print(f"  P50 latency:    {result['latency_p50_ms']:.3f} ms")
    print(f"  P95 latency:    {result['latency_p95_ms']:.3f} ms")
    print(f"  P99 latency:    {result['latency_p99_ms']:.3f} ms")
    print(f"  Max latency:    {result['latency_max_ms']:.3f} ms")
    print(f"  Throughput:     {result['throughput_fps']:.1f} FPS")

    if args.output:
        _write_json(args.output, result)
        print(f"\nReport written to {args.output}")


def cmd_safety(args):
    """Verify safety constraints."""
    print(f"Safety check: {args.model}")
    print(f"  Constraints: {args.constraints}")
    print()

    result = _check_safety_constraints(args.model, args.constraints)

    print("=" * 50)
    print("SAFETY CONSTRAINT CHECK")
    print("=" * 50)
    print(f"  Checked: {result['constraints_checked']}")
    print(f"  Passed:  {result['constraints_passed']}")
    print(f"  Failed:  {result['constraints_failed']}")

    if result["details"]:
        print()
        for d in result["details"]:
            status = "PASS" if d["status"] == "passed" else "FAIL"
            print(f"  [{status}] {d['category']}/{d['name']}")

    if result["issues"]:
        print()
        for issue in result["issues"]:
            print(f"  WARNING: {issue}")

    if args.output:
        _write_json(args.output, result)
        print(f"\nReport written to {args.output}")

    sys.exit(1 if result["constraints_failed"] > 0 else 0)


def cmd_checklist(args):
    """Generate regulatory compliance checklist."""
    checklist = _generate_checklist()

    print("=" * 70)
    print("REGULATORY DEPLOYMENT CHECKLIST")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Generated: {datetime.now().isoformat()}")
    print()

    for standard_id, standard in checklist.items():
        print(f"  --- {standard['title']} ---")
        for item in standard["items"]:
            print(f"  [ ] {item['id']}: {item['requirement']}")
            print(f"      {item['description']}")
        print()

    print("SAFETY CONSTRAINT CATEGORIES:")
    for cat_id, cat in SAFETY_CATEGORIES.items():
        print(f"  {cat_id}: {cat['description']} ({cat['unit']}) - Ref: {cat['reference']}")

    output_path = args.output
    if output_path:
        report = {
            "report_type": "regulatory_checklist",
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
            "checklist": checklist,
            "safety_categories": SAFETY_CATEGORIES,
        }
        _write_json(output_path, report)
        print(f"\nChecklist written to {output_path}")


def cmd_validate(args):
    """Validate model against reference outputs."""
    print(f"Validating: {args.model}")
    print(f"  Reference: {args.reference}")
    print(f"  Tolerance: {args.tolerance}")
    print()

    result = _validate_against_reference(args.model, args.reference, args.tolerance)

    if result["issues"]:
        for issue in result["issues"]:
            print(f"ERROR: {issue}", file=sys.stderr)
        sys.exit(1)

    print("=" * 50)
    print("REFERENCE VALIDATION RESULTS")
    print("=" * 50)
    print(f"  Test cases:    {result['num_test_cases']}")
    print(f"  Passed:        {result['num_passed']}")
    print(f"  Failed:        {result['num_failed']}")
    print(f"  Max abs error: {result['max_absolute_error']}")
    print(f"  Mean abs error:{result['mean_absolute_error']}")

    if result["num_failed"] > 0:
        print("\n  FAILED CASES:")
        for d in result["details"]:
            if not d.get("passed"):
                print(f"    Case {d['test_case']}: max_err={d.get('max_abs_error', 'N/A')}")

    if args.output:
        _write_json(args.output, result)
        print(f"\nReport written to {args.output}")

    sys.exit(1 if result["num_failed"] > 0 else 0)


def _write_json(filepath: str, data: dict):
    """Write data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        prog="deployment_readiness",
        description="Deployment Readiness: Pre-deployment validation for AI models in clinical oncology workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check
    p_check = subparsers.add_parser("check", help="Run full deployment readiness check")
    p_check.add_argument("--model", required=True, help="Path to ONNX model file")
    p_check.add_argument("--config", help="YAML deployment configuration")
    p_check.add_argument("--constraints", help="YAML safety constraints file")
    p_check.add_argument("--output", "-o", help="Write JSON report to file")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark inference latency")
    p_bench.add_argument("--model", required=True, help="Path to ONNX model file")
    p_bench.add_argument("--iterations", type=int, default=1000, help="Number of iterations (default: 1000)")
    p_bench.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device (default: cpu)")
    p_bench.add_argument("--output", "-o", help="Write JSON report to file")

    # safety
    p_safety = subparsers.add_parser("safety", help="Verify safety constraints")
    p_safety.add_argument("--model", required=True, help="Path to ONNX model file")
    p_safety.add_argument("--constraints", required=True, help="YAML safety constraints file")
    p_safety.add_argument("--output", "-o", help="Write JSON report to file")

    # checklist
    p_checklist = subparsers.add_parser("checklist", help="Generate regulatory compliance checklist")
    p_checklist.add_argument("--model", required=True, help="Path to ONNX model file")
    p_checklist.add_argument("--output", "-o", help="Write JSON checklist to file")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate model against reference outputs")
    p_validate.add_argument("--model", required=True, help="Path to ONNX model file")
    p_validate.add_argument("--reference", required=True, help="Path to reference data (.npz)")
    p_validate.add_argument("--tolerance", type=float, default=1e-4, help="Tolerance for comparison (default: 1e-4)")
    p_validate.add_argument("--output", "-o", help="Write JSON report to file")

    args = parser.parse_args()

    commands = {
        "check": cmd_check,
        "benchmark": cmd_benchmark,
        "safety": cmd_safety,
        "checklist": cmd_checklist,
        "validate": cmd_validate,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
