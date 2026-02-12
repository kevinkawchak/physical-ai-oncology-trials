"""Diagnostic test that prints environment info for CI debugging.

This test always passes but prints useful diagnostic information
that will appear in the CI logs.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path


def test_environment_info():
    """Print environment info for debugging CI failures."""
    print(f"\n{'=' * 60}")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"CWD: {os.getcwd()}")
    print(f"sys.path[0:3]: {sys.path[0:3]}")

    conftest = Path("tests/conftest.py")
    print(f"conftest.py exists: {conftest.exists()}")
    print(f"conftest.py absolute: {conftest.resolve()}")

    # Check project root resolution
    project_root = Path(__file__).resolve().parent.parent
    print(f"PROJECT_ROOT: {project_root}")
    print(f"PROJECT_ROOT exists: {project_root.exists()}")

    # Check a few key source files exist
    key_files = [
        "examples-new/01_realtime_safety_monitoring.py",
        "digital-twins/patient-modeling/tumor_twin_pipeline.py",
        "tools/dose-calculator/dose_calculator.py",
        "privacy/phi-pii-management/phi_detector.py",
    ]
    for f in key_files:
        p = project_root / f
        print(f"  {f}: exists={p.exists()}")

    # Check numpy version
    import numpy as np

    print(f"numpy version: {np.__version__}")
    print(f"numpy path: {np.__file__}")

    # Try loading a module
    name = "safety_monitoring_diag"
    rel_path = "examples-new/01_realtime_safety_monitoring.py"
    filepath = project_root / rel_path
    print("\nDiagnostic load_module test:")
    print(f"  filepath: {filepath}")
    print(f"  filepath exists: {filepath.exists()}")
    if filepath.exists():
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"  Module loaded OK: {mod}")
        print(f"  Has SafetyMonitor: {hasattr(mod, 'SafetyMonitor')}")
    print(f"{'=' * 60}")


def test_load_module_from_conftest():
    """Verify load_module works correctly."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tests.conftest import load_module

    mod = load_module(
        "safety_mon_test_diag",
        "examples-new/01_realtime_safety_monitoring.py",
    )
    assert hasattr(mod, "SafetyMonitor"), f"Module attrs: {dir(mod)[:10]}"
    print(f"load_module OK: {mod}")
