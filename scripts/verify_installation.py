#!/usr/bin/env python3
"""
Verification script for Physical AI for Oncology Clinical Trials repository.
Checks availability of required frameworks and dependencies.

Usage:
    python scripts/verify_installation.py
"""

import sys
from importlib import import_module
from typing import Dict, List, Tuple

# Framework requirements with minimum versions
REQUIREMENTS: Dict[str, Tuple[str, str]] = {
    # Core
    "numpy": ("numpy", "1.24.0"),
    "torch": ("torch", "2.1.0"),
    "scipy": ("scipy", "1.11.0"),

    # Physics Simulation
    "mujoco": ("mujoco", "3.2.0"),
    "pybullet": ("pybullet", "3.2.0"),

    # Reinforcement Learning
    "stable_baselines3": ("stable_baselines3", "2.2.0"),
    "gymnasium": ("gymnasium", "0.29.0"),

    # Agentic AI
    "langchain": ("langchain", "0.3.0"),

    # Medical Imaging
    "monai": ("monai", "1.3.0"),

    # Deep Learning
    "transformers": ("transformers", "4.35.0"),

    # Deployment
    "onnx": ("onnx", "1.15.0"),
    "onnxruntime": ("onnxruntime", "1.16.0"),
}

# Optional frameworks (not required but recommended)
OPTIONAL: Dict[str, Tuple[str, str]] = {
    "isaaclab": ("isaaclab", "2.2.0"),  # Requires separate install
    "mjx": ("mujoco.mjx", "3.2.0"),  # MuJoCo JAX backend
    "crewai": ("crewai", "0.80.0"),
    "langgraph": ("langgraph", "0.2.0"),
}


def check_version(module_name: str, min_version: str) -> Tuple[bool, str]:
    """Check if module is installed and meets minimum version."""
    try:
        module = import_module(module_name)
        version = getattr(module, "__version__", "unknown")

        # Simple version comparison (works for most cases)
        if version != "unknown":
            installed = tuple(map(int, version.split(".")[:3]))
            required = tuple(map(int, min_version.split(".")[:3]))
            if installed >= required:
                return True, version
            else:
                return False, f"{version} (need >= {min_version})"
        return True, version
    except ImportError:
        return False, "not installed"
    except Exception as e:
        return False, str(e)


def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s)"
        return False, "CUDA not available"
    except Exception as e:
        return False, str(e)


def check_ros2() -> Tuple[bool, str]:
    """Check ROS 2 availability."""
    try:
        import subprocess
        result = subprocess.run(
            ["ros2", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, "ros2 command failed"
    except FileNotFoundError:
        return False, "ros2 not found in PATH"
    except Exception as e:
        return False, str(e)


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Physical AI for Oncology Clinical Trials")
    print("Installation Verification")
    print("=" * 60)
    print()

    all_passed = True

    # Check Python version
    print("Python Version:")
    py_version = sys.version_info
    if py_version >= (3, 10):
        print(f"  ✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"  ✗ Python {py_version.major}.{py_version.minor} (need >= 3.10)")
        all_passed = False
    print()

    # Check CUDA
    print("GPU/CUDA:")
    cuda_ok, cuda_info = check_cuda()
    if cuda_ok:
        print(f"  ✓ {cuda_info}")
    else:
        print(f"  ! {cuda_info}")
    print()

    # Check required packages
    print("Required Packages:")
    for name, (module, min_ver) in REQUIREMENTS.items():
        ok, info = check_version(module, min_ver)
        if ok:
            print(f"  ✓ {name}: {info}")
        else:
            print(f"  ✗ {name}: {info}")
            all_passed = False
    print()

    # Check optional packages
    print("Optional Packages:")
    for name, (module, min_ver) in OPTIONAL.items():
        ok, info = check_version(module, min_ver)
        if ok:
            print(f"  ✓ {name}: {info}")
        else:
            print(f"  - {name}: {info}")
    print()

    # Check ROS 2
    print("ROS 2:")
    ros_ok, ros_info = check_ros2()
    if ros_ok:
        print(f"  ✓ {ros_info}")
    else:
        print(f"  - {ros_info} (optional)")
    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("✓ All required dependencies satisfied!")
        print("  Ready for physical AI oncology development.")
    else:
        print("✗ Some required dependencies missing.")
        print("  Run: pip install -r requirements.txt")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
