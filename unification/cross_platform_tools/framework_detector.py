#!/usr/bin/env python3
"""
Framework Detection and Validation for Physical AI Oncology Trials

Detects available simulation frameworks and validates their installations
for cross-platform oncology robotics development.

Usage:
    python framework_detector.py [--verbose] [--json]

    # Programmatic use
    from unification.cross_platform_tools.framework_detector import FrameworkDetector
    detector = FrameworkDetector()
    available = detector.detect_all()

Last updated: January 2026
"""

import sys
import subprocess
import importlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import argparse


@dataclass
class FrameworkInfo:
    """Information about a detected framework."""
    name: str
    available: bool
    version: str = "unknown"
    gpu_support: bool = False
    ros2_integration: bool = False
    path: Optional[str] = None
    features: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommended_use: str = ""


@dataclass
class SystemInfo:
    """System information for framework compatibility."""
    python_version: str
    platform: str
    cuda_available: bool
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    ros2_available: bool = False
    ros2_distro: str = ""


class FrameworkDetector:
    """
    Detect and validate simulation framework installations.

    Checks for:
    - NVIDIA Isaac Lab / Isaac Sim
    - MuJoCo / MJX
    - Gazebo Ionic
    - PyBullet
    - ROS 2 integration
    """

    FRAMEWORKS = ["isaac", "mujoco", "gazebo", "pybullet"]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.system_info: Optional[SystemInfo] = None
        self.frameworks: Dict[str, FrameworkInfo] = {}

    def detect_all(self) -> Dict[str, FrameworkInfo]:
        """Detect all supported frameworks."""
        self.system_info = self._get_system_info()

        if self.verbose:
            print("Detecting available frameworks...")
            print(f"Python: {self.system_info.python_version}")
            print(f"Platform: {self.system_info.platform}")
            print(f"CUDA: {self.system_info.cuda_version if self.system_info.cuda_available else 'Not available'}")
            print()

        self.frameworks["isaac"] = self._detect_isaac()
        self.frameworks["mujoco"] = self._detect_mujoco()
        self.frameworks["gazebo"] = self._detect_gazebo()
        self.frameworks["pybullet"] = self._detect_pybullet()

        return self.frameworks

    def _get_system_info(self) -> SystemInfo:
        """Get system information."""
        import platform

        info = SystemInfo(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=platform.system(),
            cuda_available=False,
        )

        # Check CUDA
        try:
            import torch
            info.cuda_available = torch.cuda.is_available()
            if info.cuda_available:
                info.cuda_version = torch.version.cuda
                info.gpu_name = torch.cuda.get_device_name(0)
                info.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except ImportError:
            pass

        # Check ROS 2
        try:
            result = subprocess.run(
                ["ros2", "--version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info.ros2_available = True
                info.ros2_distro = self._get_ros2_distro()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return info

    def _get_ros2_distro(self) -> str:
        """Get ROS 2 distribution name."""
        import os
        return os.environ.get("ROS_DISTRO", "unknown")

    def _detect_isaac(self) -> FrameworkInfo:
        """Detect NVIDIA Isaac Lab installation."""
        info = FrameworkInfo(
            name="NVIDIA Isaac Lab",
            available=False,
            recommended_use="GPU-accelerated robot training (4096+ parallel envs)"
        )

        # Check for isaaclab module
        try:
            import omni.isaac.lab as lab
            info.available = True
            info.version = getattr(lab, "__version__", "unknown")
            info.gpu_support = True
            info.features = [
                "GPU-parallel simulation",
                "Isaac Sim integration",
                "ORBIT-Surgical support",
                "USD scene format"
            ]
            info.path = str(Path(lab.__file__).parent)

        except ImportError:
            # Try alternative import
            try:
                import isaaclab
                info.available = True
                info.version = getattr(isaaclab, "__version__", "unknown")
                info.gpu_support = True
                info.features = ["Isaac Lab standalone mode"]
                info.path = str(Path(isaaclab.__file__).parent)
            except ImportError:
                info.warnings.append("Isaac Lab not installed")
                info.warnings.append("Install from: https://isaac-sim.github.io/IsaacLab/")

        # Check GPU requirements
        if info.available and self.system_info:
            if not self.system_info.cuda_available:
                info.warnings.append("CUDA not available - Isaac Lab requires NVIDIA GPU")
            elif self.system_info.gpu_memory_gb < 8:
                info.warnings.append(f"GPU memory ({self.system_info.gpu_memory_gb:.1f}GB) may be insufficient")

        return info

    def _detect_mujoco(self) -> FrameworkInfo:
        """Detect MuJoCo installation."""
        info = FrameworkInfo(
            name="MuJoCo",
            available=False,
            recommended_use="High-fidelity physics validation and MJX GPU training"
        )

        try:
            import mujoco
            info.available = True
            info.version = mujoco.__version__
            info.path = str(Path(mujoco.__file__).parent)
            info.features = [
                "Accurate contact dynamics",
                "MJCF model format",
                "Visualization",
                "Force/torque sensing"
            ]

            # Check for MJX (GPU)
            try:
                from mujoco import mjx
                info.gpu_support = True
                info.features.append("MJX GPU acceleration (JAX backend)")
            except ImportError:
                info.warnings.append("MJX not available - install with: pip install mujoco-mjx")

            # Check ROS 2 integration
            if self.system_info and self.system_info.ros2_available:
                info.ros2_integration = True
                info.features.append("ROS 2 bridge available")

        except ImportError:
            info.warnings.append("MuJoCo not installed")
            info.warnings.append("Install with: pip install mujoco")

        return info

    def _detect_gazebo(self) -> FrameworkInfo:
        """Detect Gazebo Ionic installation."""
        info = FrameworkInfo(
            name="Gazebo Ionic",
            available=False,
            recommended_use="ROS 2 integration and sensor simulation"
        )

        # Check for gz-sim command
        try:
            result = subprocess.run(
                ["gz", "sim", "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                info.available = True
                # Parse version from output
                version_line = result.stdout.strip()
                info.version = version_line.split()[-1] if version_line else "unknown"
                info.features = [
                    "ROS 2 native integration",
                    "SDF model format",
                    "Sensor simulation",
                    "Physics plugins"
                ]

                # ROS 2 integration
                if self.system_info and self.system_info.ros2_available:
                    info.ros2_integration = True
                    info.features.append("ros_gz bridge")

        except (FileNotFoundError, subprocess.TimeoutExpired):
            info.warnings.append("Gazebo not installed or not in PATH")
            info.warnings.append("Install with: sudo apt install ros-jazzy-ros-gz")

        return info

    def _detect_pybullet(self) -> FrameworkInfo:
        """Detect PyBullet installation."""
        info = FrameworkInfo(
            name="PyBullet",
            available=False,
            recommended_use="Rapid prototyping and algorithm development"
        )

        try:
            import pybullet as p
            info.available = True
            info.version = str(p.getAPIVersion())
            info.path = str(Path(p.__file__).parent)
            info.features = [
                "Pure Python API",
                "URDF support",
                "Soft body simulation",
                "Camera rendering",
                "Gymnasium interface"
            ]

            # Check for OpenGL rendering
            try:
                # Quick test of GUI capability
                info.features.append("OpenGL rendering")
            except Exception:
                info.warnings.append("OpenGL rendering may not be available")

        except ImportError:
            info.warnings.append("PyBullet not installed")
            info.warnings.append("Install with: pip install pybullet")

        return info

    def get_recommended_pipeline(self) -> Dict[str, str]:
        """Get recommended framework for each pipeline stage."""
        recommendations = {}

        # Training
        if self.frameworks.get("isaac", FrameworkInfo("", False)).available:
            recommendations["training"] = "isaac"
        elif self.frameworks.get("mujoco", FrameworkInfo("", False)).gpu_support:
            recommendations["training"] = "mujoco_mjx"
        elif self.frameworks.get("pybullet", FrameworkInfo("", False)).available:
            recommendations["training"] = "pybullet"
        else:
            recommendations["training"] = "none_available"

        # Validation
        if self.frameworks.get("mujoco", FrameworkInfo("", False)).available:
            recommendations["validation"] = "mujoco"
        elif self.frameworks.get("pybullet", FrameworkInfo("", False)).available:
            recommendations["validation"] = "pybullet"
        else:
            recommendations["validation"] = "none_available"

        # ROS 2 integration
        if self.frameworks.get("gazebo", FrameworkInfo("", False)).available:
            recommendations["ros2_integration"] = "gazebo"
        elif self.frameworks.get("mujoco", FrameworkInfo("", False)).ros2_integration:
            recommendations["ros2_integration"] = "mujoco"
        else:
            recommendations["ros2_integration"] = "none_available"

        # Rapid prototyping
        if self.frameworks.get("pybullet", FrameworkInfo("", False)).available:
            recommendations["prototyping"] = "pybullet"
        elif self.frameworks.get("mujoco", FrameworkInfo("", False)).available:
            recommendations["prototyping"] = "mujoco"
        else:
            recommendations["prototyping"] = "none_available"

        return recommendations

    def print_report(self) -> None:
        """Print detection report."""
        print("=" * 60)
        print("Physical AI Oncology Trials - Framework Detection Report")
        print("=" * 60)
        print()

        # System info
        if self.system_info:
            print("System Information:")
            print(f"  Python: {self.system_info.python_version}")
            print(f"  Platform: {self.system_info.platform}")
            if self.system_info.cuda_available:
                print(f"  CUDA: {self.system_info.cuda_version}")
                print(f"  GPU: {self.system_info.gpu_name}")
                print(f"  GPU Memory: {self.system_info.gpu_memory_gb:.1f} GB")
            else:
                print("  CUDA: Not available")
            if self.system_info.ros2_available:
                print(f"  ROS 2: {self.system_info.ros2_distro}")
            else:
                print("  ROS 2: Not available")
            print()

        # Frameworks
        print("Framework Detection:")
        print("-" * 40)
        for name, info in self.frameworks.items():
            status = "✓" if info.available else "✗"
            print(f"  {status} {info.name}")
            if info.available:
                print(f"      Version: {info.version}")
                if info.gpu_support:
                    print("      GPU: Supported")
                if info.ros2_integration:
                    print("      ROS 2: Integrated")
                for feature in info.features[:3]:  # Show first 3 features
                    print(f"      • {feature}")
            for warning in info.warnings:
                print(f"      ⚠ {warning}")
            print()

        # Recommendations
        print("Recommended Pipeline:")
        print("-" * 40)
        recommendations = self.get_recommended_pipeline()
        for stage, framework in recommendations.items():
            print(f"  {stage}: {framework}")
        print()

        # Summary
        available_count = sum(1 for f in self.frameworks.values() if f.available)
        print(f"Summary: {available_count}/{len(self.frameworks)} frameworks available")
        print("=" * 60)

    def to_json(self) -> str:
        """Export detection results as JSON."""
        data = {
            "system": asdict(self.system_info) if self.system_info else None,
            "frameworks": {
                name: asdict(info) for name, info in self.frameworks.items()
            },
            "recommendations": self.get_recommended_pipeline(),
        }
        return json.dumps(data, indent=2)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Detect available simulation frameworks for oncology robotics."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed detection progress"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    detector = FrameworkDetector(verbose=args.verbose)
    detector.detect_all()

    if args.json:
        print(detector.to_json())
    else:
        detector.print_report()

    # Exit code based on availability
    available_count = sum(1 for f in detector.frameworks.values() if f.available)
    sys.exit(0 if available_count > 0 else 1)


if __name__ == "__main__":
    main()
