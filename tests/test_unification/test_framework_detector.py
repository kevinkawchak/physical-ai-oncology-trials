"""Tests for Framework Detector.

Validates FrameworkInfo and SystemInfo dataclasses, detector
initialisation, recommendation pipeline, and JSON export.

LICENSE: MIT
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
MOD_PATH = "unification/cross_platform_tools/framework_detector.py"


@pytest.fixture()
def fd_mod():
    return load_module("framework_detector", MOD_PATH)


@pytest.fixture()
def detector(fd_mod):
    return fd_mod.FrameworkDetector(verbose=False)


# ===========================================================================
# TestFrameworkInfo
# ===========================================================================


class TestFrameworkInfo:
    def test_defaults(self, fd_mod):
        fi = fd_mod.FrameworkInfo(name="TestFW", available=False)
        assert fi.version == "unknown"
        assert fi.gpu_support is False
        assert fi.ros2_integration is False
        assert fi.features == []
        assert fi.warnings == []

    def test_custom_values(self, fd_mod):
        fi = fd_mod.FrameworkInfo(
            name="MuJoCo",
            available=True,
            version="3.2.0",
            gpu_support=True,
            features=["Contact dynamics"],
        )
        assert fi.available is True
        assert fi.version == "3.2.0"
        assert len(fi.features) == 1


# ===========================================================================
# TestSystemInfo
# ===========================================================================


class TestSystemInfo:
    def test_defaults(self, fd_mod):
        si = fd_mod.SystemInfo(
            python_version="3.10.0",
            platform="Linux",
            cuda_available=False,
        )
        assert si.cuda_version == ""
        assert si.gpu_memory_gb == 0.0
        assert si.ros2_available is False

    def test_cuda_info(self, fd_mod):
        si = fd_mod.SystemInfo(
            python_version="3.10.0",
            platform="Linux",
            cuda_available=True,
            cuda_version="12.3",
            gpu_name="RTX 4090",
            gpu_memory_gb=24.0,
        )
        assert si.cuda_available is True
        assert si.gpu_memory_gb == 24.0


# ===========================================================================
# TestDetector
# ===========================================================================


class TestDetector:
    def test_initialization(self, detector):
        assert detector.verbose is False
        assert detector.frameworks == {}

    def test_detect_all_returns_dict(self, detector):
        result = detector.detect_all()
        assert isinstance(result, dict)
        # Should have 4 framework entries
        assert len(result) == 4
        assert "isaac" in result
        assert "mujoco" in result
        assert "gazebo" in result
        assert "pybullet" in result

    def test_each_framework_is_framework_info(self, detector, fd_mod):
        result = detector.detect_all()
        for name, info in result.items():
            assert isinstance(info, fd_mod.FrameworkInfo)

    def test_system_info_populated(self, detector):
        detector.detect_all()
        assert detector.system_info is not None
        assert detector.system_info.python_version != ""

    def test_framework_list(self, fd_mod):
        assert "isaac" in fd_mod.FrameworkDetector.FRAMEWORKS
        assert "mujoco" in fd_mod.FrameworkDetector.FRAMEWORKS


# ===========================================================================
# TestRecommendedPipeline
# ===========================================================================


class TestRecommendedPipeline:
    def test_recommendations_returned(self, detector):
        detector.detect_all()
        recs = detector.get_recommended_pipeline()
        assert isinstance(recs, dict)
        assert "training" in recs
        assert "validation" in recs
        assert "ros2_integration" in recs
        assert "prototyping" in recs

    def test_recommendations_are_strings(self, detector):
        detector.detect_all()
        recs = detector.get_recommended_pipeline()
        for stage, fw in recs.items():
            assert isinstance(fw, str)


# ===========================================================================
# TestJsonExport
# ===========================================================================


class TestJsonExport:
    def test_to_json(self, detector):
        detector.detect_all()
        j = detector.to_json()
        data = json.loads(j)
        assert "system" in data
        assert "frameworks" in data
        assert "recommendations" in data

    def test_json_frameworks_structure(self, detector):
        detector.detect_all()
        data = json.loads(detector.to_json())
        for fw_name, fw_info in data["frameworks"].items():
            assert "name" in fw_info
            assert "available" in fw_info
            assert "version" in fw_info


# ===========================================================================
# TestVerboseMode
# ===========================================================================


class TestVerboseMode:
    def test_verbose_construction(self, fd_mod):
        d = fd_mod.FrameworkDetector(verbose=True)
        assert d.verbose is True


# ===========================================================================
# TestDetectMethods
# ===========================================================================


class TestDetectMethods:
    def test_detect_isaac_returns_framework_info(self, detector, fd_mod):
        detector.system_info = fd_mod.SystemInfo(
            python_version="3.10.0",
            platform="Linux",
            cuda_available=False,
        )
        info = detector._detect_isaac()
        assert isinstance(info, fd_mod.FrameworkInfo)
        assert info.name == "NVIDIA Isaac Lab"

    def test_detect_mujoco_returns_framework_info(self, detector, fd_mod):
        detector.system_info = fd_mod.SystemInfo(
            python_version="3.10.0",
            platform="Linux",
            cuda_available=False,
        )
        info = detector._detect_mujoco()
        assert isinstance(info, fd_mod.FrameworkInfo)
        assert info.name == "MuJoCo"

    def test_detect_pybullet_returns_framework_info(self, detector, fd_mod):
        detector.system_info = fd_mod.SystemInfo(
            python_version="3.10.0",
            platform="Linux",
            cuda_available=False,
        )
        info = detector._detect_pybullet()
        assert isinstance(info, fd_mod.FrameworkInfo)
        assert info.name == "PyBullet"
