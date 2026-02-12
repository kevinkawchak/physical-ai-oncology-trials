"""Unit tests for unification/cross_platform_tools/framework_detector.py.

Tests cover FrameworkInfo, SystemInfo dataclasses, FrameworkDetector,
and FRAMEWORKS constant.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "unification/cross_platform_tools/framework_detector.py"


@pytest.fixture()
def mod():
    return load_module("framework_detector", MOD_PATH)


class TestConstants:
    def test_frameworks_list(self, mod):
        for fw in ("isaac", "mujoco", "gazebo", "pybullet"):
            assert fw in mod.FrameworkDetector.FRAMEWORKS


class TestFrameworkInfo:
    def test_creation(self, mod):
        info = mod.FrameworkInfo(
            name="mujoco",
            available=True,
            version="3.4.0",
            gpu_support=False,
        )
        assert info.name == "mujoco"
        assert info.available is True

    def test_unavailable_framework(self, mod):
        info = mod.FrameworkInfo(
            name="isaac",
            available=False,
            version=None,
            gpu_support=False,
        )
        assert info.available is False


class TestSystemInfo:
    def test_creation(self, mod):
        info = mod.SystemInfo(
            python_version="3.10.0",
            platform="linux",
            cuda_available=False,
        )
        assert info.python_version == "3.10.0"


class TestFrameworkDetector:
    def test_instantiation(self, mod):
        detector = mod.FrameworkDetector(verbose=False)
        assert detector is not None

    def test_detect_all(self, mod):
        detector = mod.FrameworkDetector(verbose=False)
        results = detector.detect_all()
        assert isinstance(results, dict)

    def test_system_info(self, mod):
        detector = mod.FrameworkDetector(verbose=False)
        sys_info = detector._get_system_info()
        assert sys_info.python_version is not None
        assert sys_info.platform is not None
