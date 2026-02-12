"""Tests for examples/03_cross_framework_validation.py

Covers SimulationFramework enum, validation config, and validator.
"""

from __future__ import annotations


class TestEnums:
    def test_simulation_frameworks(self, cross_framework_mod):
        sf = cross_framework_mod.SimulationFramework
        assert sf.ISAAC_LAB.value == "isaac_lab"
        assert sf.MUJOCO.value == "mujoco"
        assert sf.PYBULLET.value == "pybullet"


class TestValidationConfig:
    def test_creation(self, cross_framework_mod):
        cfg = cross_framework_mod.ValidationConfig(policy_path="test_policy.onnx")
        assert cfg is not None


class TestCrossFrameworkValidator:
    def test_creation(self, cross_framework_mod):
        cfg = cross_framework_mod.ValidationConfig(policy_path="test_policy.onnx")
        validator = cross_framework_mod.CrossFrameworkValidator(config=cfg)
        assert validator is not None
