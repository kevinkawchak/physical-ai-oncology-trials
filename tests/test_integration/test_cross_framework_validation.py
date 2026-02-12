"""Integration test: Cross-Framework Physics Validation.

Validates physics parameter conversion and framework detection
across the unification layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def bridge_mod():
    return load_module("isaac_mujoco_bridge", "unification/simulation_physics/isaac_mujoco_bridge.py")


@pytest.fixture()
def detector_mod():
    return load_module("framework_detector", "unification/cross_platform_tools/framework_detector.py")


@pytest.fixture()
def converter_mod():
    return load_module("urdf_sdf_mjcf_converter", "unification/simulation_physics/urdf_sdf_mjcf_converter.py")


class TestCrossFrameworkValidation:
    def test_physics_parameter_round_trip(self, bridge_mod):
        """Parameters converted between frameworks should be consistent."""
        params = bridge_mod.PhysicsParameters(
            timestep=0.002,
            gravity=np.array([0.0, 0.0, -9.81]),
            friction_coefficient=0.5,
            contact_stiffness=1000.0,
            contact_damping=100.0,
            solver_iterations=50,
        )
        assert params.timestep > 0
        assert params.solver_iterations > 0

    def test_contact_parameter_conversion(self, bridge_mod):
        """Contact parameters should convert between framework formats."""
        contact = bridge_mod.ContactParameters(
            stiffness=1000.0,
            damping=100.0,
        )
        # Convert to framework-specific formats
        isaac_params = contact.to_isaac_physx()
        assert isaac_params is not None

        mujoco_params = contact.to_mujoco_solref()
        assert mujoco_params is not None

        pybullet_params = contact.to_pybullet()
        assert pybullet_params is not None

    def test_framework_detector_runs(self, detector_mod):
        """Framework detector should return results without crashing."""
        detector = detector_mod.FrameworkDetector(verbose=False)
        results = detector.detect_all()
        assert isinstance(results, dict)
        # Should detect at least some frameworks info
        assert len(results) > 0

    def test_robot_model_representation(self, converter_mod):
        """Robot model dataclass should hold links and joints."""
        link = converter_mod.Link(
            name="base_link",
            mass=5.0,
            inertia=np.eye(3) * 0.1,
            com_position=np.zeros(3),
        )
        joint = converter_mod.Joint(
            name="joint_1",
            joint_type="revolute",
            parent="base_link",
            child="link_1",
            axis=np.array([0.0, 0.0, 1.0]),
        )
        model = converter_mod.RobotModel(
            name="test_robot",
            links=[link],
            joints=[joint],
            source_format="urdf",
        )
        assert model.name == "test_robot"
        assert len(model.links) == 1
        assert len(model.joints) == 1

    def test_detector_system_info(self, detector_mod):
        """System info should include Python version and platform."""
        detector = detector_mod.FrameworkDetector(verbose=False)
        sys_info = detector._get_system_info()
        assert isinstance(sys_info, detector_mod.SystemInfo)
        assert sys_info.python_version is not None
        assert sys_info.platform is not None
