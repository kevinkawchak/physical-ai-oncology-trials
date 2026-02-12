"""Tests for NVIDIA Isaac / MuJoCo Bridge.

Validates physics parameter mapping, state conversion, bridge
initialisation, bidirectional sync support (v0.9.2 fix), and
conversion statistics tracking.

LICENSE: MIT
"""

from __future__ import annotations


import numpy as np
import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
MOD_PATH = "unification/simulation_physics/isaac_mujoco_bridge.py"


@pytest.fixture()
def bridge_mod():
    return load_module("isaac_mujoco_bridge", MOD_PATH)


@pytest.fixture()
def bridge(bridge_mod):
    return bridge_mod.IsaacMuJoCoBridge(config_path=None, validate_conversion=True)


# ===========================================================================
# TestPhysicsParameters
# ===========================================================================


class TestPhysicsParameters:
    def test_default_values(self, bridge_mod):
        p = bridge_mod.PhysicsParameters()
        assert p.timestep == 0.002
        assert p.gravity == (0.0, 0.0, -9.81)
        assert p.solver_iterations == 50

    def test_custom_values(self, bridge_mod):
        p = bridge_mod.PhysicsParameters(timestep=0.001, gravity=(0, 0, -10.0))
        assert p.timestep == 0.001
        assert p.gravity[2] == -10.0


# ===========================================================================
# TestContactParameters
# ===========================================================================


class TestContactParameters:
    def test_default_values(self, bridge_mod):
        cp = bridge_mod.ContactParameters()
        assert cp.stiffness == 1e5
        assert cp.damping == 1e3

    def test_to_isaac_physx(self, bridge_mod):
        cp = bridge_mod.ContactParameters()
        physx = cp.to_isaac_physx()
        assert "contact_offset" in physx
        assert "rest_offset" in physx
        assert physx["contact_offset"] == 0.001

    def test_to_mujoco_solref(self, bridge_mod):
        cp = bridge_mod.ContactParameters(stiffness=1e5, damping=1e3)
        solref = cp.to_mujoco_solref()
        assert len(solref) == 2
        timeconst, dampratio = solref
        assert timeconst > 0
        assert dampratio > 0
        # Verify formula: timeconst = 1/sqrt(stiffness)
        expected_tc = 1.0 / np.sqrt(1e5)
        assert abs(timeconst - expected_tc) < 1e-10

    def test_to_pybullet(self, bridge_mod):
        cp = bridge_mod.ContactParameters(stiffness=5e4, damping=500)
        pb = cp.to_pybullet()
        assert pb["contactStiffness"] == 5e4
        assert pb["contactDamping"] == 500


# ===========================================================================
# TestPhysicsParameterMapper
# ===========================================================================


class TestPhysicsParameterMapper:
    def test_same_framework_passthrough(self, bridge_mod):
        p = bridge_mod.PhysicsParameters(timestep=0.005)
        result = bridge_mod.PhysicsParameterMapper.convert_parameters(p, "isaac", "isaac")
        assert result["timestep"] == 0.005

    def test_isaac_to_mujoco(self, bridge_mod):
        p = bridge_mod.PhysicsParameters()
        result = bridge_mod.PhysicsParameterMapper.convert_parameters(p, "isaac", "mujoco")
        assert "solref" in result
        assert "timestep" in result
        assert result["timestep"] == p.timestep

    def test_isaac_to_pybullet(self, bridge_mod):
        p = bridge_mod.PhysicsParameters()
        result = bridge_mod.PhysicsParameterMapper.convert_parameters(p, "isaac", "pybullet")
        assert "contactStiffness" in result
        assert "contactDamping" in result

    def test_parameter_mapping_dict_exists(self, bridge_mod):
        mapping = bridge_mod.PhysicsParameterMapper.PARAMETER_MAPPING
        assert "timestep" in mapping
        assert "gravity" in mapping
        assert "friction" in mapping


# ===========================================================================
# TestStateConverter
# ===========================================================================


class TestStateConverter:
    def test_isaac_to_mujoco(self, bridge_mod):
        state = {
            "joint_pos": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            "joint_vel": np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]),
        }
        qpos, qvel = bridge_mod.StateConverter.isaac_to_mujoco(state)
        np.testing.assert_array_almost_equal(qpos, state["joint_pos"])
        np.testing.assert_array_almost_equal(qvel, state["joint_vel"])

    def test_mujoco_to_isaac(self, bridge_mod):
        qpos = np.array([0.1, 0.2, 0.3])
        qvel = np.array([0.01, 0.02, 0.03])
        result = bridge_mod.StateConverter.mujoco_to_isaac(qpos, qvel)
        assert "joint_pos" in result
        assert "joint_vel" in result
        np.testing.assert_array_almost_equal(result["joint_pos"], qpos)

    def test_isaac_to_pybullet(self, bridge_mod):
        state = {
            "joint_pos": np.array([0.1, 0.2]),
            "joint_vel": np.array([0.01, 0.02]),
        }
        result = bridge_mod.StateConverter.isaac_to_pybullet(state)
        assert "joint_positions" in result
        assert "joint_velocities" in result

    def test_roundtrip_isaac_mujoco(self, bridge_mod):
        """State roundtrip: Isaac -> MuJoCo -> Isaac should preserve values."""
        original = {
            "joint_pos": np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.6, 0.7]),
            "joint_vel": np.array([0.01, 0.02, -0.03, 0.04, 0.05, 0.06, -0.07]),
        }
        qpos, qvel = bridge_mod.StateConverter.isaac_to_mujoco(original)
        recovered = bridge_mod.StateConverter.mujoco_to_isaac(qpos, qvel)
        np.testing.assert_array_almost_equal(recovered["joint_pos"], original["joint_pos"])
        np.testing.assert_array_almost_equal(recovered["joint_vel"], original["joint_vel"])


# ===========================================================================
# TestIsaacMuJoCoBridge
# ===========================================================================


class TestIsaacMuJoCoBridge:
    def test_initialization(self, bridge):
        assert bridge.validate is True
        assert bridge.conversion_stats["environments_converted"] == 0
        assert bridge.conversion_stats["states_synced"] == 0

    def test_config_defaults(self, bridge):
        assert "tolerance" in bridge.config
        assert "position" in bridge.config["tolerance"]

    def test_get_conversion_stats_returns_copy(self, bridge):
        stats1 = bridge.get_conversion_stats()
        stats2 = bridge.get_conversion_stats()
        assert stats1 is not stats2

    def test_validate_conversion_returns_errors(self, bridge):
        errors = bridge.validate_conversion(None, None, num_steps=10)
        assert "position_mae" in errors
        assert "velocity_mae" in errors
        assert "force_mae" in errors

    def test_unsupported_conversion_raises(self, bridge):
        with pytest.raises(NotImplementedError):
            bridge.convert_environment(None, source_framework="gazebo", target_framework="pybullet")

    def test_v092_bidirectional_sync_direction(self, bridge_mod):
        """v0.9.2 fix: sync_state supports both isaac->mujoco and mujoco->isaac."""
        bridge = bridge_mod.IsaacMuJoCoBridge()
        # The method should handle source_framework='mujoco' without error
        # (it internally calls mujoco_to_isaac path)
        # We cannot fully test without real envs but we verify the code path exists
        # by checking the sync_state accepts mujoco as source
        assert hasattr(bridge, "sync_state")
        # Verify the method signature accepts source_framework parameter
        import inspect

        sig = inspect.signature(bridge.sync_state)
        assert "source_framework" in sig.parameters
        assert "target_framework" in sig.parameters


# ===========================================================================
# TestPolicyTransferValidator
# ===========================================================================


class TestPolicyTransferValidator:
    def test_construction(self, bridge_mod, bridge):
        ptv = bridge_mod.PolicyTransferValidator(bridge)
        assert ptv.bridge is bridge

    def test_evaluate_policy_returns_array(self, bridge_mod, bridge):
        ptv = bridge_mod.PolicyTransferValidator(bridge)
        rewards = ptv._evaluate_policy("dummy.onnx", None, num_episodes=5)
        assert isinstance(rewards, np.ndarray)
        assert len(rewards) == 5
