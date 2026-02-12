from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module("ros2_surgical_deployment", "examples-new/03_ros2_surgical_deployment.py")


# ---------------------------------------------------------------------------
# TestEnums
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for ProcedurePhase enum."""

    def test_idle_exists(self, mod):
        assert mod.ProcedurePhase.IDLE is not None

    def test_homing_exists(self, mod):
        assert mod.ProcedurePhase.HOMING is not None

    def test_registration_exists(self, mod):
        assert mod.ProcedurePhase.REGISTRATION is not None

    def test_approach_exists(self, mod):
        assert mod.ProcedurePhase.APPROACH is not None

    def test_operate_exists(self, mod):
        assert mod.ProcedurePhase.OPERATE is not None

    def test_retract_exists(self, mod):
        assert mod.ProcedurePhase.RETRACT is not None

    def test_emergency_exists(self, mod):
        assert mod.ProcedurePhase.EMERGENCY is not None

    def test_complete_exists(self, mod):
        assert mod.ProcedurePhase.COMPLETE is not None


# ---------------------------------------------------------------------------
# TestPhaseConfig
# ---------------------------------------------------------------------------


class TestPhaseConfig:
    """Tests for PhaseConfig dataclass."""

    def test_creation_defaults(self, mod):
        cfg = mod.PhaseConfig()
        assert cfg.max_velocity_m_s == 0.25
        assert cfg.max_force_n == 5.0

    def test_autonomy_default_false(self, mod):
        cfg = mod.PhaseConfig()
        assert cfg.autonomy_enabled is False

    def test_teleop_default_true(self, mod):
        cfg = mod.PhaseConfig()
        assert cfg.teleop_enabled is True

    def test_custom_values(self, mod):
        cfg = mod.PhaseConfig(max_velocity_m_s=0.1, max_force_n=2.0, autonomy_enabled=True)
        assert cfg.max_velocity_m_s == 0.1
        assert cfg.autonomy_enabled is True


# ---------------------------------------------------------------------------
# TestProcedureStateMachine
# ---------------------------------------------------------------------------


class TestProcedureStateMachine:
    """Tests for ProcedureStateMachine class."""

    def test_init_phase_is_idle(self, mod):
        sm = mod.ProcedureStateMachine()
        assert sm.current_phase == mod.ProcedurePhase.IDLE

    def test_valid_transition(self, mod):
        sm = mod.ProcedureStateMachine()
        success = sm.request_transition(mod.ProcedurePhase.HOMING)
        assert success is True
        assert sm.current_phase == mod.ProcedurePhase.HOMING

    def test_invalid_transition(self, mod):
        sm = mod.ProcedureStateMachine()
        success = sm.request_transition(mod.ProcedurePhase.OPERATE)
        assert success is False
        assert sm.current_phase == mod.ProcedurePhase.IDLE

    def test_emergency_always_allowed(self, mod):
        sm = mod.ProcedureStateMachine()
        success = sm.request_transition(mod.ProcedurePhase.EMERGENCY)
        assert success is True
        assert sm.current_phase == mod.ProcedurePhase.EMERGENCY

    def test_current_config_returns_phase_config(self, mod):
        sm = mod.ProcedureStateMachine()
        cfg = sm.current_config
        assert isinstance(cfg, mod.PhaseConfig)

    def test_transition_history_recorded(self, mod):
        sm = mod.ProcedureStateMachine()
        sm.request_transition(mod.ProcedurePhase.HOMING)
        assert len(sm._transition_history) == 1


# ---------------------------------------------------------------------------
# TestPolicyInferenceEngine
# ---------------------------------------------------------------------------


class TestPolicyInferenceEngine:
    """Tests for PolicyInferenceEngine class."""

    def test_init(self, mod):
        engine = mod.PolicyInferenceEngine(obs_dim=35, act_dim=8)
        assert engine.obs_dim == 35
        assert engine.act_dim == 8

    def test_infer_returns_array(self, mod):
        engine = mod.PolicyInferenceEngine(obs_dim=10, act_dim=4)
        obs = np.zeros(10, dtype=np.float32)
        action = engine.infer(obs)
        assert isinstance(action, np.ndarray)

    def test_infer_action_bounded(self, mod):
        engine = mod.PolicyInferenceEngine(obs_dim=10, act_dim=4)
        obs = np.random.randn(10).astype(np.float32) * 10
        action = engine.infer(obs)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)


# ---------------------------------------------------------------------------
# TestRobotHardwareInterface
# ---------------------------------------------------------------------------


class TestRobotHardwareInterface:
    """Tests for RobotHardwareInterface class."""

    def test_init(self, mod):
        hw = mod.RobotHardwareInterface(robot_type="dvrk_psm")
        assert hw.robot_type == "dvrk_psm"

    def test_enable(self, mod):
        hw = mod.RobotHardwareInterface()
        result = hw.enable()
        assert result is True

    def test_send_command_requires_enable(self, mod):
        hw = mod.RobotHardwareInterface()
        pos = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        result = hw.send_cartesian_command(pos, quat)
        assert result is False

    def test_send_command_after_enable(self, mod):
        hw = mod.RobotHardwareInterface()
        hw.enable()
        pos = np.array([0.01, 0.02, -0.15])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        result = hw.send_cartesian_command(pos, quat)
        assert result is True

    def test_get_state(self, mod):
        hw = mod.RobotHardwareInterface()
        state = hw.get_state()
        assert "position_m" in state
        assert "orientation_quat" in state
        assert "is_enabled" in state


# ---------------------------------------------------------------------------
# TestSurgicalControlLoop
# ---------------------------------------------------------------------------


class TestSurgicalControlLoop:
    """Tests for SurgicalControlLoop class."""

    def test_init(self, mod):
        hw = mod.RobotHardwareInterface()
        policy = mod.PolicyInferenceEngine(obs_dim=35, act_dim=8)
        sm = mod.ProcedureStateMachine()
        loop = mod.SurgicalControlLoop(robot=hw, policy=policy, state_machine=sm, control_rate_hz=100)
        assert loop.control_rate_hz == 100

    def test_step_runs_small_cycle(self, mod):
        hw = mod.RobotHardwareInterface()
        hw.enable()
        policy = mod.PolicyInferenceEngine(obs_dim=35, act_dim=8)
        sm = mod.ProcedureStateMachine()
        sm.request_transition(mod.ProcedurePhase.HOMING)
        loop = mod.SurgicalControlLoop(robot=hw, policy=policy, state_machine=sm, control_rate_hz=1000)
        summary = loop.run(max_cycles=5)
        assert summary["total_cycles"] == 5
        assert "max_jitter_us" in summary
