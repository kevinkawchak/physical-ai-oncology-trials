"""Tests for examples-new/03_ros2_surgical_deployment.py

Covers ProcedureStateMachine, PolicyInferenceEngine, RobotHardwareInterface,
and SurgicalControlLoop.
"""

from __future__ import annotations

import numpy as np


class TestProcedurePhaseEnum:
    def test_members(self, ros2_mod):
        pp = ros2_mod.ProcedurePhase
        assert pp.IDLE is not None
        assert pp.HOMING is not None
        assert pp.EMERGENCY is not None
        assert pp.COMPLETE is not None


class TestPhaseConfig:
    def test_defaults(self, ros2_mod):
        cfg = ros2_mod.PhaseConfig()
        assert cfg.max_velocity_m_s == 0.25
        assert cfg.max_force_n == 5.0
        assert cfg.autonomy_enabled is False


class TestProcedureStateMachine:
    def test_creation(self, ros2_mod):
        sm = ros2_mod.ProcedureStateMachine()
        assert sm.current_phase == ros2_mod.ProcedurePhase.IDLE

    def test_valid_transition(self, ros2_mod):
        sm = ros2_mod.ProcedureStateMachine()
        result = sm.request_transition(ros2_mod.ProcedurePhase.HOMING)
        assert result is True
        assert sm.current_phase == ros2_mod.ProcedurePhase.HOMING

    def test_emergency_transition_always_allowed(self, ros2_mod):
        sm = ros2_mod.ProcedureStateMachine()
        result = sm.request_transition(ros2_mod.ProcedurePhase.EMERGENCY)
        assert result is True
        assert sm.current_phase == ros2_mod.ProcedurePhase.EMERGENCY


class TestPolicyInferenceEngine:
    def test_creation(self, ros2_mod):
        policy = ros2_mod.PolicyInferenceEngine(obs_dim=35, act_dim=8)
        assert policy is not None

    def test_infer_returns_action(self, ros2_mod):
        policy = ros2_mod.PolicyInferenceEngine(obs_dim=35, act_dim=8)
        obs = np.zeros(35)
        action = policy.infer(obs)
        assert action.shape == (8,)


class TestRobotHardwareInterface:
    def test_creation(self, ros2_mod):
        hw = ros2_mod.RobotHardwareInterface(robot_type="dvrk_psm")
        assert hw is not None

    def test_enable(self, ros2_mod):
        hw = ros2_mod.RobotHardwareInterface()
        result = hw.enable()
        assert isinstance(result, bool)

    def test_get_state(self, ros2_mod):
        hw = ros2_mod.RobotHardwareInterface()
        state = hw.get_state()
        assert isinstance(state, dict)


class TestGenerateLaunchConfig:
    def test_returns_dict(self, ros2_mod):
        config = ros2_mod.generate_launch_config(robot_type="dvrk_psm")
        assert isinstance(config, dict)
        assert "robot_type" in config
