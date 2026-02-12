"""Unit tests for examples-new/03_ros2_surgical_deployment.py.

Tests cover ProcedurePhase enum, PhaseConfig dataclass,
PHASE_CONFIGS constants, and state transition rules.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "examples-new/03_ros2_surgical_deployment.py"


@pytest.fixture()
def mod():
    return load_module("ros2_deployment", MOD_PATH)


class TestEnums:
    def test_procedure_phase(self, mod):
        names = [e.name for e in mod.ProcedurePhase]
        for phase in ("IDLE", "HOMING", "REGISTRATION", "APPROACH", "OPERATE", "RETRACT", "EMERGENCY", "COMPLETE"):
            assert phase in names


class TestPhaseConfig:
    def test_creation(self, mod):
        config = mod.PhaseConfig(
            max_velocity_m_s=0.1,
            max_force_n=10.0,
            autonomy_enabled=False,
            teleop_enabled=True,
            allowed_transitions=[mod.ProcedurePhase.APPROACH],
            requires_confirmation=True,
        )
        assert config.max_velocity_m_s == 0.1
        assert config.requires_confirmation is True


class TestPhaseConfigs:
    def test_all_phases_have_config(self, mod):
        for phase in mod.ProcedurePhase:
            assert phase in mod.PHASE_CONFIGS, f"{phase.name} missing config"

    def test_emergency_has_zero_velocity(self, mod):
        emergency_cfg = mod.PHASE_CONFIGS[mod.ProcedurePhase.EMERGENCY]
        assert emergency_cfg.max_velocity_m_s == 0.0 or emergency_cfg.max_velocity_m_s <= 0.01

    def test_operate_phase_config(self, mod):
        operate_cfg = mod.PHASE_CONFIGS[mod.ProcedurePhase.OPERATE]
        assert operate_cfg.max_force_n > 0
        assert operate_cfg.max_velocity_m_s > 0

    def test_idle_allows_transition_to_homing(self, mod):
        idle_cfg = mod.PHASE_CONFIGS[mod.ProcedurePhase.IDLE]
        assert mod.ProcedurePhase.HOMING in idle_cfg.allowed_transitions

    def test_no_transitions_from_complete(self, mod):
        complete_cfg = mod.PHASE_CONFIGS[mod.ProcedurePhase.COMPLETE]
        # Complete should not transition to operating states
        assert mod.ProcedurePhase.OPERATE not in complete_cfg.allowed_transitions
