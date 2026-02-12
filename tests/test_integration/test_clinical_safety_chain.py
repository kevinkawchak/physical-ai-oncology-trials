"""Integration test: Clinical safety validation chain.

Validates safety constraint enforcement across the robot safety monitor,
dose calculator, and specimen handling modules.
"""

from __future__ import annotations

import math

import numpy as np

from tests.conftest import load_module


class TestClinicalSafetyChain:
    def test_dose_calculation_safety_bounds(self):
        dose_mod = load_module(
            "dose_calculator",
            "tools/dose-calculator/dose_calculator.py",
        )

        # Verify BED calculations produce safe values
        bed = dose_mod.calc_bed(60.0, 30, 10.0)
        assert math.isclose(bed, 72.0, rel_tol=1e-9)

        # TCP should be bounded [0, 1]
        tcp = dose_mod.calc_tcp_poisson(60.0, 30, 1e9, 0.3, 10.0)
        assert 0.0 <= tcp <= 1.0

        # NTCP should be bounded [0, 1]
        ntcp = dose_mod.calc_ntcp_lkb(60.0, 30, td50=50.0, m=0.18, n=0.12)
        assert 0.0 <= ntcp <= 1.0

    def test_robot_safety_with_force_limit(self):
        safety_mod = load_module(
            "safety_monitoring",
            "examples-new/01_realtime_safety_monitoring.py",
        )

        limits = safety_mod.SafetyLimits()
        monitor = safety_mod.SafetyMonitor(limits)

        # High force should trigger critical
        high_ft = safety_mod.ForceTorqueReading(
            force_xyz_n=np.array([10.0, 0.0, 0.0]),
            torque_xyz_nm=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        state = safety_mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.zeros(3),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=high_ft,
            timestamp_ns=1_000_000_000,
        )
        _, _, level = monitor.check_and_filter(
            np.array([0.0, 0.0, -0.15]),
            np.zeros(3),
            state,
        )
        assert level == safety_mod.SafetyLevel.CRITICAL
        assert monitor.get_status_summary()["is_stopped"] is True

    def test_specimen_cold_chain_safety(self):
        sample_mod = load_module(
            "robotic_sample_handling",
            "examples-new/06_robotic_sample_handling.py",
        )

        specimen = sample_mod.Specimen(
            specimen_id="SAFETY-001",
            barcode="BC-SAFE",
            trial_id="NCT-SAFE",
            patient_id="PT-SAFE",
            specimen_type=sample_mod.SpecimenType.TISSUE_BIOPSY,
            container_type=sample_mod.ContainerType.CRYOVIAL_2ML,
            collection_datetime="2026-03-01T09:00:00",
            storage_condition=sample_mod.StorageCondition.FROZEN_MINUS80,
            volume_ml=1.0,
        )

        monitor = sample_mod.ColdChainMonitor()
        # Correct temperature
        result = monitor.check_temperature(specimen, current_temp_c=-80.0)
        assert result["in_range"] is True

        # Temperature excursion
        result = monitor.check_temperature(specimen, current_temp_c=-20.0)
        assert result["in_range"] is False
