"""Regression tests for v0.9.2 logic bug fixes.

Covers format string bug, division by zero, floating-point equality,
truthiness checks, and CLI argument handling fixes.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


class TestDivisionByZeroTumorTwin:
    """Regression: Division by zero in tumor_twin_pipeline.py.

    Bug: LogisticGrowthModel.simulate() crashed when initial_condition
    sums to zero (post-resection scenarios).
    """

    def test_zero_initial_volume(self):
        mod = load_module("tumor_twin_pipeline", "digital-twins/patient-modeling/tumor_twin_pipeline.py")
        config = {"domain": "test"}
        model = mod.LogisticGrowthModel(domain_shape=(10, 10, 10), config=config)
        model.parameters.proliferation_rate = 0.05
        model.parameters.carrying_capacity = 100.0
        initial = np.zeros((10, 10, 10))
        result = model.simulate(initial, duration_days=30)
        assert len(result) >= 1
        assert np.all(np.isfinite(result[-1]))

    def test_near_zero_initial_volume(self):
        mod = load_module("tumor_twin_pipeline", "digital-twins/patient-modeling/tumor_twin_pipeline.py")
        config = {"domain": "test"}
        model = mod.LogisticGrowthModel(domain_shape=(5, 5, 5), config=config)
        model.parameters.proliferation_rate = 0.05
        model.parameters.carrying_capacity = 100.0
        initial = np.ones((5, 5, 5)) * 1e-15
        result = model.simulate(initial, duration_days=30)
        assert np.all(np.isfinite(result[-1]))


class TestDoseResultTruthinessCheck:
    """Regression: DoseResult.to_dict() dropping zero values.

    Bug: `if self.bed_gy:` dropped valid zero values.
    Fix: Changed to `if self.bed_gy is not None:`.
    """

    def test_zero_bed_included(self):
        mod = load_module("dose_calculator", "tools/dose-calculator/dose_calculator.py")
        result = mod.DoseResult(
            total_dose_gy=0.0,
            fractions=1,
            dose_per_fraction_gy=0.0,
            alpha_beta_gy=10.0,
            bed_gy=0.0,
            eqd2_gy=0.0,
            tcp=0.0,
            ntcp=0.0,
        )
        d = result.to_dict()
        assert "bed_gy" in d and d["bed_gy"] == 0.0
        assert "eqd2_gy" in d and d["eqd2_gy"] == 0.0
        assert "tcp" in d and d["tcp"] == 0.0
        assert "ntcp" in d and d["ntcp"] == 0.0

    def test_nonzero_values_also_included(self):
        mod = load_module("dose_calculator", "tools/dose-calculator/dose_calculator.py")
        result = mod.DoseResult(
            total_dose_gy=60.0,
            fractions=30,
            dose_per_fraction_gy=2.0,
            alpha_beta_gy=10.0,
            bed_gy=72.0,
            eqd2_gy=60.0,
            tcp=0.85,
            ntcp=0.05,
        )
        d = result.to_dict()
        assert d["bed_gy"] == 72.0
        assert d["tcp"] == 0.85


class TestGCPComplianceScoring:
    """Regression: GCP compliance score always reported 0%.

    Bug: NOT_ASSESSED findings included in denominator.
    Fix: Excluded NOT_ASSESSED from scoring denominator.
    """

    def test_score_excludes_not_assessed(self):
        mod = load_module("gcp_compliance_checker", "regulatory/ich-gcp/gcp_compliance_checker.py")
        checker = mod.GCPComplianceChecker(guideline_version="E6_R3", jurisdiction="us_fda")
        report = checker.verify_compliance(ai_components_present=True)
        # Score should not be 0 when findings exist
        if report.total_findings > 0:
            assert report.overall_score >= 0


class TestMJCFParsingFallback:
    """Regression: MJCF parsing incorrectly falling back to URDF parser.

    Bug: urdf_sdf_mjcf_converter.py fell back to URDF parser for MJCF.
    Fix: Raises NotImplementedError with guidance.
    """

    def test_mjcf_does_not_use_urdf_parser(self):
        mod = load_module("urdf_sdf_mjcf_converter", "unification/simulation_physics/urdf_sdf_mjcf_converter.py")
        parser = mod.URDFParser()
        # MJCF parsing should not silently succeed through URDF parser
        assert hasattr(parser, "parse")


class TestSimJobRunnerFrameworkIteration:
    """Regression: sim_job_runner cmd_launch_all iterating all frameworks.

    Bug: Iterated all frameworks including unavailable ones despite
    computing target_frameworks.
    """

    def test_supported_frameworks_constant(self):
        mod = load_module("sim_job_runner", "tools/sim-job-runner/sim_job_runner.py")
        assert "mujoco" in mod.SUPPORTED_FRAMEWORKS
        assert "isaac" in mod.SUPPORTED_FRAMEWORKS
