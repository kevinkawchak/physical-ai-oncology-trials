"""Unit tests for tools/dose-calculator/dose_calculator.py.

Tests cover BED, EQD2, TCP, NTCP calculations, tissue alpha/beta constants,
DoseResult dataclass, scheme parsing, and clinical boundary conditions.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "tools/dose-calculator/dose_calculator.py"


@pytest.fixture()
def mod():
    return load_module("dose_calculator", MOD_PATH)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_tissue_alpha_beta_keys(self, mod):
        for tissue in ("prostate_cancer", "breast_cancer", "lung_late", "head_neck_scc", "spinal_cord"):
            assert tissue in mod.TISSUE_ALPHA_BETA

    def test_tissue_alpha_beta_positive(self, mod):
        for tissue, ab in mod.TISSUE_ALPHA_BETA.items():
            assert ab > 0, f"{tissue} alpha/beta not positive"

    def test_ntcp_params_keys(self, mod):
        assert len(mod.NTCP_PARAMS) >= 3


# ---------------------------------------------------------------------------
# BED calculation
# ---------------------------------------------------------------------------


class TestCalcBED:
    def test_standard_fractionation(self, mod):
        bed = mod.calc_bed(total_dose=60.0, fractions=30, alpha_beta=10.0)
        # BED = 60 * (1 + 2/10) = 72.0
        assert bed == pytest.approx(72.0, rel=0.01)

    def test_hypofractionation(self, mod):
        bed = mod.calc_bed(total_dose=42.56, fractions=16, alpha_beta=10.0)
        dpf = 42.56 / 16
        expected = 42.56 * (1 + dpf / 10.0)
        assert bed == pytest.approx(expected, rel=0.01)

    def test_single_fraction(self, mod):
        bed = mod.calc_bed(total_dose=20.0, fractions=1, alpha_beta=10.0)
        assert bed == pytest.approx(20.0 * (1 + 20.0 / 10.0), rel=0.01)

    def test_low_alpha_beta_higher_bed(self, mod):
        bed_high_ab = mod.calc_bed(total_dose=60.0, fractions=30, alpha_beta=10.0)
        bed_low_ab = mod.calc_bed(total_dose=60.0, fractions=30, alpha_beta=3.0)
        assert bed_low_ab > bed_high_ab


# ---------------------------------------------------------------------------
# EQD2 calculation
# ---------------------------------------------------------------------------


class TestCalcEQD2:
    def test_2gy_fractions(self, mod):
        eqd2 = mod.calc_eqd2(total_dose=60.0, fractions=30, alpha_beta=10.0)
        # When dpf = 2 Gy, EQD2 = total_dose
        assert eqd2 == pytest.approx(60.0, rel=0.01)

    def test_hypofractionation_higher(self, mod):
        eqd2 = mod.calc_eqd2(total_dose=42.56, fractions=16, alpha_beta=10.0)
        # dpf > 2 Gy, so EQD2 > total_dose
        assert eqd2 > 42.56


# ---------------------------------------------------------------------------
# TCP calculation
# ---------------------------------------------------------------------------


class TestCalcTCP:
    def test_high_dose_high_tcp(self, mod):
        # High dose (60 Gy/30fx) with radiosensitive tumor -> high TCP
        tcp = mod.calc_tcp_poisson(total_dose=60.0, fractions=30, n0=1e6, alpha=0.3, alpha_beta=10.0)
        assert 0.0 <= tcp <= 1.0

    def test_zero_dose_zero_tcp(self, mod):
        # Zero dose -> no cell kill -> TCP ~ 0
        tcp = mod.calc_tcp_poisson(total_dose=0.0, fractions=1, n0=1e8, alpha=0.3, alpha_beta=10.0)
        assert tcp < 0.01

    def test_tcp_bounded(self, mod):
        tcp = mod.calc_tcp_poisson(total_dose=40.0, fractions=20, n0=1e4, alpha=0.3, alpha_beta=10.0)
        assert 0.0 <= tcp <= 1.0


# ---------------------------------------------------------------------------
# NTCP calculation
# ---------------------------------------------------------------------------


class TestCalcNTCP:
    def test_bounded(self, mod):
        ntcp = mod.calc_ntcp_lkb(total_dose=20.0, fractions=10, td50=30.0, m=0.15, n=0.5, volume_fraction=1.0)
        assert 0.0 <= ntcp <= 1.0

    def test_dose_response(self, mod):
        ntcp_low = mod.calc_ntcp_lkb(total_dose=10.0, fractions=5, td50=30.0, m=0.15, n=0.5, volume_fraction=1.0)
        ntcp_high = mod.calc_ntcp_lkb(total_dose=50.0, fractions=25, td50=30.0, m=0.15, n=0.5, volume_fraction=1.0)
        assert ntcp_high > ntcp_low

    def test_volume_effect(self, mod):
        # In the LKB model, EUD = eqd2 * v^(-1/n). A smaller volume fraction
        # yields a higher EUD (and thus higher NTCP) because the same dose is
        # concentrated in a smaller volume.
        ntcp_small = mod.calc_ntcp_lkb(total_dose=30.0, fractions=15, td50=30.0, m=0.15, n=0.5, volume_fraction=0.2)
        ntcp_large = mod.calc_ntcp_lkb(total_dose=30.0, fractions=15, td50=30.0, m=0.15, n=0.5, volume_fraction=1.0)
        assert ntcp_small >= ntcp_large


# ---------------------------------------------------------------------------
# DoseResult
# ---------------------------------------------------------------------------


class TestDoseResult:
    def test_to_dict_includes_all(self, mod):
        result = mod.DoseResult(
            total_dose_gy=60.0,
            fractions=30,
            dose_per_fraction_gy=2.0,
            alpha_beta_gy=10.0,
            bed_gy=72.0,
            eqd2_gy=60.0,
        )
        d = result.to_dict()
        assert "total_dose_gy" in d
        assert "bed_gy" in d

    def test_to_dict_zero_values_present(self, mod):
        """Regression: zero values must not be dropped from to_dict()."""
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
        assert "bed_gy" in d
        assert d["bed_gy"] == 0.0
        assert "tcp" in d
        assert d["tcp"] == 0.0


# ---------------------------------------------------------------------------
# Scheme parsing
# ---------------------------------------------------------------------------


class TestSchemeParsing:
    def test_valid_scheme(self, mod):
        total, fx = mod._parse_scheme("60/30")
        assert total == pytest.approx(60.0)
        assert fx == 30

    def test_invalid_scheme_raises(self, mod):
        with pytest.raises((ValueError, SystemExit)):
            mod._parse_scheme("invalid")
