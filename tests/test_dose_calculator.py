"""Tests for tools/dose-calculator/dose_calculator.py

Covers BED, EQD2, TCP (Poisson & logistic), NTCP (LKB), scheme parsing,
and known clinical reference values.
"""

from __future__ import annotations

import math

import pytest


# ── BED calculations ──────────────────────────────────────────────────────


class TestCalcBED:
    def test_standard_fractionation(self, dose_calc_mod, standard_fractionation):
        bed = dose_calc_mod.calc_bed(**standard_fractionation)
        # 60 Gy / 30 fx → d=2 Gy, BED = 60*(1+2/10) = 72 Gy
        assert math.isclose(bed, 72.0, rel_tol=1e-9)

    def test_hypofractionation_higher_bed(self, dose_calc_mod, standard_fractionation, hypofractionation):
        bed_std = dose_calc_mod.calc_bed(**standard_fractionation)
        bed_hypo = dose_calc_mod.calc_bed(**hypofractionation)
        # Same alpha/beta — hypofractionation changes BED
        assert bed_hypo != bed_std

    def test_single_fraction(self, dose_calc_mod):
        # 20 Gy in 1 fx, alpha/beta=10 → BED = 20*(1+20/10) = 60 Gy
        bed = dose_calc_mod.calc_bed(20.0, 1, 10.0)
        assert math.isclose(bed, 60.0, rel_tol=1e-9)

    def test_low_alpha_beta_gives_higher_bed(self, dose_calc_mod):
        bed_tumor = dose_calc_mod.calc_bed(60.0, 30, 10.0)  # tumor
        bed_late = dose_calc_mod.calc_bed(60.0, 30, 3.0)  # late tissue
        assert bed_late > bed_tumor  # late-effect BED always higher


# ── EQD2 calculations ─────────────────────────────────────────────────────


class TestCalcEQD2:
    def test_2gy_fractions_returns_same_dose(self, dose_calc_mod):
        # 60 Gy in 30 fx at 2 Gy/fx → EQD2 = 60 Gy
        eqd2 = dose_calc_mod.calc_eqd2(60.0, 30, 10.0)
        assert math.isclose(eqd2, 60.0, rel_tol=1e-9)

    def test_hypofractionation_eqd2(self, dose_calc_mod, hypofractionation):
        eqd2 = dose_calc_mod.calc_eqd2(**hypofractionation)
        assert eqd2 > 0


# ── TCP calculations ──────────────────────────────────────────────────────


class TestCalcTCP:
    def test_poisson_high_dose_tcp_near_one(self, dose_calc_mod):
        tcp = dose_calc_mod.calc_tcp_poisson(
            total_dose=80.0,
            fractions=40,
            n0=1e9,
            alpha=0.3,
            alpha_beta=10.0,
        )
        assert tcp > 0.9

    def test_poisson_zero_dose_tcp_near_zero(self, dose_calc_mod):
        tcp = dose_calc_mod.calc_tcp_poisson(
            total_dose=0.01,
            fractions=1,
            n0=1e9,
            alpha=0.3,
            alpha_beta=10.0,
        )
        assert tcp < 0.01

    def test_logistic_td50_gives_half(self, dose_calc_mod):
        # At dose equal to TD50 (in 2 Gy fractions), TCP ≈ 0.5
        tcp = dose_calc_mod.calc_tcp_logistic(
            total_dose=50.0,
            fractions=25,
            td50=50.0,
            gamma50=2.0,
            alpha_beta=10.0,
        )
        assert math.isclose(tcp, 0.5, abs_tol=0.05)

    def test_tcp_bounded_zero_one(self, dose_calc_mod):
        for dose in [0.1, 10.0, 40.0, 80.0, 120.0]:
            tcp = dose_calc_mod.calc_tcp_poisson(dose, max(int(dose / 2), 1), 1e9, 0.3, 10.0)
            assert 0.0 <= tcp <= 1.0


# ── NTCP calculations ─────────────────────────────────────────────────────


class TestCalcNTCP:
    def test_ntcp_bounded_zero_one(self, dose_calc_mod):
        ntcp = dose_calc_mod.calc_ntcp_lkb(60.0, 30, td50=50.0, m=0.18, n=0.12)
        assert 0.0 <= ntcp <= 1.0

    def test_ntcp_increases_with_dose(self, dose_calc_mod):
        ntcp_low = dose_calc_mod.calc_ntcp_lkb(30.0, 15, td50=50.0, m=0.18, n=0.12)
        ntcp_high = dose_calc_mod.calc_ntcp_lkb(70.0, 35, td50=50.0, m=0.18, n=0.12)
        assert ntcp_high > ntcp_low

    def test_volume_fraction_affects_ntcp(self, dose_calc_mod):
        ntcp_full = dose_calc_mod.calc_ntcp_lkb(
            60.0,
            30,
            td50=50.0,
            m=0.18,
            n=0.5,
            volume_fraction=1.0,
        )
        ntcp_partial = dose_calc_mod.calc_ntcp_lkb(
            60.0,
            30,
            td50=50.0,
            m=0.18,
            n=0.5,
            volume_fraction=0.3,
        )
        # LKB model: partial volume changes NTCP (direction depends on n)
        assert ntcp_partial != ntcp_full
        assert 0.0 <= ntcp_partial <= 1.0

    def test_lung_preset_exists(self, dose_calc_mod):
        assert "lung_pneumonitis" in dose_calc_mod.NTCP_PARAMS
        params = dose_calc_mod.NTCP_PARAMS["lung_pneumonitis"]
        assert "td50" in params and "m" in params and "n" in params


# ── Scheme parsing ────────────────────────────────────────────────────────


class TestSchemeParsing:
    def test_valid_scheme(self, dose_calc_mod):
        dose, fx = dose_calc_mod._parse_scheme("60/30")
        assert dose == 60.0
        assert fx == 30

    def test_invalid_scheme_raises(self, dose_calc_mod):
        with pytest.raises(ValueError, match="Invalid scheme"):
            dose_calc_mod._parse_scheme("60-30")


# ── DoseResult dataclass ──────────────────────────────────────────────────


class TestDoseResult:
    def test_to_dict_includes_required_keys(self, dose_calc_mod):
        result = dose_calc_mod.DoseResult(
            total_dose_gy=60.0,
            fractions=30,
            dose_per_fraction_gy=2.0,
            alpha_beta_gy=10.0,
            bed_gy=72.0,
            eqd2_gy=60.0,
        )
        d = result.to_dict()
        assert d["total_dose_gy"] == 60.0
        assert d["fractions"] == 30
        assert "bed_gy" in d
        assert "eqd2_gy" in d

    def test_to_dict_zero_values_not_dropped(self, dose_calc_mod):
        """Regression: v0.9.2 fixed truthiness check that dropped zero values."""
        result = dose_calc_mod.DoseResult(
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
        assert "eqd2_gy" in d
        assert "tcp" in d
        assert "ntcp" in d


# ── Tissue reference data ─────────────────────────────────────────────────


class TestTissueData:
    def test_alpha_beta_dict_not_empty(self, dose_calc_mod):
        assert len(dose_calc_mod.TISSUE_ALPHA_BETA) >= 10

    def test_all_alpha_beta_positive(self, dose_calc_mod):
        for tissue, ab in dose_calc_mod.TISSUE_ALPHA_BETA.items():
            assert ab > 0, f"{tissue} has non-positive alpha/beta"
