"""Unit tests for digital-twins/examples-twins/06_dt_validation_verification.py.

Tests cover ValidationLevel, RiskCategory enums, PredictionPair,
ValidationDataset dataclasses, AccuracyMetrics, and report generation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

MOD_PATH = "digital-twins/examples-twins/06_dt_validation_verification.py"


@pytest.fixture()
def mod():
    return load_module("dt_validation", MOD_PATH)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_validation_level(self, mod):
        names = [e.name for e in mod.ValidationLevel]
        assert "CREDIBILITY_ASSESSMENT" in names
        assert "COMPARISON_TO_REALITY" in names

    def test_risk_category(self, mod):
        names = [e.name for e in mod.RiskCategory]
        for cat in ("CLASS_I", "CLASS_II", "CLASS_III"):
            assert cat in names


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestPredictionPair:
    def test_creation(self, mod):
        pp = mod.PredictionPair(
            patient_id="PT-001",
            predicted=12.5,
            observed=13.0,
        )
        assert pp.patient_id == "PT-001"
        assert pp.predicted == 12.5
        assert pp.observed == 13.0


class TestValidationDataset:
    def test_creation(self, mod):
        pairs = [
            mod.PredictionPair(patient_id=f"PT-{i}", predicted=float(i), observed=float(i) + 0.5) for i in range(10)
        ]
        ds = mod.ValidationDataset(
            name="Test Dataset",
            pairs=pairs,
            endpoint="tumor_volume",
        )
        assert len(ds.pairs) == 10
        assert ds.name == "Test Dataset"


# ---------------------------------------------------------------------------
# AccuracyMetrics tests
# ---------------------------------------------------------------------------


class TestAccuracyMetrics:
    def test_mae(self, mod):
        metrics = mod.AccuracyMetrics()
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        observed = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        mae = metrics.mean_absolute_error(predicted, observed)
        assert mae >= 0
        assert mae < 1.0

    def test_rmse(self, mod):
        metrics = mod.AccuracyMetrics()
        predicted = np.array([1.0, 2.0, 3.0])
        observed = np.array([1.0, 2.0, 3.0])
        rmse = metrics.root_mean_squared_error(predicted, observed)
        assert rmse == pytest.approx(0.0)

    def test_r_squared_perfect(self, mod):
        metrics = mod.AccuracyMetrics()
        predicted = np.array([1.0, 2.0, 3.0, 4.0])
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        r2 = metrics.r_squared(predicted, observed)
        assert r2 == pytest.approx(1.0)

    def test_r_squared_range(self, mod):
        metrics = mod.AccuracyMetrics()
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        observed = np.array([1.5, 1.8, 3.2, 3.9, 5.5])
        r2 = metrics.r_squared(predicted, observed)
        assert r2 <= 1.0

    def test_mape(self, mod):
        metrics = mod.AccuracyMetrics()
        predicted = np.array([10.0, 20.0, 30.0])
        observed = np.array([11.0, 22.0, 27.0])
        mape = metrics.mean_absolute_percentage_error(predicted, observed)
        assert mape >= 0

    def test_concordance_index(self, mod):
        metrics = mod.AccuracyMetrics()
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        observed = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        ci = metrics.concordance_index(predicted, observed)
        assert 0.0 <= ci <= 1.0
        # Perfect rank ordering
        assert ci > 0.9

    def test_compute_all(self, mod):
        metrics = mod.AccuracyMetrics()
        pairs = [
            mod.PredictionPair(patient_id=f"PT-{i}", predicted=float(i), observed=float(i) + 0.1) for i in range(1, 11)
        ]
        ds = mod.ValidationDataset(name="test", pairs=pairs, endpoint="volume")
        results = metrics.compute_all(ds)
        assert "mae" in results
        assert "rmse" in results
        assert "r_squared" in results
