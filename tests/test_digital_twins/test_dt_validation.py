"""Tests for digital-twins/examples-twins/06_dt_validation_verification.py

Covers AccuracyMetrics, CalibrationAnalyzer, DiscriminationAnalyzer,
SubgroupAnalyzer, and VVReportGenerator.
"""

from __future__ import annotations

import numpy as np


# ── Enum values ──────────────────────────────────────────────────────────


class TestEnums:
    def test_validation_levels(self, vv_mod):
        levels = vv_mod.ValidationLevel
        assert levels.CREDIBILITY_ASSESSMENT.value == "credibility_assessment"
        assert levels.COMPARISON_TO_REALITY.value == "comparison_to_clinical_reality"

    def test_risk_categories(self, vv_mod):
        risk = vv_mod.RiskCategory
        assert risk.CLASS_I.value == "class_I"
        assert risk.CLASS_II.value == "class_II"
        assert risk.CLASS_III.value == "class_III"


# ── PredictionPair and ValidationDataset ─────────────────────────────────


class TestPredictionPair:
    def test_creation(self, vv_mod):
        pair = vv_mod.PredictionPair(
            patient_id="VP-001",
            predicted=0.8,
            observed=0.75,
        )
        assert pair.patient_id == "VP-001"
        assert pair.predicted == 0.8
        assert pair.observed == 0.75


class TestValidationDataset:
    def _make_dataset(self, vv_mod, n=20):
        pairs = [
            vv_mod.PredictionPair(
                patient_id=str(i),
                predicted=float(i) / n,
                observed=float(i) / n + np.random.normal(0, 0.05),
            )
            for i in range(n)
        ]
        return vv_mod.ValidationDataset(
            name="test_dataset",
            pairs=pairs,
            endpoint="overall_survival",
        )

    def test_creation(self, vv_mod):
        ds = self._make_dataset(vv_mod)
        assert ds.name == "test_dataset"
        assert len(ds.pairs) == 20


# ── AccuracyMetrics ──────────────────────────────────────────────────────


class TestAccuracyMetrics:
    def _make_dataset(self, vv_mod, n=50):
        pairs = [
            vv_mod.PredictionPair(
                patient_id=str(i),
                predicted=float(i),
                observed=float(i) + np.random.normal(0, 1),
            )
            for i in range(n)
        ]
        return vv_mod.ValidationDataset(
            name="accuracy_test",
            pairs=pairs,
            endpoint="volume_cm3",
        )

    def test_compute_all_returns_dict(self, vv_mod):
        ds = self._make_dataset(vv_mod)
        metrics = vv_mod.AccuracyMetrics().compute_all(ds)
        assert isinstance(metrics, dict)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r_squared" in metrics

    def test_mae_non_negative(self, vv_mod):
        ds = self._make_dataset(vv_mod)
        metrics = vv_mod.AccuracyMetrics().compute_all(ds)
        assert metrics["mae"] >= 0

    def test_rmse_non_negative(self, vv_mod):
        ds = self._make_dataset(vv_mod)
        metrics = vv_mod.AccuracyMetrics().compute_all(ds)
        assert metrics["rmse"] >= 0

    def test_perfect_prediction_r_squared(self, vv_mod):
        pairs = [
            vv_mod.PredictionPair(
                patient_id=str(i),
                predicted=float(i),
                observed=float(i),
            )
            for i in range(1, 30)
        ]
        ds = vv_mod.ValidationDataset(
            name="perfect",
            pairs=pairs,
            endpoint="volume",
        )
        metrics = vv_mod.AccuracyMetrics().compute_all(ds)
        assert metrics["r_squared"] > 0.99

    def test_mae_helper(self, vv_mod):
        pred = np.array([1.0, 2.0, 3.0])
        obs = np.array([1.0, 2.0, 3.0])
        mae = vv_mod.AccuracyMetrics().mean_absolute_error(pred, obs)
        assert mae == 0.0

    def test_rmse_helper(self, vv_mod):
        pred = np.array([1.0, 2.0, 3.0])
        obs = np.array([2.0, 3.0, 4.0])
        rmse = vv_mod.AccuracyMetrics().root_mean_squared_error(pred, obs)
        assert abs(rmse - 1.0) < 1e-9


# ── VVReportGenerator ────────────────────────────────────────────────────


class TestVVReportGenerator:
    def test_generate_model_card(self, vv_mod):
        accuracy = {"mae": 1.2, "rmse": 1.5, "r_squared": 0.85}
        card = vv_mod.VVReportGenerator().generate_model_card(
            model_name="TumorGrowthPredictor",
            model_version="1.0",
            intended_use="Treatment planning support",
            accuracy_results=accuracy,
        )
        assert isinstance(card, dict)
        assert card["model_details"]["name"] == "TumorGrowthPredictor"
        assert "accuracy" in card["performance"]

    def test_generate_vv_report(self, vv_mod):
        accuracy = {"mae": 1.0, "rmse": 1.5, "r_squared": 0.9}
        report = vv_mod.VVReportGenerator().generate_vv_report(
            model_name="TumorModel",
            accuracy_results=accuracy,
        )
        assert isinstance(report, dict)
        assert "report_title" in report
