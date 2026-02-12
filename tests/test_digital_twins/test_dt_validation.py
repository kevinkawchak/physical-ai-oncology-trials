"""Tests for digital-twins/examples-twins/06_dt_validation_verification.py."""

from __future__ import annotations


import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module(
        "dt_validation",
        "digital-twins/examples-twins/06_dt_validation_verification.py",
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestValidationLevel:
    def test_credibility(self, mod):
        assert mod.ValidationLevel.CREDIBILITY_ASSESSMENT.value == "credibility_assessment"

    def test_comparison_to_test(self, mod):
        assert mod.ValidationLevel.COMPARISON_TO_TEST.value == "comparison_to_test_data"

    def test_comparison_to_reality(self, mod):
        assert mod.ValidationLevel.COMPARISON_TO_REALITY.value == "comparison_to_clinical_reality"

    def test_applicability(self, mod):
        assert mod.ValidationLevel.APPLICABILITY_ASSESSMENT.value == "applicability_assessment"


class TestRiskCategory:
    def test_class_i(self, mod):
        assert mod.RiskCategory.CLASS_I.value == "class_I"

    def test_class_ii(self, mod):
        assert mod.RiskCategory.CLASS_II.value == "class_II"

    def test_class_iii(self, mod):
        assert mod.RiskCategory.CLASS_III.value == "class_III"


# ---------------------------------------------------------------------------
# PredictionPair tests
# ---------------------------------------------------------------------------


class TestPredictionPair:
    def test_creation(self, mod):
        pp = mod.PredictionPair(patient_id="V-001", predicted=150.0, observed=160.0)
        assert pp.patient_id == "V-001"
        assert pp.predicted == pytest.approx(150.0)
        assert pp.observed == pytest.approx(160.0)

    def test_default_optional_fields(self, mod):
        pp = mod.PredictionPair(patient_id="V-002", predicted=1.0, observed=2.0)
        assert pp.prediction_time is None
        assert pp.observation_time is None
        assert pp.covariates == {}

    def test_with_covariates(self, mod):
        pp = mod.PredictionPair(
            patient_id="V-003",
            predicted=100.0,
            observed=110.0,
            covariates={"age_group": "old", "sex": "F"},
        )
        assert pp.covariates["age_group"] == "old"


# ---------------------------------------------------------------------------
# ValidationDataset tests
# ---------------------------------------------------------------------------


class TestValidationDataset:
    def _make_dataset(self, mod, n=50):
        rng = np.random.default_rng(42)
        pairs = [
            mod.PredictionPair(
                patient_id=f"VP-{i:04d}",
                predicted=float(rng.normal(180, 30)),
                observed=float(rng.normal(180, 30)),
            )
            for i in range(n)
        ]
        return mod.ValidationDataset(name="test", pairs=pairs, endpoint="PFS_days")

    def test_from_arrays(self, mod):
        ds = self._make_dataset(mod, n=30)
        assert ds.name == "test"
        assert len(ds.pairs) == 30

    def test_length(self, mod):
        ds = self._make_dataset(mod, n=100)
        assert len(ds.pairs) == 100


# ---------------------------------------------------------------------------
# AccuracyMetrics tests
# ---------------------------------------------------------------------------


class TestAccuracyMetrics:
    def _make_dataset(self, mod, n=100):
        rng = np.random.default_rng(42)
        true_vals = rng.exponential(180, n)
        predicted = true_vals + rng.normal(0, 20, n)
        pairs = [
            mod.PredictionPair(patient_id=f"A-{i}", predicted=float(predicted[i]), observed=float(true_vals[i]))
            for i in range(n)
        ]
        return mod.ValidationDataset(name="acc_test", pairs=pairs, endpoint="PFS_days")

    def test_mae_nonnegative(self, mod):
        metrics = mod.AccuracyMetrics()
        ds = self._make_dataset(mod)
        result = metrics.compute_all(ds)
        assert result["mae"] >= 0

    def test_rmse_ge_mae(self, mod):
        metrics = mod.AccuracyMetrics()
        ds = self._make_dataset(mod)
        result = metrics.compute_all(ds)
        assert result["rmse"] >= result["mae"]

    def test_r_squared_bounded(self, mod):
        metrics = mod.AccuracyMetrics()
        ds = self._make_dataset(mod)
        result = metrics.compute_all(ds)
        # R^2 should be positive for correlated data
        assert result["r_squared"] > 0
        assert result["r_squared"] <= 1.0

    def test_c_index_bounded(self, mod):
        metrics = mod.AccuracyMetrics()
        ds = self._make_dataset(mod)
        result = metrics.compute_all(ds)
        assert 0.0 <= result["c_index"] <= 1.0

    def test_pearson_significant(self, mod):
        metrics = mod.AccuracyMetrics()
        ds = self._make_dataset(mod)
        result = metrics.compute_all(ds)
        assert result["pearson_p"] < 0.05  # should be highly correlated


# ---------------------------------------------------------------------------
# CalibrationAnalyzer tests
# ---------------------------------------------------------------------------


class TestCalibrationAnalyzer:
    def test_analyze_returns_dict(self, mod):
        rng = np.random.default_rng(42)
        n = 100
        true_labels = (rng.random(n) > 0.5).astype(float)
        pred_probs = np.clip(true_labels * 0.7 + (1 - true_labels) * 0.3 + rng.normal(0, 0.1, n), 0.01, 0.99)
        analyzer = mod.CalibrationAnalyzer()
        result = analyzer.analyze(pred_probs, true_labels)
        assert isinstance(result, dict)
        assert "hosmer_lemeshow_stat" in result
        assert "hosmer_lemeshow_p" in result
        assert "expected_calibration_error" in result
        assert "brier_score" in result
        assert "calibration_groups" in result

    def test_brier_score_bounded(self, mod):
        rng = np.random.default_rng(42)
        n = 100
        true_labels = (rng.random(n) > 0.5).astype(float)
        pred_probs = np.clip(true_labels * 0.7 + rng.normal(0, 0.1, n), 0.01, 0.99)
        analyzer = mod.CalibrationAnalyzer()
        result = analyzer.analyze(pred_probs, true_labels)
        assert 0.0 <= result["brier_score"] <= 1.0

    def test_perfect_calibration(self, mod):
        """If predictions exactly match labels, Brier should be low."""
        labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        preds = np.array([0.99, 0.01, 0.99, 0.01, 0.99, 0.01, 0.99, 0.01, 0.99, 0.01])
        analyzer = mod.CalibrationAnalyzer()
        result = analyzer.analyze(preds, labels)
        assert result["brier_score"] < 0.01


# ---------------------------------------------------------------------------
# DiscriminationAnalyzer tests
# ---------------------------------------------------------------------------


class TestDiscriminationAnalyzer:
    def test_compute_auc_bounded(self, mod):
        rng = np.random.default_rng(42)
        n = 100
        labels = (rng.random(n) > 0.5).astype(float)
        scores = np.clip(labels * 0.7 + (1 - labels) * 0.3 + rng.normal(0, 0.15, n), 0.01, 0.99)
        analyzer = mod.DiscriminationAnalyzer()
        result = analyzer.analyze(scores, labels)
        assert 0.0 <= result["auc"] <= 1.0

    def test_auc_above_random(self, mod):
        """With informative scores, AUC should be above 0.5."""
        rng = np.random.default_rng(42)
        n = 200
        labels = (rng.random(n) > 0.5).astype(float)
        scores = np.clip(labels * 0.8 + (1 - labels) * 0.2 + rng.normal(0, 0.1, n), 0.01, 0.99)
        analyzer = mod.DiscriminationAnalyzer()
        result = analyzer.analyze(scores, labels)
        assert result["auc"] > 0.5

    def test_confusion_matrix_sums(self, mod):
        rng = np.random.default_rng(42)
        n = 80
        labels = (rng.random(n) > 0.5).astype(float)
        scores = np.clip(labels * 0.7 + rng.normal(0, 0.2, n), 0.01, 0.99)
        analyzer = mod.DiscriminationAnalyzer()
        result = analyzer.analyze(scores, labels, threshold=0.5)
        cm = result["confusion_matrix"]
        assert cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"] == n

    def test_sensitivity_specificity_bounded(self, mod):
        rng = np.random.default_rng(42)
        n = 100
        labels = (rng.random(n) > 0.5).astype(float)
        scores = np.clip(labels * 0.6 + rng.normal(0, 0.2, n), 0.01, 0.99)
        analyzer = mod.DiscriminationAnalyzer()
        result = analyzer.analyze(scores, labels)
        assert 0.0 <= result["sensitivity"] <= 1.0
        assert 0.0 <= result["specificity"] <= 1.0


# ---------------------------------------------------------------------------
# VVReportGenerator tests
# ---------------------------------------------------------------------------


class TestVVReportGenerator:
    def test_generate_model_card_keys(self, mod):
        gen = mod.VVReportGenerator()
        acc = {"n": 100, "mae": 15.0, "rmse": 20.0, "r_squared": 0.85, "c_index": 0.75}
        card = gen.generate_model_card(
            model_name="Test DT",
            model_version="1.0",
            intended_use="Testing",
            accuracy_results=acc,
        )
        assert "model_details" in card
        assert "performance" in card
        assert "limitations" in card
        assert "regulatory" in card
        assert card["model_details"]["name"] == "Test DT"

    def test_generate_vv_report_keys(self, mod):
        gen = mod.VVReportGenerator()
        acc = {"n": 100, "mae": 15.0, "rmse": 20.0, "r_squared": 0.85, "c_index": 0.75}
        report = gen.generate_vv_report(model_name="Test DT", accuracy_results=acc)
        assert "overall_result" in report
        assert "criteria_passed" in report
        assert "criteria_failed" in report
        assert "detailed_results" in report
        assert "regulatory_references" in report

    def test_generate_vv_report_pass(self, mod):
        gen = mod.VVReportGenerator()
        acc = {"r_squared": 0.9, "c_index": 0.8}
        cal = {"calibration_adequate": True}
        disc = {"auc": 0.85}
        report = gen.generate_vv_report(
            model_name="Good Model",
            accuracy_results=acc,
            calibration_results=cal,
            discrimination_results=disc,
        )
        assert report["overall_result"] == "PASS"

    def test_generate_vv_report_fail(self, mod):
        gen = mod.VVReportGenerator()
        acc = {"r_squared": 0.3, "c_index": 0.45}
        report = gen.generate_vv_report(model_name="Bad Model", accuracy_results=acc)
        assert report["overall_result"] == "FAIL"
        assert len(report["criteria_failed"]) > 0
