"""
=============================================================================
EXAMPLE 06: Digital Twin Validation & Verification (V&V) Framework
=============================================================================

WHAT THIS CODE DOES:
    Implements a comprehensive validation and verification framework for
    oncology digital twins per FDA regulatory requirements. Assesses
    prediction accuracy, calibration, discrimination, and clinical utility
    of DT models against real patient outcomes. Generates regulatory-grade
    documentation including model cards, V&V reports, and FDA submission
    artifacts.

CLINICAL CONTEXT:
    For digital twins to be used in clinical decision-making, regulatory
    bodies (FDA, EMA) require rigorous validation demonstrating:
      - Prediction accuracy: DT predictions match actual patient outcomes
      - Calibration: predicted probabilities are well-calibrated
      - Discrimination: DT can distinguish responders from non-responders
      - Generalizability: performance across patient subgroups
      - Robustness: stability under input perturbation
    The ASME V&V 40 standard and FDA TPLC guidance for AI/ML devices
    define the validation framework implemented here.

USE CASES COVERED:
    1. Concordance index (C-index) for survival prediction accuracy
    2. Calibration analysis (Hosmer-Lemeshow, calibration curves)
    3. Discrimination metrics (AUC, sensitivity, specificity, F1)
    4. Subgroup analysis across patient demographics and biomarkers
    5. Temporal validation (train/test on different time periods)
    6. Sensitivity analysis and robustness testing
    7. Model card and V&V report generation for FDA submission

FRAMEWORK REQUIREMENTS:
    Required:
        - NumPy 1.24.0+
        - SciPy 1.11.0+ (statistics)
    Optional:
        - scikit-learn (for full ROC/AUC computation)
        - lifelines (for concordance index)

REGULATORY NOTES:
    - ASME V&V 40: Verification and Validation in Computational Modeling
    - FDA: Artificial Intelligence and Machine Learning in SaMD (2021/2025)
    - FDA TPLC: Total Product Life Cycle for AI/ML-enabled Devices
    - FDA PCCP: Predetermined Change Control Plan Guidance (Aug 2025)
    - IEC 62304: Medical device software lifecycle (Class B/C)
    - ISO 14971: Risk management for medical devices

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2, norm, pearsonr, spearmanr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: VALIDATION DATA STRUCTURES
# =============================================================================


class ValidationLevel(Enum):
    """ASME V&V 40 validation levels."""

    CREDIBILITY_ASSESSMENT = "credibility_assessment"
    COMPARISON_TO_TEST = "comparison_to_test_data"
    COMPARISON_TO_REALITY = "comparison_to_clinical_reality"
    APPLICABILITY_ASSESSMENT = "applicability_assessment"


class RiskCategory(Enum):
    """FDA Software as Medical Device risk categories."""

    CLASS_I = "class_I"     # Low risk
    CLASS_II = "class_II"   # Moderate risk (most DT applications)
    CLASS_III = "class_III"  # High risk


@dataclass
class PredictionPair:
    """A single predicted-vs-observed outcome pair.

    Attributes:
        patient_id: Patient identifier
        predicted: DT-predicted value
        observed: Actual clinical outcome
        prediction_time: When prediction was made
        observation_time: When outcome was observed
        covariates: Patient covariates for subgroup analysis
    """

    patient_id: str
    predicted: float
    observed: float
    prediction_time: datetime | None = None
    observation_time: datetime | None = None
    covariates: dict = field(default_factory=dict)


@dataclass
class ValidationDataset:
    """Dataset of prediction-outcome pairs for validation.

    Attributes:
        name: Dataset identifier
        pairs: List of prediction pairs
        endpoint: Clinical endpoint (e.g., "PFS_days", "tumor_volume")
        description: Dataset description
        collection_period: Data collection period
    """

    name: str
    pairs: list[PredictionPair]
    endpoint: str
    description: str = ""
    collection_period: str = ""


# =============================================================================
# SECTION 2: ACCURACY METRICS
# =============================================================================
# Comprehensive prediction accuracy metrics for continuous and
# time-to-event outcomes.


class AccuracyMetrics:
    """Computes prediction accuracy metrics for DT validation.

    Implements standard biostatistical metrics for comparing DT predictions
    to observed clinical outcomes.

    Instructions for engineers:
        - Use MAE/RMSE for continuous outcomes (tumor volume, biomarker)
        - Use C-index for time-to-event outcomes (PFS, OS)
        - Use AUC for binary classification (response/non-response)
        - Always report with confidence intervals (bootstrap)
        - Subgroup analysis required for FDA submission

    Example:
        >>> metrics = AccuracyMetrics()
        >>> result = metrics.compute_all(validation_dataset)
        >>> print(f"MAE: {result['mae']:.2f}, R^2: {result['r_squared']:.3f}")
    """

    def compute_all(self, dataset: ValidationDataset) -> dict[str, Any]:
        """Compute all applicable accuracy metrics.

        Args:
            dataset: Validation dataset

        Returns:
            Dictionary of computed metrics with confidence intervals
        """
        predicted = np.array([p.predicted for p in dataset.pairs])
        observed = np.array([p.observed for p in dataset.pairs])

        n = len(predicted)
        if n < 5:
            logger.warning("Insufficient data for validation: n=%d", n)
            return {"n": n, "error": "insufficient_data"}

        results = {"n": n, "endpoint": dataset.endpoint}

        # Continuous metrics
        results["mae"] = self.mean_absolute_error(predicted, observed)
        results["rmse"] = self.root_mean_squared_error(predicted, observed)
        results["mape"] = self.mean_absolute_percentage_error(predicted, observed)
        results["r_squared"] = self.r_squared(predicted, observed)
        results["pearson_r"], results["pearson_p"] = pearsonr(predicted, observed)
        results["spearman_r"], results["spearman_p"] = spearmanr(predicted, observed)

        # Bias
        results["mean_bias"] = float(np.mean(predicted - observed))
        results["bias_std"] = float(np.std(predicted - observed))

        # Concordance index (for time-to-event)
        results["c_index"] = self.concordance_index(predicted, observed)

        # Bootstrap confidence intervals
        results["ci_95"] = self._bootstrap_ci(predicted, observed)

        return results

    def mean_absolute_error(self, predicted: np.ndarray, observed: np.ndarray) -> float:
        """MAE: average absolute prediction error."""
        return float(np.mean(np.abs(predicted - observed)))

    def root_mean_squared_error(self, predicted: np.ndarray, observed: np.ndarray) -> float:
        """RMSE: root mean squared error."""
        return float(np.sqrt(np.mean((predicted - observed) ** 2)))

    def mean_absolute_percentage_error(self, predicted: np.ndarray, observed: np.ndarray) -> float:
        """MAPE: mean absolute percentage error."""
        nonzero = observed != 0
        if not np.any(nonzero):
            return float("inf")
        return float(np.mean(np.abs((predicted[nonzero] - observed[nonzero]) / observed[nonzero])) * 100)

    def r_squared(self, predicted: np.ndarray, observed: np.ndarray) -> float:
        """Coefficient of determination (R^2)."""
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)

    def concordance_index(self, predicted: np.ndarray, observed: np.ndarray) -> float:
        """Harrell's concordance index for survival predictions.

        Measures the probability that for a random pair of patients,
        the one with the worse predicted outcome actually had a worse
        observed outcome. C=0.5 is random, C=1.0 is perfect.
        """
        n = len(predicted)
        concordant = 0
        discordant = 0
        tied = 0

        for i in range(n):
            for j in range(i + 1, n):
                if observed[i] == observed[j]:
                    tied += 1
                    continue

                # Determine which has better observed outcome
                if observed[i] > observed[j]:
                    # i has longer survival → should have lower risk (higher predicted survival)
                    if predicted[i] > predicted[j]:
                        concordant += 1
                    elif predicted[i] < predicted[j]:
                        discordant += 1
                    else:
                        tied += 1
                else:
                    if predicted[j] > predicted[i]:
                        concordant += 1
                    elif predicted[j] < predicted[i]:
                        discordant += 1
                    else:
                        tied += 1

        total = concordant + discordant + tied
        if total == 0:
            return 0.5
        return float(concordant / (concordant + discordant + 0.5 * tied))

    def _bootstrap_ci(
        self,
        predicted: np.ndarray,
        observed: np.ndarray,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
    ) -> dict[str, tuple[float, float]]:
        """Compute bootstrap 95% CIs for key metrics."""
        rng = np.random.default_rng(42)
        n = len(predicted)

        mae_samples = []
        r2_samples = []
        c_samples = []

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            p_boot = predicted[idx]
            o_boot = observed[idx]

            mae_samples.append(self.mean_absolute_error(p_boot, o_boot))
            r2_samples.append(self.r_squared(p_boot, o_boot))
            c_samples.append(self.concordance_index(p_boot, o_boot))

        ci = {}
        ci["mae"] = (
            float(np.percentile(mae_samples, alpha / 2 * 100)),
            float(np.percentile(mae_samples, (1 - alpha / 2) * 100)),
        )
        ci["r_squared"] = (
            float(np.percentile(r2_samples, alpha / 2 * 100)),
            float(np.percentile(r2_samples, (1 - alpha / 2) * 100)),
        )
        ci["c_index"] = (
            float(np.percentile(c_samples, alpha / 2 * 100)),
            float(np.percentile(c_samples, (1 - alpha / 2) * 100)),
        )

        return ci


# =============================================================================
# SECTION 3: CALIBRATION ANALYSIS
# =============================================================================
# Assesses whether predicted probabilities match observed frequencies.


class CalibrationAnalyzer:
    """Calibration analysis for probabilistic DT predictions.

    Evaluates whether predicted probabilities (e.g., response probability,
    DLT probability) are well-calibrated against observed frequencies.

    Instructions for engineers:
        - Hosmer-Lemeshow test with 10 groups (standard)
        - Expected calibration error (ECE) for overall calibration
        - Calibration plot (predicted vs. observed) for visualization
        - p > 0.05 on H-L test indicates adequate calibration

    Example:
        >>> calibrator = CalibrationAnalyzer()
        >>> result = calibrator.analyze(predicted_probs, observed_binary)
    """

    def analyze(
        self,
        predicted_probabilities: np.ndarray,
        observed_binary: np.ndarray,
        n_groups: int = 10,
    ) -> dict[str, Any]:
        """Run calibration analysis.

        Args:
            predicted_probabilities: Predicted probabilities (0-1)
            observed_binary: Observed binary outcomes (0 or 1)
            n_groups: Number of groups for H-L test

        Returns:
            Dictionary with calibration metrics
        """
        n = len(predicted_probabilities)

        # Sort by predicted probability
        sorted_idx = np.argsort(predicted_probabilities)
        pred_sorted = predicted_probabilities[sorted_idx]
        obs_sorted = observed_binary[sorted_idx]

        # Group into deciles
        group_size = max(1, n // n_groups)
        groups = []

        for i in range(n_groups):
            start = i * group_size
            end = min(start + group_size, n)
            if start >= n:
                break

            group_pred = pred_sorted[start:end]
            group_obs = obs_sorted[start:end]

            groups.append({
                "mean_predicted": float(np.mean(group_pred)),
                "mean_observed": float(np.mean(group_obs)),
                "n": len(group_pred),
                "n_events": int(np.sum(group_obs)),
            })

        # Hosmer-Lemeshow statistic
        hl_stat = 0.0
        for g in groups:
            if g["n"] == 0:
                continue
            expected = g["mean_predicted"] * g["n"]
            if expected > 0 and (g["n"] - expected) > 0:
                hl_stat += (g["n_events"] - expected) ** 2 / (
                    expected * (1 - g["mean_predicted"])
                )

        hl_df = max(1, len(groups) - 2)
        hl_pvalue = float(1 - chi2.cdf(hl_stat, hl_df))

        # Expected Calibration Error
        ece = np.mean([
            abs(g["mean_predicted"] - g["mean_observed"]) * g["n"] / n
            for g in groups
        ])

        # Brier score
        brier = float(np.mean((predicted_probabilities - observed_binary) ** 2))

        return {
            "hosmer_lemeshow_stat": round(float(hl_stat), 3),
            "hosmer_lemeshow_p": round(hl_pvalue, 4),
            "calibration_adequate": hl_pvalue > 0.05,
            "expected_calibration_error": round(float(ece), 4),
            "brier_score": round(brier, 4),
            "calibration_groups": groups,
            "n_groups": len(groups),
        }


# =============================================================================
# SECTION 4: DISCRIMINATION ANALYSIS
# =============================================================================
# Binary classification metrics for DT predictions.


class DiscriminationAnalyzer:
    """Discrimination analysis for binary DT predictions.

    Evaluates the DT's ability to distinguish between outcome categories
    (e.g., responder vs. non-responder, progression vs. stable).

    Instructions for engineers:
        - AUC > 0.7 generally acceptable for clinical utility
        - Optimize threshold for clinical context (sensitivity vs. specificity)
        - Report PPV/NPV at relevant prevalence rates
        - DeLong test for comparing AUC between models

    Example:
        >>> disc = DiscriminationAnalyzer()
        >>> result = disc.analyze(predicted_scores, observed_labels)
        >>> print(f"AUC: {result['auc']:.3f}")
    """

    def analyze(
        self,
        predicted_scores: np.ndarray,
        observed_labels: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Run discrimination analysis.

        Args:
            predicted_scores: Predicted probability scores (0-1)
            observed_labels: Binary labels (0 or 1)
            threshold: Classification threshold

        Returns:
            Dictionary with discrimination metrics
        """
        # Binary predictions at threshold
        predicted_labels = (predicted_scores >= threshold).astype(int)

        # Confusion matrix
        tp = int(np.sum((predicted_labels == 1) & (observed_labels == 1)))
        tn = int(np.sum((predicted_labels == 0) & (observed_labels == 0)))
        fp = int(np.sum((predicted_labels == 1) & (observed_labels == 0)))
        fn = int(np.sum((predicted_labels == 0) & (observed_labels == 1)))

        # Metrics
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)

        # AUC (trapezoidal approximation)
        auc = self._compute_auc(predicted_scores, observed_labels)

        # Youden's J statistic for optimal threshold
        optimal_threshold = self._find_optimal_threshold(
            predicted_scores, observed_labels
        )

        return {
            "auc": round(auc, 3),
            "sensitivity": round(sensitivity, 3),
            "specificity": round(specificity, 3),
            "ppv": round(ppv, 3),
            "npv": round(npv, 3),
            "accuracy": round(accuracy, 3),
            "f1_score": round(f1, 3),
            "threshold_used": threshold,
            "optimal_threshold": round(optimal_threshold, 3),
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }

    def _compute_auc(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> float:
        """Compute AUC via Mann-Whitney U statistic."""
        pos = scores[labels == 1]
        neg = scores[labels == 0]

        if len(pos) == 0 or len(neg) == 0:
            return 0.5

        u_stat = 0.0
        for p in pos:
            u_stat += np.sum(p > neg) + 0.5 * np.sum(p == neg)

        return float(u_stat / (len(pos) * len(neg)))

    def _find_optimal_threshold(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> float:
        """Find optimal threshold via Youden's J statistic."""
        thresholds = np.linspace(0.01, 0.99, 99)
        best_j = -1
        best_thresh = 0.5

        for t in thresholds:
            pred = (scores >= t).astype(int)
            tp = np.sum((pred == 1) & (labels == 1))
            fn = np.sum((pred == 0) & (labels == 1))
            tn = np.sum((pred == 0) & (labels == 0))
            fp = np.sum((pred == 1) & (labels == 0))

            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            j = sens + spec - 1

            if j > best_j:
                best_j = j
                best_thresh = t

        return best_thresh


# =============================================================================
# SECTION 5: SUBGROUP ANALYSIS
# =============================================================================


class SubgroupAnalyzer:
    """Subgroup performance analysis for equity and generalizability.

    Evaluates DT prediction performance across patient subgroups to
    ensure equitable accuracy and identify populations where the model
    may underperform.

    Instructions for engineers:
        - FDA requires performance across demographic subgroups
        - Report metrics stratified by age, sex, race, tumor stage
        - Identify subgroups with degraded performance for labeling
        - Minimum subgroup size: 20 patients for meaningful analysis
    """

    def analyze(
        self,
        dataset: ValidationDataset,
        subgroup_variables: list[str],
    ) -> dict[str, Any]:
        """Analyze prediction performance across subgroups.

        Args:
            dataset: Validation dataset with covariates
            subgroup_variables: Variables to stratify by

        Returns:
            Dictionary with subgroup-level metrics
        """
        metrics = AccuracyMetrics()
        results = {}

        for variable in subgroup_variables:
            # Extract unique values for this variable
            values = set()
            for pair in dataset.pairs:
                if variable in pair.covariates:
                    values.add(pair.covariates[variable])

            subgroup_results = {}
            for value in sorted(values, key=str):
                subset = [
                    p for p in dataset.pairs
                    if p.covariates.get(variable) == value
                ]

                if len(subset) < 10:
                    continue

                sub_dataset = ValidationDataset(
                    name=f"{variable}={value}",
                    pairs=subset,
                    endpoint=dataset.endpoint,
                )
                sub_metrics = metrics.compute_all(sub_dataset)
                subgroup_results[str(value)] = {
                    "n": sub_metrics["n"],
                    "mae": sub_metrics.get("mae", None),
                    "r_squared": sub_metrics.get("r_squared", None),
                    "c_index": sub_metrics.get("c_index", None),
                }

            results[variable] = subgroup_results

        return results


# =============================================================================
# SECTION 6: ROBUSTNESS TESTING
# =============================================================================


class RobustnessAnalyzer:
    """Sensitivity and robustness analysis for DT predictions.

    Tests DT prediction stability under input perturbation to assess
    model robustness per FDA AI/ML guidance requirements.

    Instructions for engineers:
        - Perturb each input variable within clinically plausible ranges
        - Measure output sensitivity (partial derivatives)
        - Identify inputs with disproportionate influence
        - Report worst-case performance under perturbation
    """

    def analyze(
        self,
        predict_fn,
        baseline_inputs: dict[str, float],
        perturbation_ranges: dict[str, tuple[float, float]],
        n_samples: int = 100,
    ) -> dict[str, Any]:
        """Run robustness analysis.

        Args:
            predict_fn: Function mapping input dict to prediction
            baseline_inputs: Baseline input values
            perturbation_ranges: Min/max perturbation per variable
            n_samples: Number of perturbation samples

        Returns:
            Dictionary with sensitivity indices and stability metrics
        """
        rng = np.random.default_rng(42)
        baseline_pred = predict_fn(baseline_inputs)

        sensitivities = {}
        all_predictions = []

        for var_name, (low, high) in perturbation_ranges.items():
            predictions = []
            for _ in range(n_samples):
                perturbed = baseline_inputs.copy()
                perturbed[var_name] = rng.uniform(low, high)
                pred = predict_fn(perturbed)
                predictions.append(pred)

            pred_array = np.array(predictions)
            all_predictions.extend(predictions)

            sensitivities[var_name] = {
                "mean_prediction": round(float(np.mean(pred_array)), 3),
                "std_prediction": round(float(np.std(pred_array)), 3),
                "range": (round(float(np.min(pred_array)), 3),
                          round(float(np.max(pred_array)), 3)),
                "sensitivity_index": round(
                    float(np.std(pred_array) / max(abs(baseline_pred), 0.01)), 3
                ),
            }

        # Overall stability
        all_preds = np.array(all_predictions)
        stability = {
            "baseline_prediction": round(float(baseline_pred), 3),
            "overall_std": round(float(np.std(all_preds)), 3),
            "coefficient_of_variation": round(
                float(np.std(all_preds) / max(abs(np.mean(all_preds)), 0.01)), 3
            ),
            "robust": float(np.std(all_preds)) < 0.1 * abs(baseline_pred),
        }

        return {
            "sensitivities": sensitivities,
            "stability": stability,
        }


# =============================================================================
# SECTION 7: V&V REPORT AND MODEL CARD GENERATOR
# =============================================================================


class VVReportGenerator:
    """Generates regulatory V&V reports and model cards.

    Creates FDA-submission-ready documentation including model cards
    (per Mitchell et al. 2019), V&V reports (per ASME V&V 40), and
    PCCP documentation (per FDA Aug 2025 guidance).

    Instructions for engineers:
        1. Run all validation analyses (accuracy, calibration, etc.)
        2. Call generate_model_card() for concise model documentation
        3. Call generate_vv_report() for comprehensive V&V report
        4. Include both in FDA 510(k) or De Novo submission
        5. Update with each model modification per PCCP

    Example:
        >>> generator = VVReportGenerator()
        >>> model_card = generator.generate_model_card(
        ...     model_name="Tumor Growth DT v1.0",
        ...     accuracy_results=accuracy,
        ...     calibration_results=calibration,
        ... )
    """

    def generate_model_card(
        self,
        model_name: str,
        model_version: str,
        intended_use: str,
        accuracy_results: dict,
        calibration_results: dict | None = None,
        discrimination_results: dict | None = None,
        subgroup_results: dict | None = None,
        robustness_results: dict | None = None,
        risk_category: RiskCategory = RiskCategory.CLASS_II,
    ) -> dict[str, Any]:
        """Generate model card for the digital twin.

        Args:
            model_name: Name of the DT model
            model_version: Version string
            intended_use: Intended clinical use statement
            accuracy_results: Results from AccuracyMetrics
            calibration_results: Results from CalibrationAnalyzer
            discrimination_results: Results from DiscriminationAnalyzer
            subgroup_results: Results from SubgroupAnalyzer
            robustness_results: Results from RobustnessAnalyzer
            risk_category: FDA risk classification

        Returns:
            Model card as structured dictionary
        """
        card = {
            "model_card_version": "1.0",
            "generated": datetime.now().isoformat(),
            "model_details": {
                "name": model_name,
                "version": model_version,
                "type": "Oncology Digital Twin",
                "framework": "Physical AI Oncology Trials",
                "risk_category": risk_category.value,
            },
            "intended_use": {
                "primary_use": intended_use,
                "intended_users": "Oncologists, radiation oncologists, clinical researchers",
                "out_of_scope": "Not for autonomous treatment decisions; physician oversight required",
            },
            "performance": {
                "accuracy": {
                    "n_patients": accuracy_results.get("n", 0),
                    "mae": accuracy_results.get("mae"),
                    "rmse": accuracy_results.get("rmse"),
                    "r_squared": accuracy_results.get("r_squared"),
                    "c_index": accuracy_results.get("c_index"),
                    "confidence_intervals": accuracy_results.get("ci_95"),
                },
            },
            "limitations": [
                "Model validated on specific tumor types; generalization to other tumors not validated",
                "Performance may degrade for patients with rare genomic profiles",
                "Requires minimum imaging quality standards for accurate predictions",
                "Treatment response predictions assume protocol adherence",
            ],
            "ethical_considerations": [
                "Predictions should augment, not replace, clinical judgment",
                "Subgroup performance variations should be communicated to clinicians",
                "Patient consent required for DT creation from clinical data",
            ],
            "regulatory": {
                "fda_pathway": "510(k) or De Novo",
                "asme_vv40_level": ValidationLevel.COMPARISON_TO_REALITY.value,
                "iec_62304_class": "B",
                "pccp_applicable": True,
            },
        }

        if calibration_results:
            card["performance"]["calibration"] = {
                "hosmer_lemeshow_p": calibration_results.get("hosmer_lemeshow_p"),
                "ece": calibration_results.get("expected_calibration_error"),
                "brier_score": calibration_results.get("brier_score"),
                "adequate": calibration_results.get("calibration_adequate"),
            }

        if discrimination_results:
            card["performance"]["discrimination"] = {
                "auc": discrimination_results.get("auc"),
                "sensitivity": discrimination_results.get("sensitivity"),
                "specificity": discrimination_results.get("specificity"),
                "f1_score": discrimination_results.get("f1_score"),
            }

        if subgroup_results:
            card["performance"]["subgroup_analysis"] = subgroup_results

        if robustness_results:
            card["performance"]["robustness"] = robustness_results.get("stability")

        return card

    def generate_vv_report(
        self,
        model_name: str,
        accuracy_results: dict,
        calibration_results: dict | None = None,
        discrimination_results: dict | None = None,
        subgroup_results: dict | None = None,
        robustness_results: dict | None = None,
        validation_level: ValidationLevel = ValidationLevel.COMPARISON_TO_REALITY,
    ) -> dict[str, Any]:
        """Generate comprehensive V&V report for regulatory submission.

        Args:
            model_name: Model name
            accuracy_results: Accuracy analysis results
            calibration_results: Calibration analysis results
            discrimination_results: Discrimination analysis results
            subgroup_results: Subgroup analysis results
            robustness_results: Robustness analysis results
            validation_level: ASME V&V 40 validation level

        Returns:
            V&V report as structured dictionary
        """
        # Overall assessment
        passes = []
        fails = []

        # Accuracy check
        r2 = accuracy_results.get("r_squared", 0)
        if r2 > 0.5:
            passes.append(f"R^2 = {r2:.3f} (>0.5 threshold)")
        else:
            fails.append(f"R^2 = {r2:.3f} (<0.5 threshold)")

        c_idx = accuracy_results.get("c_index", 0.5)
        if c_idx > 0.6:
            passes.append(f"C-index = {c_idx:.3f} (>0.6 threshold)")
        else:
            fails.append(f"C-index = {c_idx:.3f} (<0.6 threshold)")

        # Calibration check
        if calibration_results:
            if calibration_results.get("calibration_adequate", False):
                passes.append("Calibration adequate (H-L p > 0.05)")
            else:
                fails.append("Calibration inadequate (H-L p < 0.05)")

        # Discrimination check
        if discrimination_results:
            auc = discrimination_results.get("auc", 0.5)
            if auc > 0.7:
                passes.append(f"AUC = {auc:.3f} (>0.7 threshold)")
            else:
                fails.append(f"AUC = {auc:.3f} (<0.7 threshold)")

        overall_pass = len(fails) == 0

        report = {
            "report_title": f"Validation & Verification Report: {model_name}",
            "report_version": "1.0",
            "generated": datetime.now().isoformat(),
            "validation_level": validation_level.value,
            "overall_result": "PASS" if overall_pass else "FAIL",
            "criteria_passed": passes,
            "criteria_failed": fails,
            "detailed_results": {
                "accuracy": accuracy_results,
                "calibration": calibration_results,
                "discrimination": discrimination_results,
                "subgroup": subgroup_results,
                "robustness": robustness_results,
            },
            "regulatory_references": [
                "ASME V&V 40-2018: Verification and Validation in Computational Modeling",
                "FDA: AI/ML-Based Software as Medical Device Action Plan (2021/2025)",
                "FDA: Predetermined Change Control Plans for ML-Enabled Devices (Aug 2025)",
                "IEC 62304: Medical device software lifecycle processes",
                "ISO 14971: Application of risk management to medical devices",
            ],
            "recommendations": [],
        }

        if fails:
            report["recommendations"].append(
                "Address failing criteria before regulatory submission"
            )
        if not subgroup_results:
            report["recommendations"].append(
                "Conduct subgroup analysis for complete regulatory package"
            )
        if not robustness_results:
            report["recommendations"].append(
                "Perform sensitivity analysis per FDA AI/ML guidance"
            )

        return report

    def export_to_json(self, report: dict, output_path: str) -> str:
        """Export report to JSON file.

        Args:
            report: Report dictionary
            output_path: Output file path

        Returns:
            Path to saved file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Report exported to %s", output_path)
        return output_path


# =============================================================================
# SECTION 8: MAIN — COMPLETE V&V DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Digital Twin Validation & Verification Framework")
    print("Physical AI Oncology Trials — Example 06")
    print("=" * 70)

    # --- Generate synthetic validation data ---
    rng = np.random.default_rng(42)
    n_patients = 200

    # Simulated DT predictions vs. actual outcomes
    true_pfs = rng.exponential(180, n_patients)  # true PFS in days
    prediction_noise = rng.normal(0, 30, n_patients)
    predicted_pfs = true_pfs + prediction_noise  # DT predictions

    # Response probabilities
    true_response = (rng.random(n_patients) > 0.6).astype(int)
    predicted_prob = np.clip(
        true_response * 0.7 + (1 - true_response) * 0.3 + rng.normal(0, 0.15, n_patients),
        0.01, 0.99,
    )

    # Create validation dataset
    pairs = []
    for i in range(n_patients):
        age_group = "young" if rng.random() > 0.5 else "old"
        sex = "M" if rng.random() > 0.45 else "F"
        stage = rng.choice(["IIIB", "IVA", "IVB"])
        pairs.append(PredictionPair(
            patient_id=f"VAL-{i:04d}",
            predicted=float(predicted_pfs[i]),
            observed=float(true_pfs[i]),
            covariates={"age_group": age_group, "sex": sex, "stage": stage},
        ))

    dataset = ValidationDataset(
        name="Tumor Growth DT Validation Set",
        pairs=pairs,
        endpoint="PFS_days",
        description="Retrospective validation against 200 NSCLC patients",
        collection_period="2024-2025",
    )

    # --- Run accuracy analysis ---
    print("\n--- 1. Accuracy Metrics ---")
    accuracy = AccuracyMetrics()
    acc_results = accuracy.compute_all(dataset)
    print(f"  N patients: {acc_results['n']}")
    print(f"  MAE: {acc_results['mae']:.1f} days")
    print(f"  RMSE: {acc_results['rmse']:.1f} days")
    print(f"  R-squared: {acc_results['r_squared']:.3f}")
    print(f"  C-index: {acc_results['c_index']:.3f}")
    print(f"  Pearson r: {acc_results['pearson_r']:.3f} (p={acc_results['pearson_p']:.2e})")
    print(f"  Mean bias: {acc_results['mean_bias']:.1f} days")
    if "ci_95" in acc_results:
        for metric, ci in acc_results["ci_95"].items():
            print(f"  95% CI ({metric}): ({ci[0]:.3f}, {ci[1]:.3f})")

    # --- Run calibration analysis ---
    print("\n--- 2. Calibration Analysis ---")
    calibrator = CalibrationAnalyzer()
    cal_results = calibrator.analyze(predicted_prob, true_response)
    print(f"  Hosmer-Lemeshow stat: {cal_results['hosmer_lemeshow_stat']}")
    print(f"  H-L p-value: {cal_results['hosmer_lemeshow_p']}")
    print(f"  Calibration adequate: {cal_results['calibration_adequate']}")
    print(f"  ECE: {cal_results['expected_calibration_error']}")
    print(f"  Brier score: {cal_results['brier_score']}")

    # --- Run discrimination analysis ---
    print("\n--- 3. Discrimination Analysis ---")
    discriminator = DiscriminationAnalyzer()
    disc_results = discriminator.analyze(predicted_prob, true_response)
    print(f"  AUC: {disc_results['auc']}")
    print(f"  Sensitivity: {disc_results['sensitivity']}")
    print(f"  Specificity: {disc_results['specificity']}")
    print(f"  PPV: {disc_results['ppv']}")
    print(f"  NPV: {disc_results['npv']}")
    print(f"  F1 score: {disc_results['f1_score']}")
    print(f"  Optimal threshold: {disc_results['optimal_threshold']}")

    # --- Run subgroup analysis ---
    print("\n--- 4. Subgroup Analysis ---")
    subgroup = SubgroupAnalyzer()
    sub_results = subgroup.analyze(dataset, ["age_group", "sex", "stage"])
    for variable, groups in sub_results.items():
        print(f"  {variable}:")
        for value, metrics in groups.items():
            print(f"    {value}: n={metrics['n']}, MAE={metrics['mae']:.1f}, R^2={metrics['r_squared']:.3f}")

    # --- Run robustness analysis ---
    print("\n--- 5. Robustness Analysis ---")

    def dummy_predict(inputs: dict) -> float:
        """Dummy DT prediction function for robustness testing."""
        return (
            inputs.get("tumor_volume", 20) * 0.5
            + inputs.get("growth_rate", 0.01) * 1000
            - inputs.get("treatment_sensitivity", 0.5) * 30
        )

    robustness = RobustnessAnalyzer()
    rob_results = robustness.analyze(
        predict_fn=dummy_predict,
        baseline_inputs={"tumor_volume": 20, "growth_rate": 0.015, "treatment_sensitivity": 0.5},
        perturbation_ranges={
            "tumor_volume": (5, 50),
            "growth_rate": (0.005, 0.03),
            "treatment_sensitivity": (0.1, 0.9),
        },
    )
    print(f"  Baseline prediction: {rob_results['stability']['baseline_prediction']}")
    print(f"  Overall CV: {rob_results['stability']['coefficient_of_variation']}")
    print(f"  Robust: {rob_results['stability']['robust']}")
    for var, sens in rob_results["sensitivities"].items():
        print(f"  {var}: sensitivity_index={sens['sensitivity_index']}")

    # --- Generate model card and V&V report ---
    print("\n--- 6. Regulatory Documentation ---")
    generator = VVReportGenerator()

    model_card = generator.generate_model_card(
        model_name="Tumor Growth Digital Twin",
        model_version="1.0.0",
        intended_use="Predict tumor progression-free survival for NSCLC patients to support treatment planning",
        accuracy_results=acc_results,
        calibration_results=cal_results,
        discrimination_results=disc_results,
        subgroup_results=sub_results,
        robustness_results=rob_results,
    )
    print(f"  Model card generated: {model_card['model_details']['name']}")
    print(f"  Risk category: {model_card['model_details']['risk_category']}")

    vv_report = generator.generate_vv_report(
        model_name="Tumor Growth Digital Twin v1.0",
        accuracy_results=acc_results,
        calibration_results=cal_results,
        discrimination_results=disc_results,
        subgroup_results=sub_results,
        robustness_results=rob_results,
    )
    print(f"  V&V report result: {vv_report['overall_result']}")
    print(f"  Criteria passed: {len(vv_report['criteria_passed'])}")
    print(f"  Criteria failed: {len(vv_report['criteria_failed'])}")

    for p in vv_report["criteria_passed"]:
        print(f"    PASS: {p}")
    for f in vv_report["criteria_failed"]:
        print(f"    FAIL: {f}")

    if vv_report["recommendations"]:
        print("  Recommendations:")
        for r in vv_report["recommendations"]:
            print(f"    - {r}")

    print("\n" + "=" * 70)
    print("V&V framework demonstration complete.")
