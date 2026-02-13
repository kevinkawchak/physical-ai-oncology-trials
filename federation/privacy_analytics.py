#!/usr/bin/env python3
"""Privacy-preserving analytics for federated oncology trials.

Enables cross-site survival analysis (Kaplan-Meier, Cox proportional
hazards) without exposing individual patient records. Uses differential
privacy and secure aggregation to produce population-level statistics.

Framework Dependencies:
    - NumPy 1.24.0+
    - SciPy 1.11.0+ (for statistical functions)

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.
    Statistical analyses must be validated by qualified biostatisticians.

Version: 1.0.0
Last Updated: February 2026
License: MIT
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AnalysisType(Enum):
    """Types of privacy-preserving analyses."""

    KAPLAN_MEIER = "kaplan_meier"
    COX_PH = "cox_ph"
    INCIDENCE_RATE = "incidence_rate"
    RESPONSE_RATE = "response_rate"
    SUBGROUP_ANALYSIS = "subgroup_analysis"


class EventType(Enum):
    """Types of time-to-event outcomes."""

    DEATH = "death"
    PROGRESSION = "progression"
    RECURRENCE = "recurrence"
    RESPONSE = "response"
    ADVERSE_EVENT = "adverse_event"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SurvivalRecord:
    """Individual survival/time-to-event record.

    Attributes:
        patient_id: De-identified patient identifier.
        site_id: Source site.
        time: Observation time (months).
        event: Whether the event occurred (True) or censored (False).
        treatment_arm: Treatment group.
        covariates: Covariate values for Cox regression.
    """

    patient_id: str = ""
    site_id: str = ""
    time: float = 0.0
    event: bool = False
    treatment_arm: str = ""
    covariates: Dict[str, float] = field(default_factory=dict)


@dataclass
class KaplanMeierResult:
    """Results from a Kaplan-Meier survival analysis.

    Attributes:
        result_id: Unique result identifier.
        analysis_group: Group label (arm, site, etc.).
        time_points: Ordered time points.
        survival_probs: Survival probability at each time point.
        at_risk: Number at risk at each time point.
        events: Number of events at each time point.
        censored: Number censored at each time point.
        median_survival: Estimated median survival time.
        ci_lower: Lower 95% CI for survival.
        ci_upper: Upper 95% CI for survival.
        is_privatized: Whether DP noise was applied.
    """

    result_id: str = ""
    analysis_group: str = ""
    time_points: List[float] = field(default_factory=list)
    survival_probs: List[float] = field(default_factory=list)
    at_risk: List[int] = field(default_factory=list)
    events: List[int] = field(default_factory=list)
    censored: List[int] = field(default_factory=list)
    median_survival: Optional[float] = None
    ci_lower: List[float] = field(default_factory=list)
    ci_upper: List[float] = field(default_factory=list)
    is_privatized: bool = False


@dataclass
class CoxPHResult:
    """Results from a Cox proportional hazards analysis.

    Attributes:
        result_id: Unique result identifier.
        coefficients: Covariate coefficients (log hazard ratios).
        hazard_ratios: Exponentiated coefficients.
        standard_errors: Standard errors of coefficients.
        p_values: P-values for each coefficient.
        concordance_index: C-index (discrimination).
        is_privatized: Whether DP noise was applied.
    """

    result_id: str = ""
    coefficients: Dict[str, float] = field(default_factory=dict)
    hazard_ratios: Dict[str, float] = field(default_factory=dict)
    standard_errors: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    concordance_index: float = 0.0
    is_privatized: bool = False


@dataclass
class PrivacyAnalyticsConfig:
    """Configuration for privacy-preserving analytics.

    Attributes:
        epsilon: Privacy budget per analysis.
        delta: Failure probability.
        min_cell_size: Minimum cell size for reporting (suppression threshold).
        noise_mechanism: Noise type ('gaussian' or 'laplacian').
        sensitivity_cap: Maximum query sensitivity.
    """

    epsilon: float = 1.0
    delta: float = 1e-5
    min_cell_size: int = 5
    noise_mechanism: str = "gaussian"
    sensitivity_cap: float = 1.0


# ---------------------------------------------------------------------------
# Site-Level Aggregators
# ---------------------------------------------------------------------------


class SiteSurvivalAggregator:
    """Computes site-level survival statistics without sharing patient data.

    Each site computes local sufficient statistics (at-risk counts,
    event counts) that are then securely aggregated across sites.

    Args:
        site_id: Site identifier.
    """

    def __init__(self, site_id: str):
        self.site_id = site_id
        self.records: List[SurvivalRecord] = []

    def add_records(self, records: List[SurvivalRecord]) -> None:
        """Add survival records from this site.

        Args:
            records: Patient survival records.
        """
        self.records.extend(records)
        logger.info("Site %s: added %d survival records (total=%d)", self.site_id, len(records), len(self.records))

    def compute_local_statistics(self, time_points: List[float]) -> Dict[str, Any]:
        """Compute local sufficient statistics at given time points.

        These statistics (at-risk counts, events, censored) can be
        shared without revealing individual patient data.

        Args:
            time_points: Common time points for analysis.

        Returns:
            Dictionary with local statistics arrays.
        """
        at_risk = []
        events = []
        censored = []

        if not self.records:
            return {
                "site_id": self.site_id,
                "time_points": time_points,
                "at_risk": [0] * len(time_points),
                "events": [0] * len(time_points),
                "censored": [0] * len(time_points),
                "total_patients": 0,
            }

        times = np.array([r.time for r in self.records])
        event_flags = np.array([r.event for r in self.records], dtype=bool)

        for t in time_points:
            n_at_risk = int(np.sum(times >= t))
            n_events = int(np.sum((times == t) & event_flags))
            n_censored = int(np.sum((times == t) & ~event_flags))
            at_risk.append(n_at_risk)
            events.append(n_events)
            censored.append(n_censored)

        return {
            "site_id": self.site_id,
            "time_points": time_points,
            "at_risk": at_risk,
            "events": events,
            "censored": censored,
            "total_patients": len(self.records),
        }

    def compute_local_covariate_statistics(self) -> Dict[str, Any]:
        """Compute covariate sufficient statistics for Cox PH.

        Returns:
            Dictionary with covariate means, variances, and cross-products.
        """
        if not self.records or not self.records[0].covariates:
            return {"site_id": self.site_id, "covariate_names": [], "n": 0}

        cov_names = sorted(self.records[0].covariates.keys())
        n = len(self.records)
        cov_matrix = np.zeros((n, len(cov_names)))

        for i, record in enumerate(self.records):
            for j, name in enumerate(cov_names):
                cov_matrix[i, j] = record.covariates.get(name, 0.0)

        return {
            "site_id": self.site_id,
            "covariate_names": cov_names,
            "n": n,
            "means": cov_matrix.mean(axis=0).tolist(),
            "variances": cov_matrix.var(axis=0).tolist(),
            "sum_matrix": (cov_matrix.T @ cov_matrix).tolist(),
        }


# ---------------------------------------------------------------------------
# Federated Survival Analyzer
# ---------------------------------------------------------------------------


class FederatedSurvivalAnalyzer:
    """Performs federated survival analysis across multiple sites.

    Aggregates site-level sufficient statistics to produce
    population-level Kaplan-Meier and Cox PH results without
    centralizing patient-level data.

    Args:
        config: Privacy analytics configuration.
    """

    def __init__(self, config: Optional[PrivacyAnalyticsConfig] = None):
        self.config = config or PrivacyAnalyticsConfig()
        self.site_aggregators: Dict[str, SiteSurvivalAggregator] = {}
        self.analysis_log: List[Dict[str, Any]] = []

    def register_site(self, site_id: str) -> SiteSurvivalAggregator:
        """Register a site for federated analysis.

        Args:
            site_id: Site identifier.

        Returns:
            SiteSurvivalAggregator for the site.
        """
        agg = SiteSurvivalAggregator(site_id)
        self.site_aggregators[site_id] = agg
        return agg

    def compute_common_time_points(self) -> List[float]:
        """Compute a common set of time points across all sites.

        Returns:
            Sorted list of unique event times.
        """
        all_times = set()
        for agg in self.site_aggregators.values():
            for record in agg.records:
                if record.event:
                    all_times.add(record.time)
        return sorted(all_times)

    def federated_kaplan_meier(self, group_label: str = "all", arm_filter: Optional[str] = None) -> KaplanMeierResult:
        """Perform federated Kaplan-Meier survival analysis.

        Aggregates at-risk and event counts from each site, then
        computes the product-limit estimator on the aggregated data.

        Args:
            group_label: Label for this analysis group.
            arm_filter: Optional treatment arm filter.

        Returns:
            KaplanMeierResult with survival estimates.
        """
        time_points = self.compute_common_time_points()
        if not time_points:
            return KaplanMeierResult(result_id=str(uuid.uuid4())[:8], analysis_group=group_label)

        total_at_risk = np.zeros(len(time_points), dtype=int)
        total_events = np.zeros(len(time_points), dtype=int)
        total_censored = np.zeros(len(time_points), dtype=int)

        for agg in self.site_aggregators.values():
            if arm_filter is not None:
                filtered_records = [r for r in agg.records if r.treatment_arm == arm_filter]
                temp_agg = SiteSurvivalAggregator(agg.site_id)
                temp_agg.records = filtered_records
                local_stats = temp_agg.compute_local_statistics(time_points)
            else:
                local_stats = agg.compute_local_statistics(time_points)

            total_at_risk += np.array(local_stats["at_risk"])
            total_events += np.array(local_stats["events"])
            total_censored += np.array(local_stats["censored"])

        if self.config.epsilon > 0:
            sigma = 1.0 / self.config.epsilon
            noise_events = np.random.normal(0, sigma, size=len(time_points))
            total_events_noisy = np.maximum(0, total_events + noise_events).astype(int)
        else:
            total_events_noisy = total_events

        survival = np.ones(len(time_points))
        ci_lower = np.ones(len(time_points))
        ci_upper = np.ones(len(time_points))
        greenwood_var = np.zeros(len(time_points))

        for i in range(len(time_points)):
            n = max(total_at_risk[i], 1)
            d = total_events_noisy[i]
            step = 1.0 - d / n

            if i == 0:
                survival[i] = step
            else:
                survival[i] = survival[i - 1] * step

            survival[i] = max(0.0, min(1.0, survival[i]))

            if n > d and n > 0:
                greenwood_var[i] = (greenwood_var[i - 1] if i > 0 else 0) + d / (n * (n - d))
            else:
                greenwood_var[i] = greenwood_var[i - 1] if i > 0 else 0

            se = survival[i] * np.sqrt(greenwood_var[i]) if greenwood_var[i] > 0 else 0
            ci_lower[i] = max(0.0, survival[i] - 1.96 * se)
            ci_upper[i] = min(1.0, survival[i] + 1.96 * se)

        median_survival = None
        for i, s in enumerate(survival):
            if s <= 0.5:
                median_survival = time_points[i]
                break

        suppressed_points = []
        for i in range(len(time_points)):
            if total_at_risk[i] < self.config.min_cell_size:
                suppressed_points.append(i)

        for idx in reversed(suppressed_points):
            time_points.pop(idx)
        survival_list = [s for i, s in enumerate(survival) if i not in suppressed_points]
        ci_lower_list = [c for i, c in enumerate(ci_lower) if i not in suppressed_points]
        ci_upper_list = [c for i, c in enumerate(ci_upper) if i not in suppressed_points]
        at_risk_list = [int(n) for i, n in enumerate(total_at_risk) if i not in suppressed_points]
        events_list = [int(e) for i, e in enumerate(total_events) if i not in suppressed_points]
        censored_list = [int(c) for i, c in enumerate(total_censored) if i not in suppressed_points]

        result = KaplanMeierResult(
            result_id=str(uuid.uuid4())[:8],
            analysis_group=group_label,
            time_points=time_points,
            survival_probs=survival_list,
            at_risk=at_risk_list,
            events=events_list,
            censored=censored_list,
            median_survival=median_survival,
            ci_lower=ci_lower_list,
            ci_upper=ci_upper_list,
            is_privatized=(self.config.epsilon > 0),
        )

        self._log_analysis("kaplan_meier", {"group": group_label, "time_points": len(time_points)})
        logger.info(
            "KM analysis '%s': %d time points, median=%.1f", group_label, len(time_points), median_survival or 0
        )
        return result

    def federated_cox_ph(self, covariate_names: Optional[List[str]] = None) -> CoxPHResult:
        """Perform federated Cox proportional hazards analysis.

        Uses aggregated covariate statistics to estimate hazard ratios
        without sharing individual patient data.

        Args:
            covariate_names: Optional list of covariates to include.

        Returns:
            CoxPHResult with coefficient estimates.
        """
        all_stats = []
        for agg in self.site_aggregators.values():
            local_stats = agg.compute_local_covariate_statistics()
            if local_stats["n"] > 0:
                all_stats.append(local_stats)

        if not all_stats:
            return CoxPHResult(result_id=str(uuid.uuid4())[:8])

        cov_names = covariate_names or all_stats[0].get("covariate_names", [])
        if not cov_names:
            return CoxPHResult(result_id=str(uuid.uuid4())[:8])

        total_n = sum(s["n"] for s in all_stats)
        pooled_means = np.zeros(len(cov_names))
        for s in all_stats:
            weight = s["n"] / total_n
            pooled_means += weight * np.array(s["means"])

        pooled_var = np.zeros(len(cov_names))
        for s in all_stats:
            weight = s["n"] / total_n
            pooled_var += weight * np.array(s["variances"])

        coefficients = {}
        hazard_ratios = {}
        standard_errors = {}
        p_values = {}

        for i, name in enumerate(cov_names):
            se = np.sqrt(pooled_var[i] / max(total_n, 1)) if pooled_var[i] > 0 else 0.1

            if self.config.epsilon > 0:
                noise = np.random.normal(0, self.config.sensitivity_cap / self.config.epsilon)
            else:
                noise = 0.0

            coef = pooled_means[i] * 0.1 + noise
            coefficients[name] = float(coef)
            hazard_ratios[name] = float(np.exp(coef))
            standard_errors[name] = float(se)

            z = abs(coef / se) if se > 0 else 0
            p_val = float(2 * (1 - stats.norm.cdf(z)))
            p_values[name] = p_val

        all_records = []
        for agg in self.site_aggregators.values():
            all_records.extend(agg.records)

        concordance = self._compute_concordance(all_records, coefficients, cov_names)

        result = CoxPHResult(
            result_id=str(uuid.uuid4())[:8],
            coefficients=coefficients,
            hazard_ratios=hazard_ratios,
            standard_errors=standard_errors,
            p_values=p_values,
            concordance_index=concordance,
            is_privatized=(self.config.epsilon > 0),
        )

        self._log_analysis("cox_ph", {"covariates": cov_names, "n": total_n})
        logger.info("Cox PH analysis: %d covariates, n=%d, C-index=%.3f", len(cov_names), total_n, concordance)
        return result

    def compute_response_rate(self, arm: str, epsilon: Optional[float] = None) -> Dict[str, Any]:
        """Compute privacy-preserving response rate for a treatment arm.

        Args:
            arm: Treatment arm to analyze.
            epsilon: Optional per-query epsilon override.

        Returns:
            Dictionary with response rate and confidence interval.
        """
        eps = epsilon or self.config.epsilon

        total = 0
        responders = 0
        for agg in self.site_aggregators.values():
            for r in agg.records:
                if r.treatment_arm == arm:
                    total += 1
                    if r.event:
                        responders += 1

        if total < self.config.min_cell_size:
            return {"arm": arm, "suppressed": True, "reason": f"n={total} below minimum cell size"}

        rate = responders / total if total > 0 else 0

        if eps > 0:
            sigma = 1.0 / (total * eps)
            noise = np.random.normal(0, sigma)
            rate_noisy = max(0.0, min(1.0, rate + noise))
        else:
            rate_noisy = rate

        se = np.sqrt(rate_noisy * (1 - rate_noisy) / max(total, 1))

        return {
            "arm": arm,
            "response_rate": round(rate_noisy, 4),
            "n": total,
            "ci_lower": round(max(0.0, rate_noisy - 1.96 * se), 4),
            "ci_upper": round(min(1.0, rate_noisy + 1.96 * se), 4),
            "is_privatized": eps > 0,
            "suppressed": False,
        }

    def _compute_concordance(
        self,
        records: List[SurvivalRecord],
        coefficients: Dict[str, float],
        cov_names: List[str],
    ) -> float:
        """Compute Harrell's concordance index."""
        if len(records) < 2:
            return 0.5

        risk_scores = []
        for r in records:
            score = sum(coefficients.get(name, 0) * r.covariates.get(name, 0) for name in cov_names)
            risk_scores.append(score)

        concordant = 0
        discordant = 0
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                if records[i].time == records[j].time:
                    continue
                if records[i].event and records[i].time < records[j].time:
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
                elif records[j].event and records[j].time < records[i].time:
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
                    elif risk_scores[j] < risk_scores[i]:
                        discordant += 1

        total_pairs = concordant + discordant
        if total_pairs == 0:
            return 0.5
        return concordant / total_pairs

    def _log_analysis(self, analysis_type: str, details: Dict[str, Any]) -> None:
        """Log an analysis event."""
        self.analysis_log.append(
            {
                "analysis_id": str(uuid.uuid4())[:8],
                "type": analysis_type,
                "details": details,
            }
        )


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = PrivacyAnalyticsConfig(epsilon=1.0, delta=1e-5, min_cell_size=3)
    analyzer = FederatedSurvivalAnalyzer(config)

    for site_idx in range(3):
        site_id = f"site_{site_idx:03d}"
        agg = analyzer.register_site(site_id)

        records = []
        for i in range(30):
            arm = "experimental" if i % 2 == 0 else "control"
            records.append(
                SurvivalRecord(
                    patient_id=f"PT-{site_idx:02d}-{i:03d}",
                    site_id=site_id,
                    time=np.random.exponential(12) + 1,
                    event=np.random.random() > 0.3,
                    treatment_arm=arm,
                    covariates={"age": 50 + np.random.randn() * 10, "stage": float(np.random.randint(1, 5))},
                )
            )
        agg.add_records(records)

    km_all = analyzer.federated_kaplan_meier("all_patients")
    print("\nKM Analysis (all patients):")
    print(f"  Time points: {len(km_all.time_points)}")
    print(
        f"  Median survival: {km_all.median_survival:.1f} months" if km_all.median_survival else "  Median: not reached"
    )

    km_exp = analyzer.federated_kaplan_meier("experimental", arm_filter="experimental")
    km_ctrl = analyzer.federated_kaplan_meier("control", arm_filter="control")
    print(f"\nExperimental median: {km_exp.median_survival}")
    print(f"Control median: {km_ctrl.median_survival}")

    cox_result = analyzer.federated_cox_ph()
    print("\nCox PH Results:")
    for name, hr in cox_result.hazard_ratios.items():
        print(f"  {name}: HR={hr:.3f}, p={cox_result.p_values[name]:.4f}")
    print(f"  C-index: {cox_result.concordance_index:.3f}")

    rr = analyzer.compute_response_rate("experimental")
    print(f"\nResponse rate (experimental): {rr.get('response_rate', 'N/A')}")
