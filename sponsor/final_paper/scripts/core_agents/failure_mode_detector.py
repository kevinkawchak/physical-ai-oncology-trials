"""Failure mode detector: real-time detection of 5 failure modes."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class FailureMode(Enum):
    ENROLLMENT_STALL = "enrollment_stall"
    PROTOCOL_DEVIATION_SURGE = "protocol_deviation_surge"
    DATA_QUALITY_DEGRADATION = "data_quality_degradation"
    SAFETY_SIGNAL_EMERGENCE = "safety_signal_emergence"
    SITE_PERFORMANCE_DECLINE = "site_performance_decline"


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class FailureAlert:
    alert_id: str
    mode: FailureMode
    severity: Severity
    site_id: str | None
    description: str
    metric_value: float
    threshold: float
    detected_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False

    @property
    def exceedance_ratio(self) -> float:
        if self.threshold == 0:
            return 0.0
        return self.metric_value / self.threshold


@dataclass
class SiteMetrics:
    site_id: str
    enrollment_rate_monthly: float = 0.0
    deviation_rate: float = 0.0
    query_rate_per_page: float = 0.0
    sae_rate: float = 0.0
    screen_fail_rate: float = 0.0
    missing_data_pct: float = 0.0


@dataclass
class DetectionThresholds:
    enrollment_rate_min: float = 1.0
    deviation_rate_max: float = 0.10
    query_rate_max: float = 0.15
    sae_rate_max: float = 0.05
    screen_fail_rate_max: float = 0.50
    missing_data_max: float = 0.05


class FailureModeDetector:
    """Detects five failure modes in real-time from site metrics."""

    def __init__(self, thresholds: DetectionThresholds | None = None) -> None:
        self.thresholds = thresholds or DetectionThresholds()
        self.alerts: list[FailureAlert] = []
        self._alert_counter = 0

    def _make_alert(self, mode: FailureMode, severity: Severity, site_id: str, desc: str,
                    value: float, threshold: float) -> FailureAlert:
        self._alert_counter += 1
        alert = FailureAlert(
            alert_id=f"FA-{self._alert_counter:05d}", mode=mode, severity=severity,
            site_id=site_id, description=desc, metric_value=value, threshold=threshold,
        )
        self.alerts.append(alert)
        return alert

    def check_enrollment_stall(self, metrics: SiteMetrics) -> FailureAlert | None:
        if metrics.enrollment_rate_monthly < self.thresholds.enrollment_rate_min:
            sev = Severity.CRITICAL if metrics.enrollment_rate_monthly == 0 else Severity.WARNING
            return self._make_alert(
                FailureMode.ENROLLMENT_STALL, sev, metrics.site_id,
                f"Enrollment rate {metrics.enrollment_rate_monthly:.1f}/mo below minimum",
                metrics.enrollment_rate_monthly, self.thresholds.enrollment_rate_min,
            )
        return None

    def check_deviation_surge(self, metrics: SiteMetrics) -> FailureAlert | None:
        if metrics.deviation_rate > self.thresholds.deviation_rate_max:
            sev = Severity.CRITICAL if metrics.deviation_rate > 0.20 else Severity.WARNING
            return self._make_alert(
                FailureMode.PROTOCOL_DEVIATION_SURGE, sev, metrics.site_id,
                f"Deviation rate {metrics.deviation_rate:.1%} exceeds threshold",
                metrics.deviation_rate, self.thresholds.deviation_rate_max,
            )
        return None

    def check_data_quality(self, metrics: SiteMetrics) -> FailureAlert | None:
        if metrics.query_rate_per_page > self.thresholds.query_rate_max:
            return self._make_alert(
                FailureMode.DATA_QUALITY_DEGRADATION, Severity.WARNING, metrics.site_id,
                f"Query rate {metrics.query_rate_per_page:.1%} per page exceeds threshold",
                metrics.query_rate_per_page, self.thresholds.query_rate_max,
            )
        return None

    def check_safety_signal(self, metrics: SiteMetrics) -> FailureAlert | None:
        if metrics.sae_rate > self.thresholds.sae_rate_max:
            return self._make_alert(
                FailureMode.SAFETY_SIGNAL_EMERGENCE, Severity.CRITICAL, metrics.site_id,
                f"SAE rate {metrics.sae_rate:.1%} above threshold",
                metrics.sae_rate, self.thresholds.sae_rate_max,
            )
        return None

    def check_site_performance(self, metrics: SiteMetrics) -> FailureAlert | None:
        if metrics.screen_fail_rate > self.thresholds.screen_fail_rate_max:
            return self._make_alert(
                FailureMode.SITE_PERFORMANCE_DECLINE, Severity.WARNING, metrics.site_id,
                f"Screen fail rate {metrics.screen_fail_rate:.1%} exceeds threshold",
                metrics.screen_fail_rate, self.thresholds.screen_fail_rate_max,
            )
        return None

    def run_all_checks(self, metrics: SiteMetrics) -> list[FailureAlert]:
        results: list[FailureAlert] = []
        for check in [self.check_enrollment_stall, self.check_deviation_surge,
                       self.check_data_quality, self.check_safety_signal, self.check_site_performance]:
            alert = check(metrics)
            if alert:
                results.append(alert)
        return results

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for a in self.alerts:
            key = a.mode.value
            counts[key] = counts.get(key, 0) + 1
        return counts


def main() -> None:
    detector = FailureModeDetector()
    metrics = SiteMetrics("S01", enrollment_rate_monthly=0.5, deviation_rate=0.15, query_rate_per_page=0.08,
                          sae_rate=0.07, screen_fail_rate=0.35)
    alerts = detector.run_all_checks(metrics)
    for a in alerts:
        print(f"[{a.severity.value}] {a.mode.value}: {a.description}")
    print(f"Summary: {detector.summary()}")


if __name__ == "__main__":
    main()
