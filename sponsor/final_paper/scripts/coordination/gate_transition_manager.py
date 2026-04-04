from __future__ import annotations

"""Seven-gate decision framework for autonomous sponsor simulation.

Each program gate evaluates go-criteria, computes KPI scores, and
produces one of four outcomes: advance, conditional, hold, or terminate.
Gate definitions mirror the autonomous-sponsor lifecycle described in
the companion paper.

Gates
-----
1. Preclinical validation      2. IND/CTA readiness
3. First-patient-in            4. Interim analysis
5. Data lock                   6. Regulatory submission
7. Post-approval commitment
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class GateID(IntEnum):
    """Canonical gate identifiers (1-based)."""

    PRECLINICAL_VALIDATION = 1
    IND_CTA_READINESS = 2
    FIRST_PATIENT_IN = 3
    INTERIM_ANALYSIS = 4
    DATA_LOCK = 5
    REGULATORY_SUBMISSION = 6
    POST_APPROVAL_COMMITMENT = 7


class GateOutcome(str, Enum):
    """Possible outcomes of a gate evaluation."""

    ADVANCE = "advance"
    CONDITIONAL = "conditional"
    HOLD = "hold"
    TERMINATE = "terminate"


# ---------------------------------------------------------------------------
# KPI specification per gate
# ---------------------------------------------------------------------------

_KPI_SPEC: dict[GateID, list[dict[str, object]]] = {
    GateID.PRECLINICAL_VALIDATION: [
        {"name": "efficacy_signal", "weight": 0.30, "threshold": 0.60},
        {"name": "toxicity_profile", "weight": 0.25, "threshold": 0.70},
        {"name": "manufacturing_feasibility", "weight": 0.20, "threshold": 0.65},
        {"name": "regulatory_alignment", "weight": 0.15, "threshold": 0.60},
        {"name": "ip_freedom_to_operate", "weight": 0.10, "threshold": 0.50},
    ],
    GateID.IND_CTA_READINESS: [
        {"name": "cmc_package_completeness", "weight": 0.25, "threshold": 0.80},
        {"name": "nonclinical_package", "weight": 0.25, "threshold": 0.75},
        {"name": "protocol_finalization", "weight": 0.20, "threshold": 0.85},
        {"name": "irb_ethics_submission", "weight": 0.15, "threshold": 0.70},
        {"name": "site_readiness", "weight": 0.15, "threshold": 0.65},
    ],
    GateID.FIRST_PATIENT_IN: [
        {"name": "site_activation_rate", "weight": 0.25, "threshold": 0.60},
        {"name": "enrollment_pipeline", "weight": 0.25, "threshold": 0.50},
        {"name": "supply_chain_readiness", "weight": 0.20, "threshold": 0.75},
        {"name": "monitoring_plan_ready", "weight": 0.15, "threshold": 0.80},
        {"name": "robot_system_qualified", "weight": 0.15, "threshold": 0.70},
    ],
    GateID.INTERIM_ANALYSIS: [
        {"name": "enrollment_target_pct", "weight": 0.20, "threshold": 0.70},
        {"name": "data_quality_score", "weight": 0.20, "threshold": 0.80},
        {"name": "safety_profile_acceptable", "weight": 0.25, "threshold": 0.75},
        {"name": "efficacy_trend", "weight": 0.20, "threshold": 0.55},
        {"name": "operational_efficiency", "weight": 0.15, "threshold": 0.65},
    ],
    GateID.DATA_LOCK: [
        {"name": "query_resolution_rate", "weight": 0.25, "threshold": 0.95},
        {"name": "protocol_deviations_resolved", "weight": 0.20, "threshold": 0.90},
        {"name": "sae_reconciliation", "weight": 0.20, "threshold": 0.95},
        {"name": "database_audit_ready", "weight": 0.20, "threshold": 0.90},
        {"name": "statistical_analysis_plan_locked", "weight": 0.15, "threshold": 0.85},
    ],
    GateID.REGULATORY_SUBMISSION: [
        {"name": "csr_completeness", "weight": 0.25, "threshold": 0.90},
        {"name": "safety_update_report", "weight": 0.20, "threshold": 0.90},
        {"name": "ectd_module_readiness", "weight": 0.20, "threshold": 0.85},
        {"name": "labeling_proposal", "weight": 0.15, "threshold": 0.80},
        {"name": "advisory_committee_prep", "weight": 0.20, "threshold": 0.75},
    ],
    GateID.POST_APPROVAL_COMMITMENT: [
        {"name": "rems_program_ready", "weight": 0.20, "threshold": 0.80},
        {"name": "pharmacovigilance_system", "weight": 0.25, "threshold": 0.85},
        {"name": "post_marketing_study_plan", "weight": 0.20, "threshold": 0.75},
        {"name": "manufacturing_scale_up", "weight": 0.20, "threshold": 0.80},
        {"name": "commercial_readiness", "weight": 0.15, "threshold": 0.70},
    ],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KPIScore:
    """A single KPI value with its evaluation result."""

    name: str
    value: float
    weight: float
    threshold: float
    passed: bool
    weighted_score: float


@dataclass(frozen=True)
class GateDecision:
    """Output of a gate evaluation."""

    decision_id: str
    gate_id: int
    gate_name: str
    outcome: GateOutcome
    composite_score: float
    kpi_scores: list[KPIScore]
    passed_count: int
    total_count: int
    rationale: str
    next_actions: list[str]
    evaluated_at: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "gate_id": self.gate_id,
            "gate_name": self.gate_name,
            "outcome": self.outcome.value,
            "composite_score": round(self.composite_score, 4),
            "kpi_scores": [
                {
                    "name": k.name,
                    "value": round(k.value, 4),
                    "weight": k.weight,
                    "threshold": k.threshold,
                    "passed": k.passed,
                    "weighted_score": round(k.weighted_score, 4),
                }
                for k in self.kpi_scores
            ],
            "passed_count": self.passed_count,
            "total_count": self.total_count,
            "rationale": self.rationale,
            "next_actions": self.next_actions,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Gate Transition Manager
# ---------------------------------------------------------------------------

# Outcome thresholds
_ADVANCE_THRESHOLD = 0.80
_CONDITIONAL_THRESHOLD = 0.60
_HOLD_THRESHOLD = 0.40


class GateTransitionManager:
    """Evaluate program gates and produce structured decisions.

    Usage::

        mgr = GateTransitionManager()
        kpi_values = {"efficacy_signal": 0.82, "toxicity_profile": 0.91, ...}
        decision = mgr.evaluate_gate(GateID.PRECLINICAL_VALIDATION, kpi_values)
    """

    def __init__(self) -> None:
        self._decisions: list[GateDecision] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_gate(
        self,
        gate_id: int,
        kpi_values: dict[str, float],
        program_data: dict[str, object] | None = None,
    ) -> GateDecision:
        """Evaluate *gate_id* against the supplied KPI values.

        Parameters
        ----------
        gate_id:
            Integer gate identifier (1-7).
        kpi_values:
            Mapping of KPI name to its measured value (0.0-1.0).
        program_data:
            Optional context dict (logged but not used for scoring).

        Returns
        -------
        GateDecision with outcome, rationale, and recommended next actions.
        """
        gate_enum = GateID(gate_id)
        specs = _KPI_SPEC[gate_enum]

        kpi_scores = self._score_kpis(specs, kpi_values)
        composite = sum(k.weighted_score for k in kpi_scores)
        passed = [k for k in kpi_scores if k.passed]
        failed = [k for k in kpi_scores if not k.passed]

        outcome = self._determine_outcome(composite, len(passed), len(kpi_scores), failed)
        rationale = self._build_rationale(gate_enum, composite, passed, failed, outcome)
        next_actions = self._determine_next_actions(gate_enum, outcome, failed, program_data)

        decision = GateDecision(
            decision_id=uuid.uuid4().hex[:12],
            gate_id=gate_id,
            gate_name=gate_enum.name.replace("_", " ").title(),
            outcome=outcome,
            composite_score=composite,
            kpi_scores=kpi_scores,
            passed_count=len(passed),
            total_count=len(kpi_scores),
            rationale=rationale,
            next_actions=next_actions,
            evaluated_at=datetime.now(timezone.utc),
        )
        self._decisions.append(decision)
        logger.info(
            "Gate %d (%s): %s (score=%.3f, %d/%d KPIs passed)",
            gate_id,
            gate_enum.name,
            outcome.value,
            composite,
            len(passed),
            len(kpi_scores),
        )
        return decision

    def decision_history(self, gate_id: int | None = None, limit: int = 50) -> list[GateDecision]:
        """Return recent gate decisions, optionally filtered by gate."""
        source = list(reversed(self._decisions))
        if gate_id is not None:
            source = [d for d in source if d.gate_id == gate_id]
        return source[:limit]

    def outcome_summary(self) -> dict[str, int]:
        """Return counts of each outcome across all evaluated gates."""
        counts: dict[str, int] = {o.value: 0 for o in GateOutcome}
        for d in self._decisions:
            counts[d.outcome.value] += 1
        return counts

    @staticmethod
    def gate_names() -> dict[int, str]:
        """Return mapping of gate id to human-readable name."""
        return {g.value: g.name.replace("_", " ").title() for g in GateID}

    @staticmethod
    def kpi_spec(gate_id: int) -> list[dict[str, object]]:
        """Return the KPI specification for a given gate."""
        return list(_KPI_SPEC[GateID(gate_id)])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_kpis(
        specs: list[dict[str, object]],
        kpi_values: dict[str, float],
    ) -> list[KPIScore]:
        scores: list[KPIScore] = []
        for spec in specs:
            name = str(spec["name"])
            weight = float(spec["weight"])  # type: ignore[arg-type]
            threshold = float(spec["threshold"])  # type: ignore[arg-type]
            value = float(kpi_values.get(name, 0.0))
            value = max(0.0, min(1.0, value))
            passed = value >= threshold
            scores.append(
                KPIScore(
                    name=name,
                    value=value,
                    weight=weight,
                    threshold=threshold,
                    passed=passed,
                    weighted_score=value * weight,
                )
            )
        return scores

    @staticmethod
    def _determine_outcome(
        composite: float,
        passed_count: int,
        total_count: int,
        failed: list[KPIScore],
    ) -> GateOutcome:
        pass_ratio = passed_count / total_count if total_count > 0 else 0.0

        if composite >= _ADVANCE_THRESHOLD and pass_ratio == 1.0:
            return GateOutcome.ADVANCE
        if composite >= _CONDITIONAL_THRESHOLD and pass_ratio >= 0.6:
            return GateOutcome.CONDITIONAL
        if composite >= _HOLD_THRESHOLD:
            # Check for critical failures
            critical_failures = [f for f in failed if f.weight >= 0.25]
            if critical_failures:
                return GateOutcome.HOLD
            return GateOutcome.HOLD
        return GateOutcome.TERMINATE

    @staticmethod
    def _build_rationale(
        gate: GateID,
        composite: float,
        passed: list[KPIScore],
        failed: list[KPIScore],
        outcome: GateOutcome,
    ) -> str:
        parts = [
            f"Gate {gate.value} ({gate.name}): composite={composite:.3f},",
            f"{len(passed)}/{len(passed) + len(failed)} KPIs passed.",
        ]
        if failed:
            fail_names = ", ".join(f.name for f in failed)
            parts.append(f"Failed: {fail_names}.")
        parts.append(f"Outcome: {outcome.value}.")
        return " ".join(parts)

    @staticmethod
    def _determine_next_actions(
        gate: GateID,
        outcome: GateOutcome,
        failed: list[KPIScore],
        program_data: dict[str, object] | None,
    ) -> list[str]:
        actions: list[str] = []

        if outcome == GateOutcome.ADVANCE:
            next_gate = gate.value + 1
            if next_gate <= 7:
                next_name = GateID(next_gate).name.replace("_", " ").title()
                actions.append(f"Proceed to Gate {next_gate}: {next_name}")
            else:
                actions.append("Program lifecycle complete - transition to post-market")
            actions.append("Archive gate decision and notify stakeholders")

        elif outcome == GateOutcome.CONDITIONAL:
            for f in failed:
                actions.append(f"Remediate KPI '{f.name}' (current={f.value:.2f}, threshold={f.threshold:.2f})")
            actions.append("Schedule conditional review within 30 days")
            actions.append("Notify risk management of conditional advance")

        elif outcome == GateOutcome.HOLD:
            for f in failed:
                actions.append(f"Develop corrective action plan for '{f.name}'")
            actions.append("Escalate to sponsor oversight committee")
            actions.append("Schedule hold review within 60 days")
            actions.append("Assess impact on program timeline and budget")

        elif outcome == GateOutcome.TERMINATE:
            actions.append("Initiate program termination review")
            actions.append("Notify regulatory affairs of potential study closure")
            actions.append("Prepare close-out documentation")
            actions.append("Assess portfolio reallocation options")

        if program_data and program_data.get("urgent"):
            actions.insert(0, "URGENT: Expedite review per program flag")

        return actions
