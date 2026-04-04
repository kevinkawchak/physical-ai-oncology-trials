from __future__ import annotations

"""Four-gate safety authorization protocol for robotic procedures.

Every robotic procedure must pass through four sequential authorization
gates before execution may begin:

1. Patient verification  - Consent validity, identity confirmation.
2. System readiness      - Calibration, instruments, alarm checks.
3. Parameter validation  - Safety envelopes, digital-twin concordance.
4. Clinical authorization - Tier 2/3 clinician sign-off.

All four gates must pass for a procedure to be authorized.  Any gate
failure blocks the procedure and triggers the appropriate remediation
workflow.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum, Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SafetyGateID(IntEnum):
    """Sequential safety authorization gates."""

    PATIENT_VERIFICATION = 1
    SYSTEM_READINESS = 2
    PARAMETER_VALIDATION = 3
    CLINICAL_AUTHORIZATION = 4


class GateStatus(str, Enum):
    """Evaluation status of a single gate."""

    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"
    WAIVED = "waived"


class ClinicalTier(str, Enum):
    """Clinician authorization tiers."""

    TIER_1 = "tier_1"  # Nurse / technician (informational only)
    TIER_2 = "tier_2"  # Attending physician
    TIER_3 = "tier_3"  # Principal investigator / medical director


# ---------------------------------------------------------------------------
# Check specifications per gate
# ---------------------------------------------------------------------------

_GATE_CHECKS: dict[SafetyGateID, list[dict[str, object]]] = {
    SafetyGateID.PATIENT_VERIFICATION: [
        {"check": "informed_consent_valid", "required": True, "description": "Signed ICF on file and not expired"},
        {"check": "patient_identity_confirmed", "required": True, "description": "Two-factor identity match"},
        {"check": "eligibility_criteria_met", "required": True, "description": "Inclusion/exclusion criteria current"},
        {"check": "no_active_withdrawal", "required": True, "description": "Patient has not withdrawn consent"},
        {
            "check": "procedure_scheduled",
            "required": False,
            "description": "Procedure appears on approved schedule",
        },
    ],
    SafetyGateID.SYSTEM_READINESS: [
        {"check": "calibration_current", "required": True, "description": "Robot calibration within 24h window"},
        {"check": "instrument_verification", "required": True, "description": "All instruments present and sterile"},
        {"check": "alarm_system_functional", "required": True, "description": "Safety alarms tested and active"},
        {"check": "communication_link_stable", "required": True, "description": "Telemetry uplink latency < 50ms"},
        {"check": "backup_power_available", "required": True, "description": "UPS charged and failover tested"},
        {
            "check": "environment_controlled",
            "required": False,
            "description": "OR temperature and humidity within range",
        },
    ],
    SafetyGateID.PARAMETER_VALIDATION: [
        {
            "check": "safety_envelope_configured",
            "required": True,
            "description": "Force/torque/velocity limits loaded",
        },
        {"check": "digital_twin_concordance", "required": True, "description": "Digital twin deviation < 2%"},
        {
            "check": "path_plan_validated",
            "required": True,
            "description": "Planned trajectory collision-free in simulation",
        },
        {
            "check": "dose_parameters_verified",
            "required": False,
            "description": "Radiation/drug dose matches prescription",
        },
        {
            "check": "anatomical_registration",
            "required": True,
            "description": "Patient anatomy registered to robot frame",
        },
    ],
    SafetyGateID.CLINICAL_AUTHORIZATION: [
        {
            "check": "clinician_sign_off",
            "required": True,
            "description": "Tier 2 or Tier 3 clinician has authorized",
        },
        {
            "check": "anesthesia_clearance",
            "required": False,
            "description": "Anesthesia team confirms patient readiness",
        },
        {
            "check": "nursing_checklist_complete",
            "required": False,
            "description": "Pre-procedure nursing checklist signed",
        },
        {
            "check": "time_out_performed",
            "required": True,
            "description": "Surgical time-out completed with full team",
        },
    ],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single authorization check."""

    check_name: str
    description: str
    required: bool
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateEvaluation:
    """Outcome of evaluating one authorization gate."""

    gate_id: int
    gate_name: str
    status: GateStatus
    checks: list[CheckResult]
    passed_required: int
    total_required: int
    passed_optional: int
    total_optional: int
    failure_reasons: list[str]


@dataclass(frozen=True)
class AuthorizationResult:
    """Final authorization decision for a procedure."""

    authorization_id: str
    procedure_id: str
    robot_id: str
    patient_id: str
    authorized: bool
    gate_evaluations: list[GateEvaluation]
    blocking_gates: list[int]
    clinician_tier: str
    clinician_id: str
    rationale: str
    evaluated_at: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "authorization_id": self.authorization_id,
            "procedure_id": self.procedure_id,
            "robot_id": self.robot_id,
            "patient_id": self.patient_id,
            "authorized": self.authorized,
            "gate_evaluations": [
                {
                    "gate_id": g.gate_id,
                    "gate_name": g.gate_name,
                    "status": g.status.value,
                    "passed_required": g.passed_required,
                    "total_required": g.total_required,
                    "passed_optional": g.passed_optional,
                    "total_optional": g.total_optional,
                    "failure_reasons": g.failure_reasons,
                }
                for g in self.gate_evaluations
            ],
            "blocking_gates": self.blocking_gates,
            "clinician_tier": self.clinician_tier,
            "clinician_id": self.clinician_id,
            "rationale": self.rationale,
            "authorized": self.authorized,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Procedure Authorization Gate Engine
# ---------------------------------------------------------------------------


class ProcedureAuthorizationGate:
    """Evaluate the four-gate safety authorization protocol.

    Usage::

        auth = ProcedureAuthorizationGate()
        check_data = {
            "informed_consent_valid": True,
            "patient_identity_confirmed": True,
            ...
        }
        result = auth.evaluate(
            procedure_id="PROC-042",
            robot_id="SR-001",
            patient_id="PT-1234",
            check_values=check_data,
            clinician_tier="tier_2",
            clinician_id="DR-SMITH",
        )
    """

    def __init__(self) -> None:
        self._history: list[AuthorizationResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        procedure_id: str,
        robot_id: str,
        patient_id: str,
        check_values: dict[str, bool | str],
        clinician_tier: str,
        clinician_id: str,
    ) -> AuthorizationResult:
        """Run all four gates and produce an authorization decision.

        Parameters
        ----------
        procedure_id:
            Unique procedure identifier.
        robot_id:
            Robot performing the procedure.
        patient_id:
            Patient receiving the procedure.
        check_values:
            Mapping of check name to pass/fail boolean or detail string.
            A truthy value means the check passed.
        clinician_tier:
            ``"tier_1"``, ``"tier_2"``, or ``"tier_3"``.
        clinician_id:
            Identifier of the authorizing clinician.
        """
        gate_evals: list[GateEvaluation] = []
        blocking: list[int] = []

        for gate_id in SafetyGateID:
            evaluation = self._evaluate_gate(gate_id, check_values)
            gate_evals.append(evaluation)
            if evaluation.status == GateStatus.FAILED:
                blocking.append(gate_id.value)

        # Clinical tier must be 2 or 3 for authorization
        tier_sufficient = clinician_tier in (ClinicalTier.TIER_2.value, ClinicalTier.TIER_3.value)
        if not tier_sufficient:
            blocking.append(SafetyGateID.CLINICAL_AUTHORIZATION.value)

        authorized = len(blocking) == 0 and tier_sufficient
        rationale = self._build_rationale(authorized, gate_evals, blocking, clinician_tier)

        result = AuthorizationResult(
            authorization_id=uuid.uuid4().hex[:12],
            procedure_id=procedure_id,
            robot_id=robot_id,
            patient_id=patient_id,
            authorized=authorized,
            gate_evaluations=gate_evals,
            blocking_gates=sorted(set(blocking)),
            clinician_tier=clinician_tier,
            clinician_id=clinician_id,
            rationale=rationale,
            evaluated_at=datetime.now(timezone.utc),
        )
        self._history.append(result)
        logger.info(
            "Procedure %s authorization: %s (blocking gates: %s)",
            procedure_id,
            "AUTHORIZED" if authorized else "DENIED",
            blocking or "none",
        )
        return result

    def authorization_history(self, limit: int = 50) -> list[AuthorizationResult]:
        """Return recent authorization results."""
        return list(reversed(self._history[-limit:]))

    def authorization_rate(self) -> float:
        """Return the fraction of procedures that were authorized."""
        if not self._history:
            return 0.0
        return sum(1 for r in self._history if r.authorized) / len(self._history)

    def most_common_blockers(self, limit: int = 10) -> list[tuple[str, int]]:
        """Return the most frequent blocking check names."""
        blocker_counts: dict[str, int] = {}
        for result in self._history:
            for gate_eval in result.gate_evaluations:
                for reason in gate_eval.failure_reasons:
                    blocker_counts[reason] = blocker_counts.get(reason, 0) + 1
        sorted_blockers = sorted(blocker_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_blockers[:limit]

    @staticmethod
    def gate_checks(gate_id: int) -> list[dict[str, object]]:
        """Return the check specification for a given gate."""
        return list(_GATE_CHECKS[SafetyGateID(gate_id)])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_gate(
        gate_id: SafetyGateID,
        check_values: dict[str, bool | str],
    ) -> GateEvaluation:
        specs = _GATE_CHECKS[gate_id]
        checks: list[CheckResult] = []
        failure_reasons: list[str] = []
        passed_req = 0
        total_req = 0
        passed_opt = 0
        total_opt = 0

        for spec in specs:
            name = str(spec["check"])
            required = bool(spec["required"])
            description = str(spec["description"])
            raw_value = check_values.get(name)

            if isinstance(raw_value, str):
                passed = raw_value.lower() in ("true", "yes", "pass", "1")
                detail = raw_value
            else:
                passed = bool(raw_value) if raw_value is not None else False
                detail = "passed" if passed else "not confirmed"

            checks.append(CheckResult(
                check_name=name,
                description=description,
                required=required,
                passed=passed,
                detail=detail,
            ))

            if required:
                total_req += 1
                if passed:
                    passed_req += 1
                else:
                    failure_reasons.append(f"{name}: {description}")
            else:
                total_opt += 1
                if passed:
                    passed_opt += 1

        status = GateStatus.PASSED if passed_req == total_req else GateStatus.FAILED

        return GateEvaluation(
            gate_id=gate_id.value,
            gate_name=gate_id.name.replace("_", " ").title(),
            status=status,
            checks=checks,
            passed_required=passed_req,
            total_required=total_req,
            passed_optional=passed_opt,
            total_optional=total_opt,
            failure_reasons=failure_reasons,
        )

    @staticmethod
    def _build_rationale(
        authorized: bool,
        gate_evals: list[GateEvaluation],
        blocking: list[int],
        clinician_tier: str,
    ) -> str:
        passed_gates = [g for g in gate_evals if g.status == GateStatus.PASSED]
        if authorized:
            return (
                f"All {len(passed_gates)} gates passed. "
                f"Clinician tier ({clinician_tier}) sufficient. Procedure authorized."
            )
        parts = [f"Authorization DENIED. {len(blocking)} gate(s) blocking."]
        for gate_eval in gate_evals:
            if gate_eval.status == GateStatus.FAILED:
                reasons = "; ".join(gate_eval.failure_reasons[:3])
                parts.append(f"Gate {gate_eval.gate_id} ({gate_eval.gate_name}): {reasons}")
        tier_sufficient = clinician_tier in (ClinicalTier.TIER_2.value, ClinicalTier.TIER_3.value)
        if not tier_sufficient:
            parts.append(f"Clinician tier ({clinician_tier}) insufficient; Tier 2 or 3 required.")
        return " ".join(parts)
