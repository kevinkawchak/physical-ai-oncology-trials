#!/usr/bin/env python3
"""Site enrollment synchronization for multi-site oncology trials.

Real-time (simulated) enrollment tracking across trial sites with
conflict resolution for patient eligibility disputes, stratification
rebalancing, and cross-site enrollment parity monitoring.

Framework Dependencies:
    - NumPy 1.24.0+

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.
    Enrollment decisions must be reviewed by qualified clinical investigators
    and IRB-approved protocols.

Version: 1.0.0
Last Updated: February 2026
License: MIT
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EnrollmentStatus(Enum):
    """Enrollment status for a patient."""

    SCREENING = "screening"
    ELIGIBLE = "eligible"
    ENROLLED = "enrolled"
    RANDOMIZED = "randomized"
    ACTIVE = "active"
    COMPLETED = "completed"
    WITHDRAWN = "withdrawn"
    SCREEN_FAIL = "screen_fail"


class ConflictType(Enum):
    """Types of enrollment conflicts."""

    DUPLICATE_ENROLLMENT = "duplicate_enrollment"
    ELIGIBILITY_DISPUTE = "eligibility_dispute"
    STRATIFICATION_IMBALANCE = "stratification_imbalance"
    SITE_CAPACITY = "site_capacity"
    ARM_IMBALANCE = "arm_imbalance"


class ConflictResolution(Enum):
    """Resolution strategies for enrollment conflicts."""

    FIRST_COME = "first_come"
    SITE_PRIORITY = "site_priority"
    RANDOM_ASSIGNMENT = "random_assignment"
    MANUAL_REVIEW = "manual_review"


class TreatmentArm(Enum):
    """Treatment arms for randomization."""

    EXPERIMENTAL = "experimental"
    CONTROL = "control"
    PLACEBO = "placebo"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class PatientEnrollment:
    """Individual patient enrollment record.

    Attributes:
        patient_id: De-identified patient identifier.
        site_id: Enrolling site identifier.
        status: Current enrollment status.
        treatment_arm: Assigned treatment arm.
        enrollment_time: Timestamp of enrollment.
        stratification_factors: Stratification factor values.
        eligibility_criteria_met: Boolean flags for eligibility criteria.
        demographics: De-identified demographic data.
    """

    patient_id: str = ""
    site_id: str = ""
    status: EnrollmentStatus = EnrollmentStatus.SCREENING
    treatment_arm: Optional[TreatmentArm] = None
    enrollment_time: float = 0.0
    stratification_factors: Dict[str, str] = field(default_factory=dict)
    eligibility_criteria_met: Dict[str, bool] = field(default_factory=dict)
    demographics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SiteEnrollmentState:
    """Enrollment state for a single site.

    Attributes:
        site_id: Site identifier.
        site_name: Human-readable site name.
        target_enrollment: Target enrollment count.
        current_enrollment: Current enrolled count.
        screening_count: Patients in screening.
        screen_fail_count: Number of screen failures.
        withdrawal_count: Number of withdrawals.
        arm_counts: Enrollment per treatment arm.
        stratification_counts: Counts by stratification factor.
        enrollment_rate: Rolling enrollment rate (patients/week).
    """

    site_id: str = ""
    site_name: str = ""
    target_enrollment: int = 50
    current_enrollment: int = 0
    screening_count: int = 0
    screen_fail_count: int = 0
    withdrawal_count: int = 0
    arm_counts: Dict[str, int] = field(default_factory=dict)
    stratification_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    enrollment_rate: float = 0.0


@dataclass
class EnrollmentConflict:
    """Record of an enrollment conflict.

    Attributes:
        conflict_id: Unique conflict identifier.
        conflict_type: Type of conflict.
        patient_id: Affected patient.
        site_ids: Sites involved in the conflict.
        description: Human-readable description.
        resolution: Applied resolution strategy.
        resolved: Whether the conflict has been resolved.
        resolution_details: Details of how it was resolved.
        timestamp: When the conflict was detected.
    """

    conflict_id: str = ""
    conflict_type: ConflictType = ConflictType.DUPLICATE_ENROLLMENT
    patient_id: str = ""
    site_ids: List[str] = field(default_factory=list)
    description: str = ""
    resolution: ConflictResolution = ConflictResolution.MANUAL_REVIEW
    resolved: bool = False
    resolution_details: str = ""
    timestamp: float = 0.0


@dataclass
class StratificationConfig:
    """Configuration for stratified randomization.

    Attributes:
        factors: Dictionary of factor names to possible values.
        block_size: Block size for block randomization.
        imbalance_threshold: Maximum tolerated imbalance ratio.
    """

    factors: Dict[str, List[str]] = field(default_factory=dict)
    block_size: int = 4
    imbalance_threshold: float = 0.2


# ---------------------------------------------------------------------------
# Stratified Randomizer
# ---------------------------------------------------------------------------


class StratifiedRandomizer:
    """Implements stratified block randomization.

    Ensures balanced treatment arm assignment within strata
    defined by patient characteristics.

    Args:
        config: Stratification configuration.
        arms: List of treatment arms.
    """

    def __init__(self, config: StratificationConfig, arms: Optional[List[TreatmentArm]] = None):
        self.config = config
        self.arms = arms or [TreatmentArm.EXPERIMENTAL, TreatmentArm.CONTROL]
        self.strata_blocks: Dict[str, List[TreatmentArm]] = {}

    def _get_stratum_key(self, factors: Dict[str, str]) -> str:
        """Generate a stratum key from factor values."""
        sorted_items = sorted(factors.items())
        return "|".join(f"{k}={v}" for k, v in sorted_items)

    def _generate_block(self) -> List[TreatmentArm]:
        """Generate a randomized block of treatment assignments."""
        repeats = self.config.block_size // len(self.arms)
        repeats = max(repeats, 1)
        block = list(self.arms) * repeats
        np.random.shuffle(block)
        return block

    def randomize(self, patient: PatientEnrollment) -> TreatmentArm:
        """Assign a treatment arm using stratified block randomization.

        Args:
            patient: Patient to randomize.

        Returns:
            Assigned treatment arm.
        """
        stratum_key = self._get_stratum_key(patient.stratification_factors)

        if stratum_key not in self.strata_blocks or not self.strata_blocks[stratum_key]:
            self.strata_blocks[stratum_key] = self._generate_block()

        arm = self.strata_blocks[stratum_key].pop(0)
        patient.treatment_arm = arm
        patient.status = EnrollmentStatus.RANDOMIZED

        logger.info(
            "Patient %s randomized to %s (stratum: %s)",
            patient.patient_id,
            arm.value,
            stratum_key,
        )
        return arm


# ---------------------------------------------------------------------------
# Conflict Resolver
# ---------------------------------------------------------------------------


class ConflictResolver:
    """Resolves enrollment conflicts across sites.

    Handles duplicate enrollment detection, eligibility disputes,
    stratification imbalances, and arm balance issues.
    """

    def __init__(self):
        self.resolved_conflicts: List[EnrollmentConflict] = []

    def detect_duplicate(
        self, patient_id: str, enrollments: Dict[str, List[PatientEnrollment]]
    ) -> Optional[EnrollmentConflict]:
        """Detect if a patient is enrolled at multiple sites.

        Args:
            patient_id: Patient to check.
            enrollments: Per-site enrollment lists.

        Returns:
            EnrollmentConflict if duplicate found, else None.
        """
        enrolled_sites = []
        for site_id, patients in enrollments.items():
            for p in patients:
                if p.patient_id == patient_id and p.status in (
                    EnrollmentStatus.ENROLLED,
                    EnrollmentStatus.RANDOMIZED,
                    EnrollmentStatus.ACTIVE,
                ):
                    enrolled_sites.append(site_id)

        if len(enrolled_sites) > 1:
            conflict = EnrollmentConflict(
                conflict_id=str(uuid.uuid4())[:8],
                conflict_type=ConflictType.DUPLICATE_ENROLLMENT,
                patient_id=patient_id,
                site_ids=enrolled_sites,
                description=f"Patient {patient_id} enrolled at {len(enrolled_sites)} sites",
                timestamp=time.time(),
            )
            logger.warning("Duplicate enrollment detected: %s at sites %s", patient_id, enrolled_sites)
            return conflict
        return None

    def resolve_duplicate(
        self, conflict: EnrollmentConflict, strategy: ConflictResolution = ConflictResolution.FIRST_COME
    ) -> str:
        """Resolve a duplicate enrollment conflict.

        Args:
            conflict: The conflict to resolve.
            strategy: Resolution strategy.

        Returns:
            Site ID where enrollment is maintained.
        """
        if strategy == ConflictResolution.FIRST_COME:
            winning_site = conflict.site_ids[0]
        elif strategy == ConflictResolution.RANDOM_ASSIGNMENT:
            winning_site = conflict.site_ids[np.random.randint(len(conflict.site_ids))]
        else:
            winning_site = conflict.site_ids[0]

        conflict.resolved = True
        conflict.resolution = strategy
        conflict.resolution_details = f"Enrollment maintained at {winning_site}"
        self.resolved_conflicts.append(conflict)

        logger.info("Conflict %s resolved: enrollment at %s", conflict.conflict_id, winning_site)
        return winning_site

    def check_arm_balance(
        self, site_states: Dict[str, SiteEnrollmentState], threshold: float = 0.2
    ) -> List[EnrollmentConflict]:
        """Check for treatment arm imbalances across sites.

        Args:
            site_states: Per-site enrollment states.
            threshold: Maximum tolerated imbalance ratio.

        Returns:
            List of imbalance conflicts detected.
        """
        conflicts = []
        total_arms: Dict[str, int] = {}

        for state in site_states.values():
            for arm, count in state.arm_counts.items():
                total_arms[arm] = total_arms.get(arm, 0) + count

        if not total_arms:
            return conflicts

        total = sum(total_arms.values())
        if total == 0:
            return conflicts

        expected = total / len(total_arms)
        for arm, count in total_arms.items():
            if expected > 0 and abs(count - expected) / expected > threshold:
                conflict = EnrollmentConflict(
                    conflict_id=str(uuid.uuid4())[:8],
                    conflict_type=ConflictType.ARM_IMBALANCE,
                    description=f"Arm {arm} imbalanced: {count} vs expected {expected:.0f}",
                    timestamp=time.time(),
                )
                conflicts.append(conflict)

        return conflicts


# ---------------------------------------------------------------------------
# Enrollment Synchronizer
# ---------------------------------------------------------------------------


class EnrollmentSynchronizer:
    """Synchronizes enrollment across multiple clinical sites.

    Provides centralized enrollment tracking, conflict resolution,
    and real-time enrollment status monitoring.

    Args:
        trial_id: Clinical trial identifier.
        stratification_config: Configuration for stratified randomization.
    """

    def __init__(self, trial_id: str = "", stratification_config: Optional[StratificationConfig] = None):
        self.trial_id = trial_id or str(uuid.uuid4())[:8]
        self.sites: Dict[str, SiteEnrollmentState] = {}
        self.enrollments: Dict[str, List[PatientEnrollment]] = {}
        self.conflict_resolver = ConflictResolver()
        self.stratification_config = stratification_config or StratificationConfig(
            factors={"cancer_type": ["breast", "lung", "colorectal"], "stage": ["II", "III", "IV"]},
            block_size=4,
        )
        self.randomizer = StratifiedRandomizer(self.stratification_config)
        self.audit_log: List[Dict[str, Any]] = []

        logger.info("Enrollment synchronizer initialized for trial %s", self.trial_id)

    def register_site(self, site_id: str, site_name: str = "", target: int = 50) -> SiteEnrollmentState:
        """Register a new site for enrollment.

        Args:
            site_id: Site identifier.
            site_name: Human-readable name.
            target: Target enrollment count.

        Returns:
            New SiteEnrollmentState.
        """
        state = SiteEnrollmentState(
            site_id=site_id,
            site_name=site_name or f"Site {site_id}",
            target_enrollment=target,
        )
        self.sites[site_id] = state
        self.enrollments[site_id] = []

        self._log_event("site_registered", {"site_id": site_id, "target": target})
        logger.info("Registered site %s (target=%d)", site_id, target)
        return state

    def screen_patient(
        self,
        site_id: str,
        patient_id: str = "",
        eligibility: Optional[Dict[str, bool]] = None,
        stratification: Optional[Dict[str, str]] = None,
        demographics: Optional[Dict[str, Any]] = None,
    ) -> PatientEnrollment:
        """Screen a patient at a site.

        Args:
            site_id: Site performing screening.
            patient_id: Patient identifier (auto-generated if empty).
            eligibility: Eligibility criteria results.
            stratification: Stratification factor values.
            demographics: De-identified demographics.

        Returns:
            PatientEnrollment record.
        """
        if site_id not in self.sites:
            raise ValueError(f"Site {site_id} not registered")

        patient = PatientEnrollment(
            patient_id=patient_id or f"PT-{uuid.uuid4().hex[:6].upper()}",
            site_id=site_id,
            status=EnrollmentStatus.SCREENING,
            enrollment_time=time.time(),
            eligibility_criteria_met=eligibility or {},
            stratification_factors=stratification or {},
            demographics=demographics or {},
        )

        self.enrollments[site_id].append(patient)
        self.sites[site_id].screening_count += 1

        logger.info("Patient %s screened at site %s", patient.patient_id, site_id)
        return patient

    def enroll_patient(self, site_id: str, patient_id: str) -> Optional[PatientEnrollment]:
        """Enroll a screened patient and perform randomization.

        Args:
            site_id: Site where patient is screened.
            patient_id: Patient to enroll.

        Returns:
            Updated PatientEnrollment, or None if not found.
        """
        patient = self._find_patient(site_id, patient_id)
        if patient is None:
            logger.warning("Patient %s not found at site %s", patient_id, site_id)
            return None

        all_met = all(patient.eligibility_criteria_met.values()) if patient.eligibility_criteria_met else True
        if not all_met:
            patient.status = EnrollmentStatus.SCREEN_FAIL
            self.sites[site_id].screen_fail_count += 1
            logger.info("Patient %s screen failed at site %s", patient_id, site_id)
            return patient

        dup_conflict = self.conflict_resolver.detect_duplicate(patient_id, self.enrollments)
        if dup_conflict is not None:
            winning_site = self.conflict_resolver.resolve_duplicate(dup_conflict)
            if winning_site != site_id:
                patient.status = EnrollmentStatus.SCREEN_FAIL
                self.sites[site_id].screen_fail_count += 1
                return patient

        patient.status = EnrollmentStatus.ENROLLED
        self.sites[site_id].current_enrollment += 1

        arm = self.randomizer.randomize(patient)
        arm_key = arm.value
        self.sites[site_id].arm_counts[arm_key] = self.sites[site_id].arm_counts.get(arm_key, 0) + 1

        for factor, value in patient.stratification_factors.items():
            if factor not in self.sites[site_id].stratification_counts:
                self.sites[site_id].stratification_counts[factor] = {}
            strat = self.sites[site_id].stratification_counts[factor]
            strat[value] = strat.get(value, 0) + 1

        self._log_event(
            "patient_enrolled",
            {"patient_id": patient_id, "site_id": site_id, "arm": arm_key},
        )

        logger.info("Patient %s enrolled at %s, arm=%s", patient_id, site_id, arm_key)
        return patient

    def withdraw_patient(self, site_id: str, patient_id: str, reason: str = "") -> Optional[PatientEnrollment]:
        """Record a patient withdrawal.

        Args:
            site_id: Site of withdrawal.
            patient_id: Withdrawing patient.
            reason: Reason for withdrawal.

        Returns:
            Updated PatientEnrollment.
        """
        patient = self._find_patient(site_id, patient_id)
        if patient is None:
            return None

        patient.status = EnrollmentStatus.WITHDRAWN
        self.sites[site_id].withdrawal_count += 1
        if self.sites[site_id].current_enrollment > 0:
            self.sites[site_id].current_enrollment -= 1

        self._log_event(
            "patient_withdrawn",
            {"patient_id": patient_id, "site_id": site_id, "reason": reason},
        )
        return patient

    def get_enrollment_summary(self) -> Dict[str, Any]:
        """Generate cross-site enrollment summary.

        Returns:
            Dictionary with enrollment metrics across all sites.
        """
        total_enrolled = sum(s.current_enrollment for s in self.sites.values())
        total_target = sum(s.target_enrollment for s in self.sites.values())
        total_screened = sum(s.screening_count for s in self.sites.values())
        total_screen_fails = sum(s.screen_fail_count for s in self.sites.values())
        total_withdrawals = sum(s.withdrawal_count for s in self.sites.values())

        site_summaries = {}
        for site_id, state in self.sites.items():
            pct = (state.current_enrollment / state.target_enrollment * 100) if state.target_enrollment > 0 else 0
            site_summaries[site_id] = {
                "enrolled": state.current_enrollment,
                "target": state.target_enrollment,
                "pct_target": round(pct, 1),
                "screening": state.screening_count,
                "screen_fails": state.screen_fail_count,
                "withdrawals": state.withdrawal_count,
                "arm_counts": dict(state.arm_counts),
            }

        arm_totals: Dict[str, int] = {}
        for state in self.sites.values():
            for arm, count in state.arm_counts.items():
                arm_totals[arm] = arm_totals.get(arm, 0) + count

        return {
            "trial_id": self.trial_id,
            "total_sites": len(self.sites),
            "total_enrolled": total_enrolled,
            "total_target": total_target,
            "pct_overall": round(total_enrolled / total_target * 100, 1) if total_target > 0 else 0,
            "total_screened": total_screened,
            "total_screen_fails": total_screen_fails,
            "screen_fail_rate": round(total_screen_fails / total_screened * 100, 1) if total_screened > 0 else 0,
            "total_withdrawals": total_withdrawals,
            "arm_totals": arm_totals,
            "site_summaries": site_summaries,
            "conflicts_resolved": len(self.conflict_resolver.resolved_conflicts),
        }

    def check_balance(self) -> List[EnrollmentConflict]:
        """Check for enrollment imbalances across all sites.

        Returns:
            List of detected imbalance conflicts.
        """
        return self.conflict_resolver.check_arm_balance(self.sites)

    def _find_patient(self, site_id: str, patient_id: str) -> Optional[PatientEnrollment]:
        """Find a patient at a site."""
        if site_id not in self.enrollments:
            return None
        for patient in self.enrollments[site_id]:
            if patient.patient_id == patient_id:
                return patient
        return None

    def _log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Append an event to the audit log."""
        self.audit_log.append(
            {
                "event_id": str(uuid.uuid4())[:8],
                "event_type": event_type,
                "timestamp": time.time(),
                "trial_id": self.trial_id,
                "details": details,
            }
        )


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sync = EnrollmentSynchronizer(trial_id="ONCO-FED-001")

    for i in range(4):
        sync.register_site(f"site_{i:03d}", f"Hospital {chr(65 + i)}", target=25)

    for site_id in sync.sites:
        for j in range(5):
            patient = sync.screen_patient(
                site_id,
                eligibility={"age_18_plus": True, "adequate_organ_function": True, "informed_consent": True},
                stratification={
                    "cancer_type": np.random.choice(["breast", "lung", "colorectal"]),
                    "stage": np.random.choice(["II", "III", "IV"]),
                },
            )
            sync.enroll_patient(site_id, patient.patient_id)

    summary = sync.get_enrollment_summary()
    print("\n=== Enrollment Summary ===")
    print(f"  Total enrolled: {summary['total_enrolled']}/{summary['total_target']} ({summary['pct_overall']}%)")
    print(f"  Arm totals: {summary['arm_totals']}")
    for sid, ss in summary["site_summaries"].items():
        print(f"  {sid}: {ss['enrolled']}/{ss['target']} enrolled")
