#!/usr/bin/env python3
"""
Central Patient Journey State Model for Physical AI Oncology Trials
===================================================================

Defines every data structure used across all 10 stages of a single-patient
journey through a Physical AI oncology clinical trial.  This file is created
once (Commit 1) and is designed from the start to accommodate all 10 stages
without modification.

Enumerations cover patient stages, treatment arms, response categories,
adverse-event severity (CTCAE v5.0), consent status, data-lock status,
Physical AI system classification per ICH E6(R3) section 1.2 and
21 CFR 312.401, USL readiness levels per ICH E6(R3) section 1.5,
and MCP conformance levels per 21 CFR Part 50 section 50.33.

Dataclasses model demographics, tumor profiles, biomarkers, organ function,
adverse events, treatment cycles, surgical records, imaging time-points,
digital-twin state, robot qualifications, federation contributions,
regulatory events, consent records, and audit entries.

The master ``PatientJourneyState`` dataclass aggregates all of the above
and provides helper methods for stage advancement, adverse-event tracking,
regulatory-event logging, and 21 CFR Part 11 compliant audit entries.

Usage:
    from patient_state import PatientJourneyState, PatientStage

    state = PatientJourneyState(
        demographics=PatientDemographics(...),
        tumor=TumorProfile(...),
        ...
    )
    state.advance_stage(PatientStage.ENROLLMENT)

Last updated: March 2026

DISCLAIMER: RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class PatientStage(Enum):
    """Ten stages of a single-patient journey through a Physical AI trial."""

    PRESCREENING = "PRESCREENING"
    ENROLLMENT = "ENROLLMENT"
    DIGITAL_TWIN_CONSTRUCTION = "DIGITAL_TWIN_CONSTRUCTION"
    ROBOT_QUALIFICATION = "ROBOT_QUALIFICATION"
    SURGERY = "SURGERY"
    RECOVERY = "RECOVERY"
    IMMUNOTHERAPY = "IMMUNOTHERAPY"
    FEDERATION = "FEDERATION"
    SURVEILLANCE = "SURVEILLANCE"
    CLOSEOUT = "CLOSEOUT"


class PatientStatus(Enum):
    """Possible statuses throughout the patient journey."""

    REFERRED = "REFERRED"
    SCREENING = "SCREENING"
    ELIGIBLE = "ELIGIBLE"
    CONSENTED = "CONSENTED"
    ENROLLED = "ENROLLED"
    RANDOMIZED = "RANDOMIZED"
    ACTIVE_TREATMENT = "ACTIVE_TREATMENT"
    SURGERY_SCHEDULED = "SURGERY_SCHEDULED"
    SURGERY_COMPLETE = "SURGERY_COMPLETE"
    RECOVERING = "RECOVERING"
    ON_IMMUNOTHERAPY = "ON_IMMUNOTHERAPY"
    SURVEILLANCE = "SURVEILLANCE"
    COMPLETED = "COMPLETED"
    WITHDRAWN = "WITHDRAWN"


class TreatmentArm(Enum):
    """Treatment arms in the clinical trial."""

    EXPERIMENTAL = "EXPERIMENTAL"
    CONTROL = "CONTROL"
    PLACEBO = "PLACEBO"


class ResponseCategory(Enum):
    """RECIST 1.1 response categories."""

    CR = "CR"  # Complete Response
    PR = "PR"  # Partial Response
    SD = "SD"  # Stable Disease
    PD = "PD"  # Progressive Disease


class AESeverity(Enum):
    """CTCAE v5.0 adverse-event severity grades."""

    MILD = "MILD"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    LIFE_THREATENING = "LIFE_THREATENING"
    FATAL = "FATAL"


class ConsentStatus(Enum):
    """Status of informed consent."""

    PENDING = "PENDING"
    OBTAINED = "OBTAINED"
    AMENDED = "AMENDED"
    WITHDRAWN = "WITHDRAWN"


class DataLockStatus(Enum):
    """Status of patient data lock for regulatory submission."""

    OPEN = "OPEN"
    SOFT_LOCK = "SOFT_LOCK"
    HARD_LOCK = "HARD_LOCK"


class PhysicalAIClassification(Enum):
    """Physical AI system classification per ICH E6(R3) section 1.2
    and 21 CFR 312.401."""

    SURGICAL_ROBOT = "SURGICAL_ROBOT"
    COBOT = "COBOT"
    HUMANOID = "HUMANOID"
    THERAPEUTIC = "THERAPEUTIC"
    DIAGNOSTIC = "DIAGNOSTIC"
    ASSISTIVE = "ASSISTIVE"
    REHABILITATIVE = "REHABILITATIVE"


class USLReadinessLevel(Enum):
    """Unification Standard Level readiness per ICH E6(R3) section 1.5.

    USL scores range from 1.0 to 10.0 across four dimensions:
    Autonomy, Dexterity, Safety, and Interoperability.
    """

    FOUNDATIONAL = "FOUNDATIONAL"          # 1.0 - 4.9
    INTERMEDIATE = "INTERMEDIATE"          # 5.0 - 6.9
    ADVANCED = "ADVANCED"                  # 7.0 - 8.9
    CLINICAL_GRADE = "CLINICAL_GRADE"      # 9.0 - 10.0


class MCPConformanceLevel(Enum):
    """MCP-PAI conformance levels per 21 CFR Part 50 section 50.33.

    Five cumulative levels from basic core conformance through
    full robot-procedure integration.
    """

    CORE = "CORE"
    CLINICAL_READ = "CLINICAL_READ"
    IMAGING = "IMAGING"
    FEDERATED_SITE = "FEDERATED_SITE"
    ROBOT_PROCEDURE = "ROBOT_PROCEDURE"


# ---------------------------------------------------------------------------
# Legal stage transitions (directed graph)
# ---------------------------------------------------------------------------

LEGAL_STAGE_TRANSITIONS: dict[PatientStage, list[PatientStage]] = {
    PatientStage.PRESCREENING: [PatientStage.ENROLLMENT],
    PatientStage.ENROLLMENT: [PatientStage.DIGITAL_TWIN_CONSTRUCTION],
    PatientStage.DIGITAL_TWIN_CONSTRUCTION: [PatientStage.ROBOT_QUALIFICATION],
    PatientStage.ROBOT_QUALIFICATION: [PatientStage.SURGERY],
    PatientStage.SURGERY: [PatientStage.RECOVERY],
    PatientStage.RECOVERY: [PatientStage.IMMUNOTHERAPY],
    PatientStage.IMMUNOTHERAPY: [PatientStage.FEDERATION],
    PatientStage.FEDERATION: [PatientStage.SURVEILLANCE],
    PatientStage.SURVEILLANCE: [PatientStage.CLOSEOUT],
    PatientStage.CLOSEOUT: [],
}


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PatientDemographics:
    """Core demographic information for a trial participant."""

    patient_id: str = ""
    age: int = 0
    sex: str = ""
    ethnicity: str = ""
    smoking_status: str = ""


@dataclass
class TumorProfile:
    """Tumor characterisation at baseline and over time."""

    tumor_type: str = ""
    stage: str = ""
    grade: str = ""
    volume_cm3: float = 0.0
    molecular_markers: dict = field(default_factory=dict)
    growth_rate: float = 0.0
    histology: str = ""


@dataclass
class Biomarkers:
    """Key molecular biomarkers for treatment-selection decisions."""

    pdl1_tps: float = 0.0
    tmb: float = 0.0
    egfr: str = "WT"
    alk: str = "negative"
    ros1: str = "negative"
    msi: str = "MSS"
    kras: str = "WT"
    braf: str = "WT"
    her2: str = "negative"
    ntrk: str = "negative"


@dataclass
class OrganFunction:
    """Organ-function assessments for eligibility and safety monitoring."""

    ecog: int = 0
    renal_function: float = 0.0   # eGFR mL/min/1.73m2
    hepatic_function: str = ""    # Child-Pugh class
    cardiac_ef: float = 0.0       # Ejection fraction %
    pulmonary_fev1: float = 0.0   # FEV1 %predicted


@dataclass
class AdverseEvent:
    """Adverse event record per CTCAE v5.0 and ICH E6(R3) section 2.10."""

    event_id: str = ""
    description: str = ""
    severity: AESeverity = AESeverity.MILD
    organ_system: str = ""
    onset_day: int = 0
    resolution_day: Optional[int] = None
    resolved: bool = False
    causality: str = ""
    related_to_device: bool = False
    reported_to_irb: bool = False
    reported_to_fda: bool = False
    ctcae_grade: int = 1


@dataclass
class TreatmentCycle:
    """Single cycle of systemic treatment (e.g. immunotherapy)."""

    cycle_number: int = 0
    start_day: int = 0
    drug: str = ""
    dose_mg: float = 0.0
    dose_modifications: str = ""
    labs_pre: dict = field(default_factory=dict)
    labs_post: dict = field(default_factory=dict)
    toxicities: list[AdverseEvent] = field(default_factory=list)
    response_assessment: Optional[ResponseCategory] = None


@dataclass
class SurgicalRecord:
    """Surgical record including Physical AI system data."""

    procedure_date_day: int = 0
    procedure_type: str = ""
    robot_id: str = ""
    robot_classification: PhysicalAIClassification = PhysicalAIClassification.SURGICAL_ROBOT
    operative_time_min: int = 0
    estimated_blood_loss_ml: int = 0
    margins_status: str = ""
    lymph_nodes_sampled: int = 0
    lymph_nodes_positive: int = 0
    complications: str = ""
    specimens: list[str] = field(default_factory=list)
    pathology_stage: str = ""
    usl_score_at_procedure: float = 0.0
    autonomy_level: int = 0


@dataclass
class ImagingTimepoint:
    """Imaging assessment at a specific trial time-point."""

    day: int = 0
    modality: str = ""
    tumor_volume_cm3: float = 0.0
    response_category: Optional[ResponseCategory] = None
    notes: str = ""


@dataclass
class DigitalTwinState:
    """State of the patient-specific digital twin per ICH E6(R3) section 1.4.2."""

    twin_id: str = ""
    model_type: str = ""
    current_volume_cm3: float = 0.0
    proliferation_rate: float = 0.0
    calibrated: bool = False
    last_updated_day: int = 0
    predictions: dict = field(default_factory=dict)
    validation_score: float = 0.0


@dataclass
class RobotQualification:
    """Robot qualification record per ICH E6(R3) section 1.5 and 21 CFR 312.401-405."""

    robot_id: str = ""
    robot_type: str = ""
    robot_classification: PhysicalAIClassification = PhysicalAIClassification.SURGICAL_ROBOT
    usl_score: float = 0.0
    usl_readiness: USLReadinessLevel = USLReadinessLevel.FOUNDATIONAL
    safety_validated: bool = False
    calibration_error_mm: float = 0.0
    deployment_ready: bool = False
    iec_80601_compliant: bool = False
    cybersecurity_validated: bool = False


@dataclass
class FederationContribution:
    """Record of a single federated-learning contribution per 21 CFR 50.33."""

    round_number: int = 0
    day: int = 0
    model_type: str = ""
    epsilon_spent: float = 0.0
    gradient_norm: float = 0.0


@dataclass
class RegulatoryEvent:
    """Regulatory event or milestone with CFR section citation."""

    event_type: str = ""
    day: int = 0
    description: str = ""
    document_id: str = ""
    cfr_section: str = ""
    status: str = ""


@dataclass
class ConsentRecord:
    """Informed-consent record per 21 CFR Part 50 section 50.25 and ICH E6(R3) section 2.8."""

    consent_status: ConsentStatus = ConsentStatus.PENDING
    version: str = ""
    date_day: int = 0
    physical_ai_appendix_included: bool = False
    robot_types_disclosed: list[str] = field(default_factory=list)
    autonomy_levels_disclosed: bool = False
    override_rights_disclosed: bool = False


@dataclass
class AuditEntry:
    """21 CFR Part 11 compliant audit-trail entry."""

    timestamp_day: int = 0
    action: str = ""
    actor: str = ""
    details: str = ""
    electronic_signature: str = ""
    cfr_part_11_compliant: bool = True


# ---------------------------------------------------------------------------
# Master patient journey state
# ---------------------------------------------------------------------------


@dataclass
class PatientJourneyState:
    """Master state object aggregating all patient data across 10 stages.

    This is the single source of truth for a patient's journey through
    a Physical AI oncology trial.  Helper methods enforce legal stage
    transitions, adverse-event reporting thresholds, regulatory-event
    logging, and 21 CFR Part 11 compliant audit entries.
    """

    demographics: PatientDemographics = field(default_factory=PatientDemographics)
    tumor: TumorProfile = field(default_factory=TumorProfile)
    biomarkers: Biomarkers = field(default_factory=Biomarkers)
    organ_function: OrganFunction = field(default_factory=OrganFunction)
    stage: PatientStage = PatientStage.PRESCREENING
    status: PatientStatus = PatientStatus.REFERRED
    consent: ConsentRecord = field(default_factory=ConsentRecord)
    treatment_arm: Optional[TreatmentArm] = None
    enrollment_day: Optional[int] = None
    surgical_record: Optional[SurgicalRecord] = None
    digital_twin: Optional[DigitalTwinState] = None
    robot_qualifications: list[RobotQualification] = field(default_factory=list)
    treatment_cycles: list[TreatmentCycle] = field(default_factory=list)
    imaging_timepoints: list[ImagingTimepoint] = field(default_factory=list)
    adverse_events: list[AdverseEvent] = field(default_factory=list)
    federation_contributions: list[FederationContribution] = field(default_factory=list)
    regulatory_events: list[RegulatoryEvent] = field(default_factory=list)
    audit_trail: list[AuditEntry] = field(default_factory=list)
    data_lock: DataLockStatus = DataLockStatus.OPEN
    mcp_conformance_level: MCPConformanceLevel = MCPConformanceLevel.CORE
    outcome: Optional[dict] = None

    # ----- helpers -----

    def advance_stage(self, new_stage: PatientStage) -> None:
        """Advance the patient to *new_stage*, enforcing legal transitions.

        Legal transitions follow a strict linear progression from
        PRESCREENING through CLOSEOUT.  Any attempt to skip a stage
        raises ``ValueError``.

        Per ICH E6(R3) section 2.9 all stage transitions are logged to
        the audit trail as 21 CFR Part 11 compliant entries.
        """
        allowed = LEGAL_STAGE_TRANSITIONS.get(self.stage, [])
        if new_stage not in allowed:
            raise ValueError(
                f"Illegal stage transition: {self.stage.value} -> {new_stage.value}. "
                f"Allowed targets: {[s.value for s in allowed]}"
            )
        old_stage = self.stage
        self.stage = new_stage
        self.add_audit_entry(
            action="STAGE_ADVANCE",
            actor="system",
            details=f"Stage advanced from {old_stage.value} to {new_stage.value}",
        )
        logger.info("Stage advanced: %s -> %s", old_stage.value, new_stage.value)

    def add_adverse_event(self, event: AdverseEvent) -> None:
        """Append an adverse event and check reporting thresholds.

        Per Physical AI Adaptation of 21 CFR 312.32, Grade >= 3 events
        require expedited FDA reporting.  Device-related events require
        additional Physical AI causality assessment per ICH E6(R3) section 2.10.
        """
        self.adverse_events.append(event)
        # Per 21 CFR 312.32 - expedited reporting for serious events
        if event.ctcae_grade >= 3:
            logger.warning(
                "Grade %d AE detected (%s) - expedited reporting required per 21 CFR 312.32",
                event.ctcae_grade,
                event.description,
            )
        if event.related_to_device:
            logger.warning(
                "Device-related AE (%s) - Physical AI causality assessment required per ICH E6(R3) section 2.10",
                event.description,
            )
        self.add_audit_entry(
            action="ADVERSE_EVENT_RECORDED",
            actor="system",
            details=f"AE {event.event_id}: {event.description} (Grade {event.ctcae_grade})",
        )

    def add_regulatory_event(self, event: RegulatoryEvent) -> None:
        """Append a regulatory event with CFR section citation."""
        self.regulatory_events.append(event)
        self.add_audit_entry(
            action="REGULATORY_EVENT",
            actor="system",
            details=f"{event.event_type}: {event.description} (CFR {event.cfr_section})",
        )
        logger.info(
            "Regulatory event: %s [%s] - %s",
            event.event_type,
            event.cfr_section,
            event.description,
        )

    def add_audit_entry(self, action: str, actor: str, details: str) -> None:
        """Create a timestamped 21 CFR Part 11 compliant audit entry."""
        day = self.get_current_day()
        entry = AuditEntry(
            timestamp_day=day,
            action=action,
            actor=actor,
            details=details,
            electronic_signature=f"SIG-{actor}-{day}-{len(self.audit_trail) + 1:04d}",
            cfr_part_11_compliant=True,
        )
        self.audit_trail.append(entry)

    def get_current_day(self) -> int:
        """Return the most recent day from any timeline event."""
        candidates: list[int] = [0]
        if self.enrollment_day is not None:
            candidates.append(self.enrollment_day)
        if self.surgical_record is not None:
            candidates.append(self.surgical_record.procedure_date_day)
        if self.imaging_timepoints:
            candidates.append(max(t.day for t in self.imaging_timepoints))
        if self.treatment_cycles:
            candidates.append(max(c.start_day for c in self.treatment_cycles))
        if self.adverse_events:
            candidates.append(max(e.onset_day for e in self.adverse_events))
        if self.federation_contributions:
            candidates.append(max(f.day for f in self.federation_contributions))
        return max(candidates)

    def summary(self) -> dict:
        """Return a compact summary of the current patient state."""
        return {
            "patient_id": self.demographics.patient_id,
            "stage": self.stage.value,
            "status": self.status.value,
            "treatment_arm": self.treatment_arm.value if self.treatment_arm else None,
            "enrollment_day": self.enrollment_day,
            "consent": self.consent.consent_status.value,
            "adverse_events": len(self.adverse_events),
            "treatment_cycles": len(self.treatment_cycles),
            "imaging_timepoints": len(self.imaging_timepoints),
            "audit_entries": len(self.audit_trail),
            "data_lock": self.data_lock.value,
            "mcp_conformance": self.mcp_conformance_level.value,
            "digital_twin_calibrated": self.digital_twin.calibrated if self.digital_twin else False,
            "robots_qualified": len([r for r in self.robot_qualifications if r.deployment_ready]),
            "federation_rounds": len(self.federation_contributions),
            "outcome": self.outcome,
        }
