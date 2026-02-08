"""
=============================================================================
Breach Response Protocol for Physical AI Oncology Trials
=============================================================================

Automated breach detection, risk assessment, and notification workflow
management per HIPAA Breach Notification Rule (45 CFR 164.400-414).

CLINICAL CONTEXT:
-----------------
In 2025, healthcare data breaches affected approximately 44.3 million
individuals (HHS OCR data). Clinical trial data breaches require:
  - Rapid detection and containment
  - Four-factor risk assessment per HIPAA guidance
  - Notification to individuals within 60 days of discovery
  - Notification to HHS OCR (immediately for 500+ affected)
  - Media notification when 500+ in a single state
  - State attorney general notification per state laws
  - Complete documentation for at least 6 years

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - Python 3.10+

Optional:
    - None (standalone module)

REFERENCES:
    - HIPAA Breach Notification Rule: 45 CFR 164.400-414
    - HHS OCR Breach Portal: https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf
    - HHS Breach Notification Requirements (2026 update)

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.
    Requires institutional validation and regulatory review before deployment.

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: BREACH DATA STRUCTURES
# =============================================================================


class IncidentType(Enum):
    """Types of security incidents."""

    UNAUTHORIZED_ACCESS = "unauthorized_access"
    UNAUTHORIZED_DISCLOSURE = "unauthorized_disclosure"
    LOSS = "loss"
    THEFT = "theft"
    RANSOMWARE = "ransomware"
    PHISHING = "phishing"
    SYSTEM_MISCONFIGURATION = "system_misconfiguration"
    IMPROPER_DISPOSAL = "improper_disposal"
    AI_SYSTEM_BREACH = "ai_system_breach"


class RiskLevel(Enum):
    """Risk levels from four-factor assessment."""

    LOW = "low_probability_of_compromise"
    MEDIUM = "medium_probability"
    HIGH = "high_probability"
    BREACH_CONFIRMED = "breach_confirmed"


class NotificationRecipient(Enum):
    """Required notification recipients."""

    INDIVIDUALS = "affected_individuals"
    HHS_OCR = "hhs_ocr"
    MEDIA = "prominent_media"
    STATE_AG = "state_attorney_general"
    SPONSOR = "trial_sponsor"
    IRB = "institutional_review_board"
    FDA = "fda_if_applicable"


@dataclass
class Incident:
    """Security incident record."""

    incident_id: str
    incident_type: IncidentType
    description: str
    phi_types_involved: list[str]
    individuals_affected: int
    discovery_date: str
    occurrence_date: str = ""
    discovery_method: str = ""
    affected_systems: list[str] = field(default_factory=list)
    containment_actions: list[str] = field(default_factory=list)
    status: str = "open"
    state_jurisdictions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "incident_type": self.incident_type.value,
            "description": self.description,
            "phi_types_involved": self.phi_types_involved,
            "individuals_affected": self.individuals_affected,
            "discovery_date": self.discovery_date,
            "status": self.status,
        }


@dataclass
class RiskAssessment:
    """HIPAA four-factor breach risk assessment."""

    incident_id: str
    assessment_date: str

    # Factor 1: Nature and extent of PHI involved
    phi_nature_score: int = 0  # 1-5
    phi_nature_notes: str = ""

    # Factor 2: Unauthorized person who used/received PHI
    unauthorized_party_score: int = 0  # 1-5
    unauthorized_party_notes: str = ""

    # Factor 3: Whether PHI was actually acquired or viewed
    acquisition_score: int = 0  # 1-5
    acquisition_notes: str = ""

    # Factor 4: Extent of mitigation
    mitigation_score: int = 0  # 1-5
    mitigation_notes: str = ""

    # Determination
    overall_risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    is_reportable_breach: bool = False
    rationale: str = ""

    def calculate_risk(self) -> None:
        """Calculate overall risk from four factors.

        Scores must be integers in the range 1-5 for each of the four
        HIPAA risk-assessment factors.  Invalid scores are clamped to the
        nearest boundary (1 or 5) and a warning is logged so that the
        assessment object is never left in an inconsistent state.
        """
        scores = [self.phi_nature_score, self.unauthorized_party_score, self.acquisition_score, self.mitigation_score]

        if not all(1 <= s <= 5 for s in scores):
            logger.warning("Risk scores out of range %s; clamping to [1, 5].", scores)
            self.phi_nature_score = max(1, min(5, self.phi_nature_score))
            self.unauthorized_party_score = max(1, min(5, self.unauthorized_party_score))
            self.acquisition_score = max(1, min(5, self.acquisition_score))
            self.mitigation_score = max(1, min(5, self.mitigation_score))

        # Higher mitigation score = lower risk, so invert
        adjusted_mitigation = 6 - self.mitigation_score
        self.overall_risk_score = (
            self.phi_nature_score * 0.3
            + self.unauthorized_party_score * 0.25
            + self.acquisition_score * 0.25
            + adjusted_mitigation * 0.2
        )

        if self.overall_risk_score <= 2.0:
            self.risk_level = RiskLevel.LOW
            self.is_reportable_breach = False
        elif self.overall_risk_score <= 3.0:
            self.risk_level = RiskLevel.MEDIUM
            self.is_reportable_breach = True
        elif self.overall_risk_score <= 4.0:
            self.risk_level = RiskLevel.HIGH
            self.is_reportable_breach = True
        else:
            self.risk_level = RiskLevel.BREACH_CONFIRMED
            self.is_reportable_breach = True


@dataclass
class NotificationDeadline:
    """A specific notification deadline."""

    recipient: str
    regulation: str
    due_date: str
    days_remaining: int
    status: str = "pending"
    sent_date: str = ""
    notes: str = ""


@dataclass
class NotificationTimeline:
    """Complete notification timeline for an incident."""

    incident_id: str
    discovery_date: str
    deadlines: list[NotificationDeadline] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "discovery_date": self.discovery_date,
            "deadlines": [
                {
                    "recipient": d.recipient,
                    "regulation": d.regulation,
                    "due_date": d.due_date,
                    "days_remaining": d.days_remaining,
                    "status": d.status,
                }
                for d in self.deadlines
            ],
        }


# =============================================================================
# SECTION 2: STATE NOTIFICATION REQUIREMENTS
# =============================================================================

# Notification requirements that differ from HIPAA federal floor
STATE_BREACH_LAWS: dict[str, dict[str, Any]] = {
    "CA": {
        "name": "California",
        "notification_days": 45,  # Days to notify (shorter than HIPAA)
        "ag_notification_threshold": 500,
        "statute": "Cal. Civ. Code § 1798.82",
        "special_requirements": [
            "Must use specific notification format",
            "AI disclosure: AB 489 effective Jan 1, 2026",
        ],
    },
    "TX": {
        "name": "Texas",
        "notification_days": 60,
        "ag_notification_threshold": 250,
        "statute": "Tex. Bus. & Com. Code § 521.053",
        "special_requirements": ["TRAIGA: AI use disclosure required effective Jan 1, 2026"],
    },
    "NY": {
        "name": "New York",
        "notification_days": 60,
        "ag_notification_threshold": 0,  # All breaches
        "statute": "N.Y. Gen. Bus. Law § 899-aa",
        "special_requirements": ["Must notify NY AG, DFS, and Division of State Police"],
    },
    "MA": {
        "name": "Massachusetts",
        "notification_days": 30,  # Shorter than HIPAA
        "ag_notification_threshold": 0,  # All breaches
        "statute": "Mass. Gen. Laws ch. 93H",
        "special_requirements": ["Must include specific content in notification", "Must notify AG and OCABR"],
    },
    "FL": {
        "name": "Florida",
        "notification_days": 30,
        "ag_notification_threshold": 500,
        "statute": "Fla. Stat. § 501.171",
        "special_requirements": [],
    },
}


# =============================================================================
# SECTION 3: BREACH RESPONSE MANAGER
# =============================================================================


class BreachResponseManager:
    """
    Manages breach detection, assessment, and notification workflows.

    Implements the HIPAA Breach Notification Rule requirements with
    state-level overlay for multi-jurisdiction compliance.

    RESPONSE WORKFLOW:
    ------------------
    1. Incident reported and documented
    2. Immediate containment actions
    3. Four-factor risk assessment
    4. Breach determination
    5. Notification timeline generated
    6. Notifications sent per timeline
    7. Post-incident review and documentation
    """

    def __init__(
        self,
        organization: str,
        hipaa_covered_entity: bool = True,
        state_jurisdictions: list[str] | None = None,
        incident_log_path: str = "incident_logs/",
    ):
        """
        Initialize breach response manager.

        Args:
            organization: Organization name
            hipaa_covered_entity: Whether org is a HIPAA covered entity
            state_jurisdictions: List of state codes for applicable laws
            incident_log_path: Directory for incident documentation
        """
        self.organization = organization
        self.hipaa_covered_entity = hipaa_covered_entity
        self.state_jurisdictions = state_jurisdictions or []
        self.incident_log_path = Path(incident_log_path)

        self._incidents: dict[str, Incident] = {}
        self._assessments: dict[str, RiskAssessment] = {}
        self._incident_counter = 0

        logger.info(
            f"BreachResponseManager initialized: org={organization}, "
            f"covered_entity={hipaa_covered_entity}, "
            f"states={state_jurisdictions}"
        )

    def report_incident(
        self,
        incident_type: str,
        description: str,
        phi_types_involved: list[str],
        individuals_affected: int,
        discovery_date: str,
        occurrence_date: str = "",
        affected_systems: list[str] | None = None,
        containment_actions: list[str] | None = None,
    ) -> Incident:
        """
        Report a new security incident.

        Args:
            incident_type: Type from IncidentType enum values
            description: Description of the incident
            phi_types_involved: Types of PHI involved
            individuals_affected: Number of individuals affected
            discovery_date: Date incident was discovered (ISO format)
            occurrence_date: Date incident occurred (if known)
            affected_systems: List of affected systems
            containment_actions: Actions taken to contain the incident

        Returns:
            Incident record
        """
        self._incident_counter += 1
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{self._incident_counter:04d}"

        incident = Incident(
            incident_id=incident_id,
            incident_type=IncidentType(incident_type),
            description=description,
            phi_types_involved=phi_types_involved,
            individuals_affected=individuals_affected,
            discovery_date=discovery_date,
            occurrence_date=occurrence_date or discovery_date,
            affected_systems=affected_systems or [],
            containment_actions=containment_actions or [],
            state_jurisdictions=self.state_jurisdictions,
        )

        self._incidents[incident_id] = incident

        logger.info(f"Incident reported: {incident_id} ({incident_type}), affecting {individuals_affected} individuals")

        return incident

    def assess_breach_risk(
        self,
        incident: Incident,
        phi_nature_score: int = 3,
        unauthorized_party_score: int = 3,
        acquisition_score: int = 3,
        mitigation_score: int = 3,
        phi_nature_notes: str = "",
        unauthorized_party_notes: str = "",
        acquisition_notes: str = "",
        mitigation_notes: str = "",
    ) -> RiskAssessment:
        """
        Perform four-factor breach risk assessment.

        The four factors (per HHS guidance) determine whether an
        impermissible use or disclosure is a reportable breach:

        1. Nature and extent of PHI (types, sensitivity, likelihood of
           re-identification)
        2. Unauthorized person who used/received PHI (obligation to
           protect, ability to retain)
        3. Whether PHI was actually acquired or viewed (vs. only
           opportunity)
        4. Extent of mitigation (actions taken to reduce harm)

        Args:
            incident: The incident to assess
            phi_nature_score: 1 (low sensitivity) to 5 (high sensitivity)
            unauthorized_party_score: 1 (low risk) to 5 (high risk)
            acquisition_score: 1 (no evidence acquired) to 5 (confirmed acquired)
            mitigation_score: 1 (no mitigation) to 5 (complete mitigation)
            *_notes: Detailed notes for each factor

        Returns:
            RiskAssessment with breach determination
        """
        assessment = RiskAssessment(
            incident_id=incident.incident_id,
            assessment_date=datetime.now().isoformat(),
            phi_nature_score=phi_nature_score,
            phi_nature_notes=phi_nature_notes or self._auto_assess_phi_nature(incident),
            unauthorized_party_score=unauthorized_party_score,
            unauthorized_party_notes=unauthorized_party_notes,
            acquisition_score=acquisition_score,
            acquisition_notes=acquisition_notes,
            mitigation_score=mitigation_score,
            mitigation_notes=mitigation_notes,
        )

        assessment.calculate_risk()

        # Generate rationale
        assessment.rationale = self._generate_rationale(assessment, incident)

        self._assessments[incident.incident_id] = assessment

        logger.info(
            f"Risk assessment for {incident.incident_id}: "
            f"score={assessment.overall_risk_score:.2f}, "
            f"level={assessment.risk_level.value}, "
            f"reportable={assessment.is_reportable_breach}"
        )

        return assessment

    def generate_notification_timeline(self, incident: Incident) -> NotificationTimeline:
        """
        Generate notification timeline based on incident details.

        Calculates deadlines for all required notifications per
        HIPAA and applicable state laws.

        Args:
            incident: The incident requiring notification

        Returns:
            NotificationTimeline with all deadlines
        """
        discovery = datetime.fromisoformat(incident.discovery_date)
        timeline = NotificationTimeline(incident_id=incident.incident_id, discovery_date=incident.discovery_date)

        # HIPAA federal requirements
        hipaa_deadline = discovery + timedelta(days=60)

        # Individual notification (always required for breaches)
        timeline.deadlines.append(
            NotificationDeadline(
                recipient="Affected Individuals",
                regulation="45 CFR 164.404",
                due_date=hipaa_deadline.isoformat(),
                days_remaining=(hipaa_deadline - datetime.now()).days,
                notes="Written notice by first-class mail or email (if consented)",
            )
        )

        # HHS OCR notification
        if incident.individuals_affected >= 500:
            timeline.deadlines.append(
                NotificationDeadline(
                    recipient="HHS OCR (large breach)",
                    regulation="45 CFR 164.408",
                    due_date=hipaa_deadline.isoformat(),
                    days_remaining=(hipaa_deadline - datetime.now()).days,
                    notes="Submit via HHS breach portal within 60 days",
                )
            )
        else:
            year_end = datetime(discovery.year, 12, 31) + timedelta(days=60)
            timeline.deadlines.append(
                NotificationDeadline(
                    recipient="HHS OCR (small breach)",
                    regulation="45 CFR 164.408",
                    due_date=year_end.isoformat(),
                    days_remaining=(year_end - datetime.now()).days,
                    notes="Submit via HHS breach portal by March 1 of following year",
                )
            )

        # Media notification (500+ in a single state)
        if incident.individuals_affected >= 500:
            timeline.deadlines.append(
                NotificationDeadline(
                    recipient="Prominent Media Outlets",
                    regulation="45 CFR 164.406",
                    due_date=hipaa_deadline.isoformat(),
                    days_remaining=(hipaa_deadline - datetime.now()).days,
                    notes="In states where 500+ individuals are affected",
                )
            )

        # State-specific requirements
        for state_code in incident.state_jurisdictions:
            state_law = STATE_BREACH_LAWS.get(state_code)
            if state_law:
                state_deadline = discovery + timedelta(days=state_law["notification_days"])
                if incident.individuals_affected >= state_law["ag_notification_threshold"]:
                    timeline.deadlines.append(
                        NotificationDeadline(
                            recipient=f"{state_law['name']} Attorney General",
                            regulation=state_law["statute"],
                            due_date=state_deadline.isoformat(),
                            days_remaining=(state_deadline - datetime.now()).days,
                            notes="; ".join(state_law.get("special_requirements", [])),
                        )
                    )

        # Clinical trial-specific notifications
        timeline.deadlines.append(
            NotificationDeadline(
                recipient="Trial Sponsor",
                regulation="ICH E6(R3) / Protocol Agreement",
                due_date=(discovery + timedelta(days=1)).isoformat(),
                days_remaining=max(0, (discovery + timedelta(days=1) - datetime.now()).days),
                notes="Immediate notification per sponsor agreement",
            )
        )

        timeline.deadlines.append(
            NotificationDeadline(
                recipient="IRB/Ethics Committee",
                regulation="45 CFR 46 / ICH E6(R3)",
                due_date=(discovery + timedelta(days=5)).isoformat(),
                days_remaining=max(0, (discovery + timedelta(days=5) - datetime.now()).days),
                notes="Prompt notification of events that may affect participant safety or rights",
            )
        )

        # Sort by due date
        timeline.deadlines.sort(key=lambda d: d.due_date)

        return timeline

    def generate_incident_report(self, incident_id: str) -> str:
        """Generate a comprehensive incident report."""
        incident = self._incidents.get(incident_id)
        if not incident:
            return f"Incident {incident_id} not found"

        assessment = self._assessments.get(incident_id)

        report = f"""
SECURITY INCIDENT REPORT
=========================
Organization: {self.organization}
Report Date: {datetime.now().isoformat()}

INCIDENT DETAILS
-----------------
Incident ID: {incident.incident_id}
Type: {incident.incident_type.value}
Description: {incident.description}
Discovery Date: {incident.discovery_date}
Occurrence Date: {incident.occurrence_date}
Individuals Affected: {incident.individuals_affected}
PHI Types Involved: {", ".join(incident.phi_types_involved)}
Affected Systems: {", ".join(incident.affected_systems)}
Containment Actions: {", ".join(incident.containment_actions)}
Status: {incident.status}
"""

        if assessment:
            report += f"""
FOUR-FACTOR RISK ASSESSMENT
-----------------------------
Factor 1 - PHI Nature/Extent: {assessment.phi_nature_score}/5
  Notes: {assessment.phi_nature_notes}

Factor 2 - Unauthorized Party: {assessment.unauthorized_party_score}/5
  Notes: {assessment.unauthorized_party_notes}

Factor 3 - PHI Acquired/Viewed: {assessment.acquisition_score}/5
  Notes: {assessment.acquisition_notes}

Factor 4 - Mitigation Extent: {assessment.mitigation_score}/5
  Notes: {assessment.mitigation_notes}

Overall Risk Score: {assessment.overall_risk_score:.2f}
Risk Level: {assessment.risk_level.value}
Reportable Breach: {assessment.is_reportable_breach}
Rationale: {assessment.rationale}
"""

        return report

    def _auto_assess_phi_nature(self, incident: Incident) -> str:
        """Auto-generate PHI nature assessment notes."""
        high_sensitivity = {"social_security_numbers", "financial_data", "genetic_data"}
        sensitive_types = set(incident.phi_types_involved) & high_sensitivity

        if sensitive_types:
            return (
                f"High-sensitivity PHI involved: {', '.join(sensitive_types)}. "
                "Significant risk of identity theft or financial harm."
            )
        return f"PHI types: {', '.join(incident.phi_types_involved)}. Moderate risk of harm from disclosure."

    def _generate_rationale(self, assessment: RiskAssessment, incident: Incident) -> str:
        """Generate rationale for breach determination."""
        if assessment.is_reportable_breach:
            return (
                f"Four-factor risk assessment score ({assessment.overall_risk_score:.2f}) "
                f"indicates {assessment.risk_level.value}. "
                f"The {incident.incident_type.value} incident affecting "
                f"{incident.individuals_affected} individuals involving "
                f"{', '.join(incident.phi_types_involved)} constitutes a "
                "reportable breach requiring notification per 45 CFR 164.400-414."
            )
        else:
            return (
                f"Four-factor risk assessment score ({assessment.overall_risk_score:.2f}) "
                "demonstrates low probability of compromise. "
                "This incident does not rise to the level of a reportable breach. "
                "Documentation retained per HIPAA 6-year retention requirement."
            )


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================


def run_breach_response_demo():
    """
    Demonstrate breach response protocol capabilities.

    Shows incident reporting, risk assessment, notification
    timeline generation, and incident reporting.
    """
    logger.info("=" * 60)
    logger.info("BREACH RESPONSE PROTOCOL DEMO")
    logger.info("=" * 60)

    # Initialize manager
    brm = BreachResponseManager(
        organization="Physical AI Oncology Consortium",
        hipaa_covered_entity=True,
        state_jurisdictions=["CA", "TX", "NY", "MA"],
    )

    # Scenario: Unauthorized access to patient imaging database
    incident = brm.report_incident(
        incident_type="unauthorized_access",
        description=(
            "Unauthorized access to patient imaging database detected via "
            "anomalous query patterns. AI training pipeline accessed "
            "non-de-identified DICOM files due to misconfigured access controls."
        ),
        phi_types_involved=["medical_record_numbers", "diagnostic_images", "dates", "institution_names"],
        individuals_affected=150,
        discovery_date="2026-02-01",
        affected_systems=["imaging_pacs", "ai_training_pipeline"],
        containment_actions=[
            "Access revoked for AI training pipeline",
            "Network segment isolated",
            "Forensic investigation initiated",
        ],
    )

    # Perform risk assessment
    assessment = brm.assess_breach_risk(
        incident,
        phi_nature_score=3,
        unauthorized_party_score=2,
        acquisition_score=4,
        mitigation_score=4,
        phi_nature_notes="MRNs and diagnostic images exposed, but no SSN or financial data",
        unauthorized_party_notes="Internal AI system, not external threat actor",
        acquisition_notes="DICOM files were processed by AI pipeline and cached",
        mitigation_notes="Access immediately revoked, cached data purged, audit confirms no exfiltration",
    )

    # Generate notification timeline
    timeline = brm.generate_notification_timeline(incident)

    # Print results
    print(brm.generate_incident_report(incident.incident_id))

    print("\nNOTIFICATION TIMELINE")
    print("-" * 70)
    for deadline in timeline.deadlines:
        print(f"  {deadline.due_date[:10]} | {deadline.recipient:35s} | {deadline.regulation}")

    return {"incident_id": incident.incident_id, "status": "demo_complete"}


if __name__ == "__main__":
    result = run_breach_response_demo()
    print(f"\nDemo completed: {result}")
