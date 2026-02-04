"""
=============================================================================
ICH E6(R3) GCP Compliance Checker for Physical AI Oncology Trials
=============================================================================

Verifies compliance with ICH E6(R3) Good Clinical Practice guidelines,
with emphasis on provisions relevant to AI-enabled clinical trials.

CLINICAL CONTEXT:
-----------------
ICH E6(R3), adopted January 2025, is the first GCP revision to
explicitly address digital technologies in clinical trials:
  - Risk-Based Quality Management (RBQM) throughout trial lifecycle
  - Digital technology validation and governance requirements
  - AI/ML tool justification, validation, and supervision
  - Data governance as shared sponsor/investigator responsibility
  - Service provider oversight (including AI/technology vendors)

Effective dates:
  - EU: July 23, 2025
  - U.S. FDA: September 9, 2025

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - Python 3.10+

REFERENCES:
    - ICH E6(R3) Step 4 Final Guideline (Jan 2025)
    - FDA Adoption of ICH E6(R3) (Sep 2025)
    - ICH E6(R3) Annex 1: Implementation guidance
    - ICH E6(R3) Annex 2: Decentralized trials (expected 2026)

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: COMPLIANCE CATEGORIES
# =============================================================================

class GuidelineVersion(Enum):
    """ICH E6 guideline versions."""
    E6_R2 = "E6_R2"  # Previous version
    E6_R3 = "E6_R3"  # Current version (Jan 2025)


class Jurisdiction(Enum):
    """Regulatory jurisdictions."""
    US_FDA = "us_fda"
    EU_EMA = "eu_ema"
    JAPAN_PMDA = "japan_pmda"
    CANADA_HC = "canada_hc"
    ICH_GLOBAL = "ich_global"


class ComplianceStatus(Enum):
    """Compliance status for each requirement."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"
    NOT_APPLICABLE = "not_applicable"


class FindingSeverity(Enum):
    """Severity of compliance findings."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    OBSERVATION = "observation"


@dataclass
class ComplianceFinding:
    """Individual compliance finding."""
    finding_id: str
    category: str
    requirement: str
    status: ComplianceStatus
    severity: FindingSeverity
    description: str
    recommendation: str = ""
    guideline_reference: str = ""
    ai_specific: bool = False


@dataclass
class ComplianceReport:
    """Complete compliance assessment report."""
    report_id: str
    guideline_version: str
    jurisdiction: str
    assessment_date: str
    overall_score: float = 0.0
    total_findings: int = 0
    critical_findings: int = 0
    major_findings: int = 0
    minor_findings: int = 0
    findings: list[ComplianceFinding] = field(default_factory=list)
    category_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "guideline_version": self.guideline_version,
            "jurisdiction": self.jurisdiction,
            "assessment_date": self.assessment_date,
            "overall_score": self.overall_score,
            "total_findings": self.total_findings,
            "critical_findings": self.critical_findings,
            "major_findings": self.major_findings,
            "minor_findings": self.minor_findings,
            "category_scores": self.category_scores
        }


# =============================================================================
# SECTION 2: GCP COMPLIANCE REQUIREMENTS
# =============================================================================

# ICH E6(R3) compliance requirements organized by category
GCP_REQUIREMENTS: dict[str, list[dict[str, Any]]] = {
    "quality_management": [
        {
            "id": "QM-001",
            "requirement": "Risk-Based Quality Management System (RBQM) documented",
            "description": "Sponsor has implemented a systematic risk-based approach to quality management, identifying critical data and processes",
            "reference": "E6(R3) Section 5",
            "ai_specific": False,
            "severity": "critical"
        },
        {
            "id": "QM-002",
            "requirement": "Critical-to-quality factors identified for trial",
            "description": "Factors essential to trial integrity identified during Quality by Design planning",
            "reference": "E6(R3) Section 5.0",
            "ai_specific": False,
            "severity": "critical"
        },
        {
            "id": "QM-003",
            "requirement": "Risk assessment documented for AI/ML components",
            "description": "Specific risks from AI/ML components identified and mitigated",
            "reference": "E6(R3) Section 5",
            "ai_specific": True,
            "severity": "major"
        },
        {
            "id": "QM-004",
            "requirement": "Quality tolerance limits defined for AI outputs",
            "description": "Acceptable performance boundaries set for AI-generated results",
            "reference": "E6(R3) Section 5.0",
            "ai_specific": True,
            "severity": "major"
        }
    ],
    "investigator_responsibilities": [
        {
            "id": "IR-001",
            "requirement": "Investigator qualifications documented",
            "description": "PI has appropriate training and experience, including understanding of AI tools used",
            "reference": "E6(R3) Section 4.1",
            "ai_specific": False,
            "severity": "critical"
        },
        {
            "id": "IR-002",
            "requirement": "AI tool training provided to site staff",
            "description": "Clinical site personnel trained on AI system operation, limitations, and override procedures",
            "reference": "E6(R3) Section 4.2",
            "ai_specific": True,
            "severity": "major"
        },
        {
            "id": "IR-003",
            "requirement": "Informed consent process includes AI disclosure",
            "description": "Consent process addresses AI involvement, limitations, and participant rights",
            "reference": "E6(R3) Section 4.8",
            "ai_specific": True,
            "severity": "critical"
        }
    ],
    "sponsor_obligations": [
        {
            "id": "SO-001",
            "requirement": "Protocol design appropriate for trial objectives",
            "description": "Trial design scientifically sound with justification for AI components",
            "reference": "E6(R3) Section 5.2",
            "ai_specific": False,
            "severity": "critical"
        },
        {
            "id": "SO-002",
            "requirement": "Service provider oversight documented",
            "description": "Sponsor oversight of AI technology vendors and service providers",
            "reference": "E6(R3) Section 5.2",
            "ai_specific": True,
            "severity": "major"
        },
        {
            "id": "SO-003",
            "requirement": "Monitoring plan includes AI system performance",
            "description": "Risk-based monitoring plan addresses AI output quality and drift",
            "reference": "E6(R3) Section 5.18",
            "ai_specific": True,
            "severity": "major"
        }
    ],
    "digital_technology_provisions": [
        {
            "id": "DT-001",
            "requirement": "Digital technology use justified in protocol",
            "description": "AI/digital tools justified in relation to trial design, endpoints, and participant population",
            "reference": "E6(R3) Annex 1",
            "ai_specific": True,
            "severity": "major"
        },
        {
            "id": "DT-002",
            "requirement": "Digital tool validation documented",
            "description": "AI/ML systems validated for intended use with documented evidence",
            "reference": "E6(R3) Annex 1",
            "ai_specific": True,
            "severity": "critical"
        },
        {
            "id": "DT-003",
            "requirement": "Digital technology risks communicated to participants",
            "description": "Risks specific to digital/AI technology included in informed consent",
            "reference": "E6(R3) Annex 1",
            "ai_specific": True,
            "severity": "major"
        },
        {
            "id": "DT-004",
            "requirement": "Computerized systems validated per 21 CFR Part 11",
            "description": "Electronic records systems validated with appropriate access controls and audit trails",
            "reference": "E6(R3) Annex 1 / 21 CFR Part 11",
            "ai_specific": False,
            "severity": "critical"
        }
    ],
    "data_governance": [
        {
            "id": "DG-001",
            "requirement": "Data governance plan documented",
            "description": "Comprehensive plan for data integrity, traceability, and security",
            "reference": "E6(R3) Section 5.5",
            "ai_specific": False,
            "severity": "critical"
        },
        {
            "id": "DG-002",
            "requirement": "Data flow from AI systems documented",
            "description": "Complete data lineage from AI input through output and clinical use",
            "reference": "E6(R3) Section 5.5",
            "ai_specific": True,
            "severity": "major"
        },
        {
            "id": "DG-003",
            "requirement": "Audit trail maintained for AI-generated data",
            "description": "All AI inputs, outputs, and clinical decisions traceable",
            "reference": "E6(R3) Section 5.5 / 21 CFR Part 11",
            "ai_specific": True,
            "severity": "critical"
        },
        {
            "id": "DG-004",
            "requirement": "Data quality controls for AI training data",
            "description": "Quality standards for data used to train AI models in the trial",
            "reference": "E6(R3) Section 5.5",
            "ai_specific": True,
            "severity": "major"
        }
    ],
    "safety_reporting": [
        {
            "id": "SR-001",
            "requirement": "Adverse event reporting procedures include AI-related events",
            "description": "AE/SAE reporting covers events potentially caused or influenced by AI systems",
            "reference": "E6(R3) Section 4.11",
            "ai_specific": True,
            "severity": "critical"
        },
        {
            "id": "SR-002",
            "requirement": "DSMB charter addresses AI performance monitoring",
            "description": "Data Safety Monitoring Board reviews AI system performance alongside safety data",
            "reference": "E6(R3) Section 5.5",
            "ai_specific": True,
            "severity": "major"
        }
    ]
}


# =============================================================================
# SECTION 3: GCP COMPLIANCE CHECKER
# =============================================================================

class GCPComplianceChecker:
    """
    Verifies ICH E6(R3) GCP compliance for AI-enabled oncology trials.

    Assesses trial documentation against E6(R3) requirements with
    emphasis on digital technology provisions, AI/ML tool governance,
    and data integrity requirements.

    ASSESSMENT WORKFLOW:
    -------------------
    1. Load compliance requirements for specified version/jurisdiction
    2. Evaluate each requirement against available documentation
    3. Generate findings with severity classification
    4. Calculate category and overall compliance scores
    5. Produce actionable compliance report
    """

    def __init__(
        self,
        guideline_version: str = "E6_R3",
        jurisdiction: str = "us_fda"
    ):
        """
        Initialize GCP compliance checker.

        Args:
            guideline_version: "E6_R2" or "E6_R3"
            jurisdiction: "us_fda", "eu_ema", etc.
        """
        self.guideline_version = GuidelineVersion(guideline_version)
        self.jurisdiction = Jurisdiction(jurisdiction)
        self._finding_counter = 0

        logger.info(
            f"GCPComplianceChecker initialized: version={guideline_version}, "
            f"jurisdiction={jurisdiction}"
        )

    def verify_compliance(
        self,
        protocol_path: str = "",
        trial_master_file: str = "",
        check_categories: list[str] | None = None,
        ai_components_present: bool = True
    ) -> ComplianceReport:
        """
        Run compliance verification against E6(R3) requirements.

        Args:
            protocol_path: Path to trial protocol document
            trial_master_file: Path to Trial Master File directory
            check_categories: Specific categories to check
            ai_components_present: Whether trial uses AI/ML components

        Returns:
            ComplianceReport with findings and scores
        """
        report_id = f"GCP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        categories = check_categories or list(GCP_REQUIREMENTS.keys())

        report = ComplianceReport(
            report_id=report_id,
            guideline_version=self.guideline_version.value,
            jurisdiction=self.jurisdiction.value,
            assessment_date=datetime.now().isoformat()
        )

        for category in categories:
            requirements = GCP_REQUIREMENTS.get(category, [])
            category_findings = []

            for req in requirements:
                # Skip AI-specific requirements if no AI components
                if req.get("ai_specific") and not ai_components_present:
                    continue

                # Simulate assessment (in production, this would check
                # actual documents against requirements)
                finding = self._assess_requirement(req, category, protocol_path, trial_master_file)
                report.findings.append(finding)
                category_findings.append(finding)

            # Calculate category score
            if category_findings:
                compliant = sum(
                    1 for f in category_findings
                    if f.status == ComplianceStatus.COMPLIANT
                )
                report.category_scores[category] = (
                    compliant / len(category_findings)
                ) * 100

        # Calculate overall metrics
        report.total_findings = len(report.findings)
        report.critical_findings = sum(
            1 for f in report.findings
            if f.severity == FindingSeverity.CRITICAL
            and f.status != ComplianceStatus.COMPLIANT
        )
        report.major_findings = sum(
            1 for f in report.findings
            if f.severity == FindingSeverity.MAJOR
            and f.status != ComplianceStatus.COMPLIANT
        )
        report.minor_findings = sum(
            1 for f in report.findings
            if f.severity == FindingSeverity.MINOR
            and f.status != ComplianceStatus.COMPLIANT
        )

        if report.total_findings > 0:
            compliant_total = sum(
                1 for f in report.findings
                if f.status == ComplianceStatus.COMPLIANT
            )
            report.overall_score = (compliant_total / report.total_findings) * 100

        logger.info(
            f"Compliance report {report_id}: score={report.overall_score:.1f}%, "
            f"findings={report.total_findings}, "
            f"critical={report.critical_findings}"
        )

        return report

    def generate_report_text(self, report: ComplianceReport) -> str:
        """Generate human-readable compliance report."""
        text = f"""
ICH E6(R3) GCP COMPLIANCE REPORT
==================================
Report ID: {report.report_id}
Guideline: {report.guideline_version}
Jurisdiction: {report.jurisdiction}
Assessment Date: {report.assessment_date}

OVERALL COMPLIANCE SCORE: {report.overall_score:.1f}%

FINDINGS SUMMARY
-----------------
Total Requirements Assessed: {report.total_findings}
Critical Non-Compliance: {report.critical_findings}
Major Non-Compliance: {report.major_findings}
Minor Non-Compliance: {report.minor_findings}

CATEGORY SCORES
-----------------
"""
        for category, score in report.category_scores.items():
            text += f"  {category:40s} {score:6.1f}%\n"

        text += "\nDETAILED FINDINGS\n-----------------\n"

        for finding in report.findings:
            if finding.status != ComplianceStatus.COMPLIANT:
                text += f"\n[{finding.severity.value.upper()}] {finding.finding_id}\n"
                text += f"  Category: {finding.category}\n"
                text += f"  Requirement: {finding.requirement}\n"
                text += f"  Status: {finding.status.value}\n"
                text += f"  Description: {finding.description}\n"
                text += f"  Reference: {finding.guideline_reference}\n"
                if finding.recommendation:
                    text += f"  Recommendation: {finding.recommendation}\n"
                if finding.ai_specific:
                    text += f"  [AI-SPECIFIC REQUIREMENT]\n"

        return text

    def _assess_requirement(
        self,
        req: dict,
        category: str,
        protocol_path: str,
        tmf_path: str
    ) -> ComplianceFinding:
        """
        Assess a single compliance requirement.

        In a production system, this would analyze actual trial
        documentation. Here it creates a structured finding template
        for manual assessment.
        """
        self._finding_counter += 1
        finding_id = f"F-{self._finding_counter:04d}"

        # Default to not_assessed (manual review required)
        finding = ComplianceFinding(
            finding_id=finding_id,
            category=category,
            requirement=req["requirement"],
            status=ComplianceStatus.NOT_ASSESSED,
            severity=FindingSeverity(req.get("severity", "major")),
            description=req["description"],
            guideline_reference=req.get("reference", ""),
            ai_specific=req.get("ai_specific", False),
            recommendation=(
                f"Review {category} documentation against "
                f"{req.get('reference', 'ICH E6(R3)')} requirements"
            )
        )

        return finding


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================

def run_gcp_compliance_demo():
    """
    Demonstrate GCP compliance checking for AI oncology trials.
    """
    logger.info("=" * 60)
    logger.info("GCP COMPLIANCE CHECKER DEMO")
    logger.info("=" * 60)

    checker = GCPComplianceChecker(
        guideline_version="E6_R3",
        jurisdiction="us_fda"
    )

    # Run compliance verification
    report = checker.verify_compliance(
        check_categories=[
            "quality_management",
            "investigator_responsibilities",
            "sponsor_obligations",
            "digital_technology_provisions",
            "data_governance",
            "safety_reporting"
        ],
        ai_components_present=True
    )

    print(f"\nGCP Compliance Report: {report.report_id}")
    print(f"Overall Score: {report.overall_score:.1f}%")
    print(f"Total Requirements: {report.total_findings}")

    print("\nCategory Scores:")
    for category, score in report.category_scores.items():
        print(f"  {category:40s} {score:6.1f}%")

    ai_findings = [f for f in report.findings if f.ai_specific]
    print(f"\nAI-Specific Requirements: {len(ai_findings)}")
    for finding in ai_findings[:5]:
        print(f"  [{finding.severity.value:8s}] {finding.requirement[:65]}")

    return {"report_id": report.report_id, "status": "demo_complete"}


if __name__ == "__main__":
    result = run_gcp_compliance_demo()
    print(f"\nDemo completed: {result}")
