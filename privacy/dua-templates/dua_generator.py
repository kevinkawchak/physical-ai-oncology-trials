"""
=============================================================================
Data Use Agreement Generator for Physical AI Oncology Trials
=============================================================================

Generates standardized Data Use Agreements (DUAs) for multi-site
clinical trial data sharing, AI model training collaborations,
and cross-institutional research partnerships.

CLINICAL CONTEXT:
-----------------
Multi-site AI oncology trials require formal DUAs before sharing:
  - Patient data (even de-identified) between institutions
  - AI model training datasets across research networks
  - Genomic and imaging data for federated learning
  - Model weights and outputs from collaborative training
DUAs are legally mandated by HIPAA (45 CFR 164.514(e)) for
Limited Data Sets and are best practice for all research data exchange.

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - Python 3.10+

Optional:
    - reportlab (for PDF generation)
    - jinja2 (for template rendering)

REFERENCES:
    - HIPAA Limited Data Set: 45 CFR 164.514(e)
    - HHS Common DUA Structure (Feb 2025)
    - CDC Core DUA Initiative (Dec 2025)
    - GA4GH Framework for Responsible Sharing

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: DUA CONFIGURATION
# =============================================================================

class DUATemplate(Enum):
    """Available DUA templates."""
    LIMITED_DATA_SET = "limited_data_set"
    DEIDENTIFIED_DATA = "deidentified_data"
    MULTI_SITE_AI_RESEARCH = "multi_site_ai_research"
    GENOMIC_DATA = "genomic_data"
    IMAGING_DATA = "imaging_data"
    FEDERATED_LEARNING = "federated_learning"


class Jurisdiction(Enum):
    """Legal jurisdictions for DUA compliance."""
    US_HIPAA = "us_hipaa"
    EU_GDPR = "eu_gdpr"
    CROSS_BORDER = "cross_border"
    UK_DPA = "uk_dpa"


class SecurityLevel(Enum):
    """Data security requirement levels."""
    STANDARD = "standard"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"


@dataclass
class DUAParty:
    """Party to a Data Use Agreement."""
    organization_name: str
    organization_type: str  # "academic", "industry", "healthcare_system", "government"
    contact_name: str = ""
    contact_title: str = ""
    contact_email: str = ""
    address: str = ""
    ehr_system: str = ""
    irb_name: str = ""
    fwa_number: str = ""  # Federalwide Assurance number


@dataclass
class DataDescription:
    """Description of data covered by the DUA."""
    description: str
    data_types: list[str] = field(default_factory=list)
    record_count_estimate: int = 0
    date_range: str = ""
    identifiability_level: str = "de-identified"  # "phi", "limited_data_set", "de-identified"
    contains_genomic: bool = False
    contains_imaging: bool = False
    file_formats: list[str] = field(default_factory=list)


@dataclass
class SecurityRequirement:
    """Security requirement for data handling."""
    requirement: str
    category: str  # "technical", "administrative", "physical"
    mandatory: bool = True
    description: str = ""


@dataclass
class DUADocument:
    """Generated Data Use Agreement document."""
    agreement_id: str
    template: str
    jurisdiction: str
    provider: DUAParty
    recipient: DUAParty
    data_description: DataDescription
    permitted_uses: list[str]
    prohibited_uses: list[str]
    security_requirements: list[SecurityRequirement]
    retention_period_years: int
    effective_date: str
    expiration_date: str
    sections: dict[str, str] = field(default_factory=dict)
    generated_date: str = ""

    def export(self, output_path: str):
        """Export DUA to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(self._to_dict(), f, indent=2)
        elif path.suffix == ".md":
            with open(path, "w") as f:
                f.write(self._to_markdown())
        elif path.suffix == ".txt":
            with open(path, "w") as f:
                f.write(self._to_text())
        else:
            # Default to markdown
            md_path = path.with_suffix(".md")
            with open(md_path, "w") as f:
                f.write(self._to_markdown())

        logger.info(f"DUA exported to {output_path}")

    def _to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "agreement_id": self.agreement_id,
            "template": self.template,
            "jurisdiction": self.jurisdiction,
            "provider": self.provider.__dict__,
            "recipient": self.recipient.__dict__,
            "data_description": self.data_description.__dict__,
            "permitted_uses": self.permitted_uses,
            "prohibited_uses": self.prohibited_uses,
            "retention_period_years": self.retention_period_years,
            "effective_date": self.effective_date,
            "expiration_date": self.expiration_date,
            "generated_date": self.generated_date,
            "sections": self.sections
        }

    def _to_markdown(self) -> str:
        """Convert to markdown document."""
        md = f"# Data Use Agreement\n\n"
        md += f"**Agreement ID**: {self.agreement_id}\n\n"
        for title, content in self.sections.items():
            md += f"## {title}\n\n{content}\n\n"
        return md

    def _to_text(self) -> str:
        """Convert to plain text document."""
        text = "DATA USE AGREEMENT\n"
        text += "=" * 60 + "\n\n"
        text += f"Agreement ID: {self.agreement_id}\n\n"
        for title, content in self.sections.items():
            text += f"{title.upper()}\n"
            text += "-" * len(title) + "\n"
            text += content + "\n\n"
        return text


# =============================================================================
# SECTION 2: SECURITY REQUIREMENT DEFINITIONS
# =============================================================================

SECURITY_REQUIREMENTS: dict[str, list[dict[str, Any]]] = {
    "standard": [
        {
            "requirement": "encryption_in_transit",
            "category": "technical",
            "description": "All data transmitted using TLS 1.2 or higher"
        },
        {
            "requirement": "access_controls",
            "category": "administrative",
            "description": "Role-based access limited to authorized personnel"
        },
        {
            "requirement": "audit_logging",
            "category": "technical",
            "description": "Access and modification logs maintained for duration of agreement"
        },
    ],
    "enhanced": [
        {
            "requirement": "encryption_at_rest",
            "category": "technical",
            "description": "AES-256 encryption for all stored data"
        },
        {
            "requirement": "encryption_in_transit",
            "category": "technical",
            "description": "All data transmitted using TLS 1.3"
        },
        {
            "requirement": "mfa",
            "category": "technical",
            "description": "Multi-factor authentication for all data access"
        },
        {
            "requirement": "access_controls",
            "category": "administrative",
            "description": "Role-based access with principle of least privilege"
        },
        {
            "requirement": "audit_logging",
            "category": "technical",
            "description": "Tamper-evident audit logs with integrity verification"
        },
        {
            "requirement": "security_training",
            "category": "administrative",
            "description": "Annual HIPAA/privacy training for all personnel with access"
        },
    ],
    "maximum": [
        {
            "requirement": "encryption_at_rest",
            "category": "technical",
            "description": "AES-256 encryption with hardware security modules (HSM)"
        },
        {
            "requirement": "encryption_in_transit",
            "category": "technical",
            "description": "TLS 1.3 with certificate pinning"
        },
        {
            "requirement": "mfa",
            "category": "technical",
            "description": "Hardware token MFA (FIDO2/WebAuthn) for all data access"
        },
        {
            "requirement": "network_segmentation",
            "category": "technical",
            "description": "Dedicated VLAN/subnet for research data with firewall controls"
        },
        {
            "requirement": "access_controls",
            "category": "administrative",
            "description": "Role-based access with quarterly access reviews"
        },
        {
            "requirement": "audit_logging",
            "category": "technical",
            "description": "Real-time SIEM integration with tamper-evident logging"
        },
        {
            "requirement": "security_training",
            "category": "administrative",
            "description": "Annual training plus quarterly security awareness updates"
        },
        {
            "requirement": "vulnerability_management",
            "category": "technical",
            "description": "Quarterly vulnerability scans and annual penetration testing"
        },
        {
            "requirement": "data_loss_prevention",
            "category": "technical",
            "description": "DLP controls preventing unauthorized data exfiltration"
        },
    ]
}


# =============================================================================
# SECTION 3: DUA GENERATOR
# =============================================================================

class DUAGenerator:
    """
    Generates standardized Data Use Agreements for clinical trial data sharing.

    Creates legally-compliant DUA documents incorporating HIPAA requirements,
    institutional policies, and AI research-specific provisions.

    DUA GENERATION WORKFLOW:
    -----------------------
    1. Select template based on data sharing scenario
    2. Configure parties, data description, and permitted uses
    3. Apply jurisdiction-specific requirements
    4. Generate security requirements based on data sensitivity
    5. Produce formatted agreement document
    """

    def __init__(
        self,
        template: str = "multi_site_ai_research",
        jurisdiction: str = "us_hipaa"
    ):
        """
        Initialize DUA generator.

        Args:
            template: DUA template type
            jurisdiction: Legal jurisdiction
        """
        self.template = DUATemplate(template)
        self.jurisdiction = Jurisdiction(jurisdiction)
        self._agreement_counter = 0

        logger.info(f"DUAGenerator initialized: template={template}, jurisdiction={jurisdiction}")

    def generate(
        self,
        data_provider: str,
        data_recipient: str,
        data_description: str,
        permitted_uses: list[str],
        retention_period_years: int = 7,
        security_requirements: list[str] | None = None,
        provider_details: dict | None = None,
        recipient_details: dict | None = None,
        data_details: dict | None = None,
        effective_date: str | None = None
    ) -> DUADocument:
        """
        Generate a Data Use Agreement.

        Args:
            data_provider: Name of data-providing organization
            data_recipient: Name of data-receiving organization
            data_description: Description of data being shared
            permitted_uses: List of permitted data uses
            retention_period_years: How long data may be retained
            security_requirements: Required security controls
            provider_details: Additional provider info
            recipient_details: Additional recipient info
            data_details: Additional data details
            effective_date: Agreement start date (ISO format)

        Returns:
            DUADocument ready for export
        """
        self._agreement_counter += 1
        agreement_id = f"DUA-{datetime.now().strftime('%Y%m%d')}-{self._agreement_counter:04d}"

        effective = effective_date or datetime.now().isoformat()[:10]
        expiration = (
            datetime.fromisoformat(effective) + timedelta(days=365 * retention_period_years)
        ).isoformat()[:10]

        # Build party objects
        provider = DUAParty(
            organization_name=data_provider,
            organization_type=(provider_details or {}).get("type", "healthcare_system"),
            contact_name=(provider_details or {}).get("contact_name", ""),
            contact_email=(provider_details or {}).get("contact_email", ""),
        )

        recipient = DUAParty(
            organization_name=data_recipient,
            organization_type=(recipient_details or {}).get("type", "academic"),
            contact_name=(recipient_details or {}).get("contact_name", ""),
            contact_email=(recipient_details or {}).get("contact_email", ""),
        )

        # Build data description
        data_desc = DataDescription(
            description=data_description,
            data_types=(data_details or {}).get("data_types", []),
            identifiability_level=(data_details or {}).get("identifiability", "de-identified"),
            contains_genomic=(data_details or {}).get("contains_genomic", False),
            contains_imaging=(data_details or {}).get("contains_imaging", False),
        )

        # Determine security level
        security_level = self._determine_security_level(data_desc, security_requirements or [])
        sec_reqs = [
            SecurityRequirement(
                requirement=r["requirement"],
                category=r["category"],
                description=r["description"]
            )
            for r in SECURITY_REQUIREMENTS.get(security_level, SECURITY_REQUIREMENTS["enhanced"])
        ]

        # Generate prohibited uses
        prohibited_uses = self._generate_prohibited_uses()

        # Build agreement sections
        sections = self._generate_sections(
            provider, recipient, data_desc, permitted_uses,
            prohibited_uses, sec_reqs, retention_period_years,
            effective, expiration
        )

        dua = DUADocument(
            agreement_id=agreement_id,
            template=self.template.value,
            jurisdiction=self.jurisdiction.value,
            provider=provider,
            recipient=recipient,
            data_description=data_desc,
            permitted_uses=permitted_uses,
            prohibited_uses=prohibited_uses,
            security_requirements=sec_reqs,
            retention_period_years=retention_period_years,
            effective_date=effective,
            expiration_date=expiration,
            sections=sections,
            generated_date=datetime.now().isoformat()
        )

        logger.info(f"DUA generated: {agreement_id} ({data_provider} -> {data_recipient})")

        return dua

    def _determine_security_level(
        self,
        data_desc: DataDescription,
        requested_requirements: list[str]
    ) -> str:
        """Determine security level based on data sensitivity."""
        if data_desc.identifiability_level == "phi":
            return "maximum"
        elif data_desc.identifiability_level == "limited_data_set":
            return "enhanced"
        elif data_desc.contains_genomic:
            return "enhanced"
        elif "encryption_at_rest" in requested_requirements:
            return "enhanced"
        return "standard"

    def _generate_prohibited_uses(self) -> list[str]:
        """Generate standard prohibited uses."""
        prohibited = [
            "Re-identification of individuals or attempt to contact individuals",
            "Use of data for purposes not specified in this agreement",
            "Sale or commercial licensing of the data to third parties",
            "Sharing data with unauthorized parties or subcontractors",
            "Use of data for insurance underwriting or employment decisions",
            "Publication of results that could identify individual participants",
        ]

        if self.template == DUATemplate.MULTI_SITE_AI_RESEARCH:
            prohibited.extend([
                "Training AI models for uses outside the specified research scope",
                "Deploying AI models trained on this data without written approval",
                "Retaining model weights that memorize individual patient data",
            ])

        return prohibited

    def _generate_sections(
        self,
        provider: DUAParty,
        recipient: DUAParty,
        data_desc: DataDescription,
        permitted_uses: list[str],
        prohibited_uses: list[str],
        sec_reqs: list[SecurityRequirement],
        retention_years: int,
        effective: str,
        expiration: str
    ) -> dict[str, str]:
        """Generate DUA document sections."""
        sections = {}

        sections["1. Parties"] = (
            f"This Data Use Agreement ('Agreement') is entered into by:\n\n"
            f"**Data Provider**: {provider.organization_name} "
            f"({provider.organization_type})\n\n"
            f"**Data Recipient**: {recipient.organization_name} "
            f"({recipient.organization_type})"
        )

        sections["2. Purpose"] = (
            f"The purpose of this Agreement is to establish the terms and conditions "
            f"under which the Data Provider will disclose certain data to the Data "
            f"Recipient for use in research related to physical AI oncology clinical trials.\n\n"
            f"**Data Description**: {data_desc.description}\n\n"
            f"**Identifiability Level**: {data_desc.identifiability_level}"
        )

        sections["3. Permitted Uses"] = (
            "The Data Recipient may use the data solely for the following purposes:\n\n"
            + "\n".join(f"- {use}" for use in permitted_uses)
        )

        sections["4. Prohibited Uses"] = (
            "The Data Recipient shall NOT:\n\n"
            + "\n".join(f"- {use}" for use in prohibited_uses)
        )

        sections["5. Data Security Requirements"] = (
            "The Data Recipient shall implement the following security safeguards:\n\n"
            + "\n".join(
                f"- **{req.requirement}** ({req.category}): {req.description}"
                for req in sec_reqs
            )
        )

        sections["6. Term and Termination"] = (
            f"**Effective Date**: {effective}\n\n"
            f"**Expiration Date**: {expiration}\n\n"
            f"**Data Retention**: Data may be retained for {retention_years} years "
            f"from the effective date. Upon termination or expiration, all data "
            f"and copies must be destroyed or returned within 30 days, with "
            f"written certification of destruction."
        )

        sections["7. Breach Notification"] = (
            "The Data Recipient shall notify the Data Provider within 24 hours "
            "of discovery of any breach or suspected breach of this Agreement, "
            "including any unauthorized use, disclosure, or access to the data. "
            "The notification shall include the nature of the breach, the data "
            "involved, and the corrective actions taken or planned."
        )

        sections["8. Compliance"] = (
            "Both parties shall comply with all applicable federal, state, and "
            "local laws, including but not limited to HIPAA (45 CFR Parts 160 "
            "and 164), FDA regulations (21 CFR Part 11), and any applicable "
            "state privacy laws."
        )

        if self.jurisdiction == Jurisdiction.CROSS_BORDER:
            sections["8. Compliance"] += (
                "\n\nFor data originating from the European Economic Area, "
                "the Data Recipient shall additionally comply with the EU "
                "General Data Protection Regulation (GDPR) and ensure "
                "appropriate transfer mechanisms are in place."
            )

        sections["9. Publication"] = (
            "The Data Recipient may publish results derived from analysis of the "
            "data, provided that:\n\n"
            "- No individual participant can be identified from published results\n"
            "- The Data Provider is given 30 days to review manuscripts prior to submission\n"
            "- The Data Provider is acknowledged as the source of the data\n"
            "- The Data Provider has the right to request removal of any content "
            "that could compromise participant privacy"
        )

        sections["10. Signatures"] = (
            "**Data Provider**\n\n"
            f"Organization: {provider.organization_name}\n"
            "Authorized Representative: _________________________\n"
            "Title: _________________________\n"
            "Date: _________________________\n"
            "Signature: _________________________\n\n"
            "**Data Recipient**\n\n"
            f"Organization: {recipient.organization_name}\n"
            "Authorized Representative: _________________________\n"
            "Title: _________________________\n"
            "Date: _________________________\n"
            "Signature: _________________________"
        )

        return sections


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================

def run_dua_generator_demo():
    """
    Demonstrate DUA generation capabilities.

    Shows creation of a multi-site AI research DUA for
    oncology clinical trial data sharing.
    """
    logger.info("=" * 60)
    logger.info("DUA GENERATOR DEMO")
    logger.info("=" * 60)

    generator = DUAGenerator(
        template="multi_site_ai_research",
        jurisdiction="us_hipaa"
    )

    dua = generator.generate(
        data_provider="Memorial Sloan Kettering Cancer Center",
        data_recipient="Physical AI Oncology Consortium",
        data_description=(
            "De-identified CT imaging datasets and corresponding treatment "
            "outcomes for non-small cell lung cancer patients enrolled in "
            "physical AI-guided surgical planning trials"
        ),
        permitted_uses=[
            "AI model training for surgical planning optimization",
            "Validation of digital twin tumor models",
            "Publication of aggregate research findings",
            "Cross-institutional model performance benchmarking"
        ],
        retention_period_years=7,
        security_requirements=["encryption_at_rest", "encryption_in_transit", "mfa"],
        data_details={
            "data_types": ["DICOM CT", "treatment outcomes", "pathology reports"],
            "identifiability": "de-identified",
            "contains_imaging": True
        }
    )

    print(f"\nGenerated DUA: {dua.agreement_id}")
    print(f"Template: {dua.template}")
    print(f"Period: {dua.effective_date} to {dua.expiration_date}")
    print(f"Security requirements: {len(dua.security_requirements)}")
    print(f"Permitted uses: {len(dua.permitted_uses)}")
    print(f"Prohibited uses: {len(dua.prohibited_uses)}")
    print(f"Sections: {len(dua.sections)}")
    print()

    # Print section titles
    for title in dua.sections:
        print(f"  Section: {title}")

    return {"agreement_id": dua.agreement_id, "status": "demo_complete"}


if __name__ == "__main__":
    result = run_dua_generator_demo()
    print(f"\nDemo completed: {result}")
