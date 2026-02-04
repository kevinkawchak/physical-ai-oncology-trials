"""
=============================================================================
Regulatory Tracker for Physical AI Oncology Trials
=============================================================================

Tracks and monitors regulatory developments across multiple jurisdictions
for AI-enabled oncology clinical trials, providing structured intelligence
on guidance documents, enforcement actions, and emerging requirements.

CLINICAL CONTEXT:
-----------------
Trial teams must track regulatory changes from:
  - FDA (CDRH, CDER, OCE): Device clearances, draft guidance, enforcement
  - EMA: MDR updates, EU AI Act implementation, clinical trial regulation
  - ICH: GCP revisions, harmonized technical requirements
  - WHO: Global AI governance guidance
  - HHS/OCR: HIPAA enforcement, privacy rule updates
  - National agencies: PMDA, TGA, Health Canada, MHRA

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - Python 3.10+

REFERENCES:
    - FDA AI/ML Device Database
    - EU AI Act Implementation Timeline
    - ICH Guidelines Database
    - WHO Regulatory Considerations on AI for Health

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
# SECTION 1: DATA STRUCTURES
# =============================================================================

class RegulatoryJurisdiction(Enum):
    """Regulatory jurisdictions tracked."""
    FDA = "fda"
    EMA = "ema"
    ICH = "ich"
    WHO = "who"
    HHS_OCR = "hhs_ocr"
    PMDA = "pmda"
    TGA = "tga"
    HEALTH_CANADA = "health_canada"
    MHRA = "mhra"


class UpdateType(Enum):
    """Types of regulatory updates."""
    GUIDANCE_DRAFT = "guidance_draft"
    GUIDANCE_FINAL = "guidance_final"
    DEVICE_CLEARANCE = "device_clearance"
    ENFORCEMENT_ACTION = "enforcement_action"
    REGULATION_PROPOSED = "regulation_proposed"
    REGULATION_FINAL = "regulation_final"
    STANDARD_UPDATE = "standard_update"
    MEETING_NOTICE = "meeting_notice"
    POLICY_STATEMENT = "policy_statement"


class ImpactLevel(Enum):
    """Impact level for trial operations."""
    CRITICAL = "critical"       # Immediate action required
    HIGH = "high"               # Plan changes needed
    MEDIUM = "medium"           # Monitor and prepare
    LOW = "low"                 # Informational
    INFORMATIONAL = "informational"


@dataclass
class RegulatoryUpdate:
    """A single regulatory update or development."""
    update_id: str
    jurisdiction: str
    title: str
    date: str
    update_type: str
    impact_level: str
    summary: str
    affected_areas: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    source_url: str = ""
    comment_deadline: str = ""
    effective_date: str = ""

    def to_dict(self) -> dict:
        return {
            "update_id": self.update_id,
            "jurisdiction": self.jurisdiction,
            "title": self.title,
            "date": self.date,
            "update_type": self.update_type,
            "impact_level": self.impact_level,
            "summary": self.summary,
            "affected_areas": self.affected_areas
        }


@dataclass
class ComplianceDeadline:
    """A regulatory compliance deadline."""
    jurisdiction: str
    regulation: str
    requirement: str
    deadline: str
    status: str = "upcoming"  # "upcoming", "imminent", "overdue", "met"
    notes: str = ""


# =============================================================================
# SECTION 2: REGULATORY TIMELINE DATABASE
# =============================================================================

# Key regulatory milestones and deadlines for AI oncology trials
REGULATORY_TIMELINE: list[dict[str, Any]] = [
    {
        "jurisdiction": "fda",
        "title": "FDA Draft Guidance: AI-Enabled Device Software Functions",
        "date": "2025-01-15",
        "type": "guidance_draft",
        "impact": "critical",
        "summary": "Comprehensive lifecycle management and submission recommendations for AI/ML-enabled devices. TPLC approach for 510(k), PMA, and De Novo submissions.",
        "affected_areas": ["device_submissions", "ai_ml_models", "post_market"],
        "action_items": ["Review guidance against current submission strategy", "Update device submission documentation"],
        "source_url": "https://www.fda.gov/news-events/press-announcements/fda-issues-comprehensive-draft-guidance-developers-artificial-intelligence-enabled-medical-devices"
    },
    {
        "jurisdiction": "fda",
        "title": "FDA Draft Guidance: AI in Drug Development (CDER)",
        "date": "2025-01-07",
        "type": "guidance_draft",
        "impact": "high",
        "summary": "Risk-based credibility assessment framework for AI models across nonclinical, clinical, postmarketing, and manufacturing phases.",
        "affected_areas": ["drug_development", "ai_ml_models", "clinical_trials"],
        "action_items": ["Assess AI model credibility per proposed framework"],
        "source_url": "https://www.fda.gov/news-events/press-announcements/fda-proposes-framework-advance-credibility-ai-models-used-drug-and-biological-product-submissions"
    },
    {
        "jurisdiction": "ich",
        "title": "ICH E6(R3) GCP Adopted (Step 4)",
        "date": "2025-01-06",
        "type": "standard_update",
        "impact": "critical",
        "summary": "Comprehensive GCP revision with RBQM, digital technology provisions, AI/ML validation requirements, and shared data governance.",
        "affected_areas": ["clinical_trials", "data_governance", "digital_technology", "quality_management"],
        "action_items": ["Update SOPs for RBQM", "Document AI tool validation", "Revise monitoring plans"],
        "source_url": "https://database.ich.org/sites/default/files/ICH_E6(R3)_Step4_FinalGuideline_2025_0106.pdf"
    },
    {
        "jurisdiction": "ema",
        "title": "ICH E6(R3) Effective in EU",
        "date": "2025-07-23",
        "type": "regulation_final",
        "impact": "critical",
        "summary": "E6(R3) GCP becomes binding for all clinical trials conducted in EU member states.",
        "affected_areas": ["clinical_trials", "eu_sites"],
        "action_items": ["Ensure EU site compliance with E6(R3)", "Update EU-specific trial documents"]
    },
    {
        "jurisdiction": "fda",
        "title": "FDA PCCP Guidance Finalized",
        "date": "2025-08-01",
        "type": "guidance_final",
        "impact": "high",
        "summary": "Predetermined Change Control Plans allow AI device updates post-market without new submissions if changes fall within pre-approved boundaries.",
        "affected_areas": ["device_submissions", "ai_ml_models", "post_market"],
        "action_items": ["Develop PCCP for adaptive AI components", "Define modification boundaries"],
        "source_url": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence"
    },
    {
        "jurisdiction": "fda",
        "title": "FDA Publishes ICH E6(R3) in U.S.",
        "date": "2025-09-09",
        "type": "regulation_final",
        "impact": "critical",
        "summary": "E6(R3) GCP requirements available for U.S. clinical trial sites. Transition period from E6(R2).",
        "affected_areas": ["clinical_trials", "us_sites"],
        "action_items": ["Begin E6(R3) transition for U.S. sites"]
    },
    {
        "jurisdiction": "fda",
        "title": "FDA RWE Guidance Finalized for Devices",
        "date": "2025-12-15",
        "type": "guidance_final",
        "impact": "medium",
        "summary": "FDA no longer requires identifiable patient-level data from all RWD sources. Operational February 16, 2026.",
        "affected_areas": ["real_world_evidence", "post_market"],
        "action_items": ["Review RWE strategy for post-market surveillance"],
        "effective_date": "2026-02-16"
    },
    {
        "jurisdiction": "fda",
        "title": "QMSR Effective (ISO 13485 Alignment)",
        "date": "2026-02-02",
        "type": "regulation_final",
        "impact": "high",
        "summary": "Quality Management System Regulation aligns U.S. requirements with ISO 13485 international standards.",
        "affected_areas": ["quality_management", "device_manufacturing"],
        "action_items": ["Verify QMS alignment with ISO 13485", "Update quality procedures"]
    },
    {
        "jurisdiction": "ema",
        "title": "EU AI Act: High-Risk AI Requirements",
        "date": "2026-08-02",
        "type": "regulation_final",
        "impact": "critical",
        "summary": "Full transparency, conformity assessment, and documentation requirements for high-risk AI systems. Penalties up to 35M EUR or 7% global turnover.",
        "affected_areas": ["ai_ml_models", "eu_compliance", "transparency"],
        "action_items": ["Complete AI Act conformity assessment", "Implement transparency requirements", "Document AI system governance"]
    },
    {
        "jurisdiction": "ema",
        "title": "EU AI Act: CE-Marked Medical Devices",
        "date": "2027-08-02",
        "type": "regulation_final",
        "impact": "critical",
        "summary": "AI systems in CE-marked medical devices under MDR/IVDR fully regulated under EU AI Act.",
        "affected_areas": ["ai_ml_models", "eu_compliance", "device_submissions"],
        "action_items": ["Prepare dual MDR + AI Act compliance", "Update technical files"]
    }
]


# =============================================================================
# SECTION 3: REGULATORY TRACKER
# =============================================================================

class RegulatoryTracker:
    """
    Monitors and tracks regulatory developments for AI oncology trials.

    Provides structured intelligence on guidance documents,
    enforcement actions, and compliance deadlines across
    multiple jurisdictions.

    TRACKING CAPABILITIES:
    ---------------------
    1. Regulatory timeline with milestone tracking
    2. Compliance deadline management
    3. Impact assessment for trial operations
    4. Jurisdiction-specific filtering
    5. Action item generation
    """

    def __init__(
        self,
        jurisdictions: list[str] | None = None,
        topics: list[str] | None = None
    ):
        """
        Initialize regulatory tracker.

        Args:
            jurisdictions: Jurisdictions to track
            topics: Topic areas to focus on
        """
        self.jurisdictions = jurisdictions or ["fda", "ema", "ich"]
        self.topics = topics or ["ai_ml_devices", "oncology", "clinical_trials"]

        # Load timeline database
        self._updates: list[RegulatoryUpdate] = []
        self._load_timeline()

        logger.info(
            f"RegulatoryTracker initialized: jurisdictions={self.jurisdictions}, "
            f"topics={self.topics}, updates_loaded={len(self._updates)}"
        )

    def _load_timeline(self):
        """Load regulatory timeline into structured updates."""
        for idx, entry in enumerate(REGULATORY_TIMELINE):
            if entry["jurisdiction"] in self.jurisdictions:
                self._updates.append(RegulatoryUpdate(
                    update_id=f"REG-{idx + 1:04d}",
                    jurisdiction=entry["jurisdiction"],
                    title=entry["title"],
                    date=entry["date"],
                    update_type=entry["type"],
                    impact_level=entry["impact"],
                    summary=entry["summary"],
                    affected_areas=entry.get("affected_areas", []),
                    action_items=entry.get("action_items", []),
                    source_url=entry.get("source_url", ""),
                    effective_date=entry.get("effective_date", "")
                ))

    def get_recent_updates(
        self,
        days: int = 90,
        jurisdiction: str | None = None,
        impact_level: str | None = None
    ) -> list[RegulatoryUpdate]:
        """
        Get recent regulatory updates.

        Args:
            days: Number of days to look back
            jurisdiction: Filter by jurisdiction
            impact_level: Filter by impact level

        Returns:
            List of matching regulatory updates
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()[:10]
        filtered = self._updates

        if jurisdiction:
            filtered = [u for u in filtered if u.jurisdiction == jurisdiction]
        if impact_level:
            filtered = [u for u in filtered if u.impact_level == impact_level]

        # Sort by date descending
        filtered.sort(key=lambda u: u.date, reverse=True)

        return filtered

    def get_upcoming_deadlines(
        self,
        days_ahead: int = 180
    ) -> list[ComplianceDeadline]:
        """
        Get upcoming compliance deadlines.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of upcoming compliance deadlines
        """
        today = datetime.now().isoformat()[:10]
        future = (datetime.now() + timedelta(days=days_ahead)).isoformat()[:10]

        deadlines = []
        for update in self._updates:
            effective = update.effective_date or update.date
            if today <= effective <= future:
                days_until = (
                    datetime.fromisoformat(effective) - datetime.now()
                ).days

                status = "upcoming"
                if days_until <= 30:
                    status = "imminent"
                elif days_until <= 0:
                    status = "overdue"

                deadlines.append(ComplianceDeadline(
                    jurisdiction=update.jurisdiction,
                    regulation=update.title,
                    requirement=update.summary[:100] + "...",
                    deadline=effective,
                    status=status,
                    notes=f"{days_until} days remaining"
                ))

        deadlines.sort(key=lambda d: d.deadline)
        return deadlines

    def generate_action_items(
        self,
        jurisdiction: str | None = None
    ) -> list[dict[str, str]]:
        """
        Generate consolidated action items from regulatory updates.

        Args:
            jurisdiction: Filter by jurisdiction

        Returns:
            List of action items with context
        """
        items = []
        filtered = self._updates

        if jurisdiction:
            filtered = [u for u in filtered if u.jurisdiction == jurisdiction]

        for update in filtered:
            for action in update.action_items:
                items.append({
                    "action": action,
                    "source": update.title,
                    "jurisdiction": update.jurisdiction,
                    "impact": update.impact_level,
                    "date": update.date
                })

        return items

    def generate_landscape_report(self) -> str:
        """Generate a comprehensive regulatory landscape report."""
        report = f"""
REGULATORY LANDSCAPE REPORT
=============================
Report Date: {datetime.now().isoformat()[:10]}
Jurisdictions: {', '.join(self.jurisdictions)}
Topics: {', '.join(self.topics)}
Total Updates Tracked: {len(self._updates)}

IMPACT SUMMARY
--------------
"""
        # Count by impact level
        for level in ["critical", "high", "medium", "low"]:
            count = sum(1 for u in self._updates if u.impact_level == level)
            if count > 0:
                report += f"  {level.upper()}: {count} updates\n"

        report += "\nUPCOMING DEADLINES (Next 180 Days)\n"
        report += "-" * 40 + "\n"

        deadlines = self.get_upcoming_deadlines()
        for dl in deadlines:
            report += f"  {dl.deadline} [{dl.status:9s}] {dl.regulation[:55]}\n"

        if not deadlines:
            report += "  No deadlines in the next 180 days.\n"

        report += "\nACTION ITEMS\n"
        report += "-" * 40 + "\n"

        actions = self.generate_action_items()
        for item in actions:
            report += f"  [{item['jurisdiction'].upper():3s}] {item['action']}\n"

        report += f"\nKEY REGULATORY UPDATES\n"
        report += "-" * 40 + "\n"

        for update in sorted(self._updates, key=lambda u: u.date, reverse=True)[:10]:
            report += (
                f"\n  [{update.jurisdiction.upper():3s}] {update.title}\n"
                f"  Date: {update.date} | Impact: {update.impact_level}\n"
                f"  {update.summary[:100]}...\n"
            )

        return report


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================

def run_regulatory_tracker_demo():
    """
    Demonstrate regulatory intelligence tracking for AI oncology trials.
    """
    logger.info("=" * 60)
    logger.info("REGULATORY TRACKER DEMO")
    logger.info("=" * 60)

    tracker = RegulatoryTracker(
        jurisdictions=["fda", "ema", "ich"],
        topics=["ai_ml_devices", "oncology", "clinical_trials"]
    )

    # Get all tracked updates
    all_updates = tracker.get_recent_updates(days=365 * 3)
    print(f"\nTotal updates tracked: {len(all_updates)}")

    # Show recent high-impact updates
    critical = tracker.get_recent_updates(days=365, impact_level="critical")
    print(f"\nCritical updates (past year): {len(critical)}")
    for update in critical[:5]:
        print(f"  [{update.jurisdiction.upper()}] {update.date}: {update.title}")

    # Show upcoming deadlines
    deadlines = tracker.get_upcoming_deadlines(days=365)
    print(f"\nUpcoming deadlines (next year): {len(deadlines)}")
    for dl in deadlines[:5]:
        print(f"  {dl.deadline} [{dl.status}]: {dl.regulation[:55]}")

    # Generate action items
    actions = tracker.generate_action_items()
    print(f"\nTotal action items: {len(actions)}")
    for item in actions[:5]:
        print(f"  [{item['jurisdiction'].upper()}] {item['action'][:65]}")

    return {"updates": len(all_updates), "status": "demo_complete"}


if __name__ == "__main__":
    result = run_regulatory_tracker_demo()
    print(f"\nDemo completed: {result}")
