"""
=============================================================================
FDA Submission Tracker for Physical AI Oncology Trials
=============================================================================

Tracks and manages FDA regulatory submissions for AI/ML-enabled oncology
devices, including pre-submission meetings, marketing authorizations,
and post-market surveillance obligations.

CLINICAL CONTEXT:
-----------------
Physical AI oncology devices may require FDA authorization through:
  - 510(k): Substantial equivalence to a predicate device
  - De Novo: Novel devices without a predicate (new risk classification)
  - PMA: Pre-market approval for high-risk (Class III) devices
  - Breakthrough Device: Priority review for serious conditions

As of Jan 2025, the FDA issued comprehensive draft guidance for
AI-enabled device software functions, requiring:
  - Model description and architecture documentation
  - Training data lineage and demographic representation
  - Performance metrics tied to intended use claims
  - Bias analysis and mitigation strategies
  - Human-AI workflow description
  - Post-market monitoring plans

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - Python 3.10+

REFERENCES:
    - FDA Draft Guidance: AI-Enabled Device Software Functions (Jan 2025)
    - FDA PCCP Guidance for AI Devices (Aug 2025, finalized)
    - FDA Q-Submission Program Guidance (2025)
    - SPIRIT-AI / CONSORT-AI Reporting Extensions
    - 21 CFR Part 11: Electronic Records

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
# SECTION 1: SUBMISSION TYPES AND STRUCTURES
# =============================================================================

class SubmissionType(Enum):
    """FDA regulatory submission pathways."""
    FIVE_TEN_K = "510k"
    DE_NOVO = "de_novo"
    PMA = "pma"
    BREAKTHROUGH = "breakthrough_device"
    PRE_SUB = "pre_submission"
    PCCP = "pccp"


class SubmissionStatus(Enum):
    """Submission lifecycle status."""
    PLANNING = "planning"
    PRE_SUBMISSION = "pre_submission_filed"
    PRE_SUB_MEETING = "pre_sub_meeting_complete"
    PREPARING = "preparing_submission"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    ADDITIONAL_INFO = "additional_info_requested"
    ACCEPTED = "accepted"
    CLEARED = "cleared"
    AUTHORIZED = "authorized"
    WITHDRAWN = "withdrawn"


class DeviceClass(Enum):
    """FDA device classification."""
    CLASS_I = "I"
    CLASS_II = "II"
    CLASS_III = "III"


@dataclass
class AIMLComponent:
    """AI/ML component within a device submission."""
    name: str
    model_type: str  # "classification", "segmentation", "prediction", "planning"
    architecture: str = ""
    training_data_size: int = 0
    training_data_sources: list[str] = field(default_factory=list)
    validation_approach: str = ""
    performance_metrics: dict[str, float] = field(default_factory=dict)
    locked_or_adaptive: str = "locked"  # "locked" or "adaptive"
    pccp_planned: bool = False


@dataclass
class ChecklistItem:
    """Individual checklist item for submission preparation."""
    category: str
    description: str
    status: str = "pending"  # "pending", "in_progress", "complete", "not_applicable"
    notes: str = ""
    regulatory_reference: str = ""
    required: bool = True


@dataclass
class SubmissionChecklist:
    """Complete checklist for a submission."""
    submission_id: str
    submission_type: str
    items: list[ChecklistItem] = field(default_factory=list)
    completion_percentage: float = 0.0

    def update_completion(self):
        """Recalculate completion percentage."""
        required_items = [i for i in self.items if i.required]
        if not required_items:
            self.completion_percentage = 100.0
            return
        complete = sum(1 for i in required_items if i.status == "complete")
        self.completion_percentage = (complete / len(required_items)) * 100


@dataclass
class Submission:
    """FDA regulatory submission record."""
    submission_id: str
    submission_type: SubmissionType
    device_name: str
    intended_use: str
    device_class: DeviceClass
    sponsor: str
    status: SubmissionStatus = SubmissionStatus.PLANNING
    ai_ml_components: list[AIMLComponent] = field(default_factory=list)
    breakthrough_designation: bool = False
    predicate_device: str = ""
    product_code: str = ""
    submission_date: str = ""
    decision_date: str = ""
    review_division: str = "CDRH"
    oncology_indication: str = ""
    created_date: str = ""
    milestones: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "submission_id": self.submission_id,
            "submission_type": self.submission_type.value,
            "device_name": self.device_name,
            "intended_use": self.intended_use,
            "device_class": self.device_class.value,
            "sponsor": self.sponsor,
            "status": self.status.value,
            "breakthrough_designation": self.breakthrough_designation,
            "ai_ml_components": len(self.ai_ml_components),
            "created_date": self.created_date
        }


# =============================================================================
# SECTION 2: CHECKLIST TEMPLATES
# =============================================================================

# AI/ML device submission checklist per Jan 2025 FDA draft guidance
AIML_SUBMISSION_CHECKLIST: dict[str, list[dict[str, Any]]] = {
    "device_description": [
        {
            "description": "Complete device description including hardware and software components",
            "reference": "21 CFR 807.87(e)",
            "required": True
        },
        {
            "description": "Software level of concern determination (per IEC 62304)",
            "reference": "IEC 62304 / FDA Guidance",
            "required": True
        },
        {
            "description": "Intended use statement with specific clinical indications",
            "reference": "21 CFR 807.87(e)",
            "required": True
        }
    ],
    "ai_ml_model": [
        {
            "description": "AI/ML model architecture description and rationale",
            "reference": "FDA AI/ML Guidance (Jan 2025)",
            "required": True
        },
        {
            "description": "Training data description: sources, size, demographic representation",
            "reference": "FDA AI/ML Guidance (Jan 2025)",
            "required": True
        },
        {
            "description": "Data preprocessing and feature engineering documentation",
            "reference": "FDA AI/ML Guidance (Jan 2025)",
            "required": True
        },
        {
            "description": "Model training methodology and hyperparameter selection",
            "reference": "FDA AI/ML Guidance (Jan 2025)",
            "required": True
        },
        {
            "description": "Performance metrics with confidence intervals",
            "reference": "FDA AI/ML Guidance (Jan 2025)",
            "required": True
        },
        {
            "description": "Bias analysis across demographic subgroups (age, sex, race/ethnicity)",
            "reference": "FDA AI/ML Guidance (Jan 2025)",
            "required": True
        },
        {
            "description": "Algorithm change protocol (locked vs. adaptive)",
            "reference": "FDA PCCP Guidance (Aug 2025)",
            "required": True
        }
    ],
    "clinical_evidence": [
        {
            "description": "Clinical validation study design and results",
            "reference": "FDA AI/ML Guidance (Jan 2025)",
            "required": True
        },
        {
            "description": "Clinical performance in intended use population",
            "reference": "21 CFR 814.20",
            "required": True
        },
        {
            "description": "Comparison to standard of care / predicate device",
            "reference": "21 CFR 807.87(f)",
            "required": True
        }
    ],
    "human_factors": [
        {
            "description": "Human-AI workflow description and user interface design",
            "reference": "FDA AI/ML Guidance (Jan 2025)",
            "required": True
        },
        {
            "description": "Labeling for healthcare providers and/or patients",
            "reference": "21 CFR 801",
            "required": True
        },
        {
            "description": "Training materials for clinical users",
            "reference": "IEC 62366",
            "required": True
        }
    ],
    "cybersecurity": [
        {
            "description": "Cybersecurity risk assessment and threat model",
            "reference": "FDA Cybersecurity Guidance (2023)",
            "required": True
        },
        {
            "description": "Software Bill of Materials (SBOM)",
            "reference": "FDA Cybersecurity Guidance (2023)",
            "required": True
        }
    ],
    "quality_system": [
        {
            "description": "Design history file (DHF) per 21 CFR 820 / QMSR",
            "reference": "21 CFR 820 / ISO 13485",
            "required": True
        },
        {
            "description": "Risk management file per ISO 14971",
            "reference": "ISO 14971:2019",
            "required": True
        },
        {
            "description": "Software lifecycle documentation per IEC 62304",
            "reference": "IEC 62304:2015",
            "required": True
        }
    ],
    "post_market": [
        {
            "description": "Post-market surveillance plan",
            "reference": "21 CFR 822 / FDA AI/ML Guidance",
            "required": True
        },
        {
            "description": "Performance monitoring plan (drift detection)",
            "reference": "FDA AI/ML Guidance (Jan 2025)",
            "required": True
        },
        {
            "description": "Adverse event reporting procedures (MDR/MedWatch)",
            "reference": "21 CFR 803",
            "required": True
        }
    ]
}

# Breakthrough Device Designation additional requirements
BREAKTHROUGH_CHECKLIST: list[dict[str, Any]] = [
    {
        "description": "Breakthrough Device Designation request letter",
        "reference": "FDA Breakthrough Device Guidance",
        "required": True
    },
    {
        "description": "Evidence device provides more effective treatment/diagnosis for life-threatening or irreversibly debilitating condition",
        "reference": "21 USC 360e-3",
        "required": True
    },
    {
        "description": "Preliminary clinical evidence or significant bench/animal data",
        "reference": "FDA Breakthrough Device Guidance",
        "required": True
    },
    {
        "description": "Data development plan for interactive review with FDA",
        "reference": "FDA Breakthrough Device Guidance",
        "required": True
    }
]


# =============================================================================
# SECTION 3: FDA SUBMISSION TRACKER
# =============================================================================

class FDASubmissionTracker:
    """
    Manages FDA regulatory submissions for AI/ML-enabled oncology devices.

    Tracks submissions through their lifecycle from planning through
    authorization, with checklist management and milestone tracking.

    SUPPORTED WORKFLOWS:
    -------------------
    1. Pre-Submission (Q-Sub) meeting preparation
    2. 510(k) / De Novo / PMA submission assembly
    3. Breakthrough Device Designation requests
    4. PCCP preparation for adaptive AI algorithms
    5. Post-market surveillance setup
    """

    def __init__(
        self,
        sponsor: str,
        device_class: str = "II",
        review_division: str = "CDRH"
    ):
        """
        Initialize FDA submission tracker.

        Args:
            sponsor: Sponsoring organization name
            device_class: FDA device classification (I, II, III)
            review_division: FDA review division (CDRH, CBER, CDER)
        """
        self.sponsor = sponsor
        self.device_class = DeviceClass(device_class)
        self.review_division = review_division

        self._submissions: dict[str, Submission] = {}
        self._submission_counter = 0

        logger.info(
            f"FDASubmissionTracker initialized: sponsor={sponsor}, "
            f"class={device_class}, division={review_division}"
        )

    def create_submission(
        self,
        submission_type: str,
        device_name: str,
        intended_use: str,
        ai_ml_components: list[str] | None = None,
        breakthrough_designation: bool = False,
        predicate_device: str = "",
        oncology_indication: str = ""
    ) -> Submission:
        """
        Create a new FDA submission record.

        Args:
            submission_type: "510k", "de_novo", "pma", "breakthrough_device"
            device_name: Name of the device
            intended_use: Intended use statement
            ai_ml_components: Names of AI/ML components in the device
            breakthrough_designation: Whether to apply for Breakthrough designation
            predicate_device: Predicate device (for 510(k))
            oncology_indication: Cancer type / clinical indication

        Returns:
            Submission record
        """
        self._submission_counter += 1
        submission_id = f"SUB-{datetime.now().strftime('%Y%m%d')}-{self._submission_counter:04d}"

        # Create AI/ML component records
        components = [
            AIMLComponent(name=name, model_type="classification")
            for name in (ai_ml_components or [])
        ]

        submission = Submission(
            submission_id=submission_id,
            submission_type=SubmissionType(submission_type),
            device_name=device_name,
            intended_use=intended_use,
            device_class=self.device_class,
            sponsor=self.sponsor,
            ai_ml_components=components,
            breakthrough_designation=breakthrough_designation,
            predicate_device=predicate_device,
            review_division=self.review_division,
            oncology_indication=oncology_indication,
            created_date=datetime.now().isoformat()
        )

        # Add initial milestone
        submission.milestones.append({
            "date": datetime.now().isoformat(),
            "event": "submission_created",
            "description": f"{submission_type} submission initiated for {device_name}"
        })

        self._submissions[submission_id] = submission

        logger.info(
            f"Submission created: {submission_id} ({submission_type}) "
            f"for {device_name}"
        )

        return submission

    def generate_presub_checklist(self, submission: Submission) -> SubmissionChecklist:
        """
        Generate a pre-submission checklist based on submission type.

        Creates a comprehensive checklist of required documentation
        per FDA guidance for AI/ML-enabled devices.

        Args:
            submission: The submission to generate a checklist for

        Returns:
            SubmissionChecklist with all required items
        """
        checklist = SubmissionChecklist(
            submission_id=submission.submission_id,
            submission_type=submission.submission_type.value
        )

        # Add standard AI/ML submission items
        for category, items in AIML_SUBMISSION_CHECKLIST.items():
            for item_def in items:
                checklist.items.append(ChecklistItem(
                    category=category,
                    description=item_def["description"],
                    regulatory_reference=item_def["reference"],
                    required=item_def["required"]
                ))

        # Add Breakthrough-specific items
        if submission.breakthrough_designation:
            for item_def in BREAKTHROUGH_CHECKLIST:
                checklist.items.append(ChecklistItem(
                    category="breakthrough_designation",
                    description=item_def["description"],
                    regulatory_reference=item_def["reference"],
                    required=item_def["required"]
                ))

        # Add PCCP items for adaptive algorithms
        has_adaptive = any(
            c.locked_or_adaptive == "adaptive"
            for c in submission.ai_ml_components
        )
        if has_adaptive:
            checklist.items.append(ChecklistItem(
                category="pccp",
                description="Predetermined Change Control Plan per Aug 2025 guidance",
                regulatory_reference="FDA PCCP Guidance (Aug 2025)",
                required=True
            ))
            checklist.items.append(ChecklistItem(
                category="pccp",
                description="Modification protocol describing permitted change types",
                regulatory_reference="FDA PCCP Guidance (Aug 2025)",
                required=True
            ))
            checklist.items.append(ChecklistItem(
                category="pccp",
                description="Impact assessment methodology for each change type",
                regulatory_reference="FDA PCCP Guidance (Aug 2025)",
                required=True
            ))

        # Add De Novo specific items
        if submission.submission_type == SubmissionType.DE_NOVO:
            checklist.items.append(ChecklistItem(
                category="de_novo_specific",
                description="Proposed special controls for the device type",
                regulatory_reference="21 USC 360c(f)(2)",
                required=True
            ))
            checklist.items.append(ChecklistItem(
                category="de_novo_specific",
                description="Proposed device classification (Class I or II)",
                regulatory_reference="21 USC 360c(f)(2)",
                required=True
            ))

        checklist.update_completion()

        logger.info(
            f"Checklist generated for {submission.submission_id}: "
            f"{len(checklist.items)} items, "
            f"{checklist.completion_percentage:.0f}% complete"
        )

        return checklist

    def update_status(
        self,
        submission_id: str,
        new_status: str,
        notes: str = ""
    ):
        """Update submission status with milestone tracking."""
        submission = self._submissions.get(submission_id)
        if not submission:
            raise ValueError(f"Submission {submission_id} not found")

        old_status = submission.status.value
        submission.status = SubmissionStatus(new_status)

        submission.milestones.append({
            "date": datetime.now().isoformat(),
            "event": f"status_change",
            "description": f"Status changed from {old_status} to {new_status}. {notes}"
        })

        logger.info(f"Submission {submission_id}: {old_status} -> {new_status}")

    def get_submission_summary(self, submission_id: str) -> str:
        """Generate a human-readable submission summary."""
        submission = self._submissions.get(submission_id)
        if not submission:
            return f"Submission {submission_id} not found"

        summary = f"""
FDA SUBMISSION SUMMARY
======================
Submission ID: {submission.submission_id}
Type: {submission.submission_type.value}
Device: {submission.device_name}
Sponsor: {submission.sponsor}
Class: {submission.device_class.value}
Status: {submission.status.value}
Intended Use: {submission.intended_use}
Oncology Indication: {submission.oncology_indication or 'Not specified'}
Breakthrough Designation: {submission.breakthrough_designation}
AI/ML Components: {len(submission.ai_ml_components)}
Review Division: {submission.review_division}
Created: {submission.created_date}

MILESTONES
----------
"""
        for milestone in submission.milestones:
            summary += f"  {milestone['date'][:10]}: {milestone['description']}\n"

        return summary


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================

def run_fda_tracker_demo():
    """
    Demonstrate FDA submission tracking for AI oncology devices.

    Shows submission creation, checklist generation, and
    status tracking for a De Novo AI device submission.
    """
    logger.info("=" * 60)
    logger.info("FDA SUBMISSION TRACKER DEMO")
    logger.info("=" * 60)

    tracker = FDASubmissionTracker(
        sponsor="Physical AI Oncology Consortium",
        device_class="II"
    )

    # Create a De Novo submission for an AI surgical planning system
    submission = tracker.create_submission(
        submission_type="de_novo",
        device_name="AI-Guided Surgical Planning System for Oncology",
        intended_use=(
            "AI-assisted surgical planning for solid tumor resection "
            "using patient-specific digital twin models derived from "
            "pre-operative CT/MRI imaging"
        ),
        ai_ml_components=[
            "tumor_segmentation_model",
            "surgical_path_optimizer",
            "outcome_prediction_model"
        ],
        breakthrough_designation=True,
        oncology_indication="Non-small cell lung cancer (NSCLC)"
    )

    # Generate submission checklist
    checklist = tracker.generate_presub_checklist(submission)

    print(f"\nSubmission: {submission.submission_id}")
    print(f"Type: {submission.submission_type.value}")
    print(f"Device: {submission.device_name}")
    print(f"Breakthrough: {submission.breakthrough_designation}")
    print(f"\nChecklist ({len(checklist.items)} items):")

    # Group by category
    categories: dict[str, list[ChecklistItem]] = {}
    for item in checklist.items:
        categories.setdefault(item.category, []).append(item)

    for category, items in categories.items():
        print(f"\n  {category.upper().replace('_', ' ')} ({len(items)} items)")
        for item in items[:3]:  # Show first 3 per category
            print(f"    [{item.status:10s}] {item.description[:70]}")
        if len(items) > 3:
            print(f"    ... and {len(items) - 3} more items")

    # Update status
    tracker.update_status(
        submission.submission_id,
        "pre_submission_filed",
        "Q-Sub request submitted to CDRH"
    )

    print(tracker.get_submission_summary(submission.submission_id))

    return {"submission_id": submission.submission_id, "status": "demo_complete"}


if __name__ == "__main__":
    result = run_fda_tracker_demo()
    print(f"\nDemo completed: {result}")
