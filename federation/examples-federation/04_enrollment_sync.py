#!/usr/bin/env python3
"""Example 04: Multi-Site Enrollment Coordination.

Demonstrates cross-site enrollment synchronization with stratified
randomization, conflict detection, and balance monitoring across
multiple clinical trial sites.

Key Concepts:
    - Multi-site enrollment registration
    - Patient screening and eligibility checking
    - Stratified block randomization
    - Duplicate enrollment detection and resolution
    - Arm balance monitoring
    - Cross-site enrollment summary

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.

Version: 1.0.0
Last Updated: February 2026
License: MIT
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from federation.site_enrollment import (
    ConflictResolution,
    EnrollmentSynchronizer,
    StratificationConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trial Configuration
# ---------------------------------------------------------------------------

TRIAL_SITES = [
    {"site_id": "mayo_clinic", "name": "Mayo Clinic Rochester", "target": 40},
    {"site_id": "mskcc", "name": "Memorial Sloan Kettering", "target": 50},
    {"site_id": "mdanderson", "name": "MD Anderson Cancer Center", "target": 45},
    {"site_id": "dana_farber", "name": "Dana-Farber Cancer Institute", "target": 35},
]

STRATIFICATION = StratificationConfig(
    factors={
        "cancer_type": ["breast", "lung", "colorectal"],
        "stage": ["II", "III", "IV"],
        "biomarker": ["positive", "negative"],
    },
    block_size=4,
    imbalance_threshold=0.2,
)

ELIGIBILITY_CRITERIA = [
    "age_18_75",
    "adequate_organ_function",
    "ecog_0_1",
    "informed_consent",
    "no_prior_immunotherapy",
    "measurable_disease",
]


# ---------------------------------------------------------------------------
# Simulation Functions
# ---------------------------------------------------------------------------


def simulate_patient_screening(site_id: str, n_patients: int) -> list:
    """Generate simulated patient screening data.

    Args:
        site_id: Site identifier.
        n_patients: Number of patients to generate.

    Returns:
        List of patient data dictionaries.
    """
    patients = []
    for _ in range(n_patients):
        eligibility = {}
        for criterion in ELIGIBILITY_CRITERIA:
            eligibility[criterion] = np.random.random() > 0.1

        cancer_type = np.random.choice(["breast", "lung", "colorectal"])
        stage = np.random.choice(["II", "III", "IV"])
        biomarker = np.random.choice(["positive", "negative"])

        patients.append(
            {
                "eligibility": eligibility,
                "stratification": {
                    "cancer_type": cancer_type,
                    "stage": stage,
                    "biomarker": biomarker,
                },
                "demographics": {
                    "age_group": np.random.choice(["18-40", "41-55", "56-65", "66-75"]),
                    "sex": np.random.choice(["M", "F"]),
                },
            }
        )
    return patients


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------


def demo_basic_enrollment() -> None:
    """Demonstrate basic multi-site enrollment."""
    print("\n--- Basic Multi-Site Enrollment ---")

    sync = EnrollmentSynchronizer(trial_id="ONCO-IMM-2026", stratification_config=STRATIFICATION)

    for site_info in TRIAL_SITES:
        sync.register_site(site_info["site_id"], site_info["name"], site_info["target"])
    print(f"  Registered {len(TRIAL_SITES)} sites")

    for site_info in TRIAL_SITES:
        site_id = site_info["site_id"]
        patients = simulate_patient_screening(site_id, 8)

        enrolled = 0
        for patient_data in patients:
            patient = sync.screen_patient(
                site_id,
                eligibility=patient_data["eligibility"],
                stratification=patient_data["stratification"],
                demographics=patient_data["demographics"],
            )
            result = sync.enroll_patient(site_id, patient.patient_id)
            if result is not None and result.treatment_arm is not None:
                enrolled += 1

        print(f"  {site_id}: {enrolled}/{len(patients)} enrolled")

    summary = sync.get_enrollment_summary()
    print(f"\n  Overall: {summary['total_enrolled']}/{summary['total_target']} ({summary['pct_overall']}%)")
    print(f"  Arm distribution: {summary['arm_totals']}")
    print(f"  Screen fail rate: {summary['screen_fail_rate']}%")


def demo_conflict_detection() -> None:
    """Demonstrate duplicate enrollment detection."""
    print("\n--- Conflict Detection ---")

    sync = EnrollmentSynchronizer(trial_id="CONFLICT-TEST")
    sync.register_site("site_a", "Hospital A", target=20)
    sync.register_site("site_b", "Hospital B", target=20)

    patient_a = sync.screen_patient(
        "site_a",
        patient_id="PT-DUPLICATE",
        eligibility={"consent": True},
        stratification={"cancer_type": "breast", "stage": "III"},
    )
    sync.enroll_patient("site_a", patient_a.patient_id)

    patient_b = sync.screen_patient(
        "site_b",
        patient_id="PT-DUPLICATE",
        eligibility={"consent": True},
        stratification={"cancer_type": "breast", "stage": "III"},
    )
    result = sync.enroll_patient("site_b", patient_b.patient_id)

    if result is not None:
        print(f"  Patient PT-DUPLICATE status at site_b: {result.status.value}")
    print(f"  Conflicts resolved: {len(sync.conflict_resolver.resolved_conflicts)}")
    for conflict in sync.conflict_resolver.resolved_conflicts:
        print(f"    [{conflict.conflict_type.value}] {conflict.description}")
        print(f"    Resolution: {conflict.resolution_details}")


def demo_withdrawal_tracking() -> None:
    """Demonstrate patient withdrawal handling."""
    print("\n--- Withdrawal Tracking ---")

    sync = EnrollmentSynchronizer(trial_id="WITHDRAW-TEST")
    sync.register_site("site_main", "Main Hospital", target=20)

    enrolled_ids = []
    for _ in range(6):
        patient = sync.screen_patient(
            "site_main",
            eligibility={"consent": True},
            stratification={"cancer_type": "lung", "stage": "IV"},
        )
        result = sync.enroll_patient("site_main", patient.patient_id)
        if result is not None and result.treatment_arm is not None:
            enrolled_ids.append(patient.patient_id)

    print(f"  Initially enrolled: {len(enrolled_ids)} patients")

    for pid in enrolled_ids[:2]:
        sync.withdraw_patient("site_main", pid, reason="Patient decision")
        print(f"  Withdrawn: {pid}")

    summary = sync.get_enrollment_summary()
    print(f"  After withdrawals: {summary['total_enrolled']} enrolled, {summary['total_withdrawals']} withdrawn")


def demo_balance_monitoring() -> None:
    """Demonstrate arm balance monitoring."""
    print("\n--- Balance Monitoring ---")

    sync = EnrollmentSynchronizer(trial_id="BALANCE-TEST")
    for site_info in TRIAL_SITES[:2]:
        sync.register_site(site_info["site_id"], site_info["name"], site_info["target"])

    for site_info in TRIAL_SITES[:2]:
        patients = simulate_patient_screening(site_info["site_id"], 12)
        for patient_data in patients:
            patient = sync.screen_patient(
                site_info["site_id"],
                eligibility=patient_data["eligibility"],
                stratification=patient_data["stratification"],
            )
            sync.enroll_patient(site_info["site_id"], patient.patient_id)

    imbalances = sync.check_balance()
    print(f"  Arm imbalances detected: {len(imbalances)}")
    for conflict in imbalances:
        print(f"    [{conflict.conflict_type.value}] {conflict.description}")

    summary = sync.get_enrollment_summary()
    for site_id, site_summary in summary["site_summaries"].items():
        print(f"  {site_id}: arms={site_summary['arm_counts']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all enrollment synchronization demonstrations."""
    print("=" * 70)
    print("Example 04: Multi-Site Enrollment Coordination")
    print("=" * 70)

    demo_basic_enrollment()
    demo_conflict_detection()
    demo_withdrawal_tracking()
    demo_balance_monitoring()

    print("\n" + "=" * 70)
    print("Enrollment synchronization demonstration completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
