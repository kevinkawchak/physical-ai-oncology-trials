#!/usr/bin/env python3
"""Example 06: Full 8-Site Multi-Cancer Consortium Coordination.

Demonstrates a complete multi-site federated oncology trial with
8 participating sites across multiple cancer types, incorporating
federated learning, differential privacy, enrollment synchronization,
data harmonization, consortium reporting, and privacy-preserving
survival analysis.

Key Concepts:
    - 8-site federation with FedProx aggregation
    - Differential privacy on gradient updates
    - Stratified enrollment with adaptive balancing
    - Cross-site DICOM/FHIR harmonization
    - DSMB package generation
    - Federated Kaplan-Meier survival analysis
    - Comprehensive audit trail

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

from federation.consortium_reporting import (
    AdverseEvent,
    AdverseEventGrade,
    AERelationship,
    DSMBPackageGenerator,
    SitePerformanceMetrics,
)
from federation.data_harmonization import DataHarmonizationEngine, DICOMMetadata
from federation.differential_privacy import (
    DifferentialPrivacyEngine,
    NoiseMechanism,
    PrivacyBudget,
    PrivacyConfig,
)
from federation.federated_coordinator import (
    AggregationStrategy,
    FederatedCoordinator,
    FederationConfig,
    SiteConfig,
)
from federation.privacy_analytics import (
    FederatedSurvivalAnalyzer,
    PrivacyAnalyticsConfig,
    SurvivalRecord,
)
from federation.site_enrollment import (
    EnrollmentSynchronizer,
    StratificationConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consortium Configuration
# ---------------------------------------------------------------------------

CONSORTIUM_SITES = [
    {
        "id": "mayo_rochester",
        "name": "Mayo Clinic Rochester",
        "institution": "Mayo Clinic",
        "patients": 200,
        "target": 40,
    },
    {"id": "mskcc_nyc", "name": "Memorial Sloan Kettering", "institution": "MSKCC", "patients": 250, "target": 50},
    {
        "id": "mdanderson_houston",
        "name": "MD Anderson Cancer Center",
        "institution": "UT MD Anderson",
        "patients": 220,
        "target": 45,
    },
    {
        "id": "dana_farber_boston",
        "name": "Dana-Farber Cancer Institute",
        "institution": "DFCI",
        "patients": 180,
        "target": 35,
    },
    {
        "id": "stanford_palo_alto",
        "name": "Stanford Cancer Institute",
        "institution": "Stanford Medicine",
        "patients": 160,
        "target": 30,
    },
    {
        "id": "johns_hopkins_baltimore",
        "name": "Johns Hopkins Sidney Kimmel",
        "institution": "JHU",
        "patients": 190,
        "target": 40,
    },
    {
        "id": "ucsf_san_francisco",
        "name": "UCSF Helen Diller",
        "institution": "UCSF Health",
        "patients": 170,
        "target": 35,
    },
    {
        "id": "cleveland_clinic_oh",
        "name": "Cleveland Clinic Taussig",
        "institution": "Cleveland Clinic",
        "patients": 150,
        "target": 30,
    },
]

CANCER_TYPES = ["breast", "lung", "colorectal", "prostate"]
STAGES = ["II", "III", "IV"]
BIOMARKERS = ["positive", "negative"]

MODEL_LAYERS = {
    "encoder_conv1": (32, 3, 3, 3),
    "encoder_conv2": (64, 32, 3, 3),
    "fc1": (256, 128),
    "fc2": (128, 64),
    "classifier": (64, len(CANCER_TYPES)),
}


# ---------------------------------------------------------------------------
# Phase 1: Federated Learning
# ---------------------------------------------------------------------------


def phase_federated_learning() -> dict:
    """Execute federated learning across all 8 sites."""
    print("\n" + "=" * 60)
    print("PHASE 1: Federated Learning (FedProx, 8 Sites)")
    print("=" * 60)

    config = FederationConfig(
        trial_name="MULTI-ONCO-CONSORTIUM-2026",
        strategy=AggregationStrategy.FEDPROX,
        num_rounds=5,
        min_sites_per_round=4,
        site_fraction=0.75,
        proximal_mu=0.01,
        convergence_threshold=1e-5,
    )
    coordinator = FederatedCoordinator(config)

    for site_info in CONSORTIUM_SITES:
        coordinator.register_site(
            SiteConfig(
                site_id=site_info["id"],
                site_name=site_info["name"],
                institution=site_info["institution"],
                patient_count=site_info["patients"],
                data_quality_score=0.85 + np.random.uniform(0, 0.15),
            )
        )

    coordinator.initialize_global_model(MODEL_LAYERS)

    dp_config = PrivacyConfig(
        mechanism=NoiseMechanism.GAUSSIAN,
        epsilon_per_round=0.5,
        delta=1e-5,
        clip_norm=1.0,
    )
    dp_budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
    dp_engine = DifferentialPrivacyEngine(dp_config, dp_budget)

    rounds = coordinator.run_federation()

    summary = coordinator.get_federation_summary()
    print("\n  Federation Summary:")
    print(f"    Sites: {summary['total_sites']}")
    print(f"    Rounds completed: {summary['completed_rounds']}")
    print(f"    Final loss: {summary['final_avg_loss']:.4f}")
    print(f"    Final accuracy: {summary['final_avg_accuracy']:.4f}")

    dp_status = dp_engine.get_budget_status()
    print(f"\n  Privacy Budget: eps_remaining={dp_status['epsilon_remaining']:.2f}")

    return summary


# ---------------------------------------------------------------------------
# Phase 2: Enrollment Synchronization
# ---------------------------------------------------------------------------


def phase_enrollment() -> dict:
    """Synchronize enrollment across all sites."""
    print("\n" + "=" * 60)
    print("PHASE 2: Enrollment Synchronization")
    print("=" * 60)

    strat_config = StratificationConfig(
        factors={
            "cancer_type": CANCER_TYPES,
            "stage": STAGES,
            "biomarker": BIOMARKERS,
        },
        block_size=4,
    )
    sync = EnrollmentSynchronizer(trial_id="MULTI-ONCO-2026", stratification_config=strat_config)

    for site_info in CONSORTIUM_SITES:
        sync.register_site(site_info["id"], site_info["name"], site_info["target"])

    for site_info in CONSORTIUM_SITES:
        n_to_screen = site_info["target"] + np.random.randint(2, 8)
        for _ in range(n_to_screen):
            patient = sync.screen_patient(
                site_info["id"],
                eligibility={
                    "age_18_75": np.random.random() > 0.05,
                    "adequate_organ_function": np.random.random() > 0.08,
                    "ecog_0_1": np.random.random() > 0.1,
                    "informed_consent": True,
                },
                stratification={
                    "cancer_type": np.random.choice(CANCER_TYPES),
                    "stage": np.random.choice(STAGES),
                    "biomarker": np.random.choice(BIOMARKERS),
                },
            )
            sync.enroll_patient(site_info["id"], patient.patient_id)

    summary = sync.get_enrollment_summary()
    print("\n  Enrollment Summary:")
    print(f"    Total enrolled: {summary['total_enrolled']}/{summary['total_target']} ({summary['pct_overall']}%)")
    print(f"    Screen fail rate: {summary['screen_fail_rate']}%")
    print(f"    Arm distribution: {summary['arm_totals']}")

    for site_id, site_sum in summary["site_summaries"].items():
        print(f"    {site_id}: {site_sum['enrolled']}/{site_sum['target']} ({site_sum['pct_target']}%)")

    imbalances = sync.check_balance()
    print(f"\n  Arm imbalances detected: {len(imbalances)}")

    return summary


# ---------------------------------------------------------------------------
# Phase 3: Data Harmonization
# ---------------------------------------------------------------------------


def phase_harmonization() -> dict:
    """Harmonize data across all sites."""
    print("\n" + "=" * 60)
    print("PHASE 3: Cross-Site Data Harmonization")
    print("=" * 60)

    engine = DataHarmonizationEngine(trial_id="MULTI-ONCO-2026")

    modalities = ["CT", "MR", "PT", "MR", "CT"]
    body_parts = ["CHEST", "brain", "wholebody", "ABD", "PELVIS"]
    icd10_codes = ["C50.9", "C34.9", "C18.9", "C61"]

    total_dicom = 0
    total_dx = 0
    total_labs = 0

    for site_info in CONSORTIUM_SITES:
        site_id = site_info["id"]

        n_studies = np.random.randint(3, 8)
        dicom_batch = []
        for _ in range(n_studies):
            dicom_batch.append(
                DICOMMetadata(
                    modality=np.random.choice(modalities),
                    body_part=np.random.choice(body_parts),
                    slice_thickness=np.random.choice([1.0, 1.25, 2.5, 3.0, 5.0]),
                    pixel_spacing=(np.random.choice([0.5, 0.7, 1.0, 1.2]), np.random.choice([0.5, 0.7, 1.0, 1.2])),
                )
            )
        result = engine.harmonize_dicom_series(site_id, dicom_batch)
        total_dicom += result.records_harmonized

        n_dx = np.random.randint(5, 15)
        diagnoses = [
            {"code": np.random.choice(icd10_codes), "patient_ref": f"Patient/{site_id}/{j}"} for j in range(n_dx)
        ]
        _, dx_result = engine.harmonize_diagnoses(site_id, diagnoses)
        total_dx += dx_result.records_harmonized

        n_labs = np.random.randint(3, 10)
        labs = [
            {
                "marker_name": np.random.choice(["CEA", "PSA", "CA125", "AFP"]),
                "value": np.random.uniform(1, 100),
                "patient_ref": f"Patient/{site_id}/{j}",
            }
            for j in range(n_labs)
        ]
        _, lab_result = engine.harmonize_lab_results(site_id, labs)
        total_labs += lab_result.records_harmonized

    print("\n  Harmonization Totals:")
    print(f"    DICOM series normalized: {total_dicom}")
    print(f"    Diagnoses harmonized: {total_dx}")
    print(f"    Lab results harmonized: {total_labs}")
    print(f"    Total events: {len(engine.harmonization_log)}")

    return {"dicom": total_dicom, "diagnoses": total_dx, "labs": total_labs}


# ---------------------------------------------------------------------------
# Phase 4: Consortium Reporting
# ---------------------------------------------------------------------------


def phase_reporting(enrollment_summary: dict) -> dict:
    """Generate DSMB package and consortium reports."""
    print("\n" + "=" * 60)
    print("PHASE 4: Consortium Reporting & DSMB Package")
    print("=" * 60)

    site_metrics = {}
    for site_info in CONSORTIUM_SITES:
        site_id = site_info["id"]
        site_sum = enrollment_summary.get("site_summaries", {}).get(site_id, {})
        site_metrics[site_id] = SitePerformanceMetrics(
            site_id=site_id,
            site_name=site_info["name"],
            enrollment_count=site_sum.get("enrolled", 0),
            enrollment_target=site_info["target"],
            enrollment_rate=np.random.uniform(1.5, 4.0),
            screen_fail_rate=float(enrollment_summary.get("screen_fail_rate", 0)),
            data_completeness=np.random.uniform(0.85, 0.98),
            query_rate=np.random.uniform(2, 15),
            protocol_deviation_count=np.random.randint(0, 4),
            ae_reporting_delay_days=np.random.uniform(0.5, 5.0),
        )

    adverse_events = []
    terms = ["Nausea", "Fatigue", "Neutropenia", "Diarrhea", "Rash", "Anemia", "Thrombocytopenia", "Hepatotoxicity"]
    socs = ["Gastrointestinal", "General", "Blood/Lymphatic", "Skin", "Hepatobiliary"]

    for i in range(50):
        ae = AdverseEvent(
            ae_id=f"AE-{i:04d}",
            patient_id=f"PT-{i:04d}",
            site_id=CONSORTIUM_SITES[i % 8]["id"],
            term=np.random.choice(terms),
            system_organ_class=np.random.choice(socs),
            grade=AdverseEventGrade(np.random.choice([1, 1, 1, 2, 2, 3, 3, 4])),
            relationship=AERelationship(np.random.choice(["unrelated", "unrelated", "possible", "probable"])),
            serious=(i % 10 == 0),
            onset_day=np.random.randint(1, 180),
        )
        adverse_events.append(ae)

    dsmb_gen = DSMBPackageGenerator()
    package = dsmb_gen.generate(
        trial_id="MULTI-ONCO-2026",
        review_number=1,
        site_metrics=site_metrics,
        adverse_events=adverse_events,
        arm_distribution=enrollment_summary.get("arm_totals", {}),
    )

    print(f"\n  DSMB Package (Review #{package.review_number}):")
    enrollment = package.enrollment_summary
    if enrollment is not None:
        print(f"    Enrollment: {enrollment.total_enrolled}/{enrollment.total_target}")
        print(f"    Projection: {enrollment.projected_completion}")

    print(f"    Total AEs: {package.ae_summary.get('total_aes', 0)}")
    print(f"    SAEs: {package.ae_summary.get('total_saes', 0)}")
    print(f"    Treatment-related: {package.ae_summary.get('treatment_related', 0)}")

    print(f"\n  Safety Signals ({len(package.safety_signals)}):")
    for signal in package.safety_signals:
        print(f"    [{signal['severity']}] {signal['description']}")

    print("\n  Recommendations:")
    for rec in package.recommendations:
        print(f"    - {rec}")

    return {"package_id": package.package_id, "aes": len(adverse_events)}


# ---------------------------------------------------------------------------
# Phase 5: Privacy-Preserving Analytics
# ---------------------------------------------------------------------------


def phase_survival_analysis() -> dict:
    """Run federated survival analysis across all sites."""
    print("\n" + "=" * 60)
    print("PHASE 5: Privacy-Preserving Survival Analysis")
    print("=" * 60)

    analytics_config = PrivacyAnalyticsConfig(
        epsilon=1.0,
        delta=1e-5,
        min_cell_size=3,
    )
    analyzer = FederatedSurvivalAnalyzer(analytics_config)

    total_patients = 0
    for site_info in CONSORTIUM_SITES:
        site_id = site_info["id"]
        agg = analyzer.register_site(site_id)

        n_patients = np.random.randint(20, 40)
        records = []
        for i in range(n_patients):
            arm = "experimental" if np.random.random() > 0.5 else "control"
            cancer = np.random.choice(CANCER_TYPES)
            stage_val = float(np.random.choice([2, 3, 4]))

            base_survival = 24 if arm == "experimental" else 18
            if cancer == "lung":
                base_survival *= 0.7
            elif cancer == "breast":
                base_survival *= 1.3

            records.append(
                SurvivalRecord(
                    patient_id=f"PT-{site_id[:4]}-{i:03d}",
                    site_id=site_id,
                    time=max(0.5, np.random.exponential(base_survival)),
                    event=np.random.random() > 0.35,
                    treatment_arm=arm,
                    covariates={
                        "age": 50 + np.random.randn() * 12,
                        "stage": stage_val,
                        "ecog": float(np.random.choice([0, 1])),
                    },
                )
            )
        agg.add_records(records)
        total_patients += n_patients

    print(f"\n  Total patients across {len(CONSORTIUM_SITES)} sites: {total_patients}")

    km_all = analyzer.federated_kaplan_meier("all_patients")
    km_exp = analyzer.federated_kaplan_meier("experimental_arm", arm_filter="experimental")
    km_ctrl = analyzer.federated_kaplan_meier("control_arm", arm_filter="control")

    print("\n  Kaplan-Meier Results:")
    print(
        f"    All patients: {len(km_all.time_points)} time points, median={km_all.median_survival:.1f}mo"
        if km_all.median_survival
        else f"    All patients: {len(km_all.time_points)} time points, median not reached"
    )
    print(f"    Experimental: median={'%.1f' % km_exp.median_survival if km_exp.median_survival else 'NR'}mo")
    print(f"    Control: median={'%.1f' % km_ctrl.median_survival if km_ctrl.median_survival else 'NR'}mo")

    cox_result = analyzer.federated_cox_ph(["age", "stage", "ecog"])
    print(f"\n  Cox PH Results (C-index={cox_result.concordance_index:.3f}):")
    for name in ["age", "stage", "ecog"]:
        hr = cox_result.hazard_ratios.get(name, 0)
        p = cox_result.p_values.get(name, 1)
        print(f"    {name}: HR={hr:.3f}, p={p:.4f}")

    rr_exp = analyzer.compute_response_rate("experimental")
    rr_ctrl = analyzer.compute_response_rate("control")
    print("\n  Response Rates:")
    print(
        f"    Experimental: {rr_exp.get('response_rate', 'N/A')} "
        f"(95% CI: {rr_exp.get('ci_lower', 'N/A')}-{rr_exp.get('ci_upper', 'N/A')})"
    )
    print(
        f"    Control: {rr_ctrl.get('response_rate', 'N/A')} "
        f"(95% CI: {rr_ctrl.get('ci_lower', 'N/A')}-{rr_ctrl.get('ci_upper', 'N/A')})"
    )

    return {
        "total_patients": total_patients,
        "km_time_points": len(km_all.time_points),
        "c_index": cox_result.concordance_index,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full 8-site multi-cancer consortium coordination."""
    print("=" * 70)
    print("Example 06: Full 8-Site Multi-Cancer Consortium Coordination")
    print("=" * 70)
    print(f"\nConsortium: {len(CONSORTIUM_SITES)} sites, {len(CANCER_TYPES)} cancer types")
    print(f"Total target enrollment: {sum(s['target'] for s in CONSORTIUM_SITES)}")

    fed_summary = phase_federated_learning()
    enrollment_summary = phase_enrollment()
    harm_summary = phase_harmonization()
    report_summary = phase_reporting(enrollment_summary)
    survival_summary = phase_survival_analysis()

    print("\n" + "=" * 70)
    print("CONSORTIUM SUMMARY")
    print("=" * 70)
    print(
        f"  Federated Learning: {fed_summary.get('completed_rounds', 0)} rounds, "
        f"accuracy={fed_summary.get('final_avg_accuracy', 0):.4f}"
    )
    print(f"  Enrollment: {enrollment_summary.get('total_enrolled', 0)}/{enrollment_summary.get('total_target', 0)}")
    print(
        f"  Harmonization: {harm_summary['dicom']} DICOM + {harm_summary['diagnoses']} Dx + {harm_summary['labs']} Labs"
    )
    print(f"  Safety: {report_summary['aes']} AEs tracked")
    print(f"  Survival: {survival_summary['total_patients']} patients, C-index={survival_summary['c_index']:.3f}")
    print(f"\n{'=' * 70}")
    print("Full consortium coordination completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
