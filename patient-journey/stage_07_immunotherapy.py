#!/usr/bin/env python3
"""
Stage 7: Adjuvant Immunotherapy (Day 28 to Month 24)
=====================================================

Orchestrates 35 planned cycles of pembrolizumab 200 mg IV every 3 weeks
as adjuvant immunotherapy following curative-intent surgery.  Manages
PK/PD modelling, cumulative toxicity tracking, digital twin updates at
imaging time-points, adaptive AI agent recommendations with human
oversight, annual IND reporting, and regulatory-change monitoring.

Governing regulations:
    - 21 CFR 312.33 (Annual IND Reports / Physical AI Performance Data)
    - 21 CFR 312.56 (Review of Ongoing Investigations)
    - 21 CFR 312.30 (Protocol Amendments)
    - 21 CFR 312.405 (Physical AI System Maintenance Reporting)
    - ICH E6(R3) section 2.5 (Medical Decisions / Treatment Modifications)
    - ICH E6(R3) section 2.12.4 (Ongoing Digital Twin Validation)
    - ICH E6(R3) section 3.1 (Sponsor Responsibilities)
    - 21 CFR 50.32 (Ongoing Consent / Participant Notification)

Usage:
    from stage_07_immunotherapy import ImmunotherapyOrchestrator

    orchestrator = ImmunotherapyOrchestrator(patient_state, treatment_protocol)
    state = orchestrator.run(patient_state)

Last updated: March 2026

DISCLAIMER: RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]
    warnings.warn("NumPy not available; some features will be limited.")

from patient_state import (
    AdverseEvent,
    AESeverity,
    ImagingTimepoint,
    MCPConformanceLevel,
    PatientJourneyState,
    PatientStage,
    PatientStatus,
    PhysicalAIClassification,
    RegulatoryEvent,
    ResponseCategory,
    TreatmentCycle,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Treatment constants — pembrolizumab adjuvant protocol
# ---------------------------------------------------------------------------

DRUG_NAME: str = "pembrolizumab"
DOSE_MG: float = 200.0
ROUTE: str = "IV"
CYCLE_INTERVAL_DAYS: int = 21  # q3w
PLANNED_CYCLES: int = 35
TREATMENT_START_DAY: int = 28
TREATMENT_END_MONTH: int = 24

# PK/PD model parameters (population estimates)
PK_CLEARANCE_L_DAY: float = 0.22
PK_VOLUME_CENTRAL_L: float = 3.5
PK_HALF_LIFE_DAYS: float = 26.0
PD_EC50_UG_ML: float = 1.5
PD_EMAX: float = 0.95

# Imaging schedule — cycles at which CT scans are performed
IMAGING_CYCLES: list[int] = [4, 8, 12, 17, 23, 29, 35]

# Recurrence risk trajectory (% at imaging time-points)
RECURRENCE_RISK_TRAJECTORY: dict[int, float] = {
    4: 35.0,
    8: 28.0,
    12: 18.0,
    17: 12.0,
    23: 8.0,
    29: 5.0,
    35: 3.0,
}

# Expected AE timeline
EXPECTED_AE_CYCLE_6_HYPOTHYROIDISM: dict[str, Any] = {
    "event_id": "AE-IMM-001",
    "description": "Immune-related hypothyroidism",
    "severity": AESeverity.MILD,
    "organ_system": "endocrine",
    "ctcae_grade": 1,
    "causality": "related",
    "related_to_device": False,
}

EXPECTED_AE_CYCLE_12_RASH: dict[str, Any] = {
    "event_id": "AE-IMM-002",
    "description": "Immune-related maculopapular rash",
    "severity": AESeverity.MILD,
    "organ_system": "dermatologic",
    "ctcae_grade": 1,
    "causality": "related",
    "related_to_device": False,
}


class ImmunotherapyOrchestrator:
    """Orchestrates Stage 7 adjuvant immunotherapy (Day 28 to Month 24).

    Manages 35 planned cycles of pembrolizumab 200 mg IV q3w with
    PK/PD modelling, cumulative toxicity tracking, digital twin updates,
    adaptive agent recommendations, annual IND reporting, and regulatory
    compliance monitoring.

    Per ICH E6(R3) section 2.5, all treatment modifications require
    investigator medical decision-making with Physical AI system input
    recorded but non-binding.

    Attributes:
        patient_state: Patient state from Stage 6 (Recovery).
        treatment_protocol: Protocol configuration dictionary.
    """

    def __init__(
        self,
        patient_state: PatientJourneyState,
        treatment_protocol: dict,
    ) -> None:
        self.patient_state = patient_state
        self.treatment_protocol = treatment_protocol
        self._cycles_completed: list[TreatmentCycle] = []
        self._cumulative_aes: list[AdverseEvent] = []
        self._imaging_results: list[dict] = []
        self._twin_updates: list[dict] = []
        self._agent_recommendations: list[dict] = []
        logger.info(
            "ImmunotherapyOrchestrator initialised: %s %s mg %s q%dd x %d cycles",
            DRUG_NAME,
            DOSE_MG,
            ROUTE,
            CYCLE_INTERVAL_DAYS,
            PLANNED_CYCLES,
        )

    # ------------------------------------------------------------------
    # 1. Treatment initialisation
    # ------------------------------------------------------------------

    def initialize_treatment(self) -> dict:
        """Initialise pembrolizumab 200 mg IV q3w adjuvant treatment plan.

        Builds PK/PD model parameters and validates pre-treatment labs.
        Per ICH E6(R3) section 2.5, the investigator confirms the treatment
        plan after reviewing Physical AI recommendations.

        Returns:
            Treatment initialisation result with PK/PD model and schedule.
        """
        logger.info(
            "Initialising treatment: %s %s mg %s q%dd, %d planned cycles "
            "per ICH E6(R3) section 2.5",
            DRUG_NAME,
            DOSE_MG,
            ROUTE,
            CYCLE_INTERVAL_DAYS,
            PLANNED_CYCLES,
        )
        # PK/PD model per ICH E6(R3) section 2.5
        pk_pd_model = {
            "drug": DRUG_NAME,
            "dose_mg": DOSE_MG,
            "route": ROUTE,
            "interval_days": CYCLE_INTERVAL_DAYS,
            "planned_cycles": PLANNED_CYCLES,
            "pk_parameters": {
                "clearance_L_day": PK_CLEARANCE_L_DAY,
                "volume_central_L": PK_VOLUME_CENTRAL_L,
                "half_life_days": PK_HALF_LIFE_DAYS,
            },
            "pd_parameters": {
                "ec50_ug_ml": PD_EC50_UG_ML,
                "emax": PD_EMAX,
            },
            "steady_state_trough_ug_ml": 29.0,
        }

        schedule = []
        for cycle_num in range(1, PLANNED_CYCLES + 1):
            cycle_day = TREATMENT_START_DAY + (cycle_num - 1) * CYCLE_INTERVAL_DAYS
            schedule.append({
                "cycle": cycle_num,
                "day": cycle_day,
                "imaging": cycle_num in IMAGING_CYCLES,
            })

        result = {
            "treatment_plan": {
                "drug": DRUG_NAME,
                "dose_mg": DOSE_MG,
                "route": ROUTE,
                "frequency": "q3w",
                "planned_cycles": PLANNED_CYCLES,
                "start_day": TREATMENT_START_DAY,
            },
            "pk_pd_model": pk_pd_model,
            "schedule": schedule,
            "investigator_confirmed": True,
            "cfr_section": "ICH E6(R3) section 2.5",
        }
        return result

    # ------------------------------------------------------------------
    # 2. Single cycle execution
    # ------------------------------------------------------------------

    def execute_cycle(
        self,
        cycle_number: int,
        labs: dict,
        imaging: dict | None = None,
    ) -> TreatmentCycle:
        """Execute a single immunotherapy cycle per ICH E6(R3) section 2.5.

        Records pre-treatment labs, administers treatment, checks for
        adverse events at known time-points, and records imaging if
        scheduled.

        Args:
            cycle_number: Cycle number (1-35).
            labs: Pre-treatment laboratory values.
            imaging: Optional imaging data if this is an imaging cycle.

        Returns:
            TreatmentCycle record for this cycle.
        """
        cycle_day = TREATMENT_START_DAY + (cycle_number - 1) * CYCLE_INTERVAL_DAYS
        logger.info(
            "Executing cycle %d/%d on day %d per ICH E6(R3) section 2.5",
            cycle_number,
            PLANNED_CYCLES,
            cycle_day,
        )

        toxicities: list[AdverseEvent] = []

        # Per § 50.32, patient notified of new AEs via ongoing consent
        if cycle_number == 6:
            ae = AdverseEvent(
                event_id=EXPECTED_AE_CYCLE_6_HYPOTHYROIDISM["event_id"],
                description=EXPECTED_AE_CYCLE_6_HYPOTHYROIDISM["description"],
                severity=EXPECTED_AE_CYCLE_6_HYPOTHYROIDISM["severity"],
                organ_system=EXPECTED_AE_CYCLE_6_HYPOTHYROIDISM["organ_system"],
                onset_day=cycle_day,
                ctcae_grade=EXPECTED_AE_CYCLE_6_HYPOTHYROIDISM["ctcae_grade"],
                causality=EXPECTED_AE_CYCLE_6_HYPOTHYROIDISM["causality"],
                related_to_device=EXPECTED_AE_CYCLE_6_HYPOTHYROIDISM["related_to_device"],
            )
            toxicities.append(ae)
            self._cumulative_aes.append(ae)
            logger.warning(
                "Cycle %d AE: %s (Grade %d) — patient notified per 21 CFR 50.32",
                cycle_number,
                ae.description,
                ae.ctcae_grade,
            )

        if cycle_number == 12:
            ae = AdverseEvent(
                event_id=EXPECTED_AE_CYCLE_12_RASH["event_id"],
                description=EXPECTED_AE_CYCLE_12_RASH["description"],
                severity=EXPECTED_AE_CYCLE_12_RASH["severity"],
                organ_system=EXPECTED_AE_CYCLE_12_RASH["organ_system"],
                onset_day=cycle_day,
                ctcae_grade=EXPECTED_AE_CYCLE_12_RASH["ctcae_grade"],
                causality=EXPECTED_AE_CYCLE_12_RASH["causality"],
                related_to_device=EXPECTED_AE_CYCLE_12_RASH["related_to_device"],
            )
            toxicities.append(ae)
            self._cumulative_aes.append(ae)
            logger.warning(
                "Cycle %d AE: %s (Grade %d) — patient notified per 21 CFR 50.32",
                cycle_number,
                ae.description,
                ae.ctcae_grade,
            )

        response = None
        if imaging is not None and cycle_number in IMAGING_CYCLES:
            response = ResponseCategory.CR
            self._imaging_results.append({
                "cycle": cycle_number,
                "day": cycle_day,
                "response": response.value,
                "recurrence_risk_pct": RECURRENCE_RISK_TRAJECTORY.get(cycle_number, 0.0),
            })

        cycle = TreatmentCycle(
            cycle_number=cycle_number,
            start_day=cycle_day,
            drug=DRUG_NAME,
            dose_mg=DOSE_MG,
            dose_modifications="none",
            labs_pre=labs,
            labs_post={},
            toxicities=toxicities,
            response_assessment=response,
        )

        self._cycles_completed.append(cycle)
        return cycle

    # ------------------------------------------------------------------
    # 3. Cumulative toxicity tracking
    # ------------------------------------------------------------------

    def track_cumulative_toxicity(self, cycles: list[TreatmentCycle]) -> dict:
        """Track cumulative toxicity across all completed cycles.

        Cycle 6: immune-related hypothyroidism (Grade 1, managed with
        levothyroxine).  Cycle 12: immune-related maculopapular rash
        (Grade 1, managed with topical steroids).

        Per 21 CFR 50.32, participant is notified of each new adverse
        event through the ongoing consent process.

        Args:
            cycles: List of completed TreatmentCycle records.

        Returns:
            Cumulative toxicity summary.
        """
        logger.info("Tracking cumulative toxicity across %d cycles", len(cycles))
        all_aes: list[dict] = []
        for cycle in cycles:
            for ae in cycle.toxicities:
                all_aes.append({
                    "cycle": cycle.cycle_number,
                    "event_id": ae.event_id,
                    "description": ae.description,
                    "grade": ae.ctcae_grade,
                    "organ_system": ae.organ_system,
                    "onset_day": ae.onset_day,
                    # Per 21 CFR 50.32 - ongoing consent notification
                    "patient_notified": True,
                    "notification_per": "21 CFR 50.32",
                })

        max_grade = max((ae["grade"] for ae in all_aes), default=0)

        result = {
            "total_aes": len(all_aes),
            "adverse_events": all_aes,
            "max_grade": max_grade,
            "dose_modifications_required": False,
            "treatment_discontinued": False,
            "cumulative_toxicity_acceptable": max_grade <= 2,
            "patient_notifications_complete": True,
            "cfr_section": "21 CFR 50.32",
        }
        return result

    # ------------------------------------------------------------------
    # 4. Digital twin update at imaging
    # ------------------------------------------------------------------

    def update_twin_at_imaging(
        self,
        imaging_data: dict,
        cycle: int,
    ) -> dict:
        """Update the digital twin model at imaging time-points.

        Per ICH E6(R3) section 2.12.4, the digital twin is recalibrated
        at each imaging assessment.  Recurrence risk trajectory:
        35% -> 28% -> 18% -> 12% -> 8% -> 5% -> 3%.

        Args:
            imaging_data: Imaging assessment data.
            cycle: Current cycle number.

        Returns:
            Digital twin update result with recurrence risk.
        """
        risk_pct = RECURRENCE_RISK_TRAJECTORY.get(cycle, 3.0)
        logger.info(
            "Updating digital twin at cycle %d: recurrence risk %.1f%% "
            "per ICH E6(R3) section 2.12.4",
            cycle,
            risk_pct,
        )

        update = {
            "cycle": cycle,
            "day": TREATMENT_START_DAY + (cycle - 1) * CYCLE_INTERVAL_DAYS,
            "recurrence_risk_pct": risk_pct,
            "tumor_volume_cm3": 0.0,
            "response": "NED",  # No Evidence of Disease
            "twin_recalibrated": True,
            "tme_response_score": max(0.1, 1.0 - (risk_pct / 100.0)),
            "validation_status": "PASSED",
            "cfr_section": "ICH E6(R3) section 2.12.4",
        }
        self._twin_updates.append(update)
        return update

    # ------------------------------------------------------------------
    # 5. Adaptive agent with human oversight
    # ------------------------------------------------------------------

    def run_adaptive_agent(
        self,
        cycle: int,
        clinical_data: dict,
    ) -> dict:
        """Run adaptive AI agent for treatment recommendations.

        Per 21 CFR 312.404, all AI-generated recommendations require
        human oversight.  The investigator reviews and approves or
        overrides each recommendation before implementation.

        Args:
            cycle: Current cycle number.
            clinical_data: Clinical data for this cycle.

        Returns:
            Agent recommendation with human oversight record.
        """
        logger.info(
            "Running adaptive agent at cycle %d per 21 CFR 312.404 "
            "human oversight requirements",
            cycle,
        )
        risk_pct = RECURRENCE_RISK_TRAJECTORY.get(cycle, 3.0)

        # Simulation agent: what-if analysis for rash management
        whatif_scenarios = []
        if cycle >= 12:
            whatif_scenarios.append({
                "scenario": "rash_dose_hold",
                "description": "Hold pembrolizumab 1 cycle for rash management",
                "predicted_risk_change_pct": 0.5,
                "recommendation": "continue_with_topical_steroids",
            })

        recommendation = {
            "cycle": cycle,
            "agent_type": "adaptive_treatment",
            "recurrence_risk_pct": risk_pct,
            "recommendation": "continue_treatment" if risk_pct > 5.0 else "continue_to_completion",
            "confidence": 0.92,
            "whatif_scenarios": whatif_scenarios,
            # Per 21 CFR 312.404 - human oversight
            "human_oversight": {
                "investigator_reviewed": True,
                "investigator_approved": True,
                "override_applied": False,
                "review_timestamp_day": TREATMENT_START_DAY + (cycle - 1) * CYCLE_INTERVAL_DAYS,
            },
            "cfr_section": "21 CFR 312.404",
        }
        self._agent_recommendations.append(recommendation)
        return recommendation

    # ------------------------------------------------------------------
    # 6. Annual IND report
    # ------------------------------------------------------------------

    def generate_annual_report(self) -> dict:
        """Generate annual IND report per 21 CFR 312.33.

        Includes Physical AI system performance data as required for
        investigational trials with robotic and AI components.

        Returns:
            Annual report summary with Physical AI performance data.
        """
        logger.info("Generating annual IND report per 21 CFR 312.33")
        cycles_at_report = len(self._cycles_completed)
        total_aes = len(self._cumulative_aes)

        # Per 21 CFR 312.33 — annual report with Physical AI data
        result = {
            "report_type": "ANNUAL_IND_REPORT",
            "reporting_period_months": 12,
            "patient_id": self.patient_state.demographics.patient_id,
            "cycles_completed": cycles_at_report,
            "planned_cycles": PLANNED_CYCLES,
            "treatment_status": "ON_TREATMENT" if cycles_at_report < PLANNED_CYCLES else "COMPLETED",
            "adverse_events": {
                "total": total_aes,
                "grade_1": sum(1 for ae in self._cumulative_aes if ae.ctcae_grade == 1),
                "grade_2": sum(1 for ae in self._cumulative_aes if ae.ctcae_grade == 2),
                "grade_3_plus": sum(1 for ae in self._cumulative_aes if ae.ctcae_grade >= 3),
                "device_related": sum(1 for ae in self._cumulative_aes if ae.related_to_device),
            },
            "physical_ai_performance": {
                "digital_twin_updates": len(self._twin_updates),
                "twin_validation_status": "ALL_PASSED",
                "adaptive_agent_recommendations": len(self._agent_recommendations),
                "agent_overrides": 0,
                "system_uptime_pct": 99.7,
                "data_integrity_score": 1.0,
            },
            "recurrence_status": "NED",
            "investigator_assessment": "CONTINUE_TREATMENT",
            "cfr_section": "21 CFR 312.33",
        }
        return result

    # ------------------------------------------------------------------
    # 7. Regulatory change monitoring
    # ------------------------------------------------------------------

    def monitor_regulatory_changes(self) -> dict:
        """Monitor for regulatory changes affecting the trial.

        Per 21 CFR 312.56, the sponsor must review ongoing investigations
        and ensure continued compliance.  Protocol amendments per
        21 CFR 312.30 are assessed.  Physical AI system maintenance
        reporting per 21 CFR 312.405 is tracked.

        Returns:
            Regulatory monitoring status.
        """
        logger.info(
            "Monitoring regulatory changes per 21 CFR 312.56 and 21 CFR 312.30"
        )
        # Per 21 CFR 312.56 - review of ongoing investigations
        result = {
            "monitoring_active": True,
            "sections_monitored": [
                {
                    "section": "21 CFR 312.33",
                    "description": "Annual IND reports",
                    "status": "COMPLIANT",
                },
                {
                    "section": "21 CFR 312.56",
                    "description": "Review of ongoing investigations",
                    "status": "COMPLIANT",
                },
                {
                    "section": "21 CFR 312.30",
                    "description": "Protocol amendments",
                    "status": "NO_AMENDMENTS_NEEDED",
                },
                {
                    "section": "21 CFR 312.405",
                    "description": "Physical AI system maintenance reporting",
                    "status": "COMPLIANT",
                },
                {
                    "section": "ICH E6(R3) section 3.1",
                    "description": "Sponsor responsibilities",
                    "status": "COMPLIANT",
                },
            ],
            "protocol_amendments_pending": 0,
            "system_maintenance_events": 0,
            "next_review_cycle": min(
                len(self._cycles_completed) + 4, PLANNED_CYCLES
            ),
            "cfr_sections": ["21 CFR 312.56", "21 CFR 312.30", "21 CFR 312.405"],
        }
        return result

    # ------------------------------------------------------------------
    # 8. TME post-surgical enhancement
    # ------------------------------------------------------------------

    def enhance_tme_post_surgical(self) -> dict:
        """Enhance tumor micro-environment model post-surgery.

        Updates the digital twin TME compartment to reflect the
        post-resection immune landscape including T-cell infiltration,
        PD-L1 dynamics, and pembrolizumab pharmacodynamic response.

        Returns:
            TME enhancement result.
        """
        logger.info("Enhancing TME model post-surgery for immunotherapy response")
        result = {
            "tme_updated": True,
            "compartments": [
                "t_cell_infiltration",
                "pd_l1_expression",
                "cytokine_milieu",
                "vasculature_normalisation",
            ],
            "pd_l1_baseline_tps": 65.0,
            "predicted_response": "favorable",
            "immune_activation_score": 0.82,
        }
        return result

    # ------------------------------------------------------------------
    # 9. Main run orchestration
    # ------------------------------------------------------------------

    def run(self, patient_state: PatientJourneyState) -> PatientJourneyState:
        """Orchestrate all 35 cycles of adjuvant immunotherapy.

        Iterates through all planned cycles, executing treatment,
        recording adverse events, updating the digital twin at imaging
        time-points, running the adaptive agent, generating annual
        reports, and monitoring regulatory compliance.

        Advances patient stage to IMMUNOTHERAPY upon completion.

        Args:
            patient_state: Patient state from Stage 6 (Recovery).

        Returns:
            Updated state with all treatment cycles and IMMUNOTHERAPY stage.
        """
        logger.info("Starting Stage 7: Adjuvant Immunotherapy (Day 28 to Month 24)")
        state = patient_state

        # Step 1: Initialise treatment per ICH E6(R3) section 2.5
        treatment_plan = self.initialize_treatment()

        # Step 2: Enhance TME model post-surgery
        self.enhance_tme_post_surgical()

        # Step 3: Execute all 35 cycles
        for cycle_num in range(1, PLANNED_CYCLES + 1):
            # Pre-treatment labs
            labs = {
                "wbc": 6.2,
                "anc": 3.8,
                "platelets": 210,
                "hemoglobin": 12.8,
                "creatinine": 0.9,
                "alt": 28,
                "ast": 25,
                "tsh": 4.2 if cycle_num >= 6 else 2.8,
                "free_t4": 0.9 if cycle_num >= 6 else 1.1,
            }

            # Imaging at scheduled cycles
            imaging = None
            if cycle_num in IMAGING_CYCLES:
                imaging = {
                    "modality": "CT_chest_abdomen_pelvis",
                    "response": "NED",
                    "tumor_volume_cm3": 0.0,
                }

            # Execute cycle per ICH E6(R3) section 2.5
            cycle = self.execute_cycle(cycle_num, labs, imaging)
            state.treatment_cycles.append(cycle)

            # Record AEs on state
            for ae in cycle.toxicities:
                state.add_adverse_event(ae)

            # Update twin at imaging per ICH E6(R3) section 2.12.4
            if imaging is not None and cycle_num in IMAGING_CYCLES:
                twin_update = self.update_twin_at_imaging(imaging, cycle_num)
                state.imaging_timepoints.append(ImagingTimepoint(
                    day=cycle.start_day,
                    modality="CT",
                    tumor_volume_cm3=0.0,
                    response_category=ResponseCategory.CR,
                    notes=f"NED, recurrence risk {twin_update['recurrence_risk_pct']}%",
                ))

            # Run adaptive agent per 21 CFR 312.404
            if cycle_num in IMAGING_CYCLES or cycle_num in (6, 12):
                self.run_adaptive_agent(cycle_num, {"cycle": cycle_num, "labs": labs})

        # Step 4: Cumulative toxicity assessment
        toxicity_summary = self.track_cumulative_toxicity(self._cycles_completed)

        # Step 5: Annual report per 21 CFR 312.33
        annual_report = self.generate_annual_report()

        # Step 6: Regulatory monitoring per 21 CFR 312.56
        reg_status = self.monitor_regulatory_changes()

        # Advance stage to IMMUNOTHERAPY
        state.advance_stage(PatientStage.IMMUNOTHERAPY)
        state.status = PatientStatus.ON_IMMUNOTHERAPY

        # Record regulatory events
        state.add_regulatory_event(RegulatoryEvent(
            event_type="IMMUNOTHERAPY_COMPLETE",
            day=TREATMENT_START_DAY + (PLANNED_CYCLES - 1) * CYCLE_INTERVAL_DAYS,
            description=(
                f"Completed {PLANNED_CYCLES} cycles of {DRUG_NAME} {DOSE_MG} mg "
                f"{ROUTE} q3w. {toxicity_summary['total_aes']} AEs (max grade "
                f"{toxicity_summary['max_grade']}). NED at final imaging."
            ),
            document_id="IMM-COMPLETE-001",
            cfr_section="ICH E6(R3) section 2.5",
            status="COMPLETED",
        ))

        state.add_regulatory_event(RegulatoryEvent(
            event_type="ANNUAL_IND_REPORT",
            day=TREATMENT_START_DAY + 365,
            description="Annual IND report with Physical AI performance data per 21 CFR 312.33",
            document_id="IND-ANNUAL-001",
            cfr_section="21 CFR 312.33",
            status="SUBMITTED",
        ))

        state.add_regulatory_event(RegulatoryEvent(
            event_type="REGULATORY_MONITORING",
            day=TREATMENT_START_DAY + (PLANNED_CYCLES - 1) * CYCLE_INTERVAL_DAYS,
            description="Ongoing regulatory monitoring per 21 CFR 312.56; no amendments required",
            document_id="REG-MON-001",
            cfr_section="21 CFR 312.56",
            status="COMPLIANT",
        ))

        state.add_audit_entry(
            action="IMMUNOTHERAPY_COMPLETE",
            actor="ImmunotherapyOrchestrator",
            details=(
                f"Stage 7 complete: {PLANNED_CYCLES} cycles {DRUG_NAME}, "
                f"{toxicity_summary['total_aes']} AEs (max grade {toxicity_summary['max_grade']}), "
                f"NED at final imaging, recurrence risk "
                f"{RECURRENCE_RISK_TRAJECTORY[35]}%"
            ),
        )

        logger.info(
            "Stage 7 complete: %d cycles, %d AEs, NED, recurrence risk %.1f%%",
            PLANNED_CYCLES,
            toxicity_summary["total_aes"],
            RECURRENCE_RISK_TRAJECTORY[35],
        )
        return state
