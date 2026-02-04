"""
=============================================================================
EXAMPLE 05: Treatment Response Prediction with Digital Twins
=============================================================================

This example demonstrates how to predict patient-specific treatment
responses using digital twin technology for oncology clinical decision support.

CLINICAL CONTEXT:
-----------------
Treatment selection in oncology is challenging due to:
  - Inter-patient variability in treatment response
  - Limited ability to predict outcomes before treatment
  - Need to balance efficacy with quality of life
  - Multiple viable treatment options

Digital twins enable:
  - Patient-specific outcome prediction
  - Virtual treatment comparison
  - Optimal treatment selection
  - Adaptive treatment planning

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - NumPy 1.24.0+
    - SciPy 1.11.0+
    - TumorTwin (conceptual integration)

Optional:
    - MONAI 1.4.0+ (for imaging)
    - matplotlib (for visualization)

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from scipy.integrate import odeint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: PATIENT AND TUMOR DATA STRUCTURES
# =============================================================================


class TumorType(Enum):
    """Common tumor types for modeling."""

    NSCLC = "non_small_cell_lung_cancer"
    SCLC = "small_cell_lung_cancer"
    BREAST_TNBC = "triple_negative_breast_cancer"
    BREAST_HR = "hormone_receptor_positive_breast"
    COLORECTAL = "colorectal_cancer"
    PANCREATIC = "pancreatic_adenocarcinoma"
    GLIOBLASTOMA = "glioblastoma"


class TreatmentModality(Enum):
    """Treatment modalities."""

    CHEMOTHERAPY = "chemotherapy"
    RADIATION = "radiation"
    IMMUNOTHERAPY = "immunotherapy"
    TARGETED = "targeted_therapy"
    SURGERY = "surgery"
    COMBINED = "combined_modality"


@dataclass
class PatientProfile:
    """
    Complete patient profile for digital twin creation.

    Attributes:
        patient_id: Unique identifier
        age: Patient age
        sex: Patient sex
        tumor_type: Cancer type
        stage: Cancer stage (I-IV)
        tumor_volume_cm3: Current tumor volume
        biomarkers: Dictionary of biomarker status
        performance_status: ECOG performance status (0-4)
        comorbidities: List of comorbidities
    """

    patient_id: str
    age: int
    sex: str
    tumor_type: TumorType
    stage: str
    tumor_volume_cm3: float
    biomarkers: dict = field(default_factory=dict)
    performance_status: int = 0
    comorbidities: list = field(default_factory=list)


@dataclass
class TumorCharacteristics:
    """
    Tumor biological characteristics for modeling.

    These parameters are estimated from imaging and biopsy data.
    """

    proliferation_rate: float = 0.05  # Cell division rate (/day)
    death_rate: float = 0.01  # Natural cell death rate (/day)
    drug_sensitivity: float = 0.5  # Drug response parameter
    radiation_sensitivity: float = 0.3  # Alpha parameter for LQ model
    immune_visibility: float = 0.5  # Immunogenicity score
    heterogeneity_index: float = 0.3  # Intra-tumor heterogeneity


@dataclass
class TreatmentProtocol:
    """Treatment protocol specification."""

    modality: TreatmentModality
    name: str
    parameters: dict = field(default_factory=dict)
    duration_days: int = 0
    cycles: int = 1


@dataclass
class TreatmentOutcome:
    """Predicted treatment outcome."""

    protocol: TreatmentProtocol
    response_category: str  # CR, PR, SD, PD
    volume_change_percent: float
    time_to_progression_days: float
    survival_probability_1yr: float
    quality_of_life_score: float
    toxicity_profile: dict = field(default_factory=dict)
    confidence_interval: tuple = (0.0, 0.0)


# =============================================================================
# SECTION 2: TUMOR GROWTH MODELS
# =============================================================================


class TumorGrowthModel:
    """
    Base class for tumor growth mathematical models.

    Implements various models for tumor dynamics:
    - Exponential growth
    - Logistic growth
    - Gompertz growth
    - Reaction-diffusion (spatial)

    CLINICAL RELEVANCE:
    ------------------
    Model selection depends on tumor type and stage:
    - Early stage: Exponential/Logistic
    - Advanced: Gompertz (growth saturation)
    - Glioblastoma: Reaction-diffusion (invasion)
    """

    def __init__(self, characteristics: TumorCharacteristics):
        self.characteristics = characteristics

    def simulate(
        self, initial_volume: float, duration_days: float, treatment: TreatmentProtocol | None = None
    ) -> np.ndarray:
        """
        Simulate tumor volume over time.

        Args:
            initial_volume: Starting tumor volume (cm^3)
            duration_days: Simulation duration
            treatment: Treatment protocol (None for natural history)

        Returns:
            Array of volumes at each day
        """
        raise NotImplementedError


class ExponentialGrowthModel(TumorGrowthModel):
    """Exponential tumor growth: dV/dt = (r - d) * V"""

    def simulate(
        self, initial_volume: float, duration_days: float, treatment: TreatmentProtocol | None = None
    ) -> np.ndarray:
        """Simulate exponential growth with optional treatment."""
        t = np.arange(0, duration_days + 1)
        net_growth = self.characteristics.proliferation_rate - self.characteristics.death_rate

        if treatment is None:
            # Natural history
            volumes = initial_volume * np.exp(net_growth * t)
        else:
            # Modified growth with treatment
            volumes = self._simulate_with_treatment(initial_volume, t, treatment)

        return volumes

    def _simulate_with_treatment(
        self, initial_volume: float, timepoints: np.ndarray, treatment: TreatmentProtocol
    ) -> np.ndarray:
        """Simulate growth with treatment effect."""
        volumes = np.zeros_like(timepoints, dtype=float)
        volumes[0] = initial_volume

        # Treatment parameters
        drug_effect = self.characteristics.drug_sensitivity
        duration = treatment.duration_days

        for i, t in enumerate(timepoints[1:], 1):
            if t <= duration:
                # During treatment: reduced proliferation + cell kill
                effective_rate = self.characteristics.proliferation_rate * (1 - drug_effect)
                kill_rate = drug_effect * 0.1
                net_rate = effective_rate - self.characteristics.death_rate - kill_rate
            else:
                # Post-treatment: regrowth
                net_rate = self.characteristics.proliferation_rate - self.characteristics.death_rate

            volumes[i] = volumes[i - 1] * np.exp(net_rate)

        return volumes


class GompertzGrowthModel(TumorGrowthModel):
    """
    Gompertz tumor growth: dV/dt = a * V * ln(K/V)

    This model captures growth saturation observed in many tumors.
    """

    def __init__(
        self,
        characteristics: TumorCharacteristics,
        carrying_capacity: float = 1000.0,  # cm^3
    ):
        super().__init__(characteristics)
        self.carrying_capacity = carrying_capacity

    def simulate(
        self, initial_volume: float, duration_days: float, treatment: TreatmentProtocol | None = None
    ) -> np.ndarray:
        """Simulate Gompertz growth."""
        t = np.arange(0, duration_days + 1)
        a = self.characteristics.proliferation_rate
        K = self.carrying_capacity

        def gompertz_ode(V, t, treatment_active):
            if V <= 0:
                return 0
            growth = a * V * np.log(K / V)

            if treatment_active:
                kill = self.characteristics.drug_sensitivity * V * 0.1
                return growth - kill
            return growth

        # Solve ODE
        treatment_active = treatment is not None
        volumes = odeint(gompertz_ode, initial_volume, t, args=(treatment_active,))

        return volumes.flatten()


# =============================================================================
# SECTION 3: TREATMENT RESPONSE MODELS
# =============================================================================


class TreatmentResponseModel:
    """
    Model treatment-specific responses.

    Implements pharmacodynamic models for:
    - Chemotherapy (cell cycle effects)
    - Radiation (linear-quadratic model)
    - Immunotherapy (immune kinetics)
    - Targeted therapy (pathway inhibition)
    """

    def __init__(self, patient: PatientProfile, tumor: TumorCharacteristics):
        self.patient = patient
        self.tumor = tumor

    def predict_response(
        self, protocol: TreatmentProtocol, baseline_volume: float, horizon_days: int = 180
    ) -> TreatmentOutcome:
        """
        Predict treatment response.

        Args:
            protocol: Treatment protocol
            baseline_volume: Starting tumor volume
            horizon_days: Prediction horizon

        Returns:
            TreatmentOutcome with predicted response
        """
        if protocol.modality == TreatmentModality.CHEMOTHERAPY:
            return self._predict_chemotherapy(protocol, baseline_volume, horizon_days)
        elif protocol.modality == TreatmentModality.RADIATION:
            return self._predict_radiation(protocol, baseline_volume, horizon_days)
        elif protocol.modality == TreatmentModality.IMMUNOTHERAPY:
            return self._predict_immunotherapy(protocol, baseline_volume, horizon_days)
        else:
            return self._predict_generic(protocol, baseline_volume, horizon_days)

    def _predict_chemotherapy(
        self, protocol: TreatmentProtocol, baseline_volume: float, horizon_days: int
    ) -> TreatmentOutcome:
        """Predict chemotherapy response using PK/PD model."""
        # Drug parameters
        drug = protocol.parameters.get("drug", "generic")
        cycles = protocol.cycles
        cycle_length = protocol.parameters.get("cycle_days", 21)

        # Simulate tumor dynamics
        model = ExponentialGrowthModel(self.tumor)
        protocol.duration_days = cycles * cycle_length
        volumes = model.simulate(baseline_volume, horizon_days, protocol)

        # Calculate metrics
        nadir_volume = np.min(volumes)
        final_volume = volumes[-1]
        volume_change = (final_volume - baseline_volume) / baseline_volume * 100

        # Response category
        response = self._classify_response(volume_change)

        # Time to progression
        ttp = self._estimate_time_to_progression(volumes, baseline_volume)

        # Survival probability (simplified model)
        survival_1yr = self._estimate_survival(response, self.patient)

        # Quality of life impact
        qol = self._estimate_quality_of_life(protocol)

        # Toxicity
        toxicity = self._estimate_chemotherapy_toxicity(drug, cycles)

        return TreatmentOutcome(
            protocol=protocol,
            response_category=response,
            volume_change_percent=volume_change,
            time_to_progression_days=ttp,
            survival_probability_1yr=survival_1yr,
            quality_of_life_score=qol,
            toxicity_profile=toxicity,
            confidence_interval=(volume_change - 15, volume_change + 15),
        )

    def _predict_radiation(
        self, protocol: TreatmentProtocol, baseline_volume: float, horizon_days: int
    ) -> TreatmentOutcome:
        """Predict radiation response using LQ model."""
        total_dose = protocol.parameters.get("total_dose_gy", 60)
        fractions = protocol.parameters.get("fractions", 30)

        # Linear-quadratic parameters
        alpha = self.tumor.radiation_sensitivity
        beta = alpha / 10  # Assume alpha/beta = 10 for tumor

        # Surviving fraction per fraction
        dose_per_fraction = total_dose / fractions
        sf_per_fraction = np.exp(-alpha * dose_per_fraction - beta * dose_per_fraction**2)
        total_sf = sf_per_fraction**fractions

        # Predicted volume after treatment
        final_volume = baseline_volume * total_sf

        # Regrowth after treatment
        growth_model = GompertzGrowthModel(self.tumor)
        regrowth_duration = horizon_days - fractions * 1.4  # Treatment duration
        regrowth = growth_model.simulate(final_volume, regrowth_duration)

        final_volume = regrowth[-1] if len(regrowth) > 0 else final_volume
        volume_change = (final_volume - baseline_volume) / baseline_volume * 100
        response = self._classify_response(volume_change)

        return TreatmentOutcome(
            protocol=protocol,
            response_category=response,
            volume_change_percent=volume_change,
            time_to_progression_days=self._estimate_time_to_progression(regrowth, final_volume),
            survival_probability_1yr=self._estimate_survival(response, self.patient),
            quality_of_life_score=0.7,
            toxicity_profile={"fatigue": 0.9, "skin_reaction": 0.6, "esophagitis": 0.3},
        )

    def _predict_immunotherapy(
        self, protocol: TreatmentProtocol, baseline_volume: float, horizon_days: int
    ) -> TreatmentOutcome:
        """Predict immunotherapy response using immune kinetics."""
        # Immune response parameters
        immune_visibility = self.tumor.immune_visibility
        pd_l1 = self.patient.biomarkers.get("PD-L1", 0)

        # Probability of response based on biomarkers
        response_prob = 0.2 + 0.3 * immune_visibility + 0.3 * (pd_l1 / 100)

        # Simulate immune-tumor dynamics
        def immune_dynamics(y, t):
            V, T = y  # Tumor volume, T-cell count
            dV = self.tumor.proliferation_rate * V - 0.1 * T * V / (V + 100)
            dT = 0.1 * T * V / (V + 1000) - 0.01 * T
            return [dV, dT]

        t = np.arange(0, horizon_days + 1)
        y0 = [baseline_volume, 100]  # Initial tumor and T-cells
        solution = odeint(immune_dynamics, y0, t)

        volumes = solution[:, 0]
        final_volume = volumes[-1]
        volume_change = (final_volume - baseline_volume) / baseline_volume * 100
        response = self._classify_response(volume_change)

        return TreatmentOutcome(
            protocol=protocol,
            response_category=response,
            volume_change_percent=volume_change,
            time_to_progression_days=self._estimate_time_to_progression(volumes, baseline_volume),
            survival_probability_1yr=self._estimate_survival(response, self.patient),
            quality_of_life_score=0.8,
            toxicity_profile={"irAE": response_prob * 0.3, "fatigue": 0.4},
        )

    def _predict_generic(
        self, protocol: TreatmentProtocol, baseline_volume: float, horizon_days: int
    ) -> TreatmentOutcome:
        """Generic treatment prediction."""
        return TreatmentOutcome(
            protocol=protocol,
            response_category="SD",
            volume_change_percent=0,
            time_to_progression_days=180,
            survival_probability_1yr=0.5,
            quality_of_life_score=0.7,
        )

    def _classify_response(self, volume_change_percent: float) -> str:
        """Classify response using RECIST-like criteria."""
        if volume_change_percent <= -100:
            return "CR"  # Complete Response
        elif volume_change_percent <= -30:
            return "PR"  # Partial Response
        elif volume_change_percent < 20:
            return "SD"  # Stable Disease
        else:
            return "PD"  # Progressive Disease

    def _estimate_time_to_progression(self, volumes: np.ndarray, baseline: float) -> float:
        """Estimate time to disease progression."""
        # Progression = 20% increase from nadir
        nadir = np.min(volumes)
        nadir_idx = np.argmin(volumes)
        threshold = nadir * 1.2

        for i in range(nadir_idx, len(volumes)):
            if volumes[i] > threshold:
                return float(i)

        return float(len(volumes))  # No progression within horizon

    def _estimate_survival(self, response: str, patient: PatientProfile) -> float:
        """Estimate 1-year survival probability."""
        base_survival = {"CR": 0.95, "PR": 0.80, "SD": 0.60, "PD": 0.30}

        # Adjust for performance status
        ps_adjustment = 1 - patient.performance_status * 0.1

        return base_survival.get(response, 0.5) * ps_adjustment

    def _estimate_quality_of_life(self, protocol: TreatmentProtocol) -> float:
        """Estimate quality of life impact (0-1 scale)."""
        base_qol = {
            TreatmentModality.SURGERY: 0.6,
            TreatmentModality.CHEMOTHERAPY: 0.5,
            TreatmentModality.RADIATION: 0.7,
            TreatmentModality.IMMUNOTHERAPY: 0.8,
            TreatmentModality.TARGETED: 0.75,
        }
        return base_qol.get(protocol.modality, 0.7)

    def _estimate_chemotherapy_toxicity(self, drug: str, cycles: int) -> dict:
        """Estimate chemotherapy toxicity profile."""
        return {
            "neutropenia": min(0.8, 0.3 + cycles * 0.1),
            "nausea": 0.6,
            "fatigue": 0.8,
            "neuropathy": min(0.5, cycles * 0.08),
            "alopecia": 0.7,
        }


# =============================================================================
# SECTION 4: TREATMENT COMPARISON AND OPTIMIZATION
# =============================================================================


class TreatmentOptimizer:
    """
    Compare and optimize treatment strategies.

    Enables:
    - Multi-treatment comparison
    - Pareto optimization (efficacy vs toxicity)
    - Personalized treatment ranking
    """

    def __init__(self, patient: PatientProfile, tumor: TumorCharacteristics):
        self.patient = patient
        self.tumor = tumor
        self.response_model = TreatmentResponseModel(patient, tumor)

    def compare_treatments(
        self, protocols: list[TreatmentProtocol], baseline_volume: float, horizon_days: int = 180
    ) -> list[TreatmentOutcome]:
        """
        Compare multiple treatment options.

        Args:
            protocols: List of treatment protocols
            baseline_volume: Current tumor volume
            horizon_days: Prediction horizon

        Returns:
            List of outcomes, sorted by predicted benefit
        """
        outcomes = []

        for protocol in protocols:
            outcome = self.response_model.predict_response(protocol, baseline_volume, horizon_days)
            outcomes.append(outcome)

        # Sort by response (CR > PR > SD > PD) then by volume change
        response_rank = {"CR": 0, "PR": 1, "SD": 2, "PD": 3}
        outcomes.sort(key=lambda x: (response_rank.get(x.response_category, 4), x.volume_change_percent))

        return outcomes

    def generate_recommendation(self, outcomes: list[TreatmentOutcome], optimization_target: str = "efficacy") -> dict:
        """
        Generate treatment recommendation.

        Args:
            outcomes: List of predicted outcomes
            optimization_target: What to optimize (efficacy, qol, balanced)

        Returns:
            Recommendation dictionary
        """
        if not outcomes:
            return {"recommendation": "Insufficient data"}

        if optimization_target == "efficacy":
            # Prioritize tumor control
            best = min(outcomes, key=lambda x: x.volume_change_percent)
        elif optimization_target == "qol":
            # Prioritize quality of life
            best = max(outcomes, key=lambda x: x.quality_of_life_score)
        else:
            # Balanced approach
            best = min(outcomes, key=lambda x: -x.quality_of_life_score + x.volume_change_percent / 100)

        return {
            "recommended_treatment": best.protocol.name,
            "expected_response": best.response_category,
            "volume_change": f"{best.volume_change_percent:.1f}%",
            "survival_1yr": f"{best.survival_probability_1yr:.1%}",
            "quality_of_life": f"{best.quality_of_life_score:.2f}",
            "confidence": "Moderate",
            "rationale": self._generate_rationale(best, optimization_target),
        }

    def _generate_rationale(self, outcome: TreatmentOutcome, target: str) -> str:
        """Generate explanation for recommendation."""
        if target == "efficacy":
            return (
                f"{outcome.protocol.name} predicted to achieve {outcome.response_category} "
                f"with {abs(outcome.volume_change_percent):.0f}% tumor reduction."
            )
        elif target == "qol":
            return (
                f"{outcome.protocol.name} offers best quality of life "
                f"(score: {outcome.quality_of_life_score:.2f}) while maintaining "
                f"{outcome.response_category} response."
            )
        else:
            return (
                f"{outcome.protocol.name} provides optimal balance of efficacy "
                f"({outcome.response_category}) and tolerability "
                f"(QoL: {outcome.quality_of_life_score:.2f})."
            )


# =============================================================================
# SECTION 5: CLINICAL DECISION SUPPORT REPORT
# =============================================================================


class ClinicalDecisionReport:
    """Generate clinical decision support report."""

    def __init__(self, patient: PatientProfile, outcomes: list[TreatmentOutcome], recommendation: dict):
        self.patient = patient
        self.outcomes = outcomes
        self.recommendation = recommendation

    def generate(self) -> str:
        """Generate formatted clinical report."""
        report = f"""
================================================================================
              TREATMENT RESPONSE PREDICTION REPORT
              Digital Twin Clinical Decision Support
================================================================================

Patient Information
-------------------
Patient ID: {self.patient.patient_id}
Age/Sex: {self.patient.age} / {self.patient.sex}
Diagnosis: {self.patient.tumor_type.value}
Stage: {self.patient.stage}
Tumor Volume: {self.patient.tumor_volume_cm3:.2f} cm³
Performance Status: ECOG {self.patient.performance_status}

Biomarkers: {self._format_biomarkers()}

Treatment Options Analyzed
--------------------------
"""
        for i, outcome in enumerate(self.outcomes, 1):
            report += f"""
{i}. {outcome.protocol.name}
   Response: {outcome.response_category}
   Volume Change: {outcome.volume_change_percent:+.1f}%
   Time to Progression: {outcome.time_to_progression_days:.0f} days
   1-Year Survival: {outcome.survival_probability_1yr:.1%}
   Quality of Life: {outcome.quality_of_life_score:.2f}/1.0
"""

        report += f"""
Recommendation
--------------
Based on the digital twin analysis:

  RECOMMENDED: {self.recommendation["recommended_treatment"]}

  Expected Response: {self.recommendation["expected_response"]}
  Volume Change: {self.recommendation["volume_change"]}
  1-Year Survival: {self.recommendation["survival_1yr"]}
  Quality of Life: {self.recommendation["quality_of_life"]}

Rationale:
  {self.recommendation["rationale"]}

Confidence Level: {self.recommendation["confidence"]}

================================================================================
DISCLAIMER: This report is generated by an AI-based clinical decision support
system and is intended to assist, not replace, clinical judgment. All treatment
decisions should be made by qualified healthcare providers in consultation with
the patient.

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Framework: Physical AI Oncology Trials v1.0
================================================================================
"""
        return report

    def _format_biomarkers(self) -> str:
        """Format biomarker information."""
        if not self.patient.biomarkers:
            return "Not available"

        return ", ".join(f"{k}: {v}" for k, v in self.patient.biomarkers.items())


# =============================================================================
# SECTION 6: MAIN PIPELINE
# =============================================================================


def predict_treatment_response(patient_id: str = "ONCO-2026-001", tumor_volume_cm3: float = 15.0) -> dict:
    """
    Predict treatment responses for a patient using digital twin.

    This function demonstrates the complete workflow for
    treatment response prediction and clinical decision support.

    Args:
        patient_id: Patient identifier
        tumor_volume_cm3: Current tumor volume

    Returns:
        Dictionary with predictions and recommendation
    """
    logger.info("=" * 60)
    logger.info("TREATMENT RESPONSE PREDICTION WITH DIGITAL TWINS")
    logger.info("=" * 60)

    # Create patient profile
    patient = PatientProfile(
        patient_id=patient_id,
        age=62,
        sex="M",
        tumor_type=TumorType.NSCLC,
        stage="IIIB",
        tumor_volume_cm3=tumor_volume_cm3,
        biomarkers={"PD-L1": 60, "EGFR": "negative", "ALK": "negative"},
        performance_status=1,
    )

    # Estimate tumor characteristics
    tumor = TumorCharacteristics(
        proliferation_rate=0.03,
        death_rate=0.005,
        drug_sensitivity=0.4,
        radiation_sensitivity=0.35,
        immune_visibility=0.6,
    )

    logger.info(f"Patient: {patient.patient_id}")
    logger.info(f"Tumor type: {patient.tumor_type.value}")
    logger.info(f"Baseline volume: {patient.tumor_volume_cm3:.2f} cm³")

    # Define treatment protocols
    protocols = [
        TreatmentProtocol(
            modality=TreatmentModality.CHEMOTHERAPY,
            name="Carboplatin/Paclitaxel",
            parameters={"drug": "carboplatin_paclitaxel", "cycle_days": 21},
            cycles=4,
        ),
        TreatmentProtocol(
            modality=TreatmentModality.RADIATION,
            name="Definitive Chemoradiation",
            parameters={"total_dose_gy": 60, "fractions": 30},
        ),
        TreatmentProtocol(
            modality=TreatmentModality.IMMUNOTHERAPY,
            name="Pembrolizumab",
            parameters={"agent": "pembrolizumab", "dose_mg_kg": 2},
            cycles=12,
            duration_days=365,
        ),
        TreatmentProtocol(
            modality=TreatmentModality.COMBINED,
            name="Chemoimmunotherapy",
            parameters={"chemotherapy": "carboplatin_paclitaxel", "immunotherapy": "pembrolizumab"},
            cycles=4,
        ),
    ]

    # Compare treatments
    optimizer = TreatmentOptimizer(patient, tumor)
    outcomes = optimizer.compare_treatments(protocols, patient.tumor_volume_cm3, horizon_days=180)

    # Generate recommendation
    recommendation = optimizer.generate_recommendation(outcomes, optimization_target="balanced")

    # Generate clinical report
    report = ClinicalDecisionReport(patient, outcomes, recommendation)
    print(report.generate())

    logger.info("=" * 60)
    logger.info("PREDICTION COMPLETE")
    logger.info(f"Recommended: {recommendation['recommended_treatment']}")
    logger.info("=" * 60)

    return {
        "patient_id": patient_id,
        "recommendation": recommendation,
        "outcomes": [
            {"treatment": o.protocol.name, "response": o.response_category, "volume_change": o.volume_change_percent}
            for o in outcomes
        ],
    }


if __name__ == "__main__":
    result = predict_treatment_response()
    print(f"\nPrediction complete: {result['recommendation']['recommended_treatment']}")
