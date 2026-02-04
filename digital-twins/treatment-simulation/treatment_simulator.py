"""
Treatment Response Simulator for Oncology Digital Twins

Multi-modality treatment simulation including chemotherapy, radiation,
immunotherapy, and surgical planning. Integrates with patient-specific
digital twins for personalized outcome prediction.

Version: 1.0.0
Last Updated: February 2026
Framework Dependencies:
    - NumPy 1.24.0+
    - SciPy 1.11.0+
    - Patient modeling from digital_twins.patient_modeling

License: MIT
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TreatmentType(Enum):
    """Supported treatment modalities."""

    CHEMOTHERAPY = "chemotherapy"
    RADIATION = "radiation"
    IMMUNOTHERAPY = "immunotherapy"
    TARGETED_THERAPY = "targeted_therapy"
    SURGERY = "surgery"
    COMBINED = "combined"


class ResponseType(Enum):
    """RECIST-style response categories."""

    COMPLETE_RESPONSE = "CR"
    PARTIAL_RESPONSE = "PR"
    STABLE_DISEASE = "SD"
    PROGRESSIVE_DISEASE = "PD"


@dataclass
class TreatmentProtocol:
    """Treatment protocol specification.

    Attributes:
        type: Treatment modality
        name: Protocol name
        parameters: Modality-specific parameters
        schedule: Treatment schedule
        duration_days: Total treatment duration
    """

    type: TreatmentType
    name: str = ""
    parameters: dict = field(default_factory=dict)
    schedule: dict = field(default_factory=dict)
    duration_days: int = 0


@dataclass
class TreatmentResponse:
    """Treatment response prediction results.

    Attributes:
        timepoints: Time points (days)
        volumes: Predicted tumor volumes
        response_category: RECIST response category
        volume_change_percent: Percent change from baseline
        control_probability: Probability of local control
        toxicity_estimate: Predicted toxicity metrics
        confidence_interval: 95% CI for predictions
    """

    timepoints: np.ndarray
    volumes: np.ndarray
    response_category: ResponseType
    volume_change_percent: float
    control_probability: float
    toxicity_estimate: dict = field(default_factory=dict)
    confidence_interval: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


class TreatmentSimulator:
    """
    Multi-modality treatment response simulator.

    Simulates patient-specific treatment responses using digital twin
    technology, enabling in-silico treatment comparison and optimization.

    Attributes:
        patient_twin: Patient-specific digital twin
        baseline_volume: Pre-treatment tumor volume

    Example:
        >>> simulator = TreatmentSimulator(patient_twin=patient_dt)
        >>> response = simulator.predict_response(
        ...     treatment={"type": "radiation", "total_dose_gy": 60},
        ...     horizon_days=90
        ... )
        >>> print(f"Response: {response.response_category.value}")

    References:
        - Nature Reviews Cancer: Digital twins in oncology (2025)
        - AAPM TG-166: Model-based dose calculation
    """

    def __init__(self, patient_twin):
        """Initialize treatment simulator with patient digital twin.

        Args:
            patient_twin: Calibrated PatientDigitalTwin instance
        """
        self.patient_twin = patient_twin
        self.baseline_volume = patient_twin.current_volume_cm3

        # Initialize modality-specific models
        self._radiation_model = LinearQuadraticModel()
        self._chemo_model = PharmacokineticModel()
        self._immuno_model = ImmunotherapyModel()

        logger.info(f"TreatmentSimulator initialized for patient {patient_twin.patient_id}")

    def predict_response(
        self, treatment: dict | TreatmentProtocol, horizon_days: float = 90, include_uncertainty: bool = True
    ) -> TreatmentResponse:
        """
        Predict treatment response for given protocol.

        Args:
            treatment: Treatment protocol specification
            horizon_days: Prediction horizon in days
            include_uncertainty: Whether to compute confidence intervals

        Returns:
            TreatmentResponse with predicted outcomes
        """
        # Parse treatment protocol
        if isinstance(treatment, dict):
            treatment_type = TreatmentType(treatment.get("type", "radiation"))
        else:
            treatment_type = treatment.type

        logger.info(f"Simulating {treatment_type.value} response over {horizon_days} days")

        # Route to appropriate model
        if treatment_type == TreatmentType.RADIATION:
            return self._simulate_radiation(treatment, horizon_days, include_uncertainty)
        elif treatment_type == TreatmentType.CHEMOTHERAPY:
            return self._simulate_chemotherapy(treatment, horizon_days, include_uncertainty)
        elif treatment_type == TreatmentType.IMMUNOTHERAPY:
            return self._simulate_immunotherapy(treatment, horizon_days, include_uncertainty)
        elif treatment_type == TreatmentType.SURGERY:
            return self._simulate_surgery(treatment, horizon_days, include_uncertainty)
        elif treatment_type == TreatmentType.COMBINED:
            return self._simulate_combined(treatment, horizon_days, include_uncertainty)
        else:
            return self._simulate_generic(treatment, horizon_days, include_uncertainty)

    def _simulate_radiation(self, treatment: dict, horizon_days: float, include_uncertainty: bool) -> TreatmentResponse:
        """Simulate radiation therapy response using LQ model."""
        total_dose = treatment.get("total_dose_gy", 60)
        fractions = treatment.get("fractions", 30)
        dose_per_fraction = total_dose / fractions

        # Get tumor-specific alpha/beta
        alpha, beta = self._radiation_model.get_tissue_parameters(self.patient_twin.clinical_data.tumor_grade)

        # Calculate surviving fraction per fraction
        sf_per_fraction = self._radiation_model.surviving_fraction(dose_per_fraction, alpha, beta)

        # Total surviving fraction
        total_sf = sf_per_fraction**fractions

        # Calculate tumor volume trajectory
        treatment_duration = fractions * 1.4  # Weekday fractions
        timepoints = np.linspace(0, horizon_days, int(horizon_days) + 1)
        volumes = np.zeros_like(timepoints)
        volumes[0] = self.baseline_volume

        # During treatment: exponential cell kill + regrowth
        proliferation = self.patient_twin.proliferation_rate

        for i, t in enumerate(timepoints[1:], 1):
            if t <= treatment_duration:
                # Fraction delivery (simplified daily)
                fractions_delivered = min(int(t / 1.4), fractions)
                sf = sf_per_fraction**fractions_delivered
                # Regrowth between fractions
                regrowth = np.exp(proliferation * t * 0.5)
                volumes[i] = self.baseline_volume * sf * regrowth
            else:
                # Post-treatment regrowth from nadir
                nadir_idx = int(treatment_duration)
                if nadir_idx < len(volumes):
                    nadir_volume = min(volumes[: nadir_idx + 1])
                else:
                    nadir_volume = volumes[i - 1]
                time_since_treatment = t - treatment_duration
                volumes[i] = nadir_volume * np.exp(proliferation * time_since_treatment)

        # Calculate metrics
        final_volume = volumes[-1]
        volume_change = (final_volume - self.baseline_volume) / self.baseline_volume * 100

        # Tumor control probability (TCP)
        tcp = self._radiation_model.tumor_control_probability(
            self.baseline_volume * 1e6,  # cells
            total_sf,
        )

        # Response category
        response_cat = self._classify_response(volume_change)

        # Uncertainty estimation
        confidence_interval = {}
        if include_uncertainty:
            confidence_interval = self._estimate_uncertainty(volumes, alpha_std=0.05, beta_std=0.01)

        return TreatmentResponse(
            timepoints=timepoints,
            volumes=volumes,
            response_category=response_cat,
            volume_change_percent=volume_change,
            control_probability=tcp,
            toxicity_estimate=self._estimate_radiation_toxicity(total_dose, fractions),
            confidence_interval=confidence_interval,
            metrics={
                "total_dose_gy": total_dose,
                "fractions": fractions,
                "surviving_fraction": total_sf,
                "nadir_volume": np.min(volumes),
                "nadir_day": timepoints[np.argmin(volumes)],
            },
        )

    def _simulate_chemotherapy(
        self, treatment: dict, horizon_days: float, include_uncertainty: bool
    ) -> TreatmentResponse:
        """Simulate chemotherapy response using PK/PD model."""
        drug = treatment.get("drug", "generic")
        dose_mg_m2 = treatment.get("dose_mg_m2", 100)
        cycles = treatment.get("cycles", 6)
        cycle_days = treatment.get("cycle_days", 21)

        # Get drug-specific parameters
        drug_params = self._chemo_model.get_drug_parameters(drug)

        timepoints = np.linspace(0, horizon_days, int(horizon_days) + 1)
        volumes = np.zeros_like(timepoints)
        volumes[0] = self.baseline_volume

        proliferation = self.patient_twin.proliferation_rate

        for i, t in enumerate(timepoints[1:], 1):
            # Determine current cycle
            current_cycle = int(t / cycle_days)

            if current_cycle < cycles:
                # Drug effect during treatment
                time_in_cycle = t % cycle_days
                concentration = self._chemo_model.concentration_profile(dose_mg_m2, time_in_cycle, drug_params)
                kill_rate = self._chemo_model.cell_kill_rate(concentration, drug_params)

                # Net growth = proliferation - kill
                net_rate = proliferation - kill_rate
                volumes[i] = volumes[i - 1] * np.exp(net_rate)
            else:
                # Post-treatment regrowth
                volumes[i] = volumes[i - 1] * np.exp(proliferation)

        # Calculate metrics
        final_volume = volumes[-1]
        volume_change = (final_volume - self.baseline_volume) / self.baseline_volume * 100
        response_cat = self._classify_response(volume_change)

        return TreatmentResponse(
            timepoints=timepoints,
            volumes=volumes,
            response_category=response_cat,
            volume_change_percent=volume_change,
            control_probability=0.0,  # Different metric for chemo
            toxicity_estimate=self._estimate_chemo_toxicity(drug, dose_mg_m2, cycles),
            confidence_interval={},
            metrics={
                "drug": drug,
                "total_dose": dose_mg_m2 * cycles,
                "cycles_completed": cycles,
                "nadir_volume": np.min(volumes),
            },
        )

    def _simulate_immunotherapy(
        self, treatment: dict, horizon_days: float, include_uncertainty: bool
    ) -> TreatmentResponse:
        """Simulate immunotherapy response."""
        agent = treatment.get("agent", "pembrolizumab")
        dose_mg_kg = treatment.get("dose_mg_kg", 2)
        cycle_weeks = treatment.get("cycle_weeks", 3)

        timepoints = np.linspace(0, horizon_days, int(horizon_days) + 1)

        # Solve immune-tumor dynamics ODE
        y0 = [self.baseline_volume, 100]  # [Tumor volume, T-cell count]

        def dynamics(y, t):
            V, T = y
            proliferation = self.patient_twin.proliferation_rate

            # Immune activation enhanced by checkpoint inhibitor
            immune_boost = 2.0  # Fold increase in immune activity

            dV = proliferation * V - self._immuno_model.kill_rate * T * V / (V + 100)
            dT = (
                self._immuno_model.activation_rate * immune_boost * T * V / (V + 1000)
                - self._immuno_model.exhaustion_rate * T
            )

            return [dV, dT]

        solution = odeint(dynamics, y0, timepoints)
        volumes = solution[:, 0]

        # Calculate metrics
        final_volume = volumes[-1]
        volume_change = (final_volume - self.baseline_volume) / self.baseline_volume * 100
        response_cat = self._classify_response(volume_change)

        return TreatmentResponse(
            timepoints=timepoints,
            volumes=volumes,
            response_category=response_cat,
            volume_change_percent=volume_change,
            control_probability=0.0,
            toxicity_estimate={"irAE_risk": 0.15},
            confidence_interval={},
            metrics={"agent": agent, "immune_response": "active" if volume_change < -20 else "limited"},
        )

    def _simulate_surgery(self, treatment: dict, horizon_days: float, include_uncertainty: bool) -> TreatmentResponse:
        """Simulate surgical resection and recurrence."""
        resection_extent = treatment.get("resection_extent", 0.95)  # 95% resection
        margin_mm = treatment.get("margin_mm", 5)

        timepoints = np.linspace(0, horizon_days, int(horizon_days) + 1)
        volumes = np.zeros_like(timepoints)

        # Pre-surgery volume
        volumes[0] = self.baseline_volume

        # Immediate post-surgery (day 1)
        residual_fraction = 1 - resection_extent
        surgery_day = 7  # Surgery on day 7

        proliferation = self.patient_twin.proliferation_rate

        for i, t in enumerate(timepoints[1:], 1):
            if t < surgery_day:
                # Pre-operative growth
                volumes[i] = volumes[i - 1] * np.exp(proliferation)
            elif t == surgery_day or (t > surgery_day and volumes[i - 1] == 0):
                # Surgery day - immediate reduction
                volumes[i] = self.baseline_volume * residual_fraction
            else:
                # Post-operative regrowth from residual
                volumes[i] = volumes[i - 1] * np.exp(proliferation)

        # Calculate metrics
        final_volume = volumes[-1]
        volume_change = (final_volume - self.baseline_volume) / self.baseline_volume * 100
        response_cat = self._classify_response(volume_change)

        # Recurrence probability based on margin
        recurrence_prob = self._estimate_recurrence_probability(margin_mm, residual_fraction)

        return TreatmentResponse(
            timepoints=timepoints,
            volumes=volumes,
            response_category=response_cat,
            volume_change_percent=volume_change,
            control_probability=1 - recurrence_prob,
            toxicity_estimate={"surgical_morbidity": 0.05},
            confidence_interval={},
            metrics={
                "resection_extent": resection_extent,
                "margin_mm": margin_mm,
                "residual_volume": self.baseline_volume * residual_fraction,
                "recurrence_probability": recurrence_prob,
            },
        )

    def _simulate_combined(self, treatment: dict, horizon_days: float, include_uncertainty: bool) -> TreatmentResponse:
        """Simulate combined modality treatment."""
        modalities = treatment.get("modalities", [])
        sequencing = treatment.get("sequencing", "sequential")

        # Accumulate effects from all modalities
        current_volume = self.baseline_volume
        timepoints = np.linspace(0, horizon_days, int(horizon_days) + 1)
        volumes = np.zeros_like(timepoints)
        volumes[0] = current_volume

        for modality in modalities:
            # Simulate each modality
            response = self.predict_response(modality, horizon_days, include_uncertainty=False)

            # Apply volume reduction
            volume_factor = response.volumes[-1] / response.volumes[0]
            current_volume *= volume_factor

        # Simplified: apply combined effect
        proliferation = self.patient_twin.proliferation_rate * 0.5  # Reduced during treatment

        for i, t in enumerate(timepoints[1:], 1):
            volumes[i] = volumes[i - 1] * np.exp(proliferation)

        volumes = volumes * (current_volume / self.baseline_volume)

        final_volume = volumes[-1]
        volume_change = (final_volume - self.baseline_volume) / self.baseline_volume * 100
        response_cat = self._classify_response(volume_change)

        return TreatmentResponse(
            timepoints=timepoints,
            volumes=volumes,
            response_category=response_cat,
            volume_change_percent=volume_change,
            control_probability=0.0,
            toxicity_estimate={},
            confidence_interval={},
            metrics={"modalities": len(modalities)},
        )

    def _simulate_generic(self, treatment: dict, horizon_days: float, include_uncertainty: bool) -> TreatmentResponse:
        """Generic treatment simulation for unsupported modalities."""
        timepoints = np.linspace(0, horizon_days, int(horizon_days) + 1)
        volumes = np.full_like(timepoints, self.baseline_volume)

        return TreatmentResponse(
            timepoints=timepoints,
            volumes=volumes,
            response_category=ResponseType.STABLE_DISEASE,
            volume_change_percent=0,
            control_probability=0,
            confidence_interval={},
            metrics={},
        )

    def _classify_response(self, volume_change_percent: float) -> ResponseType:
        """Classify response according to RECIST-like criteria."""
        if volume_change_percent <= -100:
            return ResponseType.COMPLETE_RESPONSE
        elif volume_change_percent <= -30:
            return ResponseType.PARTIAL_RESPONSE
        elif volume_change_percent < 20:
            return ResponseType.STABLE_DISEASE
        else:
            return ResponseType.PROGRESSIVE_DISEASE

    def _estimate_uncertainty(self, volumes: np.ndarray, alpha_std: float, beta_std: float) -> dict:
        """Estimate prediction uncertainty."""
        # Simplified uncertainty estimation
        relative_std = 0.15  # 15% relative uncertainty
        return {"lower": volumes * (1 - 1.96 * relative_std), "upper": volumes * (1 + 1.96 * relative_std)}

    def _estimate_radiation_toxicity(self, total_dose: float, fractions: int) -> dict:
        """Estimate radiation toxicity risk."""
        dose_per_fraction = total_dose / fractions

        return {
            "acute_mucositis_risk": min(0.8, total_dose / 70 * 0.7),
            "late_fibrosis_risk": min(0.3, dose_per_fraction / 2.5 * 0.2),
            "fatigue_probability": 0.9 if total_dose > 50 else 0.5,
        }

    def _estimate_chemo_toxicity(self, drug: str, dose_mg_m2: float, cycles: int) -> dict:
        """Estimate chemotherapy toxicity risk."""
        return {"neutropenia_risk": min(0.7, dose_mg_m2 / 150), "nausea_risk": 0.6, "neuropathy_risk": 0.3 * cycles / 6}

    def _estimate_recurrence_probability(self, margin_mm: float, residual_fraction: float) -> float:
        """Estimate recurrence probability after surgery."""
        # Simplified model
        margin_factor = np.exp(-margin_mm / 5)
        residual_factor = residual_fraction * 10
        return min(0.9, margin_factor * 0.3 + residual_factor)

    def compare_treatments(self, treatments: list[dict], metrics: list[str] | None = None) -> "TreatmentComparison":
        """
        Compare multiple treatment strategies.

        Args:
            treatments: List of treatment protocols to compare
            metrics: Metrics to include in comparison

        Returns:
            TreatmentComparison with ranked results
        """
        if metrics is None:
            metrics = ["tumor_control", "volume_reduction", "toxicity"]

        results = []
        for treatment in treatments:
            response = self.predict_response(treatment, horizon_days=90)
            results.append(
                {
                    "name": treatment.get("name", treatment.get("type", "Unknown")),
                    "protocol": treatment,
                    "response": response,
                    "volume_change": response.volume_change_percent,
                    "control_prob": response.control_probability,
                }
            )

        # Rank by volume reduction
        results.sort(key=lambda x: x["volume_change"])

        return TreatmentComparison(treatments=results)

    def monte_carlo_simulation(
        self, treatment: dict, n_samples: int = 1000, parameter_uncertainty: dict | None = None
    ) -> dict:
        """
        Monte Carlo simulation for uncertainty quantification.

        Args:
            treatment: Treatment protocol
            n_samples: Number of Monte Carlo samples
            parameter_uncertainty: Parameter uncertainty specifications

        Returns:
            Dictionary with mean, std, and confidence intervals
        """
        if parameter_uncertainty is None:
            parameter_uncertainty = {"alpha": 0.1, "beta": 0.2}

        results = []

        # Store original parameters
        original_proliferation = self.patient_twin.proliferation_rate

        for _ in range(n_samples):
            # Perturb parameters
            proliferation_sample = original_proliferation * (
                1 + np.random.randn() * parameter_uncertainty.get("proliferation", 0.1)
            )
            self.patient_twin.model.parameters.proliferation_rate = max(0.001, proliferation_sample)

            # Run simulation
            response = self.predict_response(treatment, horizon_days=90, include_uncertainty=False)
            results.append(
                {
                    "final_volume": response.volumes[-1],
                    "volume_change": response.volume_change_percent,
                    "control_prob": response.control_probability,
                }
            )

        # Restore original parameters
        self.patient_twin.model.parameters.proliferation_rate = original_proliferation

        # Compute statistics
        final_volumes = [r["final_volume"] for r in results]
        control_probs = [r["control_prob"] for r in results]

        return {
            "volume_mean": np.mean(final_volumes),
            "volume_std": np.std(final_volumes),
            "volume_ci": np.percentile(final_volumes, [2.5, 97.5]),
            "tcp_mean": np.mean(control_probs),
            "tcp_ci": np.percentile(control_probs, [2.5, 97.5]),
        }


@dataclass
class TreatmentComparison:
    """Treatment comparison results."""

    treatments: list

    def generate_report(self, format: str = "clinical") -> str:
        """Generate comparison report."""
        report_lines = ["Treatment Comparison Report", "=" * 40, ""]

        for i, t in enumerate(self.treatments, 1):
            report_lines.append(f"{i}. {t['name']}")
            report_lines.append(f"   Volume change: {t['volume_change']:.1f}%")
            report_lines.append(f"   Response: {t['response'].response_category.value}")
            report_lines.append("")

        return "\n".join(report_lines)


class LinearQuadraticModel:
    """Linear-Quadratic model for radiation response."""

    # Tissue-specific alpha/beta values (Gy)
    TISSUE_PARAMS = {
        "I": (0.20, 0.02),  # Low grade
        "II": (0.25, 0.025),
        "III": (0.30, 0.03),  # High grade
        "IV": (0.35, 0.035),  # Most aggressive
        "unknown": (0.30, 0.03),
    }

    def get_tissue_parameters(self, tumor_grade: str) -> tuple[float, float]:
        """Get alpha/beta parameters for tumor grade."""
        return self.TISSUE_PARAMS.get(tumor_grade, self.TISSUE_PARAMS["unknown"])

    def surviving_fraction(self, dose: float, alpha: float, beta: float) -> float:
        """Calculate surviving fraction for single dose."""
        return np.exp(-alpha * dose - beta * dose**2)

    def tumor_control_probability(self, initial_cells: float, total_sf: float) -> float:
        """Calculate tumor control probability."""
        surviving_cells = initial_cells * total_sf
        return np.exp(-surviving_cells)

    def biologically_effective_dose(self, total_dose: float, dose_per_fraction: float, alpha_beta: float) -> float:
        """Calculate BED for fractionated treatment."""
        return total_dose * (1 + dose_per_fraction / alpha_beta)


class PharmacokineticModel:
    """Pharmacokinetic/Pharmacodynamic model for chemotherapy."""

    # Drug-specific parameters
    DRUG_PARAMS = {
        "cisplatin": {
            "clearance": 25,  # L/hr
            "vd": 20,  # L
            "t_half": 0.5,  # hours (distribution)
            "ec50": 1.0,  # ug/mL
            "hill": 2,
        },
        "paclitaxel": {"clearance": 15, "vd": 200, "t_half": 5, "ec50": 0.5, "hill": 1.5},
        "generic": {"clearance": 20, "vd": 50, "t_half": 2, "ec50": 1.0, "hill": 1},
    }

    def get_drug_parameters(self, drug: str) -> dict:
        """Get PK parameters for drug."""
        return self.DRUG_PARAMS.get(drug.lower(), self.DRUG_PARAMS["generic"])

    def concentration_profile(self, dose_mg_m2: float, time_hours: float, params: dict) -> float:
        """Calculate drug concentration at time t."""
        k_elim = 0.693 / params["t_half"]
        c0 = dose_mg_m2 / params["vd"]
        return c0 * np.exp(-k_elim * time_hours)

    def cell_kill_rate(self, concentration: float, params: dict) -> float:
        """Calculate cell kill rate using Hill equation."""
        ec50 = params["ec50"]
        hill = params["hill"]
        emax = 0.5  # Maximum kill rate /day

        return emax * concentration**hill / (ec50**hill + concentration**hill)


class ImmunotherapyModel:
    """Immune-tumor dynamics model for immunotherapy."""

    def __init__(self):
        self.activation_rate = 0.1  # T-cell activation rate
        self.kill_rate = 0.05  # Tumor killing rate
        self.exhaustion_rate = 0.01  # T-cell exhaustion rate


# Convenience function
def create_simulator(patient_twin) -> TreatmentSimulator:
    """Create treatment simulator from patient digital twin."""
    return TreatmentSimulator(patient_twin)


if __name__ == "__main__":
    print("Treatment Simulator - Physical AI Oncology Trials")
    print("=" * 50)

    # Create mock patient twin for testing
    class MockTwin:
        patient_id = "TEST-001"
        current_volume_cm3 = 10.0
        proliferation_rate = 0.05

        class clinical_data:
            tumor_grade = "III"

        class model:
            class parameters:
                proliferation_rate = 0.05

    mock_twin = MockTwin()

    # Initialize simulator
    simulator = TreatmentSimulator(mock_twin)

    # Test radiation simulation
    print("\n1. Radiation Therapy Simulation")
    rt_protocol = {"type": "radiation", "total_dose_gy": 60, "fractions": 30}
    rt_response = simulator.predict_response(rt_protocol, horizon_days=90)
    print(f"   Response: {rt_response.response_category.value}")
    print(f"   Volume change: {rt_response.volume_change_percent:.1f}%")
    print(f"   TCP: {rt_response.control_probability:.1%}")

    # Test chemotherapy simulation
    print("\n2. Chemotherapy Simulation")
    chemo_protocol = {"type": "chemotherapy", "drug": "cisplatin", "dose_mg_m2": 75, "cycles": 4}
    chemo_response = simulator.predict_response(chemo_protocol, horizon_days=90)
    print(f"   Response: {chemo_response.response_category.value}")
    print(f"   Volume change: {chemo_response.volume_change_percent:.1f}%")

    # Test treatment comparison
    print("\n3. Treatment Comparison")
    comparison = simulator.compare_treatments(
        [{"name": "Radiation 60Gy", **rt_protocol}, {"name": "Cisplatin", **chemo_protocol}]
    )
    print(comparison.generate_report())
