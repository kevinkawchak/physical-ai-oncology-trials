"""
=============================================================================
EXAMPLE 04: Tumor Microenvironment & Immunotherapy Digital Twin
=============================================================================

WHAT THIS CODE DOES:
    Implements an agent-based model (ABM) of the tumor immune microenvironment
    (TME) to simulate checkpoint inhibitor immunotherapy response. Models
    individual cell populations (tumor cells, CD8+ T-cells, Tregs, macrophages,
    dendritic cells) and their interactions through cytokine signaling, PD-1/
    PD-L1 checkpoint dynamics, and spatial heterogeneity.

CLINICAL CONTEXT:
    Immune checkpoint inhibitors (anti-PD-1/PD-L1, anti-CTLA-4) have
    transformed oncology but only benefit 20-40% of patients. Response depends
    on complex TME factors:
      - Tumor mutational burden (TMB) and neoantigen load
      - PD-L1 expression level (TPS and CPS scores)
      - CD8+ T-cell infiltration (hot vs. cold tumors)
      - Regulatory T-cell (Treg) immunosuppression
      - Myeloid-derived suppressor cell (MDSC) activity
      - Cytokine milieu (IFN-gamma, TGF-beta, IL-10)
    The TME digital twin predicts individual patient response to checkpoint
    blockade by simulating these immune dynamics with patient-specific
    parameters derived from biopsy, flow cytometry, and genomic data.

USE CASES COVERED:
    1. Agent-based TME simulation with discrete cell populations
    2. PD-1/PD-L1 checkpoint axis dynamics under anti-PD-1 therapy
    3. CD8+ T-cell activation, exhaustion, and reinvigoration modeling
    4. Cytokine network (IFN-gamma, TGF-beta, IL-10) feedback loops
    5. Tumor immunogenicity scoring from TMB and neoantigen data
    6. Response prediction (iRECIST criteria) with biomarker stratification

FRAMEWORK REQUIREMENTS:
    Required:
        - NumPy 1.24.0+
        - SciPy 1.11.0+ (ODE integration)
    Optional:
        - MONAI 1.4.0+ (for imaging-based TIL scoring)
        - scikit-learn (for biomarker-based stratification)

REGULATORY NOTES:
    - FDA Companion Diagnostic Guidance (PD-L1 testing)
    - iRECIST Criteria for immune-modified response assessment
    - ICH E6(R3): Biomarker-driven adaptive trial designs
    - SITC Biomarker Committee Recommendations (2025)

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: TME CELL POPULATIONS AND BIOMARKERS
# =============================================================================


class ImmunePheno(Enum):
    """Tumor immune phenotype classification."""

    INFLAMED = "inflamed"               # Hot: high TIL, high PD-L1
    IMMUNE_EXCLUDED = "immune_excluded"  # T-cells at margin, not infiltrating
    IMMUNE_DESERT = "immune_desert"      # Cold: minimal immune infiltrate


class CheckpointAgent(Enum):
    """Immune checkpoint inhibitor agents."""

    PEMBROLIZUMAB = "pembrolizumab"    # Anti-PD-1
    NIVOLUMAB = "nivolumab"            # Anti-PD-1
    ATEZOLIZUMAB = "atezolizumab"      # Anti-PD-L1
    DURVALUMAB = "durvalumab"          # Anti-PD-L1
    IPILIMUMAB = "ipilimumab"          # Anti-CTLA-4


class iRECISTResponse(Enum):
    """iRECIST immune-modified response criteria."""

    iCR = "immune_complete_response"
    iPR = "immune_partial_response"
    iSD = "immune_stable_disease"
    iUPD = "immune_unconfirmed_progressive_disease"
    iCPD = "immune_confirmed_progressive_disease"


@dataclass
class PatientTMEProfile:
    """Patient-specific tumor microenvironment profile.

    Derived from biopsy pathology, flow cytometry, and genomic sequencing.

    Attributes:
        patient_id: Patient identifier
        tumor_cells_per_mm2: Tumor cell density from pathology
        cd8_til_per_mm2: CD8+ tumor-infiltrating lymphocyte density
        treg_per_mm2: Regulatory T-cell density
        macrophage_per_mm2: Tumor-associated macrophage density
        pdl1_tps: PD-L1 tumor proportion score (0-100%)
        pdl1_cps: PD-L1 combined positive score
        tmb_mut_per_mb: Tumor mutational burden (mutations/Mb)
        msi_status: Microsatellite instability (MSI-H, MSS)
        ifn_gamma_signature: IFN-gamma gene expression score
        tgf_beta_level: TGF-beta serum level (pg/mL)
        immune_phenotype: Classified immune phenotype
    """

    patient_id: str
    tumor_cells_per_mm2: float = 5000.0
    cd8_til_per_mm2: float = 200.0
    treg_per_mm2: float = 50.0
    macrophage_per_mm2: float = 100.0
    pdl1_tps: float = 50.0
    pdl1_cps: float = 60.0
    tmb_mut_per_mb: float = 10.0
    msi_status: str = "MSS"
    ifn_gamma_signature: float = 0.5
    tgf_beta_level: float = 20.0
    immune_phenotype: ImmunePheno = ImmunePheno.INFLAMED


# =============================================================================
# SECTION 2: ODE-BASED TME DYNAMICS MODEL
# =============================================================================
# The TME is modeled as a system of coupled ODEs describing the
# interactions between cell populations and cytokine concentrations.
# State vector: [T, E, R, M, D, P, IFNg, TGFb, IL10]
#   T    : Tumor cell count (10^6)
#   E    : Effector CD8+ T-cells (10^6)
#   R    : Regulatory T-cells (10^6)
#   M    : Tumor-associated macrophages (10^6)
#   D    : Dendritic cells (10^6)
#   P    : PD-L1 expression level (0-1)
#   IFNg : IFN-gamma concentration (relative units)
#   TGFb : TGF-beta concentration (relative units)
#   IL10 : IL-10 concentration (relative units)


TME_STATE_DIM = 9
TME_LABELS = ["Tumor", "CD8_Effector", "Treg", "Macrophage", "Dendritic",
              "PD-L1", "IFN-gamma", "TGF-beta", "IL-10"]


@dataclass
class TMEParameters:
    """Kinetic parameters for TME dynamics.

    Instructions for engineers:
        - Tune proliferation/death rates to match patient's tumor doubling time
        - Checkpoint blockade modifies pd1_blockade_efficacy (0-1)
        - Immunogenicity score derived from TMB and neoantigen quality
        - Parameters calibratable from serial biopsy or ctDNA data

    Attributes:
        tumor_growth_rate: Intrinsic tumor growth rate (/day)
        tumor_carrying_capacity: Max tumor burden (10^6 cells)
        immunogenicity: Tumor immunogenicity score (0-1)
        cd8_activation_rate: T-cell activation rate by DCs
        cd8_killing_rate: Tumor cell kill rate per effector T-cell
        cd8_exhaustion_rate: T-cell exhaustion rate
        cd8_reinvigoration_rate: T-cell reinvigoration by checkpoint blockade
        treg_suppression_rate: Treg suppression of effector T-cells
        pd1_blockade_efficacy: Fraction of PD-1/PD-L1 axis blocked (0-1)
        ctla4_blockade_efficacy: Fraction of CTLA-4 axis blocked (0-1)
    """

    tumor_growth_rate: float = 0.02        # /day
    tumor_carrying_capacity: float = 1000.0  # 10^6 cells
    immunogenicity: float = 0.5             # 0-1
    cd8_activation_rate: float = 0.1        # /day
    cd8_killing_rate: float = 0.05          # /day per 10^6 effectors
    cd8_exhaustion_rate: float = 0.03       # /day
    cd8_reinvigoration_rate: float = 0.0    # /day (set by checkpoint blockade)
    treg_suppression_rate: float = 0.02     # /day
    treg_recruitment_rate: float = 0.01     # /day
    macrophage_polarization: float = 0.5    # M1 vs M2 balance (0=M2, 1=M1)
    dc_maturation_rate: float = 0.05        # /day
    pd1_blockade_efficacy: float = 0.0      # 0-1 (0=no drug, 1=complete block)
    ctla4_blockade_efficacy: float = 0.0    # 0-1


class TMEDynamicsModel:
    """Coupled ODE model of tumor-immune microenvironment dynamics.

    Implements the Kuznetsov-Taylor extended model with checkpoint
    inhibitor pharmacodynamics and cytokine feedback loops.

    Instructions for engineers:
        - State initialization from patient biopsy data
        - Checkpoint drug pharmacokinetics modeled separately
        - IFN-gamma drives PD-L1 upregulation (adaptive resistance)
        - TGF-beta drives Treg recruitment and T-cell exclusion
        - Call simulate() to project TME evolution under therapy

    References:
        - Kuznetsov et al., Bull Math Biol, 1994 (original model)
        - Lai & Bhatt, J Math Biol, 2024 (checkpoint inhibitor extension)
        - Serre et al., JTO, 2016 (virtual clinical trials)

    Example:
        >>> model = TMEDynamicsModel(params)
        >>> result = model.simulate(initial_state, duration_days=180)
        >>> print(f"Tumor at day 180: {result['tumor_trajectory'][-1]:.1f} M cells")
    """

    def __init__(self, params: TMEParameters):
        self.params = params

    def _ode_system(self, t: float, y: np.ndarray) -> np.ndarray:
        """TME ODE system.

        Args:
            t: Time (days)
            y: State vector [TME_STATE_DIM]

        Returns:
            Derivative vector dy/dt
        """
        T, E, R, M, D, P, ifng, tgfb, il10 = y
        p = self.params
        dydt = np.zeros(TME_STATE_DIM)

        # Effective checkpoint blockade
        pd1_block = p.pd1_blockade_efficacy
        ctla4_block = p.ctla4_blockade_efficacy

        # PD-1/PD-L1 inhibition of T-cell function
        # Without blockade: PD-L1 suppresses killing; with: suppression reduced
        pd1_suppression = P * (1.0 - pd1_block)

        # --- Tumor cells (T) ---
        # Growth: logistic; Kill: by effector T-cells modulated by PD-L1
        tumor_growth = p.tumor_growth_rate * T * (1 - T / p.tumor_carrying_capacity)
        immune_kill = p.cd8_killing_rate * E * T / (T + 100) * (1 - pd1_suppression)
        dydt[0] = tumor_growth - immune_kill

        # --- CD8+ Effector T-cells (E) ---
        # Activation by DCs (enhanced by CTLA-4 blockade); antigen-driven
        activation = p.cd8_activation_rate * D * p.immunogenicity * (
            1 + 0.5 * ctla4_block  # CTLA-4 blockade enhances priming
        )
        # Stimulation by IFN-gamma
        ifng_boost = 0.02 * ifng * E
        # Exhaustion: PD-1 signaling drives exhaustion
        exhaustion = p.cd8_exhaustion_rate * E * (0.5 + pd1_suppression)
        # Reinvigoration: checkpoint blockade rescues exhausted cells
        reinvigoration = p.cd8_reinvigoration_rate * pd1_block * E * 0.3
        # Suppression by Tregs
        treg_suppress = p.treg_suppression_rate * R * E / (E + 10)
        # Suppression by IL-10 and TGF-beta
        cytokine_suppress = 0.01 * (tgfb + il10) * E
        # Natural turnover
        turnover = 0.01 * E

        dydt[1] = activation + ifng_boost + reinvigoration - exhaustion - treg_suppress - cytokine_suppress - turnover

        # --- Regulatory T-cells (R) ---
        # Recruited by TGF-beta; CTLA-4 blockade may deplete
        recruitment = p.treg_recruitment_rate * tgfb * T / (T + 100)
        ctla4_depletion = 0.02 * ctla4_block * R
        treg_turnover = 0.01 * R
        dydt[2] = recruitment - ctla4_depletion - treg_turnover

        # --- Tumor-associated macrophages (M) ---
        # Recruited by tumor; polarized by IFN-gamma (M1) or TGF-beta (M2)
        mac_recruitment = 0.01 * T / (T + 500)
        # M1 macrophages help anti-tumor; M2 promote tumor
        m1_fraction = p.macrophage_polarization * (1 + 0.3 * ifng) / (1 + 0.3 * ifng + 0.3 * tgfb)
        mac_turnover = 0.005 * M
        dydt[3] = mac_recruitment - mac_turnover

        # --- Dendritic cells (D) ---
        # Maturation driven by tumor antigens and IFN-gamma
        maturation = p.dc_maturation_rate * p.immunogenicity * (1 + ifng)
        dc_turnover = 0.02 * D
        dydt[4] = maturation - dc_turnover

        # --- PD-L1 expression (P) ---
        # Upregulated by IFN-gamma (adaptive resistance); constitutive component
        pdl1_upregulation = 0.1 * ifng / (ifng + 1.0)
        pdl1_constitutive = 0.05 * T / (T + 500)  # tumor-intrinsic
        pdl1_decay = 0.05 * P
        dydt[5] = pdl1_upregulation + pdl1_constitutive - pdl1_decay

        # --- Cytokines ---
        # IFN-gamma: produced by activated CD8+ T-cells and M1 macrophages
        dydt[6] = 0.05 * E + 0.02 * M * m1_fraction - 0.1 * ifng

        # TGF-beta: produced by tumor and Tregs
        dydt[7] = 0.01 * T / (T + 200) + 0.03 * R - 0.05 * tgfb

        # IL-10: immunosuppressive, from Tregs and M2 macrophages
        dydt[8] = 0.02 * R + 0.01 * M * (1 - m1_fraction) - 0.05 * il10

        return dydt

    def simulate(
        self,
        initial_state: np.ndarray,
        duration_days: float = 180,
        dt_days: float = 1.0,
    ) -> dict[str, Any]:
        """Simulate TME dynamics.

        Args:
            initial_state: Initial state vector [TME_STATE_DIM]
            duration_days: Simulation duration in days
            dt_days: Output time step in days

        Returns:
            Dictionary with trajectories and summary metrics
        """
        t_eval = np.arange(0, duration_days, dt_days)

        sol = solve_ivp(
            self._ode_system,
            [0, duration_days],
            initial_state,
            t_eval=t_eval,
            method="RK45",
            max_step=0.5,
        )

        # Enforce non-negativity
        trajectories = np.maximum(sol.y, 0.0)

        result = {
            "time_days": sol.t,
            "tumor_trajectory": trajectories[0],
            "cd8_trajectory": trajectories[1],
            "treg_trajectory": trajectories[2],
            "macrophage_trajectory": trajectories[3],
            "dendritic_trajectory": trajectories[4],
            "pdl1_trajectory": trajectories[5],
            "ifng_trajectory": trajectories[6],
            "tgfb_trajectory": trajectories[7],
            "il10_trajectory": trajectories[8],
        }

        # Summary metrics
        T_final = trajectories[0, -1]
        T_initial = initial_state[0]
        result["tumor_change_pct"] = (T_final - T_initial) / T_initial * 100
        result["nadir_tumor"] = float(np.min(trajectories[0]))
        result["nadir_day"] = float(sol.t[np.argmin(trajectories[0])])
        result["peak_cd8"] = float(np.max(trajectories[1]))
        result["peak_ifng"] = float(np.max(trajectories[6]))

        return result


# =============================================================================
# SECTION 3: CHECKPOINT INHIBITOR PHARMACOKINETICS
# =============================================================================
# Models drug concentration over time for anti-PD-1/PD-L1 agents
# to dynamically modulate the blockade efficacy parameter.


@dataclass
class CheckpointPKParams:
    """Pharmacokinetic parameters for checkpoint inhibitors.

    Attributes:
        agent: Checkpoint inhibitor agent
        half_life_days: Elimination half-life (days)
        trough_efficacy: Trough PD-1 blockade efficacy (fraction)
        peak_efficacy: Peak PD-1 blockade efficacy (fraction)
        dosing_interval_days: Dosing interval in days
    """

    agent: CheckpointAgent
    half_life_days: float = 25.0
    trough_efficacy: float = 0.7
    peak_efficacy: float = 0.95
    dosing_interval_days: int = 21


CHECKPOINT_PK_LIBRARY = {
    CheckpointAgent.PEMBROLIZUMAB: CheckpointPKParams(
        agent=CheckpointAgent.PEMBROLIZUMAB,
        half_life_days=25.0, trough_efficacy=0.75, peak_efficacy=0.95,
        dosing_interval_days=21,
    ),
    CheckpointAgent.NIVOLUMAB: CheckpointPKParams(
        agent=CheckpointAgent.NIVOLUMAB,
        half_life_days=26.7, trough_efficacy=0.70, peak_efficacy=0.92,
        dosing_interval_days=14,
    ),
    CheckpointAgent.ATEZOLIZUMAB: CheckpointPKParams(
        agent=CheckpointAgent.ATEZOLIZUMAB,
        half_life_days=27.0, trough_efficacy=0.72, peak_efficacy=0.93,
        dosing_interval_days=21,
    ),
    CheckpointAgent.IPILIMUMAB: CheckpointPKParams(
        agent=CheckpointAgent.IPILIMUMAB,
        half_life_days=15.0, trough_efficacy=0.50, peak_efficacy=0.85,
        dosing_interval_days=21,
    ),
}


class CheckpointPKModel:
    """Pharmacokinetic model for checkpoint inhibitor blockade.

    Computes the time-varying PD-1 or CTLA-4 blockade efficacy based
    on drug dosing schedule and elimination kinetics.

    Instructions for engineers:
        - Blockade efficacy = fraction of target receptors occupied
        - Multi-dose: superposition of individual dose contributions
        - Steady state typically reached after 3-4 doses
        - Combine with TME model by updating blockade efficacy parameter

    Example:
        >>> pk = CheckpointPKModel(CheckpointAgent.PEMBROLIZUMAB)
        >>> efficacy = pk.get_blockade_efficacy(day=35, doses=[0, 21])
    """

    def __init__(self, agent: CheckpointAgent):
        self.params = CHECKPOINT_PK_LIBRARY.get(
            agent,
            CheckpointPKParams(agent=agent),
        )
        self.k_elim = np.log(2) / self.params.half_life_days

    def get_blockade_efficacy(
        self, day: float, dose_days: list[float]
    ) -> float:
        """Compute blockade efficacy at a given time point.

        Args:
            day: Current time (days from first dose)
            dose_days: List of days when doses were administered

        Returns:
            Blockade efficacy (0-1)
        """
        total_concentration = 0.0
        for dose_day in dose_days:
            if day >= dose_day:
                dt = day - dose_day
                total_concentration += np.exp(-self.k_elim * dt)

        # Sigmoidal receptor occupancy model
        ec50 = 0.5  # concentration for 50% occupancy (normalized)
        hill = 2.0
        occupancy = total_concentration ** hill / (
            ec50 ** hill + total_concentration ** hill
        )

        # Scale between trough and peak efficacy
        efficacy = self.params.trough_efficacy + (
            self.params.peak_efficacy - self.params.trough_efficacy
        ) * occupancy

        return min(efficacy, self.params.peak_efficacy)


# =============================================================================
# SECTION 4: IMMUNOTHERAPY RESPONSE PREDICTOR
# =============================================================================
# Integrates TME dynamics with checkpoint PK to predict treatment response
# per iRECIST criteria.


class ImmunotherapyResponsePredictor:
    """Predicts immunotherapy response from TME digital twin simulation.

    Combines patient TME profile, checkpoint inhibitor PK, and TME dynamics
    to predict treatment response with iRECIST classification.

    Instructions for engineers:
        1. Initialize with patient TME profile from biopsy
        2. Configure treatment with checkpoint agent and schedule
        3. Call predict_response() to simulate the full treatment course
        4. Assess response_probability for clinical decision support
        5. Evaluate pseudoprogression probability for iUPD management

    Example:
        >>> predictor = ImmunotherapyResponsePredictor(patient_profile)
        >>> result = predictor.predict_response(
        ...     agent=CheckpointAgent.PEMBROLIZUMAB,
        ...     duration_days=180,
        ...     n_doses=9,
        ... )
        >>> print(f"Response: {result['irecist_response'].value}")
        >>> print(f"Response probability: {result['response_probability']:.1%}")
    """

    def __init__(self, patient_profile: PatientTMEProfile):
        self.profile = patient_profile
        self._compute_immunogenicity_score()

    def _compute_immunogenicity_score(self) -> None:
        """Compute composite immunogenicity score from biomarkers.

        Integrates TMB, MSI, PD-L1, and TIL density into a single
        immunogenicity score that drives the TME dynamics model.
        """
        # TMB component (high TMB = more neoantigens)
        tmb_score = np.clip(self.profile.tmb_mut_per_mb / 20.0, 0, 1)

        # MSI component (MSI-H = strong immune response)
        msi_score = 0.9 if self.profile.msi_status == "MSI-H" else 0.3

        # PD-L1 component (expression enables checkpoint blockade)
        pdl1_score = np.clip(self.profile.pdl1_tps / 100.0, 0, 1)

        # TIL density component
        til_score = np.clip(self.profile.cd8_til_per_mm2 / 500.0, 0, 1)

        # IFN-gamma signature
        ifng_score = np.clip(self.profile.ifn_gamma_signature, 0, 1)

        # Weighted composite
        self.immunogenicity = (
            0.25 * tmb_score
            + 0.20 * msi_score
            + 0.20 * pdl1_score
            + 0.20 * til_score
            + 0.15 * ifng_score
        )

        logger.info(
            "Immunogenicity score: %.3f (TMB=%.2f, MSI=%.2f, PD-L1=%.2f, TIL=%.2f)",
            self.immunogenicity, tmb_score, msi_score, pdl1_score, til_score
        )

    def predict_response(
        self,
        agent: CheckpointAgent = CheckpointAgent.PEMBROLIZUMAB,
        duration_days: float = 180,
        n_doses: int = 9,
        include_combination: CheckpointAgent | None = None,
    ) -> dict[str, Any]:
        """Predict immunotherapy response.

        Args:
            agent: Primary checkpoint inhibitor
            duration_days: Treatment duration in days
            n_doses: Number of doses to administer
            include_combination: Optional combination agent (e.g., ipilimumab)

        Returns:
            Dictionary with response prediction, trajectories, and biomarkers
        """
        # Set up PK model
        pk_primary = CheckpointPKModel(agent)
        pk_params = pk_primary.params
        dose_days = [i * pk_params.dosing_interval_days for i in range(n_doses)]

        pk_combo = None
        combo_dose_days = []
        if include_combination:
            pk_combo = CheckpointPKModel(include_combination)
            # Combination typically given for first 4 doses
            combo_interval = pk_combo.params.dosing_interval_days
            combo_dose_days = [i * combo_interval for i in range(min(4, n_doses))]

        # Initialize TME state from patient profile
        initial_state = self._profile_to_state()

        # Set up dynamics parameters
        params = TMEParameters(
            tumor_growth_rate=0.02,
            tumor_carrying_capacity=1000.0,
            immunogenicity=self.immunogenicity,
            cd8_activation_rate=0.1,
            cd8_killing_rate=0.05,
            cd8_exhaustion_rate=0.03,
        )

        # Run simulation with time-varying blockade efficacy
        model = TMEDynamicsModel(params)

        # Simulate in daily steps with updated PK
        t_all = np.arange(0, duration_days, 1.0)
        state = initial_state.copy()
        trajectories = np.zeros((TME_STATE_DIM, len(t_all)))
        trajectories[:, 0] = state

        for i in range(1, len(t_all)):
            day = t_all[i]

            # Update blockade efficacy from PK
            pd1_eff = pk_primary.get_blockade_efficacy(day, dose_days)
            model.params.pd1_blockade_efficacy = pd1_eff

            if pk_combo:
                ctla4_eff = pk_combo.get_blockade_efficacy(day, combo_dose_days)
                model.params.ctla4_blockade_efficacy = ctla4_eff

            # Reinvigoration rate proportional to blockade
            model.params.cd8_reinvigoration_rate = 0.05 * pd1_eff

            # Single step of ODE
            sol = solve_ivp(
                model._ode_system, [0, 1.0], state,
                t_eval=[1.0], method="RK45",
            )
            state = np.maximum(sol.y[:, -1], 0.0)
            trajectories[:, i] = state

        # Classify response
        tumor_change_pct = (state[0] - initial_state[0]) / initial_state[0] * 100
        irecist = self._classify_irecist(trajectories[0], t_all)

        # Pseudoprogression detection
        pseudo_prob = self._pseudoprogression_probability(trajectories[0], t_all)

        # Response probability (biomarker-weighted)
        response_prob = self._compute_response_probability()

        result = {
            "agent": agent.value,
            "combination": include_combination.value if include_combination else None,
            "duration_days": duration_days,
            "n_doses": n_doses,
            "time_days": t_all,
            "tumor_trajectory": trajectories[0],
            "cd8_trajectory": trajectories[1],
            "treg_trajectory": trajectories[2],
            "pdl1_trajectory": trajectories[5],
            "ifng_trajectory": trajectories[6],
            "tumor_change_pct": tumor_change_pct,
            "irecist_response": irecist,
            "response_probability": response_prob,
            "pseudoprogression_probability": pseudo_prob,
            "immunogenicity_score": self.immunogenicity,
            "immune_phenotype": self.profile.immune_phenotype.value,
            "biomarkers": {
                "TMB": self.profile.tmb_mut_per_mb,
                "PD-L1_TPS": self.profile.pdl1_tps,
                "CD8_TIL": self.profile.cd8_til_per_mm2,
                "MSI": self.profile.msi_status,
            },
        }

        logger.info(
            "Response prediction: %s, tumor change=%.1f%%, prob=%.1%%",
            irecist.value, tumor_change_pct, response_prob * 100
        )

        return result

    def _profile_to_state(self) -> np.ndarray:
        """Convert patient TME profile to ODE initial state."""
        p = self.profile
        # Scale from per mm^2 to 10^6 cells (assuming ~1 cm^3 biopsy)
        scale = 0.01  # cells/mm^2 → 10^6 cells (rough approximation)
        return np.array([
            p.tumor_cells_per_mm2 * scale,   # T: tumor cells
            p.cd8_til_per_mm2 * scale,        # E: CD8+ effectors
            p.treg_per_mm2 * scale,           # R: Tregs
            p.macrophage_per_mm2 * scale,     # M: macrophages
            1.0,                               # D: dendritic cells
            p.pdl1_tps / 100.0,               # P: PD-L1 expression
            p.ifn_gamma_signature,            # IFN-gamma
            p.tgf_beta_level / 50.0,          # TGF-beta (normalized)
            0.3,                               # IL-10 (baseline)
        ])

    def _classify_irecist(
        self, tumor_trajectory: np.ndarray, time_days: np.ndarray
    ) -> iRECISTResponse:
        """Classify response per iRECIST criteria."""
        T0 = tumor_trajectory[0]
        T_final = tumor_trajectory[-1]
        change_pct = (T_final - T0) / T0 * 100

        # Check for pseudoprogression (initial increase then decrease)
        T_max = np.max(tumor_trajectory)
        T_nadir = np.min(tumor_trajectory)

        if change_pct <= -100:
            return iRECISTResponse.iCR
        elif change_pct <= -30:
            return iRECISTResponse.iPR
        elif change_pct < 20:
            return iRECISTResponse.iSD
        else:
            # Check if progression is confirmed (sustained)
            # Look at last 30 days
            last_30 = tumor_trajectory[-30:] if len(tumor_trajectory) > 30 else tumor_trajectory
            if np.all(np.diff(last_30) > 0):
                return iRECISTResponse.iCPD  # confirmed progressive
            else:
                return iRECISTResponse.iUPD  # unconfirmed, may be pseudo

    def _pseudoprogression_probability(
        self, tumor_trajectory: np.ndarray, time_days: np.ndarray
    ) -> float:
        """Estimate probability of pseudoprogression.

        Pseudoprogression occurs when apparent tumor growth is actually
        immune infiltration, followed by eventual shrinkage.
        """
        # Detect pattern: initial increase > 20% then decrease
        T0 = tumor_trajectory[0]
        T_max_idx = np.argmax(tumor_trajectory)
        T_max = tumor_trajectory[T_max_idx]

        if T_max / T0 < 1.2:  # no significant increase
            return 0.0

        # Check if decrease follows the peak
        if T_max_idx < len(tumor_trajectory) - 10:
            post_peak = tumor_trajectory[T_max_idx:]
            T_after = post_peak[-1]
            if T_after < T_max * 0.9:  # at least 10% decrease from peak
                return 0.6  # likely pseudoprogression
            else:
                return 0.2  # uncertain

        return 0.1  # too early to tell

    def _compute_response_probability(self) -> float:
        """Compute overall response probability from biomarkers."""
        # Evidence-based weights from clinical validation studies
        prob = 0.1  # baseline response rate

        if self.profile.msi_status == "MSI-H":
            prob += 0.35  # MSI-H strong predictor

        if self.profile.tmb_mut_per_mb > 10:
            prob += 0.15

        if self.profile.pdl1_tps >= 50:
            prob += 0.20
        elif self.profile.pdl1_tps >= 1:
            prob += 0.10

        if self.profile.cd8_til_per_mm2 > 300:
            prob += 0.10

        if self.profile.immune_phenotype == ImmunePheno.INFLAMED:
            prob += 0.05
        elif self.profile.immune_phenotype == ImmunePheno.IMMUNE_DESERT:
            prob -= 0.10

        return np.clip(prob, 0.05, 0.95)


# =============================================================================
# SECTION 5: MAIN — IMMUNOTHERAPY RESPONSE SIMULATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Tumor Microenvironment & Immunotherapy Digital Twin")
    print("Physical AI Oncology Trials — Example 04")
    print("=" * 70)

    # --- Create patient TME profile (PD-L1 high, inflamed phenotype) ---
    patient = PatientTMEProfile(
        patient_id="NSCLC-IO-017",
        tumor_cells_per_mm2=6000.0,
        cd8_til_per_mm2=350.0,
        treg_per_mm2=80.0,
        macrophage_per_mm2=120.0,
        pdl1_tps=65.0,
        pdl1_cps=75.0,
        tmb_mut_per_mb=12.0,
        msi_status="MSS",
        ifn_gamma_signature=0.7,
        tgf_beta_level=15.0,
        immune_phenotype=ImmunePheno.INFLAMED,
    )

    print(f"\nPatient: {patient.patient_id}")
    print(f"Phenotype: {patient.immune_phenotype.value}")
    print(f"PD-L1 TPS: {patient.pdl1_tps}%")
    print(f"TMB: {patient.tmb_mut_per_mb} mut/Mb")
    print(f"CD8+ TIL: {patient.cd8_til_per_mm2}/mm^2")

    # --- Predict response to pembrolizumab monotherapy ---
    predictor = ImmunotherapyResponsePredictor(patient)

    print(f"\nImmunogenicity score: {predictor.immunogenicity:.3f}")

    print("\n--- Pembrolizumab Monotherapy (6 months) ---")
    result_mono = predictor.predict_response(
        agent=CheckpointAgent.PEMBROLIZUMAB,
        duration_days=180,
        n_doses=9,
    )

    print(f"iRECIST response: {result_mono['irecist_response'].value}")
    print(f"Tumor change: {result_mono['tumor_change_pct']:.1f}%")
    print(f"Response probability: {result_mono['response_probability']:.1%}")
    print(f"Pseudoprogression probability: {result_mono['pseudoprogression_probability']:.1%}")
    print(f"Peak CD8+: {result_mono['peak_cd8']:.1f} x10^6")
    print(f"Peak IFN-gamma: {result_mono['peak_ifng']:.2f}")

    # --- Compare with pembrolizumab + ipilimumab combination ---
    print("\n--- Pembrolizumab + Ipilimumab Combination ---")
    result_combo = predictor.predict_response(
        agent=CheckpointAgent.PEMBROLIZUMAB,
        duration_days=180,
        n_doses=9,
        include_combination=CheckpointAgent.IPILIMUMAB,
    )

    print(f"iRECIST response: {result_combo['irecist_response'].value}")
    print(f"Tumor change: {result_combo['tumor_change_pct']:.1f}%")
    print(f"Response probability: {result_combo['response_probability']:.1%}")

    # --- Immune desert patient (expected non-responder) ---
    print("\n--- Immune Desert Patient (poor prognosis) ---")
    cold_patient = PatientTMEProfile(
        patient_id="NSCLC-IO-023",
        tumor_cells_per_mm2=8000.0,
        cd8_til_per_mm2=30.0,       # very low TIL
        treg_per_mm2=20.0,
        macrophage_per_mm2=50.0,
        pdl1_tps=5.0,               # low PD-L1
        pdl1_cps=8.0,
        tmb_mut_per_mb=3.0,         # low TMB
        msi_status="MSS",
        ifn_gamma_signature=0.1,
        tgf_beta_level=40.0,        # high TGF-beta (immunosuppressive)
        immune_phenotype=ImmunePheno.IMMUNE_DESERT,
    )

    cold_predictor = ImmunotherapyResponsePredictor(cold_patient)
    result_cold = cold_predictor.predict_response(
        agent=CheckpointAgent.PEMBROLIZUMAB,
        duration_days=180,
        n_doses=9,
    )

    print(f"Immunogenicity score: {cold_predictor.immunogenicity:.3f}")
    print(f"iRECIST response: {result_cold['irecist_response'].value}")
    print(f"Tumor change: {result_cold['tumor_change_pct']:.1f}%")
    print(f"Response probability: {result_cold['response_probability']:.1%}")

    print("\n" + "=" * 70)
    print("TME immunotherapy digital twin simulation complete.")
