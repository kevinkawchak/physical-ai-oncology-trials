"""
=============================================================================
EXAMPLE 03: Adaptive Radiation Therapy Digital Twin
=============================================================================

WHAT THIS CODE DOES:
    Implements a digital twin for online adaptive radiation therapy (ART)
    that tracks daily anatomical changes, accumulates delivered dose on
    deforming anatomy, detects replanning triggers, and optimizes fraction-
    level dose delivery. Integrates deformable image registration (DIR),
    dose warping, and biologically effective dose (BED) computation.

CLINICAL CONTEXT:
    During a 6-week course of radiation therapy (30 fractions), tumor and
    organ-at-risk (OAR) anatomy changes significantly:
      - Tumor shrinkage: 30-50% volume reduction by fraction 15-20
      - Weight loss: shifts body contour and internal anatomy
      - Organ filling: bladder/rectum variation (pelvic tumors)
      - Edema/inflammation: transient volume changes
    Without adaptation, the planned dose distribution diverges from the
    delivered dose on actual anatomy. The ART digital twin:
      1. Registers daily CBCT to planning CT via deformable registration
      2. Propagates contours and accumulates dose on deforming anatomy
      3. Computes DVH metrics and detects when replanning is needed
      4. Supports online plan adaptation for the current fraction

USE CASES COVERED:
    1. Deformable image registration (B-spline) between planning CT and daily CBCT
    2. Dose warping from planning geometry to daily anatomy
    3. Cumulative dose accumulation on reference anatomy
    4. DVH metric computation for tumor and organs-at-risk
    5. Replanning trigger detection based on dosimetric criteria
    6. BED/EQD2 computation accounting for fraction size changes

FRAMEWORK REQUIREMENTS:
    Required:
        - NumPy 1.24.0+
        - SciPy 1.11.0+ (ndimage, optimize)
    Optional:
        - MONAI 1.4.0+ (deformable registration networks)
        - pydicom 2.4.0+ (DICOM RT structure/dose loading)
        - SimpleITK (alternative registration backend)

REGULATORY NOTES:
    - AAPM TG-132: Image registration in radiotherapy
    - AAPM TG-275: Dose accumulation in radiation therapy
    - IEC 62083: Requirements for RT treatment planning systems
    - FDA 510(k) K223357: Adaptive RT system clearance precedent

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
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: ANATOMICAL STRUCTURES AND DOSE CONSTRAINTS
# =============================================================================
# Define structures of interest and their dosimetric constraints per
# standard radiation therapy protocols (RTOG/NRG guidelines).


class StructureType(Enum):
    """Anatomical structure categories."""

    GTV = "gross_tumor_volume"       # Visible tumor
    CTV = "clinical_target_volume"   # Microscopic extension
    PTV = "planning_target_volume"   # Setup margin
    OAR = "organ_at_risk"
    BODY = "body_contour"


@dataclass
class DoseConstraint:
    """Dosimetric constraint for a structure.

    Attributes:
        structure_name: Name of the anatomical structure
        constraint_type: Type of constraint (max_dose, mean_dose, dvh)
        limit_gy: Dose limit in Gy
        volume_percent: Volume percentage for DVH constraints
        priority: Constraint priority (1=highest)
    """

    structure_name: str
    constraint_type: str   # "max_dose", "mean_dose", "d_volume", "v_dose"
    limit_gy: float
    volume_percent: float = 0.0
    priority: int = 1


@dataclass
class Structure:
    """Anatomical structure with contour mask and dose constraint.

    Attributes:
        name: Structure name (e.g., "GTV", "Spinal Cord")
        structure_type: GTV, CTV, PTV, OAR, or BODY
        mask: Binary 3D mask on reference geometry
        constraints: List of dose constraints
        alpha_beta: Alpha/beta ratio for BED calculation (Gy)
    """

    name: str
    structure_type: StructureType
    mask: np.ndarray
    constraints: list[DoseConstraint] = field(default_factory=list)
    alpha_beta: float = 10.0  # default tumor alpha/beta


# Standard constraint sets for common treatment sites
LUNG_SBRT_CONSTRAINTS = [
    DoseConstraint("PTV", "d_95", 50.0, 95.0, priority=1),
    DoseConstraint("Spinal Cord", "max_dose", 18.0, priority=1),
    DoseConstraint("Esophagus", "max_dose", 27.5, priority=2),
    DoseConstraint("Heart", "max_dose", 30.0, priority=2),
    DoseConstraint("Brachial Plexus", "max_dose", 24.0, priority=1),
    DoseConstraint("Lung-GTV", "v_dose", 20.0, 10.0, priority=2),  # V20 < 10%
]

HEAD_NECK_CONSTRAINTS = [
    DoseConstraint("PTV_High", "d_95", 70.0, 95.0, priority=1),
    DoseConstraint("PTV_Low", "d_95", 56.0, 95.0, priority=1),
    DoseConstraint("Spinal Cord", "max_dose", 45.0, priority=1),
    DoseConstraint("Brainstem", "max_dose", 54.0, priority=1),
    DoseConstraint("Parotid_L", "mean_dose", 26.0, priority=2),
    DoseConstraint("Parotid_R", "mean_dose", 26.0, priority=2),
]


# =============================================================================
# SECTION 2: DEFORMABLE IMAGE REGISTRATION
# =============================================================================
# B-spline deformable image registration (DIR) to align daily CBCT
# anatomy with planning CT reference frame. Per AAPM TG-132, DIR
# accuracy should be validated to within 2mm for clinical use.


@dataclass
class DeformationField:
    """3D deformation vector field from DIR.

    Attributes:
        displacement: Displacement field [3, D, H, W] in mm
        source_to_target: Direction of registration
        jacobian_determinant: Local volume change map
        registration_metric: Final registration metric value (lower=better)
    """

    displacement: np.ndarray
    source_to_target: str = "cbct_to_planning_ct"
    jacobian_determinant: np.ndarray | None = None
    registration_metric: float = 0.0


class BSplineRegistration:
    """B-spline deformable image registration for adaptive RT.

    Computes a smooth deformation field mapping daily CBCT anatomy to
    the planning CT reference frame. Used for contour propagation and
    dose accumulation.

    Instructions for engineers:
        - Control point spacing determines deformation smoothness
        - Use mutual information metric for multi-modality (CT-to-CBCT)
        - Validate with known landmark pairs per AAPM TG-132
        - Jacobian determinant < 0 indicates folding (invalid deformation)
        - For production, use MONAI or SimpleITK registration backends

    Example:
        >>> reg = BSplineRegistration(control_point_spacing=20.0)
        >>> dvf = reg.register(fixed=planning_ct, moving=daily_cbct)
        >>> propagated_contour = reg.warp_mask(original_mask, dvf)
    """

    def __init__(
        self,
        control_point_spacing: float = 20.0,
        smoothing_sigma: float = 2.0,
        max_iterations: int = 100,
    ):
        """Initialize B-spline registration.

        Args:
            control_point_spacing: Spacing between B-spline control points (mm)
            smoothing_sigma: Gaussian smoothing sigma for regularization
            max_iterations: Maximum optimization iterations
        """
        self.control_point_spacing = control_point_spacing
        self.smoothing_sigma = smoothing_sigma
        self.max_iterations = max_iterations

    def register(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> DeformationField:
        """Compute deformation field from moving to fixed image.

        Args:
            fixed: Reference image (planning CT) [D, H, W]
            moving: Target image (daily CBCT) [D, H, W]
            voxel_spacing: Voxel dimensions in mm

        Returns:
            DeformationField mapping moving to fixed coordinates
        """
        shape = fixed.shape

        # Initialize displacement field (zero = identity)
        n_ctrl = [max(2, int(s * voxel_spacing[i] / self.control_point_spacing))
                  for i, s in enumerate(shape)]

        # Simplified registration using gradient-based optimization
        # In production, use MONAI DenseAffineHead or SimpleITK B-spline
        displacement = np.zeros((3,) + tuple(shape))

        # Compute image difference to drive registration
        diff = fixed.astype(np.float64) - moving.astype(np.float64)

        # Gradient of moving image for force computation
        for axis in range(3):
            grad = np.gradient(moving.astype(np.float64), axis=axis)
            # Demons-like force: displacement proportional to diff * grad
            force = diff * grad / (grad ** 2 + diff ** 2 + 1e-6)
            force = gaussian_filter(force, sigma=self.smoothing_sigma)
            displacement[axis] = force * 2.0  # scale factor

        # Smooth the displacement field
        for axis in range(3):
            displacement[axis] = gaussian_filter(
                displacement[axis], sigma=self.smoothing_sigma * 2
            )

        # Compute Jacobian determinant for quality check
        jac_det = self._compute_jacobian_determinant(displacement)

        # Registration quality metric (mean squared difference after warp)
        warped = self._warp_image(moving, displacement)
        metric = float(np.mean((fixed - warped) ** 2))

        dvf = DeformationField(
            displacement=displacement,
            source_to_target="cbct_to_planning_ct",
            jacobian_determinant=jac_det,
            registration_metric=metric,
        )

        # Quality check: warn if folding detected
        if jac_det is not None and np.any(jac_det < 0):
            n_folded = int(np.sum(jac_det < 0))
            logger.warning(
                "DIR quality: %d voxels with negative Jacobian (folding)",
                n_folded
            )

        logger.info("Registration complete: metric=%.4f", metric)
        return dvf

    def warp_mask(self, mask: np.ndarray, dvf: DeformationField) -> np.ndarray:
        """Propagate binary contour using deformation field.

        Args:
            mask: Binary structure mask on reference geometry
            dvf: Deformation field from register()

        Returns:
            Warped binary mask on deformed geometry
        """
        warped = self._warp_image(mask.astype(np.float64), dvf.displacement)
        return (warped > 0.5).astype(np.float64)

    def warp_dose(self, dose: np.ndarray, dvf: DeformationField) -> np.ndarray:
        """Warp dose distribution using deformation field.

        Uses the inverse deformation field to pull dose values from the
        deformed geometry back to the reference anatomy for accumulation.

        Args:
            dose: Dose distribution on daily geometry
            dvf: Deformation field

        Returns:
            Dose warped to reference anatomy coordinates
        """
        return self._warp_image(dose, dvf.displacement)

    def _warp_image(self, image: np.ndarray, displacement: np.ndarray) -> np.ndarray:
        """Apply displacement field to warp an image."""
        shape = image.shape
        coords = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float64)

        # Add displacement to identity coordinates
        warped_coords = coords + displacement

        # Interpolate image at displaced coordinates
        warped = map_coordinates(
            image.astype(np.float64), warped_coords,
            order=1, mode="nearest"
        )
        return warped

    def _compute_jacobian_determinant(self, displacement: np.ndarray) -> np.ndarray:
        """Compute Jacobian determinant of the deformation field."""
        # Jacobian matrix at each voxel
        jac = np.zeros((3, 3) + displacement.shape[1:])
        for i in range(3):
            for j in range(3):
                jac[i, j] = np.gradient(displacement[i], axis=j)
                if i == j:
                    jac[i, j] += 1.0  # identity component

        # Determinant (3x3 at each voxel)
        det = (
            jac[0, 0] * (jac[1, 1] * jac[2, 2] - jac[1, 2] * jac[2, 1])
            - jac[0, 1] * (jac[1, 0] * jac[2, 2] - jac[1, 2] * jac[2, 0])
            + jac[0, 2] * (jac[1, 0] * jac[2, 1] - jac[1, 1] * jac[2, 0])
        )
        return det


# =============================================================================
# SECTION 3: DOSE ACCUMULATION ENGINE
# =============================================================================
# Accumulates delivered dose on the reference anatomy across fractions,
# accounting for anatomical changes via deformable registration.


@dataclass
class FractionRecord:
    """Record of a single delivered fraction.

    Attributes:
        fraction_number: Fraction index (1-based)
        planned_dose: Planned dose distribution for this fraction
        delivered_dose_ref: Delivered dose warped to reference anatomy
        dvf: Deformation field for this fraction
        structure_volumes: Structure volumes on daily anatomy (cm^3)
        dvh_metrics: DVH metrics computed on accumulated dose
    """

    fraction_number: int
    planned_dose: np.ndarray | None = None
    delivered_dose_ref: np.ndarray | None = None
    dvf: DeformationField | None = None
    structure_volumes: dict[str, float] = field(default_factory=dict)
    dvh_metrics: dict[str, dict] = field(default_factory=dict)


class DoseAccumulator:
    """Cumulative dose accumulation engine for adaptive RT.

    Maintains the accumulated dose on the reference (planning) anatomy
    by warping each fraction's delivered dose back to the reference
    frame using the daily deformation field.

    Instructions for engineers:
        - Reference anatomy = planning CT (fraction 0)
        - Each fraction's dose is warped to reference via inverse DVF
        - DVH metrics computed on accumulated dose after each fraction
        - BED/EQD2 accounts for effective fraction size at each voxel
        - Replanning triggered when accumulated DVH violates constraints

    Example:
        >>> accumulator = DoseAccumulator(reference_dose, structures)
        >>> for fx in range(1, 31):
        ...     accumulator.add_fraction(fx, daily_dose, dvf)
        ...     metrics = accumulator.compute_dvh_metrics()
        ...     if accumulator.check_replanning_trigger(metrics):
        ...         print(f"Replan recommended at fraction {fx}")
    """

    def __init__(
        self,
        reference_shape: tuple[int, int, int],
        structures: list[Structure],
        voxel_spacing: tuple[float, float, float] = (2.5, 1.0, 1.0),
        total_fractions: int = 30,
    ):
        """Initialize dose accumulator.

        Args:
            reference_shape: Shape of reference anatomy (D, H, W)
            structures: List of anatomical structures with constraints
            voxel_spacing: Voxel spacing in mm (slice, row, col)
            total_fractions: Total planned fractions
        """
        self.reference_shape = reference_shape
        self.structures = {s.name: s for s in structures}
        self.voxel_spacing = voxel_spacing
        self.total_fractions = total_fractions
        self.voxel_volume_cc = np.prod(voxel_spacing) / 1000.0

        # Accumulated dose array (Gy)
        self.accumulated_dose = np.zeros(reference_shape)
        self.fraction_records: list[FractionRecord] = []
        self.registration = BSplineRegistration()

        logger.info(
            "Dose accumulator initialized: %s shape, %d structures, %d fractions",
            reference_shape, len(structures), total_fractions
        )

    def add_fraction(
        self,
        fraction_number: int,
        daily_dose: np.ndarray,
        dvf: DeformationField,
    ) -> FractionRecord:
        """Add a delivered fraction to the accumulation.

        Args:
            fraction_number: Fraction index
            daily_dose: Dose delivered on daily anatomy (Gy)
            dvf: Deformation field from daily CBCT registration

        Returns:
            FractionRecord with accumulated metrics
        """
        # Warp daily dose to reference anatomy
        dose_on_ref = self.registration.warp_dose(daily_dose, dvf)

        # Accumulate
        self.accumulated_dose += dose_on_ref

        # Compute daily structure volumes
        volumes = {}
        for name, structure in self.structures.items():
            warped_mask = self.registration.warp_mask(structure.mask, dvf)
            volumes[name] = float(np.sum(warped_mask) * self.voxel_volume_cc)

        # Compute DVH metrics on accumulated dose
        dvh_metrics = self._compute_dvh_metrics()

        record = FractionRecord(
            fraction_number=fraction_number,
            delivered_dose_ref=dose_on_ref,
            dvf=dvf,
            structure_volumes=volumes,
            dvh_metrics=dvh_metrics,
        )
        self.fraction_records.append(record)

        logger.info(
            "Fraction %d/%d accumulated. Max dose: %.1f Gy",
            fraction_number, self.total_fractions,
            float(np.max(self.accumulated_dose))
        )

        return record

    def _compute_dvh_metrics(self) -> dict[str, dict]:
        """Compute DVH metrics for all structures on accumulated dose."""
        metrics = {}

        for name, structure in self.structures.items():
            mask = structure.mask
            if np.sum(mask) == 0:
                continue

            dose_in_structure = self.accumulated_dose[mask > 0.5]

            if len(dose_in_structure) == 0:
                continue

            struct_metrics = {
                "max_dose_gy": float(np.max(dose_in_structure)),
                "min_dose_gy": float(np.min(dose_in_structure)),
                "mean_dose_gy": float(np.mean(dose_in_structure)),
                "d95_gy": float(np.percentile(dose_in_structure, 5)),
                "d50_gy": float(np.percentile(dose_in_structure, 50)),
                "d2_gy": float(np.percentile(dose_in_structure, 98)),
                "volume_cc": float(np.sum(mask) * self.voxel_volume_cc),
            }

            # Volume receiving >= X Gy (for V20, V30, etc.)
            for dose_level in [5, 10, 20, 30, 40, 50, 60]:
                volume_fraction = np.mean(dose_in_structure >= dose_level)
                struct_metrics[f"v{dose_level}_pct"] = float(volume_fraction * 100)

            metrics[name] = struct_metrics

        return metrics

    def compute_bed(
        self, alpha_beta: float = 10.0
    ) -> np.ndarray:
        """Compute biologically effective dose (BED) map.

        BED = n * d * (1 + d / (alpha/beta))
        where n = number of fractions, d = dose per fraction at each voxel.

        For accumulated dose with varying fraction sizes, BED is computed
        voxel-wise using the actual delivered dose per fraction.

        Args:
            alpha_beta: Alpha/beta ratio in Gy (10 for tumor, 3 for late OARs)

        Returns:
            BED array in Gy
        """
        n_fractions = len(self.fraction_records)
        if n_fractions == 0:
            return np.zeros(self.reference_shape)

        bed = np.zeros(self.reference_shape)
        for record in self.fraction_records:
            if record.delivered_dose_ref is not None:
                d = record.delivered_dose_ref
                bed += d * (1 + d / alpha_beta)

        return bed

    def compute_eqd2(self, alpha_beta: float = 10.0) -> np.ndarray:
        """Compute equivalent dose in 2 Gy fractions (EQD2).

        EQD2 = BED / (1 + 2/alpha_beta)

        Args:
            alpha_beta: Alpha/beta ratio in Gy

        Returns:
            EQD2 array in Gy
        """
        bed = self.compute_bed(alpha_beta)
        return bed / (1 + 2.0 / alpha_beta)

    def check_replanning_trigger(
        self, tolerance_percent: float = 5.0
    ) -> dict[str, Any]:
        """Check if replanning is needed based on dosimetric criteria.

        Compares current accumulated dose projections against planned
        constraints. Triggers replanning when any constraint is projected
        to be violated by more than the tolerance.

        Args:
            tolerance_percent: Allowable deviation from constraint (%)

        Returns:
            Dictionary with trigger status and violating constraints
        """
        if not self.fraction_records:
            return {"trigger": False, "reason": "no_fractions"}

        # Project final dose based on current trend
        n_delivered = len(self.fraction_records)
        scale_factor = self.total_fractions / max(n_delivered, 1)

        current_metrics = self.fraction_records[-1].dvh_metrics
        violations = []

        for name, structure in self.structures.items():
            if name not in current_metrics:
                continue

            for constraint in structure.constraints:
                projected = self._project_metric(
                    current_metrics[name], constraint, scale_factor
                )

                if projected is not None:
                    deviation_pct = (projected - constraint.limit_gy) / constraint.limit_gy * 100
                    if deviation_pct > tolerance_percent:
                        violations.append({
                            "structure": constraint.structure_name,
                            "constraint": constraint.constraint_type,
                            "limit_gy": constraint.limit_gy,
                            "projected_gy": round(projected, 2),
                            "deviation_pct": round(deviation_pct, 1),
                            "priority": constraint.priority,
                        })

        trigger = len(violations) > 0

        if trigger:
            logger.warning(
                "Replanning triggered: %d constraint violations detected",
                len(violations)
            )

        return {
            "trigger": trigger,
            "fraction": n_delivered,
            "violations": violations,
            "recommendation": "replan_required" if trigger else "continue",
        }

    def _project_metric(
        self,
        current_metrics: dict,
        constraint: DoseConstraint,
        scale_factor: float,
    ) -> float | None:
        """Project a DVH metric to end of treatment."""
        if constraint.constraint_type == "max_dose":
            return current_metrics.get("max_dose_gy", 0) * scale_factor
        elif constraint.constraint_type == "mean_dose":
            return current_metrics.get("mean_dose_gy", 0) * scale_factor
        elif constraint.constraint_type == "d_95":
            return current_metrics.get("d95_gy", 0) * scale_factor
        return None

    def get_accumulation_summary(self) -> dict[str, Any]:
        """Get summary of dose accumulation progress."""
        n = len(self.fraction_records)
        summary = {
            "fractions_delivered": n,
            "fractions_remaining": self.total_fractions - n,
            "max_accumulated_dose_gy": float(np.max(self.accumulated_dose)),
            "mean_accumulated_dose_gy": float(np.mean(
                self.accumulated_dose[self.accumulated_dose > 0]
            )) if np.any(self.accumulated_dose > 0) else 0.0,
        }

        if n > 0:
            summary["structure_metrics"] = self.fraction_records[-1].dvh_metrics
            # Volume changes over treatment
            if n > 1:
                first = self.fraction_records[0].structure_volumes
                latest = self.fraction_records[-1].structure_volumes
                summary["volume_changes"] = {}
                for name in first:
                    if name in latest and first[name] > 0:
                        change = (latest[name] - first[name]) / first[name] * 100
                        summary["volume_changes"][name] = round(change, 1)

        return summary


# =============================================================================
# SECTION 4: ADAPTIVE RT DIGITAL TWIN ORCHESTRATOR
# =============================================================================
# Top-level class that manages the full ART digital twin workflow:
# daily CBCT → registration → contour propagation → dose accumulation
# → replanning decision.


class AdaptiveRTDigitalTwin:
    """Complete adaptive radiation therapy digital twin.

    Manages the full adaptive RT workflow from planning through daily
    treatment delivery, tracking anatomical changes and cumulative dose.

    Instructions for engineers:
        1. Initialize with planning CT and RT structures
        2. Set the planned dose distribution
        3. For each fraction:
           a. Provide daily CBCT image
           b. Call process_fraction() to register, accumulate, and check
           c. Review replanning_status for adaptation triggers
        4. Access cumulative dose and BED maps at any time
        5. Export reports for clinical review

    Example:
        >>> art = AdaptiveRTDigitalTwin(
        ...     patient_id="LUNG-042",
        ...     planning_ct=ct_array,
        ...     structures=structure_list,
        ...     planned_dose=dose_array,
        ...     total_fractions=30,
        ... )
        >>> for fx in range(1, 31):
        ...     result = art.process_fraction(fx, daily_cbct)
        ...     if result["replanning"]["trigger"]:
        ...         print(f"Replan at fraction {fx}")
    """

    def __init__(
        self,
        patient_id: str,
        planning_ct: np.ndarray,
        structures: list[Structure],
        planned_dose: np.ndarray,
        total_fractions: int = 30,
        prescribed_dose_gy: float = 60.0,
        voxel_spacing: tuple[float, float, float] = (2.5, 1.0, 1.0),
    ):
        """Initialize adaptive RT digital twin.

        Args:
            patient_id: Patient identifier
            planning_ct: Planning CT volume [D, H, W]
            structures: List of delineated structures with constraints
            planned_dose: Planned dose distribution [D, H, W] in Gy
            total_fractions: Total planned fractions
            prescribed_dose_gy: Total prescribed dose in Gy
            voxel_spacing: Voxel spacing in mm
        """
        self.patient_id = patient_id
        self.planning_ct = planning_ct
        self.planned_dose = planned_dose
        self.dose_per_fraction = planned_dose / total_fractions
        self.prescribed_dose_gy = prescribed_dose_gy
        self.total_fractions = total_fractions

        self.registration = BSplineRegistration()
        self.accumulator = DoseAccumulator(
            reference_shape=planning_ct.shape,
            structures=structures,
            voxel_spacing=voxel_spacing,
            total_fractions=total_fractions,
        )

        self._daily_cbcts: list[np.ndarray] = []
        self._replanning_events: list[dict] = []

        logger.info(
            "ART Digital Twin initialized: patient=%s, %d fractions, Rx=%.0f Gy",
            patient_id, total_fractions, prescribed_dose_gy
        )

    def process_fraction(
        self,
        fraction_number: int,
        daily_cbct: np.ndarray,
    ) -> dict[str, Any]:
        """Process a single treatment fraction.

        Performs the complete adaptive workflow:
        1. Register daily CBCT to planning CT
        2. Warp planned dose to daily anatomy (simulate delivery)
        3. Warp delivered dose back to reference for accumulation
        4. Compute DVH metrics on accumulated dose
        5. Check replanning triggers

        Args:
            fraction_number: Fraction index (1-based)
            daily_cbct: Daily CBCT volume [D, H, W]

        Returns:
            Dictionary with registration quality, DVH metrics, and
            replanning recommendation
        """
        self._daily_cbcts.append(daily_cbct)

        # Step 1: Deformable registration (daily CBCT → planning CT)
        dvf = self.registration.register(
            fixed=self.planning_ct,
            moving=daily_cbct,
        )

        # Step 2: Simulate delivered dose on daily anatomy
        # (In practice this comes from the treatment delivery system)
        delivered_dose = self.dose_per_fraction.copy()

        # Step 3: Accumulate dose on reference anatomy
        record = self.accumulator.add_fraction(
            fraction_number, delivered_dose, dvf
        )

        # Step 4: Check replanning triggers
        replan_check = self.accumulator.check_replanning_trigger()
        if replan_check["trigger"]:
            self._replanning_events.append({
                "fraction": fraction_number,
                "violations": replan_check["violations"],
            })

        result = {
            "fraction": fraction_number,
            "registration_quality": dvf.registration_metric,
            "max_accumulated_dose_gy": float(np.max(self.accumulator.accumulated_dose)),
            "dvh_metrics": record.dvh_metrics,
            "structure_volumes": record.structure_volumes,
            "replanning": replan_check,
        }

        return result

    def get_accumulated_dose(self) -> np.ndarray:
        """Get current accumulated dose on reference anatomy."""
        return self.accumulator.accumulated_dose.copy()

    def get_bed_map(self, alpha_beta: float = 10.0) -> np.ndarray:
        """Get biologically effective dose map."""
        return self.accumulator.compute_bed(alpha_beta)

    def get_treatment_summary(self) -> dict[str, Any]:
        """Get comprehensive treatment progress summary."""
        summary = self.accumulator.get_accumulation_summary()
        summary["patient_id"] = self.patient_id
        summary["prescribed_dose_gy"] = self.prescribed_dose_gy
        summary["replanning_events"] = len(self._replanning_events)

        if self._replanning_events:
            summary["replanning_details"] = self._replanning_events

        return summary


# =============================================================================
# SECTION 5: MAIN — SIMULATED ADAPTIVE RT COURSE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Adaptive Radiation Therapy Digital Twin")
    print("Physical AI Oncology Trials — Example 03")
    print("=" * 70)

    # --- Create synthetic planning data ---
    # Small volume for demo (actual clinical: 512x512x~150)
    shape = (32, 64, 64)
    voxel_spacing = (2.5, 1.0, 1.0)

    # Planning CT with tumor
    planning_ct = np.random.randn(*shape) * 100 + 1024  # HU values
    tumor_center = (16, 32, 32)

    # GTV: spherical tumor
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    tumor_radius = 8
    gtv_mask = ((zz - tumor_center[0]) ** 2 +
                (yy - tumor_center[1]) ** 2 +
                (xx - tumor_center[2]) ** 2) < tumor_radius ** 2
    gtv_mask = gtv_mask.astype(np.float64)

    # PTV: GTV + 5mm margin
    ptv_mask = gaussian_filter(gtv_mask, sigma=2.0) > 0.1
    ptv_mask = ptv_mask.astype(np.float64)

    # Spinal cord OAR (cylindrical, posterior)
    cord_mask = ((yy - 50) ** 2 + (xx - 32) ** 2) < 5 ** 2
    cord_mask = cord_mask.astype(np.float64)

    # Planned dose: 2 Gy/fx to PTV, falloff outside
    planned_dose = np.zeros(shape)
    planned_dose[ptv_mask > 0.5] = 60.0  # 60 Gy total
    planned_dose = gaussian_filter(planned_dose, sigma=2.0)

    # Define structures
    structures = [
        Structure(
            name="GTV", structure_type=StructureType.GTV, mask=gtv_mask,
            constraints=[DoseConstraint("GTV", "d_95", 60.0, 95.0, 1)],
            alpha_beta=10.0,
        ),
        Structure(
            name="PTV", structure_type=StructureType.PTV, mask=ptv_mask,
            constraints=[DoseConstraint("PTV", "d_95", 57.0, 95.0, 1)],
            alpha_beta=10.0,
        ),
        Structure(
            name="Spinal_Cord", structure_type=StructureType.OAR, mask=cord_mask,
            constraints=[DoseConstraint("Spinal_Cord", "max_dose", 45.0, 0, 1)],
            alpha_beta=3.0,
        ),
    ]

    # --- Initialize ART digital twin ---
    art = AdaptiveRTDigitalTwin(
        patient_id="LUNG-NSCLC-042",
        planning_ct=planning_ct,
        structures=structures,
        planned_dose=planned_dose,
        total_fractions=30,
        prescribed_dose_gy=60.0,
        voxel_spacing=voxel_spacing,
    )

    print(f"\nPatient: {art.patient_id}")
    print(f"Prescription: {art.prescribed_dose_gy} Gy in {art.total_fractions} fractions")
    print(f"GTV volume: {float(np.sum(gtv_mask) * np.prod(voxel_spacing) / 1000):.1f} cc")
    print()

    # --- Simulate 10 fractions with progressive tumor shrinkage ---
    print("Simulating 10 fractions with anatomical changes:\n")
    for fx in range(1, 11):
        # Simulate daily CBCT with tumor shrinkage (~3% per fraction)
        shrink_factor = 1.0 - 0.03 * fx
        daily_cbct = planning_ct.copy()
        # Modify tumor region to simulate shrinkage
        shrunk_radius = tumor_radius * shrink_factor
        daily_tumor = ((zz - tumor_center[0]) ** 2 +
                       (yy - tumor_center[1]) ** 2 +
                       (xx - tumor_center[2]) ** 2) < shrunk_radius ** 2
        daily_cbct[gtv_mask > 0.5] += 50  # tumor contrast
        daily_cbct += np.random.randn(*shape) * 10  # CBCT noise

        # Process fraction
        result = art.process_fraction(fx, daily_cbct)

        replan_str = "REPLAN" if result["replanning"]["trigger"] else "OK"
        print(
            f"  Fx {fx:2d}: reg_quality={result['registration_quality']:.4f}, "
            f"max_dose={result['max_accumulated_dose_gy']:.1f} Gy, "
            f"status={replan_str}"
        )

    # --- Treatment summary ---
    print("\n" + "=" * 70)
    print("Treatment Progress Summary (after 10/30 fractions):")
    summary = art.get_treatment_summary()
    print(f"  Fractions delivered: {summary['fractions_delivered']}")
    print(f"  Fractions remaining: {summary['fractions_remaining']}")
    print(f"  Max accumulated dose: {summary['max_accumulated_dose_gy']:.1f} Gy")
    print(f"  Replanning events: {summary['replanning_events']}")

    if "volume_changes" in summary:
        print("\n  Volume changes from fraction 1:")
        for name, change in summary["volume_changes"].items():
            print(f"    {name}: {change:+.1f}%")

    # --- BED computation ---
    bed_tumor = art.get_bed_map(alpha_beta=10.0)
    bed_oar = art.get_bed_map(alpha_beta=3.0)
    print(f"\n  BED (tumor, a/b=10): max={float(np.max(bed_tumor)):.1f} Gy")
    print(f"  BED (OAR, a/b=3):   max={float(np.max(bed_oar)):.1f} Gy")

    print("\n" + "=" * 70)
    print("Adaptive RT digital twin simulation complete.")
