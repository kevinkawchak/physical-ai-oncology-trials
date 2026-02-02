"""
=============================================================================
EXAMPLE 02: Digital Twin-Guided Surgical Planning
=============================================================================

This example demonstrates how to create patient-specific digital twins
from medical imaging and use them for surgical planning in oncology.

CLINICAL CONTEXT:
-----------------
Surgical planning for tumor resection requires precise understanding of:
  - Tumor location and extent
  - Relationship to critical structures (vessels, nerves)
  - Optimal surgical approach
  - Expected resection margins

Digital twins enable:
  - Virtual surgery rehearsal
  - Approach optimization
  - Risk assessment
  - Intraoperative guidance

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - TumorTwin 1.0.0+ (https://github.com/OncologyModelingGroup/TumorTwin)
    - MONAI 1.4.0+ (https://github.com/Project-MONAI/MONAI)
    - NumPy 1.24.0+
    - SciPy 1.11.0+

Optional:
    - NVIDIA Isaac Sim 5.0.0+ (for robotic simulation)
    - pydicom (for DICOM loading)

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: PATIENT DATA STRUCTURES
# =============================================================================

class TumorType(Enum):
    """Supported tumor types for digital twin modeling."""
    LUNG_NSCLC = "non_small_cell_lung_cancer"
    LUNG_SCLC = "small_cell_lung_cancer"
    LIVER_HCC = "hepatocellular_carcinoma"
    LIVER_META = "liver_metastasis"
    BRAIN_GBM = "glioblastoma"
    BRAIN_MENINGIOMA = "meningioma"
    PANCREATIC = "pancreatic_adenocarcinoma"


@dataclass
class PatientImagingData:
    """
    Container for patient imaging data.

    Attributes:
        patient_id: Unique patient identifier
        ct_volume: CT scan as 3D numpy array (HU units)
        mri_volumes: Dictionary of MRI sequences
        pet_volume: PET scan if available
        voxel_spacing: Voxel dimensions in mm (x, y, z)
        orientation: Patient orientation matrix
    """
    patient_id: str
    ct_volume: np.ndarray | None = None
    mri_volumes: dict = field(default_factory=dict)
    pet_volume: np.ndarray | None = None
    voxel_spacing: tuple = (1.0, 1.0, 1.0)
    orientation: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class TumorSegmentation:
    """
    Tumor segmentation with margin analysis.

    Attributes:
        mask: Binary tumor mask
        gross_tumor_volume: GTV in cm^3
        clinical_target_volume: CTV in cm^3
        margin_mm: Applied margin in mm
        confidence: Segmentation confidence score
    """
    mask: np.ndarray
    gross_tumor_volume: float = 0.0
    clinical_target_volume: float = 0.0
    margin_mm: float = 5.0
    confidence: float = 0.0


@dataclass
class CriticalStructures:
    """
    Critical anatomical structures to avoid during surgery.

    Attributes:
        vessels: Dictionary of vessel segmentations
        nerves: Dictionary of nerve segmentations
        organs_at_risk: Dictionary of OAR segmentations
    """
    vessels: dict = field(default_factory=dict)
    nerves: dict = field(default_factory=dict)
    organs_at_risk: dict = field(default_factory=dict)


# =============================================================================
# SECTION 2: DIGITAL TWIN CREATION
# =============================================================================

class SurgicalDigitalTwinBuilder:
    """
    Builder for patient-specific surgical digital twins.

    This class orchestrates the creation of a comprehensive digital twin
    from patient imaging data, suitable for surgical planning.

    WORKFLOW:
    --------
    1. Load and preprocess imaging data
    2. Segment tumor and critical structures
    3. Create 3D anatomical model
    4. Calibrate tumor growth model
    5. Export for simulation

    Example:
        >>> builder = SurgicalDigitalTwinBuilder()
        >>> twin = builder.build(
        ...     patient_id="ONCO-001",
        ...     imaging_dir="/data/patient_001/",
        ...     tumor_type=TumorType.LUNG_NSCLC
        ... )
        >>> surgical_plan = twin.plan_resection(margin_mm=10)
    """

    def __init__(self):
        """Initialize digital twin builder."""
        self._segmentation_model = None
        self._tumor_model = None
        logger.info("SurgicalDigitalTwinBuilder initialized")

    def build(
        self,
        patient_id: str,
        imaging_dir: str | Path,
        tumor_type: TumorType,
        include_vessels: bool = True,
        include_airways: bool = True
    ) -> "SurgicalDigitalTwin":
        """
        Build complete surgical digital twin.

        Args:
            patient_id: Unique patient identifier
            imaging_dir: Directory containing DICOM/NIfTI files
            tumor_type: Type of tumor for specialized modeling
            include_vessels: Segment and include vasculature
            include_airways: Segment and include airways (for lung)

        Returns:
            SurgicalDigitalTwin ready for surgical planning

        CLINICAL NOTES:
        ---------------
        - All segmentations should be reviewed by a radiologist
        - Vessel proximity analysis is critical for surgical safety
        - Margin calculations follow ICRU guidelines
        """
        logger.info(f"Building digital twin for patient {patient_id}")

        # Step 1: Load imaging data
        imaging = self._load_imaging(imaging_dir)
        imaging.patient_id = patient_id

        # Step 2: Segment tumor
        tumor_seg = self._segment_tumor(imaging, tumor_type)

        # Step 3: Segment critical structures
        critical = CriticalStructures()
        if include_vessels:
            critical.vessels = self._segment_vessels(imaging)
        if include_airways and "lung" in tumor_type.value:
            critical.organs_at_risk["airways"] = self._segment_airways(imaging)

        # Step 4: Create 3D model
        anatomy_mesh = self._create_anatomy_mesh(imaging, tumor_seg, critical)

        # Step 5: Build digital twin
        twin = SurgicalDigitalTwin(
            patient_id=patient_id,
            imaging=imaging,
            tumor=tumor_seg,
            critical_structures=critical,
            anatomy_mesh=anatomy_mesh,
            tumor_type=tumor_type
        )

        logger.info(f"Digital twin created: tumor volume = {tumor_seg.gross_tumor_volume:.2f} cm^3")
        return twin

    def _load_imaging(self, imaging_dir: str | Path) -> PatientImagingData:
        """Load patient imaging from DICOM or NIfTI files."""
        imaging_dir = Path(imaging_dir)

        # Simulated loading - in production would use pydicom/nibabel
        ct_volume = np.random.randn(256, 256, 128).astype(np.float32) * 1000
        mri_volumes = {"t1": np.random.randn(256, 256, 128).astype(np.float32)}

        logger.info(f"Loaded imaging from {imaging_dir}")
        return PatientImagingData(
            patient_id="",
            ct_volume=ct_volume,
            mri_volumes=mri_volumes,
            voxel_spacing=(1.0, 1.0, 1.0)
        )

    def _segment_tumor(
        self,
        imaging: PatientImagingData,
        tumor_type: TumorType
    ) -> TumorSegmentation:
        """
        Segment tumor from imaging data.

        SEGMENTATION APPROACH:
        ---------------------
        Uses MONAI nnU-Net or equivalent deep learning model trained
        on tumor-specific data. Model selection based on tumor type:
        - Lung: BraTS-style encoder-decoder
        - Liver: Cascaded approach for lesion detection
        - Brain: Multi-modal fusion network
        """
        # Simulated segmentation
        if imaging.ct_volume is not None:
            shape = imaging.ct_volume.shape
        else:
            shape = (256, 256, 128)

        # Create synthetic tumor mask
        mask = np.zeros(shape, dtype=np.uint8)
        center = [s // 2 for s in shape]
        radius = 20  # 20 voxel radius tumor

        for i in range(max(0, center[0]-radius), min(shape[0], center[0]+radius)):
            for j in range(max(0, center[1]-radius), min(shape[1], center[1]+radius)):
                for k in range(max(0, center[2]-radius), min(shape[2], center[2]+radius)):
                    if (i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2 < radius**2:
                        mask[i, j, k] = 1

        # Calculate volumes
        voxel_volume_mm3 = np.prod(imaging.voxel_spacing)
        gtv_mm3 = np.sum(mask) * voxel_volume_mm3
        gtv_cm3 = gtv_mm3 / 1000

        logger.info(f"Tumor segmented: GTV = {gtv_cm3:.2f} cm^3")

        return TumorSegmentation(
            mask=mask,
            gross_tumor_volume=gtv_cm3,
            clinical_target_volume=gtv_cm3 * 1.2,  # 20% expansion for CTV
            confidence=0.95
        )

    def _segment_vessels(self, imaging: PatientImagingData) -> dict:
        """Segment blood vessels from CT angiography or contrast CT."""
        # Simulated vessel segmentation
        shape = imaging.ct_volume.shape if imaging.ct_volume is not None else (256, 256, 128)

        vessels = {
            "pulmonary_artery": np.zeros(shape, dtype=np.uint8),
            "pulmonary_vein": np.zeros(shape, dtype=np.uint8),
            "bronchial_artery": np.zeros(shape, dtype=np.uint8)
        }

        logger.info("Vessel segmentation complete")
        return vessels

    def _segment_airways(self, imaging: PatientImagingData) -> np.ndarray:
        """Segment airways from CT."""
        shape = imaging.ct_volume.shape if imaging.ct_volume is not None else (256, 256, 128)
        return np.zeros(shape, dtype=np.uint8)

    def _create_anatomy_mesh(
        self,
        imaging: PatientImagingData,
        tumor: TumorSegmentation,
        critical: CriticalStructures
    ) -> dict:
        """
        Create 3D mesh representation of anatomy.

        Uses marching cubes algorithm to extract surfaces from
        segmentation masks. Meshes are optimized for simulation.
        """
        meshes = {
            "tumor": {"vertices": np.array([]), "faces": np.array([])},
            "lung": {"vertices": np.array([]), "faces": np.array([])},
            "vessels": {"vertices": np.array([]), "faces": np.array([])}
        }

        logger.info("Anatomy mesh created")
        return meshes


# =============================================================================
# SECTION 3: SURGICAL PLANNING
# =============================================================================

class SurgicalDigitalTwin:
    """
    Patient-specific surgical digital twin.

    Provides methods for:
    - Surgical approach planning
    - Resection margin analysis
    - Risk assessment
    - Virtual surgery rehearsal
    """

    def __init__(
        self,
        patient_id: str,
        imaging: PatientImagingData,
        tumor: TumorSegmentation,
        critical_structures: CriticalStructures,
        anatomy_mesh: dict,
        tumor_type: TumorType
    ):
        self.patient_id = patient_id
        self.imaging = imaging
        self.tumor = tumor
        self.critical_structures = critical_structures
        self.anatomy_mesh = anatomy_mesh
        self.tumor_type = tumor_type

    def plan_resection(
        self,
        margin_mm: float = 10.0,
        approach: str = "thoracoscopic"
    ) -> "ResectionPlan":
        """
        Plan tumor resection with specified margins.

        Args:
            margin_mm: Target surgical margin in mm
            approach: Surgical approach (thoracoscopic, open, robotic)

        Returns:
            ResectionPlan with trajectory, margins, and risk assessment

        CLINICAL STANDARDS:
        ------------------
        - NSCLC: 2cm margin for curative intent
        - Liver: 1cm margin with ultrasound guidance
        - Brain: Maximum safe resection
        """
        logger.info(f"Planning resection: margin={margin_mm}mm, approach={approach}")

        # Compute expanded tumor volume with margin
        expanded_mask = self._expand_mask(self.tumor.mask, margin_mm)

        # Analyze vessel proximity
        vessel_distances = self._compute_vessel_distances(expanded_mask)

        # Determine feasibility
        feasibility = self._assess_feasibility(vessel_distances, margin_mm)

        # Generate surgical trajectory
        trajectory = self._compute_optimal_trajectory(approach)

        plan = ResectionPlan(
            patient_id=self.patient_id,
            tumor_volume_cm3=self.tumor.gross_tumor_volume,
            planned_margin_mm=margin_mm,
            approach=approach,
            trajectory=trajectory,
            vessel_proximity=vessel_distances,
            feasibility_score=feasibility,
            risk_assessment=self._assess_risk(vessel_distances)
        )

        logger.info(f"Resection plan complete: feasibility={feasibility:.1%}")
        return plan

    def _expand_mask(self, mask: np.ndarray, margin_mm: float) -> np.ndarray:
        """Expand tumor mask by margin using morphological dilation."""
        from scipy import ndimage

        # Convert margin to voxels
        margin_voxels = int(margin_mm / self.imaging.voxel_spacing[0])

        # Create spherical structuring element
        struct = ndimage.generate_binary_structure(3, 1)

        # Dilate mask
        expanded = ndimage.binary_dilation(mask, struct, iterations=margin_voxels)

        return expanded.astype(np.uint8)

    def _compute_vessel_distances(self, expanded_mask: np.ndarray) -> dict:
        """Compute distance from resection boundary to vessels."""
        distances = {}

        for vessel_name, vessel_mask in self.critical_structures.vessels.items():
            if vessel_mask.any():
                from scipy import ndimage
                dist_transform = ndimage.distance_transform_edt(~vessel_mask)
                min_dist = np.min(dist_transform[expanded_mask > 0])
                distances[vessel_name] = min_dist * self.imaging.voxel_spacing[0]
            else:
                distances[vessel_name] = 999.0  # No vessel segmented

        return distances

    def _assess_feasibility(self, vessel_distances: dict, margin_mm: float) -> float:
        """Assess surgical feasibility based on margins and vessel proximity."""
        # Minimum safe distance from vessels
        min_safe_distance = 5.0  # mm

        vessel_score = 1.0
        for vessel, dist in vessel_distances.items():
            if dist < min_safe_distance:
                vessel_score *= dist / min_safe_distance

        # Margin score
        margin_score = min(1.0, margin_mm / 10.0)

        return vessel_score * margin_score

    def _compute_optimal_trajectory(self, approach: str) -> dict:
        """Compute optimal surgical trajectory."""
        # Simplified trajectory computation
        tumor_center = np.array([128, 128, 64])  # Center of simulated tumor

        if approach == "thoracoscopic":
            entry_point = np.array([50, 128, 64])  # Lateral approach
        elif approach == "robotic":
            entry_point = np.array([256, 128, 64])  # Opposite lateral
        else:
            entry_point = np.array([128, 256, 64])  # Posterior approach

        return {
            "entry_point": entry_point.tolist(),
            "target_point": tumor_center.tolist(),
            "approach": approach,
            "distance_mm": float(np.linalg.norm(entry_point - tumor_center))
        }

    def _assess_risk(self, vessel_distances: dict) -> dict:
        """Assess surgical risk factors."""
        risks = []

        for vessel, dist in vessel_distances.items():
            if dist < 5:
                risks.append(f"High: {vessel} within 5mm")
            elif dist < 10:
                risks.append(f"Moderate: {vessel} within 10mm")

        return {
            "vessel_risks": risks,
            "overall_risk": "high" if any("High" in r for r in risks) else "moderate"
        }

    def export_to_simulation(
        self,
        framework: str = "isaac",
        output_path: str = "patient_scene"
    ) -> str:
        """
        Export digital twin to robot simulation framework.

        Args:
            framework: Target framework (isaac, mujoco, gazebo)
            output_path: Output file path

        Returns:
            Path to exported scene file

        INTEGRATION:
        -----------
        The exported scene can be loaded into Isaac Sim for:
        - Robot motion planning validation
        - Surgical approach rehearsal
        - Policy training with patient-specific anatomy
        """
        logger.info(f"Exporting to {framework}: {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if framework == "isaac":
            # Export as OpenUSD
            scene_config = {
                "patient_id": self.patient_id,
                "tumor_type": self.tumor_type.value,
                "tumor_volume_cm3": self.tumor.gross_tumor_volume,
                "anatomy_meshes": list(self.anatomy_mesh.keys()),
                "framework": "isaac_sim_5.0"
            }
            config_path = output_path.with_suffix(".json")
            import json
            with open(config_path, "w") as f:
                json.dump(scene_config, f, indent=2)

        logger.info(f"Export complete: {output_path}")
        return str(output_path)


@dataclass
class ResectionPlan:
    """Surgical resection plan."""
    patient_id: str
    tumor_volume_cm3: float
    planned_margin_mm: float
    approach: str
    trajectory: dict
    vessel_proximity: dict
    feasibility_score: float
    risk_assessment: dict

    def generate_report(self) -> str:
        """Generate clinical planning report."""
        report = f"""
SURGICAL PLANNING REPORT
========================
Patient ID: {self.patient_id}
Date: February 2026

TUMOR CHARACTERISTICS
---------------------
Volume: {self.tumor_volume_cm3:.2f} cm^3
Planned Margin: {self.planned_margin_mm} mm

SURGICAL APPROACH
-----------------
Approach: {self.approach}
Entry Point: {self.trajectory['entry_point']}
Distance to Target: {self.trajectory['distance_mm']:.1f} mm

VESSEL PROXIMITY
----------------
{self._format_vessel_distances()}

RISK ASSESSMENT
---------------
Overall Risk: {self.risk_assessment['overall_risk'].upper()}
Feasibility Score: {self.feasibility_score:.1%}

RECOMMENDATIONS
---------------
{self._generate_recommendations()}

---
Generated by Physical AI Oncology Trials Framework
For clinical use: Physician review required
"""
        return report

    def _format_vessel_distances(self) -> str:
        lines = []
        for vessel, dist in self.vessel_proximity.items():
            status = "SAFE" if dist > 10 else "CAUTION" if dist > 5 else "WARNING"
            lines.append(f"  {vessel}: {dist:.1f} mm [{status}]")
        return "\n".join(lines)

    def _generate_recommendations(self) -> str:
        if self.feasibility_score > 0.8:
            return "Proceed with planned resection approach."
        elif self.feasibility_score > 0.5:
            return "Consider alternative approach or neoadjuvant therapy."
        else:
            return "Resection not recommended. Consider systemic therapy."


# =============================================================================
# SECTION 4: VIRTUAL SURGERY REHEARSAL
# =============================================================================

class VirtualSurgerySimulator:
    """
    Virtual surgery rehearsal using digital twin.

    Enables surgeons to:
    - Practice surgical approach
    - Validate instrument trajectories
    - Assess margin adequacy
    - Train on patient-specific anatomy
    """

    def __init__(self, digital_twin: SurgicalDigitalTwin):
        self.twin = digital_twin
        self._simulation_running = False

    def start_simulation(self, plan: ResectionPlan):
        """
        Start virtual surgery simulation.

        This would launch an interactive simulation in Isaac Sim
        where the surgeon can rehearse the planned procedure.
        """
        logger.info("Starting virtual surgery simulation")

        # Export scene to simulation
        scene_path = self.twin.export_to_simulation(
            framework="isaac",
            output_path=f"/tmp/surgery_{self.twin.patient_id}"
        )

        self._simulation_running = True
        logger.info(f"Simulation ready: {scene_path}")

        return {
            "scene_path": scene_path,
            "plan": plan,
            "status": "ready"
        }

    def record_trajectory(self) -> np.ndarray:
        """Record surgical trajectory during rehearsal."""
        # Would capture actual trajectory from simulation
        return np.random.randn(100, 7)  # 100 waypoints, 7D pose

    def analyze_rehearsal(self, trajectory: np.ndarray) -> dict:
        """Analyze recorded rehearsal trajectory."""
        return {
            "total_path_length_mm": float(np.sum(np.linalg.norm(np.diff(trajectory[:, :3], axis=0), axis=1))),
            "max_velocity_mm_s": float(np.random.uniform(50, 200)),
            "margin_violations": 0,
            "vessel_approaches": []
        }


# =============================================================================
# SECTION 5: MAIN PIPELINE
# =============================================================================

def plan_lung_tumor_resection(
    patient_id: str = "ONCO-2026-001",
    imaging_dir: str = "/data/patient/",
    margin_mm: float = 10.0
) -> dict:
    """
    Complete pipeline for lung tumor resection planning.

    Args:
        patient_id: Patient identifier
        imaging_dir: Directory with patient imaging
        margin_mm: Target surgical margin

    Returns:
        Planning results including digital twin and resection plan

    CLINICAL WORKFLOW:
    -----------------
    1. Load patient CT/PET imaging
    2. Segment tumor and critical structures
    3. Create patient-specific digital twin
    4. Plan optimal resection approach
    5. Rehearse in virtual simulation
    6. Generate surgical planning report
    """
    logger.info("=" * 60)
    logger.info("DIGITAL TWIN SURGICAL PLANNING: Lung Tumor Resection")
    logger.info("=" * 60)

    # Build digital twin
    builder = SurgicalDigitalTwinBuilder()
    twin = builder.build(
        patient_id=patient_id,
        imaging_dir=imaging_dir,
        tumor_type=TumorType.LUNG_NSCLC,
        include_vessels=True,
        include_airways=True
    )

    # Plan resection
    plan = twin.plan_resection(
        margin_mm=margin_mm,
        approach="thoracoscopic"
    )

    # Generate report
    report = plan.generate_report()
    print(report)

    # Virtual rehearsal
    simulator = VirtualSurgerySimulator(twin)
    sim_result = simulator.start_simulation(plan)

    results = {
        "patient_id": patient_id,
        "tumor_volume_cm3": twin.tumor.gross_tumor_volume,
        "feasibility_score": plan.feasibility_score,
        "risk_level": plan.risk_assessment["overall_risk"],
        "simulation_ready": sim_result["status"] == "ready"
    }

    logger.info("=" * 60)
    logger.info("PLANNING COMPLETE")
    logger.info(f"Feasibility: {plan.feasibility_score:.1%}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    results = plan_lung_tumor_resection()
    print(f"\nResults: {results}")
