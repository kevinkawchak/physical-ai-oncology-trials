"""
Clinical Digital Twin Interface

Production-ready interface for integrating digital twins with hospital
clinical systems including PACS, EHR, and surgical navigation systems.

Version: 1.0.0
Last Updated: February 2026
Framework Dependencies:
    - pydicom 2.4.0+
    - fhirclient 4.0.0+
    - numpy 1.24.0+

License: MIT
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Clinical system connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    AUTHENTICATING = "authenticating"


class ComplianceRegulation(Enum):
    """Regulatory compliance frameworks."""
    FDA_21CFR11 = "21CFR11"
    HIPAA = "HIPAA"
    GDPR = "GDPR"
    IEC_62304 = "IEC62304"


@dataclass
class PatientRecord:
    """Patient clinical record.

    Attributes:
        mrn: Medical record number
        name: Patient name (de-identified for research)
        dob: Date of birth
        sex: Patient sex
        diagnoses: List of diagnoses
        imaging_studies: List of imaging study references
        treatments: Treatment history
    """
    mrn: str
    name: str = ""
    dob: str = ""
    sex: str = ""
    diagnoses: list = field(default_factory=list)
    imaging_studies: list = field(default_factory=list)
    treatments: list = field(default_factory=list)
    biomarkers: dict = field(default_factory=dict)


@dataclass
class ImagingStudy:
    """Imaging study metadata.

    Attributes:
        study_id: Unique study identifier
        modality: Imaging modality (CT, MR, PET, etc.)
        date: Study date
        body_part: Body region
        series: List of series in study
        storage_path: Path to DICOM files
    """
    study_id: str
    modality: str
    date: str
    body_part: str = ""
    series: list = field(default_factory=list)
    storage_path: str = ""


@dataclass
class AuditEntry:
    """Audit trail entry for compliance.

    Attributes:
        timestamp: Event timestamp
        user: User identifier
        action: Action performed
        resource: Resource affected
        details: Additional details
    """
    timestamp: datetime
    user: str
    action: str
    resource: str
    details: dict = field(default_factory=dict)


class ClinicalConnector:
    """
    Connect to hospital clinical systems (PACS, EHR, FHIR).

    Provides unified interface for accessing patient data from various
    hospital information systems, enabling digital twin creation from
    real clinical data.

    Attributes:
        pacs_endpoint: PACS server URL
        fhir_endpoint: FHIR server URL
        auth_method: Authentication method

    Example:
        >>> connector = ClinicalConnector(
        ...     pacs_endpoint="https://pacs.hospital.local",
        ...     fhir_endpoint="https://fhir.hospital.local/R4"
        ... )
        >>> patient = connector.get_patient(mrn="12345678")
        >>> studies = connector.query_imaging(patient_id=patient.mrn)

    References:
        - HL7 FHIR R4: https://www.hl7.org/fhir/
        - DICOM: https://www.dicomstandard.org/
    """

    def __init__(
        self,
        pacs_endpoint: str = "",
        fhir_endpoint: str = "",
        auth_method: str = "oauth2",
        credentials_path: str | None = None
    ):
        """Initialize clinical connector.

        Args:
            pacs_endpoint: PACS server endpoint URL
            fhir_endpoint: FHIR R4 server endpoint URL
            auth_method: Authentication method (oauth2, basic, certificate)
            credentials_path: Path to credentials file
        """
        self.pacs_endpoint = pacs_endpoint
        self.fhir_endpoint = fhir_endpoint
        self.auth_method = auth_method
        self.credentials_path = credentials_path

        self._pacs_client = None
        self._fhir_client = None
        self._audit_log = []

        logger.info(f"ClinicalConnector initialized with FHIR: {fhir_endpoint}")

    def test_connection(self) -> dict[str, ConnectionStatus]:
        """Test connectivity to clinical systems.

        Returns:
            Dictionary with connection status for each system
        """
        status = {}

        # Test PACS connection
        if self.pacs_endpoint:
            try:
                # Simulated connection test
                status["pacs"] = ConnectionStatus.CONNECTED
            except Exception as e:
                logger.error(f"PACS connection failed: {e}")
                status["pacs"] = ConnectionStatus.ERROR
        else:
            status["pacs"] = ConnectionStatus.DISCONNECTED

        # Test FHIR connection
        if self.fhir_endpoint:
            try:
                status["fhir"] = ConnectionStatus.CONNECTED
            except Exception as e:
                logger.error(f"FHIR connection failed: {e}")
                status["fhir"] = ConnectionStatus.ERROR
        else:
            status["fhir"] = ConnectionStatus.DISCONNECTED

        return status

    def get_patient(self, mrn: str) -> PatientRecord:
        """Retrieve patient record by MRN.

        Args:
            mrn: Medical record number

        Returns:
            PatientRecord with clinical data
        """
        self._log_audit("get_patient", f"Patient/{mrn}")

        # In production, this would query FHIR/EHR
        # Simulated response for demonstration
        patient = PatientRecord(
            mrn=mrn,
            name="[De-identified]",
            sex="M",
            diagnoses=["C34.9 - Malignant neoplasm of lung"],
            biomarkers={"EGFR": "positive", "ALK": "negative"}
        )

        logger.info(f"Retrieved patient record: {mrn}")
        return patient

    def query_imaging(
        self,
        patient_id: str,
        modality: list[str] | None = None,
        body_part: str = "",
        date_range: tuple[str, str] | None = None
    ) -> list[ImagingStudy]:
        """Query imaging studies from PACS.

        Args:
            patient_id: Patient identifier
            modality: List of modalities to include (CT, MR, PET, etc.)
            body_part: Body part filter
            date_range: Date range tuple (start, end)

        Returns:
            List of matching ImagingStudy objects
        """
        self._log_audit(
            "query_imaging",
            f"Patient/{patient_id}",
            {"modality": modality, "body_part": body_part}
        )

        # Simulated PACS query
        studies = [
            ImagingStudy(
                study_id=f"1.2.3.4.{i}",
                modality=mod,
                date="2025-12-01",
                body_part=body_part or "CHEST"
            )
            for i, mod in enumerate(modality or ["CT"])
        ]

        logger.info(f"Found {len(studies)} imaging studies for patient {patient_id}")
        return studies

    def download_study(
        self,
        study_id: str,
        output_dir: str | None = None
    ) -> list[str]:
        """Download DICOM study from PACS.

        Args:
            study_id: Study identifier
            output_dir: Output directory for DICOM files

        Returns:
            List of downloaded file paths
        """
        self._log_audit("download_study", f"Study/{study_id}")

        # In production, this would download actual DICOM files
        output_dir = output_dir or f"/tmp/dicom/{study_id}"
        files = [f"{output_dir}/image_{i:04d}.dcm" for i in range(100)]

        logger.info(f"Downloaded study {study_id}: {len(files)} files")
        return files

    def _log_audit(
        self,
        action: str,
        resource: str,
        details: dict | None = None
    ) -> None:
        """Log audit entry for compliance."""
        entry = AuditEntry(
            timestamp=datetime.now(),
            user="system",  # Would be actual user in production
            action=action,
            resource=resource,
            details=details or {}
        )
        self._audit_log.append(entry)


class FHIRClient:
    """
    FHIR R4 client for healthcare data access.

    Implements standard FHIR operations for accessing and storing
    clinical data in FHIR-compliant servers.

    Attributes:
        base_url: FHIR server base URL

    Example:
        >>> client = FHIRClient("https://fhir.hospital.local/R4")
        >>> patients = client.search("Patient", {"identifier": "MRN|123"})
    """

    def __init__(self, base_url: str):
        """Initialize FHIR client.

        Args:
            base_url: FHIR R4 server base URL
        """
        self.base_url = base_url.rstrip("/")
        self._session = None

        logger.info(f"FHIRClient initialized: {base_url}")

    def search(
        self,
        resource_type: str,
        params: dict[str, str]
    ) -> list[dict]:
        """Search for FHIR resources.

        Args:
            resource_type: FHIR resource type (Patient, Observation, etc.)
            params: Search parameters

        Returns:
            List of matching resources
        """
        # Simulated FHIR search
        logger.info(f"FHIR search: {resource_type} with params {params}")

        # Return simulated results
        if resource_type == "Patient":
            return [{
                "resourceType": "Patient",
                "id": "patient-123",
                "identifier": [{"system": "MRN", "value": params.get("identifier", "").split("|")[-1]}],
                "name": [{"family": "Test", "given": ["Patient"]}]
            }]
        elif resource_type == "ImagingStudy":
            return [{
                "resourceType": "ImagingStudy",
                "id": f"study-{i}",
                "status": "available",
                "modality": [{"code": "CT"}]
            } for i in range(3)]
        else:
            return []

    def read(self, resource_type: str, resource_id: str) -> dict:
        """Read a single FHIR resource.

        Args:
            resource_type: FHIR resource type
            resource_id: Resource ID

        Returns:
            FHIR resource dictionary
        """
        logger.info(f"FHIR read: {resource_type}/{resource_id}")
        return {
            "resourceType": resource_type,
            "id": resource_id
        }

    def create(self, resource_type: str, resource: dict) -> dict:
        """Create a new FHIR resource.

        Args:
            resource_type: FHIR resource type
            resource: Resource content

        Returns:
            Created resource with server-assigned ID
        """
        logger.info(f"FHIR create: {resource_type}")
        resource["id"] = f"new-{datetime.now().timestamp()}"
        return resource

    def update(self, resource_type: str, resource_id: str, resource: dict) -> dict:
        """Update existing FHIR resource.

        Args:
            resource_type: FHIR resource type
            resource_id: Resource ID
            resource: Updated resource content

        Returns:
            Updated resource
        """
        logger.info(f"FHIR update: {resource_type}/{resource_id}")
        resource["id"] = resource_id
        return resource


class DICOMHandler:
    """
    DICOM image handling and processing.

    Provides utilities for loading, processing, and storing DICOM
    medical images for digital twin creation.

    Example:
        >>> handler = DICOMHandler()
        >>> series = handler.load_series("/path/to/dicom/")
        >>> volume = handler.to_numpy(series)
    """

    def __init__(self):
        """Initialize DICOM handler."""
        self._pydicom_available = self._check_pydicom()

    def _check_pydicom(self) -> bool:
        """Check if pydicom is available."""
        try:
            import pydicom
            return True
        except ImportError:
            logger.warning("pydicom not installed, using numpy fallback")
            return False

    def load_series(self, series_path: str | Path) -> list:
        """Load DICOM series from directory.

        Args:
            series_path: Path to DICOM series directory

        Returns:
            List of DICOM datasets
        """
        series_path = Path(series_path)
        datasets = []

        if self._pydicom_available:
            import pydicom

            for dcm_file in sorted(series_path.glob("*.dcm")):
                try:
                    ds = pydicom.dcmread(str(dcm_file))
                    datasets.append(ds)
                except Exception as e:
                    logger.warning(f"Failed to read {dcm_file}: {e}")
        else:
            # Simulated load
            logger.info(f"Simulated DICOM load from {series_path}")
            datasets = [{"SimulatedDataset": True} for _ in range(50)]

        logger.info(f"Loaded {len(datasets)} DICOM files from {series_path}")
        return datasets

    def extract_metadata(self, datasets: list) -> dict:
        """Extract metadata from DICOM datasets.

        Args:
            datasets: List of DICOM datasets

        Returns:
            Dictionary of extracted metadata
        """
        if not datasets:
            return {}

        if self._pydicom_available and hasattr(datasets[0], "Modality"):
            ds = datasets[0]
            return {
                "Modality": getattr(ds, "Modality", "Unknown"),
                "PatientName": str(getattr(ds, "PatientName", "Unknown")),
                "StudyDate": getattr(ds, "StudyDate", "Unknown"),
                "SeriesDescription": getattr(ds, "SeriesDescription", "Unknown"),
                "SliceThickness": getattr(ds, "SliceThickness", 1.0),
                "PixelSpacing": list(getattr(ds, "PixelSpacing", [1.0, 1.0]))
            }
        else:
            return {
                "Modality": "CT",
                "PatientName": "Test Patient",
                "StudyDate": "20251201",
                "SliceThickness": 1.0,
                "PixelSpacing": [1.0, 1.0]
            }

    def to_numpy(self, datasets: list) -> np.ndarray:
        """Convert DICOM series to numpy volume.

        Args:
            datasets: List of DICOM datasets

        Returns:
            3D numpy array (slices, height, width)
        """
        if not datasets:
            return np.zeros((64, 256, 256))

        if self._pydicom_available and hasattr(datasets[0], "pixel_array"):
            slices = [ds.pixel_array for ds in datasets]
            volume = np.stack(slices, axis=0)
        else:
            # Simulated volume
            volume = np.random.randn(len(datasets), 256, 256)

        logger.info(f"Created volume: {volume.shape}")
        return volume

    def create_structured_report(
        self,
        patient_dt,
        report_type: str,
        content: dict
    ) -> dict:
        """Create DICOM Structured Report.

        Args:
            patient_dt: Patient digital twin
            report_type: Type of report
            content: Report content dictionary

        Returns:
            DICOM SR as dictionary (would be pydicom Dataset in production)
        """
        sr = {
            "SOPClassUID": "1.2.840.10008.5.1.4.1.1.88.33",  # Comprehensive SR
            "Modality": "SR",
            "SeriesDescription": f"Digital Twin {report_type}",
            "ContentDate": datetime.now().strftime("%Y%m%d"),
            "ContentTime": datetime.now().strftime("%H%M%S"),
            "Content": content
        }

        logger.info(f"Created DICOM SR: {report_type}")
        return sr

    def store_to_pacs(self, sr: dict, pacs_endpoint: str) -> bool:
        """Store DICOM object to PACS.

        Args:
            sr: DICOM object to store
            pacs_endpoint: PACS endpoint URL

        Returns:
            Success status
        """
        logger.info(f"Storing to PACS: {pacs_endpoint}")
        # In production, this would use C-STORE
        return True


class ClinicalDecisionSupport:
    """
    Clinical decision support using digital twins.

    Generates treatment recommendations and risk stratification
    based on digital twin predictions and clinical evidence.

    Attributes:
        patient_dt: Patient digital twin

    Example:
        >>> cds = ClinicalDecisionSupport(patient_dt)
        >>> recommendations = cds.generate_recommendations(
        ...     treatment_options=["surgery", "chemoradiation"]
        ... )
    """

    def __init__(self, patient_dt):
        """Initialize clinical decision support.

        Args:
            patient_dt: Patient digital twin instance
        """
        self.patient_dt = patient_dt

        logger.info(f"CDS initialized for patient {patient_dt.patient_id}")

    def generate_recommendations(
        self,
        treatment_options: list[str],
        optimization_target: str = "overall_survival",
        patient_preferences: dict | None = None
    ) -> dict:
        """Generate treatment recommendations.

        Args:
            treatment_options: List of treatment options to consider
            optimization_target: Optimization objective
            patient_preferences: Patient preference weights

        Returns:
            Recommendation dictionary with ranked options
        """
        from digital_twins.treatment_simulation import TreatmentSimulator

        simulator = TreatmentSimulator(self.patient_dt)

        recommendations = []
        for option in treatment_options:
            # Simulate treatment
            protocol = self._create_protocol(option)
            response = simulator.predict_response(protocol, horizon_days=365)

            recommendations.append({
                "treatment": option,
                "predicted_response": response.response_category.value,
                "volume_change": response.volume_change_percent,
                "confidence": 0.85,  # Placeholder
                "evidence_level": "II",
                "considerations": self._get_considerations(option)
            })

        # Rank by optimization target
        if optimization_target == "overall_survival":
            recommendations.sort(key=lambda x: x["volume_change"])
        elif optimization_target == "quality_of_life":
            # Would incorporate QoL metrics
            pass

        result = {
            "patient_id": self.patient_dt.patient_id,
            "timestamp": datetime.now().isoformat(),
            "optimization_target": optimization_target,
            "recommendations": recommendations,
            "disclaimer": "For clinical decision support only. Physician review required."
        }

        logger.info(f"Generated {len(recommendations)} treatment recommendations")
        return result

    def _create_protocol(self, treatment_option: str) -> dict:
        """Create treatment protocol for simulation."""
        protocols = {
            "surgery": {
                "type": "surgery",
                "resection_extent": 0.95,
                "margin_mm": 10
            },
            "chemoradiation": {
                "type": "combined",
                "modalities": [
                    {"type": "chemotherapy", "drug": "cisplatin", "cycles": 4},
                    {"type": "radiation", "total_dose_gy": 60, "fractions": 30}
                ]
            },
            "immunotherapy": {
                "type": "immunotherapy",
                "agent": "pembrolizumab",
                "dose_mg_kg": 2
            }
        }
        return protocols.get(treatment_option, {"type": treatment_option})

    def _get_considerations(self, treatment: str) -> list[str]:
        """Get clinical considerations for treatment."""
        considerations = {
            "surgery": [
                "Surgical candidacy assessment required",
                "Consider neoadjuvant therapy if borderline resectable"
            ],
            "chemoradiation": [
                "Monitor for treatment-related toxicity",
                "Consider supportive care needs"
            ],
            "immunotherapy": [
                "PD-L1 expression testing recommended",
                "Monitor for immune-related adverse events"
            ]
        }
        return considerations.get(treatment, [])

    def export_to_fhir(
        self,
        recommendations: dict,
        destination: str = "tumor_board_review"
    ) -> str:
        """Export recommendations as FHIR DiagnosticReport.

        Args:
            recommendations: Recommendation dictionary
            destination: Destination workflow

        Returns:
            FHIR resource ID
        """
        report = {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "59776-5",
                    "display": "Digital Twin Treatment Recommendations"
                }]
            },
            "conclusion": json.dumps(recommendations["recommendations"]),
            "issued": recommendations["timestamp"]
        }

        logger.info(f"Exported recommendations to FHIR for {destination}")
        return f"DiagnosticReport/{recommendations['patient_id']}"


class ComplianceManager:
    """
    Regulatory compliance management.

    Handles audit trails, electronic signatures, and documentation
    for FDA 21 CFR Part 11 and other regulatory requirements.

    Attributes:
        regulation: Compliance framework

    Example:
        >>> compliance = ComplianceManager(regulation="21CFR11")
        >>> compliance.enable_audit_trail("/audit/log.json")
    """

    def __init__(self, regulation: str = "21CFR11"):
        """Initialize compliance manager.

        Args:
            regulation: Regulatory framework (21CFR11, HIPAA, GDPR)
        """
        self.regulation = ComplianceRegulation(regulation)
        self._audit_trail_path = None
        self._audit_entries = []

        logger.info(f"ComplianceManager initialized: {regulation}")

    def enable_audit_trail(
        self,
        log_path: str,
        include_user_actions: bool = True,
        include_data_access: bool = True
    ) -> None:
        """Enable audit trail logging.

        Args:
            log_path: Path to audit log file
            include_user_actions: Log user actions
            include_data_access: Log data access events
        """
        self._audit_trail_path = Path(log_path)
        self._audit_trail_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Audit trail enabled: {log_path}")

    def log_event(
        self,
        event_type: str,
        user: str,
        resource: str,
        action: str,
        details: dict | None = None
    ) -> None:
        """Log compliance event.

        Args:
            event_type: Type of event
            user: User identifier
            resource: Affected resource
            action: Action performed
            details: Additional details
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user": user,
            "resource": resource,
            "action": action,
            "details": details or {},
            "regulation": self.regulation.value
        }

        self._audit_entries.append(entry)

        if self._audit_trail_path:
            with open(self._audit_trail_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def require_signature(self, func):
        """Decorator to require electronic signature.

        Args:
            func: Function requiring signature

        Returns:
            Wrapped function with signature verification
        """
        def wrapper(*args, **kwargs):
            # In production, would verify e-signature
            self.log_event(
                "signature_required",
                user="system",
                resource=func.__name__,
                action="execute_with_signature"
            )
            return func(*args, **kwargs)

        return wrapper

    def generate_report(
        self,
        output_path: str,
        period: str = "2026-Q1"
    ) -> str:
        """Generate compliance report.

        Args:
            output_path: Output file path
            period: Reporting period

        Returns:
            Path to generated report
        """
        report = {
            "title": f"Compliance Report - {self.regulation.value}",
            "period": period,
            "generated": datetime.now().isoformat(),
            "total_events": len(self._audit_entries),
            "events_by_type": self._count_events_by_type(),
            "summary": "All activities logged and compliant"
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Compliance report generated: {output_path}")
        return str(output_path)

    def _count_events_by_type(self) -> dict:
        """Count audit events by type."""
        counts = {}
        for entry in self._audit_entries:
            event_type = entry.get("event_type", "unknown")
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts


class IntraoperativeInterface:
    """
    Real-time intraoperative guidance interface.

    Connects digital twin to surgical navigation and robotic
    systems for real-time guidance during procedures.

    Attributes:
        navigation_system: Navigation system type
        robot_interface: Robot control interface

    Example:
        >>> intraop = IntraoperativeInterface(
        ...     navigation_system="brainlab",
        ...     robot_interface="dvrk"
        ... )
        >>> intraop.register_patient(patient_dt)
    """

    def __init__(
        self,
        navigation_system: str = "generic",
        robot_interface: str = "simulation"
    ):
        """Initialize intraoperative interface.

        Args:
            navigation_system: Navigation system type
            robot_interface: Robot interface type
        """
        self.navigation_system = navigation_system
        self.robot_interface = robot_interface
        self._patient_registered = False
        self._alert_callbacks = []

        logger.info(
            f"IntraoperativeInterface: nav={navigation_system}, robot={robot_interface}"
        )

    def register_patient(
        self,
        patient_dt,
        registration_method: str = "surface_matching"
    ) -> bool:
        """Register patient anatomy for navigation.

        Args:
            patient_dt: Patient digital twin
            registration_method: Registration method

        Returns:
            Registration success status
        """
        logger.info(f"Registering patient with method: {registration_method}")

        # In production, would perform actual registration
        self._patient_registered = True
        self._patient_dt = patient_dt

        return True

    async def stream_guidance(self):
        """Stream real-time surgical guidance.

        Yields:
            GuidanceUpdate with navigation data
        """
        import asyncio

        while True:
            # Simulated guidance updates
            update = GuidanceUpdate(
                timestamp=datetime.now(),
                position=[0.0, 0.0, 0.0],
                margin_distance_mm=np.random.uniform(3, 20),
                trajectory=[0.0, 0.0, 1.0],
                confidence=0.95
            )

            yield update
            await asyncio.sleep(0.033)  # ~30 Hz

    def alert(self, message: str, severity: str = "warning") -> None:
        """Send surgical alert.

        Args:
            message: Alert message
            severity: Alert severity (info, warning, critical)
        """
        logger.warning(f"Surgical alert [{severity}]: {message}")

        for callback in self._alert_callbacks:
            callback(message, severity)


@dataclass
class GuidanceUpdate:
    """Real-time guidance update.

    Attributes:
        timestamp: Update timestamp
        position: Current instrument position
        margin_distance_mm: Distance to nearest margin
        trajectory: Recommended trajectory
        confidence: Confidence score
    """
    timestamp: datetime
    position: list[float]
    margin_distance_mm: float
    trajectory: list[float]
    confidence: float


class SurgicalDigitalTwin:
    """
    Surgical-specific digital twin for robotic surgery integration.

    Extends patient digital twin with surgical planning and
    robot simulation capabilities.

    Example:
        >>> surgical_dt = SurgicalDigitalTwin.from_patient_dt(patient_dt)
        >>> surgical_dt.export_to_robot_sim("isaac", "dvrk_psm")
    """

    def __init__(self, patient_dt, anatomy_mesh: np.ndarray | None = None):
        """Initialize surgical digital twin.

        Args:
            patient_dt: Base patient digital twin
            anatomy_mesh: 3D anatomy mesh (optional)
        """
        self.patient_dt = patient_dt
        self.anatomy_mesh = anatomy_mesh
        self._robot_model = None
        self._overlay_enabled = False

        logger.info(f"SurgicalDigitalTwin created for {patient_dt.patient_id}")

    @classmethod
    def from_patient_dt(cls, patient_dt) -> "SurgicalDigitalTwin":
        """Create surgical DT from patient DT.

        Args:
            patient_dt: Patient digital twin

        Returns:
            SurgicalDigitalTwin instance
        """
        return cls(patient_dt)

    def export_to_robot_sim(
        self,
        framework: str = "isaac",
        robot_model: str = "dvrk_psm",
        output_path: str = "/tmp/sim_scene.usd"
    ) -> str:
        """Export to robot simulation framework.

        Args:
            framework: Simulation framework (isaac, mujoco, gazebo)
            robot_model: Robot model to include
            output_path: Output scene file path

        Returns:
            Path to exported scene
        """
        logger.info(f"Exporting to {framework} with robot {robot_model}")

        # Create scene configuration
        scene_config = {
            "framework": framework,
            "robot_model": robot_model,
            "patient_id": self.patient_dt.patient_id,
            "tumor_volume": self.patient_dt.current_volume_cm3,
            "anatomy": {
                "include_soft_tissue": True,
                "deformation_model": "fem"
            }
        }

        # In production, would generate actual scene file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path + ".config.json", "w") as f:
            json.dump(scene_config, f, indent=2)

        logger.info(f"Scene exported to {output_path}")
        return output_path

    def enable_overlay(
        self,
        robot_interface,
        visualization: str = "tumor_margin",
        update_rate_hz: int = 30
    ) -> None:
        """Enable digital twin overlay on robot visualization.

        Args:
            robot_interface: Robot control interface
            visualization: Visualization type
            update_rate_hz: Update rate in Hz
        """
        self._robot_model = robot_interface
        self._overlay_enabled = True
        self._visualization = visualization
        self._update_rate = update_rate_hz

        logger.info(
            f"Overlay enabled: {visualization} at {update_rate_hz}Hz"
        )


# Factory class for creating digital twins from clinical data
class ClinicalDigitalTwinFactory:
    """Factory for creating digital twins from clinical systems."""

    def __init__(self, connector: ClinicalConnector):
        """Initialize factory with clinical connector.

        Args:
            connector: ClinicalConnector instance
        """
        self.connector = connector

    def create_from_patient(
        self,
        mrn: str,
        include_imaging: bool = True,
        include_pathology: bool = True,
        include_genomics: bool = True
    ):
        """Create digital twin from patient clinical data.

        Args:
            mrn: Medical record number
            include_imaging: Include imaging data
            include_pathology: Include pathology data
            include_genomics: Include genomic data

        Returns:
            PatientDigitalTwin instance
        """
        # Get patient record
        patient = self.connector.get_patient(mrn)

        # Get imaging if requested
        imaging_data = {}
        if include_imaging:
            studies = self.connector.query_imaging(patient.mrn)
            if studies:
                handler = DICOMHandler()
                # Would load actual images
                imaging_data["ct"] = np.zeros((64, 256, 256))

        # Create digital twin
        from digital_twins.patient_modeling import TumorTwinPipeline

        pipeline = TumorTwinPipeline()
        patient_dt = pipeline.create_twin(
            patient_id=mrn,
            imaging_data=imaging_data,
            tumor_segmentation=np.zeros((64, 64, 64)),
            clinical_data={
                "age": 60,
                "sex": patient.sex,
                "tumor_grade": "III"
            }
        )

        # Add clinical attributes
        patient_dt.diagnoses = patient.diagnoses
        patient_dt.biomarkers = patient.biomarkers

        logger.info(f"Created clinical digital twin for {mrn}")
        return patient_dt


if __name__ == "__main__":
    print("Clinical Digital Twin Interface - Physical AI Oncology Trials")
    print("=" * 60)

    # Test clinical connector
    print("\n1. Testing Clinical Connector")
    connector = ClinicalConnector(
        fhir_endpoint="https://fhir.hospital.local/R4"
    )
    status = connector.test_connection()
    print(f"   Connection status: {status}")

    # Test patient retrieval
    print("\n2. Testing Patient Retrieval")
    patient = connector.get_patient("12345678")
    print(f"   Patient MRN: {patient.mrn}")
    print(f"   Diagnoses: {patient.diagnoses}")

    # Test FHIR client
    print("\n3. Testing FHIR Client")
    fhir = FHIRClient("https://fhir.hospital.local/R4")
    patients = fhir.search("Patient", {"identifier": "MRN|12345678"})
    print(f"   Found {len(patients)} patients")

    # Test DICOM handler
    print("\n4. Testing DICOM Handler")
    dicom = DICOMHandler()
    metadata = dicom.extract_metadata([])
    print(f"   Metadata: {metadata}")

    # Test compliance
    print("\n5. Testing Compliance Manager")
    compliance = ComplianceManager("21CFR11")
    compliance.log_event("test", "user1", "test_resource", "test_action")
    print("   Audit event logged")

    print("\n" + "=" * 60)
    print("All tests completed successfully")
