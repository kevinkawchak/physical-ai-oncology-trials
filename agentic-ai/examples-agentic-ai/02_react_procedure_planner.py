"""
=============================================================================
EXAMPLE 02: ReAct Procedure Planner for Robotic Oncology Surgery
=============================================================================

Implements a ReAct (Reasoning + Acting) agent that plans and reasons
through robotic oncology surgical procedures step by step, integrating
patient anatomy, intraoperative imaging, and safety constraints.

CLINICAL CONTEXT:
-----------------
Robotic oncology surgery requires systematic planning that accounts for:
  - Patient-specific anatomy from preoperative imaging
  - Tumor characteristics (size, location, vascularity, margins)
  - Critical structure proximity and keep-out zone definition
  - Instrument selection and port placement optimization
  - Contingency planning for intraoperative findings

REACT PATTERN:
--------------
The ReAct pattern alternates between:
  1. THOUGHT: Reason about the current state and next action
  2. ACTION: Execute a tool or query to gather information
  3. OBSERVATION: Process the result of the action
  4. Repeat until the plan is complete

This is distinct from simple multi-agent orchestration (covered in
examples/04_agentic_clinical_workflow.py) because the agent maintains
an internal reasoning chain that builds context across steps.

ARCHITECTURE:
-------------
  Patient Data --> ReAct Agent --> Structured Procedure Plan
                      |
                  +---+---+---+
                  |   |   |   |
               Imaging Anatomy Safety Protocol
               Tools   DB   Checker Lookup

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - anthropic >= 0.40.0 or openai >= 1.60.0

Optional:
    - numpy >= 1.24.0 (spatial calculations)
    - scipy >= 1.11.0 (optimization)

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: ANATOMICAL AND PROCEDURE DATA MODELS
# =============================================================================


class TumorType(Enum):
    """Solid tumor classifications relevant to robotic surgery."""

    NSCLC = "non_small_cell_lung_cancer"
    RENAL_CELL = "renal_cell_carcinoma"
    PROSTATE = "prostate_adenocarcinoma"
    COLORECTAL = "colorectal_adenocarcinoma"
    HEPATOCELLULAR = "hepatocellular_carcinoma"
    PANCREATIC = "pancreatic_ductal_adenocarcinoma"


class RiskLevel(Enum):
    """Surgical risk classification."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnatomicalStructure:
    """Anatomical structure identified from imaging."""

    name: str
    structure_type: str  # tumor, vessel, organ, nerve, bone
    centroid: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    volume_cc: float = 0.0
    risk_if_damaged: RiskLevel = RiskLevel.MODERATE
    min_clearance_mm: float = 5.0
    segmentation_confidence: float = 0.95


@dataclass
class PatientAnatomy:
    """Patient-specific anatomical model from imaging."""

    patient_id: str
    structures: list = field(default_factory=list)
    tumor: Optional[AnatomicalStructure] = None
    imaging_date: str = ""
    imaging_modality: str = "CT"
    registration_error_mm: float = 1.0

    def get_critical_structures(self, max_distance_mm: float = 20.0) -> list:
        """Get structures within distance of tumor."""
        if self.tumor is None:
            return []
        critical = []
        for structure in self.structures:
            dist = _euclidean_distance(self.tumor.centroid, structure.centroid)
            if dist <= max_distance_mm / 10.0:  # Convert mm to cm (centroid units)
                critical.append(
                    {
                        "name": structure.name,
                        "type": structure.structure_type,
                        "distance_mm": dist * 10.0,
                        "risk": structure.risk_if_damaged.value,
                        "min_clearance_mm": structure.min_clearance_mm,
                    }
                )
        return critical


@dataclass
class SurgicalInstrument:
    """Robotic surgical instrument specification."""

    name: str
    instrument_type: str  # grasper, scissors, stapler, energy_device, scope
    diameter_mm: float = 8.0
    working_length_mm: float = 400.0
    articulation_degrees: float = 60.0
    energy_type: str = "none"  # none, monopolar, bipolar, ultrasonic, laser


@dataclass
class ProcedureStep:
    """A single step in the surgical procedure plan."""

    step_number: int
    action: str
    rationale: str
    instruments_needed: list = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    estimated_duration_minutes: int = 5
    safety_checks: list = field(default_factory=list)
    contingencies: list = field(default_factory=list)
    anatomical_targets: list = field(default_factory=list)


@dataclass
class ProcedurePlan:
    """Complete surgical procedure plan."""

    procedure_name: str
    patient_id: str
    steps: list = field(default_factory=list)
    total_estimated_minutes: int = 0
    overall_risk: RiskLevel = RiskLevel.MODERATE
    required_instruments: list = field(default_factory=list)
    critical_structures: list = field(default_factory=list)
    contingency_plans: list = field(default_factory=list)
    reasoning_trace: list = field(default_factory=list)


# =============================================================================
# SECTION 2: REACT AGENT TOOLS
# =============================================================================


def _euclidean_distance(a: list, b: list) -> float:
    """Calculate Euclidean distance between two 3D points."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class ProcedurePlanningTools:
    """
    Tool implementations for the ReAct procedure planner.

    Each tool returns structured data that the agent uses to
    reason about the next step in planning.
    """

    def __init__(self, anatomy: PatientAnatomy):
        self.anatomy = anatomy
        self._instrument_catalog = self._init_instruments()
        self._protocol_db = self._init_protocols()

    def _init_instruments(self) -> dict[str, SurgicalInstrument]:
        """Initialize available instrument catalog."""
        return {
            "maryland_bipolar": SurgicalInstrument(
                name="Maryland Bipolar Forceps",
                instrument_type="grasper",
                diameter_mm=8.0,
                articulation_degrees=60.0,
                energy_type="bipolar",
            ),
            "monopolar_scissors": SurgicalInstrument(
                name="Monopolar Curved Scissors",
                instrument_type="scissors",
                diameter_mm=8.0,
                articulation_degrees=60.0,
                energy_type="monopolar",
            ),
            "cadiere_forceps": SurgicalInstrument(
                name="Cadiere Forceps",
                instrument_type="grasper",
                diameter_mm=8.0,
                articulation_degrees=60.0,
            ),
            "prograsp_forceps": SurgicalInstrument(
                name="ProGrasp Forceps",
                instrument_type="grasper",
                diameter_mm=8.0,
                articulation_degrees=60.0,
            ),
            "endowrist_stapler": SurgicalInstrument(
                name="EndoWrist Stapler 45",
                instrument_type="stapler",
                diameter_mm=12.0,
                working_length_mm=450.0,
            ),
            "vessel_sealer": SurgicalInstrument(
                name="Vessel Sealer Extend+",
                instrument_type="energy_device",
                diameter_mm=8.0,
                energy_type="bipolar",
            ),
            "endoscope_30": SurgicalInstrument(
                name="30-degree Endoscope",
                instrument_type="scope",
                diameter_mm=8.0,
            ),
            "harmonic_scalpel": SurgicalInstrument(
                name="Harmonic ACE+7 Shears",
                instrument_type="energy_device",
                diameter_mm=5.0,
                energy_type="ultrasonic",
            ),
        }

    def _init_protocols(self) -> dict[str, dict]:
        """Initialize surgical protocol database."""
        return {
            "lobectomy": {
                "name": "Robotic-Assisted Thoracoscopic Lobectomy",
                "required_steps": [
                    "Port placement",
                    "Adhesiolysis if needed",
                    "Hilar dissection",
                    "Vascular ligation",
                    "Bronchus division",
                    "Fissure completion",
                    "Lymph node dissection",
                    "Specimen extraction",
                ],
                "contraindications": [
                    "Tumor invading chest wall (T3)",
                    "N2 disease not downstaged",
                    "FEV1 < 0.8L predicted postoperative",
                ],
                "margin_requirement_mm": 20.0,
            },
            "partial_nephrectomy": {
                "name": "Robotic Partial Nephrectomy",
                "required_steps": [
                    "Colon mobilization",
                    "Renal hilum identification",
                    "Hilar vessel control",
                    "Tumor scoring and excision",
                    "Renorrhaphy",
                    "Hilum unclamping",
                ],
                "contraindications": [
                    "Tumor in solitary kidney with high complexity (RENAL score > 10)",
                    "Renal vein thrombus extending to IVC",
                ],
                "margin_requirement_mm": 5.0,
                "max_warm_ischemia_minutes": 25,
            },
            "prostatectomy": {
                "name": "Robotic-Assisted Radical Prostatectomy",
                "required_steps": [
                    "Bladder drop and space of Retzius development",
                    "Endopelvic fascia incision",
                    "Dorsal venous complex ligation",
                    "Bladder neck dissection",
                    "Seminal vesicle dissection",
                    "Nerve-sparing dissection",
                    "Apical dissection and urethral division",
                    "Vesicourethral anastomosis",
                ],
                "contraindications": [
                    "Locally advanced T4 disease",
                    "Uncorrected coagulopathy",
                ],
                "margin_requirement_mm": 1.0,
            },
        }

    # --- Tools available to the ReAct agent ---

    def analyze_tumor(self) -> dict:
        """Analyze tumor characteristics from imaging data."""
        if self.anatomy.tumor is None:
            return {"error": "No tumor segmentation available"}

        tumor = self.anatomy.tumor
        return {
            "tumor_type": tumor.structure_type,
            "location": tumor.name,
            "centroid_cm": tumor.centroid,
            "volume_cc": tumor.volume_cc,
            "segmentation_confidence": tumor.segmentation_confidence,
            "imaging_modality": self.anatomy.imaging_modality,
            "registration_error_mm": self.anatomy.registration_error_mm,
        }

    def identify_critical_structures(self, proximity_mm: float = 20.0) -> dict:
        """Identify critical structures near the tumor."""
        critical = self.anatomy.get_critical_structures(max_distance_mm=proximity_mm)
        return {
            "proximity_threshold_mm": proximity_mm,
            "critical_structure_count": len(critical),
            "structures": critical,
            "highest_risk": max((s["risk"] for s in critical), default="none"),
        }

    def lookup_protocol(self, procedure_type: str) -> dict:
        """Look up surgical protocol for the given procedure type."""
        protocol = self._protocol_db.get(procedure_type)
        if protocol is None:
            available = list(self._protocol_db.keys())
            return {"error": f"Protocol not found: {procedure_type}", "available_protocols": available}
        return protocol

    def select_instruments(self, step_requirements: list[str]) -> dict:
        """Select optimal instruments for given surgical requirements."""
        selected = []
        for requirement in step_requirements:
            req_lower = requirement.lower()
            if "grasp" in req_lower or "retract" in req_lower:
                selected.append(self._instrument_catalog["cadiere_forceps"])
            elif "cut" in req_lower or "dissect" in req_lower:
                selected.append(self._instrument_catalog["monopolar_scissors"])
            elif "seal" in req_lower or "coagulat" in req_lower:
                selected.append(self._instrument_catalog["vessel_sealer"])
            elif "stapl" in req_lower or "divid" in req_lower:
                selected.append(self._instrument_catalog["endowrist_stapler"])
            elif "visualiz" in req_lower or "scope" in req_lower:
                selected.append(self._instrument_catalog["endoscope_30"])

        return {
            "selected_instruments": [
                {"name": inst.name, "type": inst.instrument_type, "energy": inst.energy_type} for inst in selected
            ],
            "total_ports_needed": len(set(inst.name for inst in selected)) + 1,
        }

    def evaluate_approach_safety(self, approach_vector: list[float], target: list[float]) -> dict:
        """Evaluate safety of a surgical approach vector."""
        structures_in_path = []

        for structure in self.anatomy.structures:
            # Simplified: check if structure is between approach and target
            dist_to_line = self._point_to_line_distance(structure.centroid, approach_vector, target)
            if dist_to_line < structure.min_clearance_mm / 10.0:
                structures_in_path.append(
                    {
                        "name": structure.name,
                        "clearance_mm": dist_to_line * 10.0,
                        "required_clearance_mm": structure.min_clearance_mm,
                        "risk": structure.risk_if_damaged.value,
                        "safe": dist_to_line * 10.0 >= structure.min_clearance_mm,
                    }
                )

        all_safe = all(s["safe"] for s in structures_in_path) if structures_in_path else True
        return {
            "approach_vector": approach_vector,
            "target": target,
            "structures_evaluated": len(structures_in_path),
            "structures_in_path": structures_in_path,
            "approach_safe": all_safe,
            "risk_level": "low" if all_safe else "high",
        }

    def estimate_margin(self, resection_boundary: list[list[float]]) -> dict:
        """Estimate surgical margins from proposed resection boundary."""
        if self.anatomy.tumor is None:
            return {"error": "No tumor data available"}

        # Calculate minimum distance from tumor surface to resection boundary
        min_margin = float("inf")
        for point in resection_boundary:
            dist = _euclidean_distance(self.anatomy.tumor.centroid, point)
            tumor_radius = (self.anatomy.tumor.volume_cc * 3 / (4 * math.pi)) ** (1 / 3)
            margin = (dist - tumor_radius) * 10.0  # Convert to mm
            min_margin = min(min_margin, margin)

        return {
            "minimum_margin_mm": round(max(0.0, min_margin), 1),
            "tumor_volume_cc": self.anatomy.tumor.volume_cc,
            "margin_adequate": min_margin >= 5.0,
            "recommendation": "Adequate margin" if min_margin >= 5.0 else "Consider wider resection",
        }

    @staticmethod
    def _point_to_line_distance(point: list, line_start: list, line_end: list) -> float:
        """Calculate distance from a point to a line segment in 3D."""
        if len(point) < 3 or len(line_start) < 3 or len(line_end) < 3:
            return float("inf")

        # Vector from line_start to point
        ap = [point[i] - line_start[i] for i in range(3)]
        # Vector from line_start to line_end
        ab = [line_end[i] - line_start[i] for i in range(3)]

        ab_sq = sum(x * x for x in ab)
        if ab_sq == 0:
            return _euclidean_distance(point, line_start)

        t = max(0, min(1, sum(ap[i] * ab[i] for i in range(3)) / ab_sq))
        closest = [line_start[i] + t * ab[i] for i in range(3)]
        return _euclidean_distance(point, closest)


# =============================================================================
# SECTION 3: REACT AGENT ENGINE
# =============================================================================


class ReActStep:
    """A single step in the ReAct reasoning chain."""

    def __init__(self, step_type: str, content: str, data: Any = None):
        self.step_type = step_type  # thought, action, observation
        self.content = content
        self.data = data
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        """Serialize for audit trail."""
        return {
            "type": self.step_type,
            "content": self.content,
            "timestamp": self.timestamp,
        }


class ReActProcedurePlanner:
    """
    ReAct agent for surgical procedure planning.

    Uses the Reasoning + Acting pattern to build a procedure plan
    through iterative tool use and chain-of-thought reasoning.

    PLANNING PROCESS:
    -----------------
    1. Analyze tumor and patient anatomy
    2. Identify critical structures and constraints
    3. Select appropriate surgical protocol
    4. Plan each step with instrument selection
    5. Evaluate approach safety for each step
    6. Define contingency plans
    7. Estimate margins and verify adequacy

    SAFETY PRINCIPLES:
    ------------------
    - Every step includes explicit safety checks
    - Critical structures are identified and tracked
    - Contingency plans are defined for foreseeable complications
    - Margin adequacy is verified before plan finalization
    """

    def __init__(self, anatomy: PatientAnatomy, procedure_type: str):
        self.anatomy = anatomy
        self.procedure_type = procedure_type
        self.tools = ProcedurePlanningTools(anatomy)
        self.reasoning_trace: list[ReActStep] = []
        self._step_count = 0

    def _think(self, thought: str) -> None:
        """Record a reasoning step."""
        self._step_count += 1
        step = ReActStep("thought", thought)
        self.reasoning_trace.append(step)
        logger.info(f"[THOUGHT {self._step_count}] {thought}")

    def _act(self, action: str, tool_name: str, tool_args: dict = None) -> Any:
        """Execute a tool and record the action."""
        self._step_count += 1
        step = ReActStep("action", f"{action} -> {tool_name}")
        self.reasoning_trace.append(step)
        logger.info(f"[ACTION {self._step_count}] {action}")

        # Execute tool
        tool_fn = getattr(self.tools, tool_name, None)
        if tool_fn is None:
            return {"error": f"Tool not found: {tool_name}"}

        if tool_args:
            result = tool_fn(**tool_args)
        else:
            result = tool_fn()

        return result

    def _observe(self, observation: str, data: Any = None) -> None:
        """Record an observation from tool output."""
        self._step_count += 1
        step = ReActStep("observation", observation, data)
        self.reasoning_trace.append(step)
        logger.info(f"[OBSERVATION {self._step_count}] {observation}")

    def plan_procedure(self) -> ProcedurePlan:
        """
        Execute the ReAct planning loop to generate a procedure plan.

        Returns:
            Complete procedure plan with reasoning trace
        """
        plan = ProcedurePlan(
            procedure_name=self.procedure_type,
            patient_id=self.anatomy.patient_id,
        )

        # Step 1: Analyze tumor
        self._think(
            "I need to first understand the tumor characteristics to plan the procedure. "
            "Let me analyze the tumor from the imaging data."
        )
        tumor_data = self._act("Analyzing tumor characteristics", "analyze_tumor")
        self._observe(
            f"Tumor located at {tumor_data.get('location', 'unknown')}, "
            f"volume {tumor_data.get('volume_cc', 0)} cc, "
            f"confidence {tumor_data.get('segmentation_confidence', 0):.0%}"
        )

        # Step 2: Identify critical structures
        self._think(
            "Now I need to identify critical structures near the tumor that must be "
            "preserved during resection. I'll check within 30mm of the tumor."
        )
        critical = self._act(
            "Identifying critical structures within 30mm",
            "identify_critical_structures",
            {"proximity_mm": 30.0},
        )
        plan.critical_structures = critical.get("structures", [])
        self._observe(
            f"Found {critical['critical_structure_count']} critical structures. "
            f"Highest risk level: {critical['highest_risk']}"
        )

        # Step 3: Look up protocol
        self._think(
            f"I'll look up the standard protocol for {self.procedure_type} "
            "to ensure all required steps are included in the plan."
        )
        protocol = self._act(
            f"Looking up protocol for {self.procedure_type}",
            "lookup_protocol",
            {"procedure_type": self.procedure_type},
        )

        if "error" in protocol:
            self._observe(f"Protocol lookup failed: {protocol['error']}")
            self._think("I'll use the first available protocol as a template.")
            protocol = self._act(
                "Using lobectomy protocol as template", "lookup_protocol", {"procedure_type": "lobectomy"}
            )

        self._observe(
            f"Protocol: {protocol.get('name', 'unknown')}. Required steps: {len(protocol.get('required_steps', []))}"
        )

        # Step 4: Check contraindications
        self._think(
            "Before proceeding with planning, I need to verify there are no contraindications for this procedure."
        )
        contraindications = protocol.get("contraindications", [])
        self._observe(
            f"Contraindications to verify: {len(contraindications)}. "
            "Assuming clinical team has cleared patient for surgery."
        )

        # Step 5: Plan each surgical step
        self._think(
            "Now I'll plan each step of the procedure, selecting appropriate "
            "instruments and defining safety checks for each."
        )

        required_steps = protocol.get("required_steps", [])
        step_number = 0
        all_instruments = set()

        for step_desc in required_steps:
            step_number += 1

            # Determine instruments needed
            requirements = self._determine_step_requirements(step_desc)
            instruments = self._act(
                f"Selecting instruments for: {step_desc}",
                "select_instruments",
                {"step_requirements": requirements},
            )

            selected = instruments.get("selected_instruments", [])
            for inst in selected:
                all_instruments.add(inst["name"])

            # Define safety checks
            safety_checks = self._generate_safety_checks(step_desc, plan.critical_structures)

            # Define contingencies
            contingencies = self._generate_contingencies(step_desc)

            step = ProcedureStep(
                step_number=step_number,
                action=step_desc,
                rationale=f"Required step per {protocol.get('name', 'protocol')}",
                instruments_needed=[inst["name"] for inst in selected],
                risk_level=self._assess_step_risk(step_desc, plan.critical_structures),
                estimated_duration_minutes=self._estimate_duration(step_desc),
                safety_checks=safety_checks,
                contingencies=contingencies,
            )
            plan.steps.append(step)

            self._observe(
                f"Step {step_number}: {step_desc} - "
                f"Risk: {step.risk_level.value}, "
                f"Duration: {step.estimated_duration_minutes}min, "
                f"Instruments: {len(selected)}"
            )

        # Step 6: Evaluate approach safety
        self._think(
            "I'll evaluate the overall approach safety by checking the primary "
            "approach vector against critical structures."
        )
        if self.anatomy.tumor:
            approach = self._act(
                "Evaluating approach safety",
                "evaluate_approach_safety",
                {
                    "approach_vector": [0.0, 0.0, 0.3],
                    "target": self.anatomy.tumor.centroid,
                },
            )
            self._observe(
                f"Approach safety: {'SAFE' if approach['approach_safe'] else 'UNSAFE'}. "
                f"Structures in path: {approach['structures_evaluated']}"
            )

        # Step 7: Estimate margins
        self._think("Finally, I'll verify that the planned resection margins are adequate.")
        if self.anatomy.tumor:
            margin_boundary = [
                [c + 0.02 for c in self.anatomy.tumor.centroid],
                [c - 0.02 for c in self.anatomy.tumor.centroid],
                [self.anatomy.tumor.centroid[0] + 0.02, self.anatomy.tumor.centroid[1], self.anatomy.tumor.centroid[2]],
            ]
            margins = self._act(
                "Estimating surgical margins",
                "estimate_margin",
                {"resection_boundary": margin_boundary},
            )
            self._observe(f"Minimum margin: {margins['minimum_margin_mm']}mm. Status: {margins['recommendation']}")

        # Finalize plan
        plan.total_estimated_minutes = sum(s.estimated_duration_minutes for s in plan.steps)
        plan.required_instruments = list(all_instruments)
        plan.overall_risk = max(
            (s.risk_level for s in plan.steps),
            key=lambda r: list(RiskLevel).index(r),
            default=RiskLevel.MODERATE,
        )
        plan.reasoning_trace = [step.to_dict() for step in self.reasoning_trace]

        self._think(
            f"Procedure plan complete. {len(plan.steps)} steps, "
            f"estimated {plan.total_estimated_minutes} minutes, "
            f"overall risk: {plan.overall_risk.value}"
        )

        return plan

    def _determine_step_requirements(self, step_description: str) -> list[str]:
        """Determine instrument requirements from step description."""
        desc_lower = step_description.lower()
        requirements = []

        if "dissect" in desc_lower or "mobiliz" in desc_lower:
            requirements.extend(["dissection", "grasping", "visualization"])
        if "ligat" in desc_lower or "seal" in desc_lower or "vessel" in desc_lower:
            requirements.extend(["sealing", "grasping"])
        if "divis" in desc_lower or "stapl" in desc_lower or "transect" in desc_lower:
            requirements.extend(["stapling", "grasping"])
        if "lymph" in desc_lower:
            requirements.extend(["grasping", "dissection"])
        if "extract" in desc_lower or "specimen" in desc_lower:
            requirements.extend(["grasping"])
        if "port" in desc_lower or "dock" in desc_lower:
            requirements.extend(["visualization"])
        if "sutur" in desc_lower or "anastomos" in desc_lower:
            requirements.extend(["grasping", "cutting"])

        return requirements if requirements else ["visualization", "grasping"]

    def _generate_safety_checks(self, step_description: str, critical_structures: list) -> list[str]:
        """Generate safety checks for a procedure step."""
        checks = ["Verify instrument tip position within workspace"]

        desc_lower = step_description.lower()
        if "vessel" in desc_lower or "arter" in desc_lower or "vein" in desc_lower:
            checks.append("Confirm vessel identification before ligation")
            checks.append("Verify proximal and distal control before division")

        if "bronch" in desc_lower or "airway" in desc_lower:
            checks.append("Confirm bronchial anatomy before division")
            checks.append("Verify stapler position and margin")

        if critical_structures:
            structure_names = [s["name"] for s in critical_structures[:3]]
            checks.append(f"Maintain clearance from: {', '.join(structure_names)}")

        checks.append("Monitor force/torque within safety limits")
        return checks

    def _generate_contingencies(self, step_description: str) -> list[str]:
        """Generate contingency plans for potential complications."""
        contingencies = []
        desc_lower = step_description.lower()

        if "vessel" in desc_lower or "ligat" in desc_lower:
            contingencies.append("If bleeding: apply pressure, clip, or convert to open")
            contingencies.append("If vessel anatomy variant: pause and re-image")

        if "dissect" in desc_lower:
            contingencies.append("If unexpected adhesions: careful sharp dissection")
            contingencies.append("If anatomy unclear: obtain frozen section")

        if "stapl" in desc_lower:
            contingencies.append("If stapler misfire: manual closure and re-staple")

        if not contingencies:
            contingencies.append("If unable to proceed safely: hold and reassess")

        return contingencies

    def _assess_step_risk(self, step_description: str, critical_structures: list) -> RiskLevel:
        """Assess risk level of a procedure step."""
        desc_lower = step_description.lower()

        high_risk_keywords = ["arter", "vein", "vessel", "ligat", "stapl", "divid"]
        moderate_risk_keywords = ["dissect", "mobiliz", "lymph"]

        if any(kw in desc_lower for kw in high_risk_keywords):
            if critical_structures:
                return RiskLevel.HIGH
            return RiskLevel.MODERATE

        if any(kw in desc_lower for kw in moderate_risk_keywords):
            return RiskLevel.MODERATE

        return RiskLevel.LOW

    def _estimate_duration(self, step_description: str) -> int:
        """Estimate duration in minutes for a procedure step."""
        desc_lower = step_description.lower()

        if "port" in desc_lower or "dock" in desc_lower:
            return 15
        if "lymph" in desc_lower:
            return 20
        if "dissect" in desc_lower:
            return 20
        if "extract" in desc_lower or "specimen" in desc_lower:
            return 15
        if "ligat" in desc_lower or "seal" in desc_lower:
            return 10
        if "stapl" in desc_lower or "divid" in desc_lower:
            return 10
        if "anastomos" in desc_lower or "sutur" in desc_lower:
            return 25
        return 10


# =============================================================================
# SECTION 4: PLAN OUTPUT FORMATTER
# =============================================================================


def format_procedure_plan(plan: ProcedurePlan) -> str:
    """Format procedure plan as a readable report."""
    lines = [
        "=" * 70,
        "ROBOTIC ONCOLOGY PROCEDURE PLAN",
        "=" * 70,
        f"Procedure: {plan.procedure_name}",
        f"Patient ID: {plan.patient_id}",
        f"Overall Risk: {plan.overall_risk.value.upper()}",
        f"Estimated Duration: {plan.total_estimated_minutes} minutes",
        f"Reasoning Steps: {len(plan.reasoning_trace)}",
        "",
        "REQUIRED INSTRUMENTS:",
    ]

    for inst in plan.required_instruments:
        lines.append(f"  - {inst}")

    lines.append("")
    lines.append("CRITICAL STRUCTURES:")
    for struct in plan.critical_structures:
        lines.append(f"  - {struct['name']} ({struct['risk']} risk, {struct['distance_mm']:.1f}mm from tumor)")

    lines.append("")
    lines.append("PROCEDURE STEPS:")
    lines.append("-" * 70)

    for step in plan.steps:
        lines.append(f"\nStep {step.step_number}: {step.action}")
        lines.append(f"  Risk Level: {step.risk_level.value}")
        lines.append(f"  Duration: {step.estimated_duration_minutes} min")
        lines.append(f"  Rationale: {step.rationale}")

        if step.instruments_needed:
            lines.append(f"  Instruments: {', '.join(step.instruments_needed)}")

        if step.safety_checks:
            lines.append("  Safety Checks:")
            for check in step.safety_checks:
                lines.append(f"    [_] {check}")

        if step.contingencies:
            lines.append("  Contingencies:")
            for contingency in step.contingencies:
                lines.append(f"    -> {contingency}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("REASONING TRACE:")
    lines.append("-" * 70)
    for trace in plan.reasoning_trace:
        lines.append(f"  [{trace['type'].upper():11s}] {trace['content']}")

    return "\n".join(lines)


# =============================================================================
# SECTION 5: MAIN DEMO
# =============================================================================


def run_react_planner_demo():
    """
    Demonstrate the ReAct procedure planner for robotic lobectomy.

    Creates a patient anatomy model with tumor and critical structures,
    then runs the planner to generate a complete procedure plan.
    """
    logger.info("=" * 60)
    logger.info("REACT PROCEDURE PLANNER DEMO")
    logger.info("=" * 60)

    # Create patient anatomy
    anatomy = PatientAnatomy(
        patient_id="PT-2026-0042",
        imaging_date="2026-01-15",
        imaging_modality="CT",
        registration_error_mm=1.2,
    )

    # Define tumor
    anatomy.tumor = AnatomicalStructure(
        name="Right upper lobe mass",
        structure_type="tumor",
        centroid=[0.05, -0.02, 0.15],
        volume_cc=4.2,
        risk_if_damaged=RiskLevel.LOW,
        min_clearance_mm=0.0,
        segmentation_confidence=0.97,
    )

    # Define critical structures
    anatomy.structures = [
        AnatomicalStructure(
            name="Right pulmonary artery - truncus anterior",
            structure_type="vessel",
            centroid=[0.04, -0.01, 0.14],
            volume_cc=0.5,
            risk_if_damaged=RiskLevel.CRITICAL,
            min_clearance_mm=5.0,
        ),
        AnatomicalStructure(
            name="Right upper lobe bronchus",
            structure_type="airway",
            centroid=[0.045, -0.015, 0.145],
            volume_cc=0.3,
            risk_if_damaged=RiskLevel.HIGH,
            min_clearance_mm=3.0,
        ),
        AnatomicalStructure(
            name="Superior pulmonary vein",
            structure_type="vessel",
            centroid=[0.055, -0.025, 0.155],
            volume_cc=0.4,
            risk_if_damaged=RiskLevel.CRITICAL,
            min_clearance_mm=5.0,
        ),
        AnatomicalStructure(
            name="Azygos vein",
            structure_type="vessel",
            centroid=[0.03, 0.01, 0.16],
            volume_cc=0.6,
            risk_if_damaged=RiskLevel.HIGH,
            min_clearance_mm=5.0,
        ),
        AnatomicalStructure(
            name="Vagus nerve",
            structure_type="nerve",
            centroid=[0.035, 0.005, 0.155],
            volume_cc=0.1,
            risk_if_damaged=RiskLevel.MODERATE,
            min_clearance_mm=3.0,
        ),
    ]

    # Run ReAct planner
    planner = ReActProcedurePlanner(anatomy, "lobectomy")
    plan = planner.plan_procedure()

    # Print formatted plan
    report = format_procedure_plan(plan)
    print(report)

    return {
        "patient_id": anatomy.patient_id,
        "procedure": plan.procedure_name,
        "steps": len(plan.steps),
        "estimated_minutes": plan.total_estimated_minutes,
        "overall_risk": plan.overall_risk.value,
        "reasoning_steps": len(plan.reasoning_trace),
    }


if __name__ == "__main__":
    result = run_react_planner_demo()
    print(f"\nDemo result: {result}")
