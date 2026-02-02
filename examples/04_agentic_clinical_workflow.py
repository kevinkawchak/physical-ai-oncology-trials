"""
=============================================================================
EXAMPLE 04: Agentic AI for Clinical Trial Workflows
=============================================================================

This example demonstrates how to implement multi-agent systems for
coordinating complex clinical trial workflows in oncology.

CLINICAL CONTEXT:
-----------------
Multi-site oncology clinical trials require coordination of:
  - Patient enrollment and eligibility screening
  - Treatment protocol compliance
  - Data collection and quality assurance
  - Regulatory documentation
  - Adverse event reporting

Agentic AI enables:
  - Automated workflow orchestration
  - Natural language task specification
  - Intelligent decision support
  - Real-time multi-site coordination

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - CrewAI 1.6.1+ (https://github.com/crewAIInc/crewAI)
    - LangGraph 1.1.0+ (https://github.com/langchain-ai/langgraph)
    - anthropic SDK (for Claude integration)

Optional:
    - Model Context Protocol (MCP) tools
    - FHIR client for EHR integration

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: CLINICAL TRIAL DATA STRUCTURES
# =============================================================================

class TrialPhase(Enum):
    """Clinical trial phases."""
    PHASE_I = "Phase I"
    PHASE_II = "Phase II"
    PHASE_III = "Phase III"
    PHASE_IV = "Phase IV"


class PatientStatus(Enum):
    """Patient enrollment status."""
    SCREENING = "screening"
    ENROLLED = "enrolled"
    ACTIVE = "active_treatment"
    FOLLOW_UP = "follow_up"
    COMPLETED = "completed"
    WITHDRAWN = "withdrawn"


@dataclass
class ClinicalTrial:
    """Clinical trial metadata."""
    trial_id: str
    title: str
    phase: TrialPhase
    sponsor: str
    sites: list = field(default_factory=list)
    enrollment_target: int = 0
    current_enrollment: int = 0
    start_date: str = ""
    status: str = "recruiting"


@dataclass
class Patient:
    """Patient record for clinical trial."""
    patient_id: str
    site_id: str
    status: PatientStatus
    enrollment_date: str = ""
    treatment_arm: str = ""
    visits: list = field(default_factory=list)
    adverse_events: list = field(default_factory=list)


@dataclass
class AdverseEvent:
    """Adverse event record."""
    event_id: str
    patient_id: str
    description: str
    severity: str  # mild, moderate, severe, life_threatening
    onset_date: str
    resolved: bool = False
    reported_to_irb: bool = False
    reported_to_fda: bool = False


# =============================================================================
# SECTION 2: AGENT DEFINITIONS
# =============================================================================

class AgentRole(Enum):
    """Roles for clinical trial agents."""
    COORDINATOR = "trial_coordinator"
    ELIGIBILITY = "eligibility_specialist"
    DATA_MANAGER = "data_manager"
    SAFETY_OFFICER = "safety_officer"
    REGULATORY = "regulatory_specialist"
    SITE_MONITOR = "site_monitor"


@dataclass
class AgentConfig:
    """Configuration for a clinical trial agent."""
    role: AgentRole
    name: str
    goal: str
    backstory: str
    tools: list = field(default_factory=list)
    llm_model: str = "claude-sonnet-4-20250514"


class ClinicalTrialAgent:
    """
    Base class for clinical trial workflow agents.

    Each agent has a specific role and set of capabilities
    for managing aspects of the clinical trial.

    AGENT DESIGN PRINCIPLES:
    -----------------------
    1. Single responsibility per agent
    2. Clear handoff protocols between agents
    3. Audit trail for all decisions
    4. Human-in-the-loop for critical actions
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.role = config.role
        self.name = config.name
        self._action_history = []

        logger.info(f"Agent initialized: {self.name} ({self.role.value})")

    def process_task(self, task: str, context: dict) -> dict:
        """
        Process a task with given context.

        Args:
            task: Natural language task description
            context: Relevant context dictionary

        Returns:
            Result dictionary with action and output
        """
        logger.info(f"{self.name} processing: {task[:50]}...")

        # Record action for audit trail
        self._action_history.append({
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "context_keys": list(context.keys())
        })

        # Simulated processing - in production would call LLM
        result = self._execute_role_specific_logic(task, context)

        return result

    def _execute_role_specific_logic(self, task: str, context: dict) -> dict:
        """Execute role-specific processing logic."""
        # Override in subclasses
        return {"status": "completed", "output": "Task processed"}

    def get_audit_trail(self) -> list:
        """Get complete audit trail for this agent."""
        return self._action_history


class EligibilityAgent(ClinicalTrialAgent):
    """
    Agent for patient eligibility screening.

    Evaluates patient eligibility against trial inclusion/exclusion
    criteria using clinical data and laboratory results.
    """

    def __init__(self):
        config = AgentConfig(
            role=AgentRole.ELIGIBILITY,
            name="Dr. Eligibility",
            goal="Accurately screen patients against trial criteria",
            backstory="Expert in oncology trial eligibility with 15 years experience",
            tools=["check_lab_values", "review_medical_history", "verify_diagnosis"]
        )
        super().__init__(config)

        # Trial eligibility criteria
        self.inclusion_criteria = []
        self.exclusion_criteria = []

    def set_trial_criteria(
        self,
        inclusion: list,
        exclusion: list
    ):
        """Set eligibility criteria for the trial."""
        self.inclusion_criteria = inclusion
        self.exclusion_criteria = exclusion

    def _execute_role_specific_logic(self, task: str, context: dict) -> dict:
        """Screen patient eligibility."""
        patient_data = context.get("patient_data", {})

        # Check inclusion criteria
        inclusion_met = []
        for criterion in self.inclusion_criteria:
            met = self._check_criterion(criterion, patient_data)
            inclusion_met.append({"criterion": criterion, "met": met})

        # Check exclusion criteria
        exclusion_violated = []
        for criterion in self.exclusion_criteria:
            violated = self._check_criterion(criterion, patient_data)
            if violated:
                exclusion_violated.append(criterion)

        eligible = all(c["met"] for c in inclusion_met) and not exclusion_violated

        return {
            "status": "completed",
            "eligible": eligible,
            "inclusion_results": inclusion_met,
            "exclusion_violations": exclusion_violated,
            "recommendation": "Proceed with enrollment" if eligible else "Not eligible"
        }

    def _check_criterion(self, criterion: str, patient_data: dict) -> bool:
        """Check if patient meets a criterion."""
        # Simulated criterion checking
        return bool(hash(criterion + str(patient_data)) % 2)


class SafetyOfficerAgent(ClinicalTrialAgent):
    """
    Agent for safety monitoring and adverse event management.

    Monitors patient safety, evaluates adverse events, and
    triggers appropriate reporting workflows.
    """

    def __init__(self):
        config = AgentConfig(
            role=AgentRole.SAFETY_OFFICER,
            name="Safety Monitor",
            goal="Ensure patient safety and regulatory compliance for AE reporting",
            backstory="Pharmacovigilance expert with expertise in oncology safety",
            tools=["evaluate_ae", "report_to_irb", "report_to_fda", "recommend_action"]
        )
        super().__init__(config)

    def _execute_role_specific_logic(self, task: str, context: dict) -> dict:
        """Evaluate adverse event and determine actions."""
        ae = context.get("adverse_event", {})

        # Determine severity and required actions
        severity = ae.get("severity", "mild")

        actions = []
        if severity in ["severe", "life_threatening"]:
            actions.append("Notify principal investigator immediately")
            actions.append("Submit expedited report to IRB within 24 hours")
            if severity == "life_threatening":
                actions.append("Submit IND Safety Report to FDA within 7 days")

        if ae.get("unexpected", False):
            actions.append("Update Investigator's Brochure")

        return {
            "status": "completed",
            "severity_assessment": severity,
            "required_actions": actions,
            "reporting_timeline": self._get_reporting_timeline(severity),
            "recommendation": f"Handle as {severity} adverse event"
        }

    def _get_reporting_timeline(self, severity: str) -> dict:
        """Get regulatory reporting timelines."""
        timelines = {
            "mild": {"irb": "annual_report"},
            "moderate": {"irb": "30_days"},
            "severe": {"irb": "24_hours", "sponsor": "24_hours"},
            "life_threatening": {"irb": "24_hours", "fda": "7_days", "sponsor": "immediate"}
        }
        return timelines.get(severity, timelines["mild"])


class DataManagerAgent(ClinicalTrialAgent):
    """
    Agent for data quality and management.

    Ensures data completeness, resolves queries, and
    maintains trial database integrity.
    """

    def __init__(self):
        config = AgentConfig(
            role=AgentRole.DATA_MANAGER,
            name="Data Manager",
            goal="Maintain high-quality trial data and resolve discrepancies",
            backstory="Expert in clinical data management and EDC systems",
            tools=["validate_data", "generate_query", "resolve_discrepancy"]
        )
        super().__init__(config)

    def _execute_role_specific_logic(self, task: str, context: dict) -> dict:
        """Validate and manage trial data."""
        data = context.get("patient_data", {})

        # Data quality checks
        issues = []
        queries = []

        # Check for missing required fields
        required_fields = ["patient_id", "visit_date", "vital_signs"]
        for field in required_fields:
            if field not in data or not data[field]:
                issues.append(f"Missing required field: {field}")
                queries.append({
                    "field": field,
                    "query": f"Please provide {field} value",
                    "priority": "high"
                })

        # Check for out-of-range values
        if "lab_values" in data:
            for lab, value in data.get("lab_values", {}).items():
                if not self._value_in_range(lab, value):
                    issues.append(f"Out of range: {lab} = {value}")

        return {
            "status": "completed",
            "data_quality_score": 100 - len(issues) * 10,
            "issues_found": issues,
            "queries_generated": queries,
            "action": "Data accepted" if not issues else "Queries sent to site"
        }

    def _value_in_range(self, lab: str, value: Any) -> bool:
        """Check if lab value is in acceptable range."""
        # Simplified range checking
        return True


# =============================================================================
# SECTION 3: MULTI-AGENT ORCHESTRATION
# =============================================================================

class ClinicalTrialCrew:
    """
    Multi-agent crew for clinical trial management.

    Orchestrates multiple specialized agents to handle
    complex clinical trial workflows.

    WORKFLOW EXAMPLES:
    -----------------
    1. Patient enrollment: Eligibility -> Data -> Coordinator
    2. Adverse event: Safety -> Regulatory -> Coordinator
    3. Site monitoring: Monitor -> Data -> Coordinator
    """

    def __init__(self, trial: ClinicalTrial):
        self.trial = trial
        self.agents: dict = {}
        self._workflow_history = []

        self._initialize_agents()
        logger.info(f"ClinicalTrialCrew initialized for {trial.trial_id}")

    def _initialize_agents(self):
        """Initialize all required agents."""
        self.agents["eligibility"] = EligibilityAgent()
        self.agents["safety"] = SafetyOfficerAgent()
        self.agents["data"] = DataManagerAgent()

        # Set trial-specific criteria
        self.agents["eligibility"].set_trial_criteria(
            inclusion=[
                "Age >= 18 years",
                "ECOG performance status 0-1",
                "Histologically confirmed diagnosis",
                "Adequate organ function"
            ],
            exclusion=[
                "Prior treatment with study drug",
                "Active CNS metastases",
                "Uncontrolled infection",
                "Pregnancy or lactation"
            ]
        )

    def process_enrollment(self, patient_data: dict) -> dict:
        """
        Process patient enrollment request.

        Workflow:
        1. Eligibility screening
        2. Data validation
        3. Enrollment confirmation

        Args:
            patient_data: Patient information dictionary

        Returns:
            Enrollment result with all agent outputs
        """
        logger.info(f"Processing enrollment for patient {patient_data.get('patient_id')}")

        workflow_result = {
            "workflow": "enrollment",
            "patient_id": patient_data.get("patient_id"),
            "timestamp": datetime.now().isoformat(),
            "steps": []
        }

        # Step 1: Eligibility check
        eligibility_result = self.agents["eligibility"].process_task(
            "Screen patient for trial eligibility",
            {"patient_data": patient_data}
        )
        workflow_result["steps"].append({
            "agent": "eligibility",
            "result": eligibility_result
        })

        if not eligibility_result.get("eligible", False):
            workflow_result["outcome"] = "not_eligible"
            workflow_result["reason"] = eligibility_result.get("exclusion_violations", [])
            return workflow_result

        # Step 2: Data validation
        data_result = self.agents["data"].process_task(
            "Validate patient enrollment data",
            {"patient_data": patient_data}
        )
        workflow_result["steps"].append({
            "agent": "data",
            "result": data_result
        })

        if data_result.get("data_quality_score", 0) < 80:
            workflow_result["outcome"] = "pending_queries"
            workflow_result["queries"] = data_result.get("queries_generated", [])
            return workflow_result

        # Step 3: Complete enrollment
        workflow_result["outcome"] = "enrolled"
        workflow_result["treatment_arm"] = self._assign_treatment_arm()

        self._workflow_history.append(workflow_result)
        logger.info(f"Enrollment complete: {workflow_result['outcome']}")

        return workflow_result

    def process_adverse_event(self, ae_data: dict) -> dict:
        """
        Process adverse event report.

        Workflow:
        1. Safety evaluation
        2. Determine reporting requirements
        3. Generate required reports

        Args:
            ae_data: Adverse event information

        Returns:
            AE processing result with required actions
        """
        logger.info(f"Processing AE: {ae_data.get('event_id')}")

        workflow_result = {
            "workflow": "adverse_event",
            "event_id": ae_data.get("event_id"),
            "timestamp": datetime.now().isoformat(),
            "steps": []
        }

        # Safety evaluation
        safety_result = self.agents["safety"].process_task(
            "Evaluate adverse event and determine required actions",
            {"adverse_event": ae_data}
        )
        workflow_result["steps"].append({
            "agent": "safety",
            "result": safety_result
        })

        workflow_result["severity"] = safety_result.get("severity_assessment")
        workflow_result["required_actions"] = safety_result.get("required_actions", [])
        workflow_result["reporting_timeline"] = safety_result.get("reporting_timeline", {})

        self._workflow_history.append(workflow_result)
        return workflow_result

    def _assign_treatment_arm(self) -> str:
        """Assign patient to treatment arm (randomization)."""
        import random
        arms = ["Treatment A", "Treatment B", "Control"]
        return random.choice(arms)

    def get_workflow_history(self) -> list:
        """Get complete workflow history."""
        return self._workflow_history

    def generate_report(self) -> str:
        """Generate trial status report."""
        report = f"""
CLINICAL TRIAL STATUS REPORT
============================
Trial: {self.trial.title}
ID: {self.trial.trial_id}
Phase: {self.trial.phase.value}

Enrollment Status
-----------------
Target: {self.trial.enrollment_target}
Current: {self.trial.current_enrollment}
Progress: {self.trial.current_enrollment / max(1, self.trial.enrollment_target) * 100:.1f}%

Workflow Summary
----------------
Total workflows processed: {len(self._workflow_history)}
"""

        # Count outcomes
        outcomes = {}
        for wf in self._workflow_history:
            outcome = wf.get("outcome", "unknown")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

        for outcome, count in outcomes.items():
            report += f"  {outcome}: {count}\n"

        return report


# =============================================================================
# SECTION 4: LANGGRAPH WORKFLOW (Alternative Implementation)
# =============================================================================

class LangGraphWorkflow:
    """
    Alternative workflow implementation using LangGraph.

    LangGraph provides:
    - Stateful workflow graphs
    - Conditional routing
    - Persistence
    - Streaming support
    """

    def __init__(self, trial: ClinicalTrial):
        self.trial = trial
        self._graph = None
        self._build_graph()

    def _build_graph(self):
        """Build the workflow graph."""
        # In production, this would use LangGraph's StateGraph
        # from langgraph.graph import StateGraph

        logger.info("LangGraph workflow built")

    def invoke(self, input_data: dict) -> dict:
        """Invoke the workflow."""
        # Simulated invocation
        return {"status": "completed", "result": "Workflow executed"}


# =============================================================================
# SECTION 5: NATURAL LANGUAGE INTERFACE
# =============================================================================

class ClinicalTrialAssistant:
    """
    Natural language interface for clinical trial management.

    Allows trial coordinators to manage trials using
    conversational commands.

    EXAMPLE COMMANDS:
    ----------------
    - "Enroll patient 12345 from Site A"
    - "Report adverse event for patient 12345: Grade 3 nausea"
    - "Generate enrollment report for this week"
    - "Check eligibility for patient with ECOG 2"
    """

    def __init__(self, trial: ClinicalTrial):
        self.trial = trial
        self.crew = ClinicalTrialCrew(trial)

    def process_command(self, command: str) -> str:
        """
        Process natural language command.

        Args:
            command: Natural language command

        Returns:
            Response string
        """
        logger.info(f"Processing command: {command}")

        # Parse command intent (simplified)
        command_lower = command.lower()

        if "enroll" in command_lower:
            return self._handle_enrollment(command)
        elif "adverse" in command_lower or "ae" in command_lower:
            return self._handle_adverse_event(command)
        elif "report" in command_lower:
            return self._handle_report(command)
        elif "eligibility" in command_lower or "eligible" in command_lower:
            return self._handle_eligibility_check(command)
        else:
            return f"I can help with enrollment, adverse events, eligibility checks, and reports. What would you like to do?"

    def _handle_enrollment(self, command: str) -> str:
        """Handle enrollment command."""
        # Extract patient ID (simplified)
        patient_id = f"PT-{hash(command) % 10000:04d}"

        result = self.crew.process_enrollment({
            "patient_id": patient_id,
            "site_id": "Site-A",
            "vital_signs": {"bp": "120/80", "hr": 72},
            "lab_values": {"wbc": 5.5, "hgb": 12.0}
        })

        if result["outcome"] == "enrolled":
            return f"Patient {patient_id} has been enrolled in {result.get('treatment_arm', 'unknown arm')}."
        elif result["outcome"] == "not_eligible":
            return f"Patient {patient_id} is not eligible. Reason: {result.get('reason', 'See details')}"
        else:
            return f"Enrollment pending. Queries: {result.get('queries', [])}"

    def _handle_adverse_event(self, command: str) -> str:
        """Handle adverse event command."""
        event_id = f"AE-{hash(command) % 10000:04d}"

        result = self.crew.process_adverse_event({
            "event_id": event_id,
            "patient_id": "PT-0001",
            "description": command,
            "severity": "moderate",
            "onset_date": datetime.now().isoformat()
        })

        actions = result.get("required_actions", [])
        return f"Adverse event {event_id} recorded. Severity: {result.get('severity')}. Required actions: {', '.join(actions)}"

    def _handle_report(self, command: str) -> str:
        """Handle report generation."""
        return self.crew.generate_report()

    def _handle_eligibility_check(self, command: str) -> str:
        """Handle eligibility check."""
        result = self.crew.agents["eligibility"].process_task(
            "Check eligibility",
            {"patient_data": {"ecog": 1, "age": 55}}
        )
        return f"Eligibility assessment: {'Eligible' if result.get('eligible') else 'Not eligible'}"


# =============================================================================
# SECTION 6: MAIN PIPELINE
# =============================================================================

def run_clinical_trial_workflow():
    """
    Demonstrate agentic AI for clinical trial management.

    This function shows how multi-agent systems can
    coordinate complex clinical trial workflows.
    """
    logger.info("=" * 60)
    logger.info("AGENTIC AI CLINICAL TRIAL WORKFLOW")
    logger.info("=" * 60)

    # Create trial
    trial = ClinicalTrial(
        trial_id="NCT-2026-0001",
        title="Phase II Study of Physical AI-Guided Tumor Resection",
        phase=TrialPhase.PHASE_II,
        sponsor="Physical AI Oncology Consortium",
        sites=["Site-A", "Site-B", "Site-C"],
        enrollment_target=150,
        current_enrollment=45
    )

    # Create assistant
    assistant = ClinicalTrialAssistant(trial)

    # Example commands
    commands = [
        "Enroll patient John Doe from Site A",
        "Check eligibility for patient with ECOG status 1",
        "Report adverse event for patient: Grade 2 fatigue starting yesterday",
        "Generate enrollment report"
    ]

    print("\nClinical Trial Assistant Demo")
    print("-" * 40)

    for command in commands:
        print(f"\n> {command}")
        response = assistant.process_command(command)
        print(f"Assistant: {response}")

    # Print final report
    print("\n" + "=" * 60)
    print("FINAL TRIAL STATUS")
    print("=" * 60)
    print(assistant.crew.generate_report())

    return {
        "trial_id": trial.trial_id,
        "workflows_processed": len(assistant.crew.get_workflow_history()),
        "status": "demo_complete"
    }


if __name__ == "__main__":
    result = run_clinical_trial_workflow()
    print(f"\nDemo completed: {result}")
