# Agentic AI for Physical Oncology Systems: Strengths

*Production-validated capabilities for autonomous clinical trial robotics (October 2025 - January 2026)*

---

## 1. LLM-Based Robot Control

### Natural Language Task Specification

**Core Strength**: Agentic AI enables non-programmer clinicians to specify complex robotic tasks through natural language.

**LLM-Embedded Robotic Scrub Nurse (2025)**

Demonstrated capabilities in surgical settings:

| Capability | Performance | Clinical Impact |
|------------|-------------|-----------------|
| Voice command recognition | 96.5% accuracy | Hands-free surgeon interaction |
| Multi-modal input fusion | Speech + vision | Context-aware responses |
| Real-time action execution | <1 second latency | Seamless workflow integration |
| Standalone operation | 94% success rate | Reduced staffing requirements |

**Practical Application for Oncology Trials**:

```python
# Natural language control for drug infusion robot
from agentic import LLMController, SafetyMonitor

controller = LLMController(
    model="claude-sonnet-4",
    robot_interface="unitree_g1",
    safety_constraints=ONCOLOGY_WARD_SAFETY
)

# Clinician issues natural language command
response = controller.execute(
    command="Prepare the cisplatin infusion for bed 3, patient Johnson. "
            "Verify the dosage matches the protocol before connecting.",
    context={
        "patient_id": "PT-2847",
        "protocol": "ONCO-2025-042",
        "current_location": "pharmacy_prep"
    }
)

# Agent autonomously:
# 1. Navigates to medication storage
# 2. Identifies correct medication via vision
# 3. Verifies dosage against protocol database
# 4. Prepares infusion with proper technique
# 5. Transports to patient bedside
# 6. Confirms patient identity before handoff
```

### Hierarchical Task Decomposition

**Strength**: LLM planners decompose high-level clinical goals into executable subtask sequences.

**Architecture Benefits**:

1. **Separation of concerns**: Language understanding (LLM) vs. motor execution (policy network)
2. **Interpretable plans**: Each step can be reviewed before execution
3. **Flexible replanning**: Verbal corrections immediately modify behavior
4. **Knowledge integration**: LLM incorporates medical knowledge into planning

```python
# Hierarchical planning for tumor biopsy procedure
from agentic import HierarchicalPlanner

planner = HierarchicalPlanner(
    high_level="claude-opus-4",  # Procedure planning
    mid_level="surgical_vla",    # Subtask execution
    low_level="reactive_policy"  # Real-time control
)

procedure_plan = planner.generate(
    goal="Perform CT-guided lung biopsy on right upper lobe nodule",
    patient_data=imaging_data,
    constraints=["avoid major vessels", "single needle pass preferred"]
)

# Generated plan:
# 1. Verify patient positioning and CT alignment
# 2. Identify optimal entry point (intercostal space)
# 3. Administer local anesthesia
# 4. Insert guide needle under CT guidance
# 5. Advance biopsy needle to target
# 6. Obtain tissue sample
# 7. Confirm sample adequacy
# 8. Remove needle and apply hemostasis
```

---

## 2. Multi-Agent Coordination

### Cooperative Surgical Assistance

**Key Strength**: Multi-agent reinforcement learning enables multiple robots to coordinate without explicit programming.

**Demonstrated Results (2025)**:

| Configuration | Procedure Time | Collision Reduction | Success Rate |
|--------------|----------------|---------------------|--------------|
| 1 Human + 1 Agent | -44.4% | -44.7% | 96% |
| 2 Cooperative Agents | -71.2% | -98% | 92% |

**Oncology Application**: Multi-robot coordination for complex procedures requiring simultaneous retraction, visualization, and manipulation.

```python
# Multi-agent setup for robotic nephrectomy assistance
from crewai import Crew, Agent, Task

# Define specialized agents
camera_agent = Agent(
    role="Surgical Camera Operator",
    goal="Maintain optimal visualization of surgical field",
    backstory="Expert in laparoscopic camera positioning",
    tools=[camera_control, zoom_adjustment, focus_tracking]
)

retractor_agent = Agent(
    role="Tissue Retractor",
    goal="Provide exposure while minimizing tissue trauma",
    backstory="Specialized in gentle tissue manipulation",
    tools=[retractor_control, force_sensing, position_hold]
)

surgeon_assistant = Agent(
    role="Surgical Assistant",
    goal="Anticipate surgeon needs and provide instruments",
    backstory="Experienced surgical technologist",
    tools=[instrument_selection, handover_execution, suction_irrigation]
)

# Orchestrate multi-agent cooperation
surgical_crew = Crew(
    agents=[camera_agent, retractor_agent, surgeon_assistant],
    tasks=[maintain_visualization, provide_exposure, assist_dissection],
    process="hierarchical",  # Surgeon commands propagate through hierarchy
    manager_llm="claude-sonnet-4"
)
```

### Multi-Site Clinical Trial Coordination

**Strength**: Agentic systems can coordinate activities across multiple trial sites.

**Capabilities**:
- Standardized protocol execution across sites
- Real-time data synchronization
- Automated deviation detection and reporting
- Cross-site resource allocation

```python
# Multi-site trial coordination agent
from langgraph import StateGraph, Agent

class TrialCoordinator:
    def __init__(self, sites: list[str]):
        self.graph = StateGraph()

        # Site-specific agents
        for site in sites:
            self.graph.add_node(
                f"site_{site}",
                Agent(
                    tools=[
                        enrollment_tracker,
                        protocol_checker,
                        adverse_event_reporter,
                        sample_logistics
                    ]
                )
            )

        # Central coordination
        self.graph.add_node(
            "central_coordinator",
            Agent(tools=[cross_site_analytics, regulatory_reporting])
        )

    def coordinate(self, event: str, site: str):
        """Route events through appropriate agents"""
        return self.graph.invoke({
            "event": event,
            "origin_site": site,
            "timestamp": datetime.now()
        })
```

---

## 3. Tool Use and Environment Interaction

### Model Context Protocol (MCP) Integration

**Strength**: Standardized protocol for connecting LLM agents to external tools and systems.

**MCP Ecosystem (December 2025 - January 2026)**:
- 97 million+ monthly SDK downloads across Python and TypeScript
- Donated to **Agentic AI Foundation (AAIF)** under Linux Foundation (Dec 2025)
- Co-founded by Anthropic, Block, and OpenAI with support from Google, Microsoft, AWS, Cloudflare, and Bloomberg
- Adopted by OpenAI (March 2025), Google DeepMind (April 2025), Microsoft/GitHub (May 2025)
- Official community-driven Registry for discovering MCP servers (Nov 2025)

**Oncology Trial Integration Points**:

| System | MCP Integration | Capability |
|--------|-----------------|------------|
| Hospital EMR | Read patient data | Protocol eligibility verification |
| Imaging PACS | Retrieve DICOM | Intraoperative guidance |
| Robot control | Send commands | Physical task execution |
| Lab systems | Query results | Real-time biomarker monitoring |
| Regulatory DB | Check requirements | Compliance verification |

```python
# MCP-based agentic oncology system
from mcp import MCPClient, ToolRegistry

# Register clinical trial tools
registry = ToolRegistry()

registry.register(
    name="query_patient_eligibility",
    description="Check if patient meets trial inclusion/exclusion criteria",
    input_schema={"patient_id": str, "protocol_id": str},
    handler=eligibility_checker.check
)

registry.register(
    name="schedule_imaging",
    description="Schedule CT/MRI/PET for trial protocol timepoint",
    input_schema={"patient_id": str, "modality": str, "timepoint": str},
    handler=imaging_scheduler.schedule
)

registry.register(
    name="execute_robot_task",
    description="Command robot to perform specified clinical task",
    input_schema={"task": str, "robot_id": str, "safety_level": str},
    handler=robot_controller.execute
)

# Agent uses tools through MCP
agent = MCPClient(tools=registry, model="claude-sonnet-4")
agent.run("Verify patient 2847 eligibility for ONCO-2025-042, "
          "then schedule their week 4 PET scan and prepare the "
          "sample collection robot for tomorrow's biopsy")
```

**Security Considerations for Clinical MCP Deployment**:
- Implement authentication for all MCP servers (HIPAA compliance)
- Use tool permissions and combining restrictions to prevent data exfiltration
- Validate tool inputs to prevent prompt injection attacks
- Follow security best practices from AAIF governance guidelines

### ROS 2 Agentic Frameworks

**RAI (RobotecAI) Framework**

Production-ready agentic framework for ROS 2 robots:

| Feature | Capability | Benefit |
|---------|------------|---------|
| Vendor-agnostic | Works with any ROS 2 robot | Flexibility |
| Voice interaction | Natural language commands | Accessibility |
| Multi-LLM support | OpenAI, AWS Bedrock, Anthropic | Choice |
| Safety integration | ROS 2 safety nodes | Clinical compliance |

```python
# RAI framework for hospital robot
from rai import RAIAgent, ROSInterface

agent = RAIAgent(
    ros_interface=ROSInterface(node_name="oncology_assistant"),
    llm_provider="anthropic",
    model="claude-sonnet-4",
    safety_config={
        "max_velocity": 0.5,  # m/s in patient areas
        "collision_avoidance": True,
        "emergency_stop_enabled": True
    }
)

# Voice-activated command
agent.listen_and_execute()

# Transcript: "Deliver the pathology samples from OR 3 to the lab"
# Agent automatically:
# - Plans route avoiding patient areas during quiet hours
# - Coordinates with door access systems
# - Navigates to OR 3
# - Confirms sample pickup with OR staff
# - Transports with appropriate speed/smoothness
# - Delivers to lab with chain-of-custody logging
```

**ROSA (NASA JPL)**

Natural language interface for ROS systems:

- Built on LangChain framework
- Supports inspection, diagnosis, and operation
- Extensible tool integration

---

## 4. Autonomous Task Planning

### Clinical Workflow Automation

**Strength**: Agentic systems can manage complex, multi-step clinical workflows with appropriate human oversight.

**Capabilities**:

1. **Protocol interpretation**: Parse clinical trial protocols into executable task sequences
2. **Resource scheduling**: Optimize robot/equipment utilization across procedures
3. **Exception handling**: Detect deviations and escalate appropriately
4. **Documentation**: Automatic logging of all actions for audit trail

```python
# Clinical workflow agent for trial day operations
from langgraph import StateGraph, START, END

class TrialDayAgent:
    def __init__(self):
        self.workflow = StateGraph(TrialDayState)

        # Define workflow nodes
        self.workflow.add_node("verify_schedule", self.verify_patient_schedule)
        self.workflow.add_node("prepare_equipment", self.prepare_robotic_equipment)
        self.workflow.add_node("patient_intake", self.coordinate_patient_intake)
        self.workflow.add_node("procedure_execution", self.oversee_procedure)
        self.workflow.add_node("sample_processing", self.coordinate_samples)
        self.workflow.add_node("documentation", self.complete_documentation)

        # Define conditional routing
        self.workflow.add_conditional_edges(
            "patient_intake",
            self.check_eligibility_status,
            {
                "eligible": "procedure_execution",
                "ineligible": "documentation",  # Log screen failure
                "uncertain": "escalate_to_coordinator"
            }
        )

    async def run_trial_day(self, date: datetime):
        """Execute full trial day workflow"""
        initial_state = TrialDayState(
            date=date,
            patients=self.get_scheduled_patients(date),
            status="starting"
        )
        return await self.workflow.ainvoke(initial_state)
```

### Corrective Instruction Handling

**Strength**: Agents can receive and immediately incorporate verbal corrections during execution.

**SRT-H Demonstration**:
- 95% accuracy on corrective instruction following
- Immediate policy modification without retraining
- Natural language error recovery

```python
# Corrective instruction example
agent.execute("Grasp the tissue with the right instrument")

# Surgeon observes suboptimal grip
surgeon_correction = "Stop - grip is too far from the edge, reposition 5mm proximally"

# Agent immediately:
# 1. Halts current action
# 2. Interprets correction
# 3. Adjusts grip position
# 4. Resumes with corrected approach
agent.handle_correction(surgeon_correction)
```

---

## 5. Context-Aware Decision Making

### Patient-Specific Adaptation

**Strength**: Agentic systems can incorporate comprehensive patient context into decision-making.

**Context Sources**:
- Electronic health records
- Prior imaging studies
- Genomic data
- Treatment history
- Real-time vital signs

```python
# Context-aware robotic biopsy planning
from agentic import ContextAwareAgent

agent = ContextAwareAgent(
    llm="claude-opus-4",
    context_sources=[
        EMRConnector(patient_id="PT-2847"),
        ImagingConnector(pacs_server="hospital.pacs.local"),
        GenomicsConnector(lab_system="foundation_medicine"),
        VitalsMonitor(bedside_monitor="bed_3")
    ]
)

# Agent synthesizes all context for decision
plan = agent.plan_procedure(
    procedure="liver_biopsy",
    target="segment_7_lesion"
)

# Agent considers:
# - Prior biopsy attempts (none for this lesion)
# - Coagulation status (INR 1.1 - acceptable)
# - Lesion characteristics from imaging (2.3cm, hypervascular)
# - Genomic profile of primary tumor (for expected histology)
# - Current vitals (stable, proceed)
```

### Situation Awareness

**Strength**: LLM-based agents maintain high-level situational awareness during procedures.

**Capabilities**:
- Procedure phase recognition
- Anomaly detection and alerting
- Progress estimation
- Resource anticipation

---

## 6. Human-Robot Collaboration

### Shared Autonomy

**Strength**: Agentic systems can seamlessly blend autonomous and human-guided operation.

**Collaboration Modes**:

| Mode | Robot Role | Human Role | Oncology Use Case |
|------|------------|------------|-------------------|
| Full autonomy | Complete task | Supervision | Sample transport |
| Shared control | Execute + suggest | Approve + correct | Surgical assistance |
| Teleoperation | Execute commands | Direct control | Complex dissection |
| Advisory | Suggest actions | Decide + execute | Treatment planning |

```python
# Shared autonomy controller
from agentic import SharedAutonomyController

controller = SharedAutonomyController(
    autonomy_levels={
        "transport": "full_autonomy",
        "patient_interaction": "shared_control",
        "invasive_procedure": "teleoperation",
        "planning": "advisory"
    }
)

# Controller adjusts autonomy based on task and context
current_task = "surgical_retraction"
autonomy = controller.get_autonomy_level(
    task=current_task,
    risk_level="medium",
    operator_experience="expert_surgeon"
)
# Returns: "shared_control" with surgeon override capability
```

### Intent Recognition

**Strength**: Agents can anticipate surgeon needs based on procedure context.

**Demonstrated Capabilities**:
- Next instrument prediction: 87% accuracy
- Action timing optimization: -15% idle time
- Proactive preparation: 2-3 steps ahead

---

## 7. Regulatory Compliance Automation

### Protocol Adherence Monitoring

**Strength**: Agentic systems can continuously monitor for protocol deviations.

```python
# Automated protocol compliance agent
from agentic import ComplianceAgent

compliance = ComplianceAgent(
    protocol_document="ONCO-2025-042_v3.pdf",
    monitoring_systems=[
        robot_action_log,
        medication_administration,
        imaging_schedule,
        adverse_event_reports
    ]
)

# Continuous monitoring
async def monitor_trial():
    async for event in trial_event_stream:
        compliance_status = await compliance.check_event(event)

        if compliance_status.deviation_detected:
            await notify_coordinator(compliance_status.deviation)
            await log_deviation(compliance_status)

            if compliance_status.severity == "critical":
                await pause_enrollment()
```

### Documentation Automation

**Strength**: Agents can generate regulatory-compliant documentation from actions.

- Automatic CRF completion
- Adverse event narratives
- Deviation reports
- Audit trail maintenance

---

## 8. Scalability and Deployment

### Edge Deployment

**NVIDIA Holoscan Integration**

Enables real-time agentic AI at the edge:

| Feature | Specification | Benefit |
|---------|---------------|---------|
| Latency | 5x lower than cloud | Real-time control |
| GPU direct RDMA | Sensor → GPU path | Minimal overhead |
| Multi-camera sync | Microsecond precision | Surgical visualization |

### Fleet Management

**Strength**: Agentic systems can manage fleets of clinical robots.

```python
# Hospital robot fleet management
from agentic import FleetManager

fleet = FleetManager(
    robots=[
        Robot("unitree_g1_01", location="pharmacy"),
        Robot("unitree_g1_02", location="or_suite"),
        Robot("mobile_01", location="pathology"),
        Robot("mobile_02", location="charging_station")
    ],
    coordination_strategy="task_auction"
)

# Optimal task assignment
fleet.assign_task(
    task="urgent_sample_transport",
    from_location="or_3",
    to_location="pathology",
    priority="high"
)
# Fleet manager selects best available robot considering:
# - Current location and battery status
# - Other pending tasks
# - Priority weighting
# - Path optimization
```

---

## Summary: Key Agentic AI Strengths for Oncology Trials

| Capability | Maturity | Impact |
|------------|----------|--------|
| Natural language control | Production-ready | High |
| Multi-agent coordination | Validated | High |
| MCP tool integration | Production-ready | Critical |
| ROS 2 integration | Production-ready | High |
| Workflow automation | Validated | High |
| Corrective instruction | Validated | Medium |
| Context-aware decisions | Emerging | High |
| Compliance monitoring | Validated | Critical |

---

## Recommended Implementation Path

1. **Start with**: RAI or ROSA for ROS 2 integration with existing robots
2. **Add**: MCP for clinical system integration (EMR, PACS, lab)
3. **Scale with**: CrewAI for multi-agent coordination
4. **Enhance with**: LangGraph for complex workflow automation
5. **Optimize with**: NVIDIA Holoscan for edge deployment

---

*References: CrewAI 1.6.1 (Nov 2025), LangGraph 1.1.0 (Jan 2026), Model Context Protocol/AAIF (Dec 2025, https://modelcontextprotocol.io), RAI Framework (2025), ROSA (NASA JPL), NVIDIA Holoscan (2025)*
