# Agentic AI for Physical Oncology Systems: Limitations

*Critical constraints for clinical trial deployment (October 2025 - January 2026)*

---

## 1. Reliability and Consistency

### LLM Non-Determinism

**Critical Limitation**: Language models produce variable outputs for identical inputs, problematic for medical device consistency.

**Manifestations**:

| Scenario | Variability Impact | Risk Level |
|----------|-------------------|------------|
| Task decomposition | Different subtask sequences | Medium |
| Timing decisions | Variable action delays | Medium |
| Error interpretation | Inconsistent error handling | High |
| Safety classification | Variable risk assessment | Critical |

**Quantified Variability (2025 Studies)**:
- Same prompt, 100 runs: 15-25% variation in action sequences
- Temperature=0 reduces but doesn't eliminate: 5-10% variation
- Safety-critical decisions: Unacceptable for autonomous operation

**Mitigation Approaches**:

```python
# Consensus-based decision making for safety-critical actions
from agentic import ConsensusAgent

class SafetyConsensusAgent:
    def __init__(self, n_samples: int = 5, threshold: float = 0.8):
        self.n_samples = n_samples
        self.threshold = threshold

    async def decide(self, situation: str) -> Decision:
        """Require supermajority agreement for action"""
        decisions = []
        for _ in range(self.n_samples):
            decision = await self.llm.decide(situation)
            decisions.append(decision)

        # Check for consensus
        action_counts = Counter(d.action for d in decisions)
        most_common, count = action_counts.most_common(1)[0]

        if count / self.n_samples >= self.threshold:
            return Decision(action=most_common, confidence=count/self.n_samples)
        else:
            # No consensus - escalate to human
            return Decision(action="ESCALATE", confidence=0)
```

### Instruction Following Failures

**Limitation**: LLMs occasionally ignore explicit instructions, especially under complex contexts.

**Documented Failure Modes**:
- Constraint violations: "Never exceed 10mm/s" ignored in 3-5% of cases
- Sequence errors: Steps executed out of order
- Premature termination: Tasks abandoned before completion
- Hallucinated completions: Reports success without completing task

---

## 2. Latency and Real-Time Constraints

### LLM Inference Bottleneck

**Current Latencies (January 2026)**:

| Model | Inference Time | Control Frequency | Clinical Suitability |
|-------|----------------|-------------------|---------------------|
| Claude Sonnet 4 | 200-500ms | 2-5 Hz | Planning only |
| Claude Haiku 4 | 50-100ms | 10-20 Hz | High-level decisions |
| GPT-4o | 150-300ms | 3-7 Hz | Planning only |
| Local 7B model | 20-50ms | 20-50 Hz | Simple decisions |

**Impact on Oncology Robotics**:
- Real-time surgical control (200 Hz): Impossible with current LLMs
- Reactive safety responses (<10ms): Requires separate system
- Dynamic replanning (<100ms): Marginal with edge models

**Architectural Constraint**:

```
Required architecture for real-time agentic control:

┌─────────────────────────────────────────────────────────┐
│                    LLM Planner (2-5 Hz)                 │
│            Strategic decisions, task planning           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               Mid-Level Policy (20-50 Hz)               │
│          Trajectory generation, subtask execution       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│             Reactive Controller (200-1000 Hz)           │
│         Collision avoidance, force control, safety      │
└─────────────────────────────────────────────────────────┘

LLM CANNOT be in the reactive loop - latency prohibitive
```

### Network Dependency

**Limitation**: Cloud-based LLMs require network connectivity, problematic in:
- Operating rooms with RF restrictions
- Network outage scenarios
- High-security environments

---

## 3. Context Window Limitations

### Long Procedure Context

**Limitation**: Extended oncology procedures exceed practical context limits.

**Context Requirements**:

| Procedure Type | Duration | Events/Hour | Total Context Needed |
|---------------|----------|-------------|---------------------|
| Simple biopsy | 30 min | 100 | 50 events |
| Laparoscopic surgery | 2-4 hours | 200 | 400-800 events |
| Complex resection | 6-8 hours | 200 | 1200-1600 events |
| Multi-day trial protocol | Days | Continuous | Unbounded |

**Current Model Limits**:
- Claude: 200K tokens (~150K words)
- GPT-4: 128K tokens (~100K words)
- Practical limit with tool use: ~50K tokens

**Implications**:
- Cannot maintain full procedure history in context
- Requires summarization (lossy)
- Risk of "forgetting" earlier events

```python
# Context management with summarization
from agentic import ContextManager

class ProcedureContextManager:
    def __init__(self, max_tokens: int = 50000):
        self.max_tokens = max_tokens
        self.full_history = []  # Complete log
        self.working_context = []  # LLM context

    def add_event(self, event: ProcedureEvent):
        self.full_history.append(event)

        # Check if context exceeded
        if self.estimate_tokens() > self.max_tokens:
            self.compress_context()

    def compress_context(self):
        """LIMITATION: Information loss during compression"""
        # Keep recent events detailed
        recent = self.working_context[-100:]

        # Summarize older events
        older = self.working_context[:-100]
        summary = self.summarizer.summarize(older)

        self.working_context = [summary] + recent

        # WARNING: Subtle details from older events may be lost
        # Could miss patterns spanning long time periods
```

---

## 4. Tool Use Reliability

### Tool Call Failures

**Limitation**: LLMs frequently make errors when calling external tools.

**Documented Error Rates (2025)**:

| Error Type | Frequency | Impact |
|------------|-----------|--------|
| Wrong tool selected | 5-10% | Medium |
| Incorrect parameters | 8-15% | High |
| Missing required parameters | 3-7% | High |
| Malformed output parsing | 5-10% | Medium |

**Oncology-Specific Risks**:
- Wrong patient ID in tool call → Wrong patient interaction
- Incorrect dosage parameter → Medication error
- Missing safety flags → Constraint violation

**Required Safeguards**:

```python
# Tool call validation for medical contexts
from agentic import ToolCallValidator

class MedicalToolValidator:
    def validate_call(self, tool_name: str, params: dict) -> ValidationResult:
        # Check required parameters
        if tool_name == "administer_medication":
            required = ["patient_id", "medication", "dose", "route"]
            missing = [r for r in required if r not in params]
            if missing:
                return ValidationResult(
                    valid=False,
                    error=f"Missing required parameters: {missing}"
                )

            # Validate patient ID format
            if not self.is_valid_patient_id(params["patient_id"]):
                return ValidationResult(
                    valid=False,
                    error="Invalid patient ID format"
                )

            # Cross-check with patient database
            patient = self.lookup_patient(params["patient_id"])
            if patient is None:
                return ValidationResult(
                    valid=False,
                    error="Patient not found in database"
                )

            # Verify medication is in patient's protocol
            if params["medication"] not in patient.protocol_medications:
                return ValidationResult(
                    valid=False,
                    error="Medication not in patient protocol - REQUIRES HUMAN REVIEW"
                )

        return ValidationResult(valid=True)
```

### MCP Integration Challenges

**Limitation**: While MCP standardizes tool interfaces, integration complexity remains high.

**Challenges**:
- Server availability and reliability
- Version compatibility
- Authentication management
- Error propagation

---

## 5. Multi-Agent Coordination Failures

### Emergent Behavior Unpredictability

**Limitation**: Multi-agent systems can exhibit unexpected emergent behaviors.

**Documented Issues**:
- Deadlocks: Agents waiting for each other indefinitely
- Resource conflicts: Simultaneous access to shared equipment
- Goal conflicts: Agents working at cross-purposes
- Cascade failures: One agent's error propagating through system

**Oncology Impact**:
- Scheduling conflicts for shared imaging equipment
- Robot collision in shared workspaces
- Inconsistent patient interaction

```python
# Multi-agent failure example
"""
Scenario: Two agents both need to use the same imaging system

Agent A: "I need CT scanner for patient biopsy guidance"
Agent B: "I need CT scanner for treatment planning verification"

Without proper coordination:
- Both attempt to schedule simultaneously
- Conflicting bookings created
- Neither can proceed
- Manual intervention required

Required: Explicit resource locking and negotiation protocols
"""

from agentic import ResourceCoordinator

coordinator = ResourceCoordinator(
    shared_resources=["ct_scanner", "or_robot", "pathology_processor"],
    lock_timeout_seconds=300,
    conflict_resolution="priority_based"
)

# Agents must request resources through coordinator
async with coordinator.lock("ct_scanner", agent_id="biopsy_agent", priority=HIGH):
    # Exclusive access to resource
    await perform_biopsy_with_ct_guidance()
```

### Communication Overhead

**Limitation**: Inter-agent communication adds latency and complexity.

**Observed Overhead**:
- Message passing: 10-50ms per exchange
- Consensus protocols: 100-500ms for agreement
- State synchronization: Continuous bandwidth requirement

---

## 6. Safety and Oversight

### Autonomous Action Risks

**Critical Limitation**: Fully autonomous agentic systems can take harmful actions before human intervention.

**Risk Categories**:

| Risk | Example | Mitigation |
|------|---------|------------|
| Misinterpretation | "Remove the lesion" interpreted as wrong structure | Confirmation dialogs |
| Overconfidence | Agent proceeds despite uncertainty | Confidence thresholds |
| Scope creep | Agent takes unauthorized additional actions | Explicit action boundaries |
| Failure to escalate | Agent doesn't recognize own limitations | Uncertainty monitoring |

**Required Safety Architecture**:

```python
# Mandatory human-in-the-loop for high-risk actions
from agentic import SafetyGate

class OncologyAgentWithSafetyGates:
    def __init__(self):
        self.safety_gates = {
            "medication_administration": SafetyGate(
                requires_confirmation=True,
                timeout_action="abort",
                confirmation_channel="bedside_display"
            ),
            "invasive_procedure": SafetyGate(
                requires_confirmation=True,
                required_confirmations=2,  # Surgeon + nurse
                timeout_action="abort"
            ),
            "patient_identification": SafetyGate(
                requires_confirmation=True,
                verification_method="barcode_scan"
            ),
            "sample_transport": SafetyGate(
                requires_confirmation=False,  # Lower risk
                logging_required=True
            )
        }

    async def execute_action(self, action: Action):
        gate = self.safety_gates.get(action.category)

        if gate and gate.requires_confirmation:
            confirmed = await gate.request_confirmation(
                action=action,
                context=self.current_context
            )
            if not confirmed:
                return ActionResult(status="ABORTED", reason="Human declined")

        return await self.perform_action(action)
```

### Audit Trail Gaps

**Limitation**: LLM reasoning is difficult to fully log and audit.

**Challenges**:
- Internal reasoning not fully externalized
- Token-level decisions not interpretable
- Counterfactual analysis difficult
- Reproducibility not guaranteed

---

## 7. Domain Knowledge Limitations

### Medical Knowledge Currency

**Limitation**: LLM knowledge has a training cutoff and may not reflect current guidelines.

**Implications for Oncology**:
- New drug approvals not in training data
- Updated NCCN guidelines unknown
- Recent safety alerts missing
- Emerging treatment protocols unavailable

**Required Approach**:

```python
# RAG-augmented medical knowledge
from agentic import MedicalKnowledgeAgent

agent = MedicalKnowledgeAgent(
    base_llm="claude-sonnet-4",
    knowledge_sources=[
        NCCNGuidelines(version="2025.4"),
        FDADrugDatabase(last_updated="2026-01-15"),
        InstitutionalProtocols(hospital="example_cancer_center"),
        PubMed(search_recency="6_months")
    ],
    knowledge_update_frequency="daily"
)

# Agent retrieves current information before decisions
response = agent.plan_treatment(
    patient=patient_data,
    # Agent will:
    # 1. Check current NCCN guidelines for this cancer type
    # 2. Verify drug interactions with current medications
    # 3. Confirm protocol eligibility criteria
    # 4. Then generate plan
)
```

### Procedure-Specific Expertise

**Limitation**: General LLMs lack deep procedural expertise for specialized oncology procedures.

**Gap Areas**:
- Surgical technique nuances
- Institution-specific protocols
- Equipment-specific constraints
- Rare complication management

---

## 8. Integration Complexity

### Legacy System Compatibility

**Limitation**: Healthcare IT infrastructure often uses legacy systems incompatible with modern agentic frameworks.

**Common Barriers**:

| System | Challenge | Workaround Complexity |
|--------|-----------|----------------------|
| HL7 v2 EMR | No REST API | High - requires interface engine |
| PACS (pre-WADO) | Proprietary protocols | Medium - DICOM gateway |
| Robot controllers | Real-time constraints | High - custom middleware |
| Lab instruments | Serial/proprietary | High - device-specific adapters |

### Authentication and Authorization

**Limitation**: Agentic systems require robust identity and access management not yet standardized.

**Challenges**:
- Agent identity representation
- Action-level authorization
- Audit trail for agent actions
- Credential management across systems

---

## 9. Regulatory Uncertainty

### FDA Pathway Unclear

**Limitation**: No established regulatory pathway for agentic AI in medical devices.

**Open Questions**:
- How to validate non-deterministic agent behavior?
- What constitutes a "change" requiring resubmission?
- How to handle continuous learning agents?
- Liability allocation for agent decisions

### Clinical Validation Requirements

**Limitation**: Traditional clinical validation methods don't apply well to agentic systems.

**Challenges**:
- Cannot enumerate all possible behaviors
- Statistical testing insufficient for rare events
- Generalization claims difficult to validate
- Long-term behavior changes

---

## 10. Failure Recovery

### Graceful Degradation

**Limitation**: Agentic systems often fail completely rather than degrading gracefully.

**Failure Modes Without Graceful Degradation**:
- LLM API unavailable → Complete system halt
- Tool call fails → Agent confused, may loop
- Context overflow → Incoherent behavior
- Network latency spike → Timeout cascade

**Required Resilience Patterns**:

```python
# Resilient agentic system architecture
from agentic import ResilientAgent

class ResilientOncologyAgent:
    def __init__(self):
        self.primary_llm = "claude-sonnet-4"
        self.fallback_llm = "local_llama_70b"
        self.emergency_mode = "rule_based_fallback"

    async def execute(self, task: Task):
        try:
            # Try primary LLM
            return await self.execute_with_llm(task, self.primary_llm)
        except LLMUnavailableError:
            logger.warning("Primary LLM unavailable, using fallback")
            try:
                return await self.execute_with_llm(task, self.fallback_llm)
            except LLMUnavailableError:
                logger.error("All LLMs unavailable, entering emergency mode")
                return await self.emergency_mode_execute(task)

    async def emergency_mode_execute(self, task: Task):
        """Rule-based fallback for critical operations"""
        if task.type == "safety_critical":
            # Safe default: stop and alert human
            await self.alert_human_operator(task)
            return TaskResult(status="PAUSED", reason="LLM unavailable")
        elif task.type == "routine":
            # Execute pre-programmed routine
            return await self.execute_routine(task)
```

---

## Summary: Critical Limitations Requiring Mitigation

| Limitation | Severity | Mitigation Maturity | Deployment Blocker |
|------------|----------|--------------------|--------------------|
| LLM non-determinism | Critical | Partial (consensus) | Yes - for autonomous |
| Real-time latency | High | Partial (hierarchy) | Yes - for reactive |
| Context limits | Medium | Partial (summarization) | No |
| Tool call errors | High | Good (validation) | No |
| Multi-agent failures | Medium | Partial (coordination) | No |
| Safety oversight | Critical | Good (gates) | Yes - require gates |
| Knowledge currency | Medium | Good (RAG) | No |
| Regulatory uncertainty | Critical | Low | Yes - for FDA path |

---

## Deployment Recommendations

1. **Never allow fully autonomous operation** for patient-facing tasks
2. **Implement mandatory safety gates** for all irreversible actions
3. **Use hierarchical architecture** with non-LLM reactive layer
4. **Validate all tool calls** before execution
5. **Maintain fallback modes** for LLM unavailability
6. **Update knowledge bases** continuously via RAG
7. **Engage regulatory early** before significant investment
8. **Log extensively** for audit trail and debugging

---

*References: FDA AI/ML Guidance (2024), AAMI TIR57 (2024), LangChain Best Practices (2025), CrewAI Safety Documentation (2025)*
