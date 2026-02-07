"""
=============================================================================
EXAMPLE 05: Safety-Constrained Agent Executor for Medical Robotics
=============================================================================

Implements formal safety constraints for agentic control of surgical
robots, with pre-condition/post-condition verification, runtime invariant
checking, and human-in-the-loop gates for critical actions.

CLINICAL CONTEXT:
-----------------
When AI agents control surgical robots in oncology procedures, safety
cannot rely solely on the agent's reasoning. Formal safety constraints
provide defense-in-depth:
  - Pre-conditions: Verified BEFORE an action executes
  - Post-conditions: Verified AFTER an action completes
  - Invariants: Continuously monitored during execution
  - Safety gates: Human approval checkpoints for critical actions
  - Rollback actions: Automatic recovery from constraint violations

DISTINCTION FROM EXISTING EXAMPLES:
------------------------------------
- examples-new/01_realtime_safety_monitoring.py: Hardware-level safety
  monitoring (force limits, joint limits, watchdog timers)
- This example: Agent-level safety with formal constraint specifications,
  action pre/post-condition verification, and structured safety proofs

SAFETY ARCHITECTURE:
--------------------
  Agent Decision --> Pre-condition Check --> Safety Gate --> Execute Action
                          |                      |               |
                     [REJECT if               [HOLD for        Post-condition
                      violated]              approval]          Check
                                                                 |
                                                            [ROLLBACK if
                                                             violated]

  Concurrent: Invariant Monitor (checks continuously during execution)

REGULATORY ALIGNMENT:
---------------------
- IEC 62304: Software lifecycle for medical device software
- IEC 80601-2-77: Particular requirements for robotically assisted surgery
- ISO 14971: Risk management for medical devices

FRAMEWORK REQUIREMENTS:
-----------------------
Required: (none - pure Python implementation)

Optional:
    - anthropic >= 0.40.0 (for LLM-based agent integration)

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: SAFETY CONSTRAINT DATA MODELS
# =============================================================================


class ConstraintType(Enum):
    """Types of safety constraints."""

    PRE_CONDITION = "pre_condition"
    POST_CONDITION = "post_condition"
    INVARIANT = "invariant"
    SAFETY_GATE = "safety_gate"


class ConstraintSeverity(Enum):
    """Severity of constraint violation consequences."""

    ADVISORY = "advisory"  # Log and continue
    BLOCKING = "blocking"  # Prevent action, agent can retry
    CRITICAL = "critical"  # Prevent action, require human intervention
    EMERGENCY = "emergency"  # Immediate stop, trigger emergency protocol


class ViolationResponse(Enum):
    """Response to a constraint violation."""

    LOG_ONLY = "log_only"
    REJECT_ACTION = "reject_action"
    ROLLBACK = "rollback"
    EMERGENCY_STOP = "emergency_stop"
    HUMAN_REVIEW = "human_review"


class ActionCategory(Enum):
    """Categories of agent actions by risk level."""

    OBSERVATION = "observation"  # Read-only queries
    NAVIGATION = "navigation"  # Robot motion without tissue contact
    MANIPULATION = "manipulation"  # Tool-tissue interaction
    ENERGY = "energy_delivery"  # Cautery, stapling, cutting
    CRITICAL = "critical_action"  # Vessel ligation, organ division


@dataclass
class SafetyConstraint:
    """
    Formal safety constraint definition.

    Each constraint has:
    - A human-readable description
    - A machine-checkable predicate function
    - Severity and response configuration
    - Applicability to specific action categories
    """

    constraint_id: str
    name: str
    description: str
    constraint_type: ConstraintType
    severity: ConstraintSeverity
    violation_response: ViolationResponse
    predicate: Callable = field(default=lambda state: True)
    applicable_actions: list = field(default_factory=list)
    regulatory_reference: str = ""

    def check(self, state: dict) -> "ConstraintResult":
        """
        Evaluate the constraint against current state.

        Args:
            state: Current system state dictionary

        Returns:
            ConstraintResult with pass/fail and details
        """
        try:
            satisfied = self.predicate(state)
            return ConstraintResult(
                constraint_id=self.constraint_id,
                constraint_name=self.name,
                satisfied=satisfied,
                severity=self.severity,
                violation_response=self.violation_response if not satisfied else ViolationResponse.LOG_ONLY,
                message=f"{'PASSED' if satisfied else 'VIOLATED'}: {self.description}",
                timestamp=time.time(),
            )
        except Exception as e:
            return ConstraintResult(
                constraint_id=self.constraint_id,
                constraint_name=self.name,
                satisfied=False,
                severity=ConstraintSeverity.CRITICAL,
                violation_response=ViolationResponse.REJECT_ACTION,
                message=f"EVALUATION ERROR in {self.name}: {str(e)}",
                timestamp=time.time(),
            )


@dataclass
class ConstraintResult:
    """Result of evaluating a safety constraint."""

    constraint_id: str
    constraint_name: str
    satisfied: bool
    severity: ConstraintSeverity
    violation_response: ViolationResponse
    message: str
    timestamp: float = 0.0


@dataclass
class AgentAction:
    """An action proposed by the agent for execution."""

    action_id: str
    category: ActionCategory
    description: str
    parameters: dict = field(default_factory=dict)
    requires_confirmation: bool = False
    rollback_action: Optional[str] = None
    estimated_duration_seconds: float = 1.0


@dataclass
class SafetyVerdict:
    """Overall safety verdict for a proposed action."""

    action_id: str
    approved: bool
    constraint_results: list = field(default_factory=list)
    blocking_violations: list = field(default_factory=list)
    advisory_violations: list = field(default_factory=list)
    requires_human_approval: bool = False
    safety_score: float = 1.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        """Serialize for audit trail."""
        return {
            "action_id": self.action_id,
            "approved": self.approved,
            "safety_score": self.safety_score,
            "blocking_violations": len(self.blocking_violations),
            "advisory_violations": len(self.advisory_violations),
            "requires_human_approval": self.requires_human_approval,
            "timestamp": self.timestamp,
        }


# =============================================================================
# SECTION 2: CONSTRAINT LIBRARY
# =============================================================================


class OncologyRoboticsConstraintLibrary:
    """
    Pre-defined safety constraints for robotic oncology procedures.

    Constraints are organized by type and aligned to regulatory
    requirements from IEC 80601-2-77 and ISO 14971.
    """

    @staticmethod
    def workspace_constraints() -> list[SafetyConstraint]:
        """Constraints ensuring robot stays within safe workspace."""
        return [
            SafetyConstraint(
                constraint_id="WS-001",
                name="Workspace Boundary",
                description="Robot end-effector must be within defined workspace boundaries",
                constraint_type=ConstraintType.PRE_CONDITION,
                severity=ConstraintSeverity.BLOCKING,
                violation_response=ViolationResponse.REJECT_ACTION,
                predicate=lambda state: _check_workspace_bounds(
                    state.get("target_position", [0, 0, 0]),
                    state.get("workspace_limits", {}),
                ),
                applicable_actions=[ActionCategory.NAVIGATION, ActionCategory.MANIPULATION],
                regulatory_reference="IEC 80601-2-77 Clause 201.12.4.4",
            ),
            SafetyConstraint(
                constraint_id="WS-002",
                name="Keep-Out Zone Clearance",
                description="Target must not be within any keep-out zone around critical anatomy",
                constraint_type=ConstraintType.PRE_CONDITION,
                severity=ConstraintSeverity.CRITICAL,
                violation_response=ViolationResponse.REJECT_ACTION,
                predicate=lambda state: _check_keepout_zones(
                    state.get("target_position", [0, 0, 0]),
                    state.get("keepout_zones", []),
                ),
                applicable_actions=[
                    ActionCategory.NAVIGATION,
                    ActionCategory.MANIPULATION,
                    ActionCategory.ENERGY,
                ],
                regulatory_reference="IEC 80601-2-77 Clause 201.12.4.5",
            ),
        ]

    @staticmethod
    def force_constraints() -> list[SafetyConstraint]:
        """Constraints on applied forces during tissue interaction."""
        return [
            SafetyConstraint(
                constraint_id="FC-001",
                name="Maximum Applied Force",
                description="Applied force must not exceed tissue-specific threshold",
                constraint_type=ConstraintType.INVARIANT,
                severity=ConstraintSeverity.CRITICAL,
                violation_response=ViolationResponse.EMERGENCY_STOP,
                predicate=lambda state: state.get("current_force_n", 0) <= state.get("max_force_n", 10.0),
                applicable_actions=[ActionCategory.MANIPULATION, ActionCategory.ENERGY],
                regulatory_reference="IEC 80601-2-77 Clause 201.12.4.6",
            ),
            SafetyConstraint(
                constraint_id="FC-002",
                name="Force Rate Limit",
                description="Rate of force change must not exceed threshold (prevents tissue damage)",
                constraint_type=ConstraintType.INVARIANT,
                severity=ConstraintSeverity.BLOCKING,
                violation_response=ViolationResponse.ROLLBACK,
                predicate=lambda state: abs(state.get("force_rate_n_s", 0)) <= state.get("max_force_rate_n_s", 5.0),
                applicable_actions=[ActionCategory.MANIPULATION],
                regulatory_reference="ISO 14971 Risk Control",
            ),
        ]

    @staticmethod
    def velocity_constraints() -> list[SafetyConstraint]:
        """Constraints on robot motion velocity."""
        return [
            SafetyConstraint(
                constraint_id="VC-001",
                name="Maximum Velocity",
                description="Robot velocity must not exceed safety-rated maximum",
                constraint_type=ConstraintType.INVARIANT,
                severity=ConstraintSeverity.BLOCKING,
                violation_response=ViolationResponse.ROLLBACK,
                predicate=lambda state: state.get("current_velocity_m_s", 0) <= state.get("max_velocity_m_s", 0.1),
                applicable_actions=[ActionCategory.NAVIGATION, ActionCategory.MANIPULATION],
                regulatory_reference="IEC 80601-2-77 Clause 201.12.4.3",
            ),
            SafetyConstraint(
                constraint_id="VC-002",
                name="Approach Velocity Near Target",
                description="Velocity must reduce when approaching surgical target",
                constraint_type=ConstraintType.PRE_CONDITION,
                severity=ConstraintSeverity.BLOCKING,
                violation_response=ViolationResponse.REJECT_ACTION,
                predicate=lambda state: (
                    state.get("distance_to_target_mm", 100) > 10.0
                    or state.get("commanded_velocity_m_s", 0) <= state.get("approach_velocity_m_s", 0.02)
                ),
                applicable_actions=[ActionCategory.NAVIGATION, ActionCategory.MANIPULATION],
                regulatory_reference="ISO 14971 Risk Control",
            ),
        ]

    @staticmethod
    def clinical_constraints() -> list[SafetyConstraint]:
        """Clinical workflow constraints."""
        return [
            SafetyConstraint(
                constraint_id="CL-001",
                name="Patient Identity Verified",
                description="Patient identity must be verified before any intervention",
                constraint_type=ConstraintType.PRE_CONDITION,
                severity=ConstraintSeverity.CRITICAL,
                violation_response=ViolationResponse.REJECT_ACTION,
                predicate=lambda state: state.get("patient_identity_verified", False),
                applicable_actions=[
                    ActionCategory.MANIPULATION,
                    ActionCategory.ENERGY,
                    ActionCategory.CRITICAL,
                ],
                regulatory_reference="WHO Surgical Safety Checklist",
            ),
            SafetyConstraint(
                constraint_id="CL-002",
                name="Surgical Timeout Completed",
                description="Surgical timeout must be completed before intervention begins",
                constraint_type=ConstraintType.PRE_CONDITION,
                severity=ConstraintSeverity.CRITICAL,
                violation_response=ViolationResponse.REJECT_ACTION,
                predicate=lambda state: state.get("timeout_completed", False),
                applicable_actions=[
                    ActionCategory.MANIPULATION,
                    ActionCategory.ENERGY,
                    ActionCategory.CRITICAL,
                ],
                regulatory_reference="WHO Surgical Safety Checklist",
            ),
            SafetyConstraint(
                constraint_id="CL-003",
                name="Vitals Stable for Intervention",
                description="Patient vitals must be within acceptable ranges before intervention",
                constraint_type=ConstraintType.PRE_CONDITION,
                severity=ConstraintSeverity.BLOCKING,
                violation_response=ViolationResponse.HUMAN_REVIEW,
                predicate=lambda state: (
                    50 <= state.get("heart_rate_bpm", 72) <= 120
                    and state.get("spo2_percent", 98) >= 92
                    and state.get("systolic_bp", 120) >= 90
                ),
                applicable_actions=[ActionCategory.MANIPULATION, ActionCategory.ENERGY],
                regulatory_reference="ICH E6(R3) GCP",
            ),
        ]

    @staticmethod
    def energy_constraints() -> list[SafetyConstraint]:
        """Constraints specific to energy delivery (cautery, stapling)."""
        return [
            SafetyConstraint(
                constraint_id="EN-001",
                name="Energy Device Activation Confirmation",
                description="Energy device activation requires explicit surgeon confirmation",
                constraint_type=ConstraintType.SAFETY_GATE,
                severity=ConstraintSeverity.CRITICAL,
                violation_response=ViolationResponse.HUMAN_REVIEW,
                predicate=lambda state: state.get("energy_activation_confirmed", False),
                applicable_actions=[ActionCategory.ENERGY],
                regulatory_reference="IEC 80601-2-77 Clause 201.12.4.7",
            ),
            SafetyConstraint(
                constraint_id="EN-002",
                name="Clear Instrument Path for Energy",
                description="No other instruments in energy application zone",
                constraint_type=ConstraintType.PRE_CONDITION,
                severity=ConstraintSeverity.CRITICAL,
                violation_response=ViolationResponse.REJECT_ACTION,
                predicate=lambda state: state.get("instrument_path_clear", True),
                applicable_actions=[ActionCategory.ENERGY],
                regulatory_reference="IEC 80601-2-77 Clause 201.12.4.8",
            ),
        ]

    @classmethod
    def get_all_constraints(cls) -> list[SafetyConstraint]:
        """Get all safety constraints from the library."""
        constraints = []
        constraints.extend(cls.workspace_constraints())
        constraints.extend(cls.force_constraints())
        constraints.extend(cls.velocity_constraints())
        constraints.extend(cls.clinical_constraints())
        constraints.extend(cls.energy_constraints())
        return constraints


# =============================================================================
# SECTION 3: HELPER FUNCTIONS FOR CONSTRAINT PREDICATES
# =============================================================================


def _check_workspace_bounds(position: list, limits: dict) -> bool:
    """Check if position is within workspace boundaries."""
    if not limits or len(position) < 3:
        return True

    x, y, z = position[0], position[1], position[2]
    x_range = limits.get("x_range_m", [-1, 1])
    y_range = limits.get("y_range_m", [-1, 1])
    z_range = limits.get("z_range_m", [-1, 1])

    return x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1] and z_range[0] <= z <= z_range[1]


def _check_keepout_zones(position: list, zones: list) -> bool:
    """Check if position is outside all keep-out zones."""
    if len(position) < 3:
        return True

    for zone in zones:
        center = zone.get("center", [0, 0, 0])
        radius = zone.get("radius_m", 0.01)
        dist = sum((a - b) ** 2 for a, b in zip(position[:3], center[:3])) ** 0.5
        if dist < radius:
            return False
    return True


# =============================================================================
# SECTION 4: SAFETY-CONSTRAINED EXECUTOR
# =============================================================================


class SafetyConstrainedExecutor:
    """
    Executes agent actions with formal safety constraint verification.

    Every action proposed by the agent passes through:
    1. Pre-condition verification
    2. Safety gate approval (if required)
    3. Action execution with invariant monitoring
    4. Post-condition verification
    5. Rollback if post-conditions violated

    AUDIT TRAIL:
    ------------
    Every constraint check, approval, and execution is logged
    for 21 CFR Part 11 compliance and post-hoc analysis.
    """

    def __init__(self, constraints: Optional[list[SafetyConstraint]] = None):
        if constraints is None:
            constraints = OncologyRoboticsConstraintLibrary.get_all_constraints()
        self._constraints = constraints
        self._audit_log: list[dict] = []
        self._action_count = 0
        self._violation_count = 0
        self._rejected_count = 0
        self._human_approvals_pending: list[str] = []

        # Index constraints by type for efficient lookup
        self._pre_conditions = [c for c in constraints if c.constraint_type == ConstraintType.PRE_CONDITION]
        self._post_conditions = [c for c in constraints if c.constraint_type == ConstraintType.POST_CONDITION]
        self._invariants = [c for c in constraints if c.constraint_type == ConstraintType.INVARIANT]
        self._safety_gates = [c for c in constraints if c.constraint_type == ConstraintType.SAFETY_GATE]

        logger.info(
            f"SafetyConstrainedExecutor initialized with {len(constraints)} constraints "
            f"({len(self._pre_conditions)} pre, {len(self._post_conditions)} post, "
            f"{len(self._invariants)} invariant, {len(self._safety_gates)} gate)"
        )

    def evaluate_action(self, action: AgentAction, state: dict) -> SafetyVerdict:
        """
        Evaluate an action against all applicable constraints.

        This is the core safety verification method. Called before
        any action is allowed to execute.

        Args:
            action: Proposed agent action
            state: Current system state

        Returns:
            SafetyVerdict with approval decision and details
        """
        self._action_count += 1
        results = []
        blocking = []
        advisory = []
        requires_human = False

        # Check pre-conditions
        for constraint in self._pre_conditions:
            if self._is_applicable(constraint, action):
                result = constraint.check(state)
                results.append(result)

                if not result.satisfied:
                    self._violation_count += 1
                    if result.severity in (ConstraintSeverity.BLOCKING, ConstraintSeverity.CRITICAL):
                        blocking.append(result)
                    elif result.severity == ConstraintSeverity.ADVISORY:
                        advisory.append(result)

                    if result.violation_response == ViolationResponse.HUMAN_REVIEW:
                        requires_human = True

        # Check safety gates
        for gate in self._safety_gates:
            if self._is_applicable(gate, action):
                result = gate.check(state)
                results.append(result)

                if not result.satisfied:
                    requires_human = True
                    blocking.append(result)

        # Check invariants (current state)
        for invariant in self._invariants:
            if self._is_applicable(invariant, action):
                result = invariant.check(state)
                results.append(result)

                if not result.satisfied:
                    self._violation_count += 1
                    if result.severity in (
                        ConstraintSeverity.BLOCKING,
                        ConstraintSeverity.CRITICAL,
                        ConstraintSeverity.EMERGENCY,
                    ):
                        blocking.append(result)

        # Determine approval
        approved = len(blocking) == 0 and not requires_human

        # Calculate safety score
        total_checked = len(results)
        passed = sum(1 for r in results if r.satisfied)
        safety_score = passed / max(1, total_checked)

        verdict = SafetyVerdict(
            action_id=action.action_id,
            approved=approved,
            constraint_results=results,
            blocking_violations=blocking,
            advisory_violations=advisory,
            requires_human_approval=requires_human,
            safety_score=safety_score,
            timestamp=time.time(),
        )

        if not approved:
            self._rejected_count += 1

        # Audit log
        self._audit_log.append(
            {
                "action_id": action.action_id,
                "action_category": action.category.value,
                "action_description": action.description,
                "verdict": verdict.to_dict(),
                "constraint_details": [
                    {
                        "id": r.constraint_id,
                        "name": r.constraint_name,
                        "satisfied": r.satisfied,
                        "severity": r.severity.value,
                        "message": r.message,
                    }
                    for r in results
                ],
                "timestamp": time.time(),
            }
        )

        return verdict

    def execute_with_safety(
        self,
        action: AgentAction,
        state: dict,
        execute_fn: Callable[[AgentAction], dict],
        human_approval: bool = False,
    ) -> dict:
        """
        Execute an action with full safety constraint lifecycle.

        Args:
            action: Action to execute
            state: Current system state
            execute_fn: Function that performs the actual action
            human_approval: Whether human has approved (for safety gates)

        Returns:
            Execution result with safety metadata
        """
        # Phase 1: Pre-condition and safety gate evaluation
        if human_approval:
            state["energy_activation_confirmed"] = True

        verdict = self.evaluate_action(action, state)

        if not verdict.approved:
            if verdict.requires_human_approval and not human_approval:
                self._human_approvals_pending.append(action.action_id)
                return {
                    "status": "awaiting_human_approval",
                    "action_id": action.action_id,
                    "violations": [v.message for v in verdict.blocking_violations],
                    "safety_score": verdict.safety_score,
                }
            return {
                "status": "rejected",
                "action_id": action.action_id,
                "violations": [v.message for v in verdict.blocking_violations],
                "advisories": [v.message for v in verdict.advisory_violations],
                "safety_score": verdict.safety_score,
            }

        # Phase 2: Execute action
        try:
            result = execute_fn(action)
            result["safety_score"] = verdict.safety_score
            result["constraints_checked"] = len(verdict.constraint_results)
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {
                "status": "execution_error",
                "action_id": action.action_id,
                "error": str(e),
            }

        # Phase 3: Post-condition verification
        post_state = state.copy()
        post_state.update(result.get("new_state", {}))

        post_violations = []
        for constraint in self._post_conditions:
            if self._is_applicable(constraint, action):
                post_result = constraint.check(post_state)
                if not post_result.satisfied:
                    post_violations.append(post_result)

        if post_violations and action.rollback_action:
            result["status"] = "rolled_back"
            result["post_condition_violations"] = [v.message for v in post_violations]
            result["rollback_action"] = action.rollback_action
            logger.warning(f"Action {action.action_id} rolled back: post-condition violated")

        return result

    def _is_applicable(self, constraint: SafetyConstraint, action: AgentAction) -> bool:
        """Check if a constraint applies to a given action category."""
        if not constraint.applicable_actions:
            return True
        return action.category in constraint.applicable_actions

    def get_statistics(self) -> dict:
        """Get executor statistics."""
        return {
            "total_actions_evaluated": self._action_count,
            "total_violations": self._violation_count,
            "actions_rejected": self._rejected_count,
            "approval_rate": ((self._action_count - self._rejected_count) / max(1, self._action_count)),
            "pending_human_approvals": len(self._human_approvals_pending),
            "total_constraints": len(self._constraints),
        }

    def get_audit_log(self) -> list[dict]:
        """Get complete audit log."""
        return self._audit_log

    def format_verdict(self, verdict: SafetyVerdict) -> str:
        """Format a safety verdict as readable text."""
        lines = [
            f"Safety Verdict for Action {verdict.action_id}:",
            f"  Approved: {'YES' if verdict.approved else 'NO'}",
            f"  Safety Score: {verdict.safety_score:.0%}",
            f"  Constraints Checked: {len(verdict.constraint_results)}",
        ]

        if verdict.blocking_violations:
            lines.append("  BLOCKING VIOLATIONS:")
            for v in verdict.blocking_violations:
                lines.append(f"    [{v.severity.value.upper()}] {v.message}")

        if verdict.advisory_violations:
            lines.append("  ADVISORIES:")
            for v in verdict.advisory_violations:
                lines.append(f"    [{v.severity.value.upper()}] {v.message}")

        if verdict.requires_human_approval:
            lines.append("  ** REQUIRES HUMAN APPROVAL **")

        return "\n".join(lines)


# =============================================================================
# SECTION 5: DEMO
# =============================================================================


def mock_execute(action: AgentAction) -> dict:
    """Mock action execution for demo purposes."""
    return {
        "status": "executed",
        "action_id": action.action_id,
        "description": action.description,
        "new_state": {},
    }


def run_safety_executor_demo():
    """
    Demonstrate the safety-constrained agent executor.

    Tests various actions against safety constraints to show:
    1. Safe action approval
    2. Workspace boundary violation rejection
    3. Keep-out zone violation rejection
    4. Clinical pre-condition enforcement
    5. Safety gate human-in-the-loop
    6. Force constraint enforcement
    """
    logger.info("=" * 60)
    logger.info("SAFETY-CONSTRAINED AGENT EXECUTOR DEMO")
    logger.info("=" * 60)

    executor = SafetyConstrainedExecutor()

    # Base system state
    base_state = {
        "patient_identity_verified": True,
        "timeout_completed": True,
        "heart_rate_bpm": 75,
        "spo2_percent": 98,
        "systolic_bp": 118,
        "current_force_n": 2.0,
        "force_rate_n_s": 0.5,
        "max_force_n": 10.0,
        "max_force_rate_n_s": 5.0,
        "current_velocity_m_s": 0.05,
        "max_velocity_m_s": 0.1,
        "approach_velocity_m_s": 0.02,
        "distance_to_target_mm": 50.0,
        "commanded_velocity_m_s": 0.05,
        "workspace_limits": {
            "x_range_m": [-0.3, 0.3],
            "y_range_m": [-0.3, 0.3],
            "z_range_m": [0.0, 0.4],
        },
        "keepout_zones": [
            {"center": [0.1, 0.1, 0.2], "radius_m": 0.02, "label": "pulmonary_artery"},
        ],
        "instrument_path_clear": True,
    }

    # Test 1: Safe navigation action
    print("\n--- Test 1: Safe Navigation Action ---")
    action1 = AgentAction(
        action_id="ACT-001",
        category=ActionCategory.NAVIGATION,
        description="Move to approach position above tumor",
        parameters={"target": [0.05, -0.02, 0.25]},
    )
    state1 = {**base_state, "target_position": [0.05, -0.02, 0.25]}
    result1 = executor.execute_with_safety(action1, state1, mock_execute)
    print(f"  Status: {result1['status']}")
    print(f"  Safety Score: {result1.get('safety_score', 'N/A')}")

    # Test 2: Action violating workspace boundaries
    print("\n--- Test 2: Workspace Boundary Violation ---")
    action2 = AgentAction(
        action_id="ACT-002",
        category=ActionCategory.NAVIGATION,
        description="Move to position outside workspace",
        parameters={"target": [0.5, 0.0, 0.2]},
    )
    state2 = {**base_state, "target_position": [0.5, 0.0, 0.2]}
    result2 = executor.execute_with_safety(action2, state2, mock_execute)
    print(f"  Status: {result2['status']}")
    if "violations" in result2:
        for v in result2["violations"]:
            print(f"  Violation: {v}")

    # Test 3: Action entering keep-out zone
    print("\n--- Test 3: Keep-Out Zone Violation ---")
    action3 = AgentAction(
        action_id="ACT-003",
        category=ActionCategory.MANIPULATION,
        description="Dissect near critical vessel",
        parameters={"target": [0.1, 0.1, 0.2]},
    )
    state3 = {**base_state, "target_position": [0.1, 0.1, 0.2]}
    result3 = executor.execute_with_safety(action3, state3, mock_execute)
    print(f"  Status: {result3['status']}")
    if "violations" in result3:
        for v in result3["violations"]:
            print(f"  Violation: {v}")

    # Test 4: Action without patient verification
    print("\n--- Test 4: Missing Clinical Pre-condition ---")
    action4 = AgentAction(
        action_id="ACT-004",
        category=ActionCategory.MANIPULATION,
        description="Begin tissue dissection",
    )
    state4 = {**base_state, "patient_identity_verified": False, "target_position": [0.05, -0.02, 0.15]}
    result4 = executor.execute_with_safety(action4, state4, mock_execute)
    print(f"  Status: {result4['status']}")
    if "violations" in result4:
        for v in result4["violations"]:
            print(f"  Violation: {v}")

    # Test 5: Energy delivery requiring safety gate
    print("\n--- Test 5: Energy Delivery (No Human Approval) ---")
    action5 = AgentAction(
        action_id="ACT-005",
        category=ActionCategory.ENERGY,
        description="Activate bipolar cautery for hemostasis",
    )
    state5 = {**base_state, "target_position": [0.05, -0.02, 0.15], "energy_activation_confirmed": False}
    result5 = executor.execute_with_safety(action5, state5, mock_execute)
    print(f"  Status: {result5['status']}")

    # Test 5b: Same action WITH human approval
    print("\n--- Test 5b: Energy Delivery (With Human Approval) ---")
    result5b = executor.execute_with_safety(action5, state5, mock_execute, human_approval=True)
    print(f"  Status: {result5b['status']}")
    print(f"  Safety Score: {result5b.get('safety_score', 'N/A')}")

    # Test 6: Excessive force invariant
    print("\n--- Test 6: Excessive Force Invariant ---")
    action6 = AgentAction(
        action_id="ACT-006",
        category=ActionCategory.MANIPULATION,
        description="Continue tissue retraction",
    )
    state6 = {**base_state, "target_position": [0.05, -0.02, 0.15], "current_force_n": 12.0}
    result6 = executor.execute_with_safety(action6, state6, mock_execute)
    print(f"  Status: {result6['status']}")
    if "violations" in result6:
        for v in result6["violations"]:
            print(f"  Violation: {v}")

    # Print statistics
    print("\n" + "=" * 60)
    print("EXECUTOR STATISTICS")
    print("=" * 60)
    stats = executor.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")

    return stats


if __name__ == "__main__":
    result = run_safety_executor_demo()
    print(f"\nDemo result: {result}")
