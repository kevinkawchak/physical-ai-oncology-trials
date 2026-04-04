"""Robot execution gateway: task-order generation, site matching, safety gates, procedure state machine."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ProcedureState(Enum):
    IDLE = "idle"
    PRE_CHECK = "pre_check"
    READY = "ready"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


class SafetyGateResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    OVERRIDE = "override"


@dataclass
class TaskOrder:
    order_id: str = field(default_factory=lambda: f"TO-{uuid.uuid4().hex[:8]}")
    procedure_type: str = ""
    robot_id: str = ""
    site_id: str = ""
    subject_id: str = ""
    priority: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SafetyGate:
    gate_id: str
    description: str
    required: bool = True
    result: SafetyGateResult = SafetyGateResult.FAIL
    checked_at: datetime | None = None


STANDARD_GATES: list[dict[str, str]] = [
    {"id": "SG-01", "desc": "Patient identity confirmed"},
    {"id": "SG-02", "desc": "Consent verified and current"},
    {"id": "SG-03", "desc": "Robot self-check passed"},
    {"id": "SG-04", "desc": "Sterile field confirmed"},
    {"id": "SG-05", "desc": "Surgeon override available"},
]


class RobotExecutionGateway:
    """Manages task-order lifecycle, safety gate checks, and procedure state transitions."""

    def __init__(self) -> None:
        self.orders: dict[str, TaskOrder] = {}
        self.states: dict[str, ProcedureState] = {}
        self.gates: dict[str, list[SafetyGate]] = {}

    def create_task_order(self, procedure_type: str, robot_id: str,
                          site_id: str, subject_id: str) -> TaskOrder:
        order = TaskOrder(procedure_type=procedure_type, robot_id=robot_id,
                          site_id=site_id, subject_id=subject_id)
        self.orders[order.order_id] = order
        self.states[order.order_id] = ProcedureState.IDLE
        self.gates[order.order_id] = [
            SafetyGate(gate_id=g["id"], description=g["desc"]) for g in STANDARD_GATES
        ]
        return order

    def check_safety_gate(self, order_id: str, gate_id: str, passed: bool) -> SafetyGateResult:
        result = SafetyGateResult.PASS if passed else SafetyGateResult.FAIL
        for gate in self.gates[order_id]:
            if gate.gate_id == gate_id:
                gate.result = result
                gate.checked_at = datetime.utcnow()
                break
        return result

    def all_gates_passed(self, order_id: str) -> bool:
        return all(
            g.result == SafetyGateResult.PASS for g in self.gates[order_id] if g.required
        )

    def transition(self, order_id: str, target: ProcedureState) -> bool:
        current = self.states[order_id]
        valid_transitions: dict[ProcedureState, set[ProcedureState]] = {
            ProcedureState.IDLE: {ProcedureState.PRE_CHECK},
            ProcedureState.PRE_CHECK: {ProcedureState.READY, ProcedureState.ABORTED},
            ProcedureState.READY: {ProcedureState.EXECUTING},
            ProcedureState.EXECUTING: {ProcedureState.PAUSED, ProcedureState.COMPLETED, ProcedureState.ABORTED},
            ProcedureState.PAUSED: {ProcedureState.EXECUTING, ProcedureState.ABORTED},
        }
        if target in valid_transitions.get(current, set()):
            self.states[order_id] = target
            return True
        return False

    def start_procedure(self, order_id: str) -> bool:
        self.transition(order_id, ProcedureState.PRE_CHECK)
        if self.all_gates_passed(order_id):
            self.transition(order_id, ProcedureState.READY)
            return self.transition(order_id, ProcedureState.EXECUTING)
        self.transition(order_id, ProcedureState.ABORTED)
        return False

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for state in self.states.values():
            counts[state.value] = counts.get(state.value, 0) + 1
        return counts


def main() -> None:
    gw = RobotExecutionGateway()
    order = gw.create_task_order("biopsy", "DVRK-01", "SITE-A", "SUBJ-01")
    for gate_def in STANDARD_GATES:
        gw.check_safety_gate(order.order_id, gate_def["id"], passed=True)
    started = gw.start_procedure(order.order_id)
    print(f"Procedure started: {started}, state: {gw.states[order.order_id].value}")
    gw.transition(order.order_id, ProcedureState.COMPLETED)
    print(f"Final state: {gw.states[order.order_id].value}")
    print(f"Gateway summary: {gw.summary()}")


if __name__ == "__main__":
    main()
