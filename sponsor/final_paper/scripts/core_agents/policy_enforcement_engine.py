"""Policy enforcement engine: OPA/Rego-style policy rules, safety gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PolicyDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"


class PolicyCategory(Enum):
    SAFETY = "safety"
    ACCESS_CONTROL = "access_control"
    DATA_GOVERNANCE = "data_governance"
    ROBOT_OPERATION = "robot_operation"
    REGULATORY = "regulatory"


@dataclass
class PolicyRule:
    rule_id: str
    name: str
    category: PolicyCategory
    description: str = ""
    condition: str = ""
    priority: int = 100
    enabled: bool = True

    def evaluate(self, context: dict[str, object]) -> PolicyDecision:
        """Evaluate rule against context using simple condition matching."""
        if not self.enabled:
            return PolicyDecision.ALLOW
        if self.condition == "force_limit_exceeded":
            force = float(context.get("force_n", 0))
            limit = float(context.get("force_limit_n", 50))
            return PolicyDecision.DENY if force > limit else PolicyDecision.ALLOW
        if self.condition == "consent_required":
            return PolicyDecision.ALLOW if context.get("consent_signed") else PolicyDecision.DENY
        if self.condition == "role_authorized":
            allowed_roles = context.get("allowed_roles", [])
            user_role = context.get("user_role", "")
            return PolicyDecision.ALLOW if user_role in allowed_roles else PolicyDecision.DENY
        if self.condition == "data_export_check":
            has_approval = context.get("export_approved", False)
            return PolicyDecision.ALLOW if has_approval else PolicyDecision.WARN
        if self.condition == "calibration_current":
            return PolicyDecision.ALLOW if context.get("calibrated") else PolicyDecision.DENY
        return PolicyDecision.ALLOW


@dataclass
class PolicyEvaluation:
    rule_id: str
    decision: PolicyDecision
    context_summary: str = ""
    evaluated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SafetyGate:
    gate_id: str
    name: str
    rules: list[str] = field(default_factory=list)
    all_must_pass: bool = True


class PolicyEnforcementEngine:
    """Enforces OPA/Rego-style policies and safety gates for agent operations."""

    def __init__(self) -> None:
        self.rules: dict[str, PolicyRule] = {}
        self.gates: dict[str, SafetyGate] = {}
        self.evaluations: list[PolicyEvaluation] = []

    def register_rule(self, rule: PolicyRule) -> str:
        self.rules[rule.rule_id] = rule
        return rule.rule_id

    def evaluate_rule(self, rule_id: str, context: dict[str, object]) -> PolicyEvaluation:
        rule = self.rules[rule_id]
        decision = rule.evaluate(context)
        evaluation = PolicyEvaluation(rule_id=rule_id, decision=decision, context_summary=str(list(context.keys())))
        self.evaluations.append(evaluation)
        return evaluation

    def define_gate(self, gate_id: str, name: str, rule_ids: list[str], all_must_pass: bool = True) -> SafetyGate:
        gate = SafetyGate(gate_id=gate_id, name=name, rules=rule_ids, all_must_pass=all_must_pass)
        self.gates[gate_id] = gate
        return gate

    def check_gate(self, gate_id: str, context: dict[str, object]) -> tuple[PolicyDecision, list[PolicyEvaluation]]:
        gate = self.gates[gate_id]
        results: list[PolicyEvaluation] = []
        for rule_id in gate.rules:
            ev = self.evaluate_rule(rule_id, context)
            results.append(ev)
        if gate.all_must_pass:
            if any(r.decision == PolicyDecision.DENY for r in results):
                return PolicyDecision.DENY, results
            if any(r.decision == PolicyDecision.WARN for r in results):
                return PolicyDecision.WARN, results
            return PolicyDecision.ALLOW, results
        if all(r.decision == PolicyDecision.DENY for r in results):
            return PolicyDecision.DENY, results
        return PolicyDecision.ALLOW, results

    def build_standard_policies(self) -> None:
        policies = [
            PolicyRule("POL-001", "Force Limit Check", PolicyCategory.SAFETY, condition="force_limit_exceeded"),
            PolicyRule("POL-002", "Consent Verification", PolicyCategory.REGULATORY, condition="consent_required"),
            PolicyRule("POL-003", "Role Authorization", PolicyCategory.ACCESS_CONTROL, condition="role_authorized"),
            PolicyRule("POL-004", "Data Export", PolicyCategory.DATA_GOVERNANCE, condition="data_export_check"),
            PolicyRule("POL-005", "Calibration Check", PolicyCategory.ROBOT_OPERATION, condition="calibration_current"),
        ]
        for p in policies:
            self.register_rule(p)

    def audit_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {"allow": 0, "deny": 0, "warn": 0}
        for ev in self.evaluations:
            counts[ev.decision.value] += 1
        return counts


def main() -> None:
    engine = PolicyEnforcementEngine()
    engine.build_standard_policies()
    engine.define_gate("ROBOT_START", "Robot Procedure Start", ["POL-001", "POL-002", "POL-005"])
    context = {"force_n": 30, "force_limit_n": 50, "consent_signed": True, "calibrated": True}
    decision, results = engine.check_gate("ROBOT_START", context)
    print(f"Gate decision: {decision.value}")
    context_fail = {"force_n": 60, "force_limit_n": 50, "consent_signed": True, "calibrated": False}
    decision2, results2 = engine.check_gate("ROBOT_START", context_fail)
    print(f"Gate decision (fail): {decision2.value}")
    for r in results2:
        print(f"  {r.rule_id}: {r.decision.value}")
    print(f"Audit: {engine.audit_summary()}")


if __name__ == "__main__":
    main()
