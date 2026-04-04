"""Deployment validator: pre-deployment validation of agent connections, integrations, safety gates."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class CheckCategory(Enum):
    AGENT_CONNECTIVITY = "agent_connectivity"
    INTEGRATION = "integration"
    SAFETY_GATE = "safety_gate"
    DATA_FLOW = "data_flow"
    SECURITY = "security"
    COMPLIANCE = "compliance"


class CheckResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class ValidationCheck:
    check_id: str
    name: str
    category: CheckCategory
    description: str = ""
    result: CheckResult = CheckResult.SKIP
    message: str = ""
    checked_at: datetime | None = None
    required: bool = True


@dataclass
class ValidationReport:
    report_id: str = ""
    environment: str = "staging"
    checks: list[ValidationCheck] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def passed(self) -> bool:
        return all(c.result == CheckResult.PASS for c in self.checks if c.required)

    @property
    def pass_rate(self) -> float:
        if not self.checks:
            return 0.0
        return round(sum(1 for c in self.checks if c.result == CheckResult.PASS) / len(self.checks) * 100, 1)


STANDARD_CHECKS: list[dict[str, str]] = [
    {"id": "CHK-001", "name": "Orchestrator Agent Online", "cat": "agent_connectivity"},
    {"id": "CHK-002", "name": "Safety Monitor Agent Online", "cat": "agent_connectivity"},
    {"id": "CHK-003", "name": "Data Management Agent Online", "cat": "agent_connectivity"},
    {"id": "CHK-004", "name": "Robot Gateway Agent Online", "cat": "agent_connectivity"},
    {"id": "CHK-005", "name": "FHIR R4 Endpoint Reachable", "cat": "integration"},
    {"id": "CHK-006", "name": "DICOM PACS Connectivity", "cat": "integration"},
    {"id": "CHK-007", "name": "EDC System Integration", "cat": "integration"},
    {"id": "CHK-008", "name": "MCP Server Handshake", "cat": "integration"},
    {"id": "CHK-009", "name": "Robot Safety Gate - Force Limits", "cat": "safety_gate"},
    {"id": "CHK-010", "name": "Robot Safety Gate - E-Stop", "cat": "safety_gate"},
    {"id": "CHK-011", "name": "Consent Verification Gate", "cat": "safety_gate"},
    {"id": "CHK-012", "name": "Audit Trail Integrity", "cat": "data_flow"},
    {"id": "CHK-013", "name": "PKI Certificate Validity", "cat": "security"},
    {"id": "CHK-014", "name": "RBAC Policy Loaded", "cat": "security"},
    {"id": "CHK-015", "name": "Part 11 Controls Active", "cat": "compliance"},
]


class DeploymentValidator:
    """Validates pre-deployment readiness: agent connections, integrations, safety gates."""

    def __init__(self, environment: str = "staging") -> None:
        self.environment = environment
        self.checks: list[ValidationCheck] = []
        self.reports: list[ValidationReport] = []

    def load_standard_checks(self) -> None:
        for spec in STANDARD_CHECKS:
            check = ValidationCheck(check_id=spec["id"], name=spec["name"],
                                    category=CheckCategory(spec["cat"]))
            self.checks.append(check)

    def run_check(self, check_id: str, passed: bool, message: str = "") -> ValidationCheck:
        for check in self.checks:
            if check.check_id == check_id:
                check.result = CheckResult.PASS if passed else CheckResult.FAIL
                check.message = message
                check.checked_at = datetime.utcnow()
                return check
        raise ValueError(f"Check {check_id} not found")

    def run_all(self, results: dict[str, bool] | None = None) -> ValidationReport:
        results = results or {}
        for check in self.checks:
            passed = results.get(check.check_id, True)
            check.result = CheckResult.PASS if passed else CheckResult.FAIL
            check.checked_at = datetime.utcnow()
        report = ValidationReport(report_id=f"VR-{self.environment}-{len(self.reports)+1}",
                                  environment=self.environment, checks=list(self.checks))
        self.reports.append(report)
        return report

    def failed_checks(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.result == CheckResult.FAIL]

    def summary(self) -> dict[str, object]:
        by_result = {r.value: sum(1 for c in self.checks if c.result == r) for r in CheckResult}
        return {"environment": self.environment, "total_checks": len(self.checks),
                "by_result": by_result, "reports_generated": len(self.reports)}


def main() -> None:
    validator = DeploymentValidator("staging")
    validator.load_standard_checks()
    print(f"Loaded {len(validator.checks)} standard checks")
    failures = {"CHK-006": False, "CHK-010": False}
    report = validator.run_all(failures)
    print(f"Report: {report.report_id}, passed: {report.passed}, rate: {report.pass_rate}%")
    for fc in validator.failed_checks():
        print(f"  FAILED: {fc.name}")
    print(f"Summary: {validator.summary()}")


if __name__ == "__main__":
    main()
