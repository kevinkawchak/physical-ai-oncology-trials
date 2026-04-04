"""Validation evidence generator: IQ/OQ/PQ documentation, 21 CFR Part 11 compliance, CSV/CSA."""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


class ValidationPhase(Enum):
    IQ = "installation_qualification"
    OQ = "operational_qualification"
    PQ = "performance_qualification"


class TestResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"
    BLOCKED = "blocked"


class ComplianceArea(Enum):
    PART_11 = "21_cfr_part_11"
    ANNEX_11 = "eu_annex_11"
    CSV = "computer_system_validation"
    CSA = "computer_software_assurance"


@dataclass
class TestCase:
    test_id: str = field(default_factory=lambda: f"TC-{uuid.uuid4().hex[:6]}")
    phase: ValidationPhase = ValidationPhase.OQ
    title: str = ""
    expected_result: str = ""
    actual_result: str = ""
    result: TestResult = TestResult.NOT_RUN
    executed_by: str = ""
    evidence_hash: str = ""

    def execute(self, actual: str, tester: str) -> TestResult:
        self.actual_result = actual
        self.executed_by = tester
        self.result = TestResult.PASS if actual.strip() == self.expected_result.strip() else TestResult.FAIL
        content = f"{self.test_id}|{actual}|{datetime.utcnow().isoformat()}"
        self.evidence_hash = hashlib.sha256(content.encode()).hexdigest()[:32]
        return self.result


@dataclass
class Part11Check:
    check_id: str
    requirement: str
    description: str
    compliant: bool = False
    evidence: str = ""


@dataclass
class ValidationProtocol:
    protocol_id: str = field(default_factory=lambda: f"VP-{uuid.uuid4().hex[:6]}")
    system_name: str = ""
    version: str = ""
    phase: ValidationPhase = ValidationPhase.OQ
    test_cases: list[TestCase] = field(default_factory=list)


PART_11_REQUIREMENTS: list[tuple[str, str]] = [
    ("P11-01", "Electronic signatures unique to one individual"),
    ("P11-02", "Audit trails record date/time of entries"),
    ("P11-03", "System access limited to authorized individuals"),
    ("P11-04", "Authority checks enforce permitted operations"),
    ("P11-05", "Input checks for validity of data values"),
    ("P11-06", "Records retrievable throughout retention period"),
]


class ValidationEvidenceGenerator:
    """Generates IQ/OQ/PQ evidence, performs 21 CFR Part 11 checks, and supports CSV/CSA."""

    def __init__(self, system_name: str, version: str) -> None:
        self.system_name = system_name
        self.version = version
        self.protocols: list[ValidationProtocol] = []
        self.part11_checks: list[Part11Check] = []

    def create_protocol(self, phase: ValidationPhase) -> ValidationProtocol:
        protocol = ValidationProtocol(system_name=self.system_name, version=self.version, phase=phase)
        self.protocols.append(protocol)
        return protocol

    def add_test_case(self, protocol_id: str, title: str, expected: str) -> TestCase:
        tc = TestCase(title=title, expected_result=expected)
        for p in self.protocols:
            if p.protocol_id == protocol_id:
                tc.phase = p.phase
                p.test_cases.append(tc)
                return tc
        msg = f"Protocol {protocol_id} not found"
        raise KeyError(msg)

    def run_part11_assessment(self, evidence_map: dict[str, str]) -> list[Part11Check]:
        self.part11_checks = []
        for req_id, desc in PART_11_REQUIREMENTS:
            evidence = evidence_map.get(req_id, "")
            check = Part11Check(check_id=req_id, requirement=desc,
                                description=desc, compliant=bool(evidence), evidence=evidence)
            self.part11_checks.append(check)
        return self.part11_checks

    def part11_compliance_score(self) -> float:
        if not self.part11_checks:
            return 0.0
        compliant = sum(1 for c in self.part11_checks if c.compliant)
        return round(compliant / len(self.part11_checks), 4)

    def traceability_matrix(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for p in self.protocols:
            for tc in p.test_cases:
                rows.append({"protocol": p.protocol_id, "phase": p.phase.value,
                             "test": tc.test_id, "title": tc.title, "result": tc.result.value})
        return rows

    def summary_report(self) -> dict[str, object]:
        all_tests = [tc for p in self.protocols for tc in p.test_cases]
        passed = sum(1 for t in all_tests if t.result == TestResult.PASS)
        failed = sum(1 for t in all_tests if t.result == TestResult.FAIL)
        return {
            "system": self.system_name, "version": self.version,
            "protocols": len(self.protocols), "total_tests": len(all_tests),
            "passed": passed, "failed": failed,
            "part11_score": self.part11_compliance_score(),
        }


def main() -> None:
    gen = ValidationEvidenceGenerator("RoboticTrialPlatform", "2.1.0")
    oq = gen.create_protocol(ValidationPhase.OQ)
    for title, expected in [("Login auth", "access_granted"), ("Audit trail", "entry_logged"),
                             ("RBAC", "restricted")]:
        tc = gen.add_test_case(oq.protocol_id, title, expected)
        tc.execute(expected, "QA_Lead")
    evidence = {"P11-01": "SSO integration verified", "P11-02": "Audit trail module tested",
                "P11-03": "RBAC configured", "P11-04": "Role matrix approved"}
    gen.run_part11_assessment(evidence)
    print(f"Traceability: {gen.traceability_matrix()}")
    print(f"Report: {gen.summary_report()}")


if __name__ == "__main__":
    main()
