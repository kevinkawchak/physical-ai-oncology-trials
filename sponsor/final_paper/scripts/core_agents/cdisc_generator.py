"""CDISC generator: SDTM/ADaM dataset generation, Define.xml, P21 validation checks."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class CDISCStandard(Enum):
    SDTM = "SDTM"
    ADAM = "ADaM"


class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class CDISCVariable:
    name: str
    label: str
    data_type: str = "text"
    core: str = "Req"
    origin: str = "CRF"
    code_list: str = ""


@dataclass
class CDISCDomain:
    domain_code: str
    description: str
    standard: CDISCStandard = CDISCStandard.SDTM
    variables: list[CDISCVariable] = field(default_factory=list)
    records: list[dict[str, str]] = field(default_factory=list)


@dataclass
class P21Finding:
    rule_id: str
    severity: ValidationSeverity
    domain: str
    variable: str
    message: str


SDTM_DOMAINS: dict[str, list[str]] = {
    "DM": ["STUDYID", "DOMAIN", "USUBJID", "SUBJID", "AGE", "SEX", "RACE", "ARM"],
    "AE": ["STUDYID", "DOMAIN", "USUBJID", "AETERM", "AESEV", "AESER", "AESTDTC"],
    "LB": ["STUDYID", "DOMAIN", "USUBJID", "LBTESTCD", "LBORRES", "LBDTC"],
    "VS": ["STUDYID", "DOMAIN", "USUBJID", "VSTESTCD", "VSORRES", "VSDTC"],
    "RS": ["STUDYID", "DOMAIN", "USUBJID", "RSTESTCD", "RSSTRESC", "RSDTC"],
}


class CDISCGenerator:
    """Generates SDTM/ADaM datasets, Define.xml metadata, and runs P21 validation."""

    def __init__(self, study_id: str) -> None:
        self.study_id = study_id
        self.domains: dict[str, CDISCDomain] = {}
        self.findings: list[P21Finding] = []

    def create_domain(self, code: str, standard: CDISCStandard = CDISCStandard.SDTM) -> CDISCDomain:
        template_vars = SDTM_DOMAINS.get(code, [])
        variables = [CDISCVariable(name=v, label=v.lower()) for v in template_vars]
        domain = CDISCDomain(domain_code=code, description=f"{code} domain", standard=standard, variables=variables)
        self.domains[code] = domain
        return domain

    def add_record(self, domain_code: str, record: dict[str, str]) -> None:
        domain = self.domains[domain_code]
        record.setdefault("STUDYID", self.study_id)
        record.setdefault("DOMAIN", domain_code)
        domain.records.append(record)

    def generate_define_xml(self) -> str:
        lines = [f'<?xml version="1.0"?>', f'<!-- Define.xml for {self.study_id} -->']
        for code, domain in self.domains.items():
            lines.append(f'  <ItemGroupDef OID="IG.{code}" Name="{code}">')
            for var in domain.variables:
                lines.append(f'    <ItemRef ItemOID="{code}.{var.name}" Mandatory="{var.core == "Req"}"/>')
            lines.append("  </ItemGroupDef>")
        return "\n".join(lines)

    def run_p21_checks(self) -> list[P21Finding]:
        self.findings.clear()
        for code, domain in self.domains.items():
            for rec in domain.records:
                for var in domain.variables:
                    if var.core == "Req" and not rec.get(var.name):
                        self.findings.append(P21Finding(
                            rule_id="SD0001", severity=ValidationSeverity.ERROR,
                            domain=code, variable=var.name,
                            message=f"Required variable {var.name} missing in {code}"))
        return self.findings

    def export_dataset(self, domain_code: str, output_dir: Path) -> Path:
        domain = self.domains[domain_code]
        out_path = output_dir / f"{domain_code.lower()}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"study": self.study_id, "domain": domain_code,
                   "generated": datetime.utcnow().isoformat(), "records": domain.records}
        out_path.write_text(json.dumps(payload, indent=2))
        return out_path

    def validation_summary(self) -> dict[str, int]:
        return {s.value: sum(1 for f in self.findings if f.severity == s) for s in ValidationSeverity}


def main() -> None:
    gen = CDISCGenerator("STU-001")
    gen.create_domain("DM")
    gen.create_domain("AE")
    gen.add_record("DM", {"USUBJID": "STU-001-001", "SUBJID": "001", "AGE": "58", "SEX": "M"})
    gen.add_record("AE", {"USUBJID": "STU-001-001", "AETERM": "Nausea", "AESEV": "MODERATE"})
    findings = gen.run_p21_checks()
    print(f"P21 findings: {len(findings)}")
    print(f"Validation summary: {gen.validation_summary()}")
    print(gen.generate_define_xml()[:200])


if __name__ == "__main__":
    main()
