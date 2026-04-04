"""Protocol generator: ICH M11 structured protocol generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date


ICH_M11_SECTIONS = [
    "title_page",
    "synopsis",
    "schedule_of_activities",
    "introduction",
    "study_rationale",
    "objectives_endpoints",
    "study_design",
    "study_population",
    "study_intervention",
    "discontinuation",
    "study_assessments",
    "adverse_events",
    "statistical_considerations",
    "oversight",
    "references",
    "appendices",
]


@dataclass
class Endpoint:
    name: str
    category: str  # primary, secondary, exploratory
    measure: str
    timepoint: str


@dataclass
class StudyArm:
    arm_id: str
    name: str
    intervention: str
    dose: str
    schedule: str
    n_planned: int = 0


@dataclass
class ProtocolSection:
    section_id: str
    title: str
    content: str = ""
    version: int = 1
    locked: bool = False


@dataclass
class Protocol:
    protocol_id: str
    sponsor: str
    title: str
    phase: str
    indication: str
    generated_date: date = field(default_factory=date.today)
    sections: dict[str, ProtocolSection] = field(default_factory=dict)
    arms: list[StudyArm] = field(default_factory=list)
    endpoints: list[Endpoint] = field(default_factory=list)
    version: str = "1.0"


class ProtocolGenerator:
    """Generates ICH M11-compliant structured clinical trial protocols."""

    def __init__(self, sponsor: str) -> None:
        self.sponsor = sponsor
        self.protocols: dict[str, Protocol] = {}

    def create_protocol(self, protocol_id: str, title: str, phase: str, indication: str) -> Protocol:
        proto = Protocol(protocol_id=protocol_id, sponsor=self.sponsor, title=title, phase=phase, indication=indication)
        for section_id in ICH_M11_SECTIONS:
            readable = section_id.replace("_", " ").title()
            proto.sections[section_id] = ProtocolSection(section_id=section_id, title=readable)
        self.protocols[protocol_id] = proto
        return proto

    def add_arm(self, protocol_id: str, arm: StudyArm) -> None:
        self.protocols[protocol_id].arms.append(arm)

    def add_endpoint(self, protocol_id: str, endpoint: Endpoint) -> None:
        self.protocols[protocol_id].endpoints.append(endpoint)

    def populate_synopsis(self, protocol_id: str) -> str:
        proto = self.protocols[protocol_id]
        arm_desc = "; ".join(f"{a.name}: {a.intervention} {a.dose} {a.schedule}" for a in proto.arms)
        ep_desc = "; ".join(f"{e.category}: {e.name}" for e in proto.endpoints)
        synopsis = (
            f"Protocol: {proto.protocol_id}\n"
            f"Title: {proto.title}\n"
            f"Phase: {proto.phase}\n"
            f"Indication: {proto.indication}\n"
            f"Arms: {arm_desc}\n"
            f"Endpoints: {ep_desc}"
        )
        proto.sections["synopsis"].content = synopsis
        return synopsis

    def export_json(self, protocol_id: str) -> str:
        proto = self.protocols[protocol_id]
        data = {
            "protocolId": proto.protocol_id,
            "sponsor": proto.sponsor,
            "title": proto.title,
            "phase": proto.phase,
            "version": proto.version,
            "sections": {k: {"title": v.title, "version": v.version} for k, v in proto.sections.items()},
            "arms": [{"id": a.arm_id, "name": a.name, "n": a.n_planned} for a in proto.arms],
        }
        return json.dumps(data, indent=2)

    def completeness_check(self, protocol_id: str) -> dict[str, bool]:
        proto = self.protocols[protocol_id]
        return {sid: bool(sec.content) for sid, sec in proto.sections.items()}


def main() -> None:
    gen = ProtocolGenerator("PhysicalAI Oncology Inc.")
    proto = gen.create_protocol("PAI-2026-001", "Robotic-Assisted Biopsy in NSCLC", "Phase II", "NSCLC")
    gen.add_arm(proto.protocol_id, StudyArm("A1", "Treatment", "PAI-101", "200mg", "Q3W", 120))
    gen.add_endpoint(proto.protocol_id, Endpoint("ORR", "primary", "RECIST v1.1", "Week 24"))
    synopsis = gen.populate_synopsis(proto.protocol_id)
    print(synopsis)
    completeness = gen.completeness_check(proto.protocol_id)
    filled = sum(v for v in completeness.values())
    print(f"Sections filled: {filled}/{len(completeness)}")


if __name__ == "__main__":
    main()
