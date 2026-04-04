"""Implementation planner: three-phase deployment planning, site readiness scoring, resource allocation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class DeploymentPhase(Enum):
    PHASE_1_FOUNDATION = "foundation"
    PHASE_2_INTEGRATION = "integration"
    PHASE_3_AUTONOMY = "full_autonomy"


class ReadinessLevel(Enum):
    NOT_ASSESSED = "not_assessed"
    RED = "red"
    AMBER = "amber"
    GREEN = "green"


@dataclass
class PhaseObjective:
    objective_id: str = field(default_factory=lambda: f"OBJ-{uuid.uuid4().hex[:6]}")
    phase: DeploymentPhase = DeploymentPhase.PHASE_1_FOUNDATION
    description: str = ""
    duration_weeks: int = 0
    completed: bool = False


@dataclass
class SiteReadiness:
    site_id: str = ""
    infrastructure_score: float = 0.0
    training_score: float = 0.0
    connectivity_score: float = 0.0
    regulatory_score: float = 0.0
    overall: ReadinessLevel = ReadinessLevel.NOT_ASSESSED

    def compute_overall(self) -> ReadinessLevel:
        avg = (self.infrastructure_score + self.training_score + self.connectivity_score + self.regulatory_score) / 4
        if avg >= 0.8:
            self.overall = ReadinessLevel.GREEN
        elif avg >= 0.6:
            self.overall = ReadinessLevel.AMBER
        else:
            self.overall = ReadinessLevel.RED
        return self.overall


@dataclass
class ResourceAllocation:
    resource_type: str = ""
    phase: DeploymentPhase = DeploymentPhase.PHASE_1_FOUNDATION
    fte_count: float = 0.0
    budget_usd: float = 0.0


class ImplementationPlanner:
    """Three-phase deployment planner with site readiness scoring and resource allocation."""

    def __init__(self, study_id: str) -> None:
        self.study_id = study_id
        self.objectives: list[PhaseObjective] = []
        self.sites: dict[str, SiteReadiness] = {}
        self.resources: list[ResourceAllocation] = []
        self._init_default_plan()

    def _init_default_plan(self) -> None:
        plan = [
            (DeploymentPhase.PHASE_1_FOUNDATION, "Deploy core agent infrastructure", 12),
            (DeploymentPhase.PHASE_1_FOUNDATION, "Establish trust layer and PKI", 8),
            (DeploymentPhase.PHASE_1_FOUNDATION, "Validate safety gates", 6),
            (DeploymentPhase.PHASE_2_INTEGRATION, "Integrate site systems (FHIR/DICOM/HL7)", 16),
            (DeploymentPhase.PHASE_2_INTEGRATION, "Deploy patient-facing interfaces", 10),
            (DeploymentPhase.PHASE_2_INTEGRATION, "CRO oversight automation", 8),
            (DeploymentPhase.PHASE_3_AUTONOMY, "Enable autonomous decision loops", 12),
            (DeploymentPhase.PHASE_3_AUTONOMY, "Full regulatory submission automation", 10),
        ]
        for phase, desc, weeks in plan:
            self.objectives.append(PhaseObjective(phase=phase, description=desc, duration_weeks=weeks))

    def assess_site(
        self, site_id: str, infra: float, training: float, connectivity: float, regulatory: float
    ) -> SiteReadiness:
        site = SiteReadiness(
            site_id=site_id,
            infrastructure_score=infra,
            training_score=training,
            connectivity_score=connectivity,
            regulatory_score=regulatory,
        )
        site.compute_overall()
        self.sites[site_id] = site
        return site

    def allocate_resource(
        self, resource_type: str, phase: DeploymentPhase, fte_count: float, budget_usd: float
    ) -> ResourceAllocation:
        alloc = ResourceAllocation(resource_type=resource_type, phase=phase, fte_count=fte_count, budget_usd=budget_usd)
        self.resources.append(alloc)
        return alloc

    def phase_progress(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for phase in DeploymentPhase:
            objs = [o for o in self.objectives if o.phase == phase]
            result[phase.value] = {"total": len(objs), "completed": sum(1 for o in objs if o.completed)}
        return result

    def total_budget(self) -> dict[str, float]:
        by_phase = {p.value: sum(r.budget_usd for r in self.resources if r.phase == p) for p in DeploymentPhase}
        by_phase["total"] = sum(by_phase.values())
        return by_phase

    def readiness_summary(self) -> dict[str, int]:
        return {level.value: sum(1 for s in self.sites.values() if s.overall == level) for level in ReadinessLevel}

    def summary(self) -> dict[str, object]:
        return {
            "study_id": self.study_id,
            "objectives": len(self.objectives),
            "sites_assessed": len(self.sites),
            "readiness": self.readiness_summary(),
            "progress": self.phase_progress(),
            "budget": self.total_budget(),
        }


def main() -> None:
    planner = ImplementationPlanner("STU-001")
    planner.assess_site("SITE-01", 0.9, 0.85, 0.95, 0.88)
    planner.assess_site("SITE-02", 0.6, 0.55, 0.70, 0.65)
    planner.assess_site("SITE-03", 0.4, 0.35, 0.50, 0.45)
    planner.allocate_resource("AI Engineers", DeploymentPhase.PHASE_1_FOUNDATION, 5.0, 1_200_000)
    planner.allocate_resource("Clinical Ops", DeploymentPhase.PHASE_2_INTEGRATION, 3.0, 800_000)
    planner.allocate_resource("Regulatory", DeploymentPhase.PHASE_3_AUTONOMY, 2.0, 500_000)
    print(f"Readiness: {planner.readiness_summary()}")
    print(f"Progress: {planner.phase_progress()}")
    print(f"Budget: {planner.total_budget()}")
    print(f"Summary: {planner.summary()}")


if __name__ == "__main__":
    main()
