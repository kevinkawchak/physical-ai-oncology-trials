"""Biomarker strategy engine: biomarker qualification scoring, CDx timeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class BiomarkerType(Enum):
    PREDICTIVE = "predictive"
    PROGNOSTIC = "prognostic"
    PHARMACODYNAMIC = "pharmacodynamic"
    SAFETY = "safety"
    SURROGATE = "surrogate"


class QualificationLevel(Enum):
    EXPLORATORY = "exploratory"
    PROBABLE_VALID = "probable_valid"
    KNOWN_VALID = "known_valid"
    FDA_QUALIFIED = "fda_qualified"


class AssayPlatform(Enum):
    IHC = "immunohistochemistry"
    NGS = "next_gen_sequencing"
    PCR = "polymerase_chain_reaction"
    FISH = "fluorescence_in_situ_hybridization"
    LIQUID_BIOPSY = "liquid_biopsy"
    FLOW_CYTOMETRY = "flow_cytometry"


@dataclass
class BiomarkerCandidate:
    marker_id: str
    name: str
    marker_type: BiomarkerType
    gene_target: str = ""
    assay_platform: AssayPlatform = AssayPlatform.IHC
    qualification: QualificationLevel = QualificationLevel.EXPLORATORY
    analytical_validation: float = 0.0
    clinical_validation: float = 0.0
    clinical_utility: float = 0.0
    prevalence: float = 0.0

    @property
    def qualification_score(self) -> float:
        return (
            self.analytical_validation * 0.30
            + self.clinical_validation * 0.35
            + self.clinical_utility * 0.25
            + self.prevalence * 0.10
        )


@dataclass
class CDxMilestone:
    name: str
    target_date: date
    completed: bool = False
    notes: str = ""


@dataclass
class CDxProgram:
    cdx_id: str
    biomarker: BiomarkerCandidate
    partner: str
    milestones: list[CDxMilestone] = field(default_factory=list)

    def build_default_timeline(self, start: date) -> None:
        steps = [
            ("Assay Development", 90),
            ("Analytical Validation", 180),
            ("Clinical Validation Study", 270),
            ("Bridging Study", 360),
            ("PMA/510k Submission", 450),
            ("FDA Review", 540),
            ("CDx Approval", 630),
        ]
        self.milestones = [CDxMilestone(name=name, target_date=start + timedelta(days=d)) for name, d in steps]

    @property
    def completion_pct(self) -> float:
        if not self.milestones:
            return 0.0
        return sum(1 for m in self.milestones if m.completed) / len(self.milestones)


class BiomarkerStrategyEngine:
    """Manages biomarker qualification and companion diagnostic programs."""

    def __init__(self) -> None:
        self.candidates: dict[str, BiomarkerCandidate] = {}
        self.cdx_programs: dict[str, CDxProgram] = {}

    def register_biomarker(self, candidate: BiomarkerCandidate) -> str:
        self.candidates[candidate.marker_id] = candidate
        return candidate.marker_id

    def rank_candidates(self) -> list[tuple[str, float]]:
        ranked = [(c.name, c.qualification_score) for c in self.candidates.values()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def advance_qualification(self, marker_id: str, new_level: QualificationLevel) -> None:
        self.candidates[marker_id].qualification = new_level

    def create_cdx_program(self, cdx_id: str, marker_id: str, partner: str) -> CDxProgram:
        biomarker = self.candidates[marker_id]
        program = CDxProgram(cdx_id=cdx_id, biomarker=biomarker, partner=partner)
        program.build_default_timeline(date.today())
        self.cdx_programs[cdx_id] = program
        return program

    def cdx_gap_analysis(self, cdx_id: str) -> list[str]:
        program = self.cdx_programs[cdx_id]
        gaps: list[str] = []
        bm = program.biomarker
        if bm.analytical_validation < 0.8:
            gaps.append(f"Analytical validation insufficient: {bm.analytical_validation:.0%}")
        if bm.clinical_validation < 0.6:
            gaps.append(f"Clinical validation insufficient: {bm.clinical_validation:.0%}")
        if bm.prevalence < 0.05:
            gaps.append(f"Low prevalence: {bm.prevalence:.1%}")
        return gaps


def main() -> None:
    engine = BiomarkerStrategyEngine()
    bm = BiomarkerCandidate(
        "BM-001",
        "PD-L1 TPS",
        BiomarkerType.PREDICTIVE,
        "CD274",
        AssayPlatform.IHC,
        QualificationLevel.KNOWN_VALID,
        0.9,
        0.85,
        0.7,
        0.30,
    )
    engine.register_biomarker(bm)
    print("Rankings:", engine.rank_candidates())
    cdx = engine.create_cdx_program("CDX-001", "BM-001", "Dako/Agilent")
    print(f"CDx milestones: {len(cdx.milestones)}, gaps: {engine.cdx_gap_analysis('CDX-001')}")


if __name__ == "__main__":
    main()
