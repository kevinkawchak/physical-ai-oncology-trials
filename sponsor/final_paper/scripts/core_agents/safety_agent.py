"""Safety agent: AE/SAE intake, MedDRA coding, causality assessment."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


class AESeverity(Enum):
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    LIFE_THREATENING = 4
    DEATH = 5


class Causality(Enum):
    UNRELATED = "unrelated"
    UNLIKELY = "unlikely"
    POSSIBLE = "possible"
    PROBABLE = "probable"
    DEFINITE = "definite"


class Outcome(Enum):
    RECOVERED = "recovered"
    RECOVERING = "recovering"
    NOT_RECOVERED = "not_recovered"
    FATAL = "fatal"
    UNKNOWN = "unknown"


class Seriousness(Enum):
    NON_SERIOUS = "non_serious"
    HOSPITALIZATION = "hospitalization"
    LIFE_THREATENING = "life_threatening"
    DISABILITY = "disability"
    CONGENITAL_ANOMALY = "congenital_anomaly"
    DEATH = "death"
    MEDICALLY_IMPORTANT = "medically_important"


@dataclass
class MedDRACoding:
    pt_code: int
    pt_term: str
    soc_code: int
    soc_term: str
    llt_code: int = 0
    llt_term: str = ""


@dataclass
class AdverseEvent:
    ae_id: str = field(default_factory=lambda: f"AE-{uuid.uuid4().hex[:8]}")
    subject_id: str = ""
    verbatim_term: str = ""
    meddra: MedDRACoding | None = None
    severity: AESeverity = AESeverity.MILD
    seriousness: list[Seriousness] = field(default_factory=list)
    causality: Causality = Causality.UNRELATED
    onset_date: date = field(default_factory=date.today)
    resolution_date: date | None = None
    outcome: Outcome = Outcome.UNKNOWN
    action_taken: str = ""
    reported_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_sae(self) -> bool:
        return len(self.seriousness) > 0 and Seriousness.NON_SERIOUS not in self.seriousness

    @property
    def is_related(self) -> bool:
        return self.causality in (Causality.POSSIBLE, Causality.PROBABLE, Causality.DEFINITE)


MEDDRA_LOOKUP: dict[str, MedDRACoding] = {
    "nausea": MedDRACoding(10028813, "Nausea", 10017947, "Gastrointestinal disorders"),
    "fatigue": MedDRACoding(10016256, "Fatigue", 10018065, "General disorders"),
    "neutropenia": MedDRACoding(10029354, "Neutropenia", 10005329, "Blood disorders"),
    "rash": MedDRACoding(10037844, "Rash", 10040785, "Skin disorders"),
    "pyrexia": MedDRACoding(10037660, "Pyrexia", 10018065, "General disorders"),
    "diarrhea": MedDRACoding(10012735, "Diarrhoea", 10017947, "Gastrointestinal disorders"),
}


class SafetyAgent:
    """Processes adverse events: intake, MedDRA coding, and causality assessment."""

    def __init__(self, study_id: str) -> None:
        self.study_id = study_id
        self.events: list[AdverseEvent] = []

    def code_verbatim(self, verbatim: str) -> MedDRACoding | None:
        key = verbatim.lower().strip()
        return MEDDRA_LOOKUP.get(key)

    def intake_ae(self, subject_id: str, verbatim: str, severity: AESeverity,
                  seriousness: list[Seriousness] | None = None) -> AdverseEvent:
        ae = AdverseEvent(subject_id=subject_id, verbatim_term=verbatim, severity=severity,
                          seriousness=seriousness or [Seriousness.NON_SERIOUS])
        ae.meddra = self.code_verbatim(verbatim)
        self.events.append(ae)
        return ae

    def assess_causality(self, ae_id: str, temporal: bool, dechallenge: bool,
                         rechallenge: bool, known_class: bool) -> Causality:
        score = sum([temporal, dechallenge, rechallenge, known_class])
        mapping = {0: Causality.UNRELATED, 1: Causality.UNLIKELY, 2: Causality.POSSIBLE,
                   3: Causality.PROBABLE, 4: Causality.DEFINITE}
        result = mapping.get(score, Causality.POSSIBLE)
        for ev in self.events:
            if ev.ae_id == ae_id:
                ev.causality = result
                break
        return result

    def sae_list(self) -> list[AdverseEvent]:
        return [e for e in self.events if e.is_sae]

    def safety_summary(self) -> dict[str, object]:
        total = len(self.events)
        saes = len(self.sae_list())
        related = sum(1 for e in self.events if e.is_related)
        by_pt: dict[str, int] = {}
        for e in self.events:
            pt = e.meddra.pt_term if e.meddra else e.verbatim_term
            by_pt[pt] = by_pt.get(pt, 0) + 1
        return {"total_aes": total, "saes": saes, "related": related, "by_preferred_term": by_pt}


def main() -> None:
    agent = SafetyAgent("STU-001")
    ae1 = agent.intake_ae("SUBJ-01", "Nausea", AESeverity.MODERATE)
    ae2 = agent.intake_ae("SUBJ-02", "Neutropenia", AESeverity.SEVERE, [Seriousness.HOSPITALIZATION])
    agent.assess_causality(ae1.ae_id, temporal=True, dechallenge=True, rechallenge=False, known_class=True)
    agent.assess_causality(ae2.ae_id, temporal=True, dechallenge=False, rechallenge=False, known_class=True)
    print(f"SAEs: {len(agent.sae_list())}")
    print(f"Summary: {agent.safety_summary()}")


if __name__ == "__main__":
    main()
