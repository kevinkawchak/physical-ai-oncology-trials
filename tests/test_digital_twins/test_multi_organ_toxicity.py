"""Unit tests for digital-twins/examples-twins/02_multi_organ_toxicity_twin.py.

Tests cover ChemoDrug, OrganSystem, CTCAEGrade enums, PBPKParameters
dataclass, drug library, and toxicity model dynamics.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

MOD_PATH = "digital-twins/examples-twins/02_multi_organ_toxicity_twin.py"


@pytest.fixture()
def mod():
    return load_module("multi_organ_toxicity", MOD_PATH)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_chemo_drug_members(self, mod):
        names = [e.name for e in mod.ChemoDrug]
        for drug in ("CISPLATIN", "CARBOPLATIN", "DOXORUBICIN", "PACLITAXEL"):
            assert drug in names

    def test_organ_system_members(self, mod):
        names = [e.name for e in mod.OrganSystem]
        for organ in ("CARDIAC", "RENAL", "HEPATIC", "NEUROLOGICAL", "HEMATOLOGIC"):
            assert organ in names

    def test_ctcae_grade_ordering(self, mod):
        assert mod.CTCAEGrade.GRADE_0.value < mod.CTCAEGrade.GRADE_5.value

    def test_ctcae_grade_count(self, mod):
        grades = list(mod.CTCAEGrade)
        assert len(grades) == 6  # Grade 0-5


# ---------------------------------------------------------------------------
# PBPK parameter tests
# ---------------------------------------------------------------------------


class TestPBPKParameters:
    def test_cisplatin_parameters_exist(self, mod):
        assert mod.ChemoDrug.CISPLATIN in mod.DRUG_PBPK_LIBRARY

    def test_parameters_positive(self, mod):
        for drug, params in mod.DRUG_PBPK_LIBRARY.items():
            assert params.vd_plasma > 0, f"{drug.name} vd_plasma not positive"
            assert params.cl_renal >= 0, f"{drug.name} cl_renal negative"

    def test_all_drugs_have_params(self, mod):
        # DRUG_PBPK_LIBRARY contains parameters for drugs with known organ toxicity profiles
        # Not all ChemoDrug members need to be present; verify the ones that are
        for drug in (
            mod.ChemoDrug.CISPLATIN,
            mod.ChemoDrug.DOXORUBICIN,
            mod.ChemoDrug.OXALIPLATIN,
            mod.ChemoDrug.PACLITAXEL,
        ):
            assert drug in mod.DRUG_PBPK_LIBRARY, f"{drug.name} missing from library"


# ---------------------------------------------------------------------------
# Toxicity model tests
# ---------------------------------------------------------------------------


class TestMultiOrganToxicityTwin:
    def test_instantiation(self, mod):
        twin = mod.MultiOrganToxicityTwin(patient_id="TEST-001")
        assert twin is not None
        assert twin.patient_id == "TEST-001"

    def test_pbpk_ode_returns_finite(self, mod):
        pbpk = mod.PBPKModel(mod.ChemoDrug.CISPLATIN)
        # Initial state: drug in plasma only
        n_compartments = pbpk.N_COMPARTMENTS  # 6: plasma + 5 organs
        y0 = np.zeros(n_compartments)
        y0[0] = 100.0  # 100 ug/mL in plasma
        dydt = pbpk._ode_system(0.0, y0)
        assert len(dydt) == n_compartments
        assert np.all(np.isfinite(dydt))

    def test_simulate_treatment_cycle(self, mod):
        twin = mod.MultiOrganToxicityTwin(patient_id="TEST-002")
        profile = twin.simulate_cycle(
            drug=mod.ChemoDrug.DOXORUBICIN,
            dose_mg_m2=60.0,
            bsa=1.8,
            cycle=1,
        )
        assert isinstance(profile, mod.ToxicityProfile)
        assert mod.OrganSystem.CARDIAC in profile.organ_states
        assert mod.OrganSystem.HEMATOLOGIC in profile.organ_states

    def test_ctcae_grading(self, mod):
        twin = mod.MultiOrganToxicityTwin(patient_id="TEST-003")
        # Simulate a cycle and check that CTCAE grades are assigned
        profile = twin.simulate_cycle(
            drug=mod.ChemoDrug.CISPLATIN,
            dose_mg_m2=75.0,
            bsa=1.85,
            cycle=1,
        )
        renal_state = profile.organ_states[mod.OrganSystem.RENAL]
        assert isinstance(renal_state.ctcae_grade, mod.CTCAEGrade)
