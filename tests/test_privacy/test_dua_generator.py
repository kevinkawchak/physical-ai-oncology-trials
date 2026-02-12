"""Tests for Data Use Agreement Generator.

Validates DUA document generation, template/jurisdiction selection,
security-level determination, prohibited-use logic, and export formats.

LICENSE: MIT
"""

from __future__ import annotations

from datetime import datetime

import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
MOD_PATH = "privacy/dua-templates/dua_generator.py"


@pytest.fixture()
def dua_mod():
    return load_module("dua_generator", MOD_PATH)


@pytest.fixture()
def generator(dua_mod):
    return dua_mod.DUAGenerator(template="multi_site_ai_research", jurisdiction="us_hipaa")


@pytest.fixture()
def sample_dua(generator):
    return generator.generate(
        data_provider="Provider Hospital",
        data_recipient="Research University",
        data_description="De-identified CT scans",
        permitted_uses=["AI model training", "Publication"],
        retention_period_years=5,
    )


# ===========================================================================
# TestEnums
# ===========================================================================


class TestEnums:
    def test_dua_template_values(self, dua_mod):
        names = {t.name for t in dua_mod.DUATemplate}
        assert "LIMITED_DATA_SET" in names
        assert "FEDERATED_LEARNING" in names
        assert len(names) == 6

    def test_jurisdiction_values(self, dua_mod):
        names = {j.name for j in dua_mod.Jurisdiction}
        assert "US_HIPAA" in names
        assert "EU_GDPR" in names
        assert "CROSS_BORDER" in names

    def test_security_level_values(self, dua_mod):
        names = {s.name for s in dua_mod.SecurityLevel}
        assert names == {"STANDARD", "ENHANCED", "MAXIMUM"}


# ===========================================================================
# TestDUAParty
# ===========================================================================


class TestDUAParty:
    def test_party_defaults(self, dua_mod):
        p = dua_mod.DUAParty(organization_name="Org", organization_type="academic")
        assert p.contact_name == ""
        assert p.fwa_number == ""

    def test_party_full_construction(self, dua_mod):
        p = dua_mod.DUAParty(
            organization_name="Hospital",
            organization_type="healthcare_system",
            contact_name="Dr. Smith",
            contact_email="smith@hosp.org",
            fwa_number="FWA00001234",
        )
        assert p.organization_name == "Hospital"
        assert p.fwa_number == "FWA00001234"


# ===========================================================================
# TestDataDescription
# ===========================================================================


class TestDataDescription:
    def test_defaults(self, dua_mod):
        dd = dua_mod.DataDescription(description="test")
        assert dd.identifiability_level == "de-identified"
        assert dd.contains_genomic is False
        assert dd.contains_imaging is False

    def test_custom_fields(self, dua_mod):
        dd = dua_mod.DataDescription(
            description="Genomic data",
            data_types=["WGS", "RNA-seq"],
            contains_genomic=True,
            record_count_estimate=500,
        )
        assert dd.contains_genomic is True
        assert len(dd.data_types) == 2


# ===========================================================================
# TestDUAGeneration
# ===========================================================================


class TestDUAGeneration:
    def test_agreement_id_prefix(self, sample_dua):
        assert sample_dua.agreement_id.startswith("DUA-")

    def test_template_matches(self, sample_dua):
        assert sample_dua.template == "multi_site_ai_research"

    def test_jurisdiction_matches(self, sample_dua):
        assert sample_dua.jurisdiction == "us_hipaa"

    def test_provider_name(self, sample_dua):
        assert sample_dua.provider.organization_name == "Provider Hospital"

    def test_recipient_name(self, sample_dua):
        assert sample_dua.recipient.organization_name == "Research University"

    def test_permitted_uses_passed_through(self, sample_dua):
        assert "AI model training" in sample_dua.permitted_uses
        assert len(sample_dua.permitted_uses) == 2

    def test_prohibited_uses_not_empty(self, sample_dua):
        assert len(sample_dua.prohibited_uses) > 0

    def test_ai_research_prohibited_uses(self, sample_dua):
        """Multi-site AI research template adds AI-specific prohibitions."""
        texts = " ".join(sample_dua.prohibited_uses).lower()
        assert "ai model" in texts or "training" in texts

    def test_sections_generated(self, sample_dua):
        assert len(sample_dua.sections) >= 8
        assert any("Parties" in k for k in sample_dua.sections)
        assert any("Security" in k for k in sample_dua.sections)

    def test_effective_and_expiration_dates(self, sample_dua):
        eff = datetime.fromisoformat(sample_dua.effective_date)
        exp = datetime.fromisoformat(sample_dua.expiration_date)
        assert exp > eff

    def test_retention_period_in_expiration(self, sample_dua):
        eff = datetime.fromisoformat(sample_dua.effective_date)
        exp = datetime.fromisoformat(sample_dua.expiration_date)
        diff_days = (exp - eff).days
        # 5 years ~ 1825 days
        assert 1820 <= diff_days <= 1830

    def test_to_dict(self, sample_dua):
        d = sample_dua._to_dict()
        assert "agreement_id" in d
        assert "provider" in d
        assert "sections" in d


# ===========================================================================
# TestSecurityLevelDetermination
# ===========================================================================


class TestSecurityLevel:
    def test_phi_gets_maximum(self, generator, dua_mod):
        dd = dua_mod.DataDescription(description="PHI data", identifiability_level="phi")
        level = generator._determine_security_level(dd, [])
        assert level == "maximum"

    def test_limited_data_set_gets_enhanced(self, generator, dua_mod):
        dd = dua_mod.DataDescription(description="LDS", identifiability_level="limited_data_set")
        level = generator._determine_security_level(dd, [])
        assert level == "enhanced"

    def test_genomic_gets_enhanced(self, generator, dua_mod):
        dd = dua_mod.DataDescription(description="Genomic", contains_genomic=True)
        level = generator._determine_security_level(dd, [])
        assert level == "enhanced"

    def test_deidentified_gets_standard(self, generator, dua_mod):
        dd = dua_mod.DataDescription(description="De-id")
        level = generator._determine_security_level(dd, [])
        assert level == "standard"

    def test_encryption_request_upgrades_to_enhanced(self, generator, dua_mod):
        dd = dua_mod.DataDescription(description="De-id")
        level = generator._determine_security_level(dd, ["encryption_at_rest"])
        assert level == "enhanced"


# ===========================================================================
# TestSecurityRequirements
# ===========================================================================


class TestSecurityRequirements:
    def test_standard_has_fewer_requirements(self, dua_mod):
        std = dua_mod.SECURITY_REQUIREMENTS["standard"]
        enh = dua_mod.SECURITY_REQUIREMENTS["enhanced"]
        assert len(std) < len(enh)

    def test_maximum_has_most_requirements(self, dua_mod):
        mx = dua_mod.SECURITY_REQUIREMENTS["maximum"]
        enh = dua_mod.SECURITY_REQUIREMENTS["enhanced"]
        assert len(mx) > len(enh)


# ===========================================================================
# TestCrossBorderJurisdiction
# ===========================================================================


class TestCrossBorder:
    def test_cross_border_includes_gdpr_section(self, dua_mod):
        gen = dua_mod.DUAGenerator(template="multi_site_ai_research", jurisdiction="cross_border")
        dua = gen.generate(
            data_provider="EU Hospital",
            data_recipient="US University",
            data_description="Cross-border imaging",
            permitted_uses=["Research"],
        )
        compliance_section = dua.sections.get("8. Compliance", "")
        assert "GDPR" in compliance_section


# ===========================================================================
# TestDUAExport
# ===========================================================================


class TestDUAExport:
    def test_export_json(self, sample_dua, tmp_path):
        out = str(tmp_path / "dua.json")
        sample_dua.export(out)
        import json
        from pathlib import Path

        data = json.loads(Path(out).read_text())
        assert data["agreement_id"] == sample_dua.agreement_id

    def test_export_markdown(self, sample_dua, tmp_path):
        out = str(tmp_path / "dua.md")
        sample_dua.export(out)
        from pathlib import Path

        text = Path(out).read_text()
        assert "Data Use Agreement" in text

    def test_export_txt(self, sample_dua, tmp_path):
        out = str(tmp_path / "dua.txt")
        sample_dua.export(out)
        from pathlib import Path

        text = Path(out).read_text()
        assert "DATA USE AGREEMENT" in text
