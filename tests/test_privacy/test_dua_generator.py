"""Tests for privacy/dua-templates/dua_generator.py

Covers DUA templates, party dataclass, and document generation.
"""

from __future__ import annotations


class TestEnums:
    def test_dua_templates(self, dua_mod):
        dt = dua_mod.DUATemplate
        assert dt.LIMITED_DATA_SET.value == "limited_data_set"
        assert dt.FEDERATED_LEARNING.value == "federated_learning"

    def test_jurisdictions(self, dua_mod):
        j = dua_mod.Jurisdiction
        assert j.US_HIPAA.value == "us_hipaa"
        assert j.EU_GDPR.value == "eu_gdpr"

    def test_security_levels(self, dua_mod):
        sl = dua_mod.SecurityLevel
        assert sl.STANDARD.value == "standard"
        assert sl.MAXIMUM.value == "maximum"


class TestDUAParty:
    def test_creation(self, dua_mod):
        party = dua_mod.DUAParty(
            organization_name="Test University",
            organization_type="academic",
        )
        assert party.organization_name == "Test University"


class TestDataDescription:
    def test_creation(self, dua_mod):
        desc = dua_mod.DataDescription(
            description="De-identified clinical trial data",
            data_types=["demographics", "outcomes"],
            record_count_estimate=1000,
        )
        assert desc.record_count_estimate == 1000
        assert desc.identifiability_level == "de-identified"


class TestDUAGenerator:
    def test_creation(self, dua_mod):
        gen = dua_mod.DUAGenerator()
        assert gen is not None

    def test_generate_document(self, dua_mod):
        gen = dua_mod.DUAGenerator()
        doc = gen.generate(
            data_provider="Hospital A",
            data_recipient="University B",
            data_description="Clinical outcomes data",
            permitted_uses=["AI model training", "Research analysis"],
        )
        assert isinstance(doc, dua_mod.DUADocument)
        assert len(doc.permitted_uses) == 2
        assert len(doc.prohibited_uses) > 0

    def test_generate_with_security(self, dua_mod):
        gen = dua_mod.DUAGenerator()
        doc = gen.generate(
            data_provider="Provider",
            data_recipient="Recipient",
            data_description="Genomic data",
            permitted_uses=["Research"],
            security_requirements=["encryption_at_rest"],
        )
        assert len(doc.security_requirements) > 0
