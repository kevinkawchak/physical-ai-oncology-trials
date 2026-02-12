"""Unit tests for privacy/dua-templates/dua_generator.py.

Tests cover DUATemplate, Jurisdiction, SecurityLevel enums,
DUAParty, DataDescription, DUADocument dataclasses.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "privacy/dua-templates/dua_generator.py"


@pytest.fixture()
def mod():
    return load_module("dua_generator", MOD_PATH)


class TestEnums:
    def test_dua_template_types(self, mod):
        names = [e.name for e in mod.DUATemplate]
        for t in ("LIMITED_DATA_SET", "DEIDENTIFIED_DATA", "MULTI_SITE_AI_RESEARCH"):
            assert t in names

    def test_jurisdiction(self, mod):
        names = [e.name for e in mod.Jurisdiction]
        for j in ("US_HIPAA", "EU_GDPR", "CROSS_BORDER"):
            assert j in names

    def test_security_level(self, mod):
        names = [e.name for e in mod.SecurityLevel]
        for s in ("STANDARD", "ENHANCED", "MAXIMUM"):
            assert s in names


class TestDUAParty:
    def test_creation(self, mod):
        party = mod.DUAParty(
            organization_name="Memorial Cancer Center",
            organization_type="healthcare_provider",
            contact_name="Dr. Smith",
            contact_title="Chief Data Officer",
            contact_email="smith@memorial.org",
        )
        assert party.organization_name == "Memorial Cancer Center"


class TestDataDescription:
    def test_creation(self, mod):
        desc = mod.DataDescription(
            description="De-identified imaging dataset",
            data_types=["CT", "MRI"],
            record_count_estimate=5000,
            date_range="2020-2025",
            identifiability_level="de-identified",
        )
        assert desc.record_count_estimate == 5000


class TestDUADocument:
    def test_creation(self, mod):
        provider = mod.DUAParty(
            organization_name="Provider Hospital",
            organization_type="healthcare_provider",
            contact_name="Admin",
            contact_title="DPO",
            contact_email="admin@provider.org",
        )
        recipient = mod.DUAParty(
            organization_name="Research University",
            organization_type="academic",
            contact_name="PI",
            contact_title="Professor",
            contact_email="pi@university.edu",
        )
        data_desc = mod.DataDescription(
            description="Anonymized oncology imaging",
            data_types=["CT"],
            record_count_estimate=1000,
            date_range="2023-2025",
            identifiability_level="de-identified",
        )
        doc = mod.DUADocument(
            agreement_id="DUA-001",
            template=mod.DUATemplate.DEIDENTIFIED_DATA.value,
            jurisdiction=mod.Jurisdiction.US_HIPAA.value,
            provider=provider,
            recipient=recipient,
            data_description=data_desc,
            permitted_uses=["AI model training", "Publication"],
            prohibited_uses=["Re-identification", "Commercial sale"],
            security_requirements=[],
            retention_period_years=5,
            effective_date="2026-01-01",
            expiration_date="2031-01-01",
        )
        assert doc.agreement_id == "DUA-001"
        assert doc.retention_period_years == 5
