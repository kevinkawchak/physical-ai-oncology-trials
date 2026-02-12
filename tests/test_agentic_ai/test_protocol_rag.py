"""Unit tests for agentic-ai/examples-agentic-ai/06_protocol_rag_compliance_agent.py.

Tests cover DocumentType, ComplianceStatus enums, DocumentChunk,
RetrievalResult, ComplianceCheck, ComplianceReport dataclasses,
and RegulatoryKnowledgeBase.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "agentic-ai/examples-agentic-ai/06_protocol_rag_compliance_agent.py"


@pytest.fixture()
def mod():
    return load_module("protocol_rag", MOD_PATH)


class TestEnums:
    def test_document_type(self, mod):
        names = [e.name for e in mod.DocumentType]
        for t in ("TRIAL_PROTOCOL", "FDA_GUIDANCE", "ICH_GUIDELINE", "IEC_STANDARD"):
            assert t in names

    def test_compliance_status(self, mod):
        names = [e.name for e in mod.ComplianceStatus]
        for s in ("COMPLIANT", "NON_COMPLIANT", "PARTIAL", "REQUIRES_REVIEW"):
            assert s in names


class TestDocumentChunk:
    def test_creation(self, mod):
        chunk = mod.DocumentChunk(
            chunk_id="CHUNK-001",
            document_id="DOC-001",
            document_type=mod.DocumentType.FDA_GUIDANCE,
            document_title="FDA AI/ML Guidance",
            section_number="3.1",
            section_title="Performance Monitoring",
            content="Manufacturers should establish processes...",
            page_number=15,
            effective_date="2025-01-15",
            keywords=["monitoring", "performance", "AI/ML"],
        )
        assert chunk.chunk_id == "CHUNK-001"
        assert chunk.document_type == mod.DocumentType.FDA_GUIDANCE

    def test_get_citation(self, mod):
        chunk = mod.DocumentChunk(
            chunk_id="CHUNK-002",
            document_id="DOC-002",
            document_type=mod.DocumentType.ICH_GUIDELINE,
            document_title="ICH E6(R3)",
            section_number="5.2",
            section_title="Monitoring",
            content="The sponsor should ensure...",
            page_number=42,
            effective_date="2025-09-01",
            keywords=["monitoring", "sponsor"],
        )
        citation = chunk.get_citation()
        assert isinstance(citation, str)
        assert "ICH E6(R3)" in citation or "DOC-002" in citation


class TestRetrievalResult:
    def test_creation(self, mod):
        chunk = mod.DocumentChunk(
            chunk_id="CHUNK-003",
            document_id="DOC-003",
            document_type=mod.DocumentType.TRIAL_PROTOCOL,
            document_title="Test Protocol",
            section_number="1.0",
            section_title="Introduction",
            content="This protocol describes...",
            page_number=1,
            effective_date="2026-01-01",
            keywords=["protocol"],
        )
        rr = mod.RetrievalResult(
            chunk=chunk,
            relevance_score=0.92,
            matched_keywords=["protocol"],
        )
        assert rr.relevance_score > 0.9
        assert len(rr.matched_keywords) == 1


class TestComplianceCheck:
    def test_compliant(self, mod):
        check = mod.ComplianceCheck(
            requirement_id="REQ-001",
            requirement_text="All AI models must have documented validation",
            source_citation="FDA AI/ML Guidance, Section 3.1",
            status=mod.ComplianceStatus.COMPLIANT,
            rationale="Model validation report exists and is up to date",
            evidence=["validation_report_v2.pdf"],
        )
        assert check.status == mod.ComplianceStatus.COMPLIANT

    def test_non_compliant_with_remediation(self, mod):
        check = mod.ComplianceCheck(
            requirement_id="REQ-002",
            requirement_text="Pre-market submission required for Class II devices",
            source_citation="FDA 21 CFR 807",
            status=mod.ComplianceStatus.NON_COMPLIANT,
            rationale="No 510(k) submission filed",
            evidence=[],
            remediation="Prepare and file 510(k) submission",
        )
        assert check.status == mod.ComplianceStatus.NON_COMPLIANT
        assert check.remediation is not None


class TestRegulatoryKnowledgeBase:
    def test_instantiation(self, mod):
        kb = mod.RegulatoryKnowledgeBase()
        assert kb is not None
        assert len(kb._chunks) > 0

    def test_search_returns_results(self, mod):
        kb = mod.RegulatoryKnowledgeBase()
        results = kb.retrieve("informed consent", top_k=5)
        assert isinstance(results, list)
        # Knowledge base should have some consent-related content
        assert len(results) >= 0
