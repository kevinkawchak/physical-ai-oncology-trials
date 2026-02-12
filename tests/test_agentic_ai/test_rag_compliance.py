"""Tests for agentic-ai/examples-agentic-ai/06_protocol_rag_compliance_agent.py"""

from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module(
        "protocol_rag_compliance_agent",
        "agentic-ai/examples-agentic-ai/06_protocol_rag_compliance_agent.py",
    )


# ============================================================
# Enum tests
# ============================================================


class TestDocumentType:
    def test_trial_protocol_value(self, mod):
        assert mod.DocumentType.TRIAL_PROTOCOL.value == "trial_protocol"

    def test_fda_guidance_value(self, mod):
        assert mod.DocumentType.FDA_GUIDANCE.value == "fda_guidance"

    def test_ich_guideline_value(self, mod):
        assert mod.DocumentType.ICH_GUIDELINE.value == "ich_guideline"

    def test_iec_standard_value(self, mod):
        assert mod.DocumentType.IEC_STANDARD.value == "iec_standard"

    def test_member_count(self, mod):
        assert len(mod.DocumentType) == 8


class TestComplianceStatus:
    def test_compliant_value(self, mod):
        assert mod.ComplianceStatus.COMPLIANT.value == "compliant"

    def test_non_compliant_value(self, mod):
        assert mod.ComplianceStatus.NON_COMPLIANT.value == "non_compliant"

    def test_partial_value(self, mod):
        assert mod.ComplianceStatus.PARTIAL.value == "partially_compliant"

    def test_requires_review_value(self, mod):
        assert mod.ComplianceStatus.REQUIRES_REVIEW.value == "requires_review"

    def test_member_count(self, mod):
        assert len(mod.ComplianceStatus) == 5


# ============================================================
# Dataclass tests
# ============================================================


class TestDocumentChunk:
    def test_creation(self, mod):
        chunk = mod.DocumentChunk(
            chunk_id="DOC-001",
            document_id="PROT-001",
            document_type=mod.DocumentType.TRIAL_PROTOCOL,
            document_title="Protocol v1.0",
            section_number="3.1",
            section_title="Inclusion Criteria",
            content="Eligible patients must be >= 18 years old.",
        )
        assert chunk.chunk_id == "DOC-001"
        assert chunk.content == "Eligible patients must be >= 18 years old."

    def test_content_field(self, mod):
        chunk = mod.DocumentChunk(
            chunk_id="DOC-002",
            document_id="FDA-001",
            document_type=mod.DocumentType.FDA_GUIDANCE,
            document_title="FDA Guidance",
            section_number="5.1",
            section_title="AI Requirements",
            content="AI output must be advisory.",
        )
        assert "advisory" in chunk.content

    def test_get_citation(self, mod):
        chunk = mod.DocumentChunk(
            chunk_id="DOC-003",
            document_id="IEC-001",
            document_type=mod.DocumentType.IEC_STANDARD,
            document_title="IEC 80601-2-77",
            section_number="201.12",
            section_title="Force Limiting",
            content="Force limits apply.",
        )
        citation = chunk.get_citation()
        assert "IEC 80601-2-77" in citation
        assert "201.12" in citation
        assert "Force Limiting" in citation

    def test_default_keywords(self, mod):
        chunk = mod.DocumentChunk(
            chunk_id="DOC-004",
            document_id="SOP-001",
            document_type=mod.DocumentType.INSTITUTIONAL_SOP,
            document_title="SOP",
            section_number="1.0",
            section_title="Setup",
            content="Setup steps.",
        )
        assert chunk.keywords == []
        assert chunk.page_number == 0


class TestRetrievalResult:
    def test_creation(self, mod):
        chunk = mod.DocumentChunk(
            chunk_id="R-001",
            document_id="PROT-001",
            document_type=mod.DocumentType.TRIAL_PROTOCOL,
            document_title="Protocol",
            section_number="1.0",
            section_title="Overview",
            content="Overview content.",
        )
        rr = mod.RetrievalResult(chunk=chunk, relevance_score=0.85, matched_keywords=["overview"])
        assert rr.relevance_score == 0.85
        assert rr.chunk.chunk_id == "R-001"

    def test_chunks_list_in_results(self, mod):
        chunk = mod.DocumentChunk(
            chunk_id="R-002",
            document_id="PROT-001",
            document_type=mod.DocumentType.TRIAL_PROTOCOL,
            document_title="Protocol",
            section_number="2.0",
            section_title="Background",
            content="Background content.",
        )
        rr = mod.RetrievalResult(chunk=chunk, relevance_score=0.6)
        assert rr.matched_keywords == []


class TestComplianceCheck:
    def test_creation(self, mod):
        cc = mod.ComplianceCheck(
            requirement_id="REQ-001",
            requirement_text="Patient must be >= 18",
            source_citation="Protocol Section 3.1",
            status=mod.ComplianceStatus.COMPLIANT,
            rationale="Patient age verified",
        )
        assert cc.status == mod.ComplianceStatus.COMPLIANT
        assert cc.requirement_id == "REQ-001"

    def test_non_compliant_with_remediation(self, mod):
        cc = mod.ComplianceCheck(
            requirement_id="REQ-002",
            requirement_text="ECOG <= 1",
            source_citation="Protocol Section 3.1",
            status=mod.ComplianceStatus.NON_COMPLIANT,
            rationale="ECOG is 2",
            remediation="Patient does not meet eligibility",
        )
        assert cc.status == mod.ComplianceStatus.NON_COMPLIANT
        assert cc.remediation != ""


# ============================================================
# RegulatoryKnowledgeBase tests
# ============================================================


class TestRegulatoryKnowledgeBase:
    def test_init_loads_chunks(self, mod):
        kb = mod.RegulatoryKnowledgeBase()
        assert len(kb._chunks) > 0

    def test_document_count(self, mod):
        kb = mod.RegulatoryKnowledgeBase()
        doc_ids = set(c.document_id for c in kb._chunks)
        assert len(doc_ids) >= 5

    def test_search_returns_results(self, mod):
        kb = mod.RegulatoryKnowledgeBase()
        results = kb.retrieve("What are the inclusion criteria for the trial?", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, mod.RetrievalResult) for r in results)

    def test_search_relevance_score_bounded(self, mod):
        kb = mod.RegulatoryKnowledgeBase()
        results = kb.retrieve("force limit requirements", top_k=5)
        for r in results:
            assert 0.0 <= r.relevance_score <= 1.0

    def test_search_force_related(self, mod):
        kb = mod.RegulatoryKnowledgeBase()
        results = kb.retrieve("force limiting tissue monitoring", top_k=3)
        assert len(results) > 0
        # At least one result should be from IEC standard about force
        content_combined = " ".join(r.chunk.content.lower() for r in results)
        assert "force" in content_combined

    def test_search_empty_query(self, mod):
        kb = mod.RegulatoryKnowledgeBase()
        results = kb.retrieve("", top_k=3)
        # Should still return some results from content matching
        assert isinstance(results, list)

    def test_keyword_index_built(self, mod):
        kb = mod.RegulatoryKnowledgeBase()
        assert len(kb._keyword_index) > 0


# ============================================================
# ComplianceVerifier tests
# ============================================================


class TestComplianceVerifier:
    def test_init(self, mod):
        verifier = mod.ComplianceVerifier()
        assert verifier._check_history == []

    def test_verify_action_returns_checks(self, mod):
        verifier = mod.ComplianceVerifier()
        kb = mod.RegulatoryKnowledgeBase()
        retrieved = kb.retrieve("robotic dissection with AI guidance", top_k=3)
        checks = verifier.verify_action_compliance(
            "Begin robotic dissection with AI guidance",
            {"current_force_n": 3.0, "max_force_n": 10.0, "within_workspace": True},
            retrieved,
        )
        assert len(checks) > 0
        assert all(isinstance(c, mod.ComplianceCheck) for c in checks)

    def test_verify_force_violation(self, mod):
        verifier = mod.ComplianceVerifier()
        kb = mod.RegulatoryKnowledgeBase()
        retrieved = kb.retrieve("force limit safety stopping", top_k=5)
        checks = verifier.verify_action_compliance(
            "Continue tissue retraction during safety stopping assessment",
            {"current_force_n": 18.0, "max_force_n": 10.0},
            retrieved,
        )
        non_compliant = [c for c in checks if c.status == mod.ComplianceStatus.NON_COMPLIANT]
        assert len(non_compliant) >= 1

    def test_verify_eligibility_ecog_violation(self, mod):
        verifier = mod.ComplianceVerifier()
        kb = mod.RegulatoryKnowledgeBase()
        retrieved = kb.retrieve("enrollment eligibility inclusion criteria", top_k=5)
        checks = verifier.verify_action_compliance(
            "Enroll patient who meets eligibility criteria",
            {"ecog_status": 2},
            retrieved,
        )
        non_compliant = [c for c in checks if c.status == mod.ComplianceStatus.NON_COMPLIANT]
        assert len(non_compliant) >= 1

    def test_verify_protocol_compliance(self, mod):
        verifier = mod.ComplianceVerifier()
        kb = mod.RegulatoryKnowledgeBase()
        retrieved = kb.retrieve("robotic surgery setup verification", top_k=3)
        checks = verifier.verify_action_compliance(
            "Begin robotic procedure setup",
            {"robot_self_test_passed": True, "within_workspace": True},
            retrieved,
        )
        assert len(checks) > 0


# ============================================================
# ProtocolRAGComplianceAgent tests
# ============================================================


class TestProtocolRAGComplianceAgent:
    def test_init(self, mod):
        agent = mod.ProtocolRAGComplianceAgent()
        assert agent.knowledge_base is not None
        assert agent.verifier is not None
        assert agent._query_history == []

    def test_check_compliance_returns_report(self, mod):
        agent = mod.ProtocolRAGComplianceAgent()
        report = agent.check_compliance(
            "Begin robotic dissection with AI guidance",
            {
                "current_force_n": 3.0,
                "max_force_n": 10.0,
                "within_workspace": True,
                "surgeon_override_available": True,
                "informed_consent_signed": True,
                "robot_self_test_passed": True,
            },
        )
        assert isinstance(report, mod.ComplianceReport)
        assert report.retrieved_sections > 0

    def test_query_populates_audit_trail(self, mod):
        agent = mod.ProtocolRAGComplianceAgent()
        agent.check_compliance(
            "Continue tissue retraction",
            {"current_force_n": 3.0, "max_force_n": 10.0},
        )
        history = agent.get_query_history()
        assert len(history) == 1
        assert "query" in history[0]
        assert "overall_status" in history[0]

    def test_multiple_queries_tracked(self, mod):
        agent = mod.ProtocolRAGComplianceAgent()
        agent.check_compliance("Query 1", {})
        agent.check_compliance("Query 2", {})
        history = agent.get_query_history()
        assert len(history) == 2

    def test_compliance_report_has_citations(self, mod):
        agent = mod.ProtocolRAGComplianceAgent()
        report = agent.check_compliance(
            "Enroll patient with ECOG status 2",
            {"ecog_status": 2},
        )
        assert len(report.citations) > 0

    def test_answer_regulatory_query(self, mod):
        agent = mod.ProtocolRAGComplianceAgent()
        answer = agent.answer_regulatory_query("What are the requirements for AI surgical planning?")
        assert "answer" in answer
        assert "citations" in answer
        assert len(answer["citations"]) > 0

    def test_answer_regulatory_query_has_confidence(self, mod):
        agent = mod.ProtocolRAGComplianceAgent()
        answer = agent.answer_regulatory_query("What force limits apply during robotic surgery?")
        assert "confidence" in answer
        assert answer["confidence"] > 0.0

    def test_non_compliant_generates_recommendations(self, mod):
        agent = mod.ProtocolRAGComplianceAgent()
        report = agent.check_compliance(
            "Continue tissue retraction during safety stopping assessment",
            {"current_force_n": 18.0, "max_force_n": 10.0},
        )
        if report.overall_status == mod.ComplianceStatus.NON_COMPLIANT:
            assert len(report.recommendations) > 0

    def test_overall_status_compliant(self, mod):
        agent = mod.ProtocolRAGComplianceAgent()
        report = agent.check_compliance(
            "Begin robotic dissection with AI guidance",
            {
                "current_force_n": 2.0,
                "max_force_n": 10.0,
                "within_workspace": True,
                "surgeon_override_available": True,
                "informed_consent_signed": True,
                "robot_self_test_passed": True,
                "estimated_blood_loss_ml": 50,
            },
        )
        # With all contexts properly set, should not be non-compliant
        assert report.overall_status in (
            mod.ComplianceStatus.COMPLIANT,
            mod.ComplianceStatus.REQUIRES_REVIEW,
        )
