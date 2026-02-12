"""Tests for agentic-ai/examples-agentic-ai/06_protocol_rag_compliance_agent.py

Covers regulatory knowledge base, compliance verification, and RAG agent.
"""

from __future__ import annotations


# ── Enum values ──────────────────────────────────────────────────────────


class TestEnums:
    def test_document_types(self, rag_mod):
        dt = rag_mod.DocumentType
        assert dt.TRIAL_PROTOCOL.value == "trial_protocol"
        assert dt.FDA_GUIDANCE.value == "fda_guidance"
        assert dt.ICH_GUIDELINE.value == "ich_guideline"

    def test_compliance_status(self, rag_mod):
        cs = rag_mod.ComplianceStatus
        assert cs.COMPLIANT.value == "compliant"
        assert cs.NON_COMPLIANT.value == "non_compliant"
        assert cs.REQUIRES_REVIEW.value == "requires_review"


# ── DocumentChunk dataclass ──────────────────────────────────────────────


class TestDocumentChunk:
    def test_creation(self, rag_mod):
        chunk = rag_mod.DocumentChunk(
            chunk_id="DC-001",
            document_id="DOC-001",
            document_type=rag_mod.DocumentType.FDA_GUIDANCE,
            document_title="AI/ML Guidance",
            section_number="3.1",
            section_title="Validation Requirements",
            content="Models must be validated on representative data.",
        )
        assert chunk.chunk_id == "DC-001"

    def test_get_citation(self, rag_mod):
        chunk = rag_mod.DocumentChunk(
            chunk_id="DC-002",
            document_id="DOC-002",
            document_type=rag_mod.DocumentType.ICH_GUIDELINE,
            document_title="ICH E6(R3)",
            section_number="5.2",
            section_title="Monitoring",
            content="Risk-based monitoring approach.",
        )
        citation = chunk.get_citation()
        assert isinstance(citation, str)
        assert len(citation) > 0


# ── RegulatoryKnowledgeBase ──────────────────────────────────────────────


class TestRegulatoryKnowledgeBase:
    def test_creation(self, rag_mod):
        kb = rag_mod.RegulatoryKnowledgeBase()
        assert kb is not None

    def test_retrieve(self, rag_mod):
        kb = rag_mod.RegulatoryKnowledgeBase()
        results = kb.retrieve("informed consent", top_k=3)
        assert isinstance(results, list)

    def test_retrieve_returns_relevant_results(self, rag_mod):
        kb = rag_mod.RegulatoryKnowledgeBase()
        results = kb.retrieve("safety monitoring", top_k=5)
        for r in results:
            assert isinstance(r, rag_mod.RetrievalResult)
            assert r.relevance_score >= 0


# ── ComplianceVerifier ───────────────────────────────────────────────────


class TestComplianceVerifier:
    def test_creation(self, rag_mod):
        verifier = rag_mod.ComplianceVerifier()
        assert verifier is not None


# ── ProtocolRAGComplianceAgent ───────────────────────────────────────────


class TestProtocolRAGComplianceAgent:
    def test_creation(self, rag_mod):
        agent = rag_mod.ProtocolRAGComplianceAgent()
        assert agent is not None

    def test_check_compliance(self, rag_mod):
        agent = rag_mod.ProtocolRAGComplianceAgent()
        report = agent.check_compliance(
            action_description="Deploy AI model for dose calculation",
            context={"component": "dose_calculator", "environment": "clinical"},
        )
        assert isinstance(report, rag_mod.ComplianceReport)
        assert report.overall_status is not None

    def test_answer_regulatory_query(self, rag_mod):
        agent = rag_mod.ProtocolRAGComplianceAgent()
        answer = agent.answer_regulatory_query(
            "What are the requirements for AI model validation?",
        )
        assert isinstance(answer, dict)
        assert "answer" in answer

    def test_query_history(self, rag_mod):
        agent = rag_mod.ProtocolRAGComplianceAgent()
        agent.check_compliance(
            action_description="Test query",
            context={},
        )
        history = agent.get_query_history()
        assert isinstance(history, list)
        assert len(history) >= 1
