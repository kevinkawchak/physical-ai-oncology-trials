"""Fixtures for agentic AI tests."""

from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def react_mod():
    """Load the ReAct procedure planner module."""
    return load_module(
        "react_planner",
        "agentic-ai/examples-agentic-ai/02_react_procedure_planner.py",
    )


@pytest.fixture()
def adaptive_agent_mod():
    """Load the realtime adaptive treatment agent module."""
    return load_module(
        "adaptive_treatment_agent",
        "agentic-ai/examples-agentic-ai/03_realtime_adaptive_treatment_agent.py",
    )


@pytest.fixture()
def orchestrator_mod():
    """Load the autonomous simulation orchestrator module."""
    return load_module(
        "simulation_orchestrator",
        "agentic-ai/examples-agentic-ai/04_autonomous_simulation_orchestrator.py",
    )


@pytest.fixture()
def safety_exec_mod():
    """Load the safety-constrained agent executor module."""
    return load_module(
        "safety_constrained_executor",
        "agentic-ai/examples-agentic-ai/05_safety_constrained_agent_executor.py",
    )


@pytest.fixture()
def rag_mod():
    """Load the protocol RAG compliance agent module."""
    return load_module(
        "protocol_rag",
        "agentic-ai/examples-agentic-ai/06_protocol_rag_compliance_agent.py",
    )
