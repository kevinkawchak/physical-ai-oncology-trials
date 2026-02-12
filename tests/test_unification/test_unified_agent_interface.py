"""Unit tests for unification/agentic_generative_ai/unified_agent_interface.py.

Tests cover AgentBackend, AgentRole enums, Tool dataclass format conversions,
AgentMessage, AgentConfig dataclasses.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from tests.conftest import load_module

MOD_PATH = "unification/agentic_generative_ai/unified_agent_interface.py"


@pytest.fixture()
def mod():
    return load_module("unified_agent_interface", MOD_PATH)


class TestEnums:
    def test_agent_backend(self, mod):
        names = [e.name for e in mod.AgentBackend]
        for b in ("CREWAI", "LANGGRAPH", "AUTOGEN", "CUSTOM"):
            assert b in names

    def test_agent_role(self, mod):
        names = [e.name for e in mod.AgentRole]
        for r in ("PLANNER", "EXECUTOR", "MONITOR", "SAFETY_SUPERVISOR"):
            assert r in names


class TestToolDataclass:
    def test_creation(self, mod):
        tool = mod.Tool(
            name="get_vitals",
            description="Get patient vital signs",
            function=lambda: None,
            input_schema={"type": "object", "properties": {}},
        )
        assert tool.name == "get_vitals"

    def test_to_mcp_format(self, mod):
        tool = mod.Tool(
            name="check_force",
            description="Check force reading",
            function=lambda: None,
            input_schema={"type": "object", "properties": {"threshold": {"type": "number"}}},
        )
        mcp = tool.to_mcp_format()
        assert isinstance(mcp, dict)
        assert "name" in mcp

    def test_to_openai_format(self, mod):
        tool = mod.Tool(
            name="plan_procedure",
            description="Generate surgical plan",
            function=lambda: None,
            input_schema={"type": "object", "properties": {}},
        )
        openai_fmt = tool.to_openai_format()
        assert isinstance(openai_fmt, dict)

    def test_to_anthropic_format(self, mod):
        tool = mod.Tool(
            name="verify_consent",
            description="Verify informed consent",
            function=lambda: None,
            input_schema={"type": "object", "properties": {}},
        )
        anthropic_fmt = tool.to_anthropic_format()
        assert isinstance(anthropic_fmt, dict)


class TestAgentMessage:
    def test_creation(self, mod):
        msg = mod.AgentMessage(
            role="assistant",
            content="Treatment plan generated.",
        )
        assert msg.role == "assistant"
        assert msg.content == "Treatment plan generated."

    def test_to_dict(self, mod):
        msg = mod.AgentMessage(
            role="user",
            content="What is the recommended dose?",
        )
        d = msg.to_dict()
        assert isinstance(d, dict)
        assert d["role"] == "user"


class TestAgentConfig:
    def test_creation(self, mod):
        config = mod.AgentConfig(
            name="Safety Supervisor",
            role=mod.AgentRole.SAFETY_SUPERVISOR,
            description="Monitors safety constraints",
            backend=mod.AgentBackend.CUSTOM,
            model="claude-opus-4-6",
            temperature=0.0,
            max_tokens=4096,
        )
        assert config.name == "Safety Supervisor"
        assert config.role == mod.AgentRole.SAFETY_SUPERVISOR
        assert config.temperature == 0.0
