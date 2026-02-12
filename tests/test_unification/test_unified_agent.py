"""Tests for Unified Agent Interface.

Validates Tool format conversions, AgentMessage serialisation,
AgentConfig defaults, UnifiedAgent construction, AgentTeam
coordination, and OncologyToolkit factory methods.

LICENSE: MIT
"""

from __future__ import annotations

import asyncio

import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
MOD_PATH = "unification/agentic_generative_ai/unified_agent_interface.py"


@pytest.fixture()
def agent_mod():
    return load_module("unified_agent_interface", MOD_PATH)


@pytest.fixture()
def dummy_tool(agent_mod):
    return agent_mod.Tool(
        name="test_tool",
        description="A test tool",
        function=lambda x: {"result": x},
        input_schema={"x": {"type": "string"}},
        safety_level="normal",
    )


@pytest.fixture()
def agent(agent_mod):
    return agent_mod.UnifiedAgent(
        name="test_agent",
        role="tester",
        description="Test agent",
        backend="custom",
    )


# ===========================================================================
# TestEnums
# ===========================================================================


class TestEnums:
    def test_agent_backend_values(self, agent_mod):
        names = {e.name for e in agent_mod.AgentBackend}
        assert names == {"CREWAI", "LANGGRAPH", "AUTOGEN", "CUSTOM"}

    def test_agent_role_values(self, agent_mod):
        names = {e.name for e in agent_mod.AgentRole}
        assert "PLANNER" in names
        assert "SAFETY_SUPERVISOR" in names
        assert len(names) == 6


# ===========================================================================
# TestTool
# ===========================================================================


class TestTool:
    def test_to_mcp_format(self, dummy_tool):
        mcp = dummy_tool.to_mcp_format()
        assert mcp["name"] == "test_tool"
        assert "inputSchema" in mcp
        assert mcp["inputSchema"]["type"] == "object"

    def test_to_openai_format(self, dummy_tool):
        oai = dummy_tool.to_openai_format()
        assert oai["type"] == "function"
        assert oai["function"]["name"] == "test_tool"

    def test_to_anthropic_format(self, dummy_tool):
        anth = dummy_tool.to_anthropic_format()
        assert anth["name"] == "test_tool"
        assert "input_schema" in anth

    def test_safety_level_default(self, agent_mod):
        t = agent_mod.Tool(name="t", description="d", function=lambda: None)
        assert t.safety_level == "normal"
        assert t.requires_confirmation is False


# ===========================================================================
# TestAgentMessage
# ===========================================================================


class TestAgentMessage:
    def test_to_dict(self, agent_mod):
        msg = agent_mod.AgentMessage(role="user", content="hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"
        assert "timestamp" in d

    def test_default_fields(self, agent_mod):
        msg = agent_mod.AgentMessage(role="assistant", content="hi")
        assert msg.tool_calls is None
        assert msg.tool_results is None
        assert msg.metadata == {}


# ===========================================================================
# TestAgentConfig
# ===========================================================================


class TestAgentConfig:
    def test_defaults(self, agent_mod):
        cfg = agent_mod.AgentConfig(name="a", role="r")
        assert cfg.backend == agent_mod.AgentBackend.CUSTOM
        assert cfg.model == "claude-sonnet-4"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096
        assert cfg.tools == []
        assert cfg.safety_constraints == []


# ===========================================================================
# TestUnifiedAgent
# ===========================================================================


class TestUnifiedAgent:
    def test_construction(self, agent, agent_mod):
        assert agent.config.name == "test_agent"
        assert agent.config.backend == agent_mod.AgentBackend.CUSTOM
        assert agent.execution_count == 0

    def test_add_tool(self, agent, dummy_tool):
        agent.add_tool(dummy_tool)
        assert len(agent.config.tools) == 1
        assert agent.config.tools[0].name == "test_tool"

    def test_add_safety_constraint(self, agent):
        agent.add_safety_constraint("Must verify patient identity")
        assert len(agent.config.safety_constraints) == 1

    def test_get_statistics(self, agent):
        stats = agent.get_statistics()
        assert stats["name"] == "test_agent"
        assert stats["execution_count"] == 0
        assert stats["tool_count"] == 0

    def test_execute_async(self, agent):
        response = asyncio.get_event_loop().run_until_complete(agent.execute("test prompt"))
        assert response.role == "assistant"
        assert agent.execution_count == 1
        assert len(agent.message_history) == 2  # user + assistant

    def test_backend_string_conversion(self, agent_mod):
        a = agent_mod.UnifiedAgent(name="a", role="r", backend="crewai")
        assert a.config.backend == agent_mod.AgentBackend.CREWAI


# ===========================================================================
# TestAgentTeam
# ===========================================================================


class TestAgentTeam:
    def test_construction(self, agent_mod):
        team = agent_mod.AgentTeam(name="test_team")
        assert team.name == "test_team"
        assert len(team.agents) == 0

    def test_add_agent(self, agent_mod, agent):
        team = agent_mod.AgentTeam()
        team.add_agent(agent)
        assert "test_agent" in team.agents

    def test_get_agent(self, agent_mod, agent):
        team = agent_mod.AgentTeam()
        team.add_agent(agent)
        assert team.get_agent("test_agent") is agent
        assert team.get_agent("nonexistent") is None

    def test_team_statistics(self, agent_mod, agent):
        team = agent_mod.AgentTeam(name="stats_team")
        team.add_agent(agent)
        stats = team.get_team_statistics()
        assert stats["team_name"] == "stats_team"
        assert stats["agent_count"] == 1
        assert "test_agent" in stats["agents"]

    def test_execute_sequential(self, agent_mod):
        a = agent_mod.UnifiedAgent(name="agent_a", role="r")
        team = agent_mod.AgentTeam()
        team.add_agent(a)
        tasks = [{"agent": "agent_a", "prompt": "do something"}]
        responses = asyncio.get_event_loop().run_until_complete(team.execute_sequential(tasks))
        assert len(responses) == 1

    def test_execute_parallel(self, agent_mod):
        a = agent_mod.UnifiedAgent(name="par_a", role="r")
        b = agent_mod.UnifiedAgent(name="par_b", role="r")
        team = agent_mod.AgentTeam()
        team.add_agent(a)
        team.add_agent(b)
        tasks = [
            {"agent": "par_a", "prompt": "task1"},
            {"agent": "par_b", "prompt": "task2"},
        ]
        responses = asyncio.get_event_loop().run_until_complete(team.execute_parallel(tasks))
        assert len(responses) == 2


# ===========================================================================
# TestOncologyToolkit
# ===========================================================================


class TestOncologyToolkit:
    def test_robot_control_tools(self, agent_mod):
        tools = agent_mod.OncologyToolkit.robot_control_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert "move_to_position" in names
        assert "emergency_stop" in names

    def test_clinical_tools(self, agent_mod):
        tools = agent_mod.OncologyToolkit.clinical_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert "verify_patient_identity" in names
        assert "log_adverse_event" in names

    def test_imaging_tools(self, agent_mod):
        tools = agent_mod.OncologyToolkit.imaging_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "get_current_image" in names
        assert "segment_tissue" in names

    def test_tool_safety_levels(self, agent_mod):
        robot_tools = agent_mod.OncologyToolkit.robot_control_tools()
        estop = [t for t in robot_tools if t.name == "emergency_stop"][0]
        assert estop.safety_level == "critical"
