#!/usr/bin/env python3
"""
Unified Agent Interface for Physical AI Oncology Trials

Provides a framework-agnostic interface for AI agents, enabling seamless
integration across CrewAI, LangGraph, AutoGen, and custom implementations.

Usage:
    from unification.agentic_generative_ai.unified_agent_interface import (
        UnifiedAgent, AgentTeam, OncologyToolkit
    )

    agent = UnifiedAgent(
        name="surgical_assistant",
        role="Provide surgical instruments and maintain sterility",
        backend="crewai"  # or "langgraph", "autogen", "custom"
    )

    response = await agent.execute("Prepare the next instrument")

Last updated: January 2026
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import logging
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentBackend(Enum):
    """Supported agent framework backends."""
    CREWAI = "crewai"
    LANGGRAPH = "langgraph"
    AUTOGEN = "autogen"
    CUSTOM = "custom"


class AgentRole(Enum):
    """Standard roles for oncology agents."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    MONITOR = "monitor"
    COORDINATOR = "coordinator"
    DOCUMENTER = "documenter"
    SAFETY_SUPERVISOR = "safety_supervisor"


@dataclass
class Tool:
    """Unified tool definition."""
    name: str
    description: str
    function: Callable
    input_schema: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    safety_level: str = "normal"  # normal, elevated, critical

    def to_mcp_format(self) -> Dict:
        """Convert to Model Context Protocol format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.input_schema,
            }
        }

    def to_openai_format(self) -> Dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.input_schema,
                }
            }
        }

    def to_anthropic_format(self) -> Dict:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.input_schema,
            }
        }


@dataclass
class AgentMessage:
    """Standardized message format for agent communication."""
    role: str  # user, assistant, system, tool
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_results: Optional[List[Dict]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


@dataclass
class AgentConfig:
    """Configuration for unified agent."""
    name: str
    role: str
    description: str = ""
    backend: AgentBackend = AgentBackend.CUSTOM
    model: str = "claude-sonnet-4"
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: List[Tool] = field(default_factory=list)
    system_prompt: str = ""
    safety_constraints: List[str] = field(default_factory=list)


class AgentBackendAdapter(ABC):
    """Abstract base class for backend adapters."""

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        context: Dict[str, Any],
        tools: List[Tool]
    ) -> AgentMessage:
        """Execute agent with given prompt and context."""
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Get the name of the backend."""
        pass


class CrewAIAdapter(AgentBackendAdapter):
    """Adapter for CrewAI backend."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._agent = None

    def _initialize(self):
        """Initialize CrewAI agent."""
        try:
            from crewai import Agent as CrewAgent
            self._agent = CrewAgent(
                role=self.config.role,
                goal=self.config.description,
                backstory=self.config.system_prompt,
                tools=[self._convert_tool(t) for t in self.config.tools],
                llm=self.config.model,
                verbose=True
            )
        except ImportError:
            logger.warning("CrewAI not installed, using mock implementation")
            self._agent = None

    def _convert_tool(self, tool: Tool):
        """Convert unified tool to CrewAI format."""
        # CrewAI uses LangChain-style tools
        return tool.function

    async def execute(
        self,
        prompt: str,
        context: Dict[str, Any],
        tools: List[Tool]
    ) -> AgentMessage:
        """Execute using CrewAI."""
        if self._agent is None:
            self._initialize()

        if self._agent is None:
            # Mock response if CrewAI not available
            return AgentMessage(
                role="assistant",
                content=f"[CrewAI Mock] Processing: {prompt}",
                metadata={"backend": "crewai", "mock": True}
            )

        # Execute with CrewAI
        # result = self._agent.execute_task(prompt)
        # For now, return mock
        return AgentMessage(
            role="assistant",
            content=f"[CrewAI] Executed: {prompt}",
            metadata={"backend": "crewai"}
        )

    def get_backend_name(self) -> str:
        return "crewai"


class LangGraphAdapter(AgentBackendAdapter):
    """Adapter for LangGraph backend."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._graph = None

    def _initialize(self):
        """Initialize LangGraph workflow."""
        try:
            from langgraph.graph import StateGraph
            # Create simple agent graph
            self._graph = StateGraph(dict)
            # Would add nodes and edges here
        except ImportError:
            logger.warning("LangGraph not installed, using mock implementation")
            self._graph = None

    async def execute(
        self,
        prompt: str,
        context: Dict[str, Any],
        tools: List[Tool]
    ) -> AgentMessage:
        """Execute using LangGraph."""
        if self._graph is None:
            self._initialize()

        if self._graph is None:
            return AgentMessage(
                role="assistant",
                content=f"[LangGraph Mock] Processing: {prompt}",
                metadata={"backend": "langgraph", "mock": True}
            )

        # Execute with LangGraph
        return AgentMessage(
            role="assistant",
            content=f"[LangGraph] Executed: {prompt}",
            metadata={"backend": "langgraph"}
        )

    def get_backend_name(self) -> str:
        return "langgraph"


class CustomAdapter(AgentBackendAdapter):
    """Custom adapter for direct API calls."""

    def __init__(self, config: AgentConfig):
        self.config = config

    async def execute(
        self,
        prompt: str,
        context: Dict[str, Any],
        tools: List[Tool]
    ) -> AgentMessage:
        """Execute using custom implementation."""
        # Would make direct API call here
        return AgentMessage(
            role="assistant",
            content=f"[Custom] Processed: {prompt}",
            metadata={"backend": "custom", "model": self.config.model}
        )

    def get_backend_name(self) -> str:
        return "custom"


class UnifiedAgent:
    """
    Framework-agnostic AI agent for oncology clinical trials.

    Supports multiple backends (CrewAI, LangGraph, AutoGen) through
    a unified interface, enabling seamless integration and portability.
    """

    def __init__(
        self,
        name: str,
        role: str,
        description: str = "",
        backend: Union[str, AgentBackend] = "custom",
        model: str = "claude-sonnet-4",
        tools: Optional[List[Tool]] = None,
        system_prompt: str = "",
        safety_constraints: Optional[List[str]] = None
    ):
        """
        Initialize unified agent.

        Args:
            name: Agent identifier
            role: Agent's role description
            description: Detailed description of agent's purpose
            backend: Framework backend to use
            model: LLM model identifier
            tools: List of tools available to agent
            system_prompt: System prompt for agent
            safety_constraints: List of safety constraints
        """
        if isinstance(backend, str):
            backend = AgentBackend(backend)

        self.config = AgentConfig(
            name=name,
            role=role,
            description=description,
            backend=backend,
            model=model,
            tools=tools or [],
            system_prompt=system_prompt,
            safety_constraints=safety_constraints or []
        )

        self.adapter = self._create_adapter()
        self.message_history: List[AgentMessage] = []
        self.execution_count = 0

        logger.info(f"Initialized UnifiedAgent '{name}' with backend '{backend.value}'")

    def _create_adapter(self) -> AgentBackendAdapter:
        """Create appropriate backend adapter."""
        adapters = {
            AgentBackend.CREWAI: CrewAIAdapter,
            AgentBackend.LANGGRAPH: LangGraphAdapter,
            AgentBackend.AUTOGEN: CustomAdapter,  # Would use AutoGen adapter
            AgentBackend.CUSTOM: CustomAdapter,
        }

        adapter_class = adapters.get(self.config.backend, CustomAdapter)
        return adapter_class(self.config)

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.config.tools.append(tool)
        logger.info(f"Added tool '{tool.name}' to agent '{self.config.name}'")

    def add_safety_constraint(self, constraint: str) -> None:
        """Add a safety constraint."""
        self.config.safety_constraints.append(constraint)

    async def execute(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Execute agent with given prompt.

        Args:
            prompt: User prompt/instruction
            context: Additional context for execution

        Returns:
            Agent response message
        """
        context = context or {}

        # Add safety constraints to context
        context["safety_constraints"] = self.config.safety_constraints

        # Add message history for context
        context["history"] = [m.to_dict() for m in self.message_history[-10:]]

        # Execute via adapter
        start_time = time.time()
        response = await self.adapter.execute(prompt, context, self.config.tools)
        execution_time = time.time() - start_time

        # Record execution
        self.message_history.append(AgentMessage(role="user", content=prompt))
        self.message_history.append(response)
        self.execution_count += 1

        # Add execution metadata
        response.metadata["execution_time"] = execution_time
        response.metadata["execution_count"] = self.execution_count

        logger.info(
            f"Agent '{self.config.name}' executed in {execution_time:.2f}s"
        )

        return response

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent execution statistics."""
        return {
            "name": self.config.name,
            "backend": self.config.backend.value,
            "execution_count": self.execution_count,
            "tool_count": len(self.config.tools),
            "history_length": len(self.message_history),
        }


class AgentTeam:
    """
    Coordinate multiple unified agents as a team.

    Enables multi-agent workflows for complex oncology procedures.
    """

    def __init__(self, name: str = "oncology_team"):
        """Initialize agent team."""
        self.name = name
        self.agents: Dict[str, UnifiedAgent] = {}
        self.execution_log: List[Dict] = []

    def add_agent(self, agent: UnifiedAgent) -> None:
        """Add an agent to the team."""
        self.agents[agent.config.name] = agent
        logger.info(f"Added agent '{agent.config.name}' to team '{self.name}'")

    def get_agent(self, name: str) -> Optional[UnifiedAgent]:
        """Get agent by name."""
        return self.agents.get(name)

    async def execute_sequential(
        self,
        tasks: List[Dict[str, str]]
    ) -> List[AgentMessage]:
        """
        Execute tasks sequentially with specified agents.

        Args:
            tasks: List of {"agent": name, "prompt": prompt} dicts

        Returns:
            List of responses
        """
        responses = []

        for task in tasks:
            agent_name = task["agent"]
            prompt = task["prompt"]

            agent = self.agents.get(agent_name)
            if agent is None:
                logger.warning(f"Agent '{agent_name}' not found")
                continue

            response = await agent.execute(prompt)
            responses.append(response)

            self.execution_log.append({
                "agent": agent_name,
                "prompt": prompt,
                "response": response.content,
                "timestamp": time.time()
            })

        return responses

    async def execute_parallel(
        self,
        tasks: List[Dict[str, str]]
    ) -> List[AgentMessage]:
        """
        Execute tasks in parallel.

        Args:
            tasks: List of {"agent": name, "prompt": prompt} dicts

        Returns:
            List of responses
        """
        coroutines = []

        for task in tasks:
            agent_name = task["agent"]
            prompt = task["prompt"]

            agent = self.agents.get(agent_name)
            if agent is not None:
                coroutines.append(agent.execute(prompt))

        responses = await asyncio.gather(*coroutines)
        return list(responses)

    def get_team_statistics(self) -> Dict[str, Any]:
        """Get team statistics."""
        return {
            "team_name": self.name,
            "agent_count": len(self.agents),
            "total_executions": sum(
                a.execution_count for a in self.agents.values()
            ),
            "execution_log_length": len(self.execution_log),
            "agents": {
                name: agent.get_statistics()
                for name, agent in self.agents.items()
            }
        }


class OncologyToolkit:
    """Pre-built tools for oncology clinical trial agents."""

    @staticmethod
    def robot_control_tools() -> List[Tool]:
        """Get robot control tools."""
        return [
            Tool(
                name="move_to_position",
                description="Move robot end-effector to target position",
                function=lambda pos: {"status": "moved", "position": pos},
                input_schema={
                    "position": {"type": "array", "items": {"type": "number"}},
                    "speed": {"type": "number", "default": 0.1}
                },
                safety_level="elevated"
            ),
            Tool(
                name="get_robot_state",
                description="Get current robot state including joint positions and forces",
                function=lambda: {"joints": [], "forces": [], "status": "ready"},
                input_schema={},
                safety_level="normal"
            ),
            Tool(
                name="emergency_stop",
                description="Immediately stop all robot motion",
                function=lambda: {"status": "stopped"},
                input_schema={},
                safety_level="critical"
            ),
        ]

    @staticmethod
    def clinical_tools() -> List[Tool]:
        """Get clinical workflow tools."""
        return [
            Tool(
                name="verify_patient_identity",
                description="Verify patient identity against protocol",
                function=lambda pid: {"verified": True, "patient_id": pid},
                input_schema={
                    "patient_id": {"type": "string"},
                    "protocol_id": {"type": "string"}
                },
                requires_confirmation=True,
                safety_level="elevated"
            ),
            Tool(
                name="check_protocol_step",
                description="Check current protocol step and requirements",
                function=lambda step: {"step": step, "requirements": []},
                input_schema={
                    "step_number": {"type": "integer"}
                },
                safety_level="normal"
            ),
            Tool(
                name="log_adverse_event",
                description="Log an adverse event for regulatory reporting",
                function=lambda event: {"logged": True, "event_id": "AE-001"},
                input_schema={
                    "event_type": {"type": "string"},
                    "severity": {"type": "string"},
                    "description": {"type": "string"}
                },
                requires_confirmation=True,
                safety_level="critical"
            ),
        ]

    @staticmethod
    def imaging_tools() -> List[Tool]:
        """Get imaging and visualization tools."""
        return [
            Tool(
                name="get_current_image",
                description="Get current camera/endoscope image",
                function=lambda: {"image": None, "timestamp": time.time()},
                input_schema={
                    "camera_id": {"type": "string", "default": "endoscope"}
                },
                safety_level="normal"
            ),
            Tool(
                name="segment_tissue",
                description="Segment tissue in current view",
                function=lambda: {"segments": [], "confidence": 0.95},
                input_schema={
                    "tissue_type": {"type": "string"}
                },
                safety_level="normal"
            ),
        ]


async def demo():
    """Demonstrate unified agent interface."""
    print("=" * 60)
    print("Unified Agent Interface for Oncology Clinical Trials")
    print("=" * 60)
    print()

    # Create agents with different backends
    planner = UnifiedAgent(
        name="procedure_planner",
        role="Plan surgical procedures and coordinate team",
        backend="custom",
        model="claude-opus-4",
        tools=OncologyToolkit.clinical_tools()
    )

    executor = UnifiedAgent(
        name="robot_controller",
        role="Control surgical robot during procedures",
        backend="custom",
        model="claude-sonnet-4",
        tools=OncologyToolkit.robot_control_tools()
    )

    # Create team
    team = AgentTeam(name="surgical_team")
    team.add_agent(planner)
    team.add_agent(executor)

    # Execute tasks
    print("Executing sequential tasks...")
    responses = await team.execute_sequential([
        {"agent": "procedure_planner", "prompt": "Plan biopsy procedure for lung nodule"},
        {"agent": "robot_controller", "prompt": "Prepare robot for needle insertion"},
    ])

    for response in responses:
        print(f"  Response: {response.content}")
        print(f"  Backend: {response.metadata.get('backend')}")
        print()

    # Show statistics
    print("Team Statistics:")
    stats = team.get_team_statistics()
    print(f"  Total agents: {stats['agent_count']}")
    print(f"  Total executions: {stats['total_executions']}")
    print()

    print("=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
