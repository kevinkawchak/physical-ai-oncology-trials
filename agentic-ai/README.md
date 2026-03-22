# Agentic AI for Physical AI Oncology Trials

**Version**: 2.7.1
**Last Updated**: March 2026

LLM-based robot control, multi-agent systems, and autonomous clinical workflows for oncology trials.

## Overview

This directory contains strengths, limitations, and results documentation for agentic AI approaches in physical AI oncology, along with 6 comprehensive engineering examples.

## Agentic AI Engineering Examples

The `examples-agentic-ai/` directory contains **6 comprehensive code examples** for engineers building agentic AI systems for robotic oncology trials. These cover MCP integration, reasoning agents, real-time decision support, simulation orchestration, formal safety constraints, and RAG-based regulatory compliance.

### Available Examples

| Example | Use Case | Key Techniques |
|---------|----------|---------------|
| `01_mcp_clinical_robotics_server.py` | MCP server for robot telemetry, imaging, vitals | Model Context Protocol, audit trails, keep-out zones |
| `02_react_procedure_planner.py` | ReAct surgical procedure planning with reasoning | Chain-of-thought, instrument selection, margin estimation |
| `03_realtime_adaptive_treatment_agent.py` | Real-time multi-modal anomaly detection and response | Cross-modal correlation, hemodynamic monitoring, streaming |
| `04_autonomous_simulation_orchestrator.py` | Autonomous experiment design, execution, and analysis | Parameter sweeps, cross-framework validation, sensitivity |
| `05_safety_constrained_agent_executor.py` | Formal safety constraints for agent-controlled robots | Pre/post-conditions, invariants, safety gates, rollback |
| `06_protocol_rag_compliance_agent.py` | RAG agent grounded in regulatory documents | Document retrieval, compliance verification, citations |

### Quick Start

```bash
# Run MCP server demo
python agentic-ai/examples-agentic-ai/01_mcp_clinical_robotics_server.py

# Run ReAct procedure planner
python agentic-ai/examples-agentic-ai/02_react_procedure_planner.py

# Run safety-constrained executor demo
python agentic-ai/examples-agentic-ai/05_safety_constrained_agent_executor.py

# Each example includes detailed inline documentation and architecture diagrams
```

## Directory Structure

```
agentic-ai/
├── README.md                  # This file
├── strengths.md               # Strengths of agentic AI approaches
├── limitations.md             # Limitations and challenges
├── results.md                 # Results and benchmarks
└── examples-agentic-ai/
    ├── 01_mcp_clinical_robotics_server.py
    ├── 02_react_procedure_planner.py
    ├── 03_realtime_adaptive_treatment_agent.py
    ├── 04_autonomous_simulation_orchestrator.py
    ├── 05_safety_constrained_agent_executor.py
    └── 06_protocol_rag_compliance_agent.py
```

## License

MIT License — See [LICENSE](../LICENSE) for details.
