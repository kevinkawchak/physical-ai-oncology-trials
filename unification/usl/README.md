# Unification Standard Level (USL) for Physical AI Oncology Trials

*Standardizing and Evaluating Robot Unification Readiness for Multi-Site Clinical Trials (February 2026)*

---

## Overview

The **Unification Standard Level (USL)** is a scoring framework for evaluating how ready a physical AI robot is for deployment in unified, multi-site oncology clinical trials. USL scores range from **1.0 to 10.0** (in 0.1 increments) and assess four weighted dimensions:

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **A) Simulation Framework Switching** | 25% | Ability to move trained policies between simulation engines (Isaac Lab, MuJoCo, Gazebo, PyBullet) |
| **B) Generative / Agentic AI Integration** | 25% | Integration with LLMs, VLAs, diffusion policies, Claude Code, Codex, MCP, and agentic frameworks |
| **C) Cross-Robot Progress Sharing** | 25% | Capacity to share and continue progress with other robots (intra- and inter-organization) |
| **D) Multi-Site Clinical Trial Collaboration** | 25% | Readiness for federated, regulatory-compliant deployment across clinical trial sites |

Each dimension derives from the four unification pillars defined in [`unification/`](../):
- Dimension A ← [`simulation_physics/`](../simulation_physics/)
- Dimension B ← [`agentic_generative_ai/`](../agentic_generative_ai/)
- Dimension C ← [`cross_platform_tools/`](../cross_platform_tools/) and [`surgical_robotics/`](../surgical_robotics/)
- Dimension D ← [`../../federation/`](../../federation/) and [`../../regulatory/`](../../regulatory/)

---

## USL Score Bands

| Score Range | Band | Description |
|-------------|------|-------------|
| 9.0 – 10.0 | **Exemplary** | Fully unified, multi-site clinical trial ready |
| 7.0 – 8.9 | **Advanced** | Strong unification, near clinical-trial ready |
| 5.0 – 6.9 | **Intermediate** | Partial unification, significant work remaining |
| 3.0 – 4.9 | **Foundational** | Basic interoperability, major gaps exist |
| 1.0 – 2.9 | **Initial** | Minimal unification capability |

---

## USL Level Definitions

| Level | Name | Description |
|-------|------|-------------|
| 1 | Conceptual | Robot exists; no simulation or AI integration attempted |
| 2 | Exploratory | Single framework tested; basic model available |
| 3 | Basic | 2+ frameworks; initial AI experiments conducted |
| 4 | Developing | Cross-framework transfer demonstrated; AI planning tested |
| 5 | Functional | 3+ frameworks; agentic AI operational; intra-org sharing |
| 6 | Integrated | Multi-framework validated; LLM planning; inter-org sharing |
| 7 | Advanced | GPU sim + policy transfer; MCP/VLA integration; skill sharing |
| 8 | Clinical-Ready | Multi-site tested; regulatory docs; federated learning |
| 9 | Validated | Full regulatory compliance; multi-site trials active |
| 10 | Exemplary | Production deployment; open consortium; continuous improvement |

---

## Evaluated Cobots (Category: Collaborative Robots)

This initial USL evaluation covers three state-of-the-art open-source collaborative robot arms from different manufacturers, each with active ROS 2 support, MuJoCo Menagerie models, and potential oncology applications.

| Robot | Manufacturer | USL Score | USL Level | Band |
|-------|-------------|-----------|-----------|------|
| [Franka Emika Panda](#franka-emika-panda) | Franka Robotics | **7.4** | 7 (Advanced) | Advanced |
| [Kinova Gen3 7DoF](#kinova-gen3-7dof) | Kinova Robotics | **5.7** | 5 (Functional) | Intermediate |
| [UFACTORY xArm 7](#ufactory-xarm-7) | UFACTORY | **3.4** | 3 (Basic) | Foundational |

---

## Diagram 1: General Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│               GENERAL COBOT COMPARISON — USL Category: Cobots          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐
│  │   FRANKA EMIKA PANDA │  │   KINOVA GEN3 7DoF   │  │  UFACTORY xARM 7 │
│  │   (Franka Robotics)  │  │  (Kinova Robotics)   │  │    (UFACTORY)    │
│  ├──────────────────────┤  ├──────────────────────┤  ├──────────────────┤
│  │ Heritage: Research   │  │ Heritage: Assistive  │  │ Heritage: Cost-  │
│  │  robotics leader     │  │  & rehabilitation    │  │  effective cobot  │
│  │                      │  │                      │  │                  │
│  │ Ecosystem: Largest   │  │ Ecosystem: Strong    │  │ Ecosystem:       │
│  │  open-source cobot   │  │  ROS 2 + Kortex API  │  │  Growing Python  │
│  │  community worldwide │  │  clinical research   │  │  SDK community   │
│  │                      │  │                      │  │                  │
│  │ Key Strength:        │  │ Key Strength:        │  │ Key Strength:    │
│  │  Most RL/AI research │  │  Lightest arm (8.2kg)│  │  Most affordable │
│  │  papers of any cobot │  │  + integrated vision │  │  7-DOF + built-  │
│  │                      │  │                      │  │  in collision    │
│  │ Primary Application: │  │ Primary Application: │  │ Primary App:     │
│  │  Surgical assistance │  │  Bedside care &      │  │  Lab automation  │
│  │  & lab automation    │  │  patient interaction │  │  & prototyping   │
│  │                      │  │                      │  │                  │
│  │ USL Score: 7.4 ██████│  │ USL Score: 5.7 ████  │  │ USL Score: 3.4██ │
│  │ Level 7 — Advanced   │  │ Level 5 — Functional │  │ Level 3 — Basic  │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────┘
│                                                                         │
│  Legend: Each █ ≈ 1.2 points on the 1.0–10.0 USL scale                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 2: Technical Specifications Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│           TECHNICAL SPECIFICATIONS — Side-by-Side Comparison            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Spec               Franka Panda      Kinova Gen3       xArm 7          │
│  ─────────────────  ────────────────  ────────────────  ────────────────│
│  DOF                7                 7                 7                │
│  Payload (kg)       3.0               4.0 ◄─ highest   3.5              │
│  Reach (mm)         855               902 ◄─ highest   700              │
│  Repeatability (mm) ±0.1              ±0.1              ±0.1             │
│  Weight (kg)        18.0              8.2 ◄─ lightest  11.2             │
│  Control Freq (Hz)  1000 ◄─ highest  1000 ◄─ highest  250 (500 max)   │
│  Torque Sensors     7 joints          7 joints          7 joints         │
│  IP Rating          IP30              IP22              IP51 ◄─ highest │
│  Temp Range (°C)    5–40              0–40              0–50 ◄─ widest  │
│  Integrated Vision  No                Yes ◄─ unique    No               │
│  Collision Detect   External          External          Built-in ◄─     │
│                                                                         │
│  Simulation Frameworks Supported:                                       │
│  ─────────────────────────────────                                      │
│  MuJoCo Menagerie   ✓ Official        ✓ Official        ✓ Official      │
│  NVIDIA Isaac Lab   ✓ Official        ◐ Community       ◐ Community     │
│  Gazebo + ROS 2     ✓ Official        ✓ Official        ✓ Official      │
│  PyBullet           ✓ Community       ◐ Community       ◐ Community     │
│  Frameworks (#)     5                 4                 3                │
│                                                                         │
│  Model Formats Available:                                               │
│  ─────────────────────────                                              │
│  URDF               ✓                 ✓                 ✓                │
│  MJCF               ✓                 ✓                 ✓                │
│  SDF                ✓                 ✓                 ✓                │
│  USD                ✓                 ◐                 ◐                │
│  Xacro              ✓                 ✓                 ✓                │
│                                                                         │
│  ROS 2 Package      franka_ros2       ros2_kortex       xarm_ros2       │
│  API / SDK          libfranka         Kortex API        xArm Python SDK │
│  GitHub Stars (est) 1000+             500+              300+             │
│                                                                         │
│  ✓ = Official support   ◐ = Community/partial   ✗ = Not available       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 3: USL Scoring Breakdown

```
┌─────────────────────────────────────────────────────────────────────────┐
│              USL SCORING BREAKDOWN — Dimension-by-Dimension             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Dimension A: Simulation Framework Switching (25% weight)               │
│  ────────────────────────────────────────────────                       │
│  Franka Panda  [████████░░] 8.0   5 frameworks, GPU sim, full transfer │
│  Kinova Gen3   [██████░░░░] 6.3   4 frameworks, partial GPU, MuJoCo+  │
│  xArm 7        [████░░░░░░] 4.6   3 frameworks, basic models only     │
│                                                                         │
│  Dimension B: Generative / Agentic AI Integration (25% weight)          │
│  ────────────────────────────────────────────────────────               │
│  Franka Panda  [███████░░░] 7.7   VLA, diffusion, LLM planning, NL    │
│  Kinova Gen3   [█████░░░░░] 5.2   LLM planning, agentic, safety      │
│  xArm 7        [███░░░░░░░] 3.2   Basic generative + agentic only     │
│                                                                         │
│  Dimension C: Cross-Robot Progress Sharing (25% weight)                 │
│  ──────────────────────────────────────────────                         │
│  Franka Panda  [████████░░] 8.5   Inter-org, ONNX, skills, sync       │
│  Kinova Gen3   [█████░░░░░] 5.8   Intra-org, ONNX, ROS 2 actions     │
│  xArm 7        [███░░░░░░░] 3.8   Intra-org SDK, basic ONNX          │
│                                                                         │
│  Dimension D: Multi-Site Clinical Trial Collaboration (25% weight)      │
│  ─────────────────────────────────────────────────────                  │
│  Franka Panda  [█████░░░░░] 5.5   Federated compat, safety cert, ISO  │
│  Kinova Gen3   [█████░░░░░] 5.5   Clinical workflow, safety cert, ISO │
│  xArm 7        [██░░░░░░░░] 2.1   Remote monitoring only              │
│                                                                         │
│  ═══════════════════════════════════════════════════                    │
│  FINAL USL SCORES (weighted average):                                   │
│  ────────────────────────────────────                                   │
│  Franka Panda  [███████░░░] 7.4   Level 7 — Advanced                  │
│  Kinova Gen3   [█████░░░░░] 5.7   Level 5 — Functional                │
│  xArm 7        [███░░░░░░░] 3.4   Level 3 — Basic                    │
│                                                                         │
│  Bar scale: each █ = 1.0 point (10 blocks = 10.0)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
unification/usl/
├── README.md                              # This file
├── usl_scoring_framework.py               # Core USL scoring engine
└── cobots/                                # Collaborative Robots category
    ├── franka_panda/
    │   └── franka_panda_usl.py            # Franka Panda evaluation + tools
    ├── kinova_gen3/
    │   └── kinova_gen3_usl.py             # Kinova Gen3 evaluation + tools
    └── ufactory_xarm7/
        └── ufactory_xarm7_usl.py          # xArm 7 evaluation + tools
```

---

## Franka Emika Panda

**USL Score: 7.4 / 10.0 — Level 7 (Advanced)**

The Franka Emika Panda is the most widely used research cobot in the world, with the largest open-source ecosystem of any collaborative robot arm. Its 7-DOF design with torque sensors in all joints makes it highly capable for delicate oncology tasks.

**Key USL strengths:**
- Official models in MuJoCo Menagerie, Isaac Lab, and Gazebo
- Extensive VLA and diffusion policy research (panda-gym, ORBIT-Surgical)
- Cross-manufacturer ONNX policy transfer demonstrated
- ISO 13482 safety alignment documented

**Key USL gaps:**
- No multi-site clinical trial deployment
- MCP server integration not yet available
- HIPAA/21 CFR Part 11 tools not developed

**Open-source references:**
- [franka_ros2](https://github.com/frankaemika/franka_ros2) — Official ROS 2 package
- [libfranka](https://github.com/frankaemika/libfranka) — Real-time C++ control library
- [MuJoCo Menagerie (Panda)](https://github.com/google-deepmind/mujoco_menagerie) — Official MJCF model
- [panda-gym](https://github.com/qgallouedec/panda-gym) — Gymnasium RL environments

---

## Kinova Gen3 7DoF

**USL Score: 5.7 / 10.0 — Level 5 (Functional)**

The Kinova Gen3 brings a unique combination of lightweight design (8.2 kg), integrated vision, and a heritage in assistive/rehabilitation robotics. Its Kortex API provides both high-level and low-level control suitable for clinical environments.

**Key USL strengths:**
- Lightest 7-DOF cobot (8.2 kg) — ideal for bedside deployment
- Integrated Intel RealSense depth camera
- Official MuJoCo Menagerie model and ros2_kortex package
- Kortex API provides versatile control interfaces

**Key USL gaps:**
- Smaller research community than Franka
- No official Isaac Lab integration
- Limited VLA/diffusion policy research
- Cross-framework transfer not extensively validated

**Open-source references:**
- [ros2_kortex](https://github.com/Kinovarobotics/ros2_kortex) — Official ROS 2 package
- [Kortex API](https://github.com/Kinovarobotics/kortex) — Kinova Kortex API SDK
- [MuJoCo Menagerie (Gen3)](https://github.com/google-deepmind/mujoco_menagerie) — Official MJCF model

---

## UFACTORY xArm 7

**USL Score: 3.4 / 10.0 — Level 3 (Basic)**

The UFACTORY xArm 7 is the most affordable 7-DOF cobot in its class, with built-in collision detection and an IP51 protection rating. Its lower entry cost makes it attractive for expanding trial site networks, but its open-source ecosystem is still maturing.

**Key USL strengths:**
- Most affordable 7-DOF cobot — lowest barrier for new sites
- Built-in collision detection (no external sensors needed)
- Best environmental protection (IP51) and temperature range (0–50 °C)
- Strong intra-organization sharing across xArm family

**Key USL gaps:**
- Smallest open-source research community
- No published VLA or diffusion policy experiments
- Lower control frequency (250 Hz default)
- No regulatory pathway documentation

**Open-source references:**
- [xArm-Python-SDK](https://github.com/xArm-Developer/xArm-Python-SDK) — Official Python SDK
- [xarm_ros2](https://github.com/xArm-Developer/xarm_ros2) — Official ROS 2 package
- [MuJoCo Menagerie (xArm 7)](https://github.com/google-deepmind/mujoco_menagerie) — Official MJCF model

---

## Influences and References

The USL framework draws on established technology readiness methodologies:

1. **NASA/DOD Technology Readiness Levels (TRL)** — Mankins, J.C. (2004). *Technology Readiness Assessments: A Retrospective*. White Paper, NASA. Original 9-level TRL scale for evaluating technology maturity from basic principles (TRL 1) to flight-proven systems (TRL 9). USL adapts this graduated-maturity concept to robot unification readiness.

2. **ML Technology Readiness Levels (MLTRL)** — Lavin, A., et al. (2021). *Technology Readiness Levels for Machine Learning Systems*. GitHub: [ai-infrastructure-alliance/mltrl](https://github.com/ai-infrastructure-alliance/mltrl). Extends TRL to ML systems with levels 1–9 covering data readiness, model development, deployment, and monitoring. USL incorporates MLTRL's recognition that AI system readiness requires evaluating software, data, and integration dimensions beyond hardware alone.

3. **TRL for Complex System Integration** — Tomaschek, K., Olechowski, A., Eppinger, S., & Joglekar, N. (2015). *A Survey of Technology Readiness Level Users*. Proceedings of PICMET 2015. DOI: [10.1109/PICMET.2015.7273196](https://doi.org/10.1109/PICMET.2015.7273196). Identifies challenges in applying TRL to integrated multi-technology systems, directly relevant to evaluating robot systems that span simulation, AI, and clinical deployment.

4. **LLM Recommendations for Oncology Trials** — Kawchak, K. (2025). *Physical AI for Clinical Oncology Trials*. Zenodo. DOI: [10.5281/zenodo.17451709](https://doi.org/10.5281/zenodo.17451709). Recommends LLM usage for upcoming oncology trials and motivates the need for standardized evaluation of AI-integrated robotic systems in clinical settings. Inspiration for the USL standard.

### Additional References

- [NVIDIA Isaac Lab 2.3.1](https://github.com/isaac-sim/IsaacLab) — GPU-accelerated robot learning
- [MuJoCo 3.4.0](https://github.com/google-deepmind/mujoco) — Physics simulation
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Curated robot models
- [ROS 2 Kilted Kaiju](https://docs.ros.org/en/kilted/) — Robot middleware
- [Model Context Protocol](https://modelcontextprotocol.io/) — Agent-tool communication (AAIF/Linux Foundation)
- ISO 13482:2014 — Robots and robotic devices — Safety requirements for personal care robots
- IEC 62304:2006+AMD1:2015 — Medical device software — Software life cycle processes
- FDA Guidance: *Marketing Submission Recommendations for a Predetermined Change Control Plan for AI/ML-Enabled Device Software Functions* (August 2025)

---

## Quick Start

### Run USL Scoring Demo

```bash
python unification/usl/usl_scoring_framework.py
```

### Evaluate Individual Cobots

```bash
# Franka Emika Panda
python unification/usl/cobots/franka_panda/franka_panda_usl.py

# Kinova Gen3 7DoF
python unification/usl/cobots/kinova_gen3/kinova_gen3_usl.py

# UFACTORY xArm 7
python unification/usl/cobots/ufactory_xarm7/ufactory_xarm7_usl.py
```

### Use in Code

```python
from unification.usl.usl_scoring_framework import (
    USLRating,
    DimensionAScore,
    DimensionBScore,
    DimensionCScore,
    DimensionDScore,
    generate_evaluation_report,
    compare_ratings,
)

# Create a rating for a new cobot
rating = USLRating(
    robot_name="My Research Robot",
    manufacturer="My Organization",
)

# Set dimension scores and compute
rating.dimension_a = DimensionAScore(num_frameworks_supported=3)
rating.dimension_b = DimensionBScore(llm_task_planning=True)
rating.dimension_c = DimensionCScore(onnx_policy_export=True)
rating.dimension_d = DimensionDScore(audit_trail_capable=True)

score = rating.compute_final_score()
print(rating.summary())
```

---

## Contributing

To add a new robot category or evaluate additional cobots:

1. Create a new directory under `cobots/` (or a new category directory)
2. Implement an evaluation module following the pattern in existing cobot files
3. Add the robot to this README with all three diagram sections updated
4. Validate across at least 2 simulation frameworks
5. Submit a PR with USL scores and supporting evidence

---

*Last updated: February 2026*
