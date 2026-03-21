# Unification Standard Level (USL) for Physical AI Oncology Trials

*Standardizing and Evaluating Robot Unification Readiness for Multi-Site Clinical Trials (March 2026)*

**Version**: 2.7.1
**Last Updated**: March 2026

---

## Overview

The **Unification Standard Level (USL)/Unification Standard Levels (USLs)** is a scoring framework for evaluating how ready a physical AI robot is for deployment in unified, multi-site oncology clinical trials. USL scores range from **1.0 to 10.0** (in 0.1 increments) and assess four weighted dimensions:

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **A) Simulation Framework Switching** | 25% | Ability to move trained policies between simulation engines (Isaac Lab, MuJoCo, Gazebo, PyBullet) |
| **B) Generative / Agentic AI Integration** | 25% | Integration with LLMs, VLAs, diffusion policies, Claude Code, Codex, MCP, and agentic frameworks |
| **C) Cross-Robot Progress Sharing** | 25% | Capacity to share and continue progress with other robots (intra- and inter-organization) |
| **D) Multi-Site Clinical Trial Collaboration** | 25% | Readiness for federated, regulatory-compliant deployment across clinical trial sites |

Each dimension derives from the four unification pillars defined in [`unification/`](../):
- Dimension A <- [`simulation_physics/`](../simulation_physics/)
- Dimension B <- [`agentic_generative_ai/`](../agentic_generative_ai/)
- Dimension C <- [`cross_platform_tools/`](../cross_platform_tools/) and [`surgical_robotics/`](../surgical_robotics/)
- Dimension D <- [`../../federation/`](../../federation/) and [`../../regulatory/`](../../regulatory/)

---

## USL Score Bands

| Score Range | Band | Description |
|-------------|------|-------------|
| 9.0 - 10.0 | **Exemplary** | Fully unified, multi-site clinical trial ready |
| 7.0 - 8.9 | **Advanced** | Strong unification, near clinical-trial ready |
| 5.0 - 6.9 | **Intermediate** | Partial unification, significant work remaining |
| 3.0 - 4.9 | **Foundational** | Basic interoperability, major gaps exist |
| 1.0 - 2.9 | **Initial** | Minimal unification capability |

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

## Robot Categories

The USL framework evaluates robots across multiple categories relevant to oncology clinical trials.

| Category | Robots Evaluated | Score Range | Status | Details |
|----------|-----------------|-------------|--------|---------|
| **Humanoid Robots** | Atlas (Electric), Optimus (Gen 2), Digit | 3.6 - 5.8 | v1.6.0 | [humanoids/](humanoids/) |
| **Surgical Robots** | da Vinci (dVRK), Hugo RAS, Versius | 3.4 - 7.1 | v1.5.0 | [surgical/](surgical/) |
| **Collaborative Robots (Cobots)** | Franka Panda, Kinova Gen3, xArm 7 | 3.4 - 7.4 | v1.4.0 | [cobots/](cobots/) |

---

## Directory Structure

```
unification/usl/
├── README.md                              # This file — USL standard overview
├── prompts.md                             # Development prompts archive
├── paper/                                 # ★ USL Paper (v1.8.0)
│   ├── Unification Standard Level for Physical AI Oncology Trials.pdf
│   ├── Latex Source Code.zip              # .tex, .sty, .bib, README
│   ├── usl_oncology_trials.tex            # Main LaTeX document
│   ├── usl-oncology.sty                   # Custom style package
│   ├── references.bib                     # BibTeX bibliography (28 refs)
│   └── README                             # Compilation instructions
├── humanoids/                             # Humanoid Robots category (v1.6.0)
│   ├── README.md                          # Humanoid evaluations, diagrams, results
│   ├── usl_humanoid_scoring.py            # USL scoring engine (humanoids)
│   ├── boston_dynamics_atlas/
│   │   └── boston_dynamics_atlas_usl.py    # Atlas Electric evaluation + tools
│   ├── tesla_optimus/
│   │   └── tesla_optimus_usl.py           # Optimus Gen 2 evaluation + tools
│   └── agility_digit/
│       └── agility_digit_usl.py           # Digit evaluation + tools
├── surgical/                              # Surgical Robots category (v1.5.0)
│   ├── README.md                          # Surgical evaluations, diagrams, results
│   ├── usl_surgical_scoring.py            # USL scoring engine (surgical)
│   ├── intuitive_davinci/
│   │   └── intuitive_davinci_usl.py       # da Vinci (dVRK) evaluation + tools
│   ├── medtronic_hugo/
│   │   └── medtronic_hugo_usl.py          # Hugo RAS evaluation + tools
│   └── cmr_versius/
│       └── cmr_versius_usl.py             # Versius evaluation + tools
└── cobots/                                # Collaborative Robots category (v1.4.0)
    ├── README.md                          # Cobot evaluations, diagrams, results
    ├── usl_scoring_framework.py           # Core USL scoring engine (cobots)
    ├── franka_panda/
    │   └── franka_panda_usl.py            # Franka Panda evaluation + tools
    ├── kinova_gen3/
    │   └── kinova_gen3_usl.py             # Kinova Gen3 evaluation + tools
    └── ufactory_xarm7/
        └── ufactory_xarm7_usl.py          # xArm 7 evaluation + tools
```

---

## Influences and References

The USL framework draws on established technology readiness methodologies:

1. **NASA/DOD Technology Readiness Levels (TRL)** — Mankins, J.C. (2004). *Technology Readiness Assessments: A Retrospective*. White Paper, NASA. Original 9-level TRL scale for evaluating technology maturity from basic principles (TRL 1) to flight-proven systems (TRL 9). USL adapts this graduated-maturity concept to robot unification readiness.

2. **ML Technology Readiness Levels (MLTRL)** — Lavin, A., et al. (2021). *Technology Readiness Levels for Machine Learning Systems*. GitHub: [ai-infrastructure-alliance/mltrl](https://github.com/ai-infrastructure-alliance/mltrl). Extends TRL to ML systems with levels 1-9 covering data readiness, model development, deployment, and monitoring. USL incorporates MLTRL's recognition that AI system readiness requires evaluating software, data, and integration dimensions beyond hardware alone.

3. **TRL for Complex System Integration** — Tomaschek, K., Olechowski, A., Eppinger, S., & Joglekar, N. (2015). *A Survey of Technology Readiness Level Users*. Proceedings of PICMET 2015. DOI: [10.1109/PICMET.2015.7273196](https://doi.org/10.1109/PICMET.2015.7273196). Identifies challenges in applying TRL to integrated multi-technology systems, directly relevant to evaluating robot systems that span simulation, AI, and clinical deployment.

4. **LLM Recommendations for Oncology Trials** — Kawchak, K. (2025). *Physical AI for Clinical Oncology Trials*. Zenodo. DOI: [10.5281/zenodo.17451709](https://doi.org/10.5281/zenodo.17451709). Recommends LLM usage for upcoming oncology trials and motivates the need for standardized evaluation of AI-integrated robotic systems in clinical settings. Inspiration for the USL standard.

### Additional References

- [Boston Dynamics Atlas Electric](https://bostondynamics.com/blog/electric-new-era-for-atlas/) — Next-gen electric humanoid
- [Drake](https://github.com/RobotLocomotion/drake) — Open-source simulation and planning (MIT/TRI)
- [Agility Robotics Digit](https://agilityrobotics.com/robots) — First commercial humanoid robot
- [NVIDIA GR00T N1](https://developer.nvidia.com/isaac/humanoid) — Humanoid foundation model
- [NVIDIA Isaac Lab 2.3.1](https://github.com/isaac-sim/IsaacLab) — GPU-accelerated robot learning
- [MuJoCo 3.4.0](https://github.com/google-deepmind/mujoco) — Physics simulation
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Curated robot models
- [ROS 2 Kilted Kaiju](https://docs.ros.org/en/kilted/) — Robot middleware
- [ORBIT-Surgical](https://github.com/orbit-surgical/orbit-surgical) — Surgical robot learning (Isaac Lab)
- [SurRoL](https://github.com/med-air/SurRoL) — Surgical robot RL benchmark
- [dVRK 2.4.0](https://github.com/jhu-dvrk/sawIntuitiveResearchKit) — da Vinci Research Kit
- [AMBF](https://github.com/WPI-AIM/ambf) — Asynchronous Multi-Body Framework
- [Model Context Protocol](https://modelcontextprotocol.io/) — Agent-tool communication (AAIF/Linux Foundation)
- ISO 13482:2014 — Robots and robotic devices — Safety requirements for personal care robots
- IEC 62304:2006+AMD1:2015 — Medical device software — Software life cycle processes
- IEC 80601-2-77:2019 — Particular requirements for robotically assisted surgical equipment
- FDA Guidance: *Marketing Submission Recommendations for a Predetermined Change Control Plan for AI/ML-Enabled Device Software Functions* (August 2025)

---

*Last updated: February 2026*
