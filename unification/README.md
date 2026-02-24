# Unification Framework for Physical AI Oncology Trials

*Multi-Organization Cooperation for Framework-Agnostic Physical AI Development (February 2026)*

---

## Unification Standard Level (USL)

The **[Unification Standard Level (USL)](usl/)** is a scoring framework for evaluating robot readiness for unified, multi-site oncology clinical trials. USL scores range from **1.0 to 10.0** across four weighted dimensions: simulation switching, AI integration, cross-robot sharing, and clinical trial collaboration. See [usl/README.md](usl/) for the full standard, and individual category READMEs for detailed evaluations and diagrams.

### Diagram 1: USL Results Summary — All Robot Categories

```
+-------------------------------------------------------------------------+
|              USL RESULTS SUMMARY — All Evaluated Robots                 |
+-------------------------------------------------------------------------+
|                                                                         |
|  Category: Humanoid Robots (v1.6.0)           Score Range: 3.6 - 5.8   |
|  ──────────────────────────────────────────                             |
|  Atlas (Electric)  [=====-----] 5.8  Best sim coverage (4 fwks, Drake) |
|  Digit             [====------] 4.2  GR00T N1 partner, Amazon deploy   |
|  Optimus (Gen 2)   [===-------] 3.6  Best hands (11-DOF), proprietary  |
|                                                                         |
|  Category: Surgical Robots (v1.5.0)           Score Range: 3.4 - 7.1   |
|  ──────────────────────────────────────────                             |
|  da Vinci (dVRK)   [=======---] 7.1  Open-source dVRK, ORBIT-Surgical |
|  Hugo RAS          [====------] 4.5  Medtronic ecosystem, CE marked    |
|  Versius           [===-------] 3.4  Lightest arms, most portable      |
|                                                                         |
|  Category: Cobots (v1.4.0)                    Score Range: 3.4 - 7.4   |
|  ──────────────────────────────────────────                             |
|  Franka Panda      [=======---] 7.4  Largest cobot research ecosystem  |
|  Kinova Gen3       [=====-----] 5.7  Lightest arm, integrated vision   |
|  xArm 7            [===-------] 3.4  Most affordable, IP51 rated       |
|                                                                         |
|  Bar scale: each = = 1.0 point (10 blocks = 10.0)                      |
+-------------------------------------------------------------------------+
```

### Diagram 2: USL Meaning — What Scores Reveal About Unification

```
+-------------------------------------------------------------------------+
|        USL MEANING — What Scores Reveal About Physical AI Unification   |
+-------------------------------------------------------------------------+
|                                                                         |
|  KEY FINDING 1: OPEN-SOURCE IS THE STRONGEST USL PREDICTOR              |
|  ──────────────────────────────────────────────────────────              |
|  The two highest-scoring robots (Franka Panda 7.4, da Vinci dVRK 7.1)  |
|  are also the two with the largest open-source ecosystems. Open code    |
|  enables more simulation frameworks (Dim A), more AI research (Dim B), |
|  and more cross-robot sharing (Dim C). Proprietary robots (Optimus,    |
|  Versius) consistently score lowest regardless of hardware quality.     |
|                                                                         |
|  KEY FINDING 2: CLINICAL READINESS LAGS ACROSS ALL CATEGORIES           |
|  ──────────────────────────────────────────────────────────              |
|  Dimension D (clinical trial collaboration) is the weakest dimension    |
|  for 7 of 9 evaluated robots. Even da Vinci (FDA cleared, 14M          |
|  procedures) only scores 7.0 on Dim D. Multi-site federated learning,  |
|  regulatory documentation, and audit trail infrastructure remain        |
|  underdeveloped across the field.                                       |
|                                                                         |
|  KEY FINDING 3: CATEGORY LEADERS DEFINE THE UNIFICATION FRONTIER        |
|  ──────────────────────────────────────────────────────────              |
|  Franka Panda (cobot), da Vinci dVRK (surgical), and Atlas (humanoid)  |
|  represent the current frontier of what is achievable in each category. |
|  Their scores (7.4, 7.1, 5.8) show that cobots and surgical robots     |
|  are closer to clinical trial readiness than humanoids.                 |
|                                                                         |
+-------------------------------------------------------------------------+
```

### Diagram 3: USL Impact — Future of Physical AI Oncology Trials

```
+-------------------------------------------------------------------------+
|    USL IMPACT — Shaping the Future of Physical AI Oncology Trials       |
+-------------------------------------------------------------------------+
|                                                                         |
|  PHASE 1 (2026): CATEGORY-SPECIFIC TRIALS                               |
|  ─────────────────────────────────────────                               |
|  Cobots for lab automation, surgical robots for MIS oncology,           |
|  humanoids for hospital logistics. Each category operates               |
|  independently with USL guiding platform selection per site.            |
|                                                                         |
|       Lab site ──[Franka 7.4]──> Sample handling trial                  |
|       OR site  ──[dVRK   7.1]──> Surgical autonomy trial               |
|       Ward     ──[Atlas  5.8]──> Logistics pilot                        |
|                                                                         |
|  PHASE 2 (2027): CROSS-CATEGORY INTEGRATION                             |
|  ───────────────────────────────────────────                             |
|  USL-standardized interfaces enable cobots and surgical robots to       |
|  share workspace data during procedures. Humanoids transport            |
|  specimens from OR to lab where cobots process them.                    |
|                                                                         |
|       OR [dVRK resection] --> Humanoid [specimen transport]             |
|                               --> Lab [Franka processing]               |
|                                                                         |
|  PHASE 3 (2028+): UNIFIED MULTI-SITE CONSORTIUM                         |
|  ────────────────────────────────────────────────                        |
|  Full multi-site trials where different hospitals use different          |
|  robot platforms but share policies via ONNX, coordinate via            |
|  federated learning, and report via unified audit trails.               |
|  USL ensures behavioral equivalence across heterogeneous sites.        |
|                                                                         |
|       Site A [Franka + dVRK]  <--federated-->  Site B [Kinova + Hugo]  |
|       Site C [xArm + Versius] <--federated-->  Site D [Atlas + dVRK]   |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## Overview

The Unification Framework enables seamless interoperability between core physical AI technologies for oncology clinical trials. This directory provides the tools, standards, and workflows necessary for research teams and organizations to:

- **Switch between simulation frameworks** (NVIDIA Isaac, MuJoCo, Gazebo, PyBullet) at any workflow stage
- **Integrate agentic and generative AI** across different robotic platforms
- **Share surgical robotics models** across organizations with standardized formats
- **Collaborate on multi-site clinical trials** with unified data and control interfaces

---

## Directory Structure

```
unification/
├── README.md                          # This file
├── simulation_physics/                # Simulation framework unification
│   ├── challenges.md                  # Technical barriers to cross-platform compatibility
│   ├── opportunities.md               # Potential benefits and pathways
│   ├── isaac_mujoco_bridge.py         # NVIDIA Isaac ↔ MuJoCo converter
│   ├── urdf_sdf_mjcf_converter.py     # Robot model format conversion
│   └── physics_parameter_mapping.yaml # Physics engine parameter equivalences
├── agentic_generative_ai/             # AI/ML framework unification
│   ├── challenges.md                  # Integration challenges across AI systems
│   ├── opportunities.md               # Unified AI orchestration potential
│   ├── unified_agent_interface.py     # Cross-framework agent abstraction
│   └── vla_model_adapter.py           # Vision-Language-Action model adapters
├── surgical_robotics/                 # Surgical robot platform unification
│   ├── challenges.md                  # Hardware/software compatibility issues
│   ├── opportunities.md               # Multi-organization cooperation benefits
│   ├── unified_surgical_api.py        # Standardized surgical robot control API
│   └── dvrk_orbit_bridge.py           # dVRK ↔ ORBIT-Surgical integration
├── cross_platform_tools/              # Conversion and validation utilities
│   ├── framework_detector.py          # Detect and validate framework installations
│   ├── model_converter.py             # Universal robot model converter
│   ├── policy_exporter.py             # Cross-framework policy export
│   └── validation_suite.py            # Cross-platform validation tests
├── usl/                               # ★ Unification Standard Level framework
│   ├── README.md                      # USL standard overview, scoring, influences
│   ├── prompts.md                     # Development prompts archive
│   ├── humanoids/                     # ★ Humanoid Robots category (v1.6.0)
│   │   ├── README.md                  # Humanoid evaluations, diagrams, results
│   │   ├── usl_humanoid_scoring.py    # USL scoring engine (humanoids)
│   │   ├── boston_dynamics_atlas/
│   │   │   └── boston_dynamics_atlas_usl.py  # Atlas Electric evaluation + tools
│   │   ├── tesla_optimus/
│   │   │   └── tesla_optimus_usl.py   # Optimus Gen 2 evaluation + tools
│   │   └── agility_digit/
│   │       └── agility_digit_usl.py   # Digit evaluation + tools
│   ├── surgical/                      # Surgical Robots category (v1.5.0)
│   │   ├── README.md                  # Surgical evaluations, diagrams, results
│   │   ├── usl_surgical_scoring.py    # USL scoring engine (surgical)
│   │   ├── intuitive_davinci/
│   │   │   └── intuitive_davinci_usl.py  # da Vinci (dVRK) evaluation + tools
│   │   ├── medtronic_hugo/
│   │   │   └── medtronic_hugo_usl.py  # Hugo RAS evaluation + tools
│   │   └── cmr_versius/
│   │       └── cmr_versius_usl.py     # Versius evaluation + tools
│   └── cobots/                        # Collaborative Robots category (v1.4.0)
│       ├── README.md                  # Cobot evaluations, diagrams, results
│       ├── usl_scoring_framework.py   # Core USL scoring engine (cobots)
│       ├── franka_panda/
│       │   └── franka_panda_usl.py    # Franka Panda evaluation + tools
│       ├── kinova_gen3/
│       │   └── kinova_gen3_usl.py     # Kinova Gen3 evaluation + tools
│       └── ufactory_xarm7/
│           └── ufactory_xarm7_usl.py  # xArm 7 evaluation + tools
├── standards_protocols/               # Interoperability standards
│   ├── data_formats.md                # Standardized data format specifications
│   ├── communication_protocols.md     # Inter-system communication standards
│   └── safety_standards.md            # Unified safety requirements
└── integration_workflows/             # End-to-end unified workflows
    ├── sim2real_unified.py            # Framework-agnostic sim-to-real pipeline
    ├── multi_site_coordination.py     # Multi-organization trial coordination
    └── workflow_templates.yaml        # Reusable workflow configurations
```

---

## Roadmap

### Q1 2026 ★ [Standards Available](../q1-2026-standards/)
- [ ] Complete Isaac ↔ MuJoCo bidirectional conversion → See [Objective 1](../q1-2026-standards/objective-1-bidirectional-conversion/)
- [ ] Publish unified robot model repository (50+ models) → See [Objective 2](../q1-2026-standards/objective-2-robot-model-repository/)
- [ ] Release validation benchmark suite v1.0 → See [Objective 3](../q1-2026-standards/objective-3-validation-benchmark/)

### Q1 2026 (USL Standard)
- [x] Establish Unification Standard Level (USL) scoring framework → See [usl/](usl/)
- [x] Evaluate 3 open-source cobots (Franka Panda, Kinova Gen3, xArm 7)
- [x] Extend USL to surgical robot category (da Vinci dVRK, Hugo RAS, Versius)
- [x] Extend USL to humanoid robot category (Atlas Electric, Optimus Gen 2, Digit)
- [ ] Extend USL to mobile manipulator category

### Q2 2026
- [ ] Integrate Gazebo Ionic GPU acceleration (when available)
- [ ] Add GR00T N1.6 ↔ OpenVLA ↔ π₀ model adapters
- [ ] Establish consortium data sharing infrastructure

### Q3 2026
- [ ] Multi-site clinical trial coordination platform
- [ ] Real-time policy switching during procedures
- [ ] FDA 510(k) pathway documentation for unified systems

### Q4 2026
- [ ] Production deployment at 3+ healthcare systems
- [ ] Open-source community governance establishment
- [ ] Annual consortium benchmark competition

---

## Core Principles

### 1. Framework Agnosticism

All trained policies, robot models, and data formats should be portable across supported frameworks:

| Source Framework | Target Frameworks | Conversion Fidelity |
|------------------|-------------------|---------------------|
| NVIDIA Isaac | MuJoCo, Gazebo, PyBullet | High (physics mapping required) |
| MuJoCo | Isaac, Gazebo, PyBullet | High |
| Gazebo | Isaac, MuJoCo, PyBullet | Medium-High |
| PyBullet | Isaac, MuJoCo, Gazebo | Medium |

### 2. Organization Neutrality

The framework supports collaboration across institutions without vendor lock-in:

- **Stanford/JHU**: ORBIT-Surgical, dVRK 2.4.0, dVRK-Si platforms
- **NVIDIA**: Isaac Lab 2.3.1, Isaac Lab-Arena, Isaac Sim 5.0.0, Isaac for Healthcare
- **Google DeepMind**: MuJoCo 3.4.0, MJX, MuJoCo Warp, Menagerie
- **Linux Foundation**: Newton Physics Engine (NVIDIA/DeepMind/Disney), Model Context Protocol (AAIF)
- **Open Robotics**: Gazebo Sim 10.0 (Jetty), ROS 2 Jazzy/Kilted
- **Research Community**: PyBullet 3.2.5, Gymnasium

### 3. Clinical Trial Compliance

All unified components maintain:

- FDA 21 CFR Part 11 audit trail capability
- ICH E6(R3) GCP compliance hooks (effective Sep 2025)
- ISO 13482 safety robot requirements
- IEC 62304 software lifecycle traceability

---

## Quick Start

### 1. Verify Framework Availability

```bash
python unification/cross_platform_tools/framework_detector.py
```

### 2. Convert Robot Model

```python
from unification.cross_platform_tools.model_converter import UnifiedModelConverter

converter = UnifiedModelConverter()

# Convert URDF to all formats
converter.convert(
    source_path="robots/surgical_arm.urdf",
    source_format="urdf",
    target_formats=["mjcf", "sdf", "usd"],
    output_dir="robots/converted/"
)
```

### 3. Export Policy Across Frameworks

```python
from unification.cross_platform_tools.policy_exporter import UnifiedPolicyExporter

exporter = UnifiedPolicyExporter()

# Export Isaac-trained policy for MuJoCo deployment
exporter.export(
    policy_path="checkpoints/needle_insertion_isaac.pt",
    source_framework="isaac",
    target_frameworks=["mujoco", "pybullet"],
    output_dir="policies/exported/"
)
```

### 4. Validate Cross-Platform Consistency

```python
from unification.cross_platform_tools.validation_suite import CrossPlatformValidator

validator = CrossPlatformValidator()

# Test policy behavior across frameworks
results = validator.validate_policy(
    policy_path="policies/surgical_policy.onnx",
    test_scenarios=["needle_insertion", "tissue_retraction"],
    frameworks=["isaac", "mujoco", "pybullet"],
    tolerance=0.05  # 5% performance variance allowed
)

validator.generate_report(results, "validation_report.html")
```

---

## Multi-Organization Cooperation Model

### Consortium Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Physical AI Oncology Consortium              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Academic    │  │   Industry   │  │  Healthcare  │          │
│  │  Partners     │  │   Partners   │  │   Systems    │          │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤          │
│  │ Stanford     │  │ NVIDIA       │  │ Mayo Clinic  │          │
│  │ JHU          │  │ Intuitive    │  │ MD Anderson  │          │
│  │ MIT          │  │ Medtronic    │  │ Memorial     │          │
│  │ Berkeley     │  │ Stryker      │  │ Cleveland    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Unified Data & Model Repository             │   │
│  │  • Standardized robot models (URDF/MJCF/SDF/USD)        │   │
│  │  • Trained policy checkpoints (ONNX)                     │   │
│  │  • Clinical trial datasets (anonymized)                  │   │
│  │  • Benchmark environments                                │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Sharing Agreement Template

Organizations participating in unified oncology trials should establish:

1. **Model Sharing**: Open-source robot models in standardized formats
2. **Policy Sharing**: Trained policies with documented training conditions
3. **Data Sharing**: Anonymized trial data with HIPAA/GDPR compliance → See [`privacy/`](../privacy/) for de-identification and DUA tools
4. **Benchmark Sharing**: Standardized evaluation environments and metrics
5. **Regulatory Compliance**: FDA/IRB/ICH-GCP documentation → See [`regulatory/`](../regulatory/) for compliance tools

---

## Framework Compatibility Matrix

### Simulation Features

| Feature | Isaac Lab | MuJoCo | Gazebo Ionic | PyBullet |
|---------|-----------|--------|--------------|----------|
| GPU Parallel Sim | ✓ (4096+) | ✓ (MJX) | ✗ | ✗ |
| Soft Body | ✓ | ✓ (composite) | Limited | ✓ |
| Ray Tracing | ✓ | ✗ | ✗ | ✗ |
| ROS 2 Native | ✓ | Via bridge | ✓ | Via bridge |
| ONNX Deploy | ✓ | ✓ | ✓ | ✓ |
| Deformable Tissue | ✓ | ✓ | Limited | Limited |
| Force Sensing | ✓ | ✓ | ✓ | ✓ |

### Recommended Use Cases

| Use Case | Primary Framework | Fallback | Rationale |
|----------|-------------------|----------|-----------|
| High-throughput RL training | Isaac Lab | MuJoCo MJX | GPU parallelization |
| Physics accuracy validation | MuJoCo | PyBullet | Reference dynamics |
| ROS 2 integration testing | Gazebo Ionic | Isaac | Native ROS support |
| Rapid prototyping | PyBullet | MuJoCo | Easy setup |
| Production deployment | Isaac + ROS 2 | Gazebo + ROS 2 | Clinical robustness |

---

## Getting Started by Role

### For Researchers (Academic Institutions)

1. Clone repository and install base requirements
2. Use `model_converter.py` to import existing robot models
3. Train in your preferred framework (Isaac recommended for speed)
4. Export policies to ONNX for cross-framework validation
5. Share results via standardized benchmark submissions

### For Engineers (Industry Partners)

1. Integrate `unified_surgical_api.py` with existing robot platforms
2. Use `framework_detector.py` to verify deployment environment
3. Apply `policy_exporter.py` for production deployment
4. Validate with `validation_suite.py` before clinical use

### For Clinicians (Healthcare Systems)

1. Review `safety_standards.md` for compliance requirements
2. Use `multi_site_coordination.py` for trial management
3. Access standardized reporting via workflow templates
4. Provide feedback on clinical utility metrics

---

## Contributing

Contributions to the unification framework are welcome from all organizations. Please ensure:

1. Cross-platform compatibility for all code contributions
2. Documentation with clinical trial context
3. Validation across at least 2 frameworks
4. Compliance with safety standards

See individual subdirectory READMEs for specific contribution guidelines.

---

## References

- [NVIDIA Isaac Lab 2.3.1](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.1)
- [NVIDIA Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLabArena)
- [NVIDIA Isaac Sim 5.0.0](https://docs.isaacsim.omniverse.nvidia.com/)
- [Newton Physics Engine](https://github.com/newton-physics/newton) - Linux Foundation
- [MuJoCo 3.4.0](https://github.com/google-deepmind/mujoco/releases/tag/3.4.0)
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [Gazebo Sim (Jetty)](https://gazebosim.org/docs/jetty/)
- [ORBIT-Surgical](https://orbit-surgical.github.io/)
- [dVRK 2.4.0](https://github.com/jhu-dvrk/sawIntuitiveResearchKit)
- [ROS 2 Kilted Kaiju](https://docs.ros.org/en/kilted/)
- [CrewAI 1.6.1](https://github.com/crewAIInc/crewAI)
- [LangGraph 1.1.0](https://github.com/langchain-ai/langgraph)
- [Model Context Protocol](https://modelcontextprotocol.io/) - AAIF/Linux Foundation

---

*Last updated: February 2026*
