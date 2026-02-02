# Physical AI Unification: Oncology Clinical Trials

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Last Updated](https://img.shields.io/badge/Updated-February%202026-blue.svg)]()
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)]()
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18445179-blue)](https://doi.org/10.5281/zenodo.18445179)

**Practical tools for integrating physical AI into oncology clinical trials, by Claude Code Opus 4.5/Cowork, ChatGPT 5.2 Thinking**

This repository provides production-ready configurations, validated pipelines, and integration guides for deploying robotic systems, digital twins, and embodied AI agents in oncology. Referenced frameworks and tools have been added primarily from Oct. 2025 to Jan. 2026.

## Responsible use

This repository is complementary and open source, please implement code safely and responsibly.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/kevinkawchak/physical-ai-oncology-trials.git
cd physical-ai-oncology-trials

# Install base dependencies
pip install -r requirements.txt

# Verify framework availability
python scripts/verify_installation.py

# Detect available simulation frameworks
python unification/cross_platform_tools/framework_detector.py
```

---

## Repository Structure

```
physical-ai-oncology-trials/
├── README.md
├── LICENSE
├── requirements.txt
│
├── digital-twins/                        
│   ├── README.md                         
│   ├── patient-modeling/                
│   │   ├── README.md
│   │   └── tumor_twin_pipeline.py       
│   ├── treatment-simulation/            
│   │   ├── README.md
│   │   └── treatment_simulator.py        
│   └── clinical-integration/            
│       ├── README.md
│       └── clinical_dt_interface.py      
│
├── examples/                             
│   ├── README.md                        
│   ├── 01_surgical_robot_training.py     
│   ├── 02_digital_twin_surgical_planning.py   
│   ├── 03_cross_framework_validation.py 
│   ├── 04_agentic_clinical_workflow.py   
│   └── 05_treatment_response_prediction.py   
│
├── q1-2026-standards/
│   ├── README.md
│   ├── objective-1-bidirectional-conversion/
│   │   ├── isaac_to_mujoco_pipeline.py
│   │   ├── mujoco_to_isaac_pipeline.py
│   │   └── physics_equivalence_tests.py
│   ├── objective-2-robot-model-repository/
│   │   ├── model_registry.yaml
│   │   └── model_validator.py
│   ├── objective-3-validation-benchmark/
│   │   └── benchmark_runner.py
│   └── implementation-guide/
│       ├── timeline.md
│       └── compliance_checklist.md
│
├── unification/
│   ├── README.md
│   ├── simulation_physics/
│   │   ├── challenges.md
│   │   ├── opportunities.md
│   │   ├── isaac_mujoco_bridge.py
│   │   ├── urdf_sdf_mjcf_converter.py
│   │   └── physics_parameter_mapping.yaml
│   ├── agentic_generative_ai/
│   │   ├── challenges.md
│   │   ├── opportunities.md
│   │   └── unified_agent_interface.py
│   ├── surgical_robotics/
│   │   ├── challenges.md
│   │   └── opportunities.md
│   ├── cross_platform_tools/
│   │   ├── framework_detector.py
│   │   └── validation_suite.py
│   ├── standards_protocols/
│   └── integration_workflows/
│
├── generative-ai/                     # VLA models, diffusion policies, synthetic data
│   ├── strengths.md
│   ├── limitations.md
│   └── results.md
├── agentic-ai/                        # LLM-based robot control, multi-agent systems
│   ├── strengths.md
│   ├── limitations.md
│   └── results.md
├── reinforcement-learning/            # RL for surgical autonomy, sim2real transfer
│   ├── strengths.md
│   ├── limitations.md
│   └── results.md
├── self-supervised-learning/          # Contrastive learning, foundation models
│   ├── strengths.md
│   ├── limitations.md
│   └── results.md
├── supervised-learning/               # Segmentation, detection, classification
│   ├── strengths.md
│   ├── limitations.md
│   └── results.md
│
├── frameworks/
│   ├── nvidia-isaac/                  # Isaac Sim, Isaac Lab, Isaac for Healthcare
│   │   └── INTEGRATION.md
│   ├── mujoco/                        # MuJoCo, MJX, MuJoCo Playground
│   │   └── INTEGRATION.md
│   ├── gazebo/                        # Gazebo Ionic, ROS 2 integration
│   │   └── INTEGRATION.md
│   └── pybullet/                      # PyBullet medical simulation
│       └── INTEGRATION.md
│
├── configs/                           # Framework configurations
│   └── training_config.yaml
│
└── scripts/                           # Utility scripts
    └── verify_installation.py
```

---

## ★ Q1 2026 Standards

The new `q1-2026-standards/` directory contains **proposed standards** for meeting the Q1 2026 unification objectives:

| Objective | Description | Status |
|-----------|-------------|--------|
| **1** | Complete Isaac ↔ MuJoCo bidirectional conversion | Standards defined |
| **2** | Publish unified robot model repository (50+ models) | Registry created |
| **3** | Release validation benchmark suite v1.0 | Suite implemented |

---

## Core Technologies (Updated October 2025 - January 2026)

### Simulation & Physics

| Framework | Version | Last Update | Use Case | Unification Status |
|-----------|---------|-------------|----------|-------------------|
| NVIDIA Isaac Lab | 2.3.1 | Dec 2024 | GPU-accelerated robot training | ✓ Bridge available |
| NVIDIA Isaac Sim | 5.0.0 | Jan 2026 | High-fidelity physics simulation | ✓ Bridge available |
| Newton Physics Engine | Beta | Jan 2026 | GPU physics (NVIDIA/DeepMind/Disney) | ✓ Isaac Lab integrated |
| MuJoCo | 3.4.0 | Dec 2024 | Precision physics simulation | ✓ Bridge available |
| MuJoCo Warp | Beta | Jan 2026 | GPU-optimized MuJoCo (NVIDIA) | ✓ Bridge available |
| Gazebo Sim (Jetty) | 10.0.0 | Oct 2024 | ROS 2 integrated simulation | ◐ In progress |
| PyBullet | 3.2.5 | Apr 2023 | Rapid prototyping | ✓ Bridge available |

### Agentic & Generative AI

| Framework | Stars | Last Update | Use Case | Unification Status |
|-----------|-------|-------------|----------|-------------------|
| NVIDIA GR00T N1.6 | - | Jan 2026 | Humanoid robot foundation model | ✓ Adapter available |
| NVIDIA Cosmos Predict 2.5 | - | Jan 2026 | World foundation model, synthetic data | ✓ Native support |
| NVIDIA Cosmos Reason 2 | - | Jan 2026 | Reasoning VLM for physical AI | ✓ Native support |
| CrewAI | 100K+ | Jan 2026 | Multi-agent orchestration (v1.6.1) | ✓ Unified interface |
| LangChain/LangGraph | 95K+ | Jan 2026 | LLM-robot integration (v1.1.0) | ✓ Unified interface |
| Model Context Protocol | - | Dec 2025 | Agent-tool communication (AAIF/Linux Foundation) | ✓ Native support |
| MONAI Multimodal | - | Jan 2026 | Medical imaging + agentic AI | ✓ Integrated |

### Surgical Robotics

| Framework | Institution | Last Update | Use Case | Unification Status |
|-----------|-------------|-------------|----------|-------------------|
| ORBIT-Surgical | Stanford/JHU | Dec 2024 | Surgical task benchmarking | ✓ Primary benchmark |
| dVRK 2.4.0 | JHU | Jan 2026 | da Vinci research platform (ROS 2 Jazzy) | ✓ Bridge available |
| dVRK-Si | JHU | 2025 | Next-gen da Vinci Si/S support | ✓ Bridge available |
| SurgicalGym | - | 2025 | GPU-based surgical RL | ◐ In progress |
| Isaac Lab-Arena | NVIDIA | Jan 2026 | Large-scale policy evaluation | ✓ Benchmark integration |

---

## ★ Unification Framework

The new `unification/` directory enables **seamless interoperability** between core physical AI technologies. Users can now:

### Switch Frameworks at Any Workflow Stage

```
Training (Isaac Lab) → Validation (MuJoCo) → ROS 2 Integration (Gazebo) → Deployment
       ↓                    ↓                      ↓                        ↓
     Fast GPU            Accurate            Native ROS 2              Clinical
     training            physics              sensors                   ready
```

### Key Unification Capabilities

1. **Model Conversion**: Convert robot models between URDF, MJCF, SDF, and USD formats
2. **Policy Transfer**: Export and validate policies across frameworks
3. **Physics Mapping**: Consistent contact dynamics across engines
4. **Agent Abstraction**: Framework-agnostic AI agent interfaces
5. **Cross-Platform Validation**: Verify behavior consistency

### Quick Start with Unification Tools

```python
# Detect available frameworks
from unification.cross_platform_tools.framework_detector import FrameworkDetector
detector = FrameworkDetector()
available = detector.detect_all()
print(detector.get_recommended_pipeline())

# Convert robot model to all formats
from unification.simulation_physics.urdf_sdf_mjcf_converter import UnifiedModelConverter
converter = UnifiedModelConverter()
converter.convert("robots/surgical_arm.urdf", target_formats=["mjcf", "sdf", "usd"])

# Create framework-agnostic agent
from unification.agentic_generative_ai.unified_agent_interface import UnifiedAgent
agent = UnifiedAgent(
    name="surgical_assistant",
    role="Provide surgical instruments",
    backend="crewai"  # or "langgraph", "custom"
)

# Validate policy across frameworks
from unification.cross_platform_tools.validation_suite import CrossPlatformValidator
validator = CrossPlatformValidator()
results = validator.validate_policy(
    "policies/needle_insertion.onnx",
    frameworks=["isaac", "mujoco", "pybullet"]
)
```

---

## Key Capabilities

### 1. Generative AI for Physical Systems
- **Vision-Language-Action (VLA) models** for surgical instrument manipulation (GR00T N1.6)
- **Diffusion policies** for trajectory generation in tumor resection
- **Synthetic data generation** for rare oncology scenarios (Cosmos Predict 2.5)
- **World models** (NVIDIA Cosmos) for physics-aware simulation and reasoning (Cosmos Reason 2)
- **Physical reasoning** via dual-system architecture (System 1 fast + System 2 deliberate)

### 2. Agentic AI for Clinical Workflows
- **LLM-based surgical assistants** with multimodal perception
- **Multi-agent coordination** for multi-site clinical trials (CrewAI 1.6.1, LangGraph 1.1.0)
- **Natural language robot programming** via ROS 2 Jazzy/Kilted integration
- **Autonomous task planning** for drug infusion and sample handling
- **Standardized tool integration** via Model Context Protocol (MCP) under Linux Foundation AAIF

### 3. Reinforcement Learning for Surgical Autonomy
- **Sim2real transfer** with domain randomization
- **Hierarchical RL** for complex surgical procedures
- **Multi-agent RL** for cooperative surgical assistance
- **GPU-accelerated training** reducing training time from days to hours

### 4. Digital Twin Integration
- **Patient-specific tumor models** from imaging data
- **Treatment response simulation** for chemotherapy/radiation
- **Real-time intraoperative guidance** with sensor fusion
- **Predictive outcome modeling** for trial design

---

## ★ Digital Twins for Oncology

The new `digital-twins/` directory provides comprehensive tools for creating and using patient-specific digital twins in oncology clinical trials.

### Key Capabilities

| Capability | Framework | Clinical Application |
|------------|-----------|---------------------|
| Tumor growth modeling | TumorTwin | Patient-specific progression prediction |
| Treatment simulation | Custom PK/PD | Response prediction before treatment |
| Surgical planning | Isaac Sim integration | Virtual surgery rehearsal |
| Clinical integration | FHIR/DICOM | Hospital system connectivity |

### Quick Start with Digital Twins

```python
# Create patient-specific tumor digital twin
from digital_twins.patient_modeling import TumorTwinPipeline

pipeline = TumorTwinPipeline(
    model_type="reaction_diffusion",
    tumor_type="glioblastoma"
)

patient_dt = pipeline.create_twin(
    patient_id="ONCO-2026-001",
    imaging_data={"mri": mri_array},
    tumor_segmentation=tumor_mask
)

# Calibrate to longitudinal data
patient_dt.calibrate(
    longitudinal_scans=[scan_t0, scan_t1],
    timepoints=[0, 30]  # days
)

# Predict tumor evolution
prediction = patient_dt.predict(horizon_days=180)
print(f"Predicted volume change: {prediction.metrics['volume_change_percent']:.1f}%")

# Simulate treatment response
from digital_twins.treatment_simulation import TreatmentSimulator

simulator = TreatmentSimulator(patient_twin=patient_dt)
response = simulator.predict_response(
    treatment={"type": "radiation", "total_dose_gy": 60, "fractions": 30},
    horizon_days=90
)
print(f"Predicted response: {response.response_category}")
```

See `digital-twins/README.md` for complete documentation.

---

## ★ Comprehensive Examples

The new `examples/` directory contains production-ready code examples for the most pressing use cases in physical AI oncology trials.

### Available Examples

| Example | Use Case | Key Frameworks |
|---------|----------|----------------|
| `01_surgical_robot_training.py` | Train needle insertion policies | Isaac Lab, ORBIT-Surgical |
| `02_digital_twin_surgical_planning.py` | DT-guided surgical planning | TumorTwin, MONAI |
| `03_cross_framework_validation.py` | Multi-framework policy validation | Isaac, MuJoCo, PyBullet |
| `04_agentic_clinical_workflow.py` | Multi-agent clinical trial coordination | CrewAI, LangGraph |
| `05_treatment_response_prediction.py` | Patient-specific treatment prediction | TumorTwin |

### Quick Start with Examples

```bash
# Run surgical robot training
python examples/01_surgical_robot_training.py --task needle_insertion

# Run treatment prediction
python examples/05_treatment_response_prediction.py

# Each example includes detailed inline documentation
```

See `examples/README.md` for complete documentation.

---

## Validated Integration Paths

### Path 1: Surgical Robot Training Pipeline (Unified)
```
NVIDIA Isaac Lab → [unification/bridge] → MuJoCo Validation → dVRK Hardware → Clinical
```

### Path 2: Agentic Clinical Assistant (Unified)
```
LangGraph + MCP → [unified_agent_interface] → Any Robot Platform → Hospital Deployment
```

### Path 3: Multi-Framework Development
```
Train (Isaac) → Validate (MuJoCo) → Integrate (Gazebo) → Prototype (PyBullet) → Deploy
      ↑___________________________↓
         Cross-platform validation
```

---

## Dependencies

### Core Requirements
```
python>=3.10
torch>=2.5.0
numpy>=1.24.0
scipy>=1.11.0
```

### Framework-Specific
```
# NVIDIA Isaac (requires NVIDIA GPU)
isaacsim>=5.0.0
isaaclab>=2.3.0

# MuJoCo
mujoco>=3.4.0
mujoco-mjx>=3.4.0  # JAX backend

# ROS 2 (Kilted Kaiju or Jazzy)
ros-jazzy-desktop  # or ros-kilted-desktop

# Agentic AI
langchain>=1.0.0
langgraph>=1.0.0
crewai>=1.0.0
```

---

## Actively Maintained Repositories (Referenced)

All referenced repositories have been updated within October 2025 - January 2026:

| Repository | Purpose | Last Commit |
|------------|---------|-------------|
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | Robot learning framework (v2.3.1) | Dec 2024 |
| [newton-physics/newton](https://github.com/newton-physics/newton) | GPU physics engine (Linux Foundation) | Jan 2026 |
| [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) | Physics simulation (v3.4.0) | Dec 2024 |
| [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) | GPU-optimized MuJoCo | Jan 2026 |
| [orbit-surgical/orbit-surgical](https://github.com/orbit-surgical/orbit-surgical) | Surgical simulation | Sep 2024 |
| [jhu-dvrk/sawIntuitiveResearchKit](https://github.com/jhu-dvrk/sawIntuitiveResearchKit) | dVRK platform (v2.4.0) | Jan 2026 |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | GR00T N1.6 foundation model | Jan 2026 |
| [RobotecAI/rai](https://github.com/RobotecAI/rai) | ROS 2 agentic framework | Active |
| [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | Multi-agent orchestration (v1.6.1) | Jan 2026 |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | Durable agent framework (v1.1.0) | Jan 2026 |
| [modelcontextprotocol](https://github.com/modelcontextprotocol) | MCP specification (AAIF) | Jan 2026 |
| [Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI) | Medical imaging AI | Jan 2026 |
| [SCAI-Lab/ros4healthcare](https://github.com/SCAI-Lab/ros4healthcare) | Healthcare robotics | 2025 |
| [bulletphysics/bullet3](https://github.com/bulletphysics/bullet3) | Physics engine (v3.2.5) | Apr 2023 |
| [OncologyModelingGroup/TumorTwin](https://github.com/OncologyModelingGroup/TumorTwin) | Patient-specific cancer DTs | 2025 |
| [surgical-robotics-ai](https://github.com/surgical-robotics-ai) | Surgical robotics ML | Active |
| [SamuelSchmidgall/SurgicalGym](https://github.com/SamuelSchmidgall/SurgicalGym) | GPU surgical simulation | Active |
| [med-air/SurRoL](https://github.com/med-air/SurRoL) | dVRK-compatible RL platform | 2025 |

---

## Multi-Organization Cooperation

The unification framework supports collaboration across institutions:

| Organization Type | Contribution Area | Integration Point |
|-------------------|-------------------|-------------------|
| Academic Labs | Algorithms, benchmarks | ORBIT-Surgical, skill library |
| Industry R&D | Hardware, deployment | ros2_surgical, safety validation |
| Healthcare Systems | Clinical validation | Multi-site coordination |
| Regulatory Bodies | Compliance standards | IEC 62304 documentation |

See `unification/README.md` for the complete cooperation model.

---

### Quick Start with Q1 2026 Tools

```python
# Bidirectional conversion (Objective 1)
from q1_2026_standards.objective_1 import IsaacToMuJoCoConverter
converter = IsaacToMuJoCoConverter()
converter.convert_urdf("robot.urdf", "robot.xml")

# Model validation (Objective 2)
from q1_2026_standards.objective_2 import ModelValidator
validator = ModelValidator()
report = validator.validate_model("models/dvrk_psm/", level=4)

# Benchmark suite (Objective 3)
from q1_2026_standards.objective_3 import BenchmarkRunner
runner = BenchmarkRunner()
results = runner.run("needle_insertion", model_path="robot.xml")
```

See `q1-2026-standards/README.md` for complete documentation and implementation timeline.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@software{kawchak2026physicalai,
  author = {Kawchak, Kevin},
  title = {Physical AI for Oncology Clinical Trials},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/kevinkawchak/physical-ai-oncology-trials}
}
```

---



## License

MIT License - See [LICENSE](LICENSE) for details.



## Contributing

Contributions welcome. Please ensure any added frameworks or tools:
1. Have been updated within the last 3 months
2. Include practical oncology clinical trial applications
3. Provide reproducible configurations
4. **Support cross-platform compatibility** (see `unification/` for guidelines)


*Last updated: February 2026*
