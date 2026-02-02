# Objective 2: Unified Robot Model Repository

*Standards for Publishing 50+ Validated Robot Models (Q1 2026)*

**Status**: Active Development
**Target Completion**: Q1 2026
**Last Updated**: February 2026

---

## Overview

This directory defines standards and specifications for the Unified Robot Model Repository, targeting **50+ validated robot models** in multiple formats. The repository enables cross-organization sharing of robot models for oncology clinical trial applications.

### Key Requirements

1. **Multi-Format Support**: Each model available in URDF, MJCF, USD, and SDF
2. **Validation**: All models validated across Isaac Lab and MuJoCo
3. **Oncology Focus**: Priority for surgical, clinical, and research robots
4. **Documentation**: Comprehensive specifications for each model

---

## Target Model Categories

| Category | Target Count | Priority | Notes |
|----------|--------------|----------|-------|
| **Oncology Robots** | 15+ | High | Surgical, radiotherapy, brachytherapy |
| **Surgical Instruments** | 20+ | High | Grippers, scalpels, needle drivers |
| **General Manipulators** | 15+ | Medium | Collaborative robots, research arms |
| **Total** | **50+** | - | All clinical-grade validated |

---

## Model Specification Standard

### Required Formats per Model

Each model in the repository MUST be provided in:

| Format | Extension | Primary Use | Required |
|--------|-----------|-------------|----------|
| URDF | `.urdf` | ROS, general robotics | Yes |
| MJCF | `.xml` | MuJoCo simulation | Yes |
| USD | `.usd/.usda` | Isaac Sim, Omniverse | Yes |
| SDF | `.sdf` | Gazebo simulation | Recommended |

### Required Metadata

Each model must include a `model.yaml` metadata file:

```yaml
# model.yaml - Required metadata for each robot model
name: "Robot Name"
version: "1.0.0"
category: "oncology-robots"  # or "surgical-instruments", "general-manipulators"
description: "Brief description of the robot and its use case"

# Kinematic properties
kinematics:
  dof: 7                    # Degrees of freedom
  base_type: "fixed"        # or "mobile", "floating"
  joint_types: ["revolute", "revolute", ...]
  workspace_volume: 0.5     # cubic meters (approximate)

# Physical properties
physical:
  mass: 10.5                # kg (total)
  payload: 5.0              # kg (maximum)
  reach: 0.8                # meters

# Validated formats
formats:
  urdf:
    path: "robot.urdf"
    validated: true
    validation_date: "2026-01-15"
  mjcf:
    path: "robot.xml"
    validated: true
    validation_date: "2026-01-15"
  usd:
    path: "robot.usd"
    validated: true
    validation_date: "2026-01-15"
  sdf:
    path: "robot.sdf"
    validated: false

# Validation status
validation:
  isaac_lab: true
  mujoco: true
  cross_framework: true
  last_validated: "2026-01-15"

# Source and licensing
source:
  origin: "Original CAD"      # or "Derived from X", "Community contribution"
  license: "Apache-2.0"       # or "MIT", "BSD-3-Clause", "CC-BY-4.0"
  manufacturer: "Manufacturer Name"
  references:
    - "https://example.com/robot-docs"
    - "DOI:10.1234/example"

# Clinical application
clinical:
  fda_clearance: false
  intended_use: "Research only"
  oncology_applications:
    - "Surgical assistance"
    - "Biopsy guidance"

# Contributors
contributors:
  - name: "Contributor Name"
    organization: "Organization"
    role: "Model creation"
```

---

## Repository Structure

```
robot-model-repository/
├── README.md
├── model_registry.yaml           # Index of all models
├── model_validator.py            # Validation tool
│
├── oncology-robots/              # Category: Oncology-specific
│   ├── README.md
│   ├── dvrk-psm/                 # da Vinci Research Kit PSM
│   │   ├── model.yaml
│   │   ├── dvrk_psm.urdf
│   │   ├── dvrk_psm.xml
│   │   ├── dvrk_psm.usd
│   │   ├── dvrk_psm.sdf
│   │   └── meshes/
│   ├── cyberknife-arm/
│   ├── proton-therapy-gantry/
│   └── ...
│
├── surgical-instruments/         # Category: Surgical tools
│   ├── README.md
│   ├── needle-driver/
│   ├── laparoscopic-grasper/
│   ├── cautery-hook/
│   └── ...
│
└── general-manipulators/         # Category: General-purpose
    ├── README.md
    ├── franka-panda/
    ├── ur5/
    ├── kuka-iiwa/
    └── ...
```

---

## Model Categories

### 1. Oncology Robots (15+ models)

Robots specifically designed or adapted for oncology applications:

| Model | DOF | Type | Status | Source |
|-------|-----|------|--------|--------|
| dVRK PSM (da Vinci) | 7+1 | Surgical | Planned | [JHU dVRK](https://github.com/jhu-dvrk/sawIntuitiveResearchKit) |
| dVRK ECM | 4 | Surgical | Planned | [JHU dVRK](https://github.com/jhu-dvrk/sawIntuitiveResearchKit) |
| CyberKnife Arm | 6 | Radiotherapy | Planned | Research collaboration |
| Proton Therapy Gantry | 3 | Radiotherapy | Planned | Research collaboration |
| STAR Robot | 7 | Surgical | Planned | [STAR](https://arxiv.org/abs/2106.06854) |
| Brachytherapy Needle | 3 | Brachytherapy | Planned | Research |
| UR5 Surgical Adapter | 6 | Surgical assist | Planned | [UR Robots](https://www.universal-robots.com/) |
| RAVEN II | 7 | Surgical | Planned | [RAVEN](https://github.com/raven-debridement/raven_2) |
| KUKA LBR Med | 7 | Medical | Planned | [KUKA Medical](https://www.kuka.com/en-de/industries/health-care) |
| Neuromate | 5 | Neurosurgery | Planned | Research |
| iARM | 6 | Assistive | Planned | Research |
| Mako RIO | 6 | Orthopedic | Planned | Research |
| Rosa Brain | 5 | Neurosurgery | Planned | Research |
| ExcelsiusGPS | 6 | Spine surgery | Planned | Research |
| Mazor X | 6 | Spine surgery | Planned | Research |

### 2. Surgical Instruments (20+ models)

End-effectors and instruments for surgical procedures:

| Model | Type | Status | Notes |
|-------|------|--------|-------|
| Large Needle Driver | Grasper | Planned | Standard surgical needle driving |
| Maryland Bipolar Forceps | Grasper | Planned | Tissue grasping and coagulation |
| Prograsp Forceps | Grasper | Planned | Strong tissue holding |
| Cadiere Forceps | Grasper | Planned | Delicate tissue manipulation |
| Fenestrated Bipolar | Grasper | Planned | Coagulation with fenestration |
| Hot Shears | Scissors | Planned | Monopolar cutting/coagulation |
| Curved Scissors | Scissors | Planned | Precise tissue cutting |
| Cautery Hook | Energy | Planned | Monopolar cauterization |
| Suction/Irrigator | Suction | Planned | Fluid management |
| Clip Applier | Applicator | Planned | Hemostatic clip application |
| Stapler | Applicator | Planned | Tissue stapling |
| Vessel Sealer | Energy | Planned | LigaSure-type sealing |
| Biopsy Forceps | Grasper | Planned | Tissue sampling |
| Retractor | Retraction | Planned | Tissue retraction |
| Trocar | Access | Planned | Port placement |
| Camera Port | Access | Planned | Endoscope access |
| Suture Needle (curved) | Consumable | Planned | Standard suture needle |
| Suture Needle (straight) | Consumable | Planned | Straight suture needle |
| Drain Tube | Consumable | Planned | Surgical drainage |
| Specimen Bag | Consumable | Planned | Tissue extraction |

### 3. General Manipulators (15+ models)

General-purpose robots adaptable for clinical use:

| Model | DOF | Manufacturer | Status | Source |
|-------|-----|--------------|--------|--------|
| Franka Emika Panda | 7 | Franka | Planned | [Franka ROS](https://github.com/frankaemika/franka_ros) |
| Universal Robots UR3e | 6 | UR | Planned | [UR Description](https://github.com/ros-industrial/universal_robot) |
| Universal Robots UR5e | 6 | UR | Planned | [UR Description](https://github.com/ros-industrial/universal_robot) |
| Universal Robots UR10e | 6 | UR | Planned | [UR Description](https://github.com/ros-industrial/universal_robot) |
| KUKA iiwa 7 | 7 | KUKA | Planned | [iiwa Stack](https://github.com/IFL-CAMP/iiwa_stack) |
| KUKA iiwa 14 | 7 | KUKA | Planned | [iiwa Stack](https://github.com/IFL-CAMP/iiwa_stack) |
| Sawyer | 7 | Rethink | Planned | [Sawyer Robot](https://github.com/RethinkRobotics/sawyer_robot) |
| Kinova Gen3 | 7 | Kinova | Planned | [Kinova ROS](https://github.com/Kinovarobotics/ros_kortex) |
| Kinova Gen3 Lite | 6 | Kinova | Planned | [Kinova ROS](https://github.com/Kinovarobotics/ros_kortex) |
| ABB YuMi (single arm) | 7 | ABB | Planned | Research |
| Denso VS-060 | 6 | Denso | Planned | Research |
| Fanuc LR Mate 200iD | 6 | Fanuc | Planned | Research |
| Motoman GP7 | 6 | Yaskawa | Planned | Research |
| Doosan M1013 | 6 | Doosan | Planned | [Doosan ROS](https://github.com/doosan-robotics/doosan-robot) |
| xArm 7 | 7 | UFACTORY | Planned | [xArm ROS](https://github.com/xArm-Developer/xarm_ros) |

---

## Validation Requirements

### Level 1: Format Validation
- [ ] URDF loads in ROS 2 Jazzy
- [ ] MJCF loads in MuJoCo 3.4.0
- [ ] USD loads in Isaac Sim 4.5.0
- [ ] SDF loads in Gazebo Ionic

### Level 2: Kinematic Validation
- [ ] Forward kinematics consistent across formats
- [ ] Joint limits identical
- [ ] Link masses match
- [ ] Inertia tensors valid

### Level 3: Dynamic Validation
- [ ] Physics behavior matches within tolerance
- [ ] Contact dynamics reasonable
- [ ] Actuator responses consistent

### Level 4: Cross-Framework Validation
- [ ] Policy trained in Isaac transfers to MuJoCo
- [ ] Trajectories reproducible across frameworks
- [ ] Benchmark suite passes

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `README.md` | This overview document |
| `model_specification.md` | Detailed model format specification |
| `model_registry.yaml` | Registry of all 50+ models |
| `model_validator.py` | Validation tool implementation |
| `oncology-robots/README.md` | Oncology robot category guide |
| `surgical-instruments/README.md` | Surgical instrument category guide |
| `general-manipulators/README.md` | General manipulator category guide |

---

## Contributing Models

To contribute a model to the repository:

1. **Prepare Model Files**
   - Create URDF from CAD or existing sources
   - Convert to MJCF, USD, and SDF using Objective 1 tools
   - Prepare mesh files (STL/OBJ)

2. **Create Metadata**
   - Fill out `model.yaml` with all required fields
   - Document kinematic and physical properties

3. **Validate**
   - Run `model_validator.py` on all formats
   - Ensure cross-framework validation passes

4. **Submit**
   - Create pull request with model files
   - Include validation report

---

## References

### Model Repositories
- [ROS robot_model](https://github.com/ros/robot_model) - URDF specification
- [Universal Robot Description Directory (URDD)](https://arxiv.org/html/2512.23135) - Unified standard (Dec 2025)
- [Awesome URDF](https://github.com/gbionics/awesome-urdf) - Curated URDF resources
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) - MuJoCo model collection

### Oncology-Specific
- [ORBIT-Surgical](https://orbit-surgical.github.io/) - Surgical simulation
- [JHU dVRK](https://github.com/jhu-dvrk/sawIntuitiveResearchKit) - da Vinci Research Kit
- [SurgicalGym](https://github.com/surgical-robotics-ai/SurgicalGym) - Surgical RL

### General Robotics
- [ROS-Industrial](https://github.com/ros-industrial) - Industrial robot packages
- [MoveIt Robots](https://github.com/ros-planning/moveit_resources) - MoveIt robot descriptions
