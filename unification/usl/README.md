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

## Robot Categories

The USL framework evaluates robots across multiple categories relevant to oncology clinical trials.

| Category | Robots Evaluated | Score Range | Status |
|----------|-----------------|-------------|--------|
| **Humanoid Robots** | Atlas (Electric), Optimus (Gen 2), Digit | 3.6 – 5.8 | v1.6.0 |
| **Surgical Robots** | da Vinci (dVRK), Hugo RAS, Versius | 3.4 – 7.1 | v1.5.0 |
| **Collaborative Robots (Cobots)** | Franka Panda, Kinova Gen3, xArm 7 | 3.4 – 7.4 | v1.4.0 |

---

## Evaluated Humanoid Robots (Category: Humanoid Robots)

This USL evaluation covers three bipedal humanoid robot systems from different manufacturers, each with potential for hospital logistics, patient transport, and assistive tasks in oncology clinical trials. Humanoid robots differ from cobots and surgical robots in their full-body bipedal locomotion (30-50+ DOF), whole-body coordination (locomotion + manipulation), and foundation model integration (GR00T, OpenVLA).

| Robot | Manufacturer | USL Score | USL Level | Band |
|-------|-------------|-----------|-----------|------|
| [Atlas (Electric)](#atlas-electric) | Boston Dynamics | **5.8** | 5 (Functional) | Intermediate |
| [Digit](#digit) | Agility Robotics | **4.2** | 4 (Developing) | Foundational |
| [Optimus (Gen 2)](#optimus-gen-2) | Tesla | **3.6** | 3 (Basic) | Foundational |

---

## Diagram 1: General Humanoid Robot Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│         GENERAL COMPARISON — USL Category: Humanoid Robots              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐
│  │  ATLAS (Electric)    │  │       DIGIT          │  │ OPTIMUS (Gen 2)  │
│  │  (Boston Dynamics)   │  │ (Agility Robotics)   │  │     (Tesla)      │
│  ├──────────────────────┤  ├──────────────────────┤  ├──────────────────┤
│  │ Heritage: Pioneer    │  │ Heritage: First      │  │ Heritage: Tesla  │
│  │  in dynamic bipedal  │  │  humanoid in commer- │  │  FSD AI + mass   │
│  │  locomotion since    │  │  cial deployment     │  │  manufacturing   │
│  │  2013 (hydraulic)    │  │  (Amazon logistics)  │  │  scale           │
│  │                      │  │                      │  │                  │
│  │ Ecosystem: Drake     │  │ Ecosystem: NVIDIA    │  │ Ecosystem:       │
│  │  (open-source sim),  │  │  GR00T N1 foundation │  │  Fully propri-   │
│  │  BDAII research,     │  │  model, Isaac Lab +  │  │  etary. No SDK,  │
│  │  Hyundai deployment  │  │  MuJoCo Menagerie    │  │  no open-source  │
│  │                      │  │                      │  │                  │
│  │ Key Strength:        │  │ Key Strength:        │  │ Key Strength:    │
│  │  Most dynamic ROM,   │  │  GR00T + Isaac Lab   │  │  11-DOF hands +  │
│  │  4-framework sim,    │  │  sim-to-real pipeline │  │  mass production │
│  │  Drake open-source   │  │  + 16 kg payload     │  │  at sub-$20K     │
│  │                      │  │                      │  │                  │
│  │ Oncology Focus:      │  │ Oncology Focus:      │  │ Oncology Focus:  │
│  │  Equipment handling, │  │  Supply tote delivery │  │  Pharmacy deliv- │
│  │  specimen delivery,  │  │  specimen courier,   │  │  ery, linen      │
│  │  decontamination     │  │  pharmacy restocking │  │  transport       │
│  │                      │  │                      │  │                  │
│  │ USL Score: 5.8 █████ │  │ USL Score: 4.2 ███   │  │ USL Score: 3.6███│
│  │ Level 5 — Functional │  │ Level 4 — Developing │  │ Level 3 — Basic  │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────┘
│                                                                         │
│  Legend: Each █ ≈ 1.2 points on the 1.0–10.0 USL scale                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 2: Technical Specifications — Humanoid Robots

```
┌─────────────────────────────────────────────────────────────────────────┐
│       TECHNICAL SPECIFICATIONS — Humanoid Robot Comparison              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Spec               Atlas (Electric)  Digit             Optimus (Gen2) │
│  ─────────────────  ────────────────  ────────────────  ────────────────│
│  Architecture       Bipedal electric  Bipedal backward  Bipedal full   │
│                     full-body         -bending knees    -body humanoid │
│  Height (m)         ~1.50             ~1.75             ~1.73          │
│  Weight (kg)        ~89               ~65               ~57 ◄─ light  │
│  Body DOF           ~28               ~20               ~28           │
│  Hand DOF (each)    Custom EE         4-finger          11 ◄─ most    │
│  Walking Speed(m/s) 1.5               1.5               1.3          │
│  Payload (kg)       11                16 ◄─ highest    9              │
│  Battery Life (hrs) ~1.5              ~3.0              ~4.0 ◄─ most  │
│  Perception         Stereo + LiDAR    LiDAR + stereo    FSD cameras   │
│                     + IMU + F/T       + depth + IMU     + IMU + tactile│
│  Compute            Custom onboard    Jetson AGX Orin   Tesla HW4/SoC │
│  Knee Design        Standard (hyper-  Backward-bending  Standard      │
│                     extended ROM)     (bird-inspired)                  │
│                                                                         │
│  Simulation / Open-Source Support:                                      │
│  ─────────────────────────────────                                      │
│  Isaac Lab          ✓ (URDF/USD)     ✓ Official ◄─    ◐ Community    │
│  MuJoCo             ✓ Community       ✓ Menagerie ◄─  ◐ Community    │
│  Drake              ✓ Official ◄─    ✗                 ✗              │
│  Gazebo + ROS 2     ◐ Community       ◐ Community       ✗              │
│  Open-Source Code   ◐ Drake (TRI)    ◐ Isaac Lab model ✗ None         │
│  Frameworks (#)     4                 3                 2              │
│                                                                         │
│  AI / Foundation Model Support:                                        │
│  ──────────────────────────────                                         │
│  GR00T N1           ◐ Potential       ✓ Primary ◄─    ✗              │
│  VLA Compatible     ✓                 ✓                 ✗              │
│  LLM Task Planning  ✓                 ✓                 ✗              │
│  RL Locomotion      ✓                 ✓                 ✓              │
│  Imitation Learning ✓                 ✓                 ✓              │
│  Diffusion Policy   ✓                 ✗                 ✗              │
│                                                                         │
│  Deployment / Clinical:                                                │
│  ──────────────────────                                                 │
│  Commercial Deploy  ✗ (Hyundai plan) ✓ Amazon ◄─     ✗ (factory)    │
│  Hospital Pilot     ✓ Potential       ✗                 ✗              │
│  ISO 13482 Align    ✓                 ✓                 ✗              │
│  Manufacturing      Hyundai (limited) RoboFab (10K/yr) Tesla (millions)│
│                                                                         │
│  ✓ = Supported   ◐ = Partial/Limited   ✗ = Not available               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 3: USL Scoring Breakdown — Humanoid Robots

```
┌─────────────────────────────────────────────────────────────────────────┐
│         USL SCORING BREAKDOWN — Humanoid Robot Dimension Scores         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Dimension A: Simulation Framework Switching (25% weight)               │
│  ────────────────────────────────────────────────                       │
│  Atlas(Elec)  [███████░░░] 7.0   4 frameworks, Drake open-source, GPU │
│  Digit        [█████░░░░░] 5.8   3 frameworks, Isaac Lab + Menagerie  │
│  Optimus(G2)  [███░░░░░░░] 3.4   2 frameworks, community models only │
│                                                                         │
│  Dimension B: Generative / Agentic AI Integration (25% weight)          │
│  ────────────────────────────────────────────────────────               │
│  Atlas(Elec)  [███████░░░] 7.4   RL, IL, diffusion, VLA, LLM plan    │
│  Digit        [█████░░░░░] 5.4   GR00T N1, VLA, RL, IL, LLM plan    │
│  Optimus(G2)  [█████░░░░░] 5.0   End-to-end NN, RL, IL, Dojo train  │
│                                                                         │
│  Dimension C: Cross-Robot Progress Sharing (25% weight)                 │
│  ──────────────────────────────────────────────                         │
│  Atlas(Elec)  [████░░░░░░] 4.5   Proprietary; Drake + ONNX sharing   │
│  Digit        [███░░░░░░░] 3.0   Isaac Lab models + ONNX; no inter   │
│  Optimus(G2)  [█░░░░░░░░░] 1.5   Fully proprietary; no sharing      │
│                                                                         │
│  Dimension D: Multi-Site Clinical Trial Collaboration (25% weight)      │
│  ─────────────────────────────────────────────────────                  │
│  Atlas(Elec)  [████░░░░░░] 4.2   Safety cert, ISO 13482, pilot poss  │
│  Digit        [██░░░░░░░░] 2.7   Commercial deploy but no healthcare │
│  Optimus(G2)  [████░░░░░░] 4.4   Audit trail, remote mon; no certs   │
│                                                                         │
│  ═══════════════════════════════════════════════════                    │
│  FINAL USL SCORES (weighted average):                                   │
│  ────────────────────────────────────                                   │
│  Atlas(Elec)  [█████░░░░░] 5.8   Level 5 — Functional                │
│  Digit        [████░░░░░░] 4.2   Level 4 — Developing                │
│  Optimus(G2)  [███░░░░░░░] 3.6   Level 3 — Basic                    │
│                                                                         │
│  Bar scale: each █ = 1.0 point (10 blocks = 10.0)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Atlas (Electric)

**USL Score: 5.8 / 10.0 — Level 5 (Functional)**

The Boston Dynamics Atlas Electric is the next-generation fully electric humanoid robot, announced in April 2024 as the successor to the iconic hydraulic Atlas platform. It features a compact form factor (~1.5 m, ~89 kg), exceptional range of motion exceeding human capabilities at many joints, and advanced whole-body dynamic locomotion. The Boston Dynamics AI Institute (BDAII) conducts foundational research on Atlas, and the Drake open-source simulator (MIT/TRI) serves as the primary planning and control platform.

**Key USL strengths:**
- Most dynamically capable humanoid — exceeds human range of motion at many joints
- Drake (TRI): open-source model-based planning and simulation
- 4 simulation frameworks supported (Drake, Isaac Lab, MuJoCo, Gazebo)
- BDAII publishes peer-reviewed locomotion and manipulation research
- Proven locomotion on diverse terrains (stairs, slopes, uneven ground)
- Hyundai Motor Group partnership for industrial deployment

**Key USL gaps:**
- Proprietary platform — no public SDK for Atlas Electric yet
- No clinical or hospital deployment history
- No ROS 2 interface (only Spot has community bridges)
- No GR00T or OpenVLA foundation model integration announced
- High cost limits widespread multi-site deployment

**References:**
- [Boston Dynamics Atlas Electric](https://bostondynamics.com/blog/electric-new-era-for-atlas/) — April 2024 announcement
- [Drake](https://github.com/RobotLocomotion/drake) — Open-source simulation and planning (MIT/TRI)
- Kuindersma, S. et al. (2016). Optimization-based locomotion planning for Atlas. *Autonomous Robots*. DOI: [10.1007/s10514-015-9479-3](https://doi.org/10.1007/s10514-015-9479-3)
- Tedrake, R. (2023). *Robotic Manipulation*. MIT. [manipulation.csail.mit.edu](https://manipulation.csail.mit.edu/)

---

## Digit

**USL Score: 4.2 / 10.0 — Level 4 (Developing)**

Agility Robotics Digit is the first humanoid robot to enter commercial deployment, with Amazon testing Digit for warehouse logistics. It features a unique backward-bending knee design (bird-inspired, spring-loaded for energy-efficient walking), the highest payload capacity among evaluated humanoids (16 kg), and official simulation models in both NVIDIA Isaac Lab and MuJoCo Menagerie. Digit is a primary integration target for NVIDIA's GR00T N1 humanoid foundation model (announced GTC 2025). Agility operates RoboFab in Salem, Oregon — the world's first humanoid robot factory with 10,000 units/year capacity.

**Key USL strengths:**
- First humanoid robot in commercial deployment (Amazon logistics)
- NVIDIA GR00T N1 foundation model integration — primary partner
- Official models in Isaac Lab and MuJoCo Menagerie
- Highest payload capacity (16 kg) among evaluated humanoids
- RoboFab factory with 10,000 units/year production capacity
- NVIDIA Jetson AGX Orin onboard for edge AI inference
- ONNX policy export via Isaac Lab training pipeline

**Key USL gaps:**
- No healthcare or hospital deployment experience
- No ROS 2 interface for clinical system integration
- No ISO 13482 safety certification for healthcare
- Limited hand dexterity (4 fingers vs. Optimus 11 DOF per hand)
- No federated learning or multi-site clinical trial tools

**References:**
- [Agility Robotics Digit](https://agilityrobotics.com/robots) — Official documentation
- [NVIDIA GR00T N1](https://developer.nvidia.com/isaac/humanoid) — Humanoid foundation model
- [MuJoCo Menagerie (Digit)](https://github.com/google-deepmind/mujoco_menagerie) — Official model
- [NVIDIA Isaac Lab (Digit)](https://github.com/isaac-sim/IsaacLab) — Official locomotion examples
- [RoboFab](https://agilityrobotics.com/robofab) — Humanoid robot factory

---

## Optimus (Gen 2)

**USL Score: 3.6 / 10.0 — Level 3 (Basic)**

Tesla Optimus Gen 2 is a general-purpose bipedal humanoid robot leveraging Tesla's FSD neural network architecture, Dojo supercomputer for policy training, and automotive manufacturing scale. Demonstrated in December 2023 with 30% speed improvement and 10 kg weight reduction over Gen 1, Optimus features 11-DOF dexterous hands with tactile sensing — the most capable hands among evaluated humanoids. Tesla aims for sub-$20,000 per unit at scale, which could enable wide-scale hospital deployment. However, Optimus is fully proprietary with no public SDK, simulation models, or developer program as of February 2026.

**Key USL strengths:**
- 11-DOF dexterous hands with tactile sensing — best hands in class
- FSD-derived perception and end-to-end neural network control
- Tesla Dojo supercomputer for large-scale policy training
- Mass production targeting sub-$20,000 per unit (Tesla factories)
- Lightweight (57 kg) with longest battery life (4 hours)
- Imitation learning from human teleoperation data (internal)

**Key USL gaps:**
- Fully proprietary — no public SDK, API, or open-source code
- No published simulation models (URDF/MJCF/USD)
- No peer-reviewed research publications
- No developer ecosystem or research community access
- No ROS 2 integration or standardized middleware
- No healthcare safety certifications or hospital testing

**References:**
- [Tesla AI Day 2022](https://www.youtube.com/watch?v=ODSJsviD_SU) — Optimus Gen 1 demonstration
- [Optimus Gen 2 Demo](https://www.youtube.com/watch?v=cpraXaw7dyc) — December 2023

---

## Evaluated Surgical Robots (Category: Surgical Robots)

This USL evaluation covers three teleoperated surgical robot systems from different manufacturers, each used in minimally invasive oncology surgery. These are master-slave systems where the surgeon controls robotic arms from a console.

| Robot | Manufacturer | USL Score | USL Level | Band |
|-------|-------------|-----------|-----------|------|
| [da Vinci (dVRK)](#da-vinci-dvrk) | Intuitive Surgical | **7.1** | 7 (Advanced) | Advanced |
| [Hugo RAS](#hugo-ras) | Medtronic | **4.5** | 4 (Developing) | Foundational |
| [Versius](#versius) | CMR Surgical | **3.4** | 3 (Basic) | Foundational |

---

## Diagram 4: General Surgical Robot Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│          GENERAL COMPARISON — USL Category: Surgical Robots             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐
│  │   DA VINCI (dVRK)    │  │     HUGO RAS         │  │    VERSIUS       │
│  │ (Intuitive Surgical) │  │    (Medtronic)       │  │  (CMR Surgical)  │
│  ├──────────────────────┤  ├──────────────────────┤  ├──────────────────┤
│  │ Heritage: Pioneer    │  │ Heritage: Medtronic  │  │ Heritage: UK     │
│  │  of robotic surgery  │  │  medical device      │  │  startup focused │
│  │  since 1999          │  │  ecosystem           │  │  on portability  │
│  │                      │  │                      │  │                  │
│  │ Ecosystem: Largest   │  │ Ecosystem: Touch     │  │ Ecosystem:       │
│  │  open-source via     │  │  Surgery Enterprise  │  │  Proprietary     │
│  │  dVRK (JHU, 45+     │  │  AI video analytics  │  │  with growing    │
│  │  institutions)       │  │                      │  │  clinical data   │
│  │                      │  │                      │  │                  │
│  │ Key Strength:        │  │ Key Strength:        │  │ Key Strength:    │
│  │  ORBIT-Surgical GPU  │  │  Modular cart arms + │  │  Lightest arms   │
│  │  sim + 14M clinical  │  │  Medtronic OR stack  │  │  (~10 kg) +      │
│  │  procedures          │  │  integration         │  │  portable across │
│  │                      │  │                      │  │  operating rooms │
│  │ Oncology Focus:      │  │ Oncology Focus:      │  │ Oncology Focus:  │
│  │  Prostatectomy,      │  │  Colectomy, gynecol- │  │  Gynecologic &   │
│  │  nephrectomy,        │  │  ogic, and urologic  │  │  colorectal      │
│  │  lobectomy           │  │  oncology            │  │  oncology        │
│  │                      │  │                      │  │                  │
│  │ USL Score: 7.1 ██████│  │ USL Score: 4.5 ████  │  │ USL Score: 3.4██ │
│  │ Level 7 — Advanced   │  │ Level 4 — Developing │  │ Level 3 — Basic  │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────┘
│                                                                         │
│  Legend: Each █ ≈ 1.2 points on the 1.0–10.0 USL scale                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 5: Technical Specifications — Surgical Robots

```
┌─────────────────────────────────────────────────────────────────────────┐
│       TECHNICAL SPECIFICATIONS — Surgical Robot Comparison              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Spec               da Vinci (dVRK)   Hugo RAS          Versius         │
│  ─────────────────  ────────────────  ────────────────  ────────────────│
│  Architecture       Single cart       Modular carts     Modular carts   │
│  Instrument Arms    3 PSMs            3 arms            3 arms (max 4)  │
│  Camera Arms        1 ECM             1 arm             1 arm           │
│  DOF per Arm        7 + grip          7 + grip          7 + grip        │
│  Instrument ⌀       5/8 mm            8 mm              5 mm ◄─ thin   │
│  Arm Weight (kg)    ~65 (full cart)   38 (per cart)     10 ◄─ lightest  │
│  Console Type       Immersive         Open              Open            │
│  Tremor Filtering   Yes               Yes               Yes             │
│  Motion Scaling     2:1, 3:1, 5:1    Yes               Yes             │
│  Control Freq (Hz)  1000              1000              1000            │
│  Stereo Vision      Yes               3D HD             3D HD           │
│  Portability        Fixed             Semi-portable     Portable ◄─    │
│                                                                         │
│  Regulatory Status:                                                     │
│  ─────────────────                                                      │
│  FDA Cleared        ✓ ◄─ only one    ✗ (pending)       ✗ (pending)    │
│  CE Marked          ✓                 ✓                 ✓               │
│  Countries          69                20+               25+             │
│  Systems Installed  9,000+ ◄─ most   500+              350+            │
│  Procedures Done    14M+ ◄─ most     50,000+           30,000+         │
│                                                                         │
│  Simulation / Open-Source Support:                                      │
│  ─────────────────────────────────                                      │
│  Open-Source Code   ✓ dVRK (BSD)     ✗                 ✗               │
│  ORBIT-Surgical     ✓ ◄─ unique     ✗                 ✗               │
│  SurRoL Benchmark   ✓ ◄─ unique     ✗                 ✗               │
│  AMBF Simulator     ✓ ◄─ unique     ✗                 ✗               │
│  Gazebo + ROS 2     ✓                ◐                 ◐               │
│  MuJoCo             ✓                ✗                 ◐               │
│  GPU Sim (Isaac)    ✓                ◐                 ✗               │
│                                                                         │
│  AI Research Ecosystem:                                                 │
│  ─────────────────────                                                  │
│  Surgical Video AI  ✓ JIGSAWS+       ✓ Touch Surgery   ◐ Limited      │
│  RL / IL Research   ✓ Extensive      ✗                 ✗               │
│  VLA / Diffusion    ✓ ◄─ unique     ✗                 ✗               │
│  Phase Recognition  ✓                ✓                 ◐               │
│                                                                         │
│  ✓ = Supported   ◐ = Partial/Limited   ✗ = Not available               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 6: USL Scoring Breakdown — Surgical Robots

```
┌─────────────────────────────────────────────────────────────────────────┐
│         USL SCORING BREAKDOWN — Surgical Robot Dimension Scores         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Dimension A: Simulation Framework Switching (25% weight)               │
│  ────────────────────────────────────────────────                       │
│  da Vinci(dVRK) [███████░░░] 7.5   5 frameworks, GPU sim, tissue def  │
│  Hugo RAS       [████░░░░░░] 4.5   2 frameworks, limited open-source  │
│  Versius        [███░░░░░░░] 3.2   2 frameworks, no open-source sim   │
│                                                                         │
│  Dimension B: Generative / Agentic AI Integration (25% weight)          │
│  ────────────────────────────────────────────────────────               │
│  da Vinci(dVRK) [███████░░░] 7.2   VLA, diffusion, IL, RL, video AI  │
│  Hugo RAS       [████░░░░░░] 4.0   Touch Surgery video AI + phase     │
│  Versius        [██░░░░░░░░] 2.8   Basic video AI, limited research   │
│                                                                         │
│  Dimension C: Cross-Robot Progress Sharing (25% weight)                 │
│  ──────────────────────────────────────────────                         │
│  da Vinci(dVRK) [██████░░░░] 6.8   Open-source, ONNX, ROS 2, JIGSAWS│
│  Hugo RAS       [███░░░░░░░] 3.5   Intra-org via Medtronic ecosystem │
│  Versius        [██░░░░░░░░] 2.5   Intra-org only, no open standards │
│                                                                         │
│  Dimension D: Multi-Site Clinical Trial Collaboration (25% weight)      │
│  ─────────────────────────────────────────────────────                  │
│  da Vinci(dVRK) [███████░░░] 7.0   FDA cleared, 9000+ systems, certs │
│  Hugo RAS       [█████░░░░░] 5.8   CE marked, EXPAND trials, audit   │
│  Versius        [█████░░░░░] 5.2   CE marked, 350+ hospitals, certs  │
│                                                                         │
│  ═══════════════════════════════════════════════════                    │
│  FINAL USL SCORES (weighted average):                                   │
│  ────────────────────────────────────                                   │
│  da Vinci(dVRK) [███████░░░] 7.1   Level 7 — Advanced                │
│  Hugo RAS       [████░░░░░░] 4.5   Level 4 — Developing              │
│  Versius        [███░░░░░░░] 3.4   Level 3 — Basic                   │
│                                                                         │
│  Bar scale: each █ = 1.0 point (10 blocks = 10.0)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## da Vinci (dVRK)

**USL Score: 7.1 / 10.0 — Level 7 (Advanced)**

The da Vinci Surgical System by Intuitive Surgical is the most widely deployed surgical robot in the world, with over 9,000 systems installed across 69 countries and more than 14 million procedures performed. The da Vinci Research Kit (dVRK), maintained by Johns Hopkins University, provides an open-source research platform based on the da Vinci Classic/S/Si hardware, enabling extensive academic collaboration.

**Key USL strengths:**
- ORBIT-Surgical (Isaac Lab): GPU-accelerated surgical RL with dVRK models
- SurRoL (PyBullet): 10 validated surgical task benchmarks
- AMBF: Real-time deformable tissue simulation with haptic feedback
- Open-source dVRK platform used at 45+ research institutions worldwide
- Extensive VLA, diffusion policy, and imitation learning research
- FDA-cleared clinical system with established regulatory pathway

**Key USL gaps:**
- dVRK hardware is legacy (Classic/S/Si) — not current Xi/5 system
- Instrument interchangeability limited by proprietary EndoWrist design
- No MCP server integration or HIPAA-compliant data sharing tools

**Open-source references:**
- [dVRK (sawIntuitiveResearchKit)](https://github.com/jhu-dvrk/sawIntuitiveResearchKit) — Open-source research kit
- [ORBIT-Surgical](https://github.com/orbit-surgical/orbit-surgical) — GPU-accelerated surgical RL (Isaac Lab)
- [SurRoL](https://github.com/med-air/SurRoL) — Surgical robot learning benchmark (PyBullet)
- [AMBF](https://github.com/WPI-AIM/ambf) — Asynchronous Multi-Body Framework
- [dvrk-ros](https://github.com/jhu-dvrk/dvrk-ros) — ROS 2 integration

---

## Hugo RAS

**USL Score: 4.5 / 10.0 — Level 4 (Developing)**

The Medtronic Hugo RAS is a modular surgical robot system featuring independently cart-mounted arms and an open console design. It integrates with Medtronic's Touch Surgery Enterprise platform for AI-powered surgical video analysis and is backed by Medtronic's extensive medical device ecosystem including StealthStation navigation and O-arm imaging.

**Key USL strengths:**
- Modular arm design — each arm independently positioned around patient
- Touch Surgery Enterprise — integrated AI video analysis platform
- Medtronic ecosystem integration (StealthStation, O-arm, Surgical Synergy)
- CE Mark obtained — active clinical use in Europe and other regions
- EXPAND clinical trial program demonstrating surgical outcomes

**Key USL gaps:**
- No public open-source code, models, or simulation environments
- FDA clearance not yet obtained as of February 2026
- No ROS 2 integration or standardized robot middleware
- Limited academic research community compared to dVRK

**References:**
- [Medtronic Hugo RAS](https://www.medtronic.com/covidien/en-us/robotic-assisted-surgery/hugo-ras-system.html) — Official product page
- [Touch Surgery Enterprise](https://www.medtronic.com/covidien/en-us/robotic-assisted-surgery/touch-surgery.html) — AI surgical intelligence

---

## Versius

**USL Score: 3.4 / 10.0 — Level 3 (Basic)**

The CMR Surgical Versius is a modular, portable surgical robot from Cambridge, UK, featuring the lightest surgical robot arms in the market (~10 kg each). Its biomimetic design mimics the human arm and enables portability between operating rooms. Versius is deployed in 350+ hospitals across 25+ countries.

**Key USL strengths:**
- Lightest surgical robot arms (~10 kg each) — highly portable
- 5 mm instrument diameter — thinnest instruments in class
- Biomimetic arm design — intuitive OR positioning
- Portable between operating rooms within a facility
- CE Mark obtained — deployed in 350+ hospitals across 25+ countries

**Key USL gaps:**
- No public open-source code, models, or simulation environments
- FDA clearance not yet obtained as of February 2026
- Smallest academic research community of the three surgical robots
- No published AI/ML research using Versius platform

**References:**
- [CMR Surgical Versius](https://cmrsurgical.com/) — Official site
- Puntambekar, S., et al. (2022). DOI: [10.1007/s11701-022-01390-2](https://doi.org/10.1007/s11701-022-01390-2) — Initial Versius experience
- Morton, B., et al. (2023). DOI: [10.1007/s11701-023-01611-w](https://doi.org/10.1007/s11701-023-01611-w) — Systematic review

---

## Evaluated Cobots (Category: Collaborative Robots)

This USL evaluation covers three state-of-the-art open-source collaborative robot arms from different manufacturers, each with active ROS 2 support, MuJoCo Menagerie models, and potential oncology applications.

| Robot | Manufacturer | USL Score | USL Level | Band |
|-------|-------------|-----------|-----------|------|
| [Franka Emika Panda](#franka-emika-panda) | Franka Robotics | **7.4** | 7 (Advanced) | Advanced |
| [Kinova Gen3 7DoF](#kinova-gen3-7dof) | Kinova Robotics | **5.7** | 5 (Functional) | Intermediate |
| [UFACTORY xArm 7](#ufactory-xarm-7) | UFACTORY | **3.4** | 3 (Basic) | Foundational |

---

## Diagram 7: General Cobot Comparison

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

## Diagram 8: Technical Specifications — Cobots

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

## Diagram 9: USL Scoring Breakdown — Cobots

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

## Directory Structure

```
unification/usl/
├── README.md                              # This file
├── humanoids/                             # Humanoid Robots category (v1.6.0)
│   ├── usl_humanoid_scoring.py            # USL scoring engine (humanoids)
│   ├── boston_dynamics_atlas/
│   │   └── boston_dynamics_atlas_usl.py    # Atlas Electric evaluation + tools
│   ├── tesla_optimus/
│   │   └── tesla_optimus_usl.py           # Optimus Gen 2 evaluation + tools
│   └── agility_digit/
│       └── agility_digit_usl.py           # Digit evaluation + tools
├── surgical/                              # Surgical Robots category (v1.5.0)
│   ├── usl_surgical_scoring.py            # USL scoring engine (surgical)
│   ├── intuitive_davinci/
│   │   └── intuitive_davinci_usl.py       # da Vinci (dVRK) evaluation + tools
│   ├── medtronic_hugo/
│   │   └── medtronic_hugo_usl.py          # Hugo RAS evaluation + tools
│   └── cmr_versius/
│       └── cmr_versius_usl.py             # Versius evaluation + tools
└── cobots/                                # Collaborative Robots category (v1.4.0)
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

2. **ML Technology Readiness Levels (MLTRL)** — Lavin, A., et al. (2021). *Technology Readiness Levels for Machine Learning Systems*. GitHub: [ai-infrastructure-alliance/mltrl](https://github.com/ai-infrastructure-alliance/mltrl). Extends TRL to ML systems with levels 1–9 covering data readiness, model development, deployment, and monitoring. USL incorporates MLTRL's recognition that AI system readiness requires evaluating software, data, and integration dimensions beyond hardware alone.

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

## Quick Start

### Run Humanoid Robot USL Scoring Demo

```bash
python unification/usl/humanoids/usl_humanoid_scoring.py
```

### Evaluate Individual Humanoid Robots

```bash
# Boston Dynamics Atlas (Electric)
python unification/usl/humanoids/boston_dynamics_atlas/boston_dynamics_atlas_usl.py

# Tesla Optimus (Gen 2)
python unification/usl/humanoids/tesla_optimus/tesla_optimus_usl.py

# Agility Robotics Digit
python unification/usl/humanoids/agility_digit/agility_digit_usl.py
```

### Run Surgical Robot USL Scoring Demo

```bash
python unification/usl/surgical/usl_surgical_scoring.py
```

### Evaluate Individual Surgical Robots

```bash
# Intuitive Surgical da Vinci (dVRK)
python unification/usl/surgical/intuitive_davinci/intuitive_davinci_usl.py

# Medtronic Hugo RAS
python unification/usl/surgical/medtronic_hugo/medtronic_hugo_usl.py

# CMR Surgical Versius
python unification/usl/surgical/cmr_versius/cmr_versius_usl.py
```

### Run Cobot USL Scoring Demo

```bash
python unification/usl/cobots/usl_scoring_framework.py
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

---

## Contributing

To add a new robot category or evaluate additional robots:

1. Create a new category directory under `usl/` (e.g., `mobile_manipulators/`)
2. Create a scoring framework module adapted for the category
3. Create a subdirectory for each evaluated robot with an evaluation module
4. Add the robot to this README with all diagram sections updated
5. Validate across at least 2 simulation frameworks
6. Submit a PR with USL scores and supporting evidence

---

*Last updated: February 2026*
