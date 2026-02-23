# Unification Standard Level (USL) for Physical AI Oncology Trials

*Standardizing and Evaluating Robot Unification Readiness for Multi-Site Clinical Trials (February 2026)*

---

## Overview

The **Unification Standard Level (USL)** is a scoring framework for evaluating how ready a physical AI robot is for deployment in unified, multi-site oncology clinical trials. USL scores range from **1.0 to 10.0** (in 0.1 increments) and assess four weighted dimensions:

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **A) Simulation Framework Switching** | 25% | Ability to move trained policies between simulation engines |
| **B) Generative / Agentic AI Integration** | 25% | Integration with LLMs, VLAs, diffusion policies, agentic frameworks |
| **C) Cross-Robot Progress Sharing** | 25% | Capacity to share and continue progress with other robots (intra- and inter-organization) |
| **D) Multi-Site Clinical Trial Collaboration** | 25% | Readiness for federated, regulatory-compliant deployment across clinical trial sites |

Each dimension derives from the four unification pillars defined in [`unification/`](../):
- Dimension A ← [`simulation_physics/`](../simulation_physics/)
- Dimension B ← [`agentic_generative_ai/`](../agentic_generative_ai/)
- Dimension C ← [`cross_platform_tools/`](../cross_platform_tools/) and [`surgical_robotics/`](../surgical_robotics/)
- Dimension D ← [`../../federation/`](../../federation/) and [`../../regulatory/`](../../regulatory/)

The USL evaluates robots across **multiple categories** — each category uses the same four-dimension framework with scoring criteria adapted to the unique characteristics of that robot type. Current categories include **Surgical Robots** and **Collaborative Robots (Cobots)**.

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

## Evaluated Surgical Robots (Category: Surgical Robot Systems)

This USL evaluation covers three major surgical robot systems from different manufacturers, representing the three dominant architectural paradigms in teleoperated surgery: boom-mounted (da Vinci), cart-based modular (Hugo RAS), and table-integrated (OTTAVA).

| Robot | Manufacturer | USL Score | USL Level | Band |
|-------|-------------|-----------|-----------|------|
| [da Vinci (Xi / da Vinci 5)](#da-vinci-xi--da-vinci-5) | Intuitive Surgical | **7.6** | 7 (Advanced) | Advanced |
| [Hugo RAS](#hugo-ras) | Medtronic | **4.1** | 4 (Developing) | Foundational |
| [OTTAVA](#ottava) | Johnson & Johnson MedTech | **2.3** | 2 (Exploratory) | Initial |

---

## Diagram 1: General Surgical Robot Comparison

```
┌──────────────────────────────────────────────────────────────────────────────┐
│          GENERAL SURGICAL ROBOT COMPARISON — USL Category: Surgical          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────┐  ┌────────────────────────┐  ┌─────────────────┐│
│  │ DA VINCI (Xi / dV5)    │  │ HUGO RAS               │  │ OTTAVA          ││
│  │ (Intuitive Surgical)   │  │ (Medtronic)            │  │ (J&J MedTech)   ││
│  ├────────────────────────┤  ├────────────────────────┤  ├─────────────────┤│
│  │ Heritage: Pioneer in   │  │ Heritage: Medtronic's  │  │ Heritage: J&J   ││
│  │  surgical robotics     │  │  entry into robotic    │  │  Ethicon + Auris││
│  │  (1999 — first FDA)    │  │  surgery (2021+ OUS)   │  │  Health + Verb  ││
│  │                        │  │                        │  │                 ││
│  │ Architecture:          │  │ Architecture:          │  │ Architecture:   ││
│  │  Boom-mounted unified  │  │  Modular cart-based    │  │  Table-         ││
│  │  patient cart (4 arms) │  │  (4 independent carts) │  │  integrated     ││
│  │                        │  │                        │  │  (bed-mounted)  ││
│  │ Key Strength:          │  │ Key Strength:          │  │ Key Strength:   ││
│  │  Largest evidence base │  │  Modular flexibility + │  │  Twin Motion +  ││
│  │  + open-source dVRK    │  │  standalone laparos-   │  │  zero OR foot-  ││
│  │  research ecosystem    │  │  copic fallback mode   │  │  print design   ││
│  │                        │  │                        │  │                 ││
│  │ FDA Status:            │  │ FDA Status:            │  │ FDA Status:     ││
│  │  Cleared (all indica-  │  │  Cleared Dec 2025      │  │  De Novo filed  ││
│  │  tions, established)   │  │  (urologic procedures) │  │  Jan 2026       ││
│  │                        │  │                        │  │                 ││
│  │ USL Score: 7.6 ███████ │  │ USL Score: 4.1 ████   │  │ USL: 2.3 ██    ││
│  │ Level 7 — Advanced     │  │ Level 4 — Developing   │  │ Level 2 — Expl ││
│  └────────────────────────┘  └────────────────────────┘  └─────────────────┘│
│                                                                              │
│  Legend: Each █ ≈ 1.0 point on the 1.0–10.0 USL scale                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 2: Technical Specifications — Surgical Robots

```
┌──────────────────────────────────────────────────────────────────────────────┐
│        TECHNICAL SPECIFICATIONS — Surgical Robot Side-by-Side                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Spec                   da Vinci Xi/5      Hugo RAS         OTTAVA           │
│  ─────────────────────  ─────────────────  ───────────────  ───────────────  │
│  Instrument DOF         7 EndoWrist        7 wristed        Not disclosed    │
│  Arms                   4 (shared boom)    4 (indep carts)  4 (table-integ)  │
│  Mounting               Boom-mounted       Cart-based       Bed-mounted      │
│  Console Type           Closed immersive   Open (3D screen) Not disclosed    │
│  Controller             Pincer-grip        Pistol-grip+IR   Not disclosed    │
│  Vision System          Proprietary 3D HD  Karl Storz 3D    Not disclosed    │
│  Force Feedback         Yes (dV5 only) ◄   Reported haptic  Not disclosed    │
│  Wrist Rotation         Full articulation  520° (2x multi)  Not disclosed    │
│  Electrosurgery         Integrated         Valleylab        Ethicon          │
│  Head Tracking          No                 Yes ◄─ unique    Not disclosed    │
│  Collision Avoidance    Software-based     Yes ◄─ built-in  Not disclosed    │
│  Twin Motion            No                 No               Yes ◄─ unique    │
│  Standalone Lap Mode    No                 Yes ◄─ unique    No               │
│                                                                              │
│  Open-Source Ecosystem:                                                      │
│  ─────────────────────                                                       │
│  Research Kit           dVRK (~40 sites)   None             None             │
│  Simulation Platform    ORBIT-Surgical ◄   None             None             │
│  ROS 2 Bridge           Yes (cisst/SAW)    No               No               │
│  Published Kinematics   Yes (full DH)      No               No               │
│                                                                              │
│  Communication Protocol:                                                     │
│  ──────────────────────                                                      │
│  Low-level (dVRK)       IEEE-1394/EtherCAT Proprietary      Proprietary      │
│  High-level             cisst/SAW + ROS    Proprietary      Proprietary      │
│  Control Frequency      ~2 kHz (dVRK)      Not disclosed    Not disclosed    │
│  FPGA Control           100 kHz PI-loop    Not disclosed    Not disclosed    │
│                                                                              │
│  Clinical Evidence:                                                          │
│  ────────────────                                                            │
│  Total Procedures       ~14,000,000 ◄      ~50,000          ~100 (IDE)       │
│  FDA Status             Cleared (all)      Cleared (uro)    De Novo pending  │
│  Countries              70+                30+              IDE only          │
│  Installed Base         ~9,000 systems     Growing          Pre-market       │
│                                                                              │
│  Digital Ecosystem      My Intuitive       Touch Surgery    Polyphonic       │
│                                                                              │
│  ◄ = category leader for that specification                                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 3: USL Scoring Breakdown — Surgical Robots

```
┌──────────────────────────────────────────────────────────────────────────────┐
│         USL SCORING BREAKDOWN — Surgical Robot Dimension-by-Dimension        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Dimension A: Simulation Framework Switching (25% weight)                    │
│  ────────────────────────────────────────────────                            │
│  da Vinci     [███████░░░] 7.8   dVRK + ORBIT-Surgical + SurRoL + GPU     │
│  Hugo RAS     [███░░░░░░░] 3.5   No public models; proprietary only        │
│  OTTAVA       [█░░░░░░░░░] 1.0   No simulation available; pre-market       │
│                                                                              │
│  Dimension B: Generative / Agentic AI Integration (25% weight)               │
│  ────────────────────────────────────────────────────────                    │
│  da Vinci     [███████░░░] 7.2   Extensive RL/IL; autonomous suturing      │
│  Hugo RAS     [███░░░░░░░] 3.8   Touch Surgery planning; limited AI        │
│  OTTAVA       [██░░░░░░░░] 2.3   Minimal; pre-market constraints           │
│                                                                              │
│  Dimension C: Cross-Robot Progress Sharing (25% weight)                      │
│  ──────────────────────────────────────────────                              │
│  da Vinci     [██████░░░░] 6.5   dVRK open-source; ONNX; OpenIGTLink      │
│  Hugo RAS     [███░░░░░░░] 3.2   Intra-Medtronic only; proprietary API    │
│  OTTAVA       [█░░░░░░░░░] 1.8   No sharing infrastructure yet            │
│                                                                              │
│  Dimension D: Multi-Site Clinical Trial Collaboration (25% weight)           │
│  ─────────────────────────────────────────────────────                       │
│  da Vinci     [████████░░] 8.8   14M procedures; FDA cleared; multi-site   │
│  Hugo RAS     [█████░░░░░] 5.8   FDA cleared (uro); 30+ countries          │
│  OTTAVA       [███░░░░░░░] 3.2   De Novo pending; IDE studies only         │
│                                                                              │
│  ═══════════════════════════════════════════════════                         │
│  FINAL USL SCORES (weighted average):                                        │
│  ────────────────────────────────────                                        │
│  da Vinci     [███████░░░] 7.6   Level 7 — Advanced                        │
│  Hugo RAS     [████░░░░░░] 4.1   Level 4 — Developing                     │
│  OTTAVA       [██░░░░░░░░] 2.3   Level 2 — Exploratory                    │
│                                                                              │
│  Bar scale: each █ = 1.0 point (10 blocks = 10.0)                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## da Vinci (Xi / da Vinci 5)

**USL Score: 7.6 / 10.0 — Level 7 (Advanced)**

The da Vinci system by Intuitive Surgical is the most established surgical robot in the world, with approximately 14 million procedures performed and the largest open-source research ecosystem (dVRK) of any surgical robot. The da Vinci 5 (2024) introduced the first FDA-cleared force feedback in a surgical robot.

**Key USL strengths:**
- dVRK open-source research kit deployed at ~40 institutions worldwide
- ORBIT-Surgical provides 14 GPU-accelerated benchmark tasks
- Extensive autonomous subtask research (suturing, tissue manipulation)
- Force feedback (da Vinci 5) — first FDA-cleared surgical robot with haptics
- ~14 million procedures — largest clinical evidence base

**Key USL gaps:**
- Commercial system control is fully proprietary (dVRK is research only)
- No IEEE 3177-2024 alignment
- Federated learning infrastructure not yet available
- VLA model integration not demonstrated

**Open-source references:**
- [dVRK](https://github.com/jhu-dvrk/sawIntuitiveResearchKit) — Open-source da Vinci research kit (v2.3.1)
- [ORBIT-Surgical](https://github.com/orbit-surgical/orbit-surgical) — GPU-accelerated surgical tasks
- [SurRoL](https://github.com/med-air/SurRoL) — dVRK-compatible RL platform
- [SurgicalGym](https://github.com/SamuelSchmidgall/SurgicalGym) — GPU surgical simulation
- [OpenIGTLink](https://github.com/openigtlink/OpenIGTLink) — Open network protocol for IGT

---

## Hugo RAS

**USL Score: 4.1 / 10.0 — Level 4 (Developing)**

The Hugo RAS by Medtronic is a modular cart-based surgical robot system that received FDA clearance in December 2025 for urologic procedures. Its unique modular design (four independent arm carts) offers flexible OR positioning, and its standalone laparoscopic mode provides a surgical fallback.

**Key USL strengths:**
- Modular cart-based design — each arm independently positionable
- FDA cleared (December 2025) for oncology-relevant urologic procedures
- Standalone laparoscopic fallback mode (tower without console)
- Collision avoidance and intelligent instrument tracking
- Touch Surgery digital ecosystem for surgical planning

**Key USL gaps:**
- No open-source simulation models or research kit
- Proprietary control protocols — no public API
- Limited AI research community
- Limited to urologic indications (multi-specialty pending)

**References:**
- [Hugo RAS System](https://www.medtronic.com/en-us/healthcare-professionals/specialties/surgical-robotics/hugo-robotic-assisted-surgery.html) — Medtronic
- Introducing Hugo RAS for Gynecological Surgery (PMC9218341): DOI [10.3389/fonc.2022.898060](https://doi.org/10.3389/fonc.2022.898060)
- State of the Art in Robotic Surgery with Hugo RAS (PMC10456103): DOI [10.3390/jpm13081233](https://doi.org/10.3390/jpm13081233)
- Hugo vs da Vinci Comparison (J Robotic Surg 2024): DOI [10.1007/s11701-024-01838-5](https://doi.org/10.1007/s11701-024-01838-5)

---

## OTTAVA

**USL Score: 2.3 / 10.0 — Level 2 (Exploratory)**

The OTTAVA by Johnson & Johnson MedTech is a table-integrated surgical robot with a fundamentally different architecture — robotic arms built into the surgical table that stow underneath when not in use. Its unique "Twin Motion" allows patient repositioning during surgery without undocking. OTTAVA filed a De Novo submission with the FDA in January 2026.

**Key USL strengths:**
- Table-integrated design — zero OR footprint when arms stowed
- Twin Motion: patient repositioning without undocking (unique capability)
- Ethicon instrumentation ecosystem backed by J&J MedTech
- Potential advantages for multi-quadrant oncology procedures

**Key USL gaps:**
- No public simulation models, kinematics, or research platform
- Most technical specifications not yet publicly disclosed
- No FDA clearance yet (De Novo pending)
- No AI research community or published AI experiments
- No interoperability with existing surgical robot ecosystems

**References:**
- [OTTAVA FDA Submission (Jan 2026)](https://www.jnj.com/media-center/press-releases/johnson-johnson-submits-ottava-robotic-surgical-system-to-the-u-s-food-and-drug-administration) — J&J
- [OTTAVA First Cases](https://www.jnj.com/media-center/press-releases/johnson-johnson-medtech-announces-completion-of-first-cases-with-ottava-robotic-surgical-system) — J&J
- Upcoming Multi-Visceral Robotic Systems (PMC11615118): DOI [10.1007/s00464-024-11384-8](https://doi.org/10.1007/s00464-024-11384-8)

---

## Evaluated Cobots (Category: Collaborative Robots)

This initial USL evaluation covers three state-of-the-art open-source collaborative robot arms from different manufacturers, each with active ROS 2 support, MuJoCo Menagerie models, and potential oncology applications.

| Robot | Manufacturer | USL Score | USL Level | Band |
|-------|-------------|-----------|-----------|------|
| [Franka Emika Panda](#franka-emika-panda) | Franka Robotics | **7.4** | 7 (Advanced) | Advanced |
| [Kinova Gen3 7DoF](#kinova-gen3-7dof) | Kinova Robotics | **5.7** | 5 (Functional) | Intermediate |
| [UFACTORY xArm 7](#ufactory-xarm-7) | UFACTORY | **3.4** | 3 (Basic) | Foundational |

---

## Diagram 4: General Cobot Comparison

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

## Diagram 5: Technical Specifications — Cobots

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

## Diagram 6: USL Scoring Breakdown — Cobots

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
├── README.md                                     # This file
├── surgical/                                     # Surgical Robot Systems category
│   ├── surgical_usl_scoring.py                   # Surgical USL scoring engine
│   ├── davinci/
│   │   └── davinci_usl.py                        # da Vinci evaluation + tools
│   ├── hugo_ras/
│   │   └── hugo_ras_usl.py                       # Hugo RAS evaluation + tools
│   └── ottava/
│       └── ottava_usl.py                         # OTTAVA evaluation + tools
└── cobots/                                       # Collaborative Robots category
    ├── usl_scoring_framework.py                  # Cobot USL scoring engine
    ├── franka_panda/
    │   └── franka_panda_usl.py                   # Franka Panda evaluation + tools
    ├── kinova_gen3/
    │   └── kinova_gen3_usl.py                    # Kinova Gen3 evaluation + tools
    └── ufactory_xarm7/
        └── ufactory_xarm7_usl.py                # xArm 7 evaluation + tools
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

### Surgical Robot Specific References

5. **IEEE 3177-2024** — *Standard for a Modular Framework for a Robotically-Assisted Surgical System*. IEEE, approved December 2024. Defines hierarchical modular architecture with Execution, Perception, HMI, Navigation, Planning, and Safety modules.

6. **IEC 80601-2-77** — *Safety standard for robotically assisted surgical equipment*. Addresses basic safety and essential performance requirements for surgical robots.

7. **Levels of Autonomy in Surgical Robotics (LASR)** — Yang, G.-Z., et al. (2024). *npj Digital Medicine*. DOI: [10.1038/s41746-024-01102-y](https://doi.org/10.1038/s41746-024-01102-y). Systematic classification of surgical robot autonomy from Level 1 (Robot Assistance) through Level 5 (Full Autonomy).

8. **SAGES STARSS** — *Tool for Assessing Robotic Surgery Systems*. DOI: [10.1007/s00464-025-11897-w](https://doi.org/10.1007/s00464-025-11897-w). Standardized assessment tool for comparing robotic surgery systems.

9. **dVRK Software Architecture** — Kazanzides, P., et al. (2017). IEEE Conference. Describes cisst/SAW distributed real-time framework with ROS integration.

10. **OpenIGTLink** — Tokuda, J., et al. (2009). *Int J Med Robot Comput Assist Surg*, 5(4):423-34. DOI: [10.1002/rcs.274](https://doi.org/10.1002/rcs.274). Open network protocol for image-guided therapy environment.

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

### Run Surgical Robot Scoring Demo

```bash
python unification/usl/surgical/surgical_usl_scoring.py
```

### Evaluate Individual Surgical Robots

```bash
# da Vinci (Xi / da Vinci 5)
python unification/usl/surgical/davinci/davinci_usl.py

# Medtronic Hugo RAS
python unification/usl/surgical/hugo_ras/hugo_ras_usl.py

# Johnson & Johnson OTTAVA
python unification/usl/surgical/ottava/ottava_usl.py
```

### Run Cobot Scoring Demo

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
2. Implement a scoring engine adapted to the category's unique characteristics
3. Create a subdirectory for each evaluated robot with an evaluation module
4. Add the robots to this README with diagram sections updated
5. Validate across at least 2 simulation frameworks
6. Submit a PR with USL scores and supporting evidence

---

*Last updated: February 2026*
