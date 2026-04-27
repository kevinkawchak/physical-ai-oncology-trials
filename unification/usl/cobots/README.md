# USL Collaborative Robot (Cobot) Evaluations

*Unification Standard Level — Category: Collaborative Robots (v1.4.0)*

---

## Diagram 1: USL Results — Cobots

```
+-------------------------------------------------------------------------+
|              USL RESULTS — Cobot Scores Explained                       |
+-------------------------------------------------------------------------+
|                                                                         |
|  FRANKA EMIKA PANDA — USL 7.4 / 10.0  [Level 7: Advanced]               |
|  WHY 7.4: Largest open-source cobot ecosystem worldwide — official      |
|  models in MuJoCo Menagerie, Isaac Lab, and Gazebo; extensive VLA       |
|  and diffusion policy research (panda-gym, ORBIT-Surgical). Cross-      |
|  manufacturer ONNX policy transfer demonstrated. 1 kHz torque-sensor    |
|  control across all 7 joints. Only cobot with 5 framework support.      |
|    Dim A: 8.0 — 5 frameworks, GPU sim, full policy transfer             |
|    Dim B: 7.7 — VLA, diffusion, LLM planning, natural language          |
|    Dim C: 8.5 — Inter-org ONNX, skills library, sync framework          |
|    Dim D: 5.5 — Federated compatible, safety cert, ISO aligned          |
|                                                                         |
|  KINOVA GEN3 7DoF — USL 5.7 / 10.0  [Level 5: Functional]               |
|  WHY 5.7: Lightest 7-DOF cobot (8.2 kg) with integrated Intel           |
|  RealSense depth camera — ideal for bedside deployment. Official        |
|  MuJoCo Menagerie model and ros2_kortex package. Kortex API provides    |
|  versatile high/low-level control. Heritage in assistive robotics.      |
|    Dim A: 6.3 — 4 frameworks, partial GPU, MuJoCo + Gazebo official     |
|    Dim B: 5.2 — LLM planning, agentic AI, safety-constrained            |
|    Dim C: 5.8 — Intra-org ONNX, ROS 2 actions, Kortex sharing           |
|    Dim D: 5.5 — Clinical workflow ready, safety cert, ISO aligned       |
|                                                                         |
|  UFACTORY xARM 7 — USL 3.4 / 10.0  [Level 3: Basic]                     |
|  WHY 3.4: Most affordable 7-DOF cobot with built-in collision           |
|  detection and best environmental protection (IP51, 0-50C). Scores      |
|  lowest due to smallest research community, no published VLA/diffusion  |
|  experiments, lower control frequency (250 Hz), and no regulatory       |
|  pathway documentation.                                                 |
|    Dim A: 4.6 — 3 frameworks, basic models only                         |
|    Dim B: 3.2 — Basic generative + agentic only                         |
|    Dim C: 3.8 — Intra-org SDK, basic ONNX export                        |
|    Dim D: 2.1 — Remote monitoring only, no certifications               |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## Diagram 2: Meaning — What Cobot USL Scores Indicate

```
+-------------------------------------------------------------------------+
|         MEANING — What Cobot USL Scores Tell Us                         |
+-------------------------------------------------------------------------+
|                                                                         |
|  RESEARCH COMMUNITY SIZE CORRELATES WITH USL SCORE                      |
|  ─────────────────────────────────────────────────                      |
|  Franka Panda has the largest open-source cobot research community      |
|  (1000+ GitHub stars on franka_ros2, panda-gym, libfranka). This        |
|  directly enables higher scores on Dim A (more framework support),      |
|  Dim B (more AI research), and Dim C (more sharing infrastructure).     |
|                                                                         |
|       GitHub ecosystem size vs. USL score:                              |
|       Franka  ========================  7.4  (1000+ stars, 5 fwks)      |
|       Kinova  ===============           5.7  (500+ stars, 4 fwks)       |
|       xArm    =========                3.4  (300+ stars, 3 fwks)        |
|                                                                         |
|  HARDWARE EXCELLENCE != UNIFICATION READINESS                           |
|  ────────────────────────────────────────────                           |
|  The xArm 7 has the best environmental protection (IP51) and built-in   |
|  collision detection, yet scores lowest. Hardware quality matters for   |
|  deployment but USL measures SOFTWARE interoperability: simulation      |
|  framework coverage, AI integration depth, and sharing infrastructure.  |
|                                                                         |
|  CLINICAL TRIAL READINESS LAGS ACROSS ALL COBOTS                        |
|  ────────────────────────────────────────────────                       |
|  Even Franka (7.4 overall) scores only 5.5 on Dim D. No cobot has       |
|  multi-site clinical trial deployment. This reflects a field-wide       |
|  gap: cobots are well-established in research but clinical trial        |
|  infrastructure (federated learning, regulatory docs, audit trails)     |
|  remains underdeveloped.                                                |
|                                                                         |
|       Dim D scores (clinical readiness):                                |
|       Franka  =====-----  5.5  (federated compat, ISO aligned)          |
|       Kinova  =====-----  5.5  (clinical workflow, ISO aligned)         |
|       xArm    ==--------  2.1  (remote monitoring only)                 |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## Diagram 3: Impact — Cobots and the Future of Physical AI Oncology Trials

```
+-------------------------------------------------------------------------+
|     IMPACT — Cobots & Future of Physical AI Oncology Trials             |
+-------------------------------------------------------------------------+
|                                                                         |
|  NEAR-TERM (2026-2027): LAB AUTOMATION AND SURGICAL ASSISTANCE          |
|  ────────────────────────────────────────────────────────────           |
|  Cobots will enter oncology trials first through lab automation         |
|  (sample handling, pipetting, vial management) and surgical             |
|  assistance (instrument handoff, retraction). Franka leads due to       |
|  its extensive RL/IL research for manipulation tasks.                   |
|                                                                         |
|       Lab automation        --> Franka (most RL research, 1 kHz)        |
|       Bedside assistance    --> Kinova (lightest, integrated vision)    |
|       Cost-effective sites  --> xArm (affordable, IP51 rated)           |
|                                                                         |
|  MID-TERM (2027-2028): CROSS-COBOT POLICY TRANSFER                      |
|  ──────────────────────────────────────────────────                     |
|  USL-standardized ONNX export will enable policies trained on           |
|  Franka to deploy on Kinova or xArm at different trial sites.           |
|  Multi-site trials can use different cobot hardware per site            |
|  while maintaining behavioral consistency through shared policies.      |
|                                                                         |
|       Franka (train) --ONNX--> Kinova (validate) --ONNX--> xArm         |
|       Site A (Stanford)        Site B (Mayo)         Site C (NHS)       |
|                                                                         |
|  LONG-TERM (2028+): COBOT-SURGICAL ROBOT COOPERATION                    |
|  ─────────────────────────────────────────────────────                  |
|  Cobots will work alongside surgical robots in the OR — handling        |
|  instruments, managing specimens, and supporting the surgical team.     |
|  USL enables unified interfaces between cobot (Franka) and surgical     |
|  robot (dVRK) systems sharing the same clinical workspace.              |
|                                                                         |
|       Surgical robot (dVRK) <--unified interface--> Cobot (Franka)      |
|       Tumor resection                               Instrument handoff  |
|       Suturing                                      Specimen handling   |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## Evaluated Cobots (Category: Collaborative Robots)

This USL evaluation covers three state-of-the-art open-source collaborative robot arms from different manufacturers, each with active ROS 2 support, MuJoCo Menagerie models, and potential oncology applications.

| Robot | Manufacturer | USL Score | USL Level | Band |
|-------|-------------|-----------|-----------|------|
| [Franka Emika Panda](#franka-emika-panda) | Franka Robotics | **7.4** | 7 (Advanced) | Advanced |
| [Kinova Gen3 7DoF](#kinova-gen3-7dof) | Kinova Robotics | **5.7** | 5 (Functional) | Intermediate |
| [UFACTORY xArm 7](#ufactory-xarm-7) | UFACTORY | **3.4** | 3 (Basic) | Foundational |

---

## Diagram 4: General Cobot Comparison

```
+-------------------------------------------------------------------------+
|               GENERAL COBOT COMPARISON — USL Category: Cobots           |
+-------------------------------------------------------------------------+
|                                                                         |
|  +----------------------+  +----------------------+  +------------------+
|  |   FRANKA EMIKA PANDA |  |   KINOVA GEN3 7DoF   |  |  UFACTORY xARM 7 |
|  |   (Franka Robotics)  |  |  (Kinova Robotics)   |  |    (UFACTORY)    |
|  +----------------------+  +----------------------+  +------------------+
|  | Heritage: Research   |  | Heritage: Assistive  |  | Heritage: Cost-  |
|  |  robotics leader     |  |  & rehabilitation    |  |  effective cobot |
|  |                      |  |                      |  |                  |
|  | Ecosystem: Largest   |  | Ecosystem: Strong    |  | Ecosystem:       |
|  |  open-source cobot   |  |  ROS 2 + Kortex API  |  |  Growing Python  |
|  |  community worldwide |  |  clinical research   |  |  SDK community   |
|  |                      |  |                      |  |                  |
|  | Key Strength:        |  | Key Strength:        |  | Key Strength:    |
|  |  Most RL/AI research |  |  Lightest arm (8.2kg)|  |  Most affordable |
|  |  papers of any cobot |  |  + integrated vision |  |  7-DOF + built-  |
|  |                      |  |                      |  |  in collision    |
|  | Primary Application: |  | Primary Application: |  | Primary App:     |
|  |  Surgical assistance |  |  Bedside care &      |  |  Lab automation  |
|  |  & lab automation    |  |  patient interaction |  |  & prototyping   |
|  |                      |  |                      |  |                  |
|  | USL Score: 7.4 ======|  | USL Score: 5.7 ====  |  | USL Score: 3.4== |
|  | Level 7 — Advanced   |  | Level 5 — Functional |  | Level 3 — Basic  |
|  +----------------------+  +----------------------+  +------------------+
|                                                                         |
|  Legend: Each = ~ 1.2 points on the 1.0-10.0 USL scale                  |
+-------------------------------------------------------------------------+
```

---

## Diagram 5: Technical Specifications — Cobots

```
+-------------------------------------------------------------------------+
|           TECHNICAL SPECIFICATIONS — Side-by-Side Comparison            |
+-------------------------------------------------------------------------+
|                                                                         |
|  Spec               Franka Panda      Kinova Gen3       xArm 7          |
|  -----------------  ----------------  ----------------  ----------------|
|  DOF                7                 7                 7               |
|  Payload (kg)       3.0               4.0 <-- highest   3.5             |
|  Reach (mm)         855               902 <-- highest   700             |
|  Repeatability (mm) +/-0.1            +/-0.1            +/-0.1          |
|  Weight (kg)        18.0              8.2 <-- lightest  11.2            |
|  Control Freq (Hz)  1000 <-- highest  1000 <-- highest  250 (500 max)   |
|  Torque Sensors     7 joints          7 joints          7 joints        |
|  IP Rating          IP30              IP22              IP51 <-- highest|
|  Temp Range (C)     5-40              0-40              0-50 <-- widest |
|  Integrated Vision  No                Yes <-- unique    No              |
|  Collision Detect   External          External          Built-in <--    |
|                                                                         |
|  Simulation Frameworks Supported:                                       |
|  ---------------------------------                                      |
|  MuJoCo Menagerie   Y Official        Y Official        Y Official      |
|  NVIDIA Isaac Lab   Y Official        ~ Community       ~ Community     |
|  Gazebo + ROS 2     Y Official        Y Official        Y Official      |
|  PyBullet           Y Community       ~ Community       ~ Community     |
|  Frameworks (#)     5                 4                 3               |
|                                                                         |
|  Model Formats Available:                                               |
|  -------------------------                                              |
|  URDF               Y                 Y                 Y               |
|  MJCF               Y                 Y                 Y               |
|  SDF                Y                 Y                 Y               |
|  USD                Y                 ~                 ~               |
|  Xacro              Y                 Y                 Y               |
|                                                                         |
|  ROS 2 Package      franka_ros2       ros2_kortex       xarm_ros2       |
|  API / SDK          libfranka         Kortex API        xArm Python SDK |
|  GitHub Stars (est) 1000+             500+              300+            |
|                                                                         |
|  Y = Official support   ~ = Community/partial   X = Not available       |
+-------------------------------------------------------------------------+
```

---

## Diagram 6: USL Scoring Breakdown — Cobots

```
+-------------------------------------------------------------------------+
|              USL SCORING BREAKDOWN — Dimension-by-Dimension             |
+-------------------------------------------------------------------------+
|                                                                         |
|  Dimension A: Simulation Framework Switching (25% weight)               |
|  ------------------------------------------------                       |
|  Franka Panda  [========--] 8.0   5 frameworks, GPU sim, full transfer  |
|  Kinova Gen3   [======----] 6.3   4 frameworks, partial GPU, MuJoCo+    |
|  xArm 7        [====------] 4.6   3 frameworks, basic models only       |
|                                                                         |
|  Dimension B: Generative / Agentic AI Integration (25% weight)          |
|  --------------------------------------------------------               |
|  Franka Panda  [=======---] 7.7   VLA, diffusion, LLM planning, NL      |
|  Kinova Gen3   [=====-----] 5.2   LLM planning, agentic, safety         |
|  xArm 7        [===-------] 3.2   Basic generative + agentic only       |
|                                                                         |
|  Dimension C: Cross-Robot Progress Sharing (25% weight)                 |
|  ----------------------------------------------                         |
|  Franka Panda  [========--] 8.5   Inter-org, ONNX, skills, sync         |
|  Kinova Gen3   [=====-----] 5.8   Intra-org, ONNX, ROS 2 actions        |
|  xArm 7        [===-------] 3.8   Intra-org SDK, basic ONNX             |
|                                                                         |
|  Dimension D: Multi-Site Clinical Trial Collaboration (25% weight)      |
|  ---------------------------------------------------------              |
|  Franka Panda  [=====-----] 5.5   Federated compat, safety cert, ISO    |
|  Kinova Gen3   [=====-----] 5.5   Clinical workflow, safety cert, ISO   |
|  xArm 7        [==--------] 2.1   Remote monitoring only                |
|                                                                         |
|  ===================================================                    |
|  FINAL USL SCORES (weighted average):                                   |
|  ------------------------------------                                   |
|  Franka Panda  [=======---] 7.4   Level 7 — Advanced                    |
|  Kinova Gen3   [=====-----] 5.7   Level 5 — Functional                  |
|  xArm 7        [===-------] 3.4   Level 3 — Basic                       |
|                                                                         |
|  Bar scale: each = = 1.0 point (10 blocks = 10.0)                       |
+-------------------------------------------------------------------------+
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
- Best environmental protection (IP51) and temperature range (0-50 C)
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

## Quick Start

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

To add new cobots or update existing evaluations:

1. Create a subdirectory under `cobots/` for the new robot
2. Create an evaluation module adapted from existing cobot evaluations
3. Score across all four USL dimensions (A-D) with cobot-specific criteria
4. Validate across at least 2 simulation frameworks
5. Submit a PR with USL scores and supporting evidence

---

## Directory Structure

```
cobots/
├── README.md                              # This file
├── usl_scoring_framework.py               # Core USL scoring engine (cobots)
├── franka_panda/
│   └── franka_panda_usl.py               # Franka Panda evaluation + tools
├── kinova_gen3/
│   └── kinova_gen3_usl.py                # Kinova Gen3 evaluation + tools
└── ufactory_xarm7/
    └── ufactory_xarm7_usl.py             # xArm 7 evaluation + tools
```

---

*Last updated: February 2026*
