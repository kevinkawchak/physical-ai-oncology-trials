# USL Humanoid Robot Evaluations

*Unification Standard Level — Category: Humanoid Robots (v1.6.0)*

---

## Diagram 1: USL Results — Humanoid Robots

```
+-------------------------------------------------------------------------+
|              USL RESULTS — Humanoid Robot Scores Explained              |
+-------------------------------------------------------------------------+
|                                                                         |
|  ATLAS (Electric) — USL 5.8 / 10.0  [Level 5: Functional]               |
|  WHY 5.8: Strongest simulation coverage (4 frameworks incl. Drake       |
|  open-source) and most advanced AI research (RL, IL, diffusion, VLA).   |
|  Held back by proprietary platform (no public SDK), zero healthcare     |
|  deployments, and limited cross-robot sharing outside Drake/ONNX.       |
|    Dim A: 7.0 — 4 frameworks, Drake open-source, GPU sim                |
|    Dim B: 7.4 — RL, IL, diffusion, VLA, LLM planning                    |
|    Dim C: 4.5 — Proprietary; sharing via Drake + ONNX only              |
|    Dim D: 4.2 — Safety cert potential, ISO 13482, no clinical pilots    |
|                                                                         |
|  DIGIT — USL 4.2 / 10.0  [Level 4: Developing]                          |
|  WHY 4.2: Best foundation model story (NVIDIA GR00T N1 primary          |
|  partner) and only humanoid with commercial deployment (Amazon).        |
|  Penalized for no healthcare experience, no ROS 2 interface, limited    |
|  hand dexterity, and no federated learning or clinical trial tooling.   |
|    Dim A: 5.8 — 3 frameworks, Isaac Lab + MuJoCo Menagerie official     |
|    Dim B: 5.4 — GR00T N1, VLA, RL, IL, LLM task planning                |
|    Dim C: 3.0 — Isaac Lab models + ONNX; no inter-org sharing           |
|    Dim D: 2.7 — Commercial deploy (Amazon) but zero healthcare use      |
|                                                                         |
|  OPTIMUS (Gen 2) — USL 3.6 / 10.0  [Level 3: Basic]                     |
|  WHY 3.6: Best hands (11-DOF tactile) and mass production potential     |
|  (sub-$20K target). Scores lowest because fully proprietary — no        |
|  public SDK, no simulation models, no peer-reviewed research, no        |
|  developer ecosystem. Cross-robot sharing is essentially zero.          |
|    Dim A: 3.4 — 2 frameworks (community models only)                    |
|    Dim B: 5.0 — End-to-end NN, RL, IL via Dojo (all internal)           |
|    Dim C: 1.5 — Fully proprietary; no external sharing                  |
|    Dim D: 4.4 — Audit trail, remote monitoring; no safety certs         |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## Diagram 2: Meaning — What Humanoid USL Scores Indicate

```
+-------------------------------------------------------------------------+
|         MEANING — What Humanoid USL Scores Tell Us                      |
+-------------------------------------------------------------------------+
|                                                                         |
|  OPENNESS DRIVES UNIFICATION                                            |
|  ─────────────────────────                                              |
|  Atlas leads because Boston Dynamics/TRI publish Drake (open-source     |
|  sim) and BDAII publishes peer-reviewed locomotion research. Digit      |
|  benefits from NVIDIA's open Isaac Lab models and MuJoCo Menagerie.     |
|  Optimus trails because Tesla shares nothing publicly.                  |
|                                                                         |
|       Open-source ecosystem size vs. USL score:                         |
|       Atlas  ████████████████░░░░  5.8  (Drake, 4 frameworks)           |
|       Digit  ██████████░░░░░░░░░░  4.2  (Isaac Lab + Menagerie)         |
|       Optimus ████████░░░░░░░░░░░  3.6  (community models only)         |
|                                                                         |
|  FOUNDATION MODELS ARE NOT ENOUGH                                       |
|  ────────────────────────────────                                       |
|  Digit has the strongest foundation model story (GR00T N1 primary       |
|  partner) yet scores below Atlas. Why? Foundation models improve        |
|  Dim B (AI integration) but cannot compensate for weak Dim C            |
|  (cross-robot sharing) and Dim D (clinical trial readiness).            |
|                                                                         |
|  HEALTHCARE GAP IS UNIVERSAL                                            |
|  ──────────────────────────                                             |
|  No evaluated humanoid has hospital deployment experience. All three    |
|  score below 5.0 on Dim D (clinical trial collaboration). This is       |
|  the single largest barrier to humanoid robots entering oncology        |
|  clinical trials — regulatory, safety, and clinical validation are      |
|  prerequisites that no manufacturer has yet addressed.                  |
|                                                                         |
|       Dim D scores (clinical readiness):                                |
|       Optimus ████░░░░░░  4.4  (audit trail, no certs)                  |
|       Atlas   ████░░░░░░  4.2  (ISO 13482 aligned, no pilots)           |
|       Digit   ██░░░░░░░░  2.7  (commercial only, no healthcare)         |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## Diagram 3: Impact — Humanoid Robots and the Future of Physical AI Oncology Trials

```
+-------------------------------------------------------------------------+
|     IMPACT — Humanoid Robots & Future of Physical AI Oncology Trials    |
+-------------------------------------------------------------------------+
|                                                                         |
|  NEAR-TERM (2026-2027): LOGISTICS FIRST                                 |
|  ──────────────────────────────────────                                 |
|  Humanoid robots will enter oncology settings via non-clinical          |
|  logistics tasks — supply transport, specimen delivery, pharmacy        |
|  restocking — before any patient-facing roles. Atlas and Digit are      |
|  closest due to locomotion maturity and payload capacity.               |
|                                                                         |
|       Supply transport      ──► Atlas (11 kg payload, dynamic ROM)      |
|       Specimen delivery     ──► Atlas / Digit (16 kg payload)           |
|       Pharmacy restocking   ──► Digit (GR00T autonomous nav)            |
|       Linen/waste transport ──► Optimus (4-hr battery, low cost)        |
|                                                                         |
|  MID-TERM (2027-2028): UNIFICATION ACCELERATION                         |
|  ───────────────────────────────────────────────                        |
|  As GR00T N1 and Drake mature, cross-robot policy transfer will         |
|  enable training on one humanoid and deploying on another. This is      |
|  the core USL value proposition — standardized interfaces mean          |
|  multi-site trials can mix humanoid platforms per site availability.    |
|                                                                         |
|       Drake (Atlas) ──ONNX──► Isaac Lab (Digit) ──GR00T──► Deploy       |
|       Training site A         Validation site B         Trial site C    |
|                                                                         |
|  LONG-TERM (2028+): INTEGRATED CLINICAL ROLES                           |
|  ─────────────────────────────────────────────                          |
|  Once ISO 13482 certification and hospital safety validation are        |
|  achieved, humanoid robots could take on assistive roles in oncology    |
|  wards — patient mobility support, equipment positioning, and           |
|  decontamination. Mass production (Optimus at sub-$20K) could make      |
|  multi-site deployment economically viable.                             |
|                                                                         |
|       Barrier removal timeline:                                         |
|       2026 ── ISO 13482 alignment ── Hospital pilot approval            |
|       2027 ── Multi-site logistics trials ── Safety data collection     |
|       2028 ── Assistive role certification ── Oncology ward deployment  |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## Evaluated Humanoid Robots (Category: Humanoid Robots)

This USL evaluation covers three bipedal humanoid robot systems from different manufacturers, each with potential for hospital logistics, patient transport, and assistive tasks in oncology clinical trials. Humanoid robots differ from cobots and surgical robots in their full-body bipedal locomotion (30-50+ DOF), whole-body coordination (locomotion + manipulation), and foundation model integration (GR00T, OpenVLA).

| Robot | Manufacturer | USL Score | USL Level | Band |
|-------|-------------|-----------|-----------|------|
| [Atlas (Electric)](#atlas-electric) | Boston Dynamics | **5.8** | 5 (Functional) | Intermediate |
| [Digit](#digit) | Agility Robotics | **4.2** | 4 (Developing) | Foundational |
| [Optimus (Gen 2)](#optimus-gen-2) | Tesla | **3.6** | 3 (Basic) | Foundational |

---

## Diagram 4: General Humanoid Robot Comparison

```
+-------------------------------------------------------------------------+
|         GENERAL COMPARISON — USL Category: Humanoid Robots              |
+-------------------------------------------------------------------------+
|                                                                         |
|  +----------------------+  +----------------------+  +------------------+
|  |  ATLAS (Electric)    |  |       DIGIT          |  | OPTIMUS (Gen 2)  |
|  |  (Boston Dynamics)   |  | (Agility Robotics)   |  |     (Tesla)      |
|  +----------------------+  +----------------------+  +------------------+
|  | Heritage: Pioneer    |  | Heritage: First      |  | Heritage: Tesla  |
|  |  in dynamic bipedal  |  |  humanoid in commer- |  |  FSD AI + mass   |
|  |  locomotion since    |  |  cial deployment     |  |  manufacturing   |
|  |  2013 (hydraulic)    |  |  (Amazon logistics)  |  |  scale           |
|  |                      |  |                      |  |                  |
|  | Ecosystem: Drake     |  | Ecosystem: NVIDIA    |  | Ecosystem:       |
|  |  (open-source sim),  |  |  GR00T N1 foundation |  |  Fully propri-   |
|  |  BDAII research,     |  |  model, Isaac Lab +  |  |  etary. No SDK,  |
|  |  Hyundai deployment  |  |  MuJoCo Menagerie    |  |  no open-source  |
|  |                      |  |                      |  |                  |
|  | Key Strength:        |  | Key Strength:        |  | Key Strength:    |
|  |  Most dynamic ROM,   |  |  GR00T + Isaac Lab   |  |  11-DOF hands +  |
|  |  4-framework sim,    |  |  sim-to-real pipeline|  |  mass production |
|  |  Drake open-source   |  |  + 16 kg payload     |  |  at sub-$20K     |
|  |                      |  |                      |  |                  |
|  | Oncology Focus:      |  | Oncology Focus:      |  | Oncology Focus:  |
|  |  Equipment handling, |  |  Supply tote delivery|  |  Pharmacy deliv- |
|  |  specimen delivery,  |  |  specimen courier,   |  |  ery, linen      |
|  |  decontamination     |  |  pharmacy restocking |  |  transport       |
|  |                      |  |                      |  |                  |
|  | USL Score: 5.8 ===== |  | USL Score: 4.2 ===   |  | USL Score: 3.6===|
|  | Level 5 — Functional |  | Level 4 — Developing |  | Level 3 — Basic  |
|  +----------------------+  +----------------------+  +------------------+
|                                                                         |
|  Legend: Each = ~ 1.2 points on the 1.0-10.0 USL scale                  |
+-------------------------------------------------------------------------+
```

---

## Diagram 5: Technical Specifications — Humanoid Robots

```
+-------------------------------------------------------------------------+
|       TECHNICAL SPECIFICATIONS — Humanoid Robot Comparison              |
+-------------------------------------------------------------------------+
|                                                                         |
|  Spec               Atlas (Electric)  Digit             Optimus (Gen2)  |
|  -----------------  ----------------  ----------------  ----------------|
|  Architecture       Bipedal electric  Bipedal backward  Bipedal full    |
|                     full-body         -bending knees    -body humanoid  |
|  Height (m)         ~1.50             ~1.75             ~1.73           |
|  Weight (kg)        ~89               ~65               ~57 <-- light   |
|  Body DOF           ~28               ~20               ~28             |
|  Hand DOF (each)    Custom EE         4-finger          11 <-- most     |
|  Walking Speed(m/s) 1.5               1.5               1.3             |
|  Payload (kg)       11                16 <-- highest    9               |
|  Battery Life (hrs) ~1.5              ~3.0              ~4.0 <-- most   |
|  Perception         Stereo + LiDAR    LiDAR + stereo    FSD cameras     |
|                     + IMU + F/T       + depth + IMU     + IMU + tactile |
|  Compute            Custom onboard    Jetson AGX Orin   Tesla HW4/SoC   |
|  Knee Design        Standard (hyper-  Backward-bending  Standard        |
|                     extended ROM)     (bird-inspired)                   |
|                                                                         |
|  Simulation / Open-Source Support:                                      |
|  ---------------------------------                                      |
|  Isaac Lab          Y (URDF/USD)     Y Official <--   ~ Community       |
|  MuJoCo             Y Community       Y Menagerie <-- ~ Community       |
|  Drake              Y Official <--   X                 X                |
|  Gazebo + ROS 2     ~ Community       ~ Community       X               |
|  Open-Source Code   ~ Drake (TRI)    ~ Isaac Lab model X None           |
|  Frameworks (#)     4                 3                 2               |
|                                                                         |
|  AI / Foundation Model Support:                                         |
|  ------------------------------                                         |
|  GR00T N1           ~ Potential       Y Primary <--   X                 |
|  VLA Compatible     Y                 Y                 X               |
|  LLM Task Planning  Y                 Y                 X               |
|  RL Locomotion      Y                 Y                 Y               |
|  Imitation Learning Y                 Y                 Y               |
|  Diffusion Policy   Y                 X                 X               |
|                                                                         |
|  Deployment / Clinical:                                                 |
|  ----------------------                                                 |
|  Commercial Deploy  X (Hyundai plan) Y Amazon <--    X (factory)        |
|  Hospital Pilot     Y Potential       X                 X               |
|  ISO 13482 Align    Y                 Y                 X               |
|  Manufacturing      Hyundai (limited) RoboFab (10K/yr) Tesla (millions) |
|                                                                         |
|  Y = Supported   ~ = Partial/Limited   X = Not available                |
+-------------------------------------------------------------------------+
```

---

## Diagram 6: USL Scoring Breakdown — Humanoid Robots

```
+-------------------------------------------------------------------------+
|         USL SCORING BREAKDOWN — Humanoid Robot Dimension Scores         |
+-------------------------------------------------------------------------+
|                                                                         |
|  Dimension A: Simulation Framework Switching (25% weight)               |
|  ------------------------------------------------                       |
|  Atlas(Elec)  [=======---] 7.0   4 frameworks, Drake open-source, GPU   |
|  Digit        [=====-----] 5.8   3 frameworks, Isaac Lab + Menagerie    |
|  Optimus(G2)  [===-------] 3.4   2 frameworks, community models only    |
|                                                                         |
|  Dimension B: Generative / Agentic AI Integration (25% weight)          |
|  --------------------------------------------------------               |
|  Atlas(Elec)  [=======---] 7.4   RL, IL, diffusion, VLA, LLM plan       |
|  Digit        [=====-----] 5.4   GR00T N1, VLA, RL, IL, LLM plan        |
|  Optimus(G2)  [=====-----] 5.0   End-to-end NN, RL, IL, Dojo train      |
|                                                                         |
|  Dimension C: Cross-Robot Progress Sharing (25% weight)                 |
|  ----------------------------------------------                         |
|  Atlas(Elec)  [====------] 4.5   Proprietary; Drake + ONNX sharing      |
|  Digit        [===-------] 3.0   Isaac Lab models + ONNX; no inter      |
|  Optimus(G2)  [=---------] 1.5   Fully proprietary; no sharing          |
|                                                                         |
|  Dimension D: Multi-Site Clinical Trial Collaboration (25% weight)      |
|  ---------------------------------------------------------              |
|  Atlas(Elec)  [====------] 4.2   Safety cert, ISO 13482, pilot poss     |
|  Digit        [==--------] 2.7   Commercial deploy but no healthcare    |
|  Optimus(G2)  [====------] 4.4   Audit trail, remote mon; no certs      |
|                                                                         |
|  ===================================================                    |
|  FINAL USL SCORES (weighted average):                                   |
|  ------------------------------------                                   |
|  Atlas(Elec)  [=====-----] 5.8   Level 5 — Functional                   |
|  Digit        [====------] 4.2   Level 4 — Developing                   |
|  Optimus(G2)  [===-------] 3.6   Level 3 — Basic                        |
|                                                                         |
|  Bar scale: each = = 1.0 point (10 blocks = 10.0)                       |
+-------------------------------------------------------------------------+
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

---

## Contributing

To add new humanoid robots or update existing evaluations:

1. Create a subdirectory under `humanoids/` for the new robot
2. Create an evaluation module adapted from existing humanoid evaluations
3. Score across all four USL dimensions (A–D) with humanoid-specific criteria
4. Validate across at least 2 simulation frameworks
5. Submit a PR with USL scores and supporting evidence

---

## Directory Structure

```
humanoids/
├── README.md                              # This file
├── usl_humanoid_scoring.py                # USL scoring engine (humanoids)
├── boston_dynamics_atlas/
│   └── boston_dynamics_atlas_usl.py        # Atlas Electric evaluation + tools
├── tesla_optimus/
│   └── tesla_optimus_usl.py               # Optimus Gen 2 evaluation + tools
└── agility_digit/
    └── agility_digit_usl.py               # Digit evaluation + tools
```

---

*Last updated: February 2026*
