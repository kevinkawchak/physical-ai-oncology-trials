# USL Surgical Robot Evaluations

*Unification Standard Level — Category: Surgical Robots (v1.5.0)*

---

## Diagram 1: USL Results — Surgical Robots

```
+-------------------------------------------------------------------------+
|              USL RESULTS — Surgical Robot Scores Explained               |
+-------------------------------------------------------------------------+
|                                                                         |
|  DA VINCI (dVRK) — USL 7.1 / 10.0  [Level 7: Advanced]                 |
|  WHY 7.1: Uniquely open-source surgical robot ecosystem — dVRK (JHU,    |
|  45+ institutions), ORBIT-Surgical (GPU sim), SurRoL (benchmarks),      |
|  AMBF (deformable tissue). 5 simulation frameworks, extensive VLA/      |
|  diffusion/IL/RL research, and FDA-cleared with 14M+ procedures.        |
|  No other surgical robot has comparable research infrastructure.         |
|    Dim A: 7.5 — 5 frameworks, GPU sim, tissue deformation modeling     |
|    Dim B: 7.2 — VLA, diffusion, IL, RL, surgical video AI             |
|    Dim C: 6.8 — Open-source dVRK, ONNX, ROS 2, JIGSAWS dataset       |
|    Dim D: 7.0 — FDA cleared, 9000+ systems, 69 countries              |
|                                                                         |
|  HUGO RAS — USL 4.5 / 10.0  [Level 4: Developing]                      |
|  WHY 4.5: Medtronic ecosystem integration (Touch Surgery Enterprise,    |
|  StealthStation, O-arm) provides strong clinical infrastructure.         |
|  Penalized for fully proprietary platform — no open-source code,        |
|  no simulation environments, no FDA clearance yet. Research community   |
|  access is limited to Medtronic partners only.                          |
|    Dim A: 4.5 — 2 frameworks, limited open-source simulation           |
|    Dim B: 4.0 — Touch Surgery video AI + phase recognition             |
|    Dim C: 3.5 — Intra-org via Medtronic ecosystem only                 |
|    Dim D: 5.8 — CE marked, EXPAND trials, clinical audit trail         |
|                                                                         |
|  VERSIUS — USL 3.4 / 10.0  [Level 3: Basic]                            |
|  WHY 3.4: Lightest arms (~10 kg), thinnest instruments (5 mm), and     |
|  most portable surgical robot. Scores lowest because proprietary        |
|  with minimal open-source presence, smallest research community,        |
|  no published AI/ML research, and no FDA clearance.                     |
|    Dim A: 3.2 — 2 frameworks, no open-source simulation                |
|    Dim B: 2.8 — Basic video AI, limited published research             |
|    Dim C: 2.5 — Intra-org only, no open standards or sharing           |
|    Dim D: 5.2 — CE marked, 350+ hospitals, 25+ countries               |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## Diagram 2: Meaning — What Surgical USL Scores Indicate

```
+-------------------------------------------------------------------------+
|         MEANING — What Surgical USL Scores Tell Us                      |
+-------------------------------------------------------------------------+
|                                                                         |
|  OPEN-SOURCE IS THE DECISIVE FACTOR                                     |
|  ──────────────────────────────────                                     |
|  The dVRK scores nearly double Hugo/Versius because it is the ONLY      |
|  surgical robot with a public open-source research platform. The        |
|  dVRK ecosystem (45+ institutions) created ORBIT-Surgical, SurRoL,     |
|  AMBF, and hundreds of published papers — an infrastructure that        |
|  proprietary platforms cannot replicate.                                 |
|                                                                         |
|       dVRK research papers vs. Hugo + Versius combined:                 |
|       dVRK     ================================  500+ papers            |
|       Hugo     =====                             ~50 papers             |
|       Versius  ===                               ~30 papers             |
|                                                                         |
|  CLINICAL DEPLOYMENT != UNIFICATION READINESS                           |
|  ────────────────────────────────────────────                            |
|  Hugo and Versius are clinically deployed (CE marked, active trials)    |
|  yet score below 5.0 overall. Clinical deployment improves Dim D        |
|  but cannot compensate for weak simulation (Dim A), AI research         |
|  (Dim B), and cross-robot sharing (Dim C). Unification requires         |
|  all four dimensions.                                                    |
|                                                                         |
|  GPU SIMULATION IS A DIFFERENTIATOR                                     |
|  ──────────────────────────────────                                     |
|  ORBIT-Surgical provides GPU-accelerated surgical RL training via       |
|  Isaac Lab — exclusively for dVRK. This enables 4096x parallel          |
|  environments for policy training, reducing training time from days     |
|  to hours. No other surgical robot has GPU simulation support.          |
|                                                                         |
|       GPU sim availability:                                             |
|       dVRK    [Y] ORBIT-Surgical (Isaac Lab) — 4096x parallel          |
|       Hugo    [X] No GPU simulation available                           |
|       Versius [X] No GPU simulation available                           |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## Diagram 3: Impact — Surgical Robots and the Future of Physical AI Oncology Trials

```
+-------------------------------------------------------------------------+
|     IMPACT — Surgical Robots & Future of Physical AI Oncology Trials    |
+-------------------------------------------------------------------------+
|                                                                         |
|  NEAR-TERM (2026-2027): dVRK AS RESEARCH BACKBONE                       |
|  ─────────────────────────────────────────────────                       |
|  The dVRK's open-source ecosystem will continue driving AI research     |
|  in surgical oncology. ORBIT-Surgical enables rapid prototyping of      |
|  autonomous surgical skills (needle insertion, tissue retraction,       |
|  suturing) that can later transfer to clinical-grade platforms.         |
|                                                                         |
|       Research (dVRK) ──policy transfer──> Clinical (da Vinci Xi/5)     |
|       ORBIT-Surgical      ONNX export       Intuitive production sys   |
|                                                                         |
|  MID-TERM (2027-2028): CROSS-PLATFORM SURGICAL AI                       |
|  ─────────────────────────────────────────────────                       |
|  As Hugo and Versius develop simulation capabilities, USL-driven        |
|  standardization will enable cross-platform surgical skill transfer.    |
|  A suturing policy trained on dVRK could validate on Hugo simulation   |
|  and deploy on Versius hardware — the multi-site trial model.          |
|                                                                         |
|       dVRK (train) ──> Hugo (validate) ──> Versius (deploy)            |
|       Site A (JHU)     Site B (Medtronic)   Site C (NHS)               |
|                                                                         |
|  LONG-TERM (2028+): AUTONOMOUS SURGICAL ONCOLOGY                        |
|  ────────────────────────────────────────────────                        |
|  Increasing levels of surgical autonomy (Levels 1-3 per Yang et al.)   |
|  will emerge for specific oncology subtasks. USL ensures that           |
|  autonomous capabilities developed on one platform can be validated    |
|  and certified across platforms for multi-site regulatory approval.    |
|                                                                         |
|       Autonomy Level 1: Robot assistance (current — all platforms)      |
|       Autonomy Level 2: Task autonomy (2027 — dVRK research)           |
|       Autonomy Level 3: Conditional autonomy (2028+ — FDA pathway)     |
|                                                                         |
+-------------------------------------------------------------------------+
```

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
+-------------------------------------------------------------------------+
|          GENERAL COMPARISON — USL Category: Surgical Robots             |
+-------------------------------------------------------------------------+
|                                                                         |
|  +----------------------+  +----------------------+  +------------------+
|  |   DA VINCI (dVRK)    |  |     HUGO RAS         |  |    VERSIUS       |
|  | (Intuitive Surgical) |  |    (Medtronic)       |  |  (CMR Surgical)  |
|  +----------------------+  +----------------------+  +------------------+
|  | Heritage: Pioneer    |  | Heritage: Medtronic  |  | Heritage: UK     |
|  |  of robotic surgery  |  |  medical device      |  |  startup focused |
|  |  since 1999          |  |  ecosystem           |  |  on portability  |
|  |                      |  |                      |  |                  |
|  | Ecosystem: Largest   |  | Ecosystem: Touch     |  | Ecosystem:       |
|  |  open-source via     |  |  Surgery Enterprise  |  |  Proprietary     |
|  |  dVRK (JHU, 45+     |  |  AI video analytics  |  |  with growing    |
|  |  institutions)       |  |                      |  |  clinical data   |
|  |                      |  |                      |  |                  |
|  | Key Strength:        |  | Key Strength:        |  | Key Strength:    |
|  |  ORBIT-Surgical GPU  |  |  Modular cart arms + |  |  Lightest arms   |
|  |  sim + 14M clinical  |  |  Medtronic OR stack  |  |  (~10 kg) +      |
|  |  procedures          |  |  integration         |  |  portable across |
|  |                      |  |                      |  |  operating rooms |
|  | Oncology Focus:      |  | Oncology Focus:      |  | Oncology Focus:  |
|  |  Prostatectomy,      |  |  Colectomy, gynecol- |  |  Gynecologic &   |
|  |  nephrectomy,        |  |  ogic, and urologic  |  |  colorectal      |
|  |  lobectomy           |  |  oncology            |  |  oncology        |
|  |                      |  |                      |  |                  |
|  | USL Score: 7.1 ======|  | USL Score: 4.5 ====  |  | USL Score: 3.4== |
|  | Level 7 — Advanced   |  | Level 4 — Developing |  | Level 3 — Basic  |
|  +----------------------+  +----------------------+  +------------------+
|                                                                         |
|  Legend: Each = ~ 1.2 points on the 1.0-10.0 USL scale                  |
+-------------------------------------------------------------------------+
```

---

## Diagram 5: Technical Specifications — Surgical Robots

```
+-------------------------------------------------------------------------+
|       TECHNICAL SPECIFICATIONS — Surgical Robot Comparison              |
+-------------------------------------------------------------------------+
|                                                                         |
|  Spec               da Vinci (dVRK)   Hugo RAS          Versius         |
|  -----------------  ----------------  ----------------  ----------------|
|  Architecture       Single cart       Modular carts     Modular carts   |
|  Instrument Arms    3 PSMs            3 arms            3 arms (max 4)  |
|  Camera Arms        1 ECM             1 arm             1 arm           |
|  DOF per Arm        7 + grip          7 + grip          7 + grip        |
|  Instrument dia.    5/8 mm            8 mm              5 mm <-- thin  |
|  Arm Weight (kg)    ~65 (full cart)   38 (per cart)     10 <-- lightest |
|  Console Type       Immersive         Open              Open            |
|  Tremor Filtering   Yes               Yes               Yes             |
|  Motion Scaling     2:1, 3:1, 5:1    Yes               Yes             |
|  Control Freq (Hz)  1000              1000              1000            |
|  Stereo Vision      Yes               3D HD             3D HD           |
|  Portability        Fixed             Semi-portable     Portable <--   |
|                                                                         |
|  Regulatory Status:                                                     |
|  -----------------                                                      |
|  FDA Cleared        Y <-- only one    X (pending)       X (pending)    |
|  CE Marked          Y                 Y                 Y               |
|  Countries          69                20+               25+             |
|  Systems Installed  9,000+ <-- most   500+              350+            |
|  Procedures Done    14M+ <-- most     50,000+           30,000+         |
|                                                                         |
|  Simulation / Open-Source Support:                                      |
|  ---------------------------------                                      |
|  Open-Source Code   Y dVRK (BSD)     X                 X               |
|  ORBIT-Surgical     Y <-- unique     X                 X               |
|  SurRoL Benchmark   Y <-- unique     X                 X               |
|  AMBF Simulator     Y <-- unique     X                 X               |
|  Gazebo + ROS 2     Y                ~                 ~               |
|  MuJoCo             Y                X                 ~               |
|  GPU Sim (Isaac)    Y                ~                 X               |
|                                                                         |
|  AI Research Ecosystem:                                                 |
|  ---------------------                                                  |
|  Surgical Video AI  Y JIGSAWS+       Y Touch Surgery   ~ Limited      |
|  RL / IL Research   Y Extensive      X                 X               |
|  VLA / Diffusion    Y <-- unique     X                 X               |
|  Phase Recognition  Y                Y                 ~               |
|                                                                         |
|  Y = Supported   ~ = Partial/Limited   X = Not available               |
+-------------------------------------------------------------------------+
```

---

## Diagram 6: USL Scoring Breakdown — Surgical Robots

```
+-------------------------------------------------------------------------+
|         USL SCORING BREAKDOWN — Surgical Robot Dimension Scores         |
+-------------------------------------------------------------------------+
|                                                                         |
|  Dimension A: Simulation Framework Switching (25% weight)               |
|  ------------------------------------------------                      |
|  da Vinci(dVRK) [=======---] 7.5   5 frameworks, GPU sim, tissue def  |
|  Hugo RAS       [====------] 4.5   2 frameworks, limited open-source  |
|  Versius        [===-------] 3.2   2 frameworks, no open-source sim   |
|                                                                         |
|  Dimension B: Generative / Agentic AI Integration (25% weight)          |
|  --------------------------------------------------------              |
|  da Vinci(dVRK) [=======---] 7.2   VLA, diffusion, IL, RL, video AI  |
|  Hugo RAS       [====------] 4.0   Touch Surgery video AI + phase     |
|  Versius        [==--------] 2.8   Basic video AI, limited research   |
|                                                                         |
|  Dimension C: Cross-Robot Progress Sharing (25% weight)                 |
|  ----------------------------------------------                         |
|  da Vinci(dVRK) [======----] 6.8   Open-source, ONNX, ROS 2, JIGSAWS|
|  Hugo RAS       [===-------] 3.5   Intra-org via Medtronic ecosystem |
|  Versius        [==--------] 2.5   Intra-org only, no open standards |
|                                                                         |
|  Dimension D: Multi-Site Clinical Trial Collaboration (25% weight)      |
|  ---------------------------------------------------------             |
|  da Vinci(dVRK) [=======---] 7.0   FDA cleared, 9000+ systems, certs |
|  Hugo RAS       [=====-----] 5.8   CE marked, EXPAND trials, audit   |
|  Versius        [=====-----] 5.2   CE marked, 350+ hospitals, certs  |
|                                                                         |
|  ===================================================                    |
|  FINAL USL SCORES (weighted average):                                   |
|  ------------------------------------                                   |
|  da Vinci(dVRK) [=======---] 7.1   Level 7 — Advanced                |
|  Hugo RAS       [====------] 4.5   Level 4 — Developing              |
|  Versius        [===-------] 3.4   Level 3 — Basic                   |
|                                                                         |
|  Bar scale: each = = 1.0 point (10 blocks = 10.0)                      |
+-------------------------------------------------------------------------+
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

## Quick Start

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

---

## Contributing

To add new surgical robots or update existing evaluations:

1. Create a subdirectory under `surgical/` for the new robot
2. Create an evaluation module adapted from existing surgical evaluations
3. Score across all four USL dimensions (A–D) with surgical-specific criteria
4. Validate across at least 2 simulation frameworks
5. Submit a PR with USL scores and supporting evidence

---

## Directory Structure

```
surgical/
├── README.md                              # This file
├── usl_surgical_scoring.py                # USL scoring engine (surgical)
├── intuitive_davinci/
│   └── intuitive_davinci_usl.py           # da Vinci (dVRK) evaluation + tools
├── medtronic_hugo/
│   └── medtronic_hugo_usl.py              # Hugo RAS evaluation + tools
└── cmr_versius/
    └── cmr_versius_usl.py                 # Versius evaluation + tools
```

---

*Last updated: February 2026*
