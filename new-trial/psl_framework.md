# Physical AI Standard Level (PSL) Framework

Released on 23 March 2026
CEO Kevin Kawchak, ChemicalQDevice

The original CFR documents are in the public domain. The original ICH document
is copyrighted and may be used, reproduced, incorporated into other works,
adapted, modified, translated or distributed under a public license. This
current work is not endorsed or sponsored by CFR, ICH, or FDA; and was adapted
using Claude Code Opus 4.6.

## 1. Overview

The Physical AI Standard Level (PSL) is a scoring framework that evaluates
each of the 10 robot types deployed in on-demand Physical AI oncology clinical
trials across three equally weighted regulatory dimensions. PSL measures how
close each robot type is to ideal clinical trial performance by assessing
omniscience, omnipresence, and omnipotence.

PSL complements the Unification Standard Level (USL) framework (Kawchak, 2026;
DOI: 10.5281/zenodo.18778220). While USL evaluates robot unification readiness
across simulation switching, AI integration, cross-robot sharing, and multi-site
collaboration (a technical interoperability focus), PSL evaluates robots on
their clinical trial omniscience, omnipresence, and omnipotence (a clinical
performance focus).

## 2. The Three PSL Dimensions

Each robot type receives a score from 0.0 to 10.0 (0.1 increments) on each
of the following dimensions.

### 2.1 Dimension A - Omniscient (Complete Knowledge)

Regulatory basis: ICH E6(R3) Adaption (DOI: 10.5281/zenodo.18973368)

Evaluates the robot's ability to know everything relevant to the clinical
trial. Scoring criteria include:

- Full patient data awareness and real-time access
- Real-time sensor fusion across all modalities
- Digital twin synchronization fidelity and prediction accuracy
- AI model knowledge breadth across oncology domains
- Cross-framework validation coverage (Isaac Lab, MuJoCo, Gazebo, PyBullet)
- Complete audit trail awareness per ICH E6(R3) Section 4.2
- Federated learning data access and aggregation capability
- Adverse event detection sensitivity per ICH E6(R3) Section 2.10
- Comprehensive documentation awareness per ICH E6(R3) Appendix C
- Regulatory compliance knowledge per ICH E6(R3) Section 1.1

A score of 10.0 means the robot has perfect, instantaneous knowledge of all
patient states, all sensor data, all trial parameters, all regulatory
requirements, and all concurrent activities at all times.

### 2.2 Dimension B - Omnipresent (Present Everywhere at Once)

Regulatory basis: 21 CFR Part 50 Adaption (DOI: 10.5281/zenodo.19040707)

Evaluates the robot's ability to be functionally present wherever needed.
Scoring criteria include:

- Multi-patient simultaneous coverage capability
- Zero transition time between stations
- Digital twin presence across all patient models
- Federated presence across all consortium sites
- Informed consent process availability at any moment per 21 CFR 50.25
- Pediatric and adult ward simultaneous coverage per 21 CFR 50 Subpart D
- Cybersecurity monitoring ubiquity per 21 CFR 50 Subpart C
- Audit trail presence across all data streams per 21 CFR 50.27
- Emergency response availability across all locations
- Pre-procedure safety matrix accessibility per 21 CFR 50.30

A score of 10.0 means the robot can serve every patient at every location
simultaneously with zero latency.

### 2.3 Dimension C - Omnipotent (Ability to Do Anything)

Regulatory basis: 21 CFR Part 312 Adaption (DOI: 10.5281/zenodo.19057628)

Evaluates the robot's ability to perform any needed action. Scoring criteria
include:

- Full procedural capability range for assigned clinical tasks
- Maximum force and precision envelope utilization
- Complete autonomy levels (assistive through fully autonomous)
- Investigational drug administration support per 21 CFR 312.23
- Adverse event intervention capacity per 21 CFR 312.32
- Emergency response capability per 21 CFR 312.42
- IND compliance fulfillment per 21 CFR 312.40
- Expanded access support per 21 CFR 312.300
- Annual reporting automation per 21 CFR 312.33
- Safety event remediation capability per 21 CFR 312.32

A score of 10.0 means the robot can perform any clinical action perfectly,
instantly, and without limitation.

## 3. Scoring Methodology

### 3.1 Per-Robot PSL Score

Per-Robot PSL = (Dimension A + Dimension B + Dimension C) / 3

Reported on the 0.0 to 10.0 scale with 0.1 increments.

### 3.2 Cumulative Site PSL Score

Cumulative PSL = Sum of all 10 robot type PSL scores

Reported on the 0.0 to 100.0 scale with 0.1 increments. A score of 100.0
means all 10 robot types are each perfectly omniscient, omnipresent, and
omnipotent.

### 3.3 Score Fluctuation Rules

PSL scores may fluctuate by up to 0.3 points per dimension per hour based
on robot performance, errors, calibration drift, maintenance events, or
exceptional performance during the simulation.

## 4. PSL Score Bands

### 4.1 Per-Robot Bands (0.0 to 10.0)

| Band | Score Range | Description |
|------|-------------|-------------|
| Nascent | 0.0 - 1.9 | Minimal capability |
| Foundational | 2.0 - 3.9 | Basic capability, significant limitations |
| Intermediate | 4.0 - 5.9 | Functional capability, notable gaps |
| Advanced | 6.0 - 7.9 | Strong capability, minor limitations |
| Elite | 8.0 - 9.9 | Near-ideal capability |
| Ideal | 10.0 | Theoretically perfect |

### 4.2 Cumulative Site Bands (0.0 to 100.0)

| Band | Score Range | Description |
|------|-------------|-------------|
| Nascent Site | 0.0 - 19.9 | Minimal site capability |
| Foundational Site | 20.0 - 39.9 | Basic site capability |
| Intermediate Site | 40.0 - 59.9 | Functional site capability |
| Advanced Site | 60.0 - 79.9 | Strong site capability |
| Elite Site | 80.0 - 99.9 | Near-ideal site capability |
| Ideal Site | 100.0 | Theoretically perfect site |

## 5. The 10 Robot Types Evaluated

| # | Robot Type | Cancer Types Served | Instances |
|---|-----------|---------------------|-----------|
| 1 | Surgical Robots | Mediastinal tumors, solid tumor resections | 3 suites |
| 2 | Cobots | Forearm soft-tissue sarcoma biopsies | 4 stations |
| 3 | RT Positioning Robots | Brain tumors (GBM, meningioma, mets) | 3 vaults |
| 4 | Needle-Placement Systems | Parotid/head-neck tumors | 2 suites |
| 5 | Social Companion Robots | Pediatric leukemia (anxiety mgmt) | 5 stations |
| 6 | Humanoids | Pediatric osteosarcoma (PT prep) | 3 stations |
| 7 | RT Motion-Tracking Robots | Lung tumors (NSCLC, SCLC, mets) | 3 vaults |
| 8 | Imaging Assistant Robots | Liver tumors (HCC, mets) | 4 bays |
| 9 | Steerable Needle Robots | Liver tumors (targeted ablation) | 2 suites |
| 10 | Rehab Exoskeletons | Femur osteosarcoma (post-surgical) | 3 bays |

## 6. Initial PSL Scores (Hour 00 Baseline)

Initial scores are assigned based on reading of the three regulatory documents,
the robot specifications in patient_robot_instructions_fixed.tex, and the
USL scores from usl_oncology_trials.tex.

| Robot Type | Dim A | Dim B | Dim C | PSL | Band |
|-----------|-------|-------|-------|-----|------|
| Surgical Robots | 7.2 | 5.8 | 7.5 | 6.8 | Advanced |
| Cobots | 7.0 | 6.5 | 6.2 | 6.6 | Advanced |
| RT Positioning Robots | 7.5 | 6.0 | 6.8 | 6.8 | Advanced |
| Needle-Placement | 6.8 | 5.5 | 6.5 | 6.3 | Advanced |
| Social Companion | 5.5 | 7.2 | 4.0 | 5.6 | Intermediate |
| Humanoids | 5.8 | 6.0 | 5.2 | 5.7 | Intermediate |
| RT Motion-Tracking | 7.8 | 6.2 | 7.0 | 7.0 | Advanced |
| Imaging Assistant | 7.0 | 6.8 | 5.8 | 6.5 | Advanced |
| Steerable Needle | 7.2 | 5.2 | 7.0 | 6.5 | Advanced |
| Rehab Exoskeletons | 5.5 | 5.8 | 5.5 | 5.6 | Intermediate |

Cumulative Site PSL: 63.4 (Advanced Site)

### 6.1 Scoring Rationale

Surgical Robots receive high Omniscient (7.2) and Omnipotent (7.5) scores due
to extensive sensor fusion, AI integration (da Vinci dVRK USL 7.1), and broad
procedural capability. Omnipresent (5.8) is lower because each surgical suite
serves one patient at a time.

Cobots score well on Omnipresent (6.5) due to rapid repositioning and
multi-station capability (Franka Panda USL 7.4). Omnipotent (6.2) reflects
the narrower procedural scope of biopsy tasks.

RT Motion-Tracking Robots achieve the highest initial PSL (7.0) due to
exceptional sensor fusion for breathing monitoring (Omniscient 7.8), strong
beam gating capabilities (Omnipotent 7.0), and vault-sharing efficiency
(Omnipresent 6.2).

Social Companion Robots score lower overall (5.6) because their Omnipotent
dimension is limited (4.0) - they do not perform clinical procedures. Their
Omnipresent score (7.2) is the highest among all robots because they can
engage multiple pediatric patients through digital interactions.

## 7. PSL Relationship to USL

PSL and USL are complementary scoring systems:

| Aspect | PSL | USL |
|--------|-----|-----|
| Focus | Clinical performance | Technical interoperability |
| Dimensions | 3 (omniscient, omnipresent, omnipotent) | 4 (sim switching, AI, sharing, collaboration) |
| Scale | 0.0 to 10.0 per robot | 1.0 to 10.0 per robot |
| Basis | 3 regulatory documents | Technical evaluation |
| Primary use | On-demand trial readiness | Multi-site unification readiness |

Reference: The Unification Standard Level (USL) framework (Kawchak, 2026;
DOI: 10.5281/zenodo.18778220) provides complementary robot technical
interoperability scoring. See
physical-ai-oncology-trials/unification/usl/paper/usl_oncology_trials.tex.

## 8. Future PSL Utility

Trial coordinators can use PSL to specify minimum thresholds for on-demand
trials. Examples:

- A coordinator might require a minimum per-robot PSL of 6.0 and a minimum
  cumulative site PSL of 55.0 before activating a new on-demand trial site
- High-complexity surgical trials may require surgical robot PSL of 7.5 or
  above
- Pediatric trials may require companion robot PSL of 6.0 and humanoid PSL
  of 6.0 or above
- Multi-cancer-type sites treating 15 or more cancer types simultaneously may
  require cumulative PSL of 60.0 or above

This allows customizable levels of omniscience, omnipresence, and omnipotence
based on trial complexity, cancer type severity, and patient volume
requirements.

## 9. References

- ICH E6(R3) Adaption: DOI 10.5281/zenodo.18973368
- 21 CFR Part 50 Adaption: DOI 10.5281/zenodo.19040707
- 21 CFR Part 312 Adaption: DOI 10.5281/zenodo.19057628
- USL Framework: DOI 10.5281/zenodo.18778220
- Patient Journey: DOI 10.5281/zenodo.19119939
- Patient Instructions: DOI 10.5281/zenodo.18810541
- Repository: DOI 10.5281/zenodo.18445179
