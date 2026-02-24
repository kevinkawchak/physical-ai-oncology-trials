# Releases

Release notes for the physical-ai-oncology-trials repository.

---

## Unification Standard Level (USL) — Surgical Robots
v1.5.0 - February 24, 2026

### Summary

Extends the **Unification Standard Level (USL)** framework to **Surgical Robots** — a new robot category under `unification/usl/surgical/`. Three teleoperated surgical robot systems from different manufacturers are evaluated: **Intuitive Surgical da Vinci (dVRK)** (USL 7.1), **Medtronic Hugo RAS** (USL 4.5), and **CMR Surgical Versius** (USL 3.4). Each system is scored across the same four dimensions (A–D) established for cobots: simulation framework switching, generative/agentic AI integration, cross-robot progress sharing, and multi-site clinical trial collaboration.

The existing `usl_scoring_framework.py` is moved under the `cobots/` directory, and a new `usl_surgical_scoring.py` is created for surgical robot evaluation. The USL README is restructured to cover general, surgical, and cobot information in that order, with 3 new text diagrams for surgical robots (general, technical, scoring). Each surgical robot has its own directory with comprehensive evaluation code including hardware specifications, kinematic models, simulation framework configurations, oncology-specific task definitions, cross-organization sharing interfaces, and USL scoring.

### Features

- `unification/usl/surgical/usl_surgical_scoring.py`: USL scoring engine adapted for surgical robots with `SurgicalSimFramework`, `SurgicalAICapability`, and `SurgicalProcedure` enums; `SurgicalDimAScore` through `SurgicalDimDScore` with surgical-specific scoring criteria (tissue deformation, instrument modeling, haptic feedback, surgical video AI, phase recognition, remote proctoring, IEC 80601 compliance); `SurgicalUSLRating` with weighted score computation, comparison tables, gap analysis, and report generation
- `unification/usl/surgical/intuitive_davinci/intuitive_davinci_usl.py`: Intuitive Surgical da Vinci (dVRK) evaluation module — `DVRKSpecs` with PSM/ECM/MTM configuration (7+1 DOF, 3 PSMs, stereo vision, EndoWrist articulation), `PSMKinematics` with remote center of motion (RCM) model and modified DH parameters (from Kazanzides et al., 2014), `DVRKFrameworkConfig` for 5 simulation frameworks (ORBIT-Surgical/Isaac Lab, SurRoL/PyBullet, AMBF, Gazebo, MuJoCo), `DVRKOncologyTask` definitions (tumor resection, lymph node dissection, suturing, biopsy), `DVRKCrossOrgSharing` with 5 sharing methods and 10 dVRK institution listing; USL score: 7.1
- `unification/usl/surgical/medtronic_hugo/medtronic_hugo_usl.py`: Medtronic Hugo RAS evaluation module — `HugoRASSpecs` with modular cart architecture (7 DOF per arm, open console, 8 mm instruments), `HugoArmKinematics` with DH parameters and joint validation, `TouchSurgeryInterface` with surgical phase recognition, performance metrics, and analytics, `HugoOncologyTask` definitions (colectomy, hysterectomy, prostatectomy, lymph node biopsy), `HugoCrossOrgSharing` with Medtronic ecosystem listing; USL score: 4.5
- `unification/usl/surgical/cmr_versius/cmr_versius_usl.py`: CMR Surgical Versius evaluation module — `VersiusSpecs` with biomimetic modular architecture (7 DOF, ~10 kg arms, 5 mm instruments, portable), `VersiusArmKinematics` with biomimetic DH parameters, `VersiusORSetup` configurations for 3 oncology specialties (gynecologic, colorectal, upper GI), `VersiusOncologyTask` definitions (hysterectomy, colectomy, gastrectomy, omentectomy), `VersiusCrossOrgSharing` with deployment regions; USL score: 3.4
- `unification/usl/README.md`: Restructured with general USL information first, then surgical robot evaluation (3 new text diagrams: general comparison, technical specifications, scoring breakdown), then cobot evaluation (original 3 diagrams preserved), robot category table, updated directory structure, expanded references
- Moved `unification/usl/usl_scoring_framework.py` → `unification/usl/cobots/usl_scoring_framework.py`
- Updated `prompts.md`: Added v1.5.0 USL Surgical Robots prompt
- Updated `releases.md`: Added v1.5.0 release notes
- Updated `CHANGELOG.md`: Added v1.5.0 entry
- Updated `unification/README.md`: Updated USL directory structure, added surgical robot roadmap items
- Updated `README.md`: Added surgical robot USL section, updated version to v1.5.0

### Contributors
@kevinkawchak
@claude

### Notes
- Three surgical robots selected for: different manufacturers, teleoperated MIS architecture, oncology surgical applications, and varying levels of open-source availability
- da Vinci (dVRK) scores highest due to its unique open-source ecosystem (dVRK, ORBIT-Surgical, SurRoL, AMBF) and extensive AI research community — no other surgical robot has comparable simulation and research infrastructure
- Hugo RAS and Versius score lower primarily due to proprietary platforms with limited open-source availability, which limits simulation switching, AI integration, and cross-robot sharing
- All four USL dimensions (A–D) are adapted for surgical robot-specific criteria: tissue deformation simulation, instrument articulation modeling, surgical video AI, phase recognition, remote proctoring, IEC 80601-2-77 compliance
- All code passes `ruff check` and `ruff format --check` on Python 3.10–3.12
- 4 new Python modules totaling approximately 2,400 lines of code
- Development by Claude Code Opus 4.6

---

## Unification Standard Level (USL) for Collaborative Robots
v1.4.0 - February 23, 2026

### Summary

Introduces the **Unification Standard Level (USL)** — a new scoring framework under `unification/usl/` for evaluating how ready physical AI robots are for deployment in unified, multi-site oncology clinical trials. USL scores range from 1.0 to 10.0 (in 0.1 increments) across four weighted dimensions: simulation framework switching, generative/agentic AI integration, cross-robot progress sharing, and multi-site clinical trial collaboration.

This initial release evaluates three state-of-the-art open-source collaborative robot arms from different manufacturers: **Franka Emika Panda** (Franka Robotics, USL 7.4), **Kinova Gen3 7DoF** (Kinova Robotics, USL 5.7), and **UFACTORY xArm 7** (UFACTORY, USL 3.4). Each cobot receives a comprehensive evaluation with hardware specifications, simulation framework configurations, kinematic validation tools, policy transfer interfaces, cross-organization sharing capabilities, and oncology-specific task definitions.

The USL framework is influenced by NASA/DOD TRL (Mankins, 2004), MLTRL (Lavin et al., 2021), TRL for complex systems (Tomaschek et al., 2015), and is inspired by LLM recommendations for oncology trials (Kawchak, 2025; DOI 10.5281/zenodo.17451709).

### Features

- `unification/usl/usl_scoring_framework.py`: Core USL scoring engine with four weighted dimensions (A–D), 10-level classification system, score band categorization, comparison tables, gap analysis, and JSON/text report generation
- `unification/usl/cobots/franka_panda/franka_panda_usl.py`: Franka Emika Panda evaluation module with hardware specs, DH parameters, URDF template generator, kinematic chain validator, policy transfer interface with 4 oncology tasks, cross-organization sharing manager, and simulation framework configurations for MuJoCo/Isaac Lab/Gazebo/PyBullet
- `unification/usl/cobots/kinova_gen3/kinova_gen3_usl.py`: Kinova Gen3 7DoF evaluation module with Kortex API abstraction layer, modified DH kinematic model, actuator module specifications, angular/Cartesian command interfaces, 4 oncology task definitions, and framework configurations for Gazebo/MuJoCo/Isaac Lab/PyBullet
- `unification/usl/cobots/ufactory_xarm7/ufactory_xarm7_usl.py`: UFACTORY xArm 7 evaluation module with xArm Python SDK abstraction, joint specifications with limit validation, error code mapping, 4 oncology lab automation tasks, intra-organization sharing across xArm family, and framework configurations
- `unification/usl/README.md`: Comprehensive USL standard documentation with scoring methodology, 10-level definitions, score bands, three text comparison diagrams (general, technical, scoring), individual cobot evaluations, references to TRL/MLTRL influences, and quick-start guide
- `prompts.md`: Development prompt archive for v1.4.0 USL standard creation
- `releases.md`: Release notes in standardized format
- Updated `unification/README.md`: Added USL directory to structure, added Q1 2026 USL roadmap items
- Updated `README.md`: Added USL section with cobot evaluation table, updated repository structure, updated version to v1.4.0
- Updated `CHANGELOG.md`: Added v1.4.0 entry
- Updated `ruff.toml`: Added per-file ignore for `unification/usl/**/*.py`

### Contributors
@kevinkawchak
@claude

### Notes
- USL framework is specific to this project — "Unification Standard Level" evaluates robot readiness for multi-site oncology trial unification, distinct from general-purpose TRL
- All four USL dimensions derive directly from the existing `unification/` pillars: `simulation_physics/`, `agentic_generative_ai/`, `cross_platform_tools/`, and the `federation/`+`regulatory/` directories
- The three evaluated cobots (Franka Panda, Kinova Gen3, xArm 7) were selected for: open-source availability, different manufacturers, MuJoCo Menagerie models, active ROS 2 support, and potential oncology applications
- All code passes `ruff check` and `ruff format --check` on Python 3.10–3.12
- 4 new Python modules totaling approximately 2,100 lines of code
- Development by Claude Code Opus 4.6
