# Releases

Release notes for the physical-ai-oncology-trials repository.

---

## Unification Standard Level (USL) for Surgical Robots
v1.5.0 - February 23, 2026

### Summary

Extends the **Unification Standard Level (USL)** framework to **surgical robot systems** — a new category under `unification/usl/surgical/` evaluating three major surgical robot platforms from different manufacturers for unification readiness in multi-site oncology clinical trials. The three evaluated systems represent the three dominant architectural paradigms in teleoperated surgery: boom-mounted (**da Vinci**, Intuitive Surgical, USL 7.6), cart-based modular (**Hugo RAS**, Medtronic, USL 4.1), and table-integrated (**OTTAVA**, Johnson & Johnson MedTech, USL 2.3).

The existing cobot scoring framework (`usl_scoring_framework.py`) has been moved to `cobots/` to establish a clean category-based directory structure. The USL README now covers both surgical robots and cobots with general, surgical, and cobot sections in that order, plus six text comparison diagrams (three for each category). Surgical robot scoring criteria are adapted from the cobot framework's a)–d) dimensions with considerations for teleoperated architectures, FDA clearance pathways, IEC 80601-2-77 compliance, and clinical evidence volumes.

### Features

- `unification/usl/surgical/surgical_usl_scoring.py`: Surgical robot USL scoring engine with four weighted dimensions adapted for surgical systems — includes SurgicalSimScore, SurgicalDimA/B/C/D scores, SurgicalUSLRating, comparison tables, gap analysis, level classification, and JSON/text report generation; new enums for SurgicalRobotCategory, SurgicalSimFramework, SurgicalAICapability, SurgicalSharingMethod, and RegulatoryStatus
- `unification/usl/surgical/davinci/davinci_usl.py`: Intuitive Surgical da Vinci (Xi / da Vinci 5) evaluation module — DaVinciSpecs (7 DOF EndoWrist, 4 arms, force feedback, ~14M procedures), PSM kinematic chain with modified DH parameters and joint limit validation, 5 simulation framework configurations (ORBIT-Surgical, dVRK Sim, SurRoL, SurgicalGym, AMBF), policy transfer interface with 4 oncology surgical tasks (tissue dissection, lymph node excision, suturing, vessel sealing), cross-organization sharing manager (dVRK research kit, ORBIT-Surgical benchmarks, OpenIGTLink, ONNX); USL score: 7.6 (Level 7 — Advanced)
- `unification/usl/surgical/hugo_ras/hugo_ras_usl.py`: Medtronic Hugo RAS evaluation module — HugoRASSpecs (7 DOF wristed instruments, 4 independent arm carts, 520° wrist rotation, Karl Storz 3D vision, pistol-grip + IR contact sensors, head tracking, collision avoidance), modular architecture interface with arm cart configuration and collision risk checking, simulation status tracker (no public models available), clinical readiness assessment with FDA clearance status and capability inventory; USL score: 4.1 (Level 4 — Developing)
- `unification/usl/surgical/ottava/ottava_usl.py`: Johnson & Johnson OTTAVA evaluation module — OTTAVASpecs (4 arms, table-integrated zero-footprint architecture, Twin Motion), Twin Motion repositioning simulator with 6 supported positions, regulatory pathway tracker with 6 milestones (IDE approvals through expected FDA decision), oncology expansion potential assessment for 4 specialties with estimated addressable procedure volumes; USL score: 2.3 (Level 2 — Exploratory)
- `unification/usl/README.md`: Comprehensive rewrite — now covers both surgical robots and cobots with general USL framework overview, separate surgical and cobot evaluation sections, six text comparison diagrams (3 surgical: general comparison, technical specifications, scoring breakdown; 3 cobot: unchanged), updated directory structure, surgical robot-specific references (IEEE 3177-2024, IEC 80601-2-77, LASR classification, SAGES STARSS, dVRK architecture, OpenIGTLink), and updated quick-start guide
- Moved `unification/usl/usl_scoring_framework.py` → `unification/usl/cobots/usl_scoring_framework.py` to establish category-based directory organization
- `prompts.md`: Updated with v1.5.0 USL surgical robot standard creation prompt
- `releases.md`: Updated with v1.5.0 release notes
- Updated `CHANGELOG.md`: Added v1.5.0 entry
- Updated `unification/README.md`: Added surgical robot category to USL roadmap, updated directory structure
- Updated `README.md`: Added surgical robot evaluation table, updated repository structure, updated version to v1.5.0

### Contributors
@kevinkawchak
@claude

### Notes
- USL surgical robot scoring criteria are adapted from the cobot framework but account for surgical-specific factors: teleoperated master-slave architectures, FDA clearance requirements (IEC 80601-2-77 vs ISO 13482), clinical evidence volumes (procedure counts), proprietary vs open-source control ecosystems, and the LASR autonomy classification
- The three evaluated surgical robots were selected for: different manufacturers, different architectural paradigms (boom/cart/table), FDA clearance status diversity (established/recent/pending), and relevance to oncology surgical procedures
- Key surgical robot references: IEEE 3177-2024 (modular surgical robot framework standard), dVRK v2.3.1 (open-source research kit), ORBIT-Surgical (GPU-accelerated benchmark suite), OpenIGTLink (open network protocol for IGT), LASR classification (npj Digital Medicine 2024), SAGES STARSS assessment tool
- All code passes `ruff check` and `ruff format --check` on Python 3.10–3.12
- 4 new Python modules totaling approximately 2,500 lines of code
- Development by Claude Code Opus 4.6

---

## Unification Standard Level (USL) for Collaborative Robots
v1.4.0 - February 23, 2026

### Summary

Introduces the **Unification Standard Level (USL)** — a new scoring framework under `unification/usl/` for evaluating how ready physical AI robots are for deployment in unified, multi-site oncology clinical trials. USL scores range from 1.0 to 10.0 (in 0.1 increments) across four weighted dimensions: simulation framework switching, generative/agentic AI integration, cross-robot progress sharing, and multi-site clinical trial collaboration.

This initial release evaluates three state-of-the-art open-source collaborative robot arms from different manufacturers: **Franka Emika Panda** (Franka Robotics, USL 7.4), **Kinova Gen3 7DoF** (Kinova Robotics, USL 5.7), and **UFACTORY xArm 7** (UFACTORY, USL 3.4). Each cobot receives a comprehensive evaluation with hardware specifications, simulation framework configurations, kinematic validation tools, policy transfer interfaces, cross-organization sharing capabilities, and oncology-specific task definitions.

The USL framework is influenced by NASA/DOD TRL (Mankins, 2004), MLTRL (Lavin et al., 2021), TRL for complex systems (Tomaschek et al., 2015), and is inspired by LLM recommendations for oncology trials (Kawchak, 2025; DOI 10.5281/zenodo.17451709).

### Features

- `unification/usl/cobots/usl_scoring_framework.py`: Core USL scoring engine with four weighted dimensions (A–D), 10-level classification system, score band categorization, comparison tables, gap analysis, and JSON/text report generation
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
