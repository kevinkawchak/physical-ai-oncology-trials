# Releases

Release notes for the physical-ai-oncology-trials repository.

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
