# End-to-End Physical AI Unification of Oncology Clinical Trials

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/badge/Release-v2.9.2-brightgreen.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![Last Updated](https://img.shields.io/badge/Updated-March%202026-blue.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![Protocol](https://img.shields.io/badge/Protocol-MCP-purple.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18445179-blue)](https://doi.org/10.5281/zenodo.18445179)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/)
[![Contributors](https://img.shields.io/badge/Contributors-4-blue.svg)](releases.md)




**Comprehensive developments for integrating physical AI into oncology clinical trials, by Claude Code Opus 4.6, Cowork; with Assistance from ChatGPT 5.4 Thinking and Google Gemini Search.**

This repository provides production-ready configurations, validated pipelines, and integration guides for deploying robotic systems, digital twins, and embodied AI agents in oncology. 

**3/24: v2.9.0 (Trial Site Documentation)** *Physical AI Oncology Clinical Trial Site Documentation* - 11 LaTeX documents for California's first Physical AI oncology trial site: legislation drafts, regulations. [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19176370-blue)](https://doi.org/10.5281/zenodo.19176370)

**3/23: v2.8.0 (On-Demand Trial Simulation)** *24-Hour On-Demand Physical AI Oncology Clinical Trial Simulation* - Full 24-hour simulation of an autonomous, patient-centric oncology trial serving 168 patients. [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19194724-blue)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial)


**3/20: v2.7.0 (Patient Journey Paper)** *A Cancer Patient's Journey Through a Regulated and Autonomous Physical AI Oncology Trial Illustration* - Comprehensive paper documenting the patient journey [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19119939-blue)](https://doi.org/10.5281/zenodo.19119939)

📄 **3/20: v2.6.0 (Patient Journey)** *End-to-End Physical AI Oncology Clinical Trial Unification: Single-Patient Journey Orchestration* - 10-stage patient journey for PAT-2026-0042 (58F, Stage IIIB NSCLC) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19123890-blue)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/patient-journey)

📄 **3/18: v2.5.0 (Regulatory Adaptation)** *End-to-End Physical AI Oncology Clinical Trial Unification: Adaption of 21 CFR Part 312 - Investigational New Drug Application* - Adaptation of 21 CFR Part 312 Regulation [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19057628-blue)](https://doi.org/10.5281/zenodo.19057628)

📄 **3/16: v2.4.0 (Regulatory Adaptation)** *End-to-End Physical AI Oncology Clinical Trial Unification: Adaption of 21 CFR Part 50 - Protection of Human Subjects* - Adaptation of 21 CFR Part 50 Regulation [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19040707-blue)](https://doi.org/10.5281/zenodo.19040707)

📄 **3/12: v2.2.0 (Regulatory Guidance)** *End-to-End Physical AI Oncology Clinical Trial Unification.* Comprehensive guidance adapted from prior ICH E6(R3), with Sections 1-4, Appendices A-C, and Glossary [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18973368-blue)](https://doi.org/10.5281/zenodo.18973368)

📄 **3/2: v2.1.0 (Patient Instructions)** *Patient Instructions: Physical AI Oncology Trials* - Paper content documentation with page-by-page instructions, text diagrams, and quantitative patient data [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18810541-blue)](https://doi.org/10.5281/zenodo.18810541)

📄 **2/26: New Paper (USL)** *Unification Standard Level for Physical AI Oncology Trials. Standardizing and Evaluating Robot Unification Readiness for Multi-Site Clinical Trials. USL scores range from 1.0 to 10.0* [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18778219-blue)](https://doi.org/10.5281/zenodo.18778219)

> **v1.0.0** — First stable release. 51 Python modules (40,526 LOC), 69 documentation files, 28 examples, 5 CLI tools, and complete privacy/regulatory infrastructure. CI-validated on Python 3.10, 3.11, and 3.12. See [V1_RELEASE.md](V1_RELEASE.md) for full release documentation.

## Responsible use

This repository is complementary and open source, please implement code safely and responsibly. Intended audience: engineers building physical AI systems (robotics, ML, integration, and validation) for clinical trial settings.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/kevinkawchak/physical-ai-oncology-trials.git
cd physical-ai-oncology-trials

# Install base dependencies
pip install -r requirements.txt

# Verify framework availability
python scripts/verify_installation.py

# Detect available simulation frameworks
python unification/cross_platform_tools/framework_detector.py
```

---

## Repository Structure

```
physical-ai-oncology-trials/
├── README.md
├── V1_RELEASE.md
├── LICENSE
├── requirements.txt
│
├── new-trial/                         # ★ 24-Hour On-Demand Trial Simulation (v2.8.0)
│   ├── README.md                      # Simulation overview and results
│   ├── psl_framework.md               # PSL scoring framework definition
│   ├── site_specification.md          # Facility and staffing specifications
│   ├── format_comparison.md           # On-demand vs. traditional comparison
│   ├── prompts.md                     # v2.8.0 development prompt
│   ├── hour-00/ through hour-23/      # 24 hourly simulation directories
│   │   ├── hour_XX_simulation.md      # Master simulation log
│   │   ├── hour_XX_robot_logs.md      # Per-robot telemetry
│   │   ├── hour_XX_patient_records.md # Patient vitals and records
│   │   ├── hour_XX_psl_scores.md      # PSL scores for all 10 robots
│   │   ├── hour_XX_diagram_facility.txt    # Facility layout diagram
│   │   ├── hour_XX_diagram_patient_flow.txt # Patient flow diagram
│   │   └── hour_XX_diagram_robot_status.txt # Robot status timeline
│   ├── final-commit/                  # Error review and 24-hour summaries
│   │   ├── final_error_review.md      # Consistency check
│   │   ├── final_24h_summary.md       # Performance summary
│   │   ├── final_psl_cumulative.md    # PSL trajectory analysis
│   │   └── final_diagram_*.txt        # Summary diagrams (3 files)
│   └── site/                          # ★ Trial Site Documentation (v2.9.0)
│       ├── README.md                  # Site documentation overview
│       ├── 01-legislation-authorization/  # SB 1042 authorization act
│       ├── 02-legislation-patient-rights/ # AB 2847 patient rights act
│       ├── 03-legislation-data-transparency/ # SB 892 data protection act
│       ├── 04-city-regulations/       # SF municipal code update
│       ├── 05-state-regulations/      # CA Title 22 Chapter 14
│       ├── 06-national-regulations/   # FDA compliance guide
│       ├── 07-building-code/          # Facility construction standards
│       ├── 08-premises-code/          # Site safety and access
│       ├── 09-parking-transportation/ # Parking and transit standards
│       ├── 10-site-operations/        # Activation and SOPs
│       ├── 11-emergency-preparedness/ # Emergency response plan
│       ├── all-documents/              # Combined 11-document source
│       │   ├── all_documents.tex      # Full combined LaTeX source
│       │   └── all_documents_chunk/   # ★ Chunked into 11 files (v2.9.1)
│       │       └── README.md          # Reconstruction instructions
│       └── zips/                      # LaTeX source archives (12 zips)
│
├── patient-journey/                   # ★ Single-Patient Journey Orchestration (v2.6.0)
│   ├── patient_state.py               # Central data model (10 enums, 14 dataclasses)
│   ├── stage_01_prescreening.py       # Stage 1: Pre-Screening & Referral Intake
│   ├── stage_02_enrollment.py         # Stage 2: Enrollment & Informed Consent
│   ├── stage_03_digital_twin.py       # Stage 3: Digital Twin Construction
│   ├── stage_04_robot_qualification.py # Stage 4: Robot Qualification
│   ├── stage_05_surgery.py            # Stage 5: Robot-Assisted Surgery
│   ├── stage_06_recovery.py           # Stage 6: Post-Operative Recovery
│   ├── stage_07_immunotherapy.py      # Stage 7: Immunotherapy Treatment
│   ├── stage_08_federation.py         # Stage 8: Federated Learning
│   ├── stage_09_surveillance.py       # Stage 9: Long-Term Surveillance
│   ├── stage_10_closeout.py           # Stage 10: Trial Closeout
│   ├── master_journey.py             # Master orchestrator (all 10 stages)
│   ├── diagrams/                      # 30 ASCII progress diagrams (3 x 10)
│   ├── deliverables/                  # Charts, tables, FDA analysis, guidance
│   ├── paper/                         # ★ Patient Journey Paper (v2.7.0)
│   │   ├── patient_journey_paper.tex  # LaTeX source
│   │   ├── patient_journey_paper_chunk/ # ★ Chunked into 3 files (v2.9.1)
│   │   │   └── README.md             # Reconstruction instructions
│   │   ├── patient_journey_paper.pdf  # Compiled PDF (compile from .tex)
│   │   ├── Latex_Source_Code.zip      # Source archive
│   │   ├── arxiv.sty                  # Style file
│   │   ├── orcid_icon.png             # ORCID icon
│   │   ├── README.md                  # Paper documentation
│   │   └── template/                  # Formatting template
│   └── prompts.md                     # Development prompts archive
│
├── patients/                          # ★ Patient Instructions (v2.1.0)
│   ├── patient_robot_instructions_fixed.tex  # LaTeX source
│   ├── patient_robot_instructions_fixed_chunk/ # ★ Chunked into 2 files (v2.9.1)
│   │   └── README.md                 # Reconstruction instructions
│   ├── README.md                      # Paper content, instructions, text diagrams
│   ├── research/                      # Archived generation scripts
│   │   ├── v1.9.1/
│   │   │   ├── generate_pdf.py        # reportlab + Pillow generator
│   │   │   ├── paper/README           # Paper access (Drive link)
│   │   │   └── images/README.md       # Image access (Drive link)
│   │   └── v1.9.0/
│   │       ├── README.md
│   │       ├── generate_illustrations.py
│   │       ├── generate_pdf.py
│   │       ├── paper/README
│   │       ├── svg/README.md
│   │       ├── pdf/README.md
│   │       └── png/README.md
│   └── prompts/
│       └── prompts.md                 # Development prompts archive
│
├── digital-twins/
│   ├── README.md
│   ├── patient-modeling/
│   │   ├── README.md
│   │   └── tumor_twin_pipeline.py
│   ├── treatment-simulation/
│   │   ├── README.md
│   │   └── treatment_simulator.py
│   ├── clinical-integration/
│   │   ├── README.md
│   │   └── clinical_dt_interface.py
│   └── examples-twins/
│       ├── README.md
│       ├── 01_realtime_dt_synchronization.py
│       ├── 02_multi_organ_toxicity_twin.py
│       ├── 03_adaptive_radiation_therapy_dt.py
│       ├── 04_tumor_microenvironment_immunotherapy_dt.py
│       ├── 05_virtual_trial_cohort_dt.py
│       └── 06_dt_validation_verification.py
│
├── examples/
│   ├── README.md
│   ├── 01_surgical_robot_training.py
│   ├── 02_digital_twin_surgical_planning.py
│   ├── 03_cross_framework_validation.py
│   ├── 04_agentic_clinical_workflow.py
│   └── 05_treatment_response_prediction.py
│
├── examples-new/
│   ├── README.md
│   ├── 01_realtime_safety_monitoring.py
│   ├── 02_sensor_fusion_intraoperative.py
│   ├── 03_ros2_surgical_deployment.py
│   ├── 04_hand_eye_calibration_registration.py
│   ├── 05_shared_autonomy_teleoperation.py
│   └── 06_robotic_sample_handling.py
│
├── q1-2026-standards/
│   ├── README.md
│   ├── objective-1-bidirectional-conversion/
│   │   ├── isaac_to_mujoco_pipeline.py
│   │   ├── mujoco_to_isaac_pipeline.py
│   │   └── physics_equivalence_tests.py
│   ├── objective-2-robot-model-repository/
│   │   ├── model_registry.yaml
│   │   └── model_validator.py
│   ├── objective-3-validation-benchmark/
│   │   └── benchmark_runner.py
│   └── implementation-guide/
│       ├── timeline.md
│       └── compliance_checklist.md
│
├── unification/
│   ├── README.md
│   ├── simulation_physics/
│   │   ├── challenges.md
│   │   ├── opportunities.md
│   │   ├── isaac_mujoco_bridge.py
│   │   ├── urdf_sdf_mjcf_converter.py
│   │   └── physics_parameter_mapping.yaml
│   ├── agentic_generative_ai/
│   │   ├── challenges.md
│   │   ├── opportunities.md
│   │   └── unified_agent_interface.py
│   ├── surgical_robotics/
│   │   ├── challenges.md
│   │   └── opportunities.md
│   ├── cross_platform_tools/
│   │   ├── framework_detector.py
│   │   └── validation_suite.py
│   ├── usl/                          # ★ Unification Standard Level
│   │   ├── README.md                 # USL standard overview
│   │   ├── prompts.md                # Development prompts archive
│   │   ├── paper/                    # ★ USL Paper (v1.8.0)
│   │   │   ├── Unification Standard Level for Physical AI Oncology Trials.pdf
│   │   │   ├── Latex Source Code.zip # .tex, .sty, .bib, README
│   │   │   ├── usl_oncology_trials.tex
│   │   │   ├── usl_oncology_trials_chunk/ # ★ Chunked into 2 files (v2.9.1)
│   │   │   │   └── README.md        # Reconstruction instructions
│   │   │   ├── usl-oncology.sty
│   │   │   ├── references.bib
│   │   │   └── README
│   │   ├── humanoids/                # ★ Humanoid Robots (v1.6.0)
│   │   │   ├── README.md             # Humanoid evaluations & diagrams
│   │   │   ├── usl_humanoid_scoring.py
│   │   │   ├── boston_dynamics_atlas/
│   │   │   ├── tesla_optimus/
│   │   │   └── agility_digit/
│   │   ├── surgical/                 # Surgical Robots (v1.5.0)
│   │   │   ├── README.md             # Surgical evaluations & diagrams
│   │   │   ├── usl_surgical_scoring.py
│   │   │   ├── intuitive_davinci/
│   │   │   ├── medtronic_hugo/
│   │   │   └── cmr_versius/
│   │   └── cobots/                   # Cobots (v1.4.0)
│   │       ├── README.md             # Cobot evaluations & diagrams
│   │       ├── usl_scoring_framework.py
│   │       ├── franka_panda/
│   │       ├── kinova_gen3/
│   │       └── ufactory_xarm7/
│   ├── standards_protocols/
│   └── integration_workflows/
│
├── generative-ai/                     # VLA models, diffusion policies, synthetic data
│   ├── strengths.md
│   ├── limitations.md
│   └── results.md
├── agentic-ai/                        # LLM-based robot control, multi-agent systems
│   ├── README.md
│   ├── strengths.md
│   ├── limitations.md
│   ├── results.md
│   └── examples-agentic-ai/
│       ├── 01_mcp_clinical_robotics_server.py
│       ├── 02_react_procedure_planner.py
│       ├── 03_realtime_adaptive_treatment_agent.py
│       ├── 04_autonomous_simulation_orchestrator.py
│       ├── 05_safety_constrained_agent_executor.py
│       └── 06_protocol_rag_compliance_agent.py
├── reinforcement-learning/            # RL for surgical autonomy, sim2real transfer
│   ├── strengths.md
│   ├── limitations.md
│   └── results.md
├── self-supervised-learning/          # Contrastive learning, foundation models
│   ├── strengths.md
│   ├── limitations.md
│   └── results.md
├── supervised-learning/               # Segmentation, detection, classification
│   ├── strengths.md
│   ├── limitations.md
│   └── results.md
│
├── frameworks/
│   ├── nvidia-isaac/                  # Isaac Sim, Isaac Lab, Isaac for Healthcare
│   │   └── INTEGRATION.md
│   ├── mujoco/                        # MuJoCo, MJX, MuJoCo Playground
│   │   └── INTEGRATION.md
│   ├── gazebo/                        # Gazebo Ionic, ROS 2 integration
│   │   └── INTEGRATION.md
│   └── pybullet/                      # PyBullet medical simulation
│       └── INTEGRATION.md
│
├── privacy/
│   ├── README.md
│   ├── phi-pii-management/
│   │   ├── README.md
│   │   └── phi_detector.py
│   ├── de-identification/
│   │   ├── README.md
│   │   └── deidentification_pipeline.py
│   ├── access-control/
│   │   ├── README.md
│   │   └── access_control_manager.py
│   ├── breach-response/
│   │   ├── README.md
│   │   └── breach_response_protocol.py
│   └── dua-templates/
│       ├── README.md
│       └── dua_generator.py
│
├── regulatory/
│   ├── README.md
│   ├── Adaption-21-CFR-Part-312/      # ★ Physical AI 21 CFR Part 312 Adaptation (v2.5.0)
│   │   └── source/
│   │       ├── Physical_AI_21_CFR_Part_312.tex  # LaTeX source (94 pages compiled)
│   │       ├── Physical_AI_21_CFR_Part_312_chunk/ # ★ Chunked into 5 files (v2.9.1)
│   │       │   └── README.md                    # Reconstruction instructions
│   │       ├── Physical_AI_21_CFR_Part_312.sty  # Custom style package
│   │       ├── Physical_AI_21_CFR_Part_312.bib  # Bibliography (42 references)
│   │       ├── Physical_AI_21_CFR_Part_312.pdf  # Compiled PDF
│   │       ├── Physical_AI_21_CFR_Part_312.zip  # Source archive
│   │       └── prompts.md                       # Development prompts archive
│   ├── Adaption-21-CFR-Part-50/       # ★ Physical AI 21 CFR Part 50 Adaptation (v2.4.0)
│   │   └── source/
│   │       ├── Physical_AI_21_CFR_Part_50.tex   # LaTeX source (37 pages compiled)
│   │       ├── Physical_AI_21_CFR_Part_50_chunk/ # ★ Chunked into 3 files (v2.9.1)
│   │       │   └── README.md                    # Reconstruction instructions
│   │       ├── Physical_AI_21_CFR_Part_50.sty   # Custom style package
│   │       ├── Physical_AI_21_CFR_Part_50.bib   # Bibliography (19 references)
│   │       ├── Physical_AI_21_CFR_Part_50.pdf   # Compiled PDF
│   │       ├── Physical_AI_21_CFR_Part_50.zip   # Source archive
│   │       ├── prompts.md                            # ★ Development prompts archive (v2.4.0)
│   │       └── README.md
│   ├── adaption-ich-e6r3/             # ★ Physical AI Unification Guidance (v2.2.0)
│   │   ├── prompts.md                 # Development prompts archive
│   │   └── source/
│   │       ├── main.tex               # LaTeX source (Sections 1-4, Appendices, Glossary)
│   │       ├── main_chunk/            # ★ Chunked into 4 files (v2.9.1)
│   │       │   └── README.md          # Reconstruction instructions
│   │       ├── ich_guideline_style.sty
│   │       ├── references.bib
│   │       ├── compiled.pdf
│   │       └── README.md
│   ├── fda-compliance/
│   │   ├── README.md
│   │   └── fda_submission_tracker.py
│   ├── irb-management/
│   │   ├── README.md
│   │   └── irb_protocol_manager.py
│   ├── ich-gcp/
│   │   ├── README.md
│   │   └── gcp_compliance_checker.py
│   └── regulatory-intelligence/
│       ├── README.md
│       └── regulatory_tracker.py
│
├── regulatory-submit/                    # FDA Submission Automation (v1.0.0)
│   ├── README.md
│   ├── presub_generator.py              # Pre-Submission (Q-Sub) package generation
│   ├── pccp_engine.py                   # PCCP document authoring
│   ├── classification_advisor.py        # 510(k)/De Novo/PMA pathway classification
│   ├── iec62304_generator.py            # IEC 62304 lifecycle documentation
│   ├── clinical_evidence.py             # Clinical evidence reports
│   ├── audit_trail.py                   # 21 CFR Part 11 audit trails
│   └── examples-regulatory-submit/      # 6 submission workflow examples
│
├── federation/
│   ├── README.md
│   ├── federated_coordinator.py
│   ├── differential_privacy.py
│   ├── secure_aggregation.py
│   ├── site_enrollment.py
│   ├── data_harmonization.py
│   ├── consortium_reporting.py
│   ├── privacy_analytics.py
│   └── examples-federation/
│       ├── README.md
│       ├── 01_basic_two_site.py
│       ├── 02_differential_privacy.py
│       ├── 03_secure_aggregation.py
│       ├── 04_enrollment_sync.py
│       ├── 05_data_harmonization.py
│       └── 06_full_consortium.py
│
├── tools/
│   ├── README.md
│   ├── dicom-inspector/
│   │   └── dicom_inspector.py
│   ├── dose-calculator/
│   │   └── dose_calculator.py
│   ├── trial-site-monitor/
│   │   └── trial_site_monitor.py
│   ├── sim-job-runner/
│   │   └── sim_job_runner.py
│   └── deployment-readiness/
│       └── deployment_readiness.py
│
├── images/
│   ├── README.md
│   ├── prompts/                     # Human-authored + AI-recommended prompts
│   │   ├── plan.md
│   │   ├── 1st.md
│   │   ├── 2nd.md
│   │   └── 3rd.md
│   ├── interactive/                 # Python visualization scripts
│   │   ├── 1st/                     # 10 scripts (architecture, clinical)
│   │   ├── 2nd/                     # 10 scripts (AI/ML benchmarks)
│   │   └── 3rd/                     # 10 scripts (regulatory, privacy)
│   └── png/                         # Static PNG exports (1920×1080 @2x)
│       ├── 1st/                     # 20 PNGs (10 light + 10 dark)
│       ├── 2nd/                     # 20 PNGs (10 light + 10 dark)
│       └── 3rd/                     # 20 PNGs (10 light + 10 dark)
│
├── national-platform/                 # National platform research documents
│   ├── national_mcp/                  # ★ National MCP Servers paper chunks (v2.9.2)
│   │   ├── README.md                  # Reconstruction instructions
│   │   ├── national_mcp_chunk_01_preamble_intro_methods.tex
│   │   ├── national_mcp_chunk_02_results.tex
│   │   ├── national_mcp_chunk_03_discussion_conclusion.tex
│   │   └── references.bib            # Bibliography
│   ├── federated_learning/            # ★ Federated Learning paper chunks (v2.9.2)
│   │   ├── README.md                  # Reconstruction instructions
│   │   ├── fl_chunk_01_preamble_intro_methods.tex
│   │   ├── fl_chunk_02_results_part1.tex
│   │   ├── fl_chunk_03_results_part2_discussion_conclusion.tex
│   │   └── references.bib            # Bibliography
│   ├── RESEARCH-A                     # Federal regulatory research (plain text)
│   ├── RESEARCH-A-CHUNK/              # ★ Chunked into 2 files (v2.9.1)
│   │   └── README.md                  # Reconstruction instructions
│   ├── RESEARCH-B                     # State/federal comparative research (plain text)
│   └── RESEARCH-B-CHUNK/              # ★ Chunked into 2 files (v2.9.1)
│       └── README.md                  # Reconstruction instructions
│
├── configs/
│   └── training_config.yaml
│
└── scripts/
    └── verify_installation.py
```

---

## ★ Q1 2026 Standards

The new `q1-2026-standards/` directory contains **proposed standards** for meeting the Q1 2026 unification objectives:

| Objective | Description | Status |
|-----------|-------------|--------|
| **1** | Complete Isaac ↔ MuJoCo bidirectional conversion | Standards defined |
| **2** | Publish unified robot model repository (50+ models) | Registry created |
| **3** | Release validation benchmark suite v1.0 | Suite implemented |

---

## Core Technologies (Updated October 2025 - March 2026)

### Simulation & Physics

| Framework | Version | Last Update | Use Case | Unification Status |
|-----------|---------|-------------|----------|-------------------|
| NVIDIA Isaac Lab | 2.3.1 | Dec 2024 | GPU-accelerated robot training | ✓ Bridge available |
| NVIDIA Isaac Sim | 5.0.0 | Jan 2026 | High-fidelity physics simulation | ✓ Bridge available |
| Newton Physics Engine | Beta | Jan 2026 | GPU physics (NVIDIA/DeepMind/Disney) | ✓ Isaac Lab integrated |
| MuJoCo | 3.4.0 | Dec 2024 | Precision physics simulation | ✓ Bridge available |
| MuJoCo Warp | Beta | Jan 2026 | GPU-optimized MuJoCo (NVIDIA) | ✓ Bridge available |
| Gazebo Sim (Jetty) | 10.0.0 | Oct 2024 | ROS 2 integrated simulation | ◐ In progress |
| PyBullet | 3.2.5 | Apr 2023 | Rapid prototyping | ✓ Bridge available |

### Agentic & Generative AI

| Framework | Stars | Last Update | Use Case | Unification Status |
|-----------|-------|-------------|----------|-------------------|
| NVIDIA GR00T N1.6 | - | Jan 2026 | Humanoid robot foundation model | ✓ Adapter available |
| NVIDIA Cosmos Predict 2.5 | - | Jan 2026 | World foundation model, synthetic data | ✓ Native support |
| NVIDIA Cosmos Reason 2 | - | Jan 2026 | Reasoning VLM for physical AI | ✓ Native support |
| CrewAI | 100K+ | Jan 2026 | Multi-agent orchestration (v1.6.1) | ✓ Unified interface |
| LangChain/LangGraph | 95K+ | Jan 2026 | LLM-robot integration (v1.1.0) | ✓ Unified interface |
| Model Context Protocol | - | Dec 2025 | Agent-tool communication (AAIF/Linux Foundation) | ✓ Native support |
| MONAI Multimodal | - | Jan 2026 | Medical imaging + agentic AI | ✓ Integrated |

### Surgical Robotics

| Framework | Institution | Last Update | Use Case | Unification Status |
|-----------|-------------|-------------|----------|-------------------|
| ORBIT-Surgical | Stanford/JHU | Dec 2024 | Surgical task benchmarking | ✓ Primary benchmark |
| dVRK 2.4.0 | JHU | Jan 2026 | da Vinci research platform (ROS 2 Jazzy) | ✓ Bridge available |
| dVRK-Si | JHU | 2025 | Next-gen da Vinci Si/S support | ✓ Bridge available |
| SurgicalGym | - | 2025 | GPU-based surgical RL | ◐ In progress |
| Isaac Lab-Arena | NVIDIA | Jan 2026 | Large-scale policy evaluation | ✓ Benchmark integration |

---

## ★ Patient Instructions (v2.1.0)

The `patients/` directory documents a **10-page patient-facing instructional PDF** titled *Patient Instructions: Physical AI Oncology Trials*. Each page is a self-contained instruction sheet for one robot type, with a 1-sentence introduction and 3 clear numbered instructions covering arrival, interaction, and conclusion. Pages cover 10 robot types across surgical, therapeutic, diagnostic, assistive, and rehabilitative categories, each paired with a specific cancer type.

> **[Paper (PDF) — Zenodo](https://doi.org/10.5281/zenodo.18810541)** | **[LaTeX Source Files — Zenodo](https://doi.org/10.5281/zenodo.18810541)** | **[Images — Google Drive](https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax)**


```
Robot Categories (5 Clinical Categories)
=========================================
  Surgical: Surgical Robots, Cobots
  Therapeutic: RT Positioning, RT Motion-Tracking
  Diagnostic: Needle-Placement, Imaging, Steerable Needle
  Assistive: Companion Robots, Humanoids (Pediatric)
  Rehabilitative: Rehab Exoskeletons
```

See [`patients/README.md`](patients/README.md) for page-by-page instructions, text diagrams, quantitative patient data, and image descriptions.

---

## ★ Unification Standard Level (USL)

The `unification/usl/` directory implements the **Unification Standard Level (USL)** — a scoring framework for evaluating robot readiness for unified, multi-site oncology clinical trials. USL scores range from **1.0 to 10.0** across four dimensions: simulation switching, AI integration, cross-robot sharing, and clinical trial collaboration.

> **★ USL Paper (v1.8.0)** — See [`unification/usl/paper/`](unification/usl/paper/) for the comprehensive 9-page paper: *Unification Standard Level for Physical AI Oncology Trials* (DOI: [10.5281/zenodo.18778220](https://doi.org/10.5281/zenodo.18778220)). Includes LaTeX source code, complete scoring methodology, all 9 robot evaluations with dimension-by-dimension rationale, cross-category analysis, and discussion.

### Humanoid Robot Evaluations (v1.6.0)

| Robot | Manufacturer | USL Score | Level | Band |
|-------|-------------|-----------|-------|------|
| Atlas (Electric) | Boston Dynamics | **5.8** | 5 — Functional | Intermediate |
| Digit | Agility Robotics | **4.2** | 4 — Developing | Foundational |
| Optimus (Gen 2) | Tesla | **3.6** | 3 — Basic | Foundational |

```bash
# Run humanoid robot USL scoring demo
python unification/usl/humanoids/usl_humanoid_scoring.py

# Evaluate individual humanoid robots
python unification/usl/humanoids/boston_dynamics_atlas/boston_dynamics_atlas_usl.py
python unification/usl/humanoids/tesla_optimus/tesla_optimus_usl.py
python unification/usl/humanoids/agility_digit/agility_digit_usl.py
```

### Surgical Robot Evaluations (v1.5.0)

| Robot | Manufacturer | USL Score | Level | Band |
|-------|-------------|-----------|-------|------|
| da Vinci (dVRK) | Intuitive Surgical | **7.1** | 7 — Advanced | Advanced |
| Hugo RAS | Medtronic | **4.5** | 4 — Developing | Foundational |
| Versius | CMR Surgical | **3.4** | 3 — Basic | Foundational |

```bash
# Run surgical robot USL scoring demo
python unification/usl/surgical/usl_surgical_scoring.py

# Evaluate individual surgical robots
python unification/usl/surgical/intuitive_davinci/intuitive_davinci_usl.py
python unification/usl/surgical/medtronic_hugo/medtronic_hugo_usl.py
python unification/usl/surgical/cmr_versius/cmr_versius_usl.py
```

### Cobot Evaluations (v1.4.0)

| Robot | Manufacturer | USL Score | Level | Band |
|-------|-------------|-----------|-------|------|
| Franka Emika Panda | Franka Robotics | **7.4** | 7 — Advanced | Advanced |
| Kinova Gen3 7DoF | Kinova Robotics | **5.7** | 5 — Functional | Intermediate |
| UFACTORY xArm 7 | UFACTORY | **3.4** | 3 — Basic | Foundational |

```bash
# Run cobot USL scoring demo
python unification/usl/cobots/usl_scoring_framework.py

# Evaluate individual cobots
python unification/usl/cobots/franka_panda/franka_panda_usl.py
python unification/usl/cobots/kinova_gen3/kinova_gen3_usl.py
python unification/usl/cobots/ufactory_xarm7/ufactory_xarm7_usl.py
```

See `unification/usl/README.md` for the full USL standard and scoring methodology, and individual category READMEs ([humanoids/](unification/usl/humanoids/), [surgical/](unification/usl/surgical/), [cobots/](unification/usl/cobots/)) for 18 text diagrams (6 per category: results, meaning, impact, general comparison, technical specs, scoring breakdown).

---

## ★ Unification Framework

The `unification/` directory enables **seamless interoperability** between core physical AI technologies. Users can now:

### Switch Frameworks at Any Workflow Stage

```
Training (Isaac Lab) → Validation (MuJoCo) → ROS 2 Integration (Gazebo) → Deployment
       ↓                    ↓                      ↓                        ↓
     Fast GPU            Accurate            Native ROS 2              Clinical
     training            physics              sensors                   ready
```

### Key Unification Capabilities

1. **Model Conversion**: Convert robot models between URDF, MJCF, SDF, and USD formats
2. **Policy Transfer**: Export and validate policies across frameworks
3. **Physics Mapping**: Consistent contact dynamics across engines
4. **Agent Abstraction**: Framework-agnostic AI agent interfaces
5. **Cross-Platform Validation**: Verify behavior consistency

### Quick Start with Unification Tools

```python
# Detect available frameworks
from unification.cross_platform_tools.framework_detector import FrameworkDetector
detector = FrameworkDetector()
available = detector.detect_all()
print(detector.get_recommended_pipeline())

# Convert robot model to all formats
from unification.simulation_physics.urdf_sdf_mjcf_converter import UnifiedModelConverter
converter = UnifiedModelConverter()
converter.convert("robots/surgical_arm.urdf", target_formats=["mjcf", "sdf", "usd"])

# Create framework-agnostic agent
from unification.agentic_generative_ai.unified_agent_interface import UnifiedAgent
agent = UnifiedAgent(
    name="surgical_assistant",
    role="Provide surgical instruments",
    backend="crewai"  # or "langgraph", "custom"
)

# Validate policy across frameworks
from unification.cross_platform_tools.validation_suite import CrossPlatformValidator
validator = CrossPlatformValidator()
results = validator.validate_policy(
    "policies/needle_insertion.onnx",
    frameworks=["isaac", "mujoco", "pybullet"]
)
```

---

## Key Capabilities

### 1. Generative AI for Physical Systems
- **Vision-Language-Action (VLA) models** for surgical instrument manipulation (GR00T N1.6)
- **Diffusion policies** for trajectory generation in tumor resection
- **Synthetic data generation** for rare oncology scenarios (Cosmos Predict 2.5)
- **World models** (NVIDIA Cosmos) for physics-aware simulation and reasoning (Cosmos Reason 2)
- **Physical reasoning** via dual-system architecture (System 1 fast + System 2 deliberate)

### 2. Agentic AI for Clinical Workflows
- **LLM-based surgical assistants** with multimodal perception
- **Multi-agent coordination** for multi-site clinical trials (CrewAI 1.6.1, LangGraph 1.1.0)
- **Natural language robot programming** via ROS 2 Jazzy/Kilted integration
- **Autonomous task planning** for drug infusion and sample handling
- **Standardized tool integration** via Model Context Protocol (MCP) under Linux Foundation AAIF

### 3. Reinforcement Learning for Surgical Autonomy
- **Sim2real transfer** with domain randomization
- **Hierarchical RL** for complex surgical procedures
- **Multi-agent RL** for cooperative surgical assistance
- **GPU-accelerated training** reducing training time from days to hours

### 4. Digital Twin Integration
- **Patient-specific tumor models** from imaging data
- **Treatment response simulation** for chemotherapy/radiation
- **Real-time intraoperative guidance** with sensor fusion
- **Predictive outcome modeling** for trial design

---

## ★ Digital Twins for Oncology

The new `digital-twins/` directory provides comprehensive tools for creating and using patient-specific digital twins in oncology clinical trials.

### Key Capabilities

| Capability | Framework | Clinical Application |
|------------|-----------|---------------------|
| Tumor growth modeling | TumorTwin | Patient-specific progression prediction |
| Treatment simulation | Custom PK/PD | Response prediction before treatment |
| Surgical planning | Isaac Sim integration | Virtual surgery rehearsal |
| Clinical integration | FHIR/DICOM | Hospital system connectivity |

### Quick Start with Digital Twins

```python
# Create patient-specific tumor digital twin
from digital_twins.patient_modeling import TumorTwinPipeline

pipeline = TumorTwinPipeline(
    model_type="reaction_diffusion",
    tumor_type="glioblastoma"
)

patient_dt = pipeline.create_twin(
    patient_id="ONCO-2026-001",
    imaging_data={"mri": mri_array},
    tumor_segmentation=tumor_mask
)

# Calibrate to longitudinal data
patient_dt.calibrate(
    longitudinal_scans=[scan_t0, scan_t1],
    timepoints=[0, 30]  # days
)

# Predict tumor evolution
prediction = patient_dt.predict(horizon_days=180)
print(f"Predicted volume change: {prediction.metrics['volume_change_percent']:.1f}%")

# Simulate treatment response
from digital_twins.treatment_simulation import TreatmentSimulator

simulator = TreatmentSimulator(patient_twin=patient_dt)
response = simulator.predict_response(
    treatment={"type": "radiation", "total_dose_gy": 60, "fractions": 30},
    horizon_days=90
)
print(f"Predicted response: {response.response_category}")
```

See `digital-twins/README.md` for complete documentation, and `digital-twins/examples-twins/README.md` for 6 advanced engineering examples (real-time synchronization, multi-organ toxicity, adaptive radiation therapy, immunotherapy modeling, virtual trial design, and V&V).

---

## ★ Engineering Examples

Detailed engineering examples are documented in their respective directories:

| Directory | Examples | Focus Area |
|-----------|----------|------------|
| [`agentic-ai/`](agentic-ai/README.md) | 6 examples | MCP integration, ReAct planning, safety constraints, RAG compliance |
| [`digital-twins/examples-twins/`](digital-twins/examples-twins/README.md) | 6 examples | Real-time sync, toxicity modeling, adaptive RT, virtual trials, V&V |
| [`examples/`](examples/README.md) | 5 examples | Surgical training, DT planning, cross-framework, agentic workflows |
| [`examples-new/`](examples-new/README.md) | 6 examples | Safety monitoring, sensor fusion, ROS 2 deployment, calibration |
| [`tools/`](tools/README.md) | 5 CLI tools | DICOM inspection, dose calculation, trial monitoring, simulation, deployment |
| [`federation/`](federation/README.md) | 6 examples | Multi-site federation, differential privacy, secure aggregation |
| [`regulatory-submit/`](regulatory-submit/README.md) | 6 examples | Pre-Sub packages, PCCP plans, pathway classification, IEC 62304 |

---

## Validated Integration Paths

### Path 1: Surgical Robot Training Pipeline (Unified)
```
NVIDIA Isaac Lab → [unification/bridge] → MuJoCo Validation → dVRK Hardware → Clinical
```

### Path 2: Agentic Clinical Assistant (Unified)
```
LangGraph + MCP → [unified_agent_interface] → Any Robot Platform → Hospital Deployment
```

### Path 3: Multi-Framework Development
```
Train (Isaac) → Validate (MuJoCo) → Integrate (Gazebo) → Prototype (PyBullet) → Deploy
      ↑___________________________↓
         Cross-platform validation
```

---

## Dependencies

### Core Requirements
```
python>=3.10
torch>=2.5.0
numpy>=1.24.0
scipy>=1.11.0
```

### Framework-Specific
```
# NVIDIA Isaac (requires NVIDIA GPU)
isaacsim>=5.0.0
isaaclab>=2.3.0

# MuJoCo
mujoco>=3.4.0
mujoco-mjx>=3.4.0  # JAX backend

# ROS 2 (Kilted Kaiju or Jazzy)
ros-jazzy-desktop  # or ros-kilted-desktop

# Agentic AI
langchain>=1.0.0
langgraph>=1.0.0
crewai>=1.0.0
```

---

## Actively Maintained Repositories (Referenced)

All referenced repositories have been updated within October 2025 - March 2026:

| Repository | Purpose | Last Commit |
|------------|---------|-------------|
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | Robot learning framework (v2.3.1) | Dec 2024 |
| [newton-physics/newton](https://github.com/newton-physics/newton) | GPU physics engine (Linux Foundation) | Jan 2026 |
| [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) | Physics simulation (v3.4.0) | Dec 2024 |
| [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) | GPU-optimized MuJoCo | Jan 2026 |
| [orbit-surgical/orbit-surgical](https://github.com/orbit-surgical/orbit-surgical) | Surgical simulation | Sep 2024 |
| [jhu-dvrk/sawIntuitiveResearchKit](https://github.com/jhu-dvrk/sawIntuitiveResearchKit) | dVRK platform (v2.4.0) | Jan 2026 |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | GR00T N1.6 foundation model | Jan 2026 |
| [RobotecAI/rai](https://github.com/RobotecAI/rai) | ROS 2 agentic framework | Active |
| [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | Multi-agent orchestration (v1.6.1) | Jan 2026 |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | Durable agent framework (v1.1.0) | Jan 2026 |
| [modelcontextprotocol](https://github.com/modelcontextprotocol) | MCP specification (AAIF) | Jan 2026 |
| [Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI) | Medical imaging AI | Jan 2026 |
| [SCAI-Lab/ros4healthcare](https://github.com/SCAI-Lab/ros4healthcare) | Healthcare robotics | 2025 |
| [bulletphysics/bullet3](https://github.com/bulletphysics/bullet3) | Physics engine (v3.2.5) | Apr 2023 |
| [OncologyModelingGroup/TumorTwin](https://github.com/OncologyModelingGroup/TumorTwin) | Patient-specific cancer DTs | 2025 |
| [surgical-robotics-ai](https://github.com/surgical-robotics-ai) | Surgical robotics ML | Active |
| [SamuelSchmidgall/SurgicalGym](https://github.com/SamuelSchmidgall/SurgicalGym) | GPU surgical simulation | Active |
| [med-air/SurRoL](https://github.com/med-air/SurRoL) | dVRK-compatible RL platform | 2025 |

---

## Multi-Organization Cooperation

The unification framework supports collaboration across institutions:

| Organization Type | Contribution Area | Integration Point |
|-------------------|-------------------|-------------------|
| Academic Labs | Algorithms, benchmarks | ORBIT-Surgical, skill library |
| Industry R&D | Hardware, deployment | ros2_surgical, safety validation |
| Healthcare Systems | Clinical validation | Multi-site coordination |
| Regulatory Bodies | Compliance standards | IEC 62304 documentation |
| Privacy Officers | PHI/PII management, de-identification | `privacy/` framework tools |
| Regulatory Affairs | FDA/IRB/ICH-GCP compliance | `regulatory/` framework tools |

See `unification/README.md` for the complete cooperation model.

---

### Quick Start with Q1 2026 Tools

```python
# Bidirectional conversion (Objective 1)
from q1_2026_standards.objective_1 import IsaacToMuJoCoConverter
converter = IsaacToMuJoCoConverter()
converter.convert_urdf("robot.urdf", "robot.xml")

# Model validation (Objective 2)
from q1_2026_standards.objective_2 import ModelValidator
validator = ModelValidator()
report = validator.validate_model("models/dvrk_psm/", level=4)

# Benchmark suite (Objective 3)
from q1_2026_standards.objective_3 import BenchmarkRunner
runner = BenchmarkRunner()
results = runner.run("needle_insertion", model_path="robot.xml")
```

See `q1-2026-standards/README.md` for complete documentation and implementation timeline.

---

## ★ Privacy Framework

The new `privacy/` directory provides **HIPAA-compliant patient data protection** tools for AI-enabled oncology clinical trials.

### Key Capabilities

| Module | Purpose | Regulatory Basis |
|--------|---------|-----------------|
| PHI/PII Management | Detect and classify protected health information | HIPAA 45 CFR 164.514 |
| De-Identification | Safe Harbor and Expert Determination methods | 45 CFR 164.514(b) |
| Access Control | Role-based access with audit trails | 21 CFR Part 11, HIPAA Security Rule |
| Breach Response | Automated incident response and notification | 45 CFR 164.400-414 |
| DUA Templates | Data Use Agreement generation for multi-site sharing | 45 CFR 164.514(e) |

### Quick Start with Privacy Tools

```python
# Detect PHI in clinical trial data
from privacy.phi_pii_management.phi_detector import PHIDetector
detector = PHIDetector(detection_mode="comprehensive")
result = detector.scan_dataset("trial_data/enrollment_records/")

# De-identify patient data for AI model training
from privacy.de_identification.deidentification_pipeline import DeidentificationPipeline
pipeline = DeidentificationPipeline(method="safe_harbor")
result = pipeline.deidentify("trial_data/raw/", "trial_data/deidentified/")

# Generate Data Use Agreement for multi-site collaboration
from privacy.dua_templates.dua_generator import DUAGenerator
generator = DUAGenerator(template="multi_site_ai_research")
dua = generator.generate(
    data_provider="Memorial Sloan Kettering",
    data_recipient="Physical AI Oncology Consortium",
    data_description="De-identified CT imaging for AI training",
    permitted_uses=["model_training", "validation", "publication"]
)
```

See `privacy/README.md` for complete documentation.

---

## ★ Regulatory Compliance Framework

The `regulatory/` directory provides **FDA, IRB, and ICH-GCP compliance tools** for navigating the regulatory landscape of AI-enabled oncology trials. Updated March 2026.

### Key Capabilities

| Module | Purpose | Key Regulations |
|--------|---------|----------------|
| FDA Compliance | Submission tracking (510(k), De Novo, PMA, Breakthrough) | FDA AI/ML Guidance (Jan 2025) |
| IRB Management | AI-specific protocol preparation and review | SACHRP, MRCT Framework (Jul 2025) |
| ICH-GCP | E6(R3) compliance verification and audit | ICH E6(R3) (effective Sep 2025) |
| Regulatory Intelligence | Multi-jurisdiction monitoring and deadline tracking | FDA, EMA, ICH, WHO |

### Quick Start with Regulatory Tools

```python
# Track FDA submissions for AI oncology devices
from regulatory.fda_compliance.fda_submission_tracker import FDASubmissionTracker
tracker = FDASubmissionTracker(sponsor="Physical AI Oncology Consortium")
submission = tracker.create_submission(
    submission_type="de_novo",
    device_name="AI-Guided Surgical Planning System",
    intended_use="AI-assisted tumor resection planning"
)

# Verify ICH E6(R3) GCP compliance
from regulatory.ich_gcp.gcp_compliance_checker import GCPComplianceChecker
checker = GCPComplianceChecker(guideline_version="E6_R3")
report = checker.verify_compliance(
    check_categories=["digital_technology_provisions", "data_governance"]
)

# Monitor regulatory developments
from regulatory.regulatory_intelligence.regulatory_tracker import RegulatoryTracker
tracker = RegulatoryTracker(jurisdictions=["fda", "ema", "ich"])
updates = tracker.get_recent_updates(days=90)
```

See `regulatory/README.md` for complete documentation.

---

## ★ Regulatory Guidance: Physical AI Clinical Trial Unification (v2.2.0)

The `regulatory/adaption-ich-e6r3/` directory contains the **End-to-End Physical AI Oncology Clinical Trial Unification** guidance, a comprehensive LaTeX document adapting the prior ICH E6(R3) regulation for physical AI oncology trials.

> **[Guidance (PDF) -- Zenodo](https://doi.org/10.5281/zenodo.18973368)** | **[LaTeX Source Files](regulatory/adaption-ich-e6r3/source/)**

```
Guidance Structure
==================
  Section 1: Principles of Physical AI Clinical Practice
  Section 2: Investigator Responsibilities in Physical AI Trials
  Section 3: Sponsor Responsibilities in Physical AI Trials
  Section 4: Data Governance for Physical AI Trials
  Appendix A: Physical AI System Documentation
  Appendix B: Clinical Trial Protocol for Physical AI Trials
  Appendix C: Essential Records for Physical AI Clinical Trials
  Glossary:   30 Physical AI-Specific Definitions
```

Key features: 7 robot categories, 5 AI/ML types, 4 simulation frameworks, USL scoring for 9 robots, digital twin integration, federated learning, privacy/cybersecurity requirements. All content adapted from the prior ICH E6(R3) regulation (adopted 06 January 2025).

See [`regulatory/adaption-ich-e6r3/source/README.md`](regulatory/adaption-ich-e6r3/source/README.md) for build instructions.

---

## ★ Regulatory Adaptation: 21 CFR Part 50 -- Protection of Human Subjects (v2.4.0)

The `regulatory/Adaption-21-CFR-Part-50/` directory contains the **End-to-End Physical AI Oncology Clinical Trial Unification: Adaption of 21 CFR Part 50 -- Protection of Human Subjects**, a 37-page LaTeX document that modifies the prior 21 CFR Part 50 regulation in-place to incorporate Physical AI requirements throughout.

> **[Adaptation (PDF) -- Zenodo](https://doi.org/10.5281/zenodo.19040707)** | **[LaTeX Source Files](regulatory/Adaption-21-CFR-Part-50/source/)**

```
Document Structure
===================
  Subpart A: General Provisions
    §50.1  Scope (with Physical AI expansion)
    §50.3  Definitions (18 original + 17 Physical AI definitions)
  Subpart B: Informed Consent of Human Subjects
    §50.20  General Requirements (with Physical AI adaptation)
    §50.22  Exception for Minimal Risk (with Physical AI risk mapping)
    §50.23  Exception from General Requirements (with Physical AI emergency/military)
    §50.24  Exception for Emergency Research (with Physical AI community consultation)
    §50.25  Elements of Informed Consent (8 basic + 6 additional + 8 Physical AI)
    §50.27  Documentation of Informed Consent (with MCP consent tracking)
  Subpart C: Additional Protections for Subjects in Physical AI Investigations
    §50.30  Physical AI System Safety Requirements
    §50.31  IRB Review of Physical AI Investigations
    §50.32  Ongoing Consent and Subject Notification
    §50.33  Data Protection for Physical AI Investigations
    §50.34  Physical AI System Classification and Regulatory Pathways
  Subpart D: Additional Safeguards for Children in Clinical Investigations
    §50.50-§50.56 (with Physical AI adaptations for pediatric populations)
  Glossary: 30 Physical AI-Specific Definitions
  Bibliography: 19 References
```

Key features: 5 robot types (surgical, therapeutic positioning, diagnostic needle-placement, rehabilitative exoskeletons, companion monitoring), USL minimum thresholds per procedure type, MCP consent tracking (5 servers, 23 tools), HIPAA Safe Harbor de-identification, hash-chained audit trails, pre-procedure safety matrix, task-order lifecycle, and FDA regulatory pathways (510(k), De Novo, PMA, Breakthrough). All content adapted from the prior 21 CFR Part 50 regulation (public domain under 17 U.S.C. §105).

See [`regulatory/Adaption-21-CFR-Part-50/source/README.md`](regulatory/Adaption-21-CFR-Part-50/source/README.md) for build instructions.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@software{kawchak2026physicalai,
  author = {Kawchak, Kevin},
  title = {Physical AI for Oncology Clinical Trials},
  version = {2.7.1},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/kevinkawchak/physical-ai-oncology-trials}
}
```

---



## License

MIT License - See [LICENSE](LICENSE) for details.



## Contributing

Contributions welcome. Please ensure any added frameworks or tools:
1. Have been updated within the last 3 months
2. Include practical oncology clinical trial applications
3. Provide reproducible configurations
4. **Support cross-platform compatibility** (see `unification/` for guidelines)


*Last updated: March 2026*
