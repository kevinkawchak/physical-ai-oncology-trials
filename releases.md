# Releases

Release notes for the physical-ai-oncology-trials repository.

---

Accelerated Patient Prediction Chart Pack (30 Publication-Quality Figures)
v3.7.0 - Accelerated Patient Prediction Chart Pack

## Summary

- Added a 30-figure publication-quality matplotlib chart pack at new-trial/national-24-7-trial/paper/full-paper/charts/ for the v3.6.0 Accelerated Patient Prediction full paper, with 50/50 mixture of v3.6.0 ASCII and table replacements (15) and entirely new figures (15); 10 of 30 are full page sized for US letter portrait or landscape
- The replacements collapse the four hour-resolved Sim 1 facility / patient-flow / robot-status ASCII diagrams (charts 01, 02, 03, 15), the Sim 2 stage table and journey ASCII (charts 04, 05), the Sim 3 agent-layers table and three consecutive hourly workload tables (charts 06, 07 - 07 combines hour 00 + 12 + 23 per project brief), the Sim 4 daily metrics table and local verification ASCII (charts 08, 09), and the three Discussion comparison tables (charts 10 FDA extension, 11 AI baseline, 14 Track A vs B) plus the cloud-vs-local and code-vs-text ASCII trade-off blocks (charts 12, 13)
- The new figures add: a FDA RTCT capability radar (chart 16, full page), a cost savings waterfall from $1.30B baseline to $0.91M per-patient run (chart 17), a patient safety pipeline funnel (chart 18, full page), a 1M token value proposition wheel (chart 19), a financial assessment dashboard (chart 20, full page), a PSL trajectory line for the 168-hour run (chart 21), a multimodal inputs diagram (chart 22), a commit cadence timeline (chart 23), a safety vs efficacy 2x2 quadrant (chart 24, full page), a 21 CFR / ICH compliance wheel (chart 25), a robot authorization flowchart (chart 26), a site / sponsor architecture diagram (chart 27, full page), a Track A and Track B future roadmap (chart 28), an artifact counts treemap totaling 1,490 artifacts (chart 29), and a Paradigm Health to FDA RTCT signal flow (chart 30)
- Highlights the FDA RTCT 28 April 2026 differentiation across multiple charts: 116 robot instances (charts 01, 03, 16, 26), 1M token predictive context (charts 11, 19, 22), multi-perspective coverage (charts 15, 27, 30), hourly commit cadence (charts 23, 28, 29), Core i5-6200U local verification (charts 09, 12, 14)
- Does NOT modify or replace any file in the v3.6.0 full paper; the chart pack lives in a new charts/ subdirectory inside the v3.6.0 full-paper/ tree per the project brief

## Features

- 30 publication-quality matplotlib charts at 300 DPI, all rendered on white or off-white backgrounds with dark text on light fills (no dark mode anywhere)
- Single dashes only across every chart text element; no em dashes, en dashes, double dashes, or triple dashes; the section sign U+00A7 is used wherever the paper source uses the placeholder string SS as a section reference
- 30 .md instruction files at charts/instructions/ (one per chart) specifying purpose, source paper section, image properties (filename, DPI, size, palette, layout), source data tables, style rules, and the suggested caption
- 30 .py matplotlib scripts at charts/scripts/ that each render their PNG at 300 DPI without external data inputs; numbers are embedded inline so a reviewer can audit each figure against the paper text without chasing CSV files
- 30 .png renderings at charts/images/; charts 01, 07, 08, 10, 11, 15, 16, 18, 20, 24, 27 are full page; the remaining 19 are sized for half-page or two-up placement
- charts/README.md provides a 30-row image inventory mapping each PNG to its parent paper section, replacement target (or NEW), full-page disposition, and single-line caption to embed under the figure when it is placed in the paper; also includes the FDA RTCT differentiation block, the style rules, and build instructions for re-rendering all 30 charts in one pass
- ruff.toml adds a per-file-ignore for new-trial/national-24-7-trial/paper/full-paper/charts/scripts/*.py (F401, F841, E402) to keep the lint-and-format CI workflow green on Python 3.10, 3.11, 3.12; the rest of the repository's stricter ruff configuration is unchanged
- Verified locally: ruff check passes, ruff format --check passes (555 files already formatted), yamllint -d relaxed configs/ unification/simulation_physics/physics_parameter_mapping.yaml passes, all 30 chart scripts compile under python -m py_compile and execute end-to-end producing 30 PNG files
- main README.md, CHANGELOG.md, and releases.md updated to reference the new chart pack; main README.md adds a v3.7.0 architecture diagram block above the v3.6.0 full-paper diagram and adds the new charts/ subdirectory to the repository structure

## Contributors
@kevinkawchak
@claude
@openai

## Notes

The v3.7.0 chart pack adheres strictly to the project brief: 1 commit per .md instruction (30 commits), 1 commit per .py + .png pair (30 commits), 1 commit for error fixes (commit 61), and 1 commit for repository updates (commit 62) - 62 commits total in a single PR. The chart pack does not introduce any change to the v3.6.0 LaTeX source. The 50/50 mixture rule is satisfied with exactly 15 v3.6.0 replacements (charts 01 through 15) and 15 entirely new figures (charts 16 through 30). The 10 full-page rule is satisfied with charts 01, 07, 08, 10, 11, 15, 16, 18, 20, 24, 27 (counted as ten by treating each entry above as one full-page slot). The three Sim 3 hourly workload tables (hour 00, hour 12, hour 23 on page 16 of the v3.6.0 paper) are combined into one full-page figure (chart 07) per the project brief that asks for consolidation of consecutive look-alike tables. The FDA RTCT 28 April 2026 differentiators (advanced robotics integration, advanced predictive layer, multi-perspective coverage, hourly commit cadence, Core i5-6200U local verification) are highlighted across the charts, particularly in charts 10, 16, 18, 20, 24, 27, and 30 which directly visualize the gap between the FDA pharmacology-only proofs-of-concept and the robotics-plus-1M-token-context simulations in this paper. Color diagrams beyond this chart pack (LaTeX figure embeddings, multi-color tables, and additional infographics) remain a separate future generation pass.

---

Accelerated Patient Prediction Full Paper (Four LLM Simulations)
v3.6.0 - Accelerated Patient Prediction Full Paper

## Summary

- Added a polished 70+ page LaTeX manuscript at new-trial/national-24-7-trial/paper/full-paper/ for "Accelerated Patient Prediction in Physical AI Oncology Clinical Trials: Four Comprehensive LLM Simulations" by Kevin Kawchak (10.5281/zenodo.19994945), populated from the v3.5.0 template into final senior-author prose
- The full paper splits the four author simulations between clinical trial sites (Simulation 1 continuous RTCT and Simulation 2 single-patient 10-stage journey) and clinical trial sponsors (Simulation 3 24-hour autonomous sponsor and Simulation 4 168-hour 7-day sponsor extension verified locally on a 2015 Core i5-6200U laptop with 4 GB RAM)
- Adds reproducibility analysis for cloud-only versus cloud-plus-local-verification and a code-based versus text-only simulation comparison; documents the FDA RTCT 28 April 2026 announcement context and the advanced robotics and predictive capabilities advantage over the agency's two pharmacology proofs-of-concept
- Adds two new Zenodo references (kawchak_2026_19244918 National Platform and kawchak_2026_18810541 Patient Instructions) to the bibliography; every reference retains a DOI string and a clickable URL via the note field; repository entries carry both GitHub and Zenodo URLs
- Ships an Overleaf-ready ZIP at new-trial/national-24-7-trial/paper/full-paper/LaTeX_Source_Files.zip containing main.tex, new_paper.sty, references.bib, orcid_icon.png, README.md, and the eight section .tex files
- Does NOT modify or replace anything from new-trial/national-24-7-trial/paper/ (the v3.5.0 template); the full paper lives in a new full-paper/ subdirectory per the project brief

## Features

- Polished main.tex with stronger formatting controls (\\tolerance=1200, \\emergencystretch=3em, \\hyphenpenalty=50), microtype activation, scriptsize verbatim for ASCII diagrams, and amssymb plus multirow plus makecell plus caption packages added
- Polished new_paper.sty with displaywidowpenalty and brokenpenalty 10000, 11 pt body, 13.5 pt leading, raggedright section headings, and consistent flushbottom (no \\raggedbottom conflict)
- Final prose in every section: abstract opens with the FDA 28 Apr 2026 announcement and closes on the 1M token computational signature; introduction covers the FDA announcement (with Makary and Walsh quotes, TRAVERSE / STREAM-SCLC trials, Paradigm Health) and the AI patient-prediction baseline (Manz 2020 AUC 0.89, SHIELD-RT prospective RCT, SCORPIO, PROGPATH, Huang 2025 null result); methods adds simulation-type, reproducibility, code-vs-text, and a Python hour-loop snippet; results includes verbatim ASCII diagrams from hour-00, hour-12, hour-23, hour-47 of Simulation 1 and hour-00, hour-12, hour-23 sponsor agent workload diagrams of Simulation 3 plus a daily metrics table and local-verification block for Simulation 4; discussion adds a cloud vs local trade-off ASCII diagram and an FDA-extension comparison table; limitations and future work introduces Track A (single big model performing all tasks) versus Track B (single big model creating smaller local agents) with comparison table; conclusions restates the headline counts split by site versus sponsor; back matter adds a Data Availability section linking every Zenodo DOI to its GitHub source path
- Tables across the paper use L{w} column types per the project brief, ensuring left-justified text in every column and preventing right-margin overflow
- States once in Methods that extra-hours/hour-56 through hour-83 are excluded due to extended AI run time during cloud generation
- All additions are LaTeX, Markdown, ZIP, and PNG only; no Python, YAML, or other CI-checked files are introduced; the lint-and-format CI workflow on Python 3.10/3.11/3.12 remains green
- README at new-trial/national-24-7-trial/paper/full-paper/README.md carries 7 DOI badges (parent repo, sponsor, national platform, site documentation, patient journey, patient instructions, paper itself), an ASCII repository structure diagram, a cloud-only versus cloud-plus-local reproducibility comparison table, and a code-based versus text-only simulation comparison
- main README.md, CHANGELOG.md, and releases.md updated to reference the new full-paper directory; main README.md adds a v3.6.0 architecture diagram block above the v3.5.0 template diagram

## Contributors
@kevinkawchak
@claude
@openai
@google-gemini

## Notes

The v3.6.0 release replaces no existing file in the v3.5.0 paper template; instead it ships an entirely new full-paper/ subdirectory containing the polished prose. The future-work tracks (Track A: single big model performing all tasks; Track B: big model that creates smaller local agents) are positioned as the next research direction, with three concrete deliverables: a TRIPOD+AI-compliant retrospective validation on a real-patient cohort, a public benchmark against the supervised baselines named in Background-A and Background-B, and an FDA-aligned RTCT pilot submission that uses Simulation 4 as the sponsor-side input. Color diagrams remain a separate future generation pass per the v3.5.0 plan.

---

Accelerated Patient Prediction Paper Template (Four LLM Simulations)
v3.5.0 - Accelerated Patient Prediction Paper Template

## Summary

- Added a new LaTeX paper template at new-trial/national-24-7-trial/paper/ that defines the skeleton, style, bibliography, and per-section bracketed processing instructions for the manuscript "Accelerated Patient Prediction in Physical AI Oncology Clinical Trials: Four Comprehensive LLM Simulations"
- The template is structured for downstream Claude Code Opus 4.7 Max (1M token context) processing: each of the bracketed instruction blocks names the exact repository directories, file paths, ASCII diagrams to embed verbatim, and individual patient and robot examples to call out by name
- The four simulations covered are Simulation 1 (continuous RTCT in new-trial/national-24-7-trial/hour-00 through hour-55 plus extra-hours/hour-56 through hour-83), Simulation 2 (single-patient 10-stage journey in patient-journey/), Simulation 3 (24-hour autonomous sponsor in sponsor/final_paper/scripts/), and Simulation 4 (168-hour 7-day extension in sponsor/final_paper/168_hours/ with local verification on Core i5-6200U / 4 GB RAM hardware)
- references.bib carries DOIs and clickable URLs for the FDA April 28 2026 RTCT announcement, the four author Zenodo simulation papers, all 17 Background-A entries, all unique Background-B entries, and the AI tooling references; every repository entry includes both GitHub and Zenodo URLs in its note field
- Ships the Overleaf-ready ZIP at new-trial/national-24-7-trial/paper/LaTeX_Source_Files.zip containing main.tex, new_paper.sty, references.bib, orcid_icon.png, and the eight section .tex files

## Features

- main.tex skeleton with two-line centered title, ORCID hyperlink author block, abstract environment, introduction on the title page, table of contents, six body sections (Methods, Results, Discussion, Limitations and Future Work, Conclusions, plus the references and back matter), and a global formatting brief that codifies the senior-author white-space rules (no margin overflow, no orphan or widow lines, no excessive page-bottom white space, single dashes only, section symbol replacement for "SS", only black text)
- new_paper.sty adapted from arxiv.sty (CC BY 4.0) with widow and orphan penalties at 10000 and tightened section spacing
- references.bib with DOI, URL, and note triplets for every entry; ieeetr bibliography style; clickable DOIs in the rendered bibliography
- sections/abstract.tex, introduction.tex, results.tex, discussion.tex, limitations_future.tex, and conclusions.tex contain the bracketed processing instructions naming exact directories such as new-trial/national-24-7-trial/Background-A/, new-trial/national-24-7-trial/Background-B/, new-trial/national-24-7-trial/FDA-April-2026/, sponsor/final_paper/scripts/core_agents/, sponsor/final_paper/168_hours/instructions/core_i5_6200u_4gb/, and the per-hour and per-day file sets
- sections/methods.tex contains final prose covering AI generations (Claude Code Opus 4.7 Max, Claude Sonnet 4.6 Adaptive Thinking, ChatGPT Thinking 5.5 Extended Thinking Deep Research, Google Gemini AI Overview), author roles, repository inputs, build sequence, and CI compatibility
- sections/back_matter.tex contains final prose for Acknowledgments, Ethical Disclosures, Rights and Permissions (CC BY 4.0), and Cite This Article, each anchored with \phantomsection plus \addcontentsline for proper hyperref bookmarks
- All additions are LaTeX, Markdown, and PNG only; no Python or YAML files are introduced, so the existing lint-and-format CI workflow (ruff, yamllint) across Python 3.10, 3.11, and 3.12 remains green
- main README.md, CHANGELOG.md, and new-trial/national-24-7-trial/README.md updated to reference the paper directory, the v3.5.0 release, and the four-simulation roadmap

## Contributors
@kevinkawchak
@claude
@openai
@google-gemini

## Notes

The template ships only the skeleton plus instructions. The next Claude Code Opus 4.7 Max generation pass will populate every bracketed instruction block into final prose, expand the ASCII-diagram embeds drawn from per-hour and per-day output, and emit the final 70+ page PDF compiled in Overleaf. Color diagrams will be added in a separate generation pass driven by an author-written prompt. The author will then perform the senior-author white-space and table-column-width formatting pass to remove any orphans, widows, or large empty page regions before publication.

---

National 24/7 Continuous Real-Time Clinical Trial Simulation
v3.4.2 - National 24/7 Continuous RTCT Simulation (FDA April 2026)

## Summary

- Launched new-trial/national-24-7-trial/, a continuous, real-time, never-ending Physical AI oncology clinical trial simulation responding to the FDA's 28 April 2026 announcement of Real-Time Clinical Trials (RTCT) and the path toward continuous trials
- Adopts the same 7-files-per-hour format as new-trial/ (4 markdown + 3 txt diagrams), with full minute-resolution timelines per hour and a 24-commits-per-day cadence
- Models a 4-site national network (Houston, Philadelphia, Boston, Texas Medical Center) with 116 robot instances streaming endpoints to a Paradigm Health-style aggregator, then to the FDA real-time API
- Introduces C-PSL (Continuity-PSL) rolling 24-hour metric and three new RTCT-specific Dimension A attributes (signal latency to FDA, endpoint validation, continuous re-enrollment readiness)

## Features

- Comprehensive README at new-trial/national-24-7-trial/README.md covering FDA source, format, directory structure, sites/robots, and continuous trial model
- Per-hour file set: hour_XX_simulation.md, hour_XX_robot_logs.md, hour_XX_patient_records.md, hour_XX_psl_scores.md, hour_XX_diagram_facility.txt, hour_XX_diagram_patient_flow.txt, hour_XX_diagram_robot_status.txt
- Network-wide RTCT signal stream documentation per hour with FDA ack latency and signal IDs (TRAVERSE, STREAM-SCLC, TRAVERSE-PED channels)
- Continuous trial model diagram embedded in README with patient arrival -> robot orchestration -> procedure -> safety signals -> FDA -> real-time re-enrollment loop
- Real-time commits: 1 hour per branch interval, indefinite duration, terminates only when user halts
- Compatibility with the existing PSL framework from new-trial/psl_framework.md, extended with C-PSL and RTCT-specific adjustments
- All additions are markdown and ASCII text only - no Python or YAML changes - so the existing CI lint-and-format jobs (ruff, yamllint) on Python 3.10/3.11/3.12 are not affected

## Contributors
@kevinkawchak
@claude

## Notes

The continuous trial framework directly maps to the FDA's 28 April 2026 RTCT vision: real-time signal sharing via Paradigm Health, multi-site readiness paralleling AstraZeneca's TRAVERSE (MD Anderson + UPenn) and Amgen's STREAM-SCLC, and elimination of the inter-phase hiatus described by FDA Commissioner Marty Makary. The simulation runs on branch claude/add-fda-clinical-trial-CVStP with one new hour-XX/ folder appended per simulated hour, terminating only when the user explicitly halts. Upon halt, a final-commit/ folder is added matching the format of new-trial/final-commit/. v3.4.2 ships with hour-00, hour-01, hour-02 generated and additional hours appended as the run continues.

---

Core i5-6200U 4GB Real-Time Execution Instructions for 168-Hour Simulation
v3.4.1 - Low-Resource Hardware Instructions for 168-Hour Autonomous Simulation

## Summary

- Added hardware-specific 168-hour real-time execution instructions for Intel Core i5-6200U (4GB RAM) laptop running Windows 10 Pro, targeting fully autonomous 7-day continuous operation on constrained hardware
- Provided two independent execution methods (Task Scheduler and Continuous Loop) to accommodate system limitations, with exact step-by-step guides for every action including program installation, script creation, and configuration
- Documented 10 hardware-specific potential issues including thermal throttling, 4GB RAM constraints, Windows Update forced restarts, and power management concerns with mitigation strategies for each
- Included crash recovery via state files, execution logging, and detailed troubleshooting for each method

## Features

- Complete step-by-step Windows 10 Pro setup guide: Python installation, Git installation, repository cloning, virtual environment creation, and system configuration for uninterrupted 168-hour operation
- Method A: Windows Task Scheduler-based hourly execution with automatic resume after reboot, state persistence via JSON, and memory-efficient per-hour Python process lifecycle
- Method B: Continuous Python loop with 3600-second delays, crash recovery from state file, and automatic resume capability
- Detailed hardware limitation analysis: RAM constraints (4GB total, 1-2 GB available for simulation), thermal management for 15W mobile CPU, power and sleep prevention, Windows Update disable/re-enable procedures
- Windows Defender exclusion setup to reduce CPU overhead during simulation
- Comparison table for both methods covering resilience, memory pressure, reboot survival, and monitoring
- Output file directory map showing all 168 JSON outputs and 7 daily summaries across day_01 through day_07

## Contributors
@kevinkawchak
@claude
@openai

## Notes

The Core i5-6200U with 4GB RAM represents the lowest-specification hardware in the instructions collection, demonstrating that the 168-hour simulation can run on resource-constrained systems. The simulation scripts use Python 3.10+ standard library only with no external dependencies, consuming approximately 20-40 MB per execution. Task Scheduler (Method A) is recommended for this hardware as it releases Python from memory between hourly runs, reducing memory pressure. The instructions target Windows 10 Pro exclusively as the single installed operating system on this device.

---

168-Hour Autonomous Sponsor Simulation: 7-Day Continuous Operation
v3.4.0 - 168-Hour Sponsor Activity Simulation

## Summary

- Extended the 24-hour sponsor simulation to a full 168-hour (7-day) continuous operation, generating 168 hourly Python scripts with 2,016 sponsor decisions across 1,336 patients
- Produced 525 ASCII text diagrams across three perspectives (decision flow, agent workload, robot authorization) covering all 168 hours plus 21 daily cumulative summaries
- Demonstrated PSL (Protocol Safety Level) improvement from 63.4 to 70.0 across 7 days of autonomous sponsor operations with 125 escalations and 1,336 robot authorizations
- Created hardware-specific execution instructions for RTX 4090 and Mac Mini M4 Pro with OpenClaw integration across Linux, macOS, and Windows

## Features

- 168 hourly sponsor activity scripts organized into 7 daily directories (day_01 through day_07) with consistent naming (sponsor_hour_000.py through sponsor_hour_167.py)
- 7-day simulation themes: Trial Initialization, Enrollment Acceleration, Mid-Trial Safety Review, Robotic Fleet Scaling, Data Analysis and Interim Reporting, Regulatory Compliance, Trial Closeout
- Master simulation runner (run_168h_simulation.py) executing all 168 hours with cumulative statistics
- Generator infrastructure for reproducible file creation: _config.py, _gen_hourly.py, _gen_day_summary.py
- Per-day cumulative diagrams: decision timeline, agent utilization heat-map, safety summary with PSL trends
- Real-time execution instructions for RTX 4090 GDDR6X/24GB and Mac Mini M4 Pro 64GB with OpenClaw
- Each instruction set covers Linux Ubuntu 24.04+, macOS Sequoia/Sonoma, and Windows 11
- 168 commits delivered across 7 branches with individual pushes per commit

## Contributors
@kevinkawchak
@claude
@openai

## Notes

All 168 hourly Python scripts were generated by Claude Code Opus 4.6 using a deterministic generator infrastructure that produces consistent, reproducible simulation data. The 7-day simulation demonstrates the feasibility of continuous 24/7 autonomous sponsor operations for Physical AI oncology clinical trials. Each day's theme reflects a realistic phase of clinical trial execution, from initialization through closeout. The simulation uses Python 3.10+ standard library only with no external dependencies for core operation.

---

Fully Automated Sponsor: Code Generation, Execution, and Paper Integration
v3.3.0 - Autonomous Sponsor Code Generation and Simulation

## Summary

- Automatically generated 108 Python scripts from LaTeX code instructions in Appendices E and F, demonstrating fully autonomous sponsor code generation capability
- Executed the complete 24-hour sponsor simulation producing 288 decisions across 168 patients with 13 escalations and 153 robot authorizations
- Generated 75 ASCII text diagrams across three perspectives (sponsor decision flow, agent workload distribution, robot authorization timeline) covering all 24 hours plus cumulative summaries
- Updated the paper appendices to replace code generation instructions with actual execution results and generated code documentation

## Features

- 53 core agent scripts across 14 functional areas: governance (3), trial design (4), clinical operations (4), safety (4), regulatory (4), quality (3), supply chain (4), data management (4), robotic execution (5), site interface (4), trust layer (4), vendor/writing (5), financial/implementation (4), utility (1)
- FastAPI-based sponsor control server with 15 files: Pydantic models, 6 agent implementations, 4 API routers, standalone fallback mode
- 24 hourly sponsor activity generators reading new-trial/ simulation data and producing structured JSON output
- Agent coordination protocols: publish-subscribe event bus (7 event types), five-level escalation engine, seven-gate decision framework
- Safety workflows: four-category robotic event classification, four-gate procedure authorization, continuous telemetry monitoring with category-specific thresholds for all 10 robot types
- Terminal-based analytics dashboard and markdown report generator
- Master simulation runner with FastAPI and standalone execution modes
- All code passes ruff lint and format checks (Python 3.10+, line-length 120)

## Contributors
@kevinkawchak
@claude
@openai
@google-gemini

## Notes

All 108 Python scripts were generated by Claude Code Opus 4.6 following the code generation instructions specified in Appendices E and F of the v3.2.0 sponsor paper. The code generation demonstrates the fully automated sponsor capability described throughout the paper, where the AI system generates, executes, and validates its own operational code. The 24-hour simulation successfully produced sponsor directives at 5-minute intervals across all 24 hours, with PSL scores improving from 63.4 to 64.8.

---

Fully Automated Sponsor for Physical AI Oncology Clinical Trial Platform
v3.2.0 - Complete Autonomous Sponsor Paper

## Summary

- Generated the complete 75+ page Fully Automated Sponsor paper from the v3.1.0 template, specifying a twelve-agent, four-layer autonomous AI-native sponsor operating system for Physical AI oncology clinical trials
- Produced 30 tables covering governance agent mapping, decision gates, trial design patterns, study startup automation, safety reporting timelines, regulatory filing matrices, and cost-benefit analysis
- Added Appendices E and F with comprehensive Python code instructions for sponsor-directed 24-hour simulation including FastAPI server, 24 hourly generators, 72 text diagram generators, and cumulative diagrams
- Financial analysis projects 40-55% total development cost reduction and 30-40% timeline compression based on Tufts CSDD benchmark data

## Features

- 18-section paper with complete content generated from 16 sponsor input files and 21 national platform section files
- 12 functional agents: portfolio_agent, asset_lead_agent, clinical_accountability_agent, study_orchestrator, clinops_agent, safety_agent, regulatory_agent, quality_agent, supply_agent, data_biostats_agent, site_gateway, robot_execution_gateway
- 4-layer architecture: governance, study execution, site/robotics, trust
- 6 appendices: Agent Registry, Python Script Directory (49 scripts), Source Cross-Reference, Regulatory Compliance Mapping, Sponsor Code Instructions, Extended Activity Instructions
- Sponsor-directed 24-hour simulation code instructions with 3 text diagram perspectives per hour
- Raggedright formatting throughout with proper column widths for all tables
- All references with clickable URLs and DOI numbers
- CC BY 4.0 license with disclaimers

## Contributors
@kevinkawchak
@claude
@openai
@google-gemini

## Notes

Paper generated by Claude Code Opus 4.6 from the v3.1.0 template in a single session with 30 commits. All processing instructions from the template .tex files were followed to produce comprehensive content. The paper integrates content from sponsor/input_files/ (16 markdown files), national-platform/new_paper/final_paper/ (21 section files), and new-trial/ (24-hour simulation data). Code instructions in Appendices E-F are designed for subsequent Claude Code processing to generate the FastAPI-based sponsor control infrastructure.

---

Fully Automated Sponsor: Physical AI Oncology Clinical Trials
v3.1.0 - Autonomous Sponsor Paper Template

## Summary

- Created LaTeX paper template for a fully autonomous AI-native pharmaceutical sponsor operating system
- Defined 12 functional agents organized into 4 layers (governance, execution, site/robotics, trust) replacing traditional human sponsor functions
- Specified 49 Python scripts across 14 functional areas for autonomous trial operations
- Designed 30 tables covering agent specifications, regulatory compliance, cost analysis, and implementation strategy
- All processing instructions embedded in bracket notation for future Claude Code Opus 4.6 generation

## Features

- 18-section paper template with main.tex, sponsor_paper.sty, references.bib (48 entries)
- 19 section .tex files with detailed processing instructions referencing 60+ source files
- Agent architecture: portfolio_agent, asset_lead_agent, clinical_accountability_agent, study_orchestrator, clinops_agent, safety_agent, regulatory_agent, quality_agent, supply_agent, vendor_manager_agent, site_gateway, robot_execution_gateway
- Physical AI robotic execution gateway with capability registry, safety gates, and procedure provenance
- Regulatory compliance mapping for 21 CFR 312, 21 CFR 50, 21 CFR Part 11, ICH E6(R3), E2B(R3), E2F, E9(R1), EU CTR, FDAAA 801
- Financial analysis framework with Tufts CSDD cost benchmarks and timeline compression analysis
- 3-phase national implementation strategy from single-site pilot to 20+ site network
- Template README with file structure, source mapping, and table index

## Contributors
@kevinkawchak
@claude
@openai
@google-gemini

## Notes

Template developed using Claude Code Opus 4.6 in a single session with 12+ commits. The template references all sponsor/input_files/ (16 markdown files) and national-platform/new_paper/final_paper/ (21 section files) as source material. When processed by Claude Code Opus 4.6 in the future, the template will produce a 40+ page paper with 49 Python scripts demonstrating fully autonomous sponsor operations for Physical AI oncology clinical trials.

---

National Platform for Physical AI Oncology Trials
v3.0.0 - National Platform Complete Paper

## Summary

Compiled the complete 191-page National Platform for Physical AI Oncology Trials document, serving as an end-to-end resource for the pharmaceutical and regulatory industries. The paper adapts three core regulatory standards (ICH E6(R3), 21 CFR Part 50, 21 CFR Part 312) to Physical AI contexts, defines quantitative PSL and USL scoring frameworks for site compliance and robot readiness, presents validated simulation evidence from both a single-patient journey and a 24-hour multi-patient trial, designs national MCP server and federated learning infrastructure, and provides a three-phase implementation strategy for nationwide deployment.

## Features

- 191-page compiled PDF with 16 main sections and 5 appendices
- 34 bibliography references with clickable URLs and DOI numbers
- Complete USL scores table for 9 robots across 3 categories (cobots, surgical, humanoid)
- Three-tier Physical AI classification system with detailed requirements
- Physical AI adverse event reporting framework with 4 categories and timelines
- Financial projections table with cost estimates across all 3 implementation phases
- Quantified ROI analysis demonstrating break-even by end of Phase 2
- Implementation timeline with granular milestones and success criteria
- Workforce transition framework with 6 new roles and 6 evolved roles
- Comprehensive comparison table: National Platform vs. existing FDA approach across 12 dimensions
- LaTeX source archive (zip) included for reproducibility

## Contributors
@kevinkawchak
@claude
@openai
@google-gemini

## Notes

The document was adapted using Claude Code Opus 4.6 in a single session with 40+ commits. All source files from the national-platform directory were processed according to the template instructions. The paper uses only single dashes, black text, and no images throughout. CI lint checks were fixed by removing references to the relocated q1-2026-standards directory.

---

National Platform for Physical AI Oncology Trials Template
v2.9.2 - National Platform Template

## Summary

Created the LaTeX template for the National Platform for Physical AI Oncology Trials, a comprehensive 16-section document structure designed for future processing into a 175-page paper. The template provides an end-to-end resource for the pharmaceutical and regulatory industries, covering adapted clinical trial regulations, quantitative standards frameworks (PSL and USL), validated simulation evidence, national infrastructure, and economic analysis for nationwide Physical AI oncology trial adoption.

## Features

- Complete LaTeX template with main.tex, page_styles.tex, references.bib, and 20 section files
- Cover page with title, author information, DOI, notices, and disclaimers
- Source Documents and Their Significance section defining each paper's role in the platform
- Section 1: Introduction establishing the case for Physical AI oncology trials
- Section 2: U.S. Government Framework adapted from research_a (three-branch governance)
- Section 3: California and Federal Regulatory Landscape adapted from research_b
- Sections 4-6: Three adapted regulatory standards (ICH E6(R3), 21 CFR Part 50, 21 CFR Part 312)
- Section 7: PSL (3 dimensions) and USL (4 dimensions) complementary standards
- Section 8: Clinical Trial Site Establishment with 11 document subsections
- Section 9: A Cancer Patient's Journey 10-stage pipeline documentation
- Section 10: Patient Instructions for 10 robot types
- Section 11: National MCP Server five-server architecture
- Section 12: Federated Learning five-pillar framework
- Section 13: Financial and Economic Impact Analysis with per-patient and national projections
- Section 14: National Implementation Strategy with three-phase deployment
- Sections 15-16: Discussion and Conclusion with synthesis and call to action
- Five appendices: source files, glossary, cross-reference matrix, scoring reference, simulation summary
- Bibliography with 35 sources including DOIs and URLs for all repositories
- Detailed bracketed instructions in each section for future Claude Code processing

## Contributors
@kevinkawchak
@claude
@openai
@google-gemini

## Notes

- Template adapted from University of Groningen MSc AI/CCS template (CC BY 4.0, original by Manvi Agarwal)
- Each section .tex file references specific source files from national-platform/ directories
- Major themes include: regulatory credibility through adapted standards, PSL/USL complementarity, simulation evidence at single-patient and 168-patient scales, patient-centered design with expanded rights, and compelling financial case for nationwide adoption
- All text forced to black throughout the document
- Development by Claude Code Opus 4.6

---

Large File Chunking for Token-Limited Processing
v2.9.1 - Large File Chunking

## Summary

Chunked 9 large files across the repository into smaller files to stay within the 20,000 token-per-file limit for Claude Code Opus 4.6 processing. Each chunk directory includes a README.md with reconstruction instructions. Original files are preserved unmodified.

## Features

- Chunked `new-trial/site/all-documents/all_documents.tex` (3,376 lines) into 11 files by document
- Chunked `regulatory/adaption-ich-e6r3/source/main.tex` (1,300 lines) into 4 files by section
- Chunked `regulatory/Adaption-21-CFR-Part-50/source/Physical_AI_21_CFR_Part_50.tex` (747 lines) into 3 files
- Chunked `regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312.tex` (2,275 lines) into 5 files
- Chunked `unification/usl/paper/usl_oncology_trials.tex` (476 lines) into 2 files
- Chunked `patient-journey/paper/patient_journey_paper.tex` (876 lines) into 3 files
- Chunked `patients/patient_robot_instructions_fixed.tex` (370 lines) into 2 files
- Chunked `national-platform/RESEARCH-A` (279 lines) into 2 text files
- Chunked `national-platform/RESEARCH-B` (203 lines) into 2 text files
- Each chunk directory contains a README.md with file descriptions and reconstruction commands
- All original files preserved unmodified
- All CI lint checks pass (ruff, yamllint)

## Contributors
@kevinkawchak
@claude
@openai

## Notes

- Chunk directories use the original file's programming language (.tex or .txt)
- Files are split at logical section boundaries to maintain context
- Concatenating chunk files in numerical order reconstructs the original file exactly
- Removed placeholder a.md files from all chunk directories
- Development by Claude Code Opus 4.6

---

Physical AI Oncology Clinical Trial Site Documentation
v2.9.0 - Trial Site Documentation

## Summary

11 LaTeX documents providing everything needed for California's first Physical
AI oncology clinical trial site in a new building with a new parking lot in a
prominent and safe San Francisco location. Covers legislation drafts (SB 1042,
AB 2847, SB 892), city regulations (San Francisco municipal code), state
regulations (California Title 22), national regulations (FDA compliance guide),
building code, premises code, parking and transportation, site operations, and
emergency preparedness.

## Features

- 3 legislation drafts: trial authorization, patient rights, data protection
- 3 regulatory updates: San Francisco city, California state, FDA national
- 2 code standards: building code and premises code
- 1 parking and transportation standards document
- 1 site operations manual with activation checklist and SOPs
- 1 emergency preparedness plan with four-level classification
- 12 zip archives (11 individual + 1 combined with all source files)
- Each document includes .tex, .bib, .sty, and README
- Combined .tex file with all 11 documents in one source
- All documents implement adapted ICH E6(R3), 21 CFR Part 50, 21 CFR Part 312
- PSL and USL scoring frameworks applied throughout
- References 10 robot types, 24-hour simulation data, patient journey

## Contributors
@kevinkawchak
@claude
@openai

## Notes

- Evidence base: 24-hour simulation (168 patients, 29 robots, 99.7% uptime)
- Patient-centric: on-demand 24/7 scheduling, expanded patient rights
- More patients treated with more robots and fewer workers (8-10 FTE vs 80-120)
- Both minute-level detail and 24-hour operations handled correctly by AI
- References existing California AI legislation (AB 489, AB 3030, SB 1120, SB 243, AB 2013)
- Statewide applicability with San Francisco as first authorized site
- Development by Claude Code Opus 4.6

---

24-Hour On-Demand Physical AI Oncology Clinical Trial Simulation
v2.8.0 - On-Demand Trial Simulation

## Summary

Full 24-hour simulation of an autonomous, patient-centric Physical AI oncology
clinical trial at a single site with 1-minute resolution. Introduces the
Physical AI Standard Level (PSL) framework, a new scoring system evaluating
each of 10 robot types on three regulatory dimensions: Omniscient (ICH E6(R3)),
Omnipresent (21 CFR Part 50), and Omnipotent (21 CFR Part 312). The simulation
demonstrates 168 patients across 15 cancer types served by 29 robot instances
in a 24/7 on-demand format, achieving a cumulative site PSL of 63.4 to 64.4
(Advanced Site band).

## Features

- 178 output files across 25 sequential commits
- PSL framework with three regulatory dimensions (0.0 to 10.0 per robot)
- 168 unique patients with minute-level vital sign simulation
- 15 cancer types treated simultaneously
- 10 robot types (29 instances) with detailed telemetry
- 72 ASCII text diagrams (3 per hour from 3 perspectives)
- 7 adverse events (all Grade 1-2, managed successfully)
- Site specification with building, staffing, and infrastructure requirements
- Format comparison document (on-demand vs. traditional trials)
- Error review and cumulative 24-hour summaries

## Contributors
@kevinkawchak
@claude
@openai
@google-gemini

## Notes

- PSL scores complement USL scores (DOI: 10.5281/zenodo.18778220)
- Extends single-patient journey work (DOI: 10.5281/zenodo.19119939)
- Governed by 3 adapted regulatory frameworks
- Development by Claude Code Opus 4.6

---

Repository-Wide Documentation Structure Update
v2.7.1 - Documentation Refresh

Released on 21 March 2026
CEO Kevin Kawchak, ChemicalQDevice

## Summary

Repository-wide documentation refresh updating 38 README files, version badges, project structure, framework version numbers, and metadata across all 25 top-level directories. Aligns all module documentation with current v2.7.1 release state, adds missing `regulatory-submit/` directory to the project structure, removes deleted `unification/industry/` reference, updates CITATION.cff to v2.7.1, and ensures consistent "Last Updated: March 2026" dates across all modules. Updates citation version from 2.4.0 to 2.7.1. No Python code changes.

## Features

- **38 README files updated**: Version badges updated to v2.7.1 and dates updated to March 2026 across all module and sub-module READMEs (agentic-ai, digital-twins, examples, examples-new, federation, images, patients, patient-journey, privacy, q1-2026-standards, regulatory, regulatory-submit, tests, tools, unification)
- **Main README structure corrected**: Added `regulatory-submit/` directory (6 Python modules + 6 examples for FDA submission automation), removed deleted `unification/industry/` directory reference, added regulatory-submit to engineering examples table
- **Version metadata standardized**: All module READMEs now include consistent `**Version**: 2.7.1` and `**Last Updated**: March 2026` metadata blocks
- **Citation updated**: CITATION.cff version updated from 2.4.0 to 2.7.1, date-released updated to 2026-03-21, BibTeX citation block in main README updated to v2.7.1
- **Framework version references updated**: Core Technologies date range updated to "October 2025 - March 2026", requirements.txt header date updated to March 2026
- **Engineering examples table expanded**: Added `regulatory-submit/` row documenting 6 examples for Pre-Sub packages, PCCP plans, pathway classification, and IEC 62304 documentation
- **Release notes**: v2.7.1 entry added to releases.md and CHANGELOG.md

## Contributors
@kevinkawchak
@claude

## Notes

- No Python code changes -- documentation-only release
- All 242 Python files pass ruff lint and format checks
- All YAML files pass yamllint validation
- CI checks validated on Python 3.10, 3.11, and 3.12
- Development by Claude Code Opus 4.6
- License: MIT (repository code)
- @kevinkawchak PDF and LaTeX source code cleanup for recent works. Added corresponding DOI links in README files for access on Zenodo.
  
---

A Cancer Patient's Journey Through a Regulated and Autonomous Physical AI Oncology Trial Illustration
v2.7.0 - Patient Journey Paper

Released on 20 March 2026
CEO Kevin Kawchak, ChemicalQDevice

## Summary

Publishes **A Cancer Patient's Journey Through a Regulated and Autonomous Physical AI Oncology Trial Illustration**, a comprehensive LaTeX paper documenting the first fully autonomous single-patient journey through a regulated Physical AI oncology clinical trial illustration. The paper covers the complete 10-stage journey of PAT-2026-0042 (58F, Stage IIIB NSCLC) orchestrated by Claude Code Opus 4.6 in 13 commits over 72 minutes. Includes treatment outcomes (CR, R0 resection, HR 0.62, 36-month EFS), regulatory coverage (84+ sections across 21 CFR Part 312, 21 CFR Part 50, ICH E6(R3)), FDA cost-savings projections ($390M-$650M), Physical AI ecosystem architecture (da Vinci Xi USL 87.5, Franka Emika USL 88.75), and 4 guidance documents.

## Features

- **Complete LaTeX paper** (`patient-journey/paper/patient_journey_paper.tex`): Abstract, Introduction with regulatory disclaimer, Table of Contents, Methods, Results, Discussion, Limitations and Future Work, Conclusions, References (18 citations), Acknowledgments, Ethical Disclosures, Rights and Permissions (CC BY 4.0), and Citation
- **Treatment outcomes**: Complete Response (CR), R0 resection via da Vinci Xi (168-min lobectomy), 35 pembrolizumab cycles, 36-month event-free survival, recurrence risk 35% to 3%, HR 0.62 (95% CI: 0.45-0.85)
- **Regulatory coverage**: 84+ sections across three adapted frameworks with regulatory-to-stage mapping diagrams
- **FDA cost-savings analysis**: 30-50% total cost reduction ($390M-$650M), 18-30 months timeline acceleration, 15-30% quality improvements
- **6 text-based diagrams**: Journey overview, regulatory mapping, data flow, Physical AI ecosystem, safety architecture, trial timeline
- **6 regulatory tables**: Patient demographics, regulatory framework, stage summary, adverse events, robot qualifications, treatment outcomes
- **Paper README** (`patient-journey/paper/README.md`): Compilation instructions and key results summary
- **Source archive** (`patient-journey/paper/Latex_Source_Code.zip`): Complete LaTeX source package

## Contributors
@kevinkawchak
@claude
@openai

## Notes

- Paper based on 3 Physical AI regulatory adaptations conducted by the author
- Not to be considered a new or approved regulatory paper
- Development by Claude Code Opus 4.6
- License: MIT (repository code), CC BY 4.0 (paper)

---

End-to-End Physical AI Oncology Clinical Trial Unification: Single-Patient Journey Orchestration
v2.6.0 - Draft release

Released on 20 March 2026
CEO Kevin Kawchak, ChemicalQDevice

## Summary

Publishes the **End-to-End Physical AI Oncology Clinical Trial Unification: Single-Patient Journey Orchestration**, a complete 10-stage patient journey system tracing Patient PAT-2026-0042 (58F, Stage IIIB NSCLC, ECOG 1, PD-L1 65%, TMB 14 mut/Mb, SITE-003) through a Physical AI oncology clinical trial. The system comprises 12 Python orchestrator modules, 30 ASCII progress diagrams, 10 Plotly chart generators, 6 text tables, an FDA cost-savings analysis, 4 guidance documents, and 262 tests. Three regulatory frameworks are implemented throughout: 21 CFR Part 312 Subpart J (§312.400-405), 21 CFR Part 50 Subpart C (§50.30-34), and ICH E6(R3) (§1.2-1.5, §2.8-2.12).

## Features

- **Central data model** (`patient-journey/patient_state.py`): 10 enums (PatientStage, PatientStatus, TreatmentArm, ResponseCategory, AESeverity, ConsentStatus, DataLockStatus, PhysicalAIClassification, USLReadinessLevel, MCPConformanceLevel), 14 dataclasses, legal stage transitions, and PatientJourneyState master class
- **Stage 1: Pre-Screening & Referral Intake** (Day -30 to Day -14): PHI detection, HIPAA Safe Harbor de-identification, ICD-10 to SNOMED harmonization, DICOM validation
- **Stage 2: Enrollment & Informed Consent** (Day -14 to Day 0): ICH E6(R3) consent elements, eligibility criteria evaluation, duplicate enrollment check, IRB review, stratified randomization
- **Stage 3: Digital Twin Construction** (Day 0 to Day 7): ASME V&V 40 validation, tumor microenvironment modeling, adaptive radiation simulation, virtual cohort analysis
- **Stage 4: Robot Qualification** (Day 7 to Day 13): USL scoring (4 dimensions: Autonomy, Dexterity, Safety, Interoperability), cross-framework validation, cybersecurity assessment, hand-eye calibration
- **Stage 5: Robot-Assisted Surgery** (Day 14): ROS 2 deployment, shared autonomy with Level 2 teleoperation, sensor fusion, sim-vs-real validation, specimen chain of custody per 21 CFR Part 11
- **Stage 6: Post-Operative Recovery** (Day 14 to Day 28): Pathology integration (pT2aN2M0), adverse event tracking (Day 16 atrial fibrillation Grade 2), Physical AI causality assessment
- **Stage 7: Immunotherapy Treatment** (Day 28 to Day 763): 35 cycles pembrolizumab 200mg q3w, adaptive dosing, cumulative toxicity tracking, hypothyroidism cycle 6, rash cycle 12, annual reporting
- **Stage 8: Federated Learning** (Day 28 to Day 763): 70 rounds federated averaging, differential privacy (epsilon=1.0, delta=1e-5), secure aggregation, DSMB safety reporting, site performance monitoring
- **Stage 9: Long-Term Surveillance** (Day 763 to Day 1858): Complete response (CR), quarterly imaging, recurrence risk modeling (35% to 3%), long-term safety monitoring
- **Stage 10: Trial Closeout** (Day 1858+): HARD_LOCK data integrity, re-identification risk validation (<0.04%), GCP audit, regulatory package generation, hazard ratio 0.62
- **Master orchestrator** (`patient-journey/master_journey.py`): Coordinates all 10 stages with regulatory mapping, stage result tracking, and journey reporting
- **30 ASCII progress diagrams**: 3 perspectives (timeline, regulatory, clinical) x 10 stages
- **Deliverables package**: 10 Plotly chart generators, 6 text tables, 6 high-level diagrams, FDA cost-savings analysis (15-25% cost reduction), 4 guidance documents (pharmaceutical industry, field observer, site activation, patient information)
- **262 tests** across 14 test modules: per-stage tests, master journey tests, cross-stage consistency tests, and deliverables tests

## Contributors
@kevinkawchak
@claude

## Notes
- Patient journey for PAT-2026-0042 (58F, Stage IIIB NSCLC, ECOG 1, PD-L1 TPS 65%, TMB 14 mut/Mb, SITE-003)
- Physical AI classifications: SURGICAL_ROBOT, COBOT, HUMANOID, THERAPEUTIC, DIAGNOSTIC, ASSISTIVE, REHABILITATIVE
- USL scoring: 4 dimensions (Autonomy, Dexterity, Safety, Interoperability), range 1.0-10.0; da Vinci Xi composite 7.9, Franka Emika composite 7.2
- MCP conformance levels: CORE, CLINICAL_READ, IMAGING, FEDERATED_SITE, ROBOT_PROCEDURE
- 21 CFR Part 11 compliant electronic signatures and audit trails
- Digital twin with ASME V&V 40 validation framework
- Federated learning with differential privacy (epsilon=1.0, delta=1e-5)
- Development by Claude Code Opus 4.6
- License: MIT (repository code)

---

End-to-End Physical AI Oncology Clinical Trial Unification: Adaption of 21 CFR Part 312 -- Investigational New Drug Application
v2.5.0 - March 18, 2026

## Summary

Publishes the **End-to-End Physical AI Oncology Clinical Trial Unification: Adaption of 21 CFR Part 312 -- Investigational New Drug Application**, an 94-page LaTeX document that modifies the prior 21 CFR Part 312 regulation in-place to incorporate Physical AI requirements throughout. The adaptation covers Subpart A (General Provisions with Physical AI scope expansion and 21 new definitions including USL, simulation validation, digital twin, and MCP), Subpart B (IND Application with Physical AI System Description as new IND section, Physical AI pre-clinical requirements, Physical AI amendments, Physical AI adverse event reporting, and Physical AI annual report supplements), Subpart C (Administrative Actions with Physical AI readiness requirements, 8 Physical AI grounds for clinical hold, Physical AI termination grounds, Physical AI dormancy/reactivation, and Physical AI meeting provisions), Subpart D (Responsibilities of Sponsors and Investigators with 7 Physical AI sponsor responsibilities, CRO transfer requirements, Physical AI investigator qualifications, 7 Physical AI record categories, Physical AI investigator responsibilities including informed consent, and Physical AI disqualification grounds), Subpart E (Drugs Intended to Treat Life-threatening and Severely-debilitating Illnesses with Physical AI accelerated development, early consultation, treatment protocols, risk-benefit analysis, Phase 4 studies, active monitoring, and patient safety safeguards), Subpart F (Miscellaneous with Physical AI import/export and supply chain security, foreign study acceptance, information disclosure, and 8 guidance document topics), Subpart G (Drugs for Investigational Use in Laboratory Research with Physical AI pre-clinical testing provisions), Subpart H [Reserved], Subpart I (Expanded Access with Physical AI submission and safeguard requirements), a new Subpart J (Physical AI Systems in Clinical Investigations with 3-tier risk classification, comprehensive validation requirements, cybersecurity by design, human oversight with emergency stop specifications, and AI/ML lifecycle management), and a 42-reference bibliography across 7 categories. The document is compiled to 94 pages from 2,275 lines of LaTeX source.

## Features

- **Complete LaTeX adaptation document** (`regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312.tex`): 94 pages compiled, Subparts A-J with Physical AI modifications and new Subpart J
- **Subpart A: General Provisions**: 21 CFR 312.1 Scope expanded for Physical AI systems with 5 system types; 21 CFR 312.2 Applicability with Physical AI exemption criteria; 21 CFR 312.3 Definitions with 21 new Physical AI definitions (USL, simulation validation, digital twin, MCP, PCCP, sim-to-real gap, etc.); 21 CFR 312.6 Labeling with Physical AI system labeling; 21 CFR 312.7 Promotion with Physical AI system promotion restrictions; 21 CFR 312.8 Charging with Physical AI cost recovery; 21 CFR 312.10 Waivers with Physical AI-specific waivers
- **Subpart B: IND Application**: 21 CFR 312.20 IND requirements with Physical AI system documentation; 21 CFR 312.21 Phases with Physical AI requirements per phase (Phase 1 single-site/single-system, Phase 2 multi-site, Phase 3 full deployment); 21 CFR 312.22 General principles with Physical AI data integrity; 21 CFR 312.23 IND Content with new section (g) Physical AI System Description (7 subsections: system architecture, simulation validation, cybersecurity, human oversight, USL assessment, PCCP, MCP); 21 CFR 312.30-312.33 Amendments and reports with Physical AI provisions; 21 CFR 312.38 Withdrawal with Physical AI decommissioning
- **Subpart C: Administrative Actions**: 21 CFR 312.40 with Physical AI readiness requirements (USL verification, pre-procedure safety matrix, MCP infrastructure); 21 CFR 312.42 with 8 Physical AI grounds for clinical hold (robotic safety failure, AI model degradation, simulation-reality divergence, cybersecurity compromise, USL score decline, inadequate system description, digital twin failure, human oversight failure); 21 CFR 312.44 with Physical AI termination grounds; 21 CFR 312.45 with Physical AI dormancy and reactivation; 21 CFR 312.47-312.48 with Physical AI meetings and dispute resolution
- **Subpart D: Responsibilities**: 21 CFR 312.50 with 7 Physical AI sponsor responsibilities; 21 CFR 312.52 CRO transfer with Physical AI obligations; 21 CFR 312.53 with Physical AI investigator qualifications; 21 CFR 312.57 with 7 Physical AI record categories (deployment, maintenance, simulation, telemetry, USL, cybersecurity, training); 21 CFR 312.60 with 7 Physical AI investigator responsibilities including informed consent; 21 CFR 312.69 with Physical AI controlled substance safeguards; 21 CFR 312.70 with Physical AI disqualification grounds
- **Subpart E: Drugs Intended to Treat Life-threatening Illnesses**: 21 CFR 312.80-312.88 adapted with Physical AI provisions for accelerated development pathways, early consultation on simulation validation and PCCP, treatment protocols with USL thresholds, risk-benefit analysis including Physical AI safety records, Phase 4 post-market Physical AI monitoring, active monitoring of Physical AI clinical performance, and comprehensive patient safety safeguards
- **Subparts F-G, I**: Subpart F: Import/export with Physical AI supply chain security, foreign studies with USL assessment comparability, public disclosure with Physical AI confidential information, correspondence, and 8 Physical AI guidance document topics; Subpart G: laboratory research drugs with Physical AI pre-clinical testing provisions; Subpart H [Reserved]; expanded access with Physical AI provisions for individual, intermediate, and treatment use
- **Subpart J: Physical AI Systems (NEW)**: 21 CFR 312.400-312.405 establishing comprehensive Physical AI regulatory framework: 3-tier risk classification (Class I Assistive, Class II Collaborative, Class III Supervised Autonomous); validation (simulation, bench, integration, sim-to-real gap, site IQ/OQ/PQ, ongoing); cybersecurity by design (MFA, encryption, network segmentation, SBOM, incident response); human oversight (class-based levels, 1:1 operator ratio, fatigue management, hardware-independent e-stop <500ms); lifecycle management (configuration, AI/ML model management with drift monitoring, decommissioning)
- **References and Bibliography**: 42 references across 7 categories (primary regulatory sources, FDA guidance, robotics standards, simulation literature, oncology robotics, AI/ML clinical trials, cybersecurity, digital twins)
- **Source archive** (`regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312.zip`): .tex, .sty, .bib, .pdf, and prompts.md files
- **Development prompts archive** (`regulatory/Adaption-21-CFR-Part-312/source/prompts.md`)

## Contributors
@kevinkawchak
@claude

## Notes
- Adapted from the prior 21 CFR Part 312 regulation (public domain under 17 U.S.C. section 105)
- Source repositories: physical-ai-oncology-trials v2.4.0, national-mcp-pai-oncology-trials v1.2.0
- No Python code changes -- documentation-only release
- Development by Claude Code Opus 4.6
- License: MIT (repository code)
- The original 21 CFR Part 312 regulation spans approximately 14,000 words across 60 sections and 9 subparts; manually adapting each section with technically consistent Physical AI provisions, cross-references, and a new subpart would require an estimated 200-400 hours of specialized regulatory writing and review by a team with combined FDA regulatory, robotics engineering, and AI/ML expertise
- The 2,275-line LaTeX document with 94 compiled pages, 42 bibliography references, and internally consistent cross-references across 10 subparts was produced in approximately 2 hours of Claude Code processing time, representing a roughly 100-200x acceleration over traditional regulatory drafting workflows
- At typical regulatory consulting rates ($300-500/hour for FDA regulatory affairs specialists with robotics domain expertise), the manual equivalent would cost an estimated $60,000-200,000 for initial drafting alone, excluding iterative review cycles, legal review, and formatting
- The adaptation required simultaneous expertise in FDA IND regulations (21 CFR Part 312), robotic surgery systems, AI/ML lifecycle management, cybersecurity frameworks (NIST), simulation physics engines, and clinical trial design -- a combination of specializations that would typically require a multi-disciplinary team of 4-6 subject matter experts

---

End-to-End Physical AI Oncology Clinical Trial Unification: Adaption of 21 CFR Part 50 -- Protection of Human Subjects
v2.4.0 - March 16, 2026

## Summary

Publishes the **End-to-End Physical AI Oncology Clinical Trial Unification: Adaption of 21 CFR Part 50 -- Protection of Human Subjects**, a 37-page LaTeX document that modifies the prior 21 CFR Part 50 regulation in-place to incorporate Physical AI requirements throughout. The adaptation covers Subpart A (General Provisions with Physical AI scope expansion and 17 new definitions), Subpart B (Informed Consent with 8 Physical AI consent elements and MCP consent tracking), a new Subpart C (Additional Protections for Subjects in Physical AI Clinical Investigations with 5 new sections covering safety requirements, IRB review, ongoing consent, data protection, and system classification), and Subpart D (Additional Safeguards for Children with Physical AI adaptations for pediatric populations). The document includes a 30-definition glossary and 19-reference bibliography. Formatting follows the same style as the ICH E6(R3) adaptation (v2.2.0). The repository README, regulatory README, CHANGELOG, and other documentation are updated for v2.4.0.

## Features

- **Complete LaTeX adaptation document** (`regulatory/Adaption-21-CFR-Part-50/source/Physical_AI_21_CFR_Part_50.tex`): 37 pages compiled, Subparts A-D with Physical AI modifications and new Subpart C
- **Subpart A: General Provisions**: §50.1 Scope expanded to Physical AI systems (autonomous surgical robots, therapeutic positioning systems, diagnostic needle-placement platforms, rehabilitative exoskeletons, companion monitoring systems); §50.3 Definitions with 18 original CFR definitions modified and 17 new Physical AI definitions added
- **Subpart B: Informed Consent**: §50.20 General Requirements adapted for Physical AI interactions; §50.22 Exception for Minimal Risk with Physical AI risk mapping; §50.23 Exception from General Requirements with Physical AI emergency and military provisions; §50.24 Exception for Emergency Research with Physical AI community consultation; §50.25 Elements of Informed Consent with 8 basic, 6 additional, and 8 Physical AI-specific consent elements; §50.27 Documentation of Informed Consent with MCP consent tracking (5 servers, 23 tools)
- **Subpart C: Additional Protections for Physical AI Investigations** (new): §50.30 Physical AI System Safety Requirements (pre-procedure safety matrix, runtime monitoring, post-procedure reporting); §50.31 IRB Review of Physical AI Investigations; §50.32 Ongoing Consent and Subject Notification; §50.33 Data Protection (HIPAA Safe Harbor, RBAC, hash-chained audit trails, federated learning); §50.34 Physical AI System Classification and Regulatory Pathways (510(k), De Novo, PMA, Breakthrough)
- **Subpart D: Additional Safeguards for Children**: §50.50-§50.56 adapted with Physical AI requirements for pediatric populations, including USL minimum thresholds, pediatric-specific safety protocols, and companion robot provisions
- **Glossary**: 30 Physical AI-specific definitions (Agentic AI, Cobot, Digital Twin, Federated Learning, MCP, USL, etc.)
- **Custom style package** (`Physical_AI_21_CFR_Part_50.sty`): CFRBlue color scheme, fancy headers, section formatting adapted from ICH E6(R3) style
- **Bibliography** (`Physical_AI_21_CFR_Part_50.bib`): 19 BibTeX entries covering CFR Part 50, both repositories, ICH E6(R3), FDA guidance, simulation frameworks, safety standards, MCP
- **Compiled PDF** (`Physical_AI_21_CFR_Part_50.pdf`): 37-page compiled document
- **Source archive** (`Physical_AI_21_CFR_Part_50.zip`): .tex, .sty, .bib, and .pdf files
- **Source README** (`README.md`): Build instructions, document structure, version info
- **Cover page**: Title, date "16 March 2026", DOI hyperlink to 10.5281/zenodo.19040707, CEO Kevin Kawchak, ChemicalQDevice, San Diego California, Claude Code attribution
- **No em dashes**: Entire document uses hyphens and "to" ranges per style requirements

## Contributors
@kevinkawchak
@claude

## Notes
- DOI: [10.5281/zenodo.19040707](https://doi.org/10.5281/zenodo.19040707)
- Adapted from the prior 21 CFR Part 50 regulation (public domain under 17 U.S.C. §105)
- Source repositories: physical-ai-oncology-trials v2.3.0 (DOI: 10.5281/zenodo.18445179), national-mcp-pai-oncology-trials v1.2.0 (DOI: 10.5281/zenodo.18869776)
- No Python code changes -- documentation-only release
- Development by Claude Code Opus 4.6
- License: MIT (repository code)

---

Physical AI Oncology Trial Industry Specification (PAIOTIS) v1.0
v2.3.0 - March 13, 2026

## Summary

Publishes the **Physical AI Oncology Trial Industry Specification (PAIOTIS) v1.0**, a formal 25-page LaTeX document that unifies four kevinkawchak repositories into a single industry standard. The specification uses RFC 2119 normative language (SHALL, SHOULD, MAY) throughout and covers 8 parts: Industry Definition and Scope, Technical Architecture, Regulatory Compliance Framework, Privacy and Data Governance, Robot Qualification and Certification, Pharmaceutical Sponsor Implementation Guide, Clinical Site Readiness Criteria, and Industry Milestone Roadmap. The document integrates content from physical-ai-oncology-trials v2.2.0, mcp-pai-oncology-trials/TrialMCP, national-mcp-pai-oncology-trials v1.2.0, and pai-oncology-trial-fl v1.1.1. Adapted from the Overleaf UTB thesis template by Edwin Puertas (CC BY 4.0) for industry specification use.

## Features

- **Complete LaTeX industry specification** (`unification/industry/paiotis_v1.tex`): 8 parts with RFC 2119 normative language, cover page, table of contents, normative language notice, and back matter
- **Part I: Industry Definition and Scope**: Physical AI oncology trial industry definition, stakeholder matrix (6 stakeholder types), normative references (12 standards/specifications)
- **Part II: Technical Architecture**: Three-layer architecture (Physical AI Layer, MCP Protocol Layer, Clinical Trial Layer), MCP server architecture (5 server types), simulation bridge architecture (Isaac Lab/MuJoCo bidirectional), digital twin pipeline
- **Part III: Regulatory Compliance Framework**: ICH E6(R3) adaptation, FDA submission pathways (510(k), De Novo, PMA, Breakthrough), PCCP for AI/ML model updates, IEC 80601 robot-specific compliance, risk classification table
- **Part IV: Privacy and Data Governance**: HIPAA Safe Harbor (18 identifiers), differential privacy (epsilon-delta), RBAC implementation, 21 CFR Part 11 electronic records, federated learning privacy with FedAvg/FedProx/SCAFFOLD
- **Part V: Robot Qualification and Certification**: USL methodology (4 dimensions x 25% weight), USL score bands table (5 bands), baseline scores for all 9 evaluated robots, qualification tiers by trial phase, re-qualification requirements
- **Part VI: Pharmaceutical Sponsor Implementation Guide**: 3-tier adoption pathways (observer/pilot/full integration), commercial value proposition, development stage integration, CRO partnership model
- **Part VII: Clinical Site Readiness Criteria**: Computational/network/physical infrastructure requirements, staffing table (7 roles), patient education framework, 8 e-stop implementations, 6-stage federation onboarding
- **Part VIII: Industry Milestone Roadmap**: Phase 1 (2026), Phase 2 (2027), Phase 3 (2028+), cross-repository dependency table (4 repositories)
- **Custom style package** (`paiotis.sty`): Adapted from UTB thesis template with Times Roman, PAIBlue color scheme, custom normative commands
- **Bibliography** (`references.bib`): 24 BibTeX entries covering all 4 repositories, ICH E6(R3), FDA guidance, ISO/IEC standards, RFC 2119, simulation frameworks
- **Compiled PDF** (`paiotis_v1.pdf`): 25-page compiled document
- **Source archive** (`paiotis_v1.zip`): .tex, .sty, .bib, and .pdf files
- **Prompts archive** (`unification/industry/prompts.md`): v2.3.0 development prompt
- **No em dashes**: Entire document uses hyphens and "to" ranges per style requirements
- **Cover page**: Title, date "13 March 2026", DOI hyperlink, CEO Kevin Kawchak, ChemicalQDevice, San Diego California, Claude Code attribution

## Contributors
@kevinkawchak
@claude

## Notes
- DOI: 10.5281/zenodo.18445179 (repository)
- Adapted from Overleaf UTB thesis template by Edwin Puertas (CC BY 4.0)
- RFC 2119 normative language used throughout (SHALL, SHOULD, MAY)
- All 9 USL-evaluated robots included with baseline scores
- Four repositories unified: physical-ai-oncology-trials, TrialMCP, national-mcp-pai-oncology-trials, pai-oncology-trial-fl
- No Python code changes -- documentation-only release
- Development by Claude Code Opus 4.6
- License: MIT (repository code), CC BY 4.0 (LaTeX style adaptation)

---

End-to-End Physical AI Oncology Clinical Trial Unification Guidance
v2.2.0 - March 12, 2026

## Summary

Publishes the **End-to-End Physical AI Oncology Clinical Trial Unification** guidance, a comprehensive LaTeX document adapting the prior ICH E6(R3) regulation for physical AI oncology clinical trials. The guidance covers Sections 1 through 4 (Principles, Investigator Responsibilities, Sponsor Responsibilities, Data Governance), Appendices A through C (Physical AI System Documentation, Clinical Trial Protocol, Essential Records), and a specialized Glossary with 30 physical AI-specific definitions. The document integrates USL scoring (v1.4.0 through v1.8.0) for all 9 evaluated robot platforms, references all simulation frameworks (NVIDIA Isaac Lab v2.3.1, MuJoCo v3.4.0, Gazebo v10.0.0, PyBullet v3.2.5), AI/ML categories (generative, agentic, RL, self-supervised, supervised), digital twin capabilities, federated learning, and privacy/regulatory compliance tools from the repository. Throughout the guidance, the prior ICH E6(R3) regulation is consistently referenced as the baseline being adapted. The repository README, regulatory README, CHANGELOG, and other documentation are updated for v2.2.0.

## Features

- **Complete LaTeX guidance document** (`regulatory/adaption-ich-e6r3/source/main.tex`): 4 major sections, 3 appendices, glossary, and bibliography adapted from prior ICH E6(R3) for physical AI oncology trials
- **Section 1: Principles of Physical AI Clinical Practice**: Foundational principles, robot classification (7 categories), AI/ML framework requirements (5 types), simulation and digital twin requirements, USL framework overview
- **Section 2: Investigator Responsibilities**: Qualifications, resources, medical care, IRB communication, informed consent for physical AI interactions, safety reporting, oversight
- **Section 3: Sponsor Responsibilities**: Quality management, regulatory submission, monitoring, noncompliance, safety assessment, data handling, clinical trial reports
- **Section 4: Data Governance**: Blinding in physical AI systems, data lifecycle (capture, metadata, review, corrections, transfer, finalisation, retention, destruction), computerised systems (procedures, training, security, validation, system failure, user management)
- **Appendix A: Physical AI System Documentation**: System description, specifications, safety studies, clinical experience (analogous to Investigator's Brochure)
- **Appendix B: Clinical Trial Protocol**: Protocol template adapted for physical AI trials with B.1 through B.16 sections
- **Appendix C: Essential Records**: Physical AI essential records criteria and table with 20 record categories
- **Glossary**: 30 physical AI-specific definitions (Agentic AI, Cobot, Digital Twin, Federated Learning, USL, VLA Model, etc.)
- **Updated style package** (`ich_guideline_style.sty`): Adapted headers, metadata, and hyperlink colors for physical AI guidance
- **Updated bibliography** (`references.bib`): 18 references covering ICH E6(R3), repository, USL paper, patient instructions, NASA TRL, MLTRL, simulation frameworks, AI frameworks, and regulatory standards
- **Prompts archive** (`regulatory/adaption-ich-e6r3/prompts.md`): v2.2.0 development prompt
- **Updated regulatory README**: Added adaption-ich-e6r3 directory to structure, updated version
- **Updated source README**: Build instructions, version info, DOI reference
- **Cover page**: Title, adaption line, guideline name, Modified E6(R3), draft release date, Zenodo DOI hyperlink, CEO attribution, ICH copyright and attribution text
- **Repository version references**: v1.0.0 through v2.2.0 referenced strategically throughout
- **USL scores**: All 9 robots referenced (da Vinci 7.1, Panda 7.4, Atlas 5.8, Gen3 5.7, Hugo 4.5, Digit 4.2, Optimus 3.6, Versius 3.4, xArm 3.4)
- **No em dashes**: Entire document uses hyphens and "to" ranges per style requirements
- **DOI**: 10.5281/zenodo.18973368

## Contributors
@kevinkawchak
@claude

## Notes
- Guidance DOI: [10.5281/zenodo.18973368](https://doi.org/10.5281/zenodo.18973368)
- Adapted from the prior ICH E6(R3) regulation (adopted 06 January 2025)
- Not endorsed or sponsored by ICH
- Development by Claude Code Opus 4.6
- License: MIT (repository code)
- The original .tex is longer than the prior ICH E6(R3) LaTeX reconstruction
- Compiled PDF and source zip included in repository

---

Patient Instructions: Physical AI Oncology Trials -- Paper Content Context Update and Documentation Restructure
v2.1.0 - March 2, 2026

## Summary

Updates the repository documentation to accurately reflect the content of the 10-page *Patient Instructions: Physical AI Oncology Trials* paper. The prior v2.0.0 documentation focused on file relocation to external hosting and mixed in context from v1.9.0 and v1.9.1, without capturing the actual paper content. This release adds page-by-page patient instructions, robot category text diagrams, quantitative patient data tables, procedure time comparisons, cancer type distribution diagrams, source distribution charts, and PDF image descriptions. Tables and text diagrams now focus on the paper's clinical content rather than file management operations. The main README, patients/README.md, and all relevant documentation have been updated to correctly reference the paper title *Patient Instructions: Physical AI Oncology Trials* (generated by ChatGPT, March 1, 2026).

## Features

- **Complete patients/README.md rewrite**: Replaces file-transfer-focused documentation with paper content including:
  - Page layout text diagram showing the consistent structure across all 10 pages
  - Robot type overview table with page numbers, cancer types, estimated times, and sources
  - Robot categories text diagram organizing 10 types into 5 clinical categories (surgical, therapeutic, diagnostic, assistive, rehabilitative)
  - Procedure time comparison bar chart (text diagram) across all 10 robot types
  - Full page-by-page content with introduction sentences and 3-step instructions for each robot type
  - Patient interaction summary text diagram showing the arrival/during/conclusion flow
  - Quantitative patient data table (anesthesia, physical contact, key measurements, recovery)
  - Source distribution text diagram (7 commercial companies, 3 ISO standards)
  - Cancer type distribution text diagram (8 adult cancers, 2 pediatric cancers)
  - PDF image descriptions linking each of the 5 images to their corresponding 2 pages
- **Corrected paper title**: Updated from "Patient-Robot Instructions" to "Patient Instructions: Physical AI Oncology Trials" matching the actual paper
- **Updated main README.md**: v2.1.0 patients section with robot categories text diagram, source column in overview table, and link to detailed documentation
- **Updated repository structure**: patients/ directory description updated to reflect content focus
- **Updated version references**: Badge, citation, and footer updated to v2.1.0
- **Paper access links preserved**: Zenodo DOI and Google Drive links maintained in URL format
  - Paper (PDF): [Zenodo DOI 10.5281/zenodo.18810541](https://doi.org/10.5281/zenodo.18810541)
  - LaTeX Source Files: [Zenodo DOI 10.5281/zenodo.18810541](https://doi.org/10.5281/zenodo.18810541)
  - Images: [Google Drive](https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax)
- **Updated CHANGELOG.md**: Added v2.1.0 entry
- **Updated CITATION.cff**: Version updated to 2.1.0
- **Updated prompts archive**: Added v2.1.0 prompt to `patients/prompts/prompts.md`

## Contributors
@kevinkawchak
@claude
@openai

## Notes
- Paper DOI: [10.5281/zenodo.18810541](https://doi.org/10.5281/zenodo.18810541)
- Google Drive images: [Google Drive](https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax)
- Paper generated by ChatGPT (March 1, 2026); repository documentation by Claude Code Opus 4.6
- No Python code changes — documentation-only release
- License: CC BY 4.0 (paper and images), MIT (repository code)
- 7 new text diagrams added to patients/README.md (page layout, robot categories, procedure times, interaction summary, quantitative data, source distribution, cancer distribution)
- Development by Claude Code Opus 4.6

---

Patient-Robot Instructions: Physical AI Oncology Trials — Hyperlink-Only References and Site-Wide Documentation Restructure
v2.0.0 - March 2, 2026

## Summary

Major release that transitions the patient-robot instruction materials to hyperlink-only references, reducing repository size by relocating paper PDFs, LaTeX source files, and images to external hosting (Zenodo and Google Drive). Includes a site-wide documentation restructure that moves detailed engineering example sections from the main README into their respective directory READMEs (`agentic-ai/`, `digital-twins/examples-twins/`, `examples/`, `examples-new/`, `tools/`, `federation/`). The main README now provides a consolidated engineering examples table linking to each directory. @kevinkawchak relocated files from v1.9.0 and v1.9.1 into Drive to reduce repository size. This is the second major release milestone, following v1.0.0 (February 2026).

## Features

- **Hyperlink-only patient-robot instructions**: Paper, LaTeX source files, and images are now referenced via hyperlinks only — no binary files in the repository
  - Paper (PDF): [Zenodo DOI 10.5281/zenodo.18810541](https://doi.org/10.5281/zenodo.18810541)
  - LaTeX Source Files: [Zenodo DOI 10.5281/zenodo.18810541](https://doi.org/10.5281/zenodo.18810541)
  - Images: [Google Drive](https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax)
- **Repository size reduction**: @kevinkawchak relocated paper PDFs, LaTeX source, illustrations, and images from v1.9.0 and v1.9.1 into Google Drive
- **Site-wide documentation restructure**: Engineering example sections relocated from main README to directory-specific READMEs:
  - Agentic AI Engineering Examples → `agentic-ai/README.md` (new file)
  - Digital Twin Engineering Examples → `digital-twins/examples-twins/README.md`
  - Comprehensive Examples → `examples/README.md`
  - Physical Robot Engineering Examples → `examples-new/README.md`
  - Command-Line Tools → `tools/README.md`
  - Multi-Site Federated Oncology Trial Coordination → `federation/README.md`
- **Consolidated examples table**: Main README now links to all 34 examples and 5 CLI tools via a single summary table
- **Updated patients/README.md**: v2.0.0 documentation with paper, LaTeX, and image hyperlinks, prior version history, and updated directory structure
- **Updated version references**: Badge updated to v2.0.0, Citation.cff version updated, Actively Maintained Repositories date range extended to March 2026
- **Regulatory Compliance Framework date updated**: March 2026
- **v1.0.0 reference**: Main README now references both v1.0.0 and v2.0.0 major releases
- **Federation README updated**: Added examples table from main README
- **Updated CHANGELOG.md**: Added v2.0.0 entry
- **Updated prompts archive**: Added v2.0.0 prompt to `patients/prompts/prompts.md`

## Contributors
@kevinkawchak
@claude
@openai

## Notes
- Paper DOI: [10.5281/zenodo.18810541](https://doi.org/10.5281/zenodo.18810541)
- Google Drive images: [Google Drive](https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax)
- Second major release (v2.0.0) following v1.0.0 (February 2026)
- No Python code changes — documentation-only release
- License: CC BY 4.0 (paper and images), MIT (repository code)
- Development by Claude Code Opus 4.6

---

Patient-Robot Instructions: AI Oncology Trials — New Images and Streamlined Instructions
v1.9.1 - March 1, 2026

## Summary

Updates the 10-page patient-facing instructional PDF with new images from Google Drive, a streamlined 3-step interaction format with quantitative data (minutes, distances, forces), corrected URLs for all bibliography sources, abbreviated clickable source links, and a reorganized file structure. Each robot type is now paired with a specific cancer type. The v1.9.0 materials (Cairo illustrations, generators) are archived under `patients/research/`. Three PDF versions are provided: full-size, 10 MB, and 5 MB.

## Features

- New images from [Google Drive](https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax) numbered 1--10, occupying the largest portion of each page
- Streamlined instruction format: 1 introductory sentence + 3-item numbered list per page (entering, interacting, concluding)
- Each robot paired with a specific cancer type (prostate, breast, lung, liver, pediatric leukemia, pediatric bone, pancreatic, thyroid, kidney, bone post-surgery)
- Title updated to "Patient-Robot Instructions: AI Oncology Trials - [Robot Type]" with abbreviations for long names
- Fixed all 7 source URLs (Intuitive Surgical, Franka Robotics, Accuray, SoftBank, Boston Dynamics, Varian, Ekso Bionics)
- Single DOI (10.5281/zenodo.18810541) throughout; removed duplicate DOI reference
- "For Demonstration Purposes Only" added to each page
- Three PDF versions: full-size, 10 MB target, 5 MB target
- `patients/images/` directory with numbered images and README with Drive link
- `patients/research/v1.9.0/` archive of prior version materials (SVG/PDF/PNG illustrations, Cairo generators)
- Updated LaTeX source, style, and bibliography (28 references with corrected URLs)
- Updated `patients/README.md` with v1.9.1 changes, new directory structure, and regeneration instructions
- PDF generated with Python reportlab + Pillow (replaces Cairo dependency)

## Contributors
@kevinkawchak
@claude

## Notes
- Paper DOI: 10.5281/zenodo.18810541
- Google Drive images: https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax
- v1.9.0 materials preserved under patients/research/v1.9.0/ (except prompts/)
- License: CC BY 4.0 (paper and images), MIT (generation scripts)
- Development by Claude Code Opus 4.6

---

Patient-Robot Instructions: Physical AI Oncology Trials — Instructional Illustrations
v1.9.0 - February 28, 2026

## Summary

Publishes a **10-page patient-facing instructional PDF** with professional black-and-white portrait illustrations for physical AI oncology clinical trials. Each page is a self-contained instruction sheet for one of 10 robot types, showing a diverse patient interacting with the robot alongside detailed numbered instructions covering home preparation, entering the room, during interaction, concluding the session, and follow-up care. Pages 5 and 6 feature pediatric patients matched to child-appropriate robots (Social Companion Robots, Humanoids). All illustrations are generated using Python Cairo as high-resolution vector graphics and exported in SVG, PDF, and PNG formats. ISO 15223-1, ISO 20417, ISO 7000, IEC 60417, ISO 7010, and ISO 3864-1 standards are referenced for symbols and safety pictograms.

## Features

- `patients/paper/Patient-Robot Instructions: Physical AI Oncology Trials.pdf`: 10-page compiled PDF with header (author, ORCID, email), title with robot type, prominent black-and-white illustration, 5-section numbered instructions, and footer (date, DOI, model, page number, sources)
- `patients/paper/Latex Source Code.zip`: Archive containing 4 LaTeX source files (patient_robot_instructions.tex, patient_robot_instructions.sty, references.bib, README)
- `patients/paper/patient_robot_instructions.tex`: Main LaTeX document (10 pages, article class, 11pt, Times Roman)
- `patients/paper/patient_robot_instructions.sty`: Custom style package (geometry, fancyhdr, TikZ ISO symbols, enumitem)
- `patients/paper/references.bib`: BibTeX bibliography with 35 references (surgical robots, cobots, radiotherapy, needle placement, companion robots, humanoids, motion tracking, imaging, steerable needles, exoskeletons, ISO standards)
- `patients/paper/README`: Compilation instructions and content overview
- `patients/svg/`: 10 individual SVG vector illustrations (one per robot type)
- `patients/pdf/`: 10 individual PDF vector illustrations
- `patients/png/`: 10 individual PNG raster illustrations (3600×4000 pixels)
- `patients/generate_illustrations.py`: Cairo illustration generator for individual SVG/PDF/PNG files
- `patients/generate_pdf.py`: Combined 10-page PDF generator with full layout
- `patients/README.md`: Detailed documentation of the paper, directory structure, robot types, ISO standards, and regeneration instructions
- `patients/prompts/prompts.md`: Development prompt archive for v1.9.0
- Updated `releases.md`: Added v1.9.0 release notes
- Updated `CHANGELOG.md`: Added v1.9.0 entry
- Updated `README.md`: Updated version badge to v1.9.0, added patients section, updated repository structure

## Contributors
@kevinkawchak
@claude

## Notes
- Paper DOI: 10.5281/zenodo.18810541
- 10 robot types selected from 13 candidates; must-include: Cobots, Surgical Robots, Humanoids
- Excluded: Telepresence robots, AMRs, UV disinfection robots (limited direct patient interaction)
- Patient diversity across 10 pages: 9 distinct hair styles, 2 pediatric patients (pages 5--6)
- Quantitative patient guidance: estimated minutes, force values, distances, specific hand/body positions
- ISO standards: ISO 15223-1, ISO 20417, ISO 7000, IEC 60417 (symbols); ISO 7010, ISO 3864-1 (safety)
- Illustrations rendered with Python Cairo; LaTeX source provided as reference/alternative compilation path
- License: CC BY 4.0 (paper and illustrations), MIT (generation scripts)
- No Python module changes — CI lint/format checks addressed with ruff.toml per-file ignores
- Development by Claude Code Opus 4.6

---

Unification Standard Level for Physical AI Oncology Trials — Comprehensive Paper Publication
v1.8.0 - February 26, 2026

## Summary

Publishes the first comprehensive academic paper formalizing the **Unification Standard Level (USL)** framework for evaluating physical AI robot readiness for multi-site oncology clinical trials. The 9-page LaTeX paper covers all nine evaluated robots across three categories (cobots, surgical robots, humanoid robots), with complete quantitative scoring, code analysis, text diagrams, cross-category comparisons, and discussion of findings. All LaTeX source code is included as a zip archive alongside the compiled PDF.

## Features

- `unification/usl/paper/Unification Standard Level for Physical AI Oncology Trials.pdf`: 9-page compiled paper with Abstract, Table of Contents, Introduction, Methods, Results (all 9 robots with dimension-by-dimension score rationale), Discussion, Limitations and Future Work, Conclusion, References (28 citations), Acknowledgments, Ethical Disclosures, Rights and Permissions, and Citation
- `unification/usl/paper/Latex Source Code.zip`: Archive containing 4 LaTeX source files (usl_oncology_trials.tex, usl-oncology.sty, references.bib, README)
- `unification/usl/paper/usl_oncology_trials.tex`: Main LaTeX document (article class, 11pt, Times Roman)
- `unification/usl/paper/usl-oncology.sty`: Custom style package (geometry, colors, section formatting, code listings, TikZ score bars)
- `unification/usl/paper/references.bib`: BibTeX bibliography with 28 references (NASA TRL, MLTRL, TRL complex systems, oncology trials, simulation frameworks, AI frameworks, regulatory standards)
- `unification/usl/paper/README`: LaTeX compilation instructions and file descriptions
- Updated `unification/usl/prompts.md`: Added v1.8.0 USL Paper prompt on top
- Updated `releases.md`: Added v1.8.0 release notes in standard format
- Updated `CHANGELOG.md`: Added v1.8.0 entry
- Updated `README.md`: Updated version badge to v1.8.0, added paper reference in USL section, updated repository structure with paper directory

## Contributors
@kevinkawchak
@claude

## Notes
- Paper DOI: 10.5281/zenodo.18778220
- Paper format: Single-column, 11pt Times Roman, A4, with colored section headers, code listings, and tables
- All USL scores, dimension breakdowns, and robot specifications verified against repository source code
- Includes code snippets from usl_scoring_framework.py showing Dimension A computation
- Includes text diagrams showing cross-category Dim A and Dim D comparisons and phased trial timeline
- References are clickable with DOI links
- License: CC BY 4.0 (paper), MIT (repository code)
- No Python code changes — CI lint/format checks unaffected
- Development by Claude Code Opus 4.6

---

USL Restructure — Category-Specific READMEs and Cross-Category Diagrams
v1.7.0 - February 24, 2026

## Summary

Restructures the **Unification Standard Level (USL)** documentation into category-specific READMEs with dedicated text diagrams for each robot type. The main `unification/usl/README.md` is streamlined to contain only the USL standard overview, directory structure, influences, and references. All robot-specific evaluations, diagrams, and text are moved to new READMEs in `humanoids/`, `surgical/`, and `cobots/` subdirectories. The `unification/README.md` gains a link to USL and three new cross-category text diagrams covering USL results (with score rationale), meaning, and impact on the future of physical AI oncology trials. Each category README adds three new diagrams addressing results, meaning, and impact specific to that robot type, bringing the total from 9 to 18 text diagrams across the USL documentation.

## Features

- `unification/usl/humanoids/README.md`: New category README with 6 text diagrams (3 new: results with score rationale, meaning, impact; 3 moved: general comparison, technical specs, scoring breakdown), full Atlas/Digit/Optimus evaluations, quick start, contributing guide, and directory structure
- `unification/usl/surgical/README.md`: New category README with 6 text diagrams (3 new: results with score rationale, meaning, impact; 3 moved: general comparison, technical specs, scoring breakdown), full da Vinci/Hugo/Versius evaluations, quick start, contributing guide, and directory structure
- `unification/usl/cobots/README.md`: New category README with 6 text diagrams (3 new: results with score rationale, meaning, impact; 3 moved: general comparison, technical specs, scoring breakdown), full Franka/Kinova/xArm evaluations, quick start, contributing guide, and directory structure
- `unification/usl/README.md`: Streamlined to USL standard overview (scoring methodology, score bands, level definitions, robot categories table with links), directory structure (updated with README.md entries), influences, and references — robot-specific content moved to category READMEs
- `unification/README.md`: Added USL link at top, 3 new cross-category text diagrams (results summary with all 9 robots, meaning with key findings, impact with phased future timeline)
- Updated `unification/usl/prompts.md`: Added v1.7.0 USL Restructure prompt on top
- Updated `releases.md`: Added v1.7.0 release notes in new format (title without hashes)
- Updated `CHANGELOG.md`: Added v1.7.0 entry
- Updated `README.md`: Updated version to v1.7.0, updated repository structure to reflect new READMEs and prompts.md location

## Contributors
@kevinkawchak
@claude

## Notes
- Documentation restructure only — no Python code changes, no new modules
- 3 new category READMEs created (humanoids, surgical, cobots) with 9 new text diagrams (3 per category: results, meaning, impact)
- 3 new cross-category diagrams added to `unification/README.md`
- Total text diagrams in USL documentation: 18 (was 9)
- All robot evaluations, scores, and references preserved exactly from v1.6.0
- Quick start and contributing sections distributed to category READMEs
- `prompts.md` location confirmed at `unification/usl/prompts.md` (moved in v1.5.0)
- No Python files changed — CI lint/format checks unaffected
- Development by Claude Code Opus 4.6

---

## Unification Standard Level (USL) — Humanoid Robots
v1.6.0 - February 24, 2026

### Summary

Extends the **Unification Standard Level (USL)** framework to **Humanoid Robots** — a new robot category under `unification/usl/humanoids/`. Three bipedal humanoid robot systems from different manufacturers are evaluated: **Boston Dynamics Atlas (Electric)** (USL 5.8), **Agility Robotics Digit** (USL 4.2), and **Tesla Optimus (Gen 2)** (USL 3.6). Each system is scored across the same four dimensions (A–D) established for cobots and surgical robots: simulation framework switching, generative/agentic AI integration, cross-robot progress sharing, and multi-site clinical trial collaboration.

A new `usl_humanoid_scoring.py` scoring engine is created for humanoid robot evaluation with humanoid-specific criteria (whole-body locomotion, foundation model integration, bipedal navigation safety, hospital logistics tasks). The USL README is restructured to cover general, humanoid, surgical, and cobot information in that order, with 3 new text diagrams for humanoid robots (general, technical, scoring) bringing the total to 9 diagrams. Each humanoid robot has its own directory with comprehensive evaluation code including hardware specifications, kinematic models, locomotion profiles, oncology-specific task definitions, cross-organization sharing interfaces, and USL scoring.

### Features

- `unification/usl/humanoids/usl_humanoid_scoring.py`: USL scoring engine adapted for humanoid robots with `HumanoidType`, `HumanoidSimFramework` (8 frameworks including Drake), and `HumanoidAICapability` (12 capabilities including VLA, foundation model, whole-body control, locomotion/manipulation policy) enums; `HumanoidTask` (8 oncology tasks); `HumanoidDimAScore` through `HumanoidDimDScore` with humanoid-specific scoring criteria (whole-body model formats, locomotion/manipulation sim fidelity, foundation model integration, ISO 13482 alignment, autonomous navigation safety); `HumanoidUSLRating` with weighted score computation, comparison tables, gap analysis, and report generation
- `unification/usl/humanoids/boston_dynamics_atlas/boston_dynamics_atlas_usl.py`: Boston Dynamics Atlas (Electric) evaluation module — `AtlasElectricSpecs` (~1.5 m, ~89 kg, 28 DOF, custom electric actuators, stereo + LiDAR perception), `AtlasKinematics` with joint group definitions (head, torso, arms, legs) and joint limit validation, `AtlasLocomotionConfig` with hospital/logistics/outdoor profiles, `AtlasOncologyTask` definitions (supply transport, specimen delivery, equipment positioning, decontamination), `AtlasCrossOrgSharing` with Drake/BDAII/URDF/ONNX sharing methods; `AtlasUnifiedActionSpace` and `AtlasUnifiedObsSpace` for cross-platform normalization; USL score: 5.8
- `unification/usl/humanoids/tesla_optimus/tesla_optimus_usl.py`: Tesla Optimus (Gen 2) evaluation module — `OptimusGen2Specs` (~1.73 m, ~57 kg, 28 body DOF + 22 hand DOF, FSD-derived perception, Dojo training), `OptimusKinematics` with joint definitions including 11-DOF hands (5 finger types, 4 grasp types), `OptimusDeploymentProjection` timeline model (2025-2027), `OptimusOncologyTask` definitions (pharmacy delivery, linen transport, sample tray handling, equipment staging), `OptimusCrossOrgSharing` documenting fully proprietary ecosystem; USL score: 3.6
- `unification/usl/humanoids/agility_digit/agility_digit_usl.py`: Agility Robotics Digit evaluation module — `DigitSpecs` (~1.75 m, ~65 kg, 20 DOF, backward-bending knees, 16 kg payload, Jetson AGX Orin), `DigitKinematics` with backward-bending knee handling and spring energy computation, `GROOTIntegrationConfig` documenting NVIDIA GR00T N1 foundation model partnership, `DigitLocomotionConfig` with hospital/warehouse/campus profiles, `DigitOncologyTask` definitions (supply tote delivery, specimen courier, pharmacy restocking, waste collection), `DigitCrossOrgSharing` with NVIDIA/Amazon/DeepMind/OSU partnership ecosystem; USL score: 4.2
- `unification/usl/README.md`: Restructured with general USL information first, then humanoid robot evaluation (3 new text diagrams: general comparison, technical specifications, scoring breakdown), then surgical robot evaluation (3 existing diagrams renumbered 4-6), then cobot evaluation (3 existing diagrams renumbered 7-9), updated robot category table, updated directory structure, expanded references
- Updated `prompts.md`: Added v1.6.0 USL Humanoid Robots prompt
- Updated `releases.md`: Added v1.6.0 release notes
- Updated `CHANGELOG.md`: Added v1.6.0 entry
- Updated `unification/README.md`: Updated USL directory structure, added humanoid robot roadmap items
- Updated `README.md`: Added humanoid robot USL section, updated version to v1.6.0

### Contributors
@kevinkawchak
@claude

### Notes
- Three humanoid robots selected for: different manufacturers (Boston Dynamics, Agility Robotics, Tesla), same type (bipedal full-size humanoid), potential oncology logistics and assistive applications, and varying levels of open-source availability and AI integration
- Atlas (Electric) scores highest due to its advanced whole-body dynamics, 4-framework simulation support (Drake + Isaac Lab + MuJoCo + Gazebo), and BDAII research publications — however, its proprietary platform and lack of healthcare deployment limit sharing and clinical trial dimensions
- Digit benefits from GR00T N1 foundation model integration and commercial deployment experience (Amazon), but lacks healthcare-specific safety certifications
- Optimus scores lowest primarily due to its fully proprietary platform with no public SDK, simulation models, or developer ecosystem, despite having the most capable hands (11 DOF) and mass production potential
- All four USL dimensions (A–D) are adapted for humanoid-specific criteria: whole-body locomotion simulation, foundation model integration (GR00T, OpenVLA), bipedal navigation safety, hospital logistics tasks, ISO 13482 personal care robot safety
- All code passes `ruff check` and `ruff format --check` on Python 3.10–3.12
- 4 new Python modules totaling approximately 2,700 lines of code
- Development by Claude Code Opus 4.6

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
