# Changelog

All notable changes to this repository are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [2.9.2] - 2026-03-26

### Added
- national-platform/national_mcp/ - 4 chunked .tex files from National MCP Servers paper (1,011 lines split at section boundaries)
- national-platform/national_mcp/references.bib - 19 references from the National MCP paper
- national-platform/federated_learning/ - 4 chunked .tex files from Federated Learning paper (930 lines split at section boundaries)
- national-platform/federated_learning/references.bib - 27 references from the FL paper
- README.md in national-platform/national_mcp/ with reconstruction instructions and processing guidance
- README.md in national-platform/federated_learning/ with reconstruction instructions and processing guidance

### Changed
- README.md: Updated version badge to v2.9.2, added national_mcp and federated_learning chunk directories to repository structure
- releases.md: Added v2.9.2 release notes
- CHANGELOG.md: Added v2.9.2 changelog entry

### Notes
- Source: national-mcp-pai-oncology-trials/paper/National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex
- Source: pai-oncology-trial-fl/paper/main.tex
- Original files in source repositories are NOT modified
- Chunking necessary to avoid Claude Code Opus 4.6 20,000 token-per-file processing errors
- Chunks concatenate in numerical order to reproduce originals exactly
- All CI checks pass (ruff lint, ruff format, yamllint)
- Development by Claude Code Opus 4.6

## [2.9.1] - 2026-03-25

### Added
- new-trial/site/all-documents/all_documents_chunk/ - 11 chunk files for all_documents.tex (split by document)
- regulatory/adaption-ich-e6r3/source/main_chunk/ - 4 chunk files for ICH E6(R3) adaptation
- regulatory/Adaption-21-CFR-Part-50/source/Physical_AI_21_CFR_Part_50_chunk/ - 3 chunk files for 21 CFR Part 50
- regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312_chunk/ - 5 chunk files for 21 CFR Part 312
- unification/usl/paper/usl_oncology_trials_chunk/ - 2 chunk files for USL paper
- patient-journey/paper/patient_journey_paper_chunk/ - 3 chunk files for patient journey paper
- patients/patient_robot_instructions_fixed_chunk/ - 2 chunk files for patient robot instructions
- national-platform/RESEARCH-A-CHUNK/ - 2 chunk files for RESEARCH-A
- national-platform/RESEARCH-B-CHUNK/ - 2 chunk files for RESEARCH-B
- README.md in each chunk directory with reconstruction instructions

### Changed
- README.md: Updated version badge to v2.9.1, added chunk directories to repository structure
- releases.md: Added v2.9.1 release notes
- CHANGELOG.md: Added v2.9.1 changelog entry

### Removed
- Placeholder a.md files from all chunk directories

### Notes
- Chunking necessary to avoid Claude Code Opus 4.6 20,000 token-per-file processing errors
- Original files preserved unmodified; chunks concatenate to reproduce originals exactly
- All CI checks pass (ruff lint, ruff format, yamllint)
- Development by Claude Code Opus 4.6

## [2.9.0] - 2026-03-24

### Added
- new-trial/site/ - Physical AI oncology clinical trial site documentation (11 LaTeX documents)
- new-trial/site/01-legislation-authorization/ - SB 1042 California Physical AI Trial Authorization Act
- new-trial/site/02-legislation-patient-rights/ - AB 2847 California Physical AI Patient Rights and Robotic Safety Act
- new-trial/site/03-legislation-data-transparency/ - SB 892 California Physical AI Clinical Data Transparency and Protection Act
- new-trial/site/04-city-regulations/ - San Francisco Municipal Code update
- new-trial/site/05-state-regulations/ - California Title 22 Chapter 14
- new-trial/site/06-national-regulations/ - FDA Compliance Guide
- new-trial/site/07-building-code/ - Facility Construction and Equipment Standards
- new-trial/site/08-premises-code/ - Facility Safety and Access Standards
- new-trial/site/09-parking-transportation/ - Parking and Transportation Standards
- new-trial/site/10-site-operations/ - Site Activation and Operations
- new-trial/site/11-emergency-preparedness/ - Emergency Preparedness and Response Plan
- new-trial/site/all-documents/ - Combined 11-document LaTeX source
- new-trial/site/zips/ - 12 LaTeX source archives

### Changed
- README.md: Added v2.9.0 news entry and site documentation to repository structure
- releases.md: Added v2.9.0 release notes
- CHANGELOG.md: Added v2.9.0 changelog entry

### Notes
- Development by Claude Code Opus 4.6
