# Releases

Release notes for the physical-ai-oncology-trials repository.

---

Cross-Repository Paper Chunking for National Platform
v2.9.2 - Cross-Repository Paper Chunking

## Summary

Chunked 2 large LaTeX papers from external repositories into the `national-platform/` directory for token-limited AI processing. Each paper is split into 4 chunk files at logical section boundaries, with corresponding `references.bib` files and README instructions. Original files in source repositories are preserved unmodified.

## Features

- Chunked `National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex` (1,011 lines) from [national-mcp-pai-oncology-trials](https://github.com/kevinkawchak/national-mcp-pai-oncology-trials) into 4 files in `national-platform/national_mcp/`
- Chunked `main.tex` (930 lines) from [pai-oncology-trial-fl](https://github.com/kevinkawchak/pai-oncology-trial-fl) into 4 files in `national-platform/federated_learning/`
- Included `references.bib` (19 references) in `national-platform/national_mcp/`
- Included `references.bib` (27 references) in `national-platform/federated_learning/`
- Each chunk directory contains a README.md with file descriptions, reconstruction commands, and processing instructions for maintaining context across chunked files
- All original files in source repositories preserved unmodified
- All CI lint checks pass (ruff, yamllint)

## Contributors
@kevinkawchak
@claude
@openai

## Notes

- Chunk files use the original .tex programming language
- Files are split at logical section boundaries to maintain context
- Concatenating chunk files in numerical order reconstructs the original file exactly
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

11 LaTeX documents providing everything needed for California's first Physical AI oncology trial site: legislation drafts, regulations, building codes, operations, and emergency plans.

## Features

- SB 1042 California Physical AI Trial Authorization Act
- AB 2847 California Physical AI Patient Rights and Robotic Safety Act
- SB 892 California Physical AI Clinical Data Transparency and Protection Act
- San Francisco Municipal Code update for Physical AI clinical trial facilities
- California Title 22 Chapter 14 Physical AI Oncology Clinical Trial Facilities
- FDA Compliance Guide for Physical AI Oncology Clinical Trial Systems
- Physical AI Oncology Clinical Trial Facility Construction and Equipment Standards
- Physical AI Oncology Clinical Trial Facility Safety and Access Standards
- Physical AI Oncology Clinical Trial Facility Parking and Transportation Standards
- Physical AI Oncology Clinical Trial Site Activation and Operations
- Physical AI Oncology Clinical Trial Facility Emergency Preparedness and Response Plan

## Contributors
@kevinkawchak
@claude
@openai

## Notes

- All documents in LaTeX format with arxiv-style template
- Combined source available in `new-trial/site/all-documents/`
- Individual ZIP archives in `new-trial/site/zips/`
- Development by Claude Code Opus 4.6
