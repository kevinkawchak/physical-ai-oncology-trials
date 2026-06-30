## prompt-ind

Your goal is to generate a new Phase 1 PDAC IND that aims to 1) hasten the entire Phase 1 IND document package process, and 2) provides 20 high quality grayscale mermaid figures, each from a unique perspective, in real time. Keep the current paper template color. Output files to kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind.

The main document to follow to create your new comprehensive IND is physical-ai-oncology-trials/blob/main/trial-ind/inputs/ReGARDD_IND_Template.docx. Abide to the TABLE OF CONTENTS format throughout the IND, and specific instructions regarding the Cover Letter and FDA 1571. Use separate .tex sections to clearly indicate and include supporting documents, where relevant. Utilize the following files to assist your process, where relevant, in physical-ai-oncology-trials/blob/main/trial-ind/inputs/: FDA-1571_Instructions_R14_03-21-2023.md, and ReGARDD-Regulatory-Guidance-for-Academic-Research-of-Drugs-and-Devices.md. Don’t use /background unless proof of research is required.

Utilize physical-ai-oncology-trials/tree/main/regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312.sty as the paper template (add a back matter section). Your new IND package will have 10x the number of characters as physical-ai-oncology-trials/tree/main/trial-documents/final-paper/publication. The IND is based on the two prior Phase 1 developments: physical-ai-oncology-trials/tree/main/trial-protocol/final-protocol/publication, physical-ai-oncology-trials/tree/main/trial-documents/final-paper/publication; and the principal investigator LLM large document guidance from physical-ai-oncology-trials/tree/main/trial-documents/inputs/llm-adoption.

Populate directories and subdirectories in physical-ai-oncology-trials/tree/main/trial-ind in a analogous manner as physical-ai-oncology-trials/tree/main/trial-protocol (but using new draft-ind, full-ind, final-ind directories, and current inputs). Do not include a publication directory under final-ind. Adapt your own mermaid/draft/full/final LaTeX file processing stages, which are enabled by generating specific sub-prompts and executing the same code sub-prompts. Each current and generated directory in physical-ai-oncology-trials/tree/main/trial-ind must have its own comprehensive README, and relevant badges.

“COVER PAGE”
Phase 1 PDAC IND: AI Generation
Draft 1.0
10.5281/zenodo.xxxxxxxx, 0009-0007-5457-8667 (with hyperlink https://orcid.org/0009-0007-5457-8667)
CEO Kevin Kawchak, ChemicalQDevice, kevink@chemicalqdevice.com
Independent research paper and practical adoption guide. It is not medical or regulatory advice and is not endorsed by the FDA, NIH, HHS, an IRB, ICH, or any sponsor. All figures derive from the author’s repository sources and are illustrative unless tied to a cited reference.
Disclaimer: This work is independent and is not endorsed or sponsored by any trial sponsor, CRO, site, IRB, regulator, or medical society; and was adapted using Claude Code Opus 4.8.
San Diego
July 1, 2026
Note: Both the IND (v1.0); and the repository (v4.3.0) (with hyperlink https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind) should be stated, where appropriate.
“COVER PAGE”
Be sure quantitative data and tables from author sources is sufficient enough for Phase 1 clinical trial acceptance. Adapt to the back matter from physical-ai-oncology-trials/blob/main/trial-documents/final-paper/publication/sections/sec-08-references-backmatter.tex.

physical-ai-oncology-trials/tree/main/trial-protocol/final-protocol/publication: Be sure to utilize this Phase 1 protocol, which specifies the LLM-Directed PDAC Robotic Daraxonrasib trial.

physical-ai-oncology-trials/tree/main/trial-documents/inputs/llm-adoption provides the exact guidance for principal investigators to create large documents, relevant to this IND.

physical-ai-oncology-trials/tree/main/trial-documents/final-paper/publication builds upon and formalizes /llm-adoption to visualize the new oncology trial process.

physical-ai-oncology-trials/tree/main/trial-documents/final-paper/publication: Be sure your grayscale mermaid diagrams are adapted in context and in color from the following figure contents, and mermaids from other documents, if relevant to this IND.
- Figure 6. large oncology documents prior, during, and after the Phase 1 trial.
- Figure 16. The six greatest-acceleration document targets
- Figure 17. Composition of the initial IND and IRB package.
- Figure 9. Pre-trial authoring.
- Figure 11. AI carries out its own after-trial authoring methods
- Figure 15. Figure grounding: each Mermaid source
- Figure 19. The three timeline buckets. State exactly how much time is saved per document
- Figure 24. The patient time-saved cascade.
- Figure 20. Employ Five name-matching and grounding verifications
- Figure 23. Update The real-world daraxonrasib PDAC document thread

Cite using physical-ai-oncology-trials/blob/main/trial-ind/inputs/references.bib. Add any new references using the same exact bibtex format. Make sure all references when compiled will have clickable URLs; and both the DOI text; and clickable DOI URLs, where relevant; and that no links run off of the right side of the page.

Create an auto-commit / auto-PR process in real-time that allows for the user to monitor branch progress without any user intervention. Do not hold commits from GitHub, instead commit after current files are generated. This is an extensive process, so the ability to monitor your branch progress throughout your generation is important. A single last update by you provides changelog, versioning, and other updates provided below.

“SUB-PROMPT SCHEDULE”
Each of the following instructions refer to adapting to the physical-ai-oncology-trials/tree/main/trial-protocol processing workflow. Learn and implement image formatting, white space formatting, and paper formatting code strategies from physical-ai-oncology-trials/tree/main/trial-protocol/final-protocol/publication (but don’t use this publication as the paper template).
1. 20+ Mermaid figures must be high quality, comprehensive, professional and professionally colored (20+ commits) (diagrams must be improved throughout the draft, full, final process below). All of the complexity and significance must translate from python mermaid diagrams into identical LaTeX based figures. Don’t re-use prior author figure from different works. There must be no overlap between components within each figure. No shortcuts, please. Everything must be new, comprehensive, look the same from Python to LaTeX, and relevant to this paper.
2. draft-ind: (the new paper’s first paper files that provides sets of bracketed text instructions that also identify exact physical-ai-oncology-trials repository files and directories for subsequent steps to process) (10+ commits) Adapt a table of contents, back matter and other supporting information
3. full-ind: (the second iteration paper needs to utilize the files and directories identified in draft-ind effectively to generate a full version). Optimize column widths for aesthetics based on the prior physical-ai-oncology-trials/tree/main/trial-protocol/final-protocol/publication methods and the amount of text per column. Learn and verify twice that each figure a) has no text box and arrow overlaps, b) has curved arrows that always contain the correct amount of specified looseness, and c) has proper spacings between boxes. Again LaTeX figures need to have the same complexity and completeness as the Python mermaid figures. (10+ commits)
4. final-ind: (your context and formatting quality should reach maximum quality here. You need to spend time double verifying all context and formatting is improved from full-ind) (learn from and implement corrections you identify from full-ind). Learn and implement the author’s /clearpage, table formatting column widths, and other types of /vspace and /hspace formatting methods throughout all figures and text. Learn from and implement the author’s other corrections/proof reading techniques to create the polished final-ind source files. (10+ commits)
“SUB-PROMPT SCHEDULE”

“RULES”
1. Commit to the physical-ai-oncology-trials repository with a comprehensive README. All subdirectories need to have detailed READMEs. Do not commit to other repositories
2. Only Claude Code Opus 4.8 (1M Context) Ultracode can be used throughout all of this single prompt and sub-prompts. Do not stall, ask questions, or go into plan mode
3. No png or jpg files are allowed
4. Use tables where relevant, make sure each table is the width of the body text, and column widths will yield professional formatted tables
5. Each README.md for each directory must be comprehensive and state which files from other directories were used and where
6. For each sub-prompt: 1 commit is required for each of the following (main.tex, .sty, .bib, and README); and 1 commit is required for each of the paper’s .tex sections (1 .tex file per section) that each correspond to main.tex (this is different than the paper template, with each section needing a .tex)
7. For each sub-prompt generation: the 2nd to last commit must fix all of your errors for all files. For the last commit, perform the remaining repository updates defined below
8. All commits and PRs must be submitted to GitHub in real-time the moment they are generated for user viewing. Do not hold commits and PRs from GitHub as they are completed
9. You have permission to commit, commit to main, merge, and create PRs in GitHub
10. Do not take shortcuts from sub-prompt to sub-prompt: every stage must be fully developed. All files generated from prompts must be present and working. Each set of LaTeX files must compile properly in Overleaf by the author
11. Don’t stop until all tasks are completed. The user will continue the session by using the phrase “Continue” if tokens are exhausted. This is a lengthy process
12. Leave DOI in the format: 10.5281/zenodo.xxxxxxxx (with Hyperlink: https://doi.org/10.5281/zenodo.xxxxxxxx). No orcid logo is needed
13. All draft-ind, full-ind, and final-ind developments must have their own .tex outputs and tex zip file that will run properly by the author in Overleaf, as each is generated separately, and accessed in real time
14. You will be judged on how well you followed these rules and sub-prompt schedule after you finish by the author and professionals
15. Don’t take any shortcuts
“RULES”

For each LaTeX source: avoid large white empty spaces without text. Where large spacing between words exist throughout the body of text.: modify \raggedright spacing to make positioning between words look equally and properly spaced. Make sure text doesn’t run off the right side of the page anywhere. Include instructions to avoid lines with a single or two words. All tables need to use a similar format for each column width as in this example: The contents of every table cell must be properly left aligned using the example format:{>{\raggedright\arraybackslash}p{2cm}. Every width value must have a prepended \raggedright\arraybackslash to ensure no big gaps between words in tables. It is also important that tables match the exact width of the body of the text.

Avoid single lines separate from the main paragraph on the next page. Perform the final formatting steps that a senior author would take by correcting white space formatting and removing and/or adding relevant text to make each section and page look properly formatted and self standing by itself. (Don’t overcrowd the page with text, some white space formatting is ok). Make sure to correct all incorrect symbols such as SS into “§” where relevant. Use single dashes, but no em dashes, double dashes, or triple dashes throughout the paper.

Under physical-ai-oncology-trials/tree/main/trial-ind/prompts: Create a prompt-ind.md that uses a “## prompt-ind” heading followed by only this entire prompt word-for-word. Make sure only a heading and this exact prompt text is included. Create a separate output-ind.md that uses a “## output-ind” heading followed by the entire output of this prompt (containing the Claude markdown output, not the code files). Be sure only the heading with the exact Claude Code output is included.

In later commits, update physical-ai-oncology-trials/blob/main/README.md repository structures, ASCII diagrams and toc, and other affected areas in the repository (this is the only repository that needs to be edited). Add a short 425 character (with spaces) summary for this update. Add 1 additional section towards the top of the README body that further details this version (followed by the toc, repository structure, badges, etc.) Include tables and colored mermaid diagrams from the current paper where relevant throughout the main/README.

Include v4.3.0 on repository documentation headings and release notes. Be sure to fix and address errors that would cause failed checks for the single pull request (such as for lint and Python environment issues to avoid the following error during final checks): "3 failing checks
x Cl / lint-and-format (3.10) (pull...
x Cl / lint-and-format (3.11) (pull...
x Cl / lint-and-format (3.12) (pull... " Place the new release notes in releases.md under main using the format below. Update other relevant documentation such as project structures. Update the main Readme diagrams, repository structure, etc. where necessary. Update the CHANGELOG.md (v4.3.0).

"FORMAT"
Release title
v4.3.0 - [Fill in Title Here]

## Summary

## Features

## Contributors
@kevinkawchak
@claude
@google-gemini
@openai

## Notes
“FORMAT”
