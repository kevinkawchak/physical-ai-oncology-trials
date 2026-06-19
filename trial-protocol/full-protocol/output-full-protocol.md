## output-full-protocol

Stage 3 turned the draft scaffold into the full Phase 1 protocol. Every
`[DRAFTING INSTRUCTION]` from the draft was executed: each bracketed pointer
became finished, senior-author clinical-protocol prose; each referenced Stage 1
Mermaid figure was reproduced as a TikZ `mermaidfig` carrying the same nodes,
edges, and quantitative content; and each table was filled with the numbers
carried from the author sources. The draft directory was left untouched.

The thirteen NIH sections are now complete. The Statement of Compliance carries
the dual drug/device citation (21 CFR parts 50, 54, 56, 312 for the IND; parts
11, 50, 56, 812 for the significant-risk IDE; Subpart J sections 312.400 through
312.405 for the Physical AI overlay) and renders the combined IND/IDE pathway as
TikZ. The Protocol Summary compresses the whole study into a Synopsis, renders
the trial schema, and lays out a full-width Schedule of Activities across seven
visit columns. The Introduction builds the rationale around the three
counterfactual scenarios in which withholding the LLM-plus-robot-plus-medicine
combination shortens progression-free and overall survival, weaves the three
framing questions, surveys the AI-in-trials landscape (zero approved generative
LLM devices against 1,016 to 1,450+ narrow authorizations), and presents the
eight Physical AI concerns as both a TikZ figure and a full-width
concern-to-mitigation table.

The Objectives and Endpoints section opens with the three-column
Objectives / Endpoints / Justification table and the endpoint hierarchy figure;
the three co-primary endpoints are the 30-day device or procedure-related
serious-AE incidence, the daraxonrasib MTD/RP2D by 3+3, and the proportion of
procedures completing all assigned tasks without unsafe conversion. The Study
Design fixes the open-label single-arm first-in-human structure, the
daraxonrasib dose ladder in exact doses, and the device readiness analogue
(USL at or above 7.0, at least 1000 simulated procedures across at least two
frameworks, sim-to-real below 2 mm and 0.5 N). The Study Population renders the
CONSORT flow and states the inclusion and exclusion criteria around the
synthetic exemplar PAT-PDAC-0001.

The Study Intervention is the operational heart: it describes the daraxonrasib
drug and the eight-arm platform (56 degrees of freedom, 640 sensor channels,
3 N per-arm and 18 N cumulative force caps, 10 kHz heartbeat, 100 microsecond
watchdog, 3 ms cross-arm E-stop), renders the platform, the LLM advisory loop,
the perioperative pause-and-restart advisory, and the three anastomoses as TikZ,
and fills the per-arm tool-assignment and sensor-channel tables. Discontinuation
defines the device and drug stopping rules; Assessments and Procedures carries
the efficacy and safety assessments and the full AE / SAE / Physical AI AE and
unanticipated-problem machinery with the 7-day and 15-day reporting timelines and
the minus-24-hour to plus-72-hour audit-preservation window. Statistical
Considerations defines the hypotheses, the n up to 18 sample-size logic, the four
analysis populations, and the DSMB interim stopping rules. The Oversight section
renders the governance and informed-consent figures and carries consent, privacy,
data handling, monitoring, QA/QC, deviations, publication, and conflict-of-interest
policies. The Additional section renders the VVUQ ten-gate figure and the
abbreviations glossary, and the References section cites each main document with
a brief explanation and closes the back matter with the ORCID mark and DOI.

The four section batches were authored in parallel and verified for balanced
braces and environments, fully escaped special characters, defined TikZ nodes,
single hyphens, and the section symbol for codified references. The
second-to-last commit recorded a consolidated error pass, and the stage closed
with the directory README and the Overleaf zip. The full protocol is the basis
the final stage polishes to maximum quality.
