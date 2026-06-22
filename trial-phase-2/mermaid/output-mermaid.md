## output-mermaid

This Stage 1 catalog contains the 24 GitHub-native Mermaid figures that mirror
the in-paper figures of the Phase 2, multicenter, randomized, controlled
protocol of on-premises LLM-directed robotic pancreaticoduodenectomy with
perioperative daraxonrasib (RMC-6236) in KRAS-mutated PDAC. Each figure renders
in the Phase 2 five-step palette (Burgundy `#800020` as the document color,
near-black `#2E2E2E`, medium gray `#6B6B6B`, light gray `#C9C9C9`, off-white
`#F5F5F5`) and is rebuilt with Phase 2 content rather than copied from the Phase 1
predicate catalog. The narrative below maps every figure to the protocol section
it serves.

### Compliance and regulatory spine (Section 0)

Figure 3 (combined IND / IDE pathway with the Subpart J overlay) renders the
Statement of Compliance: the daraxonrasib drug arm (IND, part 312, at the
established RP2D) and the robotic Whipple device arm (IDE, part 812,
significant-risk per &sect;812.3(m)) converge through Phase 0 simulation
validation (&ge;5000 sims, &ge;3 frameworks, trajectory &lt;1.5 mm, tip-force
&lt;0.4 N), USL readiness &ge;8.0 with fleet harmonization, the single IRB across
eight sites, and Class II classification into one randomized enrollment pathway.

### Summary and schema (Section 1)

Figure 1 (overall randomized multicenter schema) is the Synopsis schema, from
28-day screening through central 1:1 stratified randomization, the two arms,
conduct across eight harmonized sites, BICR follow-up, and the PFS and 24-month
OS endpoints. Figure 14 (Schedule of Activities visit map) renders the
screen, randomization/baseline, surgery Day 0, acute Day 1-7, Day 30, Day 90, and
long-term-to-24-month visit columns, with ctDNA at baseline and week 12 and the
Arm A flags for Phase 0 sign-off, intra-operative telemetry, and the restart
advisory.

### Introduction, rationale, and Physical AI framing (Section 2)

Figure 19 (four counterfactual scenarios) renders Scenarios A through D in which
withholding the integrated approach worsens the patient or the evidence
(scheduling-delay R0-window collapse, vascular injury, drug mistiming, and an
under-funded under-powered study versus the co-invested firewalled alternative).
Figure 20 (nine Physical AI concerns and mitigations) renders the eight device
concerns plus the ninth Phase 2 concern of financial influence and inequitable
access. Figure 21 (co-investment-to-success-likelihood) shows capital flowing
only through the firewall into the operational levers that raise power, lower the
sim-to-real gap, raise retention and equity, and so raise the probability of a
definitive answer.

### Objectives and endpoints (Section 3)

Figure 10 (objectives-to-endpoints hierarchy) renders the primary PFS objective
and the fixed-sequence key secondary hierarchy (OS, R0, ISGPS Grade B/C fistula,
MPR, week-12 KRAS ctDNA clearance), with the secondary estimation endpoints and
the exploratory analyses below the confirmatory line.

### Study design (Section 4)

Figure 4 (randomization and multicenter design) renders the central
permuted-block 1:1 randomization stratified by resectability, KRAS allele,
neoadjuvant therapy, and site, across the eight-site harmonized fleet into BICR
and blinded adjudication. Figure 18 (staged-autonomy model) renders the
Stage 1 to Stage 2 to Stage 3 graduation with the USL &ge;8.0 gate and the
prohibition of full autonomy under &sect;312.21(e).

### Study intervention (Section 6)

Figure 5 (on-premises LLM advisory control loop) renders the sensor-to-map-to-LLM
advisory-to-safety-gate-to-actuator loop with the isolated vendor stack and the
federated audit. Figure 6 (eight-arm platform architecture) renders PancreSpeed
II at 56 DOF, 640 channels, and the 10 kHz heartbeat bus. Figure 9 (daraxonrasib
pause-and-restart advisory) renders the perioperative restart keyed to ISGPS
grade and the 0.5 ng/mL trough at T+7 / T+14 / T+21. Figure 11 (three
anastomoses) renders the PJ (0.30 to 0.60 N), HJ (0.20 to 0.50 N), and GJ (0.40
to 0.80 N) ring-tension bands under Arm 5 control.

### Assessments and procedures (Section 8)

Figure 7 (five-vessel vascular safety-zone gate) renders the
soft_warning / no_fly / hard_stop zones with SMV/PV at 1.0 mm. Figure 8
(heartbeat / watchdog / E-stop architecture) renders the 10 kHz heartbeat, the
100 us watchdog, the 50 us park, the &le;3 ms cross-arm E-stop, and the &le;500 ms
hardware backup. Figure 15 (AE / Physical AI AE reporting) renders the dual AE
streams, the 7-day and 15-day timelines, the six Subpart J triggers, and the
-24 h to +72 h federated audit preservation. Figure 23 (ctDNA monitoring) renders
the KRAS clearance timeline from baseline to the week-12 key-secondary clearance
to the exploratory longitudinal dynamics.

### Statistical considerations (Section 9)

Figure 2 (CONSORT randomized participant flow) renders the approximately 245
screened to 220 randomized to roughly 110 per arm flow into the ITT, mITT,
per-protocol, and safety populations. Figure 13 (analysis populations and
interim) renders the four populations and the single group-sequential
O'Brien-Fleming efficacy/futility interim at about 60 percent of the roughly 140
events, with the binding device-safety halt overlay.

### Oversight, governance, and the capital firewall (Section 10)

Figure 16 (multicenter governance and oversight) renders the Sponsor-Investigator,
Coordinating Center, single IRB, DSMB, Physical AI Safety Review Committee, ISM,
CRO, and site PIs. Figure 17 (informed consent + Physical AI opt-out) renders the
consent and &sect;312.60(f) opt-out path to randomize-and-document. Figure 22
(capital firewall governance) renders funders/PACIF flowing through the firewall
into permitted levers and oversight while the blocked path bars any funder route
to randomization, endpoints, adjudication, analysis, or publication.

### Additional considerations (Section 11)

Figure 12 (VVUQ ten-gate assurance) renders the verification-before-generation
pipeline across 14 external standards plus 2 clinical baselines returning
ACCEPT / BLOCK / ESCALATE. Figure 24 (federated learning and hash-chained audit)
renders the eight-site federated network in which model evidence is shared while
raw data stays local, under a fleet-wide tamper-evident chain and 21 CFR part 11.

### Summary

Together the 24 figures cover every figure-bearing section of the Phase 2
protocol (Sections 0, 1, 2, 3, 4, 6, 8, 9, 10, and 11), carry the real
quantitative content of the protocol, and use the single shared five-step palette
so that the catalog renders identically in GitHub Mermaid now and as TikZ
`mermaidfig` figures in the draft, full, and final LaTeX stages.
