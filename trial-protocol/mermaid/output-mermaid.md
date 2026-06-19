## output-mermaid

Stage 1 produced a catalog of **25 new Mermaid figures** for the Phase 1,
first-in-human, combined IND/IDE protocol of on-premises LLM-directed robotic
pancreaticoduodenectomy with perioperative daraxonrasib in KRAS-mutated PDAC.
Each figure is a separate Markdown file with a real ```mermaid``` block, a
caption, its role in the protocol, and the exact `trial-protocol` source files it
draws from (Rule 5). One commit per figure was pushed in real time, exceeding the
20-figure floor of the sub-prompt schedule.

The figures were designed as a coherent set rather than decoration. They carry
the protocol's quantitative spine directly: the 28-day screening window and
24-month overall-survival endpoint (Figure 1); the 3+3 cohort sizes and n = 18
treated (Figures 2, 10, 22); the combined part 312 / part 812 regulatory pathway
with the Subpart J Physical AI overlay, Phase 0 simulation validation, and the
USL >= 7.0 surgical threshold (Figure 3); the on-premises second-opinion-oracle
LLM loop with its 640 sensor channels, 100 kHz force sampling, 3 N per-arm and
18 N cumulative force caps, 10 kHz heartbeat, 100 us watchdog, and <= 3 ms E-stop
(Figures 4, 5, 8); the eight operative phases and the three anastomoses with their
ring-tension bands (Figures 6, 11); the five-vessel safety-zone gate with its
1.0 / 1.5 / 3.0 / 5.0 mm thresholds (Figure 7); and the daraxonrasib
pause-and-restart advisory with the 29 / 3 / 0 of 32 restart distribution against
the 0.5 ng/mL trough (Figure 9).

The set also encodes the protocol's argument and governance: the
objectives-endpoints hierarchy and Schedule of Activities (Figures 13, 14); the
parallel clinical and Physical AI safety-reporting workflow with the 7-day and
15-day timelines and the minus-24-hour to plus-72-hour audit-preservation window
(Figure 15); the oversight architecture and its <= 90-day Physical AI Safety
Review Committee cadence (Figure 16); the informed-consent flow with the
Physical AI opt-out (Figure 17); and the phase-graduated staged-autonomy model
(Figure 18).

Three figures carry the heart of the case for this trial class. Figure 19 sets
out the three counterfactual scenarios in which withholding the
LLM-plus-robot-plus-medicine combination shortens progression-free and overall
survival: a resection-window collapse from human-only scheduling delay, a
vascular-injury cascade that the no-fly gate and E-stop would have averted, and a
drug-restart mistiming that the advisory would have corrected. Figure 20 pairs
each of the eight Physical AI concerns (limitations, patient safety, loss of
human workers, single-source software, proprietary-model dependency, non-domestic
open-source LLMs, overly complex workflows, and black boxes) with a concrete
mitigation. Figure 24 frames the risk-benefit balance for a population whose
five-year survival is below 13 percent.

Every figure uses the required palette - Corporate Blue `#00417A` for end goals
and the investigational system, Professional Gray `#6C757D` for process and
oversight, Classic White `#FFFFFF` for inputs, and black / grayscale for rules
and raw data - so emphasis reads by fill and stroke weight alone and translates
cleanly into the TikZ node styles of the later stages. A second-to-last commit
fixed the one GitHub-rendering risk found on review (an unescaped ampersand in
Figure 12), and the stage closed with this README and narrative.

These 25 figures are the visual contract the draft, full, and final protocols
fulfill: the draft places a bracketed pointer to each, the full stage renders
each as a TikZ `mermaidfig` with the same complexity, and the final stage polishes
every one for overlap-free boxes, correct curved-arrow looseness, and proper
spacing.
