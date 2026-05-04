# 25 - 21 CFR / ICH Regulatory Compliance Wheel (NEW)

## Purpose

Add a NEW regulatory compliance wheel chart in Section 2.3 (Methods,
Repository Inputs) and Section 4.1 (Discussion, FDA RTCT) that visualizes
the mapping of Simulation 2 ten-stage journey and Simulation 3 sponsor
decisions to the relevant 21 CFR and ICH sections.

## Source Paper Section

`sections/results.tex` Section 3.2 (Sim 2 regulatory mapping) and
`sections/discussion.tex` Section 4.1 (FDA RTCT).

## Image Properties

- Filename: `images/25_regulatory_compliance_wheel.png`
- DPI: 300
- Size: 9.5 inches wide by 9.5 inches tall (square)
- Background: white (#FFFFFF)
- Palette: ten alternating wedges in two greens (#2C7A4D, #1F4E2C),
  inner ring six wedges in two blues (#1F4E79, #4A7BAA), center hub white
  with title.

## Layout

- Outer ring: 10 wedges, one per Sim 2 stage, labeled with stage number
  and the relevant regulatory citation.
- Inner ring: 6 wedges, one per Sim 3 governance / study execution / site
  / robotics layer, labeled with the regulatory anchor.
- Center hub: "21 CFR plus ICH plus FDA RTCT Compliance Map."
- Below the wheel: a 6-line legend strip mapping each citation to its
  full title.

## Mapping Data

### Outer Ring (Sim 2 Ten Stages)

1. Prescreening - 21 CFR 312 Subpart B (IND content).
2. Enrollment - 21 CFR 50 (Informed Consent).
3. Digital twin init - 21 CFR Part 11 (electronic records).
4. Robot qualification - ICH E6(R3) GCP.
5. Surgery - 21 CFR 821 (Medical Device Tracking).
6. Recovery - 21 CFR 312 Subpart D (IND Safety Reporting).
7. Immunotherapy - ICH E2A (Clinical Safety Data Management).
8. Federated learning - 21 CFR 11 (electronic records, audit trail).
9. Surveillance - 21 CFR 312.32 (IND Safety Reports).
10. Closeout - ICH E3 (Clinical Study Reports).

### Inner Ring (Sponsor Layers and Anchors)

1. Governance - 21 CFR 312 (sponsor obligations).
2. Study Execution - ICH E6(R3) GCP.
3. Site / Robotics - 21 CFR 50 plus 821.
4. Trust - 21 CFR 11 plus FDA RTCT real-time signal-sharing.
5. Audit - 21 CFR 312.55.
6. FDA Liaison - FDA RTCT Pilot 2026 (Paradigm Health conduit).

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.

## Suggested Caption

Figure 25: 21 CFR and ICH compliance wheel covering Sim 2 stages 1 through
10 and Sim 3 sponsor governance / execution / site-robotics / trust layers.
