# On-Demand Physical AI Oncology Trial Site Specification

Released on 23 March 2026
CEO Kevin Kawchak, ChemicalQDevice

The original CFR documents are in the public domain. The original ICH document
is copyrighted and may be used, reproduced, incorporated into other works,
adapted, modified, translated or distributed under a public license. This
current work is not endorsed or sponsored by CFR, ICH, or FDA; and was adapted
using Claude Code Opus 4.6.

## 1. Building Requirements

Total facility size: approximately 85,000-100,000 sq ft (7,900-9,300 sq m),
single-story or dual-story medical building designed for 24/7 autonomous
on-demand oncology trial operations with 10 robot types.

### 1.1 Clinical Treatment Areas

| Area | Quantity | Size Each | Total | Robot Type |
|------|----------|-----------|-------|------------|
| Surgical suites | 3 | 600 sq ft | 1,800 sq ft | Robot 1: Surgical |
| Biopsy stations | 4 | 200 sq ft | 800 sq ft | Robot 2: Cobots |
| Radiotherapy vaults | 3 | 1,200 sq ft | 3,600 sq ft | Robots 3 and 7 |
| CT-guided suites | 2 | 400 sq ft | 800 sq ft | Robot 4: Needle-Placement |
| Companion play areas | 5 | 150 sq ft | 750 sq ft | Robot 5: Companion |
| Humanoid therapy | 3 | 250 sq ft | 750 sq ft | Robot 6: Humanoids |
| Imaging bays | 4 | 300 sq ft | 1,200 sq ft | Robot 8: Imaging |
| Ablation suites | 2 | 500 sq ft | 1,000 sq ft | Robot 9: Steerable Needle |
| Rehabilitation bays | 3 | 400 sq ft | 1,200 sq ft | Robot 10: Exoskeletons |

Clinical treatment subtotal: 11,900 sq ft

### 1.2 Support Areas

| Area | Size | Function |
|------|------|----------|
| Patient intake/check-in | 1,500 sq ft | Self-service kiosks, AI triage |
| Adult waiting area | 2,000 sq ft | Seating for 40-50 patients/families |
| Pediatric waiting area | 800 sq ft | Child-friendly, play-equipped |
| Recovery bays | 12 at 120 sq ft = 1,440 sq ft | Post-procedure monitoring |
| Pharmacy/drug storage | 800 sq ft | Investigational drug management |
| Server room | 600 sq ft | AI compute, digital twin, federation |
| Robot maintenance bay | 1,000 sq ft | Calibration, repair, parts storage |
| Emergency response area | 400 sq ft | Crash cart, emergency supplies |
| Staff areas | 1,200 sq ft | Offices, break room, locker room |
| Corridors/utilities | 15,000 sq ft | Hallways, HVAC, electrical, plumbing |

Support subtotal: 24,740 sq ft

### 1.3 Special Construction Requirements

The 3 radiotherapy vaults require radiation shielding with concrete walls
2-3 meters thick, maze-entry corridors, and interlocked door systems. Each
vault accommodates both RT Positioning (Robot 3) and RT Motion-Tracking
(Robot 7) systems with separate control rooms.

The surgical suites require full anesthesia capability including medical gas
supply, ventilation, sterile air handling, and uninterruptible power supply.

The server room requires redundant power, dedicated cooling (30-40 kW thermal
load), 10 Gbps internal networking, and physically secured access.

### 1.4 Parking Requirements

At full patient capacity of 150-180 patients per 24-hour period, with peak
concurrent occupancy of 60-80 patients during hours 08:00-15:00:

- Patient parking: 100 spaces (accounting for staggered arrivals/departures)
- Accessible parking: 10 spaces minimum (ADA compliance)
- Staff parking: 8 spaces (minimal human staffing model)
- Emergency vehicle access: 2 ambulance bays
- Drop-off/pickup zone: 6 spaces for patient transport
- Total parking lot size: approximately 30,000 sq ft (0.7 acres)

Full patient capacity: 150-180 unique patients per 24-hour cycle, with peak
concurrent occupancy of 60-80 patients between 08:00 and 15:00.

## 2. Building Reusability

### 2.1 Partial Repurposing Options

Existing ambulatory surgery centers (ASCs) of 15,000 sq ft or above can be
partially repurposed for non-radiation robot stations. ASCs already provide
surgical infrastructure, sterile environments, and recovery areas suitable
for surgical robots (Robot 1), cobots (Robot 2), and needle-placement systems
(Robot 4).

Existing radiation oncology centers can contribute shielded vault space for
RT Positioning (Robot 3) and RT Motion-Tracking (Robot 7). These facilities
already have linear accelerator infrastructure, radiation safety programs, and
qualified physics support.

### 2.2 New Construction

New construction is preferred for full 10-robot integration due to shielding,
power, and network requirements that are difficult to retrofit simultaneously.
Purpose-built facilities can optimize patient flow, robot logistics, and
infrastructure placement for 24/7 operations.

### 2.3 Retrofit Feasibility

Retrofit of existing 50,000 sq ft or larger medical office buildings is
feasible with vault construction additions. Key retrofit requirements include:

- Addition of 3 shielded vaults (approximately 5,000 sq ft including control
  rooms and maze corridors)
- Electrical upgrade to support robot charging, surgical equipment, and server
  room (minimum 800 kW capacity)
- Network infrastructure upgrade to 10 Gbps with segmented clinical and robot
  control networks
- HVAC modifications for surgical suite air handling and server room cooling
- Structural reinforcement for vault shielding weight

### 2.4 Modular Construction

Modular construction can accelerate site activation from 18-24 months
(traditional) to 6-9 months. Prefabricated modules for biopsy stations,
imaging bays, companion robot areas, and rehabilitation bays can be
manufactured off-site and assembled rapidly. Vault construction remains on
the critical path due to shielding requirements.

## 3. Minimal Human Staffing Model

### 3.1 Shift Structure

Three shifts of 8 hours each, with 2-3 humans per shift providing safety
oversight. The site operates continuously with no shift gaps.

| Shift | Hours | Human Staff |
|-------|-------|-------------|
| Day | 07:00 - 15:00 | 3 (safety officer + 2 oversight) |
| Evening | 15:00 - 23:00 | 3 (safety officer + 2 oversight) |
| Night | 23:00 - 07:00 | 2 (safety officer + 1 oversight) |

### 3.2 Human Roles

- Site safety officer (1 per shift): Monitors all robot operations via
  central dashboard, authorized to activate emergency stops, manages
  escalation protocols
- Emergency physician on-call (1): Available within 10 minutes for adverse
  event response, provides medical oversight for high-risk procedures
- Pharmacist on-call (1): Manages investigational drug inventory, verifies
  drug preparation, handles IND compliance per 21 CFR Part 312

### 3.3 Robot-Managed Functions

All scheduling, intake, consent, procedure execution, monitoring, recovery,
and discharge are managed by robot systems and AI. Specific functions include:

- Patient check-in via self-service kiosks with identity verification
- Informed consent delivery per 21 CFR Part 50 using interactive AI systems
- Procedure scheduling and robot assignment optimization
- Vital sign monitoring during procedures and recovery
- Drug preparation, dosing calculation, and administration tracking
- Digital twin creation, calibration, and treatment planning
- Post-procedure discharge assessment and follow-up scheduling
- Regulatory documentation and audit trail maintenance per ICH E6(R3)

### 3.4 Remote Oversight

Remote human oversight is available via secure video for escalation. Board-
certified oncologists, radiation oncologists, and surgeons can be consulted
remotely within 5 minutes for complex clinical decisions. The remote oversight
system maintains HIPAA compliance per the privacy framework.

### 3.5 Staffing Comparison

| Metric | On-Demand Site | Traditional Site |
|--------|---------------|-----------------|
| Total human FTE | 8-10 | 80-120 |
| Annual labor cost | $800K-$1.2M | $8M-$15M |
| Labor cost reduction | 90% | baseline |
| 24/7 coverage | Yes | No (8-10 hrs/day) |

## 4. Infrastructure Requirements

### 4.1 Power

- Total site power: 800-1,200 kW
- Surgical suites: 80 kW each (240 kW total)
- Radiotherapy vaults: 120 kW each (360 kW total)
- Server room: 40-60 kW
- Robot charging: 20-30 kW
- HVAC and lighting: 100-150 kW
- Uninterruptible power supply: 500 kW capacity (30 min battery backup)
- Emergency generator: 800 kW diesel with automatic transfer switch

### 4.2 Network

- Internal backbone: 10 Gbps fiber
- Robot control network: dedicated VLAN, sub-1ms latency
- Clinical data network: HIPAA-compliant, encrypted
- External connectivity: redundant 1 Gbps internet for federated learning
  and remote oversight
- Cybersecurity: network segmentation, intrusion detection, per ICH E6(R3)
  Section 4.3.3

### 4.3 Environmental

- Surgical suite air handling: HEPA filtration, positive pressure, 20 air
  changes per hour
- Temperature control: 68-72 F (20-22 C) in clinical areas, 64-68 F
  (18-20 C) in server room
- Humidity: 30-60% relative humidity in clinical areas
- Radiation vault ventilation: independent exhaust system per state
  radiation safety regulations

## 5. References

- Patient Robot Instructions: DOI 10.5281/zenodo.18810541
- ICH E6(R3) Adaption: DOI 10.5281/zenodo.18973368
- 21 CFR Part 50 Adaption: DOI 10.5281/zenodo.19040707
- 21 CFR Part 312 Adaption: DOI 10.5281/zenodo.19057628
- USL Framework: DOI 10.5281/zenodo.18778220
