# Science: A New Golden Age — Chunked Source Corpus

This directory holds a complete, word-for-word Markdown chunking of
`Science-A-New-Golden-Age.pdf`, the July 2026 report to the President by Michael
Kratsios, Director of the White House Office of Science and Technology Policy,
together with its annexed FY 2028 R&D budget priorities memorandum.

The corpus exists to be loaded by Claude Code (Opus 5) as grounding context when
drafting **physical AI pancreatic cancer funding applications**. The report is the
current administration's controlling statement of what federal science funding is
for, which mechanisms it favors, and which national missions it will pay for —
including a named Robotics mission whose stated purpose is "to initiate the era of
physical AI-driven scientific discovery." Proposals that speak this document's
language, cite its mechanisms, and attach themselves to its missions are arguing
inside the funder's own frame.

---

## 1. Source and provenance

| Property | Value |
| --- | --- |
| Source file | `Science-A-New-Golden-Age.pdf` (in this directory) |
| Pages | 123 PDF pages (printed folios i–xvii, 1–104) |
| Producer metadata | Adobe InDesign 21.4 (Macintosh) / Adobe PDF Library 18.0 |
| Creation date | 2026-07-20; modified 2026-07-21 |
| Author | Michael Kratsios, Director, White House Office of Science and Technology Policy |
| Transmittal letter | July 21, 2026, to President Donald J. Trump |
| Tasking letter | March 26, 2025, from President Donald J. Trump to Director Kratsios |
| Annex | NSTM-5 / M-26-16, July 21, 2026, joint OSTP/OMB memorandum |
| End notes | 185 numbered notes |
| Figures | 4 |
| Tables | 1 |

No external sources were consulted in producing these chunks. Everything in
`chunk-01` through `chunk-08` comes from the PDF in this directory;
`chunk-09` and `chunk-10` are BibTeX renderings of the report's own End Notes plus
the report and annex themselves.

## 2. Fidelity of the chunking

`chunk-01` through `chunk-08` reproduce the report's text **word for word**, with
no abbreviation, summarization, paraphrase, added section headings, or editorial
comment. Verification was done by character-level diff of each chunk against text
re-extracted from the PDF: chunks 04, 05, 06, 07 and 08 differ from the source by
**zero** characters after whitespace and punctuation normalization; chunks 01, 02
and 03 differ only in the deliberate, enumerated repairs listed below.

What was normalized, and nothing else:

1. **Running heads and page folios removed.** The repeated per-page running head
   (for example `Chapter II – Revitalizing America's Science and Technology
   Enterprise`) and the bare page number in the outer margin are page furniture,
   not body text. Each running head is preserved once, as the section heading of
   the chunk it governs.
2. **Line-break hyphenation closed up.** InDesign hyphenated 907 words across line
   endings (`estab-` / `lished`). These were rejoined against a vocabulary built
   from the document itself, so that genuine compounds (`self-assembling`,
   `well-defined`, `pay-for-performance`) keep their hyphen and broken words
   (`established`, `telecommunications`, `lymphoblastic`) do not.
3. **Justification artifacts closed up.** Eleven intra-line gaps produced by
   justified setting (`real- world`, `decision- making`, `whole-of- society`,
   `post- doctoral`) were closed. The one genuine suspensive hyphen — "the time-
   and resource-intensive parts" — was left alone.
4. **URLs reassembled.** End-note URLs broken across lines were rejoined into
   working addresses.
5. **Figures reconstructed.** Chart axis labels are set rotated in the PDF and
   extract mirrored (`sralloD PPP tnerruC fo snoilliB`); x-axis year ticks extract
   as column-major loose digits. Each of the four figures is rendered as a plain
   text block carrying its chart title, axis label, axis scale, tick years and
   legend, immediately followed by its verbatim caption.
6. **Table 1 reconstructed** as a Markdown table, since its cells extract
   interleaved with the wrapped row labels.

Original typographic oddities were **kept**, because they are what the document
says: the report's `R01–EQUIVELENT` in the Figure 4 chart title, `live-saving
therapies` in Chapter III, `half century,and yet` in the annex, the stray dash in
`America's premier –"AI for science" initiative`, and the fact that end note 168
(Charles Yang on the scientific instruments industry) is listed in the End Notes
but has no superscript marker in the Chapter V body text.

## 3. File manifest

| File | Source pages (PDF / printed) | Words | Contents |
| --- | --- | --- | --- |
| `chunk-01-front-matter-and-summary.md` | 1–18 / cover–xvii | ~4,400 | Title page, presidential epigraph, Table of Contents, Letter of Transmittal, President Trump's Letter, Summary of the Report |
| `chunk-02-chapter-one-introduction.md` | 19–30 / 1–12 | ~4,100 | Chapter I, Introduction. Figures 1, 2, 3. End notes 1–25 |
| `chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md` | 31–50 / 13–32 | ~7,300 | Chapter II. Figure 4, Table 1. End notes 26–82 |
| `chunk-04-chapter-three-securing-dominance-in-critical-and-emerging-technologies.md` | 51–64 / 33–46 | ~5,100 | Chapter III. End notes 83–120 |
| `chunk-05-chapter-four-science-and-technology-better-lives-of-all-americans.md` | 65–74 / 47–56 | ~3,700 | Chapter IV. End notes 121–143 |
| `chunk-06-chapter-five-a-new-golden-age.md` | 75–90 / 57–72 | ~5,800 | Chapter V. End notes 144–185 |
| `chunk-07-end-notes.md` | 91–101 / 73–83 | ~3,800 | All 185 end notes, verbatim, in order |
| `chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md` | 102–123 / 85–104 | ~6,100 | Annex divider, NSTM-5 / M-26-16 memorandum in full |
| `chunk-09-bibtex-references-part-one.md` | derived from `chunk-07` | 95 entries | BibTeX for the sources cited in End Notes 1–100 |
| `chunk-10-bibtex-references-part-two.md` | derived from `chunk-07` | 91 entries | BibTeX for the sources cited in End Notes 101–185, plus the report, the annex, three executive-order/roadmap references named only in the annex, and the RASOLUTE-302 daraxonrasib trial |

Chunk boundaries fall on the report's own structural seams. No sentence,
paragraph, bullet, figure caption, end note or memorandum section is split across
two files.

---

## 4. Chunk-by-chunk detail

### `chunk-01` — Front matter and Summary of the Report

**Sections.** Title page; the January 23, 2025 Trump epigraph; the full Table of
Contents with printed page numbers; Letter of Transmittal (July 21, 2026);
President Trump's Letter (March 26, 2025); Summary of the Report, which restates
all five chapters and carries the report's recommendations as bulleted lists.

**What is load-bearing here.** This chunk contains the report's four overarching
goals, stated in the transmittal letter: (i) prioritize the individual scientist
over legacy institutions; (ii) fundamentally change how research dollars are
allocated, distributed and assessed; (iii) set clear scientific goals and build
industrial muscle to translate discovery into technological strength; (iv) prepare
the research enterprise for the AI revolution. It also contains the measure of
success Kratsios sets for the whole program: "The vital questions I could not
pursue then, I am free to pursue now."

The President's letter poses three charges — unrivaled leadership in critical and
emerging technologies; revitalizing the enterprise by reducing administrative
burden; ensuring progress betters the lives of all Americans — which map one-to-one
onto Chapters III, II and IV.

The Summary's bullets are the most quotable compressed statements of every
mechanism the later chapters develop: portable graduate fellowships, NIH Director's
Pioneer Award-style long-horizon grants, golden tickets, fast grants, prize
challenges, advanced market commitments, regranting, X-Labs, ARPAs, curiosity-driven
institutes, indirect-cost reform, metascience units, regulatory sandboxes, CRADA
and Other Transaction Authority streamlining, SBIR/STTR focus, pre-competitive
consortia, industry Ph.D. and postdoc fellowships, hands-on training, registered
apprenticeships, Workforce Pell, regional innovation clusters, the Genesis Mission,
Gold Standard Science, verification infrastructure, autonomous experimentation,
and AI-native scientific institutions.

**Correlations.** `chunk-01` is the index to everything else. Its five `CHAPTER n`
summary blocks are compressed restatements of `chunk-02` through `chunk-06`
respectively, and its Table of Contents gives the printed page number of every
section heading in those chunks and in `chunk-08`. When a proposal needs a
one-sentence articulation of a policy the chapters argue at length, the sentence is
usually in `chunk-01`.

**For a physical AI pancreatic cancer application.** Use this chunk to align the
proposal's framing paragraph with the administration's stated goals, and to lift
mechanism names in the exact wording the funder uses.

---

### `chunk-02` — Chapter I, Introduction

**Sections.** Scientific Progress Remains Essential · As the Source of Our Triumphs ·
The Landscape Is Changing · New Frontiers and New Approaches · Growing Private
Sector R&D · The Linear Model No Longer Holds · Our Researchers Face Mounting
Challenges · A Time of Urgent Scientific Need · Regaining Leadership · The
President's Charge.

**Figures.** Figure 1, basic annual R&D spend by performer and by source,
1957–2023, in billions of 2017 dollars. Figure 2, gross domestic expenditure on
R&D, 2000–2024, United States / EU 27 / China, in billions of current PPP dollars.
Figure 3, proportion of doctorates awarded to temporary visa holders, 1980–2024.

**Diagnostic anchors.** Private industry now deploys around $700 billion annually,
more than triple government and higher education combined. The linear model of
basic → applied → development no longer describes discovery; Donald Stokes's
"Pasteur's quadrant" of use-inspired basic research does. Investigators spend
nearly half of federally funded research time on paperwork. Effective indirect
cost rates at NIH-funded institutions average more than 40 percent while the same
institutions accept 10 to 15 percent from private funders. China has reached
R&D parity on a purchasing-power-adjusted basis. Cancer appears here in the
report's own framing of scientific triumph: "Within living memory, cancer has been
transformed from a death sentence to a treatable condition for millions of
Americans."

**Correlations.** Chapter I states the problem that each later chapter answers, and
says so explicitly: it defers institutional inertia to Chapter II ("As will be
addressed in Chapter II") and the AI transformation to Chapter V ("as we will
discuss in Chapter V"). Chapter II reaches back to it twice ("as described in
Chapter I", "which we discussed in Chapter I"), and Chapter IV twice ("As discussed
in Chapter I", both times to reassert that the linear model no longer holds).
Pasteur's quadrant, introduced here, is the justification `chunk-03`
uses for portfolio construction and `chunk-08` uses for "basic and use-inspired
inquiry."

**For a physical AI pancreatic cancer application.** This is the source of the
problem statement. A physical-AI oncology program is by construction a
Pasteur's-quadrant program — fundamental questions pursued under the pressure of a
clinical use — and Chapter I is where the report authorizes exactly that shape of
work.

---

### `chunk-03` — Chapter II, Revitalizing America's Science and Technology Enterprise

**Sections.** The Scientific Machine Is Getting Bogged Down · Slowed by Growing
Frictions · The Incumbency Tax · Weakened Meritocracy · Misaligned Incentives · The
Reproducibility Crisis · A Lack of Accountability · A Better Path Forward · Adapting
to the Changing Nature of Science · Novel Performers · New Mechanisms · Better
Grantmaking · Prize Challenges · Future Ideas · A Portfolio-Based Approach · Driving
Constant Innovation.

**Figure and table.** Figure 4, average age of R01-equivalent first-time NIH
investigators, 1980–2016, by degree type. Table 1, the institutional-fit matrix
scoring university, corporate lab, federal lab and new institutions against seven
activities, with the legend `+ Well-Suited ≠ Partially-Suited × Less-Suited`.

**Mechanism inventory.** This is the longest chunk and the densest in funding
mechanics: Eroom's law and the roughly eighty-fold fall in new drugs per billion
dollars since 1950, halving about every nine years; at least 270 new federal
requirements imposed on research grants between 1991 and January 2025; negotiated
indirect rates of 50 to 60 percent; grant lead times up to 20 months; NIH principal
investigator average age rising from 39 to 51 between 1980 and 2008; the ARPA
program-manager model and ARPA-H / ARPA-E; focused research organizations (FROs) as
time-bound nonprofit research startups built to dissolve; NSF TIP Directorate's
X-Labs as the first federal program designed to fund independent research
organizations outside traditional academia; mid-scale science defined as tens of
millions of dollars, coordinated teams of ten to a hundred people, and timelines of
half a decade; golden tickets; double-blind review; HHMI-style person-based funding
of roughly $10 million over seven years, which produced high-impact publications at
nearly double the rate of comparable federally funded peers; NIH Director's Pioneer
Award; NSF CAREER; NSF GRFP with three years of portable support and more than
forty alumni Nobel laureates; fast grants decided in 48 hours rather than 6 to 9
months from applications taking 30 minutes; prize challenges and advanced market
commitments; scouts and regranting; quadratic funding; deliberate portfolio
construction across the exploration–exploitation trade-off; and an empowered
metascience unit in each agency, reporting to the director, with authority to run
controlled experiments and a regrantable budget, modeled in part on the United
Kingdom's Metascience Unit established in 2024.

**Correlations.** Chapter II is the mechanism library that Chapters III, IV and V
all draw on. Chapter III cites it three times for Other Transaction Authority,
facility-access review, and grand challenges. Chapter V cites it three times — for
the novel organizations and funding mechanisms AI-for-science requires, for the
documented failure of publication growth to produce acceleration, and for the
reproducibility crisis that Gold Standard Science must fix. Every mechanism named
in the annex's "R&D Priority Practices" (`chunk-08` §1–3) is first argued here:
long-duration grants, fast grants, prizes, golden tickets, advance market
commitments, regranting, quadratic funding, metascience capabilities, gap maps,
program-officer empowerment, administrative burden reduction. Table 1's "New
Institutions — By design" column is the analytic basis for the annex's "Support a
Diverse Portfolio of Institutions."

**For a physical AI pancreatic cancer application.** This chunk supplies the
instrument. A physical-AI pancreatic-cancer program that needs an engineering team
of ten to a hundred, five or more years, and produces platform and public-goods
outputs rather than an immediately commercializable asset sits exactly in the
"agile, mid-scale science" gap the report says no existing institution fills.
Table 1 is the argument, in the funder's own notation, for why a university lab,
a pharmaceutical company and a national lab each cannot do it alone.

---

### `chunk-04` — Chapter III, Securing U.S. Dominance in Critical and Emerging Technologies

**Sections.** We Must Choose Our Technological Future · Failure of the Passive Model ·
Choosing to Lead · Fighting in Our Own Arena · Unleashing Innovation · The Freedom to
Build · Places to Test · Opening America's Laboratories · Tapping Our Private Sector ·
Closer Partnerships · Marshaling Grand Efforts · Pre-Competitive Consortia · Our
National Character.

**Anchors.** American venture capital deployed over $200 billion in 2024, 57 percent
of global venture investment. The federal share of basic research funding has fallen
from over 70 percent in the 1960s to 40 percent, with industry above 35 percent. The
nuclear case study: five to six years of NRC pre-application discussion, a
12,000-page application supported by more than 2,000,000 pages of technical
documentation, over $600 million of DOE funding for a single reactor design, and the
May 2025 executive orders that impose an 18-month deadline on NRC rulemaking and cap
licensing timelines. The biotechnology case study is the one that matters most here:
a retinal prosthesis developed in Alameda, California had to run its clinical trials
in Europe; per-patient clinical trial costs run far higher than in other economies;
HHS has launched the largest deregulatory effort in its history; FDA now accepts
real-world evidence in regulatory reviews, has dropped the default two-trial
requirement in favor of a single well-powered study with confirmatory evidence, and
has fast-tracked review timelines for drugs supporting U.S. national interests; NIH
has published priority scientific areas without new Notices of Funding and removed
application requirements that added burden without commensurate benefit. DOE operates
28 user facilities. The National Quantum and Nanotechnology Infrastructure program
offers shared cleanroom access with more than 2,000 tools, often at a few hundred
dollars per hour; shared GMP facilities break the catch-22 that startups cannot raise
capital for clinical-grade manufacturing before they have clinical data. The
Foundation for the NIH and the Accelerating Medicines Partnership are held up as the
model: AMP on Alzheimer's Disease, one of twelve disease-focused AMPs, experimentally
validated 20 candidate drug targets. The Human Genome Project's $3.8 billion federal
investment generated an estimated $796 billion in economic activity. SEMATECH and
EUV LLC are the pre-competitive consortium precedents.

**Correlations.** Chapter III is the translation half of the argument whose discovery
half is Chapter II; it says so, deferring three times to Chapter II. It defers
forward once to Chapter V, on AI narrowing the comparative advantage conferred by
scientific excellence alone — the premise that makes Chapter V's institutional
reforms urgent. Its facility-access, CRADA, OTA and cost-share proposals reappear
almost verbatim as the annex's "User Facilities for the S&T Ecosystem" and "Expand
the Use of Non-Federal Cost Share" (`chunk-08`). Its regional-experimentation
argument (Arizona, Utah, Wyoming) is continued by Chapter IV's cluster argument.

**For a physical AI pancreatic cancer application.** This is the chunk that speaks
directly to clinical-trial economics and regulatory pathway. The FDA
real-world-evidence and single-pivotal-trial reforms, the CNPV pilot, the
observation that American discoveries are increasingly tested abroad, and the
FNIH/AMP public-private vehicle are all citable, in-document justifications for a
domestically run, instrumented, physical-AI-enabled pancreatic cancer trial.

---

### `chunk-05` — Chapter IV, Ensuring That Science and Technology Better the Lives of All Americans

**Sections.** The Marriage of Science and Craft · Our Manufacturing Base · Vast
Potential Remains Untapped · Our Educational System Tilted the Scales · We Must
Restructure Science as a Broader Endeavor · Expanding Participation · Integrated
Models of Training · New Paths for Translation · Making Progress Available to All.

**Anchors.** Technology exists in three forms — tools, explicit instructions, and
process knowledge — and process knowledge, the tacit kind, is the true keystone of
capability. Michael Polanyi's "we can know more than we can tell"; Harry Collins's
finding that no laboratory replicated a new laser from published sources alone.
Manufacturing employment peaked near 20 million in 1979 and stands at roughly 13
million, a decline of around 35 percent, with manufacturing's share of total
employment falling from nearly 22 percent to around 8 percent. Rainer Weiss, who
learned to machine, solder and weld as a laboratory technician and then built the
prototype interferometer that became LIGO, is the chapter's exemplar. Community
colleges enroll around 40 percent of all undergraduates. The Administration has
directed agencies toward more than one million active apprenticeships annually; DOL
has shifted to pay-for-performance; Workforce Pell passed in 2025. Regional clusters:
NSF Regional Innovation Engines, Commerce Tech Hubs, Manufacturing USA Institutes,
NIST's Manufacturing Extension Partnership, and DOW's eight Microelectronic Commons
regional hubs; Ohio's roughly $2 billion incentive package and 23-college curriculum
network; Taylor, Texas's nearly $5 billion semiconductor investment.

**Correlations.** Chapter IV twice invokes Chapter I's rejection of the linear model
as its premise. Chapter V then names Chapter IV explicitly as the constraint that
binds once intelligence is abundant: "as intelligence becomes more abundant, the
constraints on scientific progress shift from generating ideas to realizing them in
the physical world." That single sentence is the hinge between this chunk and the
physical-AI thesis. The workforce and hands-on-training proposals here become the
annex's "Expand Hands-On Technical Learning" and "Build Flexible Cross-Sector Talent
Pathways"; the cluster proposals become "Leverage R&D to Strengthen Regional
Manufacturing and Industry" and "Integrate Federal R&D with Non-R&D Investments to
Support Regional Ecosystems."

**For a physical AI pancreatic cancer application.** Physical AI is where the world
of atoms meets the world of bits, and this chunk is the report's argument that the
world of atoms is the binding constraint. Use it for the workforce, technician,
apprenticeship, instrumentation-fabrication and regional-siting components of a
proposal — the parts reviewers often treat as boilerplate and that this report treats
as the actual bottleneck.

---

### `chunk-06` — Chapter V, A New Golden Age

**Sections.** The Age of Intelligence · Adapting Our Institutions · Building the
Infrastructure · The Genesis Mission · Gold Standard Science · Ideas on the Horizon ·
Rethinking Scientific Publication · New Forms of Collaboration and Credit · As We
May Build.

**Anchors.** The chapter opens on Vannevar Bush's July 1945 companion essay "As We
May Think" and the memex, quoted in two block quotations. In 2025 American companies
committed more than $400 billion to AI infrastructure, more than the
inflation-adjusted cost of the Apollo Program and the Manhattan Project combined.
Three constraints bound the returns to intelligence: the speed of feedback, the
speed of atoms, and the constraint imposed by our own institutions. The
forest-fire analogy warns that per-researcher AI productivity gains do not aggregate
into collective progress.

The Genesis Mission, launched by Executive Order 14363 in November 2025, directs DOE
to build the American Science and Security Platform, connecting supercomputers, AI
systems, scientific instruments and datasets into a single discovery engine intended
to double the productivity and impact of American science and engineering within a
decade. DOE's 17 national laboratories employ roughly 40,000 scientists, engineers
and technical staff and receive approximately $20 billion in annual funding. The
Mission's four named challenges are problem selection (at least 20 national S&T
challenges, reviewed annually), institutional capacity (agreements with 24
organizations announced December 2025; the Transformational AI Models Consortium),
data infrastructure (the American Science Cloud; funding for dataset curation), and
integration of AI with experimental infrastructure (14 funded projects in robotics,
automated laboratories and autonomous control of large-scale experiments; the A-Lab
at Lawrence Berkeley; the Polybot at Argonne; NSF's initial $380 million for
programmable cloud labs, explicitly analogized to NSFNET's role in seeding the
internet). Materials discovery that takes around 20 years from laboratory to
deployment could be collapsed by an order of magnitude with closed-loop autonomous
experimentation, which the report calls a strategic imperative, noting that Canada
and China are racing forward and that consolidation and offshoring in the scientific
instruments industry have left expensive products, poor software and proprietary data
formats that lock researchers into vendor ecosystems.

Gold Standard Science, Executive Order 14303 of May 2025, establishes nine
principles for all federally funded research: reproducibility; transparency;
communication of error and uncertainty; collaboration across disciplines; skepticism
of assumptions; falsifiability of hypotheses; unbiased peer review; acceptance of
negative results; and freedom from conflicts of interest. Irreproducible findings in
preclinical biomedical research alone misdirect an estimated $28 billion annually.
The report calls for a verifier equal in rigor and scale to the Genesis Mission's
generator: open APIs and interoperability standards, machine-auditable replication
packages, and prizes for replicating or disproving influential papers.

The chapter closes on the sentence a pancreatic-cancer proposal should know by
heart: "We may begin to engineer cells as precisely as we now engineer circuits,
programming immune systems to hunt malignancies with complete specificity... If we
get all this right, within a generation, the diseases that today kill millions—like
cardiovascular failures, neurodegenerations, and cancers—may yield one by one to
instruments we are now starting to build."

**Correlations.** Chapter V is the synthesis chunk and the one that explicitly binds
the others: it names Chapter II for novel organizations and funding mechanisms and
again for the documented slowdown, Chapter III for the public-private partnerships
that keep frontier AI capability in conversation with public science, and Chapter IV
for the reconnection of science and craft. Its reproducibility argument is the same
argument made in `chunk-03`'s THE REPRODUCIBILITY CRISIS, now recast as an AI-safety
problem for the knowledge base. Everything in the annex's "Build the Foundation for a
New Era of Scientific Discovery" (`chunk-08`) is an operational restatement of this
chapter.

**For a physical AI pancreatic cancer application.** This is the closest chunk to the
proposal's subject. Autonomous and closed-loop experimentation, robotic and cloud
laboratories, instrument redesign for open interfaces and standardized data formats,
AI-ready dataset curation, and machine-checkable verification are all here, as is the
Genesis Mission structure a proposal can attach to.

---

### `chunk-07` — End Notes

All 185 end notes, verbatim and in order, each with its full citation and URL as
printed. Note 168 is present in this list even though its superscript marker does not
appear in the Chapter V body text.

**Correlations.** This chunk is the join table for the whole corpus. Superscript
numerals appear inline in `chunk-02` through `chunk-06`; every one resolves here. The
ranges are: `chunk-02` → notes 1–25; `chunk-03` → 26–82; `chunk-04` → 83–120;
`chunk-05` → 121–143; `chunk-06` → 144–185. `chunk-01` and `chunk-08` carry no
superscript markers, though the annex names Executive Orders 14363, 14413 and 14369
and the DOE Fusion Science & Technology Roadmap in running text. Every note here has a
corresponding BibTeX entry in `chunk-09` or `chunk-10`.

---

### `chunk-08` — Annex: FY 2028 Administration Research and Development Budget Priorities

**Structure.** The memorandum header (NSTM-5 / M-26-16, July 21, 2026, from Kratsios
and OMB Director Russell T. Vought to the heads of executive departments and
agencies), an introductory statement of purpose, then three parts:

*FY 2028 R&D Priority Areas* — Invest in Foundational Research to Drive Scientific
Breakthroughs for Emerging Technologies (with prioritized subfields in physical
sciences; chemistry and materials science; mathematics and computer science;
engineering sciences; and biological sciences); Advance National Science and
Technology Missions; Build the Foundation for a New Era of Scientific Discovery;
Expand World-Class R&D Infrastructure for Broad Use; Leverage R&D to Strengthen
Regional Manufacturing and Industry.

*R&D Priority Practices* — 1. Develop New Mechanisms to Support Frontier Science;
2. Identify and Develop Top Technical Talent; 3. Build a Self-Improving Scientific
Enterprise; 4. Integrate Federal R&D into Broader S&T Enterprise.

*Implementation* — standard FY 2028 budget submission to OMB, plus a requirement
that within 90 days each agency with $3 billion or more in FY 2026 R&D budget
authority submit an action plan to the Assistant to the President for Science and
Technology and the OMB Director.

**The six national missions**, named in full: AI (the Genesis Mission, pursuant to
Executive Order 14363); Quantum (QC-ADDS, pursuant to Executive Order 14413); Fusion
(commercial fusion power by the mid-2030s, following DOE's Fusion Science &
Technology Roadmap); Space (return to the lunar surface by 2028, a lunar base, the
National Initiative for American Space Nuclear Power, and a responsive and adaptive
national security space architecture, pursuant to Executive Order 14369);
**Robotics** ("General-purpose autonomous systems capable of dexterous manipulation,
mobility, and reliable operation in real-world environments, to initiate the era of
physical AI-driven scientific discovery and American reindustrialization"); and
Semiconductors (EUV-and-beyond photolithography, 3D advanced packaging, novel
materials).

**Operational specifics a proposal can be written against.** Increase the share of
foundational research relative to later-stage development, and note R&D character
classification as a percentage of the portfolio. Prioritize foundational research in
the biological sciences — molecular, cellular and structural biology; biochemistry
and chemical biology; genetics, genomics, and synthetic and engineering biology;
neuroscience and the neural basis of cognition and behavior; microbiology and
quantitative biology — over the broader life-sciences category. Fund AI as a new
instrument of discovery rather than for incremental gains, and prioritize work
industry is unlikely to pursue on its own, including pre-competitive research outputs
and enabling platform technologies. Build domain-specific scientific foundation
models. Make internal datasets available and create incentives for researchers to
curate experimental records, negative results and operational data from laboratory
procedures. Invest in robotics, automated laboratories, closed-loop AI scientific
workflows and autonomous control of large-scale experiments, and use federal
purchasing power to drive instrument redesign toward open interfaces, standardized
data formats and cross-vendor interoperability. Propose mid-scale instrumentation
fully funded within a fiscal year. Weigh innovative potential and commercial urgency
alongside scientific merit in facility access. Expand long-duration grants of ideally
five years or more, fully funded in year one. Establish or expand fast grants with
applications of a few pages and review timelines under one month. Target at least 3:1
leverage of private to federal investment on prizes. Explore golden tickets, advance
market commitments, regranting, quadratic funding and eigenfunding, ensuring funded
proposals meet a level of scientific rigor appropriate for Gold Standard Science.
Establish metascience capabilities and maintain gap maps. Prioritize non-federal cost
share from industry, philanthropy, state and local governments, or international
partners.

**Correlations.** The annex is the executable form of the report. Its introductory
paragraphs restate `chunk-01`'s transmittal-letter goals; its Priority Areas restate
`chunk-06` (AI and Genesis), `chunk-04` (infrastructure, partnerships, cost share)
and `chunk-05` (regional manufacturing); its Priority Practices restate `chunk-03`
(mechanisms, talent, metascience) almost item for item. Where the chapters argue,
the annex instructs — and it is the annex that agencies must answer within 90 days,
which makes its verbs ("agencies should propose", "agencies should prioritize") the
verbs a proposal should mirror.

---

### `chunk-09` and `chunk-10` — BibTeX

186 unique entries, no duplicate keys, covering all 185 end notes with none omitted.
`chunk-09` holds the 95 sources cited in End Notes 1–100; `chunk-10` holds the 85
sources cited in End Notes 101–185, plus:

- `kratsios2026sciencegoldenage` — the report itself
- `kratsiosvought2026fy2028priorities` — the annexed memorandum
- `eo14413quantum`, `eo14369space`, `doe_fusionroadmap` — references named in the
  annex without end notes
- `rasolute302` — the daraxonrasib metastatic pancreatic cancer trial, included as
  the clinical anchor for applications built from this corpus

Conventions: one entry per unique source, not one per end note, so a source cited by
several notes appears once with a `note` field listing every citing note (for
example `bush1945endlessfrontier` carries `End Notes 1, 2, 8, 9, 10, 11, 15, 26`).
A source is placed in the chunk of its first citation, so the two files concatenate
into a single valid, deduplicated `.bib` with no key collisions. Multi-source end
notes are split into one entry per source, each carrying the same note number.
Keys are `authoryeartopic`; corporate authors are brace-protected; titles are
double-braced to preserve capitalization; `month`/`day`/`date`/`doi`/`url` follow the
format of the `rasolute302` exemplar.

---

## 5. How the chunks correlate to one another

**The argument spine.** The report is a single argument in five moves, and the
chunks preserve it: `chunk-02` diagnoses (the linear model is dead, productivity has
slowed, competitors have caught up); `chunk-03` prescribes new funding mechanisms and
institutional forms; `chunk-04` prescribes translation, regulatory freedom and
public-private structure; `chunk-05` prescribes the human and industrial substrate;
`chunk-06` argues that AI raises the stakes on all four and adds the Genesis Mission
and Gold Standard Science; `chunk-08` converts all five into budget instructions.
`chunk-01` is the compressed version of the whole; `chunk-07` and its BibTeX
renderings in `chunk-09`/`chunk-10` are the evidence base.

**Explicit in-document cross-references.**

| From | To | What is carried across |
| --- | --- | --- |
| `chunk-02` | `chunk-03` | institutional inertia in the federal funding apparatus |
| `chunk-02` | `chunk-06` | technology-science interdependence, even in mathematics |
| `chunk-03` | `chunk-02` | competitor catch-up; Pasteur's quadrant |
| `chunk-04` | `chunk-03` | Other Transaction Authority; academic-merit facility review; grand challenges |
| `chunk-04` | `chunk-06` | AI narrowing the advantage conferred by scientific excellence alone |
| `chunk-05` | `chunk-02` | the linear model no longer holds |
| `chunk-06` | `chunk-03` | novel organizations and mechanisms; the publication-growth/discovery gap; reproducibility |
| `chunk-06` | `chunk-04` | public-private partnership as the route to frontier AI capability |
| `chunk-06` | `chunk-05` | craft and the world of atoms as the binding constraint on abundant intelligence |

**Implicit couplings worth exploiting.**

- *Reproducibility* is raised in `chunk-03` as an incentive failure and re-raised in
  `chunk-06` as an AI-training-data failure; `chunk-08` makes Gold Standard Science
  a condition on new funding mechanisms. A proposal's rigor plan should cite all
  three.
- *Mid-scale science* is defined in `chunk-03` (tens of millions, teams of ten to a
  hundred, half a decade), summarized in `chunk-01`, and named as a gap in
  `chunk-08` §1. This is the single most transferable framing for a physical-AI
  oncology program.
- *Instrumentation* runs from `chunk-04` (shared facilities, GMP, user facilities)
  through `chunk-05` (who builds and maintains instruments) to `chunk-06` (vendor
  lock-in, open interfaces, standardized formats) and lands in `chunk-08` as a
  purchasing-power directive.
- *Clinical translation* runs from `chunk-04` (trial costs, FDA reform, AMP/FNIH)
  through `chunk-06` (cloud laboratories, agent-based verification) and is
  operationalized by `chunk-08`'s biological-sciences priorities and cost-share
  expectations.
- *Talent* runs from `chunk-03` (fellowships, early-career independence, program
  officers) through `chunk-05` (technicians, apprenticeships, practitioners in
  residence) to `chunk-08` §2.

---

## 6. Using this corpus for physical AI pancreatic cancer funding applications

A suggested mapping from proposal section to source chunk:

| Proposal section | Primary chunks | What to draw |
| --- | --- | --- |
| Significance / problem statement | `chunk-02`, `chunk-06` | productivity slowdown, Eroom's law framing, the "cancers—may yield one by one" passage |
| Innovation / why now | `chunk-06`, `chunk-08` | Genesis Mission, closed-loop autonomous experimentation, the Robotics mission's "physical AI-driven scientific discovery" |
| Approach / institutional design | `chunk-03`, `chunk-08` §1 | mid-scale gap, Table 1, X-Labs and FRO models, long-duration and fast-grant mechanisms |
| Regulatory and clinical pathway | `chunk-04` | FDA real-world evidence, single-pivotal-trial default, CNPV pilot, trial-cost offshoring |
| Partnerships and leverage | `chunk-04`, `chunk-08` §4 | FNIH/AMP model, OTA, SBIR/STTR, 3:1 prize leverage, non-federal cost share |
| Facilities and instrumentation | `chunk-04`, `chunk-06`, `chunk-08` | DOE user facilities, shared GMP, open interfaces and standardized data formats, mid-scale instrumentation fully funded in year one |
| Data management and sharing | `chunk-06`, `chunk-08` | American Science Cloud, AI-ready datasets, incentives to curate negative results and laboratory operational data |
| Rigor and reproducibility | `chunk-03`, `chunk-06`, `chunk-08` | Gold Standard Science's nine principles, verification infrastructure, machine-auditable replication packages |
| Workforce and training | `chunk-05`, `chunk-08` §2 | hands-on technical learning, practitioners in residence, portable fellowships, cross-sector pathways |
| Regional impact | `chunk-05`, `chunk-08` | innovation clusters, Manufacturing USA, Regional Innovation Engines, co-location with manufacturing |
| References | `chunk-07`, `chunk-09`, `chunk-10` | verbatim end notes and ready-to-use BibTeX |

Two cautions when drafting. First, this report is a policy statement, not a
peer-reviewed source: its quantitative claims should be cited to the underlying end
note (via `chunk-07` and the matching BibTeX key), not to the report, wherever the
end note names a primary source. Second, the report's language is the
administration's own; quote it where alignment is the point, but a proposal's
scientific claims still need independent evidence.
