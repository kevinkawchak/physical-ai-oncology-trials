# inputs - sources, the money frame, and the address register (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../README.md)
[![Sources](https://img.shields.io/badge/Source%20directories-10-00417A.svg)](#the-ten-source-directories)
[![Copies](https://img.shields.io/badge/Copied%20files-none-3C7DB2.svg)](#why-nothing-is-copied-here)
[![Money frame](https://img.shields.io/badge/Frame-%24700K%20%C3%97%205%20years-6C757D.svg)](#the-money-frame-no-day-re-derives)
[![Phase 1](https://img.shields.io/badge/Phase%201-3%2B3%2C%20up%20to%2018-6C757D.svg)](#the-robotic-phase-1-in-numbers)
[![Addresses](https://img.shields.io/badge/Verified%20addresses-36-6C757D.svg)](#the-address-register)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

This directory is an index, not an archive. It names every repository source the
five business days read, fixes the numbers no day may re-derive, and holds the
register of every email address a letter in this block is sent to, with the
organization's own page that publishes it.

## Why nothing is copied here

Every source this block reads already lives in this repository under its own
directory with its own README. A copy would drift from the original, and the
original is the one the rest of the repository cites. So the sources are indexed
and linked, and the day directories cite the original path every time.

## The ten source directories

| # | Source | What the five days take from it |
|:--|:--|:--|
| 1 | [`../../auto-fund`](../../auto-fund) | The daily structure, the `.txt` letter format, `fundstyle.sty`, the three-business-day follow-up rule, and the CTEP thread that day 3 continues |
| 2 | [`../../capitalization-plan`](../../capitalization-plan) | The capital frame, the five TikZ diagram vocabularies, the base bibliography, and the six checkable quantities |
| 3 | [`../../pdac-funding-applications`](../../pdac-funding-applications) | The ten application file sets and the SBIR budget split reused on days 1, 4 and 5 |
| 4 | [`../../move-in`](../../move-in) | The eleven-role roster at 3.95 award-funded FTE and the $521,000 personnel line that day 1 maps the applicants against |
| 5 | [`../../potential-partners`](../../potential-partners) | The UC San Diego clinical trials office and ACTRI routes used on days 1 and 3 |
| 6 | [`../../science-golden-age`](../../science-golden-age) | The federal policy position the White House and NSF letters are written against |
| 7 | [`../../supplementary`](../../supplementary) | The digital twin and simulation source sets behind the six quantities |
| 8 | [`../../../trial-protocol`](../../../trial-protocol), [`../../../trial-ind`](../../../trial-ind), [`../../../trial-phase-2`](../../../trial-phase-2) | The Phase 1 protocol, the investigational new drug application, and the Phase 2 protocol |
| 9 | [`../../daraxonrasib-llm-story.md`](../../daraxonrasib-llm-story.md) | The June 2025 to August 2026 chronology quoted in §1 of all five packets |
| 10 | [`../../tripartisan-llm-support.md`](../../tripartisan-llm-support.md) | The three-model division of labor that grounds the day 2 statement about OpenAI and Anthropic |

## The money frame no day re-derives

| Quantity | Value | Source |
|:--|:--|:--|
| Program, five years, direct | $3,500,000 | `../../pdac-funding-applications/final-apply` |
| Program, per year, direct | $700,000 | Same |
| NIH SBIR Phase I, total cost | $306,000 | `../../pdac-funding-applications/applications/app-05-nih-sbir-seed` |
| NIH SBIR Phase II, total cost | $1,300,000 | Same |
| SBIR route, total cost | $1,606,000 | Sum of the two above |
| Delta the SBIR route does not buy | $2,104,000 | `../../capitalization-plan/final-capital`, §3 |
| Private capital behind the firewall | $5,900,000 | `../../capitalization-plan/final-capital`, §4 |
| Private to federal leverage | 3.67 to 1 | Same |
| Personnel inside the annual direct cost | $521,000 across 3.95 FTE | `../../move-in/final-move-in/sections/sec-14-staffing-and-roles.tex` |
| Virtual trial cost, projected | $36,330 | `../../capitalization-plan/final-capital`, Table 17 |

The virtual trial figure is described as **projected** everywhere it appears in
this directory, and never as estimated.

Four NSF figures enter this block for the first time. Each is taken from the
agency's own announcement and is cited where it is used on day 4.

| NSF quantity | Value | Published by |
|:--|:--|:--|
| SBIR/STTR Phase I ceiling, NSF 26-510 | $305,000 over 6 to 18 months | [NSF 26-510](https://www.nsf.gov/funding/opportunities/small-business-innovation-research-small-business-technology/nsf26-510/solicitation) |
| SBIR/STTR Phase II ceiling | $1,250,000, typically 24 months | Same |
| Strategic Breakthrough tier | Up to $30,000,000, Phase II awardees only, 1 to 1 match | Same |
| Commercialization Readiness Pilot (NCTI) | $20,000,000 over two years; six to eight companies at $1,000,000 to $3,750,000 each | [NSF announcement](https://www.nsf.gov/news/nsf-launches-20m-pilot-accelerate-commercialization) |

## The robotic Phase 1 in numbers

Every packet carries these, because a funder is buying this trial and not the
correspondence around it. All are fixed in the deposited protocol and
investigational new drug application and none is re-derived here.

| Parameter | Value | Source |
|:--|:--|:--|
| Design | Open-label, single-arm, 3+3 dose escalation | [10.5281/zenodo.20780121](https://doi.org/10.5281/zenodo.20780121) |
| Treated participants | Up to 18 | Same |
| Daraxonrasib dose levels | 160, 220 and 300 mg | Same |
| Dose-limiting toxicity window | 28 days | Same |
| Procedure | Staged eight-arm robotic pancreaticoduodenectomy | Same |
| Degrees of freedom and sensing | 56 degrees of freedom, 640 sensor channels | Same |
| Force limits | 3 N per arm, 18 N cumulative | Same |
| Stop latencies | 3 ms cross-arm stop, 500 ms system stop | Same |
| Vascular protection | Five-vessel no-fly gate | Same |
| LLM role | On-premises, advisory output only; no write credential to data capture and no route to the robot control network | [10.5281/zenodo.21097442](https://doi.org/10.5281/zenodo.21097442) |

## The six checkable quantities

| Source | Test arm | Comparator | Tier |
|:--|:--|:--|:--|
| RASolute 302, May 2026 | 13.2 months median overall survival | 6.6 months, RAS G12 population | Trial |
| Ten-arm QSP simulation, 250 ODEs | 12.8 months, hazard ratio 0.25 | 5.4 months | In silico |
| Digital twin, 1000 patients | 12.1 months | Not applicable | Twin |
| Digital twin, progression-free survival | Hazard ratio 0.31 | Not applicable | Twin |
| VVUQ credibility, 55 tests | Score 81.9 | V and V 40 gate | Twin |
| Empirical triplicate, 100,000 records | Grade 3 plus, 8.0 percent | 25.0 percent | In silico |

The FDA's approval announcement reports 13.2 against 6.7 months for the full
trial population; the 6.6-month comparator above is the RAS G12 population in the
peer-reviewed report. Both are correct for the population each describes, and
every packet names the population beside the number.

## The address register

Every address below was checked against the page the organization itself
publishes before it was written into a letter. The rule is at least three
addresses per email across `TO` and `CC`. Where an organization publishes no
address suitable for the letter, it is not written to by email.

| Address | Organization and office | Used on | Published at |
|:--|:--|:--|:--|
| `sanfrancisco@sec.gov` | SEC, San Francisco Regional Office | Days 2, 5 | [sec.gov regional office page](https://www.sec.gov/regional-office/san-francisco) |
| `losangeles@sec.gov` | SEC, Los Angeles Regional Office, which covers San Diego | Days 2, 5 | [sec.gov regional offices](https://www.sec.gov/about/regional-offices) |
| `smallbusiness@sec.gov` | SEC, Office of the Advocate for Small Business Capital Formation | Days 2, 5 | [sec.gov small business advocate](https://www.sec.gov/about/divisions-offices/office-advocate-small-business-capital-formation) |
| `OncologyAI@fda.hhs.gov` | FDA, Oncology Center of Excellence, Oncology AI Program | Days 2, 5 | [FDA OCE Oncology AI Program](https://www.fda.gov/about-fda/oncology-center-excellence/oce-oncology-artificial-intelligence-program) |
| `FDAOncology@fda.hhs.gov` | FDA, Oncology Center of Excellence | Days 2, 5 | [FDA Oncology Center of Excellence](https://www.fda.gov/about-fda/fda-organization/oncology-center-excellence) |
| `combination@fda.gov` | FDA, Office of Combination Products | Days 2, 5 | [FDA combination product contacts](https://www.fda.gov/combination-products/jurisdictional-information/combination-product-contacts) |
| `press@anthropic.com` | Anthropic, press | Day 2 | [anthropic.com newsroom](https://www.anthropic.com/news) |
| `Legal@anthropic.com` | Anthropic, legal | Day 2 | [Anthropic applicant and employee privacy notice](https://www-cdn.anthropic.com/74ee25a84fd2f6c97450872b9fe0b4f760d7e478.pdf) |
| `press@openai.com` | OpenAI, press | Day 2 | [openai.com announcements](https://openai.com/index/announcing-devday-2025/) |
| `support@openai.com` | OpenAI, general inquiries | Day 2 | [OpenAI help center](https://help.openai.com/en/articles/6614161-how-can-i-contact-support) |
| `IR@revmed.com` | Revolution Medicines, investor relations | Day 2 | [Revolution Medicines investor relations](https://ir.revmed.com/resources/contact-ir) |
| `media@revmed.com` | Revolution Medicines, media | Day 2 | Same |
| `medinfo@revmed.com` | Revolution Medicines, medical information | Day 2 | [Revolution Medicines clinical trials contact](https://revmedclinicaltrials.com/contact-us) |
| `scheduling@ostp.eop.gov` | White House Office of Science and Technology Policy, scheduling and general inquiries | Day 3 | [OSTP information and resources](https://www.whitehouse.gov/ostp/information-resources/) |
| `genesismission-partnerships@hq.doe.gov` | Department of Energy, Genesis Mission partnerships | Day 3 | [DOE Genesis Mission collaboration](https://www.energy.gov/undersecretaryforscience/genesis-mission/genesis-mission-collaboration) |
| `GenesisMissionNOFO@science.doe.gov` | DOE Office of Science, Genesis Mission funding opportunity | Day 3 | [DOE Office of Science Genesis Mission](https://science.osti.gov/grants/FOAs/Genesis-Mission) |
| `AIworkforce@opm.gov` | Office of Personnel Management, AI workforce programs | Day 3 | [OPM AI workforce memorandum](https://www.opm.gov/chcoc/latest-memos/building-the-ai-workforce-of-the-future.pdf) |
| `cancercto@ucsd.edu` | UC San Diego, Moores Cancer Center Clinical Trials Office | Days 1, 3 | [Moores clinical trial shared resource](https://moorescancercenter.ucsd.edu/research/shared-resources/clinical-trials/index.html) |
| `actri-ctss@health.ucsd.edu` | UC San Diego, ACTRI Clinical Trial Support Services | Days 1, 3 | [ACTRI Center for Clinical Research](https://actri.ucsd.edu/centers-services/portfolio/clinical/index.html) |
| `octa@health.ucsd.edu` | UC San Diego, Office of Clinical Trials Administration | Day 3 | [OCTA contact information](https://vchs.ucsd.edu/administration/research-contract/octa/contact-information.html) |
| `IRB@ucsd.edu` | UC San Diego, Office of IRB Administration | Day 1 | [UC San Diego IRB contact](https://irb.ucsd.edu/about/contact.html) |
| `steven.gore@nih.gov` | NCI CTEP, Investigational Drug Branch | Day 3 | The branch chief's own reply, filed at [`../../auto-fund/02Sep26/emails`](../../auto-fund/02Sep26/emails) |
| `mroczkowskib@mail.nih.gov` | NCI CTEP | Day 3 | Same thread |
| `nicole.pultar@nih.gov` | NCI CTEP | Day 3 | Same thread |
| `sbir@od.nih.gov` | NIH SBIR/STTR Program Office | Days 1, 4, 5 | [NIH SEED contact](https://seed.nih.gov/aboutseed/contact-us) |
| `SEEDinfo@nih.gov` | NIH SEED | Days 1, 5 | Same |
| `ncisbir@mail.nih.gov` | NCI SBIR Development Center | Days 1, 4, 5 | [NCI SBIR contact and staff](https://sbir.cancer.gov/about/contact-staff) |
| `sbir@nsf.gov` | NSF SBIR/STTR, America's Seed Fund | Days 4, 5 | [NSF Seed Fund contact](https://seedfund.nsf.gov/contact/) |
| `policy@nsf.gov` | NSF Policy Office | Day 4 | [NSF proposal and award policy](https://www.nsf.gov/oam/proposal-award-policy) |
| `rgov@nsf.gov` | NSF Research.gov help desk | Day 4 | [Research.gov help desk](https://www.research.gov/research-web/content/contactus) |
| `NAIRR_Pilot@nsf.gov` | NSF, National AI Research Resource Pilot | Day 4 | [NSF NAIRR](https://www.nsf.gov/focus-areas/ai/nairr) |
| `nairr-oc@nsf.gov` | NSF, NAIRR Operations Center | Day 4 | Same |
| `Ivan.Garibay@ucf.edu` | University of Central Florida, principal investigator of the NSF-funded NCTI pilot | Day 4 | [NCTI at UCF](https://cecs.ucf.edu/national-commercialization-and-translation-institute-ncti/) |
| `answerdesk@sba.gov` | U.S. Small Business Administration, Answer Desk | Day 1 | [SBA contact page](https://www.sba.gov/about-sba/organization/contact-sba) |
| `E-Verify@uscis.dhs.gov` | USCIS, E-Verify Contact Center | Day 1 | [E-Verify contact](https://www.e-verify.gov/contact-us) |
| `CalCompetes@gobiz.ca.gov` | California GO-Biz, California Competes | Day 1 | [CalCompetes](https://calcompetes.ca.gov) |

The chief executive writes from `kevink@chemicalqdevice.com`, the company
address every earlier thread in [`../../auto-fund`](../../auto-fund) was sent
from. Where an original request in a thread was sent from
`kevinkawchak@gmail.com`, the reply goes from that address instead so the thread
joins, and each letter's pre-send checklist says so.

## Channels that are not email

| Counterparty | Channel | Why not email | Where the content is |
|:--|:--|:--|:--|
| The two applicants | The LinkedIn conversation each one opened | They wrote on LinkedIn and published no address to the company; replying elsewhere would be a second, unasked-for contact | `28Sep26/linkedin/`, `02Oct26/linkedin/` |
| The White House | The whitehouse.gov contact form | The White House publishes a form, not an address | `30Sep26/forms/` |
| The brokerage | The brokerage's authenticated message center | An order instruction does not travel by open email | `investing/` in each day |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `inputs/README.md` | [`../../auto-fund`](../../auto-fund) | The index-not-archive convention and the money frame table |
| `final-capital/sections/sec-06-clinical-evidence.tex` | [`../../capitalization-plan`](../../capitalization-plan) | The six quantities and the robotic Phase 1 table |
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../move-in`](../../move-in) | The personnel line and the 3.95 FTE |
| `UC-San-Diego/priority-steps.md` | [`../../potential-partners`](../../potential-partners) | The UC San Diego rows of the address register, each re-checked against the university's own page |
| `02Sep26/emails/email-06-nci-ctep-gore-reply.txt` | [`../../auto-fund`](../../auto-fund) | The three CTEP rows of the address register |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
