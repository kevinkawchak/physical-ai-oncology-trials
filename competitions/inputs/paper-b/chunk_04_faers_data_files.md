# Section 3: FDA FAERS Data Files

## FDA Adverse Event Reporting System (FAERS)

### DEMO25Q3: Demographic Data, First 3 Entries Shown

| primaryid | caseid | i_f_code | mfr_sndr | age | age_cod | sex | wt | wt_cod | rept_cod | occp_cod | reporter_country | occr_country |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100289774 | 10028977 | F | GLAXOSMITHKLINE | 53 | YR | M | | | EXP | MD | IN | IN |
| 1005762123 | 10057621 | F | ROCHE | 57 | YR | M | 53 | KG | EXP | CN | CA | CA |
| 1006831310 | 10068313 | F | PFIZER | 69 | YR | F | 72 | KG | PER | CN | US | US |

---

### DRUG25Q3: Drug Information 7 Columns Not Shown

| primaryid | caseid | drug_seq | role_cod | drugname | prod_ai | route | dose_vbm | dechal | rechal |
|---|---|---|---|---|---|---|---|---|---|
| 100289774 | 10028977 | 1 | PS | DICLOFENAC POTASSIUM | DICLOFENAC POTASSIUM | | UNK | Y | U |
| 100289774 | 10028977 | 2 | C | ONDANSETRON | ONDANSETRON | Unknown | UNK | U | |
| 100289774 | 10028977 | 3 | C | TRAMADOL | TRAMADOL | Unknown | UNK | U | |

---

### REAC25Q3: Adverse Reactions, Column 4 Fields n/a

| primaryid | caseid | pt | drug_rec_act |
|---|---|---|---|
| 100289774 | 10028977 | Fixed eruption | |
| 100289774 | 10028977 | Stevens-Johnson syndrome | |
| 100289774 | 10028977 | Toxic epidermal necrolysis | |

*Table caption: FAERS July - September 2025 Quarterly ASCII Data Files, 3 of 7 [09BodyFAERS]*

---

## Rounds 1-4 Reference Solution (01_reference_solution.csv)

| Drug Name | Reaction | Emergence Quarter | Emergence PRR | Total Cases | Is Signal |
|---|---|---|---|---|---|
| DUPIXENT | Eczema | 2025Q1 | 5.058 | 5 | True |
| DUPIXENT | Pruritus | 2025Q1 | 4.215 | 8 | True |
| DUPIXENT | Condition aggravated | 2025Q1 | 3.793 | 5 | True |
| DUPIXENT | Dyspnoea | 2025Q1 | 2.529 | 5 | True |

*Table caption: Reference & Contestant Goal Output. Drug-Reaction Pairs Derived from DEMO, DRUG, REAC. First Calendar Quarters Where Pair Becomes a Signal (PRR ≥ Threshold AND Count ≥ Minimum Cases)*
