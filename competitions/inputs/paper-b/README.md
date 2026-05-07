# README: Chunked Paper Files — Code Generation Competition: 16 LLMs & FDA FAERS

## Purpose

This README is provided to assist Claude Code Opus 4.7 (1M context) in processing all 10 chunked markdown files derived from the original LaTeX paper (`main.tex`) and BibTeX bibliography (`references.bib`). These files form a complete and verbatim representation of the source paper by Kevin Kawchak (ChemicalQDevice, December 22, 2025), titled:

**"Code Generation Competition: 16 Proprietary vs. Open-Source LLMs & Iterative Learning Based on FDA Adverse Event Reporting System"**
DOI: 10.5281/zenodo.18029100

The paper describes a 4-round, 16-model, single-elimination LLM code generation tournament using FDA FAERS pharmacovigilance data. This README is intended to guide Claude Code Opus 4.7 in adapting this paper's framework for a **new physical AI oncology trial paper**.

---

## File Manifest

| File | Sections Covered | Approx. Lines | Key Content |
|---|---|---|---|
| `chunk_01_title_abstract.md` | Title, Author, Abstract, Keywords, TOC | ~50 | Paper identity, study summary, keyword taxonomy |
| `chunk_02_introduction.md` | Section 1 (1.1–1.3) | ~80 | Human/AI/LLM code iteration history and capabilities |
| `chunk_03_methods.md` | Section 2 | ~90 | Notebook development, LLM competition setup, 16 model list |
| `chunk_04_faers_data_files.md` | Section 3 | ~50 | FAERS dataset structure (DEMO, DRUG, REAC), reference solution |
| `chunk_05_results_round4_code.md` | Section 4.1–4.2 | ~170 | Round 4 Python code submission (152 lines), PRR algorithm |
| `chunk_06_results_scoring_tables.md` | Section 4.3–4.4 | ~130 | Notebook output, 5 scoring metric tables, competition results |
| `chunk_07_final_results_discussion.md` | Section 4.5, 5.1–5.3 | ~140 | Tournament final results, academic paper table, 5 difficulty tables |
| `chunk_08_limitations_conclusions.md` | Sections 6–7 | ~70 | Study limitations, iterative learning conclusions, round-by-round recap |
| `chunk_09_prompts_data_availability.md` | Sections 8–9 + Back Matter | ~200 | All 14 prompts verbatim, data availability index, acknowledgments |
| `chunk_10_bibtex_references.md` | references.bib | ~200 | All BibTeX entries verbatim |

---

## Detailed File Descriptions

### chunk_01_title_abstract.md
**Content:** Full paper title, author block (Kevin Kawchak, ChemicalQDevice, San Diego, CA, December 22, 2025, ORCID: 0009-0007-5457-8667), complete abstract text, keywords, and reconstructed table of contents with all section/subsection headings.

**Key facts established:**
- 16-LLM single-elimination tournament (8 proprietary vs. 8 open-source)
- Opus 4.5 Extended built the evaluation engine
- 4 rounds of competition; iterative learning observed
- Winners: Gpt-5.2-pro (proprietary champion), Kimi K2 Thinking (+0.405 single-round improvement)
- Competition task: FDA FAERS drug-reaction safety signal detection
- Total AI cost: ~$78.06

**Correlates with:** All other chunks depend on this as the root identity document. The abstract summarizes results detailed in chunks 06–08. Keywords directly connect to the oncology adaptation context.

---

### chunk_02_introduction.md
**Content:** Three subsections covering the evolution of code iteration: (1.1) Human Code Iteration covering DORA/SPACE frameworks and 400-line review limits, (1.2) AI Code Iteration covering AlphaTensor, AlphaDev, Bowtie2, and pre-LLM rigidity, (1.3) LLM Code Iteration covering AlphaEvolve, Opus 4.5 SWE-bench score, speed/cost comparisons of Gpt-5.2, Gemini 3 Pro, DeepSeek V3.2.

**Key facts established:**
- Python 3.11 yielded 1.22× speedup; AlphaDev sorting 70% faster short sequences
- AlphaEvolve hashing 30% faster than human-designed hash
- Gpt-5.2: ~187 tokens/sec (3.8× faster than Claude); DeepSeek V3.2: $0.56/$1.68 per 1M tokens
- Opus 4.5 SWE-bench: 74.4%; Gemini 3 Pro: 1M token context window
- 24 literature references ([01Intro]–[24Intro])

**Correlates with:** chunk_01 (abstract references AlphaEvolve and AlphaDev), chunk_10 (all Intro citations). For an oncology paper adaptation, this section would be replaced/extended with oncology AI background literature.

---

### chunk_03_methods.md
**Content:** Three subsections: (1) Competition Notebook Development — 12-prompt development pipeline using Opus 4.5 Extended, task selection with ChatGPT 5.1, final notebook USER_6th_FAERS_LLM_Competition_Task1_v2.ipynb; (2) LLM Competition Code Generations — scoring system development (F1, PRR correlation, quarter match, structure), final competition notebook _1RD_Tournament_FAERS.ipynb; (3) Rounds 1–4 Code Competition — bracket mechanics, Multi_Round_Prompt design, hardware (T4 GPU → v6e-1 TPU), 16 model configurations with API/playground details.

**Key facts established:**
- Task 1: "First calendar quarters where pair becomes safety signal (PRR ≥ threshold AND count ≥ minimum cases)"
- 8 proprietary models: Grok 4.1, Grok Expert, Gpt-5.1-codex-max, Gpt-5.2-pro, Gemini 3 Pro, Gemini 2.5 Pro, Opus 4.5, Sonnet 4.5
- 8 open-source models (Fireworks.ai): DeepSeek-V3.2, DeepSeek R1 05/28, Kimi K2 Thinking, Kimi K2 Instruct 0905, GLM-4.6, MiniMax-M2, gpt-oss-120b, Qwen3 Coder 480B
- Total AI cost ~$78.06; Google Colab + VS Code + Google Docs used
- Gemini API settings: Temp=1, Thinking=High, Budget=32768, Output=65536, Top P=0.95

**Correlates with:** chunk_04 (FAERS files used), chunk_05/06 (scoring metrics derived from Prompt 09-12), chunk_09 (all 12 prompts are listed verbatim), chunk_10 (model citations). For oncology adaptation: this section provides the full agentic workflow template.

---

### chunk_04_faers_data_files.md
**Content:** Section 3 table content — FAERS July–September 2025 Q3 data file structure for three files: DEMO25Q3.txt (demographic data, first 3 entries), DRUG25Q3.txt (drug information, 3 entries), REAC25Q3.txt (adverse reactions, 3 entries). Also contains the complete 4-row reference solution table (01_reference_solution.csv) showing DUPIXENT drug-reaction signals with PRR values, emergence quarters, and case counts.

**Key facts established:**
- DEMO fields: primaryid, caseid, i_f_code, mfr_sndr, age, age_cod, sex, wt, wt_cod, rept_cod, occp_cod, reporter_country, occr_country
- DRUG fields: primaryid, caseid, drug_seq, role_cod, drugname, prod_ai, route, dose_vbm, dechal, rechal (+ 7 not shown)
- REAC fields: primaryid, caseid, pt, drug_rec_act
- Reference solution: DUPIXENT with Eczema (PRR 5.058), Pruritus (4.215), Condition aggravated (3.793), Dyspnoea (2.529) — all 2025Q1, all True signals
- Signal threshold: PRR ≥ 2.0 AND count ≥ 3

**Correlates with:** chunk_05 (Python code implements exactly this schema), chunk_06 (scoring correctness measures match against this exact reference), chunk_07 (difficulty tables reference this 3-table join structure), chunk_10 [09BodyFAERS] citation. For oncology adaptation: provides the exact data schema template and PRR signal detection formula.

---

### chunk_05_results_round4_code.md
**Content:** Sections 4.1–4.2. Brief Rounds 1–3 summary, then complete verbatim Gpt-5.2-pro Round 4 Python code (152 lines) for `detect_signal_emergence_improved` function. The function implements 9 algorithms: schema normalization, suspect filtering, date parsing, inner joins, per-quarter counting (nunique), PRR calculation (expected = drug_cases × reac_cases / total_cases), signal thresholding, first-emergence selection, and output formatting.

**Key facts established:**
- 3 helper functions: `_empty()`, `_norm_cols()`, `_parse_quarter()`
- PRR formula: `expected = (drug_cases * reac_cases) / total_cases; prr = pair_cases / expected`
- Input schema: demo_df, drug_df, reac_df with min_cases=3, prr_threshold=2.0
- Output schema: ['drug_name', 'reaction', 'emergence_quarter', 'emergence_prr', 'total_cases', 'is_signal']
- Code uses inner joins, vectorized groupby/nunique, no Python row-loops
- Round 4 winner thoughts: type hints, docstrings, ≥3 functions for Code Quality bonus

**Correlates with:** chunk_04 (code directly implements FAERS schema), chunk_06 (scores against this code), chunk_07 (difficulty standards analyzed from this TASK_1_SPEC), chunk_10 ([Opus45], [GPT52] citations). For oncology adaptation: this is the core algorithmic template showing how to structure an LLM-generated signal detection function.

---

### chunk_06_results_scoring_tables.md
**Content:** Sections 4.3–4.4. Round 4 notebook narrative, head-to-head metrics table (time/speed/cost/tokens/LOC/helper functions for Gpt-5.2-pro vs. gpt-oss-120b), verbatim competition output log, and four scoring breakdown tables: Correctness (45% weight), Methodology (30% weight), Code Quality (15% weight), Algorithmic Efficiency (10% weight), plus Final Weighted Score Summary.

**Key facts established:**
- Gpt-5.2-pro: $1.42 cost, ~19.5 tokens/sec, 2m49s, 127 LOC, 3 helpers → score 0.9775
- gpt-oss-120b: $0.01 cost, 122.34 tokens/sec, 25s, 142 LOC, 0 helpers → score 0.2050 (failed)
- Correctness formula: 0.40×F1 + 0.30×r_PRR + 0.20×Q_match + 0.10×S_struct
- Methodology: Base 0.70 (execution success) + up to 3×0.10 bonuses
- Code Quality: Base 5.0 + docstrings(1.5) + type hints(0.5) + vectorized(1.0) + func≥1(0.5) + func≥3(0.5) + comment ratio(0.5)
- Algorithmic Efficiency: Base 0.50 + vectorized(0.15) + merge(0.10) + groupby(0.10) − nested≥2(0.15) − nested≥3(0.20)
- Score differential: 0.7725 (77.25% margin)

**Correlates with:** chunk_05 (code being scored), chunk_04 (reference solution being compared against), chunk_07 (final tournament results), chunk_09 (Prompt 12 redefined Methodology scoring). For oncology adaptation: this entire scoring framework can be directly reused for evaluating oncology trial simulation code.

---

### chunk_07_final_results_discussion.md
**Content:** Section 4.5 (tournament final results narrative, all 16 seeds), Section 5.1 (academic paper references table, 6 papers with relevance descriptions), Section 5.2 (difficulty benchmarking rationale), Section 5.3 (five difficulty standard tables: Software Engineering, AI/ML Benchmarks, FDA Regulatory, Data Complexity, Composite Score = 7.57/10).

**Key facts established:**
- Final: Gpt-5.2-pro 0.9775 defeated gpt-oss-120b 0.2050 (only non-0.9400 score)
- Kimi K2 Thinking: 0.5350 → 0.9400 (+0.405, largest single-round improvement)
- Gemini 2.5 Pro API 0.7756 beat Gemini 3 Pro API 0.7681 (Round 1)
- DeepSeek R1 05/28 (0.8088) beat DeepSeek-V3.2 (0.2050, Round 1)
- Anthropic models ≤0.4850 in first two rounds
- 6 key academic papers: Self-Refine, HumanEval, ICL demonstrations, ChatDev, CodeRL, CodeClash
- Composite difficulty: 7.57/10 (SE:7.17, AI/ML:7.88, Regulatory:7.43, Data:7.29, Domain:8.50)
- FDA regulatory complexity multiplier: 2.97×

**Correlates with:** chunk_01 (abstract results confirmed), chunk_06 (scores referenced), chunk_08 (conclusions reference these scores), chunk_10 (all 6 body citations, limits citations). For oncology adaptation: the difficulty benchmarking framework (Tables A–E) is directly portable to evaluating oncology simulation task complexity.

---

### chunk_08_limitations_conclusions.md
**Content:** Section 6 (Limitations and Future Work) — cost constraints ($249.99/month Google AI Ultra, $300/month SuperGrok Heavy inaccessible), Round 0 notebook baseline limitation, correctness metric version mismatch, SAMPLE_FRACTION=0.1 limit, notebook truncation issue, 7 scores at 0.2125 or below due to execution failures. Section 7 (Conclusions) — full narrative recap of all 4 rounds, all 16 model scores, tournament results, iterative learning findings.

**Key facts established:**
- Google AI Ultra "Deep Think" uses parallel processing at $249.99/month
- SuperGrok Heavy for "heavy reasoning, automation workflows" at $300/month
- CodeClash paper [03LimitsCodeClash] limitation re: end-of-match log files informed study design
- Future work: "Future studies will focus on oncology LLM-generated code at scale"
- Round 1–2 improvements: 4/8 contestants improved; Round 2–3: 3/4 maintained scores
- Semifinal: Gpt-5.2-pro (0.9775) over Gemini 2.5 Pro (0.9325); gpt-oss-120b (0.9400) over Kimi (0.2050)

**Correlates with:** chunk_01 (abstract conclusions confirmed), chunk_07 (scores referenced), chunk_10 ([03LimitsCodeClash], Kawchak paper series citations). **Critical for oncology adaptation:** The explicit statement "Future studies will focus on oncology LLM-generated code at scale" directly motivates the new physical AI oncology trial paper.

---

### chunk_09_prompts_data_availability.md
**Content:** Section 8 (all 14 prompts verbatim in two tables): Prompts 01–06 for notebook generation (ChatGPT + Opus interactions), Prompts 07–12 for LLM code generation (meta-prompt design, output validation, error fixing), Round_1_Prompt and Multi_Round_Prompt (full text with Instructions for Improvement block and RESPONSE FORMAT specification). Section 9 (Data Availability) — complete numbered index of 51 Zenodo supplementary items. Back Matter — full reference list with keys, Acknowledgments, Ethical Disclosures (CC BY 4.0), Citation.

**Key facts established:**
- Round_1_Prompt: Directs LLMs to read notebook, target 1.0 score, use def detect_signal_emergence_improved format
- Multi_Round_Prompt: Adds "INSTRUCTIONS FOR IMPROVEMENT" block with 3 analysis directives
- Response format specified: accept demo_df/drug_df/reac_df/min_cases=3/prr_threshold=2.0; return 6-column DataFrame; set result variable
- Data archived at Zenodo DOI: 10.5281/zenodo.18029100 with 51 supplementary items across Rounds 0–4
- Acknowledged: Anthropic, Google, OpenAI, xAI, Fireworks AI
- License: CC BY 4.0

**Correlates with:** chunk_03 (methods reference these prompts by number), chunk_05 (code format specified by Round_1_Prompt), chunk_10 (all cited works resolved). For oncology adaptation: Multi_Round_Prompt and Round_1_Prompt are fully reusable templates — replace TASK_1_SPEC with oncology trial simulation task specification.

---

### chunk_10_bibtex_references.md
**Content:** Complete verbatim BibTeX entries from `references.bib`, organized in the order they appear in the file. Covers 4 categories: (1) Model/Platform misc entries (Grok41 through FastAPI), (2) Kawchak Author Paper series (16KawchakPaper–22KawchakPaper, Zenodo DOIs), (3) Introduction references (01IntroAntoniou–24IntroStack), (4) Body/Discussion references (01BodyMadaan–09BodyFAERS), (5) Limitations references (01LimitsGemUltra–03LimitsCodeClash).

**Key facts established:**
- 22KawchakPaper is this paper (DOI: 10.5281/zenodo.18029100)
- Kawchak paper series (16–21) covers: pancreatic cancer trials (16–18), oncology LLM efficiency (19), glioblastoma drug synergy (20), glioblastoma patient matching (21) — all relevant prior oncology work
- CodeClash [03LimitsCodeClash]: arXiv:2511.00839, Yang et al., November 2025 — direct predecessor study
- AlphaEvolve [22IntroAlphaEvolvearXiv]: arXiv:2506.13131, June 2025 — algorithmic AI agent benchmark
- Self-Refine [01BodyMadaan]: NeurIPS 2023, pages 46534–46594 — core iterative learning paradigm

**Correlates with:** All 9 preceding chunks via citation keys. Every bracketed citation in chunks 01–09 resolves to an entry here. For oncology adaptation: the Kawchak paper series (entries 16–21) provides the full prior work bibliography for the oncology paper's Related Work section.

---

## Cross-File Correlation Map

```
chunk_01 (identity)
    ├── abstract summarizes → chunks 05, 06, 07, 08
    └── keywords anchor → oncology adaptation scope

chunk_02 (introduction literature)
    └── all 24 Intro citations → chunk_10

chunk_03 (methods)
    ├── references FAERS files → chunk_04
    ├── references scoring system → chunk_06
    ├── references 12 prompts by number → chunk_09
    └── all model citations → chunk_10

chunk_04 (data schema + reference solution)
    ├── schema implemented in code → chunk_05
    ├── reference solution scored against → chunk_06
    └── table structure analyzed in difficulty → chunk_07

chunk_05 (Round 4 Python code)
    ├── implements chunk_04 schema
    ├── scored by metrics in → chunk_06
    └── analyzed for complexity in → chunk_07

chunk_06 (scoring tables)
    ├── scores code from → chunk_05
    ├── compares to reference in → chunk_04
    └── informed by Prompt 12 in → chunk_09

chunk_07 (final results + discussion)
    ├── references scores from → chunk_06
    ├── references academic papers resolved in → chunk_10
    └── difficulty tables analyze task from → chunk_04 + chunk_05

chunk_08 (limitations + conclusions)
    ├── references all round scores from → chunk_07
    ├── states future oncology direction → motivates new paper
    └── [03LimitsCodeClash] and Kawchak series → chunk_10

chunk_09 (prompts + data + back matter)
    ├── prompts describe workflow from → chunk_03
    ├── response format specifies code in → chunk_05
    └── all 51 data items reference rounds in → chunks 05–07

chunk_10 (BibTeX)
    └── resolves all citations in chunks 01–09
```

---

## Guidance for New Physical AI Oncology Trial Paper

This paper's infrastructure (FDA FAERS pharmacovigilance signal detection) is the predecessor to an oncology-focused adaptation. Based on the explicit statement in chunk_08 ("Future studies will focus on oncology LLM-generated code at scale"), the following elements from these 10 chunks are directly reusable:

1. **Scoring framework** (chunk_06): Correctness/Methodology/Code Quality/Algorithmic Efficiency weights (45/30/15/10%) can score oncology simulation code.
2. **Competition prompt templates** (chunk_09): Round_1_Prompt and Multi_Round_Prompt replace TASK_1_SPEC and the detect_signal_emergence_improved signature with an oncology trial task specification and function signature.
3. **Difficulty benchmarking tables** (chunk_07): Tables A–E (SE complexity, AI/ML benchmarks, regulatory standards, data complexity, composite score) can be adapted with oncology-specific regulatory standards (GCP, ICH E6, ICH E9, FDA Oncology guidance).
4. **Prior work bibliography** (chunk_10): Kawchak papers [16–21] on pancreatic cancer (in silico), QSP metastatic cancer, FDA compliance, oncology LLM efficiency, glioblastoma drug synergy, and glioblastoma patient matching form the foundation literature.
5. **16-model tournament structure** (chunk_03): The 8 proprietary vs. 8 open-source bracket with iterative learning rounds is directly portable.
6. **FAERS → oncology data schema** (chunk_04): Replace DEMO/DRUG/REAC with oncology trial data tables (e.g., patient demographics, treatment arms, outcome measures, adverse event tables).
