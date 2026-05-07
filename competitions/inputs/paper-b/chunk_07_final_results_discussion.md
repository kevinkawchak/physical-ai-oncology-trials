# Section 4.5 & 5: Round 4 Final Results and Discussion

## 4.5 Round 4 Final Results

The Kimi K2 Thinking purple bubble indicates the largest round improvement, but Gemini 2.5 Pro API, DeepSeek R1 05/28, and GPT-5.2-pro all started with higher initial scores. GPT-5.2-pro won the most rounds, represented by the largest bubble size. The score progression figure illustrates score progression throughout the full 4 rounds, with GPT-5.2-pro improving after the first iteration and maintaining it's lead throughout the tournament. OpenAI gpt-oss-120b achieved the highest first round score at 0.9400, which was maintained for two additional rounds, but its code failed to execute in the final round - represented by the late orange line decline. Gemini 2.5 Pro API also improved after the first round to 0.9325, maintained it's score, but was eliminated by Gpt-5.2-pro in the semifinal round. All 16 of the tournament seeds are shown in the statistical summary, indicating 2 models with scores of 0.90 or higher between proprietary and open-source LLMs for the 8 first round winners. For those winners, the number of ≥0.90 second round scores increased to 6 based on the same FDA FAERS task. Performance was primarily constant in the semifinals and finals, although a single LLM code failed to execute in both rounds.

The final championship bracket shows contestants, competitors, and scores for each of the tournament's 15 matches. Top proprietary LLM Gpt-5.2-pro defeated the best open-source LLM OpenAI gpt-oss-120b in the final round by a score of 0.9775 to 0.2050. Gpt-5.2-pro saw an increase in its first round score from 0.8238 to 0.9775, maintaining that score throughout the tournament. gpt-oss-120b accomplished the top Round 1 score of all models at 0.9400, maintaining its score until the final round, with its code failing to execute; reflected by a score of 0.2050, and finishing runner-up. Kimi K2 Thinking saw the largest single round score improvement from 0.5350 to 0.9400, and Gemini 2.5 Pro API attained a small performance increase over the newer Gemini 3 Pro API in the first round of 0.7756 versus 0.7681, while Anthropic models struggled in the first two rounds, with scores not surpassing 0.4850 - in supplementary Round_1 model thoughts describing a lack of focus on the perfect score goal.

---

## 5.1 Discussion: Academic Paper References

### Academic Paper References for Iterative LLM Code Improvement

| # | Paper Title | Authors | Year | Relevance to Competition |
|---|---|---|---|---|
| 1 | Self-Refine: Iterative Refinement with Self-Feedback [01BodyMadaan] | Madaan et al. | 2023 | Core paradigm: Round N notebook → Round N+1 code |
| 2 | Evaluating Large Language Models Trained on Code [02BodyChen] | Chen et al. | 2021 | Correctness metric foundation (pass@k) |
| 3 | Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? [03BodyMin] | Min et al. | 2022 | Prior Code_A/Code_B as few-shot examples |
| 4 | ChatDev: Communicative Agents for Software Development [04BodyQian] | Qian et al. | 2023 | Multi-agent head-to-head code competition |
| 5 | CodeRL: Mastering Code Generation through Deep Reinforcement Learning [05BodyLe] | Le et al. | 2022 | Execution-based reward (Correctness=45% weight) |
| 6 | CodeClash: Benchmarking Goal-Oriented Software Engineering [03LimitsCodeClash] | Yang et al. | 2025 | Multi-round tournament to build the best Python function |

*Table caption: Academic Papers Relevant to Multi-Round Iterative Learning Process*

The current competition builds off of techniques established in prior LLM application papers. For instance, the current study implements Self-Refine (Paper 1) across 4 rounds [01BodyMadaan], where each model receives its prior round's notebook as in-context learning demonstrations (Paper 3) [03BodyMin]. Correctness scoring against a Reference Solution parallels CodeRL's execution feedback (Paper 5) [05BodyLe]. The head-to-head format is akin to ChatDev's multi-agent collaboration (Paper 4) [04BodyQian]. Additionally, the CodeClash goal-based benchmark from the SWE-bench Team regarding LLMs competing in multi-round game tournaments to build the best codebase for achieving a competitive objective was a framework for the study (Paper 6) [03LimitsCodeClash]; where an increased competition pool size from 8 to 16 and a medical objective were additive. The FDA FAERS data files [09BodyFAERS] used in the competition task are known in literature to be effective for providing safety assessments throughout the life cycle of a drug [06BodyPotter], providing new and unexpected signals of adverse drug reactions [07BodyYang], and discovering top drugs based on events leading to death [08BodyYu].

---

## 5.2 Difficulty Benchmarking

Answer key creation was easier for the notebook to establish than executing either Code A or Code B due to the solution being fully deterministic with no ambiguity, while competition code must consider many more rows and drug-reaction combinations. Difficulty standards were calculated by Opus 4.5 Extended.

**Reference Solution Code**

- No programmatic creativity requirements
- Already knows how the tables must be merged
- Already knows correct logic, date, and quarter parsing rules

**Competition Code Must Run**

- The full sampled FAERS subset, which can be thousands of rows
- All combinations of drug × reaction considered, yielding very large numbers
- All time-cumulative quarter calculations, which multiplies the number of operations

---

## 5.3 Difficulty Benchmarking Standards

### Software Engineering Complexity Standards

| Standard | Metric | Task 1 Value | Industry Benchmark | Difficulty Rating |
|---|---|---|---|---|
| McCabe | Cyclomatic Complexity | 25–35 | >20 = High Risk | 8.0/10 |
| Halstead | Difficulty (D) | ~32 | >30 = Challenging | 7.5/10 |
| Halstead | Effort (E) | ~90,000 | >50K = Substantial | 8.0/10 |
| Halstead | Time to Program | ~83 min | >60 min = Complex | 7.0/10 |
| COCOMO II | Effort | 0.6 person-months | 0.5–1.0 = Medium | 6.5/10 |
| Lines of Code | Implementation Size | 150–230 LOC | 100–300 = Medium | 6.0/10 |
| | | | **Subtotal Average** | **7.17/10** |

*Table caption: Software Engineering Complexity Assessment Metrics Derived from Static Analysis of detect_signal_emergence_improved Function Specification (TASK_1_SPEC)*

---

### AI/ML Benchmark Alignment Standards

| Benchmark | Criterion | Task 1 Mapping | Score (0–10) |
|---|---|---|---|
| HumanEval | Function Signature | Exact DataFrame params required | 7.0 |
| HumanEval | Docstring Comprehension | TASK_1_SPEC interpretation | 8.0 |
| HumanEval | Edge Case Handling | Empty inputs, invalid dates, missing cols | 9.0 |
| HumanEval | Algorithmic Correctness | PRR formula + temporal emergence | 9.0 |
| HumanEval | Output Validation | 6-column schema compliance | 6.0 |
| MBPP | Problem Category | Level 4–5 (Advanced/Expert) | 8.5 |
| BigCodeBench | Tier Classification | Tier 3–4 (Domain + Multi-source) | 8.0 |
| APPS | Difficulty Band | Interview-level (comparable) | 7.5 |
| | | **Subtotal Average** | **7.88/10** |

*Table caption: Established AI/ML Code Generation Benchmarks. Edge Case Handling Scored Highest (9.0)*

---

### FDA/Pharmacovigilance Regulatory Standards

| Standard | Component | Implementation Requirement | Complexity Multiplier |
|---|---|---|---|
| ICH E2B(R3) | Case Identification | primaryid linkage (3 tables) | 1.5× |
| ICH E2B(R3) | Drug Characterization | role_cod suspect filtering | 1.2× |
| ICH E2B(R3) | Reaction Coding | MedDRA PT handling | 1.1× |
| ICH E2B(R3) | Temporal Analysis | event_dt → quarter conversion | 1.8× |
| CIOMS VI | Signal Detection | PRR ≥ 2.0, N ≥ 3 threshold | 1.3× |
| EMA Guidelines | Disproportionality | Expected vs. observed ratio | 1.4× |
| | | **Cumulative Complexity Factor** | **2.97×** |
| Base Difficulty | 5.0 (standard data task) | **5.0** | — |
| Adjusted Difficulty | 5.0 × 2.97 ÷ 2 | **Derived Difficulty Score** | **7.43/10** |

*Table caption: FDA Complexity: Expected = (drug_cases * reac_cases) / total_cases; prr = pair_cases / expected*

---

### Data Complexity Assessment (FAERS-Specific)

| Data Characteristic | Description | Challenge Level | Score (0–10) |
|---|---|---|---|
| Relational Structure | 3-table inner joins required | High | 8.0 |
| Date Format Variability | YYYYMMDD, YYYYMM, decimal suffixes | High | 8.5 |
| Missing Data Prevalence | Requires dropna/fillna handling | Moderate | 6.5 |
| Case Deduplication | nunique() on primaryid | Moderate | 6.0 |
| Volume Scalability | 400K+ records (full dataset) | High | 7.5 |
| Schema Variability | Column casing, whitespace | Moderate | 6.5 |
| Temporal Granularity | Quarterly aggregation logic | High | 8.0 |
| | | **Subtotal Average** | **7.29/10** |

*Table caption: FAERS Dataset Complexity: Schema Variability (6.5) Applied via columns.str.lower().str.strip()*

---

### Composite Difficulty Score Calculation

| Dimension | Source Table | Score | Weight | Weighted Score |
|---|---|---|---|---|
| Software Engineering | Table A | 7.17 | 20% | 1.434 |
| AI/ML Benchmarks | Table B | 7.88 | 25% | 1.970 |
| Regulatory Standards | Table C | 7.43 | 25% | 1.858 |
| Data Complexity | Table D | 7.29 | 20% | 1.458 |
| Domain Expertise Required | (Expert estimate) | 8.50 | 10% | 0.850 |
| **TOTAL** | | | **100%** | **7.57/10** |

*Table caption: Weighted Composite Difficulty Score Representing Task-Level Complexity Across 4 Tournament Rounds*
