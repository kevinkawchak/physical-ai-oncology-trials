# Section 4.3–4.4: Round 4 Notebook Results & Reference-Based Scoring Metrics

## 4.3 Round 4 Notebook Results

The notebook introduction provides the cross-table temporal signal emergence detection competition task, scoring system, and output structure. It is important to note that correctness is based on a head-to-head comparison to 01_reference_solution.csv. The following table shows time, speed, and cost benefits favoring the open-source gpt-oss-120b over the proprietary Gpt-5.2-pro. It was significant that the three Gpt-5.2-pro helper functions contributed to higher Code Quality score (0.95 vs. 0.90) and decisive error handling. The Round 4 results output provides each of the four reference-based scoring metrics (correctness, methodology, code quality, and algorithmic efficiency) of Code_A (Gpt-5.2-pro) and Code_B (OpenAl gpt-oss-120b). Gpt-5.2-pro (0.9775/1.0) correctness matched the reference solution (1.0/1.0), with a top methodology score (1.0/1.0); while OpenAl gpt-oss-120b (0.2050/1.0) failed to execute, but had a code quality score of 9.0/10. The multi-metric score framework is illustrated in the Scoring Wheel figure and itemized with final weighted scores in the tables below.

---

### Round 4 Code Generation Metrics (Single Iteration)

| Metric | Gpt-5.2-pro | gpt-oss-120b |
|---|---|---|
| Generation Time | 2 min 49.37 sec | 0 min 25.65 sec |
| Generation Speed | ~19.5 tokens/sec | 122.34 tokens/sec |
| Generation Cost | $1.42 | $0.01 |
| Output Tokens | 3,292 T | ~3,134 T (est.) |
| Input Tokens | 46.8 KT | ~41.6 KT (est.) |
| Lines of Code | 127 LOC | 142 LOC |
| Helper Functions | 3 (_empty, _norm_cols, _parse_quarter) | 0 (inline logic only) |

*Table caption: Proprietary vs. Open-Source Model Differences*

---

### Round 4 Results. Code_A = Gpt-5.2-pro. Code_B = OpenAl gpt-oss-120b

```
======================================================================
STARTING HEAD-TO-HEAD COMPETITION
======================================================================
Competition ID: FAERS_25Q3_COMPETITION
Task: Task 1: Cross-Table Temporal Signal Emergence Detection
Data Sample: 10%
Reference Solution: 4 signals
Output Directory: /content/drive/MyDrive/Colab Notebooks/Inputs/FAERS_LLM_Competition/results/run_20251212_215200
======================================================================
COMPETITION ROUND: T1_TEMPORAL_SIGNAL_R1_038e016e
   Task: Task 1: Cross-Table Temporal Signal Emergence Detection
   Reference Solution: 4 signals
======================================================================
   Code_A hash: 9838497cf082
   Code_B hash: ce3db45a915c
Executing Code_A...
   Runtime: 0.123s | Success: True
   Output: 4 rows
Executing Code_B...
   Runtime: 0.001s | Success: False
======================================================================
ROUND RESULTS (Reference-Based Scoring)
======================================================================
   Code_A Score: 0.9775 [PASS]
      Correctness:           1.0000 (×0.45)
         ├─ Pair F1:         1.0000 (P:1.00 R:1.00)
         ├─ PRR Correlation:  1.0000
         ├─ Quarter Match:    1.0000
         └─ Matched Pairs:    4/4
      Methodology:           1.0000 (×0.30)
      Code Quality:          9.50/10 (×0.15)
      Algorithmic Effic.:    0.8500 (×0.10)
   Code_B Score: 0.2050 [FAILED]
      Correctness:           0.0000 (×0.45)
         ├─ Pair F1:         0.0000 (P:0.00 R:0.00)
         ├─ PRR Correlation:  0.0000
         ├─ Quarter Match:    0.0000
         └─ Matched Pairs:    0/0
      Methodology:           0.0000 (×0.30)
      Code Quality:          9.00/10 (×0.15)
      Algorithmic Effic.:    0.7000 (×0.10)
WINNER: Code_A
======================================================================
```

*Table caption: Round 4 Competition Notebook Section 6.1 Results*

---

## 4.4 Reference-Based Scoring Metrics

### Correctness Scoring Breakdown (45% Weight)

| Subcategory | Weight | Gpt-5.2-pro | gpt-oss-120b | Assessment Method |
|---|---|---|---|---|
| Pair F1 Score | 40% | 1.0000 | 0.0000 | Precision/Recall of pairs |
| PRR Correlation | 30% | 1.0000 | 0.0000 | Pearson correlation |
| Quarter Match Rate | 20% | 1.0000 | 0.0000 | Emergence quarter match |
| Structure Score | 10% | 1.0000 | 0.0000 | Column schema compliance |
| **CORRECTNESS TOTAL** | **100%** | **1.0000** | **0.0000** | **Sum of weighted scores** |
| **Contribution to Final** | **×0.45** | **0.4500** | **0.0000** | **Correctness × 0.45** |

*Table caption: Correctness Scoring Formula = 0.40 × F1 + 0.30 × r_PRR + 0.20 × Q_match + 0.10 × S_struct*

---

### Methodology Scoring Breakdown (30% Weight)

| Subcategory | Points | Gpt-5.2-pro | gpt-oss-120b | Assessment Criterion |
|---|---|---|---|---|
| Execution Success (Base) | +0.70 | ✓ +0.70 | ✗ +0.00 | Error-free execution |
| Warning Penalty | -0.10 each | 0 warnings | N/A (failed) | Max -0.30 for 3+ warnings |
| Vectorized Operations | +0.10 | ✓ +0.10 | N/A (failed) | Uses np.*, .str., .dt. |
| Function Count ≥ 2 | +0.10 | ✓ +0.10 | N/A (failed) | Modular code structure |
| Has Docstrings | +0.10 | ✓ +0.10 | N/A (failed) | Triple-quote documentation |
| **METHODOLOGY TOTAL** | **max 1.0** | **1.0000** | **0.0000** | **Capped at 1.0** |
| **Contribution to Final** | **×0.30** | **0.3000** | **0.0000** | **Methodology × 0.30** |

*Table caption: Methodology Based on Execution Success, Code Practices. Base = 0.70 + (3) +0.10 Bonuses*

---

### Code Quality Scoring Breakdown (15% Weight)

| Subcategory | Points | Gpt-5.2-pro | gpt-oss-120b | Detection Method |
|---|---|---|---|---|
| Base Score | +5.00 | +5.00 | +5.00 | All submissions start here |
| Has Docstrings | +1.50 | ✓ +1.50 | ✓ +1.50 | """ or ''' present |
| Has Type Hints | +0.50 | ✓ +0.50 | ✓ +0.50 | def func(x: Type) |
| Uses Vectorized Ops | +1.00 | ✓ +1.00 | ✓ +1.00 | np.*, .str., .dt., .agg() |
| Function Count ≥ 1 | +0.50 | ✓ +0.50 | ✓ +0.50 | At least one def |
| Function Count ≥ 3 | +0.50 | ✓ +0.50 | ✗ +0.00 | Three or more def |
| Comment Ratio 10–30% | +0.50 | ✓ +0.50 | ✓ +0.50 | # lines / code lines |
| **CODE QUALITY TOTAL** | **/10** | **9.50** | **9.00** | **Capped at 10.0** |
| **Contribution to Final** | **×0.015** | **0.1425** | **0.1350** | **(Quality/10) × 0.15** |

*Table caption: Quality Scoring via Static Analysis, ≥3 Helper Functions = Bonus*

---

### Algorithmic Efficiency Scoring Breakdown (10% Weight)

| Subcategory | Points | Gpt-5.2-pro | gpt-oss-120b | Detection Pattern |
|---|---|---|---|---|
| Base Score | +0.50 | +0.50 | +0.50 | All submissions start here |
| Uses Vectorized Ops | +0.15 | ✓ +0.15 | ✓ +0.15 | np.where, .str., .dt. |
| Uses Merge/Join | +0.10 | ✓ +0.10 | ✓ +0.10 | .merge() or .join() |
| Uses GroupBy | +0.10 | ✓ +0.10 | ✓ +0.10 | .groupby() |
| Nested Loops ≥ 2 | -0.15 | ✓ -0.00 | ✗ -0.15 | Indent-based detection |
| Nested Loops ≥ 3 | -0.20 | ✓ -0.00 | ✓ -0.00 | Deep nesting penalty |
| **EFFICIENCY TOTAL** | **max 1.0** | **0.8500** | **0.7000** | **Clamped [0.0, 1.0]** |
| **Contribution to Final** | **×0.10** | **0.0850** | **0.0700** | **Efficiency × 0.10** |

*Table caption: Algorithmic Efficiency via Static Code Analysis: ≥2 Nested Loops = Penalty*

---

### Final Weighted Score Summary (Round 4)

| Metric | Weight | Gpt-5.2-pro | gpt-oss-120b | Weighted Contribution |
|---|---|---|---|---|
| Correctness | 45% | 1.0000 | 0.0000 | 0.4500 vs. 0.0000 |
| Methodology | 30% | 1.0000 | 0.0000 | 0.3000 vs. 0.0000 |
| Code Quality | 15% | 9.50/10 | 9.00/10 | 0.1425 vs. 0.1350 |
| Alg. Efficiency | 10% | 0.8500 | 0.7000 | 0.0850 vs. 0.0700 |
| **Final Calculation** | | | | |
| Σ Weighted Scores | **100%** | **0.9775** | **0.2050** | — |
| Score Differential | — | +0.7725 (Gpt-5.2-pro advantage) | | 77.25% margin |
| **WINNER** | — | **Gpt-5.2-pro (Code_A Submission)** | | |

*Table caption: Weighted Score Aggregation Sums Prior Tables' Metrics*
