# Chunk 02 — Immunotherapy & Multimodal Models, Measurement Standards, and Open Questions

> **Source document:** Deep_Research_A_Oncology.docx  
> **Chunk scope:** Sections 4–6 of 6 body sections  
> **Companion files:** `chunk_01_baseline_and_short_horizon.md` · `chunk_03_bibtex.md`  
> **Citation keys used:** [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15] [16] [17]

---

## 4. Immunotherapy and Multimodal Precision Models

### 4.1 Multimodal PD-(L)1 Response Prediction

Immunotherapy prediction is one of the clearest examples of why multimodal AI is now clinically interesting. In non-small cell lung cancer (NSCLC), a 2022 multimodal model integrating radiology, pathology, and genomics to predict response to PD-(L)1 blockade reached:

| Model / Feature Set | AUC |
|---|---|
| Multimodal (radiology + pathology + genomics) | **0.80** |
| PD-L1 expression score alone | 0.73 |
| Tumor mutational burden (TMB) alone | 0.61 |

This is not a fixed death-horizon model, but it is a current baseline for therapy-response prediction using multimodal oncology data. [12]

### 4.2 SCORPIO — Routine Blood Tests Only

A more operationally attractive model is **SCORPIO**, because it uses only routine blood tests and basic clinical data rather than specialized molecular testing. SCORPIO ensembles:

- Ridge Cox regression
- Fast survival SVM
- Random survival forest (for overall survival)
- Ridge logistic regression, SVM, random forest (for clinical benefit)

It prognosticated overall survival after immune-checkpoint inhibitor (ICI) treatment at 6, 12, 18, 24, and 30 months:

| Cohort | Median time-dependent AUC |
|---|---|
| Hold-out test | 0.763 |
| Internal real-world test | 0.759 |
| External health-system cohort | 0.725 |

AUC for clinical benefit was 0.714 (hold-out) and 0.641 (internal real-world test). Risk stratification was clinically meaningful: low-risk patients had a **hazard ratio for death of 0.25** compared with the high-risk group; clinical-benefit rates separated from **55.96%** (low-risk) to **12.12%** (high-risk) in hold-out testing. [13]

### 4.3 Late-Fusion Multimodal Model — First-Line Metastatic NSCLC

A complementary result comes from first-line metastatic NSCLC treated with pembrolizumab-based immunotherapy. In a **317-patient** multimodal dataset combining clinical features, PET imaging, digitized pathology, and bulk RNA-seq, the best late-fusion multimodal model reached:

| Endpoint | Best Metric |
|---|---|
| Overall survival | C-index **0.75** |
| 1-year death prediction | AUC **0.81** |
| 6-month progression | AUC **0.67** (best multimodal) |

That gap between the OS/1-year AUC and the 6-month progression AUC is important: it shows that current AI often predicts death or long-run survival better than noisy short-run progression or response endpoints. [14]

---

## 5. What the Measurement Baseline Should Include

For a serious baseline comparison against future systems, the right metrics are not just AUROC headlines. Current oncology AI is measured with:

- **AUROC** — binary outcomes
- **Time-dependent AUC** — survival at fixed horizons
- **C-index** — censored time-to-event prediction
- **Brier score / calibration plots** — calibration quality
- **PPV, sensitivity, specificity, alert rate** — deployment-oriented threshold metrics [3][6][13][16][17]

### 5.1 Practical 2025/2026 Benchmark Summary

| Prediction Domain | Externally Validated AUROC / C-index Range |
|---|---|
| Short-horizon adverse events or utilization | **0.75–0.82** |
| 6-month mortality (rich EHR) | ~**0.89** |
| 6-month mortality (patient-reported data only) | ~**0.69–0.76** |
| Multimodal survival (pathology / imaging / transcriptomics) | ~**0.70–0.81** (C-index or AUC) |

Reported values above **0.90** do exist, especially in single-center toxicity or radiomics studies, but those should be treated as optimistic rather than definitive baselines unless they reproduce across external cohorts. [3][4][6][7][8][9][10][11][13][14][15]

### 5.2 Four Endpoint Families Any Future System Must Address

Current oncology AI usually predicts one of four endpoint types:

1. **Imminent adverse events** during treatment
2. **Fixed-horizon mortality** (30-day, 90-day, 180-day death)
3. **Fixed-horizon recurrence or survival** (1-year death, 5-year OS)
4. **Treatment-specific benefit** (response to immunotherapy)

A future system should therefore be judged not only on whether it is "more accurate," but on whether it can improve across **all four endpoint families** while preserving calibration, external validity, and real clinical impact. [1][2][9][16][17]

---

## 6. Open Questions and Limitations

### 6.1 Retrospective, Single-Center Bias

The main limitation of the current baseline is that much of the literature is still retrospective, single-center, and subject to data shift, selective reporting, and heterogeneous endpoint definitions. Even recent reviews that found promising performance also emphasized incomplete reporting, limited attention to outliers and drift, and inconsistent access to training or validation data. [2][9][11][16][17]

### 6.2 Surrogate Endpoint Misalignment

A second limitation is that current studies often optimize surrogate endpoints that are easier to model than the decisions clinicians actually care about. The immunotherapy literature makes this especially clear: overall survival can be predicted moderately well across 6–30 month horizons, but clinical benefit and short-run progression are materially harder. A future AI system that can reliably forecast both survival **and** treatment toxicity, dynamically over time and across cancer types, would represent a genuine advance rather than a marginal improvement on existing narrow models. [13][14]

### 6.3 Internal Validation ≠ Clinical Impact

A third limitation is that a model can look excellent on internal testing without changing care. The strongest current baseline studies are the ones that either validate externally across institutions or show clinical workflow impact, such as SHIELD-RT. In other words, the real baseline for future comparison should be **"externally validated and clinically actionable prediction"**, not simply "best retrospective score." [5][6][16][17]

---

*← Begins in `chunk_01_baseline_and_short_horizon.md`*  
*All citations resolvable via `chunk_03_bibtex.md` →*
