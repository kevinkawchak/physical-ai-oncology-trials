# Chunk 01 — Current AI Baseline & Short-Horizon Prediction in Oncology

> **Source document:** Deep_Research_A_Oncology.docx  
> **Chunk scope:** Sections 1–3 of 6 body sections  
> **Companion files:** `chunk_02_multimodal_and_limitations.md` · `chunk_03_bibtex.md`  
> **Citation keys used:** [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11]

---

## 1. What Current Oncology Prediction AI Actually Is

The present baseline is not a single "AI for cancer prognosis" system, but a collection of mostly narrow, supervised prediction models trained for specific endpoints, tumor types, and data modalities. In the recent oncology literature, the most common winning models are still tree ensembles such as random forest and XGBoost, while survival-specific work commonly uses random survival forests, survival support-vector machines, penalized Cox models, and task-specific deep learning on images or slides rather than general-purpose reasoning systems. [1][2]

A key baseline point is that, for structured real-world survival prediction, machine learning has not yet shown a consistent pooled advantage over traditional Cox proportional-hazards models. In a 2025 systematic review and meta-analysis of cancer survival prediction using real-world data, the standardized mean difference in AUC or C-index between machine-learning models and Cox models was 0.01 with a 95% confidence interval from −0.01 to 0.03, which is effectively no average gain. [1]

That matters for benchmarking future systems. If a future "more powerful and faster" AI claims a major oncology prediction breakthrough, the right comparison is not against the most optimistic internal-validation paper, but against externally validated current baselines measured on the same endpoint, the same horizon, and the same population, with calibration and clinical utility assessed alongside discrimination. [1][2][16][17]

---

## 2. Short-Horizon Prediction of Adverse Events and Near-Term Death

### 2.1 EHR-Based 180-Day Mortality

For near-term mortality, one of the strongest currently documented baselines is an electronic-health-record gradient-boosting classifier for 180-day mortality in outpatient oncology. In a prospective cohort of 24,582 patients, the model achieved an AUC of **0.89** for death within 180 days; at a prespecified 40% risk threshold, observed 180-day mortality was 45.2% in the high-risk group versus 3.1% in the low-risk group, and the model's positive predictive value was 45.2% despite an overall event rate of only 4.2%. The predictions were generated 4–8 days before the oncology encounter, making this a true operational short-horizon model rather than a retrospective score. [3]

### 2.2 Patient-Reported Outcome Models

When the input is limited to patient-reported outcomes rather than the full EHR, performance is usually lower but still clinically relevant. In women with ovarian cancer, a voting ensemble built from repeated biopsychosocial patient-reported outcome measures predicted 180-day mortality with:

| Metric | Value |
|---|---|
| Accuracy | 0.79 |
| Sensitivity | 0.71 |
| Specificity | 0.80 |
| AUROC | 0.76 |

In a separate advanced-cancer symptom-based study, XGBoost was the best model for 180-day mortality with AUROC 0.69, sensitivity 0.68, specificity 0.62, PPV 0.66, and NPV 0.64. [4]

### 2.3 Radiation Oncology — SHIELD-RT

For adverse events during active treatment, an important real-world example is the **SHIELD-RT** program in radiation oncology. The original deployed model was a gradient-boosted tree using structured EHR variables such as demographics, treatment, vitals, laboratory results, medications, and prior acute-care use; the prospective randomized implementation reduced acute-care events by **45%** and costs by **48%**. In later multi-institutional validation across more than 22,000 radiotherapy courses, the same model achieved:

| Metric | Value |
|---|---|
| AUROC (site A / site B) | 0.756 / 0.770 |
| Sensitivity | ~55%–58% |
| Specificity | ~80% |
| Brier Score | < 0.06 |

Clear separation of high-risk versus low-risk event rates was maintained. [5][6]

### 2.4 Chemotherapy Toxicity Prediction

Chemotherapy-toxicity prediction is similarly current but still mostly narrow. For colorectal cancer patients starting fluoropyrimidine-based chemotherapy, an XGBoost model predicted 30-day cardiotoxicity with:

| Metric | Value |
|---|---|
| AUC | 0.816 |
| Precision | 0.619 |
| F1 Score | 0.406 |
| Cohort event rate | 18.74% |

This outperformed random forest (AUC 0.804) and logistic regression (AUC 0.812). In early breast cancer receiving neoadjuvant systemic treatment, machine-learning models predicting grade 3–4 toxicity improved markedly when regimen and dose-intensity information were added: the elastic-net logistic model rose from AUROC 0.59 to **0.75**, and the support-vector machine rose from 0.64 to **0.75**. [7][8]

At the literature-wide level, toxicity prediction is promising but not yet uniformly robust. A 2026 systematic review and meta-analysis of radiation-induced toxicity models in head and neck cancer found a **pooled AUROC of 0.76**, judged as moderate discrimination, and reported that incorporating imaging biomarkers improved performance. The same review emphasized that most studies were retrospective and methodologically heterogeneous, which limits how aggressively one should interpret headline scores. [9]

---

## 3. Medium- and Long-Horizon Survival and Recurrence Prediction

### 3.1 Multimodal Pathology + Clinical Variables

For longer-horizon prognosis, current AI is increasingly image-driven. In resected non-small cell lung cancer, the **AIM-LCpro** model used patient-level multimodal weakly supervised learning on whole-slide pathology plus clinical variables to predict 5-year disease-free survival (DFS) and 5-year overall survival (OS):

| Cohort | DFS C-index | OS C-index |
|---|---|---|
| Internal validation | 0.785–0.804 | 0.726–0.787 |
| External validation | 0.693–0.749 | 0.658–0.711 |

Those numbers are strong enough to be useful, but the internal-to-external drop is exactly the sort of gap that should be treated as part of the real baseline. [10]

### 3.2 Image-Based AI — Literature Aggregate

The image-based literature as a whole often reports higher aggregate performance, but it is also more heterogeneous. A 2025 systematic review and meta-analysis of image-based AI for lung-cancer prognostic evaluation pooled 106 prognosis studies and found:

- Sensitivity: **0.83**
- Specificity: **0.83**
- AUC: **0.90**

In 53 studies that separated low-risk from high-risk groups, the pooled hazard ratio for overall survival was **2.53** and for progression-free survival was **2.80**, indicating clinically meaningful risk stratification. However, the same paper reported substantial heterogeneity and evidence of publication bias in some outcome groups, so these pooled values are better treated as an upper-bound literature estimate than as a deployment-grade benchmark. [11]

### 3.3 Pathology Foundation Models — PROGPATH

A particularly important 2025 development is the rise of pathology foundation models fused with routine clinical variables. **PROGPATH** is a pancancer weakly supervised deep-learning system using:

- A **Vision Transformer (ViT)**-based pathology encoder
- Attention-based multiple-instance learning (ABMIL)
- A cross-attention transformer
- A cancer-aware routing layer

It integrates whole-slide images with clinical variables, trained on **6,670 patients** across **15 cancer types** and externally tested on **17 independent cohorts** from three continents. Across those external cohorts, reported C-indexes ranged from **0.713 to 0.805**. That makes current pathology foundation models one of the most credible present-day baselines for medium- to long-horizon survival prediction across cancer types. [15]

---

*Continues in `chunk_02_multimodal_and_limitations.md` →*
