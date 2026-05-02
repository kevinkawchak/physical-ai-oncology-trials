# Chunk 03 — BibTeX Reference Library

> **Source document:** Deep_Research_A_Oncology.docx  
> **Chunk scope:** All 17 cited references extracted from source document  
> **Companion files:** `chunk_01_baseline_and_short_horizon.md` · `chunk_02_multimodal_and_limitations.md`  
> **Format:** BibTeX + inline annotation for each entry

---

## Usage Notes

- All citation keys follow the pattern `[AuthorYYYYShortTitle]`
- Numeric in-text citations `[1]`–`[17]` in Chunks 01 and 02 map to the entries below
- DOI links are canonical; URL fallbacks provided where DOIs resolve differently
- Entries marked **`[META-ANALYSIS]`** are systematic reviews — treat their pooled values as literature-level, not deployment-grade, benchmarks
- Entries marked **`[REPORTING GUIDELINE]`** provide the methodological scaffolding (TRIPOD+AI, CREMLS) against which new trial reporting should be structured

---

## BibTeX Entries

### [1] — Huang 2025 · ML vs Cox Survival Meta-Analysis `[META-ANALYSIS]`

> **Role in document:** Establishes the core null result — ML provides no significant mean gain over Cox regression for real-world cancer survival prediction (SMD in AUC/C-index = 0.01, 95% CI −0.01 to 0.03). Referenced in Sections 1, 2, 5, and 6. Anchor citation for benchmarking rationale.

```bibtex
@article{Huang2025CancerSurvivalMetaAnalysis,
  author    = {Huang, Yinan and Bazzazzadehgan, Shadi and Li, Jieni and
               Arabshomali, Arman and Li, Mai and Bhattacharya, Kaustuv and
               Bentley, John P. and others},
  title     = {Comparison of machine learning methods versus traditional Cox
               regression for survival prediction in cancer using real-world
               data: a systematic literature review and meta-analysis},
  journal   = {BMC Medical Research Methodology},
  year      = {2025},
  volume    = {25},
  pages     = {243},
  doi       = {10.1186/s12874-025-02694-z},
  url       = {https://doi.org/10.1186/s12874-025-02694-z}
}
```

---

### [2] — Smiley 2025 · Methodological & Reporting Quality Review `[META-ANALYSIS]`

> **Role in document:** Provides the methodological quality audit of the ML oncology literature. Referenced in Sections 1, 5, and 6 to establish that retrospective bias, heterogeneous endpoints, and reporting gaps limit the real-world generalizability of headline scores.

```bibtex
@article{Smiley2025OncologyMLReportingQuality,
  author    = {Smiley, A. S. Aref and Villarreal-Zegarra, David and
               Reategui-Rivera, C. Mahony and Escobar-Agreda, Stefan and
               Finkelstein, Joseph},
  title     = {Methodological and reporting quality of machine learning studies
               on cancer diagnosis, treatment, and prognosis},
  journal   = {Frontiers in Oncology},
  year      = {2025},
  volume    = {15},
  doi       = {10.3389/fonc.2025.1555247},
  url       = {https://doi.org/10.3389/fonc.2025.1555247}
}
```

---

### [3] — Manz 2020 · EHR 180-Day Mortality Prediction `[PROSPECTIVE]`

> **Role in document:** Primary benchmark for short-horizon EHR-based mortality prediction. AUC 0.89 in 24,582-patient prospective cohort; PPV 45.2% at 40% threshold; 4–8-day prospective lead time. Sections 2.1 and 5.

```bibtex
@article{Manz2020Oncology180DayMortality,
  author    = {Manz, Christopher R. and Chen, Jinbo and Liu, Manqing and others},
  title     = {Validation of a Machine Learning Algorithm to Predict 180-Day
               Mortality for Outpatients With Cancer},
  journal   = {JAMA Oncology},
  year      = {2020},
  volume    = {6},
  number    = {11},
  pages     = {1723--1730},
  doi       = {10.1001/jamaoncol.2020.4331},
  url       = {https://doi.org/10.1001/jamaoncol.2020.4331}
}
```

---

### [4] — Sidey-Gibbons 2022 · PRO-Based Ovarian Cancer Mortality `[PROSPECTIVE]`

> **Role in document:** Baseline for patient-reported-outcome-only mortality prediction in ovarian cancer (AUROC 0.76). Demonstrates performance ceiling when EHR access is absent. Section 2.2.

```bibtex
@article{SideyGibbons2022OvarianPROMortality,
  author    = {Sidey-Gibbons, Chris J. and Sun, Charlotte and Schneider, Amy
               and others},
  title     = {Predicting 180-day mortality for women with ovarian cancer using
               machine learning and patient-reported outcome data},
  journal   = {Scientific Reports},
  year      = {2022},
  volume    = {12},
  pages     = {21269},
  doi       = {10.1038/s41598-022-22614-1},
  url       = {https://doi.org/10.1038/s41598-022-22614-1}
}
```

---

### [5] — Hong 2020 · SHIELD-RT Prospective RCT `[RANDOMIZED]`

> **Role in document:** Gold-standard clinical-impact baseline — SHIELD-RT reduced acute-care events 45% and costs 48% in a prospective randomized trial. Model: gradient-boosted tree on structured EHR. Sections 2.3 and 6.

```bibtex
@article{Hong2020SHIELDRT,
  author    = {Hong, Julian C. and Eclov, Neville C. W. and Dalal, Nicole H.
               and others},
  title     = {System for High-Intensity Evaluation During Radiation Therapy
               (SHIELD-RT): A Prospective Randomized Study of Machine
               Learning-Directed Clinical Evaluations During Radiation and
               Chemoradiation},
  journal   = {Journal of Clinical Oncology},
  year      = {2020},
  volume    = {38},
  number    = {31},
  pages     = {3652--3661},
  doi       = {10.1200/JCO.20.01688},
  url       = {https://doi.org/10.1200/JCO.20.01688}
}
```

---

### [6] — Elia 2025 · SHIELD-RT Multi-Institutional Validation `[EXTERNAL VALIDATION]`

> **Role in document:** Multi-institutional external validation of SHIELD-RT across 22,000+ radiotherapy courses. AUROC 0.756–0.770; Brier < 0.06. Establishes that prospective RCT performance generalizes. Sections 2.3 and 6.

```bibtex
@article{Elia2025SHIELDRTExternalValidation,
  author    = {Elia, M. V. and Benson, R. and Bhargava, N. and Levey, J. and
               Eclov, N. and Friesner, I. and others},
  title     = {Multi-Institutional Validation of the SHIELD-RT Machine Learning
               Model to Prevent Acute Care Events during Radiotherapy},
  journal   = {International Journal of Radiation Oncology, Biology, Physics},
  year      = {2025},
  volume    = {123},
  number    = {1},
  pages     = {S74--S75},
  url       = {https://computationalhealth.berkeley.edu/multi-institutional-validation-of-the-shield-rt-machine-learning-model-to-prevent-acute-care-events-during-radiotherapy/}
}
```

---

### [7] — Li 2022 · Fluoropyrimidine Cardiotoxicity Prediction `[RETROSPECTIVE]`

> **Role in document:** Narrow-task chemotherapy toxicity baseline. XGBoost AUC 0.816 for 30-day cardiotoxicity in colorectal cancer. Section 2.4.

```bibtex
@article{Li2022FluoropyrimidineCardiotoxicity,
  author    = {Li, Chao and Chen, Li and Chou, Chiahung and Ngorsuraches,
               Surachat and Qian, Jingjing},
  title     = {Using Machine Learning Approaches to Predict Short-Term Risk of
               Cardiotoxicity Among Patients with Colorectal Cancer After
               Starting Fluoropyrimidine-Based Chemotherapy},
  journal   = {Cardiovascular Toxicology},
  year      = {2022},
  volume    = {22},
  number    = {2},
  pages     = {130--140},
  doi       = {10.1007/s12012-021-09708-4},
  url       = {https://doi.org/10.1007/s12012-021-09708-4}
}
```

---

### [8] — Cai 2024 · Breast Cancer Neoadjuvant Toxicity Prediction `[RETROSPECTIVE]`

> **Role in document:** Demonstrates feature-engineering impact on toxicity prediction: adding regimen/dose-intensity improves AUROC from 0.59 → 0.75 (elastic-net) and 0.64 → 0.75 (SVM). Section 2.4.

```bibtex
@article{Cai2024BreastCancerToxicityPrediction,
  author    = {Cai, Lie and Deutsch, Thomas M. and Sidey-Gibbons, Chris and
               others},
  title     = {Machine Learning to Predict the Individual Risk of
               Treatment-Relevant Toxicity for Patients With Breast Cancer
               Undergoing Neoadjuvant Systemic Treatment},
  journal   = {JCO Clinical Cancer Informatics},
  year      = {2024},
  doi       = {10.1200/CCI.24.00010},
  url       = {https://doi.org/10.1200/CCI.24.00010}
}
```

---

### [9] — Ugwu 2026 · Head & Neck Radiation Toxicity Meta-Analysis `[META-ANALYSIS]`

> **Role in document:** 2026 pooled baseline for radiation-induced toxicity models: pooled AUROC 0.76 (moderate). Highlights heterogeneity and retrospective design predominance. Sections 2.4 and 6.

```bibtex
@article{Ugwu2026HeadNeckRadiationToxicityMeta,
  author    = {Ugwu, Gibson C. and Jalali, Farzad and Liu, Geoffrey and
               Li, Guojun and Langendijk, Johannes Albertus and
               Alizadeh, Behrooz Z.},
  title     = {The actual performance of ML/AI models in predicting
               radiation-induced toxicity in head and neck cancer: a systematic
               review and meta-analysis},
  journal   = {Radiotherapy and Oncology},
  year      = {2026},
  volume    = {216},
  pages     = {111350},
  doi       = {10.1016/j.radonc.2025.111350},
  url       = {https://doi.org/10.1016/j.radonc.2025.111350}
}
```

---

### [10] — Li 2025 · AIM-LCpro NSCLC Multimodal Survival `[EXTERNAL VALIDATION]`

> **Role in document:** Multimodal weakly supervised pathology + clinical model for 5-year DFS/OS in resected NSCLC. C-index 0.785–0.804 (internal); 0.693–0.749 (external). Illustrates internal-to-external performance drop. Section 3.1.

```bibtex
@article{Li2025AIMLCpro,
  author    = {Li, Yongmeng and Chai, Xiaodong and Yang, Moxuan and
               Xiong, Jiahang and Zeng, Junyang and Chen, Yun and others},
  title     = {Accurate prediction of disease-free and overall survival in
               non-small cell lung cancer using patient-level multimodal weakly
               supervised learning},
  journal   = {npj Precision Oncology},
  year      = {2025},
  volume    = {9},
  pages     = {197},
  doi       = {10.1038/s41698-025-00981-y},
  url       = {https://doi.org/10.1038/s41698-025-00981-y}
}
```

---

### [11] — Yuan 2025 · Image-Based Lung Cancer Prognosis Meta-Analysis `[META-ANALYSIS]`

> **Role in document:** Aggregated image-AI lung cancer baseline from 106 studies: sensitivity 0.83, specificity 0.83, AUC 0.90 (upper-bound estimate with publication bias). Pooled HR for OS 2.53. Sections 3.2 and 6.

```bibtex
@article{Yuan2025LungImageMeta,
  author    = {Yuan, Xinyu and Xu, Heli and Zhu, Junkai and Yang, Zixuan and
               Pan, Boyue and Wu, Lin and Chen, Huanhuan and others},
  title     = {Systematic review and meta-analysis of artificial intelligence
               for image-based lung cancer classification and prognostic
               evaluation},
  journal   = {npj Precision Oncology},
  year      = {2025},
  volume    = {9},
  pages     = {300},
  doi       = {10.1038/s41698-025-01095-1},
  url       = {https://doi.org/10.1038/s41698-025-01095-1}
}
```

---

### [12] — Vanguri 2022 · Multimodal PD-(L)1 Response Prediction `[RETROSPECTIVE]`

> **Role in document:** Establishes multimodal (radiology + pathology + genomics) baseline for ICI response prediction in NSCLC: AUC 0.80 vs PD-L1 alone (0.73) vs TMB alone (0.61). Section 4.1.

```bibtex
@article{Vanguri2022MultimodalPDL1NSCLC,
  author    = {Vanguri, Rami S. and Luo, Jia and Aukerman, Andrew and others},
  title     = {Multimodal integration of radiology, pathology and genomics for
               prediction of response to PD-(L)1 blockade in patients with
               non-small cell lung cancer},
  journal   = {Nature Cancer},
  year      = {2022},
  volume    = {3},
  pages     = {1151--1164},
  doi       = {10.1038/s43018-022-00416-8},
  url       = {https://doi.org/10.1038/s43018-022-00416-8}
}
```

---

### [13] — Yoo 2025 · SCORPIO ICI Prognosis from Routine Blood Tests `[EXTERNAL VALIDATION]`

> **Role in document:** Operationally attractive ICI baseline using only routine blood tests. Median time-dependent AUC 0.763 (hold-out) to 0.725 (external). HR 0.25 for low vs high risk. Clinically meaningful benefit stratification. Sections 4.2, 5, and 6.

```bibtex
@article{Yoo2025SCORPIO,
  author    = {Yoo, Seong-Keun and Fitzgerald, Conall W. and Cho, Byuri Angela
               and Fitzgerald, Bailey G. and Han, Catherine and others},
  title     = {Prediction of checkpoint inhibitor immunotherapy efficacy for
               cancer using routine blood tests and clinical data},
  journal   = {Nature Medicine},
  year      = {2025},
  volume    = {31},
  number    = {3},
  doi       = {10.1038/s41591-024-03398-5},
  url       = {https://doi.org/10.1038/s41591-024-03398-5}
}
```

---

### [14] — Captier 2025 · Late-Fusion Multimodal Metastatic NSCLC Immunotherapy `[PROSPECTIVE]`

> **Role in document:** Late-fusion multimodal model (clinical + PET + pathology + RNA-seq) for first-line pembrolizumab in metastatic NSCLC. C-index 0.75 OS, AUC 0.81 (1-year death), AUC 0.67 (6-month progression). Demonstrates endpoint difficulty gradient. Sections 4.3 and 6.

```bibtex
@article{Captier2025MultimodalNSCLCImmunotherapy,
  author    = {Captier, Nicolas and Lerousseau, Marvin and Orlhac, Fanny and
               Hovhannisyan-Baghdasarian, Narin{\'e}e and Luporsi, Marie and
               others},
  title     = {Integration of clinical, pathological, radiological, and
               transcriptomic data improves prediction for first-line
               immunotherapy outcome in metastatic non-small cell lung cancer},
  journal   = {Nature Communications},
  year      = {2025},
  volume    = {16},
  pages     = {614},
  doi       = {10.1038/s41467-025-55847-5},
  url       = {https://doi.org/10.1038/s41467-025-55847-5}
}
```

---

### [15] — Yuan 2025 · PROGPATH Pancancer Foundation Model `[EXTERNAL VALIDATION]`

> **Role in document:** Pathology foundation model (ViT + ABMIL + cross-attention + cancer-aware routing) trained across 15 cancer types and externally validated on 17 cohorts (3 continents). C-index 0.713–0.805. Section 3.3.

```bibtex
@article{Yuan2025PROGPATH,
  author    = {Yuan, Wei and Chen, Yijiang and Zhu, Biyue and Yang, Sen and
               Zhang, Jiayu and others},
  title     = {Pancancer outcome prediction via a unified weakly supervised
               deep learning model},
  journal   = {Signal Transduction and Targeted Therapy},
  year      = {2025},
  volume    = {10},
  pages     = {285},
  doi       = {10.1038/s41392-025-02374-w},
  url       = {https://doi.org/10.1038/s41392-025-02374-w}
}
```

---

### [16] — Collins 2024 · TRIPOD+AI Reporting Guideline `[REPORTING GUIDELINE]`

> **Role in document:** Methodological standard for reporting clinical prediction models using ML. Referenced throughout Sections 1, 5, and 6 as the reporting framework against which all models — past and future — should be evaluated.

```bibtex
@article{Collins2024TRIPODAI,
  author    = {Collins, Gary S. and Moons, Karel G. M. and Dhiman, Paula and
               Riley, Richard D. and Beam, Andrew L. and others},
  title     = {TRIPOD+AI statement: updated guidance for reporting clinical
               prediction models that use regression or machine learning
               methods},
  journal   = {BMJ},
  year      = {2024},
  volume    = {385},
  pages     = {e078378},
  doi       = {10.1136/bmj-2023-078378},
  url       = {https://doi.org/10.1136/bmj-2023-078378}
}
```

---

### [17] — El Emam 2024 · CREMLS Reporting Guideline `[REPORTING GUIDELINE]`

> **Role in document:** Complementary reporting standard (CREMLS) for prognostic and diagnostic ML models. Co-cited with TRIPOD+AI throughout Sections 1, 5, and 6. Together, [16] and [17] define the methodological floor for any new trial paper.

```bibtex
@article{ElEmam2024CREMLS,
  author    = {El Emam, Khaled and Leung, Tiffany I. and Malin, Bradley and
               Klement, William and Eysenbach, Gunther},
  title     = {Consolidated Reporting Guidelines for Prognostic and Diagnostic
               Machine Learning Models (CREMLS)},
  journal   = {Journal of Medical Internet Research},
  year      = {2024},
  volume    = {26},
  pages     = {e52508},
  doi       = {10.2196/52508},
  url       = {https://doi.org/10.2196/52508}
}
```

---

## Citation-to-Section Cross-Reference Index

| Cite Key | [1] | [2] | [3] | [4] | [5] | [6] | [7] | [8] | [9] | [10] | [11] | [12] | [13] | [14] | [15] | [16] | [17] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Chunk 01 §1** | ✓ | ✓ | | | | | | | | | | | | | | ✓ | ✓ |
| **Chunk 01 §2.1** | | | ✓ | | | | | | | | | | | | | | |
| **Chunk 01 §2.2** | | | | ✓ | | | | | | | | | | | | | |
| **Chunk 01 §2.3** | | | | | ✓ | ✓ | | | | | | | | | | | |
| **Chunk 01 §2.4** | | | | | | | ✓ | ✓ | ✓ | | | | | | | | |
| **Chunk 01 §3.1** | | | | | | | | | | ✓ | | | | | | | |
| **Chunk 01 §3.2** | | | | | | | | | | | ✓ | | | | | | |
| **Chunk 01 §3.3** | | | | | | | | | | | | | | | ✓ | | |
| **Chunk 02 §4.1** | | | | | | | | | | | | ✓ | | | | | |
| **Chunk 02 §4.2** | | | | | | | | | | | | | ✓ | | | | |
| **Chunk 02 §4.3** | | | | | | | | | | | | | | ✓ | | | |
| **Chunk 02 §5** | | | ✓ | ✓ | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Chunk 02 §6** | ✓ | ✓ | | | ✓ | ✓ | | | ✓ | | ✓ | | ✓ | ✓ | | ✓ | ✓ |
