# Baseline for Current AI Prediction of Patient Outcomes in Oncology Trials
## Part 1: Foundational Baseline, Survival Prediction & Adverse Event Forecasting

---

## What Current AI in This Area Actually Is

The current baseline is not large general-purpose reasoning systems making end-to-end trial decisions, but a collection of relatively narrow supervised models trained for specific endpoints such as overall survival, progression-free survival, response, hospitalization, immune-related adverse events, or early treatment discontinuation.[1][2][4][5][6][7][8][9]

Across the strongest studies, the dominant model families are still classical or hybrid machine learning on structured data, including ridge Cox regression, fast survival support vector machines, random survival forests, XGBoost, elastic net, random forests, and modest neural networks, with deep learning used mainly for medical imaging and multimodal fusion rather than for autonomous clinical reasoning.[1][4][5][7][8]

The most common inputs are routine blood tests, basic clinical variables, baseline metastatic burden, adverse-event logs, electronic patient-reported outcomes, passive sensor streams such as smartphone step counts, and CT-derived imaging features, while the strongest trial-linked evidence usually comes from retrospective analyses of completed phase III datasets or combined real-world plus trial validation rather than from prospective trials that change care on the basis of the AI score itself.[1][2][5][6][7][8]

---

## Survival and Death Prediction

The clearest broad baseline for survival prediction is SCORPIO, a 2025 pan-cancer machine learning system for immune checkpoint inhibitor outcomes built from routine blood tests and clinical data in 9,745 ICI-treated patients across 21 cancer types.[1]

SCORPIO used an unweighted ensemble of three survival learners—ridge Cox regression, fast survival SVM, and random survival forest—trained on 33 selected variables, and it predicted overall survival at 6, 12, 18, 24, and 30 months after ICI administration with median time-dependent AUC values of 0.763 and 0.759 in two internal test sets, 0.759 in an independent real-world cohort, and 0.725 in a large external health-system cohort.[1]

That same study is especially useful as a benchmark because it also validated across 12 experimental arms from 10 phase III trial cohorts, where the best trial-specific overall-survival performance reached median AUC(t) 0.782 in IMvigor211, and because it outperformed tumor mutational burden, whose corresponding median AUC(t) values in the internal test sets were only 0.503 and 0.543.[1]

Importantly, SCORPIO also produced clinically interpretable risk strata rather than just a binary score: in the hold-out data, the low-risk group had a hazard ratio for death of 0.25 versus the high-risk group, and the moderate-risk group had a hazard ratio of 0.48, while in the independent real-world cohort the corresponding hazard ratios were 0.16 and 0.38.[1]

A different but highly relevant survival-prediction paradigm comes from on-treatment signal modeling rather than pretreatment prediction.[6]

Sun and colleagues trained a novel machine-learning adverse-event signature using all treatment-emergent adverse events from the durvalumab and chemotherapy arms of the MYSTIC clinical trial, then tested it on four independent durvalumab-containing non-small-cell lung cancer trial cohorts using only the first 60 days of treatment-emergent adverse events to predict long-term overall survival.[6]

Its performance was not reported as a standard ROC AUC but as treatment-specific survival separation: in patient-level meta-analysis, the signature identified longer survivors among durvalumab-treated patients with a meta-hazard ratio of 0.83, whereas it showed no analogous signal in chemotherapy or placebo cohorts, which had a meta-hazard ratio of 1.02.[6]

That study matters because it shows that some of the best current oncology trial prediction does not forecast the exact date of death, but instead converts early on-treatment signals into a relative long-term survival-risk stratifier that can be available before the first major imaging reassessment.[6]

The 2025 Predictive Biomarker Modeling Framework pushes this one step further toward trial enrichment rather than bedside prognosis.[7]

PBMF is a neural-network framework based on contrastive learning that searches clinicogenomic data for predictive rather than merely prognostic biomarkers, and in retrospective immuno-oncology trial analyses it identified a predictive biomarker from early study data that was associated with a 15% improvement in survival risk relative to the original trial population.[7]

This is important for a baseline because it shows that current AI is already being used not just to predict outcomes for individual patients, but to retrospectively ask whether trial eligibility or biomarker strategy could have been improved.[7]

Multimodal imaging systems are also reaching respectable, though not yet revolutionary, survival performance.[8]

In unresectable hepatocellular carcinoma treated with immune checkpoint inhibitors, a 2025 multimodal fusion system combined an ensemble deep-learning imaging signature from multiphase CT with clinical features using random survival forest and achieved external-test C-indices of 0.74 for overall survival and 0.69 for progression-free survival, while the imaging-only ensemble deep-learning core achieved 0.75 and 0.70 in the camrelizumab-plus-apatinib subgroup.[8]

That model explicitly predicted overall survival at 1, 2, and 3 years and progression-free survival at 3, 6, and 12 months, which makes it a useful example of a modern multimodal time-to-event predictor with clearly stated horizons.[8]

---

## Adverse Event and Hospitalization Prediction

When the endpoint is a rare near-term toxicity event rather than long-term survival, current AI often looks less impressive in externally validated trial data than in smaller proof-of-concept cohorts.[2][4][5]

The clearest trial-based baseline comes from the metastatic castration-resistant prostate cancer DREAM challenge, which used four phase III docetaxel trials from Project Data Sphere to predict whether a patient would discontinue treatment within 3 months, defined specifically as 91.5 days or the first four 3-week cycles, because of an adverse event or possible adverse event.[2]

Across 34 teams and 61 submitted models, the validation-set AUPRC ranged only from 0.088 to 0.178, compared with a random baseline of 0.104, and only a post-challenge weighted ensemble of the top seven models reached AUPRC 0.230.[2]

That is a valuable baseline because it shows that even with relatively clean phase III trial data, rare-event toxicity prediction can remain modest and difficult, and because AUPRC rather than AUROC is the right metric when the positive class is only around 10% of patients.[2]

The same challenge also showed that high-risk patients identified by the consensus models had roughly double the early-discontinuation event rate of low-risk patients, and that visceral liver metastases and prior analgesic or ACE-inhibitor use were among features separating risk groups.[2]

A more algorithmically explicit follow-on paper from the winning team described a machine-learning approach that integrated survival status and adverse-event severity into the discontinuation model, reinforcing that current useful trial AI in this space is still mostly tabular and feature-engineered rather than end-to-end deep learning.[3]

For immune-related adverse events, one of the most cited proof-of-concept studies used weekly electronic patient-reported symptom questionnaires from advanced cancer patients receiving checkpoint inhibitors and trained XGBoost models on ePRO and structured EHR data.[4]

In that study, patients completed baseline and then weekly symptom questionnaires until treatment discontinuation or 6 months of follow-up, and the model predicting the presence of an irAE reached accuracy 0.97, AUC 0.99, F1 score 0.94, and MCC 0.92, whereas prediction of irAE onset was harder and achieved accuracy 0.96, AUC 0.93, F1 score 0.67, and MCC 0.64.[4]

These results are best interpreted as a high-potential but small-sample proof of concept: the performance numbers are excellent, but the study involved only 34 patients and 820 questionnaires, so it is not yet a definitive operational benchmark for multicenter trial practice.[4]

Passive-sensing models are another current direction for short-term adverse-event forecasting.[5]

Brouwer and colleagues monitored daily smartphone step counts in patients receiving systemic anticancer therapy and trained elastic net, random forest, and neural network models to detect adverse events in the upcoming 7 days using the prior 2 weeks of activity data.[5]

Among 76 patients, unplanned hospitalization in the next week could be predicted with AUC 0.88 by random forest, 0.84 by neural network, and 0.83 by elastic net, but the same models failed to predict treatment modifications or the broader aggregate endpoint of "any clinically relevant adverse event," where AUCs were only 0.28 to 0.51 and 0.32 to 0.50 respectively.[5]

This is a very useful baseline because it shows that passive behavioral data can be strong for a narrow, imminent outcome such as hospitalization over a 7-day horizon, while remaining weak for more heterogeneous oncology endpoints.[5]

---

*Continues in: `chunk_02_response_metrics_conclusions.md`*
*References: `chunk_03_bibtex_references.md`*
