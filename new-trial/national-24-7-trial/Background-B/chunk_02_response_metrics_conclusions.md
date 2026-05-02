# Baseline for Current AI Prediction of Patient Outcomes in Oncology Trials
## Part 2: Response Prediction, Evaluation Metrics, Conclusions & Open Questions

---

## Response and Progression Prediction

Response prediction studies often report the highest headline numbers, but they also tend to be the most vulnerable to optimism from smaller retrospective datasets and narrower clinical settings.[8][9]

The advanced breast cancer radiomics study by Zhao and colleagues is a good example of the upper end of current reported performance for binary efficacy prediction rather than broad trial-grade survival modeling.[9]

In that multicentered study of anti-PD-1 antibody-based combination therapy, the clinical-only model achieved AUC 0.672 in the training set and 0.634 in the validation set, whereas the radiomics model and the integrated clinical-radiomics machine-learning model achieved much higher values, with the integrated model reaching AUC 0.997 in training and 0.961 in validation.[9]

Those numbers show that imaging-derived features can contain strong response-related signal, but they should not be used as the sole baseline for "current AI capability," because they come from a more limited retrospective setting and use a binary efficacy endpoint rather than long-horizon externally validated survival across many tumor types.[9]

The hepatocellular carcinoma multimodal fusion study is more conservative and probably closer to the practical baseline for progression-type endpoints in contemporary imaging AI.[8]

Its multimodal system predicted progression-free survival at 3, 6, and 12 months with external-test C-index 0.69, and the imaging-only ensemble deep-learning model was only slightly behind at 0.66 to 0.70 depending on subgroup, which is much less spectacular than some response-AUC papers but likely more realistic for broader translation.[8]

---

## What the Quality Metrics Really Mean

A major reason this literature can be hard to compare is that different endpoint types demand different quality metrics.[1][2][4][8]

For survival or progression endpoints with censoring, the most informative metrics are usually the concordance index and time-dependent AUC, because these tell you how well the model ranks or discriminates patients over a specified time window such as 6 to 30 months for overall survival or 3 to 12 months for progression-free survival.[1][8]

For rare binary events such as early discontinuation due to toxicity, AUPRC is more informative than AUROC because it punishes models that look superficially good by mostly predicting the majority class, which is exactly why the docetaxel challenge used AUPRC and why a score of 0.230 was meaningfully better than the random baseline of 0.104 even though both numbers sound small at first glance.[2]

For imbalanced onset-detection tasks such as emerging irAEs, accuracy and AUC alone can overstate quality, so F1 score and Matthews correlation coefficient are important secondary checks, and in the ePRO irAE paper they showed a meaningful drop from near-perfect detection of ongoing irAEs to much more modest discrimination of true onset events.[4]

Hazard ratios and Kaplan-Meier separation remain important in this literature because many oncology models are still most clinically useful as risk-stratification tools rather than exact event-time forecasters, which is why a study such as the 60-day durvalumab adverse-event signature can be valuable even without a headline ROC AUC.[1][6][8]

The other key baseline point is that horizon matters as much as metric.[1][4][5][6][8]

Current systems can reasonably be divided into three prediction horizon categories:

- **Short-horizon predictors** — anticipate events over roughly the next week or next few assessments
- **Medium-horizon predictors** — use the first 60 days of treatment to stratify later survival
- **Long-horizon predictors** — estimate probabilities of death or progression at intervals ranging from 6 months to 3 years

[1][4][5][6][8]

---

## Baseline Conclusion for Future AI Comparison

If the goal is to establish a sober baseline against which future, faster, and more powerful AI systems should be compared, the strongest current benchmark is not the most eye-catching single-institution AUC but the best externally validated, multicohort, clinically usable performance.[1][2][5][8][9]

On that standard, today's broad and reasonably validated oncology outcome predictors usually sit around:

| Task | Metric | Approximate Range |
|------|--------|-------------------|
| Survival (externally validated, multicohort) | Time-dependent AUC / C-index | 0.72–0.78 |
| Rare-event short-term discontinuation (phase III trial data) | AUPRC | ~0.23 |
| 7-day hospitalization (passive smartphone data) | AUC | ~0.88 |
| Imaging-based binary response (small retrospective) | AUC | >0.90 (narrow, single-site) |

[1][2][5][8][9]

Smaller or narrower proof-of-concept studies can report much higher values, including AUCs above 0.9 for irAE detection or imaging-based binary response classification, but those results generally come with weaker external validation, smaller cohorts, narrower disease settings, or more weakly standardized endpoints.[4][9]

The most honest present-day baseline is that current AI can already generate useful patient-level oncology forecasts, especially for risk stratification and short-term monitoring, but it still predicts broad outcome windows far better than it predicts exact event timing, and it usually works best when the endpoint is narrow, the input modality is clean, and the validation setting is similar to training.[1][2][4][5][6][8]

Future systems should therefore be judged not just on whether they produce a higher AUC or C-index, but on whether they can do **all** of the following at once:

1. Maintain performance under true external validation
2. Remain calibrated across tumor types and treatment regimens
3. Predict clinically actionable timing rather than only rank order
4. Show prospective evidence that acting on the prediction improves patient outcomes or trial efficiency

[1][6][7][8]

---

## Open Questions and Limitations

The literature is still limited by sparse prospective deployment, inconsistent reporting of calibration, frequent reliance on retrospective cohorts, and a mismatch between highly controlled trial datasets and the noisier clinical environments where these systems would actually run.[1][2][7][8]

It is also still uncommon for studies to predict the exact time to an adverse event or death as a well-calibrated individualized forecast; most systems instead output category membership, risk scores, hazard separation, or probability of an endpoint within a broad window.[1][4][5][6][8]

Finally, while 2025–2026 papers show movement toward contrastive learning, multimodal fusion, and trial-enrichment use cases, the field is still dominated by relatively conventional supervised models and retrospective validation rather than by prospectively embedded AI copilots for oncology trials.[1][7][8]

### Key Gaps Identified for Future Physical AI Trial Design

| Gap | Implication for New Trial |
|-----|---------------------------|
| Lack of prospective AI deployment | New trial should integrate AI predictions in real time |
| Poor exact event-time calibration | Trial design must define actionable prediction horizons clearly |
| Endpoint heterogeneity | Narrow, pre-specified primary endpoints are preferred |
| External validation inconsistency | Multi-site, multi-tumor enrollment required from inception |
| No prospective outcome-improvement evidence | Trial must include an intervention arm based on AI predictions |

---

*Preceded by: `chunk_01_baseline_and_prediction_domains.md`*
*References: `chunk_03_bibtex_references.md`*
