# Supervised Learning for Physical Oncology Systems: Limitations

*Constraints for clinical deployment (October 2025 - January 2026)*

---

## 1. Annotation Requirements

### Label Cost and Expertise

**Critical Limitation**: Medical annotation requires expensive expert time.

| Task | Time per Sample | Expert Required | Cost per 1K |
|------|-----------------|-----------------|-------------|
| Image classification | 5 seconds | Nurse | $50 |
| Bounding box | 30 seconds | Technician | $300 |
| Semantic segmentation | 10 minutes | Surgeon | $5,000 |
| Instance segmentation | 20 minutes | Surgeon | $10,000 |
| Temporal annotation | 2 hours | Surgeon | $50,000 |

### Annotation Quality Issues

**Limitation**: Human annotation is inherently inconsistent.

| Source | Inter-rater Agreement |
|--------|----------------------|
| Instrument detection | 92% (good) |
| Phase boundaries | 78% (moderate) |
| Tumor margins | 65% (poor) |
| Skill assessment | 72% (moderate) |

```python
# The annotation disagreement problem
"""
3 surgeons annotate same procedure:

Surgeon A: "Dissection phase ends at 12:34"
Surgeon B: "Dissection phase ends at 12:41"
Surgeon C: "Dissection phase ends at 12:38"

Span: 7 seconds
At 30 FPS: 210 frames of disagreement
Model cannot learn correct answer - ground truth is ambiguous
"""
```

---

## 2. Distribution Shift

### Domain Gap

**Limitation**: Models trained on one distribution fail on shifted data.

**Performance Degradation by Shift Type**:

| Shift Source | Accuracy Drop | Recovery Method |
|--------------|--------------|-----------------|
| New hospital | -15 to -25% | Domain adaptation |
| Different camera | -10 to -20% | Fine-tuning |
| New surgeon style | -5 to -15% | Augmentation |
| Patient variation | -10 to -20% | Diverse training |
| Lighting change | -5 to -15% | Augmentation |

### Long-Tail Distribution

**Limitation**: Rare events are underrepresented in training data.

| Event Category | Frequency | Model Performance |
|----------------|-----------|-------------------|
| Common phases | 80% | 95% accuracy |
| Uncommon phases | 15% | 78% accuracy |
| Rare complications | 4% | 62% accuracy |
| Very rare events | 1% | 45% accuracy |

---

## 3. Generalization Failures

### Out-of-Distribution Inputs

**Limitation**: Models fail silently on inputs outside training distribution.

**Failure Scenarios**:

| Scenario | Training Coverage | Test Performance |
|----------|-------------------|------------------|
| Standard cases | High | 95% |
| Obese patients | Medium | 78% |
| Pediatric | Low | 62% |
| Revision surgery | Low | 58% |
| Rare anatomy | Very low | 45% |

### Spurious Correlations

**Limitation**: Models learn shortcuts instead of true features.

**Documented Shortcuts**:
- Hospital watermarks → Procedure type
- Instrument brand → Surgeon skill
- Video quality → Outcome
- Time of day → Complication rate

```python
# Detecting spurious correlations
from analysis import SpuriousCorrelationDetector

detector = SpuriousCorrelationDetector(model, dataset)
correlations = detector.find_spurious()

# Example findings:
# - Model predicts "expert" when seeing blue graspers (brand correlation)
# - Model predicts "complication" for dark images (lighting artifact)
# - Model predicts "late phase" when clock visible in corner
```

---

## 4. Temporal Modeling Limitations

### Long-Range Dependencies

**Limitation**: Standard architectures struggle with long procedures.

| Procedure Length | Modeling Accuracy | Memory Required |
|-----------------|-------------------|-----------------|
| 10 min | 94% | 2 GB |
| 30 min | 88% | 8 GB |
| 1 hour | 81% | 32 GB |
| 3 hours | 72% | OOM |

### Causal Leakage

**Limitation**: Temporal models may use future information during training.

```python
# Causal leakage problem
"""
Training: Model sees entire video, can use future context
Deployment: Model only has past frames

Performance gap:
- With future context: 94% phase accuracy
- Causal (past only): 82% phase accuracy

Must train with causal masking, but this is often overlooked
"""
```

---

## 5. Class Imbalance

### Skewed Distributions

**Limitation**: Most surgical events follow extreme class imbalance.

**Typical Distribution**:

| Category | Percentage | Model Bias |
|----------|------------|------------|
| Normal operation | 95% | Over-predicted |
| Minor deviation | 4% | Under-predicted |
| Complication | 0.9% | Severely under-predicted |
| Critical event | 0.1% | Often missed |

### Imbalance Mitigation Limits

**Limitation**: Standard mitigation techniques have diminishing returns.

| Technique | Minority Class Recall | Majority Class Drop |
|-----------|----------------------|---------------------|
| No mitigation | 35% | 0% |
| Class weights | 58% | -5% |
| Oversampling | 62% | -8% |
| SMOTE | 55% | -3% |
| Focal loss | 68% | -10% |
| Ensemble | 72% | -12% |

---

## 6. Computational Constraints

### Training Costs

**Limitation**: Large supervised models require significant resources.

| Model | Training Time | GPU Memory | Dataset Size |
|-------|---------------|------------|--------------|
| ResNet-50 | 4 hours | 8 GB | 10K images |
| ViT-B | 24 hours | 16 GB | 100K images |
| Video Swin | 96 hours | 32 GB | 10K videos |
| Large ensemble | 1 week | 80 GB | Full dataset |

### Inference Latency

**Limitation**: Complex models may be too slow for real-time surgery.

| Model | Accuracy | Latency | Real-time? |
|-------|----------|---------|------------|
| MobileNet | 85% | 5ms | Yes |
| ResNet-50 | 91% | 15ms | Yes |
| ViT-B | 94% | 35ms | Marginal |
| Video Transformer | 96% | 100ms | No |

---

## 7. Calibration Issues

### Overconfidence

**Limitation**: Neural networks are typically overconfident in predictions.

**Calibration Metrics**:

| Model | ECE (lower better) | Reliability |
|-------|-------------------|-------------|
| ResNet (uncalibrated) | 0.15 | Poor |
| ResNet + temperature | 0.06 | Good |
| Ensemble | 0.04 | Very good |
| Bayesian NN | 0.03 | Excellent |

### Uncertainty Quantification

**Limitation**: Point estimates don't convey uncertainty.

```python
# The overconfidence problem
model_output = softmax([2.1, 0.3, 0.1])
# Returns: [0.85, 0.10, 0.05]

# Model claims 85% confidence, but:
# - May be OOD input
# - May be ambiguous case
# - Confidence doesn't reflect true probability

# Solution: Uncertainty estimation
from uncertainty import MCDropout
uncertainty_model = MCDropout(base_model, n_samples=20)
mean, std = uncertainty_model.predict_with_uncertainty(x)
# Now have uncertainty estimate
```

---

## 8. Interpretability Gaps

### Post-Hoc Explanations

**Limitation**: Most explanation methods are approximations.

| Method | Faithfulness | Computational Cost |
|--------|--------------|-------------------|
| Gradient-based | Low-Medium | Low |
| Attention | Medium | Low |
| SHAP | Medium-High | High |
| Integrated Gradients | High | Medium |
| Concept-based | High | High |

### Clinical Relevance

**Limitation**: Explanations may not map to clinical concepts.

```
Model explanation: "High activation in feature map 47 at position (128, 256)"
Clinician needs: "Model focused on the cystic artery near the clip site"

Translation is lossy and often impossible
```

---

## 9. Static Model Limitations

### No Continuous Learning

**Limitation**: Deployed models don't improve from new data.

**Consequences**:
- Performance degrades as procedures evolve
- New equipment not recognized
- Technique changes not captured
- Requires periodic retraining (regulatory burden)

### Version Management

**Limitation**: Multiple model versions create complexity.

| Challenge | Impact |
|-----------|--------|
| Which version at which site? | Inconsistent care |
| When to update? | Regulatory approval |
| Backward compatibility | Integration burden |
| Rollback procedures | Safety requirement |

---

## 10. Data Quality Dependencies

### Garbage In, Garbage Out

**Limitation**: Model quality bounded by data quality.

| Data Issue | Model Impact | Detection Difficulty |
|------------|--------------|---------------------|
| Label errors | Ceiling on accuracy | Medium |
| Selection bias | Poor generalization | Hard |
| Missing data | Biased predictions | Easy |
| Sensor artifacts | Spurious features | Medium |

### Dataset Biases

**Limitation**: Datasets reflect collection biases.

**Common Biases in Surgical Data**:
- Academic medical centers overrepresented
- Expert surgeons overrepresented
- Complex cases underrepresented
- Complications underreported

---

## Summary: Critical Limitations

| Limitation | Severity | Mitigation Maturity |
|------------|----------|---------------------|
| Annotation cost | High | Partial (active learning) |
| Distribution shift | Critical | Partial (adaptation) |
| Generalization | High | Partial (augmentation) |
| Temporal modeling | Medium | Improving |
| Class imbalance | High | Partial (techniques) |
| Calibration | Medium | Good |
| Interpretability | Medium | Improving |
| Static models | Medium | Low |

---

## Deployment Recommendations

1. **Validate on target domain** before deployment, not just held-out test set
2. **Implement calibration** for all classification models
3. **Monitor for distribution shift** continuously in production
4. **Plan for rare events** with specific detection mechanisms
5. **Budget for annotation** as ongoing operational cost
6. **Document biases** in training data for regulatory submission

---

*References: Medical AI Safety Guidelines (2025), FDA AI/ML SaMD Guidance (2024), Surgical ML Best Practices (2025)*
