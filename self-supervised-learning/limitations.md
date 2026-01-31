# Self-Supervised Learning for Physical Oncology Systems: Limitations

*Constraints and challenges for clinical deployment (October 2025 - January 2026)*

---

## 1. Pre-Training Data Requirements

### Scale Dependency

**Limitation**: SSL methods require massive datasets to learn useful representations.

| Model Quality | Data Required | Availability for Surgery |
|--------------|---------------|-------------------------|
| Basic | 10K hours | Achievable |
| Good | 100K hours | Difficult |
| State-of-art | 1M+ hours | Not available |

**Surgical Data Scarcity**:
- Total surgical video worldwide: ~500K hours accessible
- High-quality, diverse: ~50K hours
- Oncology-specific: ~10K hours
- With rich metadata: ~2K hours

### Data Quality Issues

**Limitation**: Self-supervised learning amplifies data quality problems.

| Quality Issue | Impact on SSL | Mitigation |
|--------------|---------------|------------|
| Corrupted frames | Learns noise | Filtering |
| Camera artifacts | Spurious features | Augmentation |
| Inconsistent labeling | N/A (unsupervised) | - |
| Selection bias | Biased representations | Diverse sourcing |

---

## 2. Representation Interpretability

### Black-Box Features

**Limitation**: SSL representations are not interpretable, problematic for medical applications.

**Challenges**:
- Cannot explain why features activate
- No guarantee of clinically meaningful features
- May encode spurious correlations
- Difficult to debug failure cases

```python
# The interpretability problem
"""
SSL encoder output: 768-dimensional vector
Questions we cannot answer:
1. Which dimensions encode instrument pose?
2. Which encode tissue type?
3. Which are spurious (e.g., surgeon identity)?
4. Why did the model fail on this case?

Current solutions: Post-hoc analysis only
- Attention visualization (incomplete)
- Probing classifiers (indirect)
- TCAV-style concept analysis (labor-intensive)
"""
```

### Semantic Gap

**Limitation**: SSL features may not align with clinically meaningful concepts.

| SSL Cluster | Clinician Interpretation | Alignment |
|-------------|-------------------------|-----------|
| Cluster 1 | Mix of liver + spleen | Poor |
| Cluster 2 | Clean tissue types | Good |
| Cluster 3 | Lighting variations | Spurious |
| Cluster 4 | Camera motion types | Irrelevant |

---

## 3. Negative Transfer

### Domain Mismatch

**Limitation**: Pre-training on wrong domain can hurt performance.

**Transfer Performance by Domain**:

| Pre-training Domain | Target: Laparoscopic Surgery |
|--------------------|------------------------------|
| ImageNet | +12% vs random |
| General video | +8% vs random |
| Open surgery video | +18% vs random |
| Laparoscopic video | +28% vs random |
| Oncology laparoscopic | +35% vs random |

**Negative transfer observed when**:
- Pre-training distribution very different from target
- SSL objective misaligned with downstream task
- Pre-training data lower quality than target

### Task Mismatch

**Limitation**: SSL objective may not produce features useful for target task.

| SSL Objective | Good For | Poor For |
|--------------|----------|----------|
| Reconstruction | Low-level features | Semantic understanding |
| Contrastive | Instance discrimination | Dense prediction |
| Masked prediction | Context understanding | Fine localization |

---

## 4. Computational Overhead

### Pre-Training Costs

**Limitation**: SSL pre-training requires substantial resources.

| Model Scale | GPU Hours | Cost (Cloud) | Time |
|-------------|-----------|--------------|------|
| ViT-Small | 200 | $600 | 1 day |
| ViT-Base | 1,000 | $3,000 | 5 days |
| ViT-Large | 5,000 | $15,000 | 3 weeks |
| Foundation (Cosmos-scale) | 100,000+ | $300,000+ | Months |

### Inference Overhead

**Limitation**: Large SSL models have high inference costs.

| Model | Parameters | Inference Time | Memory |
|-------|------------|----------------|--------|
| ViT-S/14 | 22M | 5ms | 0.5GB |
| ViT-B/14 | 86M | 12ms | 1.5GB |
| ViT-L/14 | 304M | 35ms | 4GB |
| ViT-G/14 | 1.8B | 150ms | 20GB |

**For real-time surgery**: Only smaller models viable (<20ms)

---

## 5. Evaluation Challenges

### Proxy Task Performance

**Limitation**: SSL success on pre-text task doesn't guarantee downstream success.

**Correlation Analysis**:

| Metric | Correlation with Downstream |
|--------|---------------------------|
| Pre-text loss | 0.45 |
| Linear probe accuracy | 0.72 |
| kNN accuracy | 0.68 |
| Transfer to target task | 1.00 (ground truth) |

**Problem**: Must fully fine-tune and evaluate to know if pre-training worked.

### Benchmark Limitations

**Limitation**: Standard SSL benchmarks don't reflect surgical reality.

| Benchmark | Realistic for Surgery? | Issues |
|-----------|----------------------|--------|
| ImageNet linear probe | No | Wrong domain |
| Surgical phase recognition | Partial | Limited diversity |
| Instrument segmentation | Partial | Clean lab data |
| Real clinical transfer | Yes | Expensive to evaluate |

---

## 6. Collapse and Failure Modes

### Representation Collapse

**Limitation**: SSL training can collapse to trivial solutions.

**Collapse Types**:

| Type | Symptom | Detection |
|------|---------|-----------|
| Complete collapse | All outputs identical | Easy (zero variance) |
| Dimensional collapse | Features span subspace | Moderate (rank analysis) |
| Cluster collapse | Few modes only | Hard (distribution analysis) |

**Prevention Required**:
```python
# Must carefully design SSL to prevent collapse
ssl_config = {
    # Architectural prevention
    "projector_hidden_dim": 4096,
    "predictor": True,

    # Training prevention
    "weight_decay": 0.04,
    "batch_size": 4096,
    "temperature": 0.1,

    # Monitoring
    "track_feature_variance": True,
    "track_effective_rank": True
}
```

### Shortcut Learning

**Limitation**: SSL may learn shortcuts instead of meaningful features.

**Common Shortcuts in Surgical Video**:
- Timestamp/watermarks
- Consistent camera position
- Surgeon identification cues
- OR environment (not tissue)

---

## 7. Temporal Coherence Issues

### Video SSL Challenges

**Limitation**: Learning from video introduces temporal biases.

| Issue | Impact | Mitigation |
|-------|--------|------------|
| Static camera | Position features dominate | Random cropping |
| Slow motion | Trivial temporal prediction | Speed augmentation |
| Repetitive actions | Cycle detection shortcut | Random sampling |
| Scene cuts | Discontinuous learning | Segment detection |

### Long-Range Dependencies

**Limitation**: Most SSL methods operate on short clips, missing procedure-level patterns.

| Time Scale | SSL Capture | Missed Information |
|------------|-------------|-------------------|
| Frames (ms) | Motion | - |
| Clips (s) | Actions | - |
| Segments (min) | Partial | Phase transitions |
| Procedures (hr) | Poor | Global planning |

---

## 8. Multi-Modal Integration Challenges

### Modality Alignment

**Limitation**: Aligning vision, language, and robot states is difficult.

**Alignment Challenges**:

| Modality Pair | Alignment Difficulty | Notes |
|--------------|---------------------|-------|
| Image-text | Medium | Semantic gap |
| Image-depth | Easy | Geometric |
| Video-action | Hard | Temporal offset |
| Vision-proprioception | Hard | Different rates |

### Missing Modalities

**Limitation**: Real deployments may lack modalities used in pre-training.

```python
# Pre-training vs deployment mismatch
pretraining_modalities = ["rgb", "depth", "force", "audio"]
deployment_modalities = ["rgb"]  # Often only this available

# Performance degradation:
# Full modalities: 92%
# RGB only: 74%  (-18%)
```

---

## 9. Fine-Tuning Sensitivity

### Catastrophic Forgetting

**Limitation**: Fine-tuning can destroy pre-trained representations.

**Forgetting Rate by Fine-tuning Strategy**:

| Strategy | Downstream Acc | General Repr Quality |
|----------|---------------|---------------------|
| Full fine-tune | 91% | 45% (degraded) |
| Last layer only | 82% | 95% (preserved) |
| LoRA | 88% | 88% (balanced) |
| Progressive unfreeze | 89% | 75% (moderate) |

### Optimal Fine-Tuning Unknown

**Limitation**: Best fine-tuning strategy is task-dependent.

| Task Type | Best Strategy |
|-----------|--------------|
| Similar to pre-training | Full fine-tune |
| Different but related | Progressive unfreeze |
| Very different | Feature extraction + new head |
| Few-shot | Freeze backbone |

---

## 10. Regulatory and Validation

### Pre-Training Provenance

**Limitation**: SSL models trained on unknown data create regulatory challenges.

**Required Documentation**:
- Training data sources and licenses
- Data quality assurance
- Representation validation
- Bias assessment

**Current Gap**: Most SSL models lack sufficient documentation for medical device submission.

### Reproducibility

**Limitation**: SSL training is highly sensitive to hyperparameters.

| Factor | Variance in Final Performance |
|--------|------------------------------|
| Random seed | ±3-5% |
| Batch size | ±5-10% |
| Learning rate | ±5-15% |
| Augmentation | ±3-8% |
| Architecture | ±5-20% |

---

## Summary: Critical Limitations

| Limitation | Severity | Mitigation Maturity |
|------------|----------|---------------------|
| Data scale requirements | High | Low |
| Interpretability | Critical | Low |
| Negative transfer | Medium | Medium |
| Computational cost | Medium | Improving |
| Collapse risk | Medium | Good |
| Temporal modeling | Medium | Improving |
| Multi-modal alignment | Medium | Low |
| Catastrophic forgetting | Medium | Good |
| Regulatory clarity | Critical | Low |

---

## Deployment Recommendations

1. **Validate pre-training domain match** before using SSL models
2. **Monitor for representation collapse** during training
3. **Use appropriate fine-tuning strategy** for your data regime
4. **Document data provenance** for regulatory submission
5. **Evaluate on realistic surgical benchmarks**, not just proxies
6. **Plan for interpretability analysis** of learned representations

---

*References: SSL Survey (2024), Surgical AI Regulation (2025), Foundation Model Documentation (2025)*
