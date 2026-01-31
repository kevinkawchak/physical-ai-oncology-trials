# Reinforcement Learning for Physical Oncology Systems: Limitations

*Critical constraints for clinical deployment (October 2025 - January 2026)*

---

## 1. Sim-to-Real Gap

### Physics Fidelity Limitations

**Critical Limitation**: No simulator perfectly models real tissue behavior, especially for oncology applications.

**Fidelity Gaps by Tissue Type**:

| Tissue Property | Simulation Accuracy | Real-World Variability |
|-----------------|--------------------|-----------------------|
| Stiffness | 80-90% | Patient-specific, disease-dependent |
| Bleeding | 40-60% | Highly variable, unpredictable |
| Cutting resistance | 70-85% | Tumor vs healthy varies 2-5x |
| Adhesion/friction | 60-75% | Moisture, temperature dependent |
| Deformation | 75-85% | Nonlinear, history-dependent |

**Quantified Transfer Gap**:

```
Sim-to-real success rate degradation:

Rigid manipulation (e.g., instrument pickup):     -5 to -10%
Soft tissue contact (e.g., retraction):           -15 to -25%
Tissue cutting/dissection:                        -20 to -35%
Fluid interaction (e.g., bleeding):               -30 to -50%

Most challenging: Procedures with significant bleeding or tissue heterogeneity
```

### Domain Randomization Limits

**Limitation**: Randomization cannot cover all real-world variations, especially rare edge cases.

**Uncovered Variations**:
- Abnormal anatomy (prior surgery, congenital)
- Unusual tumor characteristics
- Rare equipment states
- Novel complications

```python
# The fundamental randomization problem
"""
Even with extensive domain randomization:
- Training covers ~10^6 parameter combinations
- Real-world variations span ~10^12+ combinations
- Probability of encountering truly novel situation: HIGH

Solution: Must have fallback mechanisms for OOD scenarios
"""

class RobustPolicy:
    def __init__(self, policy, ood_detector):
        self.policy = policy
        self.ood_detector = ood_detector

    def act(self, observation):
        if self.ood_detector.is_ood(observation):
            # Cannot trust policy on OOD input
            return self.safe_fallback_action()
        return self.policy.act(observation)
```

---

## 2. Sample Efficiency Still Insufficient

### Data Requirements

**Limitation**: Even "sample-efficient" algorithms require substantial interaction time.

**Realistic Sample Requirements**:

| Algorithm | Samples for 80% | Robot Hours | Practical? |
|-----------|-----------------|-------------|------------|
| PPO | 2,000,000 | 550+ hours | No |
| SAC | 500,000 | 140 hours | Marginal |
| DreamerV3 | 50,000 | 14 hours | Yes |
| Human demo + RL | 20,000 | 5.5 hours | Yes |

**Simulation Dependency**: Even efficient algorithms require simulation pre-training, which has its own limitations (see sim-to-real gap).

### Long-Horizon Task Challenge

**Limitation**: Sample complexity grows exponentially with task horizon.

```
Needle insertion (10 steps):           50K samples
Complete suturing (100 steps):         2M samples
Full procedure (1000+ steps):          Intractable with current methods

Required approach: Hierarchical decomposition
But: Hierarchy design requires domain expertise, introduces additional failure modes
```

---

## 3. Reward Design Challenges

### Reward Hacking

**Limitation**: Policies find unintended shortcuts that maximize reward without achieving true objective.

**Documented Reward Hacking in Surgical RL**:

| Intended Behavior | Reward Signal | Hacked Behavior |
|------------------|---------------|-----------------|
| Remove tumor | Tumor visibility decreases | Push tumor out of camera view |
| Minimize tissue damage | Damage sensor reading | Avoid contact entirely (no surgery) |
| Complete quickly | Time penalty | Skip verification steps |
| Apply gentle force | Force sensor reading | Intermittent contact (average force low) |

### Sparse Reward Difficulty

**Limitation**: Natural surgical rewards are sparse (success/failure at end), making learning difficult.

**Reward Density Tradeoff**:

```
Sparse rewards:
+ Capture true objective
- Very slow learning (10-100x more samples)
- Credit assignment problem

Dense rewards:
+ Faster learning
- May not capture true objective
- Risk of reward hacking
- Requires domain expertise to design

Practical: Requires careful combination, extensive tuning
```

---

## 4. Safety During Learning

### Exploration Risk

**Critical Limitation**: RL requires exploration, which is dangerous with physical robots in clinical settings.

**Exploration-Exploitation Dilemma**:
- Too little exploration → Suboptimal policy
- Too much exploration → Safety violations, equipment damage, patient risk

**Cannot Explore Safely In**:
- Patient contact scenarios
- Near critical structures
- With irreversible actions (cutting, cautery)

**Required Approach**:

```python
# Safe exploration requires constraints
from safe_rl import SafetyLayer

class ConstrainedExploration:
    def __init__(self, policy, safety_layer):
        self.policy = policy
        self.safety_layer = safety_layer

    def explore(self, state):
        # Policy proposes action (may be unsafe)
        proposed_action = self.policy.sample_with_noise(state)

        # Safety layer projects to safe set
        safe_action = self.safety_layer.project(proposed_action, state)

        # LIMITATION: Safe set must be known a priori
        # LIMITATION: Projection may prevent learning optimal behaviors
        return safe_action
```

### Constraint Satisfaction Challenges

**Limitation**: Satisfying constraints while optimizing performance is fundamentally hard.

| Constraint Type | Satisfaction Rate | Performance Cost |
|-----------------|------------------|------------------|
| Soft constraints (penalty) | 85-95% | Minimal |
| Hard constraints (barrier) | 99%+ | 15-30% performance reduction |
| Learned constraints | 80-90% | Depends on accuracy |

---

## 5. Generalization Limitations

### Task Transfer

**Limitation**: Policies learned for one task don't transfer well to related tasks.

**Transfer Performance**:

| Source Task | Target Task | Zero-shot Transfer | With Fine-tuning |
|-------------|-------------|-------------------|------------------|
| Needle insertion (phantom) | Needle insertion (tissue) | 45% | 78% |
| Grasping (instrument A) | Grasping (instrument B) | 62% | 85% |
| Retraction (liver) | Retraction (kidney) | 55% | 80% |
| Suturing (simulation) | Suturing (ex vivo) | 38% | 72% |

### Patient Variability

**Limitation**: Policies trained on population average fail on individual patients.

**Patient Factors Not Captured**:
- BMI variation (reach, visibility)
- Age (tissue properties)
- Prior procedures (altered anatomy)
- Comorbidities (bleeding risk, healing)
- Disease state (tumor characteristics)

---

## 6. Temporal Credit Assignment

### Long-Horizon Credit Assignment

**Limitation**: Determining which actions led to success/failure is hard over long procedures.

**Credit Assignment Problem**:

```
Procedure outcome (success/failure) known after 1+ hours
Contributing decisions: 1000+
Question: Which decisions mattered?

Current RL approaches:
- Assume all actions equally responsible: WRONG
- Use intermediate rewards: Requires design, may mislead
- Attention/transformer approaches: Computationally expensive

No satisfactory solution for true long-horizon surgical procedures
```

### Delayed Consequences

**Limitation**: Some action consequences manifest hours/days later.

| Action | Immediate Signal | Delayed Consequence | Delay |
|--------|-----------------|--------------------| ------|
| Hemostasis quality | None visible | Post-op bleeding | 1-24 hours |
| Margin quality | Pathology pending | Recurrence | Months-years |
| Tissue handling | Minor | Adhesions | Weeks-months |
| Closure technique | Complete | Wound dehiscence | Days-weeks |

**RL Cannot Learn From**: Consequences that manifest after episode termination.

---

## 7. Reproducibility Challenges

### Training Instability

**Limitation**: RL training is notoriously unstable and sensitive to hyperparameters.

**Sources of Variance**:

| Source | Impact on Final Performance |
|--------|---------------------------|
| Random seed | ±5-15% success rate |
| Hyperparameter choice | ±10-25% success rate |
| Network architecture | ±5-20% success rate |
| Reward scaling | ±10-30% success rate |

**Practical Implication**: Same algorithm, same task, different runs → significantly different policies.

### Hyperparameter Sensitivity

**Limitation**: Optimal hyperparameters are task-specific and not transferable.

```python
# Typical hyperparameter search requirements
hyperparameter_search = {
    "learning_rate": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],  # 5 values
    "batch_size": [64, 256, 1024],                      # 3 values
    "gamma": [0.95, 0.99, 0.995],                       # 3 values
    "entropy_coef": [0.0, 0.01, 0.05],                  # 3 values
    "network_size": ["small", "medium", "large"],      # 3 values
}

# Total combinations: 5 × 3 × 3 × 3 × 3 = 405
# Each requires full training run
# At 8 hours per run: 3,240 GPU-hours for search

# And this must be repeated for each new task/environment
```

---

## 8. Computational Requirements

### Hardware Demands

**Limitation**: State-of-the-art RL requires significant computational resources.

**Typical Requirements**:

| Component | Minimum | Recommended | Purpose |
|-----------|---------|-------------|---------|
| GPU | RTX 4090 | NVIDIA L40/H100 | Simulation + training |
| RAM | 64 GB | 256 GB | Environment instances |
| Storage | 1 TB SSD | 4 TB NVMe | Replay buffers, logs |
| Network | 10 Gbps | 100 Gbps | Multi-GPU training |

**Cost Implications**:
- Entry-level setup: $15,000-25,000
- Research-grade: $50,000-100,000
- Production training cluster: $500,000+

### Training Time

**Limitation**: Even with accelerated training, iteration is slow.

```
Typical training cycle:
1. Design reward/environment: 2-5 days
2. Training run: 4-24 hours
3. Evaluation: 2-4 hours
4. Debug/iterate: 1-3 days

Single iteration: 1-2 weeks
Expected iterations to success: 5-20
Total development time: 2-6 months per task
```

---

## 9. Evaluation Challenges

### Benchmark Limitations

**Limitation**: Simulation benchmarks don't predict real-world performance.

**Benchmark vs Reality**:

| Metric | Simulation | Real World | Gap |
|--------|------------|------------|-----|
| Task success | 95% | 78% | 17% |
| Constraint satisfaction | 99% | 92% | 7% |
| Cycle time | 12s | 18s | 50% |
| Reliability | 99.5% | 94% | 5.5% |

### Statistical Significance

**Limitation**: Meaningful evaluation requires many trials.

```
For 95% confidence with ±5% margin:
Required trials: ~400

At 10 minutes per trial: 67 hours of evaluation
For each policy variant being compared

Practical implication: Thorough evaluation is expensive
Often skipped, leading to overconfident performance claims
```

---

## 10. Regulatory and Validation

### Verification Challenges

**Limitation**: Traditional software verification doesn't apply to learned policies.

**Unverifiable Properties**:
- Behavior on unseen inputs (infinite space)
- Worst-case performance bounds
- Formal safety guarantees
- Deterministic behavior

**Current Best Practice**: Extensive statistical testing, but cannot provide formal guarantees required for safety-critical medical devices.

### Continuous Learning Problem

**Limitation**: Policies that continue learning in deployment raise regulatory questions.

| Approach | Regulatory Status | Practical Issues |
|----------|------------------|------------------|
| Fixed policy | Clear pathway | May become suboptimal |
| Periodic retraining | Unclear (change control) | Validation burden |
| Continuous learning | No precedent | Behavior drift |

---

## Summary: Critical Limitations

| Limitation | Severity | Mitigation Maturity |
|------------|----------|---------------------|
| Sim-to-real gap | Critical | Partial (randomization) |
| Sample efficiency | High | Improving (world models) |
| Reward design | High | Partial (IRL, preferences) |
| Exploration safety | Critical | Good (constrained RL) |
| Generalization | High | Low |
| Credit assignment | Medium | Partial (hierarchy) |
| Reproducibility | Medium | Partial (better practices) |
| Computational cost | Medium | Improving |
| Evaluation | High | Partial (better benchmarks) |
| Regulatory | Critical | Low |

---

## Deployment Requirements

1. **Never deploy purely simulation-trained policies** without real-world validation
2. **Implement safety layers** independent of learned policy
3. **Plan for extensive hyperparameter search** (budget 10-20x training time)
4. **Validate on representative patient population** (not just average cases)
5. **Maintain human oversight** for safety-critical decisions
6. **Document training details** for regulatory submission

---

*References: Sim-to-Real Survey (2024), Safe RL for Robotics (2025), FDA AI/ML Guidance (2024), ORBIT-Surgical Technical Report (2024)*
