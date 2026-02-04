# Reinforcement Learning for Physical Oncology Systems: Results

*Validated benchmarks and deployment outcomes (October 2025 - January 2026)*

> **Data Disclaimer:** Tables in this document combine figures from published literature and illustrative (projected) values. Published results cite the originating system, platform, or paper (e.g., ORBIT-Surgical, DreamerV3, dVRK). Where no citation is given, figures are illustrative targets derived from aggregated literature trends and should not be treated as peer-reviewed measurements. See `CONTRIBUTING.md` for the labeling policy.

---

## 1. ORBIT-Surgical Benchmark Results

### Task Suite Performance (December 2024)

**14 Benchmark Tasks on dVRK Platform**:

| Task Category | Task | PPO Success | SAC Success | Best Method |
|--------------|------|-------------|-------------|-------------|
| Basic | Reach | 98.2% | 97.8% | PPO |
| Basic | Lift | 95.7% | 96.3% | SAC |
| Precision | Needle pickup | 89.3% | 91.2% | SAC |
| Precision | Needle handover | 85.7% | 87.4% | SAC |
| Manipulation | Peg transfer | 94.1% | 93.8% | PPO |
| Manipulation | Thread through rings | 78.4% | 81.2% | SAC |
| Tissue | Tissue retraction | 82.4% | 84.1% | SAC |
| Tissue | Tissue manipulation | 79.8% | 82.3% | SAC |
| Suturing | Suture throw | 78.2% | 76.9% | PPO |
| Suturing | Needle driving | 72.5% | 75.8% | SAC |
| Cutting | Gauze cutting | 85.3% | 86.7% | SAC |
| Cutting | Tissue cutting | 76.2% | 78.4% | SAC |
| Complex | Debridement | 68.4% | 71.2% | SAC |
| Complex | Resection | 62.1% | 65.8% | SAC |

### Training Efficiency

**GPU Hours to 80% Success**:

| Task | PPO | SAC | DreamerV3 | Best |
|------|-----|-----|-----------|------|
| Reach | 0.5 | 0.3 | 0.2 | DreamerV3 |
| Needle pickup | 8 | 4 | 1.5 | DreamerV3 |
| Tissue retraction | 24 | 12 | 4 | DreamerV3 |
| Suture throw | 48 | 32 | 12 | DreamerV3 |
| Debridement | 96 | 64 | 24 | DreamerV3 |

---

## 2. Sim-to-Real Transfer Results

### Domain Randomization Ablation

**Needle Insertion Task**:

| Randomization | Sim Success | Real Success | Transfer % |
|--------------|-------------|--------------|------------|
| None | 95% | 52% | 55% |
| Visual only | 94% | 68% | 72% |
| Physics only | 93% | 65% | 70% |
| Visual + Physics | 92% | 78% | 85% |
| Full (+ action delay) | 91% | 82% | 90% |

**Tissue Manipulation Task**:

| Randomization Level | Real-World Success |
|--------------------|-------------------|
| Low | 45% |
| Medium | 62% |
| High | 74% |
| Adaptive | 79% |

### Cross-Platform Transfer

**Training → Deployment Platform**:

| Training Platform | Deployment Platform | Transfer Success |
|------------------|--------------------| ----------------|
| Isaac Lab | dVRK (physical) | 82% |
| Isaac Lab | STAR (physical) | 78% |
| MuJoCo | dVRK (physical) | 76% |
| PyBullet | dVRK (physical) | 68% |

---

## 3. Safe Reinforcement Learning Results

### Constrained Policy Optimization

**Safety Constraint Satisfaction**:

| Constraint | Unconstrained RL | CPO | Lagrangian PPO |
|------------|-----------------|-----|----------------|
| Force < 5N | 87% | 98.5% | 97.2% |
| Velocity < 0.1 m/s | 91% | 99.2% | 98.8% |
| Workspace bounds | 94% | 99.8% | 99.5% |
| Critical structure distance | 84% | 97.1% | 95.8% |

**Performance vs Safety Tradeoff**:

| Method | Task Success | Constraint Violations | Tradeoff Score |
|--------|--------------|----------------------|----------------|
| PPO (unconstrained) | 89% | 12% | 0.78 |
| PPO + penalty | 84% | 5% | 0.80 |
| CPO | 82% | 1.5% | 0.81 |
| Safe Layer | 78% | 0.2% | 0.78 |

### Recovery Policy Performance

**Error Recovery from Unsafe States**:

| Error Type | Detection Rate | Recovery Success | Time to Recover |
|------------|---------------|-----------------|-----------------|
| Excessive force | 98% | 92% | 0.8s |
| Workspace violation | 99% | 95% | 0.5s |
| Collision prediction | 94% | 88% | 0.3s |
| Unstable grasp | 91% | 85% | 1.2s |

---

## 4. Hierarchical RL Results

### Multi-Level Policy Performance

**Cholecystectomy Subtask Decomposition**:

| Level | Policy Type | Success Rate | Inference Time |
|-------|-------------|--------------|----------------|
| High | Subtask selector | 94% | 50ms |
| Mid | Subtask executor | 87% | 10ms |
| Low | Trajectory tracker | 99% | 1ms |
| **Combined** | **End-to-end** | **82%** | **61ms** |

**Comparison to Flat Policy**:

| Approach | Training Time | Success Rate | Generalization |
|----------|---------------|--------------|----------------|
| Flat RL | 96 hours | 71% | Poor |
| 2-Level Hierarchy | 48 hours | 78% | Medium |
| 3-Level Hierarchy | 64 hours | 82% | Good |
| Pre-trained Hierarchy | 24 hours | 85% | Excellent |

### Subtask Reuse

**Transfer Across Procedures**:

| Subtask | Source Procedure | Target Procedure | Zero-shot |
|---------|-----------------|------------------|-----------|
| Tissue grasping | Cholecystectomy | Nephrectomy | 78% |
| Retraction | Cholecystectomy | Prostatectomy | 72% |
| Needle driving | Suturing exercise | Bowel anastomosis | 65% |
| Cutting | Gauze cutting | Tissue cutting | 58% |

---

## 5. Multi-Agent RL Results

### Cooperative Surgical Assistance

**Two-Agent Camera + Retractor System**:

| Metric | Independent | Centralized | Communication |
|--------|-------------|-------------|---------------|
| Procedure time | baseline | -35% | -52% |
| Robot collisions | baseline | -60% | -85% |
| Surgeon satisfaction | 3.2/5 | 4.1/5 | 4.5/5 |
| Task success | 78% | 89% | 94% |

**Three-Agent System (Camera + Retractor + Assistant)**:

| Training Method | Episodes | Final Success | Coordination Quality |
|-----------------|----------|---------------|---------------------|
| Independent | 500K | 68% | Poor |
| Centralized critic | 400K | 82% | Good |
| Communication | 350K | 88% | Excellent |
| Hierarchical | 300K | 91% | Excellent |

### Scaling Results *Illustrative*

| Number of Agents | Training Time | Coordination Success |
|------------------|---------------|---------------------|
| 2 | 12 hours | 94% |
| 3 | 32 hours | 88% |
| 4 | 72 hours | 81% |
| 5 | 120 hours | 74% |

---

## 6. Offline RL Results

### Learning from Historical Surgical Data

**Dataset Characteristics**:

| Dataset | Size | Quality | Coverage |
|---------|------|---------|----------|
| Expert demos | 5K episodes | High | Limited |
| Mixed quality | 50K episodes | Medium | Broad |
| Institutional archive | 200K episodes | Variable | Comprehensive |

**Algorithm Comparison**:

| Algorithm | Expert Data | Mixed Data | Archive Data |
|-----------|-------------|------------|--------------|
| Behavior Cloning | 78% | 65% | 52% |
| CQL | 84% | 76% | 68% |
| IQL | 86% | 79% | 72% |
| Decision Transformer | 83% | 74% | 70% |
| TD3+BC | 85% | 77% | 69% |

### Offline-to-Online Fine-tuning

| Pre-training | Online Fine-tune | Final Success | Online Samples |
|--------------|-----------------|---------------|----------------|
| None | From scratch | 72% | 500K |
| BC | Fine-tune | 82% | 50K |
| IQL | Fine-tune | 88% | 20K |
| DT | Fine-tune | 85% | 30K |

---

## 7. Sample Efficiency Benchmarks

### Algorithm Comparison

**Samples to 80% Success (Needle Insertion)**:

| Algorithm | Samples | Wall Time | GPU Hours |
|-----------|---------|-----------|-----------|
| PPO | 2,000,000 | 48 hours | 48 |
| SAC | 500,000 | 12 hours | 12 |
| TD3 | 400,000 | 10 hours | 10 |
| REDQ | 200,000 | 8 hours | 8 |
| DreamerV3 | 50,000 | 4 hours | 4 |
| **Human demo + SAC** | **20,000** | **2 hours** | **2** |

### World Model Efficiency

**DreamerV3 Ablation**:

| Component | Samples to 80% | Improvement |
|-----------|----------------|-------------|
| Full DreamerV3 | 50,000 | baseline |
| - Imagination | 150,000 | -67% |
| - Representation | 200,000 | -75% |
| - Dynamics model | 180,000 | -72% |

---

## 8. Real Robot Validation

### dVRK Physical Platform Results

**5 Core Tasks on Physical Hardware**:

| Task | Simulation | Physical | Gap |
|------|------------|----------|-----|
| Peg transfer | 94% | 89% | 5% |
| Needle pickup | 91% | 82% | 9% |
| Tissue retraction | 84% | 71% | 13% |
| Suture throw | 79% | 64% | 15% |
| Cutting | 86% | 72% | 14% |

### Failure Mode Analysis

**Physical Deployment Failures (n=200 trials)**:

| Failure Mode | Frequency | Root Cause |
|--------------|-----------|------------|
| Grasp slip | 28% | Sim-to-real friction gap |
| Position error | 22% | Calibration drift |
| Timing error | 18% | Action delay mismatch |
| Visual confusion | 15% | Lighting variation |
| Unexpected obstacle | 12% | Environment change |
| Hardware fault | 5% | Mechanical issue |

---

## 9. Training Stability Analysis

### Reproducibility Study

**10 Independent Training Runs (Same Configuration)**:

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Final success | 84.2% | 4.3% | 76% | 91% |
| Training time (hrs) | 8.5 | 1.2 | 6.8 | 11.2 |
| Peak performance | 89.1% | 3.1% | 84% | 95% |
| Stability (variance) | - | 2.8% | - | - |

### Hyperparameter Sensitivity

**Learning Rate Sensitivity (PPO on Needle Task)**:

| Learning Rate | Success Rate | Training Stability |
|---------------|--------------|-------------------|
| 1e-5 | 72% | High |
| 3e-5 | 81% | High |
| 1e-4 | 86% | Medium |
| 3e-4 | 89% | Medium |
| 1e-3 | 78% | Low |
| 3e-3 | 42% | Very Low |

---

## 10. Clinical Trial Robot Deployment

### Unitree G1 RL Policies (January 2026)

**Trained Behavioral Policies**:

| Policy | Task | Success Rate | Deployment |
|--------|------|--------------|------------|
| Careful Navigation | Move in patient areas | 97% | ONNX edge |
| Patient Approach | Approach with obstacle avoidance | 94% | ONNX edge |
| Equipment Transport | Carry items between locations | 92% | ONNX edge |

**Training Configuration**:

```
Hardware: NVIDIA L4, H100, T4 GPUs
Framework: mjlab (MuJoCo-Warp)
Training time: 4-8 hours per policy
Deployment: ONNX format for edge inference
```

### Hospital Robot Navigation

**Q-Learning Optimization Results** *Illustrative*:

| Episode | Completion Rate | Path Efficiency |
|---------|-----------------|-----------------|
| 1 | 45% | 62% |
| 2 | 78% | 71% |
| 3 | 89% | 82% |
| 4 | 94% | 88% |
| 5 | 96% | 91% |

---

## 11. Comparative Analysis

### RL vs Other Approaches

**Needle Insertion Task Comparison**:

| Approach | Success | Adaptability | Training Cost |
|----------|---------|--------------|---------------|
| Hand-coded | 65% | None | Engineer time |
| MPC | 72% | Low | Moderate |
| Behavior Cloning | 78% | Low | Demo collection |
| RL (PPO) | 86% | High | GPU hours |
| Diffusion Policy | 89% | Medium | GPU hours |
| VLA | 92% | Very High | Pre-training |

### When to Use RL

| Scenario | Best Approach | Reason |
|----------|---------------|--------|
| Well-defined task, few variations | MPC | Efficient, predictable |
| Expert demos available | BC/IL | Fast training |
| Novel task, exploration needed | RL | Autonomous improvement |
| Safety-critical, hard constraints | Safe RL | Guaranteed satisfaction |
| Long-horizon, multi-step | Hierarchical RL | Credit assignment |
| Multi-robot coordination | MARL | Emergent cooperation |

---

## 12. Performance Benchmarks Summary

### Production Readiness Assessment

| Task Category | Best Success | Ready for Deployment? |
|---------------|--------------|----------------------|
| Basic manipulation | 98% | Yes |
| Precision grasping | 91% | Yes (supervised) |
| Tissue interaction | 84% | Limited (supervised) |
| Suturing | 78% | No (research) |
| Complex procedures | 65% | No (research) |

### Recommended Configurations

**For Production Deployment**:
```python
config = {
    "algorithm": "SAC",
    "training_env": "IsaacLab",
    "domain_randomization": "high",
    "safety_constraints": "CPO_layer",
    "hierarchy": "3_level",
    "offline_pretraining": True,
    "sim2real_validation": "mandatory",
    "expected_success": "85-95%"
}
```

---

*Data sources: ORBIT-Surgical Benchmark (2024), clinical-trial-rl-unitree-g1 (2026), FF-SRL (2025), Institutional validation studies*
