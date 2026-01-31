# Generative AI for Physical Oncology Systems: Limitations

*Critical constraints and mitigation strategies for clinical trial deployment (October 2025 - January 2026)*

---

## 1. Safety and Reliability Constraints

### Hallucination in Action Space

**Critical Limitation**: Generative models can produce physically invalid or dangerous action sequences that appear plausible.

**Manifestations in Oncology Robotics**:

| Failure Mode | Risk Level | Example |
|--------------|------------|---------|
| Trajectory interpolation through tissue | Critical | Diffusion model generates smooth path that traverses critical anatomy |
| Overconfident grasp predictions | High | VLA model predicts stable grasp on slippery tissue |
| Phantom object manipulation | Medium | Model hallucinates instrument presence and attempts interaction |

**Mitigation Strategies**:

```python
# Safety-constrained generation with physics validation
from safety import PhysicsValidator, AnatomyChecker

validator = PhysicsValidator(collision_mesh=patient_anatomy)
anatomy_checker = AnatomyChecker(critical_structures=["aorta", "vena_cava", "ureter"])

# Generate with rejection sampling
valid_trajectory = None
for attempt in range(max_attempts):
    candidate = diffusion_policy.sample(observation)

    if validator.is_collision_free(candidate):
        if anatomy_checker.respects_margins(candidate, margin_mm=5):
            valid_trajectory = candidate
            break

if valid_trajectory is None:
    raise SafetyException("No valid trajectory found - human intervention required")
```

### Distributional Shift Under Rare Events

**Limitation**: Generative models trained on typical procedures fail catastrophically on rare intraoperative complications.

**Documented Failures (2025 Literature)**:
- Unexpected hemorrhage scenarios: Policy outputs frozen or erratic
- Atypical anatomy: VLA models misidentify structures
- Equipment malfunction: No learned recovery behaviors

**Required Safeguards**:
1. Real-time anomaly detection triggering human takeover
2. Explicit out-of-distribution detection before action execution
3. Conservative fallback policies for unrecognized scenarios

---

## 2. Computational Requirements

### Inference Latency Constraints

**Current State (January 2026)**:

| Model Type | Inference Time | Hardware Required | Clinical Feasibility |
|------------|----------------|-------------------|---------------------|
| GR00T N1.6 | 5ms @ 200Hz | NVIDIA L40/H100 | Feasible with edge deployment |
| Diffusion Policy (15 steps) | 50-100ms | RTX 4090+ | Marginal for real-time |
| Large VLM reasoning | 500ms-2s | Cloud or H100 | Planning only, not control |

**Bottlenecks for Oncology Deployment**:

1. **Diffusion models**: 15-step denoising is too slow for reactive control during active procedures
2. **VLM reasoning**: Cannot be in the control loop for time-critical actions
3. **Memory footprint**: Full VLA models require 20+ GB VRAM

**Architectural Solutions**:

```python
# Hierarchical architecture to manage latency
class OncologyController:
    def __init__(self):
        # Slow: High-level reasoning (VLM) - runs asynchronously
        self.planner = AsyncVLMPlanner(model="cosmos-reason-2")

        # Medium: Trajectory generation - runs at 10Hz
        self.trajectory_gen = DistilledDiffusion(steps=4)  # Distilled from 15-step

        # Fast: Reactive control - runs at 200Hz
        self.reactive_policy = LightweightMLP(latency_ms=2)

    def control_loop(self):
        while True:
            # Fast loop: reactive adjustments
            action = self.reactive_policy(observation)

            # Medium loop: trajectory updates (async)
            if self.trajectory_gen.has_new_plan():
                self.reactive_policy.update_target(self.trajectory_gen.get_plan())

            # Slow loop: replanning on significant scene changes
            if scene_changed_significantly():
                self.planner.request_replan_async(observation)
```

### Hardware Access Barriers

**Limitation**: State-of-the-art generative models require hardware that many clinical trial sites lack.

| Requirement | Typical Academic Medical Center | Gap |
|-------------|--------------------------------|-----|
| NVIDIA H100/L40 GPU | Often unavailable | Critical for GR00T, Cosmos |
| Isaac Sim workstation | Rare | Required for ORBIT-Surgical |
| Real-time networking | Variable | Edge deployment challenges |

---

## 3. Training Data Limitations

### Oncology-Specific Data Scarcity

**Core Problem**: Generative models are data-hungry, but oncology surgical data is:
- **Sparse**: Rare procedures (e.g., Whipple, pelvic exenteration) have limited recordings
- **Heterogeneous**: High variance in tumor presentation, patient anatomy, surgeon technique
- **Protected**: HIPAA/GDPR constraints limit data sharing

**Quantified Gap**:

| Data Type | Available (2025) | Required for Robust Policy | Gap Factor |
|-----------|------------------|---------------------------|------------|
| General surgical video | 100K+ hours | Sufficient | 1x |
| Oncology-specific procedures | ~5K hours | 50K+ hours | 10x |
| Robotic oncology demonstrations | ~500 hours | 10K+ hours | 20x |
| Rare complication scenarios | <100 hours | 1K+ hours | 10x+ |

### Synthetic Data Quality Ceiling

**Limitation**: While Cosmos and Isaac Lab can generate synthetic data, the sim2real gap remains significant for soft tissue.

**Unresolved Challenges**:
1. **Tissue deformation**: Synthetic models underestimate variability in tumor tissue properties
2. **Bleeding simulation**: Current physics engines poorly model blood flow and coagulation
3. **Tactile feedback**: Synthetic haptic data does not match real tissue palpation

```python
# Sim2real gap quantification for oncology
from evaluation import Sim2RealGap

gap_analysis = Sim2RealGap()

results = gap_analysis.evaluate(
    sim_policy=trained_in_isaac,
    real_environment=dvrk_testbench,
    tasks=["needle_insertion", "tissue_retraction", "suturing"]
)

# Typical results (2025):
# needle_insertion: 15% performance drop
# tissue_retraction: 35% performance drop (soft tissue gap)
# suturing: 25% performance drop
```

---

## 4. Generalization Failures

### Cross-Patient Variability

**Limitation**: Models trained on population data fail to account for individual anatomical variations.

**Documented Failure Cases**:
- Obese patients: Instrument reach and visibility differ significantly
- Pediatric oncology: Adult-trained models inappropriate
- Prior surgery: Altered anatomy from previous procedures not in training distribution

### Cross-Procedure Transfer

**Limitation**: VLA models do not generalize across procedure types as well as language understanding suggests.

| Training Procedure | Test Procedure | Success Rate Drop |
|--------------------|----------------|-------------------|
| Cholecystectomy | Nephrectomy | 40-60% |
| Prostatectomy | Hysterectomy | 30-50% |
| Lung biopsy | Liver biopsy | 20-30% |

**Root Cause**: Low-level motor policies are procedure-specific despite high-level semantic similarity.

---

## 5. Interpretability and Verification

### Black-Box Decision Making

**Critical Limitation for FDA Approval**: Generative models cannot explain why they produced a specific trajectory.

**Regulatory Implications**:
- Pre-market submissions require mechanistic explanation of device behavior
- Post-market surveillance needs traceable decision logs
- Adverse event investigation requires causal analysis

**Current State of Explainability**:

| Method | Applicability | Regulatory Acceptance |
|--------|---------------|----------------------|
| Attention visualization | VLA models | Insufficient alone |
| Gradient-based saliency | Diffusion policies | Research only |
| Concept bottleneck | Limited availability | Promising |
| Formal verification | Intractable for generative models | Required for safety-critical |

### Verification Challenges

**Limitation**: Traditional software verification methods do not apply to learned generative models.

```python
# The verification problem for generative policies
def verify_safety(policy, specification):
    """
    INTRACTABLE for generative models:
    - Infinite input space (continuous observations)
    - Stochastic outputs (diffusion sampling)
    - Emergent behaviors not in training data
    """
    # Cannot exhaustively verify
    # Statistical testing provides probabilistic bounds only

    # Best current practice: extensive simulation testing
    results = monte_carlo_safety_test(
        policy=policy,
        num_scenarios=100000,
        specification=specification
    )

    # Returns statistical confidence, not guarantee
    return results.failure_rate, results.confidence_interval
```

---

## 6. Temporal Consistency

### Action Sequence Coherence

**Limitation**: Generative models (especially diffusion) can produce temporally inconsistent action sequences.

**Manifestations**:
- Jittery trajectories from independent frame-by-frame generation
- Sudden direction reversals mid-procedure
- Inconsistent tool orientation across timesteps

**Impact on Oncology Procedures**:
- Suturing: Interrupted needle paths
- Dissection: Erratic tissue manipulation
- Retraction: Variable force application

**Mitigation (Partial)**:

```python
# Temporal smoothing for diffusion policies
from temporal import CausalSmoother, ActionChunking

# Action chunking: generate 8-16 steps at once
policy = DiffusionPolicy(action_chunk_size=16)

# Causal smoothing: low-pass filter on action sequences
smoother = CausalSmoother(
    cutoff_hz=5.0,  # Remove high-frequency jitter
    preserve_onset=True  # Maintain responsiveness to new commands
)

raw_actions = policy.sample(observation)
smooth_actions = smoother(raw_actions)
```

---

## 7. Failure Mode Unpredictability

### Novel Failure Types

**Limitation**: Generative models exhibit failure modes not seen in traditional control systems.

**Documented Novel Failures (2025)**:
1. **Mode collapse in action space**: Model repeatedly outputs single trajectory regardless of observation
2. **Adversarial sensitivity**: Small input perturbations cause large action changes
3. **Reward hacking in fine-tuning**: Model finds unintended shortcuts that satisfy reward but not intent
4. **Cascading errors**: Small early mistakes compound into catastrophic failures

### Detection Challenges

**Limitation**: Failures may not be apparent until physically manifested.

```python
# Failure detection limitations
class GenerativeFailureDetector:
    """
    LIMITATION: Can only detect some failure modes

    Detectable:
    - Out-of-distribution inputs (via density estimation)
    - High action variance (via ensemble disagreement)
    - Constraint violations (via explicit checking)

    NOT reliably detectable:
    - Semantically incorrect but plausible actions
    - Subtle trajectory suboptimality
    - Intent misalignment
    """

    def __init__(self):
        self.ood_detector = EnsembleOOD()
        self.variance_monitor = ActionVarianceMonitor()
        self.constraint_checker = PhysicsConstraints()

    def check(self, observation, action):
        alerts = []

        if self.ood_detector.is_ood(observation):
            alerts.append("OOD input detected")

        if self.variance_monitor.high_variance(action):
            alerts.append("High action uncertainty")

        if not self.constraint_checker.satisfies(action):
            alerts.append("Physics constraint violation")

        # NOTE: Many failure modes will NOT trigger alerts
        return alerts
```

---

## 8. Integration Complexity

### Multi-System Coordination

**Limitation**: Generative models must integrate with existing clinical infrastructure, creating complex failure points.

**Integration Challenges**:

| System | Interface Challenge | Failure Risk |
|--------|--------------------|--------------|
| Hospital EMR | Data format incompatibility | Patient ID mismatch |
| Imaging (DICOM) | Real-time streaming latency | Stale anatomical data |
| Surgical robot (dVRK) | Control frequency mismatch | Jerky motion |
| Safety monitoring | Alert propagation delay | Delayed intervention |

### Protocol Compliance

**Limitation**: Generative models may violate clinical trial protocols in subtle ways.

- Procedure timing requirements
- Documentation obligations
- Consent verification
- Deviation reporting

---

## 9. Regulatory Pathway Uncertainty

### FDA Classification Ambiguity

**Limitation**: No established regulatory pathway for generative AI in surgical robotics.

**Current Uncertainty**:
- Likely Class III (PMA) for autonomous functions
- Continuous learning systems: No approved precedent
- Generative model updates: Unclear change control requirements

### Liability Framework

**Limitation**: Unclear liability allocation for generative model decisions.

- Manufacturer vs. institution vs. surgeon responsibility
- Training data provenance requirements
- Model versioning and audit trails

---

## Summary: Critical Limitations Requiring Mitigation

| Limitation | Severity | Mitigation Maturity | Recommendation |
|------------|----------|--------------------|--------------|
| Action hallucination | Critical | Partial | Physics validation mandatory |
| Computational requirements | High | Improving | Hierarchical architecture |
| Data scarcity | High | Partial | Synthetic + transfer learning |
| Generalization failures | High | Low | Patient-specific fine-tuning |
| Interpretability | Critical | Low | Regulatory engagement early |
| Temporal inconsistency | Medium | Good | Action chunking, smoothing |
| Unpredictable failures | High | Low | Extensive simulation testing |
| Integration complexity | Medium | Medium | Standards-based interfaces |

---

## Recommended Risk Mitigations

1. **Never deploy generative policies without physics-based safety constraints**
2. **Maintain human-in-the-loop for all irreversible actions**
3. **Implement comprehensive out-of-distribution detection**
4. **Use hierarchical architectures to manage latency-reliability tradeoffs**
5. **Plan for 10x more training data than initially estimated**
6. **Engage regulatory consultants before significant development investment**

---

*References: FDA Guidance on AI/ML-based SaMD (2024), NVIDIA Isaac Safety Documentation (2025), Surgical Robotics Safety Standards (IEC 80601-2-77)*
