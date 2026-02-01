# Surgical Robotics Unification: Challenges

*Technical barriers to cross-platform surgical robot compatibility for oncology trials (January 2026)*

---

## Overview

Achieving interoperability between surgical robotic platforms requires addressing fundamental differences in hardware architectures, control interfaces, safety systems, and regulatory pathways. This document identifies the key challenges for multi-organization cooperation in oncology surgical robotics.

---

## 1. Hardware Architecture Incompatibilities

### Kinematic Chain Differences

**Challenge**: Surgical robots have fundamentally different kinematic structures.

| Robot | DoF | Architecture | Workspace | Primary Use |
|-------|-----|--------------|-----------|-------------|
| dVRK PSM | 7+1 | Cable-driven RCM | Cone, 30cm | MIS surgery |
| Franka Panda | 7+1 | Direct-drive | Sphere, 85cm | Lab automation |
| UR5e | 6+1 | Direct-drive | Sphere, 85cm | General purpose |
| STAR | 7 | Cable-driven | Limited | Soft tissue |
| Kinova Gen3 | 7+1 | Direct-drive | Sphere, 90cm | Collaborative |

**Impact**: Policies trained on one robot architecture cannot directly transfer to another.

```python
# Example: Action space mismatch

# dVRK PSM action (7 joints + gripper)
dvrk_action = np.array([
    0.1,    # outer_yaw
    -0.05,  # outer_pitch
    0.15,   # insertion (prismatic)
    0.02,   # outer_roll
    0.0,    # wrist_pitch
    0.1,    # wrist_yaw
    0.0,    # wrist_roll
    0.5     # gripper (0-1)
])

# Franka action (7 joints + gripper)
franka_action = np.array([
    0.0,    # joint1
    -0.785, # joint2
    0.0,    # joint3
    -2.356, # joint4
    0.0,    # joint5
    1.571,  # joint6
    0.785,  # joint7
    0.04    # gripper (meters)
])

# Different semantics, ranges, and coupling
```

**What Needs to Happen**:
1. Define task-space action representation (Cartesian + orientation)
2. Create inverse kinematics wrappers for each platform
3. Establish workspace intersection mapping
4. Develop platform-specific action scaling and limits

---

### Actuation System Differences

**Challenge**: Different actuation methods produce different dynamic behavior.

| System | Type | Characteristics | Control Implications |
|--------|------|-----------------|---------------------|
| Cable-driven | Tendon | Hysteresis, coupling, backlash | Requires compensation |
| Direct-drive | Motor | Precise, responsive | Standard control |
| Series elastic | SEA | Compliant, safe | Requires force control |
| Hydraulic | Fluid | High force, slow | Force-position hybrid |

**Impact**: Control gains, safety limits, and dynamic models differ fundamentally.

**What Needs to Happen**:
1. Characterize actuation dynamics per platform
2. Develop actuation-aware control wrappers
3. Create platform-specific safety margins
4. Build dynamic model libraries

---

## 2. Control Interface Fragmentation

### Communication Protocol Diversity

**Challenge**: No standard communication protocol for surgical robots.

| Robot | Primary Interface | Protocol | Latency | Update Rate |
|-------|-------------------|----------|---------|-------------|
| dVRK | cisst/SAW | ROS 1/2 | ~1ms | 1-2 kHz |
| Franka | libfranka | Real-time | <1ms | 1 kHz |
| UR | RTDE | TCP/UDP | ~2ms | 500 Hz |
| Kinova | Kortex API | gRPC | ~5ms | 1 kHz |
| Custom | Various | Various | Varies | Varies |

**What Needs to Happen**:
1. Create unified ROS 2 interface package
2. Implement protocol adapters with timing guarantees
3. Establish minimum latency requirements
4. Build communication health monitoring

---

### Control Mode Incompatibility

**Challenge**: Robots support different control modes.

| Robot | Position | Velocity | Torque | Impedance | Hybrid |
|-------|----------|----------|--------|-----------|--------|
| dVRK | ✓ | ✓ | Limited | Via SAW | No |
| Franka | ✓ | ✓ | ✓ | ✓ | ✓ |
| UR | ✓ | ✓ | Limited | External | No |
| Kinova | ✓ | ✓ | ✓ | ✓ | No |

**Impact**: Advanced control strategies (impedance, force) not portable across platforms.

```python
# Impedance control example - not universally available

# Franka (native impedance control)
franka.set_impedance(
    stiffness=np.array([600, 600, 600, 30, 30, 30]),  # N/m, Nm/rad
    damping_ratio=np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
)
franka.move_to_position(target_pose)

# UR5 (no native impedance - must implement in software)
def ur_impedance_control_loop(target_pose, stiffness, damping):
    while True:
        current_pose = ur.get_pose()
        current_velocity = ur.get_velocity()

        # Compute impedance force
        pose_error = target_pose - current_pose
        force = stiffness @ pose_error - damping @ current_velocity

        # Apply via force mode (if available)
        ur.force_mode(force)  # May not be available or accurate
```

**What Needs to Happen**:
1. Define minimum viable control mode subset
2. Implement software-based impedance for limited platforms
3. Create control mode fallback strategies
4. Document control mode limitations per platform

---

## 3. Sensing Capability Gaps

### Force/Torque Sensing

**Challenge**: Inconsistent force sensing capabilities.

| Robot | F/T Sensor | Location | Resolution | Noise |
|-------|------------|----------|------------|-------|
| dVRK | Indirect (motor current) | Joints | ~0.5N | High |
| Franka | 6-axis integrated | Wrist | 0.1N | Low |
| UR | External option | Tool | Varies | Varies |
| STAR | Custom | Tip | ~0.2N | Medium |

**Impact**: Force-based policies require platform-specific adaptation.

**What Needs to Happen**:
1. Standardize force sensing interface
2. Develop force estimation for sensorless robots
3. Create noise characterization and filtering
4. Establish force accuracy requirements

---

### Proprioceptive Sensing

**Challenge**: Different proprioceptive capabilities.

| Robot | Joint Position | Joint Velocity | Joint Torque | Temperature |
|-------|----------------|----------------|--------------|-------------|
| dVRK | Encoder | Derived | Current-based | No |
| Franka | Encoder | Encoder rate | Torque sensor | Yes |
| UR | Encoder | Derived | Current-based | Yes |

**What Needs to Happen**:
1. Define minimum sensing requirements
2. Implement missing sensor estimation
3. Create unified state observation format
4. Document sensing limitations

---

## 4. Safety System Heterogeneity

### Emergency Stop Mechanisms

**Challenge**: Different safety architectures.

| Robot | E-Stop Type | Response Time | Recovery | Power State |
|-------|-------------|---------------|----------|-------------|
| dVRK | Hardware relay | <10ms | Manual | Off |
| Franka | Category 1 | <10ms | Auto unlock | Brake |
| UR | Category 1 | <10ms | Resume | Brake |
| Custom | Varies | Varies | Varies | Varies |

**Impact**: Safety-critical behavior cannot be uniformly guaranteed.

**What Needs to Happen**:
1. Define minimum safety response requirements
2. Create unified safety interface abstraction
3. Implement platform-specific safety handlers
4. Establish safety testing protocols

---

### Collision Detection

**Challenge**: Varying collision detection capabilities.

| Robot | Detection Method | Sensitivity | Reaction |
|-------|------------------|-------------|----------|
| dVRK | None (open loop) | N/A | None |
| Franka | Torque monitoring | Configurable | Stop/reflex |
| UR | Force monitoring | Configurable | Stop/reduce |
| Kinova | Current monitoring | Medium | Stop |

**What Needs to Happen**:
1. Implement software-based collision detection
2. Create virtual force sensing for detection
3. Establish collision response protocols
4. Build collision testing benchmarks

---

## 5. Regulatory and Certification Barriers

### Medical Device Certification

**Challenge**: Different certification pathways and statuses.

| Robot | FDA Status | CE Mark | Clinical Use | Research Use |
|-------|------------|---------|--------------|--------------|
| da Vinci | 510(k) cleared | Yes | Yes | Limited |
| dVRK | Research only | N/A | No | Yes |
| Franka | General robot | Yes | Requires 510(k) | Yes |
| UR | General robot | Yes | Requires 510(k) | Yes |

**Impact**: Research results may not translate to clinical deployment.

**What Needs to Happen**:
1. Document regulatory requirements per platform
2. Create compliance documentation templates
3. Establish research-to-clinical pathway
4. Build regulatory-aware development guidelines

---

### Software Validation Requirements

**Challenge**: IEC 62304 software lifecycle requirements for clinical use.

| Requirement | Research Use | Clinical Use |
|-------------|--------------|--------------|
| Requirements documentation | Optional | Required |
| Design documentation | Optional | Required |
| Risk analysis | Recommended | Required |
| Verification testing | Recommended | Required |
| Validation testing | Recommended | Required |
| Change control | Recommended | Required |

**What Needs to Happen**:
1. Create IEC 62304 compliant development templates
2. Implement automated documentation generation
3. Build verification/validation test frameworks
4. Establish change control processes

---

## 6. Data Format and Recording

### Surgical Data Standards

**Challenge**: No standard for surgical robotics data.

| Organization | Format | Content | Adoption |
|--------------|--------|---------|----------|
| JIGSAWS | Custom HDF5 | Kinematics + video | Academic |
| ORBIT-Surgical | Isaac format | State + action | Emerging |
| SurgicalRL | Gym-based | Episode | Academic |
| Clinical | DICOM (imaging only) | Images | Clinical |

**Impact**: Data cannot be easily shared across organizations.

**What Needs to Happen**:
1. Define unified surgical robotics data format
2. Create format converters for existing datasets
3. Establish metadata standards
4. Build data validation tools

---

### Annotation and Labeling

**Challenge**: Inconsistent surgical phase and action labeling.

| Dataset | Phases | Actions | Instruments |
|---------|--------|---------|-------------|
| Cholec80 | 7 phases | N/A | 7 instruments |
| JIGSAWS | Task-specific | 15 gestures | 2 arms |
| HeiChole | 7 phases | N/A | 7 instruments |

**What Needs to Happen**:
1. Develop unified surgical ontology
2. Create cross-dataset mapping
3. Implement automated annotation tools
4. Establish annotation quality metrics

---

## 7. Multi-Site Deployment Challenges

### Environmental Variation

**Challenge**: Operating room environments vary significantly.

| Factor | Variation | Impact |
|--------|-----------|--------|
| Lighting | 100-10000 lux | Vision policies |
| Mounting | Floor/ceiling/cart | Kinematics |
| Space | 3-10m OR | Navigation |
| EMI | Variable | Communication |

**What Needs to Happen**:
1. Characterize environmental requirements
2. Implement domain adaptation for vision
3. Create site-specific calibration procedures
4. Build environmental monitoring

---

### Infrastructure Requirements

**Challenge**: Different computational and network infrastructure.

| Requirement | Minimum | Recommended | Clinical |
|-------------|---------|-------------|----------|
| GPU | RTX 3080 | RTX 4090 | Certified only |
| Network | 1 Gbps | 10 Gbps | Isolated |
| Real-time OS | Preferred | Required | Required |
| Backup power | Recommended | Required | Required |

**What Needs to Happen**:
1. Define minimum infrastructure requirements
2. Create deployment validation tests
3. Establish network isolation standards
4. Build infrastructure monitoring

---

## 8. Skill and Training Transfer

### Surgeon Training Implications

**Challenge**: Skills learned on one platform may not transfer.

| Factor | Platform A | Platform B | Transfer |
|--------|------------|------------|----------|
| Control mapping | Position | Velocity | Low |
| Force feedback | High fidelity | Low fidelity | Medium |
| Workspace | Large | Small | Low |
| Latency | 1ms | 50ms | Low |

**What Needs to Happen**:
1. Study skill transfer across platforms
2. Develop platform-agnostic training curricula
3. Create simulation-based pre-training
4. Establish competency validation

---

## Summary: Critical Path to Unification

### Immediate Priorities (Q1 2026)
1. **Unified control interface** - ROS 2 package for all platforms
2. **Task-space action representation** - Platform-agnostic actions
3. **Safety abstraction layer** - Consistent safety behavior

### Medium-Term (Q2-Q3 2026)
4. **Force sensing standardization** - Unified force interface
5. **Data format specification** - Interoperable surgical data
6. **Regulatory documentation** - IEC 62304 templates

### Long-Term (Q4 2026+)
7. **Cross-platform skill transfer** - Training standardization
8. **Multi-site infrastructure** - Deployment standards
9. **Clinical validation framework** - Approval pathway

---

*Last updated: January 2026*
