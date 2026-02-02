# Bidirectional Conversion Technical Specification

*Isaac ↔ MuJoCo Conversion Standard v1.0*

**Document Version**: 1.0.0
**Effective Date**: Q1 2026
**Authors**: Physical AI Oncology Trials Consortium

---

## 1. Introduction

### 1.1 Purpose

This specification defines the technical requirements for bidirectional conversion between NVIDIA Isaac Lab and Google DeepMind MuJoCo simulation frameworks. The goal is to enable researchers and engineers to:

- Train policies in Isaac Lab (GPU-accelerated) and validate in MuJoCo (physics reference)
- Share robot models across organizations using different frameworks
- Benchmark algorithms consistently across simulation platforms

### 1.2 Scope

This specification covers:
- Robot model format conversion (URDF, MJCF, USD, SDF)
- Physics parameter mapping
- Actuator and sensor conversion
- Validation requirements

This specification does NOT cover:
- Real-time synchronization during simulation
- Policy format conversion (see ONNX export guidelines)
- Visual rendering equivalence

### 1.3 Normative References

| Standard | Description |
|----------|-------------|
| URDF 1.0 | Unified Robot Description Format (ROS) |
| MJCF 3.4.0 | MuJoCo XML Format |
| USD 24.08 | Universal Scene Description |
| SDF 1.9 | Simulation Description Format |
| IEEE 1873-2015 | Robot Map Data Representation |

---

## 2. Format Definitions

### 2.1 URDF (Unified Robot Description Format)

**Source**: [ROS robot_model](https://github.com/ros/robot_model)

```xml
<?xml version="1.0"?>
<robot name="example_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="package://robot/meshes/base.stl"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="package://robot/meshes/base_collision.stl"/>
      </geometry>
    </collision>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="100" velocity="1.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>
</robot>
```

### 2.2 MJCF (MuJoCo XML Format)

**Source**: [MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)

```xml
<mujoco model="example_robot">
  <compiler angle="radian" autolimits="true"/>

  <default>
    <joint damping="0.5" armature="0.01"/>
    <geom friction="0.8 0.02 0.01"/>
  </default>

  <option timestep="0.002" integrator="implicitfast"/>

  <asset>
    <mesh name="base_mesh" file="base.stl"/>
  </asset>

  <worldbody>
    <body name="base_link" pos="0 0 0">
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
      <geom type="mesh" mesh="base_mesh"/>

      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14159 3.14159"/>
        <inertial mass="0.5" pos="0 0 0.05" diaginertia="0.005 0.005 0.005"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="joint1" gear="50" forcerange="-100 100"/>
  </actuator>
</mujoco>
```

### 2.3 Key Structural Differences

| Aspect | URDF | MJCF |
|--------|------|------|
| **Hierarchy** | Flat links + joints | Nested bodies |
| **Joint Definition** | Separate element | Inside child body |
| **Actuators** | External (ros_control) | Built-in actuator element |
| **Defaults** | None | Hierarchical defaults |
| **Options** | Limited | Comprehensive physics options |
| **Constraints** | Limited | Full constraint support |

---

## 3. Conversion Rules

### 3.1 URDF → MJCF Conversion

#### 3.1.1 Link to Body Mapping

```
URDF link → MJCF body
  - link.name → body.name
  - link.inertial.mass → inertial.mass
  - link.inertial.origin → inertial.pos
  - link.inertial.inertia → inertial.diaginertia (if diagonal)
                          → inertial.fullinertia (if non-diagonal)
```

#### 3.1.2 Joint Mapping

| URDF Joint Type | MJCF Joint Type | Notes |
|-----------------|-----------------|-------|
| `revolute` | `hinge` | Add range limits |
| `continuous` | `hinge` | No range limits |
| `prismatic` | `slide` | Add range limits |
| `fixed` | (no joint) | Bodies directly nested |
| `floating` | `free` | 6-DOF joint |
| `planar` | 3x `slide` | Decompose to primitives |

#### 3.1.3 Geometry Mapping

| URDF Geometry | MJCF Geometry | Conversion |
|---------------|---------------|------------|
| `box` | `box` | size/2 (MJCF uses half-size) |
| `cylinder` | `cylinder` | Direct |
| `sphere` | `sphere` | Direct |
| `mesh` | `mesh` | Requires asset entry |

#### 3.1.4 Dynamics Mapping

```python
# URDF dynamics
urdf_damping = 0.5      # N·m·s/rad
urdf_friction = 0.1     # N·m

# MJCF conversion
mjcf_damping = urdf_damping       # Direct mapping
mjcf_frictionloss = urdf_friction # Direct mapping
mjcf_armature = 0.01              # Default, adjust as needed
```

### 3.2 MJCF → URDF Conversion

#### 3.2.1 Body to Link Mapping

```
MJCF body → URDF link
  - body.name → link.name
  - inertial.mass → link.inertial.mass
  - inertial.pos → link.inertial.origin.xyz
  - inertial.diaginertia → link.inertial.inertia (ixx, iyy, izz)
```

#### 3.2.2 Joint Extraction

MJCF joints are defined within child bodies. Extract to URDF format:

```python
# For each body with a joint
urdf_joint = {
    "name": mjcf_joint.name,
    "type": "revolute" if mjcf_joint.type == "hinge" else "prismatic",
    "parent": parent_body.name,
    "child": current_body.name,
    "origin": body.pos,  # Position relative to parent
    "axis": mjcf_joint.axis,
    "limit": {
        "lower": mjcf_joint.range[0],
        "upper": mjcf_joint.range[1],
        "effort": actuator.forcerange[1] if actuator else 100,
        "velocity": 1.0  # Must be inferred or set default
    },
    "dynamics": {
        "damping": mjcf_joint.damping,
        "friction": mjcf_joint.frictionloss
    }
}
```

#### 3.2.3 Actuator Handling

MJCF actuators must be mapped to external control interfaces:

| MJCF Actuator | URDF Approach |
|---------------|---------------|
| `motor` | transmission + effort controller |
| `position` | transmission + position controller |
| `velocity` | transmission + velocity controller |
| `general` | Custom controller implementation |

---

## 4. Physics Parameter Mapping

### 4.1 Contact Parameters

#### 4.1.1 Isaac PhysX → MuJoCo

```python
def physx_to_mujoco_contact(
    contact_offset: float,
    rest_offset: float,
    static_friction: float,
    dynamic_friction: float,
    restitution: float
) -> dict:
    """
    Convert PhysX contact parameters to MuJoCo format.

    References:
    - PhysX: https://nvidia-omniverse.github.io/PhysX/
    - MuJoCo: https://mujoco.readthedocs.io/en/stable/modeling.html#contact
    """
    # Estimate stiffness from contact_offset
    # Lower offset = stiffer contact
    stiffness = 1.0 / (contact_offset * 1000)  # Approximate

    # timeconst for solref (assuming unit mass)
    timeconst = 1.0 / np.sqrt(stiffness)

    # dampratio (critical damping = 1.0)
    dampratio = 1.0

    # solimp parameters
    dmin = 0.9
    dmax = 0.95
    width = max(0.001, contact_offset)

    return {
        "solref": [timeconst, dampratio],
        "solimp": [dmin, dmax, width, 0.5, 2],
        "friction": [static_friction, 0.02, 0.01],
        "condim": 3  # Standard sliding friction
    }
```

#### 4.1.2 MuJoCo → Isaac PhysX

```python
def mujoco_to_physx_contact(
    solref: tuple,
    solimp: tuple,
    friction: tuple
) -> dict:
    """
    Convert MuJoCo contact parameters to PhysX format.
    """
    timeconst, dampratio = solref

    # Estimate stiffness from timeconst
    stiffness = 1.0 / (timeconst ** 2)

    # contact_offset from width in solimp
    contact_offset = solimp[2] if len(solimp) > 2 else 0.001

    return {
        "contact_offset": contact_offset,
        "rest_offset": 0.0,
        "static_friction": friction[0],
        "dynamic_friction": friction[0] * 0.8,  # Approximate
        "restitution": 0.1,  # Conservative default
        "bounce_threshold_velocity": 0.2
    }
```

### 4.2 Solver Configuration

```yaml
# Recommended solver settings for equivalent behavior

training_mode:
  isaac:
    physics:
      dt: 0.002
      substeps: 2
      gpu_found_lost_pairs: true
      gpu_collision_stack_size: 67108864
    physx:
      solver_type: 1  # TGS
      position_iterations: 4
      velocity_iterations: 1

  mujoco:
    option:
      timestep: 0.002
      iterations: 50
      tolerance: 1e-6
      integrator: "implicitfast"
      solver: "Newton"

validation_mode:
  isaac:
    physics:
      dt: 0.001
      substeps: 4
    physx:
      position_iterations: 8
      velocity_iterations: 2

  mujoco:
    option:
      timestep: 0.001
      iterations: 100
      tolerance: 1e-8
      integrator: "implicit"
      solver: "Newton"
```

---

## 5. Validation Requirements

### 5.1 Kinematic Validation

| Test | Requirement | Method |
|------|-------------|--------|
| Forward Kinematics | Position error < 1e-6 m | Compare EE positions for same joint config |
| Inverse Kinematics | Solution consistency | Verify IK solutions match |
| Jacobian | Error < 1e-6 | Compare analytical Jacobians |
| Joint Limits | Exact match | Verify limits in both directions |

### 5.2 Dynamic Validation

| Test | Tolerance | Duration |
|------|-----------|----------|
| Free-fall test | Velocity < 0.1% error | 2 seconds |
| Pendulum test | Position < 1% error | 10 seconds |
| Contact test | Force < 5% error | 1 second |
| Trajectory tracking | RMSE < 0.01 rad | 5 seconds |

### 5.3 Round-Trip Validation

```python
def validate_round_trip(original_urdf: str) -> ValidationResult:
    """
    Validate that URDF → MJCF → URDF preserves critical properties.
    """
    # Convert to MJCF
    mjcf_path = convert_urdf_to_mjcf(original_urdf)

    # Convert back to URDF
    recovered_urdf = convert_mjcf_to_urdf(mjcf_path)

    # Compare critical properties
    original = parse_urdf(original_urdf)
    recovered = parse_urdf(recovered_urdf)

    results = {
        "link_count_match": len(original.links) == len(recovered.links),
        "joint_count_match": len(original.joints) == len(recovered.joints),
        "mass_error": compare_masses(original, recovered),
        "inertia_error": compare_inertias(original, recovered),
        "limit_error": compare_limits(original, recovered),
    }

    return ValidationResult(
        passed=all([
            results["link_count_match"],
            results["joint_count_match"],
            results["mass_error"] < 1e-6,
            results["inertia_error"] < 1e-6,
            results["limit_error"] < 1e-6,
        ]),
        details=results
    )
```

---

## 6. Error Handling

### 6.1 Conversion Errors

| Error Code | Description | Recovery |
|------------|-------------|----------|
| `E001` | Unsupported joint type | Use closest equivalent + warning |
| `E002` | Missing mesh file | Generate primitive approximation |
| `E003` | Invalid inertia tensor | Recompute from geometry |
| `E004` | Non-positive mass | Set to minimum (1e-6 kg) |
| `E005` | Kinematic loop detected | Convert to constraint |

### 6.2 Validation Errors

| Error Code | Description | Action |
|------------|-------------|--------|
| `V001` | Kinematic mismatch | Report discrepancy, fail validation |
| `V002` | Dynamics deviation > tolerance | Report metrics, warn |
| `V003` | Missing actuator mapping | Add to mapping table |
| `V004` | Mesh collision mismatch | Visual inspection required |

---

## 7. Appendix

### 7.1 Reference Implementations

- **mjcf2urdf**: https://github.com/iory/mjcf2urdf
- **URDF2MJCF**: https://github.com/eric-heiden/URDF2MJCF
- **Wiki-GRx-MJCF**: https://github.com/FFTAI/Wiki-GRx-MJCF
- **Isaac Sim MJCF Importer**: Built into Isaac Sim 5.1.0+

### 7.2 Test Models

The following models are used for validation:

1. **Franka Panda** - 7-DOF manipulator (standard benchmark)
2. **dVRK PSM** - 7+1 DOF surgical robot
3. **UR5** - 6-DOF industrial manipulator
4. **Humanoid (MuJoCo)** - Complex articulated body

### 7.3 Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Feb 2026 | Initial specification |
