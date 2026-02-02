# Objective 1: Isaac ↔ MuJoCo Bidirectional Conversion

*Complete Bidirectional Conversion Standards for Q1 2026*

**Status**: Active Development
**Target Completion**: Q1 2026
**Last Updated**: February 2026

---

## Overview

This directory contains the complete specification and reference implementation for bidirectional conversion between NVIDIA Isaac Lab and MuJoCo simulation frameworks. The goal is to enable seamless round-trip conversion of robot models, physics parameters, and trained policies.

### Key Requirements

1. **Lossless Round-Trip Conversion**: Converting Isaac → MuJoCo → Isaac should preserve all critical properties
2. **Physics Equivalence**: Dynamics behavior should match within specified tolerances
3. **Asset Preservation**: All meshes, textures, and materials properly converted
4. **Actuator Mapping**: Motor parameters, joint limits, and control modes preserved

---

## Framework Versions

| Framework | Version | GPU Acceleration | Notes |
|-----------|---------|------------------|-------|
| Isaac Lab | 2.3.2+ | PhysX 5.x | Use feature/newton branch for MuJoCo Warp |
| Isaac Sim | 4.5.0+ | RTX | MJCF importer included |
| MuJoCo | 3.4.0 | MJX (JAX) | Reference physics |
| MuJoCo Warp | 0.2.0+ | CUDA 12.4+ | 70-300x GPU speedup |
| Newton | Beta 2 | Warp | Unified backend option |

**Sources**:
- [Isaac Lab Releases](https://github.com/isaac-sim/IsaacLab/releases)
- [MuJoCo Releases](https://github.com/google-deepmind/mujoco/releases)
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)

---

## Conversion Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BIDIRECTIONAL CONVERSION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Isaac Lab   │───▶│    USD      │───▶│   URDF      │───▶│   MJCF      │  │
│  │ Environment │    │ (OpenUSD)   │    │(Intermediate)│    │  (MuJoCo)   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         ▲                                                        │          │
│         │                                                        ▼          │
│         │           ┌─────────────────────────────────────────────┐         │
│         │           │         Physics Validation Layer            │         │
│         │           │  • Trajectory comparison                    │         │
│         │           │  • Contact force verification               │         │
│         │           │  • Actuator response matching               │         │
│         │           └─────────────────────────────────────────────┘         │
│         │                                                        │          │
│         ▼                                                        ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Isaac Lab   │◀───│    USD      │◀───│   URDF      │◀───│   MJCF      │  │
│  │ Environment │    │ (OpenUSD)   │    │(Intermediate)│    │  (MuJoCo)   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Conversion Specifications

### 1. Isaac → MuJoCo Conversion

**Input Formats**:
- USD/USDA (OpenUSD) - Native Isaac Sim format
- URDF - ROS-compatible format loaded into Isaac

**Output Format**:
- MJCF (XML) - MuJoCo native format

**Conversion Steps**:

1. **Export from Isaac Lab**
   ```python
   # Using Isaac Sim's USD exporter
   from omni.isaac.core.utils.stage import save_stage
   save_stage(filepath="robot.usda")
   ```

2. **USD → URDF Conversion**
   ```python
   # Using Newton's USD converter or custom pipeline
   from q1_2026_standards.objective_1 import USDToURDF
   converter = USDToURDF()
   converter.convert("robot.usda", "robot.urdf")
   ```

3. **URDF → MJCF Conversion**
   ```python
   # Using MuJoCo's built-in compiler
   import mujoco
   model = mujoco.MjModel.from_xml_path("robot_temp.xml")
   # Or using custom converter for oncology-specific features
   from q1_2026_standards.objective_1 import URDFToMJCF
   converter = URDFToMJCF()
   converter.convert("robot.urdf", "robot.xml",
                     preserve_soft_body=True)
   ```

4. **Physics Parameter Mapping**
   - Map PhysX contact parameters → MuJoCo solref/solimp
   - Map PhysX material friction → MuJoCo geom friction
   - Map PhysX solver settings → MuJoCo opt settings

### 2. MuJoCo → Isaac Conversion

**Input Format**:
- MJCF (XML) - MuJoCo native format

**Output Formats**:
- USD (OpenUSD) - For Isaac Sim
- URDF (intermediate) - For validation

**Conversion Steps**:

1. **Parse MJCF Model**
   ```python
   import mujoco
   model = mujoco.MjModel.from_xml_path("robot.xml")
   ```

2. **Generate URDF**
   ```python
   # Using mjcf2urdf or custom converter
   from q1_2026_standards.objective_1 import MJCFToURDF
   converter = MJCFToURDF()
   converter.convert("robot.xml", "robot.urdf",
                     preserve_actuators=True)
   ```

3. **Import to Isaac Sim**
   ```python
   # Using Isaac Sim's URDF importer
   from omni.isaac.urdf import _urdf
   urdf_interface = _urdf.acquire_urdf_interface()
   robot_prim = urdf_interface.parse_urdf(
       "robot.urdf",
       import_config
   )
   ```

4. **Alternative: Direct MJCF Import**
   ```python
   # Isaac Sim 5.1.0+ has native MJCF importer
   from omni.isaac.mjcf import _mjcf
   mjcf_interface = _mjcf.acquire_mjcf_interface()
   robot_prim = mjcf_interface.import_asset(
       "robot.xml",
       import_config
   )
   ```

---

## Physics Parameter Mapping

### Contact Dynamics

| Parameter | Isaac PhysX | MuJoCo | Conversion |
|-----------|-------------|--------|------------|
| Stiffness | `contact_offset` | `solref[0]` | `timeconst = 1/sqrt(stiffness)` |
| Damping | implicit | `solref[1]` | `dampratio = damping/(2*sqrt(k))` |
| Friction (static) | `static_friction` | `friction[0]` | Direct mapping |
| Friction (dynamic) | `dynamic_friction` | `friction[0]*0.75` | Approximate |
| Restitution | `restitution` | `solimp[4]` | Requires tuning |

### Solver Configuration

| Parameter | Isaac PhysX | MuJoCo | Notes |
|-----------|-------------|--------|-------|
| Timestep | `sim.dt` | `opt.timestep` | Direct mapping |
| Iterations | `position_iterations` | `opt.iterations` | Scale: PhysX 8 ≈ MuJoCo 50 |
| Solver type | TGS | Newton/CG/PGS | Newton recommended |
| GPU mode | Native | MJX/MJWarp | Both support GPU |

### Actuator Mapping

| Isaac Actuator | MuJoCo Actuator | Parameters |
|----------------|-----------------|------------|
| `DCMotor` | `motor` | gear, forcerange |
| `PositionJoint` | `position` | kp, kv |
| `VelocityJoint` | `velocity` | kv |
| `ImplicitActuator` | `general` | dyntype, dynprm |

---

## Validation Requirements

### Acceptance Criteria

| Metric | Tolerance | Test Method |
|--------|-----------|-------------|
| Joint position error | < 0.001 rad | Trajectory comparison |
| Joint velocity error | < 0.01 rad/s | Trajectory comparison |
| End-effector position | < 0.001 m | Forward kinematics |
| Contact force error | < 5% | Force sensor comparison |
| Round-trip model diff | 0 (critical params) | Schema comparison |

### Validation Test Suite

Run validation after each conversion:

```bash
# Validate Isaac → MuJoCo conversion
python physics_equivalence_tests.py \
    --source robot_isaac.usd \
    --target robot_mujoco.xml \
    --direction isaac_to_mujoco \
    --tolerance strict

# Validate round-trip
python physics_equivalence_tests.py \
    --source robot.urdf \
    --round-trip \
    --report validation_report.html
```

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `README.md` | This overview document |
| `conversion_specification.md` | Detailed technical specification |
| `isaac_to_mujoco_pipeline.py` | Isaac → MuJoCo converter implementation |
| `mujoco_to_isaac_pipeline.py` | MuJoCo → Isaac converter implementation |
| `physics_equivalence_tests.py` | Physics validation test suite |
| `format_mappings.yaml` | Parameter mapping definitions |

---

## Known Limitations

1. **Soft Body Conversion**: Isaac deformable bodies and MuJoCo composites use different underlying models. Manual tuning may be required.

2. **Sensor Models**: Camera and LIDAR sensors require separate configuration in each framework.

3. **Material Properties**: Visual materials (textures, shaders) are not fully preserved in round-trip.

4. **Custom Plugins**: Framework-specific plugins must be re-implemented.

---

## References

- [Isaac Sim MJCF Importer Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_mjcf.html)
- [MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [mjcf2urdf GitHub](https://github.com/iory/mjcf2urdf)
- [URDF2MJCF GitHub](https://github.com/eric-heiden/URDF2MJCF)
- [OpenUSD Specification](https://openusd.org/release/spec.html)
- [Newton Physics Integration](https://github.com/newton-physics/newton)
