# Q1 2026 Standards for Physical AI Oncology Trials

*Proposed Standards for Meeting Q1 2026 Unification Objectives*

**Version**: 1.0.0
**Last Updated**: February 2026
**Status**: Active Development

---

## Executive Summary

This directory contains comprehensive proposed standards, specifications, and reference implementations for achieving the three Q1 2026 objectives defined in the Physical AI Oncology Trials Unification Framework:

| Objective | Description | Target |
|-----------|-------------|--------|
| **1** | Complete Isaac ↔ MuJoCo bidirectional conversion | Full bidirectional support |
| **2** | Publish unified robot model repository | 50+ validated models |
| **3** | Release validation benchmark suite v1.0 | Production-ready benchmarks |

These standards are designed to enable multi-organization cooperation and ensure consistent, reproducible results across the physical AI ecosystem for oncology clinical trials.

---

## Framework Version Requirements

All standards in this directory are based on the following framework versions:

| Framework | Required Version | Release Date | Source |
|-----------|------------------|--------------|--------|
| NVIDIA Isaac Lab | 2.3.2+ | Jan 2026 | [GitHub](https://github.com/isaac-sim/IsaacLab/releases) |
| NVIDIA Isaac Sim | 4.5.0+ | Jan 2026 | [NVIDIA Developer](https://developer.nvidia.com/isaac-sim) |
| MuJoCo | 3.4.0 | Dec 2025 | [GitHub](https://github.com/google-deepmind/mujoco/releases) |
| MuJoCo MJX | 3.4.0 | Dec 2025 | [MJX Docs](https://mujoco.readthedocs.io/en/stable/mjx.html) |
| MuJoCo Warp | 0.2.0+ | Jan 2026 | [GitHub](https://github.com/google-deepmind/mujoco_warp) |
| Newton Physics | Beta 2 | Jan 2026 | [Newton Branch](https://github.com/isaac-sim/IsaacLab/tree/feature/newton) |
| Gazebo Ionic | 8.x | 2025 | [Gazebo](https://gazebosim.org/docs/ionic) |
| ROS 2 | Jazzy | 2024 | [ROS 2 Docs](https://docs.ros.org/en/jazzy/) |

---

## Directory Structure

```
q1-2026-standards/
├── README.md                              # This file
│
├── objective-1-bidirectional-conversion/  # Isaac ↔ MuJoCo conversion
│   ├── README.md                          # Conversion standards overview
│   ├── conversion_specification.md        # Technical specification
│   ├── isaac_to_mujoco_pipeline.py        # Isaac → MuJoCo converter
│   ├── mujoco_to_isaac_pipeline.py        # MuJoCo → Isaac converter
│   ├── physics_equivalence_tests.py       # Physics validation tests
│   └── format_mappings.yaml               # Parameter mappings
│
├── objective-2-robot-model-repository/    # Unified model repository
│   ├── README.md                          # Repository standards
│   ├── model_specification.md             # Model format specification
│   ├── model_registry.yaml                # Registry of 50+ models
│   ├── model_validator.py                 # Validation tool
│   ├── oncology-robots/                   # Oncology-specific robots
│   │   └── README.md
│   ├── surgical-instruments/              # Surgical tool models
│   │   └── README.md
│   └── general-manipulators/              # General-purpose robots
│       └── README.md
│
├── objective-3-validation-benchmark/      # Benchmark suite v1.0
│   ├── README.md                          # Benchmark overview
│   ├── benchmark_specification.md         # Benchmark requirements
│   ├── benchmark_runner.py                # Benchmark execution tool
│   ├── metrics/                           # Performance metrics
│   │   └── README.md
│   ├── test-scenarios/                    # Test case definitions
│   │   └── README.md
│   └── reference-data/                    # Reference results
│       └── README.md
│
└── implementation-guide/                  # Implementation guidance
    ├── README.md                          # Guide overview
    ├── timeline.md                        # Implementation timeline
    └── compliance_checklist.md            # Compliance requirements
```

---

## Quick Start

### 1. Verify Framework Compatibility

```bash
# Check installed framework versions
python q1-2026-standards/implementation-guide/check_compatibility.py
```

### 2. Run Bidirectional Conversion (Objective 1)

```python
from q1_2026_standards.objective_1 import IsaacMuJoCoConverter

converter = IsaacMuJoCoConverter()

# Isaac → MuJoCo
converter.isaac_to_mujoco(
    source="models/surgical_arm.usd",
    output="models/surgical_arm.xml"
)

# MuJoCo → Isaac
converter.mujoco_to_isaac(
    source="models/surgical_arm.xml",
    output="models/surgical_arm.usd"
)
```

### 3. Validate Robot Models (Objective 2)

```python
from q1_2026_standards.objective_2 import ModelValidator

validator = ModelValidator()
results = validator.validate_model(
    "models/dvrk_psm.urdf",
    target_formats=["mjcf", "usd"],
    physics_validation=True
)
```

### 4. Run Benchmark Suite (Objective 3)

```python
from q1_2026_standards.objective_3 import BenchmarkRunner

runner = BenchmarkRunner()
results = runner.run_suite(
    frameworks=["isaac", "mujoco"],
    scenarios=["needle_insertion", "tissue_manipulation"],
    output_report="benchmark_results.html"
)
```

---

## Key Standards Summary

### Objective 1: Bidirectional Conversion

| Standard | Requirement | Validation Method |
|----------|-------------|-------------------|
| **Format Preservation** | All kinematic data preserved during round-trip | Automated comparison |
| **Physics Equivalence** | <1% deviation in dynamics simulation | Trajectory comparison |
| **Asset Handling** | Mesh files converted/referenced correctly | Visual inspection + checksum |
| **Actuator Mapping** | Motor/joint limits preserved | Limit testing |

**Key Technologies**:
- Newton Physics Engine integration (Isaac Lab feature/newton branch)
- MuJoCo Warp for GPU-accelerated validation
- USD as intermediate format for lossless conversion

### Objective 2: Robot Model Repository

| Category | Target Count | Validation Level |
|----------|--------------|------------------|
| **Oncology Robots** | 15+ models | Clinical-grade |
| **Surgical Instruments** | 20+ models | Research-grade |
| **General Manipulators** | 15+ models | Standard |
| **Total** | 50+ models | Framework-validated |

**Supported Formats per Model**:
- URDF (ROS compatibility)
- MJCF (MuJoCo native)
- USD (Isaac Sim/Omniverse)
- SDF (Gazebo compatibility)

### Objective 3: Validation Benchmark Suite

| Benchmark Category | Scenarios | Metrics |
|--------------------|-----------|---------|
| **Physics Accuracy** | 5 scenarios | Position/velocity/force error |
| **Performance** | 3 scenarios | Steps/second, GPU utilization |
| **Sim-to-Real Gap** | 4 scenarios | Transfer success rate |
| **Cross-Framework** | 6 scenarios | Consistency score |

---

## Compliance Requirements

All implementations must comply with:

1. **Technical Standards**
   - IEEE 1873-2015 (Robot Map Data Representation)
   - ISO 10303-242 (STEP AP242 for CAD/simulation)
   - USD Schema 24.08+ (OpenUSD)

2. **Clinical Standards** (for oncology applications)
   - FDA 21 CFR Part 11 (Electronic Records)
   - IEC 62304 (Medical Device Software)
   - ISO 13482 (Personal Care Robots)

3. **Data Standards**
   - URDF 1.0 Specification (ROS)
   - MJCF 3.4.0 Schema (MuJoCo)
   - SDF 1.9 (SDFormat)

---

## Contributing

Contributions to these standards are welcome from all consortium members. Before contributing:

1. Review the relevant objective's README and specification
2. Ensure cross-framework compatibility
3. Include validation tests
4. Document with appropriate citations

See [implementation-guide/README.md](implementation-guide/README.md) for detailed contribution guidelines.

---

## References

### Framework Documentation
- [NVIDIA Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/) - Primary GPU-accelerated training framework
- [MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/) - Reference physics simulation
- [MuJoCo MJX Documentation](https://mujoco.readthedocs.io/en/stable/mjx.html) - JAX-based GPU acceleration
- [MuJoCo Warp Documentation](https://mujoco.readthedocs.io/en/latest/mjwarp.html) - NVIDIA GPU optimization
- [Newton Physics Engine](https://github.com/newton-physics/newton) - Next-gen GPU physics

### Conversion Tools
- [Isaac Sim MJCF Importer](https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_mjcf.html)
- [mjcf2urdf](https://github.com/iory/mjcf2urdf) - MJCF to URDF conversion
- [URDF2MJCF](https://github.com/eric-heiden/URDF2MJCF) - URDF to MJCF conversion
- [Newton URDF-USD Converter](https://github.com/newton-physics/newton)

### Model Repositories
- [Universal Robot Description Directory (URDD)](https://arxiv.org/html/2512.23135) - Dec 2025 unified standard
- [ROS robot_model](https://github.com/ros/robot_model) - URDF specification
- [Awesome URDF](https://github.com/gbionics/awesome-urdf) - Curated URDF resources

### Benchmarking
- [robosuite](https://robosuite.ai/) - MuJoCo-based benchmarks
- [Isaac Lab-Arena](https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena/) - Large-scale evaluation
- [ORBIT-Surgical](https://orbit-surgical.github.io/) - Surgical task benchmarks

---

*These standards are part of the Physical AI Oncology Trials Unification Framework.*
