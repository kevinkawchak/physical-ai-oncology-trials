# Physical AI Oncology Trials: Physical Robot Engineering Examples

*Comprehensive code for engineers working directly on robot hardware for oncology clinical trials*

**Version**: 1.0.0
**Status**: Active Development
**Last Updated**: February 2026

---

## Overview

This directory contains production-oriented code examples for the **physical robot engineering** challenges in oncology clinical trials. While `examples/` covers policy training, digital twin planning, cross-framework validation, agentic workflows, and treatment prediction, **`examples-new/` focuses on what happens when the robot meets the real world**: safety systems, sensor integration, deployment, calibration, human-robot interaction, and laboratory automation.

Every example includes detailed inline instructions written for engineers who are building, integrating, or deploying physical robots.

---

## How `examples-new/` Differs from `examples/`

| Concern | `examples/` | `examples-new/` |
|---------|-------------|------------------|
| **Focus** | AI/ML pipelines, simulation, planning | Physical hardware, deployment, real-time systems |
| **Stage** | Training and pre-deployment | Deployment and intraoperative |
| **Audience** | ML engineers, data scientists | Robotics engineers, integration engineers |
| **Hardware** | Simulation only | Physical robots, sensors, actuators |
| **Key challenge** | Learning good policies | Making policies work safely on hardware |

---

## Example Files

| Example | Use Case | Key Hardware | Difficulty |
|---------|----------|--------------|------------|
| `01_realtime_safety_monitoring.py` | Force/torque limiting, workspace boundaries, watchdog timers | F/T sensor, E-stop circuit | Advanced |
| `02_sensor_fusion_intraoperative.py` | Multi-sensor perception, tissue deformation tracking | Stereo camera, RGBD, F/T sensor | Advanced |
| `03_ros2_surgical_deployment.py` | ROS 2 node architecture, control loop, state machine | dVRK/Kinova, PREEMPT_RT kernel | Advanced |
| `04_hand_eye_calibration_registration.py` | Camera-robot calibration, patient-to-image registration | Optical tracker, calibration target | Intermediate |
| `05_shared_autonomy_teleoperation.py` | Surgeon-AI shared control, virtual fixtures, haptics | Surgeon console, haptic device | Advanced |
| `06_robotic_sample_handling.py` | Specimen handling, barcode verification, cold chain | Lab robot, barcode scanner | Intermediate |

---

## Quick Start

### Prerequisites

```bash
# Install base dependencies
pip install -r requirements.txt

# Verify frameworks
python scripts/verify_installation.py

# Run an example
python examples-new/01_realtime_safety_monitoring.py
```

### Running Examples

Each example runs standalone with simulated hardware and produces meaningful output:

```bash
# Real-time safety monitoring with force limits
python examples-new/01_realtime_safety_monitoring.py

# Multi-sensor fusion pipeline
python examples-new/02_sensor_fusion_intraoperative.py

# ROS 2 deployment architecture demo
python examples-new/03_ros2_surgical_deployment.py

# Hand-eye calibration and patient registration
python examples-new/04_hand_eye_calibration_registration.py

# Shared autonomy with virtual fixtures
python examples-new/05_shared_autonomy_teleoperation.py

# Robotic specimen handling for clinical trials
python examples-new/06_robotic_sample_handling.py
```

---

## Example Summaries

### 01: Real-Time Safety Monitoring

The safety layer between AI policy and robot hardware. Implements IEC 80601-2-77 supervisory safety monitoring with:

- Force-torque limit checking (tissue-specific thresholds)
- Workspace boundary enforcement (spherical, from surgical plan)
- Watchdog timer for control loop integrity
- Force rate-of-change detection (impact prevention)
- Joint limit checking (software limits tighter than mechanical)
- Full audit trail for regulatory compliance

### 02: Multi-Sensor Fusion for Intraoperative Perception

Real-time perception pipeline combining multiple sensors:

- Stereo/RGBD depth processing to point clouds
- Instrument segmentation from endoscope imagery
- Tissue deformation tracking via ICP registration
- Temporal synchronization of multi-rate sensor streams
- Tumor margin distance estimation
- Unified scene representation for robot controller and surgeon display

### 03: ROS 2 Surgical Robot Deployment

Complete ROS 2 node architecture for deploying trained policies:

- Procedure state machine (IDLE → HOMING → OPERATE → COMPLETE)
- Policy inference engine with ONNX Runtime
- Hardware interface abstraction (dVRK, Kinova, UR)
- Real-time control loop with jitter tracking
- Launch configuration for the full node graph
- Data recording for post-operative analysis

### 04: Hand-Eye Calibration and Patient Registration

Spatial calibration procedures required before every procedure:

- Eye-in-hand and eye-to-hand calibration (Tsai-Lenz method)
- Fiducial-based patient registration (Arun SVD method)
- Surface-based registration (ICP)
- Independent verification with test points
- Accuracy metrics: FRE, TRE, rotation error
- Clinical acceptance thresholds

### 05: Shared Autonomy and Teleoperation

Surgeon-robot collaboration during procedures:

- Five autonomy levels (raw teleop → full autonomy)
- Virtual fixtures: forbidden regions, guidance paths, boundary planes
- Command blending (configurable surgeon/AI authority)
- Surgeon input processing: scaling, tremor filtering, clutching
- Haptic force rendering for constraint feedback
- Smooth transitions between autonomy modes

### 06: Robotic Sample Handling

Laboratory automation for oncology trial specimens:

- Specimen pick-and-place with container-specific grip parameters
- Barcode/RFID verification at every transfer point
- Cold chain temperature monitoring and excursion logging
- Chain-of-custody audit trail (21 CFR Part 11 compatible)
- Batch processing workflow for clinical trial visits
- Priority sorting (time-sensitive specimens first)

---

## Regulatory Standards Referenced

| Standard | Relevance | Examples |
|----------|-----------|----------|
| IEC 80601-2-77 | Robotically assisted surgical equipment | 01, 03, 05 |
| IEC 62304 | Medical device software lifecycle | 01, 03 |
| ISO 13482 | Personal care robot safety | 01 |
| 21 CFR Part 11 | Electronic records and signatures | 06 |
| USP <797> | Sterile compounding | 06 |
| CAP/CLIA | Laboratory accreditation | 06 |
| ICH E6(R3) | Good Clinical Practice | 06 |

---

## Contributing

When adding new examples to `examples-new/`:

1. Focus on **physical robot engineering** (not simulation-only)
2. Include detailed **inline instructions** for hardware setup
3. Specify all **hardware and framework requirements** in the header
4. Ensure examples **run standalone** with simulated hardware
5. Follow the naming convention (`XX_descriptive_name.py`)
6. Do not duplicate topics already covered in `examples/`

---

*These examples are part of the Physical AI Oncology Trials Unification Framework.*
