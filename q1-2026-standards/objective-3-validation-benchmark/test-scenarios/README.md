# Benchmark Test Scenarios

*Scenario definitions for Q1 2026 benchmark suite*

## Overview

This directory contains configuration files for benchmark test scenarios.

## Scenario Categories

### Surgical Scenarios

| Scenario | Description | Robot | Difficulty |
|----------|-------------|-------|------------|
| `needle_insertion` | Insert needle into tissue | dVRK PSM | Medium |
| `tissue_manipulation` | Grasp and retract tissue | dVRK PSM | Medium |
| `suturing` | Place surgical suture | dVRK PSM | Hard |
| `tumor_resection` | Simulated tumor removal | dVRK PSM | Hard |

### Manipulation Scenarios

| Scenario | Description | Robot | Difficulty |
|----------|-------------|-------|------------|
| `peg_transfer` | Transfer peg between posts | Any | Easy |
| `pick_and_place` | Pick object, place target | Any | Easy |
| `reach` | Reach target position | Any | Easy |
| `push` | Push object to goal | Any | Medium |

### Physics Validation Scenarios

| Scenario | Description | Purpose |
|----------|-------------|---------|
| `free_fall` | Gravity-driven motion | Validate dynamics |
| `pendulum` | Oscillatory motion | Validate energy |
| `collision` | Object collision | Validate contact |

## Scenario Configuration Format

```yaml
# Example: needle_insertion.yaml
name: "needle_insertion"
version: "1.0"

environment:
  robot: "dvrk_psm"
  objects:
    - name: "tissue_phantom"
      type: "deformable"
    - name: "needle"
      type: "rigid"

task:
  goal: "Insert needle to depth of 20mm"
  max_steps: 500
  success_threshold: 0.95

metrics:
  - insertion_accuracy
  - force_profile
  - tissue_damage

randomization:
  tissue_stiffness: [0.8, 1.2]
  needle_pose: "uniform"
```

## References

- [ORBIT-Surgical Tasks](https://orbit-surgical.github.io/)
- [robosuite Tasks](https://robosuite.ai/)
