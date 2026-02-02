# Benchmark Reference Data

*Baseline results and reference data for validation*

## Overview

This directory contains reference results for validating benchmark implementations and comparing framework performance.

## Reference Baselines

### Physics Benchmarks

| Benchmark | MuJoCo 3.4.0 | Isaac Lab 2.3.2 | Tolerance |
|-----------|--------------|-----------------|-----------|
| `free_fall` | 0.0002 m error | 0.0003 m error | < 0.01 m |
| `pendulum` | 1.2% period error | 1.5% period error | < 5% |
| `contact` | Baseline | < 5% deviation | < 5% |

### Performance Benchmarks

| Benchmark | MuJoCo (CPU) | MuJoCo (MJX) | Isaac (PhysX) |
|-----------|--------------|--------------|---------------|
| `throughput` | 50K steps/s | 2M steps/s | 1.5M steps/s |
| `parallel` | N/A | 4096 envs | 4096 envs |

## Data Format

```json
{
  "benchmark": "free_fall",
  "framework": "mujoco",
  "version": "3.4.0",
  "timestamp": "2026-01-15T10:00:00Z",
  "results": {
    "max_position_error": 0.0002,
    "mean_position_error": 0.0001,
    "steps": 1000,
    "timestep": 0.002
  }
}
```

## References

- [MuJoCo Performance](https://mujoco.readthedocs.io/en/stable/mjx.html#performance)
- [Isaac Lab Benchmarks](https://isaac-sim.github.io/IsaacLab/)
