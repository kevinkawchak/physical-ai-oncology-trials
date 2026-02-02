# Validation Benchmark Metrics

*Metric definitions for Q1 2026 benchmark suite*

## Overview

This directory contains metric definitions and computation methods for the validation benchmark suite.

## Metric Categories

### Physics Metrics

| Metric | Description | Unit | Computation |
|--------|-------------|------|-------------|
| `position_error` | Deviation from expected position | m | RMSE of position trajectory |
| `velocity_error` | Deviation from expected velocity | m/s | RMSE of velocity trajectory |
| `force_error` | Deviation from expected forces | N | RMSE of contact forces |
| `energy_conservation` | Energy drift over time | J | Total energy variance |
| `momentum_conservation` | Momentum conservation | kg·m/s | Momentum change ratio |

### Performance Metrics

| Metric | Description | Unit | Computation |
|--------|-------------|------|-------------|
| `steps_per_second` | Simulation throughput | steps/s | Wall time measurement |
| `gpu_utilization` | GPU efficiency | % | CUDA profiler |
| `memory_usage` | Peak memory consumption | MB | Memory profiler |
| `parallel_efficiency` | Multi-env scaling | ratio | Throughput vs linear |

### Transfer Metrics

| Metric | Description | Unit | Computation |
|--------|-------------|------|-------------|
| `success_rate` | Task completion rate | % | Episodes succeeded / total |
| `policy_performance` | Reward achieved | ratio | Reward vs training baseline |
| `domain_gap` | Sim-to-real difference | ratio | Performance drop |

## References

- [robosuite Metrics](https://robosuite.ai/)
- [ORBIT-Surgical Evaluation](https://orbit-surgical.github.io/)
