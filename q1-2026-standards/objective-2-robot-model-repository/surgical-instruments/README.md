# Surgical Instruments

*End-effector and instrument models for surgical simulation*

**Target Count**: 20+ models
**Status**: Development
**Last Updated**: February 2026

---

## Overview

This category contains models of surgical instruments, end-effectors, and tools used in oncology procedures. Models are designed to be compatible with the dVRK and other surgical robotic platforms.

## Instrument Categories

### Grasping Instruments

| Model | Type | dVRK Compatible |
|-------|------|-----------------|
| Large Needle Driver | Grasper | Yes |
| Maryland Bipolar Forceps | Grasper | Yes |
| ProGrasp Forceps | Grasper | Yes |
| Cadiere Forceps | Grasper | Yes |
| Biopsy Forceps | Grasper | Yes |

### Cutting Instruments

| Model | Type | Energy |
|-------|------|--------|
| Hot Shears | Scissors | Monopolar |
| Curved Scissors | Scissors | None |
| Metzenbaum Scissors | Scissors | None |

### Energy Instruments

| Model | Type | Modality |
|-------|------|----------|
| Cautery Hook | Cautery | Monopolar |
| Vessel Sealer | Sealing | Bipolar |
| Harmonic Scalpel | Cutting | Ultrasonic |

### Applicators

| Model | Type | Application |
|-------|------|-------------|
| Clip Applier | Hemostasis | Clip placement |
| Endoscopic Stapler | Stapling | Tissue division |

## Format Requirements

Each instrument model should include:
- Gripper kinematics (open/close mechanism)
- Contact geometry for grasping simulation
- Mass properties for dynamics

## References

- [ORBIT-Surgical Instruments](https://orbit-surgical.github.io/)
- [Intuitive Surgical Instruments](https://www.intuitive.com/en-us/products-and-services/da-vinci/instruments)
