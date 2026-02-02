# Patient-Specific Digital Twin Modeling

*Creating personalized virtual tumor models from multimodal imaging data*

**Version**: 1.0.0
**Status**: Active Development
**Last Updated**: February 2026

---

## Overview

Patient-specific digital twin modeling transforms medical imaging data into calibrated computational models that replicate individual tumor behavior. These models enable personalized treatment planning and outcome prediction.

---

## Pipeline Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Medical Imaging │───>│  Segmentation &  │───>│  Model          │
│  (CT, MRI, PET)  │    │  Feature Extract │    │  Calibration    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Treatment      │<───│  Prediction &    │<───│  Digital Twin   │
│  Recommendations│    │  Simulation      │    │  Instance       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## Supported Cancer Types

| Cancer Type | Model Type | Validation Status | Reference |
|-------------|------------|-------------------|-----------|
| Glioblastoma (GBM) | Reaction-Diffusion | Clinical validation | TumorTwin |
| Triple-Negative Breast Cancer | Mechanistic | Research | TumorTwin |
| Non-Small Cell Lung Cancer | Logistic + Spatial | Research | Custom |
| Prostate Cancer | PSA-informed | Clinical | npj Digital Medicine |
| Pancreatic Cancer | Multi-compartment | Preclinical | Custom |

---

## Key Components

### 1. Imaging Data Integration

```python
from digital_twins.patient_modeling import ImagingPipeline
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst,
    Spacing, ScaleIntensity, CropForeground
)

# Standard preprocessing pipeline
preprocess = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Spacing(pixdim=(1.0, 1.0, 1.0)),  # Isotropic 1mm
    ScaleIntensity(minv=0.0, maxv=1.0),
    CropForeground()
])

pipeline = ImagingPipeline(
    modalities=["T1", "T1CE", "T2", "FLAIR"],
    preprocessing=preprocess,
    segmentation_model="nnunet_brain_tumor"
)

# Process patient imaging
processed = pipeline.process(
    patient_dir="/data/patient_001/",
    output_dir="/processed/patient_001/"
)
```

### 2. Tumor Segmentation

```python
from monai.networks.nets import SegResNet
from monai.inferers import SlidingWindowInferer

# Load pre-trained segmentation model
segmentor = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,  # T1, T1CE, T2, FLAIR
    out_channels=4  # Background, NCR, ED, ET
)
segmentor.load_state_dict(torch.load("brats_segresnet.pth"))

# Run inference with sliding window
inferer = SlidingWindowInferer(
    roi_size=(128, 128, 128),
    sw_batch_size=4,
    overlap=0.5
)

segmentation = inferer(processed["imaging"], segmentor)
```

### 3. Parameter Estimation

```python
from digital_twins.patient_modeling import ParameterEstimator

estimator = ParameterEstimator(
    model_type="reaction_diffusion",
    optimization="bayesian"
)

# Estimate patient-specific parameters
parameters = estimator.estimate(
    imaging_t0=processed["scan_baseline"],
    imaging_t1=processed["scan_followup"],
    time_interval_days=30,
    prior_distributions={
        "diffusion": ("lognormal", 0.1, 0.5),  # mm^2/day
        "proliferation": ("lognormal", 0.05, 0.3)  # /day
    }
)

print(f"Estimated diffusion: {parameters['diffusion']:.3f} mm^2/day")
print(f"Estimated proliferation: {parameters['proliferation']:.4f} /day")
print(f"Carrying capacity: {parameters['carrying_capacity']:.2f}")
```

---

## TumorTwin Framework Integration

The [TumorTwin](https://github.com/OncologyModelingGroup/TumorTwin) framework provides the core computational infrastructure for patient-specific digital twins.

### Key Features
- Modular architecture for different tumor models
- CPU and GPU-parallelized implementations
- Bayesian parameter calibration
- Uncertainty quantification
- Longitudinal data handling

### Installation

```bash
pip install tumortwin
# or from source
git clone https://github.com/OncologyModelingGroup/TumorTwin.git
cd TumorTwin && pip install -e .
```

### Basic Usage

```python
import tumortwin as tt

# Load patient dataset
patient = tt.PatientData(
    patient_id="GBM_001",
    imaging_dir="/data/GBM_001/",
    treatment_file="/data/GBM_001/treatment.csv"
)

# Initialize reaction-diffusion model
model = tt.ReactionDiffusionModel(
    domain=patient.brain_mask,
    resolution_mm=1.0
)

# Calibrate to longitudinal data
calibration = tt.BayesianCalibration(
    model=model,
    patient=patient,
    n_samples=1000
)
posterior = calibration.run()

# Create digital twin
digital_twin = tt.DigitalTwin(
    model=model,
    parameters=posterior.map_estimate,
    uncertainty=posterior.credible_interval(0.95)
)

# Predict tumor evolution
prediction = digital_twin.predict(
    horizon_days=180,
    output_times=[30, 60, 90, 120, 150, 180]
)
```

---

## Validation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Dice Coefficient | Segmentation overlap | > 0.85 |
| Volume Error | Predicted vs. actual volume | < 15% |
| Spatial Accuracy | COM distance | < 5mm |
| Calibration | Prediction interval coverage | 90-95% |

---

## References

- [TumorTwin: Python framework for patient-specific digital twins](https://arxiv.org/abs/2505.00670) - arXiv 2025
- [Physics-informed ML for prostate cancer DTs](https://www.nature.com/articles/s41746-025-01890-x) - npj Digital Medicine 2025
- [MONAI: Medical Open Network for AI](https://github.com/Project-MONAI/MONAI)

---

*See `tumor_twin_pipeline.py` for the complete implementation.*
