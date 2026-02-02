# NVIDIA Isaac Integration Guide for Oncology Robotics

*Production deployment guide for Isaac Lab 2.3.1 and Isaac Sim 5.0.0 (February 2026)*

**Sources:**
- Isaac Lab v2.3.1 (Dec 4, 2024): https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.1
- Isaac Sim 5.0.0: https://docs.isaacsim.omniverse.nvidia.com/
- Newton Physics Engine (Jan 2026): https://github.com/newton-physics/newton
- Isaac Lab-Arena (Jan 2026): https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena/

---

## Overview

NVIDIA Isaac provides the most comprehensive platform for physical AI development in oncology clinical trials, combining:
- **Isaac Lab 2.3.1**: GPU-accelerated robot learning framework
- **Isaac Lab-Arena**: Large-scale robot policy evaluation and benchmarking
- **Isaac Sim 5.0.0**: High-fidelity physics simulation
- **Newton Physics Engine**: Open-source GPU physics (NVIDIA/DeepMind/Disney, Linux Foundation)
- **Isaac for Healthcare**: Medical robotics-specific extensions (Johnson & Johnson partnership)
- **Omniverse**: Digital twin and synthetic data generation

---

## Quick Start

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 | RTX 4090 / L40 / Blackwell |
| VRAM | 12 GB | 24+ GB |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB SSD | 500 GB NVMe |
| OS | Ubuntu 22.04 | Ubuntu 22.04 / 24.04 |

### Installation

```bash
# 1. Install Isaac Sim (requires NVIDIA Omniverse)
# Download from: https://developer.nvidia.com/isaac-sim

# 2. Clone Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 3. Create conda environment
conda create -n isaaclab python=3.10
conda activate isaaclab

# 4. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# 5. Verify installation
python -c "import isaaclab; print(isaaclab.__version__)"
```

---

## Oncology Environment Setup

### Creating a Surgical Task Environment

```python
# surgical_env.py
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene
from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import Camera, ContactSensor

class SurgicalTaskEnv(DirectRLEnv):
    """Base environment for surgical robotics tasks."""

    def __init__(self, cfg):
        super().__init__(cfg)

        # Scene components
        self.robot = None
        self.tissue = None
        self.instruments = []

    def _setup_scene(self):
        """Configure surgical scene."""
        # Add surgical robot (dVRK-style)
        self.robot = Articulation(
            prim_path="/World/Robot",
            cfg=self.cfg.robot_cfg,
            init_state=self.cfg.robot_init_state
        )

        # Add deformable tissue
        self.tissue = DeformableObject(
            prim_path="/World/Tissue",
            cfg=self.cfg.tissue_cfg
        )

        # Add camera (endoscopic view)
        self.camera = Camera(
            prim_path="/World/Camera",
            cfg=CameraCfg(
                width=640,
                height=480,
                focal_length=4.0,
                horizontal_aperture=3.6
            )
        )

        # Add contact sensors
        self.contact_sensor = ContactSensor(
            prim_path="/World/Robot/ee_link",
            cfg=ContactSensorCfg(
                history_length=10,
                filter_prim_paths=["/World/Tissue"]
            )
        )

    def _get_observations(self):
        """Collect observations for policy."""
        obs = {
            "robot_state": self.robot.data.joint_pos,
            "robot_vel": self.robot.data.joint_vel,
            "ee_pos": self.robot.data.body_pos_w[:, self.ee_idx],
            "ee_quat": self.robot.data.body_quat_w[:, self.ee_idx],
            "contact_force": self.contact_sensor.data.net_forces_w,
            "image": self.camera.data.output["rgb"]
        }
        return obs

    def _compute_rewards(self):
        """Compute task-specific rewards."""
        # Task completion reward
        task_reward = self._check_task_completion()

        # Safety penalties
        force_penalty = torch.clamp(
            self.contact_sensor.data.net_forces_w.norm(dim=-1) - 5.0,
            min=0.0
        ) * 0.1

        # Efficiency reward
        time_penalty = 0.001

        return task_reward - force_penalty - time_penalty
```

### Configuration for Oncology Tasks

```python
# oncology_cfg.py
from dataclasses import dataclass
from isaaclab.envs import DirectRLEnvCfg

@dataclass
class NeedleInsertionEnvCfg(DirectRLEnvCfg):
    """Configuration for needle insertion task."""

    # Simulation
    sim_dt: float = 0.005  # 200 Hz physics
    control_dt: float = 0.02  # 50 Hz control
    decimation: int = 4

    # Robot configuration
    robot_cfg: ArticulationCfg = DVRK_CFG
    robot_init_state: ArticulationCfg.InitialStateCfg = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={"joint1": 0.0, "joint2": -0.5, ...}
    )

    # Tissue configuration
    tissue_cfg: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/Tissue",
        mesh_file="phantom_tissue.usd",
        material=DeformableMaterial(
            young_modulus=50000,  # Pa
            poisson_ratio=0.45,
            damping=0.01
        )
    )

    # Task parameters
    target_depth_mm: float = 15.0
    max_force_n: float = 2.0
    max_episode_length: int = 500

    # Reward weights
    reward_completion: float = 100.0
    reward_progress: float = 1.0
    penalty_force: float = 10.0
    penalty_time: float = 0.01
```

---

## Training Pipelines

### GPU-Accelerated PPO Training

```python
# train_surgical.py
from isaaclab.algos import PPO
from isaaclab.utils import get_device

def train_needle_insertion():
    # Environment setup
    env = NeedleInsertionEnv(
        cfg=NeedleInsertionEnvCfg(),
        num_envs=4096,  # Parallel environments
        device=get_device()
    )

    # Algorithm configuration
    ppo_cfg = PPOCfg(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=1.0,
        num_epochs=5,
        batch_size=8192,
        num_mini_batches=4
    )

    # Initialize algorithm
    agent = PPO(
        env=env,
        cfg=ppo_cfg,
        network_cfg=NetworkCfg(
            hidden_layers=[512, 256, 128],
            activation="elu"
        )
    )

    # Training loop
    agent.train(
        total_timesteps=10_000_000,
        log_interval=10,
        save_interval=100,
        checkpoint_dir="checkpoints/needle_insertion"
    )

if __name__ == "__main__":
    train_needle_insertion()
```

### Domain Randomization

```python
# domain_randomization.py
from isaaclab.utils import RandomizationCfg

oncology_randomization = RandomizationCfg(
    # Visual randomization
    lighting=RandomizationCfg.LightingCfg(
        intensity_range=(0.7, 1.3),
        color_temperature_range=(3000, 6500),
        position_noise=0.1
    ),

    # Physics randomization
    tissue_properties=RandomizationCfg.MaterialCfg(
        stiffness_range=(0.5, 2.0),  # Relative to nominal
        friction_range=(0.3, 0.8),
        damping_range=(0.8, 1.2)
    ),

    # Geometric randomization
    object_pose=RandomizationCfg.PoseCfg(
        position_noise_mm=5.0,
        orientation_noise_deg=5.0
    ),

    # Action randomization (for robustness)
    action=RandomizationCfg.ActionCfg(
        delay_range_ms=(0, 20),
        noise_std=0.01
    )
)

# Apply during training
env = NeedleInsertionEnv(
    cfg=NeedleInsertionEnvCfg(),
    randomization_cfg=oncology_randomization
)
```

---

## Isaac for Healthcare Integration

### Surgical Robot Models

```python
# Available surgical robot configurations
from isaaclab_healthcare.robots import (
    DVRKCfg,           # da Vinci Research Kit
    STARCfg,           # Smart Tissue Autonomous Robot
    UR5SurgicalCfg,    # UR5 with surgical end-effector
    FrankaSurgicalCfg  # Franka Emika for lab automation
)

# Load dVRK with surgical instruments
robot = Articulation(
    prim_path="/World/DVRK",
    cfg=DVRKCfg(
        end_effector="needle_driver",
        enable_force_torque_sensor=True,
        control_mode="impedance"
    )
)
```

### Pre-Built Surgical Environments

```python
# ORBIT-Surgical integration
from orbit_surgical.envs import (
    NeedlePickEnv,
    NeedleHandoverEnv,
    SuturingEnv,
    TissueRetractionEnv,
    GauzeCuttingEnv
)

# Create benchmark environment
env = NeedlePickEnv(
    num_envs=1024,
    device="cuda:0",
    headless=True
)
```

---

## Sim-to-Real Transfer

### Policy Export for Deployment

```python
# export_policy.py
import torch.onnx

def export_to_onnx(model, env, output_path):
    """Export trained policy to ONNX for edge deployment."""

    # Get sample observation
    obs = env.reset()

    # Export to ONNX
    torch.onnx.export(
        model.actor,
        obs,
        output_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch"},
            "action": {0: "batch"}
        },
        opset_version=17
    )

    print(f"Exported policy to {output_path}")

# Usage
export_to_onnx(
    model=trained_agent.policy,
    env=env,
    output_path="policies/needle_insertion.onnx"
)
```

### Deployment on Edge (Jetson/IGX)

```python
# edge_inference.py
import onnxruntime as ort
import numpy as np

class EdgePolicy:
    def __init__(self, onnx_path):
        # Load ONNX model
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, observation):
        """Run inference on observation."""
        obs_np = np.array(observation, dtype=np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np[np.newaxis, :]

        action = self.session.run(
            [self.output_name],
            {self.input_name: obs_np}
        )[0]

        return action[0]

# Deploy on robot
policy = EdgePolicy("policies/needle_insertion.onnx")
while running:
    obs = robot.get_observation()
    action = policy.predict(obs)
    robot.execute(action)
```

---

## Performance Optimization

### Multi-GPU Training

```python
# multi_gpu_train.py
from isaaclab.utils.distributed import setup_distributed

def train_distributed():
    # Setup distributed training
    rank, world_size = setup_distributed()

    # Create environment with proper GPU assignment
    env = NeedleInsertionEnv(
        cfg=NeedleInsertionEnvCfg(),
        num_envs=4096 // world_size,
        device=f"cuda:{rank}"
    )

    # Training with gradient synchronization
    agent = PPO(env=env, cfg=ppo_cfg, distributed=True)
    agent.train(total_timesteps=10_000_000)

# Launch with: torchrun --nproc_per_node=4 multi_gpu_train.py
```

### Memory Optimization

```python
# memory_efficient_training.py
from isaaclab.utils import MemoryEfficientCfg

mem_cfg = MemoryEfficientCfg(
    # Gradient checkpointing
    checkpoint_layers=True,

    # Mixed precision
    use_amp=True,
    amp_dtype="float16",

    # Replay buffer
    buffer_device="cpu",  # Keep buffer on CPU
    batch_prefetch=True,

    # Observation compression
    compress_images=True,
    image_dtype="uint8"
)
```

---

## Monitoring and Debugging

### TensorBoard Integration

```python
# Enable logging
agent = PPO(
    env=env,
    cfg=ppo_cfg,
    logger_cfg=LoggerCfg(
        log_dir="runs/needle_insertion",
        log_to_tensorboard=True,
        log_to_wandb=True,
        video_interval=1000
    )
)

# Launch TensorBoard
# tensorboard --logdir runs/
```

### Visualization

```python
# Render trained policy
from isaaclab.utils import visualize_policy

visualize_policy(
    env=env,
    policy=trained_agent.policy,
    num_episodes=10,
    record_video=True,
    output_path="videos/needle_insertion.mp4"
)
```

---

## Newton Physics Engine Integration

The Newton Physics Engine (January 2026) provides next-generation GPU-accelerated physics for robotics:

```python
# Newton integration with Isaac Lab (feature/newton branch)
from isaaclab.physics import NewtonPhysics

# Enable Newton for contact-rich manipulation
newton_config = NewtonPhysics(
    solver="implicit",
    gpu_accelerated=True,
    differentiable=True,  # Gradient propagation through simulation
    contact_model="newton_contact"
)

env = SurgicalEnv(
    physics_engine=newton_config,
    num_envs=4096
)
```

**Key Newton Features:**
- Differentiable physics for gradient-based optimization
- Contact-rich behaviors (walking on soft surfaces, manipulating delicate objects)
- Co-developed by NVIDIA, Google DeepMind, and Disney Research
- Managed by Linux Foundation for vendor neutrality

---

## Isaac Lab-Arena for Policy Evaluation

Isaac Lab-Arena provides large-scale benchmarking for generalist robot policies:

```python
# Benchmark surgical policies at scale
from isaac_lab_arena import BenchmarkRunner

runner = BenchmarkRunner(
    benchmarks=["libero", "robocasa", "surgical_tasks"],
    policy_path="policies/surgical_assistant.onnx"
)

results = runner.evaluate(
    num_episodes=10000,
    parallel_envs=4096,
    metrics=["success_rate", "completion_time", "safety_violations"]
)
```

---

## Best Practices for Oncology

1. **Start with pre-built ORBIT-Surgical environments** before creating custom
2. **Use extensive domain randomization** for robust sim-to-real transfer
3. **Validate physics parameters** against real tissue measurements
4. **Export to ONNX** for deployment on NVIDIA edge platforms
5. **Monitor force/contact metrics** to ensure safe behaviors
6. **Use hierarchical policies** for complex multi-step procedures
7. **Leverage Newton** for differentiable physics and contact-rich tasks
8. **Use Isaac Lab-Arena** for standardized policy benchmarking

---

## Resources

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLabArena)
- [Newton Physics Engine](https://github.com/newton-physics/newton)
- [ORBIT-Surgical GitHub](https://github.com/orbit-surgical/orbit-surgical)
- [Isaac for Healthcare Blog](https://developer.nvidia.com/blog/introducing-nvidia-isaac-for-healthcare/)
- [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/)

---

*Last updated: January 2026*
