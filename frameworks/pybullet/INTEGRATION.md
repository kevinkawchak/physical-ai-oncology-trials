# PyBullet Integration Guide for Oncology Robotics

*Rapid prototyping and RL development with PyBullet 3.2.5 (February 2026)*

**Sources:**
- PyBullet v3.2.5 (Apr 24, 2023): https://github.com/bulletphysics/bullet3/releases/tag/3.2.5
- PyBullet Quickstart: https://pybullet.org/wordpress/

---

## Overview

PyBullet provides:
- **Easy setup**: Pure Python, minimal dependencies
- **Fast prototyping**: Quick iteration on robot tasks
- **RL compatibility**: Gym/Gymnasium interface
- **Soft body simulation**: Deformable objects for tissue

---

## Installation

```bash
# Install PyBullet
pip install pybullet

# Verify installation
python -c "import pybullet as p; print(p.getAPIVersion())"
# Expected: 3.2.5 (April 2023)

# Optional: Install stable-baselines3 for RL
pip install stable-baselines3[extra]
```

---

## Surgical Environment

### Basic Setup

```python
# surgical_env.py
import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SurgicalEnv(gym.Env):
    """Gymnasium environment for surgical robotics."""

    def __init__(self, render_mode=None):
        super().__init__()

        # Connect to physics server
        if render_mode == "human":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32
        )

        # Load environment
        self._load_environment()

    def _load_environment(self):
        """Load surgical scene."""
        # Physics parameters
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)  # 240 Hz physics

        # Ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Operating table
        table_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.3, 0.4])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.3, 0.4],
                                           rgbaColor=[0.7, 0.7, 0.7, 1])
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_shape,
            baseVisualShapeIndex=table_visual,
            basePosition=[0, 0, 0.4]
        )

        # Surgical robot (using Kuka as proxy)
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, -0.5, 0.8],
            useFixedBase=True
        )

        # Get joint info
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = list(range(self.num_joints))

        # End effector link
        self.ee_link = self.num_joints - 1

        # Tissue phantom
        self._create_tissue()

    def _create_tissue(self):
        """Create deformable tissue phantom."""
        tissue_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 0.02]
        )
        tissue_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 0.02],
            rgbaColor=[0.9, 0.6, 0.6, 1]
        )
        self.tissue_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=tissue_shape,
            baseVisualShapeIndex=tissue_visual,
            basePosition=[0, 0, 0.85]
        )

        # Make tissue soft (lateral friction for sliding resistance)
        p.changeDynamics(
            self.tissue_id, -1,
            lateralFriction=0.8,
            spinningFriction=0.1,
            rollingFriction=0.01,
            restitution=0.1
        )

    def step(self, action):
        """Execute action and return observation."""
        # Scale action to joint velocities
        target_velocities = action * 0.5  # Max 0.5 rad/s

        # Apply action
        for i, joint in enumerate(self.joint_indices[:7]):
            p.setJointMotorControl2(
                self.robot_id, joint,
                p.VELOCITY_CONTROL,
                targetVelocity=target_velocities[i],
                force=100
            )

        # Step simulation
        p.stepSimulation()

        # Get observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        terminated = self._check_termination()
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        """Get current observation."""
        # Robot state
        joint_states = p.getJointStates(self.robot_id, self.joint_indices[:7])
        joint_positions = np.array([s[0] for s in joint_states])
        joint_velocities = np.array([s[1] for s in joint_states])

        # End effector pose
        ee_state = p.getLinkState(self.robot_id, self.ee_link)
        ee_pos = np.array(ee_state[0])
        ee_orn = np.array(ee_state[1])

        # Target position
        target_pos = np.array([0, 0, 0.9])

        # Contact force
        contact_force = self._get_contact_force()

        return np.concatenate([
            joint_positions,      # 7
            joint_velocities,     # 7
            ee_pos,               # 3
            ee_orn,               # 4
            target_pos - ee_pos,  # 3
            [contact_force]       # 1
        ]).astype(np.float32)

    def _get_contact_force(self):
        """Get contact force on end effector."""
        contacts = p.getContactPoints(self.robot_id, self.tissue_id)
        if contacts:
            return np.linalg.norm([c[9] for c in contacts])  # Normal force
        return 0.0

    def _compute_reward(self):
        """Compute task reward."""
        ee_state = p.getLinkState(self.robot_id, self.ee_link)
        ee_pos = np.array(ee_state[0])
        target_pos = np.array([0, 0, 0.9])

        # Distance reward
        distance = np.linalg.norm(ee_pos - target_pos)
        distance_reward = -distance

        # Force penalty
        force = self._get_contact_force()
        force_penalty = -0.1 * max(force - 2.0, 0)

        # Efficiency penalty
        time_penalty = -0.001

        return distance_reward + force_penalty + time_penalty

    def _check_termination(self):
        """Check if episode should terminate."""
        ee_state = p.getLinkState(self.robot_id, self.ee_link)
        ee_pos = np.array(ee_state[0])
        target_pos = np.array([0, 0, 0.9])

        # Success condition
        if np.linalg.norm(ee_pos - target_pos) < 0.01:
            return True

        # Safety violation
        force = self._get_contact_force()
        if force > 10.0:
            return True

        return False

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)

        # Reset robot to initial position
        for i, joint in enumerate(self.joint_indices[:7]):
            p.resetJointState(self.robot_id, joint, 0.0)

        # Randomize tissue position
        if seed is not None:
            np.random.seed(seed)
        tissue_pos = [
            0.0 + np.random.uniform(-0.05, 0.05),
            0.0 + np.random.uniform(-0.05, 0.05),
            0.85
        ]
        p.resetBasePositionAndOrientation(
            self.tissue_id, tissue_pos, [0, 0, 0, 1]
        )

        return self._get_observation(), {}

    def render(self):
        """Render is handled by GUI mode."""
        pass

    def close(self):
        """Disconnect from physics server."""
        p.disconnect(self.client)
```

---

## Training with Stable-Baselines3

```python
# train_surgical.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

def make_env(rank):
    def _init():
        env = SurgicalEnv()
        return env
    return _init

def train():
    # Create vectorized environment
    num_envs = 8
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # Create evaluation environment
    eval_env = SurgicalEnv(render_mode="human")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./checkpoints/',
        log_path='./logs/',
        eval_freq=10000,
        n_eval_episodes=5
    )

    # Create and train model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=eval_callback
    )

    model.save("surgical_policy")

if __name__ == "__main__":
    train()
```

---

## Soft Body Simulation

### Deformable Tissue

```python
# deformable_tissue.py
import pybullet as p

def create_deformable_tissue(position=[0, 0, 0.9]):
    """Create deformable tissue using soft body."""

    # Load soft body
    tissue_id = p.loadSoftBody(
        "tissue.vtk",  # Volumetric mesh
        basePosition=position,
        scale=0.1,
        mass=0.1,
        useNeoHookean=1,
        NeoHookeanMu=50,      # Shear modulus
        NeoHookeanLambda=100,  # Bulk modulus
        NeoHookeanDamping=0.01,
        useSelfCollision=1,
        frictionCoeff=0.5,
        useFaceContact=1
    )

    return tissue_id

def create_cloth_tissue(position=[0, 0, 0.9]):
    """Create tissue as cloth for surface interaction."""

    # Create cloth
    cloth_id = p.loadSoftBody(
        "cloth_z_up.obj",
        basePosition=position,
        scale=0.2,
        mass=0.1,
        useBendingSprings=1,
        useMassSpring=1,
        springElasticStiffness=100,
        springDampingStiffness=0.1,
        useSelfCollision=1
    )

    return cloth_id
```

---

## Camera and Sensors

### Endoscopic Camera

```python
# camera_setup.py
import pybullet as p
import numpy as np

class EndoscopicCamera:
    def __init__(self, robot_id, ee_link):
        self.robot_id = robot_id
        self.ee_link = ee_link

        # Camera parameters
        self.width = 640
        self.height = 480
        self.fov = 60
        self.aspect = self.width / self.height
        self.near = 0.01
        self.far = 1.0

        # Projection matrix
        self.projection_matrix = p.computeProjectionMatrixFOV(
            self.fov, self.aspect, self.near, self.far
        )

    def get_image(self):
        """Get camera image from end effector pose."""
        # Get EE pose
        ee_state = p.getLinkState(self.robot_id, self.ee_link)
        ee_pos = ee_state[0]
        ee_orn = ee_state[1]

        # Compute camera pose (looking along EE z-axis)
        rot_matrix = p.getMatrixFromQuaternion(ee_orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        camera_pos = np.array(ee_pos)
        target_pos = camera_pos + rot_matrix[:, 2] * 0.1  # Look forward
        up_vector = -rot_matrix[:, 1]  # Up is -Y in EE frame

        view_matrix = p.computeViewMatrix(
            camera_pos.tolist(),
            target_pos.tolist(),
            up_vector.tolist()
        )

        # Render
        _, _, rgb, depth, segmentation = p.getCameraImage(
            self.width, self.height,
            view_matrix, self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        return rgb, depth, segmentation


class ForceSensor:
    def __init__(self, robot_id, ee_link):
        self.robot_id = robot_id
        self.ee_link = ee_link

        # Enable force sensing
        p.enableJointForceTorqueSensor(robot_id, ee_link, True)

    def get_force_torque(self):
        """Get force/torque at end effector."""
        joint_state = p.getJointState(self.robot_id, self.ee_link)
        force_torque = joint_state[2]  # (Fx, Fy, Fz, Tx, Ty, Tz)
        return np.array(force_torque)
```

---

## Domain Randomization

```python
# domain_randomization.py
import pybullet as p
import numpy as np

class DomainRandomizer:
    def __init__(self, env):
        self.env = env

    def randomize(self):
        """Apply domain randomization."""
        self._randomize_physics()
        self._randomize_visual()
        self._randomize_dynamics()

    def _randomize_physics(self):
        """Randomize physics parameters."""
        # Gravity variation
        gravity = -9.81 * np.random.uniform(0.95, 1.05)
        p.setGravity(0, 0, gravity)

        # Timestep variation
        timestep = 1/240 * np.random.uniform(0.9, 1.1)
        p.setTimeStep(timestep)

    def _randomize_visual(self):
        """Randomize visual appearance."""
        # Tissue color
        r = np.random.uniform(0.7, 1.0)
        g = np.random.uniform(0.4, 0.7)
        b = np.random.uniform(0.4, 0.7)
        p.changeVisualShape(self.env.tissue_id, -1, rgbaColor=[r, g, b, 1])

        # Lighting (if using OpenGL renderer)
        # Note: PyBullet has limited lighting control

    def _randomize_dynamics(self):
        """Randomize object dynamics."""
        # Tissue friction
        friction = np.random.uniform(0.5, 1.0)
        p.changeDynamics(self.env.tissue_id, -1, lateralFriction=friction)

        # Tissue mass
        mass = 0.1 * np.random.uniform(0.8, 1.2)
        p.changeDynamics(self.env.tissue_id, -1, mass=mass)

        # Robot joint damping
        for joint in self.env.joint_indices:
            damping = np.random.uniform(0.3, 0.7)
            p.changeDynamics(self.env.robot_id, joint, jointDamping=damping)
```

---

## Best Practices

1. **Use PyBullet for rapid prototyping**, then migrate to Isaac/MuJoCo for production
2. **Set timestep ≤ 1/240** for stable contact simulation
3. **Use DIRECT mode** for training (faster than GUI)
4. **Enable OpenGL renderer** for realistic camera images
5. **Soft bodies are expensive** - use sparingly
6. **Validate against real physics** before relying on results

---

## Resources

- [PyBullet Quickstart](https://pybullet.org/wordpress/)
- [PyBullet GitHub](https://github.com/bulletphysics/bullet3)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PyBullet Examples](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet)

---

*Last updated: January 2026*
