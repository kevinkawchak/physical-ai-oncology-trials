# MuJoCo Integration Guide for Oncology Robotics

*High-fidelity physics simulation with MuJoCo 3.2+ and MJX (October 2025 - January 2026)*

---

## Overview

MuJoCo (Multi-Joint dynamics with Contact) provides:
- **Accurate physics**: Best-in-class contact dynamics
- **MJX**: JAX-based GPU acceleration
- **MuJoCo Playground**: Ready-to-train robot environments
- **ROS 2 integration**: Seamless deployment pipeline

---

## Installation

### Standard MuJoCo

```bash
# Install MuJoCo
pip install mujoco

# Verify installation
python -c "import mujoco; print(mujoco.__version__)"
# Expected: 3.2.7 (January 2026)
```

### MJX (GPU-Accelerated)

```bash
# Install JAX with CUDA support
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install MJX
pip install mujoco-mjx

# Verify MJX
python -c "from mujoco import mjx; print('MJX available')"
```

---

## Surgical Robot Models

### Creating a dVRK-Style Model

```xml
<!-- dvrk_psm.xml -->
<mujoco model="dVRK_PSM">
  <compiler angle="radian" autolimits="true"/>

  <option timestep="0.002" integrator="implicitfast"/>

  <default>
    <joint armature="0.01" damping="0.5"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
    <geom friction="0.8 0.02 0.01"/>
  </default>

  <asset>
    <mesh name="base" file="meshes/base.stl"/>
    <mesh name="link1" file="meshes/link1.stl"/>
    <!-- Additional meshes -->
  </asset>

  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom type="mesh" mesh="base"/>

      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
        <geom type="mesh" mesh="link1"/>

        <body name="link2" pos="0.2 0 0">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-1.0 1.0"/>
          <!-- Continue kinematic chain -->

          <!-- End effector with needle driver -->
          <body name="ee_link" pos="0.1 0 0">
            <joint name="grip" type="slide" axis="0 0 1" range="0 0.01"/>
            <geom name="jaw1" type="box" size="0.02 0.005 0.001"/>
            <geom name="jaw2" type="box" size="0.02 0.005 0.001" pos="0 0 0.01"/>

            <!-- Force/torque sensor -->
            <site name="ft_sensor" pos="0 0 0"/>
          </body>
        </body>
      </body>
    </body>

    <!-- Tissue phantom -->
    <body name="tissue" pos="0.3 0 0.4">
      <geom name="tissue_surface" type="box" size="0.1 0.1 0.02"
            rgba="0.9 0.7 0.7 1" solref="0.01 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="joint1" name="motor1" gear="50"/>
    <motor joint="joint2" name="motor2" gear="50"/>
    <motor joint="grip" name="motor_grip" gear="20"/>
  </actuator>

  <sensor>
    <force name="ee_force" site="ft_sensor"/>
    <torque name="ee_torque" site="ft_sensor"/>
  </sensor>
</mujoco>
```

### Soft Tissue Modeling

```xml
<!-- soft_tissue.xml -->
<mujoco>
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <worldbody>
    <!-- Deformable tissue using composite -->
    <composite type="grid" count="10 10 3" spacing="0.01"
               prefix="tissue" flatinertia="1">
      <joint kind="main" damping="0.001"/>
      <geom type="sphere" size="0.005" rgba="0.9 0.6 0.6 1"
            solref="0.005 1" mass="0.001"/>
      <plugin plugin="mujoco.elasticity.cable">
        <config key="stiffness" value="100"/>
        <config key="damping" value="0.1"/>
      </plugin>
    </composite>
  </worldbody>
</mujoco>
```

---

## Training with MJX

### GPU-Accelerated Environment

```python
# mjx_surgical_env.py
import jax
import jax.numpy as jnp
from mujoco import mjx
import mujoco

class MJXSurgicalEnv:
    """GPU-accelerated surgical environment using MJX."""

    def __init__(self, model_path: str, num_envs: int = 4096):
        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Convert to MJX for GPU
        self.mjx_model = mjx.put_model(self.mj_model)

        # Batch environments
        self.num_envs = num_envs
        self.batch_data = self._create_batch_data()

    def _create_batch_data(self):
        """Create batched simulation data."""
        data = mjx.put_data(self.mj_model, self.mj_data)
        return jax.vmap(lambda _: data)(jnp.arange(self.num_envs))

    @jax.jit
    def step(self, actions):
        """Parallel step across all environments."""
        def single_step(data, action):
            # Apply action
            data = data.replace(ctrl=action)

            # Step physics
            data = mjx.step(self.mjx_model, data)

            # Compute observation
            obs = self._get_obs(data)

            # Compute reward
            reward = self._compute_reward(data)

            # Check termination
            done = self._check_done(data)

            return data, obs, reward, done

        self.batch_data, obs, rewards, dones = jax.vmap(single_step)(
            self.batch_data, actions
        )
        return obs, rewards, dones, {}

    def _get_obs(self, data):
        """Extract observation from simulation state."""
        return jnp.concatenate([
            data.qpos,  # Joint positions
            data.qvel,  # Joint velocities
            data.sensordata  # Force/torque sensor
        ])

    def _compute_reward(self, data):
        """Compute task reward."""
        # Distance to target
        ee_pos = data.xpos[self.ee_body_id]
        target_pos = self.target_position
        distance = jnp.linalg.norm(ee_pos - target_pos)

        # Force penalty
        contact_force = jnp.linalg.norm(data.sensordata[:3])
        force_penalty = jnp.maximum(contact_force - 2.0, 0.0) * 0.1

        return -distance - force_penalty

    def reset(self, key=None):
        """Reset all environments."""
        if key is None:
            key = jax.random.PRNGKey(0)

        # Randomize initial states
        keys = jax.random.split(key, self.num_envs)
        self.batch_data = jax.vmap(self._reset_single)(keys)

        return self._get_batch_obs()
```

### Training Pipeline

```python
# train_mjx.py
from flax import linen as nn
from flax.training import train_state
import optax

class PolicyNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        return mean, log_std

def train_surgical_policy():
    # Create environment
    env = MJXSurgicalEnv("dvrk_psm.xml", num_envs=4096)

    # Initialize policy
    policy = PolicyNetwork(action_dim=env.action_dim)
    params = policy.init(jax.random.PRNGKey(0), jnp.zeros(env.obs_dim))

    # Optimizer
    tx = optax.adam(3e-4)
    state = train_state.TrainState.create(
        apply_fn=policy.apply,
        params=params,
        tx=tx
    )

    # Training loop
    for iteration in range(1000):
        # Collect rollouts (all vectorized on GPU)
        trajectories = collect_rollouts(env, state, steps=256)

        # Compute advantages
        advantages = compute_gae(trajectories)

        # PPO update
        state = ppo_update(state, trajectories, advantages)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Reward: {trajectories.rewards.mean()}")

    return state
```

---

## MuJoCo Playground Integration

### Using Pre-Built Environments

```python
# playground_integration.py
from mujoco_playground import registry

# List available environments
available_envs = registry.list_environments()
print(available_envs)
# ['humanoid_walk', 'quadruped_run', 'arm_reach', ...]

# Create environment
env = registry.make(
    "arm_reach",
    num_envs=1024,
    backend="mjx"  # GPU-accelerated
)

# Run with trained policy
policy = load_policy("checkpoints/arm_reach.pkl")
obs = env.reset()

for step in range(1000):
    actions = policy(obs)
    obs, rewards, dones, info = env.step(actions)
```

---

## ROS 2 Integration

### MuJoCo ROS2 Control

```python
# mujoco_ros2_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
import mujoco

class MuJoCoSimulator(Node):
    def __init__(self):
        super().__init__('mujoco_simulator')

        # Load model
        self.model = mujoco.MjModel.from_xml_path('surgical_robot.xml')
        self.data = mujoco.MjData(self.model)

        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.wrench_pub = self.create_publisher(WrenchStamped, 'ee_wrench', 10)

        # Subscribers
        self.cmd_sub = self.create_subscription(
            JointState, 'joint_commands', self.command_callback, 10
        )

        # Simulation timer (500 Hz)
        self.timer = self.create_timer(0.002, self.simulation_step)

    def simulation_step(self):
        # Step physics
        mujoco.mj_step(self.model, self.data)

        # Publish joint states
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.position = self.data.qpos.tolist()
        js_msg.velocity = self.data.qvel.tolist()
        self.joint_pub.publish(js_msg)

        # Publish force/torque
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_msg.wrench.force.x = self.data.sensordata[0]
        wrench_msg.wrench.force.y = self.data.sensordata[1]
        wrench_msg.wrench.force.z = self.data.sensordata[2]
        self.wrench_pub.publish(wrench_msg)

    def command_callback(self, msg):
        # Apply joint commands
        self.data.ctrl[:] = msg.position

def main():
    rclpy.init()
    node = MuJoCoSimulator()
    rclpy.spin(node)
    rclpy.shutdown()
```

---

## Sim-to-Real Validation

### Physics Parameter Identification

```python
# system_identification.py
import mujoco
import numpy as np
from scipy.optimize import minimize

def identify_tissue_parameters(real_data, model_path):
    """Identify tissue stiffness from real force-displacement data."""

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    def simulation_error(params):
        # Update model parameters
        model.geom_solref[tissue_geom_id, 0] = params[0]  # Stiffness
        model.geom_solref[tissue_geom_id, 1] = params[1]  # Damping

        # Run simulation
        sim_forces = []
        for displacement in real_data['displacements']:
            data.ctrl[0] = displacement
            mujoco.mj_step(model, data)
            sim_forces.append(data.sensordata[2])  # Z-force

        # Compute error
        error = np.mean((np.array(sim_forces) - real_data['forces'])**2)
        return error

    # Optimize parameters
    result = minimize(
        simulation_error,
        x0=[0.01, 1.0],  # Initial guess
        method='Nelder-Mead'
    )

    return result.x  # Identified parameters
```

---

## Best Practices

1. **Use MJX for training** (4096+ parallel environments)
2. **Use standard MuJoCo for validation** (accurate single-environment physics)
3. **Model soft tissue with composites** for deformable interactions
4. **Tune contact parameters** (solref, solimp) against real measurements
5. **Use implicit integrator** for stiff contact problems
6. **Export to ONNX** for deployment (MuJoCo models run on CPU)

---

## Resources

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MJX Tutorial](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [MuJoCo Playground](https://playground.mujoco.org/)
- [MuJoCo ROS2 Control](https://github.com/ros-controls/ros2_control_demos)

---

*Last updated: January 2026*
