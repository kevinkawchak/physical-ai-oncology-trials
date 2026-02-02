# Gazebo Integration Guide for Oncology Robotics

*ROS 2 integrated simulation with Gazebo Sim 10.0 (Jetty) (February 2026)*

**Sources:**
- Gazebo Sim 10.0.0 (Jetty) (Oct 14, 2024): https://github.com/gazebosim/gz-sim/releases/tag/gz-sim10_10.0.0
- Gazebo Documentation: https://gazebosim.org/docs/jetty/
- ROS 2 Kilted Kaiju (May 2025): https://docs.ros.org/en/kilted/

---

## Overview

Gazebo Sim (Jetty) provides:
- **Native ROS 2 integration**: Seamless message passing with ROS 2 Jazzy/Kilted
- **Sensor simulation**: Cameras, depth, force/torque
- **Large ecosystem**: Extensive model and plugin library
- **Medical robotics support**: dVRK, surgical tools

---

## Installation

### Gazebo Sim with ROS 2 Jazzy or Kilted

```bash
# Option 1: Install ROS 2 Jazzy (Ubuntu 24.04)
sudo apt update
sudo apt install ros-jazzy-desktop
sudo apt install ros-jazzy-ros-gz ros-jazzy-ros-gz-bridge ros-jazzy-ros-gz-sim

# Option 2: Install ROS 2 Kilted Kaiju (Ubuntu 24.04, May 2025+)
# See: https://docs.ros.org/en/kilted/Installation.html

# Install Gazebo Sim standalone (optional, for latest version)
# See: https://gazebosim.org/docs/jetty/install

# Verify installation
source /opt/ros/jazzy/setup.bash  # or /opt/ros/kilted/setup.bash
gz sim --version
# Expected: Gazebo Sim, version 10.x (Jetty)
```

---

## Surgical Robot Simulation

### URDF/SDF Model

```xml
<!-- surgical_robot.sdf -->
<?xml version="1.0"?>
<sdf version="1.9">
  <model name="surgical_arm">
    <link name="base_link">
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz>
        </inertia>
      </inertial>
      <visual name="base_visual">
        <geometry><mesh><uri>meshes/base.dae</uri></mesh></geometry>
      </visual>
      <collision name="base_collision">
        <geometry><mesh><uri>meshes/base_collision.stl</uri></mesh></geometry>
      </collision>
    </link>

    <link name="link1">
      <!-- Link definition -->
    </link>

    <joint name="joint1" type="revolute">
      <parent>base_link</parent>
      <child>link1</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>100</effort>
          <velocity>2.0</velocity>
        </limit>
        <dynamics>
          <damping>0.5</damping>
          <friction>0.1</friction>
        </dynamics>
      </axis>
    </joint>

    <!-- End effector with force sensor -->
    <link name="ee_link">
      <sensor name="ft_sensor" type="force_torque">
        <always_on>true</always_on>
        <update_rate>500</update_rate>
        <force_torque>
          <frame>child</frame>
          <measure_direction>child_to_parent</measure_direction>
        </force_torque>
      </sensor>
    </link>

    <!-- Camera (endoscopic view) -->
    <link name="camera_link">
      <sensor name="endoscope" type="camera">
        <always_on>true</always_on>
        <update_rate>30</update_rate>
        <camera>
          <horizontal_fov>1.2</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.01</near>
            <far>1.0</far>
          </clip>
        </camera>
      </sensor>
    </link>

    <!-- ROS 2 control plugin -->
    <plugin filename="gz-sim-joint-state-publisher-system"
            name="gz::sim::systems::JointStatePublisher"/>

    <plugin filename="gz-sim-joint-position-controller-system"
            name="gz::sim::systems::JointPositionController">
      <joint_name>joint1</joint_name>
      <topic>joint1_cmd</topic>
      <p_gain>100</p_gain>
      <i_gain>0.1</i_gain>
      <d_gain>10</d_gain>
    </plugin>
  </model>
</sdf>
```

### World Configuration

```xml
<!-- surgical_world.sdf -->
<?xml version="1.0"?>
<sdf version="1.9">
  <world name="surgical_environment">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Lighting (surgical OR) -->
    <light type="directional" name="surgical_light">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 2 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <direction>0 0 -1</direction>
    </light>

    <!-- Operating table -->
    <include>
      <uri>model://operating_table</uri>
      <pose>0 0 0.8 0 0 0</pose>
    </include>

    <!-- Tissue phantom -->
    <model name="tissue_phantom">
      <static>false</static>
      <link name="tissue">
        <inertial>
          <mass>0.5</mass>
        </inertial>
        <visual name="tissue_visual">
          <geometry><box><size>0.2 0.2 0.05</size></box></geometry>
          <material>
            <ambient>0.9 0.6 0.6 1</ambient>
          </material>
        </visual>
        <collision name="tissue_collision">
          <geometry><box><size>0.2 0.2 0.05</size></box></geometry>
          <surface>
            <contact>
              <ode>
                <soft_cfm>0.001</soft_cfm>
                <soft_erp>0.9</soft_erp>
              </ode>
            </contact>
          </surface>
        </collision>
      </link>
    </model>

    <!-- Include surgical robot -->
    <include>
      <uri>model://surgical_arm</uri>
      <pose>0 -0.3 0.8 0 0 0</pose>
    </include>
  </world>
</sdf>
```

---

## ROS 2 Integration

### Launch File

```python
# surgical_sim_launch.py
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Start Gazebo with surgical world
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', 'surgical_world.sdf'],
            output='screen'
        ),

        # ROS-Gazebo bridge
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
                '/endoscope/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/ft_sensor@geometry_msgs/msg/WrenchStamped[gz.msgs.Wrench',
                '/joint1_cmd@std_msgs/msg/Float64]gz.msgs.Double',
            ],
            output='screen'
        ),

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}]
        ),

        # Surgical controller node
        Node(
            package='surgical_control',
            executable='surgical_controller',
            output='screen'
        ),
    ])
```

### Controller Node

```python
# surgical_controller.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64
import numpy as np

class SurgicalController(Node):
    def __init__(self):
        super().__init__('surgical_controller')

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/endoscope/image', self.image_callback, 10
        )
        self.ft_sub = self.create_subscription(
            WrenchStamped, '/ft_sensor', self.ft_callback, 10
        )

        # Publishers
        self.joint_cmd_pub = self.create_publisher(Float64, '/joint1_cmd', 10)

        # State
        self.current_joints = None
        self.current_force = None
        self.current_image = None

        # Control loop (50 Hz)
        self.timer = self.create_timer(0.02, self.control_loop)

        # Load policy
        self.policy = self.load_policy('trained_policy.onnx')

    def control_loop(self):
        if self.current_joints is None:
            return

        # Prepare observation
        obs = self.prepare_observation()

        # Get action from policy
        action = self.policy.predict(obs)

        # Apply force limits
        if self.current_force is not None:
            force_magnitude = np.linalg.norm([
                self.current_force.wrench.force.x,
                self.current_force.wrench.force.y,
                self.current_force.wrench.force.z
            ])
            if force_magnitude > 5.0:  # N
                action = action * 0.5  # Reduce speed

        # Publish command
        cmd = Float64()
        cmd.data = float(action[0])
        self.joint_cmd_pub.publish(cmd)

    def joint_callback(self, msg):
        self.current_joints = msg

    def image_callback(self, msg):
        self.current_image = msg

    def ft_callback(self, msg):
        self.current_force = msg

def main():
    rclpy.init()
    node = SurgicalController()
    rclpy.spin(node)
    rclpy.shutdown()
```

---

## ROS 4 Healthcare Integration

### Using ros4healthcare Framework

```python
# ros4healthcare_integration.py
from ros4healthcare import HealthcareRobotNode
from ros4healthcare.sensors import PhysiologicalSensor
from ros4healthcare.safety import SafetyMonitor

class OncologyRobotNode(HealthcareRobotNode):
    def __init__(self):
        super().__init__('oncology_robot')

        # Add physiological monitoring
        self.vital_sensor = PhysiologicalSensor(
            sensor_types=['heart_rate', 'blood_pressure', 'spo2']
        )

        # Add safety monitoring
        self.safety_monitor = SafetyMonitor(
            force_threshold=5.0,  # N
            velocity_threshold=0.1,  # m/s
            patient_proximity_threshold=0.5  # m
        )

    def on_safety_violation(self, violation):
        """Handle safety violations."""
        self.get_logger().warn(f"Safety violation: {violation}")
        self.emergency_stop()

    def execute_procedure(self, procedure_plan):
        """Execute surgical procedure with safety monitoring."""
        for step in procedure_plan:
            # Check safety before each step
            if not self.safety_monitor.is_safe():
                self.pause_procedure()
                continue

            # Execute step
            result = self.execute_step(step)

            # Log for audit trail
            self.log_procedure_step(step, result)
```

---

## Sim-to-Real Pipeline

### Export Trained Policy

```python
# export_policy.py
import torch

def export_for_gazebo(policy_path, output_path):
    """Export policy for Gazebo ROS 2 deployment."""

    # Load trained policy
    policy = torch.load(policy_path)

    # Export to ONNX
    dummy_input = torch.randn(1, obs_dim)
    torch.onnx.export(
        policy,
        dummy_input,
        output_path,
        input_names=['observation'],
        output_names=['action'],
        opset_version=17
    )

    # Also export as ROS 2 parameter file
    with open(output_path.replace('.onnx', '.yaml'), 'w') as f:
        f.write(f"""
surgical_controller:
  ros__parameters:
    policy_path: {output_path}
    control_frequency: 50.0
    force_limit: 5.0
    velocity_limit: 0.1
""")

export_for_gazebo('checkpoints/best.pt', 'policies/surgical.onnx')
```

---

## Best Practices

1. **Use ros_gz_bridge** for efficient ROS-Gazebo communication
2. **Set physics timestep ≤ 1ms** for accurate contact dynamics
3. **Use sensors sparingly** to maintain real-time performance
4. **Validate sensor noise models** against real hardware
5. **Integrate ros4healthcare** for standardized medical robotics
6. **Log all actions** for regulatory compliance

---

## Resources

- [Gazebo Ionic Documentation](https://gazebosim.org/docs/ionic)
- [ros_gz Repository](https://github.com/gazebosim/ros_gz)
- [ROS 4 Healthcare](https://github.com/SCAI-Lab/ros4healthcare)
- [ROS Medical Robotics](https://rosmed.github.io/)

---

*Last updated: January 2026*
