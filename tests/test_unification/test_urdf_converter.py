"""Tests for URDF/SDF/MJCF Converter.

Validates Link/Joint/RobotModel dataclasses, URDF parsing,
MJCF/SDF generation, and the UnifiedModelConverter workflow.

LICENSE: MIT
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
MOD_PATH = "unification/simulation_physics/urdf_sdf_mjcf_converter.py"


@pytest.fixture()
def conv_mod():
    return load_module("urdf_sdf_mjcf_converter", MOD_PATH)


# ===========================================================================
# TestLink
# ===========================================================================


class TestLink:
    def test_default_values(self, conv_mod):
        link = conv_mod.Link(name="base_link")
        assert link.mass == 1.0
        assert link.visual_geometry is None
        np.testing.assert_array_almost_equal(link.com_position, np.zeros(3))

    def test_custom_values(self, conv_mod):
        link = conv_mod.Link(name="arm_link", mass=2.5, visual_geometry={"type": "box", "size": [0.1, 0.1, 0.5]})
        assert link.mass == 2.5
        assert link.visual_geometry["type"] == "box"


# ===========================================================================
# TestJoint
# ===========================================================================


class TestJoint:
    def test_default_values(self, conv_mod):
        j = conv_mod.Joint(name="j1", joint_type="revolute", parent="link0", child="link1")
        assert j.limit_lower == -np.pi
        assert j.limit_upper == np.pi
        assert j.damping == 0.1
        assert j.friction == 0.0

    def test_prismatic_joint(self, conv_mod):
        j = conv_mod.Joint(
            name="j_slide",
            joint_type="prismatic",
            parent="link0",
            child="link1",
            limit_lower=0.0,
            limit_upper=0.5,
        )
        assert j.joint_type == "prismatic"
        assert j.limit_upper == 0.5


# ===========================================================================
# TestRobotModel
# ===========================================================================


class TestRobotModel:
    def test_default_values(self, conv_mod):
        model = conv_mod.RobotModel(name="test_robot")
        assert model.links == []
        assert model.joints == []
        assert model.source_format == "unknown"

    def test_with_links_and_joints(self, conv_mod):
        link = conv_mod.Link(name="base")
        joint = conv_mod.Joint(name="j1", joint_type="fixed", parent="world", child="base")
        model = conv_mod.RobotModel(name="robot", links=[link], joints=[joint])
        assert len(model.links) == 1
        assert len(model.joints) == 1


# ===========================================================================
# TestURDFParser
# ===========================================================================


class TestURDFParser:
    def _make_urdf(self, tmp_path):
        """Create a minimal URDF file for testing."""
        urdf = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.2 0.2 0.4"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.4"/>
      </geometry>
    </collision>
  </link>
  <link name="arm_link">
    <inertial>
      <mass value="2.0"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <axis xyz="0 0 1"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>
</robot>"""
        path = tmp_path / "test_robot.urdf"
        path.write_text(urdf)
        return str(path)

    def test_parse_urdf(self, conv_mod, tmp_path):
        path = self._make_urdf(tmp_path)
        model = conv_mod.URDFParser.parse(path)
        assert model.name == "test_robot"
        assert model.source_format == "urdf"

    def test_parse_links(self, conv_mod, tmp_path):
        path = self._make_urdf(tmp_path)
        model = conv_mod.URDFParser.parse(path)
        assert len(model.links) == 2
        base = [l for l in model.links if l.name == "base_link"][0]
        assert base.mass == 5.0

    def test_parse_joint_limits(self, conv_mod, tmp_path):
        path = self._make_urdf(tmp_path)
        model = conv_mod.URDFParser.parse(path)
        j = model.joints[0]
        assert j.name == "joint1"
        assert j.joint_type == "revolute"
        assert abs(j.limit_lower - (-1.57)) < 1e-6
        assert abs(j.limit_upper - 1.57) < 1e-6
        assert j.damping == 0.5
        assert j.friction == 0.1

    def test_parse_visual_geometry(self, conv_mod, tmp_path):
        path = self._make_urdf(tmp_path)
        model = conv_mod.URDFParser.parse(path)
        base = [l for l in model.links if l.name == "base_link"][0]
        assert base.visual_geometry is not None
        assert base.visual_geometry["type"] == "box"

    def test_parse_collision_geometry(self, conv_mod, tmp_path):
        path = self._make_urdf(tmp_path)
        model = conv_mod.URDFParser.parse(path)
        base = [l for l in model.links if l.name == "base_link"][0]
        assert base.collision_geometry is not None
        assert base.collision_geometry["type"] == "cylinder"


# ===========================================================================
# TestMJCFGenerator
# ===========================================================================


class TestMJCFGenerator:
    def test_generate_mjcf(self, conv_mod, tmp_path):
        model = conv_mod.RobotModel(
            name="gen_robot",
            links=[conv_mod.Link(name="base"), conv_mod.Link(name="link1")],
            joints=[conv_mod.Joint(name="j1", joint_type="revolute", parent="base", child="link1")],
        )
        output = str(tmp_path / "robot.xml")
        conv_mod.MJCFGenerator.generate(model, output)
        tree = ET.parse(output)
        root = tree.getroot()
        assert root.tag == "mujoco"
        assert root.get("model") == "gen_robot"

    def test_mjcf_has_actuators(self, conv_mod, tmp_path):
        model = conv_mod.RobotModel(
            name="act_robot",
            links=[conv_mod.Link(name="base"), conv_mod.Link(name="link1")],
            joints=[conv_mod.Joint(name="j1", joint_type="revolute", parent="base", child="link1")],
        )
        output = str(tmp_path / "act_robot.xml")
        conv_mod.MJCFGenerator.generate(model, output)
        tree = ET.parse(output)
        actuator = tree.getroot().find("actuator")
        assert actuator is not None
        motors = actuator.findall("motor")
        assert len(motors) >= 1


# ===========================================================================
# TestSDFGenerator
# ===========================================================================


class TestSDFGenerator:
    def test_generate_sdf(self, conv_mod, tmp_path):
        model = conv_mod.RobotModel(
            name="sdf_robot",
            links=[conv_mod.Link(name="base")],
            joints=[],
        )
        output = str(tmp_path / "robot.sdf")
        conv_mod.SDFGenerator.generate(model, output)
        tree = ET.parse(output)
        root = tree.getroot()
        assert root.tag == "sdf"


# ===========================================================================
# TestUnifiedModelConverter
# ===========================================================================


class TestUnifiedModelConverter:
    def test_supported_formats(self, conv_mod):
        c = conv_mod.UnifiedModelConverter()
        assert "urdf" in c.SUPPORTED_FORMATS
        assert "mjcf" in c.SUPPORTED_FORMATS
        assert "sdf" in c.SUPPORTED_FORMATS

    def test_detect_format(self, conv_mod):
        c = conv_mod.UnifiedModelConverter()
        from pathlib import Path

        assert c._detect_format(Path("robot.urdf")) == "urdf"
        assert c._detect_format(Path("robot.xml")) == "mjcf"
        assert c._detect_format(Path("robot.sdf")) == "sdf"

    def test_conversion_log_initially_empty(self, conv_mod):
        c = conv_mod.UnifiedModelConverter()
        assert c.conversion_log == []

    def test_get_extension(self, conv_mod):
        c = conv_mod.UnifiedModelConverter()
        assert c._get_extension("urdf") == "urdf"
        assert c._get_extension("mjcf") == "xml"
        assert c._get_extension("sdf") == "sdf"
