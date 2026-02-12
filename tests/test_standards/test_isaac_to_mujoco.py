"""Tests for Isaac-to-MuJoCo Conversion Pipeline.

Validates PhysX/MuJoCo contact parameter conversion, ConversionConfig
defaults, URDFToMJCFConverter parsing and generation, and the
IsaacToMuJoCoConverter high-level interface.

LICENSE: MIT
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
MOD_PATH = "q1-2026-standards/objective-1-bidirectional-conversion/isaac_to_mujoco_pipeline.py"


@pytest.fixture()
def pipe_mod():
    return load_module("isaac_to_mujoco_pipeline", MOD_PATH)


# ===========================================================================
# TestPhysXContactParams
# ===========================================================================


class TestPhysXContactParams:
    def test_defaults(self, pipe_mod):
        p = pipe_mod.PhysXContactParams()
        assert p.contact_offset == 0.001
        assert p.static_friction == 0.8
        assert p.dynamic_friction == 0.6
        assert p.restitution == 0.1

    def test_custom_values(self, pipe_mod):
        p = pipe_mod.PhysXContactParams(static_friction=0.5, dynamic_friction=0.3)
        assert p.static_friction == 0.5


# ===========================================================================
# TestMuJoCoContactParams
# ===========================================================================


class TestMuJoCoContactParams:
    def test_defaults(self, pipe_mod):
        p = pipe_mod.MuJoCoContactParams()
        assert p.solref == (0.01, 1.0)
        assert p.condim == 3
        assert len(p.friction) == 3

    def test_custom_friction(self, pipe_mod):
        p = pipe_mod.MuJoCoContactParams(friction=(0.5, 0.01, 0.005))
        assert p.friction[0] == 0.5


# ===========================================================================
# TestConversionConfig
# ===========================================================================


class TestConversionConfig:
    def test_defaults(self, pipe_mod):
        cfg = pipe_mod.ConversionConfig()
        assert cfg.timestep == 0.002
        assert cfg.solver_iterations == 50
        assert cfg.integrator == "implicitfast"
        assert cfg.default_gear == 50.0
        assert cfg.default_damping == 0.5
        assert cfg.default_armature == 0.01
        assert cfg.validate_output is True

    def test_custom_config(self, pipe_mod):
        cfg = pipe_mod.ConversionConfig(timestep=0.001, strict_validation=True)
        assert cfg.timestep == 0.001
        assert cfg.strict_validation is True


# ===========================================================================
# TestPhysicsParameterConverter
# ===========================================================================


class TestPhysicsParameterConverter:
    def test_physx_to_mujoco(self, pipe_mod):
        physx = pipe_mod.PhysXContactParams()
        mj = pipe_mod.PhysicsParameterConverter.physx_to_mujoco_contact(physx)
        assert isinstance(mj, pipe_mod.MuJoCoContactParams)
        assert mj.condim == 3
        # Sliding friction should match static friction
        assert mj.friction[0] == physx.static_friction

    def test_mujoco_to_physx(self, pipe_mod):
        mj = pipe_mod.MuJoCoContactParams()
        physx = pipe_mod.PhysicsParameterConverter.mujoco_to_physx_contact(mj)
        assert isinstance(physx, pipe_mod.PhysXContactParams)
        assert physx.static_friction == mj.friction[0]
        assert physx.dynamic_friction == mj.friction[0] * 0.8

    def test_roundtrip_preserves_friction(self, pipe_mod):
        original = pipe_mod.PhysXContactParams(static_friction=0.6)
        mj = pipe_mod.PhysicsParameterConverter.physx_to_mujoco_contact(original)
        recovered = pipe_mod.PhysicsParameterConverter.mujoco_to_physx_contact(mj)
        assert abs(recovered.static_friction - original.static_friction) < 1e-6

    def test_solref_timeconst_clamped(self, pipe_mod):
        physx = pipe_mod.PhysXContactParams(contact_offset=0.0)
        mj = pipe_mod.PhysicsParameterConverter.physx_to_mujoco_contact(physx)
        timeconst = mj.solref[0]
        assert 0.001 <= timeconst <= 0.1


# ===========================================================================
# TestURDFToMJCFConverter
# ===========================================================================


def _make_urdf(tmp_path, name="test_robot"):
    """Create a minimal URDF file for testing."""
    urdf = f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry><box size="0.2 0.2 0.4"/></geometry>
    </visual>
    <collision>
      <geometry><box size="0.2 0.2 0.4"/></geometry>
    </collision>
  </link>
  <link name="link1">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <origin xyz="0 0 0.4"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
    <dynamics damping="0.3" friction="0.1"/>
  </joint>
</robot>"""
    path = tmp_path / f"{name}.urdf"
    path.write_text(urdf)
    return str(path)


class TestURDFToMJCFConverter:
    def test_convert_basic(self, pipe_mod, tmp_path):
        urdf = _make_urdf(tmp_path)
        output = str(tmp_path / "robot.xml")
        converter = pipe_mod.URDFToMJCFConverter()
        result = converter.convert(urdf, output)
        assert result["success"] is True
        assert result["links_converted"] == 2
        assert result["joints_converted"] == 1

    def test_joint_type_map(self, pipe_mod):
        jtm = pipe_mod.URDFToMJCFConverter.JOINT_TYPE_MAP
        assert jtm["revolute"] == "hinge"
        assert jtm["prismatic"] == "slide"
        assert jtm["fixed"] is None

    def test_output_is_valid_xml(self, pipe_mod, tmp_path):
        urdf = _make_urdf(tmp_path)
        output = str(tmp_path / "out.xml")
        converter = pipe_mod.URDFToMJCFConverter()
        converter.convert(urdf, output)
        tree = ET.parse(output)
        root = tree.getroot()
        assert root.tag == "mujoco"

    def test_mjcf_has_option_element(self, pipe_mod, tmp_path):
        urdf = _make_urdf(tmp_path)
        output = str(tmp_path / "opt.xml")
        pipe_mod.URDFToMJCFConverter().convert(urdf, output)
        tree = ET.parse(output)
        option = tree.getroot().find("option")
        assert option is not None
        assert option.get("integrator") == "implicitfast"

    def test_conversion_log(self, pipe_mod, tmp_path):
        urdf = _make_urdf(tmp_path)
        output = str(tmp_path / "log.xml")
        converter = pipe_mod.URDFToMJCFConverter()
        converter.convert(urdf, output)
        log = converter.get_conversion_log()
        assert len(log) == 1
        assert log[0]["success"] is True

    def test_custom_physx_params(self, pipe_mod, tmp_path):
        urdf = _make_urdf(tmp_path)
        output = str(tmp_path / "physx.xml")
        converter = pipe_mod.URDFToMJCFConverter()
        physx = pipe_mod.PhysXContactParams(static_friction=0.5)
        result = converter.convert(urdf, output, physx_params=physx)
        assert result["success"] is True

    def test_rpy_to_quat_identity(self, pipe_mod):
        converter = pipe_mod.URDFToMJCFConverter()
        q = converter._rpy_to_quat(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(q, np.array([1, 0, 0, 0]))


# ===========================================================================
# TestIsaacToMuJoCoConverter
# ===========================================================================


class TestIsaacToMuJoCoConverter:
    def test_convert_urdf(self, pipe_mod, tmp_path):
        urdf = _make_urdf(tmp_path)
        output = str(tmp_path / "isaac_out.xml")
        converter = pipe_mod.IsaacToMuJoCoConverter()
        result = converter.convert_urdf(urdf, output)
        assert result["success"] is True

    def test_convert_usd_not_available(self, pipe_mod, tmp_path):
        converter = pipe_mod.IsaacToMuJoCoConverter()
        result = converter.convert_usd("scene.usd", "scene.xml")
        assert result["success"] is False

    def test_default_config(self, pipe_mod):
        converter = pipe_mod.IsaacToMuJoCoConverter()
        assert converter.config.timestep == 0.002
        assert converter.config.validate_output is True

    def test_custom_config_propagation(self, pipe_mod, tmp_path):
        cfg = pipe_mod.ConversionConfig(timestep=0.001, default_gear=100.0)
        converter = pipe_mod.IsaacToMuJoCoConverter(config=cfg)
        assert converter.urdf_converter.config.default_gear == 100.0

        urdf = _make_urdf(tmp_path, name="gear_robot")
        output = str(tmp_path / "gear.xml")
        converter.convert_urdf(urdf, output)
        tree = ET.parse(output)
        motors = tree.getroot().findall(".//motor")
        assert len(motors) >= 1
        assert motors[0].get("gear") == "100.0"
