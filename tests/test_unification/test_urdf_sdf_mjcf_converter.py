"""Unit tests for unification/simulation_physics/urdf_sdf_mjcf_converter.py.

Tests cover Link, Joint, RobotModel dataclasses and URDFParser.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

MOD_PATH = "unification/simulation_physics/urdf_sdf_mjcf_converter.py"


@pytest.fixture()
def mod():
    return load_module("urdf_sdf_mjcf_converter", MOD_PATH)


class TestLink:
    def test_creation(self, mod):
        link = mod.Link(
            name="base_link",
            mass=5.0,
            inertia=np.eye(3) * 0.1,
            com_position=np.zeros(3),
        )
        assert link.name == "base_link"
        assert link.mass == 5.0

    def test_zero_mass_link(self, mod):
        link = mod.Link(
            name="visual_link",
            mass=0.0,
            inertia=np.zeros((3, 3)),
            com_position=np.zeros(3),
        )
        assert link.mass == 0.0


class TestJoint:
    def test_revolute_joint(self, mod):
        joint = mod.Joint(
            name="joint_1",
            joint_type="revolute",
            parent="base_link",
            child="link_1",
            axis=np.array([0.0, 0.0, 1.0]),
        )
        assert joint.joint_type == "revolute"
        assert joint.parent == "base_link"

    def test_prismatic_joint(self, mod):
        joint = mod.Joint(
            name="slide_joint",
            joint_type="prismatic",
            parent="link_1",
            child="link_2",
            axis=np.array([1.0, 0.0, 0.0]),
            limit_lower=-0.1,
            limit_upper=0.1,
        )
        assert joint.joint_type == "prismatic"
        assert joint.limit_upper == 0.1

    def test_fixed_joint(self, mod):
        joint = mod.Joint(
            name="fixed_joint",
            joint_type="fixed",
            parent="base",
            child="sensor",
            axis=np.array([0.0, 0.0, 1.0]),
        )
        assert joint.joint_type == "fixed"


class TestRobotModel:
    def test_creation(self, mod):
        link = mod.Link(name="base", mass=1.0, inertia=np.eye(3), com_position=np.zeros(3))
        joint = mod.Joint(name="j1", joint_type="revolute", parent="base", child="l1", axis=np.array([0, 0, 1.0]))
        model = mod.RobotModel(
            name="test_robot",
            links=[link],
            joints=[joint],
            source_format="urdf",
        )
        assert model.name == "test_robot"
        assert model.source_format == "urdf"

    def test_empty_model(self, mod):
        model = mod.RobotModel(
            name="empty",
            links=[],
            joints=[],
            source_format="urdf",
        )
        assert len(model.links) == 0


class TestURDFParser:
    def test_instantiation(self, mod):
        parser = mod.URDFParser()
        assert parser is not None
        assert hasattr(parser, "parse")
